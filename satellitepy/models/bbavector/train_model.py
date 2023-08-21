import logging

import torch
import torch.nn as nn
import os
import numpy as np
import cv2
# import func_utils
from tqdm import tqdm

import satellitepy.models.bbavector.loss as loss_utils
from satellitepy.models.utils import EarlyStopping
from satellitepy.models.bbavector.utils import save_model, load_checkpoint#, collater

class TrainModule(object):
    def __init__(self, 
        train_dataset, 
        valid_dataset, 
        model, 
        tasks,
        # decoder, 
        down_ratio,
        out_folder,
        init_lr,
        num_epoch,
        batch_size,
        num_workers,
        conf_thresh,
        ngpus,
        resume_train,
        patience):
        torch.manual_seed(317)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # self.dataset_phase = {'dota': ['train'],
        #                       'hrsc': ['train', 'test']}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        # self.decoder = decoder
        self.down_ratio = down_ratio
        self.out_folder=out_folder
        self.init_lr=init_lr
        self.num_epoch=num_epoch
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.conf_thresh=conf_thresh
        self.ngpus=ngpus
        self.resume_train=resume_train
        self.patience=patience
        self.tasks = tasks

    def train_network(self):
        logger = logging.getLogger(__name__)

        save_path = str(self.out_folder)
        if self.ngpus>1:
            if torch.cuda.device_count() > 1:
                logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        # add resume part for continuing training when break previously, 10-16-2020
        if self.resume_train:
            self.model, self.optimizer, start_epoch, valid_loss = load_checkpoint(self.model, 
                                                                        self.resume_train)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.init_lr)
            start_epoch = -1
            valid_loss = np.Inf

        # print(self.optimizer)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=start_epoch)
        # end 

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        criterion = loss_utils.LossAll(self.tasks)
        logger.info('Setting up data...')

        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=True,
                                                           num_workers=self.num_workers,
                                                           pin_memory=True,
                                                           drop_last=True,)
                                                        #    collate_fn=collater)
        if self.valid_dataset:
            valid_loader = torch.utils.data.DataLoader(self.valid_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=self.num_workers,
                                                    pin_memory=True,
                                                    drop_last=True,)
                                                    # collate_fn=collater)
        early_stopping = EarlyStopping(
            patience=self.patience, 
            verbose=True, 
            path=os.path.join(save_path, 'model_best.pth'), 
            val_loss_min=valid_loss,
            trace_func=print)
        logger.info('Starting training...')
        # train_loss = []
        # ap_list = []
        for epoch in range(0, self.num_epoch):
            logger.info('-'*10)
            logger.info('Epoch: {}/{} '.format(epoch, self.num_epoch))
            train_loss = self.run_train(
                                        data_loader=train_loader,
                                        criterion=criterion)
            self.scheduler.step()


            if self.valid_dataset:
                logger.info('Validation is starting...')
                valid_loss = self.run_valid(valid_loader, criterion)
                total_val_loss = sum([l for l in valid_loss.values()])
                early_stopping(total_val_loss, self.model, self.optimizer, epoch)
                if early_stopping.early_stop:
                    # self.logger.info("Early stopping")
                    logger.info("Early stopping")
                    break

            else:
                save_model(os.path.join(save_path, 'model_no_valid_{}.pth'.format(epoch)),
                                epoch,
                                self.model,
                                self.optimizer)

            train_loss_msg = ""
            for k, v in train_loss.items():
                train_loss_msg += f"{k}: {v:.5f}\n"
            val_loss_msg = ""
            for k, v in valid_loss.items():
                val_loss_msg += f"{k}: {v:.5f}\n"

            msg = (f'[{epoch}/{self.num_epoch}]\n' +
                    f'training_losses\n' +
                    '----------------------------\n' +
                    train_loss_msg +
                    '----------------------------\n' +
                    "valid_losses\n" +
                    '----------------------------\n' +
                    val_loss_msg +
                    '----------------------------')
                    # f'valid_acc: {valid_acc:.2f}')

            # self.logger.info(msg)
            logger.info(msg)

    def test_network(self):

        logger = logging.getLogger(__name__)

        valid_loss = self.run_valid(valid_loader, criterion)
        logger.info(f'valid_loss: {valid_loss:.5f}')

    def run_train(self,data_loader, criterion):
        self.model.train()
        running_loss = {}
        for data_dict in tqdm(data_loader):
            for name in data_dict.keys():
                if name not in ["img_w", "img_h", "img_path", "label_path"]:
                    data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            self.optimizer.zero_grad()
            pr_decs = self.model(data_dict['input'])
            loss_dict = criterion(pr_decs, data_dict)
            total_loss = sum([l for l in loss_dict.values()])
            total_loss.backward()
            self.optimizer.step()
            for k, v in loss_dict.items():
                running_loss.setdefault(k, [0., 0])
                if isinstance(v, torch.Tensor):
                    running_loss[k][0] += v.item()
                    running_loss[k][1] += 1
                else:
                    running_loss[k][0] += v
                    running_loss[k][1] += 1
        epoch_loss = {k: v[0] / v[1] for k, v in running_loss.items()}
        # print('{} loss: {}'.format(epoch_loss))
        return epoch_loss

    def run_valid(self,data_loader,criterion):
        # VALIDATE MODEL
        self.model.eval()
        acc_sums = 0
        running_loss = {}

        for data_dict in tqdm(data_loader):
            for name in data_dict.keys():
                if name not in ["img_w", "img_h", "img_path", "label_path"]:
                    data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            with torch.no_grad():
                pr_decs = self.model(data_dict['input'])
                loss_dict = criterion(pr_decs, data_dict)
                for k, v in loss_dict.items():
                    running_loss.setdefault(k, [0., 0])
                    if isinstance(v, torch.Tensor):
                        running_loss[k][0] += v.item()
                        running_loss[k][1] += 1
                    else:
                        running_loss[k][0] += v
                        running_loss[k][1] += 1
        epoch_loss = {k: v[0] / v[1] for k, v in running_loss.items()}
        return epoch_loss
            
            # if 'test' in self.dataset_phase[args.dataset] and epoch%5==0:
            #     mAP = self.dec_eval(args, dsets['test'])
            #     ap_list.append(mAP)
            #     np.savetxt(os.path.join(save_path, 'ap_list.txt'), ap_list, fmt='%.6f')

            # self.save_model(os.path.join(save_path, 'model_last.pth'),
            #                 epoch,
            #                 self.model,
            #                 self.optimizer)

    # def run_epoch(self, phase, data_loader, criterion):
    #     if phase == 'train':
    #         self.model.train()
    #     else:
    #         self.model.eval()
    #     running_loss = 0.
    #     for data_dict in tqdm(data_loader):
    #         for name in data_dict:
    #             data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
    #         if phase == 'train':
    #             self.optimizer.zero_grad()
    #             with torch.enable_grad():
    #                 pr_decs = self.model(data_dict['input'])
    #                 loss = criterion(pr_decs, data_dict)
    #                 loss.backward()
    #                 self.optimizer.step()
    #         else:
    #             with torch.no_grad():
    #                 pr_decs = self.model(data_dict['input'])
    #                 loss = criterion(pr_decs, data_dict)

    #         running_loss += loss.item()
    #     epoch_loss = running_loss / len(data_loader)
    #     print('{} loss: {}'.format(phase, epoch_loss))
    #     return epoch_loss


    # def dec_eval(self, args, dsets):
    #     result_path = 'result_'+self.dataset
    #     if not os.path.exists(result_path):
    #         os.mkdir(result_path)

    #     self.model.eval()
    #     func_utils.write_results(args,
    #                              self.model,dsets,
    #                              self.down_ratio,
    #                              self.device,
    #                              self.decoder,
    #                              result_path)
    #     ap = dsets.dec_evaluation(result_path)
    #     return ap



        # self.optimizer = torch.optim.Adam(self.model.parameters(), self.init_lr)

        # valid_loader = torch.utils.data.DataLoader(self.valid_dataset,
        #                                 batch_size=self.batch_size,
        #                                 shuffle=True,
        #                                 num_workers=self.num_workers,
        #                                 pin_memory=True,
        #                                 drop_last=True,
        #                                 collate_fn=collater)
        # criterion = loss_utils.LossAll()

        # self.model, self.optimizer, start_epoch, valid_loss = load_checkpoint(
        #     self.model, 
        #     self.optimizer, 
        #     self.resume_train, 
        #     strict=True)
        # self.model.to(self.device)
