import logging

import torch
import torch.nn as nn
import os
import numpy as np
from torch._utils import _get_all_device_indices
from tqdm import tqdm

import satellitepy.models.bbavector.loss as loss_utils
from satellitepy.models.utils import EarlyStopping
from satellitepy.models.bbavector.utils import save_model, load_checkpoint


class TrainModule(object):
    def __init__(self,
                 train_dataset,
                 valid_dataset,
                 model,
                 tasks,
                 down_ratio,
                 out_folder,
                 init_lr,
                 num_epoch,
                 batch_size,
                 num_workers,
                 conf_thresh,
                 ngpus,
                 resume_train,
                 patience,
                 target_task='coarse-class'):
        torch.manual_seed(317)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.down_ratio = down_ratio
        self.out_folder = out_folder
        self.init_lr = init_lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.conf_thresh = conf_thresh
        self.ngpus = ngpus
        self.resume_train = resume_train
        self.patience = patience
        self.tasks = tasks
        self.target_task = target_task

    def train_network(self):
        logger = logging.getLogger('')

        save_path = str(self.out_folder)
        if self.ngpus > 1:
            if torch.cuda.device_count() > 1:
                logger.info(f"{torch.cuda.device_count()} GPUs available. Let's use {self.ngpus} GPUs!")
                device_ids = _get_all_device_indices()[:self.ngpus]
                print(device_ids)
                self.model = nn.DataParallel(self.model, device_ids)
        elif self.ngpus == 0:
            self.device = 'cpu'
        self.model.to(self.device)

        if self.resume_train:
            self.model, self.optimizer, start_epoch, valid_loss = load_checkpoint(self.model,
                                                                                  self.resume_train)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr)
            start_epoch = -1
            valid_loss = np.Inf

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=start_epoch)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        criterion = loss_utils.LossAll(self.tasks, self.target_task)
        logger.info('Setting up data...')

        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True, )

        if self.valid_dataset:
            valid_loader = torch.utils.data.DataLoader(self.valid_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       num_workers=self.num_workers,
                                                       pin_memory=True,
                                                       drop_last=True, )

        early_stopping = EarlyStopping(
            patience=self.patience,
            verbose=True,
            path=os.path.join(save_path, 'model_best.pth'),
            val_loss_min=valid_loss,
            trace_func=print)
        logger.info('Starting training...')
        for epoch in range(0, self.num_epoch):
            logger.info('-' * 10)
            logger.info(f'Epoch: {epoch}/{self.num_epoch}')
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
                    logger.info("Early stopping")
                    break

            else:
                save_model(os.path.join(save_path, f'model_no_valid_{epoch}.pth'),
                           epoch,
                           self.model,
                           self.optimizer)

            train_loss_msg = ''
            for k, v in train_loss.items():
                train_loss_msg += f'{k}: {v:.5f}\n'
            val_loss_msg = ''
            for k, v in valid_loss.items():
                val_loss_msg += f'{k}: {v:.5f}\n'

            msg = (f'[{epoch}/{self.num_epoch}]\n' +
                   f'training_losses\n' +
                   '----------------------------\n' +
                   train_loss_msg +
                   '----------------------------\n' +
                   "valid_losses\n" +
                   '----------------------------\n' +
                   val_loss_msg +
                   '----------------------------')

            logger.info(msg)

    def run_train(self, data_loader, criterion):
        self.model.train()
        running_loss = {}
        for data_dict in tqdm(data_loader):
            for name in data_dict.keys():
                if name not in ['img_w', 'img_h', 'img_path', 'label_path']:
                    data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            self.optimizer.zero_grad()
            pr_decs = self.model(data_dict['input'])
            loss_dict = criterion(pr_decs, data_dict, self.target_task)
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
        return epoch_loss

    def run_valid(self, data_loader, criterion):
        self.model.eval()
        running_loss = {}

        for data_dict in tqdm(data_loader):
            for name in data_dict.keys():
                if name not in ['img_w', 'img_h', 'img_path', 'label_path']:
                    data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            with torch.no_grad():
                pr_decs = self.model(data_dict['input'])
                loss_dict = criterion(pr_decs, data_dict, self.target_task)
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
