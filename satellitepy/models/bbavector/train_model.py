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
                 checkpoint_interval=1,
                 gradient_accumulation_steps=1,
                 batch_log_interval=1,
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
        self.checkpoint_interval = checkpoint_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        if batch_log_interval < 1:
            raise ValueError('batch_log_interval must be at least 1')
        self.batch_log_interval = batch_log_interval
        self.tasks = tasks
        self.target_task = target_task

    def train_network(self):
        logger = logging.getLogger('')

        save_path = str(self.out_folder)
        if self.ngpus == 0:
            self.device = 'cpu'

        if self.resume_train:
            self.model, self.optimizer, start_epoch, valid_loss = load_checkpoint(
                self.resume_train, self.down_ratio, self.init_lr,
                map_location=self.device, tasks=self.tasks)
            if valid_loss is None:
                valid_loss = np.inf
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr)
            start_epoch = -1
            valid_loss = np.inf

        # Wrap after loading a checkpoint. Loading reconstructs the model, so
        # wrapping before it would discard DataParallel and resume on GPU 0 only.
        if self.ngpus > 1 and torch.cuda.device_count() > 1:
            logger.info(f"{torch.cuda.device_count()} GPUs available. Let's use {self.ngpus} GPUs!")
            device_ids = _get_all_device_indices()[:self.ngpus]
            print(device_ids)
            self.model = nn.DataParallel(self.model, device_ids)
        self.model.to(self.device)
        # Optimizer state loaded with a checkpoint may contain CPU tensors even
        # when its parameters have moved to CUDA. Adam requires every state
        # tensor to live beside its parameter at the first resumed step.
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=start_epoch)

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        criterion = loss_utils.LossAll(self.tasks, self.target_task)
        logger.info('Setting up data...')

        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   pin_memory=self.device != 'cpu',
                                                   drop_last=False, )

        if self.valid_dataset:
            valid_loader = torch.utils.data.DataLoader(self.valid_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=self.num_workers,
                                                       pin_memory=self.device != 'cpu',
                                                       drop_last=False, )

        early_stopping = EarlyStopping(
            patience=self.patience,
            verbose=True,
            path=os.path.join(save_path, 'model_best.pth'),
            val_loss_min=valid_loss,
            trace_func=print)
        logger.info('Starting training...')
        for epoch in range(start_epoch + 1, self.num_epoch):
            logger.info('-' * 10)
            logger.info(f'Epoch: {epoch}/{self.num_epoch}')
            train_loss = self.run_train(
                data_loader=train_loader,
                criterion=criterion,
                epoch=epoch)
            self.scheduler.step()

            if self.valid_dataset:
                logger.info('Validation is starting...')
                valid_loss = self.run_valid(valid_loader, criterion, epoch=epoch)
                total_val_loss = sum([l for l in valid_loss.values()])
                early_stopping(total_val_loss, self.model, self.optimizer, epoch)
                if early_stopping.early_stop:
                    logger.info("Early stopping")
                    break

            else:
                if ((epoch + 1) % self.checkpoint_interval == 0 or
                        epoch == self.num_epoch - 1):
                    save_model(os.path.join(save_path, f'model_no_valid_{epoch}.pth'),
                               epoch,
                               self.model,
                               self.optimizer)
                valid_loss = {}

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

    @staticmethod
    def _running_loss_values(running_loss):
        return {name: total / count for name, (total, count) in running_loss.items()}

    def _log_batch_progress(self, phase, epoch, batch_index, num_batches, running_loss):
        completed = batch_index + 1
        if completed % self.batch_log_interval != 0 and completed != num_batches:
            return
        losses = self._running_loss_values(running_loss)
        loss_text = ' '.join(f'{name}={value:.5f}' for name, value in losses.items())
        percent = 100.0 * completed / num_batches
        lr = self.optimizer.param_groups[0]['lr']
        logging.getLogger('').info(
            f'{phase} epoch={epoch + 1}/{self.num_epoch} '
            f'batch={completed}/{num_batches} progress={percent:.1f}% '
            f'lr={lr:.8g} {loss_text}')

    def run_train(self, data_loader, criterion, epoch=0):
        self.model.train()
        running_loss = {}
        self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(data_loader, desc=f'Train {epoch + 1}/{self.num_epoch}')
        for batch_index, data_dict in enumerate(progress):
            for name in data_dict.keys():
                if name not in ['img_w', 'img_h', 'img_path', 'label_path']:
                    data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            pr_decs = self.model(data_dict['input'])
            loss_dict = criterion(pr_decs, data_dict, self.target_task)
            total_loss = sum([l for l in loss_dict.values()])
            (total_loss / self.gradient_accumulation_steps).backward()
            if ((batch_index + 1) % self.gradient_accumulation_steps == 0 or
                    batch_index + 1 == len(data_loader)):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            for k, v in loss_dict.items():
                running_loss.setdefault(k, [0., 0])
                if isinstance(v, torch.Tensor):
                    running_loss[k][0] += v.item()
                    running_loss[k][1] += 1
                else:
                    running_loss[k][0] += v
                    running_loss[k][1] += 1
            current_loss = self._running_loss_values(running_loss)
            progress.set_postfix({k: f'{v:.4f}' for k, v in current_loss.items()}, refresh=True)
            self._log_batch_progress('train', epoch, batch_index, len(data_loader), running_loss)
        epoch_loss = self._running_loss_values(running_loss)
        return epoch_loss

    def run_valid(self, data_loader, criterion, epoch=0):
        self.model.eval()
        running_loss = {}

        progress = tqdm(data_loader, desc=f'Valid {epoch + 1}/{self.num_epoch}')
        for batch_index, data_dict in enumerate(progress):
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
            current_loss = self._running_loss_values(running_loss)
            progress.set_postfix({k: f'{v:.4f}' for k, v in current_loss.items()}, refresh=True)
            self._log_batch_progress('valid', epoch, batch_index, len(data_loader), running_loss)
        epoch_loss = self._running_loss_values(running_loss)
        return epoch_loss
