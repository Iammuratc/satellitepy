import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from satellitepy.models.utils import EarlyStopping
from satellitepy.models.chips.chip_models import get_model

class TrainModule(object):
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 test_loader,
                 optimizer,
                 scheduler,
                 loss_fn,
                 n_epochs,
                 save_path,
                 classes,
                 backbone_name,
                 patience=10,
                 val_by_source=False,
                 verbose_output=True):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.save_path = save_path
        self.classes = classes
        self.backbone_name = backbone_name
        self.patience = patience
        self.val_by_source = val_by_source
        self.verbose_output = verbose_output
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train_network(self):
        logger = logging.getLogger('')

        self.model = self.model.to(self.device)

        early_stopping = EarlyStopping(
            patience=self.patience,
            verbose=True,
            path=os.path.join(self.save_path, 'model_best.pth')
        )

        logger.info('Starting training...')
        for epoch in range(0, self.n_epochs):
            logger.info('-' * 10)
            logger.info(f'Epoch: {epoch}/{self.n_epochs}')
            train_loss = self.train()
            self.scheduler.step()

            logger.info('Validation is starting...')
            acc_sums, nums, val_losses = self.validate()

            val_accs = acc_sums / nums
            val_acc = acc_sums.sum(axis=(0, 1))/ nums.sum(axis=(0, 1))
            val_loss = np.average(val_losses)

            msg = (f'[{epoch + 1}/{self.n_epochs}] ' +
                   f'train_loss: {train_loss:.5f} ' +
                   f'valid_loss: {val_loss:.5f} ' +
                   f'valid_acc: {val_acc:.5f} ')

            val_accs = np.where(nums != 0, val_accs, -1)

            if not self.val_by_source:
                val_accs = np.sum(acc_sums, 1)
                nums = nums.sum(axis=1)
                val_accs = val_accs/nums

            if self.verbose_output:
                verbose_msg = (f'class names: \n{self.classes} ' +
                               f'accuracy by class: \n{val_accs}')
                logger.info(verbose_msg)

            logger.info(msg)

            early_stopping(val_loss, self.model, self.optimizer, epoch)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        logger.info('Training finished')


    def train(self):
        train_losses = []
        train_pbar = tqdm(self.train_loader)
        self.model.train()

        for (x, y, s) in train_pbar:
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)

            loss = self.loss_fn(y_hat, y)
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.average(train_losses)
        return train_loss


    def validate(self):
        val_losses = []
        acc_sums = np.zeros([len(self.classes), 3])
        nums = torch.zeros([len(self.classes), 3])
        val_pbar = tqdm(self.val_loader)
        self.model.eval()
        with torch.no_grad():
            for (x, y, s) in val_pbar:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)
                val_loss = self.loss_fn(y_hat, y)
                val_losses.append(val_loss.item())

                pred_int = torch.argmax(y_hat, dim=1).cpu().numpy()
                gt = torch.argmax(y, dim=1).cpu().numpy()

                acc_sum = np.sum(pred_int == gt)
                acc_sums[gt, s] += acc_sum
                nums[gt, s] += 1

        return acc_sums, nums, val_losses

    def test(self):
        logger = logging.getLogger('')
        best_path=os.path.join(self.save_path, 'model_best.pth')
        checkpoint = torch.load(best_path)
        model = get_model(self.backbone_name, len(self.classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)

        logger.info('Testing...')

        acc_sums = np.zeros([len(self.classes), 3])
        nums = torch.zeros([len(self.classes), 3])
        test_pbar = tqdm(self.test_loader)
        self.model.eval()
        with torch.no_grad():
            for (x, y, s) in test_pbar:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x)

                pred_int = torch.argmax(y_hat, dim=1).cpu().numpy()
                gt = torch.argmax(y, dim=1).cpu().numpy()

                acc_sum = np.sum(pred_int == gt)
                acc_sums[gt, s] += acc_sum
                nums[gt, s] += 1

        test_accs = acc_sums / nums
        test_acc = acc_sums.sum(axis=(0, 1)) / nums.sum(axis=(0, 1))
        test_accs = np.where(nums != 0, test_accs, -1)

        msg = (f'test_acc: {test_acc:.5f} ' +
               f'class names: \n{self.classes} \n' +
               f'accuracy by class and source: \n{test_accs}')

        logger.info(msg)

        logger.info('-----------------Combined-----------------')
        test_accs = np.sum(acc_sums, 1)
        nums = nums.sum(axis=1)
        test_accs = test_accs / nums
        msg = (f'accuracy by class: \n{test_accs}')

        logger.info(msg)
        logger.info('-----------------res_dict-----------------')
        res_dict = {}
        for c, a in zip(self.classes, test_accs):
            res_dict[c] = a

        logger.info(res_dict)