from torchvision.transforms import Compose
# from torchvision.models import resnet50, ResNet50_Weights, 
from torchvision.models import googlenet, GoogLeNet_Weights
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import mnasnet0_5, MNASNet0_5_Weights
from torchvision.models import shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights
from torchvision.models import regnet_y_800mf, RegNet_Y_800MF_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import convnext_small, ConvNeXt_Small_Weights 
from torchvision.models import swin_t, Swin_T_Weights 


import torch
# import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
import os
# from tqdm import tqdm
# import seaborn as sn
# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import logging
from tqdm import tqdm


from src.classifier.utils import EarlyStopping
from src.models.models import *
from src.data.dataset.classification import DatasetClassification
from src.transforms import Normalize, ToTensor, AddAxis, HorizontalFlip

# TODO: Log files
class Classifier(object):
    """docstring for Classifier"""

    def __init__(self, data_settings, exp_settings):
        # super(Classifier, self).__init__()
        self.exp_settings = exp_settings
        self.data_settings = data_settings
        # print(list(exp_settings.keys()))
        self.logger = logging.getLogger(__name__)
    def train(self, model, loss_func, optimizer, loaders,
              patience=10):  # ,load_last_state=False):
        # DATA
        loader_train = loaders['train']
        loader_val = loaders['val']

        # TRAINING HYPERPARAMETERS
        epochs = self.exp_settings['training']['epochs']
        batch_size = self.exp_settings['training']['batch_size']
        cuda_device = self.exp_settings['training']['cuda_device'] if 'cuda_device' in self.exp_settings['training'] else 0


        # READ MODEL AND MOVE IT TO GPU
        device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        loss_func.to(device)
        model_path = self.exp_settings['model']['path']
        model, optimizer, epoch, loss = self.load_checkpoint(model=model,optimizer=optimizer,model_path=model_path)
        # EARLY STOPPING
        early_stopping = EarlyStopping(
            patience=patience, 
            verbose=True, 
            path=model_path, 
            val_loss_min=loss,
            trace_func=self.logger.info)

        for epoch in range(epochs):  # loop over the dataset multiple times
            lr = self.get_lr(optimizer)
            self.logger.info(f'Learning rate: {lr}')
            # LOSS
            train_losses = []
            val_losses = []

            # TRAIN MODEL
            model.train()
            for data in tqdm(loader_train):
                # data is a dict with keys "image", "label"
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize

                outputs = model(data['image'].to(device))
                # print(data['label'])
                # print(outputs.softmax(dim=1))
                # outputs_softmax = .to(device)
                # data_label = 
                # loss = loss_func(outputs.softmax(dim=1), data['label'].to(device))
                loss = loss_func(outputs, data['label'].to(device))
                # loss = loss_func(torch.argmax(outputs,dim=1), data['label'].to(device))
                # train_acc = torch.sum(outputs_softmax == data_label)

                # writer.add_scalar("Loss/train", loss, epoch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # VALIDATE MODEL
            model.eval()
            acc_sums = 0
            for data in tqdm(loader_val):
                # forward pass: compute predicted outputs by passing inputs to
                # the model
                outputs = model(data['image'].to(device))

                # calculate the loss
                gt = data['label'].to(device)
                loss = loss_func(outputs, gt)
                # writer.add_scalar("Loss/val", loss, epoch)
                # record validation loss
                val_losses.append(loss.item())
                pred_int = torch.argmax(outputs.softmax(dim=1),dim=1)

                acc_sum = torch.sum(pred_int == gt)#/len(pred)
                acc_sums += acc_sum

            valid_acc = acc_sums/len(loader_val.dataset)
            train_loss = np.average(train_losses)
            val_loss = np.average(val_losses)

            # train_losses_avg.append(train_loss)
            # val_losses_avg.append(val_loss)

            msg = (f'[{epoch}/{epochs}] ' +
                   f'train_loss: {train_loss:.5f} ' +
                   f'valid_loss: {val_loss:.5f} ' +
                   f'valid_acc: {valid_acc:.2f}')
            self.logger.info(msg)

            early_stopping(val_loss, model, optimizer, epoch)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            # torch.save(model, model_path)
        # writer.flush()
        # writer.close()
        self.logger.info('Finished Training')

    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    def get_dataset(self, task):
        dataset_parts = self.data_settings['dataset_parts']
        if task == 'segmentation':
            # DATASET
            dataset = {dataset_part: DatasetSegmentation(
                settings=self.settings,
                dataset_part=dataset_part,
                transform=Compose(
                    [ToTensor(), Normalize(task=task), AddAxis()])
            )
                for dataset_part in dataset_parts}
            dataset_split = self.split_dataset(dataset)
        elif task == 'classification':
            dataset = {dataset_part: DatasetClassification(
                exp_settings=self.exp_settings,
                data_settings=self.data_settings,
                dataset_part=dataset_part,
                transform=Compose([ToTensor(), Normalize(task=task), HorizontalFlip(0.8)])
            )
                for dataset_part in dataset_parts}
            dataset_split = self.split_dataset(dataset)

        return dataset_split

    def split_dataset(self, dataset):
        if self.exp_settings['training']['split_ratio']:
            dataset_train = dataset['train']
            dataset_val = dataset['val']
            # MERGE DATASETS
            # dataset_full=dataset_train
            dataset_full = torch.utils.data.ConcatDataset(
                [dataset_train, dataset_val])
            len_full_dataset = len(dataset_full)

            # SPLIT RATIO
            ratio_train, ratio_test, ratio_val = self.exp_settings['training']['split_ratio']
            train_size = int(ratio_train * len_full_dataset)
            test_size = int(ratio_test * len_full_dataset)
            val_size = len_full_dataset - train_size - test_size
            # dataset_train, dataset_test, dataset_val = torch.utils.data.random_split(dataset_full, [train_size, test_size,val_size])
            dataset_train = torch.utils.data.Subset(
                dataset_full, range(train_size))
            dataset_test = torch.utils.data.Subset(
                dataset_full, range(train_size, train_size + test_size))
            dataset_val = torch.utils.data.Subset(dataset_full, range(
                train_size + test_size, train_size + test_size + val_size))
            self.logger.info(
                f'Full dataset (train+test+val) is split into:\n{len(dataset_train)},{len(dataset_test)},{len(dataset_val)}\n')
        else:
            dataset_train = dataset['train']
            dataset_val = dataset['val']
            self.logger.info(f'Train size: {len(dataset_train)}')
            self.logger.info(f'Val size: {len(dataset_val)}')
            dataset_split = {'train': dataset_train,
                             'val': dataset_val}
            if 'test' in list(dataset.keys()):
                dataset_test = dataset['test']
                self.logger.info(f'Test size:{len(dataset_test)}')
                dataset_split['test']=dataset_test
        return dataset_split

    def get_loaders(self,task):
        dataset = self.get_dataset(task)

        loaders = {dataset_part:self.get_loader(
            dataset=dataset[dataset_part],
            batch_size=self.exp_settings['training']['batch_size'],
            shuffle=True) for dataset_part in list(dataset.keys())}
        return loaders

    def get_loader(self, dataset, shuffle, batch_size, num_workers=4):
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)

        return loader



    def get_model(self):
        model_name = self.exp_settings['model']['name']
        if model_name == 'UNet':
            model = UNet(init_features=self.exp_settings['model']['init_features'])
        elif model_name == 'Custom_0':
            model = Custom_0()
        elif model_name.startswith('effnet'):
            effnet_no = model_name.split('_')[-1]
            model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', f'nvidia_efficientnet_b{effnet_no}', pretrained=True)
        elif model_name=='googlenet':
            weights = GoogLeNet_Weights.DEFAULT
            model = googlenet(weights=weights)
        elif model_name=='densenet':
            weights = DenseNet121_Weights.DEFAULT
            model = densenet121(weights=weights)
        elif model_name=='mnasnet0_5':
            weights = MNASNet0_5_Weights.DEFAULT
            model = mnasnet0_5(weights=weights)
        elif model_name=='shufflenet_v2_x1_5':
            weights = ShuffleNet_V2_X1_5_Weights.DEFAULT
            model = shufflenet_v2_x1_5(weights=weights)
        elif model_name=='regnet_y_800mf':
            weights = RegNet_Y_800MF_Weights.DEFAULT
            model = regnet_y_800mf(weights=weights)
        elif model_name=='resnet18':
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
        elif model_name=='resnet34':
            weights = ResNet34_Weights.DEFAULT
            model = resnet34(weights=weights)
        elif model_name=='efficientnet_b3':
            weights = EfficientNet_B3_Weights.DEFAULT
            model = efficientnet_b3(weights=weights)
        elif model_name=='convnext_small':
            weights = ConvNeXt_Small_Weights.DEFAULT
            model = convnext_small(weights=weights)
        elif model_name=='swin_t':
            weights = Swin_T_Weights.DEFAULT
            model = swin_t(weights=weights)
        else:
            self.logger.warn('Please define your model first.')
            return 0
        return model

    def load_checkpoint(self, model,model_path,optimizer=None,is_train=True):
        if os.path.exists(model_path):
            self.logger.info(f'Model is read from the previous version at:\n{model_path}')
            if input('Do you confirm that? [y/n] ') != 'y':
                self.logger.info('The existing model will be overwritten!\n')
                time.sleep(2)
                return 0
            else:
                # model = torch.load(model_path)
                checkpoint = torch.load(model_path)
                # print(checkpoint.keys())
                model.load_state_dict(checkpoint['model_state_dict'])
                if is_train:
                    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epoch = checkpoint['epoch']
                    loss = checkpoint['loss']

                    return model, optimizer, epoch, loss
                else:
                    return model
        else:
            return model, optimizer, None, None