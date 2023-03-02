import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose
import torch
import matplotlib.pyplot as plt
import cv2
# import numpy as np
from torchmetrics import F1Score

# from transforms import *
from .classifier import Classifier
# from dataset import SegmentationDataset
# from unet import UNet

# MOVE get_loaders to Classifier


class ClassifierSegmentation(Classifier):
    def __init__(self, utils, dataset):
        super(ClassifierSegmentation, self).__init__(utils.settings)
        self.settings = utils.settings
        self.dataset = dataset
        self.utils = utils

    def train(self):

        model = self.utils.get_model()
        # COST AND OPTIMIZER FUNCS
        loss_func = DiceLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.settings['training']['learning_rate'],
            momentum=0.9)

        # DATA
        loaders = self.get_loaders(
            batch_size=self.settings['training']['batch_size'])

        super().train(model, loss_func, optimizer, loaders)


    def get_predictions(self, model, loader):
        for i, data in enumerate(loader):
            y_pred_batch = model(data['image'])
            y_true_batch = data['label']
            image_path = data['image_path']

            yield y_pred_batch.long(), y_true_batch.long(), image_path

    def plot_images(self, dataset_part):
        # BATCH SIZE ()SUBPLOT COUNTER)
        row, col = 3, 5
        batch_size = col

        # MODEL
        model = self.utils.get_model()
        model.load_state_dict(
            torch.load(
                self.settings['model']['path'],
                map_location='cpu'))

        # LOADERS
        loader = self.get_loaders(batch_size=batch_size)[dataset_part]

        prediction_generator = self.get_predictions(model, loader)
        while True:
            fig, ax = plt.subplots(row, col)
            fig.subplots_adjust(wspace=0.1, hspace=0.1)
            # fig.suptitle(f'G.Truth: {y_true_name} Prediction: {y_pred_name}',fontsize=15)
            for ind in range(batch_size):
                y_pred, y_true, img_path = next(prediction_generator)

                img = cv2.cvtColor(
                    cv2.imread(
                        img_path[ind]),
                    cv2.COLOR_BGR2RGB)
                ax[0, ind].imshow(img)
                ax[1, ind].imshow(y_pred[ind, 0] * 255)
                ax[2, ind].imshow(y_true[ind, 0] * 255)
            plt.show()

    def get_f1_score(self, dataset_part):

        # MODEL
        model = self.utils.get_model()
        model.load_state_dict(
            torch.load(
                self.settings['model']['path'],
                map_location='cpu'))

        # LOADERS
        batch_size = 50
        loader = self.get_loaders(batch_size=batch_size)[dataset_part]

        prediction_generator = self.get_predictions(model, loader)

        batch = 5  # len(self.dataset[dataset_part])//batch_size

        f1 = F1Score(num_classes=2, average=None)
        f1_score = 0  # TOTAL
        for i in range(batch):  # batch
            y_pred, y_true, img_path = next(prediction_generator)
            y_pred = y_pred[:, 0].contiguous().view(-1)
            y_true = y_true[:, 0].contiguous().view(-1)
            f1_batch = f1(y_pred, y_true)
            f1_score += f1_batch
            ####
        f1_score /= batch
        print(
            f'f1 score for {batch*batch_size} samples of {dataset_part} set: {f1_score}')


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc
