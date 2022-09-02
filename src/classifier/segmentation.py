import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose
import torch
import matplotlib.pyplot as plt
import cv2

# from transforms import *
from .classifier import Classifier
# from dataset import SegmentationDataset 
# from unet import UNet

## MOVE get_loaders to Classifier

class ClassifierSegmentation(Classifier):
    def __init__(self,utils,dataset):    
        super(ClassifierSegmentation, self).__init__(utils.settings)
        self.settings = utils.settings
        self.dataset = dataset
        self.utils = utils

    def train(self):

        model = self.utils.get_model()
        ### COST AND OPTIMIZER FUNCS
        loss_func = DiceLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        ### DATA
        loaders = self.get_loaders(batch_size=self.settings['training']['batch_size'])

        super().train(model,loss_func,optimizer,loaders)

    def get_loaders(self,batch_size):
        if self.settings['training']['split_ratio']:
            dataset_train = self.dataset['train']
            # dataset_test = self.get_dataset('test')
            # dataset_val = self.get_dataset('val')
            ### MERGE DATASETS
            # dataset_full = torch.utils.data.ConcatDataset([dataset_train, dataset_test,dataset_val])
            dataset_full = dataset_train
            len_full_dataset = len(dataset_full)

            ### SPLIT RATIO
            ratio_train, ratio_test, ratio_val = self.settings['training']['split_ratio']
            train_size = int(ratio_train * len_full_dataset)
            test_size = int(ratio_test * len_full_dataset)
            val_size = len_full_dataset - train_size - test_size
            # dataset_train, dataset_test, dataset_val = torch.utils.data.random_split(dataset_full, [train_size, test_size,val_size])
            dataset_train = torch.utils.data.Subset(dataset_full, range(train_size))
            dataset_test = torch.utils.data.Subset(dataset_full, range(train_size, train_size + test_size))
            dataset_val = torch.utils.data.Subset(dataset_full, range(train_size + test_size,train_size + test_size + val_size))
            print(f'Full dataset (train+test+val) is split into:\n{len(dataset_train)},{len(dataset_test)},{len(dataset_val)}\n')
        else:
            dataset_train = self.dataset['train']
            # dataset_test = self.get_dataset('test')
            # dataset_val = self.get_dataset('val')

        loader_train = self.get_loader(dataset=dataset_train,batch_size=batch_size,shuffle=True)
        loader_test = self.get_loader(dataset=dataset_test,batch_size=batch_size,shuffle=False)
        loader_val = self.get_loader(dataset=dataset_val,batch_size=batch_size,shuffle=False)

        loaders = {'train':loader_train,
                    'test':loader_test,
                    'val':loader_val}

        return loaders

    # def get_dataset(self,dataset_part):
    #     dataset = SegmentationDataset(self.settings,
    #                                 dataset_part=dataset_part,
    #                                 transform=self.get_transform(dataset_part))
    #     return dataset


    def get_loader(self,dataset,shuffle,batch_size,num_workers=4):
        loader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size,
                                            shuffle=shuffle, 
                                            num_workers=num_workers)      

        return loader

    # def get_transform(self,dataset_part):
    #     transform = Compose([ToTensor(),Normalize(task='segmentation'),AddAxis()])
    #     return transform


    def get_predictions(self,model,loader):
        ### GET MY LOADER
        for i, data in enumerate(loader):
            y_pred_batch = model(data['image'])
            y_true_batch = data['label']
            image_path = data['image_path']

            yield y_pred_batch[0,0].long(), y_true_batch[0,0].long(), image_path[0]

    def plot_images(self,dataset_part):
        ### MODEL
        model=self.utils.get_model()
        model.load_state_dict(torch.load(self.settings['model']['path'],map_location='cpu'))

        ### LOADERS
        loader = self.get_loaders(batch_size=1)[dataset_part]

        prediction_generator = self.get_predictions(model,loader)        

        row, col = 3, 5
        fig_subplots = row*col

        # for i_fig in range(fig_count):
        fig, ax = plt.subplots(row,col)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        # fig.suptitle(f'G.Truth: {y_true_name} Prediction: {y_pred_name}',fontsize=15)
        for ind in range(col):
            y_pred,y_true,img_path = next(prediction_generator)
            # print(img_path)
            img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
            ax[0,ind].imshow(img)
            ax[1,ind].imshow(y_pred*255)
            ax[2,ind].imshow(y_true*255)
        plt.show()


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

