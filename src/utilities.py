import os
from torchvision.transforms import Compose
import torch

# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
from models.models import * 
from dataset.dataset import DatasetSegmentation 
from transforms import Normalize, ToTensor, AddAxis


class Utilities:
    """docstring for Utilities"""
    def __init__(self, settings):
        # super(Utilities, self).__init__()
        self.settings = settings

            
    # def get_file_paths(self,folder,sort=True):
    #     file_paths = [os.path.join(folder,file) for file in os.listdir(folder)]
    #     if sort:
    #         file_paths.sort()
    #     return file_paths 

    def get_model(self):
        model_name = self.settings['model']['name']
        if model_name == 'UNet':
            model = UNet(init_features=self.settings['model']['init_features'])
        elif model_name == 'Custom_0':
            model = Custom_0()
        else:
            print('Please define your model first.')
            return 0
        return model

    def get_dataset(self,dataset_parts,task):
        if task=='segmentation':
            ### DATASET
            dataset = {dataset_part:DatasetSegmentation(
                                            settings=self.settings,
                                            dataset_part=dataset_part,
                                            transform=Compose([ToTensor(),Normalize(task='segmentation'),AddAxis()])
                                            ) 
                                            for dataset_part in dataset_parts}
            dataset_split = self.split_dataset(dataset)
        return dataset_split

    def split_dataset(self,dataset):
        if self.settings['training']['split_ratio']:
            dataset_train = dataset['train']
            dataset_val = dataset['val']
            ### MERGE DATASETS
            # dataset_full=dataset_train
            dataset_full = torch.utils.data.ConcatDataset([dataset_train,dataset_val])
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
            dataset_train = dataset['train']
            dataset_test = dataset['test']
            dataset_val = dataset['val']

        dataset_split = {'train':dataset_train,
                    'val':dataset_val,
                    'test':dataset_test}
        return dataset_split




class ImageViewer(object):
    def __init__(self, ax, instance_table,image_data):
        self.ax = ax
        self.image_data = image_data
        self.instance_table = list(instance_table.keys())


        self.slices = len(image_data)
        self.ind = 0

        self.im = ax.imshow(self.get_next_img())
        self.update()

    # def onscroll(self, event):
    def on_press(self, event):
        # print("%s %s" % (event.button, event.step))
        # if event.button == 'up':
        if event.key == 'd':
            self.ind = (self.ind + 1) % self.slices
        # else:
        if event.key == 'a':
            self.ind = (self.ind - 1) % self.slices
        print(self.ind)
        self.update()

    def get_next_img(self):
        img = cv2.cvtColor(cv2.imread(self.image_data[self.ind][0]),cv2.COLOR_BGR2RGB)
        return img

    def update(self):

        label = self.instance_table[self.image_data[self.ind][1]]
        predicted = self.instance_table[self.image_data[self.ind][2]]

        self.im.set_data(self.get_next_img())
        # self.ax.set_ylabel('slice %s' % self.ind)
        self.ax.set_title(f"Label: {label}, Predicted: {predicted}")
        self.im.axes.figure.canvas.draw()
