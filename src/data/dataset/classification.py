import os
from torch.utils.data import Dataset
import torch

import cv2
import json
import numpy as np
from src.transforms import Normalize, ToTensor, AddAxis, HorizontalFlip


class DatasetClassification(Dataset):
    def __init__(self, exp_settings, data_settings, dataset_part, transform=None, **kwargs):
        # self.recognition = recognition_instance
        self.transform = transform
        self.hot_encoding = exp_settings['training']['hot_encoding']
        self.cutout_config = exp_settings['cutout']['config'] 
        self.cutout_size = exp_settings['cutout']['size']

        if 'image_folder' not in kwargs:
            if exp_settings['cutout']['config']!='original':
                self.image_folder = data_settings['cutout'][dataset_part][f'{self.cutout_config}_image_folder']
            else:
                self.image_folder = data_settings['cutout'][dataset_part]['image_folder']
        else:
            self.image_folder = kwargs['image_folder']

        if 'label_folder' not in kwargs:
            self.label_folder = data_settings['cutout'][dataset_part]['label_folder']
        else:
            self.label_folder = kwargs['label_folder']
        self.image_names = [os.path.splitext(image_file_name)[0] for image_file_name in os.listdir(self.image_folder)]

        self.instance_names = data_settings['instance_names']
        self.classes = len(self.instance_names.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, ind):
        if torch.is_tensor(ind):
            ind = ind.tolist()

        # READ LABEL FILE
        # GET LABEL
        image_name = self.image_names[ind]
        with open(os.path.join(self.label_folder,f'{image_name}.json'),'r') as f:
            label_dict = json.load(f)
        # print(label_dict)
        label_int = int(self.instance_names[label_dict['instance']['name']])

        # GET IMAGE
        img_path = os.path.join(self.image_folder,f'{image_name}.png')
        img = self.read_image(img_path)
        if self.hot_encoding:
            label = np.zeros([self.classes])
            label[label_int] = 1
        else:
            label = np.array(label_int)
        sample = {'image_path': img_path,
                  'image': img,
                  'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def read_image(self,img_path):
        img = cv2.cvtColor(cv2.imread(img_path, 1), cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img,
            dsize=(
                self.cutout_size,
                self.cutout_size),
            interpolation=cv2.INTER_LINEAR)
        return img