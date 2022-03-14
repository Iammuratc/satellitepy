import os
from torch.utils.data import Dataset
import torch

import cv2
import json
import numpy as np

import matplotlib.pyplot as plt


from utilities import show_sample
from geometry import RotatedRect


class AirplaneDataset(Dataset):
    def __init__(self,dataset_name,patch_size,transform=None):
        self.project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_folder = f"{self.project_folder}/DATA/Gaofen"
        # self.dataset_name = dataset_name
        self.patch_size=patch_size
        self.transform=transform

        self.img_folder = f"{self.data_folder}/{dataset_name}/images_{patch_size}"
        self.label_folder = f"{self.data_folder}/{dataset_name}/labels_{patch_size}"

        self.img_names = self.get_img_names()

    def __len__(self):
        return len(self.img_names) 

    def __getitem__(self, ind):
        if torch.is_tensor(ind):
            ind = ind.tolist()

        img_name = self.img_names[ind]

        img_path = f"{self.img_folder}/{img_name}"
        label_path = f"{self.label_folder}/{self.remove_ext(img_name)}.json"
        
        img = cv2.cvtColor(cv2.imread(img_path,1),cv2.COLOR_BGR2RGB)
        label = json.load(open(label_path,'r'))
        label['patch_size']=self.patch_size
        
        sample={}
        sample['image']=img
        sample.update(label)
        # print(sample)

        if self.transform:
            sample = self.transform(sample)

        return {'image':sample['image'],'orthogonal_bboxes':sample['orthogonal_bboxes']}

    def get_img_names(self):
        return os.listdir(self.img_folder)


    @staticmethod
    def remove_ext(file_name):
        return file_name.split('.')[0]


if __name__ == "__main__":
    import random
    from torchvision.transforms import Compose
    from transform import Augmentations, ToTensor
    airplane_dataset = AirplaneDataset(dataset_name='train',patch_size=512,transform=Compose([Augmentations()]))#,ToTensor()]))
    ind = random.randint(0,len(airplane_dataset)-1)
    sample = airplane_dataset[ind]
    show_sample(sample)
    print(ind)
    # print(sample['image'].shape)
    
