import os
from torch.utils.data import Dataset
import torch

import cv2
import json
import numpy as np



class RecognitionDataset(Dataset):
    def __init__(self,settings,dataset_part,transform=None):
        # self.recognition = recognition_instance
        self.transform=transform
        self.hot_encoding=settings['training']['hot_encoding']
        self.patch_config = settings['training']['patch_config']

        self.label_patch_folder = settings['patch'][dataset_part]['label_patch_folder']
        self.label_files = os.listdir(self.label_patch_folder)

        self.instance_table = settings['dataset']['instance_table']

        self.classes = len(self.instance_table.keys())
    
    def __len__(self):
        return len(self.label_files) 

    def __getitem__(self, ind):
        if torch.is_tensor(ind):
            ind = ind.tolist()

        ### READ LABEL FILE
        label_file_name = self.label_files[ind]
        label_dict = json.load(open(f"{self.label_patch_folder}/{label_file_name}",'r'))

        ### GET IMAGE
        img_path = label_dict[self.patch_config]['path']
        img = cv2.cvtColor(cv2.imread(img_path,1),cv2.COLOR_BGR2RGB)
        ### GET LABEL
        label_int = self.instance_table[label_dict['instance_name']]
        if self.hot_encoding:
            label = np.zeros([self.classes])
            label[label_int]=1
        else:
            label = label_int
        sample={'image_path':img_path,
                'image':img,
                'label':label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    


if __name__ == "__main__":
    import random
    from torchvision.transforms import Compose
    from transforms import Normalize, ToTensor
    from recognition import Recognition
    from settings import Settings

    patch_size=128
    dataset_part = 'test'
    exp_no='temp'
    patch_config = 'orthogonal_zoomed_patch'
    split_ratio=[0.8,0.1,0.1]

    settings = Settings(hot_encoding=True,
                        exp_no=exp_no,
                        patch_size=patch_size,
                        patch_config=patch_config,
                        update=True,
                        split_ratio=split_ratio)()
    # print(settings)
    # recognition_instance = Recognition(dataset_id,dataset_part,dataset_name,patch_size)
    recognition_dataset = RecognitionDataset(settings,dataset_part,transform=Compose([ToTensor(),Normalize()]))
    
    ## CHECK DATASET
    # print(len(recognition_dataset))
    for ind in range(10):
        # ind = 0#random.randint(0,len(recognition_dataset)-1)
        sample = recognition_dataset[ind]
        # print(sample['label'])
        print(sample['image_path'])

