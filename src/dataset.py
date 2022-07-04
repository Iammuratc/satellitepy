import os
from torch.utils.data import Dataset
import torch

import cv2
import json
import numpy as np



class RecognitionDataset(Dataset):
    def __init__(self,recognition_instance,transform=None):
        self.recognition = recognition_instance
        self.transform=transform

        self.label_folder = self.recognition.label_patch_folder
        self.label_files = os.listdir(self.label_folder)

        self.instance_table = { 'other': 0, 
                                'ARJ21': 1,
                                'Boeing737': 2, 
                                'Boeing747': 3,
                                'Boeing777': 4, 
                                'Boeing787': 5, 
                                'A220': 6, 
                                'A321': 7, 
                                'A330': 8, 
                                'A350': 9
                                }

        # self.instance_len = len(self.instance_table.keys())

    def __len__(self):
        return len(self.label_files) 

    def __getitem__(self, ind):
        if torch.is_tensor(ind):
            ind = ind.tolist()

        ### READ LABEL FILE
        label_file_name = self.label_files[ind]
        label_dict = json.load(open(f"{self.label_folder}/{label_file_name}",'r'))

        ### GET IMAGE
        img_path = label_dict['orthogonal_zoomed_patch']['path']
        img = cv2.cvtColor(cv2.imread(img_path,1),cv2.COLOR_BGR2RGB)

        ### GET LABEL
        label = self.instance_table[label_dict['instance_name']]

        sample={'image':img,
                'label':label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    


if __name__ == "__main__":
    import random
    # from torchvision.transforms import Compose
    # from transform import Augmentations, ToTensor
    from recognition import Recognition

    # airplane_dataset = AirplaneDataset(dataset_name='train',patch_size=512,transform=Compose([Augmentations()]))#,ToTensor()]))
    # sample = airplane_dataset[ind]
    # show_sample(sample)
    # print(ind)
    # print(sample['image'].shape)

    patch_size=128
    dataset_id = 'f73e8f1f-f23f-4dca-8090-a40c4e1c260e'
    dataset_name = 'Gaofen'
    dataset_part = 'train'

    recognition_instance = Recognition(dataset_id,dataset_part,dataset_name,patch_size)
    recognition_dataset = RecognitionDataset(recognition_instance)
    
    ind = random.randint(0,len(recognition_dataset)-1)
    sample = recognition_dataset[ind]
    print(sample['label'])