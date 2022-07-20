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

        self.label_patch_folder = settings['patch'][dataset_part]['label_patch_folder']
        self.label_files = os.listdir(self.label_patch_folder)

        self.instance_table = self.get_instance_table()

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
        img_path = label_dict['orthogonal_zoomed_patch']['path']
        img = cv2.cvtColor(cv2.imread(img_path,1),cv2.COLOR_BGR2RGB)

        ### GET LABEL
        label_int = self.instance_table[label_dict['instance_name']]
        if self.hot_encoding:
            label = np.zeros([self.classes])
            label[label_int]=1
        else:
            label = label_int
        sample={'image':img,
                'label':label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_instance_table(self):
        instance_names = ['other', 'ARJ21','Boeing737', 'Boeing747','Boeing777', 'Boeing787', 'A220', 'A321', 'A330', 'A350']
        instance_table = { instance_name:i for i,instance_name in enumerate(instance_names)}
        return instance_table

    


if __name__ == "__main__":
    import random
    # from torchvision.transforms import Compose
    # from transform import Augmentations, ToTensor
    from recognition import Recognition
    from settings import Settings

    patch_size=128
    dataset_part = 'train'
    exp_no = 0

    settings = Settings(hot_encoding=True,exp_no=0,patch_size=patch_size)()
    # recognition_instance = Recognition(dataset_id,dataset_part,dataset_name,patch_size)
    recognition_dataset = RecognitionDataset(settings,dataset_part,transform=Compose([ToTensor(),Normalize()]))
    
    ## CHECK DATASET
    print(len(recognition_dataset))
    ind = random.randint(0,len(recognition_dataset)-1)
    sample = recognition_dataset[ind]
    print(sample['label'])

