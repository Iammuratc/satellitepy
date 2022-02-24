import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import xml.etree.ElementTree as ET
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch

import cv2

class Data(Dataset):
    def __init__(self,dataset_name='train'):
        self.data_folder = "/home/murat/Projects/airplane_detection/DATA/Studenten/Gaofen"
        self.dataset_name = dataset_name
        self.json_file = json.load(open(f"{self.data_folder}/sequences.json",'r'))

        self.items = self.get_items()

    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = cv2.imread(self.items[idx][0],1)
        label = self.items[idx][1]
        sample = {'image':img,'labels':label}
        return sample

    def get_items(self):
        
        items=[]

        img_paths = self.get_img_paths()
        labels = self.get_labels()

        for img_name, img_path in img_paths.items():
            label = labels[img_name]
            items.append([img_path,label])
        return items

    def get_img_paths(self):
        img_paths = {}
        for i in self.json_file:
            image_dicts = i['images']
            for image_dict in image_dicts:
                relative_path = image_dict['relative_path']
                relative_path_split = relative_path.split('/')
                dataset_part = relative_path_split[0]
                img_name = relative_path_split[-1].split('.')[0]
                if dataset_part == self.dataset_name:
                    img_paths[img_name] = f"{self.data_folder}/{relative_path}"
                    
        return img_paths 

    def get_labels(self):
        labels={}

        label_folder = f"{self.data_folder}/{self.dataset_name}/label_xml"
        label_file_paths = [f"{label_folder}/{file_name}" for file_name in os.listdir(label_folder)]


        for label_file_path in label_file_paths: 
            root = ET.parse(label_file_path).getroot()
            file_name = root.findall('./source/filename')[0].text
            img_name = file_name.split('.')[0]
            point_spaces = root.findall('./objects/object/points')        
            points = []

            for point_space in point_spaces:
                my_points = point_space.findall('point')[:4] # remove the last coordinate

                coords = []
                for my_point in my_points:
                    # coord = []
                    for point in my_point.text.split(','):
                        coords.append(float(point))
                    # coords.append(coord)
                points.append(coords)

            labels[img_name] = points
        return labels

if __name__ == "__main__":

    train_data = Data(dataset_name='test')
    # print(len(train_data))
    # print(train_data.get_labels())
    # print(train_data.get_img_paths())
    # print(train_data.items)
    print(train_data[0])
