import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import xml.etree.ElementTree as ET
import os

class Data:
    def __init__(self):
        self.data_folder = "/home/inf4/Projects/aircraft_detection/Studenten"
        json_file = open(f"{self.data_folder}/Gaofen/gaofen_sequences.json",'r')

        self.training_json = json.load(json_file)

        # data folders
        self.train_data_folder = f"{self.data_folder}/Gaofen/train"
        self.val_data_folder = f"{self.data_folder}/Gaofen/val"


    def get_paths(self):

        train_img_paths = {}
        val_img_paths = {}
        for i in self.training_json:
            image_dicts = i['images']
            for image_dict in image_dicts:
                relative_path = image_dict['relative_path']
                relative_path_split = relative_path.split('/')
                dataset_part = relative_path_split[0]
                img_name = relative_path_split[-1].split('.')[0]
                if dataset_part == 'train':
                    train_img_paths[img_name] = f"{self.train_data_folder}/{relative_path}"
                elif dataset_part == 'val':
                    val_img_paths[img_name] = f"{self.val_data_folder}/{relative_path}"
        return train_img_paths, val_img_paths

    def get_labels(self):
        train_label_folder = f"{self.train_data_folder}/label_xml"
        label_file_paths = [f"{train_label_folder}/{file_name}" for file_name in os.listdir(train_label_folder)]


        root = ET.parse(label_file_paths[0]).getroot()
        filename = root.findall('./source/filename')[0].text

        point_spaces = root.findall('./objects/object/points')
        
        points = []
        for point_space in point_spaces:
            my_points = point_space.findall('point')[:4]

            coords = []
            for my_point in my_points:
                coord = []
                for point in my_point.text.split(','):
                    coord.append(float(point))
                coords.append(coord)
            points.append(coords)

        return points
if __name__ == "__main__":

    data = Data()
    # train_img_paths, val_img_paths = data.get_paths()
    # print(train_img_paths)
    data.get_labels()
