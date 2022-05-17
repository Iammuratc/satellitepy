import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
import os
import json
import math
import geometry
# from utilities import show_sample
import matplotlib.pyplot as plt


## TODO: Add patch size to json file

##TODO (Recognition): save images and labels

##NOTES: y axis of matplotlib figures are inverted, so the airplanes will be actually facing downwards, pay attention at the new datasets 
class Data:
    def __init__(self,dataset_name):
        self.project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_folder = f"{self.project_folder}/DATA/Gaofen"
        self.dataset_name = dataset_name
        self.json_file = json.load(open(f"{self.data_folder}/sequences.json",'r'))


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

    def get_label(self,label_path):
        label = {'bbox':[],'names':[]}
        root = ET.parse(label_path).getroot()

        ### IMAGE NAME
        file_name = root.findall('./source/filename')[0].text
        # img_name = file_name.split('.')[0]
         
        ### INSTANCE NAMES
        instance_names = root.findall('./objects/object/possibleresult/name')#[0].text
        for instance_name in instance_names:
            label['names'].append(instance_name.text)
        
        ### BBOX CCORDINATES
        point_spaces = root.findall('./objects/object/points')        
        for point_space in point_spaces:
            my_points = point_space.findall('point')[:4] # remove the last coordinate
            coords = []
            for my_point in my_points:
                #### [[[x1,y1],[x2,y2]],[[x1,y1]]]
                coord = []
                for point in my_point.text.split(','):
                    coord.append(float(point))
                coords.append(coord)
            label['bbox'].append(coords)
        return label#, img_name

    def get_label_paths(self):
        label_folder = f"{self.data_folder}/{self.dataset_name}/label_xml"
        label_paths = {file_name.split('.')[0]:f"{label_folder}/{file_name}" for file_name in os.listdir(label_folder)}
        return label_paths

    def plot_bboxes(self,img_path,label_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        label = self.get_label(label_path)

        fig, ax = plt.subplots(1)
        ax.imshow(img)
        for bbox in label['bbox']:
            # print(bbox)
            bbox=np.array(bbox)
            rect = geometry.Rectangle(bbox)
            rect.plot_bbox(ax=ax,bbox=bbox,c='b')
        plt.show()

    # def get_labels(self):
    #     labels={}

    #     label_paths = self.get_label_paths()
    #     for label_path in label_paths:
    #         label, img_name =self.get_label(label_path) 
    #         labels[img_name]=label
    #         # break
    #     return labels

    # def plot_rotated_box(self,ax,**kwargs):

    #     if 'bbox_params' in kwargs.keys():
    #         bbox_params = kwargs['bbox_params']
    #         geometry.Rectangle.plot_contours(params=bbox_params)

    #     elif 'bbox_corners' in kwargs.keys():
    #         corners = kwargs['bbox_corners']
    #         geometry.Rectangle.plot_corners(corners)
    #     else:
    #         print('Check the plot_rotated_box')
    #     return ax







    ### CHECK LABELS 
    # patch_size=512
    # train_data = Data(dataset_name='train')
    # val_data = Data(dataset_name='val')
    # test_data = Data(dataset_name='test')
    # labels = train_data.labels
    # img_paths = train_data.img_paths

    # file_name = random.choice(list(labels.keys())) # interesting images from train dataset: 360, 837, 512
    # # file_name = '837'
    # img_path = img_paths[file_name]
    # label = labels[file_name]
    # print(img_path)


    # img = cv2.imread(img_path)


    # fig, ax = plt.subplots(1)
    # ax.imshow(img)
    # ax = plt.gca()
    # # ax.set_xlim([0, 512])
    # # ax.set_ylim([0, 512])
    # for i,corners in enumerate(label['bbox']):
    #     corners=np.array(corners)
    #     instance_name = label["names"][i]
    #     rotated_rect = geometry.RotatedRect(corners=corners,parametized=False)
    #     print(f"Plane no: {i} Type: {instance_name} height: {rotated_rect.h} width: {rotated_rect.w}")

    #     # print(corners[0,0])
    #     plt.text(corners[0,0], corners[0,1], i, color='blue')#,bbox=dict(fill=False, edgecolor='blue', linewidth=0.5))
    #     center = np.mean(corners,axis=0)
    #     # rotated_rect.plot_contours(ax)
    #     plot_rotated_box(corners,ax)
    # plt.show()

    ### PATCH CONTROL
    # patch_coords = test_data.get_patch_start_coords(my_max=[1030,1030],patch_size=512,overlap=100)
    # print(patch_coords)
    # print(len(train_data))
    # print(train_data.get_labels())
    # print(train_data.get_img_paths())
    # print(train_data.items)
    # print(train_data[5]['labels'])
    # ind = random.randint(0,len(train_data)-1)
    # train_data.img_show(ind=ind,plot_bbox=True)