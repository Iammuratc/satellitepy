import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt

import cv2
import math


class Data():
    def __init__(self,dataset_name='train'):
        self.data_folder = "../DATA/Gaofen"
        self.dataset_name = dataset_name
        self.json_file = json.load(open(f"{self.data_folder}/sequences.json",'r'))

        img_paths = self.get_img_paths()
        labels = self.get_labels()
        patches = self.save_patches(img_paths=img_paths,labels=labels,patch_size=512)


    def save_patches(self,img_paths,labels,patch_size=512):

        patches = []
        box_corner_threshold = 2
        for img_name, img_path in img_paths.items():
            img = cv2.cvtColor(cv2.imread(img_path,1),cv2.COLOR_BGR2RGB)
            bbox_coords = labels[img_name]
            y_max, x_max, ch = img.shape[:3]
            patch_y = int(math.ceil(y_max/patch_size))
            patch_x = int(math.ceil(x_max/patch_size))

            for p_y in range(patch_y):
                for p_x in range(patch_x):
                    my_labels = []
                    patch = np.zeros(shape=(patch_size,patch_size,ch),dtype=np.uint8)
                    y_0 = p_y*patch_size
                    x_0 = p_x*patch_size

                    plot_img = False
                    for coords in bbox_coords:
                        box_corner_in_patch = 0
                        for coord in coords:
                            if (x_0<=coord[0]<=x_0+patch_size) and (y_0<=coord[1]<=y_0+patch_size):
                                box_corner_in_patch += 1    
                        if box_corner_in_patch>=box_corner_threshold:                        
                            shifted_coords = np.array(coords)-[x_0,y_0]
                            my_labels.append(shifted_coords)
                            plot_img=True
                    
                    y_limit_expanded = y_0+patch_size>=y_max
                    x_limit_expanded = x_0+patch_size>=x_max
                    limit_expanded = y_limit_expanded and x_limit_expanded
                    if limit_expanded:
                        # print(y_max,y_0)
                        # print(x_max,x_0)
                        patch[:y_max-y_0,:x_max-x_0] = img[y_0:y_max,x_0:x_max]
                    elif y_limit_expanded:
                        patch[:y_max-y_0,:] = img[y_0:y_max,x_0:x_0+patch_size]
                    elif x_limit_expanded:
                        patch[:,:x_max-x_0] = img[y_0:y_0+patch_size,x_0:x_max]                        
                    else:
                        patch[:,:] = img[y_0:y_0+patch_size,x_0:x_0+patch_size]

                    # patches.append(patch)
                    # print(patch)
                    # print(my_labels)
                    if plot_img:
                        self.img_show(img=patch,labels=my_labels)
            # break
        return patches

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
                    #### [[[x1,y1],[x2,y2]],[[x1,y1]]]
                    coord = []
                    for point in my_point.text.split(','):
                        coord.append(float(point))
                    coords.append(coord)
                    # for point in my_point.text.split(','):
                    #     coords.append(float(point))
                points.append(coords)

            labels[img_name] = points
        return labels

    def img_show(self,img,labels,plot_bbox=True):
        # sample = self.__getitem__(ind)
        # img = sample['image']
        # labels = sample['labels']
        # print(labels)
        fig, ax = plt.subplots(1)
        ax.imshow(img,'gray')

        if plot_bbox==True:
            for coords in labels:
                for i, coord in enumerate(coords):
                    # PLOT BBOX
                    ax.plot([coords[i-1][0],coord[0]],[coords[i-1][1],coord[1]],c='r')
                    # PLOT CORNERS
                    # ax.scatter(coord[0],coord[1],c='r',s=5)
        plt.show()


if __name__ == "__main__":
    import random

    train_data = Data(dataset_name='test')
    # print(len(train_data))
    # print(train_data.get_labels())
    # print(train_data.get_img_paths())
    # print(train_data.items)
    # print(train_data[5]['labels'])
    # ind = random.randint(0,len(train_data)-1)
    # train_data.img_show(ind=ind,plot_bbox=True)