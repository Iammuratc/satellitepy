import os
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import json
import numpy as np

import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
# from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage

from geometry import RotatedRect

# TODO: remove the planes outside of the image after augmentation, get the new image corners 
class DataGenerator(Dataset):

    def __init__(self,dataset_name,patch_size):
        self.project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_folder = f"{self.project_folder}/DATA/Gaofen"
        # self.dataset_name = dataset_name
        self.patch_size=patch_size

        self.img_folder = f"{self.data_folder}/{dataset_name}/images_{patch_size}"
        self.label_folder = f"{self.data_folder}/{dataset_name}/labels_{patch_size}"

        self.paths = self.get_paths()

    def __len__(self):
        return len(self.paths) 

    def __getitem__(self, ind):
        if torch.is_tensor(ind):
            ind = ind.tolist()

        img = cv2.cvtColor(cv2.imread(self.paths[ind][0],1),cv2.COLOR_BGR2RGB)
        label = json.load(open(self.paths[ind][1],'r'))
        sample = {'image':img,'labels':label}
        return sample


    def get_paths(self):
        ## GET IMAGE AND LABEL PATHS
        paths=[]

        for img_name in os.listdir(self.img_folder):
            json_path = f"{self.label_folder}/{self.remove_ext(img_name)}.json"
            img_path = f"{self.img_folder}/{img_name}"
            paths.append([img_path,json_path])
        return paths


    def transform_sample(self,sample):
        # APPLY AUGMENTATIONS AND ADJUST BBOXES ACCORDINGLY
        sample_aug = sample.copy()

        image=sample['image']
        bbox_aug = []

        seq = iaa.Sequential([
            iaa.Multiply((0.5, 1.5)), # change brightness, doesn't affect keypoints
            iaa.Affine(
                rotate=(-10,10),
                scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                shear=(-16, 16),
            ) 
        ])

        if sample['labels']['airplane_exist']: # If there is any airplane in the image, augment bboxes, else, augment image only 
            bboxes = sample['labels']['rotated_bboxes']
            # bboxes.insert(0,)
            # print(bboxes)

            kps = KeypointsOnImage([Keypoint(x=coord[0], y=coord[1]) for coords in bboxes for coord in coords], shape=image.shape)

            image_aug, kps_aug = seq(image=image, keypoints=kps)

            bbox_aug = np.array([keypoint.xy for keypoint in kps_aug]).reshape(-1,4,2) # eg: 3,4,2
        
            ## REMOVE AIRPLANES OUT OF IMAGE
            # self.remove_
            # bbox_aug_flatten = bbox_aug.reshape(bbox_aug.shape[0],-1)
            # print(bbox_aug_flatten.shape)
            # for i in bbox_aug_flatten:
            #     if not 0<all(bbox_aug_flatten[i])<self.patch_size:
            sample_aug['labels']['rotated_bboxes']=bbox_aug
        else:
            image_aug = seq(image=image)

        sample_aug['image']=image_aug
        ### SHOW IMAGE
        ax = data_generator.img_show(sample_aug,plot_bbox=True)

        for corners in bbox_aug:
            print(corners)       
            my_rect = RotatedRect(parametized=False,corners=corners)
            print(my_rect.angle)
            ax = my_rect.plot_contours(ax)
        plt.show()
        return sample_aug

    def img_show(self,sample,plot_bbox=True):
        # SHOW IMAGE AND BBOXES
        img = sample['image']
        bboxes = sample['labels']['rotated_bboxes']

        fig, ax = plt.subplots(1)
        ax.imshow(img,'gray')

        if plot_bbox==True:
            for coords in bboxes:
                for i, coord in enumerate(coords):
                    # PLOT BBOX
                    ax.plot([coords[i-1][0],coord[0]],[coords[i-1][1],coord[1]],c='r')
        # plt.show()
        return ax

    @staticmethod
    def remove_ext(file_name):
        return file_name.split('.')[0]


if __name__ == "__main__":
    import random
    data_generator = DataGenerator(dataset_name='val',patch_size=512)
    # ind = random.randint(0,len(data_generator)-1)
    # USE train-378 for your augmentation tests
    sample = data_generator[7]
    # print(ind)
    # print(sample)
    sample = data_generator.transform_sample(sample)
    
