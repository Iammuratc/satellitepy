import os
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import json

import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
# from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage

class DataGenerator(Dataset):

    def __init__(self,dataset_name,patch_size):
        self.project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_folder = f"{self.project_folder}/DATA/Gaofen"
        # self.dataset_name = dataset_name

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

    @staticmethod
    def remove_ext(file_name):
        return file_name.split('.')[0]

    def get_paths(self):
        paths=[]

        for img_name in os.listdir(self.img_folder):
            json_path = f"{self.label_folder}/{self.remove_ext(img_name)}.json"
            img_path = f"{self.img_folder}/{img_name}"
            paths.append([img_path,json_path])
        return paths


    def transform(self,sample):
        image=sample['image']
        # seq = iaa.Affine(translate_px={"x": 120})

        seq = iaa.Sequential([
            iaa.Multiply((0.5, 1.5)), # change brightness, doesn't affect keypoints
            iaa.Affine(
                # rotate=10,
                scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                shear=(-16, 16),

            ) 
        ])
        if sample['labels']['airplane_exist']:
            bboxes = sample['labels']['rotated_bboxes']

            # key_points = []

            kps = KeypointsOnImage([Keypoint(x=coord[0], y=coord[1]) for coords in bboxes for coord in coords], shape=image.shape)

            image_aug, kps_aug = seq(image=image, keypoints=kps)
            # image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
        # else:
            # image_aug = seq(image=image, keypoints=bbs)
            image_before = kps.draw_on_image(image, size=7)
            image_after = kps_aug.draw_on_image(image_aug, size=7)
            fig,ax=plt.subplots(1,2)
            ax[0].imshow(image_before)
            ax[1].imshow(image_after)
            plt.show()
    def img_show(self,ind,plot_bbox=True):
        sample = self.__getitem__(ind)
        img = sample['image']
        labels = sample['labels']
        print(labels)
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
    data_generator = DataGenerator(dataset_name='train',patch_size=512)
    ind = random.randint(0,len(data_generator)-1)
    # print(data_generator[1284])
    data_generator.transform(data_generator[ind])