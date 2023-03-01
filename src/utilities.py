import os
from torchvision.transforms import Compose
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import traceback
from src.models.models import *
# from src.data.dataset.dataset import DatasetSegmentation, DatasetRecognition
from src.transforms import Normalize, ToTensor, AddAxis
from src.data.cutout.cutout import Cutout


def resize_cutouts_by_padding(image_folder,save_folder,patch_size):
    
    for image_name in os.listdir(image_folder):
        print(image_name)
        image_path = os.path.join(image_folder,image_name)
        img = cv2.imread(image_path)
        # my_patch = np.zeros(shape=(patch_size,patch_size,3))
        pad_size_height = int(patch_size/2-img.shape[1]/2)
        if pad_size_height < 0:
            pad_size_height = 0
        pad_size_width = int(patch_size/2-img.shape[0]/2)
        if pad_size_width < 0:
            pad_size_width = 0
        img_padded = np.pad(img,((pad_size_height,pad_size_height),(pad_size_width,pad_size_width),(0,0)))
        # plt.imshow(img_padded)
        # plt.show()
        cv2.imwrite(os.path.join(save_folder,image_name),img_padded)

def count_airplanes_in_patches(folder):
    file_names = os.listdir(folder)

    patch_origins = []

    for file_name in file_names:
        # file_path = os.path.join(folder,file_name)
        # print(file_name)
        patch_origin = file_name.split('_')[0]
        patch_origins.append(patch_origin)
        # break
    print(len(set(patch_origins)))

# def count_instances(label_folder):
    

def convert_my_labels_to_imagenet(dataset_settings):
    ''' Convert JSON labels to imagenet type annotation file

    mmclassification needs imagenet type annotation files
    e.g.
    image_path class_id
    This function takes my json file labels and create an imagenet type annotation file
    '''
    # print(dataset_settings)
    instance_names = list(dataset_settings['instance_names'].keys())
    for dataset_part in dataset_settings['dataset_parts']:
        label_folder = dataset_settings['cutout'][dataset_part]['label_folder']
        imagenet_label_file_path = os.path.join(dataset_settings['cutout'][dataset_part]['root_folder'],'imagenet_labels.txt')
        print(f'imagenet annotation file will be saved at {imagenet_label_file_path}')
        imagenet_label_file = open(imagenet_label_file_path,'a+')

        try:
            for file_name in os.listdir(label_folder):
                label_file_path = os.path.join(label_folder,file_name)
                with open(label_file_path,'r') as f:
                    # print(label_file_path)
                    label_file = json.load(f)

                img_path = label_file['original_cutout']['img_path']
                # instance_id = label_file['instance']['id']
                instance_id = instance_names.index(label_file['instance']['name'])

                imagenet_line = f'{img_path} {instance_id}\n'
                print(imagenet_line)
                imagenet_label_file.write(imagenet_line)

        except Exception:
            traceback.print_exc()
        finally:
            imagenet_label_file.close()

def write_cutouts(dataset_settings,multi_process):
    for dataset_part in dataset_settings['dataset_parts']:
        my_cutout = Cutout(dataset_settings,dataset_part)
        # my_cutout.get_cutouts(save=True,plot=False,indices='all',multi_process=multi_process) # 12,13
        my_cutout.show_original_image(ind=2)

class Utilities:
    """docstring for Utilities"""

    def __init__(self, settings):
        # super(Utilities, self).__init__()
        self.settings = settings

if __name__ == '__main__':
    from src.settings.utils import get_project_folder
    project_folder = get_project_folder() 
    # data_base_folder = os.path.join(project_folder,'data','DOTA','val')
    # original_image_count = len(os.listdir(os.path.join(data_base_folder,'images')))
    # print(original_image_count)
    # instance_count = len(os.listdir(os.path.join(data_base_folder,'cutouts','images')))
    # print(instance_count)

    ### PAD CUTOUTS
    image_folder = os.path.join(project_folder,'data','fair1m','val','cutouts','orthogonal_images_unet')
    save_folder = os.path.join(project_folder,'data','fair1m','val','cutouts','orthogonal_images_unet_padded')
    os.makedirs(save_folder,exist_ok=True)
    resize_cutouts_by_padding(image_folder,save_folder,patch_size=128)
    
    # count_airplanes_in_patches(os.path.join(data_base_folder,'patches','patches_512','images'))