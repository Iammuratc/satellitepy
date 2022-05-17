import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

import geometry
from utilities import get_file_name_from_path
from data import Data


#### RECOGNITION DATA
class RecognitionData(Data):
    def __init__(self,dataset_name):
        '''
        Save patches for the recognition task
        A patch is consisted of a single airplane. The airplane is located in the middle of the patch by using the center of its rotated bounding box.
        Rotated bounding box can be masked (i.e. masked=True) 
        '''
        super(RecognitionData, self).__init__(dataset_name)

    def set_patch_folders(self,patch_size):
        ### PATCH SAVE DIRECTORIES
        self.patch_folder_base = f"{self.data_folder}/{self.dataset_name}/patches_{patch_size}_recognition"
        os.makedirs(self.patch_folder_base,exist_ok=True)
        self.img_patch_folder = f"{self.patch_folder_base}/images"
        os.makedirs(self.img_patch_folder,exist_ok=True)
        self.img_patch_rotated_folder = f"{self.patch_folder_base}/rotated_images"
        os.makedirs(self.img_patch_rotated_folder,exist_ok=True)
        self.label_patch_folder = f"{self.patch_folder_base}/labels"
        os.makedirs(self.label_patch_folder,exist_ok=True)
        # return img_patch_folder, label_patch_folder


    def get_img_patch(self,img,rect,patch_dict):
        '''
        Get the patch of the upwards facing airplane
        Original padded image (img) >> 2*patch_size image (img_1) >> rotate such that the airplane is facing upwards (img_2) >> patch_size image
        img: original img
        ''' 

        patch_size = patch_dict['patch_size']
        cx, cy = patch_dict['original']['center_padded']

        # Get the large image
        img_1=img[cy-patch_size:cy+patch_size,cx-patch_size:cx+patch_size,:]
       

        # Get the rotation angle
        angle = rect.get_atan2()
        M = cv2.getRotationMatrix2D((img_1.shape[0]/2, img_1.shape[1]/2), np.rad2deg(angle), 1.0)

        img_2 = cv2.warpAffine(img_1, M, (patch_size*2, patch_size*2))

        # Get the image patch
        patch_half_size = int(patch_size/2)
        patch_dict['img_patch_rotated']=img_2[patch_size-patch_half_size:patch_size+patch_half_size,patch_size-patch_half_size:patch_size+patch_half_size,:]
        patch_dict['img_patch']=img_1[patch_size-patch_half_size:patch_size+patch_half_size,patch_size-patch_half_size:patch_size+patch_half_size,:]

        return patch_dict


    def get_patch_label(self,img,patch_dict,bbox):
        # print(bbox)
        pad_size = patch_dict['original']['pad_size']
        patch_size=patch_dict['patch_size']
        # If not including _orig, the variable belongs to the patch 
        bbox_orig_padded =np.array(bbox)+pad_size # add initial padding
        # print(pad_size)
        # print(bbox_orig_padded)
        ### NEW CENTER OF AIRPLANE
        center = np.mean(bbox,axis=0).astype(int)
        center_padded = center+pad_size
        # print('center padded: ', center_padded)
        ### NEW BBOX
        bbox_patch = bbox_orig_padded-center_padded+pad_size/2
        # print(bbox_orig_padded-center_padded)
        # print(bbox_patch)
        patch_dict['rotated_bbox_patch']=bbox_patch.tolist()
        patch_dict['original']['center_padded']=center_padded.tolist()

        rect = geometry.Rectangle(bbox=bbox_patch)


        patch_dict['bbox_params']= [int(patch_size/2),int(patch_size/2),rect.h,rect.w,rect.angle]
        patch_dict['notes']['bbox_params'] = ['center_x,center_y,height,width,rotation_angle']

        patch_dict = self.get_img_patch(img=img,rect=rect,patch_dict=patch_dict)
        return patch_dict, rect


    def get_patches_in_file(self,img_path,label_path,patch_size,save=False,plot=False):


        
        ### GET ORIGINAL IMAGE
        img = cv2.imread(img_path)
        label = self.get_label(label_path)
        ### pad the original image, so no patching problem for the planes on the edge of the image
        pad_size = patch_size
        img = np.pad(img,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'constant',constant_values=0)#'symmetric')#
        ### GET LABELS
        bboxes = label['bbox']

        patch_name = os.path.split(img_path)
        # print(labels)
        for i,bbox in enumerate(bboxes):
            
            patch_dict = {  'img_patch':np.zeros(shape=(patch_size,patch_size,3),dtype=np.uint8),
                            'img_patch_rotated':np.zeros(shape=(patch_size,patch_size,3),dtype=np.uint8),
                            'instance_name':None,
                            'patch_path':None,
                            'rotated_bbox_patch':[],
                            'bbox_params':[],
                            'patch_size':None,
                            'notes':{'bbox_params':None},
                            'original': {'img_path':None, 'center_padded':None, 'pad_size':0}}
            
            patch_dict['original']['img_path']=img_path
            patch_dict['instance_name']=label["names"][i]
            patch_dict['original']['pad_size']=pad_size
            patch_dict['patch_size']=patch_size

            patch_dict, rect = self.get_patch_label(img,patch_dict,bbox)

            # print(patch_dict['rotated_bbox_patch'])

            if plot:
                self.plot_patches(patch_dict,rect)

            
            if save:
                # 'patch_path':None,
                self.save_patches(patch_dict,i)
            # break
    def save_patches(self,patch_dict,i):

        ## FILE NAMES
        img_path = patch_dict['original']['img_path']
        file_name = get_file_name_from_path(img_path)
        patch_name = f"{file_name}_{i}"

        ### SAVE FIGURES
        # os.makedirs(f"{self.patch_folder_base}/figures",exist_ok=True)
        # plt.savefig(f"{self.patch_folder_base}/figures/{patch_name}.png", bbox_inches='tight')

        ## ROTATED IMG
        img_patch_rotated = patch_dict['img_patch_rotated']
        cv2.imwrite(f"{self.img_patch_rotated_folder}/{patch_name}.png",img_patch_rotated)
        
        ## IMG
        img_patch = patch_dict['img_patch']
        cv2.imwrite(f"{self.img_patch_folder}/{patch_name}.png",img_patch)

        ### LABEL
        del patch_dict['img_patch_rotated']
        del patch_dict['img_patch']
        with open(f"{self.label_patch_folder}/{patch_name}.json", 'w') as f:
            json.dump(patch_dict, f,indent=4)

        # return patch_dict
    def plot_patches(self,patch_dict,rect):
        ### PLOT
        fig,ax = plt.subplots(2)
        ax[0].imshow(cv2.cvtColor(patch_dict['img_patch'],cv2.COLOR_BGR2RGB))
        # # ax[0].set_ylim(ax[0].get_ylim()[::-1]) # invert y axis
        # # self.plot_rotated_box(patch_dict['patch']['rotated_bboxes'],ax[0])
        # rect.plot_corners(ax[0])
        rect.plot_bbox(bbox=rect.bbox,ax=ax[0],c='b')

        ax[1].imshow(cv2.cvtColor(patch_dict['img_patch_rotated'],cv2.COLOR_BGR2RGB))
        # # ax[1].set_ylim(ax[1].get_ylim()[::-1])
        # rect.plot_contours(ax[1],rotate=False)
        rect.plot_bbox(bbox=rect.get_orthogonal_bbox(),ax=ax[1],c='b')
        plt.show()


class RecognitionAnalysis(RecognitionData):
    def __init__(self,dataset_name):
        '''
        Analyse the recognition data
        '''
        super(RecognitionData, self).__init__(dataset_name)


    def get_airplane_size(self,patch_size):
        self.set_patch_folders(patch_size)

        json_files = os.listdir(self.label_patch_folder)
        print(f"Airplane size will be calculated for {len(json_files)} planes")
        size_dict = {}

        for json_file in json_files:
           patch_dict = json.load(open(f"{self.label_patch_folder}/{json_file}",'r'))
           cx,cy,h,w,angle = patch_dict['bbox_params']
           instance_name=patch_dict['instance_name']

           
           if instance_name not in size_dict.keys():
                size_dict[instance_name] = {'w':[],'h':[]}
           else:
                size_dict[instance_name]['w'].append(w)
                size_dict[instance_name]['h'].append(h)
        # print(size_dict)

        for instance_name in size_dict.keys():
            total_no = len(size_dict[instance_name]['h'])
            fig,ax=plt.subplots(2)
            fig.suptitle(f'Instance:{instance_name}, total no: {total_no}')
            ax[0].set_title('Height')
            ax[1].set_title('Width')
            ax[0].hist(size_dict[instance_name]['h'],bins=50)
            ax[1].hist(size_dict[instance_name]['w'],bins=50)
            plt.show()

    def get_wrong_labels(self,patch_size):
        self.set_patch_folders(patch_size)
        json_files = os.listdir(self.label_patch_folder)
        print(f"Airplane size will be calculated for {len(json_files)} planes")
        size_dict = {}

        for json_file in json_files:
           patch_dict = json.load(open(f"{self.label_patch_folder}/{json_file}",'r'))
           img = json.load(open(f"{self.img_patch_folder}/{json_file}",'r'))



if __name__ == "__main__":
    import random
    
    import matplotlib.pyplot as plt    
    import os

    ### SAVE PATCHES (RECOGNITION)
    patch_size=256
    train_data = RecognitionData(dataset_name='train')
    train_data.set_patch_folders(patch_size=patch_size)
    # train_data.get_patches(save=False) 

    paths = train_data.get_img_paths()


    # print(paths)
    # problem ones: 111,148 ---- 200 den devam et
    for no in ['283','966','111','54']:
    # for no in ['111','148','100','56','20','5']: 
    # for no in paths.keys():
        img_path = f'/home/murat/Projects/airplane_detection/DATA/Gaofen/train/images/{no}.tif'
        label_path = f'/home/murat/Projects/airplane_detection/DATA/Gaofen/train/label_xml/{no}.xml'

        train_data.get_patches_in_file(img_path,label_path,patch_size=patch_size,save=False,plot=True)


        # train_data.plot_bboxes(img_path,label_path)

### ANALYSE
# analyse = RecognitionAnalysis(dataset_name='train')
# analyse.get_airplane_size(patch_size)