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
        self.img_patch_orthogonal_folder = f"{self.patch_folder_base}/orthogonal_images"
        os.makedirs(self.img_patch_folder,exist_ok=True)
        self.img_patch_orthogonal_zoomed_folder = f"{self.patch_folder_base}/orthogonal_zoomed_images"
        os.makedirs(self.img_patch_orthogonal_zoomed_folder,exist_ok=True)
        self.label_patch_folder = f"{self.patch_folder_base}/labels"
        os.makedirs(self.label_patch_folder,exist_ok=True)
        # os.makedirs(f"{self.patch_folder_base}/figures",exist_ok=True)


    def init_patch_dict(self,instance_name,img_path,pad_size,patch_size):
        patch_dict =   {
                'instance_name':None,
                'patch_size':None,
                'original_patch':   {   
                                    'img':np.zeros(shape=(patch_size,patch_size,3),dtype=np.uint8),
                                    'path':None,
                                    'bbox':[],
                                    'bbox_params':[]
                                    },

                'orthogonal_patch': {
                                    'img':np.zeros(shape=(patch_size,patch_size,3),dtype=np.uint8),
                                    'path':None,
                                    'bbox':[],
                                    'bbox_params':[]
                                    },

                'orthogonal_zoomed_patch': 
                                    {
                                    'img':np.zeros(shape=(patch_size,patch_size,3),dtype=np.uint8),
                                    'path':None,
                                    'bbox':[],
                                    'bbox_params':[]
                                    },

                'notes':            {
                                    'bbox_params':None,
                                    'bbox':None
                                    },

                'original_img':     {
                                    'path':None, 
                                    'center_padded':None, 
                                    'pad_size':0
                                    }
                }
        patch_dict['instance_name']=instance_name
        patch_dict['original_img']['path']=img_path
        patch_dict['original_img']['pad_size']=pad_size
        patch_dict['patch_size']=patch_size

        return patch_dict

    def set_orthogonal_zoomed_img(self,patch_dict):
        orthogonal_patch_img = patch_dict['orthogonal_patch']['img']
        orthogonal_bbox = patch_dict['orthogonal_patch']['bbox']
        patch_size=patch_dict['patch_size']

        x_min,x_max,y_min,y_max = geometry.Rectangle.get_bbox_limits(orthogonal_bbox)

        # print(x_min,x_max,y_min,y_max)
        orthogonal_img = orthogonal_patch_img[y_min:y_max,x_min:x_max,:]

        orthogonal_zoomed_img = cv2.resize(orthogonal_img,dsize=(patch_size,patch_size))

        patch_dict['orthogonal_zoomed_patch']['img'] = orthogonal_zoomed_img

        Ry,Rx = orthogonal_img.shape[0]/patch_size, orthogonal_img.shape[1]/patch_size

        orthogonal_zoomed_bbox = orthogonal_bbox * np.array([Rx,Ry])
        patch_dict['orthogonal_zoomed_patch']['bbox'] = orthogonal_zoomed_bbox
        patch_dict['orthogonal_zoomed_patch']['bbox_params'] = geometry.Rectangle.get_params(orthogonal_zoomed_bbox)
        return patch_dict


    def set_images(self,img,rect,patch_dict):
        '''
        Get the patch of the upwards facing airplane (orthogonal)
        Original padded image (img) >> 2*patch_size image (img_1) >> rotate such that the airplane is facing upwards (img_2) >> patch_size image
        img: original img
        ''' 

        patch_size = patch_dict['patch_size']
        cx, cy = patch_dict['original_img']['center_padded']

        # Get the large image
        img_1=img[cy-patch_size:cy+patch_size,cx-patch_size:cx+patch_size,:]
       

        # Get the rotation angle
        angle = rect.get_atan2()
        M = cv2.getRotationMatrix2D((img_1.shape[0]/2, img_1.shape[1]/2), np.rad2deg(angle), 1.0)

        img_2 = cv2.warpAffine(img_1, M, (patch_size*2, patch_size*2))

        # Get the image patch
        patch_half_size = int(patch_size/2)
        orthogonal_patch_img = img_2[patch_size-patch_half_size:patch_size+patch_half_size,patch_size-patch_half_size:patch_size+patch_half_size,:]
        patch_dict['orthogonal_patch']['img']=orthogonal_patch_img
        patch_dict['original_patch']['img']=img_1[patch_size-patch_half_size:patch_size+patch_half_size,patch_size-patch_half_size:patch_size+patch_half_size,:]
        patch_dict = self.set_orthogonal_zoomed_img(patch_dict)
        return patch_dict


    def set_patch_params(self,patch_dict,img,bbox):

        pad_size = patch_dict['original_img']['pad_size']
        patch_size=patch_dict['patch_size']
        # If not including _orig, the variable belongs to the patch 
        bbox_orig_padded =np.array(bbox)+pad_size # add initial padding
        ### NEW CENTER OF AIRPLANE
        center = np.mean(bbox,axis=0).astype(int)
        center_padded = center+pad_size
        patch_dict['original_img']['center_padded']=center_padded
        ### NEW BBOX
        bbox_patch = bbox_orig_padded-center_padded+pad_size/2

        rect = geometry.Rectangle(bbox=bbox_patch)

        ### ORIGINAL PATCH BBOX
        patch_dict['original_patch']['bbox']=bbox_patch
        patch_dict['original_patch']['bbox_params']= [int(patch_size/2),int(patch_size/2),rect.h,rect.w,rect.angle]
        

        ### ORTHOGONAL PATCH BBOX
        patch_dict['orthogonal_patch']['bbox']= rect.orthogonal_bbox
        patch_dict['orthogonal_patch']['bbox_params']= [int(patch_size/2),int(patch_size/2),rect.h,rect.w,rect.get_atan2()]

        ### NOTES
        patch_dict['notes']['bbox_params'] = ['center_x,center_y,height,width,rotation_angle']
        patch_dict['notes']['bbox'] = ['[airplane_top_left_xy,airplane_bottom_left_xy,airplane_bottom_right_xy,airplane_top_right_xy]']

        patch_dict = self.set_images(img=img,rect=rect,patch_dict=patch_dict)
        return patch_dict


    def get_patches(self,img_path,label_path,patch_size,save=False,plot=False):
        
        ### GET ORIGINAL IMAGE
        img = cv2.imread(img_path)
        label = self.get_label(label_path)
        ### pad the original image, so no patching problem for the planes on the edge of the image
        pad_size = patch_size
        img = np.pad(img,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'constant',constant_values=0)#'symmetric')#
        ### GET LABELS
        bboxes = label['bbox']

        for i,bbox in enumerate(bboxes):
            patch_dict = self.init_patch_dict(instance_name=label["names"][i],img_path=img_path,pad_size=pad_size,patch_size=patch_size)

            patch_dict = self.set_patch_params(patch_dict,img,bbox)

            if plot:
                self.plot_patches(patch_dict,i)
            
            if save:
                self.save_patches(patch_dict,i)
            # break


    def plot_patches(self,patch_dict,i):
        ### PLOT
        # print(patch_dict[''])
        # fig,ax = plt.subplots(2)
        # ax[0].imshow(cv2.cvtColor(patch_dict['original_patch']['img'],cv2.COLOR_BGR2RGB))
        # # # ax[0].set_ylim(ax[0].get_ylim()[::-1]) # invert y axis

        # original_patch_bbox = patch_dict['original_patch']['bbox']
        # geometry.Rectangle.plot_bbox(bbox=original_patch_bbox,ax=ax[0],c='b')

        # ax[1].imshow(cv2.cvtColor(patch_dict['orthogonal_patch']['img'],cv2.COLOR_BGR2RGB))
        # orthogonal_patch_bbox = patch_dict['orthogonal_patch']['bbox']
        # geometry.Rectangle.plot_bbox(bbox=orthogonal_patch_bbox,ax=ax[1],c='b')


        ### PLOT 2
        fig,ax = plt.subplots(1)
        ax.imshow(cv2.cvtColor(patch_dict['orthogonal_patch']['img'],cv2.COLOR_BGR2RGB))
        orthogonal_patch_bbox = patch_dict['orthogonal_patch']['bbox']
        geometry.Rectangle.plot_bbox(bbox=orthogonal_patch_bbox,ax=ax,c='b')

        instance_name = patch_dict['instance_name']
        ax.set_title(instance_name)
        ### SAVE FIGURES
        ## FILE NAMES
        # img_path = patch_dict['original_img']['path']
        # file_name = get_file_name_from_path(img_path)
        # patch_name = f"{file_name}_{i}"
        # plt.savefig(f"{self.patch_folder_base}/figures/{instance_name}_{patch_name}.png", bbox_inches='tight')
        plt.show()


    def save_patches(self,patch_dict,i):

        ## FILE NAMES
        img_path = patch_dict['original_img']['path']
        file_name = get_file_name_from_path(img_path)
        patch_name = f"{file_name}_{i}"



        ## ORIGINAL PATCH IMG
        original_patch_path = f"{self.img_patch_folder}/{patch_name}.png"
        patch_dict['original_patch']['path'] = original_patch_path
        cv2.imwrite(original_patch_path,patch_dict['original_patch']['img'])

        ## ORTHOGONAL PATCH IMG
        orthogonal_patch_path = f"{self.img_patch_orthogonal_folder}/{patch_name}.png"
        patch_dict['orthogonal_patch']['path'] = orthogonal_patch_path
        cv2.imwrite(orthogonal_patch_path,patch_dict['orthogonal_patch']['img'])
        
        ## ORTHOGONAL ZOOMED IMG
        orthogonal_zoomed_patch_path = f"{self.img_patch_orthogonal_zoomed_folder}/{patch_name}.png"
        patch_dict['orthogonal_zoomed_patch']['path'] = orthogonal_patch_path
        cv2.imwrite(orthogonal_zoomed_patch_path,patch_dict['orthogonal_zoomed_patch']['img'])

        ### LABEL
        for key_0, value_0 in patch_dict.items():
            if isinstance(value_0,dict):
                ## REMOVE IMG ARRAY
                if 'img' in value_0.keys():
                    del patch_dict[key_0]['img']
                ## CHANGE NP.ARRAY TO LIST                
                for key_1, value_1 in patch_dict[key_0].items():
                    if isinstance(value_1,np.ndarray):
                        patch_dict[key_0][key_1] = patch_dict[key_0][key_1].tolist()

        with open(f"{self.label_patch_folder}/{patch_name}.json", 'w') as f:
            json.dump(patch_dict, f,indent=4)

        # return patch_dict



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
           cx,cy,h,w,angle = patch_dict['orthogonal_patch']['bbox_params']
           instance_name=patch_dict['instance_name']

           
           if instance_name not in size_dict.keys():
                size_dict[instance_name] = {'w':[],'h':[]}
           else:
                size_dict[instance_name]['w'].append(w)
                size_dict[instance_name]['h'].append(h)
        # print(size_dict)

        for instance_name in size_dict.keys():
            total_no = len(size_dict[instance_name]['h'])
            print(f"{instance_name}: {total_no}")
            # fig,ax=plt.subplots(2)
            # fig.suptitle(f'Instance:{instance_name}, total no: {total_no}')
            # ax[0].set_title('Height')
            # ax[1].set_title('Width')
            # ax[0].hist(size_dict[instance_name]['h'],bins=50)
            # ax[1].hist(size_dict[instance_name]['w'],bins=50)
            # plt.show()

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
    patch_size=128
    train_data = RecognitionData(dataset_name='train')
    train_data.set_patch_folders(patch_size=patch_size)
    # train_data.get_patches(save=False) 

    paths = train_data.get_img_paths()


    # print(paths)
    # problem ones: 111,148 ---- 200 den devam et
    # for no in ['283','966','111','54']:
    # for no in ['111','148','100','56','20','5']: 
    for no in paths.keys():
        img_path = f'/home/murat/Projects/airplane_recognition/DATA/Gaofen/train/images/{no}.tif'
        label_path = f'/home/murat/Projects/airplane_recognition/DATA/Gaofen/train/label_xml/{no}.xml'

        train_data.get_patches(img_path,label_path,patch_size=patch_size,save=False,plot=True)

        # save_path = f"{train_data.data_folder}/{train_data.dataset_name}/figures/{no}_wrong_labels.png"
        # train_data.plot_bboxes(img_path,label_path,save_path)

### ANALYSE
# analyse = RecognitionAnalysis(dataset_name='train')
# analyse.get_airplane_size(patch_size)