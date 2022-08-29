import cv2
import numpy as np
import os
import geometry
import matplotlib.pyplot as plt
import json

### LEave some margin for the patches, because some airplane in DOTA has cutoff parts 
class PatchTools(object):
    """docstring for PatchTools"""
    def __init__(self, patch_size, task):
        super(PatchTools, self).__init__()

        self.patch_size=patch_size

        ### PAD SIZE (PAD ORIGINAL IMAGE AND FIRST CUTOUT)
        self.pad_size=int(self.patch_size/2)

        ### SEGMENTATION PATCHES
        self.segmentation_task=False
        # if 'mask_folder' in settings['patch'][dataset_part].keys():
        if task == 'segmentation':        
            self.segmentation_task = True

        self.margin=0
        ### PATCH TOOLS
    def get_original_image(self,img_path,flags=1):
        img = cv2.imread(img_path,flags=flags)
        if flags != 0:
            img = np.pad(img,((self.pad_size,self.pad_size),(self.pad_size,self.pad_size),(0,0)),'constant',constant_values=0)#'symmetric')#
        else:
            img = np.pad(img,((self.pad_size,self.pad_size),(self.pad_size,self.pad_size)),'constant',constant_values=0)#'symmetric')#
        return img

    def init_patch_dict(self,instance_name,img_path,mask_path=None):
        patch_dict =   {
                'file_path':None,
                'instance_name':None,
                'patch_size':None,
                'original':     {
                                    'img':None,
                                    'mask':None,
                                    'bbox':[],
                                    'mask_path':None,
                                    'img_path':None, 
                                    # 'center_padded':None, 
                                    'pad_size':0
                                },

                'original_padded_patch':   {   
                                    'img':None,
                                    'mask':None,
                                    'bbox':[],
                                    'bbox_params':[],
                                    },

                'original_patch':   {   
                                    'img':None,
                                    'mask':None,
                                    'bbox':[],
                                    'bbox_params':[],
                                    'img_path':None,
                                    'mask_path':None,
                                    },


                'orthogonal_patch': {
                                    'img':None,
                                    'mask':None,
                                    'bbox':[],
                                    'bbox_params':[],
                                    'img_path':None,
                                    'mask_path':None,
                                    },

                'orthogonal_zoomed_patch': 
                                    {
                                    'img':None,
                                    'mask':None,
                                    'bbox':[],
                                    'bbox_params':[],
                                    'img_path':None,
                                    'mask_path':None,
                                    },

                'notes':            {
                                    'bbox_params':None,
                                    'bbox':None
                                    },
                }
        patch_dict['instance_name']=instance_name
        patch_dict['original']['img_path']=img_path
        patch_dict['original']['mask_path']=mask_path
        patch_dict['original']['pad_size']=self.pad_size
        patch_dict['patch_size']=self.patch_size

        return patch_dict

    def set_patch_params(self,patch_dict,img,bbox,mask=None):

        patch_dict = self.set_original(patch_dict,img,bbox,mask)
        
        ### NEW BBOX
        # bbox_patch = bbox_orig_padded-center_padded+self.patch_size/2#+self.margin

        # rect = geometry.Rectangle(bbox=bbox_patch)

        # ### ORIGINAL PATCH BBOX
        # patch_dict['original_patch']['bbox']=bbox_patch
        # patch_dict['original_patch']['bbox_params']= [int(self.patch_size/2),int(self.patch_size/2),rect.h,rect.w,rect.angle]
        

        # # ### ORTHOGONAL PATCH BBOX
        # patch_dict['orthogonal_patch']['bbox']= rect.orthogonal_bbox
        # patch_dict['orthogonal_patch']['bbox_params']= [int(self.patch_size/2),int(self.patch_size/2),rect.h,rect.w,rect.get_atan2()]

        ### NOTES
        patch_dict['notes']['bbox_params'] = ['center_x,center_y,height,width,rotation_angle']
        patch_dict['notes']['bbox'] = ['[airplane_top_left_xy,airplane_bottom_left_xy,airplane_bottom_right_xy,airplane_top_right_xy]']

        # patch_dict = self.set_images(patch_dict=patch_dict,img=img,rect=rect,mask=mask)
        patch_dict = self.set_original(patch_dict,img,bbox,mask)
        patch_dict = self.set_original_padded_patch(patch_dict)
        patch_dict = self.set_original_patch(patch_dict)
        patch_dict = self.set_orthogonal_patch(patch_dict)
        patch_dict = self.set_orthogonal_zoomed_patch(patch_dict)


        # plt.imshow(patch_dict['original_padded_patch']['img'])
        # plt.show()
        return patch_dict



    def set_original(self,patch_dict,img,bbox,mask):
        ### IMAGE
        patch_dict['original']['img']=img

        ### BBOX
        bbox_orig_padded =np.array(bbox)+self.pad_size# # add initial padding
        patch_dict['original']['bbox']=bbox_orig_padded

        ## MASK
        patch_dict['original']['mask']=mask

        # center = np.mean(bbox,axis=0).astype(int)
        # center_padded = np.mean(bbox_orig_padded,axis=0).astype(int)#center+self.pad_size
        # patch_dict['original']['center_padded']=center_padded

        return patch_dict


    def set_original_padded_patch(self,patch_dict):
        # cx, cy = patch_dict['original']['center_padded']
        ### Get the large cutout image
        # y_0, y_1= cy-self.patch_size-self.margin, cy+self.patch_size+self.margin
        # x_0,x_1 = cx-self.patch_size-self.margin, cx+self.patch_size+self.margin

        img = patch_dict['original']['img']
        bbox = patch_dict['original']['bbox']
        # print(bbox)

        img_1, bbox_1 = self.cut_image_by_bbox(img,bbox,self.pad_size)

        patch_dict['original_padded_patch']['img']=img_1
        patch_dict['original_padded_patch']['bbox']=bbox_1

        return patch_dict

    def set_original_patch(self,patch_dict,margin=0):
        img = patch_dict['original_padded_patch']['img']
        bbox = patch_dict['original_padded_patch']['bbox']

        img_1, bbox_1 = self.cut_image_by_bbox(img,bbox,margin)

        patch_dict['original_patch']['img']=img_1
        patch_dict['original_patch']['bbox']=bbox_1

        return patch_dict

    def set_orthogonal_patch(self,patch_dict,margin=0):

        img = patch_dict['original_padded_patch']['img']
        bbox = patch_dict['original_padded_patch']['bbox']

        rect = geometry.Rectangle(bbox)

        angle = rect.get_atan2()
        # cv2.getRotationMatrix2D(center, angle, transform)
        M = cv2.getRotationMatrix2D((rect.cy, rect.cx), np.rad2deg(angle), 1.0) 
        # cv2.warpAffine(img, rotation, dest_size)
        img_rotated = cv2.warpAffine(img, M, (img.shape[0], img.shape[1])) 
        bbox_rotated = rect.orthogonal_bbox

        img_1, bbox_1 = self.cut_image_by_bbox(img_rotated,bbox_rotated,margin)

        patch_dict['orthogonal_patch']['img']=img_1
        patch_dict['orthogonal_patch']['bbox']=bbox_1
        return patch_dict


    def set_orthogonal_zoomed_patch(self,patch_dict,margin=0):

        img = patch_dict['orthogonal_patch']['img']
        bbox = patch_dict['orthogonal_patch']['bbox']

        img_1, bbox_1 = self.cut_image_by_bbox(img,bbox,margin)

        patch_dict['orthogonal_zoomed_patch']['img']=img_1
        patch_dict['orthogonal_zoomed_patch']['bbox']=bbox_1

        return patch_dict


    def cut_image_by_bbox(self,img,bbox,margin):
        x_min,x_max,y_min,y_max = geometry.Rectangle.get_bbox_limits(bbox)
        y_0,y_1 = np.array([y_min-margin, y_max+margin]).astype(int)
        x_0,x_1 = np.array([x_min-margin, x_max+margin]).astype(int)
        img_1=img[y_0:y_1,x_0:x_1,:] if len(img.shape)==3 else img[y_0:y_1,x_0:x_1]
        bbox_1 = bbox-[x_min,y_min]+margin
        return img_1, bbox_1


 

    def set_paths(self,settings,dataset_part,patch_dict,i):
        ### PATCH FOLDER SETTINGS
        img_patch_folder = settings['patch'][dataset_part]['img_folder'] 
        img_patch_orthogonal_folder = settings['patch'][dataset_part]['orthogonal_img_folder']
        img_patch_orthogonal_zoomed_folder = settings['patch'][dataset_part]['orthogonal_zoomed_img_folder']

        ## FILE NAMES
        img_path = patch_dict['original']['img_path']
        file_name = self.get_file_name_from_path(img_path)
        patch_name = f"{file_name}_{i}"
        patch_img_name = f"{patch_name}.png"

        patch_img_path = lambda folder: os.path.join(folder,patch_img_name)

        ### SET IMAGE PATHS
        patch_dict['original_patch']['img_path'] = patch_img_path(img_patch_folder)
        patch_dict['orthogonal_patch']['img_path'] = patch_img_path(img_patch_orthogonal_folder)
        patch_dict['orthogonal_zoomed_patch']['img_path'] = patch_img_path(img_patch_orthogonal_zoomed_folder)

        ### SET MASK PATHS

        if self.segmentation_task:
            mask_patch_folder = settings['patch'][dataset_part]['mask_folder']
            mask_patch_orthogonal_folder = settings['patch'][dataset_part]['orthogonal_mask_folder']
            mask_patch_orthogonal_zoomed_folder = settings['patch'][dataset_part]['orthogonal_zoomed_mask_folder']
            patch_dict['original_patch']['mask_path'] = patch_img_path(mask_patch_folder)
            patch_dict['orthogonal_patch']['mask_path'] = patch_img_path(mask_patch_orthogonal_folder)
            patch_dict['orthogonal_zoomed_patch']['mask_path'] = patch_img_path(mask_patch_orthogonal_zoomed_folder)

        ### SET LABEL PATH
        label_path = os.path.join(settings['patch'][dataset_part]['label_folder'],f'{patch_name}.json')
        patch_dict['file_path']=label_path
        return patch_dict

    def save_patch(self,settings,dataset_part,patch_dict,i):

        patch_dict = self.set_paths(settings,dataset_part,patch_dict,i)

        cv2.imwrite(patch_dict['original_patch']['img_path'],patch_dict['original_patch']['img'])
        cv2.imwrite(patch_dict['orthogonal_patch']['img_path'],patch_dict['orthogonal_patch']['img'])
        cv2.imwrite(patch_dict['orthogonal_zoomed_patch']['img_path'],patch_dict['orthogonal_zoomed_patch']['img'])

        if self.segmentation_task:
            cv2.imwrite(patch_dict['original_patch']['mask_path'],patch_dict['original_patch']['mask'])
            cv2.imwrite(patch_dict['orthogonal_patch']['mask_path'],patch_dict['orthogonal_patch']['mask'])
            cv2.imwrite(patch_dict['orthogonal_zoomed_patch']['mask_path'],patch_dict['orthogonal_zoomed_patch']['mask'])


        ### LABEL
        for key_0, value_0 in patch_dict.items():
            if isinstance(value_0,dict):
                ## REMOVE IMG ARRAY
                if 'img' in value_0.keys():
                    del patch_dict[key_0]['img']
                if 'mask' in value_0.keys():
                    del patch_dict[key_0]['mask']
           
                ## CHANGE NP.ARRAY TO LIST                
                for key_1, value_1 in patch_dict[key_0].items():
                    if isinstance(value_1,np.ndarray):
                        patch_dict[key_0][key_1] = patch_dict[key_0][key_1].tolist()

        with open(patch_dict['file_path'], 'w') as f:
            json.dump(patch_dict, f,indent=4)

    def plot_patch(self,ax,patch_dict,conf=['original','img']):
        ### PLOT
        # fig,ax = plt.subplots(1)
        patch_conf = conf[0]
        img_conf = conf[1]
        if img_conf=='img':
            ax.imshow(cv2.cvtColor(patch_dict[f'{patch_conf}_patch'][img_conf],cv2.COLOR_BGR2RGB)) # original_patch, orthogonal_patch, orthogonal_zoomed_patch
        elif img_conf=='mask':
            ax.imshow(patch_dict[f'{patch_conf}_patch'][img_conf]) # original_patch, orthogonal_patch, orthogonal_zoomed_patch

        geometry.Rectangle.plot_bbox(bbox=patch_dict[f'{patch_conf}_patch']['bbox'],ax=ax,c='b')

        instance_name = patch_dict['instance_name']
        ax.set_title(instance_name)
        # plt.show()
        # return ax

    def get_file_name_from_path(self,path):
        return os.path.splitext(os.path.split(path)[-1])[0]

