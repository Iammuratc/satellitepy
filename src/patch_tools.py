import cv2
import numpy as np
import os
import geometry
import matplotlib.pyplot as plt

class PatchTools(object):
    """docstring for PatchTools"""
    def __init__(self, settings, dataset_part):
        super(PatchTools, self).__init__()
        
        self.patch_size=settings['patch']['size']
        self.pad_size=self.patch_size        

        ### SEGMENTATION PATCHES
        self.segmentation_task=False
        if 'mask_folder' in settings['patch'][dataset_part].keys():        
            self.segmentation_task = True


        ### PATCH TOOLS
    def get_original_image(self,img_path):
        img = cv2.imread(img_path)
        img = np.pad(img,((self.pad_size,self.pad_size),(self.pad_size,self.pad_size),(0,0)),'constant',constant_values=0)#'symmetric')#
        return img

    def init_patch_dict(self,instance_name,img_path,mask_path=None):
        patch_dict =   {
                'instance_name':None,
                'patch_size':None,
                'original_patch':   {   
                                    'img':np.zeros(shape=(self.patch_size,self.patch_size,3),dtype=np.uint8),
                                    'mask':np.zeros(shape=(self.patch_size,self.patch_size),dtype=np.uint8),
                                    'bbox':[],
                                    'bbox_params':[],
                                    'img_path':None,
                                    'mask_path':None,
                                    },

                'orthogonal_patch': {
                                    'img':np.zeros(shape=(self.patch_size,self.patch_size,3),dtype=np.uint8),
                                    'mask':np.zeros(shape=(self.patch_size,self.patch_size),dtype=np.uint8),
                                    'bbox':[],
                                    'bbox_params':[],
                                    'img_path':None,
                                    'mask_path':None,
                                    },

                'orthogonal_zoomed_patch': 
                                    {
                                    'img':np.zeros(shape=(self.patch_size,self.patch_size,3),dtype=np.uint8),
                                    'mask':np.zeros(shape=(self.patch_size,self.patch_size),dtype=np.uint8),
                                    'bbox':[],
                                    'bbox_params':[],
                                    'img_path':None,
                                    'mask_path':None,
                                    },

                'notes':            {
                                    'bbox_params':None,
                                    'bbox':None
                                    },

                'original':     {
                                    'mask_path':None,
                                    'img_path':None, 
                                    'center_padded':None, 
                                    'pad_size':0
                                    }
                }
        patch_dict['instance_name']=instance_name
        patch_dict['original']['img_path']=img_path
        patch_dict['original']['mask_path']=mask_path
        patch_dict['original']['pad_size']=self.pad_size
        patch_dict['patch_size']=self.patch_size

        return patch_dict

    def set_orthogonal_zoomed_img(self,patch_dict):
        orthogonal_patch_img = patch_dict['orthogonal_patch']['img']
        orthogonal_bbox = patch_dict['orthogonal_patch']['bbox']

        x_min,x_max,y_min,y_max = geometry.Rectangle.get_bbox_limits(orthogonal_bbox)
        x_min,x_max = np.clip([x_min,x_max],0,orthogonal_patch_img.shape[1])
        y_min,y_max = np.clip([y_min,y_max],0,orthogonal_patch_img.shape[0])


        ### GET THE ORTHOGONAL ZOOMED IMAGE
        orthogonal_img = orthogonal_patch_img[y_min:y_max,x_min:x_max,:]
        orthogonal_zoomed_img = cv2.resize(orthogonal_img,dsize=(self.patch_size,self.patch_size))
        patch_dict['orthogonal_zoomed_patch']['img'] = orthogonal_zoomed_img

        if self.segmentation_task:
            orthogonal_patch_mask = patch_dict['orthogonal_patch']['mask']
            orthogonal_mask = orthogonal_patch_mask[y_min:y_max,x_min:x_max]
            orthogonal_zoomed_mask = cv2.resize(orthogonal_mask,dsize=(self.patch_size,self.patch_size))            
            patch_dict['orthogonal_zoomed_patch']['mask'] = orthogonal_zoomed_mask


        ### GET THE ORTHOGONAL ZOOMED BBOX
        orthogonal_zoomed_bbox = orthogonal_bbox - [x_min,y_min]
        orthogonal_zoomed_bbox = orthogonal_zoomed_bbox * [self.patch_size/(x_max-x_min),self.patch_size/(y_max-y_min)]

        patch_dict['orthogonal_zoomed_patch']['bbox'] = orthogonal_zoomed_bbox
        patch_dict['orthogonal_zoomed_patch']['bbox_params'] = geometry.Rectangle.get_params(orthogonal_zoomed_bbox)
        return patch_dict

    def get_padded_original_img(self,img,patch_dict):
        cx, cy = patch_dict['original']['center_padded']

        # Get the large cutout image
        y_0, y_1= cy-self.patch_size, cy+self.patch_size
        x_0,x_1 = cx-self.patch_size, cx+self.patch_size
        img_1=img[y_0:y_1,x_0:x_1,:] if len(img.shape)==3 else img[y_0:y_1,x_0:x_1]
        return img_1        

    def get_orthogonal_patch_img(self,img,rect,patch_dict):
       
        # Get the rotation angle
        angle = rect.get_atan2()
        M = cv2.getRotationMatrix2D((img.shape[0]/2, img.shape[1]/2), np.rad2deg(angle), 1.0)

        img_2 = cv2.warpAffine(img, M, (self.patch_size*2, self.patch_size*2))

        # Get the image patch
        orthogonal_patch_img = self.get_small_cutout(img_2)
        return orthogonal_patch_img

    def get_small_cutout(self,img):
        patch_half_size = int(self.patch_size/2)
        start = self.patch_size-patch_half_size
        end = self.patch_size+patch_half_size
        small_cutout = img[start:end,start:end,:] if len(img.shape)==3 else img[start:end,start:end]
        return small_cutout

    def set_images(self,patch_dict,img,rect,mask):
        '''
        Get the patch of the upwards facing airplane (orthogonal)
        Original padded image (padded_img) >> 2*patch_size cutout image >> rotate cutout such that the airplane is facing upwards (img_2) >> patch_size cutout image
        img: original img
        ''' 
        padded_img = self.get_padded_original_img(img,patch_dict)

        patch_dict['orthogonal_patch']['img']=self.get_orthogonal_patch_img(padded_img,rect,patch_dict)
        patch_dict['original_patch']['img']=self.get_small_cutout(padded_img)

        if self.segmentation_task:
            padded_mask = self.get_padded_original_img(mask,patch_dict)
            patch_dict['original_patch']['mask']=self.get_small_cutout(padded_mask)
            patch_dict['orthogonal_patch']['mask']=self.get_orthogonal_patch_img(padded_mask,rect,patch_dict)

        patch_dict = self.set_orthogonal_zoomed_img(patch_dict)
        return patch_dict


    def set_patch_params(self,patch_dict,img,bbox,mask=None):
        bbox_orig_padded =np.array(bbox)+self.pad_size # add initial padding
        ### NEW CENTER OF AIRPLANE
        center = np.mean(bbox,axis=0).astype(int)
        center_padded = center+self.pad_size
        patch_dict['original']['center_padded']=center_padded
        ### NEW BBOX
        bbox_patch = bbox_orig_padded-center_padded+self.pad_size/2

        rect = geometry.Rectangle(bbox=bbox_patch)

        ### ORIGINAL PATCH BBOX
        patch_dict['original_patch']['bbox']=bbox_patch
        patch_dict['original_patch']['bbox_params']= [int(self.patch_size/2),int(self.patch_size/2),rect.h,rect.w,rect.angle]
        

        ### ORTHOGONAL PATCH BBOX
        patch_dict['orthogonal_patch']['bbox']= rect.orthogonal_bbox
        patch_dict['orthogonal_patch']['bbox_params']= [int(self.patch_size/2),int(self.patch_size/2),rect.h,rect.w,rect.get_atan2()]

        ### NOTES
        patch_dict['notes']['bbox_params'] = ['center_x,center_y,height,width,rotation_angle']
        patch_dict['notes']['bbox'] = ['[airplane_top_left_xy,airplane_bottom_left_xy,airplane_bottom_right_xy,airplane_top_right_xy]']

        patch_dict = self.set_images(patch_dict=patch_dict,img=img,rect=rect,mask=mask)
        return patch_dict

    def set_paths(self,patch_dict,settings):
        ### PATCH FOLDER SETTINGS
        img_patch_folder = settings['patch'][dataset_part]['img_patch_folder'] 
        img_patch_orthogonal_folder = settings['patch'][dataset_part]['orthogonal_img_folder']
        img_patch_orthogonal_zoomed_folder = settings['patch'][dataset_part]['orthogonal_zoomed_img_folder']

        ## FILE NAMES
        img_path = patch_dict['original']['path']
        file_name = self.get_file_name_from_path(img_path)
        patch_img_name = f"{file_name}_{i}.png"

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
            patch_dict['orthogonal_patch']['mask_path'] = patch_img_path(mask_patch_orhtogonal_folder)
            patch_dict['orthogonal_zoomed_patch']['mask_path'] = patch_img_path(mask_patch_orthogonal_zoomed_folder)


        return patch_dict

    def save_patch(self,patch_dict,i):

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

        with open(f"{self.label_patch_folder}/{patch_name}.json", 'w') as f:
            json.dump(patch_dict, f,indent=4)

    def plot_patch(self,patch_dict,i,conf='original'):
        ### PLOT
        fig,ax = plt.subplots(1)
        ax.imshow(cv2.cvtColor(patch_dict[f'{conf}_patch']['img'],cv2.COLOR_BGR2RGB)) # original_patch, orthogonal_patch, orthogonal_zoomed_patch
        geometry.Rectangle.plot_bbox(bbox=patch_dict[f'{conf}_patch']['bbox'],ax=ax,c='b')

        instance_name = patch_dict['instance_name']
        ax.set_title(instance_name)
        plt.show()

    def get_file_name_from_path(self,path):
        return os.path.splitext(os.path.split(path)[-1])[0]

