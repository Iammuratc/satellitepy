import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

import geometry
from data import DataDem
import shapely.wkt

#### RECOGNITION DATA
class Recognition(DataDem):
    def __init__(self,settings,dataset_part):
        '''
        Save patches for the recognition task
        A patch is consisted of a single airplane. The airplane is located in the middle of the patch by using the center of its rotated bounding box.
        Rotated bounding box can be masked (i.e. masked=True) 
        '''
        super(Recognition, self).__init__(settings,dataset_part)

        self.patch_size=settings['patch']['size']
        self.pad_size = settings['patch']['size']
        self.img_patch_folder = settings['patch'][dataset_part]['img_patch_folder'] #f"{self.patch_folder_base}/images"
        self.img_patch_orthogonal_folder = settings['patch'][dataset_part]['img_patch_orthogonal_folder']#f"{self.patch_folder_base}/orthogonal_images"
        self.img_patch_orthogonal_zoomed_folder = settings['patch'][dataset_part]['img_patch_orthogonal_zoomed_folder']#f"{self.patch_folder_base}/orthogonal_zoomed_images"
        self.label_patch_folder = settings['patch'][dataset_part]['label_patch_folder']#f"{self.patch_folder_base}/labels"


    def save_patches(self):
        ### SAVE PATCHES
        sequences = self.data

        ans = input("Do you really want to save the patches?")
        if ans != 'y':
            print('If you want to save the patches, please confirm this with y.')
            return 0

        ### DEM ADAPTED VERSION
        for s in sequences:
            for base_image in s["base_images"]:
                ### GET ORIGINAL IMAGE
                img_path = base_image['image_path']
                img = self.get_original_image(img_path)
                for i,label in enumerate(base_image['ground_truth']):
                    self.get_patch(img,img_path,label,ind=i,save=True,plot=False)

    def get_patch(self,img,img_path,label,ind,save=False,plot=False):

        bbox_polygon = shapely.wkt.loads(label['pixel_position'])
        bbox = np.array(bbox_polygon.exterior.coords)[0:4,:]
        instance_name = label['class']

        patch_dict = self.init_patch_dict(instance_name=instance_name,img_path=img_path)
        patch_dict = self.set_patch_params(patch_dict,img,bbox)
        if plot:
            self.plot_patch(patch_dict,ind)
        
        if save:
            self.save_patch(patch_dict,ind)

        # return patch_dict

    def get_patch_by_index(self,img_path,obj_ind,save=False,plot=True):
        for s in self.data:
            for base_image in s["base_images"]:
                # print(base_image['image_path'])
                if img_path == base_image['image_path']:
                    img = self.get_original_image(img_path)
                    label = base_image['ground_truth'][obj_ind]
                    self.get_patch(img,img_path,label,obj_ind,save=save,plot=plot)


    def plot_all_bboxes_on_base_image(self,img_path):
        for s in self.data:
            for base_image in s["base_images"]:
                # print(base_image['image_path'])
                if img_path == base_image['image_path']:
                    img = cv2.imread(img_path)
                    fig,ax = plt.subplots(1)
                    ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # original_patch, orthogonal_patch, orthogonal_zoomed_patch
                    labels = base_image['ground_truth'] 
                    for label in labels:
                        bbox_polygon = shapely.wkt.loads(label['pixel_position'])
                        bbox = np.array(bbox_polygon.exterior.coords)[0:4,:]
                        geometry.Rectangle.plot_bbox(bbox=bbox,ax=ax,c='b')
                    plt.show()



    def get_original_image(self,img_path):
        img = cv2.imread(img_path)
        img = np.pad(img,((self.pad_size,self.pad_size),(self.pad_size,self.pad_size),(0,0)),'constant',constant_values=0)#'symmetric')#
        return img

    def init_patch_dict(self,instance_name,img_path):
        patch_dict =   {
                'instance_name':None,
                'patch_size':None,
                'original_patch':   {   
                                    'img':np.zeros(shape=(self.patch_size,self.patch_size,3),dtype=np.uint8),
                                    'path':None,
                                    'bbox':[],
                                    'bbox_params':[]
                                    },

                'orthogonal_patch': {
                                    'img':np.zeros(shape=(self.patch_size,self.patch_size,3),dtype=np.uint8),
                                    'path':None,
                                    'bbox':[],
                                    'bbox_params':[]
                                    },

                'orthogonal_zoomed_patch': 
                                    {
                                    'img':np.zeros(shape=(self.patch_size,self.patch_size,3),dtype=np.uint8),
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
        patch_dict['original_img']['pad_size']=self.pad_size
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



        ### GET THE ORTHOGONAL ZOOMED BBOX
        orthogonal_zoomed_bbox = orthogonal_bbox - [x_min,y_min]
        orthogonal_zoomed_bbox = orthogonal_zoomed_bbox * [self.patch_size/(x_max-x_min),self.patch_size/(y_max-y_min)]

        patch_dict['orthogonal_zoomed_patch']['bbox'] = orthogonal_zoomed_bbox
        patch_dict['orthogonal_zoomed_patch']['bbox_params'] = geometry.Rectangle.get_params(orthogonal_zoomed_bbox)
        return patch_dict


    def set_images(self,img,rect,patch_dict):
        '''
        Get the patch of the upwards facing airplane (orthogonal)
        Original padded image (img) >> 2*patch_size image (img_1) >> rotate such that the airplane is facing upwards (img_2) >> patch_size image
        img: original img
        ''' 

        cx, cy = patch_dict['original_img']['center_padded']

        # Get the large image
        img_1=img[cy-self.patch_size:cy+self.patch_size,cx-self.patch_size:cx+self.patch_size,:]
       

        # Get the rotation angle
        angle = rect.get_atan2()
        M = cv2.getRotationMatrix2D((img_1.shape[0]/2, img_1.shape[1]/2), np.rad2deg(angle), 1.0)

        img_2 = cv2.warpAffine(img_1, M, (self.patch_size*2, self.patch_size*2))

        # Get the image patch
        patch_half_size = int(self.patch_size/2)
        orthogonal_patch_img = img_2[self.patch_size-patch_half_size:self.patch_size+patch_half_size,self.patch_size-patch_half_size:self.patch_size+patch_half_size,:]
        patch_dict['orthogonal_patch']['img']=orthogonal_patch_img
        patch_dict['original_patch']['img']=img_1[self.patch_size-patch_half_size:self.patch_size+patch_half_size,self.patch_size-patch_half_size:self.patch_size+patch_half_size,:]
        patch_dict = self.set_orthogonal_zoomed_img(patch_dict)
        return patch_dict


    def set_patch_params(self,patch_dict,img,bbox):
        # If not including _orig, the variable belongs to the patch 
        bbox_orig_padded =np.array(bbox)+self.pad_size # add initial padding
        ### NEW CENTER OF AIRPLANE
        center = np.mean(bbox,axis=0).astype(int)
        center_padded = center+self.pad_size
        patch_dict['original_img']['center_padded']=center_padded
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

        patch_dict = self.set_images(img=img,rect=rect,patch_dict=patch_dict)
        return patch_dict


    def plot_patch(self,patch_dict,i):
        ### PLOT
        fig,ax = plt.subplots(1)
        ax.imshow(cv2.cvtColor(patch_dict['original_patch']['img'],cv2.COLOR_BGR2RGB)) # original_patch, orthogonal_patch, orthogonal_zoomed_patch
        geometry.Rectangle.plot_bbox(bbox=patch_dict['original_patch']['bbox'],ax=ax,c='b')

        instance_name = patch_dict['instance_name']
        ax.set_title(instance_name)
        plt.show()


    def save_patch(self,patch_dict,i):

        ## FILE NAMES
        img_path = patch_dict['original_img']['path']
        file_name = self.get_file_name_from_path(img_path)
        patch_name = f"{file_name}_{i}"

        ## ORIGINAL PATCH IMG
        original_patch_path = f"{self.img_patch_folder}/{patch_name}.png"
        patch_dict['original_patch']['path'] = original_patch_path
        cv2.imwrite(original_patch_path,patch_dict['original_patch']['img'])

        ## ORTHOGONAL PATCH IMG
        orthogonal_patch_path = f"{self.img_patch_orthogonal_folder}/{patch_name}.png"
        patch_dict['orthogonal_patch']['path'] = orthogonal_patch_path
        cv2.imwrite(orthogonal_patch_path,patch_dict['orthogonal_patch']['img'])
        # print(orthogonal_patch_path)

        ## ORTHOGONAL ZOOMED IMG
        orthogonal_zoomed_patch_path = f"{self.img_patch_orthogonal_zoomed_folder}/{patch_name}.png"
        patch_dict['orthogonal_zoomed_patch']['path'] = orthogonal_zoomed_patch_path
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


    def get_file_name_from_path(self,path):
        return os.path.splitext(os.path.split(path)[-1])[0]


class RecognitionAnalysis(Recognition):
    def __init__(self,dataset_id,dataset_part,dataset_name,patch_size):
        '''
        Analyse the recognition data
        '''
        super(RecognitionAnalysis, self).__init__(dataset_id,dataset_part,dataset_name,patch_size)
        self.set_patch_folders()

    def get_instance_number(self):
        json_files = os.listdir(self.label_patch_folder)

        print(f"{self.dataset_part} set of {self.dataset_name} has the following instances:")
        instance_number = {}
        for json_file in json_files:
            patch_dict = json.load(open(f"{self.label_patch_folder}/{json_file}",'r'))
            instance_name=patch_dict['instance_name']
            if instance_name not in instance_number.keys():
                instance_number[instance_name] = 0
            else:
                instance_number[instance_name] += 1
        ## SORT DICT
        instance_number = dict(sorted(instance_number.items()))

        instance_number['TOTAL'] = sum(instance_number.values())
        return instance_number
    def get_airplane_size(self):

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

    
    patch_size=128
    dataset_id = 'f73e8f1f-f23f-4dca-8090-a40c4e1c260e'
    dataset_name = 'Gaofen'
    dataset_part = 'train'

    ### SAVE PATCHES FOR A DATASET PART
    recognition = Recognition(dataset_id,dataset_part,dataset_name,patch_size)
    # recognition.save_patches()

    ### PLOT ALL THE BOOX ON AN IMAGE
    # recognition.plot_all_bboxes_on_base_image("/home/murat/Projects/airplane_recognition/data/Gaofen/train/images/20.tif")


    ### PLOT SINGLE PATCH BY INDEX

    # problem_ones_file = '/home/murat/Projects/airplane_recognition/docs/problem_ones.txt'
    # with open(problem_ones_file,'r') as f:
    #     lines = f.readlines()
    #     # print(img_paths)
    #     for line in lines:

    #         img_path,ind = line.split(',')
    #         ind = int(ind[:-1])
    #         print(os.path.split(img_path)[-1], ind)
    #         recognition.get_patch_by_index(img_path=img_path,obj_ind=ind,save=False,plot=True)


    ### PLOT SINGLE PATCH BY INDEX
    img_path = "/home/murat/Projects/airplane_recognition/data/Gaofen/train/images/143.tif"
    recognition.get_patch_by_index(img_path=img_path,obj_ind=0,save=False,plot=True)

    # ### ANALYSE
    # analyse = RecognitionAnalysis(dataset_id,dataset_part,dataset_name,patch_size)
    # instance_number = analyse.get_instance_number()
    # # print(instance_number)
    # for key, value in instance_number.items():
    #     print(f"{key}: {value}")