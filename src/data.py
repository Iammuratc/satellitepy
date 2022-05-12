import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
import os
import json
import cv2
import math
import geometry
from utilities import get_file_name_from_path
# from utilities import show_sample


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
            rect = geometry.Rectangle(np.array(bbox))
            rect.plot_corners(ax)
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



#### DETECTION DATA
class DetectionData(Data):
    
    def __init__(self,dataset_name):
        super(DetectionData, self).__init__(dataset_name)

    def save_patches(self,patch_size,overlap=100):
        patches = []

        box_corner_threshold = 2

        patch_folder = f"{self.data_folder}/{self.dataset_name}/patches_{patch_size}"
        os.makedirs(patch_folder,exist_ok=True)
        img_patch_folder = f"{patch_folder}/images"
        os.makedirs(img_patch_folder,exist_ok=True)
        label_patch_folder = f"{patch_folder}/labels_original"
        os.makedirs(label_patch_folder,exist_ok=True)
        label_patch_yolo_folder = f"{patch_folder}/labels"
        os.makedirs(label_patch_yolo_folder,exist_ok=True)

        for img_name, img_path in self.img_paths.items():
            # img = cv2.cvtColor(cv2.imread(img_path,1),cv2.COLOR_BGR2RGB)
            img = cv2.imread(img_path,1)
            bbox_coords = self.labels[img_name]['bbox']
            instance_names = self.labels[img_name]['names']
            y_max, x_max, ch = img.shape[:3]
            patch_y, patch_x = self.get_patch_start_coords(my_max=[y_max,x_max],patch_size=patch_size,overlap=overlap)
            for y_0 in patch_y:
                for x_0 in patch_x:
                    my_labels = {'instance_names':[],'rotated_bboxes':[] ,'orthogonal_bboxes':[],'airplane_exist':False}
                    patch = np.zeros(shape=(patch_size,patch_size,ch),dtype=np.uint8)

                    for i,coords in enumerate(bbox_coords):
                        box_corner_in_patch = 0
                        for coord in coords:
                            if (x_0<=coord[0]<=x_0+patch_size) and (y_0<=coord[1]<=y_0+patch_size):
                                box_corner_in_patch += 1
                        if box_corner_in_patch>=box_corner_threshold:
                            # INSTANCE NAMES
                            my_labels['instance_names'].append(instance_names[i])
                            # shift coords
                            shifted_coords = np.array(coords)-[x_0,y_0]
                            # ROTATED BBOXES                     
                            my_labels['rotated_bboxes'].append(shifted_coords.tolist())
                            # ORTHOGONAL BBOXES
                            x_coords = shifted_coords[:,0]
                            y_coords = shifted_coords[:,1]
                            # print(x_coords,y_coords)
                            x_coord_min, x_coord_max = np.amin(x_coords),np.amax(x_coords)
                            y_coord_min, y_coord_max = np.amin(y_coords),np.amax(y_coords)
                            h = y_coord_max - y_coord_min
                            w = x_coord_max - x_coord_min
                            my_labels['orthogonal_bboxes'].append([x_coord_min+w/2.0, y_coord_min+h/2.0, w, h])
                            
                            # AIRPLANE EXIST 
                            my_labels['airplane_exist']=True
                    
                    y_limit_expanded = y_0+patch_size>=y_max
                    x_limit_expanded = x_0+patch_size>=x_max
                    limit_expanded = y_limit_expanded and x_limit_expanded
                    if limit_expanded:
                        patch[:y_max-y_0,:x_max-x_0] = img[y_0:y_max,x_0:x_max]
                    elif y_limit_expanded:
                        patch[:y_max-y_0,:] = img[y_0:y_max,x_0:x_0+patch_size]
                    elif x_limit_expanded:
                        patch[:,:x_max-x_0] = img[y_0:y_0+patch_size,x_0:x_max]                        
                    else:
                        patch[:,:] = img[y_0:y_0+patch_size,x_0:x_0+patch_size]
                    # if my_labels['airplane_exist']:
                        # print(my_labels)
                        # print(my_labels['orthogonal_bboxes'])
                        # self.img_show(img=patch,bboxes=my_labels['rotated_bboxes'],rotated=True)#my_labels)
                        # self.img_show(img=patch,bboxes=my_labels['orthogonal_bboxes'],rotated=False)#my_labels)


                    patch_name = f"{img_name}_x_{x_0}_y_{y_0}"
                    ### save img
                    cv2.imwrite(f"{img_patch_folder}/{patch_name}.png",patch)

                    ### save label
                    with open(f"{label_patch_folder}/{patch_name}.json", 'w') as f:
                        json.dump(my_labels, f,indent=4)

                    ### save label for yolo
                    with open(f"{label_patch_yolo_folder}/{patch_name}.txt", 'w') as f:
                        for bboxes_params in my_labels['orthogonal_bboxes']:
                            params_str = ['0'] # class_id center_x center_y width height
                            for param in bboxes_params:
                                param_norm = param/patch_size
                                if param_norm>1:
                                    param_norm=1
                                elif param_norm<0:
                                    param_norm=0
                                params_str.append(str(param_norm))
                            my_line = ' '.join(params_str)
                            f.write(f"{my_line}\n")
                    # break
        # return patches

    def get_patch_start_coords(self,my_max,patch_size,overlap):

        def get_values(coord_max):
            coords=[]
            coord_0 = 0
            while coord_0 < coord_max:
                coords.append(coord_0)
                coord_0 = coord_0 + patch_size - overlap
                if coord_0+patch_size>coord_max:
                    coords.append(coord_max-patch_size)
                    break
            return coords        
        
        y_max,x_max=my_max

        patch_y = get_values(y_max)
        patch_x = get_values(x_max)

        return patch_y, patch_x

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
        M = rect.get_rotation_matrix_upwards(img_shape=img_1.shape)
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
        patch_dict['notes']['airplane_features'] = ['center_x,center_y,height,width,rotation_angle']

        patch_dict = self.get_img_patch(img=img,rect=rect,patch_dict=patch_dict)
        return patch_dict, rect


    def get_patches_in_file(self,img_path,label_path,patch_size,save=False):


        
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
                            'rotated_bbox_patch':[],
                            'bbox_params':[],
                            'patch_size':None,
                            'original': {'img_path':None, 'center_padded':None, 'pad_size':0},
                            'notes':{'airplane_features':None}}
            
            patch_dict['original']['img_path']=img_path
            patch_dict['instance_name']=label["names"][i]
            patch_dict['original']['pad_size']=pad_size
            patch_dict['patch_size']=patch_size

            patch_dict, rect = self.get_patch_label(img,patch_dict,bbox)

            # print(patch_dict['rotated_bbox_patch'])

            ### PLOT
            fig,ax = plt.subplots(2)
            ax[0].imshow(cv2.cvtColor(patch_dict['img_patch'],cv2.COLOR_BGR2RGB))
            # # ax[0].set_ylim(ax[0].get_ylim()[::-1]) # invert y axis
            # # self.plot_rotated_box(patch_dict['patch']['rotated_bboxes'],ax[0])
            rect.plot_corners(ax[0])
            # # rect.plot_contours(ax[0],rotate=False)
            ax[1].imshow(cv2.cvtColor(patch_dict['img_patch_rotated'],cv2.COLOR_BGR2RGB))
            # # ax[1].set_ylim(ax[1].get_ylim()[::-1])
            rect.plot_contours(ax[1],rotate=False)
            # plt.show()

            ### SAVE FIGURES
            if save:

                # ax.set_title(f"{instance_name}, h:{rotated_rect.h:.1f}, w:{rotated_rect.w:.1f}",fontsize=24)
                ## FILE NAMES
                file_name = get_file_name_from_path(img_path)
                patch_name = f"{file_name}_{i}"

                os.makedirs(f"{self.patch_folder_base}/figures",exist_ok=True)
                plt.savefig(f"{self.patch_folder_base}/figures/{patch_name}.png", bbox_inches='tight')

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

            # break
            
        # return patch_dict


if __name__ == "__main__":
    import random
    
    import matplotlib.pyplot as plt    
    import os



    ### SAVE PATCHES (RECOGNITION)
    patch_size=256
    train_data = RecognitionData(dataset_name='train')
    train_data.set_patch_folders(patch_size=patch_size)
    # train_data.get_patches(save=False) 

    # paths = train_data.get_img_paths()


    # print(paths)
    # problem ones: 111,148 ---- 200 den devam et
    # for no in ['283','966','111','54']:
    for no in ['111','148','100','56','20','5']: 
    # for no in paths.keys():
        # print(no)
        img_path = f'/home/murat/Projects/airplane_detection/DATA/Gaofen/train/images/{no}.tif'
        label_path = f'/home/murat/Projects/airplane_detection/DATA/Gaofen/train/label_xml/{no}.xml'


        train_data.get_patches_in_file(img_path,label_path,patch_size=patch_size,save=True)

        # train_data.plot_bboxes(img_path,label_path)


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