import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
import os
import json
import cv2
import math

# from utilities import show_sample


## TODO: Add patch size to json file

##TODO (Recognition): save images and labels
class Data:
    def __init__(self,dataset_name):
        self.project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_folder = f"{self.project_folder}/DATA/Gaofen"
        self.dataset_name = dataset_name
        self.json_file = json.load(open(f"{self.data_folder}/sequences.json",'r'))

        self.img_paths = self.get_img_paths()
        self.labels = self.get_labels()

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

    def get_label(self,label_file_path):
        label = {'bbox':[],'names':[]}
        root = ET.parse(label_file_path).getroot()

        ### IMAGE NAME
        file_name = root.findall('./source/filename')[0].text
        img_name = file_name.split('.')[0]
         
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
        return label, img_name

    def get_labels(self):
        labels={}

        label_folder = f"{self.data_folder}/{self.dataset_name}/label_xml"
        label_file_paths = [f"{label_folder}/{file_name}" for file_name in os.listdir(label_folder)]


        for label_file_path in label_file_paths:
            
            label, img_name =self.get_label(label_file_path) 
            labels[img_name]=label
            # break
        return labels

class RecognitionData(Data):
    def __init__(self,dataset_name):
        '''
        Save patches for the recognition task
        A patch is consisted of a single airplane. The airplane is located in the middle of the patch by using the center of its rotated bounding box.
        Rotated bounding box can be masked (i.e. masked=True) 
        '''
        super(RecognitionData, self).__init__(dataset_name)


    def save_patches(self,patch_size=128):
        patch_half_size=int(patch_size/2)
        patch_folder = f"{self.data_folder}/{self.dataset_name}/patches_{patch_size}_recognition"
        os.makedirs(patch_folder,exist_ok=True)
        img_patch_folder = f"{patch_folder}/images"
        os.makedirs(img_patch_folder,exist_ok=True)
        label_patch_folder = f"{patch_folder}/labels"
        os.makedirs(label_patch_folder,exist_ok=True)

        for file_name in list(self.labels.keys()):
            label_patch = {'instance_name':None,'rotated_bboxes':[],'original_img':None, 'original_center_pt':None}
            img_patch = np.zeros(shape=(patch_size,patch_size,3),dtype=np.uint8)

            img_path = self.img_paths[file_name]
            img = cv2.imread(img_path)
            img = np.pad(img,((patch_half_size,patch_half_size),(patch_half_size,patch_half_size),(0,0)),'constant',constant_values=0)
            label = self.labels[file_name]
            for i,corners in enumerate(label['bbox']):
                corners=np.array(corners)+patch_half_size
                instance_name = label["names"][i]
                center = np.mean(corners,axis=0).astype(int)
                print(center)
                label_patch['instance_name']=instance_name
                label_patch['rotated_bboxes']=corners-center+patch_half_size
                label_patch['original_img']=img_path
                label_patch['original_center_pt']=center

                img_patch=img[center[1]-patch_half_size:center[1]+patch_half_size,center[0]-patch_half_size:center[0]+patch_half_size,:]
                print(img_patch.shape)
                plt.imshow(img_patch)
                print(label_patch)
                plt.show()

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

if __name__ == "__main__":
    import random
    import geometry
    import matplotlib.pyplot as plt    

    def plot_rotated_box(corners,ax):
        # for coords in corners:
            # print(coords)
        for i, coord in enumerate(corners):
            # PLOT BBOX
            ax.plot([corners[i-1][0],coord[0]],[corners[i-1][1],coord[1]],c='r')


    ### SAVE PATCHES (RECOGNITION)
    train_data = RecognitionData(dataset_name='train')
    train_data.save_patches() 


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
    ### SAVE PATCHES
    # train_data.save_patches()
    # print('train data saved')
    # val_data.save_patches()
    # print('val data saved')
    # test_data.save_patches()
    # print('test data saved')

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