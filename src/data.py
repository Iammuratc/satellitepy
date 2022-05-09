import cv2
import numpy as np
import json
import xml.etree.ElementTree as ET
import os
import json
import cv2
import math
import geometry

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

    def plot_rotated_box(self,corners,ax):
        # for coords in corners:
            # print(coords)
        for i, coord in enumerate(corners):
            # PLOT BBOX
            ax.plot([corners[i-1][0],coord[0]],[corners[i-1][1],coord[1]],c='y')


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

    def get_patch_folder(self,patch_size):
        return f"{self.data_folder}/{self.dataset_name}/patches_{patch_size}_recognition"


    def get_patch(self,img,rect,center,patch_size):
        '''
        Get the patch of the upwards facing airplane
        Original padded image (img) >> 2*patch_size image (img_1) >> rotate such that the airplane is facing upwards (img_2) >> patch_size image
        img: original img
        center: center of the original bounding boxes of the airplane [x,y]
        patch_size: desired patch size
        ''' 

        # Get the large image
        cx, cy = center
        img_1=img[cy-patch_size:cy+patch_size,cx-patch_size:cx+patch_size,:]
       

        # Get the rotation angle
        M = rect.get_rotation_matrix_upwards(img_shape=img_1.shape)
        img_2 = cv2.warpAffine(img_1, M, (patch_size*2, patch_size*2))

        # Get the image patch
        patch_half_size = int(patch_size/2)
        img_patch = img_2[patch_size-patch_half_size:patch_size+patch_half_size,patch_size-patch_half_size:patch_size+patch_half_size,:]
        return img_patch


    def get_patches(self,save=False,patch_size=256):
        '''
        Get patches from an original image
        '''
        # patch_half_size=int(patch_size/2)
        my_patch_folder = self.get_patch_folder(patch_size)
        os.makedirs(my_patch_folder,exist_ok=True)
        img_patch_folder = f"{my_patch_folder}/images"
        os.makedirs(img_patch_folder,exist_ok=True)
        label_patch_folder = f"{my_patch_folder}/labels"
        os.makedirs(label_patch_folder,exist_ok=True)

        ###
        

        for file_name in list(self.labels.keys())[1:]:
            ## INITIATE PATCH
            label_patch = { 'patch':{'instance_name':None,'rotated_bboxes':[],'airplane_features':[0,0], 'size':None,'notes':{'airplane_features':None}},
                            'original': {'img_path':None, 'center_padded':None, 'pad_size':0}}
            img_patch = np.zeros(shape=(patch_size,patch_size,3),dtype=np.uint8)

            ### GET ORIGINAL IMAGE
            img_path = self.img_paths[file_name]
            img = cv2.imread(img_path)
            ### pad the original image, so no patching problem for the planes on the edge of the image
            pad_size = patch_size
            img = np.pad(img,((pad_size,pad_size),(pad_size,pad_size),(0,0)),'constant',constant_values=0)#'symmetric')#
            ### GET LABELS
            label = self.labels[file_name]
            
            for i,corners_orig in enumerate(label['bbox']):
                # If not including _orig, the variable belongs to the patch 
                corners_orig_padded =np.array(corners_orig)+pad_size # add initial padding
                
                center = np.mean(corners_orig,axis=0).astype(int)
                center_padded = center+pad_size
                get_patch_coords = lambda coords: coords-center_padded+pad_size/2
                corners = get_patch_coords(corners_orig_padded)
                label_patch['patch']['instance_name']=label["names"][i]
                label_patch['patch']['rotated_bboxes']=corners.tolist()
                label_patch['original']['img_path']=img_path
                label_patch['original']['center_padded']=center_padded.tolist()
                label_patch['original']['pad_size']=pad_size
                label_patch['patch']['size']=patch_size

                rect = geometry.Rectangle(corners=corners)


                label_patch['patch']['airplane_features']= [int(patch_size/2),int(patch_size/2),rect.h,rect.w,rect.angle]
                label_patch['patch']['notes']['airplane_features'] = ['center_x,center_y,height,width,rotation_angle']

                img_patch = self.get_patch(img=img,rect=rect,center=center_padded,patch_size=patch_size)
                print(label_patch)

                ### PLOT

                fig,ax = plt.subplots(2)
                ax[0].imshow(cv2.cvtColor(img_patch,cv2.COLOR_BGR2RGB))
                ax[0].set_ylim(ax[0].get_ylim()[::-1]) # invert y axis
                # self.plot_rotated_box(label_patch['rotated_bboxes'],ax[0])
                rect.plot_contours(ax[0],rotate=False)
                # ax[1].imshow(cv2.cvtColor(img_patch,cv2.COLOR_BGR2RGB))
                # ax[1].set_ylim(ax[1].get_ylim()[::-1])
                plt.show()

                ### SAVE FIGURES
                if save:
                    # ax.set_title(f"{instance_name}, h:{rotated_rect.h:.1f}, w:{rotated_rect.w:.1f}",fontsize=24)
                    # plt.savefig(f"{my_patch_folder}/figures/{file_name}_{i}.png", bbox_inches='tight')
                    patch_name = f"{file_name}_{i}"
                    cv2.imwrite(f"{img_patch_folder}/{patch_name}.png",img_patch)
                    ### save label
                    with open(f"{label_patch_folder}/{patch_name}.json", 'w') as f:
                        json.dump(label_patch, f,indent=4)

                # break
            # break

                

    def get_airplane_size(self,patch_size=128):

        my_patch_folder = self.get_patch_folder(patch_size)
        label_patch_folder = f"{my_patch_folder}/labels"

        size_dict = {}
        for json_file in os.listdir(label_patch_folder):
           label_patch = json.load(open(f"{label_patch_folder}/{json_file}",'r'))
           h,w = label_patch['size_hw']
           instance_name=label_patch['instance_name']
           
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
            # plt.savefig(f"{my_patch_folder}/figures/hist_{instance_name}.png", bbox_inches='tight')

    def rotate_airplane(self,corners):
        pass
    def rescale_airplane(self,patch_size):
        pass

if __name__ == "__main__":
    import random
    
    import matplotlib.pyplot as plt    




    ### SAVE PATCHES (RECOGNITION)
    train_data = RecognitionData(dataset_name='train')
    train_data.get_patches(save=False) 
    # train_data.get_airplane_size()


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