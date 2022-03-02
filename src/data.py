import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import xml.etree.ElementTree as ET
import os
import matplotlib.pyplot as plt
import json
import cv2
import math
import matplotlib.patches as patches

class Data():
    def __init__(self,dataset_name,patch_size):
        self.project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_folder = f"{self.project_folder}/DATA/Gaofen"
        self.dataset_name = dataset_name
        self.json_file = json.load(open(f"{self.data_folder}/sequences.json",'r'))

        img_paths = self.get_img_paths()
        labels = self.get_labels()
        self.save_patches(img_paths=img_paths,labels=labels,patch_size=patch_size)


    def save_patches(self,img_paths,labels,patch_size=512):

        patches = []
        box_corner_threshold = 2

        img_patch_folder = f"{self.data_folder}/{self.dataset_name}/images_{patch_size}"
        os.makedirs(img_patch_folder,exist_ok=True)
        label_patch_folder = f"{self.data_folder}/{self.dataset_name}/labels_{patch_size}"
        os.makedirs(label_patch_folder,exist_ok=True)

        for img_name, img_path in img_paths.items():
            # img = cv2.cvtColor(cv2.imread(img_path,1),cv2.COLOR_BGR2RGB)
            img = cv2.imread(img_path,1)
            bbox_coords = labels[img_name]['bbox']
            instance_names = labels[img_name]['names']
            y_max, x_max, ch = img.shape[:3]
            patch_y = int(math.ceil(y_max/patch_size))
            patch_x = int(math.ceil(x_max/patch_size))

            for p_y in range(patch_y):
                for p_x in range(patch_x):

                    my_labels = {'instance_names':[],'rotated_bboxes':[], 'airplane_exist':False} # ,'orthogonal_bboxes':[]
                    patch = np.zeros(shape=(patch_size,patch_size,ch),dtype=np.uint8)
                    y_0 = p_y*patch_size
                    x_0 = p_x*patch_size

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
                            # x_coords = shifted_coords[:,0]
                            # y_coords = shifted_coords[:,1]
                            # x_coord_min, x_coord_max = np.amin(x_coords),np.amax(x_coords)
                            # y_coord_min, y_coord_max = np.amin(y_coords),np.amax(y_coords)
                            # my_labels['orthogonal_bboxes'].append([[x_coord_min,y_coord_min],[x_coord_max,y_coord_max]])
                            # airplane exist 
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
                        # self.img_show(img=patch,bboxes=my_labels['rotated_bboxes'],rotated=False)#my_labels)

                    patch_name = f"{img_name}_x_{x_0}_y_{y_0}"
                    ### save img
                    cv2.imwrite(f"{img_patch_folder}/{patch_name}.png",patch)

                    ### save label
                    with open(f"{label_patch_folder}/{patch_name}.json", 'w') as f:
                        json.dump(my_labels, f,indent=4)
            # print(patch_name,my_labels)
            # break
        # return patches

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

    def get_labels(self):
        labels={}

        label_folder = f"{self.data_folder}/{self.dataset_name}/label_xml"
        label_file_paths = [f"{label_folder}/{file_name}" for file_name in os.listdir(label_folder)]


        for label_file_path in label_file_paths:
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
            labels[img_name]=label
            # break
        return labels

    def img_show(self,img,bboxes,rotated):
        # sample = self.__getitem__(ind)
        # img = sample['image']
        # labels = sample['labels']
        # print(labels)
        fig, ax = plt.subplots(1)
        ax.imshow(img,'gray')

    # if rotated:
        for coords in bboxes:
            for i, coord in enumerate(coords):
                # PLOT BBOX
                ax.plot([coords[i-1][0],coord[0]],[coords[i-1][1],coord[1]],c='r')
                # PLOT CORNERS
                # ax.scatter(coord[0],coord[1],c='r',s=5)
    # else:
        # for coords in bboxes:
        #     # for i, coord in enumerate(coords):
        #     x_0 = coords[0][0]
        #     y_0 = coords[0][1]
        #     w = coords[1][0] - x_0
        #     h = coords[1][1] - y_0
        #     rect = patches.Rectangle((x_0, y_0), w, h, linewidth=1, edgecolor='r', facecolor='none')

                # Add the patch to the Axes
            # ax.add_patch(rect)
        plt.show()


if __name__ == "__main__":
    # import random
    patch_size=512
    train_data = Data(dataset_name='train',patch_size=patch_size)
    train_data = Data(dataset_name='val',patch_size=patch_size)
    # print(len(train_data))
    # print(train_data.get_labels())
    # print(train_data.get_img_paths())
    # print(train_data.items)
    # print(train_data[5]['labels'])
    # ind = random.randint(0,len(train_data)-1)
    # train_data.img_show(ind=ind,plot_bbox=True)