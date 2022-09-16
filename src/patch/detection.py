import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import json

from . import geometry 

## TODO: 
#   - Move save steps to a method
#   - FAIR1m: Skip non airplane images
#   - DOTA: Patch creation reads FAIR1m files (.xml) only, DOTA has a different format, modify it accordingly

#### DETECTION PATCH
class PatchDetection:
    
    def __init__(self,settings,dataset_part):
        # super(DetectionData, self).__init__(dataset_name)
        self.settings=settings
        self.dataset_part=dataset_part

    def get_patches(self,save=False,plot=False):
        ### PATCH CONFIGURATIONS
        box_corner_threshold = self.settings['patch']['box_corner_threshold']
        overlap = self.settings['patch']['overlap']
        patch_size= self.settings['patch']['size']

        ### PATCH FOLDERS
        img_patch_folder = self.settings['patch'][self.dataset_part]['img_patch_folder']
        label_patch_folder = self.settings['patch'][self.dataset_part]['label_patch_folder']
        label_patch_yolo_folder = self.settings['patch'][self.dataset_part]['label_patch_yolo_folder']

        ### ORIGINAL IMAGES
        image_folder = self.settings['dataset'][self.dataset_part]['image_folder']
        img_files = os.listdir(image_folder)
        label_folder = self.settings['dataset'][self.dataset_part]['label_folder'] 
        label_files = os.listdir(label_folder)
        airplane_labels=self.settings['patch']['label_names']
        dataset_name = self.settings['dataset']['name']
        for img_file in img_files:
            ### IMAGE
            img_path = os.path.join(image_folder,img_file)
            img = cv2.imread(img_path,1)

            ### LABEL
            img_name = os.path.splitext(img_file)[0]
            if  (dataset_name == 'Gaofen' or dataset_name == 'FAIR1m'):         

                label_path = os.path.join(label_folder,f'{img_name}.xml')
            else:
                label_path = os.path.join(label_folder,f'{img_name}.txt')

            labels = self.get_labels(label_path)

            instance_names = labels['names'] # the ships are the instances
            # check and if no ship is in the list, continue with the next image
            # 
            for airplane_label in airplane_labels: #going through the labels
                if airplane_label in instance_names: # if one of the wanted labels in the instance names, the patches are done
                    ### PATCH COORDINATES IN THE ORIGINAL IMAGE
                    y_max, x_max, ch = img.shape[:3]
                    patch_y, patch_x = self.get_patch_start_coords(my_max=[y_max,x_max],patch_size=patch_size,overlap=overlap)
                    #skip images with no airships, you need the instance_names for it
                    for y_0 in patch_y:
                        for x_0 in patch_x:
                            ### PROCESS ONE PATCH
                            patch_dict = {  'image':np.zeros(shape=(patch_size,patch_size,ch),dtype=np.uint8),
                                            'instance_names':[],
                                            'rotated_bboxes':[] ,
                                            'orthogonal_bboxes':[],
                                            'orthogonal_bbox_params':[]
                                            }

                            for i,bbox in enumerate(labels['bboxes']):
                                ## CHECK IF BBOX IN PATCH
                                box_corner_in_patch = 0
                                for coord in bbox:
                                    if (x_0<=coord[0]<=x_0+patch_size) and (y_0<=coord[1]<=y_0+patch_size):
                                        box_corner_in_patch += 1
                                if box_corner_in_patch>=box_corner_threshold:
                                    # INSTANCE NAMES
                                    patch_dict['instance_names'].append(instance_names[i])
                                    # shift coords
                                    shifted_bbox = np.array(bbox)-[x_0,y_0]
                                    # ROTATED BBOXES                     
                                    patch_dict['rotated_bboxes'].append(shifted_bbox.tolist())
                                    # ORTHOGONAL BBOXES
                                    rect = geometry.Rectangle(bbox=shifted_bbox)

                                    orthogonal_bbox_param = rect.get_orthogonal_bbox_by_limits(bbox=shifted_bbox,return_params=True)
                                    orthogonal_bbox = rect.get_orthogonal_bbox_by_limits(bbox=shifted_bbox,return_params=False)

                                    patch_dict['orthogonal_bbox_params'].append(orthogonal_bbox_param)
                                    patch_dict['orthogonal_bboxes'].append(orthogonal_bbox)
                                                        
                            y_limit_expanded = y_0+patch_size>=y_max
                            x_limit_expanded = x_0+patch_size>=x_max
                            limit_expanded = y_limit_expanded and x_limit_expanded
                            if limit_expanded:
                                patch_dict['image'][:y_max-y_0,:x_max-x_0] = img[y_0:y_max,x_0:x_max]
                            elif y_limit_expanded:
                                patch_dict['image'][:y_max-y_0,:] = img[y_0:y_max,x_0:x_0+patch_size]
                            elif x_limit_expanded:
                                patch_dict['image'][:,:x_max-x_0] = img[y_0:y_0+patch_size,x_0:x_max]                        
                            else:
                                patch_dict['image'][:,:] = img[y_0:y_0+patch_size,x_0:x_0+patch_size]
                        
                            ### PLOT
                            if plot:
                                self.plot_patch_dict(patch_dict)

                            ### SAVE
                            if save:
                                # self.save_plot_dict(patch_dict)
                                patch_name = f"{img_name}_x_{x_0}_y_{y_0}"
                                ### SAVE IMAGE
                                cv2.imwrite(os.path.join(img_patch_folder,f"{patch_name}.png"),patch_dict['image'])
                                del patch_dict['image'] # remove image before saving labels
                                ### SAVE LABEL
                                with open(os.path.join(label_patch_folder,f"{patch_name}.json"), 'w') as f:
                                    json.dump(patch_dict, f,indent=4)

                                ### SAVE YOLO LABEL
                                with open(os.path.join(label_patch_yolo_folder,f"{patch_name}.txt"), 'w') as f:
                                    for bbox_params in patch_dict['orthogonal_bbox_params']:
                                        params_str = ['0'] # class_id center_x center_y width height
                                        for param in bbox_params:
                                            param_norm = param/patch_size
                                            if param_norm>1:
                                                param_norm=1
                                            elif param_norm<0:
                                                param_norm=0
                                            params_str.append(str(param_norm))
                                        my_line = ' '.join(params_str)
                                        f.write(f"{my_line}\n")
                else:
                    continue
    def plot_patch_dict(self,patch_dict):
        if patch_dict['instance_names']:
            fig, ax = plt.subplots(1)
            ax.imshow(cv2.cvtColor(patch_dict['image'],cv2.COLOR_BGR2RGB)) # original_patch, orthogonal_patch, orthogonal_zoomed_patch
            for i,rotated_bbox in enumerate(patch_dict['rotated_bboxes']):
                rect = geometry.Rectangle(bbox=rotated_bbox)
                # rect.plot_bbox(bbox=rect.orthogonal_bbox,ax=ax,c='r')
                rect.plot_bbox(bbox=rotated_bbox,ax=ax,c='b')
                rect.plot_bbox(bbox=patch_dict['orthogonal_bboxes'][i],ax=ax,c='g')
            plt.show()

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


    def get_labels(self,label_path):
        # dota has only text files, so we need to change the code for the labels
        def is_float(number_as_string):
            try:
                float(number_as_string)
                return True
            except ValueError:
                return False
        label = {'bboxes':[],'names':[]}
        dataset_name = self.settings['dataset']['name']
        if  (dataset_name == 'Gaofen' or dataset_name == 'FAIR1m'):         
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
                label['bboxes'].append(coords)

        elif (dataset_name =='DOTA'):
            f = open(label_path,'r') # is label path direct to label?
            boxes=[]
            names=[]
            with open(label_path,'r') as f:
                bbox_labels = [line[:-1].split(' ') for line in f.readlines()[2:]]
                for bbox_label in bbox_labels:
                    category = bbox_label[-2] # get label of this box
                    bbox_label=[[float(i) for i in bbox_label[:-2]]] # turning the box coords to floats, but delete the last two indices because there are the label and the difficulty
                    bboxToSave=np.reshape(bbox_label,(4,2)).tolist()
                    names.append(category) # put them to lists
                    boxes.append(bboxToSave)
                    label['bboxes']=boxes
                    label['names']=names
            pass
        else:
            print('No label parsing function found.')
        return label#, img_name


if __name__ == '__main__':
    from settings import SettingsDetection
    settings = SettingsDetection(patch_size=256)()


    for dataset_part in ['train','test','val']:
        detection_patch = DetectionPatch(settings,dataset_part=dataset_part)
        detection_patch.save_patches()