import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

import geometry
# from data import DataDem
# import shapely.wkt
from patch_tools import PatchTools


#### SEGMENTATION DATA
class SegmentationPatch(PatchTools):
    def __init__(self,settings,dataset_part):

        patch_size=settings['patch']['size']
        super(SegmentationPatch, self).__init__(patch_size,task='segmentation')

        self.settings=settings
        self.dataset_part=dataset_part
        ### ORIGINAL DATASET FOLDER SETTINGS
        self.original_image_folder = settings['dataset'][dataset_part]['image_folder']
        # self.original_instance_mask_folder = settings['dataset'][dataset_part]['instance_mask_folder']
        self.original_binary_mask_folder = settings['dataset'][dataset_part]['binary_mask_folder']
        # self.original_label_path = settings['dataset'][dataset_part]['label_path']
        self.original_bbox_folder = settings['dataset'][dataset_part]['bounding_box_folder']

        self.bbox_rotation = settings['dataset']['bbox_rotation']

    def get_patch_dict(self,img,img_path,mask,mask_path,bbox_label,ind):

        category = bbox_label[-2]
        if self.bbox_rotation=='clockwise':
            bbox = np.array(bbox_label[:8]).astype(int).reshape(4,2)
            # print(bbox)
            bbox_copy = bbox.copy()
            coord_1 = bbox_copy[1,:]
            coord_3 = bbox_copy[3,:] 
            bbox[3,:] = coord_1
            bbox[1,:] = coord_3
            # print(bbox)
        elif self.bbox_rotation=='counter-clockwise':
            bbox = np.array(bbox_label[:8]).astype(int).reshape(4,2)

        # rectangle = geometry.Rectangle(bbox)

        patch_dict = self.init_patch_dict(instance_name=category,img_path=img_path,mask_path=mask_path)
        patch_dict = self.set_patch_params(patch_dict,img,bbox,mask)
        return patch_dict

    def get_patches(self,save,plot):

        plane_pixel_value = 103.0 # the pixel value of airplanes in gray scale mask image

        image_paths = self.get_file_paths(self.original_image_folder)
        mask_paths = self.get_file_paths(self.original_binary_mask_folder)
        bbox_paths = self.get_file_paths(self.original_bbox_folder)

        for i,img_path in enumerate(image_paths):
            ## IMAGE 
            ## Add padding before passing to the PatchTools because of the cropping steps 
            img = self.get_original_image(img_path)#cv2.imread(img_path)
            ### BBOXES
            bbox_path = bbox_paths[i]
            with open(bbox_path,'r') as f:
                bbox_labels = [line[:-1].split(' ') for line in f.readlines()[2:]] #[x1, y1, x2, y2, x3, y3, x4, y4, category, difficult]
            ### MASK
            mask_path = mask_paths[i]
            mask = self.get_original_image(mask_path,flags=0)
            binary_mask = np.zeros_like(mask)
            cv2.inRange(mask,plane_pixel_value,plane_pixel_value,binary_mask)

            ### PLOT
            # fig,ax = plt.subplots(1)
            # ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # original_patch, orthogonal_patch, orthogonal_zoomed_patch

            bbox_ind=0
            for bbox_label in bbox_labels:
                category = bbox_label[-2]
                if category != 'plane':
                    continue

                patch_dict = self.get_patch_dict(   img=img,
                                                    img_path=img_path,
                                                    mask=mask,
                                                    mask_path=mask_path,
                                                    bbox_label=bbox_label,
                                                    ind=bbox_ind
                                                    )

                
        
                if save:
                    self.save_patch(settings,dataset_part,patch_dict,bbox_ind)

                # ### PLOT
                if plot:
                    fig,ax = plt.subplots(2,sharex=True,sharey=True)
                    self.plot_patch(ax[0],patch_dict,conf=['orthogonal_zoomed','img'])
                    self.plot_patch(ax[1],patch_dict,conf=['orthogonal_zoomed','mask'])
                    plt.show()
                    
                bbox_ind += 1
                # break
            # break
        
    def get_labels(self):
        with open(self.original_label_path, 'r') as fp:
            labels = json.load(fp)
        return labels

    def get_file_paths(self,folder):

        file_paths = [os.path.join(folder,file) for file in os.listdir(folder)]
        file_paths.sort()
        return file_paths        

    def show_original_image(self,ind,mask=True):
        ### IMAGE PATHS
        image_paths = self.get_image_paths(self.original_image_folder)

        ### MASK PATHS
        mask_paths = self.get_image_paths(self.original_binary_mask_folder)

        ### IMAGE
        img_path = image_paths[ind]
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)

        ### MASK
        mask_path = mask_paths[ind]
        mask = cv2.imread(mask_path,0)

        ### SHOW
        fig,ax = plt.subplots(1)
        ax.imshow(img)
        ax.imshow(mask,alpha=0.5)
        plt.show()

if __name__ == '__main__':
    from settings import SettingsSegmentation

    settings = SettingsSegmentation(dataset_name='DOTA',
                                    patch_size=128)()
    
    dataset_part='train'
    segmentation_patch = SegmentationPatch(settings,dataset_part)
    # segmentation_patch.show_original_image(1)

    ### SAVE PATCHES
    segmentation_patch.get_patches(save=True,plot=False)


    ### CHECK LARGE JSON LABEL DATA
    # labels = segmentation_patch.get_labels() # dict_keys(['images', 'categories', 'annotations'])
    #labels['images'] = [{'id': 0, 'file_name': 'P0000.png', 'ins_file_name': 'P0000_instance_id_RGB.png', 'seg_file_name': 'P0000_instance_color_RGB.png'}]
    #labels['annotations'] =
    # print(labels['images'][0])
    # print(labels['annotations'][0])
