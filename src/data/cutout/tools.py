import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json
# import copy

from . import geometry
# LEave some margin for the cutoutes, because some airplane in DOTA has
# cutoff parts

class Tools(object):
    """docstring for Tools"""

    def __init__(self, settings):
        super(Tools, self).__init__()

        self.settings = settings
        # self.cutout_size = self.utils.settings['cutout']['size']

        # PAD SIZE (PAD ORIGINAL IMAGE AND FIRST CUTOUT)
        self.pad_size = 500#int(self.cutout_size * 1.5)

        # SEGMENTATION PATCHES
        self.segmentation_task = 'seg' in settings['tasks']

        # PATCH TOOLS
    def get_original_image(self, img_path, flags=1):
        if img_path is None:
            return None
        img = cv2.imread(img_path, flags=flags)
        if flags != 0:
            img = np.pad(
                img,
                ((self.pad_size,
                  self.pad_size),
                 (self.pad_size,
                  self.pad_size),
                    (0,
                     0)),
                'constant',
                constant_values=0)  # 'symmetric')#
        else:
            img = np.pad(
                img,
                ((self.pad_size,
                  self.pad_size),
                 (self.pad_size,
                  self.pad_size)),
                'constant',
                constant_values=0)  # 'symmetric')#
        return img

    def init_cutout_dict(self, 
        instance_name, 
        instance_id,
        img_path, 
        mask_path=None):
        cutout_dict = {
            'file_path': None,
            'instance':{
                'name': None,
                'id':None,
                },
            # 'cutout_size': None,
            'original': {
                'img': None,
                'mask': None,
                'bbox': 
                    {'corners':[],
                    'params':[]
                    },
                'mask_path': None,
                'img_path': None,
                # 'center_padded':None,
                'margin': 0
            },

            'original_padded_cutout': {
                'img': None,
                'mask': None,
                'bbox': 
                    {'corners':[],
                    'params':[]
                    },
                'bbox_params': [],
                'margin': 0
            },

            'original_cutout': {
                'img': None,
                'mask': None,
                'bbox': 
                    {'corners':[],
                    'params':[]
                    },
                'img_path': None,
                'mask_path': None,
                'margin': 0
            },


            'orthogonal_cutout': {
                'img': None,
                'mask': None,
                'bbox': 
                    {'corners':[],
                    'params':[]
                    },
                'img_path': None,
                'mask_path': None,
                'margin': 0
            },

            'orthogonal_zoomed_cutout':
            {
                'img': None,
                'mask': None,
                'bbox': 
                    {'corners':[],
                    'params':[]
                    },
                'img_path': None,
                'mask_path': None,
                'margin': 0
            },

            'notes': {
                # 'bbox_params':None,
                'bbox_corners': None,
            },
        }
        cutout_dict['instance']['name'] = instance_name
        cutout_dict['instance']['id'] = instance_id
        cutout_dict['original']['img_path'] = img_path
        cutout_dict['original']['mask_path'] = mask_path
        # cutout_dict['cutout_size'] = self.cutout_size
        cutout_dict['notes']['bbox_corners'] = [
            '[airplane_top_left_xy,airplane_bottom_left_xy,airplane_bottom_right_xy,airplane_top_right_xy]']

        return cutout_dict

    def set_cutout_params(self, cutout_dict, img, bbox, mask=None):


        # NOTES
        # cutout_dict['notes']['bbox_params'] = ['center_x,center_y,height,width,rotation_angle']

        cutout_dict = self.set_original(cutout_dict, img, bbox, mask, margin=self.pad_size) # self.pad_size
        cutout_dict = self.set_original_padded_cutout(cutout_dict, margin=100)
        cutout_dict = self.set_original_cutout(cutout_dict, margin=20)
        cutout_dict = self.set_orthogonal_cutout(cutout_dict, margin=20)
        cutout_dict = self.set_orthogonal_zoomed_cutout(cutout_dict, margin=10)

        return cutout_dict

    def set_original(self, cutout_dict, img, bbox, mask, margin):
        # IMAGE
        cutout_dict['original']['img'] = img

        # BBOX
        bbox_orig_padded = np.array(bbox) + margin  # add initial padding
        cutout_dict['original']['bbox']['corners'] = bbox_orig_padded
        cutout_dict['original']['bbox']['params'] = self.get_bbox_params(bbox_orig_padded)

        # print(cutout_dict['original']['bbox']['corners'])

        # MASK
        cutout_dict['original']['mask'] = mask

        # MARGIN
        cutout_dict['original']['margin'] = margin
        # center = np.mean(bbox,axis=0).astype(int)
        # center_padded = np.mean(bbox_orig_padded,axis=0).astype(int)#center+self.pad_size
        # cutout_dict['original']['center_padded']=center_padded

        return cutout_dict

    def set_original_padded_cutout(self, cutout_dict, margin):
        img = cutout_dict['original']['img']
        mask = cutout_dict['original']['mask']
        bbox = cutout_dict['original']['bbox']['corners']

        mask = self.remove_other_instances_from_mask(mask, bbox) if self.segmentation_task else mask
        img_1, mask_1, bbox_1 = self.cut_image_by_bbox(img, mask, bbox, margin)


        cutout_dict['original_padded_cutout']['img'] = img_1
        cutout_dict['original_padded_cutout']['mask'] = mask_1
        cutout_dict['original_padded_cutout']['bbox']['corners'] = bbox_1
        cutout_dict['original_padded_cutout']['bbox']['params'] = self.get_bbox_params(bbox_1)
        cutout_dict['original_padded_cutout']['margin'] = margin

        return cutout_dict

    def set_original_cutout(self, cutout_dict, margin=0):
        img = cutout_dict['original_padded_cutout']['img']
        mask = cutout_dict['original_padded_cutout']['mask']
        bbox = cutout_dict['original_padded_cutout']['bbox']['corners']

        img_1, mask_1, bbox_1 = self.cut_image_by_bbox(img, mask, bbox, margin)

        cutout_dict['original_cutout']['img'] = img_1
        cutout_dict['original_cutout']['mask'] = mask_1
        cutout_dict['original_cutout']['bbox']['corners'] = bbox_1
        cutout_dict['original_cutout']['bbox']['params'] = self.get_bbox_params(bbox_1)
        cutout_dict['original_cutout']['margin'] = margin

        return cutout_dict

    def set_orthogonal_cutout(self, cutout_dict, margin=0):

        img = cutout_dict['original_padded_cutout']['img']
        mask = cutout_dict['original_padded_cutout']['mask']
        bbox = cutout_dict['original_padded_cutout']['bbox']['corners']

        rect = geometry.BBox(corners=bbox)


        angle = rect.get_orth_angle()
        cx,cy = rect.get_params()[0:2]
        ### cv2.getRotationMatrix2D(center, angle, transform)
        M = cv2.getRotationMatrix2D((cx, cy), np.rad2deg(angle), 1.0)
        ### cv2.warpAffine(img, rotation, dest_size)
        img_rotated = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]))
        mask_rotated = cv2.warpAffine(
            mask,
            M,
            (img.shape[0],
             img.shape[1]),
            flags=cv2.INTER_NEAREST) if self.segmentation_task else None
        bbox_rotated = rect.get_orthogonal_bbox()

        img_1, mask_1, bbox_1 = self.cut_image_by_bbox(
            img_rotated, mask_rotated, bbox_rotated, margin)

        cutout_dict['orthogonal_cutout']['img'] = img_1
        cutout_dict['orthogonal_cutout']['mask'] = mask_1
        cutout_dict['orthogonal_cutout']['bbox']['corners'] = bbox_1
        cutout_dict['orthogonal_cutout']['bbox']['params'] = self.get_bbox_params(bbox_1) 
        cutout_dict['orthogonal_cutout']['margin'] = margin
        return cutout_dict

    def set_orthogonal_zoomed_cutout(self, cutout_dict, margin=0):

        img = cutout_dict['orthogonal_cutout']['img']
        mask = cutout_dict['orthogonal_cutout']['mask']
        bbox = cutout_dict['orthogonal_cutout']['bbox']['corners']

        img_1, mask_1, bbox_1 = self.cut_image_by_bbox(img, mask, bbox, margin)

        cutout_dict['orthogonal_zoomed_cutout']['img'] = img_1
        cutout_dict['orthogonal_zoomed_cutout']['mask'] = mask_1
        cutout_dict['orthogonal_zoomed_cutout']['bbox']['corners'] = bbox_1
        cutout_dict['orthogonal_zoomed_cutout']['bbox']['params'] = self.get_bbox_params(bbox_1) 
        cutout_dict['orthogonal_zoomed_cutout']['margin'] = margin

        return cutout_dict

    def cut_image_by_bbox(self, img, mask, bbox, margin):
        x_min, x_max, y_min, y_max = geometry.BBox.get_bbox_limits(corners=bbox)
        # print(f"Limits: {x_min, x_max, y_min, y_max}")
        y_0, y_1 = np.array([y_min - margin, y_max + margin]).astype(int)
        x_0, x_1 = np.array([x_min - margin, x_max + margin]).astype(int)
        img_1 = img[y_0:y_1, x_0:x_1, :]
        bbox_1 = bbox - [x_min, y_min] + margin
        mask_1 = mask[y_0:y_1, x_0:x_1] if self.segmentation_task else None
        return img_1, mask_1, bbox_1

    def set_paths(self, dataset_part, cutout_dict, i):
        # PATCH FOLDER SETTINGS
        img_cutout_folder = self.settings['cutout'][dataset_part]['image_folder']
        img_cutout_orthogonal_folder = self.settings['cutout'][dataset_part]['orthogonal_image_folder']
        img_cutout_orthogonal_zoomed_folder = self.settings['cutout'][dataset_part]['orthogonal_zoomed_image_folder']

        # FILE NAMES
        img_path = cutout_dict['original']['img_path']
        file_name = self.get_file_name_from_path(img_path)
        cutout_name = f"{file_name}_{i}"
        cutout_img_name = f"{cutout_name}.png"

        def cutout_img_path(folder): return os.path.join(folder, cutout_img_name)

        # SET IMAGE PATHS
        cutout_dict['original_cutout']['img_path'] = cutout_img_path(
            img_cutout_folder)
        cutout_dict['orthogonal_cutout']['img_path'] = cutout_img_path(
            img_cutout_orthogonal_folder)
        cutout_dict['orthogonal_zoomed_cutout']['img_path'] = cutout_img_path(
            img_cutout_orthogonal_zoomed_folder)

        # SET MASK PATHS

        if self.segmentation_task:
            mask_cutout_folder = self.settings['cutout'][dataset_part]['mask_folder']
            mask_cutout_orthogonal_folder = self.settings[
                'cutout'][dataset_part]['orthogonal_mask_folder']
            mask_cutout_orthogonal_zoomed_folder = self.settings[
                'cutout'][dataset_part]['orthogonal_zoomed_mask_folder']
            cutout_dict['original_cutout']['mask_path'] = cutout_img_path(
                mask_cutout_folder)
            cutout_dict['orthogonal_cutout']['mask_path'] = cutout_img_path(
                mask_cutout_orthogonal_folder)
            cutout_dict['orthogonal_zoomed_cutout']['mask_path'] = cutout_img_path(
                mask_cutout_orthogonal_zoomed_folder)

        # SET LABEL PATH
        label_path = os.path.join(
            self.settings['cutout'][dataset_part]['label_folder'],
            f'{cutout_name}.json')
        cutout_dict['file_path'] = label_path
        return cutout_dict

    # def skip_existing_cutout(self,cutout_dict):
    #     return os.path.exists(cutout_dict['original_cutout']['img_path'])

    def save_cutout(self, dataset_part, cutout_dict, i):

        cutout_dict = self.set_paths(dataset_part, cutout_dict, i)

        cv2.imwrite(
            cutout_dict['original_cutout']['img_path'],
            cutout_dict['original_cutout']['img'])
        cv2.imwrite(
            cutout_dict['orthogonal_cutout']['img_path'],
            cutout_dict['orthogonal_cutout']['img'])
        cv2.imwrite(
            cutout_dict['orthogonal_zoomed_cutout']['img_path'],
            cutout_dict['orthogonal_zoomed_cutout']['img'])

        if self.segmentation_task:
            cv2.imwrite(
                cutout_dict['original_cutout']['mask_path'],
                cutout_dict['original_cutout']['mask'])
            cv2.imwrite(
                cutout_dict['orthogonal_cutout']['mask_path'],
                cutout_dict['orthogonal_cutout']['mask'])
            cv2.imwrite(
                cutout_dict['orthogonal_zoomed_cutout']['mask_path'],
                cutout_dict['orthogonal_zoomed_cutout']['mask'])

        # LABEL
        image_folder_keys = [
            'original',
            'original_padded_cutout',
            'original_cutout',
            'orthogonal_cutout',
            'orthogonal_zoomed_cutout']

        for folder_key in image_folder_keys:
            # Delete img and mask arrays before saving cutout_dict
            for img_key in ['img','mask']:
                del cutout_dict[folder_key][img_key]

            ## Bbox corners to list
            cutout_dict[folder_key]['bbox']['corners']=cutout_dict[folder_key]['bbox']['corners'].tolist()

        # print(cutout_dict)
        # print(cutout_dict)
        with open(cutout_dict['file_path'], 'w') as f:
            json.dump(cutout_dict, f, indent=4)


    def plot_cutout(self, ax, cutout_dict, conf=['original', 'img']):
        # PLOT
        # fig,ax = plt.subplots(1)
        cutout_conf = conf[0]
        img_conf = conf[1]
        if img_conf == 'img':
            # original_cutout, orthogonal_cutout, orthogonal_zoomed_cutout
            ax.imshow(
                cv2.cvtColor(
                    cutout_dict[f'{cutout_conf}_cutout'][img_conf],
                    cv2.COLOR_BGR2RGB))
        elif img_conf == 'mask':
            # original_cutout, orthogonal_cutout, orthogonal_zoomed_cutout
            ax.imshow(cutout_dict[f'{cutout_conf}_cutout'][img_conf])

        geometry.BBox.plot_bbox(
            corners=cutout_dict[f'{cutout_conf}_cutout']['bbox']['corners'], ax=ax, c='b')

        instance_name = cutout_dict['instance']['name']
        ax.set_title(instance_name)

    def get_file_name_from_path(self, path):
        return os.path.splitext(os.path.split(path)[-1])[0]

    def remove_other_instances_from_mask(self, mask, bbox):
        label_image = np.zeros(
            shape=(
                mask.shape[0],
                mask.shape[1],
                3))  # ,dtype=np.uint8)

        center = np.mean(bbox, axis=0).astype(int)
        try:
            plane_index = np.array([mask[center[1], center[0], :]])
        except BaseException:
            plane_index = np.array([0, 0, 0])
        # np.logical_and(plane_index,np.array([0,0,0]))
        is_background = (plane_index == np.array([0, 0, 0])).all()
        # print(is_background)
        if is_background:
            _, bbox_image, bbox_1 = self.cut_image_by_bbox(
                mask, mask, bbox, margin=-10)

            label_values = np.unique(
                bbox_image.reshape(-1, bbox_image.shape[2]), axis=0)
            label_counts = {}
            for i, label_value in enumerate(label_values):  # Remove [0,0,0]
                # print(label_value)
                label_counts[i] = np.count_nonzero(
                    (bbox_image == label_value).all(axis=2))
            try:
                max_count_index = np.argmax(list(label_counts.values())[1:])
                plane_index = label_values[max_count_index + 1]
            except BaseException:
                print(f'No plane found, so the label is {plane_index}')
        # else:
        label_image = np.where(
            mask == plane_index, (255, 255, 255), 0)[:, :, 0]

        # fig, ax = plt.subplots(2)
        # ax[0].imshow(img)
        # ax[1].imshow(label_image)
        # ax[2].imshow(mask_2)
        # plt.show()
        # print(num_labels)
        # print(label_image.shape)
        # print(plane_index)
        # print(bbox)

        return label_image.astype(np.uint8)

    def get_bbox_params(self,bbox_corners):
        bbox = geometry.BBox(corners=bbox_corners)
        return bbox.params