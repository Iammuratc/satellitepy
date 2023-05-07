import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import traceback
import xml.etree.ElementTree as ET
from multiprocessing import Pool

from satellitepy.data.cutout.geometry import BBox
from satellitepy.data.cutout.tools import Tools

# SEGMENTATION DATA


class Cutout(Tools):
    def __init__(self, settings, dataset_part):
        super(Cutout, self).__init__(settings)

        self.project_folder = settings['project_folder']
        self.dataset_part = dataset_part
        # ORIGINAL DATASET FOLDER SETTINGS
        self.original_image_folder = self.settings['original'][dataset_part]['image_folder']
        self.bbox_rotation = self.settings['bbox_rotation']
        # self.original_binary_mask_folder = utils.settings['original'][dataset_part]['binary_mask_folder']
        # self.original_label_path = utils.settings['original'][dataset_part]['label_path']
        self.original_bbox_folder = self.settings['original'][dataset_part]['bounding_box_folder']

        if self.segmentation_task:
            self.original_instance_mask_folder = self.settings['original'][dataset_part]['instance_mask_folder']

    def get_cutout_dict(self, 
                        img, 
                        img_path, 
                        mask, 
                        mask_path, 
                        bbox_label, 
                        ind, 
                        instance_name,
                        #instance_id = 1
                        ):

        if self.bbox_rotation == 'clockwise':
            bbox = np.array(bbox_label[:8]).astype(float).astype(int).reshape(4, 2)
            # print(bbox)
            bbox_copy = bbox.copy()
            coord_1 = bbox_copy[1, :]
            coord_3 = bbox_copy[3, :]
            bbox[3, :] = coord_1
            bbox[1, :] = coord_3
            # print(bbox)
        elif self.bbox_rotation == 'counter-clockwise':
            bbox = np.array(bbox_label[:8]).astype(int).reshape(4, 2)

        # rectangle = geometry.BBox(bbox)

        cutout_dict = self.init_cutout_dict(
            instance_name=instance_name,
            # instance_id=instance_id,
            img_path=img_path,
            mask_path=mask_path)
        cutout_dict = self.set_cutout_params(cutout_dict, img, bbox, mask)
        return cutout_dict

    def get_cutouts(self, save, plot, indices='all', multi_process=False):  # skip_existing

        image_paths = self.get_file_paths(self.original_image_folder)
        mask_paths = self.get_file_paths(self.original_instance_mask_folder) if self.segmentation_task else [None]*len(image_paths)
        bbox_paths = self.get_file_paths(self.original_bbox_folder)

        if save:
            ans = input(f'Cutouts are saved for the following folder:\n{self.original_image_folder}\nDo you confirm? [y/n] ')

            if ans != 'y':
                print('Please confirm it if you want to save the cutouts')
                return 0

            # if 'class' in settings['tasks']:
            #     ans = input(f'imagenet type labels will be appended to {self.settings['cutout']['imagenet_label_file']}')
            #     if ans != 'y':
            #         print('Please confirm if you want to save the cutouts')
            #     return 0
        if multi_process:
            with Pool(os.cpu_count()) as pool:
                pool.starmap(self.get_cutout,[[img_path,mask_paths[i],bbox_paths[i],i,save,plot,indices] for i, img_path in enumerate(image_paths)])

        else:
            for i, img_path in enumerate(image_paths):
                self.get_cutout(img_path,mask_paths[i],bbox_paths[i],i,save,plot,indices)
                # break
            # break

    def get_cutout(self,img_path,mask_path,bbox_path,i,save,plot,indices):
            if indices == 'all':
                pass
            elif i in indices:
                pass
            else:
                return 1

            try:
                # IMAGE
                # Add padding before passing to the CutoutTools because of the
                # cropping steps
                img = self.get_original_image(img_path)  # cv2.imread(img_path)
                print(img_path)
                # BBOXES
                # bbox_path = bbox_paths[i]
                bbox_labels = self.get_bbox_labels(bbox_path)
                # MASK
                # mask_path = mask_paths[i]
                mask = self.get_original_image(mask_path, flags=1)
                # binary_mask = np.zeros_like(mask)
                # cv2.inRange(mask,plane_pixel_value,plane_pixel_value,binary_mask)

                bbox_ind = 0
                for bbox_label in bbox_labels:
                    # print(bbox_label)
                    instance_name = bbox_label[-1]
                    if instance_name not in self.settings['instance_names'].keys():
                        continue

                    # instance_id = self.settings['instance_names'][instance_name]
                    cutout_dict = self.get_cutout_dict(
                        img=img,
                        img_path=img_path,
                        mask=mask,
                        mask_path=mask_path,
                        bbox_label=bbox_label,
                        ind=bbox_ind,
                        instance_name=instance_name,
                        # instance_id=instance_id
                        )
                    # print(cutout_dict)
                    if plot:
                        fig, ax = plt.subplots(2, 3)  # ,sharex=True,sharey=True)
                        self.plot_cutout(ax[0, 0], cutout_dict,conf=['original', 'img'])
                        self.plot_cutout(ax[0, 1], cutout_dict,conf=['orthogonal', 'img'])
                        self.plot_cutout(ax[0, 2], cutout_dict, conf=['orthogonal_zoomed', 'img'])
                        # print(cutout_dict['original_padded_cutout']['bbox']['corners'])
                        # print(cutout_dict['orthogonal_cutout']['bbox']['corners'])
                        # self.plot_cutout(ax[1, 0], cutout_dict,conf=['original', 'mask'])
                        # self.plot_cutout(ax[1, 1], cutout_dict, conf=['orthogonal', 'mask'])
                        # self.plot_cutout(ax[1, 2], cutout_dict, conf=['orthogonal_zoomed', 'mask'])
                        plt.show()
                    if save:
                        self.save_cutout(self.dataset_part, cutout_dict, bbox_ind)

                    # ### PLOT

                    bbox_ind += 1
            except Exception:
                traceback.print_exc()


    def get_bbox_labels(self,bbox_path):
        """
        Return dota type bbox labels
        bbox_labels: list([x1, y1, x2, y2, x3, y3, x4, y4, instance_name, difficult])
        # Note: difficult is removed
        """
        bbox_labels=[]

        if self.settings['dataset_name'] == 'DOTA':
            with open(bbox_path, 'r') as f:
                for line in f.readlines()[2:]:
                    bbox_labels.append(line[:-1].split(' ')[:-1])
        elif self.settings['dataset_name'] == 'fair1m':
            root = ET.parse(bbox_path).getroot()

            # IMAGE NAME
            file_name = root.findall('./source/filename')[0].text
            # img_name = file_name.split('.')[0]

            # INSTANCE NAMES
            instance_names = root.findall(
                './objects/object/possibleresult/name')  # [0].text
            # BBOX CCORDINATES
            point_spaces = root.findall('./objects/object/points')
            for i,point_space in enumerate(point_spaces):
                my_points = point_space.findall(
                    'point')[:4]  # remove the last coordinate
                coords = []
                for my_point in my_points:
                    # [[[x1,y1],[x2,y2]],[[x1,y1]]]
                    coord = []
                    for point in my_point.text.split(','):
                        coord.append(float(point))
                    coords.append(coord)
                # label['bboxes'].append(coords)
                    bbox_label = [item for sublist in coords for item in sublist]
                    bbox_label.append(instance_names[i].text)
                bbox_labels.append(bbox_label)
        elif self.settings['dataset_name'] == 'rarePlanes':
            # INSTANCE NAMES
            roles = []
            point_spaces = []

            labels = json.load(open(bbox_path, 'r'))
            for annotation in labels['annotations']:
                roles.append(annotation['role'])
                point_spaces.append(annotation['segmentation'][0])
            coords = []

            for i, bbox in enumerate(point_spaces):
                A = (bbox[0], bbox[1])
                B = (bbox[2], bbox[3])
                C = (bbox[4], bbox[5])
                D = (bbox[6], bbox[7])

                vecBD = np.subtract(D, B)

                middle = np.add(B, np.divide(vecBD, 2))
                vec_to_C = np.subtract(C, middle)
                vec_to_A = np.subtract(A, middle)

                if len(bbox) == 10:
                    coord = [np.add(D, vec_to_A), np.add(D, vec_to_C), np.add(B, vec_to_C), np.add(B, vec_to_A)]  # real
                else:
                    coord = [np.add(B, vec_to_A), np.add(D, vec_to_A), np.add(D, vec_to_C), np.add(B, vec_to_C)] # synthetic
                coord = [item for arrays in coord for item in arrays]
                print(coord)
                coords.append(coord)
                bbox_label = coord
                bbox_label.append(roles[i])
                bbox_labels.append(bbox_label)

        # print(bbox_labels)
        return bbox_labels

    def count_instances(self):
        label_file_names = os.listdir(self.original_bbox_folder)
        instance_names = list(self.settings['instance_names'].keys())
        instance_number_dict = {instance_name:0 for instance_name in instance_names}
        for label_file_name in label_file_names:
            bbox_path = os.path.join(self.original_bbox_folder,label_file_name)
            bbox_labels = self.get_bbox_labels(bbox_path)

            for bbox_label in bbox_labels:
                instance_name = bbox_label[-1]
                if instance_name not in instance_names:
                    continue
                instance_number_dict[instance_name] += 1

        # print(instance_number_dict)
        instance_number_average_dict = {instance_name:0 for instance_name in list(self.settings['instance_names'].keys())}
        ### AVERAGE
        total_number = sum(instance_number_dict.values())
        for instance_name in instance_number_dict.keys():
            instance_number_average_dict[instance_name] = instance_number_dict[instance_name]/total_number

        print(instance_number_average_dict)
                # break
            # break

    def show_original_image_from_name(self,img_name, show_labels=True):
        img_path = os.path.join(self.original_image_folder,f'{img_name}.tif')
        print(img_path)
        bbox_path = os.path.join(self.original_bbox_folder,f'{img_name}.xml')
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # fig, ax = plt.subplots(1)
        fig = plt.figure(frameon=False)
        # fig.set_size_inches(w,h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img)
        # ax.imshow(mask, alpha=0.5)
        bbox_labels=self.get_bbox_labels(bbox_path)
        for bbox_label in bbox_labels:
            bbox_corners = np.array(bbox_label[:8]).astype(int).reshape(4, 2)
            instance_name=bbox_label[-1] if show_labels else None
            BBox.plot_bbox(corners=bbox_corners, ax=ax, c='b', s=5, instance_name=None)
        # plt.show()
        # if save_dir:
        plt.axis('off')
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        # img_name = os.path.splitext(os.path.split(img_path)[1])[0]
        plt.show()
        return fig

    def show_original_image(self, ind, show_labels=False, save_dir=None):
        # IMAGE PATHS
        image_paths = self.get_file_paths(self.original_image_folder)
        # print(image_paths)
        # MASK PATHS
        # mask_paths = self.get_file_paths(self.original_instance_mask_folder)

        # BBOX PATHS
        bbox_paths = self.get_file_paths(self.original_bbox_folder)

        # IMAGE
        img_path = image_paths[ind]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        print(img_path)

        # MASK
        # mask_path = mask_paths[ind]
        # mask = cv2.imread(mask_path, 1)

        # BBOXES
        bbox_path = bbox_paths[ind]
        bbox_labels=self.get_bbox_labels(bbox_path)
        # SHOW
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        # ax.imshow(mask, alpha=0.5)

        for bbox_label in bbox_labels:
            bbox_corners = np.array(bbox_label[:8]).astype(int).reshape(4, 2)
            instance_name=bbox_label[-1] if show_labels else None
            BBox.plot_bbox(corners=bbox_corners, ax=ax, c='b', s=5, instance_name=instance_name)
        # plt.show()
        # if save_dir:
        plt.axis('off')
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        # img_name = os.path.splitext(os.path.split(img_path)[1])[0]
        plt.show()
        # plt.savefig(os.path.join(save_dir,f'{img_name}.png'))

    def get_file_paths(self, folder, sort=True):
        file_paths = [os.path.join(folder, file)
                      for file in os.listdir(folder)]
        if sort:
            file_paths.sort()
        return file_paths

if __name__ == '__main__':
    pass
    # from satellitepy.settings.dataset import SettingsDataset
    # fair1m_settings = SettingsDataset(
    # dataset_name='fair1m',
    # dataset_parts=['train','val'], # 'train',
    # tasks=['bbox'],
    # bbox_rotation='counter-clockwise',
    # instance_names=[
    #     'Boeing787',
    #     'Boeing737',
    #     'Boeing747',
    #     'Boeing777',
    #     'A220',
    #     'A321',
    #     'A330',
    #     'A350',
    #     'ARJ21',
    #     'C919',
    #     'other-airplane'])()
    # print(fair1m_settings)
    # cutout = Cutout(fair1m_settings,'val')
    ### COUNT INSTANCES
    # cutout.count_instances()


    ### SAVE BBOX PLOTTED ORIGINAL IMAGE
    # cutout.show_original_image(ind=12)
    # fig = cutout.show_original_image_from_name('14912')
    # fig.savefig('temp.png')
    # PRINT FILE PATH
    # print(utils.get_file_paths(segmentation_cutout.original_image_folder))

    # SHOW ORIGINAL IMAGE
    # train index=561 image_name=P1114
    # train index=923 image_name=P1872
    # segmentation_cutout.show_original_image(5)

    # GET PATCHES
    # Skip 923 for train, there airplanes labeled but no image
    # segmentation_cutout.get_cutoutes(save=True,plot=False,indices=range(924,1000))

    # CHECK LARGE JSON LABEL DATA
    # labels = segmentation_cutout.get_labels() # dict_keys(['images', 'categories', 'annotations'])
    #labels['images'] = [{'id': 0, 'file_name': 'P0000.png', 'ins_file_name': 'P0000_instance_id_RGB.png', 'seg_file_name': 'P0000_instance_color_RGB.png'}]
    # labels['annotations'] =
    # print(labels['images'][0])
    # print(labels['annotations'][0])
