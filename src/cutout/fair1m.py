import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt

from . import geometry
# from data import DataDem
import shapely.wkt


from .tools import PatchTools

# RECOGNITION DATA


class RecognitionPatch(PatchTools):
    def __init__(self, settings, dataset_part):
        '''
        Save patches for the recognition task
        A patch is consisted of a single airplane. The airplane is located in the middle of the patch by using the center of its rotated bounding box.
        Rotated bounding box can be masked (i.e. masked=True)
        '''
        self.patch_size = settings['patch']['size']
        super(
            RecognitionPatch,
            self).__init__(
            self.patch_size,
            task='recognition')

        # self.img_patch_folder = settings['patch'][dataset_part]['img_patch_folder'] #f"{self.patch_folder_base}/images"
        # self.img_patch_orthogonal_folder = settings['patch'][dataset_part]['img_patch_orthogonal_folder']#f"{self.patch_folder_base}/orthogonal_images"
        # self.img_patch_orthogonal_zoomed_folder = settings['patch'][dataset_part]['img_patch_orthogonal_zoomed_folder']#f"{self.patch_folder_base}/orthogonal_zoomed_images"
        # self.label_patch_folder =
        # settings['patch'][dataset_part]['label_patch_folder']#f"{self.patch_folder_base}/labels"
        with open(settings['patch'][dataset_part]['json_file_path'], 'r') as fp:
            self.sequences = json.load(fp)

    def save_patches(self):
        # SAVE PATCHES

        ans = input("Do you really want to save the patches? ")
        if ans != 'y':
            print('If you want to save the patches, please confirm this with y.')
            return 0

        # DEM ADAPTED VERSION
        for s in self.sequences:
            for base_image in s["base_images"]:
                # GET ORIGINAL IMAGE
                img_path = base_image['image_path']
                img = self.get_original_image(img_path)
                for i, label in enumerate(base_image['ground_truth']):
                    self.get_patch(
                        img, img_path, label, ind=i, save=True, plot=False)

    def get_patch(self, img, img_path, label, ind, save=False, plot=False):

        bbox_polygon = shapely.wkt.loads(label['pixel_position'])
        bbox = np.array(bbox_polygon.exterior.coords)[0:4, :]
        instance_name = label['class']

        patch_dict = self.init_patch_dict(
            instance_name=instance_name, img_path=img_path)
        patch_dict = self.set_patch_params(patch_dict, img, bbox)
        if plot:
            self.plot_patch(patch_dict, ind)

        if save:
            self.save_patch(patch_dict, ind)

        # return patch_dict

    def get_patch_by_index(self, img_path, obj_ind, save=False, plot=True):
        for s in self.sequences:
            for base_image in s["base_images"]:
                # print(base_image['image_path'])
                if img_path == base_image['image_path']:
                    img = self.get_original_image(img_path)
                    label = base_image['ground_truth'][obj_ind]
                    self.get_patch(
                        img, img_path, label, obj_ind, save=save, plot=plot)

    def plot_all_bboxes_on_base_image(self, img_path):
        fig, ax = plt.subplots(1)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        for s in self.sequences:
            for base_image in s["base_images"]:
                # print(base_image['image_path'])
                if img_path == base_image['image_path']:
                    img = cv2.imread(img_path)
                    # original_patch, orthogonal_patch, orthogonal_zoomed_patch
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    labels = base_image['ground_truth']
                    for label in labels:
                        bbox_polygon = shapely.wkt.loads(
                            label['pixel_position'])
                        bbox = np.array(bbox_polygon.exterior.coords)[0:4, :]
                        geometry.Rectangle.plot_bbox(bbox=bbox, ax=ax, c='b')
        plt.show()


if __name__ == "__main__":
    import random

    import matplotlib.pyplot as plt
    import os

    from settings import SettingsRecognition

    settings = SettingsRecognition(patch_size=128, dataset_name='Gaofen')()
    recognition = RecognitionPatch(settings, dataset_part='train')

    # SAVE PATCHES FOR A DATASET PART
    # recognition.save_patches()

    # PLOT ALL THE BOOX ON AN IMAGE
    recognition.plot_all_bboxes_on_base_image(
        "/home/murat/Projects/airplane_recognition/data/Gaofen/train/images/20.tif")

    # PLOT SINGLE PATCH BY INDEX

    # problem_ones_file = '/home/murat/Projects/airplane_recognition/docs/problem_ones.txt'
    # with open(problem_ones_file,'r') as f:
    #     lines = f.readlines()
    #     # print(img_paths)
    #     for line in lines:

    #         img_path,ind = line.split(',')
    #         ind = int(ind[:-1])
    #         print(os.path.split(img_path)[-1], ind)
    #         recognition.get_patch_by_index(img_path=img_path,obj_ind=ind,save=False,plot=True)

    # PLOT SINGLE PATCH BY INDEX
    img_path = "/home/murat/Projects/airplane_recognition/data/Gaofen/train/images/143.tif"
    recognition.get_patch_by_index(
        img_path=img_path,
        obj_ind=0,
        save=False,
        plot=True)

    # ### ANALYSE
    # analyse = RecognitionAnalysis(dataset_id,dataset_part,dataset_name,patch_size)
    # instance_number = analyse.get_instance_number()
    # # print(instance_number)
    # for key, value in instance_number.items():
    #     print(f"{key}: {value}")
