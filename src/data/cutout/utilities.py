import os

import cv2
import numpy as np

from src.data.cutout.cutout import Cutout


def filter_truncated_images(settings):
    """
    Filters out truncated images for every dataset part and 'cutout folder'
    by checking if the corners of an image are black
    Filtered images are saved in full_'cutout folder'
    """
    for dataset_part in settings['dataset_parts']:
        for folder in ['image_folder', 'orthogonal_image_folder', 'orthogonal_zoomed_image_folder']:

            src_folder = settings['cutout'][dataset_part][folder]
            full_images_folder = settings['cutout'][dataset_part]['full_' + folder]
            img_paths = Cutout.get_file_paths('', src_folder, False)

            for img_path in img_paths:
                img_name = img_path.split('\\')[-1]
                img_name = img_name.split('/')[-1]
                print("Img_name: " + img_name)

                im = cv2.imread(img_path)
                width, height, x = im.shape

                width_offset = int(width/3)
                height_offset = int(height/3)

                if ((np.array_equal(im[0, 0], [0, 0, 0]) and np.array_equal(im[width_offset, height_offset], [0, 0, 0])) or
                        (np.array_equal(im[0, height - 1], [0, 0, 0]) and np.array_equal(im[width_offset, height - height_offset], [0, 0, 0])) or
                        (np.array_equal(im[width - 1, 0], [0, 0, 0]) and np.array_equal(im[width - width_offset, height_offset], [0, 0, 0])) or
                        (np.array_equal(im[width - 1, height - 1], [0, 0, 0]) and np.array_equal(im[width - width_offset, height - height_offset], [0, 0, 0]))):
                    print(f'Image {img_name} is truncated')
                else:
                    cv2.imwrite(os.path.join(full_images_folder, img_name), im)
