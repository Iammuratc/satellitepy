import json
import logging
import os
import numpy as np
from pathlib import Path

import cv2

from satellitepy.data.labels import read_label
from satellitepy.data.patch import get_patches
from satellitepy.utils.path_utils import create_folder, get_file_paths
from satellitepy.data.cutout.geometry import BBox


def save_patches(
    image_folder,
    label_folder,
    label_format,
    out_folder,
    truncated_object_thr,
    patch_size,
    patch_overlap,
    mask_folder = None
    ):
    """
    Save patches from the original images
    Parameters
    ----------
    image_folder : Path
        Input image folder. Images in this folder will be processed.
    label_folder : Path
        Input label folder. Labels in this folder will be used to create patch labels.
    mask_folder : Path
        Input mask folder. Masks in this folder will be used to create patch masks
    label_format : str
        Input label format.
    out_folder : Path
        Output folder. Patches and corresponding labels will be saved into <out-folder>/patch_<patch-size>/images and <out-folder>/patch_<patch-size>/labels
    truncated_object_thr : float
        Truncated object threshold
    patch_size : int
        Patch size
    patch_overlap : int
        Patch overlap
    Returns
    -------
    Save patches in <out-folder>/patch_<patch-size>/images and <out-folder>/patch_<patch-size>/labels
    """

    # Create output folders
    out_image_folder = Path(out_folder) / f'patch_{patch_size}' / 'images'
    out_label_folder = Path(out_folder) / f'patch_{patch_size}' / 'labels'

    assert create_folder(out_image_folder)
    assert create_folder(out_label_folder)
    img_paths = get_file_paths(image_folder)
    label_paths = get_file_paths(label_folder)
    if mask_folder:
         mask_paths = get_file_paths(mask_folder)
    else:
        mask_paths = [None] * len(img_paths)
   
    if (len(img_paths)==len(label_paths)==len(mask_paths)):
        for img_path, label_path, mask_path in zip(img_paths,label_paths,mask_paths):
            # Image
            img = cv2.imread(str(img_path))
            # Labels
            if (mask_folder):
                gt_labels = read_label(label_path,label_format,mask_path)
            else:
                gt_labels = read_label(label_path,label_format)
            # Mask
            if mask_path:
                mask = cv2.imread(str(mask_path))
            else: None

            # Save results with the corresponding ground truth
            patches = get_patches(
            img,
            gt_labels,
            truncated_object_thr,
            patch_size,
            patch_overlap,
            mask,
            )

            count_patches = len(patches['images'])
            for i in range(count_patches):
                # Get original image name for naming patch files
                img_name = img_path.stem

                # Patch starting coordinates
                patch_x0, patch_y0 = patches['start_coords'][i]

                # Save patch image
                patch_img = patches['images'][i]
                patch_image_path = Path(out_image_folder) / f"{img_name}_x_{patch_x0}_y_{patch_y0}.png" 
                cv2.imwrite(str(patch_image_path),patch_img)

                # Save patch labels
                patch_label = patches['labels'][i]
                patch_label_path = Path(out_label_folder) / f"{img_name}_x_{patch_x0}_y_{patch_y0}.json"
                with open(str(patch_label_path),'w') as f:
                    json.dump(patch_label,f,indent=4)

        # if mask_path:
        #     out_label_paths = get_file_paths(out_label_folder)
        #     out_mask_paths = get_file_paths(out_mask_folder)
        #     test_num = 0
        #     bbx = 'bboxes'
        #     for l,m in zip(out_label_paths, out_mask_paths):
        #         test_num1 = 0
        #         print("#######" + str(test_num) + "#######")
        #         test_num +=1
        #         img = cv2.imread(str(m))
        #         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #         with open(l,'r') as f:
        #             l_data = json.load(f)
        #             if ('obboxes' in list(l_data)):
        #                 bbx = 'obboxes'
        #             elif ('hbboxes' in list(l_data)):
        #                 bbx = 'hbboxes'
        #             for v in range(len(l_data[bbx])):
        #                 print(test_num1)
        #                 test_num1 +=1
        #                 tmp_mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
        #                 pts = np.array([l_data[bbx][v]], dtype = np.int32)
        #                 cv2.fillPoly(tmp_mask, pts, 255)
        #                 coord = np.argwhere((tmp_mask == 255) & (img != 0)).tolist()
        #                 l_data['mask-indices'][v] = coord
        #                 with open(l,'w') as f:
        #                     json.dump(l_data,f,indent=4)


def split_rareplanes_labels(
        label_file,
        out_folder
    ):
    """
        Save patches from the original images
        Parameters
        ----------
        label_file : Path
            Input label file. This single label file will be split up.
        out_folder : Path
            Output folder. New labels will be saved into <out-folder>/labels
        Returns
        -------
        Save labels in <out-folder>/labels
        """

    logger = logging.getLogger(__name__)

    # Create output folder
    out_label_folder = Path(out_folder)
    assert create_folder(out_label_folder)

    label_path = Path(label_file)

    file = json.load(open(label_path, 'r'))

    id_to_img = {}
    for image in file['images']:
        id_to_img[image['id']] = image['file_name']
        label_file_path = os.path.join(out_label_folder, image['file_name'][:-3] + 'json')
        label_file = open(label_file_path, 'w')
        logger.info(f'Initializing annotations for {label_file_path}')
        annotations = {'annotations': []}
        json.dump(annotations, label_file, indent=4)
        label_file.close()

    for new_annotation in file['annotations']:
        img_name = id_to_img[new_annotation['image_id']]
        label_file_path = os.path.join(out_label_folder, img_name[:-3] + 'json')
        label_file = open(label_file_path, 'r', encoding="utf-8")
        logger.info(f'Saving annotation for {label_file_path}')
        annotations = json.load(label_file)
        label_file.close()
        annotations['annotations'].append(new_annotation)
        file = open(label_file_path, 'w', encoding="utf-8")

        json.dump(annotations, file, ensure_ascii=False, indent=4)
        file.close()