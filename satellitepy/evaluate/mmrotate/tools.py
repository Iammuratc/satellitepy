import mmcv
# import os
from pathlib import Path
import pathlib
import json
from mmdet.apis.inference import init_detector, inference_detector
import logging
import numpy as np
import matplotlib.pyplot as plt

from satellitepy.evaluate.mmrotate.utils import *
from satellitepy.utils.path_utils import create_folder, get_file_paths, is_file_names_match
from satellitepy.data.patch import get_patches, merge_patch_results

# TODO:
#   set_conf_mat_from_result reads the gt instance names from class_1, implement a way to read different class levels

def save_mmrotate_patch_results(
    out_folder,
    in_image_folder,
    in_label_folder,
    in_label_format,
    config_path,
    weights_path,
    device,
    class_names,
    nms_on_multiclass_thr
    ):
    """
    Pass patch images to a mmrotate model and save the detected bounding boxes as json files in satellitepy format
    Parameters
    ----------
    out_folder : Path
        Results will be saved here
    in_image_folder : Path
        Test image folder
    in_label_folder : Path
        Test label folder
    in_label_format : str
        Test label file format (e.g., dota, fair1m)
    config_path : Path
        MMRotate config path
    weights_path : Path
        MMRotate model weights path
    device : str
        cpu or cuda:0
    class_names : list
        Class names
    Returns
    -------
    None
    """
    logger = logging.getLogger(__name__)
    mmrotate_model = get_mmrotate_model(config_path,weights_path,device)

    # Create result patch folder
    patch_result_folder = Path(out_folder) / 'results' / 'patch_labels'
    assert create_folder(patch_result_folder)

    image_paths = get_file_paths(in_image_folder)
    label_paths = get_file_paths(in_label_folder)

    for img_path,label_path in zip(image_paths,label_paths):

        img_name = img_path.stem
        logger.info(f'{img_name} will be processed...')

        # Check if label and image names match
        is_match = is_file_names_match(img_path,label_path)
        if not is_match:
            logger.error('Label and image names do not match!')
            exit(1)

        # Image
        img = mmcv.imread(img_path)

        # Labels
        gt_labels = get_gt_labels(label_path,in_label_format)

        # Save results with the corresponding ground truth
        result = get_result(    
            img=img,
            gt_labels=gt_labels,
            mmrotate_model=mmrotate_model,
            class_names=class_names,
            nms_on_multiclass_thr=nms_on_multiclass_thr)

        # Save labels to json file
        with open(Path(patch_result_folder) / f"{img_name}.json",'w') as f:
            json.dump(result, f, indent=4)


def save_mmrotate_original_results(
    in_image_folder,
    in_label_folder,
    in_label_format,
    out_folder,
    config_path,
    weights_path,
    nms_on_multiclass_thr,
    device,
    patch_size,
    patch_overlap,
    truncated_object_thr,
    class_names
    ):
    """
    Pass original images to a mmrotate model and save the detected bounding boxes in satellitepy dict format
    Parameters
    ----------
    out_folder : Path
        Results will be saved here
    in_image_folder : Path
        Test image folder
    in_label_folder : Path
        Test label folder
    in_label_format : str
        Test label file format (e.g., dota, fair1m)
    config_path : Path
        MMRotate config path
    weights_path : Path
        MMRotate model weights path
    device : str
        cpu or cuda:0
    class_names : list
        Class names
    truncated_object_thr : float
        Overlapping part of the object with the patch. 1 if the object is completely in the patch, 0 if not.
    patch_size : int
        Patch size.
    patch_overlap : int
        Patch overlap equals to stride - patch_size
    Returns
    -------
        None
    """
    logger = logging.getLogger(__name__)
    mmrotate_model = get_mmrotate_model(config_path,weights_path,device)


    # Create result original folder
    original_result_folder = Path(out_folder) / 'results' / 'original_labels'
    assert create_folder(original_result_folder)


    image_paths = get_file_paths(in_image_folder)
    label_paths = get_file_paths(in_label_folder)
    prog_bar = mmcv.ProgressBar(len(image_paths))

    temp_count = 0
    for img_path,label_path in zip(image_paths,label_paths):
        # Check if label and image names match
        img_name = img_path.stem
        logger.info(f'{img_name} will be processed...')

        # Check if label and image names match
        is_match = is_file_names_match(img_path,label_path)
        if not is_match:
            logger.error('Label and image names do not match!')
            exit(1)

        # Image
        img = mmcv.imread(img_path)
        # logger.info(img_path)

        # Labels
        gt_labels = get_gt_labels(label_path,in_label_format)

        # Get patches
        patch_dict = get_patches(
            img=img,
            gt_labels=gt_labels,
            truncated_object_thr=truncated_object_thr,
            patch_size=patch_size,
            patch_overlap=patch_overlap
            )

        # Detected labels from patches
        det_labels = []

        for patch_img in patch_dict['images']:            
            # mmrotate result
            mmrotate_result = inference_detector(mmrotate_model,patch_img)

            # Detected labels
            det_label = get_det_labels(mmrotate_result,class_names,nms_on_multiclass_thr)

            det_labels.append(det_label)

        patch_dict['det_labels'] = det_labels
        # Merge patch results into original results standards
        merged_det_labels = merge_patch_results(patch_dict)

        # Find matches of original image with merged patch results
        matches = match_gt_and_det_bboxes(gt_labels,merged_det_labels)

        # Results
        result = {
            'gt_labels':gt_labels,
            'det_labels':merged_det_labels,
            'matches':matches
                    }

        # Save labels to json file
        with open(Path(original_result_folder) / f"{img_name}.json",'w') as f:
            json.dump(result, f, indent=4)


def calculate_map(
    in_result_folder,
    instance_names,
    conf_score_thresholds,
    iou_thresholds,
    out_folder,
    plot_pr):

    # Get logger
    logger = logging.getLogger(__name__)

    # Add background to instance_names
    instance_names = instance_names + ['Background']

    # Init confusion matrix
    conf_mat = np.zeros(shape=(len(iou_thresholds),len(conf_score_thresholds),len(instance_names),len(instance_names)))

    # Result paths
    result_paths = get_file_paths(in_result_folder)

    for result_path in result_paths:
        logger.info(f'The following result file will be evaluated: {result_path}')
        # Result json file
        with open(result_path,'r') as result_file:
            result = json.load(result_file) # dict of 'gt_labels', 'det_labels', 'matches' 
        
        conf_mat = set_conf_mat_from_result(
            conf_mat,
            result,
            instance_names,
            conf_score_thresholds,
            iou_thresholds)

    precision, recall = get_precision_recall(conf_mat,sort_values=True)
    print('Precision at all confidence score thresholds and iuo threshold = 0.5')
    print(precision[0,:])
    print('Recall at all confidence score thresholds and iou threshold = 0.5 ')
    print(recall[0,:])
    ap = get_average_precision(precision,recall)
    print(ap)
    if plot_pr:
        fig, ax = plt.subplots()
        ax.plot(recall[0,:],precision[0,:])
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        plt.show()
    # print('Instance names are')
    # print(instance_names)
    # print('Precision at confidence score threshold = 0.5 and iou threshold = 0.5 ')
    # print(precision[0,10])
    # print('Recall at confidence score threshold = 0.5 and iou threshold = 0.5 ')
    # print(recall[0,10])

