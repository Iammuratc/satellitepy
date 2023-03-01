import mmcv
import os
from pathlib import Path
import pathlib
import json
from mmdet.apis.inference import init_detector, inference_detector
import logging
import pprint


from src.evaluate.mmrotate.utils import get_mmrotate_model, get_result, get_gt_labels, get_det_labels, match_gt_and_det_bboxes
from src.settings.utils import create_folder, get_file_paths, get_file_name_from_path
from src.data.patch import get_patches, merge_patch_results


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
    patch_result_folder = pathlib.PurePath(out_folder,'results','patch_labels')
    assert create_folder(patch_result_folder)

    image_paths = get_file_paths(in_image_folder)
    label_paths = get_file_paths(in_label_folder)
    prog_bar = mmcv.ProgressBar(len(image_paths))


    for img_path,label_path in zip(image_paths,label_paths):

        # Check if label and image names match
        img_name = get_file_name_from_path(img_path)
        label_name = get_file_name_from_path(label_path)
        logger.info(f'{img_name} will be processed...')

        is_match = img_name == label_name
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

        prog_bar.update()


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
    original_result_folder = pathlib.PurePath(out_folder,'results','original_labels')
    assert create_folder(original_result_folder)


    image_paths = get_file_paths(in_image_folder)
    label_paths = get_file_paths(in_label_folder)
    prog_bar = mmcv.ProgressBar(len(image_paths))

    temp_count = 0
    for img_path,label_path in zip(image_paths,label_paths):
        # Check if label and image names match
        img_name = get_file_name_from_path(img_path)
        label_name = get_file_name_from_path(label_path)
        logger.info(f'{img_name} will be processed...')

        is_match = img_name == label_name
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



