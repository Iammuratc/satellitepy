import mmcv
from pathlib import Path
import json

from satellitepy.data.labels import read_label
from satellitepy.evaluate.mmrotate.utils import *
from satellitepy.utils.path_utils import create_folder, get_file_paths, is_file_names_match
from satellitepy.data.patch import get_patches, merge_patch_results
from satellitepy.data.utils import read_img



def save_mmrotate_patch_results(
        out_folder,
        in_image_folder,
        in_label_folder,
        in_label_format,
        config_path,
        weights_path,
        device,
        class_names,
        nms_on_multiclass_thr,
        task_name
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
    task_name: str
        name of the trained task
    Returns
    -------
    None
    """
    logger = logging.getLogger('')
    mmrotate_model = get_mmrotate_model(config_path, weights_path, device)

    patch_result_folder = Path(out_folder) / 'results' / 'patch_labels'
    assert create_folder(patch_result_folder)

    image_paths = get_file_paths(in_image_folder)
    label_paths = get_file_paths(in_label_folder)

    for img_path, label_path in zip(image_paths, label_paths):

        img_name = img_path.stem
        logger.info(f'{img_name} will be processed...')

        is_match = is_file_names_match(img_path, label_path)
        if not is_match:
            logger.error('Label and image names do not match!')
            exit(1)

        img = mmcv.imread(img_path)
        gt_labels = read_label(label_path, in_label_format)

        result = get_result(
            img=img,
            gt_labels=gt_labels,
            mmrotate_model=mmrotate_model,
            class_names=class_names,
            task_name=task_name,
            nms_on_multiclass_thr=nms_on_multiclass_thr)

        with open(Path(patch_result_folder) / f'{img_name}.json', 'w') as f:
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
        class_names,
        task_name,
        image_read_module):
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
    logger = logging.getLogger('')
    mmrotate_model = get_mmrotate_model(config_path, weights_path, device)

    original_result_folder = Path(out_folder)
    assert create_folder(original_result_folder)

    image_paths = get_file_paths(in_image_folder)
    label_paths = get_file_paths(in_label_folder)

    for img_path, label_path in zip(image_paths, label_paths):
        img_name = img_path.name
        logger.info(f'Processing {img_name}...')

        is_match = is_file_names_match(img_path, label_path)
        if not is_match:
            logger.error('Label and image names do not match!')
            exit(1)

        img = read_img(str(img_path), module=image_read_module)

        gt_labels = read_label(label_path, in_label_format)

        patch_dict = get_patches(
            img=img,
            gt_labels=gt_labels,
            truncated_object_thr=truncated_object_thr,
            patch_size=patch_size,
            patch_overlap=patch_overlap
        )

        patch_dict['det_labels'] = []

        for patch_img in patch_dict['images']:
            mmrotate_result = inference_detector(mmrotate_model, patch_img)
            det_label = get_det_labels(mmrotate_result, class_names, task_name, nms_on_multiclass_thr)

            patch_dict['det_labels'].append(det_label)

        merged_det_labels, mask = merge_patch_results(patch_dict, patch_size, shape=img.shape[0:2])
        matches = mmrotate_match_gt_and_det_bboxes(gt_labels, merged_det_labels)

        result = {
            'gt_labels': gt_labels,
            'det_labels': merged_det_labels,
            'matches': matches
        }

        with open(Path(original_result_folder) / f"{img_name}.json", 'w') as f:
            json.dump(result, f, indent=4)
