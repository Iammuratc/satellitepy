import json
import os
import shutil
from pathlib import Path
import numpy as np
import logging

import cv2
from tqdm import tqdm

from satellitepy.data.labels import read_label, init_satellitepy_label, fill_none_to_empty_keys, \
    get_all_satellitepy_keys, satellitepy_labels_empty
from satellitepy.data.patch import get_patches
from satellitepy.data.chip import get_chips
from satellitepy.data.utils import get_xview_classes, get_task_dict, read_img, rescale_labels
from satellitepy.models.bbavector.utils import decode_masks
from satellitepy.utils.path_utils import create_folder, get_file_paths
from satellitepy.data.bbox import BBox
from satellitepy.data.utils import get_satellitepy_dict_values
from satellitepy.evaluate.utils import remove_low_conf_results, match_gt_and_det_bboxes
from satellitepy.evaluate.bbavector.utils import apply_nms



logger = logging.getLogger('')


def create_satellitepy_labels(
        image_folder,
        label_folder,
        label_format,
        out_folder,
        mask_folder=None
):
    out_label_folder = Path(out_folder) / 'labels'
    assert create_folder(out_label_folder)
    label_paths = get_file_paths(label_folder)

    if image_folder:
        out_image_folder = Path(out_folder) / 'images'
        assert create_folder(out_image_folder)
        img_paths = get_file_paths(image_folder)
    else:
        img_paths = [None] * len(label_paths)

    if mask_folder:
        mask_paths = get_file_paths(mask_folder)
    else:
        mask_paths = [None] * len(label_paths)

    assert len(label_paths) == len(mask_paths) == len(img_paths)

    for label_path, mask_path, img_path in tqdm(zip(label_paths, mask_paths, img_paths), total=len(label_paths)):
        label_name = label_path.stem

        gt_labels = read_label(label_path, label_format, mask_path)

        labels = init_satellitepy_label()
        for i in range(len(gt_labels['hbboxes'])):

            for key in get_all_satellitepy_keys():
                keys = key.split("_")
                if len(keys) == 1:
                    labels[keys[0]].append(gt_labels[keys[0]][i])
                elif len(keys) == 2:
                    labels[keys[0]][keys[1]].append(gt_labels[keys[0]][keys[1]][i])
                elif len(keys) == 3:
                    labels[keys[0]][keys[1]][keys[2]].append(gt_labels[keys[0]][keys[1]][keys[2]][i])

        res_label_path = Path(out_label_folder) / f"{label_format}_{label_name}.json"
        with open(str(res_label_path), 'w') as f:
            json.dump(labels, f, indent=4)
        if image_folder:
            img = cv2.imread(str(img_path))
            copy_image_path = Path(out_image_folder) / f"{label_format}_{label_name}.png"
            cv2.imwrite(str(copy_image_path), img)


def save_patches(
        image_folder,
        label_folder,
        label_format,
        out_folder,
        truncated_object_thr,
        patch_size,
        patch_overlap,
        image_read_module,
        mask_folder=None,
        rescaling=1,
        interpolation_method=cv2.INTER_LINEAR,
        keep_empty=False,
        subset_data=None
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
    image_read_module : str
        cv2 or rasterio to read the images
    rescaling : float
        Images and the corresponding labels will be rescaled.
    interpolation_method : str
        <interpolation-method> will be used to rescale images.
    keep_patches : bool
        Empty patches are kept if true.
    Returns
    -------
    Save patches in <out-folder>/images and <out-folder>/labels
    """
    logger.info('Saving patches')

    assert out_folder.exists(), 'Make sure the out-folder exists or is created before calling this function.'

    if subset_data is not None:
        out_image_folder = Path(out_folder)
        out_label_folder = Path(out_folder)
    else:
        out_image_folder = Path(out_folder) / 'images'
        out_label_folder = Path(out_folder) / 'labels'

    assert create_folder(out_image_folder, ask_permission=False)
    img_paths = get_file_paths(image_folder)
    if label_folder:
        assert create_folder(out_label_folder, ask_permission=False)
        label_paths = get_file_paths(label_folder)
    else:
        label_paths = [None] * len(img_paths)
    if mask_folder:
        mask_paths = get_file_paths(mask_folder)
    else:
        mask_paths = [None] * len(img_paths)

    assert len(img_paths) == len(label_paths) == len(mask_paths)

    for img_path, label_path, mask_path in zip(img_paths, label_paths, mask_paths):
        mask_name = mask_path.name if mask_path != None else None
        label_name = label_path.name if label_path is not None else None
        logger.info(f"{img_path.name}, {label_name}, {mask_name}")

        # Change output directory to patches if subset_data is not None
        if subset_data is not None:
            subset_type = ""
            for col in subset_data.values:
                if img_path.name in col[0]:
                    subset_type = col[2]

            if subset_type != "":
                out_image_folder = Path(out_image_folder) / subset_type / 'images'
                assert create_folder(out_image_folder)
                out_label_folder = Path(out_label_folder) / subset_type / 'labels'
                assert create_folder(out_label_folder)

        gt_labels = read_label(label_path, label_format, mask_path, rescaling)

        img = read_img(str(img_path), module=image_read_module, rescaling=rescaling, interpolation_method=interpolation_method)

        patches = get_patches(
            img,
            gt_labels,
            truncated_object_thr,
            patch_size,
            patch_overlap
        )

        count_patches = len(patches['images'])
        logger.info(f'Number of patches: {count_patches}')
        count_skipped_patches = [0, 0]
        for i in tqdm(range(count_patches)):

            img_name = img_path.stem

            patch_x0, patch_y0 = patches['start_coords'][i]
            patch_name = f"{img_name}_x_{patch_x0}_y_{patch_y0}"

            patch_img = patches['images'][i]
            patch_image_path = Path(out_image_folder) / f"{patch_name}.png"

            if len(np.unique(patch_img)) == 1:
                count_skipped_patches[0] += 1
                continue

            patch_label = patches['labels'][i]
            patch_label_path = Path(out_label_folder) / f"{patch_name}.json"

            if not keep_empty and not (any(patch_label['obboxes']) or any(patch_label['hbboxes'])):
                count_skipped_patches[1] += 1
                continue

            cv2.imwrite(str(patch_image_path), patch_img)
            with open(str(patch_label_path), 'w') as f:
                json.dump(patch_label, f, indent=4)

        logger.info(f"{count_skipped_patches[0]} patches are skipped because images consist of only zeros.")
        if not keep_empty:
            logger.info(f"{count_skipped_patches[1]} patches are skipped because no bounding boxes are defined.")
        logger.info(f"{sum(count_skipped_patches)} patches are skipped in total.")
        logger.info(f"{count_patches - sum(count_skipped_patches)} patches are created in total.")
        logger.info(f"Patch labels are created at: {out_label_folder}")


def save_class_chips(
        label_format,
        image_folder,
        label_folder,
        out_folder,
        task,
        mask_folder=None):
    """
    Save chips from the original images, sorted by classes and displaying additional information.
    Parameters
    ----------
    label_format : str,
        Resembles the label format (e.g. dota, fair1m, etc.)
    image_folder : Path
        Input image folder. Images in this folder will be processed.
    label_folder : Path
        Input label folder. Labels in this folder will be used to create patch labels.
    out_folder : Path
        Output folder. Patches and corresponding labels will be saved into <out-folder>/patch_<patch-size>/images and <out-folder>/patch_<patch-size>/labels
    task : str
        task by which the chips will be sorted
    mask_folder : Path
        Input mask folder. Masks in this folder will be used to create chip masks
    Returns
    -------
    """

    image_paths = get_file_paths(image_folder)
    label_paths = get_file_paths(label_folder)
    if mask_folder:
        mask_paths = get_file_paths(mask_folder)
    else:
        mask_paths = [None] * len(image_paths)

    assert len(image_paths) == len(label_paths) == len(mask_paths), 'image/label/mask folders do not match in size'

    classes = []
    class_cnt = []
    class_lengths = []
    class_widths = []

    logger.info('Creating chips and writing individual image information:')
    for img_path, label_path, mask_path in tqdm(zip(image_paths, label_paths, mask_paths), total=len(image_paths)):
        img = cv2.imread(str(img_path))
        label = read_label(label_path, label_format, mask_path)

        chips = get_chips(
            img,
            label,
            task
        )

        count_chips = len(chips['images'])
        img_name = img_path.stem

        for i in range(count_chips):
            instance_name = chips['attributes']['task'][i]
            instance_name = instance_name.replace(' ', '_') if instance_name else 'None'
            if instance_name not in classes:
                classes.append(instance_name)
                class_cnt.append(0)
                class_lengths.append(0)
                class_widths.append(0)
            instance_idx = classes.index(instance_name)
            class_cnt[instance_idx] += 1
            chip_height = chips['attributes']['lengths'][i]
            class_lengths[instance_idx] += chip_height
            chip_width = chips['attributes']['widths'][i]
            class_widths[instance_idx] += chip_width

            chip_img = chips['images'][i]
            center = chips['attributes']['center'][i]
            height, width, _ = chip_img.shape
            chip_img = cv2.copyMakeBorder(chip_img, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

            cv2.putText(chip_img,
                        text=f'height={int(chip_height)}, width={int(chip_width)}',
                        org=(3, height+13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=1)

            cv2.putText(chip_img,
                        text=f'x:{center[0]}',
                        org=(width-40, height + 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=1)

            cv2.putText(chip_img,
                        text=f'y:{center[1]}',
                        org=(width-40, height + 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=1)

            cv2.line(chip_img, (0, height + 25), (width-45, height+25), (0, 0, 0), 1)
            cv2.line(chip_img, (width-45, height), (width-45, height+50), (0, 0, 0), 1)

            instance_folder = out_folder / instance_name
            assert create_folder(instance_folder, ask_permission=False)

            center = chips['attributes']['center'][i]
            chip_img_path = instance_folder / f'{img_name}_x_{center[0]}_y_{center[1]}.png'

            if not chip_img.size == 0:
                cv2.imwrite(str(chip_img_path), chip_img)
            else:
                continue

    class_lengths = np.array(class_lengths)/np.array(class_cnt)
    class_widths = np.array(class_widths)/np.array(class_cnt)

    logger.info('Adding average information for all classes:')
    for i, instance_name in tqdm(enumerate(classes), total=len(classes)):
        instance_folder = out_folder / instance_name
        image_paths = get_file_paths(instance_folder)

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            height, width, _ = img.shape

            cv2.putText(img,
                        text=f'avg_h={int(class_lengths[i])}, avg_w={int(class_widths[i])}',
                        org=(3, height - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=1)

            cv2.imwrite(str(img_path), img)


def save_chips(
        label_format,
        image_folder,
        label_folder,
        out_folder,
        chip_size,
        mask_folder,
        img_read_module,
        rescaling,
        interpolation_method,
        orient_objects=False,
        mask_objects=False,
):
    """
    Save chips from the original images
    Parameters
    ----------
    label_format : str,
        Resembles the label format (e.g. dota, fair1m, etc.)
    image_folder : Path
        Input image folder. Images in this folder will be processed.
    label_folder : Path
        Input label folder. Labels in this folder will be used to create patch labels.
    out_folder : Path
        Output folder. Patches and corresponding labels will be saved into <out-folder>/patch_<patch-size>/images and <out-folder>/patch_<patch-size>/labels
    chip_size : int
        Chip size of chip_size * chip_size will be created.
    mask_folder : Path
        Input mask folder. Masks in this folder will be used to create chip masks
    orient_objects : boolean
        If True, Objects in chips will be oriented facing upwards
    Returns
    -------
    """
    out_folder_images = out_folder / 'images'
    out_folder_labels = out_folder / 'labels'

    assert create_folder(out_folder_images)
    assert create_folder(out_folder_labels)

    image_paths = get_file_paths(image_folder)
    label_paths = get_file_paths(label_folder)
    if mask_folder:
        mask_paths = get_file_paths(mask_folder)
    else:
        mask_paths = [None] * len(image_paths)

    chip_counter = [0,0] # per folder, per image
    if len(image_paths) == len(label_paths) == len(mask_paths):
        for img_path, label_path, mask_path in zip(image_paths, label_paths, mask_paths):
            label = read_label(label_path, label_format, mask_path)
            if not (any(label['obboxes']) or any(label['hbboxes'])):
                continue
            chip_counter[1] = 0
            logger.info(f"Reading the image: {img_path}")
            img = read_img(str(img_path), img_read_module, rescaling=rescaling, interpolation_method=interpolation_method)
            logger.info(f"Image read successfull!")

            chips = get_chips(
                img,
                label,
                chip_size=chip_size,
                orient_objects=orient_objects,
                mask_objects=mask_objects
            )
            
            count_chips = len(chips['images'])
            img_name = img_path.stem
            for i in range(count_chips):

                chip_img = chips['images'][i]
                if chip_img is None:
                    logger.warning('Image is None!')
                    continue
                if not chip_img.size == 0:
                    chip_img_path = out_folder_images / f'{img_name}_{i}.png'
                    # cv2.imwrite(str(chip_img_path), chip_img)
                    save_with_padding(chip_img, chip_img_path, (chip_size, chip_size))
                    chip_counter[0] += 1
                    chip_counter[1] += 1
                else:
                    continue

                chip_label = get_label_by_idx(chips['labels'], i)
                chip_label_path = out_folder_labels / f'{img_name}_{i}.json'

                with open(str(chip_label_path), 'w') as f:
                    json.dump(chip_label, f, indent=4)
            logger.info(f"{img_path.stem} has {chip_counter[1]} chips.")
        logger.info(f"{chip_counter[0]} chips are saved in total!")
        logger.info(f"Chips are saved at: {out_folder}")


def save_with_padding(chip_img, chip_img_path, target_size=(256, 256), color=(0, 0, 0)):
    """
    Save an image with padding to ensure it meets the target size.

    Args:
        chip_img (numpy.ndarray): Input image.
        chip_img_path (str): Path to save the padded image.
        target_size (tuple): Desired (width, height) size, default is (256, 256).
        color (tuple): Color of the padding (default is black).
    """
    height, width = chip_img.shape[:2]
    target_width, target_height = target_size

    # Compute padding values
    pad_top = max(0, (target_height - height) // 2)
    pad_bottom = max(0, target_height - height - pad_top)
    pad_left = max(0, (target_width - width) // 2)
    pad_right = max(0, target_width - width - pad_left)

    # Add padding
    padded_img = cv2.copyMakeBorder(
        chip_img,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=color  # Padding color (black by default)
    )

    # Save the padded image
    cv2.imwrite(str(chip_img_path), padded_img)

def get_label_by_idx(satpy_labels: dict, i: int):
    """
    Creates a copy of the satpy_labels dict by doing the following:
    Sets each list to a singleton list corresponding to the item at position i.
    """

    def inner(input_dict, output_dict):
        for k in input_dict.keys():
            if isinstance(input_dict[k], dict):
                output_dict.setdefault(k, {})
                inner(input_dict[k], output_dict[k])
            else:
                value = input_dict[k][i]
                if isinstance(value, list):
                    output_dict[k] = value
                else:
                    output_dict[k] = [value]

    result = {}
    inner(satpy_labels, result)
    return result


def show_labels_on_images(
        image_folder,
        label_folder,
        mask_folder,
        label_format,
        img_read_module,
        out_folder,
        tasks,
        rescaling,
        interpolation_method,
    ):
    """
    Images visualizing given tasks (e.g., bounding boxes in polygons, classification tasks in text, masks in contours)
    will be stored under out_folder/images
    Parameters
    ----------
    image_folder : str
        Input image directory
    label_folder : str
        Input label directory
    mask_folder : str
        Input mask directroy
    img_read_module : str
        Module to read image, e.g., cv2, rasterio
    out_folder : Path
        Output directory. Image will be saved here.
    tasks : list
        List of tasks
    label_format : str
        Label format, e.g., dota, satellitepy
    """
    logger.info('Drawing labels on images, and saving the images.')

    out_image_folder = out_folder / "images"
    assert create_folder(out_image_folder)

    img_paths = get_file_paths(image_folder)
    label_paths = get_file_paths(label_folder)
    mask_paths = get_file_paths(mask_folder) if mask_folder else [None] * len(img_paths)

    assert len(img_paths) == len(label_paths) == len(mask_paths)

    requested_classes = [task for task in tasks if task not in ['dbboxes', 'obboxes', 'hbboxes', 'masks']]
    # Force user to pass only one bbox type
    requested_bbox_names = [task for task in tasks if task.endswith('bboxes')]
    assert len(requested_bbox_names) == 1, logger.error('There has to be only type of bounding boxes in tasks!')
    bbox_name = requested_bbox_names[0]

    for label_path, mask_path, img_path in tqdm(zip(label_paths, mask_paths, img_paths), total=len(label_paths)):
        img = read_img(str(img_path), img_read_module, rescaling=rescaling, interpolation_method=interpolation_method)
        labels = read_label(label_path, label_format, rescaling=rescaling)


        for i, bbox_corners in enumerate(labels[bbox_name]):
            if bbox_name == 'dbboxes':
                bbox = BBox(diamond_corners=bbox_corners)
                img = bbox.draw_bbox_to_img(img, corners=[bbox.diamond_corners], thickness=1)
            else:
                bbox = BBox(corners=bbox_corners)
                img = bbox.draw_bbox_to_img(img, corners=[bbox.corners], thickness=1)

            available_tasks = requested_classes.copy()
            if labels['coarse-class'][i] != 'airplane':
                available_tasks = [task for task in available_tasks if 'attributes' not in task]
            if labels['coarse-class'][i] not in ['airplane', 'ship'] and 'very-fine-class' in available_tasks:
                available_tasks.remove('very-fine-class')

            if requested_classes:
                cx, cy, width, height, angle = bbox.params
                center = (cx.astype(int), cy.astype(int))
                img = cv2.circle(img, center, 1, color=(255, 255, 255), thickness=-1)
            for j, task in enumerate(available_tasks):
                task_keys = task.split('_')
                task_text = task_keys[-1]

                task_result = get_satellitepy_dict_values(labels, task)[i]

                if type(task_result) is float:
                    task_result = round(task_result, 2)

                text = task_text + ': ' + str(task_result)

                cv2.putText(img,
                    text=str(text),
                    org=(cx.astype(int), cy.astype(int) + j*15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0,0,255),
                    thickness=1,
                    lineType=1)

            if 'masks' in tasks and labels['masks'][0] is not None:
                if label_format == 'satellitepy':
                    mask = labels['masks']
                else:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                img_mask = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
                for i in range(len(labels[bbox_name])):
                    x, y = mask[i]
                    img_mask[y, x] = 1

                contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

            out_img_path = out_image_folder / f'{img_path.stem}.png'
            cv2.imwrite(str(out_img_path), img)


def show_results_on_image(img_dir,
                          result_dir,
                          mask_dir,
                          mask_threshold,
                          mask_adaptive_size,
                          out_dir,
                          tasks,
                          target_task,
                          all_tasks_flag,
                          image_read_module,
                          iou_th=0.5,
                          conf_th=0.5,
                          no_probability=False,
):
    """
    Visualize results on images
    """
    logger.info(tasks)
    img_paths = get_file_paths(img_dir)
    label_paths = get_file_paths(result_dir)
    mask_paths = get_file_paths(mask_dir) if mask_dir else [None] * len(img_paths)
    assert len(img_paths) == len(label_paths) == len(mask_paths)
    for img_path, label_path, mask_path in tqdm(zip(img_paths, label_paths, mask_paths), total=len(img_paths)):
        img = read_img(str(img_path), module=image_read_module)
        results = read_label(label_path, label_format='satellitepy')

        if len(results['det_labels'][target_task]) == 0:
            logger.info(f'Skipping {label_path.stem}: No detections!')
            continue

        # results = remove_low_conf_results(results, target_task, conf_th, no_probability)
        results['det_labels'] = apply_nms(results['det_labels'], nms_iou_threshold=iou_th, target_task=target_task, no_probability=no_probability)
        results['matches'] = match_gt_and_det_bboxes(results['gt_labels'], results['det_labels'])


        if satellitepy_labels_empty(results):
            continue

        available_tasks = list(results['det_labels'].keys())

        if not all_tasks_flag:
            available_tasks = tasks.copy()

        available_tasks = [task for task in available_tasks if task not in ['obboxes','hbboxes','masks','confidence-scores']]

        if results['det_labels']['obboxes'][0] is None:
            bboxes = 'hbboxes'
        else:
            bboxes = 'obboxes'

        for i, bbox_corners in enumerate(results['det_labels'][bboxes]):
            bbox_corners = np.array(bbox_corners, np.int32)
            x_min, x_max, y_min, y_max = BBox.get_bbox_limits(bbox_corners)
            cv2.polylines(img, [bbox_corners], True, color=(255, 0, 0))

            for j, task in enumerate(available_tasks):

                task_result = np.argmax(results['det_labels'][task][i])

                task_text = task.split('_')[-1]
                task_dict = get_task_dict(task)

                if task_text not in ['length', 'wing-span']:
                    idx2name = {v: k for k, v in task_dict.items()}
                    task_result = idx2name[task_result]

                if type(task_result) is float:
                    task_result = round(task_result, 2)

                text = task_text + ': ' + str(task_result)
                cv2.putText(img, str(text), (x_max, y_min + j * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if mask_dir and (all_tasks_flag or 'masks' in tasks):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = 255 - mask
            mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, mask_adaptive_size, mask_threshold)
            img_mask = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)

            for i, bbox in enumerate(results['det_labels'][bboxes]):
                conf_score = results['det_labels']['confidence-scores'][i]
                iou_score = results['matches']['iou']['scores'][i]
                if conf_score < conf_th or iou_score < iou_th:
                    continue

                mask_values = decode_masks(bbox, mask)
                x, y = mask_values
                img_mask[y, x] = 1

            contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

        cv2.imwrite(str(Path(out_dir) / f"{img_path.stem}.png"), img)


def save_xview_in_satellitepy_format(out_folder, label_path):
    """
    Parameters
    ----------
    label_path : Path
        One big annotation file for all the images
    out_folder : Path
        Output folder. New labels will be saved into <out-folder>
    Returns
    -------
    Save a label file for each image
    """

    logger.info(f'Initializing save_xview_in_satellitepy_format')

    labels = json.load(open(label_path, 'r'))
    all_image_names = []
    for feature in labels['features']:
        all_image_names.append(feature['properties']['image_id'])

    image_dicts = {img_name: init_satellitepy_label() for img_name in set(all_image_names)}
    available_tasks = ['hbboxes', 'coarse-class', 'fine-class', 'role']
    all_tasks = get_all_satellitepy_keys()
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]

    classes = get_xview_classes()

    img_name_for_log = list(image_dicts.keys())[0]
    for feature in tqdm(labels['features']):
        img_name = feature['properties']['image_id']
        if img_name != img_name_for_log:
            img_name_for_log = img_name


        type_class = int(feature['properties']['type_id'])

        if type_class not in classes['vehicles'] and \
           type_class not in classes['ships'] and \
           type_class not in classes['helicopter'] and \
           type_class not in classes['airplanes']:
            continue

        if type_class in classes['vehicles']:
            image_dicts[img_name]['coarse-class'].append('vehicle')
            image_dicts[img_name]['fine-class'].append(classes['vehicles'][type_class])
            if type_class in [17, 23, 53, 19, 24, 25, 26, 28, 29, 54, 55, 56,
                              57, 59, 60, 61, 62, 63, 64, 65, 66, 32]:
                image_dicts[img_name]['role'].append('Large Vehicle')
            elif type_class in [18, 20, 21]:
                image_dicts[img_name]['role'].append('Small Vehicle')
            else:
                image_dicts[img_name]['role'].append(None)
        elif type_class in classes['ships']:
            image_dicts[img_name]['coarse-class'].append('ship')
            image_dicts[img_name]['fine-class'].append(classes['ships'][type_class])
            image_dicts[img_name]['role'].append('Merchant Ship')
        elif type_class in classes['airplanes']:
            image_dicts[img_name]['coarse-class'].append('airplane')
            image_dicts[img_name]['fine-class'].append(None)
            if type_class == 11:
                image_dicts[img_name]['role'].append('Small Civil Transport/Utility')
            elif type_class == 12:
                image_dicts[img_name]['role'].append('Medium Civil Transport/Utility')
            elif type_class == 13:
                image_dicts[img_name]['role'].append('Large Civil Transport/Utility')
            else:
                image_dicts[img_name]['role'].append(None)
        elif type_class in classes['helicopter']:
            image_dicts[img_name]['coarse-class'].append('helicopter')
            image_dicts[img_name]['fine-class'].append(None)
            image_dicts[img_name]['role'].append(None)

        coords = feature['properties']['bounds_imcoords'].split(',')
        xmin = int(coords[0])
        ymin = int(coords[1])
        xmax = int(coords[2])
        ymax = int(coords[3])
        image_dicts[img_name]['hbboxes'].append([[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax]])

        fill_none_to_empty_keys(image_dicts[img_name], not_available_tasks)

    for img_name, satellitepy_dict in image_dicts.items():
        label_name = f'{Path(img_name).stem}.json'
        label_path = out_folder / label_name
        with open(str(label_path), 'w') as f:
            json.dump(satellitepy_dict, f, indent=4)


def separate_dataset_parts(out_folder, label_folder, image_folder, dataset_part, dataset_name):
    """
    Parameters
    -------
    out_folder : Path
        Output folder. Images and labels will be saved into <out-folder>
    Returns
    -------
    Split the images and labels according to the dataset_part file
    """

    logger.info(f'Initializing separate_shipnet_data')

    data_part_name = dataset_part.stem
    out_image_folder = os.path.join(out_folder, Path('images'))
    assert create_folder(Path(out_image_folder))

    if dataset_name != 'shipnet' or data_part_name != 'test':
        out_label_folder = os.path.join(out_folder, Path('labels'))
        assert (create_folder(Path(out_label_folder)))

    with open(dataset_part, 'r') as dataset:
        for line in tqdm(dataset.readlines()):

            if dataset_name == 'shipnet':
                extension = '.bmp'
                name = line.split('.')[0]
            else:
                extension = '.jpg'
                name = line.split('.')[0][:-1]

            image_path = os.path.join(image_folder, Path(name + extension))

            shutil.copy(image_path, out_image_folder)

            if dataset_name != 'shipnet' or data_part_name != 'test':
                padding = max(0, 6 - name.__len__())
                if dataset_name == 'dior':
                    padding = 0
                label_path = os.path.join(label_folder, Path(padding * '0' + name + '.xml'))
                shutil.copy(label_path, out_label_folder)
    logger.info(f'Images and labels saved for dataset-part {data_part_name}')


def split_rareplanes_labels(
        label_file,
        out_folder
):
    """
        Split single Rareplanes annotation file into one label file per image
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
        label_file = open(label_file_path, 'r', encoding='utf-8')
        logger.info(f'Saving annotation for {label_file_path}')
        annotations = json.load(label_file)
        label_file.close()
        annotations['annotations'].append(new_annotation)
        file = open(label_file_path, 'w', encoding='utf-8')

        json.dump(annotations, file, ensure_ascii=False, indent=4)
        file.close()


def copy_files(files, src_folder, dst_folder):
    """
        Copys a list of files from one directory to another one
        Parameters
        ----------
        files : List of Paths
            Files which should be copied
        src_folder : Path
            Path to source folder of the files
        dst_folder : Path
            Path to new destination folder
    """
    for file in files:
        logger.info(f'Copying {src_folder / file.name} to {dst_folder / file.name}')
        shutil.copy(src_folder / file.name, dst_folder / file.name)