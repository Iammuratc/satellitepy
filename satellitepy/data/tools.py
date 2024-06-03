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
from satellitepy.evaluate.utils import remove_low_conf_results
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
        interpolation_method=cv2.INTER_LINEAR
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
    Returns
    -------
    Save patches in <out-folder>/images and <out-folder>/labels
    """
    logger.info('Saving patches')

    out_image_folder = Path(out_folder) / 'images'
    out_label_folder = Path(out_folder) / 'labels'

    assert create_folder(out_image_folder)
    img_paths = get_file_paths(image_folder)
    if label_folder:
        assert create_folder(out_label_folder)
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
        for i in tqdm(range(count_patches), leave=True):

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

            if not (any(patch_label['obboxes']) or any(patch_label['hbboxes'])):
                count_skipped_patches[1] += 1
                continue

            cv2.imwrite(str(patch_image_path), patch_img)
            with open(str(patch_label_path), 'w') as f:
                json.dump(patch_label, f, indent=4)

        logger.info(f"{count_skipped_patches[0]} patches are skipped because images consist of only zeros.")
        logger.info(f"{count_skipped_patches[1]} patches are skipped because no bounding boxes are defined.")
        logger.info(f"{sum(count_skipped_patches)} patches are skipped in total.")
        logger.info(f"{count_patches - sum(count_skipped_patches)} patches are created in total.")


def save_chips(
        label_format,
        image_folder,
        label_folder,
        out_folder,
        margin_size,
        include_object_classes,
        exclude_object_classes,
        mask_folder=None
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
    include_object_classes : list
        Classes that will be saved,
    exclude_object_classes : list
        Classes that wont be saved
    mask_folder : Path
        Input mask folder. Masks in this folder will be used to create chip masks
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

    if len(image_paths) == len(label_paths) == len(mask_paths):
        for img_path, label_path, mask_path in zip(image_paths, label_paths, mask_paths):
            img = cv2.imread(str(img_path))
            label = read_label(label_path, label_format, mask_path)

            chips = get_chips(
                img,
                label,
                margin_size,
                include_object_classes,
                exclude_object_classes
            )

            count_chips = len(chips['images'])
            img_name = img_path.stem

            for i in range(count_chips):

                chip_img_path = out_folder_images / f'{img_name}_{i}.png'
                chip_img = chips['images'][i]

                if not chip_img.size == 0:
                    cv2.imwrite(str(chip_img_path), chip_img)
                else:
                    continue

                chip_label = get_label_by_idx(chips['labels'], i)
                chip_label_path = out_folder_labels / f'{img_name}_{i}.txt'

                with open(str(chip_label_path), 'w') as f:
                    json.dump(chip_label, f, indent=4)


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

    requested_classes = [task for task in tasks if task not in ['obboxes', 'hbboxes', 'masks']]
    for label_path, mask_path, img_path in tqdm(zip(label_paths, mask_paths, img_paths), total=len(label_paths)):
        img = read_img(str(img_path), img_read_module, rescaling=rescaling, interpolation_method=interpolation_method)
        labels = read_label(label_path, label_format, rescaling=rescaling)

        bboxes = 'hbboxes'

        if 'dbboxes' in tasks and np.array(labels['dbboxes']).any():
            bboxes = 'dbboxes'
        elif 'obboxes' in tasks and np.array(labels['obboxes']).any():
            bboxes = 'hbboxes'

        for i, bbox_corners in enumerate(labels[bboxes]):
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

                task_result = labels
                for x in task_keys:
                    task_result = task_result[x]
                task_result = task_result[i]

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
                for i in range(len(labels[bboxes])):
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
                          iou_th=0.5,
                          conf_th=0.5):
    """
    Visualize results on images
    """
    logger.info(tasks)
    img_paths = get_file_paths(img_dir)
    label_paths = get_file_paths(result_dir)
    mask_paths = get_file_paths(mask_dir) if mask_dir else [None] * len(img_paths)
    assert len(img_paths) == len(label_paths) == len(mask_paths)
    for img_path, label_path, mask_path in tqdm(zip(img_paths, label_paths, mask_paths), total=len(img_paths)):
        img = cv2.imread(str(img_path))
        results0 = read_label(label_path, label_format='satellitepy')

        if len(results0['det_labels'][target_task]) == 0:
            print('skipping: No detections at all')
            continue

        results1 = remove_low_conf_results(results0, target_task, conf_th)
        results = apply_nms(results1['det_labels'], nms_iou_threshold=iou_th, target_task=target_task)

        if satellitepy_labels_empty(results):
            continue

        available_tasks = list(results.keys())

        if not all_tasks_flag:
            available_tasks = tasks.copy()

        if 'masks' in tasks:
            available_tasks.remove('masks')

        if 'obboxes' in available_tasks:
            available_tasks.remove('obboxes')
        if 'hbboxes' in available_tasks:
            available_tasks.remove('hbboxes')

        if results['obboxes'][0] is None:
            bboxes = 'hbboxes'
        else:
            bboxes = 'obboxes'

        for i, bbox_corners in enumerate(results[bboxes]):
            bbox_corners = np.array(bbox_corners, np.int32)
            x_min, x_max, y_min, y_max = BBox.get_bbox_limits(bbox_corners)
            cv2.polylines(img, [bbox_corners], True, color=(255, 0, 0))

            for j, task in enumerate(available_tasks):

                task_result = np.argmax(results[task][i])

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
    for feature in labels['features']:
        img_name = feature['properties']['image_id']
        if img_name != img_name_for_log:
            img_name_for_log = img_name
            logger.info(f'Following image will be written: {img_name_for_log}')

        type_class = int(feature['properties']['type_id'])

        if type_class not in classes['vehicles'] and \
           type_class not in classes['ships'] and \
           type_class not in classes['helicopter']:
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
