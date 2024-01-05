import json
import logging
import os
import shutil

import numpy as np
from pathlib import Path
import numpy as np
import logging

import cv2

from satellitepy.data.labels import read_label, init_satellitepy_label, fill_none_to_empty_keys, get_all_satellitepy_keys, satellitepy_labels_empty
from satellitepy.data.patch import get_patches
from satellitepy.data.chip import get_chips
from satellitepy.data.utils import get_xview_classes
from satellitepy.utils.path_utils import create_folder, zip_matched_files, get_file_paths
from satellitepy.data.bbox import BBox

def save_patches(
    image_folder,
    label_folder,
    label_format,
    out_folder,
    truncated_object_thr,
    patch_size,
    patch_overlap,
    include_object_classes,
    exclude_object_classes,
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
    include_object_classes: list[str]
        A list of object class names that shall be included as ground truth for the patches. 
        Takes precedence over exclude_object_classes, i.e. if both are provided, 
        only include_object_classes will be considered.
    exclude_object_classes: list[str]
        A list of object class names that shall be excluded as ground truth for the patches.
        include_object_classes takes precedence and overrides the behaviour of this parameter.
    Returns
    -------
    Save patches in <out-folder>/patch_<patch-size>/images and <out-folder>/patch_<patch-size>/labels
    """
    logger = logging.getLogger(__name__)

    # Create output folders
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

    assert len(img_paths)==len(label_paths)==len(mask_paths)

    for img_path, label_path, mask_path in zip(img_paths,label_paths,mask_paths):
        mask_name = mask_path.name if mask_path != None else None
        label_name = label_path.name if label_path is not None else None
        logger.info(f"{img_path.name}, {label_name}, {mask_name}")
        # Image
        img = cv2.imread(str(img_path))
        # Labels
        if label_path is not None:
            gt_labels = read_label(label_path,label_format,mask_path)
        else:
            gt_labels = None
        # Save results with the corresponding ground truth
        patches = get_patches(
            img,
            gt_labels,
            truncated_object_thr,
            patch_size,
            patch_overlap,
            include_object_classes,
            exclude_object_classes,
            label_path is not None
        )

        count_patches = len(patches['images'])
        logger.info(f'Number of patches: {count_patches}')
        for i in range(count_patches):
            # if satellitepy_labels_empty(patches["labels"][i]):
            #     continue

            # Get original image name for naming patch files
            img_name = img_path.stem

            # Patch starting coordinates
            patch_x0, patch_y0 = patches['start_coords'][i]

            # Save patch image
            patch_img = patches['images'][i]
            patch_image_path = Path(out_image_folder) / f"{label_format}_{img_name}_x_{patch_x0}_y_{patch_y0}.png"

            patch_label = patches['labels'][i]
            if patch_label is None or len(patch_label['hbboxes']) != 0:
                cv2.imwrite(str(patch_image_path), patch_img)

            # Save patch labels
            if patch_label is not None and len(patch_label['hbboxes']) != 0:
                patch_label_path = Path(out_label_folder) / f"{label_format}_{img_name}_x_{patch_x0}_y_{patch_y0}.json"
                with open(str(patch_label_path),'w') as f:
                    json.dump(patch_label,f,indent=4)

def save_chips(
    label_format,
    image_folder,
    label_folder,
    out_folder,
    margin_size,
    include_object_classes,
    exclude_object_classes,
    mask_folder = None
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
    out_folder_images = out_folder / "images"
    out_folder_labels = out_folder / "labels"

    assert create_folder(out_folder_images)
    assert create_folder(out_folder_labels)

    image_paths = get_file_paths(image_folder)
    label_paths = get_file_paths(label_folder)
    if mask_folder:
        mask_paths = get_file_paths(mask_folder)
    else:
        mask_paths = [None] * len(image_paths)

    if (len(image_paths)==len(label_paths)==len(mask_paths)):
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

                chip_img_path = out_folder_images / f"{img_name}_{i}.png"
                chip_img = chips['images'][i]

                if not chip_img.size == 0:
                    cv2.imwrite(str(chip_img_path), chip_img)
                else:
                    continue

                chip_label = get_label_by_idx(chips['labels'], i)
                chip_label_path = out_folder_labels / f"{img_name}_{i}.txt"

                with open(str(chip_label_path), 'w') as f:
                    json.dump(chip_label, f, indent=4)

def get_label_by_idx(satpy_labels: dict, i: int):
    """
    Creates a copy of the satpy_labels dict by doing the following:
    Sets each list to a singleton list correponding to the item at position i.
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
      
def show_results_on_image(img_dir, 
    result_dir, 
    out_dir, 
    tasks, 
    iou_th=0.5, 
    conf_th=0.5):
    """
    Visualize results on images
    """
    logger = logging.getLogger(__name__)
    logger.info(tasks)
    img_paths = get_file_paths(img_dir)
    label_paths = get_file_paths(result_dir)
    assert len(img_paths) == len(label_paths)#
        # for img_path, label_path, mask_path in zip(img_paths, label_paths, mask_paths):
    for img_path, label_path in zip(img_paths, label_paths):
        img = cv2.imread(str(img_path))
        logger.info(img_path)
        labels = read_label(label_path, label_format='satellitepy')

        # logger.info('Adding bounding boxes/labels to image')

        if satellitepy_labels_empty(labels):
            continue

        if labels['obboxes'][0] == None:
            bboxes = 'hbboxes'
        else:
            bboxes = 'obboxes'

        for i, bbox_corners in enumerate(labels[bboxes]):
            conf_score = labels['confidence-scores'][i]
            iou_score = labels['matches']['iou']['scores'][i]
            if conf_score < conf_th or iou_score < iou_th:
                continue
            bbox_corners = np.array(bbox_corners, np.int32)
            # bbox_corners = bbox_corners.reshape((-1, 1, 2))
            x_min, x_max, y_min, y_max = BBox.get_bbox_limits(bbox_corners)
            # ax.text(x=(x_max+x_min)/2,y=(y_max+y_min)/2 - 5 ,s=labels[classes[0]][i], fontsize=8, color='r', alpha=1, horizontalalignment='center', verticalalignment='bottom')
            cv2.putText(img,str(round(conf_score, 2)),(x_max,y_min),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
            cv2.polylines(img, [bbox_corners], True, color=(0,0,255))
                # BBox.plot_bbox(corners=bbox_corners, ax=ax, c='b', s=5)
            # fig.canvas.draw()
    
        if 'masks' in tasks:
            img_mask = np.zeros(shape=(img.shape[0],img.shape[1]),dtype=np.uint8)
            for mask in labels['masks']:
                if mask is not None:
                    x, y = mask
                    img_mask[y,x] = 1
            contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0,255,0), 1)

        logger.info(Path(out_dir) / f"{img_path.stem}.png")
        cv2.imwrite(str(Path(out_dir) / f"{img_path.stem}.png"), img)

            # plt.axis('off')
            # plt.show()
            # plt.savefig(out_dir / Path(img_folder.stem + ".png"))
            # logger.info(f'Saved labels on {out_dir / Path(img_folder.stem + ".png")}')
  

# def split_rareplanes_labels(
#         label_file,
#         out_folder
#     ):
#     """
#         Save patches from the original images
#         Parameters
#         ----------
#         label_file : Path
#             Input label file. This single label file will be split up.
#         out_folder : Path
#             Output folder. New labels will be saved into <out-folder>/labels
#         Returns
#         -------
#         Save labels in <out-folder>/labels
#         """

#     logger = logging.getLogger(__name__)

#     # Create output folder
#     out_label_folder = Path(out_folder)
#     assert create_folder(out_label_folder)

#     label_path = Path(label_file)

#     file = json.load(open(label_path, 'r'))

#     id_to_img = {}
#     for image in file['images']:
#         id_to_img[image['id']] = image['file_name']
#         label_file_path = os.path.join(out_label_folder, image['file_name'][:-3] + 'json')
#         label_file = open(label_file_path, 'w')
#         logger.info(f'Initializing annotations for {label_file_path}')
#         annotations = {'annotations': []}
#         json.dump(annotations, label_file, indent=4)
#         label_file.close()

#     for new_annotation in file['annotations']:
#         img_name = id_to_img[new_annotation['image_id']]
#         label_file_path = os.path.join(out_label_folder, img_name[:-3] + 'json')
#         label_file = open(label_file_path, 'r', encoding="utf-8")
#         logger.info(f'Saving annotation for {label_file_path}')
#         annotations = json.load(label_file)
#         label_file.close()
#         annotations['annotations'].append(new_annotation)
#         file = open(label_file_path, 'w', encoding="utf-8")

#         json.dump(annotations, file, ensure_ascii=False, indent=4)
#         file.close()

# def split_xview_labels(
#         label_file,
#         out_folder
#     ):
#     """
#         Save patches from the original images
#         Parameters
#         ----------
#         label_file : Path
#             Input label file. This single label file will be split up.
#         out_folder : Path
#             Output folder. New labels will be saved into <out-folder>/labels
#         Returns
#         -------
#         Save labels in <out-folder>/labels
#         """

#     logger = logging.getLogger(__name__)

#     # Create output folder
#     out_label_folder = Path(out_folder)
#     assert create_folder(out_label_folder)

#     label_path = Path(label_file)

#     file = json.load(open(label_path, 'r'))

#     id_to_img = {}
#     for image in file['features']:
#         id_to_img[image['properties']["image_id"]] = image['properties']['image_id']
#         label_file_path = os.path.join(out_label_folder, image['properties']["image_id"][:-3] + 'geojson')
#         label_file = open(label_file_path, 'w')
#         logger.info(f'Initializing annotations for {label_file_path}')
#         annotations = {'annotations': [image]}
#         json.dump(annotations, label_file, indent=4)
#         label_file.close()

#     for new_annotation in file['features']:
#         img_name = id_to_img[new_annotation["properties"]['image_id']]
#         label_file_path = out_label_folder / f"{img_name[:-3]}geojson"
#         if label_file_path.exists():
#             continue
#         label_file = open(label_file_path, 'r', encoding="utf-8")
#         logger.info(f'Saving annotation for {label_file_path}')
#         annotations = json.load(label_file)
#         label_file.close()
#         annotations['annotations'].append(new_annotation)

#         file = open(label_file_path, 'w', encoding="utf-8")

#         json.dump(annotations, file, ensure_ascii=False, indent=4)
#         file.close()

def save_xview_in_satellitepy_format(out_folder,label_path):
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
    # Create output folder
    logger = logging.getLogger(__name__)
    logger.info(f'Initializing save_xview_in_satellitepy_format')


    labels = json.load(open(label_path, 'r'))
    all_image_names = []
    for feature in labels['features']:
        all_image_names.append(feature['properties']['image_id'])

    image_dicts = {img_name:init_satellitepy_label() for img_name in set(all_image_names)}
    # Get all not available tasks so we can append None to those tasks
    ## Default available tasks for dota
    available_tasks=['hbboxes', 'coarse-class', 'fine-class', 'role']
    ## All possible tasks
    all_tasks = get_all_satellitepy_keys()
    ## Not available tasks
    not_available_tasks = [task for task in all_tasks if not task in available_tasks or available_tasks.remove(task)]

    classes = get_xview_classes()

    img_name_for_log = list(image_dicts.keys())[0]        
    for feature in labels['features']:
        img_name = feature['properties']['image_id']
        if img_name != img_name_for_log:
            img_name_for_log = img_name
            logger.info(f"Following image will be written: {img_name_for_log}")
        coords = feature['properties']['bounds_imcoords'].split(',')
        xmin = int(coords[0])
        ymin = int(coords[1])
        xmax = int(coords[2])
        ymax = int(coords[3])
        image_dicts[img_name]['hbboxes'].append([[xmin, ymax], [xmin, ymin], [xmax, ymin], [xmax, ymax]])

        type_class = int(feature['properties']['type_id'])
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
        elif type_class in classes['objects']:
            image_dicts[img_name]['coarse-class'].append('other')
            image_dicts[img_name]['fine-class'].append(classes['objects'][type_class])
            image_dicts[img_name]['role'].append(None)
        else:
            image_dicts[img_name]['coarse-class'].append('other')
            image_dicts[img_name]['fine-class'].append(None)
            image_dicts[img_name]['role'].append(None)

        fill_none_to_empty_keys(image_dicts[img_name],not_available_tasks)

    # Save satellitepy labels
    for img_name, satellitepy_dict in image_dicts.items():
        label_name = f"{Path(img_name).stem}.json"
        label_path = out_folder / label_name
        with open(str(label_path),'w') as f:
            json.dump(satellitepy_dict,f,indent=4)


def separate_dataset_parts(out_folder, label_folder, image_folder, dataset_part, dataset):
    """
    Parameters
    -------
    out_folder : Path
        Output folder. Images and labels will be saved into <out-folder>
    Returns
    -------
    Split the images and labels according to the dataset_part file
    """
    # Create outut folder
    logger = logging.getLogger(__name__)
    logger.info(f'Initializing separate_shipnet_data')

    dataset_name = dataset_part.stem
    out_image_folder = os.path.join(out_folder, Path('images'))
    assert create_folder(Path(out_image_folder))

    if dataset != 'shipnet' or dataset_name != 'test':
        out_label_folder = os.path.join(out_folder, Path('labels'))
        assert(create_folder(Path(out_label_folder)))

    with open(dataset_part, 'r') as dataset:
        for line in dataset.readlines():
            name = line.split('.')[0][:-1]

            if dataset == 'shipnet':
                extension = '.bmp'
            else:
                extension = '.jpg'

            image_path = os.path.join(image_folder, Path(name + extension))

            shutil.copy(image_path, out_image_folder)

            if dataset != 'shipnet' or dataset_name != 'test':
                padding = max(0, 6 - name.__len__())
                if dataset == 'dior':
                    padding = 0
                label_path = os.path.join(label_folder, Path(padding * '0' + name + '.xml'))
                shutil.copy(label_path, out_label_folder)
    logger.info(f'Images and labels saved for dataset-part {dataset_name}')



