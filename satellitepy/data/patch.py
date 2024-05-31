import numpy as np
import logging
import shapely
from shapely.geometry import Polygon

from satellitepy.data.labels import init_satellitepy_label, get_all_satellitepy_keys, set_image_keys

logger = logging.getLogger('')


def get_patches(
        img,
        gt_labels,
        truncated_object_thr,
        patch_size,
        patch_overlap,
):
    """
    Produce patches from the original image using the labels. 
    Labels in patches exist if the bounding box of the object is in the patch.
    Original Bounding box coordinates of the objects are adjusted with respect to the patch coordinates 
    Parameters
    ----------
    img : np.ndarray
        Image.
    gt_labels : dict
        Label in satellitepy format.
    truncated_object_thr : float
        Truncated object threshold
    patch_size : int
        Patch size
    patch_overlap : int
        Patch overlap
    Returns
    -------
    patch_dict : dict
        This dict includes patches and the corresponding labels in satellitepy format
    """

    y_max, x_max, ch = img.shape

    x_pad_size = get_pad_size(x_max, patch_size, patch_overlap)
    y_pad_size = get_pad_size(y_max, patch_size, patch_overlap)
    img_padded = np.pad(img, pad_width=((0, y_pad_size), (0, x_pad_size), (0, 0)))

    y_max_padded, x_max_padded, ch = img_padded.shape

    y_start_coords = get_patch_start_coords(y_max_padded, patch_size, patch_overlap)
    x_start_coords = get_patch_start_coords(x_max_padded, patch_size, patch_overlap)
    patch_start_coords = [[x, y] for x in x_start_coords for y in y_start_coords]

    label_file_exist = gt_labels is not None

    patch_dict = {
        'images': [np.empty(shape=(patch_size, patch_size, ch), dtype=np.uint8) for _ in
                   range(len(patch_start_coords))],
        'labels': [init_satellitepy_label() for _ in range(len(patch_start_coords))],
        'start_coords': patch_start_coords,
    }

    for i, patch_start_coord in enumerate(patch_start_coords):
        x_0, y_0 = patch_start_coord
        patch_dict['images'][i] = img_padded[y_0:y_0 + patch_size, x_0:x_0 + patch_size, :]

        if label_file_exist:
            for j, (hbbox, obbox) in enumerate(zip(gt_labels['hbboxes'], gt_labels['obboxes'])):
                hbb_defined = np.array(hbbox).any()
                obb_defined = np.array(obbox).any()

                if hbb_defined and obb_defined:
                    shift_bboxes(patch_dict, gt_labels, j, i, 'obboxes', patch_start_coord, obbox, patch_size,
                                 truncated_object_thr, consider_additional=True)

                elif hbb_defined:
                    shift_bboxes(patch_dict, gt_labels, j, i, 'hbboxes', patch_start_coord, hbbox, patch_size,
                                 truncated_object_thr)

                elif obb_defined:
                    shift_bboxes(patch_dict, gt_labels, j, i, 'obboxes', patch_start_coord, obbox, patch_size,
                                 truncated_object_thr)
                else:
                    logger.error('No bounding boxes found!')
    return patch_dict


def shift_bboxes(patch_dict, gt_labels, j, i, bboxes, patch_start_coord, bbox_corners, patch_size, truncated_object_thr,
                 consider_additional=False, additional='hbboxes'):
    x_0, y_0 = patch_start_coord
    is_truncated_bbox = is_truncated(bbox_corners=bbox_corners, x_0=x_0, y_0=y_0, patch_size=patch_size,
                                     relative_area_threshold=truncated_object_thr)
    if not is_truncated_bbox:
        patch_dict['labels'][i] = set_image_keys(get_all_satellitepy_keys(), patch_dict['labels'][i], gt_labels, j)

        bbox_corners_shifted = np.array(patch_dict['labels'][i][bboxes][-1]) - [x_0, y_0]
        patch_dict['labels'][i][bboxes][-1] = bbox_corners_shifted.tolist()
        if patch_dict['labels'][i]['masks'][-1] is not None:
            mask_shifted = np.array(patch_dict['labels'][i]['masks'][-1]) - np.array([x_0, y_0]).reshape(2, 1)

            mask_shifted[0][mask_shifted[0] >= patch_size] = patch_size - 1
            mask_shifted[0][mask_shifted[0] < 0] = 0
            mask_shifted[1][mask_shifted[1] >= patch_size] = patch_size - 1
            mask_shifted[1][mask_shifted[1] < 0] = 0
            patch_dict['labels'][i]['masks'][-1] = mask_shifted.tolist()
        if consider_additional:
            bbox_corners_shifted = np.array(patch_dict['labels'][i][additional][-1]) - [x_0, y_0]
            patch_dict['labels'][i][additional][-1] = bbox_corners_shifted.tolist()


def get_pad_size(coord_max, patch_size, patch_overlap):
    """
    Get patch starting coordinates with respect to original image size.
    Parameters
    ----------
    coord_max : int
        Maximum value of the given axis.
    patch_size : int
        Patch size
    patch_overlap : int
        Overlapping part between the neighboring patches
    Returns
    -------
    pad_size : int
        Pad size of the given axis
        E.g., img size = 1000, patch size = 512, overlap = 0, resulting image = 1024 
        E.g., img size = 1000, patch size = 512, overlap = 10, resulting image = 1014
    """
    pad_size = 0
    quotient, remainder = divmod(coord_max, patch_size - patch_overlap)
    if quotient == 0 and remainder == 0:
        msg = 'No pixels are found in image!'
        logger.error(msg)
        raise Exception(msg)
    elif remainder != 0:
        pad_size = patch_size - remainder
    elif coord_max <= patch_size:
        pad_size = patch_size - coord_max
    return pad_size


def get_patch_start_coords(coord_max, patch_size, patch_overlap):
    """
    Get patch starting coordinates with respect to original image size.
    Parameters
    ----------
    coord_max : int
        Maximum value of the given axis.
    patch_size : int
        Patch size
    patch-overlap : int
        Overlapping part between the neighboring patches
    Returns
    -------
    coords : list
        Starting coordinates of the given axis
    """
    quotient, remainder = divmod(coord_max, patch_size - patch_overlap)

    coords = []
    for i in range(quotient):
        coords.append(i * (patch_size - patch_overlap))

    if coords[-1] + patch_size >= coord_max:
        del coords[-1]
    coords.append(coord_max - patch_size)

    return coords


def is_truncated(bbox_corners, x_0, y_0, patch_size, relative_area_threshold):
    """
    Check if bbox is in the patch
    Parameters
    ----------
    bbox_corners : list
        Bounding box corners in shape (4,2)
    x_0 : int
        x coordinate of patch start
    y_0 : int
        y coordinate of patch start
    patch_size : int
        Patch size
    relative_area_threshold : float
        % of object that should be in the image in range of [0.0, 1.0]
    Returns
    ------
    is_truncated : bool
        False if the part of the object inside the patch is smaller than the threshold
    """
    patch_coords = ((x_0, y_0), (x_0, y_0 + patch_size), (x_0 + patch_size, y_0 + patch_size), (x_0 + patch_size, y_0))
    patch = Polygon(patch_coords)
    bbox = Polygon(bbox_corners)
    return relative_area_threshold >= shapely.area(shapely.intersection(bbox, patch)) / shapely.area(bbox)


def merge_patch_results(patch_dict, patch_size, shape):
    """
    Merge the patch results into original image standards
    Parameters
    ----------
    patch_dict : dict
        Dict of patches of an original image with the following keys. 'images', 'labels', 'start_coords', 'det_labels' 
        'det_labels' and 'start_coords' will be used.
    patch_size : int
        Patch size
    shape: (int, int)
        Shape of the patch
    Returns
    -------
    merged_det_labels : dict
        Merged 'det_labels' dict in satellitepy format
    """
    merged_det_labels = {key: [] for key in patch_dict['det_labels'][0].keys()}
    for i, (x_0, y_0) in enumerate(patch_dict['start_coords']):
        for key in merged_det_labels.keys():
            if (key == 'hbboxes' or key == 'obboxes') and patch_dict['det_labels'][i][key] != []:
                bbox_corners_shifted = np.array(patch_dict['det_labels'][i][key]) + np.array([x_0, y_0])
                merged_det_labels[key].extend(bbox_corners_shifted.tolist())
            else:
                merged_det_labels[key].extend(patch_dict['det_labels'][i][key])

    mask = np.zeros((shape[0], shape[1]), dtype=float)
    if 'masks' not in patch_dict:
        return merged_det_labels, None

    elif np.array(patch_dict['masks']).any():
        for i, (x_0, y_0) in enumerate(patch_dict['start_coords']):
            offset_x = 0 if x_0 == 0 else 1
            offset_y = 0 if y_0 == 0 else 1

            patch_mask = patch_dict['masks'][i]

            x_min = x_0
            x_max = np.min([x_0 + patch_size, shape[1]]) - 1
            y_min = y_0
            y_max = np.min([y_0 + patch_size, shape[0]]) - 1

            new_mask = np.zeros(mask.shape)
            patch_mask = patch_mask[offset_y:y_max - y_min, offset_x:x_max - x_min]
            new_mask[y_min + offset_y:y_max, x_min + offset_x:x_max] = patch_mask

            mask = np.maximum(mask, new_mask)

    return merged_det_labels, mask
