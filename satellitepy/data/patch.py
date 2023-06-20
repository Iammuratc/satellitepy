import numpy as np
import logging
import shapely
from shapely.geometry import Polygon

from satellitepy.data.labels import init_satellitepy_label, get_all_satellitepy_keys, set_image_keys
# TODO: 
#   Filter out the truncated objects using the object area. truncated_object_thr is not use at the moment. Edit the is_truncated function.

# Init log
logger = logging.getLogger(__name__)

def get_patches(
    img,
    gt_labels,
    truncated_object_thr,
    patch_size,
    patch_overlap,
    include_object_classes,
    exclude_object_classes,
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
    include_object_classes: list[str]
        A list of object class names that shall be included as ground truth for the patches. 
        Takes precedence over exclude_object_classes, i.e. if both are provided, 
        only include_object_classes will be considered.
    exclude_object_classes: list[str]
        A list of object class names that shall be excluded as ground truth for the patches.
        include_object_classes takes precedence and overrides the behaviour of this parameter.
    mask : np.ndarray
        Mask Image
    Returns
    -------
    patch_dict : dict
        This dict includes patches and the corresponding labels in satellitepy format
    """

    # Get image and mask image shape 
    y_max, x_max, ch = img.shape

    # Pad image and mask image so full patches are possible
    x_pad_size = get_pad_size(x_max,patch_size,patch_overlap)
    y_pad_size = get_pad_size(y_max,patch_size,patch_overlap)
    img_padded = np.pad(img,pad_width=((0,y_pad_size),(0,x_pad_size),(0,0)))

    y_max_padded, x_max_padded, ch = img_padded.shape

    # Patch coordinates in the padded image and mask image
    y_start_coords =  get_patch_start_coords(y_max_padded,patch_size,patch_overlap)
    x_start_coords =  get_patch_start_coords(x_max_padded,patch_size,patch_overlap)
    patch_start_coords = [[x,y] for x in x_start_coords for y in y_start_coords]

    # Init patch dictionary
    patch_dict = {
      'images':[np.empty(shape=(patch_size, patch_size, ch), dtype=np.uint8) for _ in range(len(patch_start_coords))],
      'labels':[init_satellitepy_label() for _ in range(len(patch_start_coords))], # label_key:[] for label_key in gt_labels.keys()
      'start_coords': patch_start_coords,
      }

    for i,patch_start_coord in enumerate(patch_start_coords):
        # Patch starting coordinates
        x_0,y_0 = patch_start_coord
        patch_dict['images'][i] = img_padded[y_0:y_0+patch_size,x_0:x_0+patch_size,:]

        # Patch labels
        for j, (hbbox, obbox) in enumerate(zip(gt_labels['hbboxes'],gt_labels['obboxes'])):
            hbb_defined = hbbox != None
            obb_defined = obbox != None

            class_targets = [gt_labels['coarse-class'][j], gt_labels['fine-class'][j], gt_labels['very-fine-class'][j]]
            if not any([
                is_valid_object_class(
                    ct, include_object_classes, exclude_object_classes
                ) for ct in class_targets]
            ):
                continue

            if hbb_defined and obb_defined:
                shift_bboxes(patch_dict, gt_labels, j, i , 'obboxes', patch_start_coord, obbox, patch_size, truncated_object_thr, consider_additional=True)

            elif hbb_defined:
                shift_bboxes(patch_dict, gt_labels, j, i , 'hbboxes', patch_start_coord, hbbox, patch_size, truncated_object_thr)

            elif obb_defined:
                shift_bboxes(patch_dict, gt_labels, j, i , 'obboxes', patch_start_coord, obbox, patch_size, truncated_object_thr)
                
            else:
                logger.error('Error reading bounding boxes! No bounding boxes found')
                exit(1)
            
    return patch_dict
    
def shift_bboxes(patch_dict, gt_labels, j, i, bboxes, patch_start_coord, bbox_corners, patch_size, truncated_object_thr, consider_additional=False, additional='hbboxes'):
    x_0, y_0 = patch_start_coord
    is_truncated_bbox = is_truncated(bbox_corners=bbox_corners, x_0=x_0, y_0=y_0, patch_size=patch_size, relative_area_threshold=truncated_object_thr)
    if not is_truncated_bbox:
        # for key in keys_with_values:
        # patch_dict['labels'][i][key].append(gt_labels[key][i_label])
        patch_dict['labels'][i] = set_image_keys(get_all_satellitepy_keys(), patch_dict['labels'][i], gt_labels, j)
        # Since patches are cropped out, the image patch coordinates shift, so Bbox values should be shifted as well.
        bbox_corners_shifted = np.array(patch_dict['labels'][i][bboxes][-1]) - [x_0, y_0]
        patch_dict['labels'][i][bboxes][-1] = bbox_corners_shifted.tolist()
        if patch_dict["labels"][i]["masks"][-1] is not None:
            mask_shifted = np.array(patch_dict['labels'][i]['masks'][-1]) - np.array([x_0, y_0]).reshape(2,1)
            patch_dict['labels'][i]['masks'][-1] = mask_shifted.tolist()
        if consider_additional:
            patch_dict['labels'][i] = set_image_keys(get_all_satellitepy_keys(), patch_dict['labels'][i], gt_labels, j)
            bbox_corners_shifted = np.array(patch_dict['labels'][i][additional][-1]) - [x_0, y_0]
            patch_dict['labels'][i][additional][-1] = bbox_corners_shifted.tolist()

def is_valid_object_class(object_class_name, include_object_classes, exclude_object_classes):
    """
    Checks whether the given object class name is valid w.r.t. 
    the given include_object_classes and exclude_object_classes lists.
    If include_object_classes is not None, this function will return whether the object_class_name
    is included in that list.
    If exclude_object_classes is not None and include_object_classes is None, this function will
    return whether the given object_class_name is not in the exclude list.
    Parameters
    ----------
    object_class_name : str
        The name of the object class.
    include_object_classes: list[str]
        A list of object class names that shall be included as ground truth for the patches. 
        Takes precedence over exclude_object_classes, i.e. if both are provided, 
        only include_object_classes will be considered.
    exclude_object_classes: list[str]
        A list of object class names that shall be excluded as ground truth for the patches.
        include_object_classes takes precedence and overrides the behaviour of this parameter.
    """
    if object_class_name is None:
        return False
    elif include_object_classes is not None:
        return object_class_name in include_object_classes
    elif exclude_object_classes is not None:
        return object_class_name not in exclude_object_classes
    else:
        return True

def get_pad_size(coord_max, patch_size, patch_overlap):
    """
    Get patch starting coordinates with respect to original image size.
    Parameters
    ----------
    coord_max : int
        Maximum value of the given axis.
    patch_size : int
        Patch size
    overlap : int
        Overlapping part between the neighboring patches
    Returns
    -------
    pad_size : int
        Pad size of the given axis
        E.g., img size = 1000, patch size = 512, overlap = 0, resulting image = 1024 
        E.g., img size = 1000, patch size = 512, overlap = 10, resulting image = 1014
    """
    pad_size = 0
    quotient, remainder = divmod(coord_max+patch_overlap, patch_size)
    if quotient==0 and remainder==0:
        msg = 'No pixels are found in image!'
        logger.error(msg)
        raise Exception(msg)
    if remainder != 0:
        new_coord_size = (quotient+1)*patch_size-patch_overlap
        pad_size = new_coord_size - coord_max
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
    overlap : int
        Overlapping part between the neighboring patches
    Returns
    -------
    coords : list
        Starting coordinates of the given axis
    """
    quotient, remainder = divmod(coord_max+patch_overlap, patch_size)
    if remainder != 0:
        msg = f"{remainder} number of pixels will be discarded"
        logger.warning(msg)
    
    coords = [0]
    for i in range(1,quotient):
        coords.append(i*patch_size-patch_overlap)
    return coords

def is_truncated(bbox_corners,x_0,y_0,patch_size,relative_area_threshold):
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
    patch_coords = ((x_0,y_0),(x_0,y_0+patch_size),(x_0+patch_size,y_0+patch_size),(x_0+patch_size,y_0))
    patch = Polygon(patch_coords)
    bbox = Polygon(bbox_corners)
    return relative_area_threshold >= shapely.area(shapely.intersection(bbox, patch))/shapely.area(bbox)


def merge_patch_results(patch_dict):
    """
    Merge the patch results into original image standards
    Parameters
    ----------
    patch_dict : dict
        Dict of patches of an original image with the following keys. 'images', 'labels', 'start_coords', 'det_labels' 
        'det_labels' and 'start_coords' will be used.
    Returns
    -------
    merged_det_labels : dict
        Merged 'det_labels' dict in satellitepy format
    """
    merged_det_labels = { key:[] for key in patch_dict['det_labels'][0].keys()}
    for i,(x_0,y_0) in enumerate(patch_dict['start_coords']):
        # Merge all the keys to merged_det_labels
        # If key=='bboxes', first shift, then merge
        for key in merged_det_labels.keys():
            patch_bbox_corners = patch_dict['det_labels'][i]['bboxes']
            if key == 'bboxes' and patch_bbox_corners != []:
                bbox_corners_shifted = np.array(patch_bbox_corners) + np.array([x_0,y_0])
                merged_det_labels[key].extend(bbox_corners_shifted.tolist())
            else:
                merged_det_labels[key].extend(patch_dict['det_labels'][i][key])

    return merged_det_labels
