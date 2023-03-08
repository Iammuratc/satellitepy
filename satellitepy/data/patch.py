import numpy as np
import logging
# TODO: 
#   Shift the segmentation masks in get_patches and merge_patch_results
#   Filter out the truncated objects using the object area. truncated_object_thr is not use at the moment. Edit the is_truncated function.

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

    # Patch coordinates in the original image
    logger = logging.getLogger(__name__)
    y_max, x_max, ch = img.shape
    y_start_coords =  get_patch_start_coords(y_max,patch_size,patch_overlap)
    x_start_coords =  get_patch_start_coords(x_max,patch_size,patch_overlap)
    patch_start_coords = [[x,y] for x in x_start_coords for y in y_start_coords]
    patch_dict = {
      'images':[np.empty(shape=(patch_size, patch_size, ch), dtype=np.uint8) for _ in range(len(patch_start_coords))],
      'labels':[{label_key:[] for label_key in gt_labels.keys()} for _ in range(len(patch_start_coords))], # label_key:[] for label_key in gt_labels.keys()
      'start_coords': patch_start_coords
      }

    # Not every dataset has values for all possible keys in labels. 
    # E.g., fair1m does not have difficulty. 
    # Remove keys with empty lists
    keys_with_values = [key for key in gt_labels.keys() if gt_labels[key]!=[]]
    logger.info("{} have values for this image".format(",".join(keys_with_values)))
    for i,patch_start_coord in enumerate(patch_start_coords):
        # Patch starting coordinates
        x_0,y_0 = patch_start_coord

        # Patch image
        patch_dict['images'][i] = img[y_0:y_0+patch_size,x_0:x_0+patch_size,:]

        # Patch labels
        for i_label, bbox_corners in enumerate(gt_labels['bboxes']):
            # Check if object s bbox is in patch
            is_truncated_bbox = is_truncated(
                bbox_corners=bbox_corners,
                x_0=x_0,
                y_0=y_0,
                patch_size=patch_size,
                bbox_corner_threshold=2)
            if not is_truncated_bbox:
                for key in keys_with_values:
                    patch_dict['labels'][i][key].append(gt_labels[key][i_label])

                # Since patches are cropped out, the image patch coordinates shift, so Bbox values should be shifted as well.
                bbox_corners_shifted = np.array(patch_dict['labels'][i]['bboxes'][-1]) - [x_0,y_0]
                patch_dict['labels'][i]['bboxes'][-1] = bbox_corners_shifted.tolist()
            else:
                continue
    return patch_dict

def get_patch_start_coords(coord_max, patch_size, overlap):
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
    coords = []
    stride = patch_size - overlap
    coord_i = 0
    while (coord_i + stride) < coord_max:
        coords.append(coord_i)
        coord_i = coord_i + stride
    return coords

def is_truncated(bbox_corners,x_0,y_0,patch_size,bbox_corner_threshold):
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
        bbox_corner_threshold : int
            Number of corners that should be in the patch
    Returns
    ------
        is_truncated : bool
            False if the object is in the patch
    """
    bbox_corners_in_patch = 0
    for coord in bbox_corners:
        if (x_0 <= coord[0] <= x_0 + patch_size) and (
                y_0 <= coord[1] <= y_0 + patch_size):
            bbox_corners_in_patch += 1
    if bbox_corners_in_patch >= bbox_corner_threshold:
        return False
    else:
        return True


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

