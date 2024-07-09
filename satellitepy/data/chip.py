"""
Toolset for creating chips
"""
import math

import cv2
import numpy as np
from satellitepy.data.bbox import BBox
from satellitepy.data.labels import init_satellitepy_label, get_all_satellitepy_keys, set_image_keys
import logging

logger = logging.getLogger('')


def remove_neighbor_masks(mask):
    '''
    Remove the neighboring masks from the mask images.
    11----1111----1 >> ------1111-----
    Parameters
    -----------
    mask : np.ndarray
        Binary mask image
    Returns
    -------
    mask_middle : np.ndarray
        Mask image only with the mask in the center
    '''
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if labels is None:
        return mask
    # Get the image center
    image_center = (mask.shape[1] // 2, mask.shape[0] // 2)

    # Get the label of the component that includes the center of the image
    central_label = labels[image_center[1], image_center[0]]

    # Create a mask to keep only the central component
    mask_middle = (labels == central_label).astype(np.uint8) * 255

    # Zero out other components
    # chip_img = cv2.bitwise_and(chip_img, chip_img, mask=mask_middle)
    return mask_middle


def create_chip(img, mask, bbox, chip_size, draw_corners=False, orient_objects=False):
    center_x = np.mean(bbox[:, 0])
    center_y = np.mean(bbox[:, 1])
    center = (center_x, center_y)

    if (mask is None) or (not np.any(mask)):
        mask_objects = False
    else:
        mask_objects = True
    # Set chip mask to None
    if draw_corners:
        img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=0)
        center_x += 200
        center_y += 200
        bbox += 200
        for coords in bbox:
            cv2.circle(img, (coords[0], coords[1]), 1, (0, 0, 255), 2)

    if orient_objects:
        angle = BBox(corners=bbox).get_orth_angle()
        M = cv2.getRotationMatrix2D(center, math.degrees(angle)-90, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        max_y, max_x, _ = rotated.shape
        rot_bbox = np.array(BBox(corners=bbox).rotate_corners(-angle), dtype=int)
        rot_hbbox = BBox.get_bbox_limits(rot_bbox)
        rot_hbbox = get_chip_coords(rot_hbbox, max_x, max_y, chip_size)
        chip_img = rotated[rot_hbbox[2]:rot_hbbox[3], rot_hbbox[0]:rot_hbbox[1], :]
        
        if mask_objects:
            rotated_mask = cv2.warpAffine(mask, M, (img.shape[1], img.shape[0]))
            chip_mask = rotated_mask[rot_hbbox[2]:rot_hbbox[3], rot_hbbox[0]:rot_hbbox[1]]
            chip_mask = remove_neighbor_masks(chip_mask)
            chip_img = cv2.bitwise_and(chip_img, chip_img, mask=chip_mask)

    else:
        max_x, max_y, _ = img.shape
        chip_coords = get_chip_coords(bbox, max_x, max_y, chip_size)
        chip_img = img[chip_coords[2]:chip_coords[3], chip_coords[0]:chip_coords[1], :]
        if mask_objects:
            chip_mask = mask[chip_coords[2]:chip_coords[3], chip_coords[0]:chip_coords[1]]
            chip_mask = remove_neighbor_masks(chip_mask)
            chip_img = cv2.bitwise_and(chip_img, chip_img, mask=chip_mask)

    return chip_img, (int(center_x), int(center_y))


def get_chip_coords(bbox, max_y, max_x, chip_size):
    x_c, y_c, _, _, _ = BBox(corners=bbox).params
    chip_coords = np.array([x_c-chip_size/2, x_c+chip_size/2, y_c-chip_size/2, y_c+chip_size/2]).astype(int)
    assert any(chip_coords) > 0, "Chip coordinates can not be less than 0."
    return chip_coords

def adjust_line_length(x1, x2, desired_length): 
    center = (x1 + x2) / 2
    half_desired_length = desired_length / 2
    
    # Calculate the new points while keeping the center fixed
    new_x1 = center - half_desired_length
    new_x2 = center + half_desired_length
    
    return int(new_x1), int(new_x2)

def get_chips(img, 
    labels, 
    task=None, 
    chip_size=128,
    orient_objects=False,
    mask_objects=False):
    
    all_satellitepy_keys = get_all_satellitepy_keys()

    chips_dict = {
        'images': [],
        'labels': init_satellitepy_label(),
        'attributes': {
            'center': [],
            'lengths': [],
            'widths': [],
            'task': []
        }
    }

    # Pad images, masks and labels to avoid minus values in bbox limits
    pad_size = int(chip_size/2)
    pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0, 0)) # y,x,ch

    if any(labels['obboxes']):
        bbox_type = "obboxes"
    else:
        bbox_type = "hbboxes"

    bboxes = labels[bbox_type]
    if mask_objects:
        logger.info("Mask will be applied to the image")
        mask = np.zeros(shape=(img.shape[0],img.shape[1])).astype(np.uint8)
        for i, bbox in enumerate(bboxes):
            cv2.fillPoly(mask, np.array([bbox], dtype=int), 255)
        ## Pad mask
        mask = np.pad(mask, pad_width[:2], mode='constant', constant_values=0)
    else:
        mask = None
    ## Pad image
    img = np.pad(img, pad_width, mode='constant', constant_values=0)

    ## Pad bboxes
    bboxes = np.array(bboxes) + np.array([pad_width[1][0], pad_width[0][0]])
    for i, bbox in enumerate(bboxes):
        chip_img, center = create_chip(img=img,
            mask = mask,
            bbox=np.array(bbox).astype(int), 
            chip_size=chip_size, 
            draw_corners=False,
            orient_objects=orient_objects)

        set_image_keys(all_satellitepy_keys, chips_dict['labels'], labels, i)

        if task:
            chips_dict['attributes']['task'].append(labels[task][i])

        chips_dict['attributes']['center'].append(center)
        params = BBox(corners=bbox).get_params()
        o_length = params[3]
        o_width = params[2]
        chips_dict['attributes']['lengths'].append(o_length)
        chips_dict['attributes']['widths'].append(o_width)
        chips_dict['images'].append(chip_img)
    
    return chips_dict

