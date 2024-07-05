"""
Toolset for creating chips
"""
import math

import cv2
import numpy as np
from satellitepy.data.bbox import BBox
from satellitepy.data.labels import init_satellitepy_label, get_all_satellitepy_keys, set_image_keys


def create_chip(img, bbox, chip_size, margin=0, draw_corners=False):
    center_x = np.mean(bbox[:, 0])
    center_y = np.mean(bbox[:, 1])
    angle = BBox(corners=bbox).get_orth_angle()

    if draw_corners:
        img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=0)
        center_x += 200
        center_y += 200
        bbox += 200
        for coords in bbox:
            cv2.circle(img, (coords[0], coords[1]), 1, (0, 0, 255), 2)

    center = (center_x, center_y)

    M = cv2.getRotationMatrix2D(center, math.degrees(angle)-90, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    width, height, _ = rotated.shape
    rot_bbox = np.array(BBox(corners=bbox).rotate_corners(-angle), dtype=int)
    rot_hbbox = BBox.get_bbox_limits(rot_bbox)

    rot_hbbox = apply_margin(rot_hbbox, margin, height, width, chip_size)

    chip_img = rotated[rot_hbbox[2]:rot_hbbox[3], rot_hbbox[0]:rot_hbbox[1], :]

    return chip_img, (int(center_x), int(center_y))


def apply_margin(hbbox, margin_size, max_y, max_x, chip_size):
    x_0 = min(max(hbbox[0] - margin_size, 0), max_x)
    x_1 = min(max(hbbox[1] + margin_size, 0), max_x)
    y_0 = min(max(hbbox[2] - margin_size, 0), max_y)
    y_1 = min(max(hbbox[3] + margin_size, 0), max_y)

    x_0, x_1 = adjust_line_length(x_0, x_1, chip_size)
    y_0, y_1 = adjust_line_length(y_0, y_1, chip_size)
    
    return np.array([x_0, x_1, y_0, y_1])

def adjust_line_length(x1, x2, desired_length): 
    center = (x1 + x2) / 2
    half_desired_length = desired_length / 2
    
    # Calculate the new points while keeping the center fixed
    new_x1 = center - half_desired_length
    new_x2 = center + half_desired_length
    
    return int(new_x1), int(new_x2)

def get_chips(img, labels, task=None, margin_size=50, chip_size=128):
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

    if any(labels['obboxes']):
        bbox_type = "obboxes"
    else:
        bbox_type = "hbboxes"

    bboxes = labels[bbox_type]

    for i, bbox in enumerate(bboxes):

        chip_img, center = create_chip(img=img, bbox=np.array(bbox).astype(int), chip_size=chip_size, margin=margin_size, draw_corners=False)

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

