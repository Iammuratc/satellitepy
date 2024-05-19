"""
Toolset for creating chips
"""
import numpy as np
from satellitepy.data.bbox import BBox
from satellitepy.data.labels import init_satellitepy_label, get_all_satellitepy_keys, set_image_keys


def create_chip(img, hbbox):
    chip_img = img[hbbox[2]:hbbox[3], hbbox[0]:hbbox[1], :]

    return chip_img


def apply_margin(hbbox, margin_size, max_y, max_x):
    x_0 = min(max(hbbox[0] - margin_size, 0), max_x)
    x_1 = min(max(hbbox[1] + margin_size, 0), max_x)
    y_0 = min(max(hbbox[2] - margin_size, 0), max_y)
    y_1 = min(max(hbbox[3] + margin_size, 0), max_y)

    return np.array([x_0, x_1, y_0, y_1])


def get_chips(img, gt_labels, margin_size=100):
    height, width, channels = img.shape

    all_satellitepy_keys = get_all_satellitepy_keys()

    chips_dict = {
        'images': [],
        'labels': init_satellitepy_label()
    }

    if [None] * len(gt_labels['obboxes']) == gt_labels['obboxes']:
        bbox_type = "hbboxes"
    else:
        bbox_type = "obboxes"

    bboxes = gt_labels[bbox_type]

    for i, bbox in enumerate(bboxes):
        bbox = np.array(bbox).astype(int)

        if bbox_type == "obboxes":
            hbbox = BBox.get_bbox_limits(bbox)
        else:
            hbbox = bbox

        hbbox = apply_margin(hbbox, margin_size, height, width)

        chip_img = create_chip(img, hbbox)
        chip_bbox = bbox - [hbbox[0], hbbox[2]]

        set_image_keys(all_satellitepy_keys, chips_dict['labels'], gt_labels, i)

        chips_dict['labels'][bbox_type][-1] = [chip_bbox.tolist()]
        chips_dict['images'].append(chip_img)

        if gt_labels['masks'][i] != None:
            chip_mask = np.array(gt_labels['masks'][i])

            chip_mask[0] -= hbbox[0]
            chip_mask[1] -= hbbox[2]

            chips_dict['labels']['masks'][-1] = chip_mask.tolist()

    return chips_dict

