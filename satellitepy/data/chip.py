'''
Toolset for creating chips
'''
import numpy as np
from satellitepy.data.cutout.geometry import BBox
from satellitepy.data.labels import init_satellitepy_label, get_all_satellitepy_keys

def create_chip(img, hbbox):
    chip_img = img[hbbox[2]:hbbox[3], hbbox[0]:hbbox[1], :]

    return chip_img


def get_hbbox(bbox, margin_size, max_y, max_x):
    x_min, x_max, y_min, y_max = BBox.get_bbox_limits(bbox)

    x_0 = min(max(x_min - margin_size, 0), max_x)
    x_1 = min(max(x_max + margin_size, 0), max_x)
    y_0 = min(max(y_min - margin_size, 0), max_y)
    y_1 = min(max(y_max + margin_size, 0), max_y)

    return x_0, x_1, y_0, y_1

def get_chips(img, gt_labels, margin_size=100):
    height, width, channels = img.shape
    
    all_satellitepy_keys = get_all_satellitepy_keys()

    chips_dict = {
        'images' : [],
        'labels' : init_satellitepy_label()
    }

    bbox_type = ""

    if "obboxes" in gt_labels:
        bbox_type = "obboxes"
    else:
        bbox_type = "hbboxes"

    bboxes = gt_labels[bbox_type]

    for i, bbox in enumerate(bboxes):
        bbox = np.array(bbox)
        hbbox = get_hbbox(bbox, margin_size, height, width)

        chip_img = create_chip(img, hbbox)
        chip_bbox = bbox - [hbbox[0], hbbox[2]]
        
        set_chip_keys(all_satellitepy_keys, chips_dict['labels'], gt_labels, i)

        chips_dict[bbox_type][-1] = chip_bbox.tolist()
        chips_dict['images'].append(chip_img)

    return chips_dict


def set_chip_keys(
    all_satellitepy_keys,
    chip_labels,
    gt_labels,
    i):

    for task in all_satellitepy_keys:
        keys = task.split("_")

        if len(keys) == 1:
            chip_labels[keys[0]].append(gt_labels[keys[0]][i])
        elif len(keys) == 2:
            chip_labels[keys[0]][keys[1]].append(gt_labels[keys[0]][keys[1]][i])
        elif len(keys) == 3:
            chip_labels[keys[0]][keys[1]][keys[2]].append(gt_labels[keys[0]][keys[1]][keys[2]][i])
