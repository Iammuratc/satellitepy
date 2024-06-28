import numpy as np
import torch
import logging

from satellitepy.data.bbox import BBox
from satellitepy.data.utils import get_satellitepy_table
from satellitepy.evaluate.utils import get_ious

from mmdet.apis.inference import init_detector, inference_detector
from mmcv.ops import nms_rotated


def get_result(
        img,
        gt_labels,
        mmrotate_model,
        class_names,
        task_name,
        nms_on_multiclass_thr):
    """
    Get the detected bounding box results from mmrotate model
    Parameters
    ----------
        img_path : Path
            Image path.
        label_path : Path
            Ground truth label path.
        mmrotate_model : mmrotate.models.detectors.<model_name>
            MMRotate model
        label_format : str
            Label format. E.g., dota, fair1m
        class_names : list
            Class names to be used to classify the detected bounding box.
        task_name:
            Name of the trained task
        nms_on_multiclass_thr : float
            IUO threshold for filtering out multiple detection results oj one object.
    Returns
    -------
        result : dict
            Detected bounding box with their corresponding ground truth and IOU values 
    """
    logger = logging.getLogger('')

    mmrotate_result = inference_detector(mmrotate_model, img)

    det_labels = get_det_labels(mmrotate_result, class_names, task_name, nms_on_multiclass_thr)
    matches = mmrotate_match_gt_and_det_bboxes(gt_labels, det_labels)

    result = {
        'gt_labels': gt_labels,
        'matches': matches,
        'det_labels': det_labels
    }
    
    return result


def get_mmrotate_model(config_path, model_path, device):
    model = init_detector(config_path, model_path, device=device)
    return model


def get_det_labels(mmrotate_result, class_names, task_name, nms_on_multiclass_thr):
    """
    Convert mmrotate results to dict in satellitepy format
    Parameters
    ----------
    mmrotate_result : list
        Detected bounding boxes from mmrotate models
    class_names : list
        Class names
    multiclass_nms_iou_thr : float
        MMRotate models detect bounding boxes for each class, this results in several bounding boxes for one object
        nms_on_multiclass filters out the lower score bounding boxes, and keep the best
    """
    det_labels = {
        'obboxes': [],
        task_name: [],
        'confidence-scores': []}
    for class_bboxes_ind, class_bboxes in enumerate(mmrotate_result):
        if not class_bboxes.any():
            continue

        for class_bbox in class_bboxes:
            my_bbox = BBox(params=np.array(class_bbox)[:5])
            if class_names[class_bboxes_ind] == 'other':
                continue
            det_labels[task_name].append(get_satellitepy_table()[task_name][class_names[class_bboxes_ind]])
            det_labels['obboxes'].append(my_bbox.corners)
            det_labels['confidence-scores'].append(float(class_bbox[-1]))
    return det_labels


def nms_on_multi_class(result, nms_iou_thr):
    """
    Merge all the bboxes of each class and apply nms
    MMRotate returns overlapping bounding boxes for different classes for some reason, so this function is recommended for multiclass mmrotate models.

    Parameters
    ---------
    result : ndarray
        Detection results, has shape
        (num_classes, num_bboxes, 6).
    nms_iou_thr : float
        nms IoU threshold, the detection results
        have done nms in the detector, only applied when users want to
        change the nms IoU threshold.
    """
    result_len = sum([len(det_bboxes) for det_bboxes in result])
    result_squeezed = np.zeros(shape=(result_len, 6))
    labels_squeezed = np.zeros(shape=result_len)
    i = 0
    for det_label, det_bboxes in enumerate(result):
        for det_bbox in det_bboxes:
            if len(det_bbox) == 0:
                continue
            result_squeezed[i, :] = det_bbox
            labels_squeezed[i] = det_label
            i += 1

    result_squeezed_nms, keep_ind = nms_rotated(
        dets=torch.from_numpy(result_squeezed[:, :5]),
        scores=torch.from_numpy(result_squeezed[:, -1]),
        iou_threshold=nms_iou_thr,
        labels=torch.from_numpy(labels_squeezed))

    if keep_ind is None:
        return result

    result_nms = [[] for _ in range(len(result))]

    for ind in keep_ind:
        ind_cls = int(labels_squeezed[ind])
        bbox = result_squeezed[ind]
        result_nms[ind_cls].append(bbox)

    result_nms_to_numpy = []
    for det_bboxes in result_nms:
        if not det_bboxes:
            result_nms_to_numpy.append(np.empty(shape=(0, 6)))
        else:
            result_nms_to_numpy.append(np.array(det_bboxes))

    return result_nms_to_numpy


def mmrotate_match_gt_and_det_bboxes(gt_labels, det_labels):
    matches = {'iou': {'scores': [], 'indexes': []}}
    ious = get_ious(det_labels['obboxes'], gt_labels['obboxes'])

    for i, iou in enumerate(ious):
        try:
            bbox_ind_gt = np.argmax(iou)
            iou_score = iou[bbox_ind_gt]
        except:
            continue

        matches['iou']['scores'].append(iou_score.item())
        matches['iou']['indexes'].append(bbox_ind_gt.item())
    return matches
