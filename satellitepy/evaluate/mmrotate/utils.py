import os
import numpy as np
import torch
import logging

from satellitepy.data.labels import read_label
from satellitepy.data.cutout.geometry import BBox

from shapely.geometry import Polygon

# from mmrotate.core.bbox import rbbox_overlaps
from mmdet.apis.inference import init_detector, inference_detector
from mmcv.ops import nms_rotated


def get_result(
    img,
    gt_labels,
    mmrotate_model,
    class_names,
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
        nms_on_multiclass_thr : float
            IUO threshold for filtering out multiple detection results oj one object.
    Returns
    -------
        result : dict
            Detected bounding box with their corresponding ground truth and IOU values 
    """
    logger = logging.getLogger(__name__)

    # Inference mmrotate model
    mmrotate_result = inference_detector(mmrotate_model,img)

    # Results
    det_labels = get_det_labels(mmrotate_result,class_names,nms_on_multiclass_thr)
    # IOU matching between ground truth and detected bboxes 
    matches = match_gt_and_det_bboxes(gt_labels,det_labels)

    result = {
        'gt_labels':gt_labels,
        'det_labels':det_labels,
        'matches':matches
                }

    return result 

def get_mmrotate_model(config_path,model_path,device):
    # build the model from a config file and a checkpoint file
    model = init_detector(config_path, model_path, device=device)
    return model

def get_gt_labels(label_path,label_format):
    """
    Convert ground truth labels to dict in satellitepy format
    """
    gt_labels = read_label(label_path,label_format)
    return gt_labels

def get_det_labels(mmrotate_result,class_names,nms_on_multiclass_thr):
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
    if nms_on_multiclass_thr!=0:
        mmrotate_result = nms_on_multi_class(mmrotate_result,nms_iou_thr=nms_on_multiclass_thr)
    det_labels = {
        'bboxes':[],
        'instance_names':[],
        'confidence_scores':[]}
    for class_bboxes_ind, class_bboxes in enumerate(mmrotate_result):
        if class_bboxes==[]:
            continue
        if not isinstance(class_bboxes,np.ndarray):
            class_bboxes = np.array(class_bboxes)

        for class_bbox in class_bboxes:
            my_bbox = BBox(params=class_bbox[:5])
            det_labels['instance_names'].append(class_names[class_bboxes_ind])
            det_labels['bboxes'].append(my_bbox.corners)
            det_labels['confidence_scores'].append(class_bbox[-1])
    return det_labels

def match_gt_and_det_bboxes(gt_labels,det_labels):
    """
    Match ground truth and detected bboxes
    Get the matching indexes and store iou scores
    Parameters
    ----------
    gt_labels : dict
        ground truth labels in satellitepy format
    det_labels : dict
        detected bboxes labels in satellitepy format
    Returns
    -------
    matches : dict
        'iou' : intersection over union
            'scores' :  Highest value of matching bboxes of gt and det
            'indexes' : Matching index of det label. The order of the labels in 'indexes' is the same as the det_labels['bboxes'] order, and the value is the index in gt_labels['bboxes'].   
    """
    matches = {'iou':{'scores':[],'indexes':[]}}

    det_label_params = [BBox(corners=my_bbox).params for my_bbox in det_labels['bboxes']] if len(det_labels['bboxes'])!=0 else []
    gt_label_params = [BBox(corners=my_bbox).params for my_bbox in gt_labels['bboxes']] if len(gt_labels['bboxes'])!=0 else []

    ## Old IOU calculation (keep it for now 12.05)
    # ious = rbbox_overlaps(torch.FloatTensor(det_label_params), torch.FloatTensor(gt_label_params))
    ## New IOU calculation
    ious = get_ious(det_labels['bboxes'], gt_labels['bboxes'])
    for i,iou in enumerate(ious):
        # ROW: detected bboxes
        # COL: gt bboxes
        try:
            bbox_ind_gt = np.argmax(iou)
            iou_score = iou[bbox_ind_gt]
        except:
            continue

        matches['iou']['scores'].append(iou_score.item())
        matches['iou']['indexes'].append(bbox_ind_gt.item())
    return matches

def nms_on_multi_class(result,nms_iou_thr):
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
    result_squeezed = np.zeros(shape=(result_len,6))
    labels_squeezed = np.zeros(shape=(result_len))
    i=0
    for det_label, det_bboxes in enumerate(result):
        for det_bbox in det_bboxes:
            if len(det_bbox) == 0:
                continue
            result_squeezed[i,:] = det_bbox#[:5]
            labels_squeezed[i] = det_label
            i+=1

    result_squeezed_nms, keep_ind = nms_rotated(
        dets=torch.from_numpy(result_squeezed[:,:5]),
        scores=torch.from_numpy(result_squeezed[:,-1]),
        iou_threshold= nms_iou_thr,
        labels=torch.from_numpy(labels_squeezed))

    if keep_ind is None:
        return result
    ### Put the remaining bboxes into their class lists 
    result_nms = [[] for _ in range(len(result))]

    for ind in keep_ind:
        ind_cls = int(labels_squeezed[ind])
        bbox = result_squeezed[ind]
        result_nms[ind_cls].append(bbox)

    result_nms_to_numpy = [] 
    for det_bboxes in result_nms:
        if det_bboxes == []:
            result_nms_to_numpy.append(np.empty(shape=(0,6)))
        else:
            result_nms_to_numpy.append(np.array(det_bboxes))

    return result_nms_to_numpy

def get_ious(bboxes_1,bboxes_2):
    '''
    This functions returns the IOUs for two bbox sets, e.g., ground truth and detected bboxes
    Parameters
    ----------
    bboxes_1 : list
        List of bounding box corners
    bboxes_2 : list
        List of bounding box corners
    Returns
    -------
    ious : np.ndarray
        IOU matrix with the shape [len(bboxes_1),len(bboxes_2)]
    '''
    polygons_1 = [Polygon(bbox) for bbox in bboxes_1]
    polygons_2 = [Polygon(bbox) for bbox in bboxes_2]

    ious = np.zeros(shape=(len(bboxes_1),len(bboxes_2)))

    for i, p1 in enumerate(polygons_1):
        for j, p2 in enumerate(polygons_2):
            intersection_area = p1.intersection(p2).area 
            iou = intersection_area / (p1.area + p2.area - intersection_area)
            ious[i,j] = iou

    return ious
