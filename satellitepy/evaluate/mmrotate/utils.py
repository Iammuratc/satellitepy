import os
import numpy as np
import torch
import logging

from satellitepy.data.labels import read_label
from satellitepy.data.cutout.geometry import BBox

from mmrotate.core.bbox import rbbox_overlaps
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
            det_labels['bboxes'].append(my_bbox.corners.tolist())
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

    ious = rbbox_overlaps(torch.FloatTensor(det_label_params), torch.FloatTensor(gt_label_params))
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


def set_conf_mat_from_result(
    conf_mat,
    result,
    instance_names,
    conf_score_thresholds,
    iou_thresholds):

    for i_iou_th, iou_th in enumerate(iou_thresholds):
        for i_conf_score_th, conf_score_th in enumerate(conf_score_thresholds):
            # (Surely) Detected gt label indices
            ## These indices have greater values than both iou and confidence score thresholds
            det_gt_bbox_indices = []

            # Iterate over the confidence scores of the detected bounding boxes
            for i_conf_score, conf_score in enumerate(result['det_labels']['confidence_scores']):
                ## If the confidence score is lower than threshold, skip the object
                if conf_score<conf_score_th:
                    continue
                ## Check if iou score is greater than iou_threshold
                iou_score = result['matches']['iou']['scores'][i_conf_score]
                if iou_score < iou_th:
                    continue

                gt_index = result['matches']['iou']['indexes'][i_conf_score]
                det_gt_bbox_indices.append(gt_index)
                det_gt_instance_name = result['gt_labels']['classes']['1'][gt_index]
                ## Set instance name to Background if it is not defined by the user
                det_gt_instance_name = 'Background' if det_gt_instance_name not in instance_names else det_gt_instance_name 
                det_gt_index = instance_names.index(det_gt_instance_name)
                ## Det index
                det_index = instance_names.index(result['det_labels']['instance_names'][i_conf_score])
                conf_mat[i_iou_th,i_conf_score_th,det_gt_index,det_index] += 1

            # If a ground truth label is undetected, add it as a detected Background label
            undet_gt_bbox_indices = set(range(len(result['gt_labels']['classes']['1']))) - set(det_gt_bbox_indices) 
            for undet_gt_bbox_ind in undet_gt_bbox_indices:
                undet_gt_instance_name = result['gt_labels']['classes']['1'][undet_gt_bbox_ind]
                ## Set instance name to Background if it is not defined by the user
                undet_gt_instance_name = 'Background' if undet_gt_instance_name not in instance_names else undet_gt_instance_name
                undet_gt_index = instance_names.index(undet_gt_instance_name)

                conf_mat[i_iou_th, i_conf_score_th, undet_gt_index, instance_names.index('Background')] += 1

    return conf_mat

def get_precision_recall(conf_mat,sort_values=True,complete_curve=True):
    """
    Calculate precision,recall and average precision from confusion matrix
    Parameters
    ----------
    conf_mat : np.ndarray
        Confusion matrix with shape=len(iou_thresholds),len(conf_score_thresholds),len(instance_names),len(instance_names). 
        Rows are ground truth, columns are predictions.
    sorted : bool
        If True, precision and recall values will be modified such that they are in the ascending/descending order.
        This prevents the up-down ziczacs in PR plots
    complete_curve : bool
        If True, precision and recall values will be bound to x- and y-axis on the PR curves
    Returns
    -------
    precision : np.ndarray
        Precision values at every IOU and confidence thresholds
    recall : np.ndarray
        Recall values at every IOU and confidence thresholds
    """

    len_iou_thresholds = conf_mat.shape[0]
    len_conf_score_thresholds = conf_mat.shape[1]
    len_instance_names = conf_mat.shape[2]

    precision = np.zeros(shape=(len_iou_thresholds,len_conf_score_thresholds,len_instance_names))
    recall = np.zeros(shape=(len_iou_thresholds,len_conf_score_thresholds,len_instance_names))

    for i_iou in range(len_iou_thresholds):
        for i_conf_score_th in range(len_conf_score_thresholds):
            for i in range(len_instance_names): ## Row is GT
                tp = 0
                fp = 0
                fn = 0
                for j in range(len_instance_names):
                    if i == j:
                        tp = conf_mat[i_iou,i_conf_score_th,i,j]
                    else:
                        fn += conf_mat[i_iou,i_conf_score_th,i,j]
                        fp += conf_mat[i_iou,i_conf_score_th,j,i]
                precision[i_iou,i_conf_score_th,i] = tp/(tp+fp)
                recall[i_iou,i_conf_score_th,i] = tp/(tp+fn)

    if sort_values:
        # The precision recall values wave. We do not want that. For example, recall should be in the descending order.
        # Let's say the recall list is [0.6, 0.6, 0.58, 0.7, 0.7, 0.6, 0.5, 0.4]
        # The loop below should return this: [0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.4]
        for i_iou in range(len_iou_thresholds):
            for i_conf_score_th in range(1,len_conf_score_thresholds):
                for i in range(len_instance_names): ## Row is GT
                    # Sort precision
                    ## If the current recall value is higher than the previous value
                    ## Set the current recall value to the previous value
                    p_0 = precision[i_iou,i_conf_score_th-1,i]
                    p_1 = precision[i_iou,i_conf_score_th,i]
                    if p_1 < p_0:
                        precision[i_iou,i_conf_score_th,i] = p_0
                    # Sort recall
                    r_0 = recall[i_iou,len_conf_score_thresholds-i_conf_score_th,i]
                    r_1 = recall[i_iou,len_conf_score_thresholds-i_conf_score_th-1,i]
                    ## If the previous recall value is higher than the current value
                    ## Set the current value to the previous value
                    if r_0 > r_1:
                        recall[i_iou,len_conf_score_thresholds-i_conf_score_th-1,i] = r_0
    if complete_curve:
        precision = np.pad(precision,((0,0),(1,0),(0,0)))
        precision = np.pad(precision,((0,0),(0,1),(0,0)),'edge')
        recall = np.pad(recall,((0,0),(0,1),(0,0)))
        recall = np.pad(recall,((0,0),(1,0),(0,0)),'edge')
    return precision, recall

def get_average_precision(precision,recall):
    '''
    Calculate average precision from precision and recall values.
    PAY ATTENTION that the values are sorted in those lists.
    Parameters
    ----------
    precision : np.ndarray
        Ascending precision values at every IOU and confidence thresholds
    recall : np.ndarray
        Descending recall values at every IOU and confidence thresholds
    Returns
    -------
    ap : np.ndarray
        Average precision
    '''

    len_iou_thresholds = precision.shape[0]
    len_conf_score_thresholds = precision.shape[1]
    len_instance_names = precision.shape[2]

    ap = np.zeros(shape=(precision.shape[0],precision.shape[2]))

    for i_iou in range(len_iou_thresholds):
        for i_conf_score_th in range(1,len_conf_score_thresholds):
            for i in range(len_instance_names): ## Row is GT
                ap_i = precision[i_iou,i_conf_score_th,i] * (recall[i_iou,i_conf_score_th-1,i] - recall[i_iou,i_conf_score_th,i])
                ap[i_iou,i] += ap_i
    return ap