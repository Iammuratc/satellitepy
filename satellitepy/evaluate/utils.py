from satellitepy.data.bbox import BBox
from shapely.geometry import Polygon
import numpy as np
# from torch import Tensor
# from typing import Optional, Tuple
import torch

from satellitepy.data.utils import get_task_dict, get_satellitepy_dict_values


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

    ## New IOU calculation
    if "obboxes" in det_labels:
        ious = get_ious(det_labels['obboxes'], gt_labels['obboxes'])
    else:
        ious = get_ious(det_labels['hbboxes'], gt_labels['hbboxes'])
    for i,iou in enumerate(ious):
        # ROW: detected obboxes
        # COL: gt obboxes
        try:
            bbox_ind_gt = np.argmax(iou)
            iou_score = iou[bbox_ind_gt]
        except:
            continue

        matches['iou']['scores'].append(iou_score.item())
        matches['iou']['indexes'].append(bbox_ind_gt.item())
    return matches


def set_conf_mat_from_result(
    conf_mat,
    task,
    result,
    instance_names,
    conf_score_thresholds,
    iou_thresholds,
    ignore_other_instances):

    ignored_instances_ret = set
    ignored_cnt = 0

    task_dict = get_task_dict(task)
    idx2name = {v: k for k, v in task_dict.items()}
    taskResult = get_satellitepy_dict_values(result['gt_labels'], task)
    if len(taskResult) == 0:
        return conf_mat
    for i_iou_th, iou_th in enumerate(iou_thresholds):
        for i_conf_score_th, conf_score_th in enumerate(conf_score_thresholds):
            # (Surely) Detected gt label indices
            ## These indices have greater values than both iou and confidence score thresholds
            det_gt_bbox_indices = []

            # Iterate over the confidence scores of the detected bounding boxes
            for i_conf_score, conf_score in enumerate(result['det_labels']['confidence-scores']):
                ## If the confidence score is lower than threshold, skip the object
                if conf_score<conf_score_th:
                    continue
                ## Check if iou score is greater than iou_threshold
                iou_score = result['matches']['iou']['scores'][i_conf_score]
                if iou_score < iou_th:
                    continue

                gt_index = result['matches']['iou']['indexes'][i_conf_score]

                # det_gt_instance_name = result['gt_labels'][task][gt_index]
                det_gt_instance_name = taskResult[gt_index]

                if det_gt_instance_name is None:
                    continue

                det_name = str(idx2name[result['det_labels'][task][i_conf_score]])

                if ignore_other_instances and det_name not in instance_names:
                    ignored_cnt+=1
                    ignored_instances_ret.add(det_name)
                    continue

                det_gt_bbox_indices.append(gt_index)

                ## Set instance name to Background if it is not defined by the user
                det_gt_instance_name = 'Background' if str(det_gt_instance_name) not in instance_names else det_gt_instance_name
                det_gt_index = instance_names.index(str(det_gt_instance_name))
                ## Det index
                det_index = instance_names.index(det_name)
                conf_mat[i_iou_th,i_conf_score_th,det_gt_index,det_index] += 1

            # If a ground truth label is undetected, add it as a detected Background label

            # undet_gt_bbox_indices = set(range(len(result['gt_labels'][task]))) - set(det_gt_bbox_indices)
            undet_gt_bbox_indices = set(range(len(taskResult))) - set(det_gt_bbox_indices)

            for undet_gt_bbox_ind in undet_gt_bbox_indices:

                # undet_gt_instance_name = result['gt_labels'][task][undet_gt_bbox_ind]
                undet_gt_instance_name = taskResult[undet_gt_bbox_ind]

                ## Set instance name to Background if it is not defined by the user
                undet_gt_instance_name = 'Background' if str(undet_gt_instance_name) not in instance_names else undet_gt_instance_name
                undet_gt_index = instance_names.index(str(undet_gt_instance_name))

                conf_mat[i_iou_th, i_conf_score_th, undet_gt_index, instance_names.index('Background')] += 1

    return conf_mat, ignored_instances_ret, ignored_cnt

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

    precision = np.nan_to_num(precision, True, nan=1.0)
    recall = np.nan_to_num(recall, True, nan = 0.0)

    ap = np.zeros(shape=(precision.shape[0],precision.shape[2]))

    for i_iou in range(len_iou_thresholds):
        for i_conf_score_th in range(1,len_conf_score_thresholds):
            for i in range(len_instance_names): ## Row is GT
                ap_i = precision[i_iou,i_conf_score_th,i] * (recall[i_iou,i_conf_score_th-1,i] - recall[i_iou,i_conf_score_th,i])
                ap[i_iou,i] += ap_i
    return ap

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
