import json
import logging

from shapely.geometry import Polygon
import numpy as np
from satellitepy.evaluate.bbavector.utils import apply_nms
from satellitepy.utils.path_utils import get_file_paths
from tqdm import tqdm

from satellitepy.data.utils import get_task_dict, get_satellitepy_dict_values


def match_gt_and_det_bboxes(gt_labels, det_labels):
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
            'indexes' : Matching index of det label. The order of the labels in 'indexes' is the same as
                        the det_labels['bboxes'] order, and the value is the index in gt_labels['bboxes'].
    """
    matches = {'iou': {'scores': [], 'indexes': []}}

    if gt_labels is None:
        return matches

    if 'obboxes' in det_labels and any(gt_labels['obboxes']):
        ious = get_ious(det_labels['obboxes'], gt_labels['obboxes'])
    else:
        ious = get_ious(det_labels['hbboxes'], gt_labels['hbboxes'])
    for i, iou in enumerate(ious):
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
        nms_iou_thresh,
        ignore_other_instances,
        no_probability):
    ignored_instances_ret = []
    ignored_cnt = 0

    task_dict = get_task_dict(task)
    idx2name = {v: k for k, v in task_dict.items()}
    task_result = get_satellitepy_dict_values(result['gt_labels'], task)

    result = remove_low_conf_results(result, task, conf_score_thresholds[0], no_probability)
    det_results = apply_nms(result['det_labels'], nms_iou_threshold=nms_iou_thresh, target_task=task, no_probability=no_probability)

    if no_probability:
        det_inds = det_results[task]
        conf_scores = det_results['confidence-scores']
    else:
        det_inds = np.argmax(det_results[task], axis=1) if len(det_results[task]) > 0 else []
        conf_scores = np.max(det_results[task], axis=1) if len(det_inds) > 0 else []

    matches = match_gt_and_det_bboxes(result['gt_labels'], det_results)

    if len(task_result) == 0:
        return conf_mat, ignored_instances_ret, ignored_cnt
    for i_iou_th, iou_th in enumerate(iou_thresholds):
        for i_conf_score_th, conf_score_th in enumerate(conf_score_thresholds):
            det_gt_bbox_indices = []

            for i_conf_score, conf_score in enumerate(conf_scores):
                if conf_score < conf_score_th:
                    continue

                iou_score = matches['iou']['scores'][i_conf_score]
                if iou_score < iou_th:
                    continue

                gt_index = matches['iou']['indexes'][i_conf_score]

                det_gt_instance_name = task_result[gt_index]

                if det_gt_instance_name is None:
                    continue

                det_name = str(idx2name[det_inds[i_conf_score]])

                if ignore_other_instances and det_name not in instance_names:
                    ignored_cnt += 1
                    ignored_instances_ret.append(det_name)
                    continue

                det_gt_bbox_indices.append(gt_index)

                det_gt_instance_name = 'Background' if str(
                    det_gt_instance_name) not in instance_names else det_gt_instance_name
                det_gt_index = instance_names.index(str(det_gt_instance_name))

                det_index = instance_names.index(det_name)
                conf_mat[i_iou_th, i_conf_score_th, det_gt_index, det_index] += 1

            undet_gt_bbox_indices = set(range(len(task_result))) - set(det_gt_bbox_indices)

            for undet_gt_bbox_ind in undet_gt_bbox_indices:
                undet_gt_instance_name = task_result[undet_gt_bbox_ind]

                undet_gt_instance_name = 'Background' if str(
                    undet_gt_instance_name) not in instance_names else undet_gt_instance_name
                undet_gt_index = instance_names.index(str(undet_gt_instance_name))

                conf_mat[i_iou_th, i_conf_score_th, undet_gt_index, instance_names.index('Background')] += 1

    return conf_mat, ignored_instances_ret, ignored_cnt


def get_precision_recall(conf_mat, sort_values=True, complete_curve=True):
    """
    Calculate precision,recall and average precision from confusion matrix
    Parameters
    ----------
    conf_mat : np.ndarray
        Confusion matrix with shape=len(iou_thresholds),len(conf_score_thresholds),len(instance_names),len(instance_names). 
        Rows are ground truth, columns are predictions.
    sort_values : bool
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

    precision = np.zeros(shape=(len_iou_thresholds, len_conf_score_thresholds, len_instance_names))
    recall = np.zeros(shape=(len_iou_thresholds, len_conf_score_thresholds, len_instance_names))

    for i_iou in range(len_iou_thresholds):
        for i_conf_score_th in range(len_conf_score_thresholds):
            for i in range(len_instance_names):
                tp = 0
                fp = 0
                fn = 0
                for j in range(len_instance_names):
                    if i == j:
                        tp = conf_mat[i_iou, i_conf_score_th, i, j]
                    else:
                        fn += conf_mat[i_iou, i_conf_score_th, i, j]
                        fp += conf_mat[i_iou, i_conf_score_th, j, i]
                precision[i_iou, i_conf_score_th, i] = tp / (tp + fp)
                recall[i_iou, i_conf_score_th, i] = tp / (tp + fn)

    if sort_values:
        for i_iou in range(len_iou_thresholds):
            for i_conf_score_th in range(1, len_conf_score_thresholds):
                for i in range(len_instance_names):
                    p_0 = precision[i_iou, i_conf_score_th - 1, i]
                    p_1 = precision[i_iou, i_conf_score_th, i]
                    if p_1 < p_0:
                        precision[i_iou, i_conf_score_th, i] = p_0

                    r_0 = recall[i_iou, len_conf_score_thresholds - i_conf_score_th, i]
                    r_1 = recall[i_iou, len_conf_score_thresholds - i_conf_score_th - 1, i]

                    if r_0 > r_1:
                        recall[i_iou, len_conf_score_thresholds - i_conf_score_th - 1, i] = r_0

    if complete_curve:
        precision = np.pad(precision, ((0, 0), (1, 0), (0, 0)))
        precision = np.pad(precision, ((0, 0), (0, 1), (0, 0)), 'edge')
        recall = np.pad(recall, ((0, 0), (0, 1), (0, 0)))
        recall = np.pad(recall, ((0, 0), (1, 0), (0, 0)), 'edge')
    return precision, recall


def get_average_precision(precision, recall):
    """
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
    """

    len_iou_thresholds = precision.shape[0]
    len_conf_score_thresholds = precision.shape[1]
    len_instance_names = precision.shape[2]

    precision = np.nan_to_num(precision, True, nan=1.0)
    recall = np.nan_to_num(recall, True, nan=0.0)

    ap = np.zeros(shape=(precision.shape[0], precision.shape[2]))

    for i_iou in range(len_iou_thresholds):
        for i_conf_score_th in range(1, len_conf_score_thresholds):
            for i in range(len_instance_names):
                ap_i = precision[i_iou, i_conf_score_th, i] * (
                            recall[i_iou, i_conf_score_th - 1, i] - recall[i_iou, i_conf_score_th, i])
                ap[i_iou, i] += ap_i
    return ap


def get_ious(bboxes_1, bboxes_2):
    """
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
    """
    polygons_1 = [Polygon(bbox) for bbox in bboxes_1]
    polygons_2 = [Polygon(bbox) for bbox in bboxes_2]

    ious = np.zeros(shape=(len(bboxes_1), len(bboxes_2)))

    for i, p1 in enumerate(polygons_1):
        for j, p2 in enumerate(polygons_2):
            intersection_area = p1.intersection(p2).area
            iou = intersection_area / (p1.area + p2.area - intersection_area)
            ious[i, j] = iou

    return ious


def remove_low_conf_results(results, task, conf_score, no_probability):
    if conf_score == 0:
        return results

    if no_probability:
        conf_scores = np.array(results['det_labels'][task])
    else:
        conf_scores = np.max(results['det_labels'][task], axis=1) if len(results['det_labels'][task]) > 0 else []
    idx = np.argwhere(conf_scores > conf_score).flatten() if len(results['det_labels'][task]) > 0 else []

    filtered_results = {
        'det_labels': {},
        'matches': {
            'iou': {
                'scores': [],
                'indexes': []
            }
        }
    }

    for key in results['det_labels'].keys():
        filtered_results['det_labels'][key] = np.array(results['det_labels'][key])[idx]

    if len(results['gt_labels'][task]) != 0:
        filtered_results['matches']['iou']['scores'] = np.array(results['matches']['iou']['scores'])[idx]
        filtered_results['matches']['iou']['indexes'] = np.array(results['matches']['iou']['indexes'])[idx]

    return filtered_results


def get_instance_names(in_label_path, task):
    logger = logging.getLogger('')
    logger.info('Infering instances based on ground truth labels.')

    label_paths = get_file_paths(in_label_path)

    indices = get_task_dict(task)
    idx2name = {v: k for k, v in indices.items()}

    instances = set()

    for label_path in tqdm(label_paths):
        with open(label_path, 'r') as file:
            labels = json.load(file)
            for instance in get_satellitepy_dict_values(labels['gt_labels'], task):
                if instance is None:
                    continue
                idx = indices[instance]
                instances.add(idx2name[idx])

    logger.info(f'Found instances: {list(instances)}')
    return list(instances)