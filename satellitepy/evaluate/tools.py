import cv2
import os

import numpy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from satellitepy.data.utils import get_satellitepy_dict_values
from satellitepy.utils.path_utils import create_folder, get_file_paths, is_file_names_match
# from satellitepy.data.patch import get_patches, merge_patch_results
import logging
import json
from satellitepy.evaluate.utils import set_conf_mat_from_result, get_precision_recall, get_average_precision
from tqdm import tqdm


def calculate_map(
    in_result_folder,
    task,
    instance_names,
    conf_score_thresholds,
    iou_thresholds,
    out_folder,
    plot_pr):

    # Get logger
    logger = logging.getLogger(__name__)

    # Add background to instance_names
    instance_names = instance_names + ['Background']

    # Init confusion matrix
    conf_mat = np.zeros(shape=(len(iou_thresholds),len(conf_score_thresholds),len(instance_names),len(instance_names)))

    # Result paths
    result_paths = get_file_paths(in_result_folder)

    for result_path in tqdm(result_paths):
        # logger.info(f'The following result file will be evaluated: {result_path}')
        # Result json file
        if result_path.suffix != ".json":
            continue
        with open(result_path,'r') as result_file:
            result = json.load(result_file) # dict of 'gt_labels', 'det_labels', 'matches' 
        
        conf_mat = set_conf_mat_from_result(
            conf_mat,
            task,
            result,
            instance_names,
            conf_score_thresholds,
            iou_thresholds)

    pr_threshold_ind = 0
    precision, recall = get_precision_recall(conf_mat,sort_values=True)
    with numpy.printoptions(threshold=numpy.inf):
        logger.info('Confusion matrix:')
        logger.info(conf_mat)
        logger.info(50*'-')
        logger.info(f'Precision at all confidence score thresholds and iuo threshold = {iou_thresholds[pr_threshold_ind]}')
        logger.info(precision[pr_threshold_ind,:])
        logger.info(50 * '-')
        logger.info(f'Recall at all confidence score thresholds and iou threshold = {iou_thresholds[pr_threshold_ind]} ')
        logger.info(recall[pr_threshold_ind,:])
        logger.info(50 * '-')
        logger.info('AP')
        ap = get_average_precision(precision,recall)
        logger.info(ap)
        logger.info('mAP')
        mAP = np.sum(np.transpose(np.transpose(ap)[:-1]), axis=1)/(len(ap[0])-1)
        logger.info(mAP)
    if plot_pr:
        fig, ax = plt.subplots()
        ax.plot(recall[pr_threshold_ind,:],precision[pr_threshold_ind,:])
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        plt.savefig(str(out_folder) + '/plot_AP.png')
        plt.show()


def calc_iou(gt_mask, det_mask):
    gt_set = set(zip(gt_mask[0], gt_mask[1]))
    det_set = set(zip(det_mask[0], det_mask[1]))

    intersection = gt_set.intersection(det_set)
    union = gt_set.union(det_set)

    int_cnt = len(intersection)
    un_cnt = len(union)

    score = int_cnt/un_cnt if un_cnt != 0 else 0
    return score



def calculate_iou_score(in_result_folder, out_folder, iou_thresholds, conf_score_threshold):
    # Get logger
    logger = logging.getLogger(__name__)

    # Result paths
    result_paths = get_file_paths(in_result_folder)

    ious = np.zeros(len(iou_thresholds))
    cnt = np.zeros(len(iou_thresholds))

    for result_path in tqdm(result_paths):
        if result_path.suffix != ".json":
            continue

        with open(result_path,'r') as result_file:
            result = json.load(result_file) # dict of 'gt_labels', 'det_labels', 'matches'
            gt_results = get_satellitepy_dict_values(result['gt_labels'], "masks")

            for i_iou_th, iou_th in enumerate(iou_thresholds):
                # Iterate over the confidence scores of the detected bounding boxes

                for i_conf_score, conf_score in enumerate(result['confidence-scores']):
                    ## If the confidence score is lower than threshold, skip the object
                    if conf_score < conf_score_threshold:
                        continue
                    ## Check if iou score is greater than iou_threshold
                    iou_score = result['matches']['iou']['scores'][i_conf_score]
                    if iou_score < iou_th:
                        continue

                    gt_index = result['matches']['iou']['indexes'][i_conf_score]
                    det_gt_value = gt_results[gt_index]
                    if det_gt_value is None:
                        continue
                    ## Det index
                    det_value = result["masks"][i_conf_score]
                    iou = calc_iou(det_gt_value, det_value)

                    cnt[i_iou_th] += 1
                    ious[i_iou_th] += iou

    ious = ious / cnt
    logger.info(ious)


def calculate_relative_score(in_result_folder, task, conf_score_threshold, iou_thresholds, out_folder):
    # Get logger
    logger = logging.getLogger(__name__)

    # Result paths
    result_paths = get_file_paths(in_result_folder)

    score = np.zeros(len(iou_thresholds))
    cnt = np.zeros(len(iou_thresholds))

    for result_path in tqdm(result_paths):
        # logger.info(f'The following result file will be evaluated: {result_path}')
        # Result json file
        if result_path.suffix != ".json":
            continue
        with open(result_path,'r') as result_file:
            result = json.load(result_file) # dict of 'gt_labels', 'det_labels', 'matches'
            gt_results = get_satellitepy_dict_values(result['gt_labels'], task)
            for i_iou_th, iou_th in enumerate(iou_thresholds):
                # Iterate over the confidence scores of the detected bounding boxes

                for i_conf_score, conf_score in enumerate(result['confidence-scores']):
                    ## If the confidence score is lower than threshold, skip the object
                    if conf_score < conf_score_threshold:
                        continue
                    ## Check if iou score is greater than iou_threshold
                    iou_score = result['matches']['iou']['scores'][i_conf_score]
                    if iou_score < iou_th:
                        continue

                    gt_index = result['matches']['iou']['indexes'][i_conf_score]
                    det_gt_value = gt_results[gt_index]
                    if det_gt_value is None:
                        continue
                    ## Det index
                    det_value = result[task][i_conf_score][0]

                    error = abs(det_gt_value - det_value)/det_gt_value
                    cnt[i_iou_th] += 1
                    score[i_iou_th] += (1 - error)
    score = score / cnt
    logger.info(score)