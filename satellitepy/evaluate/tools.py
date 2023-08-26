import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
    logger.info(f'Precision at all confidence score thresholds and iuo threshold = {iou_thresholds[pr_threshold_ind]}')
    logger.info(precision[pr_threshold_ind,:])
    logger.info(f'Recall at all confidence score thresholds and iou threshold = {iou_thresholds[pr_threshold_ind]} ')
    logger.info(recall[pr_threshold_ind,:])
    logger.info('AP')
    ap = get_average_precision(precision,recall)
    logger.info(ap)
    if plot_pr:
        fig, ax = plt.subplots()
        ax.plot(recall[pr_threshold_ind,:],precision[pr_threshold_ind,:])
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        plt.show()
