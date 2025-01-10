import cv2
import numpy as np
import matplotlib.pyplot as plt

from satellitepy.data.utils import get_satellitepy_dict_values, remove_repetitive_values
from satellitepy.models.bbavector.utils import decode_masks
from satellitepy.utils.path_utils import get_file_paths
import logging
import json
from satellitepy.evaluate.utils import set_conf_mat_from_result, get_precision_recall, get_average_precision, \
    remove_low_conf_results, match_gt_and_det_bboxes, filter_out_fp_airplanes
from tqdm import tqdm
from satellitepy.evaluate.bbavector.utils import apply_nms

logger = logging.getLogger('')

def calculate_map(
        in_result_folder,
        task,
        instance_dict,
        conf_score_thresholds,
        iou_thresholds,
        out_folder,
        plot_pr,
        nms_iou_thresh,
        ignore_other_instances=False,
        no_probability=False,
        by_source=False,
        store_undetected_objects=False,
        norm_conf_scores=False):
    '''
    instance_dict : dict
        Dictionary of class names with indices.
    '''


    instance_dict['Background'] = len(set(instance_dict.values())) # Set background an index
    background_index = instance_dict['Background']
    logger.info(f'Background added to the index {background_index}')

    instance_names = list(instance_dict.keys())
    instance_indices = set(instance_dict.values()) 
    

    conf_mat = np.zeros(
        shape=(len(iou_thresholds), len(conf_score_thresholds), len(instance_indices), len(instance_indices)))

    result_paths = get_file_paths(in_result_folder)

    ignored_instances = []
    ignored_cnt = 0

    gt_instance_names = [] # Store all instance names in ground truth to filter the AP values later

    undet_obj_indices = {}
    for i, result_path in enumerate(tqdm(result_paths)):
        if result_path.suffix != '.json':
            continue
        with open(result_path, 'r') as result_file:
            result = json.load(result_file)
        conf_mat, ignored_instances_ret, ignored_cnt_ret, undet_gt_bbox_index_dict = set_conf_mat_from_result(
            conf_mat,
            task,
            result,
            instance_dict,
            conf_score_thresholds,
            iou_thresholds,
            nms_iou_thresh,
            ignore_other_instances,
            no_probability,
            norm_conf_scores,
            by_source)
        for instance_name in result['gt_labels'][task]:
            gt_instance_names.append(instance_name)
        ignored_instances += ignored_instances_ret
        ignored_cnt += ignored_cnt_ret
        if i == 0:
            for th_values, undet_gt_bbox_indices in undet_gt_bbox_index_dict.items():
                undet_obj_indices[th_values] = ["{}_{}".format(str(result_path.stem),str(undet_gt_bbox_ind)) for undet_gt_bbox_ind in undet_gt_bbox_indices ]
        else:
            for th_values, undet_gt_bbox_indices in undet_gt_bbox_index_dict.items():
                for undet_gt_bbox_ind in undet_gt_bbox_indices:
                    undet_obj_indices[th_values].append("{}_{}".format(str(result_path.stem),str(undet_gt_bbox_ind)))
    gt_instance_names = list(set(gt_instance_names))
    pr_threshold_ind = 0
    precision, recall = get_precision_recall(conf_mat, sort_values=True)
    ap = get_average_precision(precision, recall)
    mAP = np.sum(np.transpose(np.transpose(ap)[:-1]), axis=1) / (len(ap[0]) - 1)

    if plot_pr:
        precision_recall_curve(out_folder, precision[pr_threshold_ind, :], recall[pr_threshold_ind, :])

    # Store results into a text file
    with np.printoptions(threshold=np.inf):
        logger.info('AP')
        logger.info('Instance names')
        logger.info(instance_names)
        logger.info(ap)
        logger.info('mAP')
        logger.info(mAP)
        if ignore_other_instances:
            logger.info(f'ignored {ignored_cnt} other instances. Ignored instance names: {set(ignored_instances)}')

        # Remove the instances that are not in ground truth
        instance_dict = {key:value for key,value in instance_dict.items() if key in gt_instance_names+ ['Background']}
        ## Remove repetitive indices for printing
        instance_dict = remove_repetitive_values(instance_dict)
        instance_names = list(instance_dict.keys())
        ap_50_filtered = [ap[0][ind] for ind in instance_dict.values()]
        # Print AP_50 with the sorted instance names for articles/papers by assuming that iou_thresholds[0]=0.5
        logger.info(f'AP results at {iou_thresholds[0]}')
        sorted_instance_names,ap_50 = expand_sort_lists(instance_names,ap_50_filtered)

        logger.info(sorted_instance_names)
        logger.info(ap_50)
        evaluation_file_path = out_folder / 'mAP_values.txt'
        with open(evaluation_file_path, 'w') as file:
            np.savetxt(file, ap, fmt='%.2f', delimiter=',',
                       header=f'AP (Columns: {instance_names}, Rows: IoUs {iou_thresholds}):')
            np.savetxt(file, [mAP], fmt='%.2f', delimiter='\t', header=f'mAP (Columns: {iou_thresholds}):')
            np.savetxt(file, [sorted_instance_names], fmt='%s', delimiter=',',header=f'AP{iou_thresholds[0]}):')
            np.savetxt(file, [ap_50], fmt='%.2f', delimiter=',')
            conf_mat_iou_th = 0.5
            conf_mat_conf_sc_th = 0.5

            conf_mat_iou_th_ind = iou_thresholds.index(conf_mat_iou_th)
            conf_mat_conf_sc_ind = conf_score_thresholds.index(conf_mat_conf_sc_th)

            np.savetxt(file, conf_mat[conf_mat_iou_th_ind][conf_mat_conf_sc_ind], fmt='%.2f', delimiter=',',header=f'Confusion matrix (IoU={conf_mat_iou_th}, Conf. Score={conf_mat_conf_sc_th})')
            
            if store_undetected_objects:
                np.savetxt(file,[len(undet_obj_indices[conf_mat_iou_th,conf_mat_conf_sc_th])], fmt='%.0f', delimiter=',',header=f'Number of FN at (IoU={conf_mat_iou_th}, Conf. Score={conf_mat_conf_sc_th})')
                np.savetxt(file,undet_obj_indices[conf_mat_iou_th,conf_mat_conf_sc_th],delimiter='\n',fmt='%s',header='Undetected object indices')
        logger.info(f'AP calculations are saved into: {evaluation_file_path}')


    return mAP

def expand_sort_lists(*lists):
    # Combine the lists into a list of tuples and sort by the first list
    combined = sorted(zip(*lists))
    # Unzip the sorted list of tuples back into two lists
    sorted_lists = [list(my_list) for my_list in zip(*combined)]
    # Convert them back to lists if needed
    return sorted_lists

def precision_recall_curve(out_folder, precision, recall):
    rec_values = [np.ones(recall.shape[1])]
    prec_values = [np.zeros(precision.shape[1])]
    prec_max = np.zeros(precision.shape[1])

    for i in range(len(recall)):
        rec_i = recall[i]
        rec_values.append(rec_i)
        prec_i = precision[i]
        rec_values.append(rec_i)
        prec_i_next = precision[i + 1] if i < len(recall) - 1 else precision[i]
        prec_max = np.max([prec_i, prec_i_next, prec_max], 0)
        prec_values.append(prec_i)
        prec_values.append(prec_max)

    fig, ax = plt.subplots()
    ax.plot(rec_values, prec_values)

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

    score = int_cnt / un_cnt if un_cnt != 0 else 0
    return score


def calculate_iou_score(in_result_folder,
                        in_mask_folder,
                        out_folder,
                        iou_thresholds,
                        conf_score_threshold,
                        nms_iou_thresh,
                        mask_threshold,
                        mask_adaptive_size,
                        target_task):

    logger = logging.getLogger('')

    result_paths = get_file_paths(in_result_folder)
    mask_paths = get_file_paths(in_mask_folder)

    ious = np.zeros(len(iou_thresholds))
    cnt = np.zeros(len(iou_thresholds))

    for result_path, mask_path in tqdm(zip(result_paths, mask_paths), total=len(result_paths)):
        if result_path.suffix != '.json' or mask_path.suffix != '.png':
            continue

        with open(result_path, 'r') as result_file:
            result = json.load(result_file)
            gt_results = get_satellitepy_dict_values(result['gt_labels'], 'masks')
            if len(gt_results) == 0:
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = 255 - mask
            mask = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                         mask_adaptive_size, mask_threshold)

            from satellitepy.data.utils import get_satellitepy_table

            instance_dict = get_satellitepy_table()[target_task]

            result = filter_out_fp_airplanes(result, target_task, instance_dict)
            # result = remove_low_conf_results(result, target_task, conf_score_threshold, no_probability=False)

            det_results = apply_nms(result['det_labels'], nms_iou_threshold=nms_iou_thresh, target_task=target_task)
            conf_scores = np.max(det_results[target_task], axis=1) if len(det_results[target_task]) > 0 else []

            matches = match_gt_and_det_bboxes(result['gt_labels'], det_results)

            for i_iou_th, iou_th in enumerate(iou_thresholds):

                for i_conf_score, conf_score in enumerate(conf_scores):
                    iou_score = matches['iou']['scores'][i_conf_score]
                    if iou_score < iou_th:
                        continue

                    gt_index = matches['iou']['indexes'][i_conf_score]
                    det_gt_value = gt_results[gt_index]
                    if det_gt_value is None:
                        continue

                    bbox = det_results['obboxes'][i_conf_score] if det_results['obboxes'][i_conf_score] != None else det_results['hbboxes'][i_conf_score]
                    det_value = decode_masks(bbox, mask)

                    iou = calc_iou(det_gt_value, det_value)

                    cnt[i_iou_th] += 1
                    ious[i_iou_th] += iou

    ious = ious / cnt
    logger.info(ious)
    return ious


def calculate_relative_score(in_result_folder, task, target_task, conf_score_threshold, iou_thresholds, nms_iou_thresh,
                             out_folder):
    logger = logging.getLogger('')

    result_paths = get_file_paths(in_result_folder)

    score = np.zeros(len(iou_thresholds))
    cnt = np.zeros(len(iou_thresholds))

    for result_path in tqdm(result_paths):
        if result_path.suffix != '.json':
            continue
        with open(result_path, 'r') as result_file:
            result = json.load(result_file)
            gt_results = get_satellitepy_dict_values(result['gt_labels'], task)
            result = remove_low_conf_results(result, target_task, conf_score_threshold)
            det_results = apply_nms(result['det_labels'], nms_iou_threshold=nms_iou_thresh, target_task=target_task)
            conf_scores = np.max(det_results[target_task], axis=1) if len(det_results[target_task]) > 0 else []

            if len(gt_results) == 0:
                continue

            for i_iou_th, iou_th in enumerate(iou_thresholds):
                for i_conf_score, conf_score in enumerate(conf_scores):
                    if conf_score < conf_score_threshold:
                        continue
                    iou_score = result['matches']['iou']['scores'][i_conf_score]
                    if iou_score < iou_th:
                        continue

                    gt_index = result['matches']['iou']['indexes'][i_conf_score]
                    det_gt_value = gt_results[gt_index]
                    if det_gt_value is None:
                        continue

                    det_value = det_results[task][i_conf_score][0]

                    error = abs(det_gt_value - det_value) / det_gt_value
                    cnt[i_iou_th] += 1
                    score[i_iou_th] += (1 - error)
    score = score / cnt
    logger.info(score)
    return score
