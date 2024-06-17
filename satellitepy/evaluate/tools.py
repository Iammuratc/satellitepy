import cv2
import numpy as np
import matplotlib.pyplot as plt

from satellitepy.data.utils import get_satellitepy_dict_values
from satellitepy.models.bbavector.utils import decode_masks
from satellitepy.utils.path_utils import get_file_paths
import logging
import json
from satellitepy.evaluate.utils import set_conf_mat_from_result, get_precision_recall, get_average_precision, \
    remove_low_conf_results, match_gt_and_det_bboxes
from tqdm import tqdm
from satellitepy.evaluate.bbavector.utils import apply_nms


def calculate_map(
        in_result_folder,
        task,
        instance_names,
        conf_score_thresholds,
        iou_thresholds,
        out_folder,
        plot_pr,
        nms_iou_thresh,
        ignore_other_instances=False):

    logger = logging.getLogger('')

    instance_names = instance_names + ['Background']

    conf_mat = np.zeros(
        shape=(len(iou_thresholds), len(conf_score_thresholds), len(instance_names), len(instance_names)))

    result_paths = get_file_paths(in_result_folder)

    ignored_instances = []
    ignored_cnt = 0

    for result_path in tqdm(result_paths):
        if result_path.suffix != '.json':
            continue
        with open(result_path, 'r') as result_file:
            result = json.load(result_file)
        conf_mat, ignored_instances_ret, ignored_cnt_ret = set_conf_mat_from_result(
            conf_mat,
            task,
            result,
            instance_names,
            conf_score_thresholds,
            iou_thresholds,
            nms_iou_thresh,
            ignore_other_instances)

        ignored_instances += ignored_instances_ret
        ignored_cnt += ignored_cnt_ret

    pr_threshold_ind = 0
    precision, recall = get_precision_recall(conf_mat, sort_values=True)
    with np.printoptions(threshold=np.inf):
        logger.info('AP')
        ap = get_average_precision(precision, recall)
        logger.info('Instance names')
        logger.info(instance_names)
        logger.info(ap)
        logger.info('mAP')
        mAP = np.sum(np.transpose(np.transpose(ap)[:-1]), axis=1) / (len(ap[0]) - 1)
        logger.info(mAP)
        if ignore_other_instances:
            logger.info(f'ignored {ignored_cnt} other instances. Ignored instance names: {set(ignored_instances)}')

        evaluation_file_path = out_folder / 'mAP_values.txt'
        with open(evaluation_file_path, 'w') as file:
            np.savetxt(file, ap, fmt='%.2f', delimiter=',',
                       header=f'AP (Columns: {instance_names}, Rows: IoUs {iou_thresholds}):')
            np.savetxt(file, mAP, fmt='%.2f', delimiter=',', header=f'mAP (Columns: {iou_thresholds}):')

    if plot_pr:
        precision_recall_curve(out_folder, precision[pr_threshold_ind, :], recall[pr_threshold_ind, :])

    return mAP


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

            result = remove_low_conf_results(result, target_task, conf_score_threshold)
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

                    bbox = det_results['hbboxes'][i_conf_score] if det_results['hbboxes'][
                        i_conf_score].any() else det_results['hbboxes'][i_conf_score]
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
