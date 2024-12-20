"""
Calculate mean average precision from result json files for mmrotate models.
See the details of the result json files at satellitepy.evaluate.mmrotate.utils.get_result
"""

import configargparse
from pathlib import Path
import logging

from satellitepy.evaluate.tools import calculate_map
from satellitepy.evaluate.utils import get_instance_names
from satellitepy.utils.path_utils import create_folder, init_logger, get_default_log_path, get_default_log_config
from satellitepy.data.utils import get_satellitepy_table

def get_args():
    """Arguments parser."""

    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-result-folder', type=str,
                        help='Folder of results. The results in this folder will be processed.')
    parser.add_argument('--task', required=True, type=str, help='Task name. This task will be evaluated.')
    parser.add_argument('--out-folder',
                        type=str,
                        help='Save folder of result evaluations. It will be asked to create if not exists.')
    parser.add_argument('--instance-names', type=str, help='Instance names. The instance name --Background-- will be '
                                                           'added automatically. All other instance names (e.g., '
                                                           'bridge), that are not defined here but in the result '
                                                           'files, will be treated as Background.')
    parser.add_argument('--ignore-other-instances', action='store_true', help='If True, ignore instances not in instance names.')
    parser.add_argument('--confidence-score-thresholds', default=None, type=str, help='Confidence score threshold. If '
                                                                                      'the detected object has a '
                                                                                      'lower confidence score than '
                                                                                      'this threshold, the object '
                                                                                      'will be ignored. Default value '
                                                                                      'is range(0,1.01,0.05)')
    parser.add_argument('--iou-thresholds', default=None, type=str, help='Iou thresholds. Default value is range(0.5,'
                                                                         '0.96,0.05)')
    parser.add_argument('--no-probability', action='store_true', 
        help='If True, results already consist of max values of probabilities,' 
            'i.e., shape is [N,1], where, N number of instances.'
            'By default, each instance will have a probability of each class, shape is [N,C], where C number of classes.')
    parser.add_argument('--norm-conf-scores', action='store_true', 
        help='BBAVector produces conf score results larger than 1. If True, the confidence score sum will be normalized to 1. ')
    parser.add_argument('--plot-pr', action='store_true', help='Plot the PR curve.')
    parser.add_argument('--nms-iou-thresh', type=float, default=0.3,
                        help='Non-maximum suppression IOU threshold. Overlapping predictions will be removed '
                             'according to this value.')
    parser.add_argument('--by-source', action='store_true', help='If True, the calculations will be done for each annotation source.')
    parser.add_argument('--store-undetected-objects', action='store_true', help='If True, the ground truth indices of the undetected objects will be stored in the result file.')
    parser.add_argument('--log-config-path', default=None, help='Log config file.')
    parser.add_argument('--log-path', default=None, help='Log will be saved here.')
    return parser


def main(parser):
    """Application entry point."""
    args = parser.parse_args()

    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)

    log_config_path = get_default_log_config() if args.log_config_path is None else Path(args.log_config_path)
    log_path = get_default_log_path(Path(__file__).resolve().stem) if args.log_path is None else Path(args.log_path)

    init_logger(config_path=log_config_path, log_path=log_path)
    logger = logging.getLogger('')
    logger.info('mAP values will be calculated...')
    logger.info(f'Log will be stored at: {log_path}')

    config_path = out_folder / f'{Path(__file__).resolve().stem}.ini'
    parser.write_config_file(args, [str(config_path)])
    logger.info(f'Configs will be stored at {config_path}')

    in_result_folder = Path(args.in_result_folder)
    task = args.task

    # instance_names = [instance_name for instance_name in args.instance_names.split(',')] if args.instance_names is not None else get_instance_names(in_result_folder, task)
    instance_dict = get_satellitepy_table()[task]
    # print(instance_names)
    ignore_other_instances = args.ignore_other_instances
    iou_thresholds = [float(iou_threshold) for iou_threshold in
                      args.iou_thresholds.split(',')] if args.iou_thresholds is not None else [x / 100.0 for x in
                                                                                               range(50, 96, 5)]
    conf_score_thresholds = [float(confidence_score_threshold) for confidence_score_threshold in
                             args.confidence_score_thresholds.split(
                                 ',')] if args.confidence_score_thresholds is not None else [x / 100.0 for x in
                                                                                             range(0, 96, 5)]
    plot_pr = args.plot_pr
    out_folder = Path(args.out_folder)

    assert create_folder(out_folder)

    nms_iou_thresh = args.nms_iou_thresh

    calculate_map(
        in_result_folder,
        task,
        instance_dict,
        conf_score_thresholds,
        iou_thresholds,
        out_folder,
        plot_pr,
        nms_iou_thresh,
        ignore_other_instances,
        no_probability=args.no_probability,
        by_source=args.by_source,
        norm_conf_scores=args.norm_conf_scores,
        store_undetected_objects=args.store_undetected_objects
    )


if __name__ == '__main__':
    args = get_args()
    main(args)
