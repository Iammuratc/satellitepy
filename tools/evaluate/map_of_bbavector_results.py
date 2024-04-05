"""
Calculate mean average precision from result json files for mmrotate models.
See the details of the result json files at satellitepy.evaluate.mmrotate.utils.get_result
"""

import configargparse
from pathlib import Path
import logging

from satellitepy.evaluate.tools import calculate_map
from satellitepy.utils.path_utils import create_folder, init_logger, get_default_log_path, get_default_log_config


def get_args():
    """Arguments parser."""

    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-result-folder', type=str,
                        help='Folder of results. The results in this folder will be processed.')
    parser.add_argument('--task', required=True, type=str, help='Task name. This task will be evaluated.')
    parser.add_argument('--out-folder',
                        type=str,
                        help='Save folder of result evaluations. It will be asked to create if not exists.')
    parser.add_argument('--instance-names', type=str, help='Instance names. The instance name --Background-- will be added automatically.'
        'All other instance names (e.g., bridge), that are not defined here but in the result files, will be treated as Background.')
    parser.add_argument('--ignore-other-instances', default=False, type=bool,
                        help='Ignores instances not in instance names if set. Default is False.')
    parser.add_argument('--confidence-score-thresholds', default=None, type=str, help='Confidence score threshold. If the detected object has a lower' 
        'confidence score than this threshold, the object will be ignored. Default value is range(0,1.01,0.05).')
    parser.add_argument('--iou-thresholds', default=None, type=str, help='Iou thresholds. Default value is range(0.5,0.96,0.05)')
    parser.add_argument('--plot-pr', default=False, type=bool, help='Plot the PR curve.')
    parser.add_argument('--log-config-path', default=None, help='Log config file.')
    parser.add_argument('--log-path', default=None, help='Log will be saved here.')
    args = parser.parse_args()
    return args

def main(args):
    """Application entry point."""
    # Logger configs
    log_config_path = get_default_log_config() if args.log_config_path==None else Path(args.log_config_path)
    log_path = get_default_log_path(Path(__file__).resolve().stem) if args.log_path==None else Path(args.log_path)

    # Init arguments
    in_result_folder = Path(args.in_result_folder)
    instance_names = [instance_name for instance_name in args.instance_names.split(',')]
    ignore_other_instances = args.ignore_other_instances
    iou_thresholds = [float(iou_threshold) for iou_threshold in args.iou_thresholds.split(',')] if args.iou_thresholds != None else [x / 100.0 for x in range(50, 96, 5)]
    conf_score_thresholds = [float(confidence_score_threshold) for confidence_score_threshold in args.confidence_score_thresholds.split(',')] if args.confidence_score_thresholds != None else [x / 100.0 for x in range(0, 96, 5)]
    plot_pr = args.plot_pr
    out_folder = Path(args.out_folder)
    task = args.task
    assert create_folder(out_folder)
    
    # Init logger
    init_logger(config_path=log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info(f'Log will be stored at: {log_path}')
    logger.info('Saving patches from original images...')

    # Calculate mAP
    calculate_map(
        in_result_folder,
        task,
        instance_names,
        conf_score_thresholds,
        iou_thresholds,
        out_folder,
        plot_pr,
        ignore_other_instances)

if __name__ == '__main__':
    args = get_args()
    main(args)
