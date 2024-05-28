from pathlib import Path

import configargparse

import logging

from satellitepy.evaluate.tools import calculate_iou_score
from satellitepy.utils.path_utils import create_folder, get_project_folder, init_logger


def get_args():
    """Arguments parser."""
    project_folder = get_project_folder()

    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-result-folder', type=str,
                        help='Folder of results. The results in this folder will be processed.')
    parser.add_argument('--in-mask-folder', type=str, help='Folder of masks. The masks in this folder will be '
                                                           'processed.')
    parser.add_argument('--mask-threshold', type=float, default=10,
                        help='C for cv2.adaptiveThreshold. Value is subtracted from the threshold')
    parser.add_argument('--mask-adaptive-size', type=float, default=101,
                        help='The threshold is the weighted sum of values in a neighbourhood of this size. Must be '
                             'odd, default is 51.')
    parser.add_argument('--out-folder',
                        type=str,
                        help='Save folder of result evaluations. It will be asked to create if not exists.')
    parser.add_argument('--confidence-score-threshold', default=None, type=str,
                        help='Confidence score threshold. If the detected object has a lower'
                             'confidence score than this threshold, the object will be ignored. Default value is 0.5')
    parser.add_argument('--iou-thresholds', default=None, type=str,
                        help='Iou thresholds. Default value is range(0.5,0.96,0.05)')
    parser.add_argument('--target-task', type=str, default='coarse-class',
                        help='The model will be trained for the given target task. Needs to be a classification task. '
                             'Default is coarse-class')
    parser.add_argument('--log-config-path', default=project_folder /
                                                     Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, help='Log will be saved here. Default value is '
                                                      '<out-folder>/evaluations.log')
    args = parser.parse_args()
    return args


def main(args):
    """Application entry point."""

    in_result_folder = Path(args.in_result_folder)
    in_mask_folder = Path(args.in_mask_folder)
    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)

    iou_thresholds = [float(iou_threshold) for iou_threshold in
                      args.iou_thresholds.split(',')] if args.iou_thresholds is not None else [x / 100.0 for x in
                                                                                               range(50, 96, 5)]
    conf_score_threshold = float(args.confidence_score_threshold) if args.confidence_score_threshold else 0.5
    mask_threshold = int(args.mask_threshold) if args.mask_threshold else 10
    mask_adaptive_size = int(args.mask_adaptive_size) if args.mask_adaptive_size else 51
    assert mask_adaptive_size % 2 == 1, 'mask-adaptive-size must be odd'
    out_folder = Path(args.out_folder)

    target_task = args.target_task

    log_path = Path(
        out_folder) / 'evaluations.log' if args.log_path is None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger('')
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    logger.info('Calculating IoU-score...')

    calculate_iou_score(
        in_result_folder,
        in_mask_folder,
        out_folder,
        iou_thresholds,
        conf_score_threshold,
        mask_threshold,
        mask_adaptive_size,
        target_task
    )


if __name__ == '__main__':
    args = get_args()
    main(args)
