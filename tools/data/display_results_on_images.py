import configargparse
from pathlib import Path
from satellitepy.data.tools import show_results_on_image
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
import logging

"""
Show results (e.g., bounding boxes) on an image
"""

project_folder = get_project_folder()


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-image-dir', type=Path, required=True,
                        help='Images that should be displayed')
    parser.add_argument('--in-result-dir', type=Path, required=True,
                        help='Labels that corresponds to the given images')
    parser.add_argument('--in-mask-dir', type=Path, required=False,
                        help='Masks to display on the images. Required if masks in tasks.')
    parser.add_argument('--mask-threshold', type= float, default=0.02,
                        help='Only pixels with intensity above this value will be set as mask. Range 0 to 1.')
    parser.add_argument('--out-dir', type=Path, required=True,
                        help='dir where the generated image should be saved to.')
    parser.add_argument('--tasks', type=str, nargs='+',
                        help='Which information to show on generated images. E.g.: bboxes, masks, labels')
    parser.add_argument('--conf-score-threshold', type=float, default=0.5,
                        help='Confidence score threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IOU threshold')
    parser.add_argument('--log-config-path', default=project_folder /
                                                     Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, default=None, help='Log file path.')
    args = parser.parse_args()
    return args


def run(args):
    image_dir = Path(args.in_image_dir)
    mask_dir = Path(args.in_mask_dir) if args.in_mask_dir else None
    mask_threshold = args.mask_threshold if args.mask_threshold else 0.02
    result_dir = Path(args.in_result_dir)
    output_dir = Path(args.out_dir)
    iou_th = args.iou_threshold
    conf_th = args.conf_score_threshold

    assert create_folder(output_dir)

    tasks = args.tasks

    if 'masks' in tasks:
        assert mask_dir, 'in-mask-dir must be specified if masks is in tasks!'

    log_path = output_dir / f'display_labels.log' if args.log_path == None else args.log_path

    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')

    logger.info(f'Displaying results of {image_dir.name}...')

    show_results_on_image(
        img_dir = image_dir,
        mask_dir = mask_dir,
        mask_threshold = mask_threshold,
        result_dir = result_dir,
        out_dir = output_dir,
        tasks = tasks,
        iou_th = iou_th,
        conf_th = conf_th)


if __name__ == '__main__':
    args = get_args()
    run(args)
