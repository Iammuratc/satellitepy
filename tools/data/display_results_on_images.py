import configargparse
from pathlib import Path
from satellitepy.data.tools import show_results_on_image
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder, get_default_log_path
import logging

"""
Show results (e.g., bounding boxes) on an image
"""

project_folder = get_project_folder()


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-image-folder', type=Path, required=True,
                        help='Images that should be displayed')
    parser.add_argument('--in-result-folder', type=Path, required=True,
                        help='Labels that corresponds to the given images')
    parser.add_argument('--in-mask-folder', type=Path, required=False,
                        help='Masks to display on the images. Required if masks in tasks.')
    parser.add_argument('--mask-threshold', type=float, default=10,
                        help='C for cv2.adaptiveThreshold. Value is subtracted from the threshold.')
    parser.add_argument('--mask-adaptive-size', type=float, default=101,
                        help='The threshold is the weighted sum of values in a neighbourhood of this size. Must be '
                             'odd, default is 51.')
    parser.add_argument('--out-folder', type=Path, required=True,
                        help='dir where the generated image should be saved to.')
    parser.add_argument('--tasks', type=str, nargs='+',
                        help='Which information to show on generated images. E.g.: bboxes, masks, labels. Default is '
                             'all, which shows all available information.',
                        default='all')
    parser.add_argument('--no-probability', action='store_true', 
        help='If True, results already consist of max values of probabilities,' 
            'i.e., shape is [N,1], where, N number of instances.'
            'By default, each instance will have a probability of each class, shape is [N,C], where C number of classes.')
    parser.add_argument('--target-task', type=str, default='coarse-class', help='Target task used for drawing bboxes and nms. Default is coarse-class.')
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
    image_dir = Path(args.in_image_folder)
    mask_dir = Path(args.in_mask_folder) if args.in_mask_folder else None
    mask_threshold = args.mask_threshold if args.mask_threshold else 10
    mask_adaptive_size = int(args.mask_adaptive_size) if args.mask_adaptive_size else 51
    result_dir = Path(args.in_result_folder)
    output_dir = Path(args.out_folder)
    iou_th = args.iou_threshold
    conf_th = args.conf_score_threshold

    assert create_folder(output_dir)

    tasks = args.tasks
    target_task = args.target_task

    all_tasks_flag = False
    if tasks[0] == 'all' or tasks == 'all':
        all_tasks_flag = True
    elif 'masks' in tasks:
        assert mask_dir, 'in-mask-dir must be specified if masks is in tasks!'

    logger = logging.getLogger('')
    logger.info(f'Displaying results of {image_dir.name}...')
    log_path = Path(
        output_dir) / 'display_results_on_images.log' if args.log_path is None else args.log_path

    init_logger(config_path=args.log_config_path, log_path=log_path)

    show_results_on_image(
        img_dir=image_dir,
        mask_dir=mask_dir,
        mask_threshold=mask_threshold,
        mask_adaptive_size=mask_adaptive_size,
        result_dir=result_dir,
        out_dir=output_dir,
        tasks=tasks,
        target_task=target_task,
        all_tasks_flag=all_tasks_flag,
        iou_th=iou_th,
        conf_th=conf_th,
        no_probability=args.no_probability)


if __name__ == '__main__':
    args = get_args()
    run(args)
