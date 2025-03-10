"""
Test BBAVector models on the folder with patches. 
Save detected bbox details (i.e., corners, class_name) with the corresponding ground truth labels.
"""

import configargparse
from satellitepy.evaluate.bbavector.tools import save_patch_results
from satellitepy.utils.path_utils import create_folder, init_logger, get_default_log_path, get_default_log_config
from pathlib import Path
import logging


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-path', help='Path to BBAVector model weights file.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--input-h', type=int, default=608, help='Resized image height. Equivalent of patch height.')
    parser.add_argument('--input-w', type=int, default=608, help='Resized image width. Equivalent of patch width.')
    parser.add_argument('--truncated-object-thr', default=0.5, type=float,
                        help='If (truncated-object-thr x object area) is not in the patch area, the object will be '
                             'filtered out. 1 if the object is completely in the patch, 0 if not.')
    parser.add_argument('--device', default='cuda:0', help='Device to load the model.')
    parser.add_argument('--tasks', default=['coarse-class'], nargs="+",
                        help='The model will be trained for the given tasks. Find the other task names at '
                             'satellitepy.data.utils.get_satellitepy_table. If it is fine-class or very-fine class, '
                             'None values in those keys will be filled from one upper level')
    parser.add_argument('--target-task', type=str, default='coarse-class',
                        help='The model will be trained for the given target task. Needs to be a classification task. '
                             'Default is coarse-class')
    parser.add_argument('--log-config-path', default=Path('./configs/log.config'), type=Path, help='Log config file.')
    parser.add_argument('--in-image-folder', help='Test image folder. The images in this folder will be tested.')
    parser.add_argument('--in-label-folder', required=False,
                        help='Test label folder. The labels in this folder will be used for evaluation purposes.')
    parser.add_argument('--in-label-format', default='satellitepy', help='Test label file format. e.g., dota, fair1m.')
    parser.add_argument('--in-mask-folder', type=Path, required=False,
                        help='Folder of original mask images. The mask images in this folder will be used to set mask '
                             'pixel coordinates in out labels.')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf-thresh', type=float, default=0.18,
                        help='Confidence threshold, 0.1 for general evaluation')
    parser.add_argument('--out-folder',
                        help='Save folder of detected bounding boxes. Bounding box labels will be saved into '
                             '<out-folder>/results/patch_labels.')
    parser.add_argument('--log-path', type=Path, default=None, help='Log will be saved here.')
    args = parser.parse_args()
    return args


def main(args):
    """Application entry point."""
    log_config_path = get_default_log_config() if args.log_config_path is None else Path(args.log_config_path)
    log_path = get_default_log_path(Path(__file__).resolve().stem) if args.log_path is None else Path(args.log_path)

    init_logger(config_path=log_config_path, log_path=log_path)
    logger = logging.getLogger('')
    logger.info('BBAVector model will process the images...')
    logger.info(f'Log will be stored at: {log_path}')

    weights_path = args.weights_path
    device = args.device
    out_folder = Path(args.out_folder)
    in_image_folder = Path(args.in_image_folder)
    in_label_folder = Path(args.in_label_folder) if args.in_label_folder else None
    if not in_label_folder:
        logger.info('No label folder is given. Results will have no ground truth labels.')

    in_mask_folder = Path(args.in_mask_folder) if args.in_mask_folder else None
    if not in_mask_folder:
        logger.info('No mask folder is given. Results will have no ground truth mask.')

    in_label_format = args.in_label_format
    tasks = args.tasks
    target_task = args.target_task
    assert target_task in tasks, 'target task must be part of the tasks'

    conf_thresh = args.conf_thresh
    K = args.K
    input_h = args.input_h
    input_w = args.input_w
    num_workers = args.num_workers
    down_ratio = 4

    assert create_folder(out_folder)

    init_logger(config_path=log_config_path, log_path=log_path)
    logger = logging.getLogger('')
    logger.info(f'Log will be stored at: {log_path}')
    logger.info('BBAVector model will process the images...')

    save_patch_results(
        out_folder=out_folder,
        in_image_folder=in_image_folder,
        in_label_folder=in_label_folder,
        in_label_format=in_label_format,
        checkpoint_path=weights_path,
        device=device,
        tasks=tasks,
        target_task=target_task,
        K=K,
        conf_thresh=conf_thresh,
        num_workers=num_workers,
        input_h=input_h,
        input_w=input_w,
        down_ratio=down_ratio)


if __name__ == '__main__':
    args = get_args()
    main(args)
