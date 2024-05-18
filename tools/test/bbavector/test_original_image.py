
"""
Test BBAVector models on the folder with original images.
Save detected bbox details (i.e., corners, class_name) with the corresponding ground truth labels.
"""

import configargparse
from satellitepy.evaluate.bbavector.tools import save_original_image_results
from satellitepy.utils.path_utils import create_folder, init_logger, get_default_log_path, get_default_log_config
from pathlib import Path
import logging

def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', is_config_file=True, help='Path to the configuration file')
    parser.add_argument('--weights-path',  help='Path to BBAVector model weights file.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--input-h', type=int, default=600, help='Resized image height. Equivalent of patch height.')
    parser.add_argument('--input-w', type=int, default=600, help='Resized image width. Equivalent of patch width.')
    parser.add_argument('--patch-size', type=int, help='Patch size. Patches with patch_size will be created from the original image to be tested by the MMRotate model.' 
        'Be sure patch_size is the same as the input image size of your model, if not it will be resized to the input size.')
    parser.add_argument('--patch-overlap',  type=int, help='Overlapping size of neighboring patches. In CNN terminology, stride = patch_size - patch_overlap')
    parser.add_argument('--truncated-object-thresh', default=0.5, type=float, help='If (truncated-object-thr x object area) is not in the patch area, the object will be filtered out. 1 if the object is completely in the patch, 0 if not.')
    parser.add_argument('--device', default='cuda:0', help='Device to load the model.')
    parser.add_argument('--tasks', default=['coarse-class'], nargs="+", help='The model will be trained for the given tasks.' 
            'Find the other task names at satellitepy.data.utils.get_satellitepy_table.'
            'If it is fine-class or very-fine class, None values in those keys will be filled from one upper level')
    parser.add_argument('--target-task', type=str, default='coarse-class',
                        help='The model will be trained for the given target task. Needs to be a classification task. Default is coarse-class')
    # Path configs
    parser.add_argument('--log-config-path', default=None ,type=Path, help='Log config file.')
    parser.add_argument('--in-image-folder',  help='Test image folder. The images in this folder will be tested.')
    parser.add_argument('--in-label-folder', required=False, help='Test label folder. The labels in this folder will be used for evaluation purposes.')
    parser.add_argument('--in-label-format', default="satellitepy", help='Test label file format. e.g., dota, fair1m.')
    parser.add_argument('--in-mask-folder', type=Path, required = False,
        help='Folder of original mask images. The mask images in this folder will be used to set mask pixel coordinates in out labels.')
    parser.add_argument('--out-folder', help='Save folder of detected bounding boxes. Predictions will be saved into <out-folder>.')
    parser.add_argument('--K', type=int, default=500, help='Maximum of objects')
    parser.add_argument('--conf-thresh', type=float, default=0.25,
                        help='Confidence threshold, 0.25 for general evaluation')
    parser.add_argument('--log-path', type=Path, default=None, help='Log will be saved here.')
    parser.add_argument('--image-read-module', type=str, default='cv2', help='This module will be used to read the image. rasterio is suggested for large TIF images.')
    return parser


def main(parser):
    """Application entry point."""
    args = parser.parse_args()

    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)


    # Logger configs
    log_config_path = get_default_log_config() if args.log_config_path==None else Path(args.log_config_path)
    log_path = get_default_log_path(Path(__file__).resolve().stem) if args.log_path==None else Path(args.log_path)

    # Initiate logger
    init_logger(config_path=log_config_path,log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info('BBAVector model will process the images...')
    logger.info(f'Log will be stored at: {log_path}')

    # configargparse config
    config_path = out_folder / f"{Path(__file__).resolve().stem}.ini"
    parser.write_config_file(args, [str(config_path)])
    logger.info(f"Configs will be stored at {config_path}")

    # Model arguments
    tasks = args.tasks
    target_task = args.target_task
    assert target_task in tasks, "target task must be part of the tasks"

    down_ratio = 4
    patch_overlap = args.patch_overlap
    patch_size = args.patch_size
    truncated_object_threshold = args.truncated_object_thresh

    # Data arguments
    in_image_folder = Path(args.in_image_folder)
    in_label_format = args.in_label_format

    in_label_folder = Path(args.in_label_folder) if args.in_label_folder else None
    if not in_label_folder:
        logger.info("No label folder is given. Results will have no ground truth labels.")

    in_mask_folder = Path(args.in_mask_folder) if args.in_mask_folder else None
    if not in_mask_folder:
        logger.info("No mask folder is given. Results will have no ground truth mask.")
 
    # Call the testing function
    save_original_image_results(
        out_folder=out_folder,
        in_image_folder=in_image_folder,
        in_mask_folder=in_mask_folder,
        in_label_folder=in_label_folder,
        in_label_format=in_label_format,
        truncated_object_threshold=args.truncated_object_thresh,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        checkpoint_path=args.weights_path,
        device=args.device,
        tasks=tasks,
        K=args.K,
        conf_thresh=args.conf_thresh,
        num_workers=args.num_workers,
        input_h=args.input_h,
        input_w=args.input_w,
        down_ratio = down_ratio,
        nms_iou_threshold=args.nms_iou_thresh,
        target_task=target_task,
        img_read_module=args.image_read_module)
        # nms_on_multiclass_thr)


if __name__ == '__main__':
    args = get_args()
    main(args)
