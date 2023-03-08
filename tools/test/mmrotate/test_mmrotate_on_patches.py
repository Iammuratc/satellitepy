
"""
Test MMRotate models on the folder with patches. 
Save detected bbox details (i.e., corners, class_name) with the corresponding ground truth labels.
"""

import os
import configargparse
from satellitepy.evaluate.mmrotate.tools import save_mmrotate_patch_results
from satellitepy.utils.path_utils import create_folder, init_logger
from pathlib import Path
import logging

def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-path', required=True, help='Path to MMRotate config file. Please refer to https://github.com/open-mmlab/mmrotate for more details.')
    parser.add_argument('--weights-path', required=True, help='Path to MMRotate model weights file. Please refer to https://github.com/open-mmlab/mmrotate for more details.')
    parser.add_argument('--nms-on-multiclass-thr', default=0.5, type=float, help='nms_on_multiclass_thr value is used to filter out the overlapping bounding boxes with lower scores, and keep the best. Set 0.0 to cancel it.')
    parser.add_argument('--device', default='cuda:0', help='Device to load the model.')
    parser.add_argument('--class-names', required=True, type=str, help='Class names. MMRotate does not include class names in config files, but class indexes. Be sure that the indices match with the class names')
    parser.add_argument('--log-config-path', default=Path("./configs/log.config") ,type=Path, help='Log config file.')
    parser.add_argument('--in-image-folder', required=True, help='Test image folder. The images in this folder will be tested.')
    parser.add_argument('--in-label-folder', required=True, help='Test label folder. The labels in this folder will be used for evaluation purposes.')
    parser.add_argument('--in-label-format', required=True, help='Test label file format. e.g., dota, fair1m.')
    parser.add_argument('--out-folder',
        required=True,
        help='Save folder of detected bounding boxes. Bounding box labels will be saved into <out-folder>/results/patch_labels.')
    args = parser.parse_args()
    return args


def main(args):
    """Application entry point."""

    log_config_path = Path(args.log_config_path)
    config_path = args.config_path
    weights_path = args.weights_path
    nms_on_multiclass_thr = args.nms_on_multiclass_thr
    device = args.device
    out_folder = Path(args.out_folder)
    in_image_folder = Path(args.in_image_folder)
    in_label_folder = Path(args.in_label_folder)
    in_label_format = args.in_label_format
    class_names = [class_name for class_name in args.class_names.split(',')]

    assert create_folder(out_folder)
    # Initiate logger
    init_logger(config_path=log_config_path,log_path=os.path.join(out_folder,'results.log'))
    logger = logging.getLogger(__name__)
    logger.info('MMRotate model will process the images...')

    save_mmrotate_patch_results(
        in_image_folder=in_image_folder,
        in_label_folder=in_label_folder,
        in_label_format=in_label_format,
        out_folder=out_folder,
        config_path=config_path,
        weights_path=weights_path,
        nms_on_multiclass_thr=nms_on_multiclass_thr,
        device=device,
        class_names=class_names)


if __name__ == '__main__':
    args = get_args()
    main(args)