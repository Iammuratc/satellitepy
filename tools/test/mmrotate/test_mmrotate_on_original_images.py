
"""
Test MMRotate models on the folder with patches. 
Save detected bbox details (e.g., corners, class_name) with the corresponding ground truth labels.
"""

import os
import configargparse
from pathlib import Path
import logging

from satellitepy.evaluate.mmrotate.tools import save_mmrotate_original_results
from satellitepy.utils.path_utils import create_folder, init_logger
from utils import add_shared_args


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser = add_shared_args(parser)
    parser.add_argument('--patch-size', required=True, type=int, help='Patch size. Patches with patch_size will be created from the original image to be tested by the MMRotate model.' 
        'Be sure patch_size is the same as the input image size of the MMRotate model.')
    parser.add_argument('--patch-overlap', required=True, type=int, help='Overlapping size of neighboring patches. In CNN terminology, stride = patch_size - patch_overlap')
    parser.add_argument('--truncated-object-thr', default=0.5, type=float, help='If (truncated-object-thr x object area) is not in the patch area, the object will be filtered out. 1 if the object is completely in the patch, 0 if not.')
    parser.add_argument('--out-folder',
        required=True,
        help='Save folder of detected bounding boxes. Bounding box labels will be saved into <out-folder>/results/original_labels.')
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
    patch_overlap = args.patch_overlap
    patch_size = args.patch_size
    truncated_object_thr = args.truncated_object_thr
    class_names = [class_name for class_name in args.class_names.split(',')]

    assert create_folder(out_folder)
    # Initiate logger
    init_logger(config_path=log_config_path,log_path=os.path.join(out_folder,'original_results.log'))
    logger = logging.getLogger(__name__)
    logger.info('MMRotate model will process the images...')

    save_mmrotate_original_results(
        in_image_folder=in_image_folder,
        in_label_folder=in_label_folder,
        in_label_format=in_label_format,
        out_folder=out_folder,
        config_path=config_path,
        weights_path=weights_path,
        nms_on_multiclass_thr=nms_on_multiclass_thr,
        device=device,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        truncated_object_thr=truncated_object_thr,
        class_names=class_names)


if __name__ == '__main__':
    args = get_args()
    main(args)