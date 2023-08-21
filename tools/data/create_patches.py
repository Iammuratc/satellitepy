import configargparse
from pathlib import Path
from satellitepy.data.tools import save_patches
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
import logging
import sys
"""
Create patches from original image and label folders 
Save patch labels in json files that are in satellitepy format.
"""


def get_args():
    """Arguments parser."""
    project_folder = get_project_folder()
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch-size', type=int,
                        help='Patch size. Patches with patch-size will be created from the original images.')
    parser.add_argument('--in-image-folder', type=Path,
                        help='Folder of original images. The images in this folder will be processed.')
    parser.add_argument('--in-label-folder', type=Path, required=False,
                        help='Folder of original labels. The labels in this folder will be used to create patches.')
    parser.add_argument('--in-mask-folder', type=Path, required = False,
                        help='Folder of original mask images. The mask images in this folder will be used to set mask pixel coordinates in out labels.')
    parser.add_argument('--in-label-format', type=Path, default="satellitepy",
                        help='Label file format. e.g., dota, fair1m.')
    parser.add_argument('--out-folder',
                        type=Path,
                        help='Save folder of patches. Patches and corresponding labels will be saved into <out-folder>/patch_<patch-size>/images and <out-folder>/patch_<patch-size>/labels.')
    parser.add_argument('--truncated-object-thr', default=0.5, type=float, help='If (truncated-object-thr x object area) is not in the patch area, the object will be filtered out.'
                        '1 if the object is completely in the patch, 0 if not.')
    parser.add_argument('--patch-overlap', type=int, help='Overlapping size of neighboring patches. In CNN terminology, stride = patch-size - patch-overlap.'
                        'If necessary, the original image will be padded with zeros to create full size patches.')
    parser.add_argument('--log-config-path', default=project_folder /
                        Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, required=False, help='Log path.')
    parser.add_argument('--include-object-classes', nargs="*", type=str, default=None,
                        help='A list of object class names that shall be included. Ignores all other object classes if not None. Takes precedence over --exclude-object-classes.')
    parser.add_argument('--exclude-object-classes', nargs="*", type=str, default=None, 
                        help='A list of object class names that shall be excluded. Includes all other object classes. Overriden by --include-object-classes if set.')
    args = parser.parse_args()
    return args


def run(args):
    in_image_folder = Path(args.in_image_folder)
    if args.in_label_folder:
        in_label_folder = Path(args.in_label_folder)
    else:
        in_label_folder = None
    in_label_format = str(args.in_label_format)
    if (args.in_mask_folder != None):
        in_mask_folder = Path(args.in_mask_folder)
    else:
        in_mask_folder = None
    out_folder = Path(args.out_folder)
    patch_overlap = int(args.patch_overlap)
    patch_size = int(args.patch_size)
    truncated_object_thr = float(args.truncated_object_thr)
    include_object_classes = list(args.include_object_classes) if args.include_object_classes is not None else None
    exclude_object_classes = list(args.exclude_object_classes) if args.exclude_object_classes is not None else None

    assert create_folder(out_folder)

    # Init logger
    log_path = Path(
        out_folder) / 'create_patches.log' if args.log_path == None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    logger.info('Saving patches from original images...')

    if not in_label_folder :
        logger.warn("in_label_folder is not set, this is correct, if patches for the test set shall be created")

    # Save patches
    save_patches(
        image_folder=in_image_folder,
        label_folder=in_label_folder,
        label_format=in_label_format,
        out_folder=out_folder,
        truncated_object_thr=truncated_object_thr,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        include_object_classes=include_object_classes,
        exclude_object_classes=exclude_object_classes,
        mask_folder = in_mask_folder
    )
if __name__ == '__main__':
    args = get_args()
    run(args)
