import configargparse
from pathlib import Path
from satellitepy.data.tools import save_patches
from satellitepy.utils.path_utils import create_folder, init_logger, get_default_log_path, get_default_log_config
import logging
import logging.config

"""
Create patches from original image and label folders 
Save patch labels as json files in the satellitepy format.
"""


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', is_config_file=True, help='Path to the configuration file')
    parser.add_argument('--patch-size', type=int,
                        help='Patch size. Patches with patch-size will be created from the original images.')
    parser.add_argument('--in-image-folder', type=Path,
                        help='Folder of original images. The images in this folder will be processed.')
    parser.add_argument('--in-label-folder', type=Path, required=False,
                        help='Folder of original labels. The labels in this folder will be used to create patches.')
    parser.add_argument('--in-mask-folder', type=Path, required=False,
                        help='Folder of original mask images. The mask images in this folder will be used to set mask '
                             'pixel coordinates in out labels.')
    parser.add_argument('--in-label-format', type=str, help='Label file format. e.g., dota, fair1m.')
    parser.add_argument('--out-folder', type=Path,
                        help='Save folder of patches. Patches and corresponding labels will be saved into '
                             '<out-folder>/patch_<patch-size>/images and <out-folder>/patch_<patch-size>/labels.')
    parser.add_argument('--truncated-object-thr', default=0.5, type=float, help='If (truncated-object-thr x object '
                                                                                'area) is not in the patch area, '
                                                                                'the object will be filtered out. '
                                                                                '1 if the object is completely in the patch, 0 if not.')
    parser.add_argument('--patch-overlap', type=int, help='Overlapping size of neighboring patches. In CNN '
                                                          'terminology, stride = patch-size - patch-overlap. '
                                                          'If necessary, the original image will be padded with zeros to create full size patches.')
    parser.add_argument('--log-config-path', default=None, type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, default=None, help='Log path.')
    # Reading image
    parser.add_argument('--image-read-module', type=str, default='cv2',
                        help='This module will be used to read the image. rasterio is suggested for large TIF images.')
    parser.add_argument('--rescaling', type=float, default=1, help='Scale the images before creating patches. This is helpful to unify the spatial resolution over patches from different datasets. '
                        'For example, fair1m (spatial resolution ~0.8) can be rescaled to the spatial resolution of 0.5, rescaling equals 0.8/0.5=1.6. The tasks will be rescaled by the factor of <rescaling>, too, if applicable.')
    parser.add_argument('--interpolation-method', type=str, default='INTER_LINEAR', 
        help='Interpolation method to scale the images before creating patches. This is used only if rescaling is different than 1.0.',
        choices=['INTER_NEAREST','INTER_LINEAR','INTER_CUBIC','INTER_AREA'])
    return parser


def run(parser):
    """Application entry point."""
    args = parser.parse_args()

    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)

    log_config_path = get_default_log_config() if args.log_config_path == None else Path(args.log_config_path)
    log_path = get_default_log_path(Path(__file__).resolve().stem) if args.log_path == None else Path(args.log_path)

    init_logger(config_path=log_config_path, log_path=log_path)

    logger = logging.getLogger('')
    logger.info('Saving patches from original images...')
    logger.info(f'Log will be stored at: {log_path}')

    config_path = out_folder / f"{Path(__file__).resolve().stem}.ini"
    parser.write_config_file(args, [str(config_path)])
    logger.info(f'Configs will be stored at {config_path}')

    in_image_folder = Path(args.in_image_folder)

    in_label_folder = Path(args.in_label_folder) if args.in_label_folder else None
    if not in_label_folder:
        logger.info('No label folder is given. Patches will have no ground truth labels.')

    in_mask_folder = Path(args.in_mask_folder) if args.in_mask_folder else None
    if not in_mask_folder:
        logger.info('No mask folder is given. Patches will have no ground truth mask.')

    save_patches(
        image_folder=in_image_folder,
        label_folder=in_label_folder,
        label_format=args.in_label_format,
        out_folder=out_folder,
        truncated_object_thr=args.truncated_object_thr,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        mask_folder=in_mask_folder,
        image_read_module=args.image_read_module,
        rescaling=args.rescaling,
        interpolation_method=args.interpolation_method,
    )


if __name__ == '__main__':
    args = get_args()
    run(args)
