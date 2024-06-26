import configargparse
from pathlib import Path
from satellitepy.data.tools import show_labels_on_images
from satellitepy.utils.path_utils import create_folder, init_logger, get_default_log_path, get_default_log_config
import logging

"""
Show labels (e.g., bounding boxes) on an image
"""


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', is_config_file=True, help='Path to the configuration file')
    parser.add_argument('--in-image-folder', type=Path, required=True,
                        help='Images that should be displayed')
    parser.add_argument('--in-label-folder', type=Path, required=True,
                        help='Labels that corresponds to the given images')
    parser.add_argument('--in-mask-folder', type=Path, required=False,
                        help='Path to masks')
    parser.add_argument('--in-label-format', type=str, required=True,
                        help='Label file format. e.g. dota, fair1m.')
    parser.add_argument('--out-folder', type=Path, required=True,
                        help='Folder where the generated image should be saved to.')
    parser.add_argument('--tasks', type=str, nargs='+',
                        help='Which information to show on generated images. E.g.: bboxes, masks, labels')
    parser.add_argument('--log-config-path', default=None, type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, default=None, help='Log path.')
    parser.add_argument('--image-read-module', type=str, default='cv2', help='This module will be used to read the '
                                                                             'image. rasterio is suggested for large '
                                                                             'TIF images.')
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

    log_config_path = get_default_log_config() if args.log_config_path is None else Path(args.log_config_path)
    log_path = get_default_log_path(Path(__file__).resolve().stem) if args.log_path is None else Path(args.log_path)

    init_logger(config_path=log_config_path, log_path=log_path)

    logger = logging.getLogger('')
    logger.info('Visualizing labels on images...')
    logger.info(f'Log will be stored at: {log_path}')

    config_path = out_folder / f'{Path(__file__).resolve().stem}.ini'
    parser.write_config_file(args, [str(config_path)])
    logger.info(f'Configs will be stored at {config_path}')

    in_image_folder = Path(args.in_image_folder)

    in_label_folder = Path(args.in_label_folder) if args.in_label_folder else None
    if not in_label_folder:
        logger.info('No label folder is given. No label will be displayed.')

    in_mask_folder = Path(args.in_mask_folder) if args.in_mask_folder else None
    if not in_mask_folder:
        logger.info('No mask folder is given. No mask will be displayed.')

    if args.image_read_module == 'rasterio':
        logger.info('Images will be normalized and clipped to [0,255].')

    show_labels_on_images(
        image_folder=in_image_folder,
        label_folder=in_label_folder,
        mask_folder=in_mask_folder,
        label_format=args.in_label_format,
        img_read_module=args.image_read_module,
        out_folder=out_folder,
        tasks=args.tasks,
        rescaling=args.rescaling,
        interpolation_method=args.interpolation_method
        )


if __name__ == '__main__':
    args = get_args()
    run(args)
