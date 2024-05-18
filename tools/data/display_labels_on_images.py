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
    parser.add_argument('--image-read-module', type=str, default='cv2', help='This module will be used to read the image. rasterio is suggested for large TIF images.')
    return parser


def run(parser):
    """Application entry point."""
    args = parser.parse_args()

    out_folder = Path(args.out_folder)
    assert create_folder(out_folder)

    # Logger configs
    log_config_path = get_default_log_config() if args.log_config_path==None else Path(args.log_config_path)
    log_path = get_default_log_path(Path(__file__).resolve().stem) if args.log_path==None else Path(args.log_path)

    # Initiate logger
    init_logger(config_path=log_config_path,log_path=log_path)

    logger = logging.getLogger('')
    logger.info('Visualizing labels on images...')
    logger.info(f'Log will be stored at: {log_path}')
    
    # configargparse config
    config_path = out_folder / f"{Path(__file__).resolve().stem}.ini"
    parser.write_config_file(args, [str(config_path)])
    logger.info(f"Configs will be stored at {config_path}")

    # Data arguments
    in_image_folder = Path(args.in_image_folder)

    in_label_folder = Path(args.in_label_folder) if args.in_label_folder else None
    if not in_label_folder:
        logger.info("No label folder is given. Patches will have no ground truth labels.")

    in_mask_folder = Path(args.in_mask_folder) if args.in_mask_folder else None
    if not in_mask_folder:
        logger.info("No mask folder is given. Patches will have no ground truth mask.")
    
    if args.image_read_module == 'rasterio':
        logger.info("Images will be normalized and clipped to [0,255].")

    show_labels_on_images(
        image_folder=in_image_folder,
        label_folder=in_label_folder,
        mask_folder=in_mask_folder,
        label_format=args.in_label_format,
        img_read_module=args.image_read_module,
        out_folder=out_folder,
        tasks=args.tasks)
    
if __name__ == '__main__':
    args = get_args()
    run(args)
