import logging
from pathlib import Path
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
from satellitepy.data.tools import save_chips
import configargparse

project_folder = get_project_folder()


def get_args():
    """Arguments parser."""

    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-label-format', type=str, required=True,
                        help='Label file format. e.g., dota, fair1m')
    parser.add_argument('--in-image-folder', type=Path, required=True,
                        help='Input folder which contains the images.')
    parser.add_argument('--in-label-folder', type=Path, required=True,
                        help='Input folder which contains the labels.')
    parser.add_argument('--out-folder', type=Path, required=True,
                        help='Save folder for chips. Images will be saved in <output-folder>/images and corresponding '
                             'labels in <output-folder>/labels')
    parser.add_argument('--margin-size', type=int, required=False, default=50,
                        help='Margin size of the chip to be created.')
    parser.add_argument('--chip-size', type=int, required=False, default=128,
                        help='Chip size.')
    parser.add_argument('--log-config-path', default=project_folder /
                        Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--in-mask-folder', type=Path, required=False,
                        help='Folder of original mask images. The mask images in this folder will be used to set mask '
                             'pixel coordinates in out labels.')
    parser.add_argument('--image-read-module', type=str, default='cv2',
                        help='This module will be used to read the image. rasterio is suggested for large TIF images.')
    parser.add_argument('--rescaling', type=float, default=1, help='Scale the images before creating patches. This is helpful to unify the spatial resolution over patches from different datasets. '
                        'For example, fair1m (spatial resolution ~0.8) can be rescaled to the spatial resolution of 0.5, rescaling equals 0.8/0.5=1.6. The tasks will be rescaled by the factor of <rescaling>, too, if applicable.')
    parser.add_argument('--interpolation-method', type=str, default='INTER_LINEAR', 
        help='Interpolation method to scale the images before creating patches. This is used only if rescaling is different than 1.0.',
        choices=['INTER_NEAREST','INTER_LINEAR','INTER_CUBIC','INTER_AREA'])
    parser.add_argument('--orient-objects',action='store_true', help='Orient the objects in chips')
    parser.add_argument('--mask-objects',action='store_true', help='Mask the objects in chips. If true, background will be removed in the chips')
    args = parser.parse_args()
    return args


def run(args):
    if args.in_mask_folder is not None:
        in_mask_folder = Path(args.in_mask_folder)
    else:
        in_mask_folder = None

    out_folder = Path(args.out_folder)

    margin_size = int(args.margin_size)

    assert create_folder(out_folder)

    log_path = Path(out_folder) / 'chip_creation.log'
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger('')

    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    logger.info(f'Saving chips with margin-size: {margin_size}')

    save_chips(
        label_format=args.in_label_format,
        image_folder=Path(args.in_image_folder),
        label_folder=Path(args.in_label_folder),
        out_folder=out_folder,
        margin_size=margin_size,
        chip_size=args.chip_size,
        mask_folder=in_mask_folder,
        img_read_module=args.image_read_module,
        rescaling=args.rescaling,
        interpolation_method=args.interpolation_method,
        orient_objects=args.orient_objects,
        mask_objects=args.mask_objects
    )


if __name__ == '__main__':
    args = get_args()
    run(args)
