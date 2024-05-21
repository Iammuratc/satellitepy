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
    parser.add_argument('--in-img-folder', type=Path, required=True,
                        help='Input folder which contains the images.')
    parser.add_argument('--in-label-folder', type=Path, required=True,
                        help='Input folder which contains the labels.')
    parser.add_argument('--out-folder', type=Path, required=True,
                        help='Save folder for chips. Images will be saved in <output-folder>/images and corresponding '
                             'labels in <output-folder>/labels')
    parser.add_argument('--margin-size', type=int, required=True,
                        help='Margin size of the chip to be created.')
    parser.add_argument('--log-config-path', default=project_folder /
                        Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--include-object-classes', nargs="*", type=str, default=None,
                        help='A list of object class names that shall be included. Ignores all other object classes '
                             'if not None. Takes precedence over --exclude-object-classes.')
    parser.add_argument('--exclude-object-classes', nargs="*", type=str, default=None,
                        help='A list of object class names that shall be excluded. Includes all other object classes. '
                             'Overriden by --include-object-classes if set.')
    parser.add_argument('--in-mask-folder', type=Path, required=False,
                        help='Folder of original mask images. The mask images in this folder will be used to set mask '
                             'pixel coordinates in out labels.')

    args = parser.parse_args()
    return args


def run(args):
    in_label_format = args.in_label_format
    in_img_folder = Path(args.in_img_folder)
    in_label_folder = Path(args.in_label_folder)

    if args.in_mask_folder is not None:
        in_mask_folder = Path(args.in_mask_folder)
    else:
        in_mask_folder = None

    out_folder = Path(args.out_folder)

    include_object_classes = list(args.include_object_classes) if args.include_object_classes is not None else None
    exclude_object_classes = list(args.exclude_object_classes) if args.exclude_object_classes is not None else None

    margin_size = int(args.margin_size)

    assert create_folder(out_folder)

    log_path = Path(out_folder) / 'chip_creation.log'
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger('')

    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    logger.info(f'Saving chips with margin-size: {margin_size}')

    save_chips(
        label_format=in_label_format,
        image_folder=in_img_folder,
        label_folder=in_label_folder,
        out_folder=out_folder,
        margin_size=margin_size,
        include_object_classes=include_object_classes,
        exclude_object_classes=exclude_object_classes,
        mask_folder=in_mask_folder
    )


if __name__ == '__main__':
    args = get_args()
    run(args)
