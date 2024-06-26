"""
Create labels in satellitpy format from original labels.
"""
import logging
from pathlib import Path

import configargparse

from satellitepy.data.tools import create_satellitepy_labels
from satellitepy.utils.path_utils import get_project_folder, create_folder, init_logger


def get_args():
    """Arguments parser."""
    project_folder = get_project_folder()
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-image-folder', type=Path, required=False,
                        help='Folder of original images.')
    parser.add_argument('--in-label-folder', type=Path, required=True,
                        help='Folder of original labels. The labels in this folder will be converted to satellitepy '
                             'format.')
    parser.add_argument('--in-mask-folder', type=Path, required = False,
                        help='Folder of original mask images. The mask images in this folder will be used to set mask '
                             'pixel coordinates in out labels.')
    parser.add_argument('--in-label-format', type=Path,
                        help='Label file format. e.g., dota, fair1m.')
    parser.add_argument('--out-folder',
                        type=Path,
                        help='Save folder of labels.')
    parser.add_argument('--log-config-path', default=project_folder /
                        Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, required=False, help='Log path.')
    args = parser.parse_args()
    return args


def run(args):
    in_image_folder = Path(args.in_image_folder) if args.in_image_folder else None
    in_label_folder = Path(args.in_label_folder)

    in_label_format = str(args.in_label_format)
    assert in_label_format != 'satellitepy', 'Label format is already satellitepy.'

    if args.in_mask_folder is not None:
        in_mask_folder = Path(args.in_mask_folder)
    else:
        in_mask_folder = None
    out_folder = Path(args.out_folder)

    assert create_folder(out_folder)

    log_path = Path(
        out_folder) / 'create_satellitepy_labels.log' if args.log_path is None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger('')
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    logger.info('Saving satellitepy labels from original labels...')

    create_satellitepy_labels(
        image_folder=in_image_folder,
        label_folder=in_label_folder,
        label_format=in_label_format,
        out_folder=out_folder,
        mask_folder=in_mask_folder
    )


if __name__ == '__main__':
    args = get_args()
    run(args)
