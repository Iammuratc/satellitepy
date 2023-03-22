import configargparse
from pathlib import Path

from satellitepy.data.tools import split_rareplanes_labels
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
import logging

"""
Split original rareplanes label file into one label file for each image.
"""


def get_args():
    """Arguments parser."""
    project_folder = get_project_folder()
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-label-file', type=Path, required=True,
                        help='The original label file. Will be split up for every image.')
    parser.add_argument('--out-folder', type=Path, required=True,
                        help='Save folder of new labels. Labels will be saved into <out-folder>/labels.')
    parser.add_argument('--log-config-path', default=project_folder /
                        Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, help='Log config file.')
    args = parser.parse_args()
    return args


def run(args):
    in_label_file = Path(args.in_label_file)
    out_folder = Path(args.out_folder)
    out_labels_folder = Path(args.out_folder) / 'labels'

    assert create_folder(out_folder)
    assert create_folder(out_labels_folder)

    # Init logger
    log_path = Path(
        out_folder) / 'split_rareplanes_labels.log' if args.log_path == None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    logger.info('Saving split rareplanes labels')

    # Split label
    split_rareplanes_labels(
        label_file=in_label_file,
        out_folder=out_labels_folder
    )


if __name__ == '__main__':
    args = get_args()
    run(args)
