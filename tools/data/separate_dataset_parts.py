import configargparse
from pathlib import Path

from satellitepy.data.tools import separate_dataset_parts
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
import logging

project_folder = get_project_folder()


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-label-folder', type=Path, required=False,
                        help='The original label folder.')
    parser.add_argument('--in-image-folder', type=Path,
                        help='The original image folder.')
    parser.add_argument('--dataset-part', type=Path,
                        help='Save folder of new labels. Labels will be saved into <out-folder>/labels.')
    parser.add_argument('--out-folder', type=Path,
                        help='Save folder of images and labels. Will be saved into <out-folder>/dataset-part.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset, to split, either shipnet or dior')
    parser.add_argument('--log-config-path', default=project_folder /
                                                     Path('configs/log.config'), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, help='Log file path.')
    args = parser.parse_args()
    return args


def run(args):
    in_label_folder = Path(args.in_label_folder)
    in_image_folder = Path(args.in_image_folder)
    dataset_part = Path(args.dataset_part)
    out_folder = Path(args.out_folder)
    dataset = args.dataset

    assert create_folder(out_folder)

    assert dataset == 'dior' or dataset == 'shipnet'

    log_path = project_folder / f'separate_shipnet_data.log' if args.log_path is None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger('')
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')

    separate_dataset_parts(out_folder, in_label_folder, in_image_folder, dataset_part, dataset)


if __name__ == '__main__':
    args = get_args()
    run(args)
