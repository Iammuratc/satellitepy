import configargparse
from pathlib import Path

from satellitepy.data.tools import save_xview_in_satellitepy_format
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
import logging

project_folder = get_project_folder()


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-label-file', type=Path,
                        help='The original label file. Will be split up for every image.')
    parser.add_argument('--out-folder', type=Path,
                        help='Save folder of new labels. Labels will be saved into <out-folder>/labels.')
    parser.add_argument('--log-config-path', default=project_folder /
                                                     Path("configs/log.config"), type=Path, help='Log config file.')
    parser.add_argument('--log-path', type=Path, help='Log file path.')
    args = parser.parse_args()
    return args


def run(args):
    in_label_file = Path(args.in_label_file)
    out_folder = Path(args.out_folder)

    assert create_folder(out_folder)

    log_path = project_folder / f'save_xview_in_satellitepy_format.log' if args.log_path is None else args.log_path
    init_logger(config_path=args.log_config_path, log_path=log_path)
    logger = logging.getLogger('')
    logger.info(
        f'No log path is given, the default log path will be used: {log_path}')
    logger.info('Saving satellitepy labels')

    save_xview_in_satellitepy_format(out_folder, in_label_file)


if __name__ == '__main__':
    args = get_args()
    run(args)
