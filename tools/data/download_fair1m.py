import configargparse
from pathlib import Path
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
import logging

"""
Download the Fair1m Dataset
"""

def get_args():
    """Argument Parser"""
    project_folder = get_project_folder()
    parser = configargparse.ArgumentParser(description = __doc__)
    parser.add_argument('--download', required = True, type = str,
                        help = 'Which dataset to download. Type fair1m for fair1m dataset')
    parser.add_argument('--in-folder', default = project_folder/Path('in_folder/fair1m/'), type = Path,
                        help = 'Which folder to save the dataset into')
    parser.add_argument('--log-config-path', default = project_folder/Path("configs/log.config"),
                        type=Path, help='Log config file.')
    parser.add_argument('--log-path',  type=Path, help='Log file.')
    args = parser.parse_args()
    return args

def main(args):
    if args.download == 'fair1m':
        parent_folder = Path(args.in_folder)
        assert create_folder(parent_folder)

        print("#########################################################################\n",
            "It is currently not possible to download the Fair1m dataset automated.\n",
            "To download the dataset go to https://www.gaofen-challenge.com/benchmark\n",
            "You have to log in to download the dataset\n",
            "#########################################################################")

        # Init logger
        log_path = Path(
            parent_folder) / 'download_fair1m.log' if args.log_path == None else args.log_path
        init_logger(config_path=args.log_config_path, log_path=log_path)
        logger = logging.getLogger(__name__)
        if args.log_path == None:
            logger.info(
                    f'No log path is given, the default log path will be used: {log_path}')
        logger.info('Info on how to download the dataset was given')

if __name__ == '__main__':
    args = get_args()
    main(args)