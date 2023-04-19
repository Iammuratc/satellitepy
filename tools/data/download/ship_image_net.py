import configargparse
from pathlib import Path
from satellitepy.utils.path_utils import init_logger, unzip_files_in_folder
import logging
import wget
"""
Automatically downloads ShipImageNet Dataset and saves it in given folder.
If needed can also extract downloaded zip directly
"""

DATASET_NAME = "ship_image_net"
DATASET_URL="https://drive.google.com/u/0/uc?id=1wApkaSoa9mXRfXQiq6lTtlVrv4cSc6vv&export=download&confirm=t&uuid=fdda2666-9f14-49dc-aa32-c30ade4dec2b&at=ANzk5s7sHOt1Wse-QKKJXk39p54n:1681843273234"

def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-folder', type=Path, required=True,
                        help='Folder where the dataset should be downloaded to.')
    parser.add_argument('--log-config', type=Path, required=True,
                        help='Path to log config file.')
    parser.add_argument('--log-path', type=Path, required=False,
                        help='Where the log config should be saved.')
    parser.add_argument('--unzip', type=bool, required=True, help="If datasets with zipped files should be automatically unzipped")
    args = parser.parse_args()
    return args

def run(args):
    in_folder = Path(args.in_folder) / Path(DATASET_NAME)
    log_config = Path(args.log_config)
    unzip = bool(args.unzip)

    in_folder.mkdir(parents=True, exist_ok=True)
    log_path = Path(in_folder) / "download.log" if args.log_path == None else args.log_path
    init_logger(config_path=log_config, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info(f'Using path: {log_path}')
    logger.info('Start downloading...')

    logger.info('Downloading Dataset...')
    wget.download(DATASET_URL, out = str(in_folder))
    logger.info('Finished downloading ' + DATASET_NAME)

    if unzip:
        logger.info('Start unzipping')
        unzip_files_in_folder(in_folder)
        logger.info('Finished unzipping')

if __name__ == '__main__':
    args = get_args()
    run(args)