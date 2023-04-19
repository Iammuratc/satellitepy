import configargparse
from pathlib import Path
from satellitepy.utils.path_utils import init_logger, unzip_files_in_folder
from shutil import rmtree
from torrentp import TorrentDownloader
import logging
import wget
"""
Automatically downloads UCAS-AOD Dataset and saves it in given folder.
If needed can also extract downloaded zip directly
"""

DATASET_NAME = "ucas_aod"
DATASET_URL="https://hyper.ai/tracker/download?torrent=6626"

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

    torrent_file = TorrentDownloader(str(in_folder)+"/UCAS-AOD.torrent", str(in_folder))
    torrent_file.start_download()
    logger.info('Finished downloading ' + DATASET_NAME)
    logger.info('Removing useless files')
    zip_path = in_folder / Path("UCAS-AOD/data/UCAS_AOD.zip")
    zip_path.rename(in_folder / "UCAS-AOD.zip")
    rmtree(in_folder / "UCAS-AOD")

    if unzip:
        logger.info('Start unzipping')
        unzip_files_in_folder(in_folder)
        logger.info('Finished unzipping')

if __name__ == '__main__':
    args = get_args()
    run(args)