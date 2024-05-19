import configargparse
from pathlib import Path
from satellitepy.utils.path_utils import init_logger, unzip_files_in_folder, create_folder
import logging
import wget

"""
This script will download real and synthetic part of OnlyPlanes at the same time
If needed can also extract downloaded zip directly
"""

DATASET_NAME = 'only_planes'
DATASET_URL = 'https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/OnlyPlanes_dataset_08122022.zip'
SYNTHATIC_URL = 'https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/final_aug22'


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-folder', type=Path, required=True,
                        help='Folder where the dataset should be downloaded to.')
    parser.add_argument('--log-config', type=Path, required=True,
                        help='Path to log config file.')
    parser.add_argument('--log-path', type=Path, required=False,
                        help='Where the log config should be saved.')
    parser.add_argument('--unzip', type=bool, required=True,
                        help='If datasets with zipped files should be automatically unzipped')
    args = parser.parse_args()
    return args


def run(args):
    in_folder = Path(args.in_folder) / Path(DATASET_NAME + '_real')
    log_config = Path(args.log_config)
    unzip = bool(args.unzip)
    synthetic = Path(args.in_folder) / Path(DATASET_NAME + '_synthetic')

    create_folder(in_folder)
    create_folder(synthetic)

    log_path = Path(in_folder) / 'download.log' if args.log_path is None else args.log_path
    init_logger(config_path=log_config, log_path=log_path)
    logger = logging.getLogger('')
    logger.info(f'Using path: {log_path}')
    logger.info('Start downloading...')

    logger.info('Downloading Synthetic...')
    logger.info('Downloading 1/4')
    wget.download(SYNTHATIC_URL + '/onlyplanes_faster_rcnn_r50-config.yaml',
                  out=str(synthetic))  # binary object detection config
    logger.info('Downloading 2/4')
    wget.download(SYNTHATIC_URL + '/onlyplanes_mask_rcnn_r50-config.yaml',
                  out=str(synthetic))  # instance segmentation config
    logger.info('Downloading 3/4')
    wget.download(SYNTHATIC_URL + '/onlyplanes_mask_rcnn_r50-0024999.pth',
                  out=str(synthetic))  # instance segmentation model
    logger.info('Downloading 4/4')
    wget.download(SYNTHATIC_URL + '/onlyplanes_faster_rcnn_r50-0034999.pth',
                  out=str(synthetic))  # binary object detection model
    logger.info('Finished Downloading Synthetic')

    logger.info('Downloading Dataset...')
    wget.download(DATASET_URL, out=str(in_folder))
    logger.info('Finished downloading ' + DATASET_NAME)

    if unzip:
        logger.info('Start unzipping')
        unzip_files_in_folder(in_folder)
        logger.info('Finished unzipping')


if __name__ == '__main__':
    args = get_args()
    run(args)
