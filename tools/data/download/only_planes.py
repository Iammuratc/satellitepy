import configargparse
from pathlib import Path
from satellitepy.data.tools import save_patches
from satellitepy.utils.path_utils import create_folder, init_logger, get_project_folder
from shutil import unpack_archive
import logging
import wget

DATASET_NAME = "only_planes"
DATASET="https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/OnlyPlanes_dataset_08122022.zip"
SRC="https://msdsdiag.blob.core.windows.net/naivelogicblob/OnlyPlanes/final_aug22"

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
    in_folder = Path(args.in_folder) / Path(DATASET_NAME + "_real")
    log_config = Path(args.log_config)
    unzip = bool(args.unzip)
    synthatic = Path(args.in_folder) / Path(DATASET_NAME + "_synthatic")

    in_folder.mkdir(parents=True, exist_ok=True)
    synthatic.mkdir(parents=True, exist_ok=True)

    log_path = Path(in_folder) / "download.log" if args.log_path == None else args.log_path
    init_logger(config_path=log_config, log_path=log_path)
    logger = logging.getLogger(__name__)
    logger.info(f'Using path: {log_path}')
    logger.info('Start downloading...')

    logger.info('Downloading Synthatic...')
    logger.info('Downloading 1/4')
    wget.download(SRC + "/onlyplanes_faster_rcnn_r50-0034999.pth", out = synthatic.resolve()) # binary object detection model
    logger.info('Downloading 2/4')
    wget.download(SRC + "/onlyplanes_faster_rcnn_r50-config.yaml", out = synthatic.resolve()) # binary object detection config
    logger.info('Downloading 3/4')
    wget.download(SRC + "/onlyplanes_mask_rcnn_r50-0024999.pth", out = synthatic.resolve()) # instance segmentation model
    logger.info('Downloading 4/4')
    wget.download(SRC + "/onlyplanes_mask_rcnn_r50-config.yaml", out = synthatic.resolve()) # instance segmentation config
    logger.info('Finished Downloading Synthatic')

    logger.info('Downloading Dataset...')
    wget.download(DATASET, out = in_folder.resolve())

    if unzip:
        zip_files = in_folder.rglob("*.zip")
        while True:
            try:
                path = next(zip_files)
            except StopIteration:
                break # no more files
            except PermissionError:
                logging.exception("Permission error! Cant open file")
            else:
                extract_dir = path.with_name(path.stem)
                logging.info("Unzipping " + str(path))
                unpack_archive(str(path), str(extract_dir), 'zip')  

    logger.info('Finished downloading ' + DATASET_NAME)

if __name__ == '__main__':
    args = get_args()
    run(args)