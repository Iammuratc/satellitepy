import urllib.request
import patoolib
import logging
import configargparse
import os
from pathlib import Path
from satellitepy.utils.path_utils import init_logger, create_folder

DATASET_NAME = 'VEDAI'
DATASET_URL = 'https://downloads.greyc.fr/vedai/'


def get_args():
    """Arguments parser."""
    parser = configargparse.ArgumentParser(description=__doc__)
    parser.add_argument('--in-folder', type=Path, required=True,
                        help='Folder where the dataset should be downloaded to.')
    parser.add_argument('--log-config', type=Path, required=True, help='Whether to write to log to file.')
    parser.add_argument('--log-path', type=Path, required=False, help='Where the log config should be saved.')
    parser.add_argument('--unzip', type=bool, required=True, help='If datasets with compressed files should be '
                                                                  'automatically unzipped')
    args = parser.parse_args()
    return args


def download_VEDAI(in_path, log_config, log_path=None, unzip=True):
    in_path = str(Path(in_path) / DATASET_NAME)
    if not (Path(in_path).is_dir()):
        create_folder(Path(in_path))
    if log_config:
        if log_path is None:
            log_path = str(Path(in_path) / 'download.log')
        init_logger('configs/log.config', log_path)
    try:
        files1024 = []
        files512 = []
        logging.info('Attempting to download files from server. This might take a while.')
        annotation512 = \
            urllib.request.urlretrieve(DATASET_URL + 'Annotations512.tar', str(Path(in_path) / 'Annotations512.tar'))[
                0]
        annotations1024 = \
            urllib.request.urlretrieve(DATASET_URL + 'Annotations1024.tar', str(Path(in_path) / 'Annotations1024.tar'))[
                0]
        files512.append(urllib.request.urlretrieve(DATASET_URL + 'Vehicules512.tar.001',
                                                   str(Path(in_path) / 'Vehicules512.tar.001'))[
                            0])
        files512.append(urllib.request.urlretrieve(DATASET_URL + 'Vehicules512.tar.002',
                                                   str(Path(in_path) / 'Vehicules512.tar.002'))[
                            0])
        files1024.append(urllib.request.urlretrieve(DATASET_URL + 'Vehicules1024.tar.001',
                                                    str(Path(in_path) / 'Vehicules1024.tar.001'))[
                             0])
        files1024.append(urllib.request.urlretrieve(DATASET_URL + 'Vehicules1024.tar.002',
                                                    str(Path(in_path) / 'Vehicules1024.tar.002'))[
                             0])
        files1024.append(urllib.request.urlretrieve(DATASET_URL + 'Vehicules1024.tar.003',
                                                    str(Path(in_path) / 'Vehicules1024.tar.003'))[
                             0])
        files1024.append(urllib.request.urlretrieve(DATASET_URL + 'Vehicules1024.tar.004',
                                                    str(Path(in_path) / 'Vehicules1024.tar.004'))[
                             0])
        files1024.append(urllib.request.urlretrieve(DATASET_URL + 'Vehicules1024.tar.005',
                                                    str(Path(in_path) / 'Vehicules1024.tar.005'))[
                             0])
        devkit = urllib.request.urlretrieve(DATASET_URL + 'DevKit.tar', str(Path(in_path) / 'DevKit.tar'))[0]
        tos = urllib.request.urlretrieve(DATASET_URL + 'TermsandConditionsofUseVeDAI2014.pdf',
                                         str(Path(in_path) / 'TermsandConditionsofUseVeDAI2014.pdf'))[0]
    except Exception as e:
        if logging:
            logging.error('Failed to download dataset with the following error:\n' + str(e))
    else:
        if logging:
            logging.info('Download complete.')
        if logging:
            logging.info('Merging files.')
        try:
            file512 = os.path.join(in_path, 'Vehicules512.tar')
            file1024 = os.path.join(in_path, 'Vehicules1024.tar')
            for part in files512:
                with open(file512, 'ab') as whole, open(part, 'rb') as fragment:
                    whole.write(fragment.read())
                os.remove(part)
            for part in files1024:
                with open(file1024, 'ab') as whole, open(part, 'rb') as fragment:
                    whole.write(fragment.read())
                os.remove(part)
        except Exception:
            if logging:
                logging.error('Failed to merge files.')
        if unzip:
            try:
                patoolib.extract_archive(annotation512, outdir=in_path)
                patoolib.extract_archive(annotations1024, outdir=in_path)
                patoolib.extract_archive(devkit, outdir=in_path)
                patoolib.extract_archive(file512, outdir=in_path)
                patoolib.extract_archive(file1024, outdir=in_path)
            except patoolib.util.PatoolError as e:
                if logging:
                    logging.warning(
                        'There has been an error when attempting to extract tape archive(s):\n' + str(e))
            else:
                if logging:
                    logging.info('Archive extracted.')


if __name__ == '__main__':
    args = get_args()
    download_VEDAI(args.in_folder, args.log_config, args.log_path, args.unzip)
