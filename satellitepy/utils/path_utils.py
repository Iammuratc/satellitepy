import logging
from pathlib import Path
import logging.config


def init_logger(config_path, log_path):
    """
    Initate the logger. This function is mostly called from the tools scripts.
    Parameters
    ----------
    config_path : Path
        Configuration file for logging. There should be a sample at configs/log.config
    log_path : Path
        Write your log messages in this file.
    Returns
    -------
    Initiate the log file at log_path using config_path
    """
    logging.config.fileConfig(config_path, defaults={'logfilename': log_path})

def create_folder(folder):
    """
    Create the given folder
    Parameters
    ----------
    folder : Path
        This folder path will be created if the user confirms it.
    Returns
    -------
    bool
    True if folder is confirmed to be created or if it already exists. Else, False.
    """
    if not folder.exists():
        msg = f'The following folder will be created:\n{folder}\nDo you confirm?[y/n] '
        ans = input(msg)
        if ans == 'y':
            Path(folder).mkdir(parents=True, exist_ok=True)
            return 1
        raise AssertionError('Please confirm it.\n')
    else:
        return 1

def get_file_paths(folder,sort=True):
    """
    Get file paths in a folder.
    Parameters
    ---------
    folder : Path
        The absolute file paths in this folder will return.
    sort : bool
        if True, Sort file paths. E.g., sort is necessary to match image and label files from two folders.
    Returns
    -------
    file_paths : list of Path
        Absolute file paths in folder
    """
    file_paths = [Path(file_path) for file_path in folder.glob('**/*') if file_path.is_file()]
    if sort:
        file_paths.sort()
    return file_paths

def is_file_names_match(*file_paths):
    """
    Check if file names (without extensions) match
    Parameters
    ----------
    file_paths : list of Path
        Absolute file paths or file names
    Returns
    -------
    bool
    True if file names match
    """
    file_names = []
    for file_path in file_paths:
        file_names.append(file_path.stem)
    return file_names.count(file_names[0]) == len(file_names)

def zip_matched_files(*folders):
    """
    Yield matching file paths from folders
    Parameters
    ---------
    folders : list of Path
        File paths will be generated from folders
    Returns
    -------
    file_paths : list of Path
        Matched file paths
    """
    logger = logging.getLogger(__name__)

    all_file_paths = [get_file_paths(folder) for folder in folders]
    all_file_paths_zipped = zip(*all_file_paths)
    for file_paths in all_file_paths_zipped:
        file_name = file_paths[0].stem
        logger.info(f'{file_name} will be processed...')
        # Check if file names match
        is_match = is_file_names_match(*file_paths)
        if not is_match:
            logger.error('File names do not match!')
            exit(1)
        yield file_paths