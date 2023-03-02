import os
import logging
import pathlib
import logging.config


def init_logger(config_path, log_path):
    logging.config.fileConfig(config_path, defaults={'logfilename': log_path})

def create_folder(folder):
    if not os.path.exists(folder):
        msg = f'The following folder will be created:\n{folder}\nDo you confirm?[y/n] '
        ans = input(msg)
        if ans == 'y':
            # os.makedirs(folder, exist_ok=True)
            pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
            return 1
        raise AssertionError('Please confirm it.\n')
    else:
        return 1

def get_project_folder(my_folder=None):
    """Set project folder
    Para
    my_folder : str
        If my_folder is assigned, then project_folder is my_folder, if not, it is the default project folder
    """
    # if not my_folder:

    project_folder = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
    # else:
    #     project_folder = my_folder
    assert create_folder(project_folder)
    return project_folder


def get_file_paths(folder,sort=True):
    file_paths = [os.path.join(folder,file_name) for file_name in os.listdir(folder)]
    if sort:
        file_paths.sort()
    return file_paths

def get_file_name_from_path(file_path):
    return os.path.splitext(os.path.split(file_path)[1])[0] 