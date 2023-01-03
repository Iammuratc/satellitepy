import os
import logging

def get_logger(name, file):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)

    fh = logging.FileHandler(filename=file)
    fh.setLevel(logging.DEBUG)

    # create formatter
    sh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh_formatter = logging.Formatter('%(message)s')

    # add formatter to ch
    sh.setFormatter(sh_formatter)
    fh.setFormatter(fh_formatter)

    # add ch to logger
    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger

def create_folder(folder):
    if not os.path.exists(folder):
        msg = f'The following folder will be created:\n{folder}\nDo you confirm?[y/n] '
        ans = input(msg)
        if ans == 'y':
            os.makedirs(folder, exist_ok=True)
            return 1
        raise AssertionError('Please confirm it.\n')
    else:
        return 1


def get_project_folder():
    if get_project_folder.folder != '':
        return get_project_folder.folder
    default_folder = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
    msg = f'Do you want to use the default project folder({default_folder})?[y/n] '
    ans = input(msg)
    if ans == 'y':
        project_folder = default_folder
    else:
        msg = f'Please specify the path of the project folder: '
        ans = input(msg)
        project_folder = os.path.abspath(ans)
    assert create_folder(project_folder)
    get_project_folder.folder = project_folder
    return project_folder


get_project_folder.folder = ''
