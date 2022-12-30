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


# def get_project_folder():
#     project_folder = os.path.dirname(
#         os.path.dirname(
#             os.path.dirname(
#                 os.path.abspath(__file__))))
#     assert create_folder(project_folder)
#     return project_folder

# alternative project folder
def get_project_folder():
    project_folder = os.path.abspath('F:/working')
    assert create_folder(project_folder)
    return project_folder
