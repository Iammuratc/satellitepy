import os


def get_project_folder():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_file_name_from_path(path):
    return os.path.splitext(os.path.split(path)[-1])[0]




