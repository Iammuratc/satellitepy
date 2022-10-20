import os


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
    project_folder = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
    assert create_folder(project_folder)
    return project_folder
