import os


def make_dir_if_not_exist(path):
    """Make dir if it does not already exist

    Args:
        path (str): path to check if exists and make if does not exist
    """
    if not os.path.exists(path):
        os.makedirs(path)
