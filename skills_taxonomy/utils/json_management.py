import json
from skills_taxonomy.utils.dir_management import make_dir_if_not_exist


def load_json(json_path):
    """Load json file as dictionary

    Args:
        json (str): path to json file

    Returns:
        dict: dictionary representation of json file
    """
    with open(json_path, "r") as file:
        return json.load(file)


def convert_dict_key_type(dict, type):
    """Convert keys in dictionary to ints
    (loading json loads keys as strs)"""
    return {type(k): v for k, v in dict.items()}


def save_json(save_dir, file_name, json_to_save):
    """Save dictionary as json

    Args:
        save_dir (str): path to directory to save json
        file_name (str): file name to save json
        json_to_save(json): file to save
    """
    make_dir_if_not_exist(save_dir)
    with open(f"{save_dir}{file_name}", "w") as fp:
        json.dump(
            json_to_save,
            fp,
            sort_keys=True,
            indent=4,
        )
