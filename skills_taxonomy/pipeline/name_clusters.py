from skills_taxonomy import PROJECT_DIR
from pprint import pprint
from skills_taxonomy.utils.json_management import load_json, save_json


def input_name(class_id, informative_words):
    """Print most informative words, and then ask for
    user input to assign a suitable name.

    Args:
        class_id(str): id for class assigned after clustiner
        informative_words (list of str): most informative words

    Returns:
        str: user input name for class
    """
    pprint(informative_words)
    print(f"Using the most informative words for class {class_id} above,")
    return input(f"enter name for class {class_id}:")


def name_clusters(informative_words_path):
    """Iterate through json file of class_ids and most informative words,
    printing info to user and asking for user input to name the classes.
    Return a dictionary of the user given names.

    Args:
        informative_words_path (str): path to json file of most
                                      informative words

    Returns:
        dict: dict with keys of class ids and values of user
              assigned class name
    """
    informative_words = load_json(informative_words_path)
    named_clusters = {}
    for k, v in informative_words.items():
        named_clusters[k] = input_name(k, v)
    return named_clusters


def save_named_clusters(save_dir=f"{PROJECT_DIR}/outputs/named_classes/"):
    """Run name_clusters on most_informative_words/class.json and
    most_informative_words/subclass.json and save json file.

    Args:
        save_dir ([type], optional): [description]. Defaults to f"{PROJECT_DIR}/outputs/named_classes/".
    """
    save_json(
        save_dir,
        file_name="named_classes.json",
        json_to_save=name_clusters(
            f"{PROJECT_DIR}/outputs/most_informative_words/class.json"
        ),
    )
    save_json(
        save_dir,
        file_name="named_subclasses.json",
        json_to_save=name_clusters(
            f"{PROJECT_DIR}/outputs/most_informative_words/subclass.json"
        ),
    )
