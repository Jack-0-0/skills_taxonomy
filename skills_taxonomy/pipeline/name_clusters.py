from skills_taxonomy import PROJECT_DIR
from pprint import pprint
from skills_taxonomy.utils.json_management import load_json, save_json


def input_name(cluster, informative_words):
    pprint(informative_words)
    print(f"Using the most informative words for cluster {cluster} above,")
    name = input(f"enter name for cluster {cluster}:")
    return name


def name_clusters(informative_words_path):
    informative_words = load_json(informative_words_path)
    named_clusters = {}
    for k, v in informative_words.items():
        named_clusters[k] = input_name(k, v)
    return named_clusters


def save_named_clusters(save_dir=f"{PROJECT_DIR}/outputs/named_clusters/"):
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


# save_named_clusters()
print(load_json(f"{PROJECT_DIR}/outputs/named_clusters/named_subclasses.json"))
