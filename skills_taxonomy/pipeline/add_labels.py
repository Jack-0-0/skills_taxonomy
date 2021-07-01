from skills_taxonomy.utils.json_management import load_json
from skills_taxonomy import PROJECT_DIR


def map_lbls_id(skills, label, id, lookup):
    """Map the skills labels to the skills ids

    Args:
        skills (df): skills dataframe
        label (str): col to create, either 'class_lbl' or 'subclass_lbl'
        id (str): col to use for mapping, either 'class_id' or 'subclass_id'
        lookup (dict): to use for mapping - keys of ids, values of labels

    Returns:
        df: skills dataframe with additional class label column
    """
    skills[label] = skills[id].astype(str).map(lookup)
    if label == "class_id":
        skills[id] = skills[id].astype(int)
    if label == "subclass_id":
        skills[id] = skills[id].astype(float)
    return skills


def add_labels(skills):
    """Add 'class_lbl' and 'subclass_lbl' cols to skills dataframe

    Args:
        skills (df): skills dataframe

    Returns:
        df: skills dataframe with additional 'class_lbl' and 'subclass_lbl' cols
    """
    map_lbls_id(
        skills,
        label="class_lbl",
        id="class_id",
        lookup=load_json(f"{PROJECT_DIR}/outputs/named_classes/named_classes_x.json"),
    )
    return map_lbls_id(
        skills,
        label="subclass_lbl",
        id="subclass_id",
        lookup=load_json(
            f"{PROJECT_DIR}/outputs/named_classes/named_subclasses_x.json"
        ),
    )
