from skills_taxonomy.getters.esco_skills import get_output_skills
from skills_taxonomy.utils.json_management import load_json, convert_dict_key_type
from skills_taxonomy.utils.dir_management import make_dir_if_not_exist
import os
from treelib import Tree
import numpy as np
from skills_taxonomy import PROJECT_DIR


def counts(skills, id):
    """Create dictionary with class or subclass id as keys and
    counts of skills within that class or subclass as values

    Args:
        skills (df): skills dataframe
        id (str): choose 'class_id' or 'subclass_id'

    Returns:
        dict: keys = class or subclass id,
            values = counts of skills within that class or subclass
    """
    ids, counts = np.unique(skills[id], return_counts=True)
    return dict(zip(ids, counts))


def add_class_nodes(cls_lbls, cls_counts):
    """Add nodes to tree for class level

    Args:
        cls_lbls (dict): keys of class id, values of class label
        cls_counts (dict): keys of class id, values of count
    """
    for cls in cls_lbls.keys():
        tree.create_node(
            f"{cls}: {cls_lbls[cls]}({cls_counts[cls]})",
            cls_lbls[cls],
            parent="skills_taxonomy",
        )


def add_subclass_nodes(subcls_lbls, subcls_counts, cls_lbls):
    """Add nodes to tree for subclass level

    Args:
        subcls_lbls (dict): keys of subclass id, values of subclass label
        subcls_counts (dict): keys of subclass id, values of count
        cls_lbls (dict): keys of class id, values of class label
    """
    for sub_cls in subcls_lbls.keys():
        tree.create_node(
            f"{sub_cls}: {subcls_lbls[sub_cls]}({subcls_counts[sub_cls]})",
            subcls_lbls[sub_cls],
            parent=cls_lbls[int(sub_cls)],
        )


def save_tree(save_path=f"{PROJECT_DIR}/outputs/tree/tree.txt"):
    """Save tree if dir for save_path exists, if not make dir"""
    make_dir_if_not_exist(os.path.dirname(save_path))
    tree.save2file(save_path)


def make_skills_taxonomy_tree():
    """Add class and subclass nodes to the tree"""
    tree.create_node(f"skills_taxonomy({len(skills)})", "skills_taxonomy")
    # add class nodes
    cls_lbls = convert_dict_key_type(
        load_json(f"{PROJECT_DIR}/outputs/named_classes/named_classes_x.json"), type=int
    )
    add_class_nodes(cls_lbls, cls_counts=counts(skills, id="class_id"))
    # add sub class nodes
    add_subclass_nodes(
        subcls_lbls=convert_dict_key_type(
            load_json(f"{PROJECT_DIR}/outputs/named_classes/named_subclasses_x.json"),
            type=float,
        ),
        subcls_counts=counts(skills, id="subclass_id"),
        cls_lbls=cls_lbls,
    )


if __name__ == "__main__":
    skills = get_output_skills()
    tree = Tree()
    make_skills_taxonomy_tree()
    save_tree()
