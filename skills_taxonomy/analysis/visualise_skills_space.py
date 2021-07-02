from skills_taxonomy.getters.esco_skills import get_output_skills
from skills_taxonomy.pipeline.embedding import load_embedding
from skills_taxonomy.utils.dir_management import make_dir_if_not_exist
import os.path
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from skills_taxonomy import PROJECT_DIR


def reduce_dims(embedding):
    """Using TSNE, reduce dimensions of skills embedding

    Args:
        embedding (array): skills embedding with 768 dimensions

    Returns:
        array: skills embedding with 2 dimensions
    """
    return TSNE(n_components=2).fit_transform(embedding)


def add_points_to_skills(skills, embedding_2d):
    """Add x and y points from 2 dimensional skills embedding
    to skills dataframe

    Args:
        skills (df): skills dataframe
        embedding_2d (array): skills embedding with 2 dimensions

    Returns:
        df: skills dataframe with additional columns for x and y
    """
    return skills.assign(x=embedding_2d[:, 0], y=embedding_2d[:, 1])


def plot_skills_space(
    skills, save_path=f"{PROJECT_DIR}/outputs/figures/skills_space.jpeg"
):
    """Save two plots of the skills space, one colouring the points by
    class, one colouring the points by subclass

    Args:
        skills (df): skills dataframe
        save_path (str): path to save the plot,
            defaults to f"{PROJECT_DIR}/outputs/figures/skills_space.jpeg"
    """
    plt.subplots(figsize=(30, 15))
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        data=skills, x="x", y="y", hue="class_lbl", palette="tab10", s=10, legend="full"
    )
    plt.subplot(1, 2, 2)
    sns.scatterplot(
        data=skills,
        x="x",
        y="y",
        hue="subclass_lbl",
        palette="husl",
        s=10,
        legend="full",
    )
    plt.legend(
        bbox_to_anchor=(1.01, 0.465), loc=2, borderaxespad=0.0, title="subclass_lbl"
    )
    make_dir_if_not_exist(os.path.dirname(save_path))
    plt.savefig(save_path)


def plot_subclass_skills_space(
    skills,
    nrows=3,
    ncols=3,
    save_path=f"{PROJECT_DIR}/outputs/figures/subclass_skills_space.jpeg",
):
    """Save figure with multiple sub plots for the skills space
    for each subclass

    Args:
        skills (df): skills dataframe
        nrows (int): number of rows in the subplot, defaults to 3.
        ncols (int): number of columns in the subplot, defaults to 3.
        save_path (str): path to save the plot, defaults
                to f"{PROJECT_DIR}/outputs/figures/subclass_skills_space.jpeg".
    """
    plt.subplots(figsize=(18, 18))
    for i in np.unique(skills["class_id"]):
        plt.subplot(nrows, ncols, i + 1)
        sub_points = skills[
            (skills["subclass_id"] < i + 1) & (skills["subclass_id"] >= i)
        ]
        sns.scatterplot(
            data=sub_points,
            x="x",
            y="y",
            hue="subclass_lbl",
            palette="tab10",
            s=10,
            legend="full",
        )
    make_dir_if_not_exist(os.path.dirname(save_path))
    plt.savefig(save_path)


if __name__ == "__main__":
    skills = get_output_skills()
    embedding = load_embedding()
    embedding_2d = reduce_dims(embedding)
    skills = add_points_to_skills(skills, embedding_2d)
    plot_skills_space(skills)
    plot_subclass_skills_space(skills)
