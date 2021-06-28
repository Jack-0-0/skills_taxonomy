"""Functions to create, load and save skills embedding
"""
from sentence_transformers import SentenceTransformer
import pickle
from skills_taxonomy import config, logger, PROJECT_DIR
from skills_taxonomy.utils.dir_management import make_dir_if_not_exist


def create_embedding(skills):
    """Create embedding using sentence transformer

    Args:
        skills (df): skills dataframe containing column containing
                description of the skill

    Returns:
        array: skill descriptions embedding
    """
    corpus = list(skills["description"].values)
    embedder = SentenceTransformer(config["sentence_transformer"]["model"])
    embedding = embedder.encode(corpus, show_progress_bar=True)
    return embedding


def save_embedding(embedding, save_path=PROJECT_DIR / "outputs/models/embedding.pkl"):
    """Save skills embedding as pickle file

    Args:
        embedding (array): skills embedding to be saved as pickle file
        save_path (str, optional): path to save embedding,
                            defaults to PROJECT_DIR/"outputs/models/embedding.pkl"
    """

    make_dir_if_not_exist(save_path.parent)
    with open(save_path, "wb") as out:
        pickle.dump(embedding, out)


def load_embedding(path=PROJECT_DIR / "outputs/models/embedding.pkl"):
    """Load embedding from pickle file

    Args:
        path (str): path to pickle file to load

    Returns:
        array: skills embedding loaded from pickle file
    """
    try:
        with open(path, "rb") as inp:
            embedding = pickle.load(inp)
        return embedding

    except FileNotFoundError:
        logger.error(f"There is no embedding to load at {path}")
