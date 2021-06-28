"""Functions to create, load and save skills embedding
"""
from sentence_transformers import SentenceTransformer
import pickle
from skills_taxonomy import config, logger, PROJECT_DIR
from skills_taxonomy.utils.dir_management import make_dir_if_not_exist


def create_embedding(skills):
    """Create embedding using sentence transformer

    Returns:
        embedding: skill descriptions embedding
    """
    corpus = list(skills["description"].values)
    embedder = SentenceTransformer(config["sentence_transformer"]["model"])
    embedding = embedder.encode(corpus, show_progress_bar=True)
    return embedding


def save_embedding(embedding, save_path=PROJECT_DIR / "outputs/models/embedding.pkl"):
    """Save embedding as pickle file

    Args:
        embedding: skills embedding to be saved as pickle file
        save_path: path to save embedding
    """
    make_dir_if_not_exist(save_path.parent)
    with open(save_path, "wb") as out:
        pickle.dump(embedding, out)


def load_embedding(path=PROJECT_DIR / "outputs/models/embedding.pkl"):
    """Load embedding from pickle file

    Args:
        path: path to pickle file to load

    Returns:
        embedding: skills embedding loaded from pickle file
    """
    try:
        with open(path, "rb") as inp:
            embedding = pickle.load(inp)
        return embedding

    except FileNotFoundError:
        logger.error(f"There is no embedding to load at {path}")
