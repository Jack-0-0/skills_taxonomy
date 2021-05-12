# File: pipeline/embedding.py
"""Functions to create, load and save skills embedding
"""

from sentence_transformers import SentenceTransformer
from skills_taxonomy import config
import pickle
import logging
from skills_taxonomy import PROJECT_DIR

logger = logging.getLogger(__name__)


def create_embedding(skills):
    """Create embedding using sentence transformer

    Returns:
        embedding: skill descriptions embedding
    """
    corpus = list(skills["description"].values)
    embedder = SentenceTransformer(config["sentence_transformer"]["model"])
    embedding = embedder.encode(corpus, show_progress_bar=True)
    return embedding


def save_embedding(embedding):
    """Save embedding as pickle file

    Args:
        embedding: skills embedding to be saved as pickle file
    """
    with open(f"{PROJECT_DIR}/outputs/models/embedding.pkl", "wb") as out:
        pickle.dump(embedding, out)


def load_embedding(path=f"{PROJECT_DIR}/outputs/models/embedding.pkl"):
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
        logger.info(f"There is no embedding to load at {path}")
