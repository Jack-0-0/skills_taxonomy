# File: pipeline/embedding.py
"""Create embedding using skills descriptions
"""

from sentence_transformers import SentenceTransformer
from skills_taxonomy import config


def create_embedding(skills):
    """Create embedding using sentence transformer

    Returns:
        embedding: skill descriptions embedding
    """
    corpus = list(skills["description"].values)
    embedder = SentenceTransformer(config["sentence_transformer"]["model"])
    embedding = embedder.encode(corpus, show_progress_bar=True)
    return embedding
