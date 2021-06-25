"""Functions to cluster the embedding to find meaningful groups of skills.

Agglomerative clustering is used to find skill classes and sub classes. The parameter distance_threshold influences the size of the clusters produced. A higher distance threshold results in fewer clusters.

Firstly, agglomerative clustering is performed on the whole dataset to find high level skill classes. Then, agglomerative clustering is performed on each of these classes to create sub classes.
"""
import numpy as np
from skills_taxonomy import config
from sklearn.cluster import AgglomerativeClustering


def normalise_embedding(embedding):
    """Normalise embedding"""
    return embedding / np.linalg.norm(embedding, axis=1, keepdims=True)


def cluster(
    embedding,
    n_clusters=config["clustering"]["n_clusters"],
    distance_threshold=config["clustering"]["distance_threshold"],
    affinity=config["clustering"]["affinity"],
    linkage=config["clustering"]["linkage"],
):
    """Creates, fits and returns a agglomerative clustering model

    Args:
        embedding (np.array): skill descriptions embedding
        n_clusters (int or None, optional): number of clusters to find.
        distance_threshold (int, optional): linkage distance threshold above which,
                                        clusters will not be merged.
        affinity (str, optional): metric used to compute the linkage.
        linkage (str, optional): linkage criteria to use.
    """
    clustering_model = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        affinity=affinity,
        linkage=linkage,
    )
    return clustering_model.fit(embedding)


def find_indices(cluster_assignment, class_id):
    """Find class indices"""
    indices = np.argwhere(cluster_assignment == class_id)
    return np.squeeze(indices)


def find_sub_embeddings(embedding, indices):
    """Find embeddings for class"""
    return np.asarray([embedding[i] for i in indices])


def sub_cluster_assignment(sub_clustering_model, class_id):
    """Assign sub clusters"""
    return np.asarray([(class_id + c / 10) for c in sub_clustering_model.labels_])


def full_sub_cluster_assignment(
    embedding,
    cluster_assignments,
    num_clusters,
    distances=config["clustering"]["sub_clustering_distances"],
):
    """Assign subclusters to all skills

    Args:
        embedding (np.array): skill descriptions embedding
        cluster_assignment (np.array): [description]
        num_clusters (int): [description]
        distances (list of ints, optional): linkage distance thresholds for each subcluster
                                        above which, clusters will not be merged.

    Returns:
        (np.array): subcluster assignments for all skills
    """
    full_sub_cluster_assignments = np.empty(shape=(len(embedding),))
    for class_id in range(num_clusters):
        indices = find_indices(cluster_assignments, class_id)
        sub_embeddings = find_sub_embeddings(embedding, indices)
        sub_clustering_model = cluster(
            sub_embeddings, distance_threshold=distances[class_id]
        )
        sub_cluster_assignments = sub_cluster_assignment(sub_clustering_model, class_id)
        for i, a in zip(indices, sub_cluster_assignments):
            full_sub_cluster_assignments[i] = a
    return full_sub_cluster_assignments
