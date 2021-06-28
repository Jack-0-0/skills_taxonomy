"""Functions to cluster the embedding to find meaningful groups of skills.

Agglomerative clustering is used to find skill classes and sub classes. The parameter distance_threshold influences the size of the clusters produced. A higher distance threshold results in fewer clusters.

Firstly, agglomerative clustering is performed on the whole dataset to find high level skill classes. Then, agglomerative clustering is performed on each of these classes to create sub classes.
"""
import numpy as np
from skills_taxonomy import config
from sklearn.cluster import AgglomerativeClustering


def normalise_embedding(embedding):
    """Normalise embedding to unit length

    Args:
        embedding (array): embedding to normalise

    Returns:
        array: normalised embedding
    """
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
        embedding (array): skills embedding
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
    """Find class indices

    Args:
        cluster_assignment (array): cluster assignments after clustering
        class_id (str): class identifier

    Returns:
        array: indices of skills that have cluster_assignment matching class_id
    """
    indices = np.argwhere(cluster_assignment == class_id)
    return np.squeeze(indices)


def find_sub_embeddings(embedding, indices):
    """Find sub embeddings for class

    Args:
        embedding (array): skills embedding
        indices (array): indices of skills to select

    Returns:
        array: sub emebedding containing skills that match provided indices
    """
    return np.asarray([embedding[i] for i in indices])


def sub_cluster_assignment(sub_clustering_model, class_id):
    """Assign sub clusters

    Args:
        sub_clustering_model (AgglomerativeClustering):  model that has been fit on sub cluster
        class_id (str)): class identifier

    Returns:
        array: sub cluster assignments, example: if assigned to class 1 and subclass 2,
            it would have sub cluster assignment of 1.2
    """
    return np.asarray([(class_id + c / 10) for c in sub_clustering_model.labels_])


def full_sub_cluster_assignment(
    embedding,
    cluster_assignments,
    num_clusters,
    distances=config["clustering"]["sub_clustering_distances"],
):
    """Assign subclusters to all skills

    Args:
        embedding (array): skill descriptions embedding
        cluster_assignment (array): clusters assigned to skills
        num_clusters (int): total number of clusters
        distances (list of ints, optional): linkage distance thresholds for each subcluster
                                        above which, clusters will not be merged.

    Returns:
        array: subcluster assignments for all skills
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
