from skills_taxonomy.pipeline.preprocessing import preprocess_skills
from skills_taxonomy.pipeline.embedding import (
    create_embedding,
    save_embedding,
    load_embedding,
)
from skills_taxonomy.pipeline.clustering import (
    normalise_embedding,
    cluster,
    full_sub_cluster_assignment,
)
from skills_taxonomy.pipeline.informative_words import save_informative_words
from skills_taxonomy.pipeline.name_clusters import save_named_clusters
from skills_taxonomy.pipeline.add_labels import add_labels
from skills_taxonomy import PROJECT_DIR

# load skills
skills = preprocess_skills()

# create embedding
embedding = create_embedding(skills)
save_embedding(embedding)
embedding = load_embedding()

# cluster skills
embedding = normalise_embedding(embedding)
clustering_model = cluster(embedding)
cluster_assignments = clustering_model.labels_
num_clusters = clustering_model.n_clusters_
full_sub_cluster_assignments = full_sub_cluster_assignment(
    embedding, cluster_assignments, num_clusters
)
skills = skills.assign(
    class_id=cluster_assignments, subclass_id=full_sub_cluster_assignments
)

# import pandas as pd
# skills = pd.read_csv(
#     f"{PROJECT_DIR}/outputs/skills/skills_after_ids.csv", index_col=[0]
# )

# create informative words
save_informative_words()

# name clusters
save_named_clusters()

# add labels
add_labels(skills)

# save updated skills
skills.to_csv(f"{PROJECT_DIR}/outputs/skills/skills_after_labels.csv")
