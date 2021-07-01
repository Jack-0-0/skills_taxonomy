from skills_taxonomy.pipeline.preprocessing import preprocess_skills
from skills_taxonomy.pipeline.embedding import (
    create_embedding,
    save_embedding,
    load_embedding,
)
from skills_taxonomy.pipeline.clustering import cluster_add_ids
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

# cluster skills and add cluster and subcluster id cols to skills
skills = cluster_add_ids(skills, embedding)

# create informative words
save_informative_words()

# name clusters
save_named_clusters()

# add labels
add_labels(skills)

# save updated skills
skills.to_csv(f"{PROJECT_DIR}/outputs/skills/skills_after_labels2.csv")
