from skills_taxonomy.getters.esco_skills import get_output_skills
from skills_taxonomy.pipeline.embedding import load_embedding
from skills_taxonomy.utils.dir_management import make_dir_if_not_exist
import os
import random
from sentence_transformers import util
from skills_taxonomy import PROJECT_DIR


def compare_skill(embedding, idx=None):
    """Display a skill its most similar skills in the embedding.

    Args:
        embedding (array): skills embedding
        idx (int): index to select skill,
            defaults to None (if None, a random index is chosen)

    Returns:
        df: dataframe of a skill and the skills it is closest
            to in the embedding by cosine similarity
    """
    if idx is None:
        description = embedding[random.randint(0, len(embedding))]
    else:
        description = embedding[idx]
    return (
        skills[["preferredLabel", "description"]]
        .assign(cosine_scores=util.pytorch_cos_sim(description, embedding)[0])
        .sort_values(by=["cosine_scores"], ascending=False)
        .head(10)
    )


def save_closest_skills(closest_skills):
    """Saves csv of skill and its closest skills in the skill embedding

    Args:
        closest_skills (df): dataframe of a skill and the skills it is closest
            to in the embedding by cosine similarity
    """
    skill_label = closest_skills["preferredLabel"].iloc[0].replace(" ", "_")
    save_path = (
        f"{PROJECT_DIR}/outputs/closest_skills/closest_skills_to_{skill_label}.csv"
    )
    make_dir_if_not_exist(os.path.dirname(save_path))
    closest_skills.to_csv(save_path)


if __name__ == "__main__":
    skills = get_output_skills()
    embedding = load_embedding()
    save_closest_skills(compare_skill(embedding, idx=None))
