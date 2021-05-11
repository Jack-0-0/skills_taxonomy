# File: pipeline/preprocessing.py
"""Preprocessing of skills file
"""

from skills_taxonomy.getters.esco_skills import get_skills


def preprocess_skills():
    """Preprocess skills by removing rows with descriptions
    containing two or fewer words and dropping unused columns

    Returns:
        skills: preprocessed skills dataframe
    """
    skills = get_skills()
    skills["n_words_desc"] = skills["description"].apply(lambda x: len(x.split()))
    cond = skills["n_words_desc"] > 2
    skills = skills[cond]
    skills = skills.reset_index(drop=True)
    return skills[["preferredLabel", "altLabels", "description"]]
