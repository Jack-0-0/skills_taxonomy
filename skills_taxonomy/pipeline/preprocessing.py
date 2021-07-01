"""Preprocessing of skills file
"""
from skills_taxonomy.getters.esco_skills import get_skills


def preprocess_skills():
    """Preprocess skills by removing rows with descriptions
    containing two or fewer words and dropping unused columns

    Returns:
        df: processed skills
    """
    skills = get_skills()
    return (
        skills.assign(
            n_words_desc=skills["description"].apply(lambda x: len(x.split()))
        )
        .query("n_words_desc > 2")
        .reset_index(drop=True)
        .loc[:, ["preferredLabel", "altLabels", "description"]]
    )
