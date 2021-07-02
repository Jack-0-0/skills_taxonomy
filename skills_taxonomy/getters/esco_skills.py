"""Data getters for the esco skills data.

Data source: https://ec.europa.eu/esco/portal
"""
import pandas as pd
from skills_taxonomy import PROJECT_DIR, logger


def get_skills() -> pd.DataFrame:
    """Load esco skills.

    Returns:
        df: dataframe of esco skills
    """
    return pd.read_csv(f"{PROJECT_DIR}/inputs/data/skills_en.csv")


def get_output_skills(
    skills_path=f"{PROJECT_DIR}/outputs/skills/labelled_skills.csv",
) -> pd.DataFrame:
    try:
        return pd.read_csv(skills_path, index_col=[0])
    except FileNotFoundError:
        logger.error(f"There is no file to read at {skills_path}")
