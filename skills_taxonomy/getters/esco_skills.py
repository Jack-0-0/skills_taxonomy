"""Data getters for the esco skills data.

Data source: https://ec.europa.eu/esco/portal
"""
import pandas as pd
from skills_taxonomy import PROJECT_DIR


def get_skills() -> pd.DataFrame:
    """Load esco skills.

    Returns:
        df: dataframe of esco skills
    """
    return pd.read_csv(f"{PROJECT_DIR}/inputs/data/skills_en.csv")
