"""Clean the raw dataset: build the target, exclude invalid rows,
fix known data-quality bugs, and drop unusable columns.
"""

import pandas as pd

from src.config import DEATH_CODES, HOSPICE_CODES


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary target Y: 1 if readmitted within 30 days, else 0."""
    df = df.copy()
    df["Y"] = (df["readmitted"] == "<30").astype(int)
    return df


def exclude_death_and_hospice(df: pd.DataFrame) -> pd.DataFrame:
    """Remove encounters where the patient died or was discharged to hospice —
    these cannot validly be labeled as 'not readmitted'.
    """
    exclude_codes = DEATH_CODES + HOSPICE_CODES
    return df[~df["discharge_disposition_id"].isin(exclude_codes)].copy()


def fix_mislabeled_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Restore the 'None' category in max_glu_serum and A1Cresult.

    pandas' default read_csv na_values handling silently converts the literal
    string "None" (meaning 'test not administered') into NaN. The dataset's
    own metadata confirms these two columns have no genuine missing values,
    so every NaN here is this loading artifact, not real missingness.
    """
    df = df.copy()
    df["max_glu_serum"] = df["max_glu_serum"].fillna("None")
    df["A1Cresult"] = df["A1Cresult"].fillna("None")
    return df


def drop_unusable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are unusable regardless of leakage concerns:
    weight (96.8% missing), citoglipton and examide (constant, nunique == 1).
    """
    return df.drop(columns=["weight", "citoglipton", "examide"])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline in the correct order."""
    df = build_target(df)
    df = exclude_death_and_hospice(df)
    df = fix_mislabeled_missing(df)
    df = drop_unusable_columns(df)
    return df