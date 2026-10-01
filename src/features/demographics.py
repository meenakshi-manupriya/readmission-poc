"""Demographic features: race, gender, age."""

import re

import pandas as pd


def age_band_to_midpoint(age_band: str) -> int:
    """Convert an age range string like '[50-60)' to its numeric midpoint, e.g. 55."""
    low, high = re.findall(r"\d+", age_band)
    return (int(low) + int(high)) // 2


def build_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build demographic features from race, gender, and age.

    - Drops the ~3 'Unknown/Invalid' gender rows (negligible, ambiguous).
    - Converts age band strings to a numeric midpoint (ordinal encoding).
    - One-hot encodes race and gender.

    Returns a new DataFrame with only the engineered demographic columns
    (plus the original index, so it can be joined back to other feature sets).
    """
    df = df[df["gender"] != "Unknown/Invalid"].copy()

    df["age_midpoint"] = df["age"].apply(age_band_to_midpoint)

    demo = pd.get_dummies(
        df[["race", "gender"]],
        prefix=["race", "gender"],
        dummy_na=True,   # keep missing race as its own explicit column, don't silently drop it
    )

    demo = demo.drop(columns=["gender_nan"])  # drop the gender NaN column, since we already dropped those rows above

    demo["age_midpoint"] = df["age_midpoint"]

    return demo