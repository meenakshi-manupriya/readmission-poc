"""Demographic features: race, gender, age."""

import re

import pandas as pd


def age_band_to_midpoint(age_band: str) -> int:
    """Convert an age range string like '[50-60)' to its numeric midpoint, e.g. 55."""
    low, high = re.findall(r"\d+", age_band)
    return (int(low) + int(high)) // 2


def build_demographic_features(
    df: pd.DataFrame,
    reference_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build demographic features from race, gender, and age.
    """
    df = df[df["gender"] != "Unknown/Invalid"].copy()

    df["age_midpoint"] = df["age"].apply(age_band_to_midpoint)

    demo = pd.get_dummies(
        df[["race", "gender"]],
        prefix=["race", "gender"],
        dummy_na=True,
    )
    demo = demo.drop(columns=["gender_nan"])

    demo["age_midpoint"] = df["age_midpoint"]

    if reference_columns is not None:
        demo = demo.reindex(columns=reference_columns, fill_value=0)

    return demo