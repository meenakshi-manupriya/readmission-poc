"""Discharge destination features."""

import pandas as pd

# Chosen from training-set counts (>= ~1% of rows), then frozen
KEPT_DISPOSITIONS = [1, 3, 6, 18, 2, 22, 5, 25]
DISPOSITION_CATEGORIES = [str(c) for c in KEPT_DISPOSITIONS] + ["Other"]


def map_disposition(code) -> str:
    if pd.isna(code):
        raise ValueError("Unexpected missing discharge_disposition_id")
    return str(code) if code in KEPT_DISPOSITIONS else "Other"


def build_disposition_features(df: pd.DataFrame) -> pd.DataFrame:
    """9 fixed one-hot columns: 8 kept destinations plus Other."""
    grouped = df["discharge_disposition_id"].apply(map_disposition)
    grouped = grouped.astype(pd.CategoricalDtype(DISPOSITION_CATEGORIES))
    return pd.get_dummies(grouped, prefix="disposition")