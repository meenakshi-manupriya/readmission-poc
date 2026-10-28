"""Finer-grained primary-diagnosis features (3-digit ICD-9 codes)."""

import pandas as pd

# Codes with >= ~1% of training rows, chosen by frequency (not outcome), then frozen
KEPT_DIAG1_CODES = ['250', '428', '414', '786', '410', '486', '427', '491', '715', '682', '780', '996', '434', '276', '599', '38', '584', 'V57', '493', '577', '820', '435']   # paste the 22 codes from the notebook, as strings
DIAG1_CATEGORIES = KEPT_DIAG1_CODES + ["Other", "Missing"]


def map_diag_code(value) -> str:
    if pd.isna(value):
        return "Missing"
    code = str(value).split(".")[0]
    return code if code in KEPT_DIAG1_CODES else "Other"


def build_diagnosis_detail_features(df: pd.DataFrame) -> pd.DataFrame:
    """24 fixed one-hot columns for diag_1 (22 codes + Other + Missing)."""
    grouped = df["diag_1"].apply(map_diag_code)
    grouped = grouped.astype(pd.CategoricalDtype(DIAG1_CATEGORIES))
    return pd.get_dummies(grouped, prefix="diag1code")