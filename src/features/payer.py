"""Payer (insurance) features.

Kept in its own module so it can be switched on and off for the
with/without-payer comparison. The timing of when payer_code is finalized
relative to discharge is unverified, so we measure its impact instead.
"""

import pandas as pd

# Codes with >= ~1% of training rows (chosen from the training set, then fixed)
KEPT_PAYERS = ["MC", "HM", "SP", "BC", "MD", "CP", "UN", "CM", "OG"]
PAYER_CATEGORIES = KEPT_PAYERS + ["Other", "Missing"]


def map_payer(code) -> str:
    """Map a raw payer code to a kept code, 'Other', or 'Missing'."""
    if pd.isna(code):
        return "Missing"
    if code in KEPT_PAYERS:
        return code
    return "Other"


def build_payer_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode payer_code into 11 fixed categories."""
    grouped = df["payer_code"].apply(map_payer)
    grouped = grouped.astype(pd.CategoricalDtype(PAYER_CATEGORIES))
    return pd.get_dummies(grouped, prefix="payer")