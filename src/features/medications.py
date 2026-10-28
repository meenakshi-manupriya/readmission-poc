"""Medication features: per-drug status for common drugs, plus regimen summaries."""

import pandas as pd

# All 21 drug columns remaining after cleaning (citoglipton and examide were dropped)
DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

# Drugs common enough to keep as their own columns (chosen from training-set rates)
KEPT_DRUGS = [
    "metformin", "repaglinide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "insulin",
]

DRUG_STATUSES = ["No", "Steady", "Up", "Down"]


def build_medication_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build medication features.

    - One-hot status (No/Steady/Up/Down) for the 8 kept drugs: 32 columns.
    - n_drugs_prescribed and n_drugs_changed, computed across ALL 21 drugs,
      so the rare drugs still contribute without needing their own columns.
    - Binary flags for 'change' and 'diabetesMed'.

    Columns are identical for any dataset, because every category list is
    fixed in advance rather than learned from the data.
    """
    parts = []

    for col in KEPT_DRUGS:
        unexpected = set(df[col].dropna().unique()) - set(DRUG_STATUSES)
        if unexpected:
            raise ValueError(f"Unexpected status values in {col}: {unexpected}")

        status = df[col].astype(pd.CategoricalDtype(DRUG_STATUSES))
        parts.append(pd.get_dummies(status, prefix=col))

    summary = pd.DataFrame(index=df.index)
    summary["n_drugs_prescribed"] = (df[DRUG_COLUMNS] != "No").sum(axis=1)
    summary["n_drugs_changed"] = df[DRUG_COLUMNS].isin(["Up", "Down"]).sum(axis=1)
    summary["change_flag"] = (df["change"] == "Ch").astype(int)
    summary["diabetes_med_flag"] = (df["diabetesMed"] == "Yes").astype(int)
    parts.append(summary)

    return pd.concat(parts, axis=1)