"""Assemble all feature modules into a single feature matrix."""

import pandas as pd

from src.features.demographics import build_demographic_features
from src.features.encounter import build_encounter_features
from src.features.diagnoses import build_diagnosis_features
from src.features.medications import build_medication_features
from src.features.payer import build_payer_features
from src.features.disposition import build_disposition_features

# Columns that must never appear in the feature matrix (see the leakage audit)
FORBIDDEN_COLUMNS = {"Y", "readmitted", "encounter_id", "patient_nbr"}


def build_features(
    df: pd.DataFrame,
    include_payer: bool = False,
    include_disposition: bool = True,
    reference_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data (train or test).
    include_payer : bool
        Payer features. Off by default: tested and dropped (D11).
    include_disposition : bool
        Discharge destination features. On by default (D13).
    reference_columns : list[str], optional
        Column list learned from the training matrix. Pass it when building
        the test matrix so test is forced to match train exactly.
    """
    parts = {
        "demographics": build_demographic_features(df),
        "encounter": build_encounter_features(df),
        "diagnoses": build_diagnosis_features(df),
        "medications": build_medication_features(df),
    }
    if include_payer:
        parts["payer"] = build_payer_features(df)
    if include_disposition:
        parts["disposition"] = build_disposition_features(df)

    # pd.concat(axis=1) matches rows by index label, so a module that dropped or reordered rows would misalign silently. Check before gluing.
    for name, part in parts.items():
        if not part.index.equals(df.index):
            raise ValueError(f"Row mismatch: '{name}' features don't match the input rows.")

    X = pd.concat(parts.values(), axis=1)

    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0)

    leaked = FORBIDDEN_COLUMNS & set(X.columns)
    if leaked:
        raise ValueError(f"Forbidden columns in feature matrix: {leaked}")
    if X.columns.duplicated().any():
        raise ValueError("Duplicate column names in feature matrix.")
    if X.isna().any().any():
        raise ValueError("NaN values in feature matrix.")

    return X.astype(int)