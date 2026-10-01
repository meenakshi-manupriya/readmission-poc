"""Encounter-level features: admission details, utilization counts."""

import pandas as pd

NUMERIC_COLUMNS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CATEGORICAL_COLUMNS = [
    "admission_type_id",
    "admission_source_id",
]


def build_encounter_features(
    df: pd.DataFrame,
    reference_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build encounter-level features.

    Parameters
    ----------
    df : pd.DataFrame
        The data to build features from (train or test).
    reference_columns : list[str], optional
        The exact set of one-hot columns to conform to (learned from
        training data). If None, columns are derived from `df` itself —
        only safe to do this when `df` IS the training set.

    Returns
    -------
    pd.DataFrame with the numeric columns unchanged, and one-hot encoded
    categorical columns aligned to `reference_columns` if provided.
    """
    numeric = df[NUMERIC_COLUMNS].copy()

    categorical = pd.get_dummies(
        df[CATEGORICAL_COLUMNS].astype(str),
        prefix=CATEGORICAL_COLUMNS,
    )

    result = pd.concat([numeric, categorical], axis=1)

    if reference_columns is not None:
        result = result.reindex(columns=reference_columns, fill_value=0)

    return result