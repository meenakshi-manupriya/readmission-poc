"""Validate the raw dataset against known expectations before cleaning."""

import pandas as pd

EXPECTED_SHAPE = (101766, 50)

EXPECTED_COLUMNS = [
    "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty",
    "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "readmitted",
]


def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Run sanity checks on the freshly loaded raw dataset.

    Raises
    ------
    ValueError
        If the data doesn't match known structural expectations.
    """
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Unexpected shape: got {df.shape}, expected {EXPECTED_SHAPE}. "
            "The source dataset may have changed — investigate before proceeding."
        )

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    if df["encounter_id"].duplicated().any():
        raise ValueError("Duplicate encounter_id values found — expected one row per encounter.")