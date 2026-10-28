"""Discharge destination, admitting specialty, and lab-test features."""

import pandas as pd

# Chosen from training-set counts (>= ~1% of rows), then frozen
KEPT_DISPOSITIONS = [1, 3, 6, 18, 2, 22, 5, 25]
DISPOSITION_CATEGORIES = [str(c) for c in KEPT_DISPOSITIONS] + ["Other"]

KEPT_SPECIALTIES = [
    "InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology",
    "Surgery-General", "Nephrology", "Orthopedics", "Orthopedics-Reconstructive",
    "Radiologist",
]
SPECIALTY_CATEGORIES = KEPT_SPECIALTIES + ["Other", "Missing"]

A1C_CATEGORIES = ["None", "Norm", ">7", ">8"]
GLUCOSE_CATEGORIES = ["None", "Norm", ">200", ">300"]


def map_disposition(code) -> str:
    if pd.isna(code):
        raise ValueError("Unexpected missing discharge_disposition_id")
    return str(code) if code in KEPT_DISPOSITIONS else "Other"


def map_specialty(value) -> str:
    if pd.isna(value):
        return "Missing"
    return value if value in KEPT_SPECIALTIES else "Other"


def _check_values(series: pd.Series, allowed: list, name: str) -> None:
    """Fail loudly instead of letting unexpected values silently become NaN."""
    unexpected = set(series.dropna().unique()) - set(allowed)
    if unexpected or series.isna().any():
        raise ValueError(f"Unexpected or missing values in {name}: {unexpected}")


def _one_hot(series: pd.Series, categories: list, prefix: str) -> pd.DataFrame:
    series = series.astype(pd.CategoricalDtype(categories))
    return pd.get_dummies(series, prefix=prefix)


def build_discharge_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """28 fixed columns: disposition (9), specialty (11), A1C (4), glucose (4)."""
    parts = [
        _one_hot(df["discharge_disposition_id"].apply(map_disposition),
                 DISPOSITION_CATEGORIES, "disposition"),
        _one_hot(df["medical_specialty"].apply(map_specialty),
                 SPECIALTY_CATEGORIES, "specialty"),
    ]
    for col, cats, prefix in [("A1Cresult", A1C_CATEGORIES, "a1c"),
                              ("max_glu_serum", GLUCOSE_CATEGORIES, "glucose")]:
        _check_values(df[col], cats, col)
        parts.append(_one_hot(df[col], cats, prefix))

    result = pd.concat(parts, axis=1)
    # Keep column names free of characters some libraries reject
    result.columns = [c.replace(">", "gt").replace("/", "_") for c in result.columns]
    return result