"""Diagnosis features: ICD-9 codes grouped into broad clinical categories."""

import pandas as pd

DIAG_COLUMNS = ["diag_1", "diag_2", "diag_3"]

DIAG_CATEGORIES = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
    "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing",
]


def map_icd9_to_category(code) -> str:
    """Map an ICD-9 diagnosis code to one of the broad clinical categories."""
    if pd.isna(code):
        return "Missing"

    code = str(code)

    if code.startswith("V") or code.startswith("E"):
        return "Other"

    try:
        code_num = float(code)
    except ValueError:
        return "Other"

    if 390 <= code_num <= 459 or code_num == 785:
        return "Circulatory"
    if 460 <= code_num <= 519 or code_num == 786:
        return "Respiratory"
    if 520 <= code_num <= 579 or code_num == 787:
        return "Digestive"
    if 250 <= code_num < 251:
        return "Diabetes"
    if 800 <= code_num <= 999:
        return "Injury"
    if 710 <= code_num <= 739:
        return "Musculoskeletal"
    if 580 <= code_num <= 629 or code_num == 788:
        return "Genitourinary"
    if 140 <= code_num <= 239:
        return "Neoplasms"

    return "Other"


def build_diagnosis_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode each diagnosis slot into broad clinical categories.

    Produces 10 columns per slot (9 categories + Missing), 30 in total.
    Columns are identical for any dataset, because the category list is
    fixed in advance rather than learned from the data.
    """
    parts = []
    for col in DIAG_COLUMNS:
        grouped = df[col].apply(map_icd9_to_category)
        grouped = grouped.astype(pd.CategoricalDtype(DIAG_CATEGORIES))
        parts.append(pd.get_dummies(grouped, prefix=col))

    return pd.concat(parts, axis=1)