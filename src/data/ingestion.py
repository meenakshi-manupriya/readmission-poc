"""Load the raw dataset from UCI."""

import pandas as pd

from src.config import RAW_DATA_DIR


def load_raw_data() -> pd.DataFrame:
    """
    Load the diabetes readmission dataset from UCI.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified dataset.
    """
    from ucimlrepo import fetch_ucirepo

    diabetes = fetch_ucirepo(id=296)
    df = diabetes.data.original.copy()

    return df