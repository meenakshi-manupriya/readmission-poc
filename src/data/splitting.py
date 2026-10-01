"""Split the dataset into train/test sets without leaking patients across splits."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import RANDOM_SEED


def split_by_patient(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train/test sets, grouped by patient_nbr so that no patient's
    encounters appear in both sets.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'patient_nbr' column.
    test_size : float
        Fraction of *patients* (not rows) to hold out for testing.

    Returns
    -------
    (train_df, test_df) : tuple of pd.DataFrame
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_SEED)

    train_idx, test_idx = next(
        splitter.split(df, groups=df["patient_nbr"])
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    return train_df, test_df