"""Save and load the final train/test datasets as a versioned, self-describing artifact."""

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.config import PROCESSED_DATA_DIR


def _hash_obj(obj) -> str:
    """Stable content hash of a DataFrame or Series (values and index)."""
    row_hashes = pd.util.hash_pandas_object(obj, index=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def save_dataset_version(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    groups_train: pd.Series,
    groups_test: pd.Series,
    settings: dict,
) -> Path:
    """Write the datasets plus a manifest to data/processed/v_<hash>/ and return that folder."""

    # Integrity checks: refuse to save a dataset that breaks our own rules
    if not X_train.index.equals(y_train.index) or not X_test.index.equals(y_test.index):
        raise ValueError("Features and labels are misaligned.")
    if not X_train.index.equals(groups_train.index) or not X_test.index.equals(groups_test.index):
        raise ValueError("Features and patient groups are misaligned.")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train and test columns differ.")
    overlap = set(groups_train) & set(groups_test)
    if overlap:
        raise ValueError(f"{len(overlap)} patients appear in both train and test.")

    hashes = {
        "X_train": _hash_obj(X_train), "X_test": _hash_obj(X_test),
        "y_train": _hash_obj(y_train), "y_test": _hash_obj(y_test),
    }
    version = hashlib.sha256("".join(hashes.values()).encode()).hexdigest()[:8]

    out_dir = PROCESSED_DATA_DIR / f"v_{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_parquet(out_dir / "X_train.parquet")
    X_test.to_parquet(out_dir / "X_test.parquet")
    y_train.to_frame("Y").to_parquet(out_dir / "y_train.parquet")
    y_test.to_frame("Y").to_parquet(out_dir / "y_test.parquet")
    groups_train.to_frame("patient_nbr").to_parquet(out_dir / "groups_train.parquet")
    groups_test.to_frame("patient_nbr").to_parquet(out_dir / "groups_test.parquet")

    manifest = {
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": settings,
        "shapes": {"X_train": list(X_train.shape), "X_test": list(X_test.shape)},
        "patients": {"train": int(groups_train.nunique()), "test": int(groups_test.nunique()),
                     "overlap": 0},
        "positive_rate": {"train": round(float(y_train.mean()), 6),
                          "test": round(float(y_test.mean()), 6)},
        "columns": list(X_train.columns),
        "hashes": hashes,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return out_dir


def load_dataset_version(version_dir: Path) -> dict:
    """Load every artifact from a saved version, and re-check it against its manifest."""
    version_dir = Path(version_dir)
    with open(version_dir / "manifest.json") as f:
        manifest = json.load(f)

    data = {
        "X_train": pd.read_parquet(version_dir / "X_train.parquet"),
        "X_test": pd.read_parquet(version_dir / "X_test.parquet"),
        "y_train": pd.read_parquet(version_dir / "y_train.parquet")["Y"],
        "y_test": pd.read_parquet(version_dir / "y_test.parquet")["Y"],
        "groups_train": pd.read_parquet(version_dir / "groups_train.parquet")["patient_nbr"],
        "groups_test": pd.read_parquet(version_dir / "groups_test.parquet")["patient_nbr"],
    }

    for name in ["X_train", "X_test", "y_train", "y_test"]:
        if _hash_obj(data[name]) != manifest["hashes"][name]:
            raise ValueError(f"{name} does not match the manifest hash. The file was modified.")

    data["manifest"] = manifest
    return data