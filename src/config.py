"""Shared constants used across the data pipeline."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Cohort exclusion codes (confirmed against IDs_mapping.csv) ---
DEATH_CODES = [11, 19, 20, 21]
HOSPICE_CODES = [13, 14]