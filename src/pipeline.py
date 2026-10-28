"""Run the full data pipeline: raw data -> versioned train/test dataset."""

from src.config import RANDOM_SEED
from src.data.ingestion import load_raw_data
from src.data.validation import validate_raw_data
from src.data.cleaning import clean_data
from src.data.splitting import split_by_patient
from src.features.build_features import build_features
from src.data.versioning import save_dataset_version


def run_pipeline(
    test_size: float = 0.2,
    include_payer: bool = False,
    include_disposition: bool = True,
):
    """Execute every stage in order and return the saved dataset folder."""
    df = load_raw_data()
    validate_raw_data(df)
    df = clean_data(df)

    train_df, test_df = split_by_patient(df, test_size=test_size)

    X_train = build_features(
        train_df,
        include_payer=include_payer,
        include_disposition=include_disposition,
    )
    X_test = build_features(
        test_df,
        include_payer=include_payer,
        include_disposition=include_disposition,
        reference_columns=X_train.columns.tolist(),
    )

    return save_dataset_version(
        X_train, X_test,
        train_df["Y"], test_df["Y"],
        train_df["patient_nbr"], test_df["patient_nbr"],
        settings={
            "include_payer": include_payer,
            "include_disposition": include_disposition,
            "split": "GroupShuffleSplit by patient_nbr",
            "test_size": test_size,
            "random_seed": RANDOM_SEED,
            "note": "Payer excluded (D11); disposition included (D13).",
        },
    )


if __name__ == "__main__":
    out_dir = run_pipeline()
    print("Dataset version saved to:", out_dir)