"""Dataset loader for all three credit risk datasets.

Handles IEEE-CIS Fraud Detection, Give Me Some Credit, and
Credit Card Fraud (ULB). Merges transaction + identity for IEEE-CIS.
"""

import argparse
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from requests import HTTPError

np.random.seed(42)
random.seed(42)

# Resolve project root (research-credit-drift/)
ROOT_DIR = Path(__file__).resolve().parents[2]

# Load environment overrides if present
load_dotenv(ROOT_DIR / ".env")


def _resolve_dir(env_var: str, default: Path) -> Path:
    """Resolve directory from env var (if set), else fall back to default."""
    override = os.environ.get(env_var)
    if override:
        path = Path(override).expanduser()
        # If the override is relative, anchor it to the project root
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path.resolve()
    return default.resolve()


RAW_DIR = _resolve_dir("DATA_RAW_DIR", ROOT_DIR / "data" / "raw")
PROCESSED_DIR = _resolve_dir("DATA_PROCESSED_DIR", ROOT_DIR / "data" / "processed")


def _report_kaggle_http_error(err: Exception, dataset: str) -> bool:
    """Return True if error was handled and a user-friendly message was printed."""
    if isinstance(err, HTTPError):
        status = getattr(err.response, "status_code", None)
        if status == 403:
            print(
                f"[ERROR] {dataset}: Kaggle returned 403 Forbidden. "
                "Make sure you have joined the competition/dataset and accepted the rules in the browser, "
                "and that ~/.kaggle/kaggle.json contains the correct username/key."
            )
            return True
    return False


def _extract_members(zip_path: Path, members: list[str], dest: Path) -> None:
    """Extract only selected members from a zip file."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in members:
            try:
                zf.extract(member, path=dest)
            except KeyError:
                print(f"[WARN] {member} not found inside {zip_path.name}")


def download_all(raw_dir: Path = RAW_DIR, only_missing: bool = True) -> None:
    """Download datasets from Kaggle using the Kaggle API.

    Args:
        raw_dir: Destination directory for raw CSVs.
        only_missing: If True, download only datasets that are missing.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    raw_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    # IEEE-CIS
    ieee_files = ["train_transaction.csv", "train_identity.csv"]
    ieee_missing = any(not (raw_dir / f).exists() for f in ieee_files)
    if ieee_missing or not only_missing:
        print("Downloading IEEE-CIS Fraud Detection ...")
        zip_path = raw_dir / "ieee-fraud-detection.zip"
        try:
            api.competition_download_files(
                "ieee-fraud-detection", path=raw_dir, force=True, quiet=False
            )
            _extract_members(zip_path, ieee_files, raw_dir)
            print("IEEE-CIS: done.")
        except Exception as err:
            if _report_kaggle_http_error(err, "IEEE-CIS"):
                return
            raise
    else:
        print("IEEE-CIS already present; skipping download.")

    # GMSC
    gmsc_file = "cs-training.csv"
    gmsc_missing = not (raw_dir / gmsc_file).exists()
    if gmsc_missing or not only_missing:
        print("\nDownloading Give Me Some Credit ...")
        zip_path = raw_dir / "GiveMeSomeCredit.zip"
        try:
            api.competition_download_files(
                "GiveMeSomeCredit", path=raw_dir, force=True, quiet=False
            )
            _extract_members(zip_path, [gmsc_file], raw_dir)
            print("GMSC: done.")
        except Exception as err:
            if _report_kaggle_http_error(err, "GMSC"):
                return
            raise
    else:
        print("GMSC already present; skipping download.")

    # Credit Card Fraud
    ccfraud_file = "creditcard.csv"
    ccfraud_missing = not (raw_dir / ccfraud_file).exists()
    if ccfraud_missing or not only_missing:
        print("\nDownloading Credit Card Fraud (ULB) ...")
        zip_path = raw_dir / "creditcardfraud.zip"
        try:
            api.dataset_download_files(
                "mlg-ulb/creditcardfraud", path=raw_dir, force=True, quiet=False
            )
            _extract_members(zip_path, [ccfraud_file], raw_dir)
            print("CC Fraud: done.")
        except Exception as err:
            if _report_kaggle_http_error(err, "CC Fraud"):
                return
            raise
    else:
        print("CC Fraud already present; skipping download.")

    print("\nAll requested downloads complete.")


def load_ieee_cis(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and merge IEEE-CIS transaction and identity tables.

    Returns:
        Merged DataFrame with all transactions and available identity features.
    """
    tx_path = raw_dir / "train_transaction.csv"
    id_path = raw_dir / "train_identity.csv"

    print(f"Loading IEEE-CIS transaction data from {tx_path} ...")
    transaction = pd.read_csv(tx_path)

    print(f"Loading IEEE-CIS identity data from {id_path} ...")
    identity = pd.read_csv(id_path)

    print("Merging on TransactionID (left join) ...")
    df = transaction.merge(identity, on="TransactionID", how="left")

    # Fill identity-side NaNs
    identity_cols = [c for c in identity.columns if c != "TransactionID"]
    cat_id_cols = identity.select_dtypes(include="object").columns.tolist()
    num_id_cols = [c for c in identity_cols if c not in cat_id_cols]

    for col in cat_id_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    for col in num_id_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-999)

    print(f"IEEE-CIS loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def load_gmsc(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load Give Me Some Credit dataset.

    Returns:
        DataFrame with borrower records.
    """
    path = raw_dir / "cs-training.csv"
    print(f"Loading Give Me Some Credit from {path} ...")

    # Some Kaggle downloads save as a zip even when the filename is .csv
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise FileNotFoundError("cs-training.csv zip did not contain a CSV")
            with zf.open(members[0]) as fh:
                df = pd.read_csv(fh, index_col=0)
    else:
        df = pd.read_csv(path, index_col=0)

    print(f"GMSC loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def load_ccfraud(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load Credit Card Fraud Detection (ULB) dataset.

    Returns:
        DataFrame with anonymized transaction features.
    """
    path = raw_dir / "creditcard.csv"
    print(f"Loading Credit Card Fraud from {path} ...")

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise FileNotFoundError("creditcard.csv zip did not contain a CSV")
            with zf.open(members[0]) as fh:
                df = pd.read_csv(fh)
    else:
        df = pd.read_csv(path)

    print(f"CC Fraud loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def _validate_schema(df: pd.DataFrame, required_cols: list, name: str) -> bool:
    """Check that all required columns are present."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[FAIL] {name}: missing columns: {missing}")
        return False
    print(f"[OK] {name}: schema valid ({df.shape[0]:,} rows, {df.shape[1]} cols)")
    return True


def validate_all(raw_dir: Path = RAW_DIR) -> None:
    """Validate that all raw datasets are present and have expected schemas."""
    results = {}

    # IEEE-CIS
    if (raw_dir / "train_transaction.csv").exists():
        df = load_ieee_cis(raw_dir)
        ok = _validate_schema(
            df,
            ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"],
            "IEEE-CIS",
        )
        results["ieee_cis"] = ok
    else:
        print(
            f"[MISSING] IEEE-CIS: train_transaction.csv not found in {raw_dir.resolve()}"
        )
        results["ieee_cis"] = False

    # GMSC
    if (raw_dir / "cs-training.csv").exists():
        df = load_gmsc(raw_dir)
        ok = _validate_schema(
            df,
            ["SeriousDlqin2yrs", "RevolvingUtilizationOfUnsecuredLines", "age"],
            "GMSC",
        )
        results["gmsc"] = ok
    else:
        print(f"[MISSING] GMSC: cs-training.csv not found in {raw_dir.resolve()}")
        results["gmsc"] = False

    # CC Fraud
    if (raw_dir / "creditcard.csv").exists():
        df = load_ccfraud(raw_dir)
        ok = _validate_schema(df, ["Time", "Amount", "Class"], "CC Fraud")
        results["ccfraud"] = ok
    else:
        print(f"[MISSING] CC Fraud: creditcard.csv not found in {raw_dir.resolve()}")
        results["ccfraud"] = False

    print("\n--- Validation summary ---")
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")

    if not all(results.values()):
        print("\nRun data/raw/download_all.sh to download missing datasets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset loader and validator")
    parser.add_argument(
        "--validate", action="store_true", help="Validate all raw datasets"
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download any missing datasets via Kaggle API",
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Force re-download of all datasets via Kaggle API",
    )
    args = parser.parse_args()

    if args.download_all:
        download_all(only_missing=False)
    elif args.download_missing:
        download_all(only_missing=True)

    if args.validate:
        validate_all()
