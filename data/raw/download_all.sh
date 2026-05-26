#!/bin/bash
# Download all three datasets from Kaggle.
# Requires: kaggle CLI configured with ~/.kaggle/kaggle.json

set -e

DATA_DIR="$(dirname "$0")"

echo "Downloading IEEE-CIS Fraud Detection..."
kaggle competitions download -c ieee-fraud-detection -p "$DATA_DIR"
unzip -o "$DATA_DIR/ieee-fraud-detection.zip" -d "$DATA_DIR" \
    train_transaction.csv train_identity.csv
echo "IEEE-CIS: done."

echo ""
echo "Downloading Give Me Some Credit..."
kaggle competitions download -c GiveMeSomeCredit -p "$DATA_DIR"
unzip -o "$DATA_DIR/GiveMeSomeCredit.zip" -d "$DATA_DIR" cs-training.csv
echo "GMSC: done."

echo ""
echo "Downloading Credit Card Fraud Detection (ULB)..."
kaggle datasets download -d mlg-ulb/creditcardfraud -p "$DATA_DIR"
unzip -o "$DATA_DIR/creditcardfraud.zip" -d "$DATA_DIR" creditcard.csv
echo "CC Fraud: done."

echo ""
echo "All datasets downloaded. Verifying sizes..."
ls -lh "$DATA_DIR"/*.csv
echo "Done."
