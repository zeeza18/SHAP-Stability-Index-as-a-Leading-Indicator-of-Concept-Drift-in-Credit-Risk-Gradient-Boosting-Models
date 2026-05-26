# Dataset Download Instructions

## Prerequisites

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account → API → Create New API Token → downloads `kaggle.json`
3. Place `kaggle.json` in `~/.kaggle/kaggle.json`
4. Install CLI: `pip install kaggle`

## Manual Commands

### IEEE-CIS Fraud Detection (PRIMARY — ~550MB)
```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip train_transaction.csv train_identity.csv
```
Files needed: `train_transaction.csv` (~500MB), `train_identity.csv` (~50MB)

### Give Me Some Credit (SECONDARY — ~7MB)
```bash
kaggle competitions download -c GiveMeSomeCredit
unzip GiveMeSomeCredit.zip cs-training.csv
```
File needed: `cs-training.csv`

### Credit Card Fraud (SUPPLEMENTARY — ~144MB)
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip creditcard.csv
```
File needed: `creditcard.csv`

## Automated Download

```bash
bash data/raw/download_all.sh
```

## Expected File Sizes

| File | Size | Rows |
|------|------|------|
| train_transaction.csv | ~490MB | 590,540 |
| train_identity.csv | ~48MB | 144,233 |
| cs-training.csv | ~7MB | 150,000 |
| creditcard.csv | ~144MB | 284,807 |

## Validate After Download

```bash
python src/data/loader.py --validate
```
