# PaySim dataset (download separately)

Download [Synthetic Financial Datasets for Fraud Detection (PaySim) on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1). Extract the full transaction CSV, rename it to `PaySim.csv`, and place it here:

```text
data/raw/PaySim.csv
```

The original local file is **493,534,783 bytes (470.67 MiB)**, exceeding GitHub's 100 MiB per-file limit. Raw and derived transaction datasets are excluded from Git; no download credentials are required by or stored in this repository. Kaggle may require an account to obtain the download. Follow the dataset provider's usage terms.

The notebook expects 6,362,620 rows with these columns:

```text
step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,isFraud,isFlaggedFraud
```

Run Jupyter from the `notebooks/` directory so that `../data/raw/PaySim.csv` resolves correctly. No dataset subset is bundled.
