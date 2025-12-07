# src/eda_script.py
from pathlib import Path
import pandas as pd
import numpy as np

RAW = Path("data/raw/MachineLearningRating_v3.txt")
OUT = Path("data/processed/insurance_cleaned.csv")

def run():
    df = pd.read_csv(RAW, sep='|', dtype=str, low_memory=False)
    # (apply same cleaning as notebook; here is minimal)
    df.columns = [c.strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df['TotalPremium'] = pd.to_numeric(df['TotalPremium'].str.replace(',',''), errors='coerce')
    df['TotalClaims'] = pd.to_numeric(df['TotalClaims'].str.replace(',',''), errors='coerce')
    df['transaction_date'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    # compute simple KPIs as in notebook
    df['has_claim'] = (df['TotalClaims'].fillna(0) > 0).astype(int)
    def safe_lr(r): 
        tp = r['TotalPremium']; tc=r['TotalClaims']
        return np.nan if pd.isna(tp) or tp==0 else tc/tp
    df['loss_ratio'] = df.apply(safe_lr, axis=1)
    df['margin'] = df['TotalPremium'] - df['TotalClaims']
    # subset important columns
    cols = ['UnderwrittenCoverID','PolicyID','transaction_date','TotalPremium','TotalClaims','has_claim','loss_ratio','margin','Province','PostalCode','Gender','VehicleType','make','Model']
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(OUT, index=False)
    print("Saved:", OUT)

if __name__ == "__main__":
    run()
