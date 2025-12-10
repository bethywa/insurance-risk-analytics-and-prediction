# src/task3_ab_test/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # repo root
DATA_PATH = BASE_DIR / "data" / "processed" / "insurance_cleaned.csv"

# When testing zip codes, choose top N to compare (avoid too many tiny groups)
TOP_N_ZIPCODES = 10

# Stats thresholds
ALPHA = 0.05
