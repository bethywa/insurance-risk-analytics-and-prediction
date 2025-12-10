# src/task3_ab_test/utils.py
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def load_data(path):
    """
    Loads the CSV safely without assuming a date column exists.
    """
    df = pd.read_csv(path)

    # Try to detect a date column automatically
    date_cols = [c for c in df.columns if "date" in c.lower() or "month" in c.lower()]

    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors="ignore")
        except:
            pass

    return df



def compute_metrics(df):
    """
    Returns a DataFrame with added columns:
      - has_claim (bool)
      - claim_severity (TotalClaims where TotalClaims>0, else NaN)
      - margin = TotalPremium - TotalClaims
    """
    df = df.copy()
    # ensure numeric
    df["TotalClaims"] = pd.to_numeric(df["TotalClaims"], errors="coerce").fillna(0.0)
    df["TotalPremium"] = pd.to_numeric(df["TotalPremium"], errors="coerce").fillna(0.0)

    df["has_claim"] = df["TotalClaims"] > 0
    df["claim_severity"] = df["TotalClaims"].where(df["TotalClaims"] > 0, np.nan)
    df["margin"] = df["TotalPremium"] - df["TotalClaims"]
    return df

def group_metrics(df, group_col):
    """
    For a grouping column (e.g., Province), returns:
      - count
      - claim frequency (proportion)
      - mean severity (among claims)
      - mean margin
    """
    g = df.groupby(group_col).agg(
        total_policies=("TotalPremium", "size"),
        claims_count=("has_claim", "sum"),
        freq=("has_claim", "mean"),
        mean_severity=("claim_severity", "mean"),
        median_severity=("claim_severity", "median"),
        mean_margin=("margin", "mean")
    ).reset_index()
    return g

# ----- ASSUMPTION CHECKS -----
def check_normality(series, alpha=0.05):
    """Shapiro-Wilk test for normality (useful for n < ~5000). Returns (stat, p, is_normal)"""
    series = series.dropna()
    if len(series) < 3:
        return None, None, False
    stat, p = stats.shapiro(series) if len(series) <= 5000 else stats.normaltest(series)  # use D'Agostino for big N
    return stat, p, p > alpha

def check_variance_levene(groups, alpha=0.05):
    """Levene test - check equal variances across groups. groups is list of arrays"""
    stat, p = stats.levene(*groups, center="median")
    return stat, p, p > alpha

# ----- STAT TESTS -----
def chi2_test_for_freq(df, group_col):
    """
    Build contingency table (has_claim vs group)
    Run Chi-square test of independence.
    """
    contingency = pd.crosstab(df[group_col], df["has_claim"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {"chi2": chi2, "p": p, "dof": dof, "expected": expected, "contingency": contingency}

def ttest_independent(groupA, groupB, equal_var=True):
    """Two-sample t-test for means"""
    res = stats.ttest_ind(groupA.dropna(), groupB.dropna(), equal_var=equal_var)
    return {"stat": res.statistic, "p": res.pvalue}

def anova_oneway(groups):
    """One-way ANOVA for 3+ groups (assumes normal & equal var)"""
    stat, p = stats.f_oneway(*groups)
    return {"stat": stat, "p": p}

def kruskal_test(groups):
    """Non-parametric Kruskal-Wallis for 3+ groups"""
    stat, p = stats.kruskal(*groups)
    return {"stat": stat, "p": p}

# ----- EFFECT SIZE (simple) -----
def effect_size_cohens_d(a, b):
    """Cohen's d for two groups"""
    a = a.dropna()
    b = b.dropna()
    n1, n2 = len(a), len(b)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    s = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if s == 0:
        return np.nan
    return (a.mean() - b.mean()) / s

def eta_squared_from_f(f_stat, df_between, df_within):
    """Estimate eta-squared from F-statistic"""
    return (f_stat * df_between) / (f_stat * df_between + df_within)
