# src/task3_ab_test/run_ab_tests.py
import pandas as pd
from config import DATA_PATH, TOP_N_ZIPCODES, ALPHA
from utils import load_data, compute_metrics, group_metrics, chi2_test_for_freq, \
                  check_normality, check_variance_levene, anova_oneway, kruskal_test, \
                  ttest_independent, effect_size_cohens_d
import warnings
warnings.filterwarnings("ignore")

def test_province_risk(df):
    print("\n=== Test 1: Risk differences across Province ===")
    g = group_metrics(df, "Province").sort_values("total_policies", ascending=False)
    print(g[["Province", "total_policies", "claims_count", "freq", "mean_severity", "mean_margin"]].head(20).to_string(index=False))

    # Frequency: chi-square across provinces
    chi = chi2_test_for_freq(df, "Province")
    print("\nChi-square for claim frequency across Province: chi2={:.3f}, p={:.4f}".format(chi["chi2"], chi["p"]))
    if chi["p"] < ALPHA:
        print("=> Reject H0: claim frequency differs across provinces (p < {})".format(ALPHA))
    else:
        print("=> Fail to reject H0: no evidence of difference in frequency across provinces")

    # For severity, use ANOVA/Kruskal among provinces with enough claims
    # prepare severity groups for provinces with min sample size
    provinces = g[g["claims_count"] >= 20]["Province"].tolist()  # only provinces with >=20 claims
    groups = [df.loc[(df["Province"] == p) & (df["has_claim"]), "claim_severity"].dropna() for p in provinces]
    if len(groups) >= 3:
        # Normality check on pooled? We'll check variances and normality on each
        normals = [check_normality(gp)[2] if len(gp) >= 3 else False for gp in groups]
        lev = check_variance_levene(groups)
        print("\nLevene equal variance p={:.4f} -> equal_var={}".format(lev[1], lev[2]))
        if all(normals) and lev[2]:
            an = anova_oneway(groups)
            print("ANOVA severity: stat={:.3f}, p={:.4f}".format(an["stat"], an["p"]))
            if an["p"] < ALPHA:
                print("=> Reject H0: mean severity differs among provinces")
            else:
                print("=> Fail to reject H0 (severity)")
        else:
            kr = kruskal_test(groups)
            print("Kruskal-Wallis severity: stat={:.3f}, p={:.4f}".format(kr["stat"], kr["p"]))
            if kr["p"] < ALPHA:
                print("=> Reject H0 (non-parametric): severity differs among provinces")
            else:
                print("=> Fail to reject H0 (severity non-parametric)")
    else:
        print("Not enough provinces with sufficient claims for severity test (>=3 groups).")

def test_zipcode_risk_and_margin(df, top_n=TOP_N_ZIPCODES):
    print("\n=== Test 2: Risk & Margin differences between top Zip/Postal codes (Top {}) ===".format(top_n))
    # pick top N postal codes by policy count
    zip_counts = df["PostalCode"].value_counts().dropna()
    top_zips = zip_counts.head(top_n).index.tolist()
    sub = df[df["PostalCode"].isin(top_zips)].copy()
    print("Top zip codes sample sizes:")
    print(sub["PostalCode"].value_counts())

    # Frequency: chi2
    chi = chi2_test_for_freq(sub, "PostalCode")
    print("\nChi-square for frequency across top zip codes: chi2={:.3f}, p={:.4f}".format(chi["chi2"], chi["p"]))
    if chi["p"] < ALPHA:
        print("=> Reject H0: frequency differs across these zip codes")
    else:
        print("=> Fail to reject H0: no evidence of freq diff in top zip codes")

    # Margin differences: prepare groups
    groups_margin = [sub.loc[sub["PostalCode"] == z, "margin"].dropna() for z in top_zips]
    # choose ANOVA or Kruskal depending on normality/variance
    normals = [check_normality(gp)[2] if len(gp)>=3 else False for gp in groups_margin]
    lev = check_variance_levene(groups_margin)
    print("\nLevene for margin groups p={:.4f}, equal_var={}".format(lev[1], lev[2]))
    if all(normals) and lev[2]:
        an = anova_oneway(groups_margin)
        print("ANOVA margin: stat={:.3f}, p={:.4f}".format(an["stat"], an["p"]))
        if an["p"] < ALPHA:
            print("=> Reject H0: mean margin differs across top zip codes")
        else:
            print("=> Fail to reject H0 (margin)")
    else:
        kr = kruskal_test(groups_margin)
        print("Kruskal-Wallis margin: stat={:.3f}, p={:.4f}".format(kr["stat"], kr["p"]))
        if kr["p"] < ALPHA:
            print("=> Reject H0 (non-parametric): margin differs across top zip codes")
        else:
            print("=> Fail to reject H0 (margin non-parametric)")

def test_gender_risk(df):
    print("\n=== Test 3: Risk difference by Gender ===")
    # Frequency: chi-square
    print(df["Gender"].value_counts(dropna=False).head(20))
    chi = chi2_test_for_freq(df, "Gender")
    print("\nChi-square for frequency by Gender: chi2={:.3f}, p={:.4f}".format(chi["chi2"], chi["p"]))
    if chi["p"] < ALPHA:
        print("=> Reject H0: claim frequency differs by Gender")
    else:
        print("=> Fail to reject H0: no evidence freq diff by Gender")

    # Severity among claimants: t-test between two biggest gender groups
    counts = df["Gender"].value_counts()
    top2 = counts[counts>0].index[:2].tolist()
    if len(top2) >= 2:
        g1 = df.loc[(df["Gender"]==top2[0]) & (df["has_claim"]), "claim_severity"].dropna()
        g2 = df.loc[(df["Gender"]==top2[1]) & (df["has_claim"]), "claim_severity"].dropna()
        print(f"\nComparing severity between {top2[0]} (n={len(g1)}) and {top2[1]} (n={len(g2)})")
        if len(g1) >= 10 and len(g2) >= 10:
            # variance check
            _, pvar, eq = check_variance_levene([g1, g2])
            tt = ttest_independent(g1, g2, equal_var=eq)
            d = effect_size_cohens_d(g1, g2)
            print("t-test stat={:.3f}, p={:.4f}, equal_var={}".format(tt["stat"], tt["p"], eq))
            print("Cohen d = {:.3f}".format(d))
            if tt["p"] < ALPHA:
                print("=> Reject H0: mean severity differs between genders")
            else:
                print("=> Fail to reject H0 (gender severity)")
        else:
            print("Not enough claimants per gender for t-test.")
    else:
        print("Not enough gender groups to compare severity.")

def main():
    print("Loading data from:", DATA_PATH)
    df = load_data(DATA_PATH)
    df = compute_metrics(df)
    print("Rows:", len(df))
    test_province_risk(df)
    test_zipcode_risk_and_margin(df)
    test_gender_risk(df)
    print("\nAll tests done. Interpret each result & consider effect sizes before action.")

if __name__ == "__main__":
    main()
