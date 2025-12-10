# Insurance Risk Analytics & Prediction

## 📊 Project Overview
Analysis of South African car insurance data to identify low-risk customer segments and optimize premium pricing for AlphaCare Insurance Solutions.

## 🎯 Tasks Completed

### ✅ Task 1: EDA & Data Understanding
**GitHub Branch:** `task-1`

**What was done:**
1. Set up project repository with CI/CD workflow
2. Performed comprehensive Exploratory Data Analysis
3. Created visualizations including:
   - Monthly Premiums vs Claims trends
   - Provincial risk analysis
   - Vehicle type risk profiling
4. Answered key business questions about loss ratios, outliers, and temporal trends


---

### ✅ Task 2: Data Version Control with DVC
**GitHub Branch:** `task-2`

**What was done:**
1. Implemented DVC for data versioning
2. Set up local storage for large dataset management
3. Tracked cleaned insurance data (127MB) using DVC
4. Configured `.gitignore` to exclude large files from Git



## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/bethywa/insurance-risk-analytics-and-prediction.git

# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Pull data with DVC
dvc pull

# Run analysis
jupyter notebook notebooks/eda_analysis.ipynb



  # Task 3 — A/B Hypothesis Testing (Short README)

**Goal (one line):**
Statistically validate / reject risk-related hypotheses using the cleaned insurance dataset.

**Dataset (input):**

* `data/processed/insurance_cleaned.csv` (cleaned & preprocessed) — ensure this file exists.

**Key metrics (KPIs):**

* **Claim Frequency**: proportion of policies with `TotalClaims > 0` (per group)
* **Claim Severity**: average `TotalClaims` among policies with `TotalClaims > 0`
* **Margin**: `TotalPremium - TotalClaims` (average per group)

**Hypotheses to test (null = no difference):**

1. H₀ — no risk differences across **provinces** (frequency/severity/margin)
2. H₀ — no risk differences between **postal/zip codes** (top-N areas)
3. H₀ — no margin (profit) difference between postal/zip codes
4. H₀ — no risk difference between **genders**

## Quick step-by-step (what I did 

1. Create a branch for Task 3: `git checkout -b task-3`
2. Make sure cleaned data exists at `data/processed/insurance_cleaned.csv` (or update `DATA_PATH` in scripts).
3. Reproduce/Run tests script (example):

   ```bash
   python src/task3_ab_test/run_ab_tests.py
   ```

   * Script outputs per-group tables and test results (chi-square, Levene, t-test / Kruskal-Wallis).
4. Visualize results (recommended notebooks):

   * `notebooks/task3_ab_test.ipynb` — loads results, plots: frequency bar per province, severity boxplots, margin violin/box for postal codes, and a gender comparison chart.
5. Document findings: for each rejected H₀ report — `What?`, `So what?`, `Now what?` (short business recommendation).

## Tests & Methods (mapping -> when to use)

* **Claim frequency (categorical counts)**: Chi-square test across groups.
* **Claim severity / margin (continuous, likely non-normal)**:

  * Check variance with **Levene** test.
  * If normal and equal-variance → use **ANOVA** / **t-test**.
  * If not → use **Kruskal–Wallis** (non-parametric) or Welch `t-test` for two groups.

## Example expected outputs

* Group summary table: `province | total_policies | claims_count | freq | mean_severity | mean_margin`
* Tests: `chi2` and `p-value` for frequency; `stat` and `p-value` for severity/margin.
* Interpretation guidance: reject H₀ when `p < 0.05`, then measure effect size (Cohen's d or differences in means/proportions).

## Deliverables for Task 3 (minimum)

* `src/task3_ab_test/run_ab_tests.py` (script) — reproducible tests
* `notebooks/task3_ab_test.ipynb` — interactive analysis + charts
* Short report / README (this file) with: findings, top 2 actionable recommendations per rejected hypothesis, and tracked code on branch `task-3`.

## Quick tips

* Always check group sample sizes — small groups give unstable estimates.
* Report both p-values and effect sizes (so results are actionable).
* If you aggregate postal codes, use top-N by policy count (e.g., top 10) to keep tests meaningful.


