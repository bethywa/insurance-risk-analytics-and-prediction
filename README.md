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