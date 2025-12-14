🎯 Tasks Completed
✅ Task 1: EDA & Data Understanding
GitHub Branch: task-1

Performed comprehensive Exploratory Data Analysis
Created visualizations for trends and risk profiling
Analyzed monthly premiums vs claims
✅ Task 2: Data Version Control with DVC
GitHub Branch: task-2

Implemented DVC for data versioning
Tracked cleaned insurance data (127MB)
Configured .gitignore for large files
✅ Task 3: A/B Hypothesis Testing
GitHub Branch: task-3

Tested risk differences across provinces and postal codes
Analyzed claim frequency and severity
Generated statistical reports and visualizations
✅ Task 4: Risk Modeling & Premium Calculation
GitHub Branch: task-4

Implemented regression models (Linear, Random Forest, XGBoost)
Added SHAP-based model interpretation
Developed premium calculation with configurable margins
Generated model performance reports
   
     🚀 Quick Start
      # Clone repository
git clone https://github.com/bethywa/insurance-risk-analytics-and-prediction.git
cd insurance-risk-analytics-and-prediction

# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Pull data with DVC
dvc pull

# Run Jupyter notebook
jupyter notebook notebooks/task4_risk_modeling.ipynb
              
              
   # 📂 Project Structure

              insurance-risk-analytics/
├── data/
│   ├── processed/      # Cleaned data (tracked with DVC)
│   └── raw/            # Raw data files
├── models/             # Trained models (.joblib)
├── notebooks/          # Jupyter notebooks
├── reports/            # Reports and visualizations
└── src/                # Source code
    ├── data_preprocessing.py
    ├── modeling.py
    └── task3_ab_test/
    
    📊 Model Performance
Check reports/regression_metrics.csv for detailed performance metrics.

# 📝 Requirements
Python 3.8+

DVC (for data versioning)

Jupyter Notebook

See requirements.txt for full dependency list
