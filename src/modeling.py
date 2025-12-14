import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import shap

class ModelTrainer:
    def __init__(self, model_type='regression'):
        self.model_type = model_type.lower()
        self.models = {}
        self.feature_importances_ = {}
        self.metrics = {}
        
    def train_models(self, X_train, y_train, **kwargs):
        """Train models with parameters from kwargs"""
        if self.model_type == 'regression':
            self._train_regression_models(X_train, y_train, **kwargs)
        else:
            raise ValueError("Only 'regression' model type is currently supported")
    
    def _train_regression_models(self, X_train, y_train, **kwargs):
        """Train regression models with parameters from kwargs"""
        # Get model parameters from kwargs
        rf_params = kwargs.get('random_forest', {})
        xgb_params = kwargs.get('xgboost', {})
        
        # Train Linear Regression
        print("\nTraining Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models['Linear Regression'] = lr
        print("✓ Linear Regression trained")
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        rf_defaults = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        rf_params = {**rf_defaults, **rf_params}  # Update defaults with provided params
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        print("✓ Random Forest trained")
        
        # Train XGBoost
        print("\nTraining XGBoost...")
        xgb_defaults = {
            'n_estimators': 1000,  # Large number, we'll use early stopping
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        # Handle eval_set separately if provided
        eval_set = None
        if 'eval_set' in xgb_params:
            eval_set = xgb_params.pop('eval_set')
        
        xgb_params = {**xgb_defaults, **xgb_params}  # Update defaults with provided params
        xgb = XGBRegressor(**xgb_params)
        
        # Fit with or without validation set
        if eval_set:
            xgb.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=xgb_params.get('early_stopping_rounds', 10),
                verbose=10
            )
        else:
            xgb.fit(X_train, y_train)
        
        self.models['XGBoost'] = xgb
        print("✓ XGBoost trained")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test data."""
        results = {}
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                
                if self.model_type == 'regression':
                    results[name] = {
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'R-squared': r2_score(y_test, y_pred),
                        'MSE': mean_squared_error(y_test, y_pred)
                    }
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
                results[name] = {'Error': str(e)}
        
        self.metrics = results
        return pd.DataFrame.from_dict(results, orient='index')
    
    def get_feature_importances(self, feature_names=None, n_features=10):
        """Get and optionally plot feature importances."""
        importances = {}
        for name, model in self.models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    importances[name] = pd.Series(
                        model.feature_importances_,
                        index=feature_names
                    ).sort_values(ascending=False)
                elif hasattr(model, 'coef_'):
                    # Linear models
                    importances[name] = pd.Series(
                        np.abs(model.coef_.flatten()),
                        index=feature_names
                    ).sort_values(ascending=False)
                
                # Plot top features
                if n_features > 0 and name in importances:
                    plt.figure(figsize=(10, 6))
                    importances[name].head(n_features).plot(kind='barh')
                    plt.title(f'Top {n_features} Features - {name}')
                    plt.tight_layout()
                    plt.show()
                    
            except Exception as e:
                print(f"Could not get feature importances for {name}: {str(e)}")
        
        return importances
    
    def save_models(self, output_dir):
        """Save all trained models to disk."""
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.models.items():
            try:
                filename = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}.joblib")
                joblib.dump(model, filename)
                print(f"✓ Saved {name} to {filename}")
            except Exception as e:
                print(f"Error saving {name}: {str(e)}")
    
    def explain_with_shap(self, X, feature_names=None, model_name=None):
        """Generate SHAP explanations for model predictions."""
        if model_name and model_name in self.models:
            models = {model_name: self.models[model_name]}
        else:
            models = self.models
            
        for name, model in models.items():
            try:
                print(f"\nGenerating SHAP explanations for {name}...")
                
                # Sample data if too large for SHAP
                if len(X) > 1000:
                    X_sample = X.sample(1000, random_state=42)
                    print("  Using 1000 samples for SHAP explanation")
                else:
                    X_sample = X
                
                # Use TreeExplainer for tree models, else KernelExplainer
                if hasattr(model, 'predict_proba'):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    explainer = shap.KernelExplainer(model.predict, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                
                # Summary plot
                plt.figure()
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                plt.title(f'SHAP Summary - {name}')
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                print(f"Could not generate SHAP for {name}: {str(e)}")
    
    def calculate_premiums(self, X, expense_ratio=0.2, profit_margin=0.1):
        """Calculate risk-based premiums using the best model."""
        if not self.models:
            raise ValueError("No models trained yet. Call train_models() first.")
            
        # Use the best model based on R-squared
        best_model_name = max(self.metrics.items(), 
                             key=lambda x: x[1].get('R-squared', -np.inf))[0]
        best_model = self.models[best_model_name]
        
        # Predict claim amounts
        predicted_claims = best_model.predict(X)
        
        # Calculate premium: predicted claim amount + expenses + profit
        premiums = predicted_claims * (1 + expense_ratio + profit_margin)
        
        return pd.DataFrame({
            'predicted_claim': predicted_claims,
            'expense_loading': predicted_claims * expense_ratio,
            'profit_margin': predicted_claims * profit_margin,
            'premium': premiums
        })