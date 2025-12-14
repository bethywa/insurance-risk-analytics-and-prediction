import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import re

def load_data(file_path='../data/processed/insurance_cleaned.csv'):
    """
    Load the cleaned insurance data.
    
    Parameters:
    -----------
    file_path : str, optional
        Path to the cleaned data file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded DataFrame
    """
    # Ensure the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find data file at {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path)
    print(f"Loaded data with shape: {df.shape}")
    return df

def preprocess_dates(df):
    """
    Convert date columns to datetime and then to numerical format.
    Returns the processed DataFrame and a list of date column names.
    """
    df = df.copy()
    date_columns = []
    
    # Check each column for date-like strings
    for col in df.select_dtypes(include=['object']).columns:
        # Check if column contains date-like strings
        if df[col].dropna().empty:
            continue
        sample = df[col].dropna().iloc[0]
        if isinstance(sample, str) and re.match(r'\d{4}-\d{2}-\d{2}', str(sample)):
            date_columns.append(col)
            # Convert to datetime and then to Unix timestamp
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].astype('int64') / 10**9  # Convert to seconds
    
    print(f"Processed {len(date_columns)} date columns: {date_columns}")
    return df, date_columns

def prepare_data(X, y_reg, y_clf, test_size=0.2, random_state=42):
    """
    Prepare data for modeling by splitting into train/test sets and preprocessing.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y_reg : pandas.Series
        Target variable for regression
    y_clf : pandas.Series
        Target variable for classification
    test_size : float, optional
        Proportion of data to use for testing
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, feature_names
    """
    # Preprocess dates first
    X_processed, date_columns = preprocess_dates(X)
    
    # Identify numeric and categorical columns
    numeric_cols = X_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    if date_columns:
        print(f"Processed date columns as numeric: {date_columns}")
    
    # Convert categorical columns to string type
    for col in categorical_cols:
        X_processed[col] = X_processed[col].astype(str)
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Split data
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X_processed, y_reg, y_clf, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_clf
    )
    
    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names
    numeric_features = numeric_cols
    categorical_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([numeric_features, categorical_features])
    
    # Convert back to DataFrame with column names
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    
    return X_train_df, X_test_df, y_reg_train, y_reg_test, y_clf_train, y_clf_test, feature_names

# Keep the rest of your existing functions as they are
def get_feature_names(column_transformer):
    """
    Get feature names from a ColumnTransformer.
    """
    output_features = []
    
    for name, transformer, features in column_transformer.transformers_:
        if name == 'remainder':
            continue
        if hasattr(transformer, 'get_feature_names_out'):
            output_features.extend(transformer.get_feature_names_out(features))
        else:
            output_features.extend(features)
            
    return output_features

def convert_categorical_to_numeric(X, categorical_cols=None):
    """
    Convert categorical columns to numeric codes.
    """
    X = X.copy()
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_cols:
        if col in X.columns:
            X[col] = pd.Categorical(X[col]).codes
    
    return X