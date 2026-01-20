"""
Custom transformers for the data pipeline
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List
import warnings

class DataTypeHandler(BaseEstimator, TransformerMixin):
    """
    Transformer to check and enforce data types
    """
    def __init__(self, numerical_cols: Optional[List[str]] = None,
                 categorical_cols: Optional[List[str]] = None,
                 date_cols: Optional[List[str]] = None):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.date_cols = date_cols
        self.column_types = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Learn column types from data"""
        if self.numerical_cols is None:
            self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if self.categorical_cols is None:
            self.categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        if self.date_cols is None:
            # Try to detect date columns
            date_candidates = []
            for col in X.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_candidates.append(col)
                elif X[col].dtype == 'object':
                    try:
                        pd.to_datetime(X[col], errors='raise')
                        date_candidates.append(col)
                    except:
                        pass
            self.date_cols = date_candidates
        
        # Store original dtypes
        self.original_dtypes = X.dtypes.to_dict()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Enforce data types"""
        X_transformed = X.copy()
        
        # Handle date columns
        for col in self.date_cols:
            if col in X.columns:
                try:
                    X_transformed[col] = pd.to_datetime(X_transformed[col], errors='coerce')
                except:
                    warnings.warn(f"Could not convert column '{col}' to datetime")
        
        # Handle categorical columns
        for col in self.categorical_cols:
            if col in X.columns and col not in self.date_cols:
                X_transformed[col] = X_transformed[col].astype('category')
        
        # Handle numerical columns
        for col in self.numerical_cols:
            if col in X.columns:
                try:
                    X_transformed[col] = pd.to_numeric(X_transformed[col], errors='coerce')
                except:
                    warnings.warn(f"Could not convert column '{col}' to numeric")
        
        return X_transformed
    
    def get_dtype_report(self, X: pd.DataFrame) -> dict:
        """Generate data type report"""
        report = {
            'original_dtypes': dict(X.dtypes),
            'proposed_dtypes': {
                'numerical': self.numerical_cols,
                'categorical': self.categorical_cols,
                'date': self.date_cols
            }
        }
        return report


class SmartNullHandler(BaseEstimator, TransformerMixin):
    """
    Intelligently handle missing values with smart imputation logic
    """
    def __init__(self, drop_threshold: float = 0.5, 
                 impute_strategy: str = 'median'):
        self.drop_threshold = drop_threshold
        self.impute_strategy = impute_strategy
        self.columns_to_drop = []
        self.imputation_values = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Learn which columns to drop and imputation values"""
        # Identify columns with too many nulls
        null_percentage = X.isnull().mean()
        self.columns_to_drop = null_percentage[null_percentage > self.drop_threshold].index.tolist()
        
        # Calculate imputation values for remaining columns
        for col in X.columns:
            if col not in self.columns_to_drop and X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    if self.impute_strategy == 'mean':
                        self.imputation_values[col] = X[col].mean()
                    elif self.impute_strategy == 'median':
                        self.imputation_values[col] = X[col].median()
                    else:
                        self.imputation_values[col] = X[col].mode()[0] if not X[col].mode().empty else 0
                else:
                    # For categorical columns, use mode
                    mode_values = X[col].mode()
                    self.imputation_values[col] = mode_values[0] if not mode_values.empty else 'Unknown'
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        X_transformed = X.copy()
        
        # Drop columns with too many nulls
        X_transformed = X_transformed.drop(columns=self.columns_to_drop)
        
        # Impute remaining nulls
        for col, value in self.imputation_values.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].fillna(value)
        
        return X_transformed
    
    def get_null_report(self, X: pd.DataFrame) -> dict:
        """Generate missing values report"""
        null_counts = X.isnull().sum()
        null_percentage = (null_counts / len(X)) * 100
        
        report = {
            'total_rows': len(X),
            'total_columns': len(X.columns),
            'total_nulls': null_counts.sum(),
            'columns_to_drop': self.columns_to_drop,
            'null_summary': {
                'counts': null_counts.to_dict(),
                'percentages': null_percentage.round(2).to_dict()
            }
        }
        return report


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Detect and handle outliers using IQR method
    """
    def __init__(self, factor: float = 1.5, clip_method: str = 'iqr'):
        self.factor = factor
        self.clip_method = clip_method
        self.bounds = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Calculate outlier bounds for numerical columns"""
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numerical_cols:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - (self.factor * IQR)
            upper_bound = Q3 + (self.factor * IQR)
            
            self.bounds[col] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'min': X[col].min(),
                'max': X[col].max(),
                'q1': Q1,
                'q3': Q3,
                'iqr': IQR
            }
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers"""
        X_transformed = X.copy()
        
        for col, bounds in self.bounds.items():
            if col in X.columns:
                if self.clip_method == 'iqr':
                    X_transformed[col] = X_transformed[col].clip(
                        lower=bounds['lower'],
                        upper=bounds['upper']
                    )
                elif self.clip_method == 'remove':
                    mask = (X_transformed[col] >= bounds['lower']) & (X_transformed[col] <= bounds['upper'])
                    X_transformed = X_transformed[mask]
        
        return X_transformed
    
    def get_outlier_report(self, X: pd.DataFrame) -> dict:
        """Generate outlier statistics report"""
        report = {}
        
        for col, bounds in self.bounds.items():
            if col in X.columns:
                outliers_mask = (X[col] < bounds['lower']) | (X[col] > bounds['upper'])
                outlier_count = outliers_mask.sum()
                outlier_percentage = (outlier_count / len(X)) * 100
                
                report[col] = {
                    'bounds': {'lower': bounds['lower'], 'upper': bounds['upper']},
                    'original_range': {'min': bounds['min'], 'max': bounds['max']},
                    'quartiles': {'q1': bounds['q1'], 'q3': bounds['q3']},
                    'outliers': {
                        'count': int(outlier_count),
                        'percentage': round(outlier_percentage, 2)
                    }
                }
        
        return report