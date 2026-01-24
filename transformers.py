"""
Custom transformers for the data pipeline
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List
import warnings
import re

# Suppress the specific dateutil warning
warnings.filterwarnings('ignore', message='Could not infer format')

class DataTypeHandler(BaseEstimator, TransformerMixin):
    """
    Transformer to check and enforce data types
    """
    def __init__(self, numerical_cols: Optional[List[str]] = None,
                 categorical_cols: Optional[List[str]] = None,
                 date_cols: Optional[List[str]] = None,
                 date_formats: Optional[dict] = None):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.date_cols = date_cols
        self.date_formats = date_formats or {}
        self.column_types = {}
        
    def fit(self, X: pd.DataFrame, y=None):
        """Learn column types from data"""
        if self.numerical_cols is None:
            self.numerical_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        
        if self.categorical_cols is None:
            self.categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        if self.date_cols is None:
            # Try to detect date columns
            date_candidates = []
            for col in X.columns:
                # Check column name for date/time indicators
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['date', 'time', 'datetime', 'timestamp', 'created', 'updated']):
                    date_candidates.append(col)
                elif X[col].dtype == 'object':
                    # Try to parse sample values as dates
                    sample_values = X[col].dropna().head(100)  # Check first 100 non-null values
                    if len(sample_values) > 0:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                # Try to parse as datetime
                                parsed = pd.to_datetime(sample_values, errors='coerce', infer_datetime_format=True)
                                # Consider it a date column if at least 80% of values can be parsed
                                if parsed.notna().sum() / len(sample_values) > 0.8:
                                    date_candidates.append(col)
                        except:
                            pass
            
            self.date_cols = list(set(date_candidates))  # Remove duplicates
        
        # Remove date columns from numerical/categorical if they were auto-detected there
        self.numerical_cols = [col for col in self.numerical_cols if col not in self.date_cols]
        self.categorical_cols = [col for col in self.categorical_cols if col not in self.date_cols]
        
        # Store original dtypes
        self.original_dtypes = X.dtypes.to_dict()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Enforce data types"""
        X_transformed = X.copy()
        
        # Handle date columns - suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            
            for col in self.date_cols:
                if col in X.columns:
                    try:
                        # Try common date formats
                        if col in self.date_formats:
                            X_transformed[col] = pd.to_datetime(
                                X_transformed[col], 
                                format=self.date_formats[col],
                                errors='coerce'
                            )
                        else:
                            # Try to infer format from first non-null value
                            sample = X_transformed[col].dropna().iloc[0] if X_transformed[col].notna().any() else None
                            if sample:
                                # Try to determine format from sample
                                format_str = self._infer_date_format(str(sample))
                                if format_str:
                                    X_transformed[col] = pd.to_datetime(
                                        X_transformed[col],
                                        format=format_str,
                                        errors='coerce'
                                    )
                                else:
                                    # Use pandas default parsing with infer_datetime_format
                                    X_transformed[col] = pd.to_datetime(
                                        X_transformed[col],
                                        infer_datetime_format=True,
                                        errors='coerce'
                                    )
                            else:
                                X_transformed[col] = pd.to_datetime(
                                    X_transformed[col],
                                    errors='coerce'
                                )
                    except Exception as e:
                        print(f"Warning: Could not convert column '{col}' to datetime: {e}")
                        # Keep as string if conversion fails
                        X_transformed[col] = X_transformed[col].astype(str)
        
        # Handle categorical columns
        for col in self.categorical_cols:
            if col in X.columns and col not in self.date_cols:
                # Convert to category only if not too many unique values (performance)
                unique_count = X_transformed[col].nunique()
                if unique_count <= 1000:  # Limit categories to 1000 unique values
                    X_transformed[col] = X_transformed[col].astype('category')
                else:
                    # Too many unique values, keep as object
                    X_transformed[col] = X_transformed[col].astype(str)
        
        # Handle numerical columns
        for col in self.numerical_cols:
            if col in X.columns:
                try:
                    # Try to convert to numeric, coerce errors to NaN
                    X_transformed[col] = pd.to_numeric(X_transformed[col], errors='coerce')
                    
                    # Optimize numeric type if possible
                    if X_transformed[col].notna().any():
                        col_min = X_transformed[col].min()
                        col_max = X_transformed[col].max()
                        
                        # Check if can be converted to integer
                        if pd.api.types.is_float_dtype(X_transformed[col]):
                            # Check if all values are essentially integers
                            if (X_transformed[col].dropna() % 1 == 0).all():
                                # Convert to appropriate integer type
                                if col_min >= 0:
                                    if col_max <= 255:
                                        X_transformed[col] = X_transformed[col].astype('uint8')
                                    elif col_max <= 65535:
                                        X_transformed[col] = X_transformed[col].astype('uint16')
                                    elif col_max <= 4294967295:
                                        X_transformed[col] = X_transformed[col].astype('uint32')
                                    else:
                                        X_transformed[col] = X_transformed[col].astype('uint64')
                                else:
                                    if col_min >= -128 and col_max <= 127:
                                        X_transformed[col] = X_transformed[col].astype('int8')
                                    elif col_min >= -32768 and col_max <= 32767:
                                        X_transformed[col] = X_transformed[col].astype('int16')
                                    elif col_min >= -2147483648 and col_max <= 2147483647:
                                        X_transformed[col] = X_transformed[col].astype('int32')
                                    else:
                                        X_transformed[col] = X_transformed[col].astype('int64')
                            else:
                                # Keep as float, but optimize precision
                                if np.abs(col_max) <= 3.4e38:
                                    X_transformed[col] = X_transformed[col].astype('float32')
                                else:
                                    X_transformed[col] = X_transformed[col].astype('float64')
                except Exception as e:
                    print(f"Warning: Could not convert column '{col}' to numeric: {e}")
        
        return X_transformed
    
    def _infer_date_format(self, date_str: str) -> Optional[str]:
        """Infer date format from a string"""
        if not isinstance(date_str, str):
            return None
        
        # Common date patterns
        patterns = {
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}': '%Y-%m-%d %H:%M:%S',
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}': '%Y-%m-%d %H:%M',
            r'\d{4}-\d{2}-\d{2}': '%Y-%m-%d',
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}': '%m/%d/%Y %H:%M:%S',
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}': '%m/%d/%Y %H:%M',
            r'\d{2}/\d{2}/\d{4}': '%m/%d/%Y',
            r'\d{2}-\d{2}-\d{4}': '%d-%m-%Y',
            r'\d{2}\.\d{2}\.\d{4}': '%d.%m.%Y',
            r'\d{4}/\d{2}/\d{2}': '%Y/%m/%d',
        }
        
        for pattern, format_str in patterns.items():
            if re.match(pattern, date_str):
                return format_str
        
        return None
    
    def get_dtype_report(self, X: pd.DataFrame) -> dict:
        """Generate data type report"""
        report = {
            'original_dtypes': {col: str(dtype) for col, dtype in X.dtypes.items()},
            'proposed_dtypes': {
                'numerical': self.numerical_cols,
                'categorical': self.categorical_cols,
                'date': self.date_cols
            },
            'memory_usage_before': X.memory_usage(deep=True).sum() / (1024**2),  # MB
        }
        
        # Get memory usage after transformation
        X_transformed = self.transform(X)
        report['memory_usage_after'] = X_transformed.memory_usage(deep=True).sum() / (1024**2)
        report['memory_savings'] = report['memory_usage_before'] - report['memory_usage_after']
        
        return report


class SmartNullHandler(BaseEstimator, TransformerMixin):
    """
    Intelligently handle missing values with smart imputation logic
    """
    def __init__(self, drop_threshold: float = 0.5, 
                 impute_strategy: str = 'median',
                 numeric_impute: str = 'median',
                 categorical_impute: str = 'mode'):
        self.drop_threshold = drop_threshold
        self.impute_strategy = impute_strategy
        self.numeric_impute = numeric_impute
        self.categorical_impute = categorical_impute
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
                    if self.numeric_impute == 'mean' or self.impute_strategy == 'mean':
                        self.imputation_values[col] = X[col].mean()
                    elif self.numeric_impute == 'median' or self.impute_strategy == 'median':
                        self.imputation_values[col] = X[col].median()
                    elif self.numeric_impute == 'mode' or self.impute_strategy == 'most_frequent':
                        mode_values = X[col].mode()
                        self.imputation_values[col] = mode_values[0] if not mode_values.empty else 0
                    else:
                        # Default to median
                        self.imputation_values[col] = X[col].median()
                elif pd.api.types.is_datetime64_any_dtype(X[col]):
                    # For datetime columns, use the most recent date or median
                    if self.impute_strategy == 'median':
                        # Convert to timestamp for median calculation
                        timestamps = X[col].dropna().astype('int64')
                        if len(timestamps) > 0:
                            median_timestamp = timestamps.median()
                            self.imputation_values[col] = pd.Timestamp(median_timestamp)
                        else:
                            self.imputation_values[col] = pd.Timestamp.now()
                    else:
                        # Default to most recent date
                        recent_dates = X[col].dropna()
                        self.imputation_values[col] = recent_dates.max() if len(recent_dates) > 0 else pd.Timestamp.now()
                else:
                    # For categorical/object columns
                    if self.categorical_impute == 'mode' or self.impute_strategy == 'most_frequent':
                        mode_values = X[col].mode()
                        self.imputation_values[col] = mode_values[0] if not mode_values.empty else 'Unknown'
                    else:
                        # Default to 'Unknown'
                        self.imputation_values[col] = 'Unknown'
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        X_transformed = X.copy()
        
        # Drop columns with too many nulls
        columns_to_drop_actual = [col for col in self.columns_to_drop if col in X_transformed.columns]
        X_transformed = X_transformed.drop(columns=columns_to_drop_actual)
        
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
            'imputation_values': {k: str(v) for k, v in self.imputation_values.items()},
            'null_summary': {
                'counts': null_counts.to_dict(),
                'percentages': null_percentage.round(2).to_dict()
            }
        }
        
        # Calculate additional statistics
        columns_with_nulls = null_counts[null_counts > 0]
        report['columns_with_nulls'] = len(columns_with_nulls)
        report['null_distribution'] = {
            '0%': len(null_percentage[null_percentage == 0]),
            '1-10%': len(null_percentage[(null_percentage > 0) & (null_percentage <= 10)]),
            '10-50%': len(null_percentage[(null_percentage > 10) & (null_percentage <= 50)]),
            '>50%': len(null_percentage[null_percentage > 50])
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
        numerical_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32', 'uint8', 'int8']).columns
        
        for col in numerical_cols:
            col_data = X[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Only calculate bounds if IQR is meaningful
                    lower_bound = Q1 - (self.factor * IQR)
                    upper_bound = Q3 + (self.factor * IQR)
                    
                    self.bounds[col] = {
                        'lower': lower_bound,
                        'upper': upper_bound,
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'q1': Q1,
                        'q3': Q3,
                        'iqr': IQR,
                        'mean': col_data.mean(),
                        'std': col_data.std()
                    }
                else:
                    # If IQR is 0, use standard deviation method
                    mean = col_data.mean()
                    std = col_data.std()
                    if std > 0:
                        lower_bound = mean - (3 * std)
                        upper_bound = mean + (3 * std)
                        
                        self.bounds[col] = {
                            'lower': lower_bound,
                            'upper': upper_bound,
                            'min': col_data.min(),
                            'max': col_data.max(),
                            'q1': Q1,
                            'q3': Q3,
                            'iqr': IQR,
                            'mean': mean,
                            'std': std,
                            'method': 'std'
                        }
                    else:
                        # All values are the same, no outliers
                        self.bounds[col] = {
                            'lower': col_data.min(),
                            'upper': col_data.max(),
                            'min': col_data.min(),
                            'max': col_data.max(),
                            'q1': Q1,
                            'q3': Q3,
                            'iqr': IQR,
                            'mean': mean,
                            'std': std,
                            'method': 'no_variation'
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
                elif self.clip_method == 'winsorize':
                    # Winsorize: replace outliers with bounds
                    X_transformed[col] = np.where(
                        X_transformed[col] < bounds['lower'],
                        bounds['lower'],
                        np.where(
                            X_transformed[col] > bounds['upper'],
                            bounds['upper'],
                            X_transformed[col]
                        )
                    )
        
        return X_transformed
    
    def get_outlier_report(self, X: pd.DataFrame) -> dict:
        """Generate outlier statistics report"""
        report = {}
        
        for col, bounds in self.bounds.items():
            if col in X.columns:
                col_data = X[col].dropna()
                if len(col_data) > 0:
                    outliers_mask = (col_data < bounds['lower']) | (col_data > bounds['upper'])
                    outlier_count = outliers_mask.sum()
                    outlier_percentage = (outlier_count / len(col_data)) * 100 if len(col_data) > 0 else 0
                    
                    # Calculate statistics for outliers
                    outlier_values = col_data[outliers_mask]
                    outlier_stats = {
                        'count': int(outlier_count),
                        'percentage': round(outlier_percentage, 2),
                        'min': float(outlier_values.min()) if outlier_count > 0 else None,
                        'max': float(outlier_values.max()) if outlier_count > 0 else None,
                        'mean': float(outlier_values.mean()) if outlier_count > 0 else None,
                        'std': float(outlier_values.std()) if outlier_count > 0 else None
                    }
                    
                    report[col] = {
                        'bounds': {'lower': float(bounds['lower']), 'upper': float(bounds['upper'])},
                        'original_range': {'min': float(bounds['min']), 'max': float(bounds['max'])},
                        'quartiles': {'q1': float(bounds['q1']), 'q3': float(bounds['q3'])},
                        'iqr': float(bounds['iqr']),
                        'method': bounds.get('method', 'iqr'),
                        'outliers': outlier_stats,
                        'non_outlier_stats': {
                            'count': len(col_data) - outlier_count,
                            'min': float(col_data[~outliers_mask].min()) if len(col_data) > outlier_count else None,
                            'max': float(col_data[~outliers_mask].max()) if len(col_data) > outlier_count else None,
                            'mean': float(col_data[~outliers_mask].mean()) if len(col_data) > outlier_count else None
                        }
                    }
        
        # Add summary statistics
        total_outliers = sum(stats['outliers']['count'] for stats in report.values())
        columns_with_outliers = sum(1 for stats in report.values() if stats['outliers']['count'] > 0)
        
        report['_summary'] = {
            'total_outliers': total_outliers,
            'columns_with_outliers': columns_with_outliers,
            'total_columns_analyzed': len(report),
            'outlier_factor': self.factor,
            'clip_method': self.clip_method
        }
        
        return report