"""
Pipeline manager for orchestrating the data cleaning process
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from transformers import DataTypeHandler, SmartNullHandler, OutlierHandler
from config import settings
from datetime import datetime
import json

class PipelineManager:
    """
    Manages the complete data cleaning pipeline
    """
    def __init__(self, config=None):
        self.config = config or settings
        self.pipeline = None
        self.reports = {}
        self.data_preview = None
        
    def get_data_cleaning_pipeline(self) -> Pipeline:
        """Create and return the complete pipeline"""
        self.pipeline = Pipeline([
            ('data_type_handler', DataTypeHandler(
                numerical_cols=self.config.NUMERICAL_COLUMNS,
                categorical_cols=self.config.CATEGORICAL_COLUMNS,
                date_cols=self.config.DATE_COLUMNS
            )),
            ('null_handler', SmartNullHandler(
                drop_threshold=self.config.DROP_THRESHOLD,
                impute_strategy=self.config.IMPUTE_STRATEGY
            )),
            ('outlier_handler', OutlierHandler(
                factor=self.config.OUTLIER_FACTOR
            ))
        ])
        return self.pipeline
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load data from file"""
        path = file_path or self.config.DATA_PATH
        
        try:
            if path.endswith('.csv'):
                data = pd.read_csv(path)
            elif path.endswith('.xlsx'):
                data = pd.read_excel(path)
            elif path.endswith('.json'):
                data = pd.read_json(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            print(f"✓ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        
        except FileNotFoundError:
            print(f"❌ File not found: {path}")
            print("Please create a sample data file or update DATA_PATH in config.py")
            print("Creating sample data for demonstration...")
            return self._create_sample_data()
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'customer_id': range(1, n_samples + 1),
            'age': np.random.normal(35, 10, n_samples).astype(int),
            'salary': np.random.normal(50000, 15000, n_samples),
            'purchase_amount': np.random.exponential(100, n_samples),
            'city': np.random.choice(['New York', 'London', 'Tokyo', 'Paris', None], n_samples),
            'membership_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'is_premium': np.random.choice([True, False, None], n_samples),
            'rating': np.random.choice(['A', 'B', 'C', None], n_samples),
            'score': np.random.uniform(0, 100, n_samples),
            'visits': np.random.poisson(5, n_samples)
        })
        
        # Add some null values
        null_mask = np.random.random(data.shape) < 0.05
        data = data.mask(null_mask)
        
        # Add some outliers
        outlier_indices = np.random.choice(n_samples, size=20, replace=False)
        data.loc[outlier_indices, 'salary'] *= 5
        data.loc[outlier_indices, 'purchase_amount'] *= 10
        
        # Save sample data
        data.to_csv(self.config.DATA_PATH, index=False)
        print(f"✓ Sample data created: {self.config.DATA_PATH}")
        
        return data
    
    def check_data_schema(self, data: pd.DataFrame) -> dict:
        """Check and report data types"""
        dtype_handler = DataTypeHandler(
            numerical_cols=self.config.NUMERICAL_COLUMNS,
            categorical_cols=self.config.CATEGORICAL_COLUMNS
        )
        dtype_handler.fit(data)
        
        report = dtype_handler.get_dtype_report(data)
        self.reports['schema'] = report
        
        print("\n" + "="*60)
        print("DATA SCHEMA REPORT")
        print("="*60)
        
        print("\nOriginal Data Types:")
        for col, dtype in report['original_dtypes'].items():
            print(f"  {col}: {dtype}")
        
        print("\nProposed Data Types:")
        for dtype_type, cols in report['proposed_dtypes'].items():
            if cols:
                print(f"\n  {dtype_type.upper()}:")
                for col in cols:
                    print(f"    • {col}")
        
        return report
    
    def enforce_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enforce proper data types"""
        dtype_handler = DataTypeHandler(
            numerical_cols=self.config.NUMERICAL_COLUMNS,
            categorical_cols=self.config.CATEGORICAL_COLUMNS
        )
        dtype_handler.fit(data)
        transformed_data = dtype_handler.transform(data)
        
        print("\n" + "="*60)
        print("DATA TYPE ENFORCEMENT")
        print("="*60)
        
        print("\nUpdated Data Types:")
        for col, dtype in transformed_data.dtypes.items():
            print(f"  {col}: {dtype}")
        
        return transformed_data
    
    def missing_values_report(self, data: pd.DataFrame) -> dict:
        """Generate missing values quality audit"""
        null_handler = SmartNullHandler(
            drop_threshold=self.config.DROP_THRESHOLD
        )
        null_handler.fit(data)
        
        report = null_handler.get_null_report(data)
        self.reports['missing_values'] = report
        
        print("\n" + "="*60)
        print("MISSING VALUES QUALITY AUDIT")
        print("="*60)
        
        print(f"\nDataset Summary:")
        print(f"  • Total Rows: {report['total_rows']:,}")
        print(f"  • Total Columns: {report['total_columns']}")
        print(f"  • Total Null Values: {report['total_nulls']:,}")
        
        if report['columns_to_drop']:
            print(f"\n⚠️  Columns to drop (> {self.config.DROP_THRESHOLD*100:.0f}% nulls):")
            for col in report['columns_to_drop']:
                null_pct = report['null_summary']['percentages'][col]
                print(f"    • {col}: {null_pct:.1f}% missing")
        
        print(f"\nMissing Values by Column:")
        for col, null_pct in report['null_summary']['percentages'].items():
            if null_pct > 0:
                print(f"  • {col}: {null_pct:.1f}% missing ({report['null_summary']['counts'][col]:,} values)")
        
        return report
    
    def handle_nulls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle null values with smart imputation"""
        null_handler = SmartNullHandler(
            drop_threshold=self.config.DROP_THRESHOLD,
            impute_strategy=self.config.IMPUTE_STRATEGY
        )
        null_handler.fit(data)
        transformed_data = null_handler.transform(data)
        
        print("\n" + "="*60)
        print("NULL VALUE HANDLING")
        print("="*60)
        
        remaining_nulls = transformed_data.isnull().sum().sum()
        if remaining_nulls == 0:
            print("✓ All null values have been handled!")
        else:
            print(f"⚠️  {remaining_nulls:,} null values remain")
        
        print(f"\nColumns dropped: {len(null_handler.columns_to_drop)}")
        print(f"Columns imputed: {len(null_handler.imputation_values)}")
        
        return transformed_data
    
    def outlier_statistics(self, data: pd.DataFrame) -> dict:
        """Calculate outlier statistics"""
        outlier_handler = OutlierHandler(
            factor=self.config.OUTLIER_FACTOR
        )
        outlier_handler.fit(data)
        
        report = outlier_handler.get_outlier_report(data)
        self.reports['outliers'] = report
        
        print("\n" + "="*60)
        print("OUTLIER STATISTICS")
        print("="*60)
        
        total_outliers = 0
        for col, stats in report.items():
            outliers = stats['outliers']
            total_outliers += outliers['count']
            
            print(f"\n{col}:")
            print(f"  • Original Range: [{stats['original_range']['min']:.2f}, {stats['original_range']['max']:.2f}]")
            print(f"  • IQR Bounds: [{stats['bounds']['lower']:.2f}, {stats['bounds']['upper']:.2f}]")
            print(f"  • Outliers: {outliers['count']:,} ({outliers['percentage']:.1f}%)")
        
        print(f"\n📊 Total outliers detected: {total_outliers:,}")
        
        return report
    
    def clip_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers using IQR method"""
        outlier_handler = OutlierHandler(
            factor=self.config.OUTLIER_FACTOR
        )
        outlier_handler.fit(data)
        transformed_data = outlier_handler.transform(data)
        
        print("\n" + "="*60)
        print("OUTLIER CLIPPING")
        print("="*60)
        
        print("✓ Outliers have been clipped to IQR bounds")
        
        for col, bounds in outlier_handler.bounds.items():
            if col in transformed_data.columns:
                print(f"\n{col}:")
                print(f"  • New Range: [{transformed_data[col].min():.2f}, {transformed_data[col].max():.2f}]")
                print(f"  • Clipped to: [{bounds['lower']:.2f}, {bounds['upper']:.2f}]")
        
        return transformed_data
    
    def preview_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preview current state of data"""
        self.data_preview = data
        
        print("\n" + "="*60)
        print("DATA PREVIEW")
        print("="*60)
        
        print(f"\nDataset Shape: {data.shape[0]} rows × {data.shape[1]} columns")
        
        print(f"\nFirst {self.config.PREVIEW_ROWS} rows:")
        pd.set_option('display.max_columns', self.config.MAX_COLS_TO_DISPLAY)
        print(data.head(self.config.PREVIEW_ROWS))
        
        print(f"\nData Types:")
        for col, dtype in data.dtypes.items():
            print(f"  {col}: {dtype}")
        
        print(f"\nSummary Statistics:")
        if data.select_dtypes(include=[np.number]).shape[1] > 0:
            print(data.describe())
        
        return data
    
    def run_full_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run end-to-end automated pipeline"""
        print("\n" + "="*60)
        print("RUNNING FULL AUTOMATED PIPELINE")
        print("="*60)
        
        pipeline = self.get_data_cleaning_pipeline()
        cleaned_data = pipeline.fit_transform(data)
        
        # Generate final report
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'original_shape': data.shape,
            'cleaned_shape': cleaned_data.shape,
            'columns_removed': list(set(data.columns) - set(cleaned_data.columns)),
            'pipeline_steps': [name for name, _ in pipeline.steps],
            'config': {
                'outlier_factor': self.config.OUTLIER_FACTOR,
                'drop_threshold': self.config.DROP_THRESHOLD,
                'impute_strategy': self.config.IMPUTE_STRATEGY
            }
        }
        
        self.reports['final'] = final_report
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"   • Original: {data.shape[0]:,} rows × {data.shape[1]:,} columns")
        print(f"   • Cleaned: {cleaned_data.shape[0]:,} rows × {cleaned_data.shape[1]:,} columns")
        print(f"   • Columns removed: {len(final_report['columns_removed'])}")
        
        return cleaned_data
    
    def export_clean_dataset(self, data: pd.DataFrame, file_path: str = None):
        """Export cleaned dataset to CSV"""
        path = file_path or self.config.OUTPUT_PATH
        
        try:
            data.to_csv(path, index=False)
            print(f"\n✅ Cleaned dataset exported to: {path}")
            print(f"   • File size: {data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            print(f"   • Rows: {data.shape[0]:,}")
            print(f"   • Columns: {data.shape[1]:,}")
            
            # Save reports
            self._save_reports()
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
    
    def _save_reports(self):
        """Save all generated reports"""
        report_path = self.config.REPORT_PATH
        
        try:
            with open(report_path, 'w') as f:
                f.write("DATA PIPELINE QUALITY REPORT\n")
                f.write("=" * 40 + "\n\n")
                
                for report_name, report_data in self.reports.items():
                    f.write(f"\n{report_name.upper()} REPORT\n")
                    f.write("-" * 30 + "\n")
                    f.write(json.dumps(report_data, indent=2, default=str))
                    f.write("\n\n")
            
            print(f"✓ Quality report saved to: {report_path}")
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def exit_system(self):
        """Exit the system gracefully"""
        print("\n" + "="*60)
        print("EXITING DATA PIPELINE SYSTEM")
        print("="*60)
        
        print("\n📊 Pipeline Summary:")
        if 'final' in self.reports:
            report = self.reports['final']
            print(f"  • Processed: {report['original_shape'][0]:,} rows")
            print(f"  • Cleaned: {report['cleaned_shape'][0]:,} rows")
            print(f"  • Columns removed: {len(report['columns_removed'])}")
        
        print("\n👋 Thank you for using the Modern Data Analysis Pipeline!")
        print("Files saved:")
        print(f"  • Cleaned data: {self.config.OUTPUT_PATH}")
        print(f"  • Quality report: {self.config.REPORT_PATH}")
        
        exit(0)