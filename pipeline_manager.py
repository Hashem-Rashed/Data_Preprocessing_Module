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
import os
import warnings

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
                # Try reading with optimized parameters
                try:
                    data = pd.read_csv(path)
                except MemoryError:
                    # Try with lower memory usage
                    data = pd.read_csv(path, low_memory=False)
                except Exception:
                    # Try with different encoding
                    data = pd.read_csv(path, encoding='latin-1')
            elif path.endswith('.xlsx'):
                data = pd.read_excel(path)
            elif path.endswith('.json'):
                data = pd.read_json(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            print(f"✓ Data loaded successfully: {data.shape[0]:,} rows, {data.shape[1]:,} columns")
            return data
        
        except FileNotFoundError:
            print(f"❌ File not found: {path}")
            print("Please create a sample data file or update DATA_PATH in config.py")
            return None
                
        except MemoryError:
            print(f"❌ Memory error loading file: {path}")
            print("The file is too large for available memory.")
            print("Suggestions:")
            print("1. Use a smaller subset of the data")
            print("2. Increase system memory")
            print("3. Use chunked reading (see load_data_in_chunks method)")
            return None
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print(f"File path: {path}")
            return None
    
    def load_data_in_chunks(self, file_path: str = None, chunk_size: int = 10000, max_chunks: int = None) -> pd.DataFrame:
        """Load large data in chunks"""
        path = file_path or self.config.DATA_PATH
        
        try:
            if not path.endswith('.csv'):
                raise ValueError("Chunked reading only supports CSV files")
            
            print(f"Loading {path} in chunks of {chunk_size:,} rows...")
            
            chunks = []
            total_rows = 0
            
            for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size, low_memory=False)):
                chunks.append(chunk)
                total_rows += len(chunk)
                print(f"  Loaded chunk {i+1}: {len(chunk):,} rows (Total: {total_rows:,})")
                
                # Stop if we've reached max_chunks
                if max_chunks is not None and i >= max_chunks - 1:
                    print(f"  Stopping at {max_chunks} chunks as requested")
                    break
            
            if not chunks:
                print("❌ No data loaded from file")
                return None
            
            data = pd.concat(chunks, ignore_index=True)
            print(f"✓ Data loaded successfully: {data.shape[0]:,} rows, {data.shape[1]:,} columns")
            print(f"  Memory usage: {data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading data in chunks: {e}")
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
        try:
            data.to_csv(self.config.DATA_PATH, index=False)
            print(f"✓ Sample data created: {self.config.DATA_PATH}")
        except Exception as e:
            print(f"Note: Could not save sample data: {e}")
        
        return data
    
    def check_data_schema(self, data: pd.DataFrame) -> dict:
        """Check and report data types"""
        try:
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
            
            print(f"\nDataset Shape: {data.shape[0]:,} rows × {data.shape[1]:,} columns")
            
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
        except Exception as e:
            print(f"❌ Error checking data schema: {e}")
            return {'error': str(e)}
    
    def enforce_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enforce proper data types"""
        try:
            dtype_handler = DataTypeHandler(
                numerical_cols=self.config.NUMERICAL_COLUMNS,
                categorical_cols=self.config.CATEGORICAL_COLUMNS
            )
            dtype_handler.fit(data)
            transformed_data = dtype_handler.transform(data)
            
            print("\n" + "="*60)
            print("DATA TYPE ENFORCEMENT")
            print("="*60)
            
            print(f"\nUpdated Data Types:")
            dtype_summary = transformed_data.dtypes.value_counts()
            for dtype, count in dtype_summary.items():
                print(f"  {dtype}: {count} columns")
            
            # Show specific column changes
            original_dtypes = data.dtypes
            new_dtypes = transformed_data.dtypes
            
            changed_columns = []
            for col in data.columns:
                if col in new_dtypes and original_dtypes[col] != new_dtypes[col]:
                    changed_columns.append((col, original_dtypes[col], new_dtypes[col]))
            
            if changed_columns:
                print(f"\nColumns with type changes:")
                for col, old_type, new_type in changed_columns[:10]:  # Show first 10
                    print(f"  • {col}: {old_type} → {new_type}")
                if len(changed_columns) > 10:
                    print(f"  ... and {len(changed_columns) - 10} more")
            
            return transformed_data
        except Exception as e:
            print(f"❌ Error enforcing data types: {e}")
            return data
    
    def missing_values_report(self, data: pd.DataFrame) -> dict:
        """Generate missing values quality audit"""
        try:
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
            
            total_cells = report['total_rows'] * report['total_columns']
            null_percentage = (report['total_nulls'] / total_cells * 100) if total_cells > 0 else 0
            print(f"  • Null Percentage: {null_percentage:.2f}%")
            
            if report['columns_to_drop']:
                print(f"\n⚠️  Columns to drop (> {self.config.DROP_THRESHOLD*100:.0f}% nulls):")
                for col in report['columns_to_drop']:
                    null_pct = report['null_summary']['percentages'][col]
                    print(f"    • {col}: {null_pct:.1f}% missing")
            
            # Show top columns with missing values
            null_percentages = report['null_summary']['percentages']
            sorted_columns = sorted(null_percentages.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\nTop columns with missing values:")
            for col, null_pct in sorted_columns[:10]:  # Show top 10
                if null_pct > 0:
                    null_count = report['null_summary']['counts'][col]
                    print(f"  • {col}: {null_pct:.1f}% ({null_count:,} values)")
            
            return report
        except Exception as e:
            print(f"❌ Error generating missing values report: {e}")
            return {'error': str(e)}
    
    def handle_nulls(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle null values with smart imputation"""
        try:
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
            original_nulls = data.isnull().sum().sum()
            
            if remaining_nulls == 0:
                print("✓ All null values have been handled!")
            else:
                print(f"⚠️  {remaining_nulls:,} null values remain (from {original_nulls:,} originally)")
            
            print(f"\nColumns dropped: {len(null_handler.columns_to_drop)}")
            if null_handler.columns_to_drop:
                print(f"  • {', '.join(null_handler.columns_to_drop[:5])}")
                if len(null_handler.columns_to_drop) > 5:
                    print(f"  • ... and {len(null_handler.columns_to_drop) - 5} more")
            
            print(f"Columns imputed: {len(null_handler.imputation_values)}")
            if null_handler.imputation_values:
                print("Imputation values (first 5):")
                for i, (col, value) in enumerate(list(null_handler.imputation_values.items())[:5]):
                    print(f"  • {col}: {value}")
            
            return transformed_data
        except Exception as e:
            print(f"❌ Error handling null values: {e}")
            return data
    
    def outlier_statistics(self, data: pd.DataFrame) -> dict:
        """Calculate outlier statistics"""
        try:
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
            columns_with_outliers = []
            
            for col, stats in report.items():
                outliers = stats['outliers']
                total_outliers += outliers['count']
                
                if outliers['count'] > 0:
                    columns_with_outliers.append((col, outliers['count'], outliers['percentage']))
            
            print(f"\n📊 Total outliers detected: {total_outliers:,}")
            print(f"📊 Columns with outliers: {len(columns_with_outliers)} out of {len(report)} numerical columns")
            
            if columns_with_outliers:
                # Sort by number of outliers
                columns_with_outliers.sort(key=lambda x: x[1], reverse=True)
                
                print(f"\nTop columns with outliers:")
                for col, count, pct in columns_with_outliers[:10]:  # Show top 10
                    stats = report[col]
                    print(f"\n{col}:")
                    print(f"  • Outliers: {count:,} ({pct:.1f}%)")
                    print(f"  • Original Range: [{stats['original_range']['min']:.2f}, {stats['original_range']['max']:.2f}]")
                    print(f"  • IQR Bounds: [{stats['bounds']['lower']:.2f}, {stats['bounds']['upper']:.2f}]")
            
            return report
        except Exception as e:
            print(f"❌ Error calculating outlier statistics: {e}")
            return {'error': str(e)}
    
    def clip_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers using IQR method"""
        try:
            outlier_handler = OutlierHandler(
                factor=self.config.OUTLIER_FACTOR
            )
            outlier_handler.fit(data)
            transformed_data = outlier_handler.transform(data)
            
            print("\n" + "="*60)
            print("OUTLIER CLIPPING")
            print("="*60)
            
            print("✓ Outliers have been clipped to IQR bounds")
            
            # Show changes for columns that had outliers
            for col, bounds in outlier_handler.bounds.items():
                if col in transformed_data.columns:
                    original_min = data[col].min()
                    original_max = data[col].max()
                    new_min = transformed_data[col].min()
                    new_max = transformed_data[col].max()
                    
                    if original_min != new_min or original_max != new_max:
                        print(f"\n{col}:")
                        print(f"  • Original Range: [{original_min:.2f}, {original_max:.2f}]")
                        print(f"  • New Range: [{new_min:.2f}, {new_max:.2f}]")
                        print(f"  • Clipped to: [{bounds['lower']:.2f}, {bounds['upper']:.2f}]")
            
            return transformed_data
        except Exception as e:
            print(f"❌ Error clipping outliers: {e}")
            return data
    
    def preview_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preview current state of data"""
        self.data_preview = data
        
        print("\n" + "="*60)
        print("DATA PREVIEW")
        print("="*60)
        
        print(f"\nDataset Shape: {data.shape[0]:,} rows × {data.shape[1]:,} columns")
        
        print(f"\nFirst {self.config.PREVIEW_ROWS} rows:")
        pd.set_option('display.max_columns', min(self.config.MAX_COLS_TO_DISPLAY, data.shape[1]))
        pd.set_option('display.width', 120)
        print(data.head(self.config.PREVIEW_ROWS))
        
        print(f"\nData Types:")
        dtype_summary = data.dtypes.value_counts()
        for dtype, count in dtype_summary.items():
            print(f"  {dtype}: {count} columns")
        
        print(f"\nSummary Statistics:")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(data[numeric_cols].describe().round(2))
        else:
            print("No numerical columns found for summary statistics.")
        
        # Show memory usage
        memory_mb = data.memory_usage(deep=True).sum() / (1024**2)
        print(f"\nMemory Usage: {memory_mb:.2f} MB")
        
        # Show null counts
        null_count = data.isnull().sum().sum()
        if null_count > 0:
            print(f"Null Values: {null_count:,}")
        else:
            print("Null Values: None ✓")
        
        return data
    
    def run_full_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run end-to-end automated pipeline"""
        print("\n" + "="*60)
        print("RUNNING FULL AUTOMATED PIPELINE")
        print("="*60)
        
        try:
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
            
            if final_report['columns_removed']:
                print(f"   • Columns removed: {len(final_report['columns_removed'])}")
                if len(final_report['columns_removed']) <= 10:
                    print(f"     - {', '.join(final_report['columns_removed'])}")
            
            # Memory reduction
            original_memory = data.memory_usage(deep=True).sum() / (1024**2)
            cleaned_memory = cleaned_data.memory_usage(deep=True).sum() / (1024**2)
            memory_reduction = ((original_memory - cleaned_memory) / original_memory * 100) if original_memory > 0 else 0
            
            print(f"   • Memory: {original_memory:.2f} MB → {cleaned_memory:.2f} MB ({memory_reduction:.1f}% reduction)")
            
            # Null reduction
            original_nulls = data.isnull().sum().sum()
            cleaned_nulls = cleaned_data.isnull().sum().sum()
            null_reduction = ((original_nulls - cleaned_nulls) / original_nulls * 100) if original_nulls > 0 else 100
            
            print(f"   • Null values: {original_nulls:,} → {cleaned_nulls:,} ({null_reduction:.1f}% reduction)")
            
            return cleaned_data
            
        except Exception as e:
            print(f"❌ Error running pipeline: {e}")
            print("Falling back to original data...")
            return data
    
    def export_clean_dataset(self, data: pd.DataFrame, file_path: str = None):
        """Export cleaned dataset to CSV"""
        path = file_path or self.config.OUTPUT_PATH
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            data.to_csv(path, index=False)
            print(f"\n✅ Cleaned dataset exported to: {path}")
            print(f"   • File size: {os.path.getsize(path) / (1024**2):.2f} MB")
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
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("DATA PIPELINE QUALITY REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                
                for report_name, report_data in self.reports.items():
                    f.write(f"\n{report_name.upper().replace('_', ' ')} REPORT\n")
                    f.write("-" * 40 + "\n")
                    
                    if isinstance(report_data, dict):
                        if 'error' in report_data:
                            f.write(f"Error: {report_data['error']}\n")
                        else:
                            # Format the report nicely
                            f.write(json.dumps(report_data, indent=2, default=str))
                    else:
                        f.write(str(report_data))
                    
                    f.write("\n\n")
            
            print(f"✓ Quality report saved to: {report_path}")
            print(f"   • Report size: {os.path.getsize(report_path) / 1024:.2f} KB")
            
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
        
        # Show saved files
        if os.path.exists(self.config.OUTPUT_PATH):
            file_size = os.path.getsize(self.config.OUTPUT_PATH) / (1024**2)
            print(f"Files saved:")
            print(f"  • Cleaned data: {self.config.OUTPUT_PATH} ({file_size:.2f} MB)")
        
        if os.path.exists(self.config.REPORT_PATH):
            report_size = os.path.getsize(self.config.REPORT_PATH) / 1024
            print(f"  • Quality report: {self.config.REPORT_PATH} ({report_size:.2f} KB)")
        
        exit(0)
    
    def get_pipeline_summary(self) -> dict:
        """Get summary of pipeline operations"""
        summary = {
            'pipeline_configured': self.pipeline is not None,
            'reports_generated': list(self.reports.keys()),
            'data_preview_available': self.data_preview is not None,
            'config': {
                'outlier_factor': self.config.OUTLIER_FACTOR,
                'drop_threshold': self.config.DROP_THRESHOLD,
                'impute_strategy': self.config.IMPUTE_STRATEGY,
                'data_path': self.config.DATA_PATH,
                'output_path': self.config.OUTPUT_PATH
            }
        }
        
        if 'final' in self.reports:
            final = self.reports['final']
            summary.update({
                'original_shape': final['original_shape'],
                'cleaned_shape': final['cleaned_shape'],
                'columns_removed': final['columns_removed'],
                'pipeline_steps': final.get('pipeline_steps', [])
            })
        
        return summary