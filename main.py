# -*- coding: utf-8 -*-
"""
Main application for the Modern Data Analysis Pipeline
Improved version with better error handling, logging, and UX
"""
import pandas as pd
import sys
import logging
from pathlib import Path
from pipeline_manager import PipelineManager
from config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DataPipelineApp:
    """
    Interactive command-line application for data pipeline with improved UX
    """
    
    # UI Constants
    HEADER_WIDTH = 70
    SEPARATOR = "=" * HEADER_WIDTH
    SUB_SEPARATOR = "-" * HEADER_WIDTH
    
    def __init__(self):
        self.manager = PipelineManager()
        self.data = None
        self.cleaned_data = None
        self.current_step = 0
        
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
        
    def display_header(self):
        """Display application header"""
        print("\n" + self.SEPARATOR)
        print("DATA ANALYSIS PIPELINE".center(self.HEADER_WIDTH))
        print("v2.0 - Improved Edition".center(self.HEADER_WIDTH))
        print(self.SEPARATOR)
        
    def display_menu(self):
        """Display the main menu with logical grouping"""
        self.display_header()
        
        print("\n📋 DATA OPERATIONS:")
        print("  1. Load/Reload Dataset")
        print("  2. Preview Current Data")
        print("  3. Show Data Status")
        
        print("\n🔍 DATA INSPECTION:")
        print("  4. Check Schema & Data Types")
        print("  5. Missing Values Report")
        print("  6. Outlier Analysis")
        
        print("\n⚙️  DATA PROCESSING:")
        print("  7. Enforce Data Types")
        print("  8. Handle Missing Values")
        print("  9. Clip Outliers")
        print("  10. Run Full Pipeline (Automated)")
        
        print("\n💾 EXPORT & EXIT:")
        print("  11. Export Cleaned Dataset")
        print("  0. Exit System")
        
        print("\n" + self.SUB_SEPARATOR)
        
        # Show current data status
        if self.data is not None:
            rows, cols = self.data.shape
            nulls = self.data.isnull().sum().sum()
            print(f"📊 Current: {rows:,} rows × {cols} cols | {nulls:,} null values")
        else:
            print("⚠️  No data loaded")
        print(self.SUB_SEPARATOR)
    
    def run(self):
        """Run the main application loop"""
        logger.info("Starting Data Analysis Pipeline")
        print("\n🚀 Welcome to the Modern Data Analysis Pipeline!")
        print(f"📂 Default data path: {settings.DATA_PATH}")
        
        # Initial data load attempt
        if Path(settings.DATA_PATH).exists():
            print("\n📥 Loading initial dataset...")
            self.data = self.manager.load_data()
        else:
            print(f"\n⚠️  Data file not found: {settings.DATA_PATH}")
            print("💡 You can load a different file using option 1")
        
        while True:
            try:
                self.display_menu()
                choice = self._get_user_choice()
                
                if choice is None:
                    continue
                
                # Process user choice
                self._handle_choice(choice)
                
                # Pause for user to see results
                if choice != '0':
                    input("\n⏸️  Press Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                if self._confirm_exit():
                    self.manager.exit_system()
                    
            except Exception as e:
                logger.exception("Unexpected error in main loop")
                print(f"\n❌ Unexpected error: {str(e)}")
                print("💡 Check logs/pipeline.log for details")
                input("\n⏸️  Press Enter to continue...")
    
    def _get_user_choice(self) -> str:
        """Get and validate user choice"""
        valid_choices = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        
        while True:
            choice = input("\n👉 Select option: ").strip()
            
            if choice in valid_choices:
                return choice
            else:
                print(f"❌ Invalid choice '{choice}'. Please select 0-11.")
    
    def _handle_choice(self, choice: str):
        """Handle user menu choice with validation"""
        
        # Options that require data to be loaded
        data_required_options = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        
        if choice in data_required_options and self.data is None:
            print("\n❌ No data loaded!")
            print("💡 Please load data first (option 1)")
            return
        
        # Route to appropriate handler
        if choice == '0':
            self.manager.exit_system()
            
        elif choice == '1':
            self._handle_load_data()
            
        elif choice == '2':
            self._handle_preview_data()
            
        elif choice == '3':
            self._handle_show_status()
            
        elif choice == '4':
            self.manager.check_data_schema(self.data)
            
        elif choice == '5':
            self.manager.missing_values_report(self.data)
            
        elif choice == '6':
            self.manager.outlier_statistics(self.data)
            
        elif choice == '7':
            self.data = self.manager.enforce_data_types(self.data)
            
        elif choice == '8':
            self.data = self.manager.handle_nulls(self.data)
            
        elif choice == '9':
            self.data = self.manager.clip_outliers(self.data)
            
        elif choice == '10':
            self._handle_full_pipeline()
            
        elif choice == '11':
            self._handle_export()
    
    def _handle_load_data(self):
        """Handle data loading with error recovery"""
        print("\n" + self.SUB_SEPARATOR)
        print("LOAD DATASET")
        print(self.SUB_SEPARATOR)
        
        print(f"\nCurrent path: {settings.DATA_PATH}")
        new_path = input("Enter new file path (or press Enter to use current): ").strip()
        
        if new_path:
            # Validate path exists
            if not Path(new_path).exists():
                print(f"\n❌ File not found: {new_path}")
                create = input("Would you like to create sample data instead? (y/n): ").lower()
                if create == 'y':
                    self.data = self.manager._create_sample_data()
                return
            settings.DATA_PATH = new_path
        
        logger.info(f"Loading data from {settings.DATA_PATH}")
        self.data = self.manager.load_data()
        
        if self.data is not None:
            self.cleaned_data = None  # Reset cleaned data
            logger.info(f"Data loaded successfully: {self.data.shape}")
    
    def _handle_preview_data(self):
        """Handle data preview"""
        self.manager.preview_data(self.data)
    
    def _handle_show_status(self):
        """Show comprehensive pipeline status"""
        print("\n" + self.SEPARATOR)
        print("PIPELINE STATUS".center(self.HEADER_WIDTH))
        print(self.SEPARATOR)
        
        if self.data is not None:
            print("\n📊 CURRENT DATA:")
            print(f"  • Shape: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
            print(f"  • Memory: {self.data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            print(f"  • Null Values: {self.data.isnull().sum().sum():,}")
            print(f"  • Duplicate Rows: {self.data.duplicated().sum():,}")
            
            # Data type summary
            dtype_counts = self.data.dtypes.value_counts()
            print(f"\n  • Data Types:")
            for dtype, count in dtype_counts.items():
                print(f"    - {dtype}: {count} columns")
        
        if self.cleaned_data is not None:
            print("\n✨ CLEANED DATA:")
            print(f"  • Shape: {self.cleaned_data.shape[0]:,} rows × {self.cleaned_data.shape[1]} columns")
            print(f"  • Memory: {self.cleaned_data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            print(f"  • Null Values: {self.cleaned_data.isnull().sum().sum():,}")
            
            # Calculate improvements
            if self.data is not None:
                rows_removed = self.data.shape[0] - self.cleaned_data.shape[0]
                cols_removed = self.data.shape[1] - self.cleaned_data.shape[1]
                nulls_removed = self.data.isnull().sum().sum() - self.cleaned_data.isnull().sum().sum()
                
                print(f"\n  📈 Improvements:")
                print(f"    - Rows removed: {rows_removed:,}")
                print(f"    - Columns removed: {cols_removed}")
                print(f"    - Nulls removed: {nulls_removed:,}")
        
        print(f"\n⚙️  CONFIGURATION:")
        print(f"  • Data Path: {settings.DATA_PATH}")
        print(f"  • Output Path: {settings.OUTPUT_PATH}")
        print(f"  • Outlier Factor: {settings.OUTLIER_FACTOR}")
        print(f"  • Drop Threshold: {settings.DROP_THRESHOLD*100:.0f}%")
        print(f"  • Impute Strategy: {settings.IMPUTE_STRATEGY}")
    
    def _handle_full_pipeline(self):
        """Handle full pipeline execution with confirmation"""
        print("\n" + self.SUB_SEPARATOR)
        print("⚠️  This will run the complete data cleaning pipeline:")
        print("  1. Enforce data types")
        print("  2. Handle missing values")
        print("  3. Remove/clip outliers")
        print(self.SUB_SEPARATOR)
        
        confirm = input("\nProceed with full pipeline? (y/n): ").lower()
        if confirm == 'y':
            logger.info("Starting full pipeline execution")
            self.cleaned_data = self.manager.run_full_pipeline(self.data)
            logger.info("Pipeline execution completed")
        else:
            print("❌ Pipeline cancelled")
    
    def _handle_export(self):
        """Handle data export with options"""
        if self.cleaned_data is not None:
            data_to_export = self.cleaned_data
            print("\n✅ Exporting cleaned data...")
        else:
            print("\n⚠️  No cleaned data available.")
            export_current = input("Export current (uncleaned) data instead? (y/n): ").lower()
            if export_current == 'y':
                data_to_export = self.data
            else:
                print("❌ Export cancelled")
                return
        
        # Ask for custom path
        custom_path = input(f"\nOutput path [{settings.OUTPUT_PATH}]: ").strip()
        if custom_path:
            self.manager.export_clean_dataset(data_to_export, custom_path)
        else:
            self.manager.export_clean_dataset(data_to_export)
        
        logger.info(f"Data exported successfully")
    
    def _confirm_exit(self) -> bool:
        """Confirm user wants to exit"""
        confirm = input("Are you sure you want to exit? (y/n): ").lower()
        return confirm == 'y'


def main():
    """Main entry point with error handling"""
    try:
        app = DataPipelineApp()
        app.run()
    except Exception as e:
        logger.exception("Critical error in application")
        print(f"\n❌ Critical error: {str(e)}")
        print("💡 Check logs/pipeline.log for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
