"""
Main application for the Modern Data Analysis Pipeline
"""
import pandas as pd
import sys
from pipeline_manager import PipelineManager
from config import settings

class DataPipelineApp:
    """
    Interactive command-line application for data pipeline
    """
    def __init__(self):
        self.manager = PipelineManager()
        self.data = None
        self.cleaned_data = None
        self.current_step = 0
        
    def display_menu(self):
        """Display the main menu"""
        print("\n" + "="*60)
        print("MODERN DATA ANALYSIS PIPELINE (SR. SOFTWARE MODE)")
        print("="*60)
        
        menu_items = [
            ("1", "Check Data Schema (dtypes)"),
            ("2", "Enforce Data Types (Casting)"),
            ("3", "Missing Values Report (Quality Audit)"),
            ("4", "Handle Nulls (Smart Drop/Impute Logic)"),
            ("5", "Outlier Statistics (Numerical Check)"),
            ("6", "Clip Outliers (IQR Statistical Method)"),
            ("7", "Preview Data (Current State)"),
            ("8", "RUN FULL AUTOMATED PIPELINE (End-to-End)"),
            ("9", "Export Clean Dataset (CSV Persistence)"),
            ("10", "Exit System"),
            ("L", "Load New Dataset"),
            ("S", "Show Current Status")
        ]
        
        for item in menu_items:
            print(f"{item[0]}. {item[1]}")
        
        print("\n" + "-"*60)
    
    def run(self):
        """Run the main application loop"""
        print("🚀 Welcome to the Modern Data Analysis Pipeline!")
        print("📂 Loading data from:", settings.DATA_PATH)
        
        # Initial data load
        self.data = self.manager.load_data()
        
        if self.data is None:
            print("❌ Failed to load data. Exiting...")
            return
        
        while True:
            try:
                self.display_menu()
                choice = input("\nSelect your processing step (1-10, L, S): ").strip().upper()
                
                if choice == 'L':
                    new_path = input("Enter new file path (or press Enter to reload current): ").strip()
                    if new_path:
                        settings.DATA_PATH = new_path
                    self.data = self.manager.load_data()
                
                elif choice == 'S':
                    self._show_status()
                
                elif choice == '1':
                    self.manager.check_data_schema(self.data)
                
                elif choice == '2':
                    self.data = self.manager.enforce_data_types(self.data)
                
                elif choice == '3':
                    self.manager.missing_values_report(self.data)
                
                elif choice == '4':
                    self.data = self.manager.handle_nulls(self.data)
                
                elif choice == '5':
                    self.manager.outlier_statistics(self.data)
                
                elif choice == '6':
                    self.data = self.manager.clip_outliers(self.data)
                
                elif choice == '7':
                    self.manager.preview_data(self.data)
                
                elif choice == '8':
                    self.cleaned_data = self.manager.run_full_pipeline(self.data)
                
                elif choice == '9':
                    if self.cleaned_data is not None:
                        self.manager.export_clean_dataset(self.cleaned_data)
                    else:
                        print("⚠️  No cleaned data available. Run the full pipeline (option 8) first.")
                        export_current = input("Export current data instead? (y/n): ").lower()
                        if export_current == 'y':
                            self.manager.export_clean_dataset(self.data)
                
                elif choice == '10':
                    self.manager.exit_system()
                
                elif choice.isdigit() and 1 <= int(choice) <= 10:
                    pass  # Already handled above
                
                else:
                    print("❌ Invalid choice. Please select 1-10, L, or S.")
                
                # Pause for user to see results
                if choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                confirm = input("Are you sure you want to exit? (y/n): ").lower()
                if confirm == 'y':
                    self.manager.exit_system()
            except Exception as e:
                print(f"❌ Error: {e}")
                input("Press Enter to continue...")
    
    def _show_status(self):
        """Show current pipeline status"""
        print("\n" + "="*60)
        print("CURRENT PIPELINE STATUS")
        print("="*60)
        
        if self.data is not None:
            print(f"\n📊 Data Status:")
            print(f"  • Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
            print(f"  • Memory: {self.data.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            print(f"  • Null Values: {self.data.isnull().sum().sum():,}")
            print(f"  • Data Types: {len(set(self.data.dtypes))} unique types")
        
        if self.cleaned_data is not None:
            print(f"\n✨ Cleaned Data:")
            print(f"  • Shape: {self.cleaned_data.shape[0]} rows × {self.cleaned_data.shape[1]} columns")
            print(f"  • Null Values: {self.cleaned_data.isnull().sum().sum():,}")
        
        print(f"\n⚙️  Configuration:")
        print(f"  • Data Path: {settings.DATA_PATH}")
        print(f"  • Output Path: {settings.OUTPUT_PATH}")
        print(f"  • Outlier Factor: {settings.OUTLIER_FACTOR}")
        print(f"  • Drop Threshold: {settings.DROP_THRESHOLD*100:.0f}%")
        print(f"  • Impute Strategy: {settings.IMPUTE_STRATEGY}")


def main():
    """Main entry point"""
    app = DataPipelineApp()
    app.run()


if __name__ == "__main__":
    main()