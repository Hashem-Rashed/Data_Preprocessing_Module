#!/usr/bin/env python3
"""
Runner script for the Modern Data Analysis Pipeline
Choose between CLI, Web, and Batch processing modes
"""
import sys
import os
import argparse
import webbrowser
import subprocess
from datetime import datetime
import signal

def print_banner():
    """Print application banner"""
    print("\n" + "="*70)
    print("""\033[1;36m
███╗   ███╗ ██████╗ ██████╗ ███████╗███████╗███╗   ██╗    
████╗ ████║██╔═══██╗██╔══██╗██╔════╝██╔════╝████╗  ██║    
██╔████╔██║██║   ██║██║  ██║█████╗  █████╗  ██╔██╗ ██║    
██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██╔══╝  ██║╚██╗██║    
██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗██║ ╚████║    
╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═══╝    
                                                           
██████╗  █████╗ ████████╗ █████╗                          
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗                         
██║  ██║███████║   ██║   ███████║                         
██║  ██║██╔══██║   ██║   ██╔══██║                         
██████╔╝██║  ██║   ██║   ██║  ██║                         
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝                         
\033[0m""")
    print("="*70)
    print("MODERN DATA ANALYSIS PIPELINE v2.0")
    print("Automated Data Cleaning, Analysis, and Visualization")
    print("="*70)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'dash', 'plotly']
    missing_packages = []
    
    print("\n📦 Checking dependencies...")
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package} (missing)")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        install = input("Do you want to install missing packages? (y/n): ").lower()
        if install == 'y':
            print("Installing missing packages...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                print("✅ All packages installed successfully!")
            except subprocess.CalledProcessError:
                print("❌ Failed to install packages. Please install manually:")
                print(f"pip install {' '.join(missing_packages)}")
                return False
        else:
            print("Please install missing packages to continue:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def get_dash_version():
    """Get Dash version to handle compatibility"""
    try:
        import dash
        dash_version = getattr(dash, '__version__', 'unknown')
        print(f"📊 Dash version: {dash_version}")
        return dash_version
    except:
        return 'unknown'

def run_cli_mode():
    """Run the command line interface"""
    print("\n" + "="*70)
    print("STARTING COMMAND LINE INTERFACE")
    print("="*70)
    
    try:
        from main import DataPipelineApp
        app = DataPipelineApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\n👋 CLI mode interrupted by user")
    except Exception as e:
        print(f"\n❌ Error starting CLI: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

def run_web_mode(host='0.0.0.0', port=8050, debug=False):
    """Run the web dashboard with version compatibility"""
    print("\n" + "="*70)
    print("STARTING WEB DASHBOARD")
    print("="*70)
    
    print(f"\n🌐 Dashboard will be available at:")
    print(f"   Local: http://localhost:{port}")
    print(f"   Network: http://{host}:{port}")
    
    # Try to open browser automatically
    try:
        webbrowser.open(f"http://localhost:{port}")
        print("✅ Browser opened automatically")
    except:
        print("⚠️  Could not open browser automatically")
        print("   Please open the URL manually in your browser")
    
    print("\n🔄 Starting server... (Press Ctrl+C to stop)")
    
    try:
        from app import app
        
        # Try both methods for compatibility with different Dash versions
        try:
            # Try new method first (Dash 3.0+)
            app.run(debug=debug, host=host, port=port)
        except AttributeError:
            # Fallback to old method (Dash < 3.0)
            app.run_server(debug=debug, host=host, port=port)
                
    except KeyboardInterrupt:
        print("\n\n👋 Web server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting web server: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

def run_batch_mode(config_file=None, input_file=None, output_file=None):
    """Run in batch processing mode"""
    print("\n" + "="*70)
    print("STARTING BATCH PROCESSING MODE")
    print("="*70)
    
    try:
        from pipeline_manager import PipelineManager
        from config import settings
        
        # Update config if parameters provided
        if config_file:
            print(f"📁 Loading config from: {config_file}")
            # You would load custom config here
        
        if input_file:
            print(f"📁 Processing file: {input_file}")
            settings.DATA_PATH = input_file
        
        if output_file:
            print(f"💾 Output will be saved to: {output_file}")
            settings.OUTPUT_PATH = output_file
        
        # Initialize pipeline
        manager = PipelineManager()
        
        print("\n📊 Loading data...")
        data = manager.load_data()
        
        if data is None:
            print("❌ Failed to load data. Exiting...")
            return
        
        print(f"✅ Data loaded: {data.shape[0]:,} rows × {data.shape[1]:,} columns")
        
        # Run full pipeline
        print("\n🔄 Running data pipeline...")
        cleaned_data = manager.run_full_pipeline(data)
        
        if cleaned_data is not None:
            print("\n💾 Exporting results...")
            manager.export_clean_dataset(cleaned_data)
            
            # Generate report
            report_path = settings.REPORT_PATH
            if os.path.exists(report_path):
                report_size = os.path.getsize(report_path) / 1024
                print(f"✅ Quality report: {report_path} ({report_size:.2f} KB)")
            
            output_path = settings.OUTPUT_PATH
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024**2)
                print(f"✅ Cleaned data: {output_path} ({file_size:.2f} MB)")
            
            print("\n🎉 Batch processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Batch processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in batch processing: {e}")
        import traceback
        traceback.print_exc()

def run_test_mode():
    """Run test mode with sample data"""
    print("\n" + "="*70)
    print("STARTING TEST MODE WITH SAMPLE DATA")
    print("="*70)
    
    print("\n🧪 Creating sample dataset...")
    
    try:
        from pipeline_manager import PipelineManager
        from config import settings
        
        # Create sample data
        manager = PipelineManager()
        sample_data = manager._create_sample_data()
        
        print(f"✅ Sample data created: {sample_data.shape[0]:,} rows × {sample_data.shape[1]:,} columns")
        
        # Ask user which mode to test
        print("\nChoose test mode:")
        print("1. Quick pipeline test")
        print("2. Full pipeline with reports")
        print("3. Test web dashboard")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            print("\n🚀 Running quick pipeline test...")
            cleaned = manager.run_full_pipeline(sample_data)
            print(f"✅ Test completed. Cleaned data: {cleaned.shape[0]:,} rows")
            
        elif choice == '2':
            print("\n📊 Generating comprehensive reports...")
            
            # Check schema
            print("\n1. Checking data schema...")
            manager.check_data_schema(sample_data)
            
            # Missing values
            print("\n2. Analyzing missing values...")
            manager.missing_values_report(sample_data)
            
            # Outliers
            print("\n3. Detecting outliers...")
            manager.outlier_statistics(sample_data)
            
            # Run pipeline
            print("\n4. Running full pipeline...")
            cleaned = manager.run_full_pipeline(sample_data)
            
            # Export
            print("\n5. Exporting results...")
            manager.export_clean_dataset(cleaned)
            
            print("\n🎉 Comprehensive test completed!")
            
        elif choice == '3':
            print("\n🌐 Launching web dashboard with sample data...")
            # Update config to use sample data
            settings.DATA_PATH = 'data/sample_data.csv'
            run_web_mode()
            
        else:
            print("❌ Invalid choice")
            
    except Exception as e:
        print(f"\n❌ Error in test mode: {e}")

def show_system_info():
    """Display system information"""
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    
    import platform
    import pandas as pd
    import numpy as np
    import sklearn
    
    print(f"\n📋 Platform: {platform.system()} {platform.release()}")
    print(f"🐍 Python: {platform.python_version()}")
    print(f"📦 Pandas: {pd.__version__}")
    print(f"🔢 NumPy: {np.__version__}")
    print(f"🤖 Scikit-learn: {sklearn.__version__}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 Memory: {memory.available / (1024**3):.1f} GB available / {memory.total / (1024**3):.1f} GB total")
    except:
        print("💾 Memory: psutil not available for memory info")
    
    # Check disk space
    try:
        import shutil
        disk = shutil.disk_usage(".")
        print(f"💽 Disk: {disk.free / (1024**3):.1f} GB free / {disk.total / (1024**3):.1f} GB total")
    except:
        print("💽 Disk: Disk info not available")

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    print("\n\n⚠️  Interrupted by user")
    print("👋 Exiting gracefully...")
    sys.exit(0)

def main():
    """Main entry point"""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Modern Data Analysis Pipeline')
    parser.add_argument('--mode', choices=['cli', 'web', 'batch', 'test', 'info'],
                       help='Run mode: cli, web, batch, test, or info')
    parser.add_argument('--host', default='0.0.0.0', help='Web server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8050, help='Web server port (default: 8050)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--input', help='Input file for batch mode')
    parser.add_argument('--output', help='Output file for batch mode')
    parser.add_argument('--config', help='Config file for batch mode')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install required packages.")
        return
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Determine run mode
    if args.mode:
        # Run in specified mode
        if args.mode == 'cli':
            run_cli_mode()
        elif args.mode == 'web':
            run_web_mode(host=args.host, port=args.port, debug=args.debug)
        elif args.mode == 'batch':
            run_batch_mode(config_file=args.config, input_file=args.input, output_file=args.output)
        elif args.mode == 'test':
            run_test_mode()
        elif args.mode == 'info':
            show_system_info()
    else:
        # Interactive mode selection
        while True:
            print("\n" + "="*70)
            print("SELECT RUN MODE")
            print("="*70)
            
            print("\nAvailable modes:")
            print("1. 📟 Command Line Interface (Interactive)")
            print("2. 🌐 Web Dashboard (Visual)")
            print("3. ⚙️  Batch Processing (Automated)")
            print("4. 🧪 Test Mode (Sample Data)")
            print("5. 📊 System Information")
            print("6. ❌ Exit")
            
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                run_cli_mode()
                break
            elif choice == '2':
                print("\nWeb Dashboard Options:")
                print("1. Default settings (localhost:8050)")
                print("2. Custom settings")
                
                web_choice = input("\nEnter choice (1-2): ").strip()
                
                if web_choice == '1':
                    run_web_mode()
                elif web_choice == '2':
                    host = input("Host (default: 0.0.0.0): ").strip() or '0.0.0.0'
                    port = input("Port (default: 8050): ").strip() or '8050'
                    debug = input("Debug mode? (y/n, default: n): ").strip().lower() == 'y'
                    run_web_mode(host=host, port=int(port), debug=debug)
                break
            elif choice == '3':
                print("\nBatch Processing Options:")
                input_file = input("Input file path (press Enter for default): ").strip()
                output_file = input("Output file path (press Enter for default): ").strip()
                config_file = input("Config file path (press Enter for default): ").strip()
                
                run_batch_mode(
                    config_file=config_file if config_file else None,
                    input_file=input_file if input_file else None,
                    output_file=output_file if output_file else None
                )
                break
            elif choice == '4':
                run_test_mode()
                break
            elif choice == '5':
                show_system_info()
                continue  # Show menu again
            elif choice == '6':
                print("\n👋 Thank you for using the Modern Data Analysis Pipeline!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
                continue

if __name__ == "__main__":
    main()