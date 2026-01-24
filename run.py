#!/usr/bin/env python3
"""
Runner script for the Data Analysis Pipeline
Choose between CLI and Web interface
"""
import sys
import os

def main():
    print("="*60)
    print("MODERN DATA ANALYSIS PIPELINE")
    print("="*60)
    print("\nChoose interface:")
    print("1. Command Line Interface (main.py)")
    print("2. Web Dashboard (app.py)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        print("\nStarting CLI interface...")
        from main import main as cli_main
        cli_main()
    elif choice == '2':
        print("\nStarting web dashboard...")
        print("Open http://localhost:8050 in your browser")
        try:
            from app import app
            app.run_server(debug=True, host='0.0.0.0', port=8050)
        except Exception as e:
            print(f"Error starting web server: {e}")
            print("Make sure you have installed all requirements:")
            print("pip install dash plotly pandas numpy scikit-learn")
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()