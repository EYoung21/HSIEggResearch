#!/usr/bin/env python3
"""
M1 Experiment Runner: MSC + SG + LightGBM + SMOTE for Mortality Classification

This script runs the complete M1 experiment pipeline including:
1. Dependency checking and installation guidance
2. Data preprocessing with MSC and Savitzky-Golay filtering
3. LightGBM model training with SMOTE-ENN class balancing
4. Comprehensive evaluation and results analysis
"""

import subprocess
import sys
import os

def check_and_install_dependencies():
    """Check if required packages are installed and provide installation guidance"""
    print("="*80)
    print("M1 EXPERIMENT: DEPENDENCY CHECK")
    print("="*80)
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'lightgbm': 'lightgbm',
        'imblearn': 'imbalanced-learn',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            print(f"✗ {package_name} is missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("\nTo install missing packages, run:")
        print(f"pip3 install {' '.join(missing_packages)}")
        print("\nOr install all requirements:")
        print("pip3 install -r requirements.txt")
        
        response = input("\nWould you like to attempt automatic installation? (y/n): ").lower()
        if response == 'y':
            try:
                print("Installing packages...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
                print("✓ Installation completed!")
            except subprocess.CalledProcessError:
                print("✗ Installation failed. Please install manually.")
                return False
        else:
            return False
    
    print("✓ All dependencies are available!")
    return True

def run_preprocessing():
    """Run the preprocessing pipeline"""
    print("\n" + "="*80)
    print("STEP 1: PREPROCESSING")
    print("="*80)
    
    try:
        print("Running M1 preprocessing...")
        result = subprocess.run([sys.executable, 'M1_preprocessing.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Preprocessing completed successfully!")
            print("\nPreprocessing output:")
            print(result.stdout)
        else:
            print("✗ Preprocessing failed!")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running preprocessing: {e}")
        return False
    
    return True

def run_model_training():
    """Run the model training and evaluation"""
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING & EVALUATION")
    print("="*80)
    
    try:
        print("Running M1 model training...")
        result = subprocess.run([sys.executable, 'M1_model.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Model training completed successfully!")
            print("\nModel training output:")
            print(result.stdout)
        else:
            print("✗ Model training failed!")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Error running model training: {e}")
        return False
    
    return True

def display_results():
    """Display the experiment results"""
    print("\n" + "="*80)
    print("M1 EXPERIMENT RESULTS")
    print("="*80)
    
    # Check if results files exist
    results_files = [
        'M1_experimental_results.json',
        'M1_performance_summary.txt',
        'feature_importance.csv'
    ]
    
    print("Generated files:")
    for file in results_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file)
            print(f"✓ {file} ({file_size:,} bytes)")
        else:
            print(f"✗ {file} (missing)")
    
    # Display performance summary if available
    summary_file = 'M1_performance_summary.txt'
    if os.path.exists(summary_file):
        print(f"\n{summary_file}:")
        print("-" * 80)
        with open(summary_file, 'r') as f:
            print(f.read())
    
    # Display feature importance info
    feature_file = 'feature_importance.csv'
    if os.path.exists(feature_file):
        print(f"\nTop features from {feature_file}:")
        print("-" * 50)
        try:
            import pandas as pd
            df = pd.read_csv(feature_file)
            print(df.head(10).to_string(index=False))
        except Exception as e:
            print(f"Could not display feature importance: {e}")

def validate_data_files():
    """Validate that required data files exist"""
    print("Validating data files...")
    
    required_files = [
        '../../data/reference_metadata.csv',
        '../../data/spectral_data_D0.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✓ {file}")
    
    if missing_files:
        print(f"\n✗ Missing data files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure data files are available before running the experiment.")
        return False
    
    return True

def main():
    """Main experiment runner"""
    print("="*80)
    print("M1 MORTALITY CLASSIFICATION EXPERIMENT")
    print("MSC + Savitzky-Golay + LightGBM + SMOTE-ENN")
    print("="*80)
    
    # Step 0: Validate data files
    if not validate_data_files():
        sys.exit(1)
    
    # Step 1: Check dependencies
    if not check_and_install_dependencies():
        print("\n✗ Please install required dependencies before proceeding.")
        sys.exit(1)
    
    # Step 2: Run preprocessing
    if not run_preprocessing():
        print("\n✗ Preprocessing failed. Experiment stopped.")
        sys.exit(1)
    
    # Step 3: Run model training
    if not run_model_training():
        print("\n✗ Model training failed. Experiment stopped.")
        sys.exit(1)
    
    # Step 4: Display results
    display_results()
    
    print("\n" + "="*80)
    print("M1 EXPERIMENT COMPLETED SUCCESSFULLY! 🎉")
    print("="*80)
    print("\nExperiment Summary:")
    print("- Preprocessing: MSC + Savitzky-Golay filtering")
    print("- Feature Selection: Optimized mutual information")
    print("- Class Balancing: SMOTE-ENN")
    print("- Algorithm: LightGBM + Ensemble Voting")
    print("- Target: Mortality Classification (Alive vs Dead)")
    print("\nCheck the generated files for detailed results!")

if __name__ == "__main__":
    main() 