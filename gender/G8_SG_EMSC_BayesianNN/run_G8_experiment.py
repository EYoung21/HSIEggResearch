#!/usr/bin/env python3
"""
G8 Experiment Runner: SG + EMSC + Bayesian Neural Network
Advanced preprocessing with Bayesian deep learning and uncertainty quantification
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas', 
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'tensorflow': 'tensorflow',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} not found")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ All dependencies satisfied")
    return True

def check_data_files():
    """Check if required data files exist"""
    print("\nChecking data files...")
    
    required_files = [
        '../../data/reference_metadata.csv',
        '../../data/spectral_data_D0.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} not found")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing data files: {missing_files}")
        print("Please ensure data files are in the correct location")
        return False
    
    print("✓ All data files found")
    return True

def run_preprocessing():
    """Run the G8 preprocessing pipeline"""
    print("\n" + "="*60)
    print("RUNNING G8 PREPROCESSING")
    print("="*60)
    
    try:
        result = subprocess.run([sys.executable, 'G8_preprocessing.py'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        print("✓ Preprocessing completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Preprocessing failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def run_bayesian_nn_training():
    """Run the Bayesian neural network training"""
    print("\n" + "="*60)
    print("RUNNING G8 BAYESIAN NEURAL NETWORK TRAINING")
    print("="*60)
    
    try:
        result = subprocess.run([sys.executable, 'G8_model.py'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        print("✓ Bayesian NN training completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Bayesian NN training failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def display_results():
    """Display the final results"""
    print("\n" + "="*60)
    print("G8 EXPERIMENT RESULTS")
    print("="*60)
    
    # Check if result files exist
    result_files = [
        'G8_experimental_results.json',
        'G8_performance_summary.txt',
        'G8_bayesian_nn_model.h5'
    ]
    
    print("Generated files:")
    for file_path in result_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} ({size:,} bytes)")
        else:
            print(f"✗ {file_path} not found")
    
    # Display performance summary if available
    summary_file = 'G8_performance_summary.txt'
    if os.path.exists(summary_file):
        print(f"\n{'-'*60}")
        print("PERFORMANCE SUMMARY:")
        print(f"{'-'*60}")
        
        with open(summary_file, 'r') as f:
            content = f.read()
            # Display key results
            lines = content.split('\n')
            in_results = False
            
            for line in lines:
                if 'PERFORMANCE RESULTS:' in line:
                    in_results = True
                    print(line)
                elif in_results and line.strip():
                    if line.startswith('UNCERTAINTY') or line.startswith('RELIABILITY'):
                        break
                    print(line)
                elif 'Test Accuracy:' in line or 'AUC Score:' in line or 'CV Mean:' in line:
                    print(line)
    
    # Load and display key metrics from JSON
    import json
    results_file = 'G8_experimental_results.json'
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"\n{'-'*60}")
            print("KEY METRICS:")
            print(f"{'-'*60}")
            print(f"🎯 Test Accuracy: {results['best_accuracy']:.4f}")
            print(f"📊 AUC Score: {results['auc_score']:.4f}")
            print(f"🔄 CV Mean: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            print(f"❓ Mean Uncertainty: {results['uncertainty_metrics']['mean_total']:.4f}")
            print(f"⏱️ Training Time: {results['training_time_seconds']:.1f} seconds")
            print(f"🧠 Model Parameters: {results['model_architecture']['total_parameters']:,}")
            print(f"📐 Coverage: {results['uncertainty_metrics']['prediction_interval_coverage']:.4f}")
            
        except Exception as e:
            print(f"Could not load results: {e}")

def main():
    """Main experiment runner"""
    print("="*80)
    print("G8 EXPERIMENT: SG + EMSC + Bayesian Neural Network")
    print("Advanced preprocessing with uncertainty quantification")
    print("="*80)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies not satisfied. Please install required packages.")
        return False
    
    # Check data files  
    if not check_data_files():
        print("\n❌ Data files not found. Please ensure data is available.")
        return False
    
    # Run preprocessing
    if not run_preprocessing():
        print("\n❌ Preprocessing failed. Please check errors above.")
        return False
    
    # Check if preprocessed files were created
    preprocessed_files = [
        'X_train_processed.npy',
        'X_test_processed.npy', 
        'y_train.npy',
        'y_test.npy',
        'emsc_sg_preprocessor.pkl'
    ]
    
    missing_preprocessed = []
    for file_path in preprocessed_files:
        if not os.path.exists(file_path):
            missing_preprocessed.append(file_path)
    
    if missing_preprocessed:
        print(f"\n❌ Missing preprocessed files: {missing_preprocessed}")
        return False
    
    print(f"\n✓ All preprocessed files created successfully")
    
    # Run Bayesian neural network training
    if not run_bayesian_nn_training():
        print("\n❌ Bayesian NN training failed. Please check errors above.")
        return False
    
    # Display results
    display_results()
    
    print(f"\n" + "="*80)
    print("✅ G8 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("🎯 Bayesian Neural Network with Uncertainty Quantification")
    print("📁 Check generated files for detailed results")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 