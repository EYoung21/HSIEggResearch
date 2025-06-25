#!/usr/bin/env python3
"""
G2 Experiment Runner: SNV + SG 2nd Derivative + Ensemble
========================================================

This script runs the complete G2 experiment pipeline:
1. Data preprocessing with SNV and Savitzky-Golay 2nd derivative
2. Spectral ratio feature extraction
3. Ensemble modeling with Random Forest, SVM, and XGBoost
4. Bayesian hyperparameter optimization
5. Model evaluation and results reporting

Experiment Design:
- Preprocessing: SNV + SG 2nd derivative + spectral ratios
- Algorithm: Ensemble (RF + SVM + XGB) with soft voting
- Features: Biologically meaningful spectral ratios
- Optimization: Bayesian optimization (30 calls)
- Validation: 5-fold stratified cross-validation

Author: HSI Egg Research Team
Date: 2024
"""

import sys
import os
import time
import subprocess
from pathlib import Path

def print_header():
    """Print experiment header"""
    print("=" * 70)
    print("🧪 G2 EXPERIMENT: SNV + SG 2nd Derivative + Ensemble")
    print("=" * 70)
    print("📋 Methodology:")
    print("   • Preprocessing: SNV + Savitzky-Golay 2nd derivative")
    print("   • Features: Spectral ratios (biological regions)")
    print("   • Algorithm: Ensemble (Random Forest + SVM + XGBoost)")
    print("   • Optimization: Bayesian optimization")
    print("   • Priority: HIGH")
    print("=" * 70)

def check_data_availability():
    """Check if required data files are available"""
    print("\n🔍 Checking data availability...")
    
    required_files = [
        '../../data/reference_metadata.csv',
        '../../data/spectral_data_D0.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"   ✓ Found: {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing required data files:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        print("\n💡 Please run the data preparation script first:")
        print("   python ../../convert_data_to_csv.py")
        return False
    
    print("   ✅ All required data files found")
    return True

def run_preprocessing():
    """Run G2 preprocessing pipeline"""
    print("\n" + "=" * 50)
    print("🔄 STEP 1: PREPROCESSING PIPELINE")
    print("=" * 50)
    print("• Applying SNV (Standard Normal Variate) normalization")
    print("• Computing Savitzky-Golay 2nd derivative")
    print("• Creating spectral ratio features")
    print("• Splitting train/test sets")
    
    start_time = time.time()
    
    try:
        # Run preprocessing
        result = subprocess.run([sys.executable, 'G2_preprocessing.py'], 
                              capture_output=True, text=True, check=True)
        
        print("✅ Preprocessing completed successfully!")
        print(f"⏱️  Time elapsed: {time.time() - start_time:.1f} seconds")
        
        # Print key outputs
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:  # Show last 10 lines
                if any(keyword in line for keyword in ['shape', 'samples', 'features', 'classes']):
                    print(f"   {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Preprocessing failed with error:")
        print(f"   {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def run_ensemble_modeling():
    """Run G2 ensemble modeling with optimization"""
    print("\n" + "=" * 50)
    print("🤖 STEP 2: ENSEMBLE MODELING")
    print("=" * 50)
    print("• Creating Random Forest, SVM, and XGBoost models")
    print("• Bayesian hyperparameter optimization (30 calls)")
    print("• Training optimized ensemble with soft voting")
    print("• Evaluating on test set")
    
    start_time = time.time()
    
    try:
        # Run ensemble modeling
        result = subprocess.run([sys.executable, 'G2_model.py'], 
                              capture_output=True, text=True, check=True)
        
        print("✅ Ensemble modeling completed successfully!")
        print(f"⏱️  Time elapsed: {time.time() - start_time:.1f} seconds")
        
        # Extract and display key results
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['CV Score:', 'Test Accuracy:', '🎯']):
                    print(f"   {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ensemble modeling failed with error:")
        print(f"   {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_outputs():
    """Check and summarize generated outputs"""
    print("\n" + "=" * 50)
    print("📁 CHECKING GENERATED OUTPUTS")
    print("=" * 50)
    
    expected_files = {
        'X_train_processed.npy': 'Processed training features',
        'X_test_processed.npy': 'Processed test features',
        'y_train.npy': 'Training labels',
        'y_test.npy': 'Test labels',
        'spectral_ratio_features.csv': 'Feature names and descriptions',
        'snv_sg_preprocessor.pkl': 'Fitted preprocessor',
        'label_encoder.pkl': 'Label encoder',
        'G2_ensemble_model.pkl': 'Trained ensemble model',
        'feature_importance.csv': 'Feature importance analysis',
        'test_predictions.csv': 'Test set predictions',
        'G2_experimental_results.json': 'Complete experimental results',
        'G2_performance_summary.txt': 'Human-readable summary'
    }
    
    generated_files = []
    missing_files = []
    
    for filename, description in expected_files.items():
        if Path(filename).exists():
            file_size = Path(filename).stat().st_size
            generated_files.append((filename, description, file_size))
            print(f"   ✓ {filename} ({file_size:,} bytes) - {description}")
        else:
            missing_files.append((filename, description))
            print(f"   ❌ {filename} - {description}")
    
    print(f"\n📊 Summary: {len(generated_files)}/{len(expected_files)} files generated")
    
    if missing_files:
        print(f"\n⚠️  Missing files:")
        for filename, description in missing_files:
            print(f"   • {filename} - {description}")
        return False
    
    return True

def display_results():
    """Display final results and summary"""
    print("\n" + "=" * 70)
    print("📊 G2 EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    # Try to read and display key results
    try:
        import json
        with open('G2_experimental_results.json', 'r') as f:
            results = json.load(f)
        
        print(f"🧬 Experiment: {results['experiment']}")
        print(f"⚙️  Preprocessing: {results['preprocessing']}")
        print(f"🤖 Algorithm: {results['algorithm']}")
        print(f"📈 Features: {results['feature_count']} spectral ratios")
        print(f"🎯 Test Accuracy: {results['test_results']['accuracy']:.4f}")
        
        if 'cross_validation_scores' in results:
            cv_scores = results['cross_validation_scores']
            print(f"\n📊 Cross-Validation Scores:")
            for model, scores in cv_scores.items():
                print(f"   • {model.upper()}: {scores['mean']:.4f} ± {scores['std']:.4f}")
        
        if 'individual_accuracies' in results['test_results']:
            individual_acc = results['test_results']['individual_accuracies']
            print(f"\n🔍 Individual Model Test Accuracies:")
            for model, acc in individual_acc.items():
                print(f"   • {model.upper()}: {acc:.4f}")
        
    except Exception as e:
        print(f"⚠️  Could not read results file: {e}")
        print("   Please check G2_experimental_results.json manually")
    
    # Display file locations
    print(f"\n📁 Results saved in: {os.getcwd()}")
    print("   📋 G2_performance_summary.txt - Detailed analysis")
    print("   📊 G2_experimental_results.json - Complete results")
    print("   📈 feature_importance.csv - Feature analysis")
    print("   🔮 test_predictions.csv - Model predictions")

def main():
    """Main experiment execution"""
    print_header()
    
    # Step 0: Check data availability
    if not check_data_availability():
        print("\n❌ Experiment aborted due to missing data")
        sys.exit(1)
    
    # Step 1: Preprocessing
    if not run_preprocessing():
        print("\n❌ Experiment aborted due to preprocessing failure")
        sys.exit(1)
    
    # Step 2: Ensemble modeling
    if not run_ensemble_modeling():
        print("\n❌ Experiment aborted due to modeling failure")
        sys.exit(1)
    
    # Step 3: Check outputs
    if not check_outputs():
        print("\n⚠️  Some output files are missing")
    
    # Step 4: Display results
    display_results()
    
    print("\n" + "=" * 70)
    print("🎉 G2 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("📝 Next steps:")
    print("   • Review G2_performance_summary.txt for detailed analysis")
    print("   • Compare results with G1 experiment")
    print("   • Consider running G3-G8 experiments")
    print("   • Analyze feature importance for biological insights")

if __name__ == "__main__":
    main() 