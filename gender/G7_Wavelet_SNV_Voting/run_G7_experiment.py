#!/usr/bin/env python3
"""
G7 Experiment Runner: Wavelet + SNV + Voting Ensemble
Complete pipeline for advanced voting ensemble experiment
"""

import os
import sys
import time
import subprocess
import importlib.util
from pathlib import Path

def check_dependencies():
    """Check required dependencies"""
    print("Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'scipy', 'pywt', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'pywt':
                import pywt
            elif package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} - MISSING")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ All dependencies satisfied")
    return True

def check_data_availability():
    """Check if required data files exist"""
    print("\nChecking data availability...")
    
    required_files = [
        '../../data/reference_metadata.csv',
        '../../data/spectral_data_D0.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"✗ {file_path} - MISSING")
    
    if missing_files:
        print(f"\nMissing data files: {missing_files}")
        return False
    
    print("✓ All data files available")
    return True

def run_preprocessing():
    """Run the G7 preprocessing pipeline"""
    print("\n" + "="*70)
    print("RUNNING G7 PREPROCESSING PIPELINE")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # Import and run preprocessing
        import G7_preprocessing
        result = subprocess.run([sys.executable, 'G7_preprocessing.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            preprocessing_time = time.time() - start_time
            print(f"✓ Preprocessing completed in {preprocessing_time:.1f} seconds")
            return True
        else:
            print(f"✗ Preprocessing failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        return False

def run_model_training():
    """Run the G7 model training and evaluation"""
    print("\n" + "="*70)
    print("RUNNING G7 MODEL TRAINING & EVALUATION")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # Import and run model training
        import G7_model
        result = subprocess.run([sys.executable, 'G7_model.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            training_time = time.time() - start_time
            print(f"✓ Model training completed in {training_time:.1f} seconds")
            return True
        else:
            print(f"✗ Model training failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Model training error: {e}")
        return False

def validate_results():
    """Validate generated results"""
    print("\nValidating results...")
    
    expected_files = [
        'G7_experimental_results.json',
        'G7_performance_summary.txt',
        'G7_voting_ensemble_models.pkl',
        'ensemble_feature_info.json'
    ]
    
    missing_results = []
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✓ {file_path} ({file_size:,} bytes)")
        else:
            missing_results.append(file_path)
            print(f"✗ {file_path} - MISSING")
    
    if missing_results:
        print(f"\nMissing result files: {missing_results}")
        return False
    
    print("✓ All result files generated successfully")
    return True

def display_summary():
    """Display experiment summary"""
    print("\n" + "="*70)
    print("G7 EXPERIMENT SUMMARY")
    print("="*70)
    
    try:
        import json
        
        # Load experimental results
        with open('G7_experimental_results.json', 'r') as f:
            results = json.load(f)
        
        print(f"Experiment: {results['experiment']}")
        print(f"Preprocessing: {results['preprocessing']}")
        print(f"Algorithm: {results['algorithm']}")
        print(f"Best Model: {results['best_model']}")
        print(f"Best Accuracy: {results['best_accuracy']:.4f}")
        print(f"Feature Sets: {results['feature_sets']}")
        print(f"Total Models: {results['total_models']}")
        print(f"Training Samples: {results['training_samples']}")
        print(f"Test Samples: {results['test_samples']}")
        
        # Load feature set info
        with open('ensemble_feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        print(f"\nFeature Sets ({len(feature_info['feature_sets'])}):")
        for name, info in feature_info['feature_sets'].items():
            print(f"  - {name}: {info['n_features']} features")
        
        print(f"\nPreprocessing Methods:")
        for method in feature_info['preprocessing_methods']:
            print(f"  - {method}")
        
        # Top performing models
        model_results = results['model_results']
        sorted_models = sorted(model_results.items(), 
                             key=lambda x: x[1]['accuracy'], 
                             reverse=True)
        
        print(f"\nTop 5 Models:")
        for i, (model_name, result) in enumerate(sorted_models[:5], 1):
            print(f"  {i}. {model_name}: {result['accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error loading results: {e}")
        return False

def main():
    """Main experiment runner"""
    print("="*80)
    print("G7 EXPERIMENT: Wavelet + SNV + Voting Ensemble")
    print("Advanced ensemble methods with multiple preprocessing paths")
    print("="*80)
    
    overall_start_time = time.time()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return False
    
    # Step 2: Check data availability
    if not check_data_availability():
        print("❌ Data availability check failed")
        return False
    
    # Step 3: Run preprocessing
    if not run_preprocessing():
        print("❌ Preprocessing failed")
        return False
    
    # Step 4: Run model training
    if not run_model_training():
        print("❌ Model training failed")
        return False
    
    # Step 5: Validate results
    if not validate_results():
        print("❌ Result validation failed")
        return False
    
    # Step 6: Display summary
    if not display_summary():
        print("❌ Summary generation failed")
        return False
    
    total_time = time.time() - overall_start_time
    
    print("\n" + "="*70)
    print("G7 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Total runtime: {total_time:.1f} seconds")
    print(f"Generated files:")
    
    # List all generated files
    generated_files = [
        'G7_experimental_results.json',
        'G7_performance_summary.txt', 
        'G7_voting_ensemble_models.pkl',
        'ensemble_feature_info.json'
    ]
    
    total_size = 0
    for file_path in generated_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            total_size += size
            print(f"  - {file_path} ({size:,} bytes)")
    
    print(f"\nTotal output size: {total_size:,} bytes")
    print("\n✅ G7 voting ensemble experiment ready for analysis!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 