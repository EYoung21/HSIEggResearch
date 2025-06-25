#!/usr/bin/env python3
"""
G6 Experiment Runner: SNV + Optimized + Transfer Learning
Advanced transfer learning approach for gender prediction using HSI egg data
"""

import os
import sys
import time
import subprocess
import importlib.util
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'scipy', 'tensorflow', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package} (missing)")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + ' '.join(missing_packages))
        return False
    
    print("✅ All dependencies satisfied!")
    return True

def check_data_files():
    """Check if required data files exist"""
    print("\n🔍 Checking data files...")
    
    required_files = [
        '../../data/reference_metadata.csv',
        '../../data/spectral_data_D0.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"  ✓ {file_path} ({file_size:.1f} MB)")
        else:
            missing_files.append(file_path)
            print(f"  ✗ {file_path} (missing)")
    
    if missing_files:
        print(f"\n❌ Missing data files: {missing_files}")
        return False
    
    print("✅ All data files found!")
    return True

def run_preprocessing():
    """Run the G6 preprocessing pipeline"""
    print("\n" + "="*60)
    print("🔬 RUNNING G6 PREPROCESSING")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Import and run preprocessing
        from G6_preprocessing import main as preprocessing_main
        preprocessing_main()
        
        preprocessing_time = time.time() - start_time
        print(f"\n✅ Preprocessing completed in {preprocessing_time:.1f} seconds")
        
        return True, preprocessing_time
        
    except Exception as e:
        print(f"\n❌ Preprocessing failed: {str(e)}")
        return False, 0

def run_transfer_learning():
    """Run the G6 transfer learning model"""
    print("\n" + "="*60)
    print("🧠 RUNNING G6 TRANSFER LEARNING")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Import and run transfer learning
        from G6_model import main as model_main
        model_main()
        
        modeling_time = time.time() - start_time
        print(f"\n✅ Transfer learning completed in {modeling_time:.1f} seconds")
        
        return True, modeling_time
        
    except Exception as e:
        print(f"\n❌ Transfer learning failed: {str(e)}")
        return False, 0

def validate_outputs():
    """Validate that all expected output files were generated"""
    print("\n🔍 Validating output files...")
    
    expected_files = [
        'X_train_processed.npy',
        'X_test_processed.npy', 
        'y_train.npy',
        'y_test.npy',
        'optimized_snv_preprocessor.pkl',
        'label_encoder.pkl',
        'feature_info.json',
        'G6_autoencoder.h5',
        'G6_encoder.h5',
        'G6_transfer_classifier.h5',
        'G6_classical_models.pkl',
        'G6_experimental_results.json',
        'G6_performance_summary.txt'
    ]
    
    missing_files = []
    total_size = 0
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            total_size += file_size
            size_mb = file_size / (1024*1024)
            print(f"  ✓ {file_path} ({size_mb:.2f} MB)")
        else:
            missing_files.append(file_path)
            print(f"  ✗ {file_path} (missing)")
    
    if missing_files:
        print(f"\n⚠️  Missing output files: {missing_files}")
        return False, len(expected_files) - len(missing_files), len(expected_files)
    
    total_size_mb = total_size / (1024*1024)
    print(f"\n✅ All {len(expected_files)} files generated ({total_size_mb:.1f} MB total)")
    return True, len(expected_files), len(expected_files)

def display_results():
    """Display the final experimental results"""
    print("\n" + "="*60)
    print("📊 G6 EXPERIMENTAL RESULTS")
    print("="*60)
    
    try:
        # Read performance summary
        if os.path.exists('G6_performance_summary.txt'):
            with open('G6_performance_summary.txt', 'r') as f:
                summary = f.read()
                print(summary)
        else:
            print("❌ Performance summary not found")
            
        # Read experimental results
        if os.path.exists('G6_experimental_results.json'):
            import json
            with open('G6_experimental_results.json', 'r') as f:
                results = json.load(f)
                
            print("\n" + "="*40)
            print("🎯 KEY RESULTS")
            print("="*40)
            print(f"Best Model: {results.get('best_model', 'Unknown')}")
            print(f"Best Accuracy: {results.get('best_accuracy', 'Unknown'):.4f}")
            print(f"Input Features: {results.get('input_features', 'Unknown')}")
            print(f"Training Samples: {results.get('training_samples', 'Unknown')}")
            print(f"Test Samples: {results.get('test_samples', 'Unknown')}")
            
            # Model comparison
            print(f"\n📈 MODEL COMPARISON:")
            if 'model_results' in results:
                for model_name, model_result in results['model_results'].items():
                    accuracy = model_result.get('accuracy', 0)
                    print(f"  - {model_name.replace('_', ' ').title()}: {accuracy:.4f}")
        else:
            print("❌ Experimental results not found")
            
    except Exception as e:
        print(f"❌ Error displaying results: {str(e)}")

def cleanup_intermediate_files():
    """Clean up intermediate files (optional)"""
    print("\n🧹 Cleaning up intermediate files...")
    
    intermediate_files = [
        'X_train_processed.npy',
        'X_test_processed.npy',
        'y_train.npy', 
        'y_test.npy'
    ]
    
    cleaned_count = 0
    for file_path in intermediate_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"  ✓ Removed {file_path}")
                cleaned_count += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {file_path}: {e}")
    
    if cleaned_count > 0:
        print(f"✅ Cleaned up {cleaned_count} intermediate files")
    else:
        print("ℹ️  No intermediate files to clean")

def main():
    """
    Main experiment runner for G6: SNV + Optimized + Transfer Learning
    """
    print("="*70)
    print("🚀 G6 EXPERIMENT: SNV + Optimized + Transfer Learning")
    print("="*70)
    print("Advanced transfer learning approach for gender prediction")
    print("Methodology: SNV + Feature Optimization + Autoencoder + Deep/Ensemble")
    print("="*70)
    
    total_start_time = time.time()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Step 2: Check data files
    if not check_data_files():
        print("\n❌ Data file check failed. Please ensure data files are available.")
        sys.exit(1)
    
    # Step 3: Run preprocessing
    preprocessing_success, preprocessing_time = run_preprocessing()
    if not preprocessing_success:
        print("\n❌ Preprocessing failed. Experiment terminated.")
        sys.exit(1)
    
    # Step 4: Run transfer learning
    modeling_success, modeling_time = run_transfer_learning()
    if not modeling_success:
        print("\n❌ Transfer learning failed. Experiment terminated.")
        sys.exit(1)
    
    # Step 5: Validate outputs
    validation_success, files_generated, files_expected = validate_outputs()
    
    # Step 6: Display results
    display_results()
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Final summary
    print("\n" + "="*60)
    print("📋 EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Preprocessing time: {preprocessing_time:.1f} seconds")
    print(f"Transfer learning time: {modeling_time:.1f} seconds")
    print(f"Total execution time: {total_time:.1f} seconds")
    print(f"Files generated: {files_generated}/{files_expected}")
    
    if validation_success and preprocessing_success and modeling_success:
        print("\n🎉 G6 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("✅ All components executed without errors")
        print("✅ All expected files generated")
        print("✅ Transfer learning pipeline complete")
        
        # Ask about cleanup
        try:
            response = input("\n🧹 Clean up intermediate files? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                cleanup_intermediate_files()
        except KeyboardInterrupt:
            print("\n\nExperiment completed. Intermediate files retained.")
        
    else:
        print("\n⚠️  EXPERIMENT COMPLETED WITH ISSUES")
        if not preprocessing_success:
            print("❌ Preprocessing failed")
        if not modeling_success:
            print("❌ Transfer learning failed")
        if not validation_success:
            print(f"❌ Output validation failed ({files_generated}/{files_expected} files)")
    
    print(f"\n📁 Results saved in: {os.getcwd()}")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        print("Partial results may be available in the current directory")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("Please check the error messages above for details")
        sys.exit(1) 