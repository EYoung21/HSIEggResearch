#!/usr/bin/env python3
"""
G3 Experiment Runner: EMSC + Wavelets + CNN
===========================================

This script runs the complete G3 experiment pipeline:
1. EMSC (Extended Multiplicative Scatter Correction) preprocessing
2. Wavelet transform for time-frequency decomposition
3. CNN training with hyperparameter optimization
4. Model evaluation and results reporting

Experiment Design:
- Preprocessing: EMSC + Wavelet Transform (Daubechies 4, 4 levels)
- Algorithm: Convolutional Neural Network
- Input: 2D wavelet coefficient grids
- Optimization: Grid search with cross-validation
- Features: Multi-resolution time-frequency analysis

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
    print("🧠 G3 EXPERIMENT: EMSC + Wavelets + CNN")
    print("=" * 70)
    print("📋 Methodology:")
    print("   • Preprocessing: EMSC + Wavelet Transform")
    print("   • Features: 2D Wavelet coefficient grids")
    print("   • Algorithm: Convolutional Neural Network")
    print("   • Optimization: Grid search with cross-validation")
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

def check_dependencies():
    """Check if required packages are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        ('pywt', 'PyWavelets'),
        ('tensorflow', 'TensorFlow'),
        ('scipy', 'SciPy'),
    ]
    
    missing_packages = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {name} is installed")
        except ImportError:
            missing_packages.append((package, name))
            print(f"   ❌ {name} is missing")
    
    if missing_packages:
        print(f"\n❌ Missing required packages:")
        for package, name in missing_packages:
            print(f"   • {name} (pip install {package})")
        print("\n💡 Install missing packages:")
        print("   pip install PyWavelets tensorflow")
        return False
    
    print("   ✅ All required packages found")
    return True

def run_preprocessing():
    """Run G3 EMSC + Wavelet preprocessing pipeline"""
    print("\n" + "=" * 50)
    print("🔄 STEP 1: EMSC + WAVELET PREPROCESSING")
    print("=" * 50)
    print("• Applying EMSC (Extended Multiplicative Scatter Correction)")
    print("• Computing Wavelet Transform (Daubechies 4, 4 levels)")
    print("• Reshaping for CNN input (2D grids)")
    print("• Splitting train/test sets")
    
    start_time = time.time()
    
    try:
        # Run preprocessing
        result = subprocess.run([sys.executable, 'G3_preprocessing.py'], 
                              capture_output=True, text=True, check=True)
        
        print("✅ EMSC + Wavelet preprocessing completed successfully!")
        print(f"⏱️  Time elapsed: {time.time() - start_time:.1f} seconds")
        
        # Print key outputs
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-15:]:  # Show last 15 lines
                if any(keyword in line for keyword in ['shape', 'samples', 'CNN', 'dimensions']):
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

def run_cnn_modeling():
    """Run G3 CNN modeling with optimization"""
    print("\n" + "=" * 50)
    print("🧠 STEP 2: CNN MODELING")
    print("=" * 50)
    print("• Creating Convolutional Neural Network architecture")
    print("• Grid search hyperparameter optimization")
    print("• Training optimized CNN model")
    print("• Evaluating on test set")
    
    start_time = time.time()
    
    try:
        # Run CNN modeling
        result = subprocess.run([sys.executable, 'G3_model.py'], 
                              capture_output=True, text=True, check=True)
        
        print("✅ CNN modeling completed successfully!")
        print(f"⏱️  Time elapsed: {time.time() - start_time:.1f} seconds")
        
        # Extract and display key results
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['CV Score:', 'Test Accuracy:', '🎯']):
                    print(f"   {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ CNN modeling failed with error:")
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
        'X_train_processed.npy': 'Processed training features (4D CNN input)',
        'X_test_processed.npy': 'Processed test features (4D CNN input)',
        'y_train.npy': 'Training labels',
        'y_test.npy': 'Test labels',
        'cnn_input_shape.npy': 'CNN input dimensions',
        'emsc_wavelet_preprocessor.pkl': 'EMSC + Wavelet preprocessor',
        'label_encoder.pkl': 'Label encoder',
        'G3_cnn_model.h5': 'Trained CNN model',
        'G3_model_metadata.json': 'Model metadata and architecture',
        'test_predictions.csv': 'Test set predictions',
        'G3_experimental_results.json': 'Complete experimental results',
        'G3_performance_summary.txt': 'Human-readable summary'
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
    print("📊 G3 EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    # Try to read and display key results
    try:
        import json
        with open('G3_experimental_results.json', 'r') as f:
            results = json.load(f)
        
        print(f"🧬 Experiment: {results['experiment']}")
        print(f"⚙️  Preprocessing: {results['preprocessing']}")
        print(f"🧠 Algorithm: {results['algorithm']}")
        print(f"📐 Input Shape: {results['input_shape']}")
        print(f"🎯 Test Accuracy: {results['test_results']['accuracy']:.4f}")
        
        if 'hyperparameter_optimization' in results:
            opt_results = results['hyperparameter_optimization']
            print(f"\n📊 Hyperparameter Optimization:")
            print(f"   • Method: {opt_results['method']}")
            print(f"   • Trials: {opt_results['trials']}")
            print(f"   • Best CV Score: {opt_results['best_cv_score']:.4f}")
        
        if 'best_parameters' in results:
            best_params = results['best_parameters']
            print(f"\n🏆 Best CNN Architecture:")
            print(f"   • Conv Layers: {best_params.get('conv_layers', 'N/A')}")
            print(f"   • Starting Filters: {best_params.get('filters_start', 'N/A')}")
            print(f"   • Dropout Rate: {best_params.get('dropout_rate', 'N/A')}")
            print(f"   • Dense Units: {best_params.get('dense_units', 'N/A')}")
            print(f"   • Learning Rate: {best_params.get('learning_rate', 'N/A')}")
        
        if 'training_history' in results and results['training_history']:
            history = results['training_history']
            print(f"\n📈 Training Results:")
            print(f"   • Training Epochs: {history.get('epochs', 'N/A')}")
            if 'val_accuracy' in history and history['val_accuracy']:
                print(f"   • Final Val Accuracy: {history['val_accuracy'][-1]:.4f}")
        
    except Exception as e:
        print(f"⚠️  Could not read results file: {e}")
        print("   Please check G3_experimental_results.json manually")
    
    # Display file locations
    print(f"\n📁 Results saved in: {os.getcwd()}")
    print("   📋 G3_performance_summary.txt - Detailed analysis")
    print("   📊 G3_experimental_results.json - Complete results")
    print("   🧠 G3_cnn_model.h5 - Trained CNN model")
    print("   🔮 test_predictions.csv - Model predictions")

def main():
    """Main experiment execution"""
    print_header()
    
    # Step 0: Check dependencies
    if not check_dependencies():
        print("\n❌ Experiment aborted due to missing dependencies")
        sys.exit(1)
    
    # Step 1: Check data availability
    if not check_data_availability():
        print("\n❌ Experiment aborted due to missing data")
        sys.exit(1)
    
    # Step 2: Preprocessing
    if not run_preprocessing():
        print("\n❌ Experiment aborted due to preprocessing failure")
        sys.exit(1)
    
    # Step 3: CNN modeling
    if not run_cnn_modeling():
        print("\n❌ Experiment aborted due to modeling failure")
        sys.exit(1)
    
    # Step 4: Check outputs
    if not check_outputs():
        print("\n⚠️  Some output files are missing")
    
    # Step 5: Display results
    display_results()
    
    print("\n" + "=" * 70)
    print("🎉 G3 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("📝 Next steps:")
    print("   • Review G3_performance_summary.txt for detailed analysis")
    print("   • Compare results with G1 and G2 experiments")
    print("   • Analyze CNN learning patterns and convergence")
    print("   • Consider running G4-G8 experiments")
    print("   • Evaluate wavelet coefficients for biological insights")

if __name__ == "__main__":
    main() 