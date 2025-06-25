#!/usr/bin/env python3
"""
G5 Experiment Runner: MSC + Multi-Scale Derivatives + Multi-Task Learning
=========================================================================

This script runs the complete G5 experiment pipeline:
1. MSC correction (proven successful from G1)
2. Multi-scale Savitzky-Golay derivatives (1st, 2nd, 3rd order)
3. Multi-task deep learning with shared feature extraction
4. Simultaneous gender and mortality prediction
5. Model evaluation and results reporting

Experiment Design:
- Preprocessing: MSC + Multi-scale SG derivatives
- Algorithm: Multi-Task Deep Learning (shared + task-specific heads)
- Tasks: Gender classification + Mortality prediction
- Features: 4x derivative scales (original + 1st + 2nd + 3rd)
- Optimization: Grid search with cross-validation

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
    print("🧬 G5 EXPERIMENT: MSC + Multi-Scale Derivatives + Multi-Task Learning")
    print("=" * 70)
    print("📋 Methodology:")
    print("   • Preprocessing: MSC + Multi-scale SG derivatives")
    print("   • Algorithm: Multi-Task Deep Learning")
    print("   • Tasks: Gender + Mortality (simultaneous)")
    print("   • Features: 4x derivative scales")
    print("   • Architecture: Shared + Task-specific heads")
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
        ('tensorflow', 'TensorFlow'),
        ('sklearn', 'scikit-learn'),
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
            if package == 'sklearn':
                print(f"   • {name} (pip install scikit-learn)")
            else:
                print(f"   • {name} (pip install {package})")
        print("\n💡 Install missing packages:")
        print("   pip install tensorflow scikit-learn scipy")
        return False
    
    print("   ✅ All required packages found")
    return True

def run_preprocessing():
    """Run G5 multi-scale derivatives preprocessing pipeline"""
    print("\n" + "=" * 50)
    print("🔄 STEP 1: MULTI-SCALE DERIVATIVES PREPROCESSING")
    print("=" * 50)
    print("• Applying MSC (Multiplicative Scatter Correction)")
    print("• Computing multi-scale SG derivatives (1st, 2nd, 3rd)")
    print("• Preparing multi-task dataset (Gender + Mortality)")
    print("• Creating stratified train/test splits")
    
    start_time = time.time()
    
    try:
        # Run preprocessing
        result = subprocess.run([sys.executable, 'G5_preprocessing.py'], 
                              capture_output=True, text=True, check=True)
        
        print("✅ Multi-scale derivatives preprocessing completed successfully!")
        print(f"⏱️  Time elapsed: {time.time() - start_time:.1f} seconds")
        
        # Print key outputs
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-20:]:  # Show last 20 lines
                if any(keyword in line for keyword in ['shape', 'samples', 'features', 'Task', 'Ready']):
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

def run_multi_task_modeling():
    """Run G5 multi-task deep learning modeling"""
    print("\n" + "=" * 50)
    print("🧠 STEP 2: MULTI-TASK DEEP LEARNING")
    print("=" * 50)
    print("• Creating shared feature extraction network")
    print("• Adding task-specific heads for Gender + Mortality")
    print("• Grid search hyperparameter optimization")
    print("• Training optimized multi-task model")
    print("• Evaluating both tasks on test set")
    
    start_time = time.time()
    
    try:
        # Run multi-task modeling
        result = subprocess.run([sys.executable, 'G5_model.py'], 
                              capture_output=True, text=True, check=True)
        
        print("✅ Multi-task modeling completed successfully!")
        print(f"⏱️  Time elapsed: {time.time() - start_time:.1f} seconds")
        
        # Extract and display key results
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['CV Score:', 'Test Results:', '🎯']):
                    print(f"   {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Multi-task modeling failed with error:")
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
        'X_train_processed.npy': 'Processed training features (multi-scale derivatives)',
        'X_test_processed.npy': 'Processed test features (multi-scale derivatives)',
        'y_gender_train.npy': 'Gender training labels',
        'y_gender_test.npy': 'Gender test labels',
        'y_mortality_train.npy': 'Mortality training labels',
        'y_mortality_test.npy': 'Mortality test labels',
        'multi_scale_preprocessor.pkl': 'MSC + Multi-scale derivatives preprocessor',
        'gender_encoder.pkl': 'Gender label encoder',
        'mortality_encoder.pkl': 'Mortality label encoder',
        'feature_info.json': 'Feature information and preprocessing details',
        'G5_multitask_model.h5': 'Trained multi-task deep learning model',
        'G5_model_metadata.json': 'Model metadata and architecture',
        'test_predictions.csv': 'Test set predictions for both tasks',
        'G5_experimental_results.json': 'Complete experimental results',
        'G5_performance_summary.txt': 'Human-readable summary'
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
    print("📊 G5 EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    # Try to read and display key results
    try:
        import json
        with open('G5_experimental_results.json', 'r') as f:
            results = json.load(f)
        
        print(f"🧬 Experiment: {results['experiment']}")
        print(f"⚙️  Preprocessing: {results['preprocessing']}")
        print(f"🧠 Algorithm: {results['algorithm']}")
        print(f"📋 Tasks: {', '.join(results['tasks'])}")
        print(f"📏 Input Features: {results['input_features']}")
        
        # Multi-task results
        test_results = results['test_results']
        print(f"\n🎯 Multi-Task Test Results:")
        print(f"   • Gender Task: {test_results['gender_accuracy']:.4f}")
        print(f"   • Mortality Task: {test_results['mortality_accuracy']:.4f}")
        print(f"   • Combined: {test_results['combined_accuracy']:.4f}")
        
        if 'hyperparameter_optimization' in results:
            opt_results = results['hyperparameter_optimization']
            print(f"\n📊 Hyperparameter Optimization:")
            print(f"   • Method: {opt_results['method']}")
            print(f"   • Trials: {opt_results['trials']}")
            print(f"   • Best Combined CV Score: {opt_results['best_combined_cv_score']:.4f}")
        
        if 'best_parameters' in results:
            best_params = results['best_parameters']
            print(f"\n🏆 Best Multi-Task Architecture:")
            print(f"   • Shared Layers: {best_params.get('hidden_layers', 'N/A')}")
            print(f"   • Dropout Rate: {best_params.get('dropout_rate', 'N/A')}")
            print(f"   • Learning Rate: {best_params.get('learning_rate', 'N/A')}")
            print(f"   • Task Weights: Gender={best_params.get('task_weight_gender', 'N/A')}, "
                  f"Mortality={best_params.get('task_weight_mortality', 'N/A')}")
        
        if 'training_history' in results and results['training_history']:
            history = results['training_history']
            print(f"\n📈 Training Results:")
            print(f"   • Training Epochs: {history.get('epochs', 'N/A')}")
            if 'val_gender_output_accuracy' in history and history['val_gender_output_accuracy']:
                print(f"   • Final Val Gender Accuracy: {history['val_gender_output_accuracy'][-1]:.4f}")
            if 'val_mortality_output_accuracy' in history and history['val_mortality_output_accuracy']:
                print(f"   • Final Val Mortality Accuracy: {history['val_mortality_output_accuracy'][-1]:.4f}")
        
        # Data summary
        print(f"\n🔄 Multi-Task Dataset:")
        print(f"   • Training samples: {results['training_samples']}")
        print(f"   • Test samples: {results['test_samples']}")
        print(f"   • Feature scaling: 4x (original + 1st + 2nd + 3rd derivatives)")
        
    except Exception as e:
        print(f"⚠️  Could not read results file: {e}")
        print("   Please check G5_experimental_results.json manually")
    
    # Display file locations
    print(f"\n📁 Results saved in: {os.getcwd()}")
    print("   📋 G5_performance_summary.txt - Detailed analysis")
    print("   📊 G5_experimental_results.json - Complete results")
    print("   🧠 G5_multitask_model.h5 - Trained multi-task model")
    print("   🔮 test_predictions.csv - Multi-task predictions")

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
    
    # Step 2: Multi-scale derivatives preprocessing
    if not run_preprocessing():
        print("\n❌ Experiment aborted due to preprocessing failure")
        sys.exit(1)
    
    # Step 3: Multi-task deep learning modeling
    if not run_multi_task_modeling():
        print("\n❌ Experiment aborted due to modeling failure")
        sys.exit(1)
    
    # Step 4: Check outputs
    if not check_outputs():
        print("\n⚠️  Some output files are missing")
    
    # Step 5: Display results
    display_results()
    
    print("\n" + "=" * 70)
    print("🎉 G5 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("📝 Next steps:")
    print("   • Review G5_performance_summary.txt for detailed analysis")
    print("   • Compare multi-task results with single-task experiments")
    print("   • Analyze shared vs task-specific feature importance")
    print("   • Consider running G6-G8 experiments")
    print("   • Explore joint prediction patterns and correlations")
    print("   • Evaluate transfer learning potential")

if __name__ == "__main__":
    main() 