#!/usr/bin/env python3
"""
G4 Experiment Runner: Raw + Augmentation + Transformer
======================================================

This script runs the complete G4 experiment pipeline:
1. Minimal preprocessing (preserving raw spectral information)
2. Comprehensive data augmentation to address class imbalance
3. Transformer training with attention mechanism
4. Model evaluation and results reporting

Experiment Design:
- Preprocessing: Minimal (outlier removal + normalization only)
- Data Augmentation: Comprehensive suite (noise, scaling, shifting, etc.)
- Algorithm: Transformer with Multi-Head Self-Attention
- Input: Raw spectral sequences with positional encoding
- Optimization: Grid search with cross-validation
- Features: Complete preservation of spectral information

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
    print("🚀 G4 EXPERIMENT: Raw + Augmentation + Transformer")
    print("=" * 70)
    print("📋 Methodology:")
    print("   • Preprocessing: Minimal (preserve raw information)")
    print("   • Data Augmentation: Comprehensive suite")
    print("   • Algorithm: Transformer (Multi-Head Attention)")
    print("   • Features: Raw spectral sequences")
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
        ('tensorflow', 'TensorFlow'),
        ('sklearn', 'scikit-learn'),
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
        print("   pip install tensorflow scikit-learn")
        return False
    
    print("   ✅ All required packages found")
    return True

def run_preprocessing():
    """Run G4 augmentation preprocessing pipeline"""
    print("\n" + "=" * 50)
    print("🔄 STEP 1: RAW + AUGMENTATION PREPROCESSING")
    print("=" * 50)
    print("• Applying minimal preprocessing (preserve raw info)")
    print("• Comprehensive data augmentation suite")
    print("• Balancing classes with synthetic samples")
    print("• Preparing for Transformer input")
    
    start_time = time.time()
    
    try:
        # Run preprocessing
        result = subprocess.run([sys.executable, 'G4_preprocessing.py'], 
                              capture_output=True, text=True, check=True)
        
        print("✅ Raw + Augmentation preprocessing completed successfully!")
        print(f"⏱️  Time elapsed: {time.time() - start_time:.1f} seconds")
        
        # Print key outputs
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-15:]:  # Show last 15 lines
                if any(keyword in line for keyword in ['shape', 'samples', 'Transformer', 'distribution']):
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

def run_transformer_modeling():
    """Run G4 Transformer modeling with optimization"""
    print("\n" + "=" * 50)
    print("🧠 STEP 2: TRANSFORMER MODELING")
    print("=" * 50)
    print("• Creating Transformer architecture")
    print("• Grid search hyperparameter optimization")
    print("• Training optimized Transformer model")
    print("• Evaluating on test set")
    
    start_time = time.time()
    
    try:
        # Run Transformer modeling
        result = subprocess.run([sys.executable, 'G4_model.py'], 
                              capture_output=True, text=True, check=True)
        
        print("✅ Transformer modeling completed successfully!")
        print(f"⏱️  Time elapsed: {time.time() - start_time:.1f} seconds")
        
        # Extract and display key results
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['CV Score:', 'Test Accuracy:', '🎯']):
                    print(f"   {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Transformer modeling failed with error:")
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
        'X_train_processed.npy': 'Processed training features (3D Transformer input)',
        'X_test_processed.npy': 'Processed test features (3D Transformer input)',
        'y_train_augmented.npy': 'Augmented training labels',
        'y_test.npy': 'Test labels',
        'transformer_sequence_length.npy': 'Transformer sequence length',
        'raw_augmentation_preprocessor.pkl': 'Raw + Augmentation preprocessor',
        'label_encoder.pkl': 'Label encoder',
        'G4_transformer_model.h5': 'Trained Transformer model',
        'G4_model_metadata.json': 'Model metadata and architecture',
        'test_predictions.csv': 'Test set predictions',
        'G4_experimental_results.json': 'Complete experimental results',
        'G4_performance_summary.txt': 'Human-readable summary'
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
    print("📊 G4 EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    # Try to read and display key results
    try:
        import json
        with open('G4_experimental_results.json', 'r') as f:
            results = json.load(f)
        
        print(f"🧬 Experiment: {results['experiment']}")
        print(f"⚙️  Preprocessing: {results['preprocessing']}")
        print(f"🧠 Algorithm: {results['algorithm']}")
        print(f"📏 Sequence Length: {results['sequence_length']}")
        print(f"🎯 Test Accuracy: {results['test_results']['accuracy']:.4f}")
        
        if 'hyperparameter_optimization' in results:
            opt_results = results['hyperparameter_optimization']
            print(f"\n📊 Hyperparameter Optimization:")
            print(f"   • Method: {opt_results['method']}")
            print(f"   • Trials: {opt_results['trials']}")
            print(f"   • Best CV Score: {opt_results['best_cv_score']:.4f}")
        
        if 'best_parameters' in results:
            best_params = results['best_parameters']
            print(f"\n🏆 Best Transformer Architecture:")
            print(f"   • Embedding Dim: {best_params.get('embed_dim', 'N/A')}")
            print(f"   • Attention Heads: {best_params.get('num_heads', 'N/A')}")
            print(f"   • Feed-Forward Dim: {best_params.get('ff_dim', 'N/A')}")
            print(f"   • Transformer Layers: {best_params.get('num_layers', 'N/A')}")
            print(f"   • Dropout Rate: {best_params.get('dropout_rate', 'N/A')}")
            print(f"   • Learning Rate: {best_params.get('learning_rate', 'N/A')}")
        
        if 'training_history' in results and results['training_history']:
            history = results['training_history']
            print(f"\n📈 Training Results:")
            print(f"   • Training Epochs: {history.get('epochs', 'N/A')}")
            if 'val_accuracy' in history and history['val_accuracy']:
                print(f"   • Final Val Accuracy: {history['val_accuracy'][-1]:.4f}")
        
        # Data augmentation impact
        print(f"\n🔄 Data Augmentation Impact:")
        print(f"   • Training samples: {results['training_samples']}")
        print(f"   • Test samples: {results['test_samples']}")
        
    except Exception as e:
        print(f"⚠️  Could not read results file: {e}")
        print("   Please check G4_experimental_results.json manually")
    
    # Display file locations
    print(f"\n📁 Results saved in: {os.getcwd()}")
    print("   📋 G4_performance_summary.txt - Detailed analysis")
    print("   📊 G4_experimental_results.json - Complete results")
    print("   🧠 G4_transformer_model.h5 - Trained Transformer model")
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
    
    # Step 2: Preprocessing with augmentation
    if not run_preprocessing():
        print("\n❌ Experiment aborted due to preprocessing failure")
        sys.exit(1)
    
    # Step 3: Transformer modeling
    if not run_transformer_modeling():
        print("\n❌ Experiment aborted due to modeling failure")
        sys.exit(1)
    
    # Step 4: Check outputs
    if not check_outputs():
        print("\n⚠️  Some output files are missing")
    
    # Step 5: Display results
    display_results()
    
    print("\n" + "=" * 70)
    print("🎉 G4 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("📝 Next steps:")
    print("   • Review G4_performance_summary.txt for detailed analysis")
    print("   • Compare results with G1, G2, and G3 experiments")
    print("   • Analyze attention weights and spectral patterns")
    print("   • Evaluate data augmentation effectiveness")
    print("   • Consider running G5-G8 experiments")
    print("   • Test different augmentation strategies")

if __name__ == "__main__":
    main() 