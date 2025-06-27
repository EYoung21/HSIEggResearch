"""
M7 Experiment Runner: SNV + Mixup + Semi-Supervised Learning for Mortality Classification
Complete experiment pipeline combining SNV preprocessing with mixup augmentation and semi-supervised learning
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from M7_preprocessing import SNVMixupPreprocessor
from M7_model import M7MortalityClassifier

def load_mortality_data():
    """Load mortality data from CSV files"""
    print("Loading mortality classification data...")
    
    # Load reference metadata
    reference_path = "../../data/reference_metadata.csv"
    reference_df = pd.read_csv(reference_path)
    print(f"✓ Reference metadata loaded: {reference_df.shape}")
    
    # Load wavelengths 
    wavelengths_path = "../../gender/G1_MSC_SG_LightGBM/wavelengths.csv"
    wavelengths_df = pd.read_csv(wavelengths_path)
    wavelengths = wavelengths_df['wavelength'].values
    print(f"✓ Wavelengths loaded: {len(wavelengths)} features")
    
    # Load spectral data for all days
    all_data = []
    days = ['D0', 'D1', 'D2', 'D3', 'D4']
    
    for day in days:
        spectral_path = f"../../data/spectral_data_{day}.csv"
        if os.path.exists(spectral_path):
            day_data = pd.read_csv(spectral_path)
            day_data['Day'] = day
            
            # Merge with reference data
            merged_day = pd.merge(
                reference_df[['HSI sample ID', 'Mortality status']],
                day_data,
                on='HSI sample ID',
                how='inner'
            )
            
            if len(merged_day) > 0:
                merged_day = merged_day.rename(columns={
                    'HSI sample ID': 'ID',
                    'Mortality status': 'Mortality'
                })
                all_data.append(merged_day)
                print(f"✓ {day} loaded: {merged_day.shape}")
    
    # Combine all days
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"✓ Combined data: {combined_data.shape}")
    
    # Filter for mortality classification
    mortality_data = combined_data.dropna(subset=['Mortality'])
    
    # Binary classification mapping
    alive_statuses = ['live', 'Live', 'Still alive', 'Possibly still alive - left in incubator']
    dead_statuses = ['Early dead', 'Dead embryo', 'Did not hatch', 'Late dead; cannot tell']
    
    mortality_data = mortality_data[mortality_data['Mortality'].isin(alive_statuses + dead_statuses)]
    mortality_data['Mortality_Binary'] = mortality_data['Mortality'].apply(
        lambda x: 'Alive' if x in alive_statuses else 'Dead'
    )
    
    print(f"✓ Mortality data: {mortality_data.shape}")
    print(f"  Distribution: {mortality_data['Mortality_Binary'].value_counts().to_dict()}")
    
    # Extract features
    feature_columns = [col for col in mortality_data.columns 
                      if str(col).replace('.', '').replace('-', '').replace(' ', '').isdigit()]
    
    X = mortality_data[feature_columns].values
    y_labels = mortality_data['Mortality_Binary'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    
    print(f"✓ Features: {X.shape}")
    print(f"✓ Classes: {dict(zip(label_encoder.classes_, np.bincount(y)))}")
    
    return X, y, wavelengths, label_encoder

def run_M7_experiment():
    """Run complete M7 experiment"""
    print("="*80)
    print("M7 EXPERIMENT: SNV + MIXUP + SEMI-SUPERVISED LEARNING")
    print("="*80)
    
    start_time = time.time()
    results = {}
    
    try:
        # Load data
        X_raw, y, wavelengths, label_encoder = load_mortality_data()
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training: {X_train.shape} | Test: {X_test.shape}")
        
        # Preprocessing
        preprocessor = SNVMixupPreprocessor(
            enable_derivatives=True,
            mixup_alpha=0.2,
            augmentation_ratio=0.5,
            semi_supervised_ratio=0.3
        )
        
        X_labeled, y_labeled, X_unlabeled, labeled_idx, unlabeled_idx = preprocessor.preprocess_mortality_data(
            X_train, y_train, wavelengths
        )
        
        X_test_processed = preprocessor.transform_new_data(X_test, wavelengths)
        
        # Model training
        model = M7MortalityClassifier(random_state=42)
        model.fit(X_labeled, y_labeled, X_unlabeled)
        
        # Evaluation
        cv_results = model.cross_validate(X_labeled, y_labeled, cv=5)
        primary_metrics = model.evaluate_model(X_test_processed, y_test.astype(int), use_ensemble=False)
        ensemble_metrics = model.evaluate_model(X_test_processed, y_test.astype(int), use_ensemble=True)
        
        # Store results
        total_time = time.time() - start_time
        
        results = {
            'experiment_metadata': {
                'experiment_name': 'M7_SNV_Mixup_SemiSupervised',
                'total_runtime_seconds': total_time,
                'timestamp': datetime.now().isoformat(),
                'n_samples_total': X_raw.shape[0],
                'n_features_original': X_raw.shape[1],
                'n_features_processed': X_labeled.shape[1]
            },
            'preprocessing': preprocessor.preprocessing_stats,
            'training': model.training_stats,
            'cross_validation': cv_results,
            'test_primary': primary_metrics,
            'test_ensemble': ensemble_metrics
        }
        
        print(f"\n📊 M7 RESULTS")
        print(f"Runtime: {total_time:.2f}s")
        print(f"CV: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        print(f"Primary: {primary_metrics['accuracy']:.4f} accuracy")
        print(f"Ensemble: {ensemble_metrics['accuracy']:.4f} accuracy")
        
        print(f"\n✅ M7 experiment completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        results['error'] = {'message': str(e), 'timestamp': datetime.now().isoformat()}
    
    # Save results
    with open("M7_experimental_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    if 'test_ensemble' in results:
        create_summary(results)
    
    return results

def create_summary(results):
    """Create performance summary"""
    summary = [
        "M7 EXPERIMENT SUMMARY: SNV + Mixup + Semi-Supervised Learning",
        "=" * 60,
        "",
        f"Runtime: {results['experiment_metadata']['total_runtime_seconds']:.2f} seconds",
        f"Dataset: {results['experiment_metadata']['n_samples_total']} samples",
        f"Features: {results['experiment_metadata']['n_features_original']} → {results['experiment_metadata']['n_features_processed']}",
        "",
        "PREPROCESSING:",
        f"  • SNV normalization + Mixup augmentation",
        f"  • Semi-supervised split: {results['preprocessing']['semi_supervised_ratio']:.1%} unlabeled",
        f"  • Mixup samples: {results['preprocessing']['mixup_samples']}",
        "",
        "PERFORMANCE:",
        f"  • Cross-validation: {results['cross_validation']['cv_mean']:.4f} ± {results['cross_validation']['cv_std']:.4f}",
        f"  • Primary model: {results['test_primary']['accuracy']:.4f} accuracy",
        f"  • Ensemble model: {results['test_ensemble']['accuracy']:.4f} accuracy",
        f"  • AUC: {results['test_ensemble']['auc']:.4f}",
        "",
        "COMPARISON:",
        "  • M2: 93.48% accuracy (SNV + Derivatives + Ensemble)",
        "  • M5: 80.33% accuracy (Wavelets + MSC + MultiTask)",
        f"  • M7: {results['test_ensemble']['accuracy']:.2%} accuracy (SNV + Mixup + Semi-Supervised)",
        ""
    ]
    
    with open("M7_performance_summary.txt", 'w') as f:
        f.write('\n'.join(summary))
    
    print("📝 Summary saved to M7_performance_summary.txt")

def main():
    return run_M7_experiment()

if __name__ == "__main__":
    main() 