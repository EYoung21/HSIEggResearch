"""
M8 Experiment Runner: Multi-Derivatives + Conformal Prediction
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from M8_preprocessing import MultiDerivativeProcessor
from M8_model import M8MortalityClassifier

def load_mortality_data():
    """Load mortality data"""
    print("Loading mortality data...")
    
    reference_path = "../../data/reference_metadata.csv"
    reference_df = pd.read_csv(reference_path)
    print(f"✓ Reference loaded: {reference_df.shape}")
    
    wavelengths_path = "../../gender/G1_MSC_SG_LightGBM/wavelengths.csv"
    wavelengths_df = pd.read_csv(wavelengths_path)
    wavelengths = wavelengths_df['wavelength'].values
    print(f"✓ Wavelengths loaded: {len(wavelengths)}")
    
    all_data = []
    days = ['D0', 'D1', 'D2', 'D3', 'D4']
    
    for day in days:
        spectral_path = f"../../data/spectral_data_{day}.csv"
        if os.path.exists(spectral_path):
            day_data = pd.read_csv(spectral_path)
            day_data['Day'] = day
            
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
                print(f"✓ {day}: {merged_day.shape}")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    mortality_data = combined_data.dropna(subset=['Mortality'])
    
    alive_statuses = ['live', 'Live', 'Still alive', 'Possibly still alive - left in incubator']
    dead_statuses = ['Early dead', 'Dead embryo', 'Did not hatch', 'Late dead; cannot tell']
    
    mortality_data = mortality_data[mortality_data['Mortality'].isin(alive_statuses + dead_statuses)]
    mortality_data['Mortality_Binary'] = mortality_data['Mortality'].apply(
        lambda x: 'Alive' if x in alive_statuses else 'Dead'
    )
    
    print(f"✓ Mortality data: {mortality_data.shape}")
    
    feature_columns = [col for col in mortality_data.columns 
                      if str(col).replace('.', '').replace('-', '').replace(' ', '').isdigit()]
    
    X = mortality_data[feature_columns].values
    y_labels = mortality_data['Mortality_Binary'].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    
    return X, y, wavelengths, label_encoder

def run_M8_experiment():
    """Run M8 experiment"""
    print("="*80)
    print("M8 EXPERIMENT: MULTI-DERIVATIVES + CONFORMAL PREDICTION")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Load data
        X_raw, y, wavelengths, label_encoder = load_mortality_data()
        
        # Initial split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_raw, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Preprocessing with conformal splits
        preprocessor = MultiDerivativeProcessor(
            conformal_alpha=0.1,
            use_robust_scaling=True
        )
        
        X_train, X_cal, y_train, y_cal = preprocessor.preprocess_mortality_data(
            X_temp, y_temp, wavelengths
        )
        
        X_test_processed = preprocessor.transform_new_data(X_test, wavelengths)
        
        # Model training
        model = M8MortalityClassifier(random_state=42)
        model.fit(X_train, X_cal, y_train, y_cal)
        
        # Evaluation
        X_combined_train = np.vstack([X_train, X_cal])
        y_combined_train = np.hstack([y_train, y_cal])
        cv_results = model.cross_validate(X_combined_train, y_combined_train, cv=5)
        
        individual_metrics = model.evaluate_model(X_test_processed, y_test.astype(int), use_ensemble=False)
        ensemble_metrics = model.evaluate_model(X_test_processed, y_test.astype(int), use_ensemble=True)
        uncertainty_metrics = model.get_uncertainty_metrics(X_test_processed, y_test.astype(int))
        
        total_time = time.time() - start_time
        
        results = {
            'experiment_metadata': {
                'experiment_name': 'M8_MultiDerivatives_Conformal',
                'total_runtime_seconds': total_time,
                'timestamp': datetime.now().isoformat(),
                'n_samples_total': X_raw.shape[0],
                'n_features_original': X_raw.shape[1],
                'n_features_processed': X_train.shape[1]
            },
            'preprocessing': preprocessor.preprocessing_stats,
            'training': model.training_stats,
            'cross_validation': cv_results,
            'test_individual': individual_metrics,
            'test_ensemble': ensemble_metrics,
            'uncertainty_analysis': uncertainty_metrics
        }
        
        print(f"\n📊 M8 RESULTS")
        print(f"Runtime: {total_time:.2f}s")
        print(f"CV: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        print(f"Ensemble: {ensemble_metrics['accuracy']:.4f} accuracy")
        print(f"Uncertainty coverage: {uncertainty_metrics.get('empirical_coverage', 0):.4f}")
        
        print(f"\n✅ M8 completed!")
        
        # Save results
        with open("M8_experimental_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        create_summary(results)
        return results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {'error': str(e)}

def create_summary(results):
    """Create summary"""
    summary = [
        "M8 EXPERIMENT SUMMARY: Multi-Derivatives + Conformal Prediction",
        "=" * 60,
        "",
        f"Runtime: {results['experiment_metadata']['total_runtime_seconds']:.2f}s",
        f"Features: {results['experiment_metadata']['n_features_original']} → {results['experiment_metadata']['n_features_processed']}",
        f"Ensemble accuracy: {results['test_ensemble']['accuracy']:.4f}",
        f"Uncertainty coverage: {results['uncertainty_analysis'].get('empirical_coverage', 0):.4f}",
        ""
    ]
    
    with open("M8_performance_summary.txt", 'w') as f:
        f.write('\n'.join(summary))

if __name__ == "__main__":
    run_M8_experiment()
