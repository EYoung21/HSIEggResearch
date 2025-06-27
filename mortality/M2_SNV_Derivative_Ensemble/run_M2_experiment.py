"""
Run M2 Experiment: SNV + Derivatives + Ensemble for Mortality Classification
Complete pipeline execution for mortality prediction using advanced preprocessing and ensemble models
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from M2_preprocessing import MortalityPreprocessor
from M2_model import MortalityEnsemble

def load_mortality_data():
    """Load and prepare mortality classification data"""
    print("Loading mortality classification data...")
    
    # Load reference metadata
    reference_path = "../../data/reference_metadata.csv"
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference data not found: {reference_path}")
    
    reference_df = pd.read_csv(reference_path)
    print(f"✓ Reference metadata loaded: {reference_df.shape}")
    
    # Load wavelengths from G1 experiment
    wavelengths_path = "../../gender/G1_MSC_SG_LightGBM/wavelengths.csv"
    if not os.path.exists(wavelengths_path):
        raise FileNotFoundError(f"Wavelengths data not found: {wavelengths_path}")
    
    wavelengths_df = pd.read_csv(wavelengths_path)
    wavelengths = wavelengths_df['wavelength'].values
    print(f"✓ Wavelengths loaded: {len(wavelengths)} wavelengths ({wavelengths.min():.2f} - {wavelengths.max():.2f} nm)")
    
    # Load spectral data for all days
    all_spectral_data = []
    days = ['D0', 'D1', 'D2', 'D3', 'D4']
    
    for day in days:
        spectral_path = f"../../data/spectral_data_{day}.csv"
        if os.path.exists(spectral_path):
            day_df = pd.read_csv(spectral_path)
            all_spectral_data.append(day_df)
            print(f"✓ {day} spectral data loaded: {day_df.shape}")
        else:
            print(f"Warning: {spectral_path} not found, skipping...")
    
    if not all_spectral_data:
        raise FileNotFoundError("No spectral data files found")
    
    # Combine all spectral measurements
    combined_spectral = pd.concat(all_spectral_data, ignore_index=True)
    print(f"✓ Combined spectral data: {combined_spectral.shape}")
    
    # Merge with reference data on sample ID
    if 'sample_id' in combined_spectral.columns and 'Sample ID' in reference_df.columns:
        merged_df = combined_spectral.merge(reference_df, left_on='sample_id', right_on='Sample ID', how='inner')
    elif 'Sample ID' in combined_spectral.columns:
        merged_df = combined_spectral.merge(reference_df, on='Sample ID', how='inner')
    else:
        # Use index-based merging if no sample ID column
        combined_spectral.reset_index(drop=True, inplace=True)
        reference_df.reset_index(drop=True, inplace=True)
        merged_df = pd.concat([combined_spectral, reference_df], axis=1)
    
    print(f"✓ Data merged: {merged_df.shape}")
    
    # Filter samples with mortality status
    mortality_column = None
    for col in ['Mortality status', 'Mortality', 'mortality_status', 'mortality']:
        if col in merged_df.columns:
            mortality_column = col
            break
    
    if mortality_column is None:
        raise ValueError("No mortality status column found")
    
    # Clean mortality data
    mortality_df = merged_df.dropna(subset=[mortality_column])
    # Map mortality status to binary labels
    mortality_mapping = {
        'Live': 0, 'live': 0, 'Live ': 0,  # Alive
        'Early dead': 1, 'Early_dead': 1, 'Dead': 1  # Dead
    }
    
    # Filter for valid mortality status
    valid_mortality = mortality_df[mortality_column].isin(mortality_mapping.keys())
    mortality_df = mortality_df[valid_mortality]
    
    # Apply mapping
    mortality_df[mortality_column] = mortality_df[mortality_column].map(mortality_mapping)
    
    print(f"✓ Mortality data cleaned: {mortality_df.shape}")
    print(f"  - Mortality distribution: {mortality_df[mortality_column].value_counts().to_dict()}")
    
    # Extract spectral features (wavelength columns)
    wavelength_columns = []
    for col in mortality_df.columns:
        try:
            if col.replace('.', '').replace('_', '').isdigit() or \
               ('nm' in str(col).lower()) or \
               (isinstance(col, (int, float)) and 300 <= float(col) <= 1100):
                wavelength_columns.append(col)
        except:
            continue
    
    if not wavelength_columns:
        # Use wavelength range as backup
        wavelength_columns = [col for col in mortality_df.columns 
                            if col not in [mortality_column, 'Sample ID', 'sample_id', 'day', 'Day'] and
                            mortality_df[col].dtype in ['float64', 'int64']]
    
    print(f"✓ Identified {len(wavelength_columns)} wavelength features")
    
    # Extract features and labels
    X = mortality_df[wavelength_columns].values
    y_raw = mortality_df[mortality_column].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    print(f"✓ Data prepared for mortality classification:")
    print(f"  - Features: {X.shape}")
    print(f"  - Labels: {len(np.unique(y))} classes")
    print(f"  - Classes: {label_encoder.classes_}")
    print(f"  - Class distribution: {np.bincount(y)}")
    
    return X, y, wavelengths, label_encoder

def main():
    """Run complete M2 experiment"""
    print("="*80)
    print("M2 EXPERIMENT: SNV + DERIVATIVES + ENSEMBLE FOR MORTALITY")
    print("="*80)
    
    try:
        # Load data
        X, y, wavelengths, label_encoder = load_mortality_data()
        
        # Train-test split
        print(f"\nSplitting data (80/20 train/test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        print(f"  - Train classes: {np.bincount(y_train)}")
        print(f"  - Test classes: {np.bincount(y_test)}")
        
        # Initialize preprocessor
        print(f"\nInitializing M2 preprocessor...")
        preprocessor = MortalityPreprocessor(
            derivative_window=15,
            polynomial_order=3
        )
        
        # Preprocess training data
        X_train_processed = preprocessor.preprocess_mortality_data(X_train, wavelengths)
        
        # Preprocess test data
        print("\nPreprocessing test data...")
        X_test_processed = preprocessor.transform_new_data(X_test, wavelengths)
        print(f"✓ Test data processed: {X_test_processed.shape}")
        
        # Initialize and train model
        print(f"\nInitializing M2 ensemble model...")
        model = MortalityEnsemble(random_state=42)
        
        # Train the ensemble
        training_results = model.train_mortality_ensemble(X_train_processed, y_train)
        
        # Evaluate on test set
        evaluation_results = model.evaluate_mortality_prediction(
            X_test_processed, y_test, label_encoder
        )
        
        # Analyze feature importance
        feature_analysis = model.analyze_feature_importance(
            preprocessor.preprocessing_stats, top_k=15
        )
        
        # Save results
        model.save_model_and_results(
            evaluation_results, feature_analysis, preprocessor.preprocessing_stats
        )
        
        # Create feature importance file
        if 'feature_importance' in feature_analysis:
            importance_df = pd.DataFrame(feature_analysis['feature_importance'])
            importance_df.to_csv('feature_importance.csv', index=False)
            print("✓ Feature importance saved to feature_importance.csv")
        
        # Print final summary
        print("\n" + "="*80)
        print("M2 EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        
        best_model = evaluation_results['best_model']['name'].replace('_', ' ').title()
        best_accuracy = evaluation_results['best_model']['accuracy']
        best_auc = evaluation_results['best_model']['auc']
        
        print(f"Best Model: {best_model}")
        print(f"Test Accuracy: {best_accuracy:.4f}")
        print(f"Test AUC: {best_auc:.4f}")
        print(f"Training Time: {training_results['training_time']:.2f} seconds")
        print(f"Enhanced Features: {X_train_processed.shape[1]} (from {X_train.shape[1]} wavelengths)")
        
        # Individual model performance
        print(f"\nIndividual Model Performance:")
        for model_name in ['random_forest', 'svm', 'xgboost', 'ensemble']:
            if model_name in evaluation_results:
                acc = evaluation_results[model_name]['accuracy']
                auc = evaluation_results[model_name]['auc']
                print(f"  {model_name.replace('_', ' ').title()}: {acc:.4f} (AUC: {auc:.4f})")
        
        return True
        
    except Exception as e:
        print(f"\n❌ M2 experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 M2 experiment completed successfully!")
    else:
        print("\n💥 M2 experiment failed. Check error messages above.")
        sys.exit(1) 