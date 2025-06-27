"""
Run M4 Experiment: Raw + SMOTE + Transfer Learning for Mortality Classification
Complete pipeline execution for mortality prediction using raw spectral data and transfer learning
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

from M4_preprocessing import RawTransferPreprocessor
from M4_model import MortalityTransferModel

def load_mortality_data():
    """Load and prepare mortality classification data"""
    print("Loading mortality classification data...")
    
    # Load reference metadata
    reference_path = "../../data/reference_metadata.csv"
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference data not found: {reference_path}")
    
    reference_df = pd.read_csv(reference_path)
    print(f"✓ Reference metadata loaded: {reference_df.shape}")
    print(f"Columns: {list(reference_df.columns)}")
    
    # Load wavelengths from G1 experiment
    wavelengths_path = "../../gender/G1_MSC_SG_LightGBM/wavelengths.csv"
    if not os.path.exists(wavelengths_path):
        raise FileNotFoundError(f"Wavelengths data not found: {wavelengths_path}")
    
    wavelengths_df = pd.read_csv(wavelengths_path)
    wavelengths = wavelengths_df['wavelength'].values
    print(f"✓ Wavelengths loaded: {len(wavelengths)} wavelengths ({wavelengths.min():.2f} - {wavelengths.max():.2f} nm)")
    
    # Load spectral data for all days
    all_data = []
    days = ['D0', 'D1', 'D2', 'D3', 'D4']
    
    for day in days:
        spectral_path = f"../../data/spectral_data_{day}.csv"
        if os.path.exists(spectral_path):
            day_data = pd.read_csv(spectral_path)
            print(f"✓ {day} spectral data loaded: {day_data.shape}")
            
            # Add day information
            day_data['Day'] = day
            
            # Merge with reference data for this day's samples
            # Use the original column names that exist in both files
            merged_day = pd.merge(
                reference_df[['HSI sample ID', 'Mortality status']],
                day_data,
                on='HSI sample ID',
                how='inner'
            )
            
            if len(merged_day) > 0:
                # Rename columns for consistency
                merged_day = merged_day.rename(columns={
                    'HSI sample ID': 'ID',
                    'Mortality status': 'Mortality'
                })
                all_data.append(merged_day)
                print(f"✓ {day} merged data: {merged_day.shape}")
            else:
                print(f"⚠ Warning: No matching data for {day}")
        else:
            print(f"⚠ Warning: {day} spectral data not found")
    
    if not all_data:
        raise FileNotFoundError("No merged spectral data found")
    
    # Combine all days
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"✓ Combined data from all days: {combined_data.shape}")
    
    # Filter for mortality classification (remove unknown/NaN)
    mortality_data = combined_data.dropna(subset=['Mortality'])
    
    # Map mortality status to binary classification
    # Alive: 'live', 'Live', 'Still alive', 'Possibly still alive - left in incubator'
    # Dead: 'Early dead', 'Dead embryo', 'Did not hatch', 'Late dead; cannot tell'
    alive_statuses = ['live', 'Live', 'Still alive', 'Possibly still alive - left in incubator']
    dead_statuses = ['Early dead', 'Dead embryo', 'Did not hatch', 'Late dead; cannot tell']
    
    # Create binary classification
    mortality_data = mortality_data[mortality_data['Mortality'].isin(alive_statuses + dead_statuses)]
    mortality_data['Mortality_Binary'] = mortality_data['Mortality'].apply(
        lambda x: 'Alive' if x in alive_statuses else 'Dead'
    )
    
    print(f"✓ Mortality data filtered: {mortality_data.shape}")
    mortality_counts = mortality_data['Mortality_Binary'].value_counts()
    print(f"  - Binary mortality distribution: {mortality_counts.to_dict()}")
    print(f"  - Original status distribution: {mortality_data['Mortality'].value_counts().to_dict()}")
    
    # Prepare features and target
    # Get wavelength columns (numeric column names)
    feature_columns = [col for col in mortality_data.columns 
                      if str(col).replace('.', '').replace('-', '').replace(' ', '').isdigit()]
    
    print(f"✓ Feature columns identified: {len(feature_columns)}")
    if len(feature_columns) == 0:
        print("Available columns:", mortality_data.columns.tolist())
        raise ValueError("No wavelength feature columns found")
    
    X = mortality_data[feature_columns].values
    y_labels = mortality_data['Mortality_Binary'].values  # Use binary classification
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    
    print(f"✓ Features prepared: {X.shape}")
    print(f"✓ Labels encoded: {len(np.unique(y))} classes")
    print(f"  - Class distribution: {dict(zip(label_encoder.classes_, np.bincount(y)))}")
    
    return X, y, wavelengths, label_encoder

def main():
    """Run complete M4 experiment"""
    print("="*80)
    print("M4 EXPERIMENT: RAW + SMOTE + TRANSFER LEARNING FOR MORTALITY")
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
        
        # Initialize preprocessor with minimal preprocessing
        print(f"\nInitializing M4 raw transfer preprocessor...")
        preprocessor = RawTransferPreprocessor(
            normalize_method='standard',
            pca_components=100,  # Use PCA for dimensionality reduction
            spectral_regions=True
        )
        
        # Preprocess training data
        X_train_processed = preprocessor.preprocess_mortality_data(X_train, wavelengths)
        
        # Preprocess test data
        print("\nPreprocessing test data...")
        X_test_processed = preprocessor.transform_new_data(X_test, wavelengths)
        print(f"✓ Test data processed: {X_test_processed.shape}")
        
        # Initialize and train transfer learning model
        print(f"\nInitializing M4 transfer learning model...")
        model = MortalityTransferModel(
            transfer_strategy='feature_extraction',
            random_state=42
        )
        
        # Train the transfer learning models
        training_results = model.train_mortality_transfer_models(X_train_processed, y_train)
        
        # Evaluate on test set
        evaluation_results = model.evaluate_mortality_prediction(
            X_test_processed, y_test, label_encoder
        )
        
        # Analyze transfer learning features
        feature_analysis = model.analyze_transfer_features(
            preprocessor.preprocessing_stats, top_k=15
        )
        
        # Save results
        model.save_model_and_results(
            evaluation_results, feature_analysis, preprocessor.preprocessing_stats
        )
        
        # Print final summary
        print("\n" + "="*80)
        print("M4 EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        
        best_model = evaluation_results['best_model']['name']
        best_accuracy = evaluation_results['best_model']['accuracy']
        best_auc = evaluation_results['best_model']['auc']
        
        print(f"Best Model: {best_model.replace('_', ' ').title() if best_model else 'None'}")
        print(f"Test Accuracy: {best_accuracy:.4f}")
        print(f"Test AUC: {best_auc:.4f}")
        print(f"Training Time: {training_results['training_time']:.2f} seconds")
        print(f"Enhanced Features: {X_train_processed.shape[1]} (from {X_train.shape[1]} raw wavelengths)")
        
        # Individual model performance
        print(f"\nIndividual Model Performance:")
        for model_name in training_results['models_trained']:
            if model_name in evaluation_results:
                acc = evaluation_results[model_name]['accuracy']
                auc = evaluation_results[model_name]['auc']
                print(f"  {model_name.replace('_', ' ').title()}: {acc:.4f} (AUC: {auc:.4f})")
        
        # Transfer learning insights
        if 'transfer_components' in training_results:
            print(f"\nTransfer Learning Summary:")
            print(f"  Transfer Components: {training_results['transfer_components']}")
            if 'explained_variance' in feature_analysis:
                print(f"  Explained Variance: {feature_analysis['explained_variance']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ M4 experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 M4 experiment completed successfully!")
    else:
        print("\n💥 M4 experiment failed!")
        sys.exit(1) 