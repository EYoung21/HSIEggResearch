import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_data(day='D0'):
    """
    Load reference metadata and spectral data for specified day
    """
    print(f"Loading data for Day {day}...")
    
    # Load reference data
    ref_df = pd.read_csv('data/reference_metadata.csv')
    
    # Load spectral data for specified day
    spectral_df = pd.read_csv(f'data/spectral_data_{day}.csv')
    
    # Merge on HSI sample ID
    merged_df = pd.merge(ref_df, spectral_df, on='HSI sample ID', how='inner')
    
    print(f"✓ Merged dataset shape: {merged_df.shape}")
    print(f"✓ Samples with both reference and spectral data: {len(merged_df)}")
    
    return merged_df

def prepare_gender_dataset(merged_df):
    """
    Prepare clean dataset for gender classification
    Focus on clearly labeled Male/Female eggs only
    """
    print("\n" + "="*50)
    print("PREPARING GENDER DATASET")
    print("="*50)
    
    # Filter for clear gender labels only
    gender_df = merged_df[merged_df['Gender'].isin(['Male', 'Female'])].copy()
    
    print(f"Clear gender labels: {len(gender_df)} samples")
    print("Gender distribution:", gender_df['Gender'].value_counts().to_dict())
    
    # Extract features (wavelengths) and target
    wavelength_cols = [col for col in gender_df.columns if isinstance(col, (int, float))]
    
    X_gender = gender_df[wavelength_cols].values
    y_gender = gender_df['Gender'].values
    
    # Encode labels
    le_gender = LabelEncoder()
    y_gender_encoded = le_gender.fit_transform(y_gender)
    
    print(f"Feature matrix shape: {X_gender.shape}")
    print(f"Label encoding: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")
    
    return X_gender, y_gender_encoded, le_gender, wavelength_cols, gender_df

def prepare_mortality_dataset(merged_df):
    """
    Prepare clean dataset for mortality classification
    Combine live/alive vs dead/embryo categories
    """
    print("\n" + "="*50)
    print("PREPARING MORTALITY DATASET")
    print("="*50)
    
    # Clean and standardize mortality labels
    mortality_mapping = {
        'live': 'Live',
        'Live': 'Live', 
        'Still alive': 'Live',
        'Possibly still alive - left in incubator': 'Live',
        'Dead embryo': 'Dead',
        'Early dead': 'Dead',
        'Late dead; cannot tell': 'Dead'
    }
    
    # Apply mapping and filter for clear mortality status
    mortality_df = merged_df.copy()
    mortality_df['Mortality_Clean'] = mortality_df['Mortality status'].map(mortality_mapping)
    mortality_df = mortality_df.dropna(subset=['Mortality_Clean'])
    
    # Remove infertile eggs for mortality analysis (focus on fertile eggs only)
    mortality_df = mortality_df[~mortality_df['Fertility status'].str.contains('Infertile', na=False, case=False)]
    
    print(f"Clean mortality labels: {len(mortality_df)} samples")
    print("Mortality distribution:", mortality_df['Mortality_Clean'].value_counts().to_dict())
    
    # Extract features and target
    wavelength_cols = [col for col in mortality_df.columns if isinstance(col, (int, float))]
    
    X_mortality = mortality_df[wavelength_cols].values
    y_mortality = mortality_df['Mortality_Clean'].values
    
    # Encode labels
    le_mortality = LabelEncoder()
    y_mortality_encoded = le_mortality.fit_transform(y_mortality)
    
    print(f"Feature matrix shape: {X_mortality.shape}")
    print(f"Label encoding: {dict(zip(le_mortality.classes_, le_mortality.transform(le_mortality.classes_)))}")
    
    return X_mortality, y_mortality_encoded, le_mortality, wavelength_cols, mortality_df

def create_train_test_splits(X, y, test_size=0.2, random_state=42):
    """
    Create train/test splits with stratification
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def save_processed_data(X_train, X_test, y_train, y_test, wavelength_cols, label_encoder, prefix):
    """
    Save processed data for modeling
    """
    # Create processed data directory
    import os
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    
    # Save feature data
    np.save(f'data/processed/{prefix}_X_train.npy', X_train)
    np.save(f'data/processed/{prefix}_X_test.npy', X_test)
    np.save(f'data/processed/{prefix}_y_train.npy', y_train)
    np.save(f'data/processed/{prefix}_y_test.npy', y_test)
    
    # Save wavelength information
    wavelength_df = pd.DataFrame({'wavelength': wavelength_cols})
    wavelength_df.to_csv(f'data/processed/{prefix}_wavelengths.csv', index=False)
    
    # Save label encoder
    import joblib
    joblib.dump(label_encoder, f'data/processed/{prefix}_label_encoder.pkl')
    
    print(f"✓ Saved processed {prefix} data to data/processed/")

def main():
    """
    Main data preparation pipeline
    """
    print("="*60)
    print("HSI EGG CLASSIFICATION - DATA PREPARATION")
    print("="*60)
    
    # Load and merge data (using Day 0 for pre-incubation prediction)
    merged_df = load_and_merge_data(day='D0')
    
    # Prepare gender dataset
    X_gender, y_gender, le_gender, wavelength_cols, gender_df = prepare_gender_dataset(merged_df)
    
    # Create train/test splits for gender
    print("\nCreating train/test splits for gender classification:")
    X_train_gender, X_test_gender, y_train_gender, y_test_gender = create_train_test_splits(
        X_gender, y_gender
    )
    
    # Save gender data
    save_processed_data(X_train_gender, X_test_gender, y_train_gender, y_test_gender, 
                       wavelength_cols, le_gender, 'gender')
    
    # Prepare mortality dataset
    X_mortality, y_mortality, le_mortality, wavelength_cols, mortality_df = prepare_mortality_dataset(merged_df)
    
    # Create train/test splits for mortality
    print("\nCreating train/test splits for mortality classification:")
    X_train_mortality, X_test_mortality, y_train_mortality, y_test_mortality = create_train_test_splits(
        X_mortality, y_mortality
    )
    
    # Save mortality data
    save_processed_data(X_train_mortality, X_test_mortality, y_train_mortality, y_test_mortality, 
                       wavelength_cols, le_mortality, 'mortality')
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Start with gender classification model")
    print("2. Try different preprocessing (SNV, MSC, SG filtering)")
    print("3. Test various ML algorithms (PLS-DA, XGBoost, CatBoost, RF)")
    print("4. Use SHAP for feature importance and explainability")
    print("5. Consider SMOTE for handling class imbalance if needed")

if __name__ == "__main__":
    main() 