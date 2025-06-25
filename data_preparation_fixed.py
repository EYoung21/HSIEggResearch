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

def clean_labels(df):
    """
    Clean and standardize gender and mortality labels
    """
    print("\nCleaning labels...")
    
    # Gender cleaning
    gender_mapping = {
        'M': 'Male', 'F': 'Female', 'Male': 'Male', 'Female': 'Female'
    }
    df['Gender_clean'] = df['Gender'].map(gender_mapping)
    
    # Mortality cleaning  
    mortality_mapping = {
        'L': 'Live', 'D': 'Dead', 'Live': 'Live', 'Dead': 'Dead',
        'live': 'Live', 'Still alive': 'Live',
        'Possibly still alive - left in incubator': 'Live',
        'Dead embryo': 'Dead', 'Early dead': 'Dead',
        'Late dead; cannot tell': 'Dead'
    }
    df['Mortality_clean'] = df['Mortality status'].map(mortality_mapping)
    
    # Print cleaning results
    print(f"Gender - Before: {df['Gender'].value_counts().to_dict()}")
    print(f"Gender - After: {df['Gender_clean'].value_counts().to_dict()}")
    print(f"Mortality - Before: {df['Mortality status'].value_counts().to_dict()}")
    print(f"Mortality - After: {df['Mortality_clean'].value_counts().to_dict()}")
    
    return df

def extract_features(df):
    """
    Extract wavelength features from spectral data
    """
    print("\nExtracting spectral features...")
    
    # Find wavelength columns (they should be numeric)
    wavelength_cols = []
    for col in df.columns:
        if col not in ['HSI sample ID', 'Gender', 'Mortality', 'Fertility', 
                      'Fresh egg wt. (g)', 'Major axis (mm)', 'Minor axis (mm)',
                      'Gender_clean', 'Mortality_clean']:
            try:
                # Try to convert to float to see if it's a wavelength
                float(col)
                wavelength_cols.append(col)
            except:
                continue
    
    print(f"Found {len(wavelength_cols)} wavelength features")
    
    if len(wavelength_cols) == 0:
        print("No wavelength columns found - checking all numeric columns")
        wavelength_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove metadata columns
        exclude_cols = ['HSI sample ID', 'Fresh egg wt. (g)', 'Major axis (mm)', 'Minor axis (mm)']
        wavelength_cols = [col for col in wavelength_cols if col not in exclude_cols]
        print(f"Using {len(wavelength_cols)} numeric features")
    
    return df[wavelength_cols].values, wavelength_cols

def prepare_gender_dataset(df):
    """
    Prepare dataset for gender classification
    """
    print("\n" + "="*50)
    print("PREPARING GENDER CLASSIFICATION DATASET")
    print("="*50)
    
    # Filter for samples with gender labels
    gender_df = df.dropna(subset=['Gender_clean']).copy()
    
    print(f"Samples with gender labels: {len(gender_df)}")
    print(f"Gender distribution: {gender_df['Gender_clean'].value_counts().to_dict()}")
    
    # Extract features
    X, feature_names = extract_features(gender_df)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(gender_df['Gender_clean'])
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {dict(zip(le.classes_, np.bincount(y)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def prepare_mortality_dataset(df):
    """
    Prepare dataset for mortality classification
    """
    print("\n" + "="*50)
    print("PREPARING MORTALITY CLASSIFICATION DATASET")
    print("="*50)
    
    # Filter for samples with mortality labels
    mortality_df = df.dropna(subset=['Mortality_clean']).copy()
    
    print(f"Samples with mortality labels: {len(mortality_df)}")
    print(f"Mortality distribution: {mortality_df['Mortality_clean'].value_counts().to_dict()}")
    
    # Extract features
    X, feature_names = extract_features(mortality_df)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(mortality_df['Mortality_clean'])
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {dict(zip(le.classes_, np.bincount(y)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def save_processed_data(X_train, X_test, y_train, y_test, scaler, le, dataset_type):
    """
    Save processed data for modeling
    """
    import os
    
    # Create processed data directory
    os.makedirs('data/processed', exist_ok=True)
    
    # Save arrays
    np.save(f'data/processed/{dataset_type}_X_train.npy', X_train)
    np.save(f'data/processed/{dataset_type}_X_test.npy', X_test)
    np.save(f'data/processed/{dataset_type}_y_train.npy', y_train)
    np.save(f'data/processed/{dataset_type}_y_test.npy', y_test)
    
    # Save preprocessors
    import joblib
    joblib.dump(scaler, f'data/processed/{dataset_type}_scaler.pkl')
    joblib.dump(le, f'data/processed/{dataset_type}_label_encoder.pkl')
    
    print(f"✓ Saved {dataset_type} dataset to data/processed/")

def main():
    """
    Main data preparation pipeline
    """
    print("HSI Egg Classification - Data Preparation")
    print("="*50)
    
    # Load and merge data
    df = load_and_merge_data(day='D0')
    
    # Clean labels
    df = clean_labels(df)
    
    # Prepare gender dataset
    try:
        X_train, X_test, y_train, y_test, scaler, le = prepare_gender_dataset(df)
        save_processed_data(X_train, X_test, y_train, y_test, scaler, le, 'gender')
    except Exception as e:
        print(f"Error preparing gender dataset: {e}")
    
    # Prepare mortality dataset
    try:
        X_train, X_test, y_train, y_test, scaler, le = prepare_mortality_dataset(df)
        save_processed_data(X_train, X_test, y_train, y_test, scaler, le, 'mortality')
    except Exception as e:
        print(f"Error preparing mortality dataset: {e}")
    
    print("\nData preparation complete!")

if __name__ == "__main__":
    main() 