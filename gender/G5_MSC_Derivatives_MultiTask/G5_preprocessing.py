import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class MultiScaleDerivativePreprocessor:
    """
    Advanced preprocessing for multi-task learning with MSC and multi-scale derivatives
    Combines successful G1 approach with enhanced derivative features
    """
    
    def __init__(self, window_length=21, polyorder=3):
        """
        Initialize preprocessor with Savitzky-Golay parameters
        
        Args:
            window_length: Window length for SG filter (must be odd)
            polyorder: Polynomial order for SG filter
        """
        self.window_length = window_length
        self.polyorder = polyorder
        self.scaler = StandardScaler()
        self.wavelengths = None
        
    def apply_msc(self, spectra):
        """
        Apply Multiplicative Scatter Correction (MSC)
        Same proven approach from G1 experiment
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            MSC corrected spectra
        """
        print("Applying MSC (Multiplicative Scatter Correction)...")
        
        # Calculate mean spectrum
        mean_spectrum = np.mean(spectra, axis=0)
        
        # Apply MSC correction
        msc_corrected = np.zeros_like(spectra)
        
        for i, spectrum in enumerate(spectra):
            # Fit linear regression: spectrum = a * mean_spectrum + b
            fit = np.polyfit(mean_spectrum, spectrum, deg=1)
            a, b = fit
            
            # Apply correction: corrected = (spectrum - b) / a
            msc_corrected[i] = (spectrum - b) / a
        
        print(f"✓ MSC correction applied to {spectra.shape[0]} spectra")
        return msc_corrected
    
    def compute_multi_scale_derivatives(self, spectra):
        """
        Compute multi-scale Savitzky-Golay derivatives
        Enhanced version with 1st, 2nd, and 3rd order derivatives
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Multi-scale derivative features
        """
        print("Computing multi-scale Savitzky-Golay derivatives...")
        
        n_samples, n_wavelengths = spectra.shape
        
        # Original spectra
        original = spectra
        
        # 1st derivative
        deriv_1st = np.array([
            savgol_filter(spectrum, self.window_length, self.polyorder, deriv=1)
            for spectrum in spectra
        ])
        
        # 2nd derivative  
        deriv_2nd = np.array([
            savgol_filter(spectrum, self.window_length, self.polyorder, deriv=2)
            for spectrum in spectra
        ])
        
        # 3rd derivative
        deriv_3rd = np.array([
            savgol_filter(spectrum, self.window_length, self.polyorder, deriv=3)
            for spectrum in spectra
        ])
        
        print(f"✓ Computed derivatives: Original + 1st + 2nd + 3rd order")
        print(f"  - Original spectra: {original.shape}")
        print(f"  - 1st derivative: {deriv_1st.shape}")
        print(f"  - 2nd derivative: {deriv_2nd.shape}")
        print(f"  - 3rd derivative: {deriv_3rd.shape}")
        
        return {
            'original': original,
            '1st_derivative': deriv_1st,
            '2nd_derivative': deriv_2nd,
            '3rd_derivative': deriv_3rd
        }
    
    def create_multi_scale_features(self, derivative_dict):
        """
        Create comprehensive feature set from multi-scale derivatives
        
        Args:
            derivative_dict: Dictionary with different derivative orders
            
        Returns:
            Combined feature matrix
        """
        print("Creating multi-scale feature matrix...")
        
        # Concatenate all derivative orders
        feature_list = []
        feature_names = []
        
        for deriv_name, deriv_data in derivative_dict.items():
            feature_list.append(deriv_data)
            n_features = deriv_data.shape[1]
            feature_names.extend([f"{deriv_name}_wl_{i}" for i in range(n_features)])
        
        # Combine all features
        combined_features = np.concatenate(feature_list, axis=1)
        
        print(f"✓ Multi-scale feature matrix: {combined_features.shape}")
        print(f"  - Total features: {combined_features.shape[1]}")
        print(f"  - Feature breakdown:")
        
        current_idx = 0
        for deriv_name, deriv_data in derivative_dict.items():
            n_features = deriv_data.shape[1]
            print(f"    • {deriv_name}: {n_features} features ({current_idx}:{current_idx + n_features})")
            current_idx += n_features
        
        return combined_features, feature_names
    
    def fit_transform(self, spectra, wavelengths):
        """
        Fit preprocessor and transform spectra with multi-scale derivatives
        
        Args:
            spectra: 2D array (samples x wavelengths)
            wavelengths: Wavelength values
            
        Returns:
            Processed multi-scale features
        """
        self.wavelengths = wavelengths
        
        print("Applying MSC + Multi-Scale Derivatives preprocessing...")
        
        # Step 1: MSC correction
        msc_corrected = self.apply_msc(spectra)
        
        # Step 2: Multi-scale derivatives
        derivatives = self.compute_multi_scale_derivatives(msc_corrected)
        
        # Step 3: Create feature matrix
        features, feature_names = self.create_multi_scale_features(derivatives)
        
        # Step 4: Standardization
        print("Applying standardization...")
        features_scaled = self.scaler.fit_transform(features)
        
        print(f"✓ Final processed features: {features_scaled.shape}")
        
        return features_scaled, feature_names
    
    def transform(self, spectra):
        """
        Transform new spectra using fitted parameters
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Processed multi-scale features
        """
        # Apply same preprocessing pipeline
        msc_corrected = self.apply_msc(spectra)
        derivatives = self.compute_multi_scale_derivatives(msc_corrected)
        features, _ = self.create_multi_scale_features(derivatives)
        features_scaled = self.scaler.transform(features)
        
        return features_scaled

def load_and_merge_data(day='D0'):
    """
    Load reference metadata and spectral data for specified day
    """
    print(f"Loading data for Day {day}...")
    
    # Load reference data
    ref_df = pd.read_csv('../../data/reference_metadata.csv')
    
    # Load spectral data for specified day
    spectral_df = pd.read_csv(f'../../data/spectral_data_{day}.csv')
    
    # Merge on HSI sample ID
    merged_df = pd.merge(ref_df, spectral_df, on='HSI sample ID', how='inner')
    
    print(f"✓ Merged dataset shape: {merged_df.shape}")
    print(f"✓ Samples with both reference and spectral data: {len(merged_df)}")
    
    return merged_df

def prepare_multi_task_dataset(merged_df):
    """
    Prepare dataset for multi-task learning (Gender + Mortality)
    """
    print("\n" + "="*50)
    print("PREPARING MULTI-TASK DATASET - G5 EXPERIMENT")
    print("="*50)
    
    # Filter for samples with both gender and mortality labels
    multi_task_df = merged_df[
        (merged_df['Gender'].isin(['Male', 'Female'])) &
        (merged_df['Mortality status'].isin(['Live', 'Dead']))
    ].copy()
    
    print(f"Samples with both gender and mortality labels: {len(multi_task_df)}")
    
    # Check label distributions
    print("\nTask 1 - Gender distribution:")
    gender_counts = multi_task_df['Gender'].value_counts()
    print(gender_counts.to_dict())
    
    print("\nTask 2 - Mortality distribution:")
    mortality_counts = multi_task_df['Mortality status'].value_counts()
    print(mortality_counts.to_dict())
    
    print("\nJoint distribution (Gender × Mortality):")
    joint_dist = pd.crosstab(multi_task_df['Gender'], multi_task_df['Mortality status'])
    print(joint_dist)
    
    # Extract wavelength features
    metadata_cols = ['HSI sample ID', 'Date_x', 'Date_y', 'Exp. No._x', 'Exp. No._y', 
                    'Gender', 'Fertility status', 'Mortality status', 'Mass (g)', 
                    'Major dia. (mm)', 'Minor dia. (mm)', 'Comment']
    
    # Get wavelength columns
    wavelength_cols = []
    for col in multi_task_df.columns:
        if col not in metadata_cols:
            try:
                float(col)
                wavelength_cols.append(col)
            except ValueError:
                continue
    
    print(f"\nFound {len(wavelength_cols)} wavelength features")
    print(f"Wavelength range: {wavelength_cols[0]} to {wavelength_cols[-1]}")
    
    # Extract features and labels
    X = multi_task_df[wavelength_cols].values
    y_gender = multi_task_df['Gender'].values
    y_mortality = multi_task_df['Mortality status'].values
    
    # Encode labels
    gender_encoder = LabelEncoder()
    mortality_encoder = LabelEncoder()
    
    y_gender_encoded = gender_encoder.fit_transform(y_gender)
    y_mortality_encoded = mortality_encoder.fit_transform(y_mortality)
    
    print(f"\nLabel encodings:")
    print(f"Gender: {dict(zip(gender_encoder.classes_, gender_encoder.transform(gender_encoder.classes_)))}")
    print(f"Mortality: {dict(zip(mortality_encoder.classes_, mortality_encoder.transform(mortality_encoder.classes_)))}")
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Gender labels shape: {y_gender_encoded.shape}")
    print(f"Mortality labels shape: {y_mortality_encoded.shape}")
    
    return (X, y_gender_encoded, y_mortality_encoded, 
            gender_encoder, mortality_encoder, wavelength_cols, multi_task_df)

def create_stratified_splits(X, y_gender, y_mortality, test_size=0.2, random_state=42):
    """
    Create stratified train/test splits for multi-task learning
    Ensures balanced representation of both tasks
    """
    print(f"\nCreating stratified multi-task splits...")
    
    # Create combined stratification labels (gender + mortality)
    # This ensures both tasks are balanced in train/test splits
    combined_labels = [f"{g}_{m}" for g, m in zip(y_gender, y_mortality)]
    
    # Check combined label distribution
    unique_combined, counts_combined = np.unique(combined_labels, return_counts=True)
    print(f"Combined label distribution:")
    for label, count in zip(unique_combined, counts_combined):
        print(f"  {label}: {count} samples")
    
    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, combined_labels))
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_gender_train = y_gender[train_idx]
    y_gender_test = y_gender[test_idx]
    y_mortality_train = y_mortality[train_idx]
    y_mortality_test = y_mortality[test_idx]
    
    print(f"\nStratified Multi-Task Split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    print(f"\nTraining set distributions:")
    print(f"Gender: {np.bincount(y_gender_train)}")
    print(f"Mortality: {np.bincount(y_mortality_train)}")
    
    print(f"\nTest set distributions:")
    print(f"Gender: {np.bincount(y_gender_test)}")
    print(f"Mortality: {np.bincount(y_mortality_test)}")
    
    return (X_train, X_test, y_gender_train, y_gender_test, 
            y_mortality_train, y_mortality_test)

def main():
    """
    Main preprocessing pipeline for G5 multi-task experiment
    """
    print("="*70)
    print("G5 EXPERIMENT: MSC + Multi-Scale Derivatives + Multi-Task Learning")
    print("="*70)
    
    # Load and merge data
    merged_df = load_and_merge_data(day='D0')
    
    # Prepare multi-task dataset
    (X, y_gender, y_mortality, gender_encoder, mortality_encoder, 
     wavelength_cols, multi_task_df) = prepare_multi_task_dataset(merged_df)
    
    # Create stratified splits
    (X_train_raw, X_test_raw, y_gender_train, y_gender_test, 
     y_mortality_train, y_mortality_test) = create_stratified_splits(
        X, y_gender, y_mortality
    )
    
    # Initialize and apply preprocessing
    print("\n" + "="*50)
    print("APPLYING MSC + MULTI-SCALE DERIVATIVES PREPROCESSING")
    print("="*50)
    
    preprocessor = MultiScaleDerivativePreprocessor(
        window_length=21,  # Same as successful G1
        polyorder=3        # Same as successful G1
    )
    
    # Fit on training data and transform
    X_train_processed, feature_names = preprocessor.fit_transform(X_train_raw, wavelength_cols)
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test_raw)
    
    print(f"✓ Processed training set shape: {X_train_processed.shape}")
    print(f"✓ Processed test set shape: {X_test_processed.shape}")
    
    # Save processed data
    print("\nSaving processed data...")
    np.save('X_train_processed.npy', X_train_processed)
    np.save('X_test_processed.npy', X_test_processed)
    np.save('y_gender_train.npy', y_gender_train)
    np.save('y_gender_test.npy', y_gender_test)
    np.save('y_mortality_train.npy', y_mortality_train)
    np.save('y_mortality_test.npy', y_mortality_test)
    
    # Save preprocessor and encoders
    import joblib
    joblib.dump(preprocessor, 'multi_scale_preprocessor.pkl')
    joblib.dump(gender_encoder, 'gender_encoder.pkl')
    joblib.dump(mortality_encoder, 'mortality_encoder.pkl')
    
    # Save feature information
    feature_info = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'wavelength_cols': wavelength_cols,
        'preprocessing': 'MSC + Multi-Scale SG Derivatives (1st, 2nd, 3rd order)'
    }
    
    import json
    with open('feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("✓ Saved processed data and preprocessor")
    
    # Summary statistics
    print("\n" + "="*50)
    print("MULTI-TASK PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Original features: {X.shape[1]} wavelengths")
    print(f"Multi-scale features: {X_train_processed.shape[1]} (4x derivatives)")
    print(f"Training samples: {X_train_processed.shape[0]}")
    print(f"Test samples: {X_test_processed.shape[0]}")
    print(f"Task 1 (Gender): {len(gender_encoder.classes_)} classes - {gender_encoder.classes_}")
    print(f"Task 2 (Mortality): {len(mortality_encoder.classes_)} classes - {mortality_encoder.classes_}")
    print("Preprocessing: MSC + Multi-scale SG derivatives")
    print("Ready for multi-task deep learning!")

if __name__ == "__main__":
    main() 