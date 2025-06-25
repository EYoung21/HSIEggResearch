import pandas as pd
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SNV_SG_Preprocessor:
    """
    SNV + Savitzky-Golay 2nd Derivative Preprocessor for HSI data
    
    SNV (Standard Normal Variate): Normalizes each spectrum individually
    SG 2nd Derivative: Enhances spectral features and removes baseline/linear trends
    Spectral Ratios: Creates ratio features for enhanced discrimination
    """
    
    def __init__(self, sg_window=15, sg_polyorder=3):
        """
        Initialize preprocessor
        
        Args:
            sg_window: Window length for Savitzky-Golay filter (odd number)
            sg_polyorder: Polynomial order for SG filter (< sg_window)
        """
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        self.wavelengths = None
        self.spectral_ratio_indices = None
        
    def apply_snv(self, spectra):
        """
        Apply Standard Normal Variate normalization
        
        SNV normalizes each spectrum to have mean=0 and std=1
        Formula: (spectrum - mean) / std
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            SNV normalized spectra
        """
        snv_spectra = np.zeros_like(spectra)
        
        for i in range(spectra.shape[0]):
            spectrum = spectra[i]
            
            # Handle problematic spectra
            if np.std(spectrum) < 1e-10:
                print(f"Warning: Spectrum {i} has very low variance, using original")
                snv_spectra[i] = spectrum
                continue
                
            # Apply SNV: (x - mean) / std
            mean_spectrum = np.mean(spectrum)
            std_spectrum = np.std(spectrum)
            snv_spectra[i] = (spectrum - mean_spectrum) / std_spectrum
            
        return snv_spectra
    
    def apply_sg_second_derivative(self, spectra):
        """
        Apply Savitzky-Golay 2nd derivative
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            2nd derivative spectra
        """
        derivative_spectra = np.zeros_like(spectra)
        
        for i in range(spectra.shape[0]):
            try:
                derivative_spectra[i] = signal.savgol_filter(
                    spectra[i], 
                    window_length=self.sg_window,
                    polyorder=self.sg_polyorder,
                    deriv=2  # 2nd derivative
                )
            except ValueError as e:
                print(f"Warning: SG filter failed for spectrum {i}, using zeros. Error: {e}")
                derivative_spectra[i] = np.zeros_like(spectra[i])
                
        return derivative_spectra
    
    def create_spectral_ratios(self, spectra, wavelengths):
        """
        Create spectral ratio features for enhanced discrimination
        
        Ratios are particularly useful for:
        - Reducing instrument variability
        - Highlighting relative differences between spectral regions
        - Biological interpretation (protein/lipid ratios, etc.)
        
        Args:
            spectra: 2D array (samples x wavelengths)
            wavelengths: List of wavelength values
            
        Returns:
            Spectral ratio features
        """
        n_samples = spectra.shape[0]
        
        # Define meaningful spectral regions for egg analysis
        regions = {
            'blue': (400, 500),      # Carotenoids, pigments
            'green': (500, 600),     # Chlorophyll, other pigments  
            'red': (600, 700),       # Protein absorption
            'nir1': (700, 850),      # Water, organic compounds
            'nir2': (850, 1000),     # Lipids, proteins
        }
        
        # Find wavelength indices for each region
        region_indices = {}
        for region_name, (min_wl, max_wl) in regions.items():
            indices = [i for i, wl in enumerate(wavelengths) 
                      if float(wl) >= min_wl and float(wl) <= max_wl]
            if indices:
                region_indices[region_name] = indices
        
        # Calculate mean intensity for each region
        region_means = {}
        for region_name, indices in region_indices.items():
            if indices:
                region_means[region_name] = np.mean(spectra[:, indices], axis=1)
        
        # Create ratio features
        ratio_features = []
        ratio_names = []
        
        # Region-to-region ratios
        region_pairs = [
            ('red', 'blue'),      # Protein to carotenoid ratio
            ('nir1', 'red'),      # NIR to protein ratio  
            ('nir2', 'nir1'),     # High NIR to low NIR ratio
            ('green', 'blue'),    # Green to blue pigment ratio
            ('nir2', 'red'),      # Lipid to protein ratio
            ('red', 'green'),     # Red to green ratio
        ]
        
        for region1, region2 in region_pairs:
            if region1 in region_means and region2 in region_means:
                # Avoid division by zero
                denominator = region_means[region2]
                ratio = np.where(np.abs(denominator) > 1e-10, 
                               region_means[region1] / denominator, 
                               0)
                ratio_features.append(ratio)
                ratio_names.append(f'{region1}_to_{region2}_ratio')
        
        # Individual wavelength ratios (select key wavelengths)
        key_wavelengths = [450, 550, 650, 750, 850, 950]  # Representative wavelengths
        key_indices = []
        
        for key_wl in key_wavelengths:
            # Find closest wavelength
            closest_idx = min(range(len(wavelengths)), 
                             key=lambda i: abs(float(wavelengths[i]) - key_wl))
            key_indices.append(closest_idx)
        
        # Create key wavelength ratios
        for i, idx1 in enumerate(key_indices):
            for j, idx2 in enumerate(key_indices):
                if i < j:  # Avoid duplicate ratios
                    wl1, wl2 = wavelengths[idx1], wavelengths[idx2]
                    denominator = spectra[:, idx2]
                    ratio = np.where(np.abs(denominator) > 1e-10,
                                   spectra[:, idx1] / denominator,
                                   0)
                    ratio_features.append(ratio)
                    ratio_names.append(f'wl_{wl1}_to_{wl2}_ratio')
        
        # Combine all ratio features
        if ratio_features:
            ratio_matrix = np.column_stack(ratio_features)
            self.spectral_ratio_indices = ratio_names
            print(f"Created {len(ratio_names)} spectral ratio features")
            return ratio_matrix
        else:
            print("Warning: No spectral ratios could be created")
            return spectra  # Fallback to original spectra
    
    def fit_transform(self, spectra, wavelengths):
        """
        Fit preprocessor and transform spectra
        
        Args:
            spectra: 2D array (samples x wavelengths)
            wavelengths: List of wavelength values
            
        Returns:
            Preprocessed spectra with ratio features
        """
        self.wavelengths = wavelengths
        
        print("Applying SNV normalization...")
        snv_spectra = self.apply_snv(spectra)
        
        print("Applying Savitzky-Golay 2nd derivative...")
        sg_spectra = self.apply_sg_second_derivative(snv_spectra)
        
        print("Creating spectral ratio features...")
        ratio_features = self.create_spectral_ratios(sg_spectra, wavelengths)
        
        return ratio_features
    
    def transform(self, spectra):
        """
        Transform new spectra using fitted parameters
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Preprocessed spectra with ratio features
        """
        if self.wavelengths is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
            
        snv_spectra = self.apply_snv(spectra)
        sg_spectra = self.apply_sg_second_derivative(snv_spectra)
        ratio_features = self.create_spectral_ratios(sg_spectra, self.wavelengths)
        
        return ratio_features

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

def prepare_gender_dataset(merged_df):
    """
    Prepare clean dataset for gender classification
    Focus on clearly labeled Male/Female eggs only
    """
    print("\n" + "="*50)
    print("PREPARING GENDER DATASET - G2 EXPERIMENT")
    print("="*50)
    
    # Filter for clear gender labels only
    gender_df = merged_df[merged_df['Gender'].isin(['Male', 'Female'])].copy()
    
    print(f"Clear gender labels: {len(gender_df)} samples")
    print("Gender distribution:", gender_df['Gender'].value_counts().to_dict())
    
    # Extract wavelength features (numeric columns that represent wavelengths)
    metadata_cols = ['HSI sample ID', 'Date_x', 'Date_y', 'Exp. No._x', 'Exp. No._y', 'Gender', 'Fertility status', 
                    'Mortality status', 'Mass (g)', 'Major dia. (mm)', 'Minor dia. (mm)', 'Comment']
    
    # Get wavelength columns (should be numeric and represent actual wavelengths)
    wavelength_cols = []
    for col in gender_df.columns:
        if col not in metadata_cols:
            try:
                # Check if column name can be converted to float (wavelength value)
                float(col)
                wavelength_cols.append(col)
            except ValueError:
                # Skip non-numeric column names
                continue
    
    print(f"Found {len(wavelength_cols)} wavelength features")
    print(f"Wavelength range: {wavelength_cols[0]} to {wavelength_cols[-1]}")
    
    # Extract features and labels
    X_raw = gender_df[wavelength_cols].values
    y = gender_df['Gender'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Raw feature matrix shape: {X_raw.shape}")
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return X_raw, y_encoded, le, wavelength_cols, gender_df

def create_train_test_splits(X, y, test_size=0.2, random_state=42):
    """
    Create train/test splits with stratification
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain/Test Split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def main():
    """
    Main preprocessing pipeline for G2 experiment
    """
    print("="*60)
    print("G2 EXPERIMENT: SNV + SG 2nd Derivative + Spectral Ratios")
    print("="*60)
    
    # Load and merge data (Day 0 for pre-incubation prediction)
    merged_df = load_and_merge_data(day='D0')
    
    # Prepare gender dataset
    X_raw, y, le, wavelength_cols, gender_df = prepare_gender_dataset(merged_df)
    
    # Create train/test splits
    X_train_raw, X_test_raw, y_train, y_test = create_train_test_splits(X_raw, y)
    
    # Initialize and apply SNV + SG preprocessing
    print("\n" + "="*50)
    print("APPLYING SNV + SAVITZKY-GOLAY 2ND DERIVATIVE + RATIOS")
    print("="*50)
    
    preprocessor = SNV_SG_Preprocessor(sg_window=15, sg_polyorder=3)
    
    # Fit on training data and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train_raw, wavelength_cols)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    print(f"✓ Processed training set shape: {X_train_processed.shape}")
    print(f"✓ Processed test set shape: {X_test_processed.shape}")
    
    # Save processed data
    print("\nSaving processed data...")
    np.save('X_train_processed.npy', X_train_processed)
    np.save('X_test_processed.npy', X_test_processed)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save feature names and preprocessor
    if preprocessor.spectral_ratio_indices:
        ratio_df = pd.DataFrame({'feature_name': preprocessor.spectral_ratio_indices})
        ratio_df.to_csv('spectral_ratio_features.csv', index=False)
    
    import joblib
    joblib.dump(preprocessor, 'snv_sg_preprocessor.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    
    print("✓ Saved processed data and preprocessor")
    
    # Summary statistics
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Original features: {X_raw.shape[1]} wavelengths")
    print(f"Processed features: {X_train_processed.shape[1]} ratio features")
    print(f"Training samples: {X_train_processed.shape[0]}")
    print(f"Test samples: {X_test_processed.shape[0]}")
    print(f"Classes: {le.classes_}")
    print("Preprocessing: SNV + Savitzky-Golay 2nd derivative + Spectral ratios")
    print("Ready for Ensemble modeling (RF + SVM + XGB)!")

if __name__ == "__main__":
    main() 