import pandas as pd
import numpy as np
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class MSC_SG_Preprocessor:
    """
    MSC + Savitzky-Golay 1st Derivative Preprocessor for HSI data
    
    MSC (Multiplicative Scatter Correction): Corrects for scattering effects
    SG 1st Derivative: Removes baseline drift and enhances spectral features
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
        self.msc_mean = None
        self.wavelengths = None
        
    def apply_msc(self, spectra, reference=None):
        """
        Apply Multiplicative Scatter Correction with numerical safeguards
        
        Args:
            spectra: 2D array (samples x wavelengths)
            reference: Reference spectrum (if None, uses mean)
            
        Returns:
            MSC corrected spectra
        """
        # Handle NaN values
        spectra = np.nan_to_num(spectra, nan=0.0, posinf=0.0, neginf=0.0)
        
        if reference is None:
            reference = np.mean(spectra, axis=0)
            self.msc_mean = reference
        else:
            reference = self.msc_mean
            
        # Ensure reference is not all zeros or constant
        if np.std(reference) < 1e-10:
            print("Warning: Reference spectrum is nearly constant. Skipping MSC correction.")
            return spectra.copy()
            
        corrected_spectra = np.zeros_like(spectra)
        
        for i in range(spectra.shape[0]):
            try:
                # Check for problematic spectra
                if np.std(spectra[i]) < 1e-10:
                    # Spectrum is nearly constant, skip correction
                    corrected_spectra[i] = spectra[i]
                    continue
                    
                # Linear regression: spectrum = a + b * reference
                coeffs = np.polyfit(reference, spectra[i], 1)
                
                # Avoid division by very small slopes
                if abs(coeffs[0]) < 1e-10:
                    corrected_spectra[i] = spectra[i]
                    continue
                    
                # Correct: (spectrum - a) / b
                corrected_spectra[i] = (spectra[i] - coeffs[1]) / coeffs[0]
                
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Warning: MSC failed for spectrum {i}, using original spectrum. Error: {e}")
                corrected_spectra[i] = spectra[i]
            
        return corrected_spectra
    
    def apply_sg_derivative(self, spectra):
        """
        Apply Savitzky-Golay 1st derivative
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            1st derivative spectra
        """
        derivative_spectra = np.zeros_like(spectra)
        
        for i in range(spectra.shape[0]):
            derivative_spectra[i] = signal.savgol_filter(
                spectra[i], 
                window_length=self.sg_window,
                polyorder=self.sg_polyorder,
                deriv=1  # 1st derivative
            )
            
        return derivative_spectra
    
    def fit_transform(self, spectra):
        """
        Fit preprocessor and transform spectra
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Preprocessed spectra
        """
        print("Applying MSC correction...")
        msc_spectra = self.apply_msc(spectra)
        
        print("Applying Savitzky-Golay 1st derivative...")
        processed_spectra = self.apply_sg_derivative(msc_spectra)
        
        return processed_spectra
    
    def transform(self, spectra):
        """
        Transform new spectra using fitted parameters
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Preprocessed spectra
        """
        if self.msc_mean is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
            
        msc_spectra = self.apply_msc(spectra, reference=self.msc_mean)
        processed_spectra = self.apply_sg_derivative(msc_spectra)
        
        return processed_spectra

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
    print("PREPARING GENDER DATASET - G1 EXPERIMENT")
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
    Main preprocessing pipeline for G1 experiment
    """
    print("="*60)
    print("G1 EXPERIMENT: MSC + SG 1st Derivative Preprocessing")
    print("="*60)
    
    # Load and merge data (Day 0 for pre-incubation prediction)
    merged_df = load_and_merge_data(day='D0')
    
    # Prepare gender dataset
    X_raw, y, le, wavelength_cols, gender_df = prepare_gender_dataset(merged_df)
    
    # Create train/test splits
    X_train_raw, X_test_raw, y_train, y_test = create_train_test_splits(X_raw, y)
    
    # Initialize and apply MSC + SG preprocessing
    print("\n" + "="*50)
    print("APPLYING MSC + SAVITZKY-GOLAY 1ST DERIVATIVE")
    print("="*50)
    
    preprocessor = MSC_SG_Preprocessor(sg_window=15, sg_polyorder=3)
    
    # Fit on training data and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train_raw)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    print(f"✓ Processed training set shape: {X_train_processed.shape}")
    print(f"✓ Processed test set shape: {X_test_processed.shape}")
    
    # Save processed data
    print("\nSaving processed data...")
    np.save('X_train_processed.npy', X_train_processed)
    np.save('X_test_processed.npy', X_test_processed)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save wavelength info and preprocessor
    wavelength_df = pd.DataFrame({'wavelength': wavelength_cols})
    wavelength_df.to_csv('wavelengths.csv', index=False)
    
    import joblib
    joblib.dump(preprocessor, 'msc_sg_preprocessor.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    
    print("✓ Saved processed data and preprocessor")
    
    # Summary statistics
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Original features: {X_raw.shape[1]} wavelengths")
    print(f"Processed features: {X_train_processed.shape[1]} wavelengths")
    print(f"Training samples: {X_train_processed.shape[0]}")
    print(f"Test samples: {X_test_processed.shape[0]}")
    print(f"Classes: {le.classes_}")
    print("Preprocessing: MSC + Savitzky-Golay 1st derivative")
    print("Ready for LightGBM with Bayesian optimization!")

if __name__ == "__main__":
    main() 