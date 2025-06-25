import pandas as pd
import numpy as np
import pywt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import signal
from scipy.optimize import lsq_linear
import warnings
warnings.filterwarnings('ignore')

class EMSC_Wavelet_Preprocessor:
    """
    EMSC + Wavelet Transform Preprocessor for HSI data with CNN preparation
    
    EMSC (Extended Multiplicative Scatter Correction): Advanced scatter correction
    Wavelet Transform: Time-frequency decomposition for CNN feature extraction
    """
    
    def __init__(self, wavelet='db4', decomp_levels=4, emsc_degree=2):
        """
        Initialize preprocessor
        
        Args:
            wavelet: Wavelet type for decomposition ('db4', 'haar', 'coif2', etc.)
            decomp_levels: Number of decomposition levels
            emsc_degree: Polynomial degree for EMSC correction
        """
        self.wavelet = wavelet
        self.decomp_levels = decomp_levels
        self.emsc_degree = emsc_degree
        self.emsc_reference = None
        self.wavelengths = None
        self.scaler = StandardScaler()
        self.wavelet_length = None
        
    def compute_emsc_reference(self, spectra):
        """
        Compute reference spectrum for EMSC correction
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Reference spectrum (mean spectrum)
        """
        # Use mean spectrum as reference
        self.emsc_reference = np.mean(spectra, axis=0)
        return self.emsc_reference
    
    def apply_emsc(self, spectra):
        """
        Apply Extended Multiplicative Scatter Correction
        
        EMSC corrects for:
        - Multiplicative scatter effects
        - Additive baseline effects
        - Polynomial baseline trends
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            EMSC corrected spectra
        """
        if self.emsc_reference is None:
            raise ValueError("EMSC reference not computed. Call compute_emsc_reference first.")
        
        n_samples, n_wavelengths = spectra.shape
        corrected_spectra = np.zeros_like(spectra)
        
        # Create polynomial basis for baseline correction
        x = np.arange(n_wavelengths)
        x_norm = (x - x.mean()) / x.std()  # Normalize for numerical stability
        
        # Create design matrix: [reference, constant, x, x^2, ...]
        design_matrix = np.column_stack([
            self.emsc_reference,
            np.ones(n_wavelengths)
        ])
        
        # Add polynomial terms
        for degree in range(1, self.emsc_degree + 1):
            design_matrix = np.column_stack([design_matrix, x_norm**degree])
        
        print(f"Applying EMSC with polynomial degree {self.emsc_degree}...")
        
        for i in range(n_samples):
            spectrum = spectra[i]
            
            try:
                # Solve: spectrum = a*reference + b + c*x + d*x^2 + ... + residual
                # Using least squares with bounds to ensure physical constraints
                result = lsq_linear(design_matrix, spectrum, bounds=([-np.inf, -np.inf] + [-np.inf]*self.emsc_degree, [np.inf, np.inf] + [np.inf]*self.emsc_degree))
                
                if result.success:
                    coefficients = result.x
                    
                    # Extract multiplicative factor (should be ~1.0)
                    multiplicative_factor = coefficients[0]
                    
                    # Extract additive baseline
                    baseline = np.dot(design_matrix[:, 1:], coefficients[1:])
                    
                    # Apply EMSC correction: (spectrum - baseline) / multiplicative_factor
                    if abs(multiplicative_factor) > 1e-6:
                        corrected_spectra[i] = (spectrum - baseline) / multiplicative_factor
                    else:
                        print(f"Warning: Small multiplicative factor for spectrum {i}, using original")
                        corrected_spectra[i] = spectrum
                else:
                    print(f"Warning: EMSC failed for spectrum {i}, using original")
                    corrected_spectra[i] = spectrum
                    
            except Exception as e:
                print(f"Warning: EMSC error for spectrum {i}: {e}, using original")
                corrected_spectra[i] = spectrum
        
        return corrected_spectra
    
    def apply_wavelet_transform(self, spectra):
        """
        Apply Discrete Wavelet Transform for time-frequency decomposition
        
        Wavelets are excellent for:
        - Multi-resolution analysis
        - Time-frequency localization
        - Feature extraction for CNNs
        - Noise reduction
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Wavelet coefficients organized for CNN input
        """
        n_samples = spectra.shape[0]
        
        print(f"Applying {self.wavelet} wavelet transform with {self.decomp_levels} levels...")
        
        # Apply wavelet transform to first spectrum to determine dimensions
        coeffs = pywt.wavedec(spectra[0], self.wavelet, level=self.decomp_levels, mode='symmetric')
        
        # Calculate total length of concatenated coefficients
        total_length = sum(len(c) for c in coeffs)
        self.wavelet_length = total_length
        
        # Initialize output array
        wavelet_features = np.zeros((n_samples, total_length))
        
        for i in range(n_samples):
            try:
                # Decompose signal into wavelets
                coeffs = pywt.wavedec(spectra[i], self.wavelet, level=self.decomp_levels, mode='symmetric')
                
                # Concatenate all coefficients (approximation + details)
                concatenated = np.concatenate(coeffs)
                
                # Handle potential length differences due to signal length
                if len(concatenated) <= total_length:
                    wavelet_features[i, :len(concatenated)] = concatenated
                else:
                    wavelet_features[i] = concatenated[:total_length]
                    
            except Exception as e:
                print(f"Warning: Wavelet transform failed for spectrum {i}: {e}")
                # Use zeros if transform fails
                wavelet_features[i] = np.zeros(total_length)
        
        print(f"✓ Wavelet features shape: {wavelet_features.shape}")
        return wavelet_features
    
    def reshape_for_cnn(self, wavelet_features):
        """
        Reshape wavelet features for CNN input
        
        CNNs expect input with specific dimensions for convolution operations
        
        Args:
            wavelet_features: 2D array (samples x wavelet_coefficients)
            
        Returns:
            Reshaped features for CNN (samples x height x width x channels)
        """
        n_samples, n_features = wavelet_features.shape
        
        # Try to create a roughly square 2D representation
        # This allows CNN to learn spatial patterns in the wavelet domain
        
        # Find factors of n_features to create 2D grid
        factors = []
        for i in range(1, int(np.sqrt(n_features)) + 1):
            if n_features % i == 0:
                factors.append((i, n_features // i))
        
        if factors:
            # Choose factor pair closest to square
            height, width = min(factors, key=lambda x: abs(x[0] - x[1]))
        else:
            # If no perfect factors, pad to nearest square
            side_length = int(np.ceil(np.sqrt(n_features)))
            height, width = side_length, side_length
            
            # Pad features to fit square
            padded_features = np.zeros((n_samples, height * width))
            padded_features[:, :n_features] = wavelet_features
            wavelet_features = padded_features
        
        # Reshape to 2D grid format for CNN
        # Shape: (samples, height, width, 1) - single channel
        cnn_features = wavelet_features.reshape(n_samples, height, width, 1)
        
        print(f"✓ CNN input shape: {cnn_features.shape}")
        print(f"  - Height: {height}, Width: {width}, Channels: 1")
        
        return cnn_features, (height, width)
    
    def fit_transform(self, spectra, wavelengths):
        """
        Fit preprocessor and transform spectra
        
        Args:
            spectra: 2D array (samples x wavelengths)
            wavelengths: List of wavelength values
            
        Returns:
            CNN-ready wavelet features
        """
        self.wavelengths = wavelengths
        
        print("Computing EMSC reference spectrum...")
        self.compute_emsc_reference(spectra)
        
        print("Applying EMSC correction...")
        emsc_spectra = self.apply_emsc(spectra)
        
        print("Applying wavelet transformation...")
        wavelet_features = self.apply_wavelet_transform(emsc_spectra)
        
        print("Normalizing features...")
        normalized_features = self.scaler.fit_transform(wavelet_features)
        
        print("Reshaping for CNN...")
        cnn_features, input_shape = self.reshape_for_cnn(normalized_features)
        
        return cnn_features, input_shape
    
    def transform(self, spectra):
        """
        Transform new spectra using fitted parameters
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            CNN-ready wavelet features
        """
        if self.emsc_reference is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        emsc_spectra = self.apply_emsc(spectra)
        wavelet_features = self.apply_wavelet_transform(emsc_spectra)
        normalized_features = self.scaler.transform(wavelet_features)
        cnn_features, _ = self.reshape_for_cnn(normalized_features)
        
        return cnn_features

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
    print("PREPARING GENDER DATASET - G3 EXPERIMENT")
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
    Main preprocessing pipeline for G3 experiment
    """
    print("="*60)
    print("G3 EXPERIMENT: EMSC + Wavelets + CNN Preprocessing")
    print("="*60)
    
    # Load and merge data (Day 0 for pre-incubation prediction)
    merged_df = load_and_merge_data(day='D0')
    
    # Prepare gender dataset
    X_raw, y, le, wavelength_cols, gender_df = prepare_gender_dataset(merged_df)
    
    # Create train/test splits
    X_train_raw, X_test_raw, y_train, y_test = create_train_test_splits(X_raw, y)
    
    # Initialize and apply EMSC + Wavelet preprocessing
    print("\n" + "="*50)
    print("APPLYING EMSC + WAVELET PREPROCESSING")
    print("="*50)
    
    preprocessor = EMSC_Wavelet_Preprocessor(
        wavelet='db4',        # Daubechies 4 wavelet
        decomp_levels=4,      # 4 levels of decomposition
        emsc_degree=2         # Quadratic baseline correction
    )
    
    # Fit on training data and transform both sets
    X_train_processed, input_shape = preprocessor.fit_transform(X_train_raw, wavelength_cols)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    print(f"✓ Processed training set shape: {X_train_processed.shape}")
    print(f"✓ Processed test set shape: {X_test_processed.shape}")
    print(f"✓ CNN input dimensions: {input_shape}")
    
    # Save processed data
    print("\nSaving processed data...")
    np.save('X_train_processed.npy', X_train_processed)
    np.save('X_test_processed.npy', X_test_processed)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save input shape and preprocessor
    import joblib
    joblib.dump(preprocessor, 'emsc_wavelet_preprocessor.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    
    # Save input shape for CNN model
    np.save('cnn_input_shape.npy', np.array(input_shape))
    
    print("✓ Saved processed data and preprocessor")
    
    # Summary statistics
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Original features: {X_raw.shape[1]} wavelengths")
    print(f"Wavelet features: {X_train_processed.shape[1:]} (2D grid + channel)")
    print(f"CNN input shape: {input_shape} + 1 channel")
    print(f"Training samples: {X_train_processed.shape[0]}")
    print(f"Test samples: {X_test_processed.shape[0]}")
    print(f"Classes: {le.classes_}")
    print("Preprocessing: EMSC + Wavelet decomposition")
    print("Ready for CNN modeling!")

if __name__ == "__main__":
    main() 