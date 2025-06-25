import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class SpectralAugmentationSuite:
    """
    Comprehensive data augmentation suite for hyperspectral data
    Designed to address class imbalance while preserving spectral characteristics
    """
    
    def __init__(self, random_state=42):
        """
        Initialize augmentation suite
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def add_gaussian_noise(self, spectrum, noise_factor=0.01):
        """
        Add Gaussian noise to spectrum
        
        Args:
            spectrum: 1D spectral array
            noise_factor: Standard deviation of noise relative to signal std
            
        Returns:
            Augmented spectrum
        """
        noise = np.random.normal(0, noise_factor * np.std(spectrum), spectrum.shape)
        return spectrum + noise
    
    def spectral_shift(self, spectrum, max_shift=5):
        """
        Randomly shift spectrum values (simulating wavelength calibration variations)
        
        Args:
            spectrum: 1D spectral array
            max_shift: Maximum number of wavelength positions to shift
            
        Returns:
            Shifted spectrum
        """
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return spectrum
        
        shifted = np.zeros_like(spectrum)
        if shift > 0:
            shifted[shift:] = spectrum[:-shift]
            shifted[:shift] = spectrum[0]  # Pad with first value
        else:
            shifted[:shift] = spectrum[-shift:]
            shifted[shift:] = spectrum[-1]  # Pad with last value
        
        return shifted
    
    def intensity_scaling(self, spectrum, scale_range=(0.9, 1.1)):
        """
        Random intensity scaling (simulating illumination variations)
        
        Args:
            spectrum: 1D spectral array
            scale_range: (min_scale, max_scale) tuple
            
        Returns:
            Scaled spectrum
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return spectrum * scale
    
    def baseline_drift(self, spectrum, drift_amplitude=0.05):
        """
        Add polynomial baseline drift
        
        Args:
            spectrum: 1D spectral array
            drift_amplitude: Maximum drift amplitude relative to signal range
            
        Returns:
            Spectrum with baseline drift
        """
        n_points = len(spectrum)
        x = np.linspace(0, 1, n_points)
        
        # Random polynomial coefficients
        a = np.random.uniform(-drift_amplitude, drift_amplitude)
        b = np.random.uniform(-drift_amplitude, drift_amplitude)
        c = np.random.uniform(-drift_amplitude, drift_amplitude)
        
        # Create polynomial baseline
        signal_range = np.max(spectrum) - np.min(spectrum)
        baseline = signal_range * (a * x**2 + b * x + c)
        
        return spectrum + baseline
    
    def spectral_smoothing(self, spectrum, window_size=5):
        """
        Apply random smoothing (simulating different acquisition settings)
        
        Args:
            spectrum: 1D spectral array
            window_size: Size of smoothing window
            
        Returns:
            Smoothed spectrum
        """
        if np.random.random() < 0.5:  # 50% chance to apply
            # Simple moving average
            kernel = np.ones(window_size) / window_size
            return np.convolve(spectrum, kernel, mode='same')
        return spectrum
    
    def spectral_warping(self, spectrum, warp_factor=0.02):
        """
        Non-linear spectral warping (simulating instrument variations)
        
        Args:
            spectrum: 1D spectral array
            warp_factor: Maximum warping strength
            
        Returns:
            Warped spectrum
        """
        n_points = len(spectrum)
        
        # Create random warping field
        warp = np.random.uniform(-warp_factor, warp_factor, n_points)
        
        # Apply warping by interpolation
        original_indices = np.arange(n_points)
        warped_indices = original_indices + warp * n_points
        
        # Ensure indices stay within bounds
        warped_indices = np.clip(warped_indices, 0, n_points - 1)
        
        # Interpolate to get warped spectrum
        warped_spectrum = np.interp(original_indices, warped_indices, spectrum)
        
        return warped_spectrum
    
    def mixup_augmentation(self, spectrum1, spectrum2, alpha=0.2):
        """
        Mixup augmentation between two spectra
        
        Args:
            spectrum1: First spectrum
            spectrum2: Second spectrum  
            alpha: Mixup parameter
            
        Returns:
            Mixed spectrum
        """
        lam = np.random.beta(alpha, alpha)
        return lam * spectrum1 + (1 - lam) * spectrum2
    
    def cutout_augmentation(self, spectrum, cutout_size=20, num_cutouts=2):
        """
        Spectral cutout (zeroing random spectral regions)
        
        Args:
            spectrum: 1D spectral array
            cutout_size: Size of each cutout region
            num_cutouts: Number of cutout regions
            
        Returns:
            Spectrum with cutouts
        """
        augmented = spectrum.copy()
        n_points = len(spectrum)
        
        for _ in range(num_cutouts):
            if np.random.random() < 0.5:  # 50% chance per cutout
                start = np.random.randint(0, max(1, n_points - cutout_size))
                end = min(start + cutout_size, n_points)
                augmented[start:end] = 0
        
        return augmented

class RawSpectralPreprocessor:
    """
    Minimal preprocessing for raw spectral data with augmentation
    Preserves maximum information while enabling Transformer processing
    """
    
    def __init__(self, augmentation_factor=3, augmentation_suite=None):
        """
        Initialize preprocessor
        
        Args:
            augmentation_factor: How many augmented samples per original minority sample
            augmentation_suite: SpectralAugmentationSuite instance
        """
        self.augmentation_factor = augmentation_factor
        self.augmentation_suite = augmentation_suite or SpectralAugmentationSuite()
        self.scaler = StandardScaler()
        self.wavelengths = None
        
    def minimal_preprocessing(self, spectra):
        """
        Apply minimal preprocessing to preserve spectral information
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Minimally processed spectra
        """
        print("Applying minimal preprocessing (preserving raw information)...")
        
        # Remove extreme outliers only (3-sigma rule)
        def remove_outliers(spectrum):
            mean = np.mean(spectrum)
            std = np.std(spectrum)
            return np.clip(spectrum, mean - 3*std, mean + 3*std)
        
        processed = np.array([remove_outliers(spectrum) for spectrum in spectra])
        
        # Basic normalization to prevent numerical issues
        processed = self.scaler.fit_transform(processed)
        
        return processed
    
    def augment_minority_class(self, X_train, y_train):
        """
        Apply comprehensive augmentation to minority class
        
        Args:
            X_train: Training spectra
            y_train: Training labels
            
        Returns:
            Augmented training set
        """
        print("Applying data augmentation to address class imbalance...")
        
        # Identify minority class
        unique, counts = np.unique(y_train, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        
        print(f"Original class distribution: {dict(zip(unique, counts))}")
        print(f"Minority class: {minority_class}, Majority class: {majority_class}")
        
        # Get minority and majority samples
        minority_mask = y_train == minority_class
        majority_mask = y_train == majority_class
        
        minority_X = X_train[minority_mask]
        majority_X = X_train[majority_mask]
        
        # Calculate target minority samples
        target_minority_samples = len(majority_X)  # Balance classes
        current_minority_samples = len(minority_X)
        samples_to_generate = target_minority_samples - current_minority_samples
        
        print(f"Generating {samples_to_generate} augmented minority samples...")
        
        if samples_to_generate <= 0:
            print("Classes already balanced or minority is majority")
            return X_train, y_train
        
        # Generate augmented samples
        augmented_X = []
        augmented_y = []
        
        for i in range(samples_to_generate):
            # Select random minority sample
            base_idx = np.random.randint(0, len(minority_X))
            base_spectrum = minority_X[base_idx]
            
            # Apply random combination of augmentations
            augmented = base_spectrum.copy()
            
            # Gaussian noise (always apply with varying intensity)
            augmented = self.augmentation_suite.add_gaussian_noise(
                augmented, noise_factor=np.random.uniform(0.005, 0.02)
            )
            
            # Random selection of other augmentations
            if np.random.random() < 0.7:  # 70% chance
                augmented = self.augmentation_suite.intensity_scaling(
                    augmented, scale_range=(0.95, 1.05)
                )
            
            if np.random.random() < 0.5:  # 50% chance
                augmented = self.augmentation_suite.spectral_shift(
                    augmented, max_shift=3
                )
            
            if np.random.random() < 0.4:  # 40% chance
                augmented = self.augmentation_suite.baseline_drift(
                    augmented, drift_amplitude=0.03
                )
            
            if np.random.random() < 0.3:  # 30% chance
                augmented = self.augmentation_suite.spectral_smoothing(
                    augmented, window_size=np.random.randint(3, 8)
                )
            
            if np.random.random() < 0.3:  # 30% chance
                augmented = self.augmentation_suite.spectral_warping(
                    augmented, warp_factor=0.01
                )
            
            if np.random.random() < 0.2:  # 20% chance
                augmented = self.augmentation_suite.cutout_augmentation(
                    augmented, cutout_size=np.random.randint(10, 25), num_cutouts=1
                )
            
            # Mixup with another minority sample (20% chance)
            if np.random.random() < 0.2 and len(minority_X) > 1:
                other_idx = np.random.randint(0, len(minority_X))
                if other_idx != base_idx:
                    augmented = self.augmentation_suite.mixup_augmentation(
                        augmented, minority_X[other_idx], alpha=0.3
                    )
            
            augmented_X.append(augmented)
            augmented_y.append(minority_class)
        
        # Combine original and augmented data
        X_augmented = np.vstack([X_train, np.array(augmented_X)])
        y_augmented = np.concatenate([y_train, np.array(augmented_y)])
        
        # Shuffle the augmented dataset
        shuffle_idx = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[shuffle_idx]
        y_augmented = y_augmented[shuffle_idx]
        
        # Print final distribution
        unique_aug, counts_aug = np.unique(y_augmented, return_counts=True)
        print(f"Final class distribution: {dict(zip(unique_aug, counts_aug))}")
        
        return X_augmented, y_augmented
    
    def prepare_for_transformer(self, spectra):
        """
        Prepare spectral data for Transformer input
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Transformer-ready sequences
        """
        # Transformers expect (batch_size, sequence_length, features)
        # For spectral data: (samples, wavelengths, 1)
        transformer_input = spectra.reshape(spectra.shape[0], spectra.shape[1], 1)
        
        print(f"✓ Transformer input shape: {transformer_input.shape}")
        print(f"  - Batch dimension: {transformer_input.shape[0]} samples")
        print(f"  - Sequence length: {transformer_input.shape[1]} wavelengths") 
        print(f"  - Feature dimension: {transformer_input.shape[2]} (single feature)")
        
        return transformer_input
    
    def fit_transform(self, spectra, labels, wavelengths):
        """
        Fit preprocessor and transform spectra with augmentation
        
        Args:
            spectra: 2D array (samples x wavelengths)
            labels: Label array
            wavelengths: Wavelength values
            
        Returns:
            Processed and augmented data ready for Transformer
        """
        self.wavelengths = wavelengths
        
        print("Applying minimal preprocessing...")
        processed_spectra = self.minimal_preprocessing(spectra)
        
        print("Applying data augmentation...")
        augmented_X, augmented_y = self.augment_minority_class(processed_spectra, labels)
        
        print("Preparing for Transformer...")
        transformer_ready = self.prepare_for_transformer(augmented_X)
        
        return transformer_ready, augmented_y
    
    def transform(self, spectra):
        """
        Transform new spectra using fitted parameters (no augmentation)
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Transformer-ready sequences
        """
        processed = self.scaler.transform(spectra)
        return self.prepare_for_transformer(processed)

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
    print("PREPARING GENDER DATASET - G4 EXPERIMENT")
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
    Main preprocessing pipeline for G4 experiment
    """
    print("="*60)
    print("G4 EXPERIMENT: Raw + Augmentation + Transformer Preprocessing")
    print("="*60)
    
    # Load and merge data (Day 0 for pre-incubation prediction)
    merged_df = load_and_merge_data(day='D0')
    
    # Prepare gender dataset
    X_raw, y, le, wavelength_cols, gender_df = prepare_gender_dataset(merged_df)
    
    # Create train/test splits
    X_train_raw, X_test_raw, y_train, y_test = create_train_test_splits(X_raw, y)
    
    # Initialize and apply preprocessing with augmentation
    print("\n" + "="*50)
    print("APPLYING RAW + AUGMENTATION PREPROCESSING")
    print("="*50)
    
    preprocessor = RawSpectralPreprocessor(
        augmentation_factor=3  # Generate 3x minority samples
    )
    
    # Fit on training data and transform with augmentation
    X_train_processed, y_train_augmented = preprocessor.fit_transform(
        X_train_raw, y_train, wavelength_cols
    )
    
    # Transform test data (no augmentation)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    print(f"✓ Processed training set shape: {X_train_processed.shape}")
    print(f"✓ Processed test set shape: {X_test_processed.shape}")
    print(f"✓ Augmented training labels shape: {y_train_augmented.shape}")
    
    # Save processed data
    print("\nSaving processed data...")
    np.save('X_train_processed.npy', X_train_processed)
    np.save('X_test_processed.npy', X_test_processed)
    np.save('y_train_augmented.npy', y_train_augmented)
    np.save('y_test.npy', y_test)
    
    # Save preprocessor and other components
    import joblib
    joblib.dump(preprocessor, 'raw_augmentation_preprocessor.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    
    # Save sequence length for Transformer model
    sequence_length = X_train_processed.shape[1]
    np.save('transformer_sequence_length.npy', np.array([sequence_length]))
    
    print("✓ Saved processed data and preprocessor")
    
    # Summary statistics
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Original features: {X_raw.shape[1]} wavelengths")
    print(f"Transformer sequence length: {sequence_length}")
    print(f"Original training samples: {X_train_raw.shape[0]}")
    print(f"Augmented training samples: {X_train_processed.shape[0]}")
    print(f"Test samples: {X_test_processed.shape[0]}")
    print(f"Classes: {le.classes_}")
    print("Preprocessing: Minimal + Comprehensive augmentation")
    print("Ready for Transformer modeling!")

if __name__ == "__main__":
    main()