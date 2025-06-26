"""
M3 Preprocessing: EMSC + Augmentation for Mortality Classification
Advanced spectral preprocessing with Extended Multiplicative Scatter Correction and data augmentation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

class MortalityEMSCProcessor:
    """EMSC + Augmentation preprocessing for mortality classification"""
    
    def __init__(self, n_components=5, reference_spectrum=None, augmentation_factor=3):
        self.n_components = n_components
        self.reference_spectrum = reference_spectrum
        self.augmentation_factor = augmentation_factor
        self.scaler = StandardScaler()
        self.pca_interferents = PCA(n_components=n_components)
        self.emsc_coefficients = None
        self.preprocessing_stats = {}
        
    def calculate_reference_spectrum(self, spectra):
        """Calculate mean reference spectrum for EMSC"""
        if self.reference_spectrum is None:
            self.reference_spectrum = np.mean(spectra, axis=0)
            print(f"✓ Reference spectrum calculated from {spectra.shape[0]} samples")
        return self.reference_spectrum
    
    def fit_interferent_basis(self, spectra):
        """Fit PCA basis for interferent removal in EMSC"""
        print(f"Fitting interferent basis with {self.n_components} components...")
        
        # Center the spectra around reference
        centered_spectra = spectra - self.reference_spectrum
        
        # Fit PCA to extract major variation patterns (interferents)
        self.pca_interferents.fit(centered_spectra)
        
        print(f"✓ Interferent basis fitted - Explained variance: {np.sum(self.pca_interferents.explained_variance_ratio_):.4f}")
        return self.pca_interferents.components_
    
    def apply_emsc(self, spectra):
        """Apply Extended Multiplicative Scatter Correction"""
        print("Applying EMSC correction...")
        
        if self.reference_spectrum is None:
            raise ValueError("Reference spectrum must be calculated first")
        
        n_samples, n_wavelengths = spectra.shape
        corrected_spectra = np.zeros_like(spectra)
        coefficients = np.zeros((n_samples, 2 + self.n_components))  # offset, scale, interferents
        
        # Get interferent basis
        interferent_basis = self.pca_interferents.components_
        
        for i in range(n_samples):
            spectrum = spectra[i, :]
            
            # Build design matrix: [ones, reference, interferent_components]
            design_matrix = np.column_stack([
                np.ones(n_wavelengths),
                self.reference_spectrum,
                interferent_basis.T
            ])
            
            # Solve least squares: spectrum = a + b*reference + c1*int1 + c2*int2 + ...
            try:
                coeffs = np.linalg.lstsq(design_matrix, spectrum, rcond=None)[0]
                coefficients[i, :] = coeffs
                
                # Correct spectrum: (spectrum - offset - interferents) / scale
                offset = coeffs[0]
                scale = coeffs[1] if abs(coeffs[1]) > 1e-8 else 1.0
                interferents = np.dot(interferent_basis.T, coeffs[2:])
                
                corrected_spectra[i, :] = (spectrum - offset - interferents) / scale
                
            except np.linalg.LinAlgError:
                # Fallback to simple scaling if EMSC fails
                corrected_spectra[i, :] = spectrum / np.mean(spectrum)
                coefficients[i, 1] = np.mean(spectrum)
        
        self.emsc_coefficients = coefficients
        print(f"✓ EMSC applied to {n_samples} spectra")
        
        return corrected_spectra
    
    def augment_spectral_data(self, spectra, labels, noise_level=0.01, shift_range=2, 
                            scale_range=0.05, baseline_range=0.02):
        """Apply data augmentation techniques for spectral data"""
        print(f"Applying data augmentation (factor={self.augmentation_factor})...")
        
        original_size = spectra.shape[0]
        augmented_spectra = [spectra]
        augmented_labels = [labels]
        
        for aug_idx in range(self.augmentation_factor - 1):
            aug_spectra = np.copy(spectra)
            
            for i in range(spectra.shape[0]):
                spectrum = spectra[i, :].copy()
                
                # 1. Add Gaussian noise
                noise = np.random.normal(0, noise_level * np.std(spectrum), spectrum.shape)
                spectrum += noise
                
                # 2. Spectral shift (wavelength calibration variations)
                if shift_range > 0:
                    shift = np.random.randint(-shift_range, shift_range + 1)
                    if shift != 0:
                        spectrum = np.roll(spectrum, shift)
                        # Handle edge effects
                        if shift > 0:
                            spectrum[:shift] = spectrum[shift]
                        else:
                            spectrum[shift:] = spectrum[shift-1]
                
                # 3. Intensity scaling (illumination variations)
                if scale_range > 0:
                    scale_factor = 1 + np.random.uniform(-scale_range, scale_range)
                    spectrum *= scale_factor
                
                # 4. Baseline drift
                if baseline_range > 0:
                    baseline_shift = np.random.uniform(-baseline_range, baseline_range)
                    spectrum += baseline_shift * np.mean(spectrum)
                
                # 5. Smoothing variations (instrument response)
                if np.random.random() < 0.3:  # 30% chance
                    window_size = np.random.choice([5, 7, 9, 11])
                    if window_size < len(spectrum):
                        spectrum = savgol_filter(spectrum, window_size, 2)
                
                aug_spectra[i, :] = spectrum
            
            augmented_spectra.append(aug_spectra)
            augmented_labels.append(labels.copy())
        
        # Combine all augmented data
        final_spectra = np.vstack(augmented_spectra)
        final_labels = np.concatenate(augmented_labels)
        
        print(f"✓ Data augmented: {original_size} → {final_spectra.shape[0]} samples")
        return final_spectra, final_labels
    
    def create_cnn_features(self, spectra, window_size=15):
        """Create features suitable for CNN processing"""
        print("Creating CNN-compatible features...")
        
        # 1. Raw EMSC spectra
        raw_features = spectra.copy()
        
        # 2. Smoothed versions for multi-scale analysis
        smooth_features = []
        for window in [5, 11, 21]:
            if window < spectra.shape[1]:
                smoothed = np.array([
                    savgol_filter(spectra[i, :], window, 2) 
                    for i in range(spectra.shape[0])
                ])
                smooth_features.append(smoothed)
        
        # 3. Derivative features
        first_deriv = np.array([
            savgol_filter(spectra[i, :], window_size, 2, deriv=1)
            for i in range(spectra.shape[0])
        ])
        
        second_deriv = np.array([
            savgol_filter(spectra[i, :], window_size, 2, deriv=2)
            for i in range(spectra.shape[0])
        ])
        
        # 4. Stack features for CNN channels
        if smooth_features:
            # Create multi-channel input: [raw, smooth1, smooth2, smooth3, 1st_deriv, 2nd_deriv]
            cnn_features = np.stack([
                raw_features,
                smooth_features[0] if len(smooth_features) > 0 else raw_features,
                smooth_features[1] if len(smooth_features) > 1 else raw_features,
                smooth_features[2] if len(smooth_features) > 2 else raw_features,
                first_deriv,
                second_deriv
            ], axis=-1)  # Shape: (samples, wavelengths, channels)
        else:
            cnn_features = np.stack([raw_features, first_deriv, second_deriv], axis=-1)
        
        print(f"✓ CNN features created: {cnn_features.shape}")
        return cnn_features
    
    def preprocess_mortality_data(self, spectral_data, labels, wavelengths):
        """Complete EMSC + Augmentation preprocessing pipeline"""
        print("\n" + "="*60)
        print("M3 PREPROCESSING: EMSC + AUGMENTATION + CNN")
        print("="*60)
        
        # Calculate reference spectrum
        reference = self.calculate_reference_spectrum(spectral_data)
        
        # Fit interferent basis
        interferent_basis = self.fit_interferent_basis(spectral_data)
        
        # Apply EMSC correction
        emsc_corrected = self.apply_emsc(spectral_data)
        
        # Apply data augmentation
        augmented_spectra, augmented_labels = self.augment_spectral_data(
            emsc_corrected, labels
        )
        
        # Create CNN-compatible features
        cnn_features = self.create_cnn_features(augmented_spectra)
        
        # Final normalization for CNN training
        print("Applying final normalization...")
        n_samples, n_wavelengths, n_channels = cnn_features.shape
        
        # Normalize each channel separately
        normalized_features = np.zeros_like(cnn_features)
        for channel in range(n_channels):
            channel_data = cnn_features[:, :, channel].reshape(-1, n_wavelengths)
            normalized_channel = self.scaler.fit_transform(channel_data)
            normalized_features[:, :, channel] = normalized_channel.reshape(n_samples, n_wavelengths)
        
        # Store preprocessing information
        self.preprocessing_stats = {
            'n_samples_original': spectral_data.shape[0],
            'n_samples_augmented': augmented_spectra.shape[0],
            'n_wavelengths': spectral_data.shape[1],
            'n_channels': n_channels,
            'augmentation_factor': self.augmentation_factor,
            'emsc_components': self.n_components,
            'wavelength_range': f"{wavelengths.min():.2f} - {wavelengths.max():.2f} nm",
            'emsc_explained_variance': np.sum(self.pca_interferents.explained_variance_ratio_),
            'cnn_input_shape': normalized_features.shape,
            'class_distribution_original': np.bincount(labels),
            'class_distribution_augmented': np.bincount(augmented_labels)
        }
        
        print(f"\n✓ M3 preprocessing completed:")
        print(f"  - Original samples: {spectral_data.shape[0]}")
        print(f"  - Augmented samples: {augmented_spectra.shape[0]}")
        print(f"  - CNN input shape: {normalized_features.shape}")
        print(f"  - EMSC explained variance: {self.preprocessing_stats['emsc_explained_variance']:.4f}")
        
        return normalized_features, augmented_labels
    
    def transform_new_data(self, spectral_data, wavelengths):
        """Transform new data using fitted preprocessing"""
        if self.reference_spectrum is None:
            raise ValueError("Preprocessor must be fitted first")
        
        # Apply EMSC correction
        emsc_corrected = self.apply_emsc(spectral_data)
        
        # Create CNN features (no augmentation for test data)
        cnn_features = self.create_cnn_features(emsc_corrected)
        
        # Apply fitted normalization
        n_samples, n_wavelengths, n_channels = cnn_features.shape
        normalized_features = np.zeros_like(cnn_features)
        
        for channel in range(n_channels):
            channel_data = cnn_features[:, :, channel].reshape(-1, n_wavelengths)
            normalized_channel = self.scaler.transform(channel_data)
            normalized_features[:, :, channel] = normalized_channel.reshape(n_samples, n_wavelengths)
        
        return normalized_features

def main():
    """Test preprocessing pipeline"""
    print("Testing M3 preprocessing pipeline...")
    
    # Load wavelengths
    wavelengths_df = pd.read_csv('../../gender/G1_MSC_SG_LightGBM/wavelengths.csv')
    wavelengths = wavelengths_df['wavelength'].values
    
    # Test with sample data
    n_samples = 100
    n_wavelengths = len(wavelengths)
    test_data = np.random.randn(n_samples, n_wavelengths) * 0.1 + 1.0
    test_labels = np.random.randint(0, 2, n_samples)
    
    # Initialize preprocessor
    preprocessor = MortalityEMSCProcessor(
        n_components=5, 
        augmentation_factor=3
    )
    
    # Apply preprocessing
    processed_data, processed_labels = preprocessor.preprocess_mortality_data(
        test_data, test_labels, wavelengths
    )
    
    print(f"\nTest completed:")
    print(f"Input shape: {test_data.shape}")
    print(f"Output shape: {processed_data.shape}")
    print(f"Augmentation: {len(test_labels)} → {len(processed_labels)} samples")

if __name__ == "__main__":
    main() 