"""
M5 Preprocessing: Wavelets + MSC + MultiTask for Mortality Classification
Advanced preprocessing combining wavelet transforms and multiplicative scatter correction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import signal
import pywt
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class WaveletMSCPreprocessor:
    """M5 Preprocessor: Wavelet transforms + MSC for multi-task learning"""
    
    def __init__(self, wavelet='db4', levels=3, msc_reference='mean', enable_derivatives=True):
        self.wavelet = wavelet
        self.levels = levels
        self.msc_reference = msc_reference
        self.enable_derivatives = enable_derivatives
        
        # Preprocessing components
        self.msc_reference_spectrum = None
        self.scaler = StandardScaler()
        self.preprocessing_stats = {}
        
    def apply_msc_correction(self, spectra):
        """Apply Multiplicative Scatter Correction"""
        print("Applying MSC (Multiplicative Scatter Correction)...")
        
        if self.msc_reference_spectrum is None:
            if self.msc_reference == 'mean':
                self.msc_reference_spectrum = np.mean(spectra, axis=0)
            elif self.msc_reference == 'median':
                self.msc_reference_spectrum = np.median(spectra, axis=0)
            else:
                raise ValueError("msc_reference must be 'mean' or 'median'")
        
        msc_corrected = np.zeros_like(spectra)
        
        for i, spectrum in enumerate(spectra):
            # Avoid division by zero and numerical instability
            ref_safe = self.msc_reference_spectrum.copy()
            ref_safe[np.abs(ref_safe) < 1e-10] = 1e-10
            
            # Linear regression: spectrum = a + b * reference
            X = np.vstack([np.ones(len(ref_safe)), ref_safe]).T
            try:
                coeffs = np.linalg.lstsq(X, spectrum, rcond=None)[0]
                a, b = coeffs[0], coeffs[1]
                
                # Avoid division by very small b values
                if np.abs(b) < 1e-10:
                    b = 1.0
                
                # MSC correction: (spectrum - a) / b
                msc_corrected[i] = (spectrum - a) / b
                
            except np.linalg.LinAlgError:
                # If regression fails, use original spectrum
                msc_corrected[i] = spectrum
        
        print(f"✓ MSC correction applied: {msc_corrected.shape}")
        return msc_corrected
    
    def wavelet_decomposition(self, spectra):
        """Apply discrete wavelet transform decomposition"""
        print(f"Applying wavelet decomposition (wavelet: {self.wavelet}, levels: {self.levels})...")
        
        wavelet_features = []
        feature_names = []
        
        for spectrum in spectra:
            # Discrete Wavelet Transform
            coeffs = pywt.wavedec(spectrum, self.wavelet, level=self.levels)
            
            # Extract features from each decomposition level
            spectrum_features = []
            
            for level, coeff in enumerate(coeffs):
                # Statistical features for each level
                level_features = [
                    np.mean(coeff),           # Mean
                    np.std(coeff),            # Standard deviation
                    np.max(coeff),            # Maximum
                    np.min(coeff),            # Minimum
                    np.median(coeff),         # Median
                    np.percentile(coeff, 25), # 25th percentile
                    np.percentile(coeff, 75), # 75th percentile
                    np.sum(np.abs(coeff)),    # Energy (L1 norm)
                    np.sum(coeff**2),         # Power (L2 norm)
                    len(coeff)                # Coefficient count
                ]
                
                spectrum_features.extend(level_features)
                
                # Feature names (only create once)
                if len(feature_names) == 0:
                    level_name = 'approx' if level == 0 else f'detail_{level}'
                    for stat_name in ['mean', 'std', 'max', 'min', 'median', 'q25', 'q75', 'energy', 'power', 'count']:
                        feature_names.append(f'wavelet_{level_name}_{stat_name}')
            
            wavelet_features.append(spectrum_features)
        
        wavelet_features = np.array(wavelet_features)
        print(f"✓ Wavelet features extracted: {wavelet_features.shape[1]} features from {self.levels+1} decomposition levels")
        
        return wavelet_features, feature_names
    
    def wavelet_reconstruction_features(self, spectra):
        """Extract features from selective wavelet reconstruction"""
        print("Extracting wavelet reconstruction features...")
        
        reconstruction_features = []
        feature_names = []
        
        for spectrum in spectra:
            # Discrete Wavelet Transform
            coeffs = pywt.wavedec(spectrum, self.wavelet, level=self.levels)
            
            # Reconstruct from different levels
            spectrum_features = []
            
            # Reconstruction from approximation only (low-frequency)
            approx_only = coeffs.copy()
            for i in range(1, len(approx_only)):
                approx_only[i] = np.zeros_like(approx_only[i])
            
            reconstructed_approx = pywt.waverec(approx_only, self.wavelet)
            # Ensure same length as original
            if len(reconstructed_approx) != len(spectrum):
                reconstructed_approx = reconstructed_approx[:len(spectrum)]
            
            # Features from low-frequency reconstruction
            approx_features = [
                np.mean(reconstructed_approx),
                np.std(reconstructed_approx),
                np.sum(reconstructed_approx),
                np.max(reconstructed_approx) - np.min(reconstructed_approx)
            ]
            spectrum_features.extend(approx_features)
            
            # Reconstruction from details only (high-frequency)
            details_only = coeffs.copy()
            details_only[0] = np.zeros_like(details_only[0])  # Zero out approximation
            
            reconstructed_details = pywt.waverec(details_only, self.wavelet)
            # Ensure same length as original
            if len(reconstructed_details) != len(spectrum):
                reconstructed_details = reconstructed_details[:len(spectrum)]
            
            # Features from high-frequency reconstruction
            detail_features = [
                np.mean(np.abs(reconstructed_details)),
                np.std(reconstructed_details),
                np.sum(np.abs(reconstructed_details)),
                np.max(np.abs(reconstructed_details))
            ]
            spectrum_features.extend(detail_features)
            
            # Feature names (only create once)
            if len(feature_names) == 0:
                for stat_name in ['mean', 'std', 'sum', 'range']:
                    feature_names.append(f'wavelet_approx_{stat_name}')
                for stat_name in ['mean_abs', 'std', 'sum_abs', 'max_abs']:
                    feature_names.append(f'wavelet_detail_{stat_name}')
            
            reconstruction_features.append(spectrum_features)
        
        reconstruction_features = np.array(reconstruction_features)
        print(f"✓ Wavelet reconstruction features: {reconstruction_features.shape[1]} features")
        
        return reconstruction_features, feature_names
    
    def extract_derivative_features(self, spectra):
        """Extract derivative features using Savitzky-Golay filter"""
        if not self.enable_derivatives:
            return np.array([]), []
        
        print("Extracting derivative features...")
        
        derivative_features = []
        feature_names = []
        
        for spectrum in spectra:
            # 1st derivative
            try:
                first_deriv = savgol_filter(spectrum, window_length=min(15, len(spectrum)//4*2+1), polyorder=2, deriv=1)
            except:
                first_deriv = np.gradient(spectrum)
            
            # 2nd derivative
            try:
                second_deriv = savgol_filter(spectrum, window_length=min(15, len(spectrum)//4*2+1), polyorder=2, deriv=2)
            except:
                second_deriv = np.gradient(first_deriv)
            
            # Statistical features from derivatives
            spectrum_features = [
                # 1st derivative features
                np.mean(first_deriv),
                np.std(first_deriv),
                np.max(first_deriv),
                np.min(first_deriv),
                np.sum(np.abs(first_deriv)),
                
                # 2nd derivative features
                np.mean(second_deriv),
                np.std(second_deriv),
                np.max(second_deriv),
                np.min(second_deriv),
                np.sum(np.abs(second_deriv))
            ]
            
            derivative_features.append(spectrum_features)
            
            # Feature names (only create once)
            if len(feature_names) == 0:
                for deriv_order in ['1st', '2nd']:
                    for stat_name in ['mean', 'std', 'max', 'min', 'sum_abs']:
                        feature_names.append(f'deriv_{deriv_order}_{stat_name}')
        
        if len(derivative_features) > 0:
            derivative_features = np.array(derivative_features)
            print(f"✓ Derivative features extracted: {derivative_features.shape[1]} features")
        else:
            derivative_features = np.array([])
        
        return derivative_features, feature_names
    
    def extract_spectral_statistics(self, spectra):
        """Extract basic statistical features from spectra"""
        print("Extracting spectral statistical features...")
        
        statistical_features = []
        feature_names = []
        
        for spectrum in spectra:
            features = [
                np.mean(spectrum),           # Mean intensity
                np.std(spectrum),            # Standard deviation
                np.min(spectrum),            # Minimum intensity
                np.max(spectrum),            # Maximum intensity
                np.median(spectrum),         # Median intensity
                np.percentile(spectrum, 25), # 25th percentile
                np.percentile(spectrum, 75), # 75th percentile
                np.ptp(spectrum),            # Peak-to-peak range
                np.var(spectrum),            # Variance
                np.sum(spectrum),            # Total area
                np.argmax(spectrum),         # Peak location
                np.argmin(spectrum),         # Valley location
            ]
            
            statistical_features.append(features)
            
            # Feature names (only create once)
            if len(feature_names) == 0:
                feature_names = [
                    'spec_mean', 'spec_std', 'spec_min', 'spec_max', 'spec_median',
                    'spec_q25', 'spec_q75', 'spec_range', 'spec_var', 'spec_area',
                    'spec_peak_loc', 'spec_valley_loc'
                ]
        
        statistical_features = np.array(statistical_features)
        print(f"✓ Statistical features extracted: {statistical_features.shape[1]} features")
        
        return statistical_features, feature_names
    
    def preprocess_mortality_data(self, X_raw, wavelengths):
        """Main preprocessing pipeline for M5"""
        print("\n" + "="*60)
        print("M5 PREPROCESSING: WAVELETS + MSC + MULTITASK")
        print("="*60)
        
        print(f"Input data shape: {X_raw.shape}")
        print(f"Wavelength range: {wavelengths.min():.2f} - {wavelengths.max():.2f} nm")
        
        # Store original data stats
        self.preprocessing_stats = {
            'n_samples': X_raw.shape[0],
            'n_wavelengths': X_raw.shape[1],
            'wavelength_range': f"{wavelengths.min():.2f} - {wavelengths.max():.2f} nm",
            'original_shape': X_raw.shape,
            'preprocessing_method': 'Wavelets + MSC + MultiTask features',
            'wavelet': self.wavelet,
            'wavelet_levels': self.levels
        }
        
        # 1. Apply MSC correction
        X_msc = self.apply_msc_correction(X_raw)
        
        # 2. Wavelet decomposition features
        wavelet_features, wavelet_names = self.wavelet_decomposition(X_msc)
        
        # 3. Wavelet reconstruction features
        reconstruction_features, reconstruction_names = self.wavelet_reconstruction_features(X_msc)
        
        # 4. Extract derivative features
        derivative_features, derivative_names = self.extract_derivative_features(X_msc)
        
        # 5. Extract statistical features
        statistical_features, statistical_names = self.extract_spectral_statistics(X_msc)
        
        # 6. Combine all features
        print("\nCombining all feature types...")
        feature_sets = [X_msc, wavelet_features, reconstruction_features, statistical_features]
        feature_name_sets = [
            [f'MSC_{wavelengths[i]:.2f}nm' for i in range(len(wavelengths))],
            wavelet_names,
            reconstruction_names,
            statistical_names
        ]
        
        if derivative_features.size > 0:
            feature_sets.append(derivative_features)
            feature_name_sets.append(derivative_names)
        
        X_combined = np.hstack(feature_sets)
        all_feature_names = []
        for name_set in feature_name_sets:
            all_feature_names.extend(name_set)
        
        print(f"✓ Combined features: {X_combined.shape}")
        
        # 7. Final scaling
        print("Applying final feature scaling...")
        X_scaled = self.scaler.fit_transform(X_combined)
        print(f"✓ Features scaled: {X_scaled.shape}")
        
        # Store final preprocessing stats
        self.preprocessing_stats.update({
            'n_enhanced_features': X_scaled.shape[1],
            'feature_names': all_feature_names,
            'enhancement_ratio': X_scaled.shape[1] / X_raw.shape[1],
            'wavelet_features': len(wavelet_names),
            'reconstruction_features': len(reconstruction_names),
            'derivative_features': len(derivative_names) if derivative_features.size > 0 else 0,
            'statistical_features': len(statistical_names),
            'msc_applied': True,
            'final_scaling': 'StandardScaler'
        })
        
        print(f"\n✓ M5 preprocessing completed:")
        print(f"  - Original features: {X_raw.shape[1]}")
        print(f"  - Enhanced features: {X_scaled.shape[1]}")
        print(f"  - Enhancement ratio: {X_scaled.shape[1]/X_raw.shape[1]:.2f}x")
        print(f"  - Wavelet: {self.wavelet} ({self.levels} levels)")
        print(f"  - MSC reference: {self.msc_reference}")
        
        return X_scaled
    
    def transform_new_data(self, X_new, wavelengths):
        """Transform new data using fitted preprocessors"""
        print(f"Transforming new data: {X_new.shape}")
        
        # Apply same preprocessing steps
        X_msc = self.apply_msc_correction(X_new)
        
        wavelet_features, _ = self.wavelet_decomposition(X_msc)
        reconstruction_features, _ = self.wavelet_reconstruction_features(X_msc)
        derivative_features, _ = self.extract_derivative_features(X_msc)
        statistical_features, _ = self.extract_spectral_statistics(X_msc)
        
        # Combine features
        feature_sets = [X_msc, wavelet_features, reconstruction_features, statistical_features]
        if derivative_features.size > 0:
            feature_sets.append(derivative_features)
        
        X_combined = np.hstack(feature_sets)
        
        # Apply scaling
        X_scaled = self.scaler.transform(X_combined)
        
        print(f"✓ New data transformed: {X_scaled.shape}")
        return X_scaled

def main():
    """Example usage"""
    print("M5 Wavelet MSC Preprocessor")
    print("Usage: Import and use with raw spectral data")
    
    # Example parameters
    print("\nPreprocessing Configuration:")
    print("- Multiplicative Scatter Correction (MSC)")
    print("- Discrete Wavelet Transform (DWT)")
    print("- Wavelet reconstruction features")
    print("- Derivative analysis")
    print("- Statistical feature extraction")
    print("- Final standardization")

if __name__ == "__main__":
    main() 