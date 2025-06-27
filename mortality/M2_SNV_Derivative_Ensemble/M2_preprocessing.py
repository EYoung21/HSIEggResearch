"""
M2 Preprocessing: SNV + Derivatives for Mortality Classification
Enhanced spectral preprocessing with derivative features for mortality prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

class MortalityPreprocessor:
    """SNV + Derivatives preprocessing for mortality classification"""
    
    def __init__(self, derivative_window=15, polynomial_order=3):
        self.derivative_window = derivative_window
        self.polynomial_order = polynomial_order
        self.scaler = StandardScaler()
        self.preprocessing_stats = {}
        
    def apply_snv(self, spectra):
        """Apply Standard Normal Variate normalization"""
        print("Applying SNV normalization...")
        
        snv_spectra = np.zeros_like(spectra)
        
        for i in range(spectra.shape[0]):
            spectrum = spectra[i, :]
            
            # Calculate mean and std for each spectrum
            mean_spectrum = np.mean(spectrum)
            std_spectrum = np.std(spectrum)
            
            # Avoid division by zero
            if std_spectrum > 1e-8:
                snv_spectra[i, :] = (spectrum - mean_spectrum) / std_spectrum
            else:
                snv_spectra[i, :] = spectrum - mean_spectrum
        
        print(f"✓ SNV applied to {spectra.shape[0]} spectra")
        return snv_spectra
    
    def calculate_derivatives(self, spectra):
        """Calculate 1st and 2nd derivatives using Savitzky-Golay"""
        print(f"Calculating derivatives (window={self.derivative_window}, poly={self.polynomial_order})...")
        
        # 1st derivative
        first_derivative = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            first_derivative[i, :] = savgol_filter(
                spectra[i, :], 
                window_length=self.derivative_window,
                polyorder=self.polynomial_order,
                deriv=1
            )
        
        # 2nd derivative
        second_derivative = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            second_derivative[i, :] = savgol_filter(
                spectra[i, :], 
                window_length=self.derivative_window,
                polyorder=self.polynomial_order,
                deriv=2
            )
        
        print(f"✓ Derivatives calculated: {first_derivative.shape}")
        return first_derivative, second_derivative
    
    def create_spectral_features(self, original, first_deriv, second_deriv, wavelengths):
        """Create comprehensive spectral features"""
        print("Creating enhanced spectral features...")
        
        features = []
        feature_names = []
        
        # 1. Original SNV spectra
        features.append(original)
        feature_names.extend([f'SNV_{w:.2f}nm' for w in wavelengths])
        
        # 2. First derivative
        features.append(first_deriv)
        feature_names.extend([f'1stDeriv_{w:.2f}nm' for w in wavelengths])
        
        # 3. Second derivative  
        features.append(second_deriv)
        feature_names.extend([f'2ndDeriv_{w:.2f}nm' for w in wavelengths])
        
        # 4. Spectral ratios (biological regions)
        blue_region = (wavelengths >= 400) & (wavelengths <= 500)  # Carotenoids
        red_region = (wavelengths >= 600) & (wavelengths <= 700)   # Proteins
        nir_region = (wavelengths >= 700) & (wavelengths <= 1000)  # Water/Lipids
        
        if np.any(blue_region) and np.any(red_region) and np.any(nir_region):
            blue_mean = np.mean(original[:, blue_region], axis=1)
            red_mean = np.mean(original[:, red_region], axis=1)
            nir_mean = np.mean(original[:, nir_region], axis=1)
            
            # Ratio features
            ratio_features = np.column_stack([
                red_mean / (blue_mean + 1e-8),    # Protein/Carotenoid
                nir_mean / (red_mean + 1e-8),     # Water-Lipid/Protein
                nir_mean / (blue_mean + 1e-8),    # Water-Lipid/Carotenoid
                (red_mean + nir_mean) / (blue_mean + 1e-8)  # Combined/Carotenoid
            ])
            
            features.append(ratio_features)
            feature_names.extend(['Protein_Carotenoid_Ratio', 'NIR_Protein_Ratio', 
                                'NIR_Carotenoid_Ratio', 'Combined_Carotenoid_Ratio'])
        
        # 5. Statistical features from derivatives
        # First derivative stats
        first_deriv_stats = np.column_stack([
            np.mean(first_deriv, axis=1),      # Mean slope
            np.std(first_deriv, axis=1),       # Slope variability
            np.max(first_deriv, axis=1),       # Max positive slope
            np.min(first_deriv, axis=1),       # Max negative slope
        ])
        
        features.append(first_deriv_stats)
        feature_names.extend(['1stDeriv_Mean', '1stDeriv_Std', '1stDeriv_Max', '1stDeriv_Min'])
        
        # Second derivative stats
        second_deriv_stats = np.column_stack([
            np.mean(second_deriv, axis=1),     # Mean curvature
            np.std(second_deriv, axis=1),      # Curvature variability
            np.max(second_deriv, axis=1),      # Max curvature
            np.min(second_deriv, axis=1),      # Min curvature
        ])
        
        features.append(second_deriv_stats)
        feature_names.extend(['2ndDeriv_Mean', '2ndDeriv_Std', '2ndDeriv_Max', '2ndDeriv_Min'])
        
        # Combine all features
        combined_features = np.concatenate(features, axis=1)
        
        print(f"✓ Enhanced features created: {combined_features.shape}")
        print(f"  - Original SNV: {original.shape[1]} features")
        print(f"  - 1st Derivative: {first_deriv.shape[1]} features") 
        print(f"  - 2nd Derivative: {second_deriv.shape[1]} features")
        print(f"  - Ratio features: 4 features")
        print(f"  - Derivative stats: 8 features")
        
        return combined_features, feature_names
    
    def preprocess_mortality_data(self, spectral_data, wavelengths):
        """Complete preprocessing pipeline for mortality classification"""
        print("\n" + "="*60)
        print("M2 PREPROCESSING: SNV + DERIVATIVES")
        print("="*60)
        
        # Apply SNV normalization
        snv_spectra = self.apply_snv(spectral_data)
        
        # Calculate derivatives
        first_deriv, second_deriv = self.calculate_derivatives(snv_spectra)
        
        # Create comprehensive features
        enhanced_features, feature_names = self.create_spectral_features(
            snv_spectra, first_deriv, second_deriv, wavelengths
        )
        
        # Apply final scaling
        print("Applying final StandardScaler...")
        scaled_features = self.scaler.fit_transform(enhanced_features)
        
        # Store preprocessing information
        self.preprocessing_stats = {
            'n_samples': spectral_data.shape[0],
            'n_wavelengths': spectral_data.shape[1],
            'n_enhanced_features': enhanced_features.shape[1],
            'derivative_window': self.derivative_window,
            'polynomial_order': self.polynomial_order,
            'feature_names': feature_names,
            'wavelength_range': f"{wavelengths.min():.2f} - {wavelengths.max():.2f} nm",
            'snv_mean': np.mean(snv_spectra),
            'snv_std': np.std(snv_spectra),
            'final_feature_mean': np.mean(scaled_features),
            'final_feature_std': np.std(scaled_features)
        }
        
        print(f"\n✓ M2 preprocessing completed:")
        print(f"  - Input: {spectral_data.shape}")
        print(f"  - Enhanced features: {enhanced_features.shape}")
        print(f"  - Scaled features: {scaled_features.shape}")
        
        return scaled_features
    
    def transform_new_data(self, spectral_data, wavelengths):
        """Transform new data using fitted preprocessing"""
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Preprocessor must be fitted first")
        
        # Apply same preprocessing steps
        snv_spectra = self.apply_snv(spectral_data)
        first_deriv, second_deriv = self.calculate_derivatives(snv_spectra)
        enhanced_features, _ = self.create_spectral_features(
            snv_spectra, first_deriv, second_deriv, wavelengths
        )
        
        # Apply fitted scaler
        return self.scaler.transform(enhanced_features)

def main():
    """Test preprocessing pipeline"""
    print("Testing M2 preprocessing pipeline...")
    
    # Load wavelengths
    wavelengths_df = pd.read_csv('../../gender/G1_MSC_SG_LightGBM/wavelengths.csv')
    wavelengths = wavelengths_df['wavelength'].values
    
    # Test with sample data
    n_samples = 100
    n_wavelengths = len(wavelengths)
    test_data = np.random.randn(n_samples, n_wavelengths) * 0.1 + 1.0
    
    # Initialize preprocessor
    preprocessor = MortalityPreprocessor(derivative_window=15, polynomial_order=3)
    
    # Apply preprocessing
    processed_data = preprocessor.preprocess_mortality_data(test_data, wavelengths)
    
    print(f"\nTest completed:")
    print(f"Input shape: {test_data.shape}")
    print(f"Output shape: {processed_data.shape}")
    print(f"Feature enhancement: {processed_data.shape[1] / test_data.shape[1]:.2f}x")

if __name__ == "__main__":
    main() 