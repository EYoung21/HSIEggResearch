"""
M8 Preprocessing: Multi-Derivatives + Conformal Prediction for Mortality Classification
Advanced preprocessing with multiple derivative transformations and conformal prediction setup
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

class MultiDerivativeProcessor:
    """M8 Preprocessor: Multiple derivative transformations for conformal prediction"""
    
    def __init__(self, 
                 enable_savgol_derivatives=True,
                 enable_gaussian_derivatives=True,
                 enable_numerical_derivatives=True,
                 enable_finite_differences=True,
                 savgol_window=15,
                 savgol_polyorder=3,
                 gaussian_sigma=2.0,
                 conformal_alpha=0.1,
                 use_robust_scaling=True):
        
        self.enable_savgol_derivatives = enable_savgol_derivatives
        self.enable_gaussian_derivatives = enable_gaussian_derivatives
        self.enable_numerical_derivatives = enable_numerical_derivatives
        self.enable_finite_differences = enable_finite_differences
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.gaussian_sigma = gaussian_sigma
        self.conformal_alpha = conformal_alpha
        self.use_robust_scaling = use_robust_scaling
        
        # Preprocessing components
        if use_robust_scaling:
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        self.preprocessing_stats = {}
        
    def apply_savgol_derivatives(self, spectra):
        """Extract Savitzky-Golay derivatives up to 3rd order"""
        if not self.enable_savgol_derivatives:
            return np.array([]), []
        
        print("Extracting Savitzky-Golay derivatives...")
        
        derivatives = []
        feature_names = []
        
        for spectrum in spectra:
            spectrum_derivatives = []
            
            # Adjust window length if needed
            window_length = min(self.savgol_window, len(spectrum)//4*2+1)
            
            try:
                # 1st derivative
                first_deriv = savgol_filter(spectrum, window_length=window_length, 
                                          polyorder=self.savgol_polyorder, deriv=1)
                
                # 2nd derivative
                second_deriv = savgol_filter(spectrum, window_length=window_length, 
                                           polyorder=self.savgol_polyorder, deriv=2)
                
                # 3rd derivative
                third_deriv = savgol_filter(spectrum, window_length=window_length, 
                                          polyorder=self.savgol_polyorder, deriv=3)
                
                # Statistical features from each derivative
                for deriv_order, deriv_data in enumerate([first_deriv, second_deriv, third_deriv], 1):
                    stats = [
                        np.mean(deriv_data),
                        np.std(deriv_data),
                        np.max(deriv_data),
                        np.min(deriv_data),
                        np.median(deriv_data),
                        np.var(deriv_data),
                        np.sum(np.abs(deriv_data)),
                        np.percentile(deriv_data, 25),
                        np.percentile(deriv_data, 75),
                        np.ptp(deriv_data),  # peak-to-peak
                        len(np.where(np.diff(np.sign(deriv_data)))[0]),  # zero crossings
                        np.sum(deriv_data**2)  # energy
                    ]
                    spectrum_derivatives.extend(stats)
                    
                    # Feature names (only create once)
                    if len(feature_names) == len(spectrum_derivatives) - len(stats):
                        stat_names = ['mean', 'std', 'max', 'min', 'median', 'var', 
                                    'sum_abs', 'q25', 'q75', 'range', 'zero_cross', 'energy']
                        for stat_name in stat_names:
                            feature_names.append(f'savgol_{deriv_order}d_{stat_name}')
                
            except Exception as e:
                print(f"Warning: Savitzky-Golay derivatives failed: {str(e)}")
                # Fallback to zeros
                spectrum_derivatives.extend([0.0] * 36)  # 12 stats * 3 derivatives
                if len(feature_names) == 0:
                    stat_names = ['mean', 'std', 'max', 'min', 'median', 'var', 
                                'sum_abs', 'q25', 'q75', 'range', 'zero_cross', 'energy']
                    for deriv_order in [1, 2, 3]:
                        for stat_name in stat_names:
                            feature_names.append(f'savgol_{deriv_order}d_{stat_name}')
            
            derivatives.append(spectrum_derivatives)
        
        derivatives = np.array(derivatives)
        if len(derivatives) > 0:
            print(f"✓ Savitzky-Golay derivatives extracted: {derivatives.shape[1]} features")
        
        return derivatives, feature_names
    
    def apply_gaussian_derivatives(self, spectra):
        """Extract Gaussian smoothed derivatives"""
        if not self.enable_gaussian_derivatives:
            return np.array([]), []
        
        print("Extracting Gaussian derivatives...")
        
        derivatives = []
        feature_names = []
        
        for spectrum in spectra:
            spectrum_derivatives = []
            
            try:
                # Gaussian smoothing
                smoothed = gaussian_filter1d(spectrum, sigma=self.gaussian_sigma)
                
                # 1st derivative using numerical gradient
                first_deriv = np.gradient(smoothed)
                
                # 2nd derivative
                second_deriv = np.gradient(first_deriv)
                
                # Statistical features from each derivative
                for deriv_order, deriv_data in enumerate([first_deriv, second_deriv], 1):
                    stats = [
                        np.mean(deriv_data),
                        np.std(deriv_data),
                        np.max(deriv_data),
                        np.min(deriv_data),
                        np.median(deriv_data),
                        np.sum(np.abs(deriv_data)),
                        np.percentile(deriv_data, 25),
                        np.percentile(deriv_data, 75),
                        np.ptp(deriv_data),
                        np.sum(deriv_data**2)
                    ]
                    spectrum_derivatives.extend(stats)
                    
                    # Feature names (only create once)
                    if len(feature_names) == len(spectrum_derivatives) - len(stats):
                        stat_names = ['mean', 'std', 'max', 'min', 'median', 
                                    'sum_abs', 'q25', 'q75', 'range', 'energy']
                        for stat_name in stat_names:
                            feature_names.append(f'gaussian_{deriv_order}d_{stat_name}')
                
            except Exception as e:
                print(f"Warning: Gaussian derivatives failed: {str(e)}")
                # Fallback to zeros
                spectrum_derivatives.extend([0.0] * 20)  # 10 stats * 2 derivatives
                if len(feature_names) == 0:
                    stat_names = ['mean', 'std', 'max', 'min', 'median', 
                                'sum_abs', 'q25', 'q75', 'range', 'energy']
                    for deriv_order in [1, 2]:
                        for stat_name in stat_names:
                            feature_names.append(f'gaussian_{deriv_order}d_{stat_name}')
            
            derivatives.append(spectrum_derivatives)
        
        derivatives = np.array(derivatives)
        if len(derivatives) > 0:
            print(f"✓ Gaussian derivatives extracted: {derivatives.shape[1]} features")
        
        return derivatives, feature_names
    
    def apply_numerical_derivatives(self, spectra):
        """Extract numerical derivatives with different methods"""
        if not self.enable_numerical_derivatives:
            return np.array([]), []
        
        print("Extracting numerical derivatives...")
        
        derivatives = []
        feature_names = []
        
        for spectrum in spectra:
            spectrum_derivatives = []
            
            try:
                # Forward difference
                forward_diff = np.diff(spectrum, prepend=spectrum[0])
                
                # Backward difference  
                backward_diff = np.diff(spectrum, append=spectrum[-1])
                
                # Central difference
                central_diff = np.gradient(spectrum)
                
                # Statistical features from each method
                for method_name, deriv_data in [('forward', forward_diff), 
                                              ('backward', backward_diff),
                                              ('central', central_diff)]:
                    stats = [
                        np.mean(deriv_data),
                        np.std(deriv_data),
                        np.max(deriv_data),
                        np.min(deriv_data),
                        np.sum(np.abs(deriv_data)),
                        np.percentile(deriv_data, 25),
                        np.percentile(deriv_data, 75),
                        np.sum(deriv_data**2)
                    ]
                    spectrum_derivatives.extend(stats)
                    
                    # Feature names (only create once)
                    if len(feature_names) == len(spectrum_derivatives) - len(stats):
                        stat_names = ['mean', 'std', 'max', 'min', 'sum_abs', 'q25', 'q75', 'energy']
                        for stat_name in stat_names:
                            feature_names.append(f'numerical_{method_name}_{stat_name}')
                
            except Exception as e:
                print(f"Warning: Numerical derivatives failed: {str(e)}")
                # Fallback to zeros
                spectrum_derivatives.extend([0.0] * 24)  # 8 stats * 3 methods
                if len(feature_names) == 0:
                    stat_names = ['mean', 'std', 'max', 'min', 'sum_abs', 'q25', 'q75', 'energy']
                    for method_name in ['forward', 'backward', 'central']:
                        for stat_name in stat_names:
                            feature_names.append(f'numerical_{method_name}_{stat_name}')
            
            derivatives.append(spectrum_derivatives)
        
        derivatives = np.array(derivatives)
        if len(derivatives) > 0:
            print(f"✓ Numerical derivatives extracted: {derivatives.shape[1]} features")
        
        return derivatives, feature_names
    
    def apply_finite_differences(self, spectra):
        """Extract finite difference derivatives with multiple orders"""
        if not self.enable_finite_differences:
            return np.array([]), []
        
        print("Extracting finite difference derivatives...")
        
        derivatives = []
        feature_names = []
        
        for spectrum in spectra:
            spectrum_derivatives = []
            
            try:
                # 1st order finite difference
                first_fd = np.diff(spectrum, n=1)
                
                # 2nd order finite difference
                second_fd = np.diff(spectrum, n=2)
                
                # Higher order differences (if possible)
                if len(spectrum) > 3:
                    third_fd = np.diff(spectrum, n=3)
                else:
                    third_fd = np.array([0.0])
                
                # Statistical features from each order
                for order, deriv_data in enumerate([first_fd, second_fd, third_fd], 1):
                    if len(deriv_data) > 0:
                        stats = [
                            np.mean(deriv_data),
                            np.std(deriv_data),
                            np.max(deriv_data),
                            np.min(deriv_data),
                            np.sum(np.abs(deriv_data)),
                            np.var(deriv_data),
                            np.sum(deriv_data**2)
                        ]
                    else:
                        stats = [0.0] * 7
                    
                    spectrum_derivatives.extend(stats)
                    
                    # Feature names (only create once)
                    if len(feature_names) == len(spectrum_derivatives) - len(stats):
                        stat_names = ['mean', 'std', 'max', 'min', 'sum_abs', 'var', 'energy']
                        for stat_name in stat_names:
                            feature_names.append(f'finite_diff_{order}o_{stat_name}')
                
            except Exception as e:
                print(f"Warning: Finite differences failed: {str(e)}")
                # Fallback to zeros
                spectrum_derivatives.extend([0.0] * 21)  # 7 stats * 3 orders
                if len(feature_names) == 0:
                    stat_names = ['mean', 'std', 'max', 'min', 'sum_abs', 'var', 'energy']
                    for order in [1, 2, 3]:
                        for stat_name in stat_names:
                            feature_names.append(f'finite_diff_{order}o_{stat_name}')
            
            derivatives.append(spectrum_derivatives)
        
        derivatives = np.array(derivatives)
        if len(derivatives) > 0:
            print(f"✓ Finite difference derivatives extracted: {derivatives.shape[1]} features")
        
        return derivatives, feature_names
    
    def extract_spectral_features(self, spectra, wavelengths):
        """Extract basic spectral features for baseline comparison"""
        print("Extracting spectral features...")
        
        spectral_features = []
        feature_names = []
        
        for spectrum in spectra:
            features = [
                # Basic statistics
                np.mean(spectrum),
                np.std(spectrum),
                np.max(spectrum),
                np.min(spectrum),
                np.median(spectrum),
                np.var(spectrum),
                np.ptp(spectrum),
                
                # Percentiles
                np.percentile(spectrum, 10),
                np.percentile(spectrum, 25),
                np.percentile(spectrum, 75),
                np.percentile(spectrum, 90),
                
                # Advanced metrics
                np.sum(spectrum),
                np.sum(spectrum**2),
                np.argmax(spectrum),
                np.argmin(spectrum),
                len(np.where(np.diff(np.sign(spectrum - np.mean(spectrum))))[0])  # crossings
            ]
            spectral_features.append(features)
            
            # Feature names (only create once)
            if len(feature_names) == 0:
                feature_names = [
                    'spec_mean', 'spec_std', 'spec_max', 'spec_min', 'spec_median', 
                    'spec_var', 'spec_range', 'spec_p10', 'spec_p25', 'spec_p75', 
                    'spec_p90', 'spec_sum', 'spec_energy', 'spec_argmax', 
                    'spec_argmin', 'spec_crossings'
                ]
        
        spectral_features = np.array(spectral_features)
        print(f"✓ Spectral features extracted: {spectral_features.shape[1]} features")
        
        return spectral_features, feature_names
    
    def prepare_conformal_split(self, X, y, calibration_ratio=0.2):
        """Prepare data splits for conformal prediction"""
        print(f"Preparing conformal prediction splits...")
        
        from sklearn.model_selection import train_test_split
        
        # Split into training and calibration sets
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=calibration_ratio, random_state=42, stratify=y
        )
        
        print(f"✓ Conformal splits created:")
        print(f"  - Training: {X_train.shape}")
        print(f"  - Calibration: {X_cal.shape}")
        
        return X_train, X_cal, y_train, y_cal
    
    def preprocess_mortality_data(self, X_raw, y, wavelengths):
        """Main preprocessing pipeline for M8"""
        print("\n" + "="*60)
        print("M8 PREPROCESSING: MULTI-DERIVATIVES + CONFORMAL")
        print("="*60)
        
        print(f"Input data shape: {X_raw.shape}")
        print(f"Wavelength range: {wavelengths.min():.2f} - {wavelengths.max():.2f} nm")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Store original data stats
        self.preprocessing_stats = {
            'n_samples': X_raw.shape[0],
            'n_wavelengths': X_raw.shape[1],
            'wavelength_range': f"{wavelengths.min():.2f} - {wavelengths.max():.2f} nm",
            'original_shape': X_raw.shape,
            'preprocessing_method': 'Multi-Derivatives + Conformal',
            'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
        }
        
        # Extract all derivative features
        savgol_features, savgol_names = self.apply_savgol_derivatives(X_raw)
        gaussian_features, gaussian_names = self.apply_gaussian_derivatives(X_raw)
        numerical_features, numerical_names = self.apply_numerical_derivatives(X_raw)
        finite_diff_features, finite_diff_names = self.apply_finite_differences(X_raw)
        spectral_features, spectral_names = self.extract_spectral_features(X_raw, wavelengths)
        
        # Combine all features
        print("\nCombining all feature types...")
        feature_sets = [spectral_features]
        feature_name_sets = [spectral_names]
        
        if len(savgol_features) > 0:
            feature_sets.append(savgol_features)
            feature_name_sets.append(savgol_names)
        
        if len(gaussian_features) > 0:
            feature_sets.append(gaussian_features)
            feature_name_sets.append(gaussian_names)
        
        if len(numerical_features) > 0:
            feature_sets.append(numerical_features)
            feature_name_sets.append(numerical_names)
        
        if len(finite_diff_features) > 0:
            feature_sets.append(finite_diff_features)
            feature_name_sets.append(finite_diff_names)
        
        X_combined = np.hstack(feature_sets)
        all_feature_names = []
        for name_set in feature_name_sets:
            all_feature_names.extend(name_set)
        
        print(f"✓ Combined features: {X_combined.shape}")
        
        # Apply scaling
        print("Applying feature scaling...")
        X_scaled = self.scaler.fit_transform(X_combined)
        print(f"✓ Features scaled using {'RobustScaler' if self.use_robust_scaling else 'StandardScaler'}")
        
        # Prepare conformal prediction splits
        X_train, X_cal, y_train, y_cal = self.prepare_conformal_split(X_scaled, y)
        
        # Store final preprocessing stats
        self.preprocessing_stats.update({
            'n_enhanced_features': X_combined.shape[1],
            'feature_names': all_feature_names,
            'enhancement_ratio': X_combined.shape[1] / X_raw.shape[1],
            'savgol_features': len(savgol_names),
            'gaussian_features': len(gaussian_names),
            'numerical_features': len(numerical_names),
            'finite_diff_features': len(finite_diff_names),
            'spectral_features': len(spectral_names),
            'scaler_type': 'RobustScaler' if self.use_robust_scaling else 'StandardScaler',
            'conformal_alpha': self.conformal_alpha,
            'calibration_ratio': 0.2,
            'train_samples': len(X_train),
            'calibration_samples': len(X_cal)
        })
        
        print(f"\n✓ M8 preprocessing completed:")
        print(f"  - Original features: {X_raw.shape[1]}")
        print(f"  - Enhanced features: {X_combined.shape[1]}")
        print(f"  - Enhancement ratio: {X_combined.shape[1]/X_raw.shape[1]:.2f}x")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Calibration samples: {len(X_cal)}")
        
        return X_train, X_cal, y_train, y_cal
    
    def transform_new_data(self, X_new, wavelengths):
        """Transform new data using fitted preprocessors"""
        print(f"Transforming new data: {X_new.shape}")
        
        # Apply same preprocessing steps
        savgol_features, _ = self.apply_savgol_derivatives(X_new)
        gaussian_features, _ = self.apply_gaussian_derivatives(X_new)
        numerical_features, _ = self.apply_numerical_derivatives(X_new)
        finite_diff_features, _ = self.apply_finite_differences(X_new)
        spectral_features, _ = self.extract_spectral_features(X_new, wavelengths)
        
        # Combine features
        feature_sets = [spectral_features]
        if len(savgol_features) > 0:
            feature_sets.append(savgol_features)
        if len(gaussian_features) > 0:
            feature_sets.append(gaussian_features)
        if len(numerical_features) > 0:
            feature_sets.append(numerical_features)
        if len(finite_diff_features) > 0:
            feature_sets.append(finite_diff_features)
        
        X_combined = np.hstack(feature_sets)
        
        # Apply scaling
        X_scaled = self.scaler.transform(X_combined)
        
        print(f"✓ New data transformed: {X_scaled.shape}")
        return X_scaled

def main():
    """Example usage"""
    print("M8 Multi-Derivative Conformal Preprocessor")
    print("Usage: Import and use with raw spectral data")
    
    # Example parameters
    print("\nPreprocessing Configuration:")
    print("- Savitzky-Golay derivatives (1st, 2nd, 3rd order)")
    print("- Gaussian smoothed derivatives")
    print("- Numerical derivatives (forward, backward, central)")
    print("- Finite difference derivatives")
    print("- Conformal prediction data splits")
    print("- Robust scaling for outlier resistance")

if __name__ == "__main__":
    main() 