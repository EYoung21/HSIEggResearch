"""
M7 Preprocessing: SNV + Mixup + Semi-Supervised for Mortality Classification
Advanced preprocessing combining Standard Normal Variate with mixup augmentation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class SNVMixupPreprocessor:
    """M7 Preprocessor: SNV + Mixup augmentation for semi-supervised learning"""
    
    def __init__(self, 
                 enable_derivatives=True, 
                 derivative_window=15, 
                 derivative_order=2,
                 mixup_alpha=0.2,
                 augmentation_ratio=0.5,
                 semi_supervised_ratio=0.3):
        
        self.enable_derivatives = enable_derivatives
        self.derivative_window = derivative_window
        self.derivative_order = derivative_order
        self.mixup_alpha = mixup_alpha
        self.augmentation_ratio = augmentation_ratio
        self.semi_supervised_ratio = semi_supervised_ratio
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.preprocessing_stats = {}
        
    def apply_snv_normalization(self, spectra):
        """Apply Standard Normal Variate (SNV) normalization"""
        print("Applying SNV (Standard Normal Variate) normalization...")
        
        snv_normalized = np.zeros_like(spectra)
        
        for i, spectrum in enumerate(spectra):
            # Calculate mean and standard deviation for each spectrum
            mean = np.mean(spectrum)
            std = np.std(spectrum)
            
            # Avoid division by zero
            if std < 1e-10:
                std = 1.0
            
            # SNV normalization: (spectrum - mean) / std
            snv_normalized[i] = (spectrum - mean) / std
        
        print(f"✓ SNV normalization applied: {snv_normalized.shape}")
        return snv_normalized
    
    def extract_derivative_features(self, spectra):
        """Extract derivative features using Savitzky-Golay filter"""
        if not self.enable_derivatives:
            return np.array([]), []
        
        print("Extracting derivative features...")
        
        derivative_features = []
        feature_names = []
        
        for spectrum in spectra:
            spectrum_features = []
            
            # 1st derivative
            try:
                window_length = min(self.derivative_window, len(spectrum)//4*2+1)
                first_deriv = savgol_filter(spectrum, window_length=window_length, 
                                          polyorder=self.derivative_order, deriv=1)
            except:
                first_deriv = np.gradient(spectrum)
            
            # 2nd derivative
            try:
                second_deriv = savgol_filter(spectrum, window_length=window_length, 
                                           polyorder=self.derivative_order, deriv=2)
            except:
                second_deriv = np.gradient(first_deriv)
            
            # Statistical features from derivatives
            derivative_stats = [
                # 1st derivative features
                np.mean(first_deriv),
                np.std(first_deriv),
                np.max(first_deriv),
                np.min(first_deriv),
                np.sum(np.abs(first_deriv)),
                np.percentile(first_deriv, 25),
                np.percentile(first_deriv, 75),
                
                # 2nd derivative features
                np.mean(second_deriv),
                np.std(second_deriv),
                np.max(second_deriv),
                np.min(second_deriv),
                np.sum(np.abs(second_deriv)),
                np.percentile(second_deriv, 25),
                np.percentile(second_deriv, 75)
            ]
            
            spectrum_features.extend(derivative_stats)
            derivative_features.append(spectrum_features)
            
            # Feature names (only create once)
            if len(feature_names) == 0:
                deriv_names = ['mean', 'std', 'max', 'min', 'sum_abs', 'q25', 'q75']
                for deriv_order in ['1st', '2nd']:
                    for stat_name in deriv_names:
                        feature_names.append(f'deriv_{deriv_order}_{stat_name}')
        
        derivative_features = np.array(derivative_features)
        if len(derivative_features) > 0:
            print(f"✓ Derivative features extracted: {derivative_features.shape[1]} features")
        
        return derivative_features, feature_names
    
    def extract_spectral_regions(self, spectra, wavelengths):
        """Extract features from specific spectral regions"""
        print("Extracting spectral region features...")
        
        # Define biologically relevant spectral regions
        regions = {
            'blue': (400, 500),      # Carotenoids, pigments
            'green': (500, 600),     # Chlorophyll, proteins
            'red': (600, 700),       # Chlorophyll, blood
            'nir1': (700, 850),      # Water, proteins
            'nir2': (850, 1000),     # Lipids, water
            'swir': (1000, 1050)     # Moisture content
        }
        
        region_features = []
        feature_names = []
        
        for spectrum in spectra:
            spectrum_features = []
            
            for region_name, (start_wl, end_wl) in regions.items():
                # Find wavelength indices for this region
                region_mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
                
                if np.any(region_mask):
                    region_spectrum = spectrum[region_mask]
                    
                    # Calculate regional features
                    regional_stats = [
                        np.mean(region_spectrum),          # Mean intensity
                        np.std(region_spectrum),           # Variability
                        np.max(region_spectrum),           # Peak intensity
                        np.min(region_spectrum),           # Minimum intensity
                        np.sum(region_spectrum),           # Total area
                        np.argmax(region_spectrum),        # Peak location
                        np.ptp(region_spectrum),           # Range
                        np.median(region_spectrum)         # Median
                    ]
                    
                    spectrum_features.extend(regional_stats)
                    
                    # Feature names (only create once)
                    if len(feature_names) == len(spectrum_features) - len(regional_stats):
                        stat_names = ['mean', 'std', 'max', 'min', 'sum', 'peak_loc', 'range', 'median']
                        for stat_name in stat_names:
                            feature_names.append(f'region_{region_name}_{stat_name}')
                else:
                    # If region not found, add zeros
                    spectrum_features.extend([0.0] * 8)
                    if len(feature_names) == len(spectrum_features) - 8:
                        stat_names = ['mean', 'std', 'max', 'min', 'sum', 'peak_loc', 'range', 'median']
                        for stat_name in stat_names:
                            feature_names.append(f'region_{region_name}_{stat_name}')
            
            region_features.append(spectrum_features)
        
        region_features = np.array(region_features)
        print(f"✓ Regional features extracted: {region_features.shape[1]} features from {len(regions)} regions")
        
        return region_features, feature_names
    
    def extract_spectral_ratios(self, spectra, wavelengths):
        """Extract spectral ratio features for biological interpretation"""
        print("Extracting spectral ratio features...")
        
        # Define key wavelengths for ratios (approximate)
        key_wavelengths = {
            'blue_450': 450,
            'green_550': 550,
            'red_650': 650,
            'red_edge_720': 720,
            'nir_800': 800,
            'nir_900': 900,
            'water_980': 980
        }
        
        ratio_features = []
        feature_names = []
        
        for spectrum in spectra:
            spectrum_features = []
            
            # Find intensities at key wavelengths
            intensities = {}
            for name, target_wl in key_wavelengths.items():
                # Find closest wavelength
                closest_idx = np.argmin(np.abs(wavelengths - target_wl))
                intensities[name] = spectrum[closest_idx]
            
            # Calculate meaningful ratios
            ratios = [
                # Normalized difference indices
                (intensities['nir_800'] - intensities['red_650']) / (intensities['nir_800'] + intensities['red_650']),  # NDVI-like
                (intensities['nir_900'] - intensities['water_980']) / (intensities['nir_900'] + intensities['water_980']),  # Water content
                intensities['red_650'] / intensities['nir_800'],     # Red/NIR
                intensities['blue_450'] / intensities['green_550'],  # Blue/Green
                intensities['green_550'] / intensities['red_650'],   # Green/Red
                intensities['red_edge_720'] / intensities['red_650'], # Red edge position
                (intensities['nir_800'] + intensities['red_650']) / 2, # Overall intensity
                intensities['nir_900'] / intensities['blue_450'],    # NIR/Blue
            ]
            
            # Handle division by zero and invalid values
            ratios = [ratio if np.isfinite(ratio) else 0.0 for ratio in ratios]
            spectrum_features.extend(ratios)
            
            # Feature names (only create once)
            if len(feature_names) == 0:
                feature_names = [
                    'ndvi_like', 'water_index', 'red_nir_ratio', 'blue_green_ratio',
                    'green_red_ratio', 'red_edge_ratio', 'overall_intensity', 'nir_blue_ratio'
                ]
            
            ratio_features.append(spectrum_features)
        
        ratio_features = np.array(ratio_features)
        print(f"✓ Spectral ratios extracted: {ratio_features.shape[1]} ratio features")
        
        return ratio_features, feature_names
    
    def generate_mixup_samples(self, X, y, num_samples):
        """Generate synthetic samples using Mixup data augmentation"""
        print(f"Generating {num_samples} mixup samples...")
        
        if len(X) < 2:
            print("Warning: Not enough samples for mixup. Returning original data.")
            return X, y
        
        mixup_X = []
        mixup_y = []
        
        for _ in range(num_samples):
            # Randomly select two samples
            idx1, idx2 = np.random.choice(len(X), 2, replace=False)
            
            # Sample mixing coefficient from Beta distribution
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            
            # Mix the samples
            mixed_x = lam * X[idx1] + (1 - lam) * X[idx2]
            
            # For classification, we'll use the label of the dominant sample
            mixed_y = y[idx1] if lam > 0.5 else y[idx2]
            
            mixup_X.append(mixed_x)
            mixup_y.append(mixed_y)
        
        mixup_X = np.array(mixup_X)
        mixup_y = np.array(mixup_y)
        
        print(f"✓ Mixup samples generated: {mixup_X.shape}")
        return mixup_X, mixup_y
    
    def create_semi_supervised_split(self, X, y):
        """Create labeled and unlabeled splits for semi-supervised learning"""
        print(f"Creating semi-supervised data split...")
        
        # Randomly select samples to keep labeled
        n_samples = len(X)
        n_labeled = int(n_samples * (1 - self.semi_supervised_ratio))
        
        # Stratified sampling to maintain class balance in labeled set
        unique_classes = np.unique(y)
        labeled_indices = []
        
        for class_label in unique_classes:
            class_indices = np.where(y == class_label)[0]
            n_class_labeled = int(len(class_indices) * (1 - self.semi_supervised_ratio))
            class_labeled_indices = np.random.choice(class_indices, n_class_labeled, replace=False)
            labeled_indices.extend(class_labeled_indices)
        
        labeled_indices = np.array(labeled_indices)
        unlabeled_indices = np.setdiff1d(np.arange(n_samples), labeled_indices)
        
        X_labeled = X[labeled_indices]
        y_labeled = y[labeled_indices]
        X_unlabeled = X[unlabeled_indices]
        
        print(f"✓ Semi-supervised split created:")
        print(f"  - Labeled samples: {len(X_labeled)} ({len(X_labeled)/n_samples:.1%})")
        print(f"  - Unlabeled samples: {len(X_unlabeled)} ({len(X_unlabeled)/n_samples:.1%})")
        
        return X_labeled, y_labeled, X_unlabeled, labeled_indices, unlabeled_indices
    
    def preprocess_mortality_data(self, X_raw, y, wavelengths):
        """Main preprocessing pipeline for M7"""
        print("\n" + "="*60)
        print("M7 PREPROCESSING: SNV + MIXUP + SEMI-SUPERVISED")
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
            'preprocessing_method': 'SNV + Mixup + Semi-Supervised',
            'class_distribution': dict(zip(*np.unique(y, return_counts=True)))
        }
        
        # 1. Apply SNV normalization
        X_snv = self.apply_snv_normalization(X_raw)
        
        # 2. Extract derivative features
        derivative_features, derivative_names = self.extract_derivative_features(X_snv)
        
        # 3. Extract spectral region features
        region_features, region_names = self.extract_spectral_regions(X_snv, wavelengths)
        
        # 4. Extract spectral ratio features
        ratio_features, ratio_names = self.extract_spectral_ratios(X_snv, wavelengths)
        
        # 5. Combine all features
        print("\nCombining all feature types...")
        feature_sets = [X_snv, region_features, ratio_features]
        feature_name_sets = [
            [f'SNV_{wavelengths[i]:.2f}nm' for i in range(len(wavelengths))],
            region_names,
            ratio_names
        ]
        
        if len(derivative_features) > 0:
            feature_sets.append(derivative_features)
            feature_name_sets.append(derivative_names)
        
        X_combined = np.hstack(feature_sets)
        all_feature_names = []
        for name_set in feature_name_sets:
            all_feature_names.extend(name_set)
        
        print(f"✓ Combined features: {X_combined.shape}")
        
        # 6. Create semi-supervised split
        X_labeled, y_labeled, X_unlabeled, labeled_indices, unlabeled_indices = self.create_semi_supervised_split(X_combined, y)
        
        # 7. Apply mixup augmentation to labeled data
        if self.augmentation_ratio > 0:
            num_mixup_samples = int(len(X_labeled) * self.augmentation_ratio)
            X_mixup, y_mixup = self.generate_mixup_samples(X_labeled, y_labeled, num_mixup_samples)
            
            # Combine original labeled data with mixup samples
            X_labeled_augmented = np.vstack([X_labeled, X_mixup])
            y_labeled_augmented = np.hstack([y_labeled, y_mixup])
        else:
            X_labeled_augmented = X_labeled
            y_labeled_augmented = y_labeled
        
        print(f"✓ Augmented labeled data: {X_labeled_augmented.shape}")
        
        # 8. Final scaling
        print("Applying final feature scaling...")
        X_labeled_scaled = self.scaler.fit_transform(X_labeled_augmented)
        X_unlabeled_scaled = self.scaler.transform(X_unlabeled) if len(X_unlabeled) > 0 else np.array([])
        
        print(f"✓ Features scaled - Labeled: {X_labeled_scaled.shape}, Unlabeled: {X_unlabeled_scaled.shape}")
        
        # Store final preprocessing stats
        self.preprocessing_stats.update({
            'n_enhanced_features': X_combined.shape[1],
            'feature_names': all_feature_names,
            'enhancement_ratio': X_combined.shape[1] / X_raw.shape[1],
            'derivative_features': len(derivative_names) if len(derivative_features) > 0 else 0,
            'region_features': len(region_names),
            'ratio_features': len(ratio_names),
            'snv_applied': True,
            'mixup_samples': len(X_mixup) if self.augmentation_ratio > 0 else 0,
            'labeled_samples': len(X_labeled_scaled),
            'unlabeled_samples': len(X_unlabeled_scaled),
            'semi_supervised_ratio': self.semi_supervised_ratio,
            'final_scaling': 'StandardScaler'
        })
        
        print(f"\n✓ M7 preprocessing completed:")
        print(f"  - Original features: {X_raw.shape[1]}")
        print(f"  - Enhanced features: {X_combined.shape[1]}")
        print(f"  - Enhancement ratio: {X_combined.shape[1]/X_raw.shape[1]:.2f}x")
        print(f"  - Labeled samples: {len(X_labeled_scaled)} (including {self.preprocessing_stats['mixup_samples']} mixup)")
        print(f"  - Unlabeled samples: {len(X_unlabeled_scaled)}")
        
        return X_labeled_scaled, y_labeled_augmented, X_unlabeled_scaled, labeled_indices, unlabeled_indices
    
    def transform_new_data(self, X_new, wavelengths):
        """Transform new data using fitted preprocessors"""
        print(f"Transforming new data: {X_new.shape}")
        
        # Apply same preprocessing steps
        X_snv = self.apply_snv_normalization(X_new)
        
        derivative_features, _ = self.extract_derivative_features(X_snv)
        region_features, _ = self.extract_spectral_regions(X_snv, wavelengths)
        ratio_features, _ = self.extract_spectral_ratios(X_snv, wavelengths)
        
        # Combine features
        feature_sets = [X_snv, region_features, ratio_features]
        if len(derivative_features) > 0:
            feature_sets.append(derivative_features)
        
        X_combined = np.hstack(feature_sets)
        
        # Apply scaling
        X_scaled = self.scaler.transform(X_combined)
        
        print(f"✓ New data transformed: {X_scaled.shape}")
        return X_scaled

def main():
    """Example usage"""
    print("M7 SNV Mixup Semi-Supervised Preprocessor")
    print("Usage: Import and use with raw spectral data")
    
    # Example parameters
    print("\nPreprocessing Configuration:")
    print("- Standard Normal Variate (SNV) normalization")
    print("- Mixup data augmentation")
    print("- Semi-supervised learning setup")
    print("- Spectral region analysis")
    print("- Derivative feature extraction")
    print("- Spectral ratio calculation")

if __name__ == "__main__":
    main() 