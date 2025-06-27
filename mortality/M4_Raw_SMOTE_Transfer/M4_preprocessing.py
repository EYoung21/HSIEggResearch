"""
M4 Preprocessing: Raw + SMOTE + Transfer for Mortality Classification
Minimal preprocessing approach focusing on raw spectral data with transfer learning preparation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RawTransferPreprocessor:
    """M4 Preprocessor: Raw spectral data with minimal preprocessing for transfer learning"""
    
    def __init__(self, normalize_method='standard', pca_components=None, spectral_regions=True):
        self.normalize_method = normalize_method
        self.pca_components = pca_components
        self.spectral_regions = spectral_regions
        
        # Preprocessing components
        self.scaler = None
        self.pca = None
        self.preprocessing_stats = {}
        
        # Initialize scaler
        if normalize_method == 'standard':
            self.scaler = StandardScaler()
        elif normalize_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif normalize_method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError("normalize_method must be 'standard', 'minmax', or 'robust'")
    
    def define_spectral_regions(self, wavelengths):
        """Define biological spectral regions for analysis"""
        regions = {
            'UV': (wavelengths >= 300) & (wavelengths < 400),
            'Blue': (wavelengths >= 400) & (wavelengths < 500),
            'Green': (wavelengths >= 500) & (wavelengths < 600),
            'Red': (wavelengths >= 600) & (wavelengths < 700),
            'NIR1': (wavelengths >= 700) & (wavelengths < 850),
            'NIR2': (wavelengths >= 850) & (wavelengths <= 1100)
        }
        
        return regions
    
    def extract_statistical_features(self, X):
        """Extract statistical features from raw spectra"""
        print("Extracting statistical features from raw spectra...")
        
        statistical_features = []
        
        for spectrum in X:
            features = {
                'mean': np.mean(spectrum),
                'std': np.std(spectrum),
                'min': np.min(spectrum),
                'max': np.max(spectrum),
                'median': np.median(spectrum),
                'q25': np.percentile(spectrum, 25),
                'q75': np.percentile(spectrum, 75),
                'iqr': np.percentile(spectrum, 75) - np.percentile(spectrum, 25),
                'skewness': stats.skew(spectrum),
                'kurtosis': stats.kurtosis(spectrum),
                'peak_to_peak': np.ptp(spectrum),
                'rms': np.sqrt(np.mean(spectrum**2)),
                'energy': np.sum(spectrum**2),
                'zero_crossings': np.sum(np.diff(np.signbit(spectrum - np.mean(spectrum)))),
                'slope': np.polyfit(range(len(spectrum)), spectrum, 1)[0]
            }
            statistical_features.append(list(features.values()))
        
        statistical_features = np.array(statistical_features)
        feature_names = list(features.keys())
        
        print(f"✓ Statistical features extracted: {statistical_features.shape[1]} features")
        return statistical_features, feature_names
    
    def extract_regional_features(self, X, wavelengths):
        """Extract features from specific spectral regions"""
        print("Extracting spectral regional features...")
        
        regions = self.define_spectral_regions(wavelengths)
        regional_features = []
        feature_names = []
        
        for spectrum in X:
            region_stats = []
            
            for region_name, mask in regions.items():
                if np.any(mask):
                    region_data = spectrum[mask]
                    
                    # Basic statistics for each region
                    stats = [
                        np.mean(region_data),
                        np.std(region_data),
                        np.max(region_data),
                        np.min(region_data),
                        np.sum(region_data)
                    ]
                    region_stats.extend(stats)
                    
                    # Feature names (only on first iteration)
                    if len(feature_names) < len(regions) * 5:
                        for stat_name in ['mean', 'std', 'max', 'min', 'sum']:
                            feature_names.append(f'{region_name}_{stat_name}')
            
            regional_features.append(region_stats)
        
        regional_features = np.array(regional_features)
        print(f"✓ Regional features extracted: {regional_features.shape[1]} features")
        return regional_features, feature_names
    
    def extract_spectral_ratios(self, X, wavelengths):
        """Extract meaningful spectral ratios for biological interpretation"""
        print("Extracting spectral ratios...")
        
        regions = self.define_spectral_regions(wavelengths)
        ratio_features = []
        feature_names = []
        
        for spectrum in X:
            ratios = []
            
            # Calculate regional means
            region_means = {}
            for region_name, mask in regions.items():
                if np.any(mask):
                    region_means[region_name] = np.mean(spectrum[mask])
            
            # Calculate biologically meaningful ratios
            if 'Red' in region_means and 'NIR1' in region_means and region_means['Red'] > 0:
                ratios.append(region_means['NIR1'] / region_means['Red'])  # NDVI-like
                if len(feature_names) < 1: feature_names.append('NIR1_Red_ratio')
            else:
                ratios.append(0)
                if len(feature_names) < 1: feature_names.append('NIR1_Red_ratio')
            
            if 'Blue' in region_means and 'Green' in region_means and region_means['Green'] > 0:
                ratios.append(region_means['Blue'] / region_means['Green'])  # Blue/Green
                if len(feature_names) < 2: feature_names.append('Blue_Green_ratio')
            else:
                ratios.append(0)
                if len(feature_names) < 2: feature_names.append('Blue_Green_ratio')
            
            if 'NIR2' in region_means and 'NIR1' in region_means and region_means['NIR1'] > 0:
                ratios.append(region_means['NIR2'] / region_means['NIR1'])  # NIR2/NIR1
                if len(feature_names) < 3: feature_names.append('NIR2_NIR1_ratio')
            else:
                ratios.append(0)
                if len(feature_names) < 3: feature_names.append('NIR2_NIR1_ratio')
            
            if 'Red' in region_means and 'Green' in region_means and region_means['Green'] > 0:
                ratios.append(region_means['Red'] / region_means['Green'])  # Red/Green
                if len(feature_names) < 4: feature_names.append('Red_Green_ratio')
            else:
                ratios.append(0)
                if len(feature_names) < 4: feature_names.append('Red_Green_ratio')
            
            if 'NIR1' in region_means and 'Blue' in region_means and region_means['Blue'] > 0:
                ratios.append(region_means['NIR1'] / region_means['Blue'])  # NIR/Blue
                if len(feature_names) < 5: feature_names.append('NIR1_Blue_ratio')
            else:
                ratios.append(0)
                if len(feature_names) < 5: feature_names.append('NIR1_Blue_ratio')
            
            ratio_features.append(ratios)
        
        ratio_features = np.array(ratio_features)
        print(f"✓ Spectral ratios extracted: {ratio_features.shape[1]} features")
        return ratio_features, feature_names
    
    def apply_pca_transformation(self, X_combined):
        """Apply PCA for dimensionality reduction if specified"""
        if self.pca_components is not None:
            print(f"Applying PCA transformation (components: {self.pca_components})...")
            
            self.pca = PCA(n_components=self.pca_components, random_state=42)
            X_pca = self.pca.fit_transform(X_combined)
            
            explained_variance = np.sum(self.pca.explained_variance_ratio_)
            print(f"✓ PCA applied: {X_pca.shape[1]} components, {explained_variance:.3f} variance explained")
            
            # Create PCA feature names
            pca_names = [f'PCA_{i+1}' for i in range(self.pca_components)]
            
            return X_pca, pca_names
        
        return X_combined, None
    
    def preprocess_mortality_data(self, X_raw, wavelengths):
        """Main preprocessing pipeline for M4"""
        print("\n" + "="*60)
        print("M4 PREPROCESSING: RAW + MINIMAL PROCESSING")
        print("="*60)
        
        print(f"Input data shape: {X_raw.shape}")
        print(f"Wavelength range: {wavelengths.min():.2f} - {wavelengths.max():.2f} nm")
        
        # Store original data stats
        self.preprocessing_stats = {
            'n_samples': X_raw.shape[0],
            'n_wavelengths': X_raw.shape[1],
            'wavelength_range': f"{wavelengths.min():.2f} - {wavelengths.max():.2f} nm",
            'original_shape': X_raw.shape,
            'preprocessing_method': 'Raw spectral data with minimal processing'
        }
        
        # 1. Basic normalization of raw spectra
        print(f"\nApplying {self.normalize_method} normalization...")
        X_normalized = self.scaler.fit_transform(X_raw)
        print(f"✓ Raw spectra normalized: {X_normalized.shape}")
        
        # 2. Extract statistical features
        statistical_features, stat_names = self.extract_statistical_features(X_normalized)
        
        # 3. Extract regional features if enabled
        regional_features, regional_names = [], []
        if self.spectral_regions:
            regional_features, regional_names = self.extract_regional_features(X_normalized, wavelengths)
        
        # 4. Extract spectral ratios
        ratio_features, ratio_names = self.extract_spectral_ratios(X_normalized, wavelengths)
        
        # 5. Combine all features
        print("\nCombining all feature types...")
        feature_sets = [X_normalized, statistical_features, ratio_features]
        feature_name_sets = [
            [f'Raw_{wavelengths[i]:.2f}nm' for i in range(len(wavelengths))],
            [f'Stat_{name}' for name in stat_names],
            [f'Ratio_{name}' for name in ratio_names]
        ]
        
        if len(regional_features) > 0:
            feature_sets.append(regional_features)
            feature_name_sets.append([f'Region_{name}' for name in regional_names])
        
        X_combined = np.hstack(feature_sets)
        all_feature_names = []
        for name_set in feature_name_sets:
            all_feature_names.extend(name_set)
        
        print(f"✓ Combined features: {X_combined.shape}")
        
        # 6. Apply PCA if specified
        X_final, pca_names = self.apply_pca_transformation(X_combined)
        
        if pca_names is not None:
            final_feature_names = pca_names
        else:
            final_feature_names = all_feature_names
        
        # Store final preprocessing stats
        self.preprocessing_stats.update({
            'n_enhanced_features': X_final.shape[1],
            'feature_names': final_feature_names,
            'enhancement_ratio': X_final.shape[1] / X_raw.shape[1],
            'statistical_features': len(stat_names),
            'regional_features': len(regional_names),
            'ratio_features': len(ratio_names),
            'pca_applied': self.pca_components is not None,
            'scaler_type': self.normalize_method
        })
        
        print(f"\n✓ M4 preprocessing completed:")
        print(f"  - Original features: {X_raw.shape[1]}")
        print(f"  - Enhanced features: {X_final.shape[1]}")
        print(f"  - Enhancement ratio: {X_final.shape[1]/X_raw.shape[1]:.2f}x")
        print(f"  - Normalization: {self.normalize_method}")
        if self.pca_components:
            print(f"  - PCA components: {self.pca_components}")
        
        return X_final
    
    def transform_new_data(self, X_new, wavelengths):
        """Transform new data using fitted preprocessors"""
        print(f"Transforming new data: {X_new.shape}")
        
        # Apply same normalization
        X_normalized = self.scaler.transform(X_new)
        
        # Extract same features
        statistical_features, _ = self.extract_statistical_features(X_normalized)
        
        regional_features = []
        if self.spectral_regions:
            regional_features, _ = self.extract_regional_features(X_normalized, wavelengths)
        
        ratio_features, _ = self.extract_spectral_ratios(X_normalized, wavelengths)
        
        # Combine features
        feature_sets = [X_normalized, statistical_features, ratio_features]
        if len(regional_features) > 0:
            feature_sets.append(regional_features)
        
        X_combined = np.hstack(feature_sets)
        
        # Apply PCA if fitted
        if self.pca is not None:
            X_final = self.pca.transform(X_combined)
        else:
            X_final = X_combined
        
        print(f"✓ New data transformed: {X_final.shape}")
        return X_final

def main():
    """Example usage"""
    print("M4 Raw Transfer Preprocessor")
    print("Usage: Import and use with raw spectral data")
    
    # Example parameters
    print("\nPreprocessing Configuration:")
    print("- Raw spectral data with minimal processing")
    print("- Statistical feature extraction")
    print("- Spectral regional analysis")
    print("- Biologically meaningful ratios")
    print("- Optional PCA transformation")
    print("- Standard/MinMax/Robust scaling options")

if __name__ == "__main__":
    main() 