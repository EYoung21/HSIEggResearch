import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.signal import savgol_filter
from scipy import stats
import pywt
import warnings
warnings.filterwarnings('ignore')

class WaveletSNVEnsemblePreprocessor:
    """Advanced dual-path preprocessing for voting ensemble"""
    
    def __init__(self, window_length=15, polyorder=2, wavelet='db4', decomp_levels=3):
        self.window_length = window_length
        self.polyorder = polyorder
        self.wavelet = wavelet
        self.decomp_levels = decomp_levels
        
        # Scalers for different preprocessing paths
        self.snv_scaler = StandardScaler()
        self.wavelet_scaler = StandardScaler()
        self.combined_scaler = StandardScaler()
        
        # Feature selectors
        self.snv_selector = None
        self.wavelet_selector = None
        self.combined_selector = None
        
    def apply_snv_normalization(self, spectra):
        """Apply Standard Normal Variate normalization"""
        print("Applying SNV normalization...")
        
        snv_corrected = np.zeros_like(spectra)
        
        for i, spectrum in enumerate(spectra):
            mean_spectrum = np.mean(spectrum)
            std_spectrum = np.std(spectrum)
            
            if std_spectrum > 1e-10:
                snv_corrected[i] = (spectrum - mean_spectrum) / std_spectrum
            else:
                snv_corrected[i] = spectrum - mean_spectrum
        
        print(f"✓ SNV normalization applied to {spectra.shape[0]} spectra")
        return snv_corrected
    
    def apply_wavelet_decomposition(self, spectra):
        """Apply wavelet decomposition for multi-resolution analysis"""
        print(f"Applying wavelet decomposition (wavelet={self.wavelet}, levels={self.decomp_levels})...")
        
        wavelet_features = []
        
        for spectrum in spectra:
            try:
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(spectrum, self.wavelet, level=self.decomp_levels)
                
                # Extract features from coefficients
                features = []
                
                # Approximation coefficients (low-frequency)
                approx = coeffs[0]
                features.extend([
                    np.mean(approx),
                    np.std(approx),
                    np.max(approx),
                    np.min(approx),
                    np.median(approx),
                    stats.skew(approx),
                    stats.kurtosis(approx),
                    np.ptp(approx)
                ])
                
                # Detail coefficients (high-frequency)
                for level, detail in enumerate(coeffs[1:], 1):
                    features.extend([
                        np.mean(detail),
                        np.std(detail),
                        np.max(detail),
                        np.min(detail),
                        np.sum(detail**2),  # Energy
                        stats.skew(detail),
                        stats.kurtosis(detail)
                    ])
                
                # Energy distribution across levels
                total_energy = sum(np.sum(coeff**2) for coeff in coeffs)
                if total_energy > 0:
                    energy_ratios = [np.sum(coeff**2) / total_energy for coeff in coeffs]
                    features.extend(energy_ratios)
                else:
                    features.extend([0] * len(coeffs))
                
                wavelet_features.append(features)
                
            except Exception as e:
                print(f"Warning: Wavelet decomposition failed, using zeros: {e}")
                # Fallback: create zero features
                n_features = 8 + 7 * self.decomp_levels + (self.decomp_levels + 1)
                wavelet_features.append([0] * n_features)
        
        wavelet_features = np.array(wavelet_features)
        wavelet_features = np.nan_to_num(wavelet_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"✓ Wavelet decomposition completed: {wavelet_features.shape}")
        return wavelet_features
    
    def apply_savitzky_golay_preprocessing(self, spectra):
        """Apply Savitzky-Golay smoothing and derivatives"""
        print("Applying Savitzky-Golay preprocessing...")
        
        # Original smoothed spectra
        smoothed = np.array([
            savgol_filter(spectrum, self.window_length, self.polyorder)
            for spectrum in spectra
        ])
        
        # 1st derivative
        derivative_1st = np.array([
            savgol_filter(spectrum, self.window_length, self.polyorder, deriv=1)
            for spectrum in spectra
        ])
        
        # 2nd derivative  
        derivative_2nd = np.array([
            savgol_filter(spectrum, self.window_length, self.polyorder, deriv=2)
            for spectrum in spectra
        ])
        
        # Combine features
        sg_features = np.concatenate([smoothed, derivative_1st, derivative_2nd], axis=1)
        
        # Handle NaN values
        sg_features = np.nan_to_num(sg_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"✓ SG preprocessing completed: {sg_features.shape}")
        return sg_features
    
    def create_statistical_features(self, spectra):
        """Create comprehensive statistical features"""
        print("Creating statistical features...")
        
        statistical_features = []
        
        for spectrum in spectra:
            # Basic statistics
            stats_basic = [
                np.mean(spectrum),
                np.std(spectrum),
                np.max(spectrum),
                np.min(spectrum),
                np.median(spectrum),
                np.ptp(spectrum),
                stats.skew(spectrum),
                stats.kurtosis(spectrum)
            ]
            
            # Percentiles
            percentiles = np.percentile(spectrum, [10, 25, 75, 90])
            
            # Spectral characteristics
            spectral_stats = [
                np.argmax(spectrum),  # Peak position
                np.argmin(spectrum),  # Valley position
                np.sum(spectrum > np.mean(spectrum)),  # Above-average count
                np.sum(spectrum < np.mean(spectrum)),  # Below-average count
                np.sum(np.diff(spectrum) > 0),  # Increasing segments
                np.sum(np.diff(spectrum) < 0),  # Decreasing segments
            ]
            
            # Combine all statistical features
            all_stats = stats_basic + percentiles.tolist() + spectral_stats
            statistical_features.append(all_stats)
        
        statistical_features = np.array(statistical_features)
        statistical_features = np.nan_to_num(statistical_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"✓ Statistical features created: {statistical_features.shape}")
        return statistical_features
    
    def optimize_feature_selection(self, X, y, n_features_range=(50, 200)):
        """Optimize feature selection using cross-validation"""
        print(f"Optimizing feature selection for {X.shape[1]} features...")
        
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        min_features, max_features = n_features_range
        max_possible = min(X.shape[1], max_features)
        feature_counts = list(range(min_features, max_possible + 1, 25))
        
        if X.shape[1] < min_features:
            feature_counts = [X.shape[1]]
        
        best_score = -1
        best_n_features = feature_counts[0]
        best_selector = None
        
        for n_feat in feature_counts:
            print(f"  Testing {n_feat} features...")
            
            selector = SelectKBest(score_func=mutual_info_classif, k=n_feat)
            X_selected = selector.fit_transform(X, y)
            
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            cv_scores = cross_val_score(rf, X_selected, y, cv=3, scoring='accuracy')
            mean_score = np.mean(cv_scores)
            
            print(f"    CV Score: {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_n_features = n_feat
                best_selector = selector
        
        print(f"✓ Optimal features: {best_n_features} (CV score: {best_score:.4f})")
        return best_n_features, best_selector
    
    def create_ensemble_feature_sets(self, spectra, labels):
        """Create multiple feature sets for ensemble voting"""
        print("\n" + "="*50)
        print("CREATING ENSEMBLE FEATURE SETS")
        print("="*50)
        
        feature_sets = {}
        
        # Path 1: SNV + SG preprocessing
        print("\n--- SNV + Savitzky-Golay Path ---")
        snv_spectra = self.apply_snv_normalization(spectra)
        sg_features = self.apply_savitzky_golay_preprocessing(snv_spectra)
        
        # Optimize SNV+SG features
        n_snv_features, snv_selector = self.optimize_feature_selection(sg_features, labels)
        self.snv_selector = snv_selector
        snv_features_selected = snv_selector.fit_transform(sg_features, labels)
        snv_features_scaled = self.snv_scaler.fit_transform(snv_features_selected)
        
        feature_sets['snv_sg'] = {
            'features': snv_features_scaled,
            'name': 'SNV + Savitzky-Golay',
            'n_features': n_snv_features,
            'preprocessing': 'SNV normalization + SG derivatives'
        }
        
        # Path 2: Wavelet decomposition
        print("\n--- Wavelet Decomposition Path ---")
        wavelet_features = self.apply_wavelet_decomposition(spectra)
        
        # Optimize wavelet features
        n_wavelet_features, wavelet_selector = self.optimize_feature_selection(
            wavelet_features, labels, n_features_range=(10, 50)
        )
        self.wavelet_selector = wavelet_selector
        wavelet_features_selected = wavelet_selector.fit_transform(wavelet_features, labels)
        wavelet_features_scaled = self.wavelet_scaler.fit_transform(wavelet_features_selected)
        
        feature_sets['wavelet'] = {
            'features': wavelet_features_scaled,
            'name': 'Wavelet Decomposition',
            'n_features': n_wavelet_features,
            'preprocessing': f'{self.wavelet} wavelet, {self.decomp_levels} levels'
        }
        
        # Path 3: Combined approach
        print("\n--- Combined Features Path ---")
        statistical_features = self.create_statistical_features(spectra)
        
        # Combine all feature types
        combined_features = np.concatenate([
            sg_features,
            wavelet_features,
            statistical_features
        ], axis=1)
        
        # Optimize combined features
        n_combined_features, combined_selector = self.optimize_feature_selection(
            combined_features, labels, n_features_range=(100, 300)
        )
        self.combined_selector = combined_selector
        combined_features_selected = combined_selector.fit_transform(combined_features, labels)
        combined_features_scaled = self.combined_scaler.fit_transform(combined_features_selected)
        
        feature_sets['combined'] = {
            'features': combined_features_scaled,
            'name': 'Combined Features',
            'n_features': n_combined_features,
            'preprocessing': 'SNV+SG + Wavelets + Statistical'
        }
        
        # Path 4: Multi-scale normalized features
        print("\n--- Multi-scale Normalization Path ---")
        minmax_scaler = MinMaxScaler()
        robust_scaler = RobustScaler()
        
        snv_minmax = minmax_scaler.fit_transform(snv_features_selected)
        snv_robust = robust_scaler.fit_transform(snv_features_selected)
        
        # Store additional scalers
        self.minmax_scaler = minmax_scaler
        self.robust_scaler = robust_scaler
        
        feature_sets['snv_minmax'] = {
            'features': snv_minmax,
            'name': 'SNV + MinMax Scaling',
            'n_features': n_snv_features,
            'preprocessing': 'SNV + SG + MinMax normalization'
        }
        
        feature_sets['snv_robust'] = {
            'features': snv_robust,
            'name': 'SNV + Robust Scaling',
            'n_features': n_snv_features,
            'preprocessing': 'SNV + SG + Robust scaling'
        }
        
        print(f"\n✓ Created {len(feature_sets)} feature sets for ensemble voting")
        
        # Store feature set metadata
        self.feature_sets_info = {
            name: {
                'n_features': info['n_features'],
                'preprocessing': info['preprocessing']
            }
            for name, info in feature_sets.items()
        }
        
        return feature_sets
    
    def transform_new_data(self, spectra):
        """Transform new spectral data using fitted preprocessors"""
        feature_sets = {}
        
        # SNV + SG path
        snv_spectra = self.apply_snv_normalization(spectra)
        sg_features = self.apply_savitzky_golay_preprocessing(snv_spectra)
        snv_features_selected = self.snv_selector.transform(sg_features)
        snv_features_scaled = self.snv_scaler.transform(snv_features_selected)
        
        feature_sets['snv_sg'] = snv_features_scaled
        feature_sets['snv_minmax'] = self.minmax_scaler.transform(snv_features_selected)
        feature_sets['snv_robust'] = self.robust_scaler.transform(snv_features_selected)
        
        # Wavelet path
        wavelet_features = self.apply_wavelet_decomposition(spectra)
        wavelet_features_selected = self.wavelet_selector.transform(wavelet_features)
        wavelet_features_scaled = self.wavelet_scaler.transform(wavelet_features_selected)
        
        feature_sets['wavelet'] = wavelet_features_scaled
        
        # Combined path
        statistical_features = self.create_statistical_features(spectra)
        combined_features = np.concatenate([
            sg_features, wavelet_features, statistical_features
        ], axis=1)
        combined_features_selected = self.combined_selector.transform(combined_features)
        combined_features_scaled = self.combined_scaler.transform(combined_features_selected)
        
        feature_sets['combined'] = combined_features_scaled
        
        return feature_sets

def load_and_merge_data(day='D0'):
    """Load and merge data"""
    print(f"Loading data for Day {day}...")
    
    ref_df = pd.read_csv('../../data/reference_metadata.csv')
    spectral_df = pd.read_csv(f'../../data/spectral_data_{day}.csv')
    merged_df = pd.merge(ref_df, spectral_df, on='HSI sample ID', how='inner')
    
    print(f"✓ Merged dataset shape: {merged_df.shape}")
    return merged_df

def prepare_voting_ensemble_dataset(merged_df):
    """Prepare dataset for voting ensemble"""
    print("\n" + "="*50)
    print("PREPARING VOTING ENSEMBLE DATASET - G7 EXPERIMENT")
    print("="*50)
    
    gender_df = merged_df[merged_df['Gender'].isin(['Male', 'Female'])].copy()
    print(f"Samples with gender labels: {len(gender_df)}")
    
    # Gender distribution
    gender_counts = gender_df['Gender'].value_counts()
    print(f"Gender distribution: {gender_counts.to_dict()}")
    
    # Extract wavelength features
    metadata_cols = ['HSI sample ID', 'Date_x', 'Date_y', 'Exp. No._x', 'Exp. No._y', 
                    'Gender', 'Fertility status', 'Mortality status', 'Mass (g)', 
                    'Major dia. (mm)', 'Minor dia. (mm)', 'Comment']
    
    wavelength_cols = []
    for col in gender_df.columns:
        if col not in metadata_cols:
            try:
                float(col)
                wavelength_cols.append(col)
            except ValueError:
                continue
    
    print(f"Found {len(wavelength_cols)} wavelength features")
    
    X = gender_df[wavelength_cols].values
    y = gender_df['Gender'].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Feature matrix shape: {X.shape}")
    return X, y_encoded, label_encoder, wavelength_cols, gender_df

def main():
    """Main preprocessing pipeline"""
    print("="*70)
    print("G7 EXPERIMENT: Wavelet + SNV + Voting Ensemble")
    print("="*70)
    
    # Load data
    merged_df = load_and_merge_data(day='D0')
    X, y, label_encoder, wavelength_cols, gender_df = prepare_voting_ensemble_dataset(merged_df)
    
    # Stratified split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train_raw.shape[0]} samples")
    print(f"Test set: {X_test_raw.shape[0]} samples")
    
    # Initialize preprocessor
    print("\n" + "="*50)
    print("WAVELET + SNV ENSEMBLE PREPROCESSING")
    print("="*50)
    
    preprocessor = WaveletSNVEnsemblePreprocessor(
        window_length=15,
        polyorder=2,
        wavelet='db4',
        decomp_levels=3
    )
    
    # Create ensemble feature sets
    train_feature_sets = preprocessor.create_ensemble_feature_sets(X_train_raw, y_train)
    
    # Transform test data
    print("\nTransforming test data...")
    test_feature_sets = preprocessor.transform_new_data(X_test_raw)
    
    # Save all feature sets
    print("\nSaving ensemble feature sets...")
    
    import joblib
    
    # Save training feature sets
    for name, feature_data in train_feature_sets.items():
        np.save(f'X_train_{name}.npy', feature_data['features'])
        print(f"✓ Saved X_train_{name}.npy ({feature_data['features'].shape})")
    
    # Save test feature sets
    for name, features in test_feature_sets.items():
        np.save(f'X_test_{name}.npy', features)
        print(f"✓ Saved X_test_{name}.npy ({features.shape})")
    
    # Save labels
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save preprocessor and encoders
    joblib.dump(preprocessor, 'wavelet_snv_ensemble_preprocessor.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    # Save feature set information
    feature_sets_info = {
        'feature_sets': preprocessor.feature_sets_info,
        'total_sets': len(train_feature_sets),
        'preprocessing_methods': [
            'SNV + Savitzky-Golay',
            'Wavelet Decomposition',
            'Combined Features',
            'Multi-scale Normalization'
        ],
        'wavelet_config': {
            'wavelet': preprocessor.wavelet,
            'decomp_levels': preprocessor.decomp_levels
        }
    }
    
    import json
    with open('ensemble_feature_info.json', 'w') as f:
        json.dump(feature_sets_info, f, indent=2, default=str)
    
    # Summary
    print("\n" + "="*50)
    print("VOTING ENSEMBLE PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Original features: {X.shape[1]} wavelengths")
    print(f"Feature sets created: {len(train_feature_sets)}")
    print(f"Training samples: {X_train_raw.shape[0]}")
    print(f"Test samples: {X_test_raw.shape[0]}")
    
    print("\nFeature Set Details:")
    for name, info in train_feature_sets.items():
        print(f"  - {info['name']}: {info['n_features']} features")
    
    print("\nReady for voting ensemble models!")

if __name__ == "__main__":
    main() 