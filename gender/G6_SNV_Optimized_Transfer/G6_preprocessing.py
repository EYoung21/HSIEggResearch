import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import savgol_filter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class OptimizedSNVPreprocessor:
    """
    Advanced SNV preprocessing with optimized wavelength selection for transfer learning
    """
    
    def __init__(self, window_length=15, polyorder=2):
        self.window_length = window_length
        self.polyorder = polyorder
        self.scaler = StandardScaler()
        self.feature_selector = None
        
    def apply_snv(self, spectra):
        """Apply Standard Normal Variate (SNV) normalization"""
        print("Applying SNV (Standard Normal Variate) normalization...")
        
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
    
    def apply_savitzky_golay_smoothing(self, spectra):
        """Apply Savitzky-Golay smoothing"""
        print("Applying Savitzky-Golay smoothing...")
        
        smoothed_spectra = np.array([
            savgol_filter(spectrum, self.window_length, self.polyorder)
            for spectrum in spectra
        ])
        
        print(f"✓ SG smoothing applied (window={self.window_length}, poly={self.polyorder})")
        return smoothed_spectra
    
    def compute_spectral_features(self, spectra):
        """Compute advanced spectral features"""
        print("Computing advanced spectral features...")
        
        # Original smoothed spectra
        original = spectra
        
        # 1st derivative
        derivative_1st = np.array([
            savgol_filter(spectrum, self.window_length, self.polyorder, deriv=1)
            for spectrum in spectra
        ])
        
        # Statistical features with NaN handling
        spectral_stats = []
        for spectrum in spectra:
            # Use nanmean, nanstd, etc. to handle NaN values
            stats_features = [
                np.nanmean(spectrum),
                np.nanstd(spectrum),
                np.nanmax(spectrum),
                np.nanmin(spectrum),
                np.nanmedian(spectrum),
                stats.skew(spectrum, nan_policy='omit'),
                stats.kurtosis(spectrum, nan_policy='omit'),
                np.nanmax(spectrum) - np.nanmin(spectrum),  # Peak-to-peak with NaN handling
            ]
            spectral_stats.append(stats_features)
        
        spectral_stats = np.array(spectral_stats)
        
        # Combine features
        combined_features = np.concatenate([
            original,
            derivative_1st,
            spectral_stats
        ], axis=1)
        
        # Replace any remaining NaN values with 0
        combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"✓ Enhanced features: {combined_features.shape}")
        return combined_features
    
    def optimize_wavelength_selection(self, X, y):
        """Optimize wavelength selection"""
        print("Optimizing wavelength selection...")
        
        feature_counts = [50, 100, 150, 200, 250, 300]
        feature_counts = [k for k in feature_counts if k <= X.shape[1]]
        
        best_score = -1
        best_n_features = feature_counts[0]
        best_selector = None
        
        from sklearn.model_selection import cross_val_score
        
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
        
        # Store the best selector
        self.feature_selector = best_selector
        print(f"✓ Optimal features: {best_n_features} (CV score: {best_score:.4f})")
        return best_n_features
    
    def create_transfer_learning_features(self, X_selected):
        """Create transfer learning optimized features"""
        print("Creating transfer learning optimized features...")
        
        # Multiple scaling approaches
        minmax_scaler = MinMaxScaler()
        X_minmax = minmax_scaler.fit_transform(X_selected)
        
        X_zscore = self.scaler.fit_transform(X_selected)
        
        from sklearn.preprocessing import RobustScaler
        robust_scaler = RobustScaler()
        X_robust = robust_scaler.fit_transform(X_selected)
        
        # Store scalers
        self.minmax_scaler = minmax_scaler
        self.robust_scaler = robust_scaler
        
        # Combine scaled features
        transfer_features = np.concatenate([X_minmax, X_zscore, X_robust], axis=1)
        
        print(f"✓ Transfer learning features: {transfer_features.shape}")
        return transfer_features
    
    def fit_transform(self, spectra, labels, wavelengths):
        """Fit and transform spectra for transfer learning"""
        print("Applying SNV + Optimized Transfer Learning preprocessing...")
        
        # SNV normalization
        snv_corrected = self.apply_snv(spectra)
        
        # Smoothing
        smoothed_spectra = self.apply_savitzky_golay_smoothing(snv_corrected)
        
        # Enhanced features
        enhanced_features = self.compute_spectral_features(smoothed_spectra)
        
        # Optimize selection
        self.optimize_wavelength_selection(enhanced_features, labels)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.fit_transform(enhanced_features, labels)
        else:
            # Fallback if no selector was found
            X_selected = enhanced_features
        
        # Transfer learning features
        transfer_features = self.create_transfer_learning_features(X_selected)
        
        print(f"✓ Final transfer learning features: {transfer_features.shape}")
        return transfer_features
    
    def transform(self, spectra):
        """Transform new spectra"""
        snv_corrected = self.apply_snv(spectra)
        smoothed_spectra = self.apply_savitzky_golay_smoothing(snv_corrected)
        enhanced_features = self.compute_spectral_features(smoothed_spectra)
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(enhanced_features)
        else:
            X_selected = enhanced_features
        
        X_minmax = self.minmax_scaler.transform(X_selected)
        X_zscore = self.scaler.transform(X_selected)
        X_robust = self.robust_scaler.transform(X_selected)
        
        transfer_features = np.concatenate([X_minmax, X_zscore, X_robust], axis=1)
        return transfer_features

def load_and_merge_data(day='D0'):
    """Load and merge data"""
    print(f"Loading data for Day {day}...")
    
    ref_df = pd.read_csv('../../data/reference_metadata.csv')
    spectral_df = pd.read_csv(f'../../data/spectral_data_{day}.csv')
    merged_df = pd.merge(ref_df, spectral_df, on='HSI sample ID', how='inner')
    
    print(f"✓ Merged dataset shape: {merged_df.shape}")
    return merged_df

def prepare_transfer_learning_dataset(merged_df):
    """Prepare dataset for transfer learning"""
    print("\n" + "="*50)
    print("PREPARING TRANSFER LEARNING DATASET - G6 EXPERIMENT")
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
    print("G6 EXPERIMENT: SNV + Optimized + Transfer Learning")
    print("="*70)
    
    merged_df = load_and_merge_data(day='D0')
    X, y, label_encoder, wavelength_cols, gender_df = prepare_transfer_learning_dataset(merged_df)
    
    # Stratified split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train_raw.shape[0]} samples")
    print(f"Test set: {X_test_raw.shape[0]} samples")
    
    # Preprocessing
    print("\n" + "="*50)
    print("APPLYING SNV + OPTIMIZED TRANSFER LEARNING PREPROCESSING")
    print("="*50)
    
    preprocessor = OptimizedSNVPreprocessor(window_length=15, polyorder=2)
    
    X_train_processed = preprocessor.fit_transform(X_train_raw, y_train, wavelength_cols)
    X_test_processed = preprocessor.transform(X_test_raw)
    
    print(f"✓ Processed training set shape: {X_train_processed.shape}")
    print(f"✓ Processed test set shape: {X_test_processed.shape}")
    
    # Save processed data
    print("\nSaving processed data...")
    np.save('X_train_processed.npy', X_train_processed)
    np.save('X_test_processed.npy', X_test_processed)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    import joblib
    joblib.dump(preprocessor, 'optimized_snv_preprocessor.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    # Feature info
    feature_info = {
        'selected_features': int(X_train_processed.shape[1] // 3),
        'transfer_learning_features': int(X_train_processed.shape[1]),
        'wavelength_cols': wavelength_cols,
        'preprocessing': 'SNV + SG Smoothing + Feature Selection + Transfer Learning Scaling'
    }
    
    import json
    with open('feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2, default=str)
    
    print("✓ Saved processed data and preprocessor")
    print(f"Training samples: {X_train_processed.shape[0]}")
    print(f"Test samples: {X_test_processed.shape[0]}")
    print(f"Classes: {len(label_encoder.classes_)} - {label_encoder.classes_}")
    print("Ready for transfer learning models!")

if __name__ == "__main__":
    main() 