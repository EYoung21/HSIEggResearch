import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from scipy.signal import savgol_filter
from scipy import stats
from scipy.linalg import pinv
import warnings
warnings.filterwarnings('ignore')

class EMSCSavitzkyGolayPreprocessor:
    """
    Advanced preprocessing combining EMSC and Savitzky-Golay for Bayesian neural networks
    Extended Multiplicative Scatter Correction (EMSC) + derivatives + feature optimization
    """
    
    def __init__(self, window_length=15, polyorder=2, deriv_orders=[0, 1, 2], emsc_order=2):
        """
        Initialize EMSC + Savitzky-Golay preprocessor
        
        Args:
            window_length: Window length for SG filtering
            polyorder: Polynomial order for SG filtering
            deriv_orders: Derivative orders to compute [0, 1, 2]
            emsc_order: Polynomial order for EMSC correction
        """
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv_orders = deriv_orders
        self.emsc_order = emsc_order
        
        # EMSC reference spectrum and correction matrix
        self.reference_spectrum = None
        self.emsc_correction_matrix = None
        
        # Scalers for different feature types
        self.main_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # Feature selectors
        self.feature_selector = None
        self.selected_features = None
        
        # Preprocessing metadata
        self.wavelength_range = None
        self.n_original_features = None
        
    def compute_emsc_reference(self, spectra):
        """
        Compute EMSC reference spectrum (median spectrum)
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Reference spectrum for EMSC correction
        """
        print("Computing EMSC reference spectrum...")
        
        # Use median spectrum as reference (more robust than mean)
        reference = np.median(spectra, axis=0)
        
        # Smooth the reference spectrum
        reference_smoothed = savgol_filter(reference, self.window_length, self.polyorder)
        
        print(f"✓ EMSC reference computed from {spectra.shape[0]} spectra")
        return reference_smoothed
    
    def create_emsc_correction_matrix(self, wavelengths, reference_spectrum):
        """
        Create EMSC correction matrix for polynomial baseline correction
        
        Args:
            wavelengths: Wavelength array
            reference_spectrum: Reference spectrum for EMSC
            
        Returns:
            EMSC correction matrix
        """
        print("Creating EMSC correction matrix...")
        
        n_wavelengths = len(wavelengths)
        
        # Normalize wavelengths to [-1, 1] for numerical stability
        norm_wavelengths = 2 * (wavelengths - np.min(wavelengths)) / (np.max(wavelengths) - np.min(wavelengths)) - 1
        
        # Create polynomial terms for baseline correction
        # Columns: [reference, constant, linear, quadratic, ...]
        n_terms = self.emsc_order + 2  # reference + polynomial terms
        correction_matrix = np.zeros((n_wavelengths, n_terms))
        
        # Reference spectrum (multiplicative term)
        correction_matrix[:, 0] = reference_spectrum
        
        # Polynomial terms (additive baseline)
        for order in range(self.emsc_order + 1):
            correction_matrix[:, order + 1] = norm_wavelengths ** order
        
        # Handle NaN and inf values
        correction_matrix = np.nan_to_num(correction_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        
        print(f"✓ EMSC matrix created: {correction_matrix.shape} ({n_terms} correction terms)")
        return correction_matrix
    
    def apply_emsc_correction(self, spectra):
        """
        Apply Extended Multiplicative Scatter Correction
        Removes both multiplicative scattering and additive baseline effects
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            EMSC corrected spectra
        """
        print("Applying EMSC (Extended Multiplicative Scatter Correction)...")
        
        n_samples, n_wavelengths = spectra.shape
        corrected_spectra = np.zeros_like(spectra)
        
        # Use pseudoinverse for robust solving
        correction_matrix_pinv = pinv(self.emsc_correction_matrix)
        
        for i, spectrum in enumerate(spectra):
            try:
                # Solve for correction coefficients: spectrum = matrix @ coefficients
                coefficients = correction_matrix_pinv @ spectrum
                
                # Extract multiplicative coefficient (first coefficient)
                multiplicative_coeff = coefficients[0]
                
                # Extract additive polynomial coefficients
                additive_baseline = self.emsc_correction_matrix[:, 1:] @ coefficients[1:]
                
                # Apply EMSC correction
                if abs(multiplicative_coeff) > 1e-10:  # Avoid division by zero
                    corrected_spectra[i] = (spectrum - additive_baseline) / multiplicative_coeff
                else:
                    # Fallback: just remove baseline
                    corrected_spectra[i] = spectrum - additive_baseline
                    
            except Exception as e:
                print(f"Warning: EMSC failed for spectrum {i}, using original: {e}")
                corrected_spectra[i] = spectrum
        
        print(f"✓ EMSC correction applied to {n_samples} spectra")
        return corrected_spectra
    
    def apply_savitzky_golay_derivatives(self, spectra):
        """
        Apply Savitzky-Golay filtering with multiple derivatives
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Combined SG features (original + derivatives)
        """
        print(f"Applying Savitzky-Golay derivatives (orders: {self.deriv_orders})...")
        
        sg_features_list = []
        
        for deriv_order in self.deriv_orders:
            print(f"  Computing {deriv_order}-order derivative...")
            
            if deriv_order == 0:
                # Original smoothed spectra
                sg_features = np.array([
                    savgol_filter(spectrum, self.window_length, self.polyorder, deriv=0)
                    for spectrum in spectra
                ])
            else:
                # Derivatives
                sg_features = np.array([
                    savgol_filter(spectrum, self.window_length, self.polyorder, deriv=deriv_order)
                    for spectrum in spectra
                ])
            
            sg_features_list.append(sg_features)
        
        # Combine all derivative orders
        combined_sg_features = np.concatenate(sg_features_list, axis=1)
        
        # Handle NaN values
        combined_sg_features = np.nan_to_num(combined_sg_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"✓ SG derivatives completed: {combined_sg_features.shape}")
        return combined_sg_features
    
    def create_advanced_spectral_features(self, spectra):
        """
        Create advanced spectral features for enhanced discrimination
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Advanced feature matrix
        """
        print("Creating advanced spectral features...")
        
        advanced_features = []
        
        for spectrum in spectra:
            # Basic statistical features
            basic_stats = [
                np.mean(spectrum),
                np.std(spectrum),
                np.max(spectrum),
                np.min(spectrum),
                np.median(spectrum),
                np.ptp(spectrum),  # Peak-to-peak
                stats.skew(spectrum),
                stats.kurtosis(spectrum)
            ]
            
            # Percentiles
            percentiles = np.percentile(spectrum, [5, 10, 25, 75, 90, 95])
            
            # Spectral moments
            x = np.arange(len(spectrum))
            total_intensity = np.sum(spectrum)
            
            if total_intensity > 0:
                # Centroid (first moment)
                centroid = np.sum(x * spectrum) / total_intensity
                
                # Spread (second moment)
                spread = np.sqrt(np.sum((x - centroid)**2 * spectrum) / total_intensity)
                
                # Skewness (third moment)
                spectral_skew = np.sum((x - centroid)**3 * spectrum) / (total_intensity * spread**3)
                
                # Kurtosis (fourth moment)
                spectral_kurt = np.sum((x - centroid)**4 * spectrum) / (total_intensity * spread**4) - 3
            else:
                centroid = spread = spectral_skew = spectral_kurt = 0
            
            spectral_moments = [centroid, spread, spectral_skew, spectral_kurt]
            
            # Peak characteristics
            peak_indices = []
            for i in range(1, len(spectrum) - 1):
                if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                    peak_indices.append(i)
            
            peak_features = [
                len(peak_indices),  # Number of peaks
                np.argmax(spectrum),  # Main peak position
                np.max(spectrum),  # Main peak intensity
                np.sum(spectrum > np.mean(spectrum)),  # Above-average points
            ]
            
            # Spectral ratios (important for chemical analysis)
            n_points = len(spectrum)
            third = n_points // 3
            
            ratio_features = [
                np.mean(spectrum[:third]) / (np.mean(spectrum[third:2*third]) + 1e-10),  # Low/mid ratio
                np.mean(spectrum[2*third:]) / (np.mean(spectrum[third:2*third]) + 1e-10),  # High/mid ratio
                np.mean(spectrum[:third]) / (np.mean(spectrum[2*third:]) + 1e-10),  # Low/high ratio
            ]
            
            # Combine all advanced features
            all_features = basic_stats + percentiles.tolist() + spectral_moments + peak_features + ratio_features
            advanced_features.append(all_features)
        
        advanced_features = np.array(advanced_features)
        
        # Handle NaN values
        advanced_features = np.nan_to_num(advanced_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"✓ Advanced features created: {advanced_features.shape}")
        return advanced_features
    
    def optimize_feature_selection(self, X, y, selection_methods=['mutual_info', 'f_classif'], n_features_range=(100, 500)):
        """
        Optimize feature selection using multiple methods and cross-validation
        
        Args:
            X: Feature matrix
            y: Target labels
            selection_methods: List of selection methods to try
            n_features_range: Range of features to test
            
        Returns:
            Best selector and number of features
        """
        print(f"Optimizing feature selection for {X.shape[1]} features...")
        
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        min_features, max_features = n_features_range
        max_possible = min(X.shape[1], max_features)
        
        # Adjust range based on available features
        if max_possible < min_features:
            feature_counts = [max_possible]
        else:
            step_size = max(1, (max_possible - min_features) // 10)
            feature_counts = list(range(min_features, max_possible + 1, step_size))
        
        best_score = -1
        best_selector = None
        best_n_features = feature_counts[0]
        best_method = selection_methods[0]
        
        # Test different selection methods
        for method in selection_methods:
            print(f"  Testing {method} feature selection...")
            
            if method == 'mutual_info':
                score_func = mutual_info_classif
            elif method == 'f_classif':
                score_func = f_classif
            else:
                continue
            
            # Test different numbers of features
            for n_feat in feature_counts:
                print(f"    Testing {n_feat} features with {method}...")
                
                try:
                    selector = SelectKBest(score_func=score_func, k=n_feat)
                    X_selected = selector.fit_transform(X, y)
                    
                    # Quick evaluation with Random Forest
                    rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
                    cv_scores = cross_val_score(rf, X_selected, y, cv=3, scoring='accuracy')
                    mean_score = np.mean(cv_scores)
                    
                    print(f"      CV Score: {mean_score:.4f}")
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_n_features = n_feat
                        best_selector = selector
                        best_method = method
                        
                except Exception as e:
                    print(f"      Failed: {e}")
        
        print(f"✓ Best selection: {best_method} with {best_n_features} features (CV: {best_score:.4f})")
        return best_selector, best_n_features, best_method
    
    def preprocess_for_bayesian_nn(self, spectra, labels, wavelengths):
        """
        Complete preprocessing pipeline for Bayesian neural network training
        
        Args:
            spectra: Raw spectral data
            labels: Target labels
            wavelengths: Wavelength array
            
        Returns:
            Processed features and metadata
        """
        print("\n" + "="*60)
        print("EMSC + SAVITZKY-GOLAY PREPROCESSING FOR BAYESIAN NN")
        print("="*60)
        
        self.wavelength_range = (wavelengths.min(), wavelengths.max())
        self.n_original_features = spectra.shape[1]
        
        # Step 1: Compute EMSC reference spectrum
        self.reference_spectrum = self.compute_emsc_reference(spectra)
        
        # Step 2: Create EMSC correction matrix
        self.emsc_correction_matrix = self.create_emsc_correction_matrix(
            wavelengths, self.reference_spectrum
        )
        
        # Step 3: Apply EMSC correction
        emsc_corrected_spectra = self.apply_emsc_correction(spectra)
        
        # Step 4: Apply Savitzky-Golay derivatives
        sg_features = self.apply_savitzky_golay_derivatives(emsc_corrected_spectra)
        
        # Step 5: Create advanced spectral features
        advanced_features = self.create_advanced_spectral_features(emsc_corrected_spectra)
        
        # Step 6: Combine all features
        combined_features = np.concatenate([sg_features, advanced_features], axis=1)
        
        print(f"\nFeature combination:")
        print(f"  - SG derivatives: {sg_features.shape[1]} features")
        print(f"  - Advanced spectral: {advanced_features.shape[1]} features")
        print(f"  - Total combined: {combined_features.shape[1]} features")
        
        # Step 7: Optimize feature selection
        best_selector, n_features, method = self.optimize_feature_selection(
            combined_features, labels, n_features_range=(50, min(300, combined_features.shape[1]))
        )
        
        self.feature_selector = best_selector
        selected_features = best_selector.fit_transform(combined_features, labels)
        
        # Step 8: Apply scaling
        print(f"\nApplying scaling...")
        
        # Standard scaling for main features
        main_scaled = self.main_scaler.fit_transform(selected_features)
        
        # Robust scaling for additional robustness
        robust_scaled = self.robust_scaler.fit_transform(selected_features)
        
        # Create multi-scale features for enhanced Bayesian learning
        multi_scale_features = np.concatenate([main_scaled, robust_scaled], axis=1)
        
        print(f"✓ Multi-scale features: {multi_scale_features.shape}")
        
        # Store preprocessing metadata
        preprocessing_info = {
            'emsc_order': self.emsc_order,
            'sg_window_length': self.window_length,
            'sg_polyorder': self.polyorder,
            'derivative_orders': self.deriv_orders,
            'n_original_features': self.n_original_features,
            'n_selected_features': n_features,
            'n_final_features': multi_scale_features.shape[1],
            'feature_selection_method': method,
            'wavelength_range': self.wavelength_range,
            'scaling_methods': ['StandardScaler', 'RobustScaler']
        }
        
        return multi_scale_features, preprocessing_info
    
    def transform_new_data(self, spectra, wavelengths):
        """
        Transform new spectral data using fitted preprocessors
        
        Args:
            spectra: New spectral data
            wavelengths: Wavelength array
            
        Returns:
            Transformed features
        """
        # Apply EMSC correction
        emsc_corrected = self.apply_emsc_correction(spectra)
        
        # Apply SG derivatives
        sg_features = self.apply_savitzky_golay_derivatives(emsc_corrected)
        
        # Create advanced features
        advanced_features = self.create_advanced_spectral_features(emsc_corrected)
        
        # Combine features
        combined_features = np.concatenate([sg_features, advanced_features], axis=1)
        
        # Apply feature selection
        selected_features = self.feature_selector.transform(combined_features)
        
        # Apply scaling
        main_scaled = self.main_scaler.transform(selected_features)
        robust_scaled = self.robust_scaler.transform(selected_features)
        
        # Create multi-scale features
        multi_scale_features = np.concatenate([main_scaled, robust_scaled], axis=1)
        
        return multi_scale_features

def load_and_merge_data(day='D0'):
    """Load and merge spectral and metadata"""
    print(f"Loading data for Day {day}...")
    
    ref_df = pd.read_csv('../../data/reference_metadata.csv')
    spectral_df = pd.read_csv(f'../../data/spectral_data_{day}.csv')
    merged_df = pd.merge(ref_df, spectral_df, on='HSI sample ID', how='inner')
    
    print(f"✓ Merged dataset shape: {merged_df.shape}")
    return merged_df

def prepare_bayesian_dataset(merged_df):
    """Prepare dataset for Bayesian neural network training"""
    print("\n" + "="*60)
    print("PREPARING BAYESIAN NN DATASET - G8 EXPERIMENT")
    print("="*60)
    
    # Filter for gender prediction
    gender_df = merged_df[merged_df['Gender'].isin(['Male', 'Female'])].copy()
    print(f"Samples with gender labels: {len(gender_df)}")
    
    # Class distribution
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
    
    # Extract features and labels
    X = gender_df[wavelength_cols].values
    y = gender_df['Gender'].values
    wavelengths = np.array([float(col) for col in wavelength_cols])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
    print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    return X, y_encoded, label_encoder, wavelengths, gender_df

def main():
    """Main preprocessing pipeline for G8 Bayesian neural network"""
    print("="*80)
    print("G8 EXPERIMENT: SG + EMSC + Bayesian Neural Network")
    print("="*80)
    
    # Load data
    merged_df = load_and_merge_data(day='D0')
    X, y, label_encoder, wavelengths, gender_df = prepare_bayesian_dataset(merged_df)
    
    # Stratified split for robust evaluation
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"Training set: {X_train_raw.shape[0]} samples")
    print(f"Test set: {X_test_raw.shape[0]} samples")
    
    # Initialize preprocessor
    preprocessor = EMSCSavitzkyGolayPreprocessor(
        window_length=15,
        polyorder=2,
        deriv_orders=[0, 1, 2],  # Original + 1st + 2nd derivatives
        emsc_order=2  # Quadratic baseline correction
    )
    
    # Preprocess training data
    X_train_processed, preprocessing_info = preprocessor.preprocess_for_bayesian_nn(
        X_train_raw, y_train, wavelengths
    )
    
    # Transform test data
    print("\nTransforming test data...")
    X_test_processed = preprocessor.transform_new_data(X_test_raw, wavelengths)
    
    # Save processed data
    print("\nSaving processed data...")
    
    import joblib
    import json
    
    # Save feature arrays
    np.save('X_train_processed.npy', X_train_processed)
    np.save('X_test_processed.npy', X_test_processed)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save preprocessor and encoders
    joblib.dump(preprocessor, 'emsc_sg_preprocessor.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    # Save preprocessing information
    with open('preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2, default=str)
    
    # Summary
    print("\n" + "="*60)
    print("BAYESIAN NN PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Original features: {X.shape[1]} wavelengths")
    print(f"Processed features: {X_train_processed.shape[1]}")
    print(f"Training samples: {X_train_processed.shape[0]}")
    print(f"Test samples: {X_test_processed.shape[0]}")
    print(f"Classes: {len(label_encoder.classes_)} - {label_encoder.classes_}")
    
    print(f"\nPreprocessing details:")
    for key, value in preprocessing_info.items():
        print(f"  - {key}: {value}")
    
    print(f"\nFiles generated:")
    print(f"  - X_train_processed.npy: {X_train_processed.shape}")
    print(f"  - X_test_processed.npy: {X_test_processed.shape}")
    print(f"  - Preprocessor and metadata files")
    
    print("\nReady for Bayesian neural network training!")

if __name__ == "__main__":
    main() 