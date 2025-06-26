import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy.signal import savgol_filter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MSCSavitzkyGolayPreprocessor:
    """
    MSC + Savitzky-Golay preprocessing for mortality classification
    Multiplicative Scatter Correction followed by SG filtering for enhanced mortality prediction
    """
    
    def __init__(self, window_length=15, polyorder=2, apply_derivatives=True):
        """
        Initialize MSC + Savitzky-Golay preprocessor
        
        Args:
            window_length: Window length for SG filtering
            polyorder: Polynomial order for SG filtering  
            apply_derivatives: Whether to compute derivatives
        """
        self.window_length = window_length
        self.polyorder = polyorder
        self.apply_derivatives = apply_derivatives
        
        # MSC reference spectrum
        self.reference_spectrum = None
        
        # Scalers
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        # Feature selector
        self.feature_selector = None
        
        # Preprocessing metadata
        self.wavelength_range = None
        self.n_original_features = None
        
    def compute_msc_reference(self, spectra):
        """
        Compute MSC reference spectrum (mean spectrum)
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Reference spectrum for MSC correction
        """
        print("Computing MSC reference spectrum...")
        
        # Use mean spectrum as reference for MSC
        reference = np.mean(spectra, axis=0)
        
        print(f"✓ MSC reference computed from {spectra.shape[0]} spectra")
        return reference
    
    def apply_msc_correction(self, spectra):
        """
        Apply Multiplicative Scatter Correction
        Removes multiplicative scattering effects
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            MSC corrected spectra
        """
        print("Applying MSC (Multiplicative Scatter Correction)...")
        
        n_samples, n_wavelengths = spectra.shape
        corrected_spectra = np.zeros_like(spectra)
        
        for i, spectrum in enumerate(spectra):
            # Linear regression: spectrum = a * reference + b
            try:
                # Add constant term for offset
                X = np.column_stack([self.reference_spectrum, np.ones(n_wavelengths)])
                coefficients = np.linalg.lstsq(X, spectrum, rcond=None)[0]
                
                # Extract slope and intercept
                slope, intercept = coefficients[0], coefficients[1]
                
                # Apply MSC correction
                if abs(slope) > 1e-10:  # Avoid division by zero
                    corrected_spectra[i] = (spectrum - intercept) / slope
                else:
                    corrected_spectra[i] = spectrum - intercept
                    
            except Exception as e:
                print(f"Warning: MSC failed for spectrum {i}, using original: {e}")
                corrected_spectra[i] = spectrum
        
        print(f"✓ MSC correction applied to {n_samples} spectra")
        return corrected_spectra
    
    def apply_savitzky_golay_filtering(self, spectra):
        """
        Apply Savitzky-Golay filtering with optional derivatives
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            SG filtered features
        """
        print(f"Applying Savitzky-Golay filtering (window={self.window_length}, poly={self.polyorder})...")
        
        sg_features_list = []
        
        # Original smoothed spectra (0th derivative)
        print("  Computing smoothed spectra...")
        sg_smoothed = np.array([
            savgol_filter(spectrum, self.window_length, self.polyorder, deriv=0)
            for spectrum in spectra
        ])
        sg_features_list.append(sg_smoothed)
        
        if self.apply_derivatives:
            # First derivative
            print("  Computing 1st derivative...")
            sg_first_deriv = np.array([
                savgol_filter(spectrum, self.window_length, self.polyorder, deriv=1)
                for spectrum in spectra
            ])
            sg_features_list.append(sg_first_deriv)
            
            # Second derivative
            print("  Computing 2nd derivative...")
            sg_second_deriv = np.array([
                savgol_filter(spectrum, self.window_length, self.polyorder, deriv=2)
                for spectrum in spectra
            ])
            sg_features_list.append(sg_second_deriv)
        
        # Combine all SG features
        combined_sg_features = np.concatenate(sg_features_list, axis=1)
        
        # Handle NaN values
        combined_sg_features = np.nan_to_num(combined_sg_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"✓ SG filtering completed: {combined_sg_features.shape}")
        return combined_sg_features
    
    def create_mortality_specific_features(self, spectra):
        """
        Create mortality-specific spectral features
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Mortality-specific feature matrix
        """
        print("Creating mortality-specific spectral features...")
        
        mortality_features = []
        
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
            
            # Percentile features for mortality assessment
            percentiles = np.percentile(spectrum, [10, 25, 50, 75, 90])
            
            # Spectral shape features (important for biological changes)
            x = np.arange(len(spectrum))
            total_intensity = np.sum(spectrum)
            
            if total_intensity > 0:
                # Spectral centroid (center of mass)
                centroid = np.sum(x * spectrum) / total_intensity
                
                # Spectral spread (width)
                spread = np.sqrt(np.sum((x - centroid)**2 * spectrum) / total_intensity)
                
                # Spectral slope (overall trend)
                slope = np.polyfit(x, spectrum, 1)[0]
            else:
                centroid = spread = slope = 0
            
            spectral_shape = [centroid, spread, slope]
            
            # Biological indicators for mortality
            # Peak characteristics (protein and lipid regions)
            n_points = len(spectrum)
            
            # Divide spectrum into biological regions
            protein_region = spectrum[:n_points//3]  # Early wavelengths
            lipid_region = spectrum[n_points//3:2*n_points//3]  # Mid wavelengths  
            water_region = spectrum[2*n_points//3:]  # Late wavelengths
            
            biological_features = [
                np.mean(protein_region),
                np.std(protein_region),
                np.mean(lipid_region),
                np.std(lipid_region),
                np.mean(water_region),
                np.std(water_region),
                # Ratios that may indicate mortality
                np.mean(protein_region) / (np.mean(lipid_region) + 1e-10),
                np.mean(water_region) / (np.mean(protein_region) + 1e-10),
                np.std(protein_region) / (np.mean(protein_region) + 1e-10),  # CV
            ]
            
            # Peak detection for mortality markers
            peak_count = 0
            peak_heights = []
            
            for i in range(1, len(spectrum) - 1):
                if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                    peak_count += 1
                    peak_heights.append(spectrum[i])
            
            peak_features = [
                peak_count,
                np.mean(peak_heights) if peak_heights else 0,
                np.std(peak_heights) if len(peak_heights) > 1 else 0,
            ]
            
            # Combine all mortality-specific features
            all_features = (basic_stats + percentiles.tolist() + 
                          spectral_shape + biological_features + peak_features)
            mortality_features.append(all_features)
        
        mortality_features = np.array(mortality_features)
        
        # Handle NaN values
        mortality_features = np.nan_to_num(mortality_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"✓ Mortality-specific features created: {mortality_features.shape}")
        return mortality_features
    
    def optimize_feature_selection_for_mortality(self, X, y, n_features_range=(50, 200)):
        """
        Optimize feature selection specifically for mortality classification
        
        Args:
            X: Feature matrix
            y: Mortality labels
            n_features_range: Range of features to test
            
        Returns:
            Best selector and number of features
        """
        print(f"Optimizing feature selection for mortality classification...")
        
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        min_features, max_features = n_features_range
        max_possible = min(X.shape[1], max_features)
        
        # Adjust range based on available features
        if max_possible < min_features:
            feature_counts = [max_possible]
        else:
            step_size = max(1, (max_possible - min_features) // 8)
            feature_counts = list(range(min_features, max_possible + 1, step_size))
        
        best_score = -1
        best_selector = None
        best_n_features = feature_counts[0]
        best_method = 'f_classif'
        
        # Test different selection methods
        selection_methods = {
            'f_classif': f_classif,
            'mutual_info': mutual_info_classif
        }
        
        for method_name, score_func in selection_methods.items():
            print(f"  Testing {method_name} feature selection...")
            
            # Test different numbers of features
            for n_feat in feature_counts:
                print(f"    Testing {n_feat} features with {method_name}...")
                
                try:
                    selector = SelectKBest(score_func=score_func, k=n_feat)
                    X_selected = selector.fit_transform(X, y)
                    
                    # Quick evaluation with Random Forest
                    rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
                    cv_scores = cross_val_score(rf, X_selected, y, cv=3, scoring='accuracy')
                    mean_score = np.mean(cv_scores)
                    
                    print(f"      CV Score: {mean_score:.4f}")
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_n_features = n_feat
                        best_selector = selector
                        best_method = method_name
                        
                except Exception as e:
                    print(f"      Failed: {e}")
        
        print(f"✓ Best selection: {best_method} with {best_n_features} features (CV: {best_score:.4f})")
        return best_selector, best_n_features, best_method
    
    def preprocess_for_mortality_classification(self, spectra, labels, wavelengths):
        """
        Complete preprocessing pipeline for mortality classification
        
        Args:
            spectra: Raw spectral data
            labels: Mortality labels
            wavelengths: Wavelength array
            
        Returns:
            Processed features and metadata
        """
        print("\n" + "="*60)
        print("MSC + SAVITZKY-GOLAY PREPROCESSING FOR MORTALITY")
        print("="*60)
        
        self.wavelength_range = (wavelengths.min(), wavelengths.max())
        self.n_original_features = spectra.shape[1]
        
        # Step 1: Compute MSC reference spectrum
        self.reference_spectrum = self.compute_msc_reference(spectra)
        
        # Step 2: Apply MSC correction
        msc_corrected_spectra = self.apply_msc_correction(spectra)
        
        # Step 3: Apply Savitzky-Golay filtering
        sg_features = self.apply_savitzky_golay_filtering(msc_corrected_spectra)
        
        # Step 4: Create mortality-specific features
        mortality_features = self.create_mortality_specific_features(msc_corrected_spectra)
        
        # Step 5: Combine all features
        combined_features = np.concatenate([sg_features, mortality_features], axis=1)
        
        print(f"\nFeature combination:")
        print(f"  - SG filtered: {sg_features.shape[1]} features")
        print(f"  - Mortality-specific: {mortality_features.shape[1]} features")
        print(f"  - Total combined: {combined_features.shape[1]} features")
        
        # Step 6: Optimize feature selection for mortality
        best_selector, n_features, method = self.optimize_feature_selection_for_mortality(
            combined_features, labels, n_features_range=(30, min(150, combined_features.shape[1]))
        )
        
        self.feature_selector = best_selector
        selected_features = best_selector.fit_transform(combined_features, labels)
        
        # Step 7: Apply scaling
        print(f"\nApplying dual scaling...")
        
        # Standard scaling for primary features
        standard_scaled = self.standard_scaler.fit_transform(selected_features)
        
        # MinMax scaling for additional robustness
        minmax_scaled = self.minmax_scaler.fit_transform(selected_features)
        
        # Create dual-scale feature set for enhanced mortality detection
        dual_scale_features = np.concatenate([standard_scaled, minmax_scaled], axis=1)
        
        print(f"✓ Dual-scale features: {dual_scale_features.shape}")
        
        # Store preprocessing metadata
        preprocessing_info = {
            'msc_applied': True,
            'sg_window_length': self.window_length,
            'sg_polyorder': self.polyorder,
            'derivatives_applied': self.apply_derivatives,
            'n_original_features': self.n_original_features,
            'n_selected_features': n_features,
            'n_final_features': dual_scale_features.shape[1],
            'feature_selection_method': method,
            'wavelength_range': self.wavelength_range,
            'scaling_methods': ['StandardScaler', 'MinMaxScaler'],
            'mortality_specific_features': mortality_features.shape[1]
        }
        
        return dual_scale_features, preprocessing_info
    
    def transform_new_data(self, spectra, wavelengths):
        """
        Transform new spectral data using fitted preprocessors
        
        Args:
            spectra: New spectral data
            wavelengths: Wavelength array
            
        Returns:
            Transformed features
        """
        # Apply MSC correction
        msc_corrected = self.apply_msc_correction(spectra)
        
        # Apply SG filtering
        sg_features = self.apply_savitzky_golay_filtering(msc_corrected)
        
        # Create mortality features
        mortality_features = self.create_mortality_specific_features(msc_corrected)
        
        # Combine features
        combined_features = np.concatenate([sg_features, mortality_features], axis=1)
        
        # Apply feature selection
        selected_features = self.feature_selector.transform(combined_features)
        
        # Apply scaling
        standard_scaled = self.standard_scaler.transform(selected_features)
        minmax_scaled = self.minmax_scaler.transform(selected_features)
        
        # Create dual-scale features
        dual_scale_features = np.concatenate([standard_scaled, minmax_scaled], axis=1)
        
        return dual_scale_features

def load_and_merge_data(day='D0'):
    """Load and merge spectral and metadata for mortality analysis"""
    print(f"Loading data for Day {day}...")
    
    ref_df = pd.read_csv('../../data/reference_metadata.csv')
    spectral_df = pd.read_csv(f'../../data/spectral_data_{day}.csv')
    merged_df = pd.merge(ref_df, spectral_df, on='HSI sample ID', how='inner')
    
    print(f"✓ Merged dataset shape: {merged_df.shape}")
    return merged_df

def prepare_mortality_dataset(merged_df):
    """Prepare dataset for mortality classification"""
    print("\n" + "="*60)
    print("PREPARING MORTALITY CLASSIFICATION DATASET - M1 EXPERIMENT")
    print("="*60)
    
    # Show available mortality status values
    print("Available mortality status values:")
    print(merged_df['Mortality status'].value_counts())
    
    # Define live vs dead categories
    live_labels = ['live', 'Live', 'Still alive', 'Possibly still alive - left in incubator']
    dead_labels = ['Dead embryo', 'Early dead', 'Late dead; cannot tell', 'Did not hatch']
    
    # Filter for mortality classification
    mortality_df = merged_df[
        merged_df['Mortality status'].isin(live_labels + dead_labels)
    ].copy()
    
    # Create binary mortality labels
    mortality_df['Binary_Mortality'] = mortality_df['Mortality status'].apply(
        lambda x: 'Alive' if x in live_labels else 'Dead'
    )
    
    print(f"Samples with mortality labels: {len(mortality_df)}")
    
    # Class distribution
    mortality_counts = mortality_df['Binary_Mortality'].value_counts()
    print(f"Binary mortality distribution: {mortality_counts.to_dict()}")
    
    # Check for class imbalance
    alive_count = mortality_counts.get('Alive', 0)
    dead_count = mortality_counts.get('Dead', 0)
    total_count = alive_count + dead_count
    
    if total_count > 0:
        imbalance_ratio = max(alive_count, dead_count) / min(alive_count, dead_count)
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 2.0:
            print("⚠️  Significant class imbalance detected - SMOTE will be beneficial")
        else:
            print("✓ Relatively balanced classes")
    
    # Extract wavelength features
    metadata_cols = ['HSI sample ID', 'Date_x', 'Date_y', 'Exp. No._x', 'Exp. No._y', 
                    'Gender', 'Fertility status', 'Mortality status', 'Mass (g)', 
                    'Major dia. (mm)', 'Minor dia. (mm)', 'Comment']
    
    wavelength_cols = []
    for col in mortality_df.columns:
        if col not in metadata_cols:
            try:
                float(col)
                wavelength_cols.append(col)
            except ValueError:
                continue
    
    print(f"Found {len(wavelength_cols)} wavelength features")
    
    # Extract features and labels
    X = mortality_df[wavelength_cols].values
    y = mortality_df['Binary_Mortality'].values
    wavelengths = np.array([float(col) for col in wavelength_cols])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
    print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    return X, y_encoded, label_encoder, wavelengths, mortality_df

def main():
    """Main preprocessing pipeline for M1 mortality classification"""
    print("="*80)
    print("M1 EXPERIMENT: MSC + SG + LightGBM + SMOTE for Mortality Classification")
    print("="*80)
    
    # Load data
    merged_df = load_and_merge_data(day='D0')
    X, y, label_encoder, wavelengths, mortality_df = prepare_mortality_dataset(merged_df)
    
    # Stratified split for robust evaluation
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"Training set: {X_train_raw.shape[0]} samples")
    print(f"Test set: {X_test_raw.shape[0]} samples")
    
    # Check training set class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Training class distribution: {dict(zip(label_encoder.classes_[unique], counts))}")
    
    # Initialize preprocessor
    preprocessor = MSCSavitzkyGolayPreprocessor(
        window_length=15,
        polyorder=2,
        apply_derivatives=True
    )
    
    # Preprocess training data
    X_train_processed, preprocessing_info = preprocessor.preprocess_for_mortality_classification(
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
    joblib.dump(preprocessor, 'msc_sg_preprocessor.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    # Save preprocessing information
    with open('preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2, default=str)
    
    # Summary
    print("\n" + "="*60)
    print("MORTALITY PREPROCESSING SUMMARY")
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
    
    print("\nReady for LightGBM + SMOTE training!")

if __name__ == "__main__":
    main() 