"""
M6 Preprocessing: SG + Optimization for Mortality Classification
Optimized Savitzky-Golay preprocessing with morphological and spectral features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy.signal import savgol_filter, find_peaks
from scipy import stats
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
import warnings
warnings.filterwarnings('ignore')

class OptimizedSGMorphologicalPreprocessor:
    """
    Optimized Savitzky-Golay preprocessing with morphological analysis
    Advanced spectral and morphological feature engineering for mortality prediction
    """
    
    def __init__(self, optimize_sg_params=True, apply_morphology=True):
        """
        Initialize optimized SG + morphological preprocessor
        
        Args:
            optimize_sg_params: Whether to optimize SG parameters automatically
            apply_morphology: Whether to include morphological features
        """
        self.optimize_sg_params = optimize_sg_params
        self.apply_morphology = apply_morphology
        
        # Optimized SG parameters
        self.best_window_length = 15
        self.best_polyorder = 2
        
        # Scalers
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        
        # Feature selector
        self.feature_selector = None
        
        # Preprocessing metadata
        self.wavelength_range = None
        self.n_original_features = None
        self.morphological_features_names = []
        
    def optimize_sg_parameters(self, spectra, labels):
        """
        Optimize Savitzky-Golay parameters for mortality classification
        Tests different window lengths and polynomial orders
        
        Args:
            spectra: 2D array (samples x wavelengths)
            labels: 1D array of mortality labels
            
        Returns:
            Best window_length and polyorder
        """
        print("Optimizing Savitzky-Golay parameters for mortality classification...")
        
        # Parameter ranges to test
        window_lengths = [9, 11, 13, 15, 17, 19, 21]  # Must be odd
        poly_orders = [2, 3, 4]
        
        best_score = 0
        best_window = 15
        best_poly = 2
        
        for window in window_lengths:
            for poly in poly_orders:
                if window > poly:  # Window must be larger than polynomial order
                    try:
                        # Apply SG filtering with current parameters
                        sg_spectra = np.array([
                            savgol_filter(spectrum, window, poly, deriv=0)
                            for spectrum in spectra
                        ])
                        
                        # Create simple features for evaluation
                        simple_features = np.column_stack([
                            np.mean(sg_spectra, axis=1),
                            np.std(sg_spectra, axis=1),
                            np.max(sg_spectra, axis=1),
                            np.min(sg_spectra, axis=1)
                        ])
                        
                        # Evaluate with F-classification score
                        selector = SelectKBest(f_classif, k='all')
                        selector.fit(simple_features, labels)
                        score = np.mean(selector.scores_)
                        
                        if score > best_score:
                            best_score = score
                            best_window = window
                            best_poly = poly
                            
                    except Exception as e:
                        print(f"  Warning: Failed for window={window}, poly={poly}: {e}")
                        continue
        
        self.best_window_length = best_window
        self.best_polyorder = best_poly
        
        print(f"✓ Optimal SG parameters: window={best_window}, poly={best_poly}")
        print(f"  Best F-score: {best_score:.4f}")
        
        return best_window, best_poly
    
    def apply_optimized_sg_filtering(self, spectra):
        """
        Apply optimized Savitzky-Golay filtering with multiple derivatives
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Enhanced SG features
        """
        print(f"Applying optimized SG filtering (window={self.best_window_length}, poly={self.best_polyorder})...")
        
        sg_features_list = []
        
        # Original smoothed spectra (0th derivative)
        print("  Computing optimized smoothed spectra...")
        sg_smoothed = np.array([
            savgol_filter(spectrum, self.best_window_length, self.best_polyorder, deriv=0)
            for spectrum in spectra
        ])
        sg_features_list.append(sg_smoothed)
        
        # First derivative (slope information)
        print("  Computing optimized 1st derivative...")
        sg_first_deriv = np.array([
            savgol_filter(spectrum, self.best_window_length, self.best_polyorder, deriv=1)
            for spectrum in spectra
        ])
        sg_features_list.append(sg_first_deriv)
        
        # Second derivative (curvature information)
        print("  Computing optimized 2nd derivative...")
        sg_second_deriv = np.array([
            savgol_filter(spectrum, self.best_window_length, self.best_polyorder, deriv=2)
            for spectrum in spectra
        ])
        sg_features_list.append(sg_second_deriv)
        
        # Third derivative (fine variations) - NEW for M6
        print("  Computing 3rd derivative for fine spectral variations...")
        sg_third_deriv = np.array([
            savgol_filter(spectrum, self.best_window_length, self.best_polyorder, deriv=3)
            for spectrum in spectra
        ])
        sg_features_list.append(sg_third_deriv)
        
        # Combine all SG features
        combined_sg_features = np.concatenate(sg_features_list, axis=1)
        
        # Handle NaN values
        combined_sg_features = np.nan_to_num(combined_sg_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"✓ Enhanced SG filtering completed: {combined_sg_features.shape}")
        return combined_sg_features
    
    def extract_morphological_features(self, spectra):
        """
        Extract morphological features from spectral data
        Includes peak analysis, shape descriptors, and structural patterns
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Morphological feature matrix
        """
        print("Extracting morphological features from spectra...")
        
        morphological_features = []
        feature_names = []
        
        for spectrum in spectra:
            features = []
            
            # Peak-related morphological features
            peaks, peak_properties = find_peaks(spectrum, height=np.mean(spectrum))
            
            # Number of peaks
            n_peaks = len(peaks)
            features.append(n_peaks)
            
            # Peak characteristics
            if n_peaks > 0:
                peak_heights = peak_properties['peak_heights']
                features.extend([
                    np.mean(peak_heights),  # Average peak height
                    np.std(peak_heights),   # Peak height variability
                    np.max(peak_heights),   # Highest peak
                    np.sum(peak_heights)    # Total peak intensity
                ])
                
                # Peak spacing (regularity)
                if n_peaks > 1:
                    peak_spacings = np.diff(peaks)
                    features.extend([
                        np.mean(peak_spacings),  # Average peak spacing
                        np.std(peak_spacings)    # Peak spacing variability
                    ])
                else:
                    features.extend([0, 0])  # No spacing for single peak
            else:
                features.extend([0, 0, 0, 0, 0, 0])  # No peaks found
            
            # Spectral shape morphology
            # Valley detection (inverted peaks)
            valleys, _ = find_peaks(-spectrum, height=-np.mean(spectrum))
            n_valleys = len(valleys)
            features.append(n_valleys)
            
            # Spectral roughness (local variability)
            diff_spectrum = np.diff(spectrum)
            roughness = np.mean(np.abs(diff_spectrum))
            features.append(roughness)
            
            # Spectral symmetry (mortality indicator)
            mid_point = len(spectrum) // 2
            left_half = spectrum[:mid_point]
            right_half = spectrum[mid_point:]
            if len(left_half) == len(right_half):
                symmetry = np.corrcoef(left_half, right_half[::-1])[0, 1]
                features.append(symmetry if not np.isnan(symmetry) else 0)
            else:
                features.append(0)
            
            # Morphological moments (shape descriptors)
            x = np.arange(len(spectrum))
            if np.sum(spectrum) > 0:
                # First moment (centroid)
                centroid = np.sum(x * spectrum) / np.sum(spectrum)
                features.append(centroid)
                
                # Second moment (spread/width)
                spread = np.sqrt(np.sum((x - centroid)**2 * spectrum) / np.sum(spectrum))
                features.append(spread)
                
                # Third moment (skewness of distribution)
                skewness = np.sum((x - centroid)**3 * spectrum) / (np.sum(spectrum) * spread**3)
                features.append(skewness if not np.isnan(skewness) else 0)
                
                # Fourth moment (kurtosis of distribution)
                kurtosis = np.sum((x - centroid)**4 * spectrum) / (np.sum(spectrum) * spread**4)
                features.append(kurtosis if not np.isnan(kurtosis) else 0)
            else:
                features.extend([0, 0, 0, 0])
            
            # Binary morphological operations (for pattern analysis)
            # Threshold spectrum for morphological analysis
            threshold = np.mean(spectrum)
            binary_spectrum = spectrum > threshold
            
            if np.any(binary_spectrum):
                # Erosion and dilation patterns
                structure = np.ones(3)  # 3-point structuring element
                
                eroded = binary_erosion(binary_spectrum, structure)
                dilated = binary_dilation(binary_spectrum, structure)
                opened = binary_opening(binary_spectrum, structure)
                closed = binary_closing(binary_spectrum, structure)
                
                # Morphological features from binary operations
                features.extend([
                    np.sum(eroded),     # Erosion result
                    np.sum(dilated),    # Dilation result  
                    np.sum(opened),     # Opening result
                    np.sum(closed)      # Closing result
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            morphological_features.append(features)
        
        # Feature names for interpretability
        if not self.morphological_features_names:
            self.morphological_features_names = [
                'n_peaks', 'mean_peak_height', 'std_peak_height', 'max_peak_height', 
                'total_peak_intensity', 'mean_peak_spacing', 'std_peak_spacing',
                'n_valleys', 'roughness', 'symmetry', 'centroid', 'spread', 
                'skewness', 'kurtosis', 'erosion', 'dilation', 'opening', 'closing'
            ]
        
        morphological_matrix = np.array(morphological_features)
        print(f"✓ Morphological features extracted: {morphological_matrix.shape}")
        print(f"  Features: {len(self.morphological_features_names)} morphological descriptors")
        
        return morphological_matrix
    
    def create_advanced_spectral_features(self, spectra):
        """
        Create advanced spectral features for mortality classification
        Includes biological indicators and statistical descriptors
        
        Args:
            spectra: 2D array (samples x wavelengths)
            
        Returns:
            Advanced spectral feature matrix
        """
        print("Creating advanced spectral features...")
        
        spectral_features = []
        
        for spectrum in spectra:
            # Statistical descriptors
            stats_features = [
                np.mean(spectrum),
                np.std(spectrum),
                np.max(spectrum),
                np.min(spectrum),
                np.median(spectrum),
                np.ptp(spectrum),  # Peak-to-peak
                stats.skew(spectrum),
                stats.kurtosis(spectrum)
            ]
            
            # Percentile features (mortality assessment)
            percentiles = np.percentile(spectrum, [5, 10, 25, 50, 75, 90, 95])
            
            # Energy and power features
            total_energy = np.sum(spectrum**2)
            spectral_energy = [
                total_energy,
                np.sqrt(np.mean(spectrum**2)),  # RMS
                np.mean(np.abs(spectrum))       # MAE
            ]
            
            # Biological region analysis (enhanced for mortality)
            n_wavelengths = len(spectrum)
            
            # Blue region (400-500nm) - Carotenoids/oxidative stress
            blue_start = int(0.1 * n_wavelengths)
            blue_end = int(0.3 * n_wavelengths)
            blue_region = spectrum[blue_start:blue_end]
            
            # Green region (500-600nm) - Chlorophyll/plant matter
            green_start = int(0.3 * n_wavelengths)
            green_end = int(0.5 * n_wavelengths)
            green_region = spectrum[green_start:green_end]
            
            # Red region (600-700nm) - Protein content
            red_start = int(0.5 * n_wavelengths)
            red_end = int(0.7 * n_wavelengths)
            red_region = spectrum[red_start:red_end]
            
            # NIR region (700-1000nm) - Water/lipid content
            nir_start = int(0.7 * n_wavelengths)
            nir_end = n_wavelengths
            nir_region = spectrum[nir_start:nir_end]
            
            # Regional statistics
            regional_features = []
            for region, name in [(blue_region, 'blue'), (green_region, 'green'), 
                               (red_region, 'red'), (nir_region, 'nir')]:
                if len(region) > 0:
                    regional_features.extend([
                        np.mean(region),
                        np.std(region),
                        np.max(region)
                    ])
                else:
                    regional_features.extend([0, 0, 0])
            
            # Biological ratios (mortality indicators)
            if np.mean(blue_region) > 0 and np.mean(red_region) > 0 and np.mean(nir_region) > 0:
                bio_ratios = [
                    np.mean(red_region) / np.mean(blue_region),    # Protein/Carotenoid
                    np.mean(nir_region) / np.mean(red_region),     # Water-Lipid/Protein
                    np.mean(nir_region) / np.mean(blue_region),    # NIR/Carotenoid
                    np.mean(green_region) / np.mean(red_region)    # Green/Red
                ]
            else:
                bio_ratios = [1, 1, 1, 1]  # Default ratios
            
            # Combine all spectral features
            combined_features = (stats_features + list(percentiles) + 
                               spectral_energy + regional_features + bio_ratios)
            spectral_features.append(combined_features)
        
        spectral_matrix = np.array(spectral_features)
        print(f"✓ Advanced spectral features created: {spectral_matrix.shape}")
        
        return spectral_matrix
    
    def optimize_feature_selection(self, X, y, max_features=200):
        """
        Optimize feature selection for mortality classification
        Uses mutual information and F-classification
        
        Args:
            X: Feature matrix
            y: Labels
            max_features: Maximum number of features to select
            
        Returns:
            Optimized feature matrix
        """
        print("Optimizing feature selection for mortality prediction...")
        
        # Determine optimal number of features
        n_features_to_test = min(max_features, X.shape[1])
        
        # Use mutual information for feature selection
        selector = SelectKBest(mutual_info_classif, k=n_features_to_test)
        self.feature_selector = selector
        
        X_selected = selector.fit_transform(X, y)
        
        selected_features = selector.get_support(indices=True)
        scores = selector.scores_
        
        print(f"✓ Feature selection completed: {X.shape[1]} → {X_selected.shape[1]} features")
        print(f"  Selection method: Mutual Information")
        print(f"  Average feature score: {np.mean(scores):.4f}")
        
        return X_selected
    
    def preprocess_for_mortality_classification(self, spectra, labels, wavelengths):
        """
        Complete preprocessing pipeline for M6 mortality classification
        
        Args:
            spectra: Raw spectral data
            labels: Mortality labels
            wavelengths: Wavelength array
            
        Returns:
            Preprocessed features and metadata
        """
        print("\n" + "="*60)
        print("M6 PREPROCESSING: SG + OPTIMIZATION + MORPHOLOGICAL")
        print("="*60)
        
        self.wavelength_range = (wavelengths[0], wavelengths[-1])
        self.n_original_features = spectra.shape[1]
        
        # Step 1: Optimize SG parameters if requested
        if self.optimize_sg_params:
            self.optimize_sg_parameters(spectra, labels)
        
        # Step 2: Apply optimized SG filtering with derivatives
        sg_features = self.apply_optimized_sg_filtering(spectra)
        
        # Step 3: Extract morphological features
        if self.apply_morphology:
            morphological_features = self.extract_morphological_features(spectra)
        else:
            morphological_features = np.array([]).reshape(spectra.shape[0], 0)
        
        # Step 4: Create advanced spectral features
        spectral_features = self.create_advanced_spectral_features(spectra)
        
        # Step 5: Combine all features
        if morphological_features.size > 0:
            combined_features = np.concatenate([sg_features, morphological_features, spectral_features], axis=1)
        else:
            combined_features = np.concatenate([sg_features, spectral_features], axis=1)
        
        print(f"Combined features shape: {combined_features.shape}")
        
        # Step 6: Optimize feature selection
        optimized_features = self.optimize_feature_selection(combined_features, labels)
        
        # Step 7: Dual scaling (Standard + Robust)
        print("Applying dual scaling (Standard + Robust)...")
        
        # Standard scaling
        features_standard = self.standard_scaler.fit_transform(optimized_features)
        
        # Robust scaling
        features_robust = self.robust_scaler.fit_transform(optimized_features)
        
        # Combine both scalings
        final_features = np.concatenate([features_standard, features_robust], axis=1)
        
        print(f"✓ Final preprocessed features: {final_features.shape}")
        
        # Create preprocessing info
        preprocessing_info = {
            'method': 'SG_Optimization_Morphological',
            'wavelength_range': self.wavelength_range,
            'n_original_features': self.n_original_features,
            'n_final_features': final_features.shape[1],
            'sg_window_length': self.best_window_length,
            'sg_polyorder': self.best_polyorder,
            'sg_optimized': self.optimize_sg_params,
            'morphological_enabled': self.apply_morphology,
            'n_morphological_features': len(self.morphological_features_names),
            'feature_selection_method': 'mutual_info_classif',
            'scaling_methods': ['StandardScaler', 'RobustScaler']
        }
        
        return final_features, preprocessing_info

def load_and_merge_data(day='D0'):
    """Load and merge spectral and metadata"""
    print(f"Loading HSI data for {day}...")
    
    # Load data
    metadata_df = pd.read_csv('../../data/reference_metadata.csv')
    spectral_df = pd.read_csv(f'../../data/spectral_data_{day}.csv')
    
    # Merge on HSI sample ID
    merged_df = pd.merge(metadata_df, spectral_df, on='HSI sample ID', how='inner')
    
    print(f"✓ Loaded {len(merged_df)} samples with {spectral_df.shape[1]-1} wavelengths")
    return merged_df

def prepare_mortality_dataset(merged_df):
    """Prepare mortality classification dataset"""
    print("Preparing mortality classification dataset...")
    
    # Define mortality classes
    alive_labels = ['live', 'Live', 'Still alive', 'Possibly still alive - left in incubator']
    dead_labels = ['Dead embryo', 'Early dead', 'Late dead; cannot tell', 'Did not hatch']
    
    # Filter samples with valid mortality data
    mortality_mask = merged_df['Mortality status'].isin(alive_labels + dead_labels)
    mortality_df = merged_df[mortality_mask].copy()
    
    # Binary encoding
    mortality_df['Mortality_Binary'] = mortality_df['Mortality status'].apply(
        lambda x: 'Alive' if x in alive_labels else 'Dead'
    )
    
    # Extract features and labels
    wavelength_cols = [col for col in mortality_df.columns if col.replace('.', '').replace('-', '').isdigit()]
    
    spectra = mortality_df[wavelength_cols].values
    labels = mortality_df['Mortality_Binary'].values
    wavelengths = np.array([float(col) for col in wavelength_cols])
    
    print(f"✓ Mortality dataset prepared:")
    print(f"  Total samples: {len(spectra)}")
    print(f"  Wavelengths: {len(wavelengths)} ({wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm)")
    print(f"  Class distribution: {np.bincount(LabelEncoder().fit_transform(labels))}")
    
    return spectra, labels, wavelengths, mortality_df

def main():
    """Main preprocessing function for M6"""
    print("Starting M6 Preprocessing: SG + Optimization + Morphological")
    
    # Load data
    merged_df = load_and_merge_data('D0')
    spectra, labels, wavelengths, mortality_df = prepare_mortality_dataset(merged_df)
    
    # Initialize preprocessor
    preprocessor = OptimizedSGMorphologicalPreprocessor(
        optimize_sg_params=True,
        apply_morphology=True
    )
    
    # Apply preprocessing
    processed_features, preprocessing_info = preprocessor.preprocess_for_mortality_classification(
        spectra, labels, wavelengths
    )
    
    # Train/test split
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        processed_features, encoded_labels, 
        test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    np.save('X_train_processed.npy', X_train)
    np.save('X_test_processed.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save preprocessing artifacts
    import joblib
    joblib.dump(preprocessor, 'optimized_sg_morphological_preprocessor.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    
    # Save preprocessing info
    import json
    with open('preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    
    print(f"\n✓ M6 Preprocessing completed successfully!")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Features: {X_train.shape[1]} (optimized)")
    
    return X_train, X_test, y_train, y_test, preprocessing_info

if __name__ == "__main__":
    main() 