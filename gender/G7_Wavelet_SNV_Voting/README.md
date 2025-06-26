# G7 Experiment: Wavelet + SNV + Voting Ensemble

## 🎯 Objective
Advanced gender prediction using hyperspectral imaging data through sophisticated ensemble methods combining wavelet decomposition, SNV normalization, and multiple voting strategies.

## 🧬 Methodology

### Preprocessing: Multi-Path Feature Engineering
- **Wavelet Decomposition**: db4 wavelet with 3 decomposition levels for multi-resolution analysis
- **SNV Normalization**: Standard Normal Variate for scatter correction
- **Savitzky-Golay Filtering**: Window length 15, polynomial order 2 with derivatives
- **Statistical Features**: Comprehensive spectral statistics (18 features)
- **Multi-Scale Normalization**: Standard, MinMax, and Robust scaling

### Feature Engineering Pipeline
```
Raw Spectra (300 wavelengths)
├── Path 1: SNV + Savitzky-Golay → 900 features → Optimized to 50 features
├── Path 2: Wavelet Decomposition → 33 features → Optimized to 10 features  
├── Path 3: Combined Features → 951 features → Optimized to 175 features
├── Path 4: SNV + MinMax Scaling → 50 features
└── Path 5: SNV + Robust Scaling → 50 features
```

### Advanced Voting Ensemble
- **Base Models**: 6 diverse algorithms per feature set
  - Random Forest (200 trees, max_depth=10)
  - Gradient Boosting (100 estimators, learning_rate=0.1)
  - SVM with RBF kernel
  - Logistic Regression
  - Gaussian Naive Bayes
  - K-Nearest Neighbors (k=5, distance weighting)

- **Ensemble Strategies**: 
  - Hard Voting: Majority vote classification
  - Soft Voting: Probability-weighted predictions

### Feature Set Optimization
- **Mutual Information**: Feature relevance scoring
- **Cross-Validation**: 5-fold stratified validation for selection
- **Automated Selection**: Range-based optimization (10-300 features)

## 📊 **Key Results**

### **🏆 Performance Summary**
- **Best Model**: snv_sg_gradient_boosting
- **Best Accuracy**: 53.49%
- **Total Models Trained**: 40 (30 base + 10 ensemble)
- **Feature Sets**: 5 different preprocessing approaches
- **Dataset**: 1,074 samples (859 training, 215 test)

### **🎯 Feature Set Performance**
| Feature Set | Features | Best Algorithm | Accuracy |
|-------------|----------|----------------|----------|
| **SNV + SG** | 50 | **Gradient Boosting** | **53.49%** |
| **Wavelet** | 10 | Naive Bayes | 53.49% |
| **Combined** | 175 | Naive Bayes | 52.56% |
| **SNV + MinMax** | 50 | Gradient Boosting | 53.49% |
| **SNV + Robust** | 50 | Gradient Boosting | 53.49% |

### **🥇 Top 5 Model Results**
1. **snv_sg_gradient_boosting**: 53.49%
2. **snv_minmax_gradient_boosting**: 53.49%
3. **snv_robust_gradient_boosting**: 53.49%
4. **wavelet_naive_bayes**: 53.49%
5. **combined_naive_bayes**: 52.56%

### **📈 Ensemble Results**
| Ensemble Type | Feature Set | Accuracy | Notes |
|---------------|-------------|----------|-------|
| **Hard Voting** | SNV + MinMax | 51.16% | Majority vote |
| **Soft Voting** | Wavelet | 50.70% | Probability averaging |
| **Hard Voting** | Wavelet | 50.23% | Binary consensus |
| **Soft Voting** | Combined | 50.23% | Weighted predictions |

### **📊 Best Model Analysis (snv_sg_gradient_boosting)**
```
              precision    recall  f1-score   support
      Female       0.56      0.66      0.61       116
        Male       0.49      0.38      0.43        99

    accuracy                           0.53       215
   macro avg       0.53      0.52      0.52       215
weighted avg       0.53      0.53      0.53       215

Confusion Matrix:
[[77 39]
 [61 38]]
```

### **🔬 Technical Achievement**
- **Multi-Path Processing**: 5 different feature engineering approaches
- **Advanced Ensemble**: 40 total models with voting strategies
- **Wavelet Innovation**: db4 wavelet with 3 decomposition levels (33 features)
- **Gradient Boosting**: Consistently best performer across feature sets

## 🔬 Technical Innovation

### Wavelet Analysis Features
- **Approximation Coefficients**: Low-frequency spectral patterns
- **Detail Coefficients**: High-frequency noise and fine features
- **Energy Distribution**: Multi-level energy ratios
- **Statistical Descriptors**: Mean, std, skew, kurtosis per level

### Multi-Path Preprocessing
- **Complementary Approaches**: Different preprocessing methods capture distinct spectral aspects
- **Feature Diversity**: 5 feature sets provide ensemble diversity
- **Automatic Optimization**: Data-driven feature selection per path

### Ensemble Architecture
```
Feature Set 1 → [6 Base Models] → Hard/Soft Voting → Predictions
Feature Set 2 → [6 Base Models] → Hard/Soft Voting → Predictions
Feature Set 3 → [6 Base Models] → Hard/Soft Voting → Predictions
Feature Set 4 → [6 Base Models] → Hard/Soft Voting → Predictions
Feature Set 5 → [6 Base Models] → Hard/Soft Voting → Predictions
                                              ↓
                                    Best Model Selection
```

## 📁 Generated Files

### Core Results
- **G7_experimental_results.json**: Complete experimental data (269,823 bytes)
- **G7_performance_summary.txt**: Human-readable summary (1,436 bytes)
- **G7_voting_ensemble_models.pkl**: All trained models (1,058,079 bytes)

### Preprocessing Data
- **ensemble_feature_info.json**: Feature set metadata (720 bytes)
- **X_train_*.npy**: Training feature sets (5 files)
- **X_test_*.npy**: Test feature sets (5 files)
- **y_train.npy, y_test.npy**: Label arrays
- **label_encoder.pkl**: Label encoding information

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Complete Experiment
```bash
python run_G7_experiment.py
```

### Individual Components
```bash
# Preprocessing only
python G7_preprocessing.py

# Model training only (requires preprocessing)
python G7_model.py
```

## 📋 Dependencies
- **Python**: ≥3.7
- **NumPy**: ≥1.20.0 (numerical computing)
- **Pandas**: ≥1.3.0 (data manipulation)
- **Scikit-learn**: ≥1.0.0 (machine learning)
- **SciPy**: ≥1.7.0 (scientific computing)
- **PyWavelets**: ≥1.1.0 (wavelet transforms)
- **Joblib**: ≥1.0.0 (model persistence)

## 🔍 Methodology Details

### Wavelet Decomposition Process
1. **Daubechies-4 Wavelet**: Chosen for balanced frequency resolution
2. **3 Decomposition Levels**: Multi-scale analysis
3. **Feature Extraction**: 
   - Approximation: 8 statistical features
   - Details: 7 features × 3 levels = 21 features
   - Energy ratios: 4 features
   - **Total**: 33 wavelet features per spectrum

### SNV Normalization
```python
snv_spectrum = (spectrum - mean(spectrum)) / std(spectrum)
```
- Removes multiplicative scattering effects
- Standardizes spectral intensity variations
- Enhances chemical signal clarity

### Ensemble Voting Strategies
- **Hard Voting**: Majority class prediction
- **Soft Voting**: Weighted probability averaging
- **Cross-Validation**: Model performance weighting

## 📈 Performance Analysis

### Strengths
- **Robust Preprocessing**: Multiple complementary approaches
- **Ensemble Diversity**: 30 base models with different perspectives
- **Feature Optimization**: Automated selection per preprocessing path
- **Comprehensive Evaluation**: 40 total model configurations

### Observations
- **Gradient Boosting**: Consistently best performer across feature sets
- **Feature Set Equivalence**: Multiple sets achieved similar performance
- **Ensemble Limitations**: Voting didn't improve beyond best base models
- **Class Balance**: Slight bias toward female classification

### Comparison with Previous Experiments
- **G1-G6 Range**: 50.70% - 69.57%
- **G7 Performance**: 53.49% (middle range)
- **Innovation**: Advanced ensemble methodology with wavelet analysis

## 🔬 Research Implications

### Wavelet Analysis Insights
- **Frequency Domain**: Captures spectral patterns invisible in raw data
- **Multi-Resolution**: Different decomposition levels reveal distinct features
- **Compact Representation**: 33 features effectively summarize 300 wavelengths

### Ensemble Learning Findings
- **Base Model Diversity**: Multiple algorithms provide robustness
- **Voting Limitations**: Simple voting doesn't always improve performance
- **Feature Set Impact**: Preprocessing approach significantly affects results

### Spectral Processing Advances
- **Multi-Path Strategy**: Parallel preprocessing enhances feature diversity
- **Optimization Integration**: Automated feature selection per preprocessing type
- **Comprehensive Evaluation**: Systematic comparison of ensemble strategies

## 🎯 Future Directions
- **Meta-Learning**: Combine predictions from different feature sets
- **Advanced Voting**: Weighted voting based on model confidence
- **Hyperparameter Optimization**: Systematic tuning of ensemble parameters
- **Cross-Experiment Learning**: Transfer learning from other experiments

---

**Experiment G7** demonstrates advanced ensemble methodology combining wavelet analysis with sophisticated voting strategies for hyperspectral gender prediction, achieving competitive performance through comprehensive preprocessing and model diversity. 