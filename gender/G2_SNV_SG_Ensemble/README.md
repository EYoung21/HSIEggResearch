# G2 Experiment: SNV + SG 2nd Derivative + Ensemble

## 🧪 Experiment Overview

**G2** implements an ensemble approach for gender classification using advanced preprocessing and multiple algorithm families. This experiment focuses on **spectral ratios** as features after applying SNV normalization and Savitzky-Golay 2nd derivative preprocessing.

## 📋 Methodology

### Preprocessing Pipeline
1. **SNV (Standard Normal Variate)**: Normalizes each spectrum individually to remove multiplicative scatter effects
2. **Savitzky-Golay 2nd Derivative**: Removes baseline trends and enhances spectral features
3. **Spectral Ratios**: Creates biologically meaningful ratio features from different spectral regions

### Machine Learning Algorithm
- **Ensemble Method**: Combines three complementary algorithms
  - **Random Forest**: Tree-based ensemble with feature importance
  - **Support Vector Machine (SVM)**: Kernel-based classification with RBF kernel
  - **XGBoost**: Gradient boosting with regularization
- **Voting Strategy**: Soft voting (probability averaging)
- **Optimization**: Bayesian optimization for hyperparameter tuning

### Key Features
- **Priority**: HIGH
- **Feature Type**: Spectral ratios (reduced dimensionality)
- **Validation**: 5-fold stratified cross-validation
- **Optimization Calls**: 30 (configurable)

## 🎯 Scientific Rationale

### Why SNV?
- Removes multiplicative scatter effects from sample-to-sample variations
- More robust than MSC for heterogeneous samples
- Normalizes each spectrum to mean=0, std=1

### Why 2nd Derivative?
- Removes baseline drift and linear trends
- Enhances narrow spectral features
- Better separation of overlapping peaks
- Reduces instrument noise effects

### Why Spectral Ratios?
- **Biological Relevance**: Ratios between different spectral regions (protein/lipid, NIR/visible)
- **Instrument Independence**: Reduces systematic instrument variations
- **Dimensionality Reduction**: Fewer but more meaningful features
- **Interpretability**: Direct biological interpretation possible

### Why Ensemble?
- **Diversity**: Combines different algorithm families (tree-based, kernel-based, boosting)
- **Robustness**: Reduces overfitting and improves generalization
- **Performance**: Often outperforms individual models
- **Uncertainty**: Probability averaging provides confidence estimates

## 📁 File Structure

```
G2_SNV_SG_Ensemble/
├── G2_preprocessing.py          # SNV + SG preprocessing + ratio features
├── G2_model.py                  # Ensemble classifier with optimization
├── run_G2_experiment.py         # Complete experiment pipeline
├── requirements.txt             # Dependencies
├── README.md                    # This documentation
└── [Generated Files]
    ├── X_train_processed.npy    # Processed training features
    ├── X_test_processed.npy     # Processed test features
    ├── y_train.npy              # Training labels
    ├── y_test.npy               # Test labels
    ├── spectral_ratio_features.csv  # Feature names
    ├── snv_sg_preprocessor.pkl  # Fitted preprocessor
    ├── label_encoder.pkl        # Label encoder
    ├── G2_ensemble_model.pkl    # Trained ensemble model
    ├── feature_importance.csv   # Feature importance analysis
    ├── test_predictions.csv     # Test predictions
    ├── G2_experimental_results.json  # Complete results
    └── G2_performance_summary.txt    # Summary report
```

## 🚀 Usage

### Quick Start
```bash
# Navigate to G2 directory
cd gender/G2_SNV_SG_Ensemble

# Install dependencies
pip install -r requirements.txt

# Run complete experiment
python run_G2_experiment.py
```

### Step-by-Step Execution
```bash
# 1. Preprocessing only
python G2_preprocessing.py

# 2. Modeling only (after preprocessing)
python G2_model.py
```

## 📊 Expected Outputs

### Performance Metrics
- **Cross-Validation Scores**: Individual models + ensemble
- **Test Accuracy**: Final model performance
- **Confusion Matrix**: Detailed classification breakdown
- **Individual Model Comparison**: RF vs SVM vs XGBoost performance

### Feature Analysis
- **Spectral Ratio Importance**: Most discriminative ratios
- **Biological Interpretation**: Regional importance (protein, lipid, NIR regions)
- **Feature Rankings**: Combined importance from tree-based models

### Model Artifacts
- **Trained Ensemble**: Complete voting classifier
- **Individual Models**: RF, SVM, XGBoost with optimized parameters
- **Preprocessor**: Fitted SNV + SG pipeline for new data

## 🔬 Spectral Ratio Features

The experiment creates biologically meaningful ratios between:

### Regional Ratios
- **Protein/Carotenoid**: Red (600-700nm) / Blue (400-500nm)
- **NIR/Protein**: NIR1 (700-850nm) / Red (600-700nm)
- **Lipid/Protein**: NIR2 (850-1000nm) / Red (600-700nm)
- **High NIR/Low NIR**: NIR2 (850-1000nm) / NIR1 (700-850nm)

### Key Wavelength Ratios
- Selected representative wavelengths: 450, 550, 650, 750, 850, 950 nm
- All pairwise ratios between key wavelengths
- Total: ~21 ratio features (6 regional + 15 wavelength pairs)

## ⚙️ Configuration Options

### Preprocessing Parameters
```python
# In G2_preprocessing.py
sg_window = 15        # Savitzky-Golay window length
sg_polyorder = 3      # Polynomial order for SG filter
```

### Optimization Parameters
```python
# In G2_model.py
n_calls = 30          # Bayesian optimization calls
cv_folds = 5          # Cross-validation folds
voting = 'soft'       # 'soft' or 'hard' voting
```

## 📈 Performance Expectations

Based on the methodology:

### Strengths
- **Robust preprocessing** removes instrument variability
- **Ensemble approach** provides reliable predictions
- **Spectral ratios** enhance biological interpretability
- **Bayesian optimization** finds optimal hyperparameters

### Considerations
- Fewer features may limit information content
- SVM requires feature scaling (handled automatically)
- Ensemble training takes longer than single models
- Performance depends on quality of spectral regions

## 🔧 Troubleshooting

### Common Issues
1. **Missing Data**: Ensure CSV files exist in `../../data/`
2. **Memory Errors**: Reduce `n_calls` in optimization
3. **Long Runtime**: Normal for Bayesian optimization (30+ minutes)
4. **Import Errors**: Install all requirements: `pip install -r requirements.txt`

### Environment Setup
```bash
# For macOS users (XGBoost support)
brew install libomp
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
```

## 📚 References

- **SNV**: Standard Normal Variate for spectral preprocessing
- **Savitzky-Golay**: Digital filtering for spectral smoothing/derivatives
- **Ensemble Learning**: Combining multiple learners for better performance
- **Bayesian Optimization**: Efficient hyperparameter optimization
- **Spectral Ratios**: Chemometric approach for robust analysis

## 🎯 Next Steps

After G2 completion:
1. Compare with G1 results
2. Analyze feature importance for biological insights
3. Consider G3-G8 experiments for comprehensive comparison
4. Optimize spectral regions based on importance rankings
5. Validate on external datasets if available

---

*Part of the HSI Egg Research Gender Classification Study* 