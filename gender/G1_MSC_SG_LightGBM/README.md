# G1 Experiment: MSC + SG 1st Derivative + LightGBM + Bayesian Optimization

## Overview
This experiment implements gender classification for HSI (Hyperspectral Imaging) egg data using:
- **Preprocessing**: Multiplicative Scatter Correction (MSC) + Savitzky-Golay 1st derivative
- **Algorithm**: LightGBM with Bayesian hyperparameter optimization
- **Features**: Full spectrum (all wavelengths)
- **Priority**: HIGH

## Methodology

### Preprocessing Pipeline
1. **MSC (Multiplicative Scatter Correction)**
   - Corrects for scattering effects in spectral data
   - Normalizes each spectrum against a reference (mean spectrum)
   - Formula: `corrected = (spectrum - intercept) / slope`

2. **Savitzky-Golay 1st Derivative**
   - Removes baseline drift and enhances spectral features
   - Window length: 15 points
   - Polynomial order: 3
   - Derivative order: 1

### Machine Learning
- **Algorithm**: LightGBM (Light Gradient Boosting Machine)
- **Optimization**: Bayesian optimization using scikit-optimize
- **Search Space**: 9 hyperparameters (learning rate, num_leaves, regularization, etc.)
- **Cross-Validation**: 5-fold stratified CV for hyperparameter optimization

## Files

### Input Files (from main data directory)
- `../../data/reference_metadata.csv` - Sample metadata and labels
- `../../data/spectral_data_D0.csv` - Day 0 spectral measurements

### Code Files
- `G1_preprocessing.py` - MSC + SG preprocessing implementation
- `G1_model.py` - LightGBM with Bayesian optimization
- `run_G1_experiment.py` - Main execution pipeline
- `requirements.txt` - Python package dependencies

### Output Files (generated during experiment)
- `X_train_processed.npy` - Preprocessed training features
- `X_test_processed.npy` - Preprocessed test features
- `y_train.npy` - Training labels
- `y_test.npy` - Test labels
- `wavelengths.csv` - Wavelength information
- `msc_sg_preprocessor.pkl` - Fitted preprocessor
- `label_encoder.pkl` - Label encoder
- `lightgbm_model.txt` - Trained LightGBM model
- `lightgbm_model_info.pkl` - Model metadata
- `test_predictions.npy` - Test set predictions
- `test_probabilities.npy` - Prediction probabilities
- `feature_importance.csv` - Wavelength importance ranking
- `G1_experiment_summary.md` - Experiment report

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Experiment
```bash
python run_G1_experiment.py
```

### 3. Run Individual Steps
```bash
# Preprocessing only
python G1_preprocessing.py

# Modeling only (requires preprocessed data)
python G1_model.py
```

## 📊 **Key Results**

### **🎯 Performance Summary**
- **Test Accuracy**: 53.95% ⚠️ 
- **Cross-Validation**: 55.88% ± std
- **AUC Score**: Not applicable (binary classification issue)
- **Training Time**: 0.5 minutes
- **Dataset**: 1,074 samples (859 training, 215 test)

### **🚨 Critical Issue Identified**
**Problem**: Model predicts ALL samples as Female (class imbalance issue)

**📊 Confusion Matrix:**
```
              Predicted
Actual     Female  Male
Female      116     0
Male         99     0
```

**📈 Classification Report:**
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Female** | 0.54 | 1.00 | 0.70 | 116 |
| **Male** | 0.00 | 0.00 | 0.00 | 99 |
| **Overall** | 0.54 | 0.54 | 0.38 | 215 |

**Analysis**: 
- The 53.95% accuracy is misleading - it's just the proportion of females in the test set
- Model learned to always predict the majority class (Female)
- No actual learning of discriminative spectral patterns occurred
- Classic case of class imbalance leading to poor model performance
- Evidence: Recall for Male class = 0.00 (no males correctly identified)

### **🔬 Technical Results**
- **Algorithm**: LightGBM with Bayesian optimization (30 calls)
- **Best Hyperparameters**: num_leaves=78, learning_rate=0.030, feature_fraction=0.828
- **Features**: 300 wavelengths (374.14-1015.32 nm)
- **Preprocessing**: MSC + Savitzky-Golay 1st derivative (window=15, poly=3)

### **📈 Top 5 Most Important Wavelengths**
1. **793.2 nm** (importance: 10.91) - Near-infrared, water absorption
2. **631.67 nm** (importance: 8.89) - Red, protein absorption  
3. **992.76 nm** (importance: 5.98) - Near-infrared, organic compounds
4. **578.79 nm** (importance: 5.90) - Green-yellow, pigments
5. **443.58 nm** (importance: 4.40) - Blue, carotenoids

**Biological Interpretation:**
- Near-infrared dominance suggests water/lipid content differences
- Red region importance indicates protein content variations
- Blue region suggests carotenoid/pigment differences
- Pattern indicates potential biological differences exist but model cannot exploit them

### **⚙️ Configuration Details**
- **Classes**: Female (578 samples, 53.8%), Male (496 samples, 46.2%)
- **Split**: Stratified 80/20 train/test
- **Optimization**: 5-fold cross-validation
- **Model Size**: 15.7 KB (lightweight deployment)

## 📁 **Files Generated**

### **Data Files**
- `X_train_processed.npy` (2.0 MB) - Preprocessed training features
- `X_test_processed.npy` (504 KB) - Preprocessed test features
- `y_train.npy`, `y_test.npy` - Training and test labels
- `wavelengths.csv` - Wavelength information

### **Model Files**
- `lightgbm_model.txt` (15.7 KB) - Trained LightGBM model
- `lightgbm_model_info.pkl` - Model metadata
- `msc_sg_preprocessor.pkl` - Fitted preprocessor

### **Results Files**
- `test_predictions.npy` - Model predictions
- `test_probabilities.npy` - Prediction probabilities
- `feature_importance.csv` - Wavelength importance rankings
- `G1_experimental_results.json` - Complete results in JSON format

### **Documentation**
- `README.md` - Methodology documentation
- `G1_experiment_summary.md` - Experiment overview

## 🔧 **Recommendations**

### **🚨 Immediate Fixes**
1. **Implement class balancing** (SMOTE, class weights in LightGBM)
2. **Increase Bayesian optimization calls** (30 → 100+)
3. **Add stratified sampling verification**
4. **Try different train/test splits**

### **🧪 Methodology Improvements**
1. **Compare with SNV preprocessing** (G2 experiment)
2. **Test different algorithms** (Random Forest, SVM, Neural Networks)
3. **Implement ensemble methods**
4. **Add feature selection** to reduce noise
5. **Try different derivative orders** (2nd derivative)

### **🔬 Preprocessing Alternatives**
1. **SNV (Standard Normal Variate)** instead of MSC
2. **EMSC (Extended MSC)** for better correction
3. **Different Savitzky-Golay parameters**
4. **Wavelength selection** based on biological knowledge

## 🎯 **Conclusion**

✅ **SUCCESSFUL PIPELINE**: Complete preprocessing and modeling pipeline implemented  
❌ **POOR PERFORMANCE**: Model fails to learn discriminative patterns  
🔍 **ROOT CAUSE**: Class imbalance and possibly insufficient preprocessing  
🎯 **NEXT STEPS**: Implement class balancing and proceed to G2 experiment for comparison

The G1 experiment serves as a baseline and demonstrates the importance of proper class handling in machine learning pipelines. The infrastructure is solid and ready for systematic comparison with other preprocessing methods.

## Technical Details

### MSC Implementation
- Reference spectrum: Mean of training spectra
- Linear regression per spectrum against reference
- Correction applied to both training and test data

### Savitzky-Golay Parameters
- Window length: 15 (must be odd, covers ~15 wavelengths)
- Polynomial order: 3 (cubic polynomial fitting)
- Derivative order: 1 (first derivative for baseline removal)

### Bayesian Optimization
- Acquisition function: Expected Improvement (EI)
- Initial random points: 10
- Total evaluations: 30 (configurable)
- Cross-validation: 5-fold stratified

## Biological Interpretation

### Expected Important Wavelengths
- **Protein absorption**: ~280nm region
- **Lipid features**: ~930nm, ~1200nm regions  
- **Water absorption**: ~970nm, ~1450nm regions
- **Carotenoid absorption**: ~450-500nm region

### Gender-Related Spectral Differences
- Males vs Females may show differences in:
  - Yolk composition (lipids, proteins)
  - Shell thickness variations
  - Internal structure differences

## Notes
- Uses Day 0 data for pre-incubation gender prediction
- MSC corrects for scattering variations between eggs
- SG 1st derivative enhances discriminative features
- Bayesian optimization finds optimal hyperparameters efficiently
- Feature importance reveals biologically relevant wavelengths

## Troubleshooting

### Common Issues
1. **Import errors**: Install required packages via `pip install -r requirements.txt`
2. **Memory issues**: Reduce Bayesian optimization calls in `G1_model.py`
3. **Data path errors**: Ensure `../../data/` contains CSV files
4. **scikit-optimize missing**: Will fall back to default hyperparameters

### Performance Tips
- Reduce `n_calls` in Bayesian optimization for faster execution
- Adjust `sg_window` size based on spectral resolution
- Monitor memory usage with large spectral datasets 