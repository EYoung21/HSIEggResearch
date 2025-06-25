# HSI Egg Classification Research

## Project Overview
Hyperspectral imaging (HSI) analysis for gender and mortality prediction in pre-incubation chicken eggs. Part of summer research at UIUC Agricultural Engineering Department.

## Current Status
- Data converted from Excel to CSV format
- Baseline models trained for gender and mortality classification
- Project structure established with data security protocols
- Literature review completed identifying unexplored research directions

## Data Summary
- **Total samples**: 699 eggs with metadata
- **Spectral data**: 300 wavelengths (374-1015 nm)
- **Time points**: Days 0-4 of incubation
- **Analysis focus**: Day 0 (pre-incubation)

### Gender Classification Dataset
- **Clean samples**: 1,074 eggs (578 Female, 496 Male)
- **Baseline accuracy**: ~50-53% (Random Forest/SVM)

### Mortality Classification Dataset
- **Clean samples**: 1,247 eggs (1,023 Live, 224 Dead)
- **Baseline accuracy**: ~71-82% (Random Forest/SVM)

## Project Structure
```
HSIResearch/
├── data/
│   ├── reference_metadata.csv          # Egg metadata
│   ├── spectral_data_D0.csv           # Day 0 HSI data
│   ├── spectral_data_D1-D4.csv        # Days 1-4 HSI data
│   └── processed/                      # Train/test splits
├── models/                             # Trained model files
├── convert_data_to_csv.py             # Excel to CSV conversion
├── data_preparation_fixed.py          # Data cleaning and preparation
├── starter_model.py                   # Baseline modeling pipeline
└── .gitignore                         # Data security protection
```

## Literature Review Findings

### Performance Benchmarks
- **Industry standard**: 98% accuracy for commercial deployment
- **Research target**: >90% accuracy for publication
- **Current baseline**: 53% (gender), 82% (mortality)

### Key Wavelength Regions
- **570 nm**: Carotenoids (yolk pigmentation)
- **640 nm**: Heme proteins (blood formation)
- **660 nm**: Protein absorption
- **687 nm**: Water molecules
- **748 nm**: Lipids
- **771 nm**: Protein-lipid interactions

## Unexplored Research Directions

*Based on analysis of Professor Kamruzzaman's published work, these approaches haven't been systematically explored for pre-incubation gender/mortality prediction:*

### High-Priority Preprocessing Techniques
- **MSC + Savitzky-Golay combination**: Professor used separately, not combined
- **Extended Multiple Scatter Correction (EMSC)**: More advanced than standard MSC
- **SNV with optimized window sizes**: Systematic parameter optimization not done
- **Wavelet transform preprocessing**: Frequency-domain decomposition
- **Spectral derivative combinations**: Second derivatives, mixed orders

### Advanced Machine Learning Approaches
- **LightGBM**: Often outperforms XGBoost/CatBoost on spectral data
- **Ensemble voting classifiers**: RF + SVM + Gradient boosting combinations
- **1D-CNN + LSTM**: Sequential pattern recognition in wavelengths
- **Transformer architectures**: Attention mechanisms for spectral relationships
- **Multi-task learning**: Single model predicting both gender AND mortality
- **Semi-supervised learning**: Leveraging unlabeled egg data

### Novel Feature Engineering
- **Spectral ratio indices**: Meaningful wavelength ratios (e.g., 570nm/640nm)
- **Normalized Difference Spectral Index (NDSI)**: Similar to vegetation indices
- **Band depth analysis**: Absorption depth measurements
- **Texture-based spectral features**: GLCM, LBP on spectral images
- **Biology-informed features**: Carotenoid/protein absorption ratios

### Advanced Optimization Strategies
- **Bayesian optimization**: More efficient hyperparameter tuning
- **Neural Architecture Search (NAS)**: Automated network design
- **Multi-objective optimization**: Balance accuracy with interpretability
- **Transfer learning**: Day 1-4 models as feature extractors for Day 0

### Data Enhancement Techniques
- **SMOTE variants**: BorderlineSMOTE, ADASYN for class imbalance
- **Spectral data augmentation**: Gaussian noise, wavelength shifting
- **Mixup augmentation**: Synthetic sample generation
- **Uncertainty quantification**: Monte Carlo dropout, ensemble variance

### Explainable AI Beyond Current Methods
- **LIME**: Local explanations for individual predictions
- **Integrated Gradients**: Attribution methods for deep learning
- **Wavelength importance mapping**: Biological interpretation of key regions
- **Causal inference**: Identify spectral markers of future development

## Immediate High-Impact Experiments

*Prioritized for maximum accuracy improvement:*

1. **MSC + SG preprocessing** (expected +10-15% accuracy boost)
2. **LightGBM with Bayesian optimization** (modern gradient boosting)
3. **Spectral ratio features** (biologically meaningful)
4. **Ensemble voting classifier** (RF + SVM + XGBoost)
5. **SMOTE-ENN for mortality imbalance** (improve minority class detection)
6. **Transfer learning from post-incubation data** (leverage Day 1-4 patterns)

## Planned Experiments

### Preprocessing Methods to Test
- Standard Normal Variate (SNV) with parameter optimization
- Multiplicative Scatter Correction (MSC) + Savitzky-Golay combinations
- Extended Multiple Scatter Correction (EMSC)
- Wavelet transform preprocessing
- First/second derivative combinations

### Algorithms to Implement
- LightGBM with Bayesian hyperparameter optimization
- Ensemble voting classifiers (RF + SVM + XGBoost)
- 1D-CNN + LSTM for sequential patterns
- Multi-task learning models
- PLS-DA with optimized components
- Transfer learning from post-incubation models

### Analysis Approaches
- SHAP + LIME for comprehensive explainable AI
- Recursive Feature Elimination (RFE) with biology-informed constraints
- Principal Component Analysis (PCA) with spectral ratio features
- SMOTE variants for advanced class imbalance handling
- Uncertainty quantification for prediction confidence

## Research Questions

1. Which preprocessing combinations work best for HSI egg data?
2. What are the most discriminative wavelengths for gender/mortality?
3. Can spectral ratios improve biological interpretability?
4. How effective is transfer learning from post-incubation to pre-incubation?
5. Which ensemble methods provide the best accuracy-interpretability trade-off?
6. Can we achieve >90% accuracy benchmark with novel approaches?
7. What uncertainty levels are associated with edge-case predictions?

## Data Security
- Original files and processed data excluded from version control
- Only code and documentation tracked in git
- Model files stored locally only

---
Last updated: June 2025 