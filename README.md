# HSI Egg Classification Research

## Project Overview
Hyperspectral imaging (HSI) analysis for gender and mortality prediction in pre-incubation chicken eggs. Part of summer research at UIUC Agricultural Engineering Department.

## Current Status
- Data converted from Excel to CSV format
- Baseline models trained for gender and mortality classification
- Project structure established with data security protocols

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

## Planned Experiments

### Preprocessing Methods to Test
- Standard Normal Variate (SNV)
- Multiplicative Scatter Correction (MSC)
- Savitzky-Golay filtering
- First/second derivatives

### Algorithms to Implement
- XGBoost
- CatBoost
- PLS-DA (Partial Least Squares Discriminant Analysis)
- Neural Networks

### Analysis Approaches
- SHAP for explainable AI
- Recursive Feature Elimination (RFE)
- Principal Component Analysis (PCA)
- SMOTE for class imbalance

## Research Questions

1. Which preprocessing combinations work best for HSI egg data?
2. What are the most discriminative wavelengths for gender/mortality?
3. How does prediction accuracy vary across incubation days?
4. Can we achieve >90% accuracy benchmark?

## Data Security
- Original files and processed data excluded from version control
- Only code and documentation tracked in git
- Model files stored locally only

---
Last updated: June 2025 