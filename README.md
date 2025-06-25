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

## Experimental Design Matrix

*Systematic approach to testing unexplored combinations for maximum accuracy improvement:*

### Gender Classification Experiments
```
ID | Preprocessing           | Algorithm              | Features                | Priority
---|------------------------|------------------------|-------------------------|----------
G1 | MSC + SG 1st deriv     | LightGBM + Bayes opt   | Full spectrum          | HIGH
G2 | SNV + SG 2nd deriv     | Ensemble (RF+SVM+XGB) | Spectral ratios        | HIGH  
G3 | EMSC + wavelets        | 1D-CNN + LSTM          | Band depths            | MED
G4 | Raw + augmentation     | Transformer            | GLCM texture           | MED
G5 | MSC + derivatives      | Multi-task (G+M)       | Biology-informed       | HIGH
G6 | SNV optimized          | Transfer from Day1-4   | PCA + ratios           | MED
G7 | Wavelet + SNV          | Voting classifier      | Key wavelengths only   | LOW
G8 | SG + EMSC              | Bayesian NN            | Carotenoid indices     | LOW
```

### Mortality Classification Experiments  
```
ID | Preprocessing           | Algorithm              | Features                | Priority
---|------------------------|------------------------|-------------------------|----------
M1 | MSC + SG filtering     | LightGBM + SMOTE-ENN   | Full + class balance   | HIGH
M2 | SNV + 2nd derivative   | Ensemble + uncertainty | Protein/lipid ratios   | HIGH
M3 | EMSC + augmentation    | 1D-CNN + dropout       | Band depth analysis    | MED
M4 | Raw + SMOTE variants   | Transfer learning      | Heme protein indices   | MED
M5 | Wavelets + MSC         | Multi-task (G+M)       | Water absorption       | HIGH
M6 | SG + optimization      | Gradient boosting      | Morphological + spec   | MED
M7 | SNV + mixup            | Semi-supervised        | Texture + spectral     | LOW
M8 | Multiple derivatives   | Conformal prediction   | Uncertainty bounds     | LOW
```

### Combined (Gender + Mortality) Experiments
```
ID | Preprocessing           | Algorithm              | Features                | Priority  
---|------------------------|------------------------|-------------------------|----------
C1 | MSC + SG combination   | Multi-task deep net    | Shared representations | HIGH
C2 | SNV + derivatives      | Ensemble multi-output  | Biology-informed       | HIGH
C3 | EMSC + wavelets        | Transfer + fine-tune   | Cross-task features    | MED
C4 | Augmentation suite     | Attention mechanisms   | Wavelength importance  | MED
C5 | Optimized preprocessing| Hierarchical learning  | Task-specific heads    | HIGH
C6 | Multiple corrections   | Meta-learning          | Few-shot adaptation    | LOW
C7 | Advanced derivatives   | Neural architecture    | Automated design       | LOW
C8 | Domain adaptation      | Uncertainty + ensemble | Confidence intervals   | MED
```

**Priority Key:**
- **HIGH**: Expected >10% accuracy improvement, established techniques
- **MED**: Moderate improvement potential, some novelty  
- **LOW**: Exploratory, high novelty but uncertain gains

**Recommended execution order:** G1, M1, C1, G2, M2, C2

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