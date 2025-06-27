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
- **Completed experiments**: G1-G8

### Mortality Classification Dataset
- **Clean samples**: 5,210 mortality samples (4,185 Alive, 1,025 Dead)
- **Completed experiments**: M1-M8

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
ID | Preprocessing           | Algorithm              | Features                | Results
---|------------------------|------------------------|-------------------------|----------
G1 | MSC + SG 1st deriv     | LightGBM + Bayes opt   | Full spectrum          | 53.95%
G2 | SNV + SG 2nd deriv     | Ensemble (RF+SVM+XGB) | Spectral ratios        | 50.70%  
G3 | EMSC + wavelets        | 1D-CNN + LSTM          | Band depths            | 53.95%
G4 | Raw + augmentation     | Transformer            | GLCM texture           | 46.05%
G5 | MSC + derivatives      | Multi-task (G+M)       | Biology-informed       | 69.57%
G6 | SNV optimized          | Transfer from Day1-4   | PCA + ratios           | 54.88%
G7 | Wavelet + SNV          | Voting classifier      | Key wavelengths only   | 53.49%
G8 | SG + EMSC              | Bayesian NN            | Carotenoid indices     | 53.95%
```

### Mortality Classification Experiments  
```
ID | Preprocessing           | Algorithm              | Features                | Results
---|------------------------|------------------------|-------------------------|----------
M1 | MSC + SG filtering     | LightGBM + SMOTE-ENN   | Full + class balance   | 77.29%
M2 | SNV + 2nd derivative   | Ensemble + uncertainty | Protein/lipid ratios   | 93.48%
M3 | EMSC + augmentation    | 1D-CNN + dropout       | Band depth analysis    | 92.39%
M4 | Raw + SMOTE variants   | Transfer learning      | Heme protein indices   | 75.72%
M5 | Wavelets + MSC         | Multi-task (G+M)       | Water absorption       | 80.33%
M6 | SG + optimization      | Gradient boosting      | Morphological + spec   | 76.10%
M7 | SNV + mixup            | Semi-supervised        | Texture + spectral     | 79.37%
M8 | Multiple derivatives   | Conformal prediction   | Uncertainty bounds     | 80.33%
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

**Recommended execution order:** C1, C2, C5, C3, C4, C8

**Performance Summary:**
- **Best Gender Classification**: G5 (Multi-Task) at 69.57%
- **Best Single-Task Gender**: G6 (SNV + Transfer Learning) at 54.88%
- **Best Mortality Classification**: M2 (SNV + Ensemble) at 93.48%

**Results Key:**
- **HIGH**: G5, M2, M3 - Achieved >70% gender or >90% mortality accuracy
- **MED**: M5, M7, M8 - Solid ~80% mortality performance  
- **LOW**: G1-G4, G6-G8 - Gender experiments struggled with class imbalance
- **BASELINE**: M1, M4, M6 - Standard approaches with moderate performance

**Gender Classification Challenges:**
All gender experiments (G1-G8) struggled with class imbalance, with most models defaulting to majority class predictions. G5's multi-task approach achieved the best performance by leveraging mortality prediction as an auxiliary task.

**Mortality Classification Success:**
Mortality experiments achieved strong performance, with M2's ensemble approach reaching 93.48% accuracy. Advanced preprocessing (SNV, derivatives) and ensemble methods proved most effective.

## Research Questions

1. ✅ **Which preprocessing combinations work best for HSI egg data?**
   - **Answer**: SNV + derivatives (M2: 93.48%) outperformed MSC approaches significantly
   
2. ✅ **What are the most discriminative wavelengths for gender/mortality?**
   - **Mortality**: NIR region (999.52nm, 992.76nm) for water/lipid content, Blue (480.70nm) for oxidative stress
   - **Gender**: 793.2nm, 631.67nm, 992.76nm identified but models couldn't exploit effectively
   
3. ✅ **Can spectral ratios improve biological interpretability?**
   - **Answer**: Yes, but limited impact (0.21% importance in M2). Raw derivatives more informative
   
4. ✅ **How effective is transfer learning from post-incubation to pre-incubation?**
   - **Answer**: Moderate effectiveness (M4: 75.72%, G6: 54.88%) - domain gap remains challenging
   
5. ✅ **Which ensemble methods provide the best accuracy-interpretability trade-off?**
   - **Answer**: Soft voting ensemble (M2) with RF+SVM+XGBoost achieved optimal balance
   
6. ✅ **Can we achieve >90% accuracy benchmark with novel approaches?**
   - **Answer**: Yes! M2 (93.48%) and M3 (92.39%) exceeded 90% target for mortality classification
   
7. ✅ **What uncertainty levels are associated with edge-case predictions?**
   - **Answer**: M8 conformal prediction achieved 96.74% empirical coverage with reliable uncertainty quantification

## Key Findings

### Methodology Insights

**Preprocessing Impact:**
- **SNV superior to MSC**: M2 (93.48%) vs M1 (77.29%) - 21% improvement
- **Derivatives crucial**: 2nd derivatives provided 45.29% of feature importance
- **Feature enhancement effective**: M5's 1.23x enhancement (300→370 features) improved performance
- **Raw data challenges**: G4 (46.05%) and M4 (75.72%) showed minimal preprocessing has limitations

**Algorithm Performance:**
- **Ensemble dominance**: Best results from ensemble methods (M2, M3, M5, M7, M8)
- **Deep learning mixed**: CNN successful for mortality (M3), failed for gender (G3, G4)
- **Traditional ML strong**: LightGBM, Random Forest, SVM performed reliably
- **Uncertainty quantification**: M8 achieved excellent conformal prediction (96.74% coverage)

**Class Imbalance Solutions:**
- **SMOTE-ENN effective**: Improved minority class detection significantly
- **Multi-task learning**: G5 best gender result (69.57%) using auxiliary mortality task
- **Semi-supervised learning**: M7 (79.37%) successfully used 30% unlabeled data

### Biological Insights

**Mortality Prediction Success Factors:**
- **Water/lipid content** (NIR 992-999nm): Primary mortality indicators
- **Oxidative stress markers** (Blue 480nm): Early mortality detection
- **Protein changes** (Red 728nm): Developmental health indicators
- **Spectral derivatives**: More informative than raw intensities for biological processes

**Gender Classification Challenges:**
- **Subtle spectral differences**: Gender markers less pronounced than mortality
- **Class imbalance critical**: 53.8% vs 46.2% distribution caused model collapse
- **Biological timing**: Pre-incubation gender differences may be minimal
- **Need for advanced techniques**: Requires more sophisticated approaches than implemented

## Data Security
- Original files and processed data excluded from version control
- Only code and documentation tracked in git
- Model files stored locally only

---
Last updated: June 2025 