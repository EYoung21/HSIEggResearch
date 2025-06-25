# G5 Experiment: MSC + Multi-Scale Derivatives + Multi-Task Learning

## 🎯 Overview

The **G5 experiment** implements advanced **multi-task deep learning** for simultaneous gender and mortality prediction from hyperspectral imaging (HSI) data of chicken eggs. This experiment combines the proven MSC preprocessing from G1 with enhanced multi-scale derivative features and a sophisticated multi-task neural architecture.

**Key Innovation**: Shared feature extraction with task-specific prediction heads, enabling the model to learn common spectral patterns while capturing task-specific information for both gender and mortality prediction.

---

## 🧬 Methodology

### Preprocessing Pipeline
- **MSC (Multiplicative Scatter Correction)**: Light scattering correction (proven successful in G1)
- **Multi-Scale Savitzky-Golay Derivatives**: 
  - Original spectra (baseline features)
  - 1st derivative (spectral slopes)
  - 2nd derivative (curvature patterns)  
  - 3rd derivative (fine spectral variations)
- **Feature Scaling**: 4x feature expansion (300 → 1,200 features)
- **Standardization**: Z-score normalization for stable training

### Multi-Task Architecture
```
Input (1200 features)
    ↓
Shared Feature Extraction
├── Dense(512) + BatchNorm + Dropout
├── Dense(256) + BatchNorm + Dropout  
├── Dense(128) + BatchNorm + Dropout
└── Shared Representation(64)
    ↓
Task-Specific Heads
├── Gender Head → Dense(32) → Dense(1, sigmoid)
└── Mortality Head → Dense(32) → Dense(1, sigmoid)
```

### Advanced Features
- **Shared Learning**: Common spectral features benefit both tasks
- **Task-Specific Heads**: Capture unique patterns for each prediction
- **Weighted Loss Functions**: Balance training between tasks
- **Batch Normalization**: Stable training with derivative features
- **Dropout Regularization**: Prevent overfitting in multi-task setting

---

## 📊 Dataset

- **Training Set**: ~859 samples (stratified by joint gender×mortality labels)
- **Test Set**: ~215 samples (balanced representation)
- **Tasks**: 
  - Task 1: Gender Classification (Male/Female)
  - Task 2: Mortality Prediction (Live/Dead)
- **Input Features**: 1,200 (300 wavelengths × 4 derivative orders)
- **Joint Labels**: 4 combinations (Male-Live, Male-Dead, Female-Live, Female-Dead)

---

## 🏗️ Architecture Details

### Shared Feature Extraction
- **Purpose**: Learn common spectral representations useful for both tasks
- **Layers**: 3-4 dense layers with decreasing sizes (512→256→128→64)
- **Regularization**: L2 penalty + batch normalization + dropout
- **Activation**: ReLU for non-linear feature learning

### Task-Specific Heads
- **Gender Head**: Focused on sex-related spectral differences
- **Mortality Head**: Captures vitality-related patterns
- **Architecture**: Dense(32) + BatchNorm + Dropout + Dense(1)
- **Output**: Sigmoid activation for binary classification

### Loss Function
```python
Total Loss = w₁ × Binary_CrossEntropy(Gender) + w₂ × Binary_CrossEntropy(Mortality)
```
- **Task Weights**: Configurable (default: w₁=1.0, w₂=1.0)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Metrics**: Individual accuracy for each task + combined score

---

## 📁 File Structure

```
G5_MSC_Derivatives_MultiTask/
├── G5_preprocessing.py           # Multi-scale derivatives preprocessing
├── G5_model.py                  # Multi-task deep learning model
├── run_G5_experiment.py         # Complete experiment pipeline
├── requirements.txt             # Python dependencies
├── README.md                    # This documentation
└── Generated Files:
    ├── X_train_processed.npy    # Multi-scale training features
    ├── X_test_processed.npy     # Multi-scale test features
    ├── y_gender_train.npy       # Gender training labels
    ├── y_gender_test.npy        # Gender test labels
    ├── y_mortality_train.npy    # Mortality training labels  
    ├── y_mortality_test.npy     # Mortality test labels
    ├── multi_scale_preprocessor.pkl  # Fitted preprocessor
    ├── gender_encoder.pkl       # Gender label encoder
    ├── mortality_encoder.pkl    # Mortality label encoder
    ├── feature_info.json        # Feature metadata
    ├── G5_multitask_model.h5    # Trained multi-task model
    ├── G5_model_metadata.json   # Model architecture info
    ├── test_predictions.csv     # Multi-task predictions
    ├── G5_experimental_results.json  # Complete results
    └── G5_performance_summary.txt    # Human-readable summary
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd gender/G5_MSC_Derivatives_MultiTask
pip install -r requirements.txt
```

### 2. Run Complete Experiment
```bash
python run_G5_experiment.py
```

### 3. Run Individual Components
```bash
# Preprocessing only
python G5_preprocessing.py

# Modeling only (requires preprocessed data)
python G5_model.py
```

---

## 🔬 Key Innovations

### 1. Multi-Task Learning
- **Advantage**: Leverages correlations between gender and mortality
- **Implementation**: Shared feature extraction + task-specific heads
- **Benefit**: Improved feature utilization and potential performance gains

### 2. Multi-Scale Derivative Features
- **Original**: Baseline spectral information
- **1st Derivative**: Spectral slopes and peak positions
- **2nd Derivative**: Curvature and band characteristics
- **3rd Derivative**: Fine spectral variations and noise patterns

### 3. Advanced Architecture
- **Shared Layers**: Learn common biological signatures
- **Task Heads**: Capture gender-specific and mortality-specific patterns
- **Regularization**: Prevent overfitting with multi-scale features

### 4. Stratified Multi-Task Splitting
- **Challenge**: Balance both gender AND mortality in train/test splits
- **Solution**: Combined stratification on joint labels
- **Result**: Balanced representation for all gender×mortality combinations

---

## 📈 Expected Outcomes

### Performance Targets
- **Individual Tasks**: >50% accuracy (better than random)
- **Combined Score**: Average of both task accuracies
- **Comparative**: Compare against single-task G1-G4 experiments

### Analysis Dimensions
- **Task Performance**: Individual accuracy for gender and mortality
- **Joint Analysis**: Performance on gender×mortality combinations
- **Feature Importance**: Shared vs task-specific feature patterns
- **Training Dynamics**: Convergence patterns for multi-task loss

---

## 🔍 Hyperparameter Optimization

### Grid Search Parameters
- **Hidden Layers**: [512,256,128], [256,128,64], [512,256], etc.
- **Dropout Rate**: 0.2, 0.25, 0.3, 0.4
- **Learning Rate**: 0.0005, 0.001, 0.002
- **Task Weights**: Balanced (1.0,1.0), Gender-priority (1.2,0.8), Mortality-priority (0.8,1.2)

### Optimization Strategy
- **Method**: 3-fold cross-validation with stratified sampling
- **Metric**: Combined accuracy (average of both tasks)
- **Trials**: 8-10 parameter combinations
- **Early Stopping**: Prevent overfitting during search

---

## 🎯 Technical Advantages

### 1. Proven Preprocessing
- **MSC**: Successful light scattering correction from G1
- **SG Derivatives**: Robust spectral feature enhancement
- **Multi-Scale**: Comprehensive frequency domain coverage

### 2. Advanced Deep Learning
- **Multi-Task**: Simultaneous prediction improves efficiency
- **Shared Learning**: Common biological patterns benefit both tasks
- **Regularization**: Batch normalization + dropout for stability

### 3. Comprehensive Evaluation
- **Individual Metrics**: Task-specific performance analysis
- **Joint Analysis**: Gender×mortality combination insights
- **Statistical Rigor**: Cross-validation and stratified sampling

---

## 🔮 Future Directions

### Model Enhancements
- **Attention Mechanisms**: Focus on task-relevant spectral regions
- **Dynamic Task Weighting**: Adaptive loss balancing during training
- **Ensemble Multi-Task**: Combine multiple architectures

### Feature Engineering
- **Wavelength Selection**: Task-specific important regions
- **Cross-Task Features**: Interaction terms between tasks
- **Time Series**: Multi-day spectral evolution patterns

### Applications
- **Transfer Learning**: Apply to other biological classification tasks
- **Hierarchical Tasks**: Multi-level prediction (species→gender→mortality)
- **Real-Time Inference**: Optimize for production deployment

---

## 📚 References & Related Work

### Previous Experiments
- **G1**: MSC + SG 1st derivative + LightGBM (53.95% accuracy)
- **G2**: SNV + SG 2nd derivative + Ensemble (50.70% accuracy)
- **G3**: EMSC + Wavelets + CNN (53.95% accuracy, model collapse)
- **G4**: Raw + Augmentation + Transformer (incomplete, computational issues)

### Scientific Context
- **Multi-Task Learning**: Caruana (1997), Ruder (2017)
- **Hyperspectral Classification**: Ghamisi et al. (2017)
- **Savitzky-Golay Derivatives**: Rinnan et al. (2009)
- **Poultry Gender Prediction**: Various HSI studies 2018-2024

---

## 📞 Contact & Support

For questions about the G5 experiment implementation:
- Check `G5_performance_summary.txt` for detailed results
- Review `G5_experimental_results.json` for raw data
- Examine training logs for debugging
- Compare with previous G1-G4 experiment outcomes

**Experiment Priority**: HIGH - Multi-task learning represents significant methodological advancement for simultaneous gender and mortality prediction from hyperspectral egg data. 