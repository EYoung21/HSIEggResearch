# G3 Experiment: EMSC + Wavelets + CNN

## 🧠 **Advanced Deep Learning for HSI Gender Classification**

This experiment implements a cutting-edge approach combining **Extended Multiplicative Scatter Correction (EMSC)**, **Wavelet Transform**, and **Convolutional Neural Networks (CNN)** for gender classification of chicken eggs using hyperspectral imaging.

---

## 📋 **Experimental Design**

| Component | Method | Details |
|-----------|---------|---------|
| **Preprocessing** | EMSC + Wavelet Transform | Extended scatter correction + Multi-resolution analysis |
| **Algorithm** | Convolutional Neural Network | Deep learning with optimized architecture |
| **Features** | 2D Wavelet Coefficient Grids | Time-frequency domain representation |
| **Optimization** | Grid Search + Cross-Validation | Automated hyperparameter tuning |
| **Priority** | HIGH | Advanced methodology for complex patterns |

---

## 🔬 **Scientific Methodology**

### **1. EMSC (Extended Multiplicative Scatter Correction)**

EMSC is an advanced preprocessing technique that corrects for:

```
Spectrum = Multiplicative_Factor × Reference + Polynomial_Baseline + Residual
```

**Advantages over MSC/SNV:**
- **Polynomial baseline correction** (quadratic trends)
- **Physical constraint enforcement**
- **Robust reference spectrum estimation**
- **Superior scatter artifact removal**

**Mathematical Model:**
```python
# EMSC correction equation
corrected_spectrum = (raw_spectrum - baseline) / multiplicative_factor

# Where baseline includes:
# - Additive offset
# - Linear trend
# - Quadratic curvature
```

### **2. Wavelet Transform (Daubechies 4)**

Wavelets provide **multi-resolution time-frequency analysis**:

**Key Benefits:**
- **Localized features** in both spectral and frequency domains
- **Multi-scale decomposition** (4 levels)
- **Noise reduction** through coefficient thresholding
- **Edge preservation** for sharp spectral features

**Decomposition Structure:**
```
Level 0: Original signal (300 wavelengths)
Level 1: Approximation (150) + Detail (150)
Level 2: Approximation (75) + Detail (75)
Level 3: Approximation (38) + Detail (38)
Level 4: Approximation (19) + Detail (19)
```

### **3. CNN Architecture**

**Optimized Design:**
- **2D Convolutional layers** for spatial pattern recognition
- **Batch normalization** for stable training
- **Global average pooling** to reduce overfitting
- **Dropout regularization** for generalization
- **Automatic architecture search**

**Input Format:**
```python
# CNN expects 4D input: (samples, height, width, channels)
# Wavelet coefficients reshaped to 2D grids
input_shape = (height, width, 1)  # Single channel grayscale
```

---

## 📁 **File Structure**

```
G3_EMSC_Wavelets_CNN/
├── G3_preprocessing.py      # EMSC + Wavelet preprocessing
├── G3_model.py             # CNN architecture and training
├── run_G3_experiment.py    # Complete experiment pipeline
├── requirements.txt        # Python dependencies
├── README.md              # This documentation
├── [Generated Files]
│   ├── X_train_processed.npy        # 4D CNN training data
│   ├── X_test_processed.npy         # 4D CNN test data
│   ├── emsc_wavelet_preprocessor.pkl # Fitted preprocessor
│   ├── G3_cnn_model.h5             # Trained CNN model
│   ├── G3_experimental_results.json # Complete results
│   └── G3_performance_summary.txt   # Human-readable summary
```

---

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
cd gender/G3_EMSC_Wavelets_CNN
pip install -r requirements.txt
```

### **2. Run Complete Experiment**
```bash
python run_G3_experiment.py
```

### **3. Run Individual Components**
```bash
# Preprocessing only
python G3_preprocessing.py

# CNN modeling only (requires preprocessed data)
python G3_model.py
```

---

## ⚙️ **Configuration Options**

### **EMSC Parameters**
```python
EMSC_Wavelet_Preprocessor(
    wavelet='db4',          # Wavelet type: 'db4', 'haar', 'coif2'
    decomp_levels=4,        # Decomposition levels: 3-6
    emsc_degree=2           # Polynomial degree: 1-3
)
```

### **CNN Hyperparameters**
```python
# Automatically optimized parameters:
conv_layers         # Number of conv layers: 2-4
filters_start       # Starting filters: 16-64
dropout_rate        # Dropout rate: 0.2-0.4
dense_units         # Dense layer units: 64-256
learning_rate       # Learning rate: 0.0005-0.002
```

---

## 📊 **Expected Performance**

Based on methodology complexity and feature representation:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| **CV Accuracy** | 55-65% | Cross-validation performance |
| **Test Accuracy** | 52-62% | Final model evaluation |
| **Training Time** | 15-45 min | Depends on hyperparameter search |
| **Model Size** | 1-10 MB | CNN architecture dependent |

**Performance Factors:**
- **Wavelet coefficients** capture non-linear spectral patterns
- **CNN spatial learning** from 2D coefficient grids
- **EMSC preprocessing** reduces instrumental artifacts
- **Hyperparameter optimization** finds optimal architecture

---

## 🔧 **Technical Requirements**

### **Hardware**
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8+ GB (16 GB for larger datasets)
- **Storage**: 2+ GB free space
- **GPU**: Optional (TensorFlow GPU for acceleration)

### **Software**
- **Python**: 3.8+
- **TensorFlow**: 2.8+
- **PyWavelets**: 1.3+
- **SciPy**: 1.7+ (for EMSC optimization)

---

## 🧪 **Scientific Insights**

### **Advantages of G3 Approach**

1. **Multi-Resolution Analysis**
   - Wavelets capture features at different scales
   - Both high-frequency (edges) and low-frequency (trends) information

2. **Advanced Scatter Correction**
   - EMSC superior to MSC/SNV for complex baselines
   - Physical constraints ensure realistic corrections

3. **Deep Learning Capabilities**
   - CNNs learn complex non-linear patterns
   - Spatial relationships in wavelet domain
   - Automatic feature extraction

4. **Robustness**
   - Wavelet denoising reduces measurement artifacts
   - CNN regularization prevents overfitting
   - Extensive hyperparameter optimization

### **Biological Relevance**

- **Multi-scale biological processes** captured by wavelets
- **Spatial tissue patterns** learned by CNN convolutions
- **Instrument-independent features** through EMSC correction

---

## 📈 **Results Interpretation**

### **Output Files**

1. **G3_performance_summary.txt**
   - Human-readable results summary
   - Architecture details and performance metrics

2. **G3_experimental_results.json**
   - Complete experimental data (JSON format)
   - Hyperparameter search results
   - Training history and metrics

3. **G3_cnn_model.h5**
   - Trained CNN model (TensorFlow/Keras format)
   - Can be loaded for predictions or analysis

4. **test_predictions.csv**
   - Individual predictions for test samples
   - Includes probabilities and confidence scores

### **Performance Metrics**

- **Accuracy**: Overall classification performance
- **Confusion Matrix**: Class-specific prediction analysis
- **Training History**: Learning curves and convergence
- **Architecture**: Optimal CNN structure found

---

## 🐛 **Troubleshooting**

### **Common Issues**

1. **TensorFlow Installation**
   ```bash
   # For CPU-only version
   pip install tensorflow
   
   # For GPU version (if CUDA available)
   pip install tensorflow-gpu
   ```

2. **Memory Issues**
   ```python
   # Reduce batch size in G3_model.py
   batch_size=16  # Instead of 32
   
   # Reduce CNN architecture complexity
   max_trials=10  # Fewer hyperparameter trials
   ```

3. **Wavelet Transform Errors**
   ```bash
   # Ensure PyWavelets is installed
   pip install PyWavelets>=1.3.0
   ```

4. **EMSC Convergence Issues**
   ```python
   # Adjust polynomial degree in G3_preprocessing.py
   emsc_degree=1  # Simpler baseline model
   ```

### **Performance Optimization**

1. **GPU Acceleration**
   ```python
   # Check GPU availability
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

2. **Parallel Processing**
   ```python
   # CNN training uses all available cores
   # Set TensorFlow thread limits if needed
   tf.config.threading.set_intra_op_parallelism_threads(4)
   ```

---

## 📚 **References & Theory**

### **EMSC Methodology**
- Martens, H. & Stark, E. (1991). "Extended multiplicative signal correction and spectral interference subtraction"
- Geladi, P. et al. (1985). "Linearization and scatter-correction for near-infrared reflectance spectra"

### **Wavelet Analysis**
- Daubechies, I. (1992). "Ten Lectures on Wavelets"
- Mallat, S. (1999). "A Wavelet Tour of Signal Processing"

### **CNN for Spectral Data**
- LeCun, Y. et al. (1998). "Gradient-based learning applied to document recognition"
- Goodfellow, I. et al. (2016). "Deep Learning"

---

## 🎯 **Next Steps**

After running G3 experiment:

1. **Compare with G1/G2** - Analyze performance differences
2. **Visualize wavelet coefficients** - Understand feature extraction
3. **CNN interpretation** - Analyze learned convolutional filters
4. **Run G4-G8 experiments** - Complete methodology comparison
5. **Ensemble methods** - Combine G1-G3 predictions

---

## 📞 **Support**

For questions or issues:
- Review troubleshooting section above
- Check output files for error messages
- Verify all dependencies are installed correctly
- Ensure data files are available in `../../data/`

**Experiment Status**: Ready for execution
**Priority**: HIGH - Advanced deep learning methodology
**Expected Runtime**: 20-60 minutes (depending on hardware) 