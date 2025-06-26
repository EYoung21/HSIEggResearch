"""
M3 Model: EMSC + Augmentation + CNN for Mortality Classification
Convolutional Neural Network for mortality prediction with advanced preprocessing
"""

import numpy as np
import pandas as pd
import time
import json
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, 
    BatchNormalization, Activation, SeparableConv1D, GlobalMaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

class MortalityCNN:
    """M3 Model: CNN for mortality classification with EMSC features"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Model components
        self.cnn_model = None
        self.backup_model = None  # Fallback classical model
        
        # Training metadata
        self.training_time = 0
        self.training_history = None
        self.cv_scores = {}
        self.model_architecture = None
        
        # Class balancing
        self.smote = SMOTE(random_state=random_state, k_neighbors=3)
        
    def create_cnn_architecture(self, input_shape, num_classes=2):
        """Create CNN architecture for spectral data"""
        print(f"Creating CNN architecture for input shape: {input_shape}")
        
        model = Sequential([
            # First convolutional block
            Conv1D(32, kernel_size=15, padding='same', input_shape=input_shape),
            BatchNormalization(),
            Activation('relu'),
            Conv1D(32, kernel_size=15, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # Second convolutional block
            Conv1D(64, kernel_size=11, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            Conv1D(64, kernel_size=11, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Third convolutional block
            Conv1D(128, kernel_size=7, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            SeparableConv1D(128, kernel_size=7, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),
            
            # Fourth convolutional block
            Conv1D(256, kernel_size=5, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            SeparableConv1D(256, kernel_size=5, padding='same'),
            BatchNormalization(),
            Activation('relu'),
            
            # Global pooling and classification
            GlobalAveragePooling1D(),
            
            # Dense layers
            Dense(512, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            
            Dense(256, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.5),
            
            Dense(128, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Activation('relu'),
            Dropout(0.3),
            
            # Output layer
            Dense(1 if num_classes == 2 else num_classes, 
                  activation='sigmoid' if num_classes == 2 else 'softmax')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        loss = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        print(f"✓ CNN architecture created:")
        print(f"  - Total parameters: {model.count_params():,}")
        # Get trainable parameters count (compatible with different TensorFlow versions)
        try:
            trainable_params = sum([tf.keras.utils.count_params(w) for w in model.trainable_weights])
        except AttributeError:
            trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        return model
    
    def create_backup_model(self, input_shape):
        """Create simple backup model for comparison"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        # Flatten CNN features for classical ML
        class CNNFlattener:
            def fit(self, X, y=None):
                return self
            
            def transform(self, X):
                return X.reshape(X.shape[0], -1)
            
            def fit_transform(self, X, y=None):
                return self.transform(X)
        
        backup_model = Pipeline([
            ('flatten', CNNFlattener()),
            ('scale', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state))
        ])
        
        return backup_model
    
    def train_mortality_cnn(self, X_train, y_train, validation_split=0.2):
        """Train CNN for mortality classification"""
        print("\n" + "="*60)
        print("TRAINING M3: EMSC + AUGMENTATION + CNN")
        print("="*60)
        
        start_time = time.time()
        
        # Apply SMOTE for additional class balancing if needed
        original_dist = np.bincount(y_train)
        print(f"Original class distribution: {original_dist}")
        
        # For CNN, we might need additional samples if very imbalanced
        if len(original_dist) > 1 and original_dist.min() / original_dist.max() < 0.3:
            print("Applying additional SMOTE balancing...")
            # Flatten for SMOTE, then reshape back
            X_flat = X_train.reshape(X_train.shape[0], -1)
            X_resampled_flat, y_resampled = self.smote.fit_resample(X_flat, y_train)
            X_resampled = X_resampled_flat.reshape(-1, X_train.shape[1], X_train.shape[2])
            print(f"Resampled distribution: {np.bincount(y_resampled)}")
        else:
            X_resampled, y_resampled = X_train, y_train
            print("Class distribution acceptable, no additional balancing needed")
        
        # Create CNN model
        input_shape = (X_resampled.shape[1], X_resampled.shape[2])
        self.cnn_model = self.create_cnn_architecture(input_shape)
        
        # Store architecture info
        self.model_architecture = {
            'input_shape': input_shape,
            'total_params': self.cnn_model.count_params(),
            'layers': len(self.cnn_model.layers)
        }
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        print("\nTraining CNN model...")
        history = self.cnn_model.fit(
            X_resampled, y_resampled,
            validation_split=validation_split,
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        self.training_history = history.history
        
        # Train backup model
        print("\nTraining backup Random Forest model...")
        self.backup_model = self.create_backup_model(input_shape)
        self.backup_model.fit(X_resampled, y_resampled)
        
        # Cross-validation evaluation
        print("\nPerforming CNN cross-validation...")
        cv_scores = self.evaluate_cnn_cv(X_resampled, y_resampled, cv_folds=3)
        
        self.cv_scores = cv_scores
        self.training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {self.training_time:.2f} seconds")
        print(f"✓ CNN CV Accuracy: {cv_scores['cnn_cv_mean']:.4f} ± {cv_scores['cnn_cv_std']:.4f}")
        print(f"✓ Backup CV Accuracy: {cv_scores['backup_cv_mean']:.4f} ± {cv_scores['backup_cv_std']:.4f}")
        
        return {
            'training_time': self.training_time,
            'cv_scores': cv_scores,
            'model_architecture': self.model_architecture,
            'training_samples': X_resampled.shape[0],
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
    
    def evaluate_cnn_cv(self, X, y, cv_folds=3):
        """Perform cross-validation for CNN"""
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cnn_scores = []
        backup_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            print(f"Training fold {fold + 1}/{cv_folds}...")
            
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Train CNN for this fold
            fold_model = self.create_cnn_architecture((X.shape[1], X.shape[2]))
            fold_model.fit(
                X_fold_train, y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
            )
            
            # Evaluate CNN - fixed prediction handling
            cnn_pred = fold_model.predict(X_fold_val, verbose=0)
            cnn_pred = (cnn_pred.flatten() > 0.5).astype(int)
            
            cnn_accuracy = accuracy_score(y_fold_val, cnn_pred)
            cnn_scores.append(cnn_accuracy)
            
            # Evaluate backup model
            fold_backup = self.create_backup_model((X.shape[1], X.shape[2]))
            fold_backup.fit(X_fold_train, y_fold_train)
            backup_pred = fold_backup.predict(X_fold_val)
            backup_accuracy = accuracy_score(y_fold_val, backup_pred)
            backup_scores.append(backup_accuracy)
            
            print(f"  Fold {fold + 1}: CNN={cnn_accuracy:.4f}, Backup={backup_accuracy:.4f}")
        
        return {
            'cnn_cv_mean': np.mean(cnn_scores),
            'cnn_cv_std': np.std(cnn_scores),
            'backup_cv_mean': np.mean(backup_scores),
            'backup_cv_std': np.std(backup_scores),
            'cnn_cv_scores': cnn_scores,
            'backup_cv_scores': backup_scores
        }
    
    def evaluate_mortality_prediction(self, X_test, y_test, label_encoder):
        """Evaluate mortality prediction models"""
        print("\n" + "="*60)
        print("EVALUATING MORTALITY CNN")
        print("="*60)
        
        results = {}
        
        # Evaluate CNN model - fixed prediction handling
        print("Evaluating CNN model...")
        cnn_pred_proba = self.cnn_model.predict(X_test, verbose=0)
        cnn_probabilities = cnn_pred_proba.flatten()
        cnn_predictions = (cnn_probabilities > 0.5).astype(int)
        
        cnn_accuracy = accuracy_score(y_test, cnn_predictions)
        cnn_auc = roc_auc_score(y_test, cnn_probabilities)
        
        results['cnn'] = {
            'predictions': cnn_predictions.tolist(),
            'probabilities': cnn_probabilities.tolist(),
            'accuracy': cnn_accuracy,
            'auc': cnn_auc,
            'confusion_matrix': confusion_matrix(y_test, cnn_predictions),
            'classification_report': classification_report(
                y_test, cnn_predictions, target_names=label_encoder.classes_, output_dict=True
            )
        }
        
        print(f"CNN - Accuracy: {cnn_accuracy:.4f}, AUC: {cnn_auc:.4f}")
        
        # Evaluate backup model
        print("Evaluating backup model...")
        backup_predictions = self.backup_model.predict(X_test)
        backup_probabilities = self.backup_model.predict_proba(X_test)[:, 1]
        
        backup_accuracy = accuracy_score(y_test, backup_predictions)
        backup_auc = roc_auc_score(y_test, backup_probabilities)
        
        results['backup'] = {
            'predictions': backup_predictions.tolist(),
            'probabilities': backup_probabilities.tolist(),
            'accuracy': backup_accuracy,
            'auc': backup_auc,
            'confusion_matrix': confusion_matrix(y_test, backup_predictions),
            'classification_report': classification_report(
                y_test, backup_predictions, target_names=label_encoder.classes_, output_dict=True
            )
        }
        
        print(f"Backup - Accuracy: {backup_accuracy:.4f}, AUC: {backup_auc:.4f}")
        
        # Determine best model
        best_model_name = 'cnn' if cnn_accuracy >= backup_accuracy else 'backup'
        best_accuracy = max(cnn_accuracy, backup_accuracy)
        
        results['best_model'] = {
            'name': best_model_name,
            'accuracy': best_accuracy,
            'auc': results[best_model_name]['auc']
        }
        
        print(f"\nMortality CNN Results:")
        print(f"  Best Model: {best_model_name.upper()}")
        print(f"  Best Accuracy: {best_accuracy:.4f}")
        print(f"  Best AUC: {results[best_model_name]['auc']:.4f}")
        
        return results
    
    def analyze_cnn_features(self, X_sample, preprocessing_info, top_k=10):
        """Analyze CNN feature learning and importance"""
        print(f"\nAnalyzing CNN feature learning...")
        
        # Get intermediate layer outputs
        layer_outputs = []
        for i, layer in enumerate(self.cnn_model.layers):
            if 'conv1d' in layer.name:
                try:
                    # Build the model by calling it first
                    _ = self.cnn_model(X_sample[:1])
                    intermediate_model = tf.keras.Model(
                        inputs=self.cnn_model.input,
                        outputs=layer.output
                    )
                    output = intermediate_model.predict(X_sample[:5], verbose=0)  # Use first 5 samples
                except:
                    # Skip if model input not defined
                    continue
                layer_outputs.append({
                    'layer_name': layer.name,
                    'output_shape': output.shape,
                    'activation_stats': {
                        'mean': float(np.mean(output)),
                        'std': float(np.std(output)),
                        'max': float(np.max(output)),
                        'min': float(np.min(output))
                    }
                })
        
        # Analyze filter responses
        conv_layers = [layer for layer in self.cnn_model.layers if 'conv1d' in layer.name]
        filter_analysis = []
        
        for layer in conv_layers[:3]:  # Analyze first 3 conv layers
            weights = layer.get_weights()[0]  # Get filter weights
            filter_analysis.append({
                'layer_name': layer.name,
                'n_filters': weights.shape[-1],
                'filter_size': weights.shape[0],
                'weight_stats': {
                    'mean': float(np.mean(weights)),
                    'std': float(np.std(weights)),
                    'max': float(np.max(weights)),
                    'min': float(np.min(weights))
                }
            })
        
        # Training history analysis
        history_analysis = {}
        if self.training_history:
            history_analysis = {
                'final_train_loss': float(self.training_history['loss'][-1]),
                'final_val_loss': float(self.training_history['val_loss'][-1]),
                'final_train_acc': float(self.training_history['accuracy'][-1]),
                'final_val_acc': float(self.training_history['val_accuracy'][-1]),
                'epochs_trained': len(self.training_history['loss']),
                'best_val_acc': float(max(self.training_history['val_accuracy'])),
                'overfitting_indicator': float(self.training_history['accuracy'][-1] - self.training_history['val_accuracy'][-1])
            }
        
        return {
            'layer_outputs': layer_outputs,
            'filter_analysis': filter_analysis,
            'training_analysis': history_analysis,
            'model_complexity': {
                'total_parameters': self.model_architecture['total_params'],
                'input_channels': preprocessing_info.get('n_channels', 'Unknown'),
                'input_length': preprocessing_info.get('n_wavelengths', 'Unknown')
            }
        }
    
    def save_model_and_results(self, results, feature_analysis, preprocessing_info):
        """Save CNN models and results"""
        print("\nSaving M3 CNN models and results...")
        
        # Save CNN model
        if self.cnn_model:
            self.cnn_model.save('M3_cnn_model.h5')
            print("✓ CNN model saved to M3_cnn_model.h5")
        
        experiment_results = {
            'experiment_name': 'M3_EMSC_Augmentation_CNN',
            'methodology': {
                'preprocessing': 'EMSC + Data Augmentation + Multi-scale CNN Features',
                'model': 'Convolutional Neural Network',
                'augmentation': f"{preprocessing_info.get('augmentation_factor', 'Unknown')}x data augmentation",
                'emsc_components': preprocessing_info.get('emsc_components', 'Unknown')
            },
            'data_info': {
                'n_samples_original': preprocessing_info.get('n_samples_original', 0),
                'n_samples_augmented': preprocessing_info.get('n_samples_augmented', 0),
                'n_channels': preprocessing_info.get('n_channels', 0),
                'wavelength_range': preprocessing_info.get('wavelength_range', 'Unknown'),
                'cnn_input_shape': preprocessing_info.get('cnn_input_shape', 'Unknown')
            },
            'training_results': {
                'training_time': self.training_time,
                'cv_scores': self.cv_scores,
                'model_architecture': self.model_architecture
            },
            'evaluation_results': results,
            'feature_analysis': feature_analysis,
            'preprocessing_stats': preprocessing_info
        }
        
        # Convert numpy arrays and complex objects to JSON-serializable format
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            else:
                return str(obj)
        
        with open('M3_experimental_results.json', 'w') as f:
            json.dump(experiment_results, f, indent=2, default=convert_numpy)
        
        # Performance summary
        best_model = results['best_model']['name'].upper()
        best_accuracy = results['best_model']['accuracy']
        best_auc = results['best_model']['auc']
        
        summary_text = f"""
M3 MORTALITY CLASSIFICATION - PERFORMANCE SUMMARY
===============================================

METHODOLOGY:
- Preprocessing: EMSC + Data Augmentation ({preprocessing_info.get('augmentation_factor', 'Unknown')}x)
- Model: Convolutional Neural Network + Random Forest Backup
- Features: Multi-scale CNN features with {preprocessing_info.get('n_channels', 'Unknown')} channels
- EMSC Components: {preprocessing_info.get('emsc_components', 'Unknown')}

DATASET:
- Original Samples: {preprocessing_info.get('n_samples_original', 'Unknown')}
- Augmented Samples: {preprocessing_info.get('n_samples_augmented', 'Unknown')}
- CNN Input Shape: {preprocessing_info.get('cnn_input_shape', 'Unknown')}
- Wavelength Range: {preprocessing_info.get('wavelength_range', 'Unknown')}

TRAINING PERFORMANCE:
- Training Time: {self.training_time:.2f} seconds
- CNN CV: {self.cv_scores['cnn_cv_mean']:.4f} ± {self.cv_scores['cnn_cv_std']:.4f}
- Backup CV: {self.cv_scores['backup_cv_mean']:.4f} ± {self.cv_scores['backup_cv_std']:.4f}
- Model Parameters: {self.model_architecture['total_params']:,}

TEST PERFORMANCE:
- Best Model: {best_model}
- Test Accuracy: {best_accuracy:.4f}
- Test AUC: {best_auc:.4f}

MODEL COMPARISON:
- CNN: {results['cnn']['accuracy']:.4f} (AUC: {results['cnn']['auc']:.4f})
- Backup RF: {results['backup']['accuracy']:.4f} (AUC: {results['backup']['auc']:.4f})

CNN ARCHITECTURE:
- Total Parameters: {self.model_architecture['total_params']:,}
- Input Shape: {self.model_architecture['input_shape']}
- Total Layers: {self.model_architecture['layers']}

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open('M3_performance_summary.txt', 'w') as f:
            f.write(summary_text)
        
        print("✓ Results saved:")
        print("  - M3_experimental_results.json")
        print("  - M3_performance_summary.txt")
        print("  - M3_cnn_model.h5")

def main():
    """Example usage"""
    print("M3 Mortality CNN Model")
    print("Usage: Import and use with preprocessed CNN-compatible data")
    
    # Example architecture info
    print("\nCNN Architecture:")
    print("- Multi-scale Conv1D layers (32, 64, 128, 256 filters)")
    print("- Separable convolutions for efficiency")
    print("- Batch normalization and dropout for regularization")
    print("- Global average pooling + dense layers")
    print("- Early stopping and learning rate scheduling")

if __name__ == "__main__":
    main() 