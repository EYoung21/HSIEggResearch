"""
M5 Model: Wavelets + MSC + MultiTask Learning for Mortality Classification
Multi-task neural network that predicts mortality and auxiliary tasks simultaneously
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class MultiTaskNeuralNetwork(BaseEstimator, ClassifierMixin):
    """Multi-task neural network for mortality classification with auxiliary tasks"""
    
    def __init__(self, 
                 hidden_layers=[256, 128, 64], 
                 dropout_rate=0.3,
                 learning_rate=0.001,
                 epochs=100,
                 batch_size=32,
                 validation_split=0.2,
                 early_stopping_patience=15):
        
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        
        self.model = None
        self.history = None
        self.feature_importance_ = None
        
    def _create_model(self, n_features):
        """Create multi-task neural network architecture"""
        
        # Input layer
        inputs = keras.Input(shape=(n_features,), name='spectral_input')
        
        # Shared backbone layers
        x = inputs
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, activation='relu', name=f'shared_dense_{i+1}')(x)
            x = layers.BatchNormalization(name=f'shared_bn_{i+1}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'shared_dropout_{i+1}')(x)
        
        # Shared feature representation
        shared_features = layers.Dense(64, activation='relu', name='shared_features')(x)
        
        # Primary task: Mortality classification
        mortality_branch = layers.Dense(32, activation='relu', name='mortality_dense')(shared_features)
        mortality_branch = layers.Dropout(self.dropout_rate, name='mortality_dropout')(mortality_branch)
        mortality_output = layers.Dense(1, activation='sigmoid', name='mortality_prediction')(mortality_branch)
        
        # Auxiliary task 1: Spectral quality assessment (binary classification)
        quality_branch = layers.Dense(16, activation='relu', name='quality_dense')(shared_features)
        quality_output = layers.Dense(1, activation='sigmoid', name='quality_prediction')(quality_branch)
        
        # Auxiliary task 2: Spectral intensity regression
        intensity_branch = layers.Dense(16, activation='relu', name='intensity_dense')(shared_features)
        intensity_output = layers.Dense(1, activation='linear', name='intensity_prediction')(intensity_branch)
        
        # Auxiliary task 3: Spectral variance regression
        variance_branch = layers.Dense(16, activation='relu', name='variance_dense')(shared_features)
        variance_output = layers.Dense(1, activation='linear', name='variance_prediction')(variance_branch)
        
        # Create model with multiple outputs
        model = keras.Model(
            inputs=inputs,
            outputs={
                'mortality': mortality_output,
                'quality': quality_output,
                'intensity': intensity_output,
                'variance': variance_output
            },
            name='MultiTaskMortalityNet'
        )
        
        # Compile with multi-task losses
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                'mortality': 'binary_crossentropy',
                'quality': 'binary_crossentropy',
                'intensity': 'mse',
                'variance': 'mse'
            },
            loss_weights={
                'mortality': 1.0,    # Primary task weight
                'quality': 0.3,      # Auxiliary task weight
                'intensity': 0.2,    # Auxiliary task weight
                'variance': 0.2      # Auxiliary task weight
            },
            metrics={
                'mortality': ['accuracy'],
                'quality': ['accuracy'],
                'intensity': ['mae'],
                'variance': ['mae']
            }
        )
        
        return model
    
    def _create_auxiliary_targets(self, X, y):
        """Create auxiliary task targets from spectral data"""
        
        # Auxiliary target 1: Quality (based on spectral noise level)
        spectral_std = np.std(X, axis=1)
        quality_threshold = np.median(spectral_std)
        y_quality = (spectral_std <= quality_threshold).astype(int)  # 1 = high quality, 0 = low quality
        
        # Auxiliary target 2: Intensity (mean spectral intensity)
        y_intensity = np.mean(X, axis=1)
        
        # Auxiliary target 3: Variance (spectral variance)
        y_variance = np.var(X, axis=1)
        
        return {
            'mortality': y,
            'quality': y_quality,
            'intensity': y_intensity,
            'variance': y_variance
        }
    
    def fit(self, X, y):
        """Train the multi-task neural network"""
        print(f"Training MultiTask Neural Network...")
        print(f"Input shape: {X.shape}")
        
        # Create auxiliary targets
        targets = self._create_auxiliary_targets(X, y)
        
        # Create model
        self.model = self._create_model(X.shape[1])
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_mortality_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_mortality_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X, targets,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✓ MultiTask Neural Network training completed")
        return self
    
    def predict(self, X):
        """Predict mortality (primary task only)"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        predictions = self.model.predict(X, verbose=0)
        # Return only mortality predictions (primary task)
        mortality_proba = predictions['mortality'].flatten()
        return (mortality_proba > 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict mortality probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        predictions = self.model.predict(X, verbose=0)
        mortality_proba = predictions['mortality'].flatten()
        return np.column_stack([1 - mortality_proba, mortality_proba])
    
    def predict_all_tasks(self, X):
        """Predict all tasks (mortality + auxiliary)"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        return self.model.predict(X, verbose=0)

class M5MortalityClassifier:
    """M5 Complete classifier combining preprocessing and multi-task learning"""
    
    def __init__(self, use_multitask=True, backup_models=True):
        self.use_multitask = use_multitask
        self.backup_models = backup_models
        
        # Primary model
        if use_multitask:
            self.primary_model = MultiTaskNeuralNetwork()
        else:
            self.primary_model = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.2,
                random_state=42
            )
        
        # Backup models for ensemble
        if backup_models:
            self.backup_model_1 = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            self.backup_model_2 = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        
        self.is_fitted = False
        self.training_stats = {}
    
    def fit(self, X, y):
        """Train all models"""
        print("\n" + "="*60)
        print("M5 MODEL TRAINING: MULTITASK NEURAL NETWORK")
        print("="*60)
        
        print(f"Training data shape: {X.shape}")
        print(f"Target distribution: {np.bincount(y)}")
        
        # Train primary model
        print("\n1. Training primary model...")
        self.primary_model.fit(X, y)
        
        # Train backup models
        if self.backup_models:
            print("\n2. Training backup models...")
            self.backup_model_1.fit(X, y)
            self.backup_model_2.fit(X, y)
            print("✓ Backup models trained")
        
        self.is_fitted = True
        
        # Store training stats
        self.training_stats = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
            'primary_model': 'MultiTaskNeuralNetwork' if self.use_multitask else 'MLPClassifier',
            'backup_models': self.backup_models
        }
        
        print(f"\n✓ M5 model training completed")
        return self
    
    def predict(self, X):
        """Predict using primary model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.primary_model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities using primary model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.primary_model.predict_proba(X)
    
    def predict_ensemble(self, X):
        """Predict using ensemble of all models"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if not self.backup_models:
            return self.predict(X)
        
        # Get predictions from all models
        pred_primary = self.primary_model.predict_proba(X)[:, 1]
        pred_backup1 = self.backup_model_1.predict_proba(X)[:, 1]
        pred_backup2 = self.backup_model_2.predict_proba(X)[:, 1]
        
        # Weighted ensemble (primary model gets more weight)
        ensemble_proba = (0.6 * pred_primary + 0.25 * pred_backup1 + 0.15 * pred_backup2)
        
        return (ensemble_proba > 0.5).astype(int)
    
    def predict_ensemble_proba(self, X):
        """Predict ensemble probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if not self.backup_models:
            return self.predict_proba(X)
        
        # Get predictions from all models
        pred_primary = self.primary_model.predict_proba(X)[:, 1]
        pred_backup1 = self.backup_model_1.predict_proba(X)[:, 1]
        pred_backup2 = self.backup_model_2.predict_proba(X)[:, 1]
        
        # Weighted ensemble
        ensemble_proba = (0.6 * pred_primary + 0.25 * pred_backup1 + 0.15 * pred_backup2)
        
        return np.column_stack([1 - ensemble_proba, ensemble_proba])
    
    def evaluate_model(self, X, y, use_ensemble=False):
        """Comprehensive model evaluation"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Get predictions
        if use_ensemble and self.backup_models:
            y_pred = self.predict_ensemble(X)
            y_proba = self.predict_ensemble_proba(X)[:, 1]
            model_name = "Ensemble"
        else:
            y_pred = self.predict(X)
            y_proba = self.predict_proba(X)[:, 1]
            model_name = "Primary"
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        print(f"Performing {cv}-fold cross-validation...")
        
        # Use simpler model for cross-validation to avoid memory issues
        simple_model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42
        )
        
        cv_scores = cross_val_score(simple_model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_folds': cv
        }
        
        print(f"✓ Cross-validation completed: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
        return cv_results

def main():
    """Example usage"""
    print("M5 MultiTask Mortality Classifier")
    print("Features:")
    print("- Multi-task neural network")
    print("- Mortality + auxiliary tasks")
    print("- Ensemble backup models")
    print("- Comprehensive evaluation")

if __name__ == "__main__":
    main() 