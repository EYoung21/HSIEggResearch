import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import joblib
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Set TensorFlow logging
tf.get_logger().setLevel('ERROR')

class BayesianNeuralNetwork:
    """Bayesian Neural Network with Monte Carlo Dropout for uncertainty quantification"""
    
    def __init__(self, input_dim, hidden_units=[128, 64, 32], dropout_rate=0.2, 
                 learning_rate=0.001, random_state=42):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        # Model components
        self.model = None
        self.history = None
        self.training_time = 0
        
    def build_model(self):
        """Build Bayesian neural network with MC Dropout"""
        print("Building Bayesian Neural Network with MC Dropout...")
        
        # Input layer
        inputs = tf.keras.Input(shape=(self.input_dim,), name='input_features')
        
        # Hidden layers with MC Dropout
        x = inputs
        for i, units in enumerate(self.hidden_units):
            x = tf.keras.layers.Dense(
                units=units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name=f'dense_{i+1}'
            )(x)
            
            # Batch normalization
            x = tf.keras.layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
            
            # MC Dropout (applied during both training and inference)
            x = tf.keras.layers.Dropout(
                self.dropout_rate, 
                name=f'mc_dropout_{i+1}'
            )(x, training=True)  # Always active for MC sampling
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            1, 
            activation='sigmoid',
            name='output'
        )(x)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='BayesianNN_MCDropout')
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print(f"✓ Bayesian NN architecture:")
        print(f"  - Input dimension: {self.input_dim}")
        print(f"  - Hidden layers: {self.hidden_units}")
        print(f"  - Dropout rate: {self.dropout_rate}")
        print(f"  - Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train_with_uncertainty(self, X_train, y_train, X_val=None, y_val=None, 
                              epochs=100, batch_size=32, verbose=1):
        """Train Bayesian neural network"""
        print(f"\nTraining Bayesian Neural Network...")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        
        start_time = time.time()
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            print(f"Validation samples: {X_val.shape[0]}")
        else:
            validation_data = None
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_time = time.time() - start_time
        
        print(f"✓ Training completed in {self.training_time:.1f} seconds")
        print(f"✓ Final loss: {self.history.history['loss'][-1]:.4f}")
        
        if validation_data:
            print(f"✓ Final val_accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
        
        return self.history
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """Make predictions with uncertainty using MC Dropout"""
        print(f"Making predictions with uncertainty (MC samples: {n_samples})...")
        
        # Generate multiple predictions using MC Dropout
        predictions_samples = []
        
        for i in range(n_samples):
            # Enable dropout during inference for MC sampling
            pred = self.model(X, training=True)
            predictions_samples.append(pred.numpy().flatten())
        
        predictions_samples = np.array(predictions_samples)
        
        # Calculate statistics
        mean_predictions = np.mean(predictions_samples, axis=0)
        prediction_std = np.std(predictions_samples, axis=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = prediction_std
        
        # Aleatoric uncertainty approximation
        aleatoric_uncertainty = mean_predictions * (1 - mean_predictions)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        print(f"✓ Predictions generated with uncertainty estimates")
        print(f"  - Mean epistemic uncertainty: {np.mean(epistemic_uncertainty):.4f}")
        print(f"  - Mean aleatoric uncertainty: {np.mean(aleatoric_uncertainty):.4f}")
        print(f"  - Mean total uncertainty: {np.mean(total_uncertainty):.4f}")
        
        return mean_predictions, {
            'epistemic': epistemic_uncertainty,
            'aleatoric': aleatoric_uncertainty,
            'total': total_uncertainty
        }, predictions_samples
    
    def evaluate_with_uncertainty(self, X_test, y_test, n_samples=100):
        """Evaluate model with uncertainty analysis"""
        print(f"\nEvaluating Bayesian Neural Network...")
        
        # Get predictions with uncertainty
        pred_probs, uncertainties, pred_samples = self.predict_with_uncertainty(X_test, n_samples)
        
        # Convert to binary predictions
        pred_binary = (pred_probs > 0.5).astype(int)
        
        # Standard metrics
        accuracy = accuracy_score(y_test, pred_binary)
        
        try:
            auc_score = roc_auc_score(y_test, pred_probs)
        except:
            auc_score = 0.5
        
        # Uncertainty-based analysis
        high_uncertainty_mask = uncertainties['total'] > np.percentile(uncertainties['total'], 75)
        low_uncertainty_mask = uncertainties['total'] <= np.percentile(uncertainties['total'], 25)
        
        # Accuracy on different uncertainty levels
        if np.sum(low_uncertainty_mask) > 0:
            low_uncertainty_accuracy = accuracy_score(
                y_test[low_uncertainty_mask], 
                pred_binary[low_uncertainty_mask]
            )
        else:
            low_uncertainty_accuracy = 0.0
        
        if np.sum(high_uncertainty_mask) > 0:
            high_uncertainty_accuracy = accuracy_score(
                y_test[high_uncertainty_mask], 
                pred_binary[high_uncertainty_mask]
            )
        else:
            high_uncertainty_accuracy = 0.0
        
        # Prediction intervals
        lower_bound = np.percentile(pred_samples, 2.5, axis=0)
        upper_bound = np.percentile(pred_samples, 97.5, axis=0)
        interval_width = upper_bound - lower_bound
        
        # Coverage
        coverage = np.mean((y_test >= lower_bound) & (y_test <= upper_bound))
        
        results = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'predictions': pred_probs,
            'binary_predictions': pred_binary,
            'uncertainties': uncertainties,
            'low_uncertainty_accuracy': low_uncertainty_accuracy,
            'high_uncertainty_accuracy': high_uncertainty_accuracy,
            'prediction_intervals': {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': interval_width,
                'mean_width': np.mean(interval_width),
                'coverage': coverage
            },
            'uncertainty_stats': {
                'mean_epistemic': np.mean(uncertainties['epistemic']),
                'mean_aleatoric': np.mean(uncertainties['aleatoric']),
                'mean_total': np.mean(uncertainties['total']),
                'std_epistemic': np.std(uncertainties['epistemic']),
                'std_aleatoric': np.std(uncertainties['aleatoric']),
                'std_total': np.std(uncertainties['total'])
            }
        }
        
        print(f"✓ Evaluation completed:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - AUC Score: {auc_score:.4f}")
        print(f"  - Low uncertainty accuracy: {low_uncertainty_accuracy:.4f}")
        print(f"  - High uncertainty accuracy: {high_uncertainty_accuracy:.4f}")
        print(f"  - Prediction interval coverage: {coverage:.4f}")
        
        return results

def load_processed_data():
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    
    try:
        X_train = np.load('X_train_processed.npy')
        X_test = np.load('X_test_processed.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        
        label_encoder = joblib.load('label_encoder.pkl')
        
        with open('preprocessing_info.json', 'r') as f:
            preprocessing_info = json.load(f)
        
        print(f"✓ Data loaded successfully:")
        print(f"  - Training: {X_train.shape}")
        print(f"  - Test: {X_test.shape}")
        print(f"  - Classes: {label_encoder.classes_}")
        
        return X_train, X_test, y_train, y_test, label_encoder, preprocessing_info
        
    except FileNotFoundError as e:
        print(f"✗ Failed to load data: {e}")
        print("Please run G8_preprocessing.py first")
        return None, None, None, None, None, None

def cross_validate_bayesian_nn(X, y, cv_folds=3, epochs=50):
    """Perform cross-validation"""
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []
    cv_uncertainties = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{cv_folds} ---")
        
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create model for this fold
        bnn = BayesianNeuralNetwork(
            input_dim=X.shape[1],
            hidden_units=[64, 32],
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        bnn.build_model()
        bnn.train_with_uncertainty(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            epochs=epochs,
            batch_size=16,
            verbose=0
        )
        
        # Evaluate
        results = bnn.evaluate_with_uncertainty(X_val_fold, y_val_fold, n_samples=50)
        
        cv_scores.append(results['accuracy'])
        cv_uncertainties.append(results['uncertainty_stats']['mean_total'])
        
        print(f"  Fold {fold + 1} accuracy: {results['accuracy']:.4f}")
    
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    print(f"\n✓ CV results: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    
    return cv_scores, cv_uncertainties

def main():
    """Main Bayesian neural network pipeline"""
    print("="*80)
    print("G8 BAYESIAN NEURAL NETWORK: SG + EMSC + Uncertainty Quantification")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test, label_encoder, preprocessing_info = load_processed_data()
    
    if X_train is None:
        return
    
    print(f"\nDataset summary:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Cross-validation
    cv_scores, cv_uncertainties = cross_validate_bayesian_nn(X_train, y_train, cv_folds=3, epochs=50)
    
    # Train final model
    print(f"\n" + "="*60)
    print("TRAINING FINAL BAYESIAN NEURAL NETWORK")
    print("="*60)
    
    bnn = BayesianNeuralNetwork(
        input_dim=X_train.shape[1],
        hidden_units=[128, 64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        random_state=42
    )
    
    bnn.build_model()
    
    # Split for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    history = bnn.train_with_uncertainty(
        X_train_split, y_train_split,
        X_val_split, y_val_split,
        epochs=100,
        batch_size=32,
        verbose=1
    )
    
    # Final evaluation
    print(f"\n" + "="*60)
    print("FINAL EVALUATION WITH UNCERTAINTY")
    print("="*60)
    
    final_results = bnn.evaluate_with_uncertainty(X_test, y_test, n_samples=100)
    
    # Save results
    print(f"\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save model
    bnn.model.save('G8_bayesian_nn_model.h5')
    
    # Experimental results
    experiment_results = {
        'experiment': 'G8_SG_EMSC_BayesianNN',
        'preprocessing': 'EMSC + Savitzky-Golay + Advanced features',
        'algorithm': 'Bayesian Neural Network (MC Dropout)',
        'best_accuracy': float(final_results['accuracy']),
        'auc_score': float(final_results['auc_score']),
        'training_time_seconds': float(bnn.training_time),
        'cv_scores': [float(score) for score in cv_scores],
        'cv_mean': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores)),
        'uncertainty_metrics': {
            'mean_epistemic': float(final_results['uncertainty_stats']['mean_epistemic']),
            'mean_aleatoric': float(final_results['uncertainty_stats']['mean_aleatoric']),
            'mean_total': float(final_results['uncertainty_stats']['mean_total']),
            'prediction_interval_coverage': float(final_results['prediction_intervals']['coverage']),
            'mean_interval_width': float(final_results['prediction_intervals']['mean_width'])
        },
        'model_architecture': {
            'input_dim': int(X_train.shape[1]),
            'hidden_units': [128, 64, 32],
            'dropout_rate': 0.2,
            'total_parameters': int(bnn.model.count_params())
        },
        'preprocessing_info': preprocessing_info,
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0])
    }
    
    # Add predictions
    experiment_results['predictions'] = {
        'probabilities': final_results['predictions'].tolist(),
        'binary_predictions': final_results['binary_predictions'].tolist(),
        'true_labels': y_test.tolist(),
        'epistemic_uncertainty': final_results['uncertainties']['epistemic'].tolist(),
        'aleatoric_uncertainty': final_results['uncertainties']['aleatoric'].tolist(),
        'total_uncertainty': final_results['uncertainties']['total'].tolist()
    }
    
    # Save experimental results
    with open('G8_experimental_results.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    # Performance summary
    summary = f"""G8 EXPERIMENT SUMMARY: Bayesian Neural Network with Uncertainty Quantification
==============================================================================

METHODOLOGY:
- Preprocessing: EMSC + Savitzky-Golay derivatives + Advanced spectral features  
- Algorithm: Bayesian Neural Network with Monte Carlo Dropout
- Architecture: {bnn.model.count_params():,} parameters
- Uncertainty: Epistemic + Aleatoric uncertainty quantification

DATASET:
- Training samples: {X_train.shape[0]}
- Test samples: {X_test.shape[0]}
- Features: {X_train.shape[1]} (multi-scale processed)
- Classes: {', '.join(label_encoder.classes_)}

PERFORMANCE RESULTS:
- Test Accuracy: {final_results['accuracy']:.4f}
- AUC Score: {final_results['auc_score']:.4f}
- CV Mean: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}

UNCERTAINTY QUANTIFICATION:
- Mean Epistemic Uncertainty: {final_results['uncertainty_stats']['mean_epistemic']:.4f}
- Mean Aleatoric Uncertainty: {final_results['uncertainty_stats']['mean_aleatoric']:.4f}
- Mean Total Uncertainty: {final_results['uncertainty_stats']['mean_total']:.4f}
- Prediction Interval Coverage: {final_results['prediction_intervals']['coverage']:.4f}
- Mean Interval Width: {final_results['prediction_intervals']['mean_width']:.4f}

RELIABILITY ANALYSIS:
- Low Uncertainty Accuracy: {final_results['low_uncertainty_accuracy']:.4f}
- High Uncertainty Accuracy: {final_results['high_uncertainty_accuracy']:.4f}

TECHNICAL DETAILS:
- Training Time: {bnn.training_time:.1f} seconds
- Architecture: {' → '.join(map(str, [X_train.shape[1]] + [128, 64, 32] + [1]))}
- Dropout Rate: {bnn.dropout_rate}
- Monte Carlo Samples: 100

FILES GENERATED:
- G8_bayesian_nn_model.h5 (trained model)
- G8_experimental_results.json (complete results)
"""
    
    with open('G8_performance_summary.txt', 'w') as f:
        f.write(summary)
    
    print("✓ Saved G8_bayesian_nn_model.h5")
    print("✓ Saved G8_experimental_results.json") 
    print("✓ Saved G8_performance_summary.txt")
    
    print(f"\n" + "="*80)
    print("G8 BAYESIAN NEURAL NETWORK RESULTS")
    print("="*80)
    print(f"🎯 Test Accuracy: {final_results['accuracy']:.4f}")
    print(f"📊 AUC Score: {final_results['auc_score']:.4f}")
    print(f"🔄 CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"❓ Mean Uncertainty: {final_results['uncertainty_stats']['mean_total']:.4f}")
    print(f"⏱️ Training Time: {bnn.training_time:.1f} seconds")
    print("✅ G8 Bayesian NN experiment completed!")

if __name__ == "__main__":
    main() 