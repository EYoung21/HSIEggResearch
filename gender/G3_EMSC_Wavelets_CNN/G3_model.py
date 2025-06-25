import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
import joblib
import json
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class WaveletCNNClassifier:
    """
    Convolutional Neural Network for wavelet-transformed spectral data
    Optimized for gender classification from HSI egg data
    """
    
    def __init__(self, input_shape, random_state=42):
        """
        Initialize CNN classifier
        
        Args:
            input_shape: Tuple of (height, width) for 2D input
            random_state: Random seed for reproducibility
        """
        self.input_shape = input_shape + (1,)  # Add channel dimension
        self.random_state = random_state
        self.model = None
        self.history = None
        self.best_params = {}
        
    def create_cnn_architecture(self, 
                               conv_layers=3,
                               filters_start=32,
                               kernel_size=3,
                               pool_size=2,
                               dropout_rate=0.3,
                               dense_units=128,
                               learning_rate=0.001):
        """
        Create CNN architecture optimized for wavelet features
        
        Args:
            conv_layers: Number of convolutional layers
            filters_start: Starting number of filters (doubles each layer)
            kernel_size: Convolution kernel size
            pool_size: Max pooling size
            dropout_rate: Dropout rate for regularization
            dense_units: Number of units in dense layer
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        
        print(f"Creating CNN with {conv_layers} conv layers, {filters_start} start filters")
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=self.input_shape))
        
        # Convolutional layers with increasing filters
        current_filters = filters_start
        for i in range(conv_layers):
            # Convolution + Activation
            model.add(layers.Conv2D(
                filters=current_filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                name=f'conv_{i+1}'
            ))
            
            # Batch normalization for stable training
            model.add(layers.BatchNormalization(name=f'bn_conv_{i+1}'))
            
            # Max pooling (skip for last layer if input is small)
            if i < conv_layers - 1 or min(self.input_shape[:2]) > 8:
                model.add(layers.MaxPooling2D(
                    pool_size=pool_size,
                    name=f'pool_{i+1}'
                ))
            
            # Dropout for regularization
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name=f'dropout_conv_{i+1}'))
            
            # Double filters for next layer
            current_filters = min(current_filters * 2, 512)  # Cap at 512
        
        # Global average pooling to reduce parameters
        model.add(layers.GlobalAveragePooling2D(name='global_avg_pool'))
        
        # Dense layers
        if dense_units > 0:
            model.add(layers.Dense(dense_units, activation='relu', name='dense_1'))
            model.add(layers.BatchNormalization(name='bn_dense'))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate, name='dropout_dense'))
        
        # Output layer for binary classification
        model.add(layers.Dense(1, activation='sigmoid', name='output'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def hyperparameter_search(self, X_train, y_train, cv_folds=5, max_trials=20):
        """
        Manual hyperparameter search for CNN architecture
        
        Args:
            X_train: Training features (4D array)
            y_train: Training labels
            cv_folds: Number of CV folds
            max_trials: Maximum number of parameter combinations
            
        Returns:
            Best parameters and CV score
        """
        print(f"Starting hyperparameter search with {max_trials} trials...")
        
        # Define parameter grid (simplified for computational efficiency)
        param_combinations = [
            # (conv_layers, filters_start, dropout_rate, dense_units, learning_rate)
            (2, 16, 0.2, 64, 0.001),
            (2, 32, 0.3, 128, 0.001),
            (3, 16, 0.2, 64, 0.001),
            (3, 32, 0.3, 128, 0.001),
            (3, 32, 0.4, 64, 0.0005),
            (4, 16, 0.3, 128, 0.001),
            (4, 32, 0.2, 64, 0.001),
            (2, 32, 0.3, 64, 0.002),
            (3, 16, 0.4, 128, 0.0005),
            (3, 64, 0.2, 128, 0.001),
            (2, 16, 0.3, 128, 0.001),
            (4, 16, 0.4, 64, 0.0005),
            (2, 64, 0.2, 64, 0.001),
            (3, 32, 0.2, 256, 0.001),
            (3, 16, 0.3, 64, 0.002),
            (4, 32, 0.3, 128, 0.0005),
            (2, 32, 0.4, 128, 0.001),
            (3, 64, 0.3, 64, 0.001),
            (4, 16, 0.2, 128, 0.001),
            (2, 16, 0.4, 64, 0.002),
        ]
        
        best_score = 0
        best_params = None
        results = []
        
        for trial, params in enumerate(param_combinations[:max_trials]):
            conv_layers, filters_start, dropout_rate, dense_units, learning_rate = params
            
            print(f"\nTrial {trial+1}/{max_trials}: conv={conv_layers}, filters={filters_start}, "
                  f"dropout={dropout_rate}, dense={dense_units}, lr={learning_rate}")
            
            try:
                # Cross-validation
                cv_scores = []
                kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                
                for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
                    X_train_fold = X_train[train_idx]
                    X_val_fold = X_train[val_idx]
                    y_train_fold = y_train[train_idx]
                    y_val_fold = y_train[val_idx]
                    
                    # Create and train model
                    model = self.create_cnn_architecture(
                        conv_layers=conv_layers,
                        filters_start=filters_start,
                        dropout_rate=dropout_rate,
                        dense_units=dense_units,
                        learning_rate=learning_rate
                    )
                    
                    # Early stopping
                    early_stop = EarlyStopping(
                        monitor='val_accuracy',
                        patience=5,
                        restore_best_weights=True,
                        verbose=0
                    )
                    
                    # Train model
                    model.fit(
                        X_train_fold, y_train_fold,
                        validation_data=(X_val_fold, y_val_fold),
                        epochs=30,  # Reduced for faster search
                        batch_size=32,
                        callbacks=[early_stop],
                        verbose=0
                    )
                    
                    # Evaluate
                    val_score = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
                    cv_scores.append(val_score)
                    
                    # Clean up
                    del model
                    tf.keras.backend.clear_session()
                
                # Calculate mean CV score
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                print(f"  CV Score: {mean_score:.4f} ± {std_score:.4f}")
                
                results.append({
                    'params': params,
                    'cv_score': mean_score,
                    'cv_std': std_score
                })
                
                # Update best parameters
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        'conv_layers': conv_layers,
                        'filters_start': filters_start,
                        'dropout_rate': dropout_rate,
                        'dense_units': dense_units,
                        'learning_rate': learning_rate
                    }
                    print(f"  ✓ New best score!")
                
            except Exception as e:
                print(f"  Error in trial {trial+1}: {e}")
                continue
        
        self.best_params = best_params
        print(f"\n✓ Hyperparameter search completed!")
        print(f"✓ Best CV score: {best_score:.4f}")
        print(f"✓ Best parameters: {best_params}")
        
        return best_params, best_score, results
    
    def train_optimized_model(self, X_train, y_train, X_val=None, y_val=None, epochs=100):
        """
        Train CNN with optimized hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Maximum training epochs
            
        Returns:
            Trained model and training history
        """
        if not self.best_params:
            raise ValueError("No optimized parameters found. Run hyperparameter_search first.")
        
        print("Training optimized CNN model...")
        
        # Create model with best parameters
        self.model = self.create_cnn_architecture(**self.best_params)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            # Use 20% of training data for validation
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
            )
            X_train, y_train = X_train_split, y_train_split
            validation_data = (X_val_split, y_val_split)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✓ Model training completed!")
        return self.model, self.history
    
    def evaluate_on_test(self, X_test, y_test, label_encoder):
        """
        Comprehensive evaluation on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            label_encoder: Label encoder for class names
            
        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_optimized_model first.")
        
        print("\n" + "="*50)
        print("CNN TEST SET EVALUATION")
        print("="*50)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Test accuracy
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"CNN Test Accuracy: {test_accuracy:.4f}")
        
        # Detailed classification report
        class_names = label_encoder.classes_
        print(f"\nCNN Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion matrix
        print(f"\nCNN Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Class-wise analysis
        print(f"\nClass-wise Analysis:")
        for i, class_name in enumerate(class_names):
            class_mask = y_test == i
            if np.sum(class_mask) > 0:
                class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
                print(f"{class_name}: {np.sum(class_mask)} samples, accuracy: {class_acc:.4f}")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
            'predicted_probability': y_pred_proba.flatten(),
        })
        
        predictions_df.to_csv('test_predictions.csv', index=False)
        
        return {
            'test_accuracy': test_accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba.flatten(),
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=class_names, output_dict=True)
        }
    
    def get_training_history(self):
        """
        Get training history for analysis
        
        Returns:
            Training history dictionary
        """
        if self.history is None:
            return None
        
        return {
            'loss': self.history.history['loss'],
            'accuracy': self.history.history['accuracy'],
            'val_loss': self.history.history['val_loss'],
            'val_accuracy': self.history.history['val_accuracy'],
            'epochs': len(self.history.history['loss'])
        }
    
    def save_model(self, filepath='G3_cnn_model.h5'):
        """
        Save trained CNN model
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        self.model.save(filepath)
        
        # Save additional metadata
        metadata = {
            'best_params': self.best_params,
            'input_shape': self.input_shape,
            'training_history': self.get_training_history()
        }
        
        with open('G3_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Model saved to {filepath}")
        print(f"✓ Metadata saved to G3_model_metadata.json")

def main():
    """
    Main training pipeline for G3 CNN experiment
    """
    print("="*60)
    print("G3 CNN MODEL: Convolutional Neural Network for Wavelets")
    print("="*60)
    
    # Load processed data
    print("Loading processed wavelet data...")
    X_train = np.load('X_train_processed.npy')
    X_test = np.load('X_test_processed.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    # Load input shape and label encoder
    input_shape = tuple(np.load('cnn_input_shape.npy'))
    label_encoder = joblib.load('label_encoder.pkl')
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ CNN input shape: {input_shape}")
    print(f"✓ Classes: {label_encoder.classes_}")
    
    # Initialize CNN classifier
    classifier = WaveletCNNClassifier(input_shape=input_shape, random_state=42)
    
    # Hyperparameter optimization
    print("\n" + "="*50)
    print("CNN HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    
    best_params, best_score, search_results = classifier.hyperparameter_search(
        X_train, y_train, 
        cv_folds=3,  # Reduced for faster execution
        max_trials=15  # Reduced for faster execution
    )
    
    # Train optimized model
    print("\n" + "="*50)
    print("TRAINING OPTIMIZED CNN")
    print("="*50)
    
    model, history = classifier.train_optimized_model(X_train, y_train, epochs=100)
    
    # Test set evaluation
    test_results = classifier.evaluate_on_test(X_test, y_test, label_encoder)
    
    # Save model and results
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    classifier.save_model('G3_cnn_model.h5')
    
    # Save comprehensive results
    results = {
        'experiment': 'G3_EMSC_Wavelets_CNN',
        'preprocessing': 'EMSC + Wavelet Transform (db4, 4 levels)',
        'algorithm': 'Convolutional Neural Network',
        'hyperparameter_optimization': {
            'method': 'Grid search with cross-validation',
            'trials': len(search_results),
            'best_cv_score': float(best_score)
        },
        'best_parameters': best_params,
        'test_results': {
            'accuracy': float(test_results['test_accuracy']),
            'confusion_matrix': test_results['confusion_matrix'].tolist(),
        },
        'training_history': classifier.get_training_history(),
        'input_shape': list(input_shape),
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0])
    }
    
    with open('G3_experimental_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary report
    training_hist = classifier.get_training_history()
    final_val_acc = training_hist['val_accuracy'][-1] if training_hist else 0
    
    summary = f"""
G3 EXPERIMENT SUMMARY: EMSC + Wavelets + CNN
===========================================

METHODOLOGY:
- Preprocessing: EMSC + Wavelet Transform (Daubechies 4, 4 levels)
- Algorithm: Convolutional Neural Network
- Input: 2D wavelet coefficient grids
- Optimization: Grid search with cross-validation

DATASET:
- Training samples: {X_train.shape[0]}
- Test samples: {X_test.shape[0]}
- Input shape: {input_shape} + 1 channel
- Classes: {', '.join(label_encoder.classes_)}

CNN ARCHITECTURE:
- Convolutional layers: {best_params.get('conv_layers', 'N/A')}
- Starting filters: {best_params.get('filters_start', 'N/A')}
- Dropout rate: {best_params.get('dropout_rate', 'N/A')}
- Dense units: {best_params.get('dense_units', 'N/A')}
- Learning rate: {best_params.get('learning_rate', 'N/A')}

TRAINING RESULTS:
- Best CV Score: {best_score:.4f}
- Final Validation Accuracy: {final_val_acc:.4f}
- Training Epochs: {training_hist['epochs'] if training_hist else 'N/A'}

TEST SET RESULTS:
- CNN Test Accuracy: {test_results['test_accuracy']:.4f}

CONFUSION MATRIX:
{test_results['confusion_matrix']}

OPTIMIZATION RESULTS:
- Search method: Grid search with 3-fold CV
- Trials completed: {len(search_results)}
- Best parameters: {best_params}

FILES GENERATED:
- G3_cnn_model.h5 (trained CNN model)
- G3_experimental_results.json (complete results)
- G3_model_metadata.json (model metadata)
- test_predictions.csv (test set predictions)

NOTES:
- EMSC provides superior scatter correction compared to MSC/SNV
- Wavelets capture multi-resolution time-frequency information
- CNN learns spatial patterns in wavelet coefficient space
- Deep learning approach for complex non-linear relationships
"""
    
    with open('G3_performance_summary.txt', 'w') as f:
        f.write(summary)
    
    print("✓ Saved G3_experimental_results.json")
    print("✓ Saved G3_performance_summary.txt")
    print("✓ Saved test_predictions.csv")
    
    print(f"\n🎯 G3 CNN TEST ACCURACY: {test_results['test_accuracy']:.4f}")
    print("✅ G3 experiment completed successfully!")

if __name__ == "__main__":
    main()