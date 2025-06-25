import numpy as np
import pandas as pd
import tensorflow as tf
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

class SpectralTransformerClassifier:
    """
    Transformer-based classifier for spectral sequences
    Implements attention mechanism for gender classification from HSI egg data
    """
    
    def __init__(self, sequence_length, random_state=42):
        """
        Initialize Transformer classifier
        
        Args:
            sequence_length: Length of input spectral sequences
            random_state: Random seed for reproducibility
        """
        self.sequence_length = sequence_length
        self.random_state = random_state
        self.model = None
        self.history = None
        self.best_params = {}
        
    def positional_encoding(self, length, depth):
        """
        Create positional encoding for Transformer
        
        Args:
            length: Sequence length
            depth: Embedding dimension
            
        Returns:
            Positional encoding tensor
        """
        depth = depth / 2
        
        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :] / depth   # (1, depth)
        
        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)
        
        pos_encoding = np.concatenate([
            np.sin(angle_rads), 
            np.cos(angle_rads)
        ], axis=-1)
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def create_transformer_architecture(self,
                                      embed_dim=128,
                                      num_heads=8,
                                      ff_dim=256,
                                      num_layers=4,
                                      dropout_rate=0.1,
                                      learning_rate=0.001):
        """
        Create Transformer architecture for spectral classification
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_layers: Number of transformer layers
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            
        Returns:
            Compiled Transformer model
        """
        
        print(f"Creating Transformer with {num_layers} layers, {num_heads} heads, {embed_dim}D embedding")
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, 1))
        
        # Embedding layer (project 1D spectral values to higher dimension)
        x = layers.Dense(embed_dim)(inputs)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding(self.sequence_length, embed_dim)
        x = x + pos_encoding
        
        # Transformer layers
        for i in range(num_layers):
            # Multi-head self-attention
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                name=f'attention_{i+1}'
            )(x, x)
            
            # Add & Norm
            attention_output = layers.Dropout(dropout_rate)(attention_output)
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization(name=f'norm_1_{i+1}')(x)
            
            # Feed-forward network
            ff_output = tf.keras.Sequential([
                layers.Dense(ff_dim, activation='relu'),
                layers.Dropout(dropout_rate),
                layers.Dense(embed_dim)
            ], name=f'ffn_{i+1}')(x)
            
            # Add & Norm
            x = layers.Add()([x, ff_output])
            x = layers.LayerNormalization(name=f'norm_2_{i+1}')(x)
        
        # Global pooling (aggregate sequence information)
        # Try both average and max pooling
        avg_pool = layers.GlobalAveragePooling1D()(x)
        max_pool = layers.GlobalMaxPooling1D()(x)
        
        # Concatenate different pooling strategies
        pooled = layers.Concatenate()([avg_pool, max_pool])
        
        # Classification head
        x = layers.Dense(ff_dim // 2, activation='relu', name='classifier_dense')(pooled)
        x = layers.Dropout(dropout_rate, name='classifier_dropout')(x)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = tf.keras.Model(inputs, outputs, name='SpectralTransformer')
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def hyperparameter_search(self, X_train, y_train, cv_folds=3, max_trials=15):
        """
        Hyperparameter search for Transformer architecture
        
        Args:
            X_train: Training features (3D array)
            y_train: Training labels
            cv_folds: Number of CV folds
            max_trials: Maximum number of parameter combinations
            
        Returns:
            Best parameters and CV score
        """
        print(f"Starting Transformer hyperparameter search with {max_trials} trials...")
        
        # Define parameter grid (optimized for spectral data)
        param_combinations = [
            # (embed_dim, num_heads, ff_dim, num_layers, dropout_rate, learning_rate)
            (64, 4, 128, 2, 0.1, 0.001),
            (128, 8, 256, 3, 0.1, 0.001),
            (64, 4, 128, 3, 0.2, 0.001),
            (128, 4, 256, 2, 0.1, 0.0005),
            (96, 6, 192, 3, 0.15, 0.001),
            (128, 8, 256, 4, 0.1, 0.001),
            (64, 8, 128, 2, 0.2, 0.002),
            (96, 4, 192, 2, 0.1, 0.001),
            (128, 4, 256, 3, 0.2, 0.0005),
            (64, 6, 128, 3, 0.15, 0.001),
            (160, 8, 320, 2, 0.1, 0.0005),
            (96, 8, 192, 3, 0.1, 0.001),
            (128, 6, 256, 2, 0.15, 0.001),
            (64, 4, 256, 4, 0.2, 0.001),
            (96, 4, 192, 4, 0.1, 0.0005),
        ]
        
        best_score = 0
        best_params = None
        results = []
        
        for trial, params in enumerate(param_combinations[:max_trials]):
            embed_dim, num_heads, ff_dim, num_layers, dropout_rate, learning_rate = params
            
            print(f"\nTrial {trial+1}/{max_trials}: embed={embed_dim}, heads={num_heads}, "
                  f"ff={ff_dim}, layers={num_layers}, dropout={dropout_rate}, lr={learning_rate}")
            
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
                    model = self.create_transformer_architecture(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        ff_dim=ff_dim,
                        num_layers=num_layers,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate
                    )
                    
                    # Early stopping
                    early_stop = EarlyStopping(
                        monitor='val_accuracy',
                        patience=8,
                        restore_best_weights=True,
                        verbose=0
                    )
                    
                    # Train model
                    model.fit(
                        X_train_fold, y_train_fold,
                        validation_data=(X_val_fold, y_val_fold),
                        epochs=40,  # Reduced for faster search
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
                        'embed_dim': embed_dim,
                        'num_heads': num_heads,
                        'ff_dim': ff_dim,
                        'num_layers': num_layers,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate
                    }
                    print(f"  ✓ New best score!")
                
            except Exception as e:
                print(f"  Error in trial {trial+1}: {e}")
                continue
        
        self.best_params = best_params
        print(f"\n✓ Transformer hyperparameter search completed!")
        print(f"✓ Best CV score: {best_score:.4f}")
        print(f"✓ Best parameters: {best_params}")
        
        return best_params, best_score, results
    
    def train_optimized_model(self, X_train, y_train, X_val=None, y_val=None, epochs=100):
        """
        Train Transformer with optimized hyperparameters
        
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
        
        print("Training optimized Transformer model...")
        
        # Create model with best parameters
        self.model = self.create_transformer_architecture(**self.best_params)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy' if X_val is not None else 'accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
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
        
        print("✓ Transformer training completed!")
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
        print("TRANSFORMER TEST SET EVALUATION")
        print("="*50)
        
        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Test accuracy
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Transformer Test Accuracy: {test_accuracy:.4f}")
        
        # Detailed classification report
        class_names = label_encoder.classes_
        print(f"\nTransformer Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion matrix
        print(f"\nTransformer Confusion Matrix:")
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
    
    def save_model(self, filepath='G4_transformer_model.h5'):
        """
        Save trained Transformer model
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        self.model.save(filepath)
        
        # Save additional metadata
        metadata = {
            'best_params': self.best_params,
            'sequence_length': self.sequence_length,
            'training_history': self.get_training_history()
        }
        
        with open('G4_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Model saved to {filepath}")
        print(f"✓ Metadata saved to G4_model_metadata.json")

def main():
    """
    Main training pipeline for G4 Transformer experiment
    """
    print("="*60)
    print("G4 TRANSFORMER MODEL: Attention Mechanism for Spectral Data")
    print("="*60)
    
    # Load processed data
    print("Loading processed spectral data...")
    X_train = np.load('X_train_processed.npy')
    X_test = np.load('X_test_processed.npy')
    y_train = np.load('y_train_augmented.npy')
    y_test = np.load('y_test.npy')
    
    # Load sequence length and label encoder
    sequence_length = int(np.load('transformer_sequence_length.npy')[0])
    label_encoder = joblib.load('label_encoder.pkl')
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Sequence length: {sequence_length}")
    print(f"✓ Classes: {label_encoder.classes_}")
    
    # Initialize Transformer classifier
    classifier = SpectralTransformerClassifier(sequence_length=sequence_length, random_state=42)
    
    # Hyperparameter optimization
    print("\n" + "="*50)
    print("TRANSFORMER HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    
    best_params, best_score, search_results = classifier.hyperparameter_search(
        X_train, y_train, 
        cv_folds=3,  # Reduced for faster execution
        max_trials=12  # Reduced for faster execution
    )
    
    # Train optimized model
    print("\n" + "="*50)
    print("TRAINING OPTIMIZED TRANSFORMER")
    print("="*50)
    
    model, history = classifier.train_optimized_model(X_train, y_train, epochs=100)
    
    # Test set evaluation
    test_results = classifier.evaluate_on_test(X_test, y_test, label_encoder)
    
    # Save model and results
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    classifier.save_model('G4_transformer_model.h5')
    
    # Save comprehensive results
    results = {
        'experiment': 'G4_Raw_Augmentation_Transformer',
        'preprocessing': 'Minimal + Comprehensive Data Augmentation',
        'algorithm': 'Transformer (Multi-Head Self-Attention)',
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
        'sequence_length': sequence_length,
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0])
    }
    
    with open('G4_experimental_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary report
    training_hist = classifier.get_training_history()
    final_val_acc = training_hist['val_accuracy'][-1] if training_hist else 0
    
    summary = f"""
G4 EXPERIMENT SUMMARY: Raw + Augmentation + Transformer
======================================================

METHODOLOGY:
- Preprocessing: Minimal (preserving raw information)
- Data Augmentation: Comprehensive suite (noise, scaling, shifting, etc.)
- Algorithm: Transformer with Multi-Head Self-Attention
- Input: Raw spectral sequences with positional encoding
- Optimization: Grid search with cross-validation

DATASET:
- Training samples: {X_train.shape[0]} (augmented from {X_train.shape[0]//2} original)
- Test samples: {X_test.shape[0]}
- Sequence length: {sequence_length} wavelengths
- Classes: {', '.join(label_encoder.classes_)}

TRANSFORMER ARCHITECTURE:
 - Embedding dimension: {best_params.get('embed_dim', 'N/A') if best_params else 'N/A'}
 - Attention heads: {best_params.get('num_heads', 'N/A') if best_params else 'N/A'}
 - Feed-forward dimension: {best_params.get('ff_dim', 'N/A') if best_params else 'N/A'}
 - Transformer layers: {best_params.get('num_layers', 'N/A') if best_params else 'N/A'}
 - Dropout rate: {best_params.get('dropout_rate', 'N/A') if best_params else 'N/A'}
 - Learning rate: {best_params.get('learning_rate', 'N/A') if best_params else 'N/A'}

TRAINING RESULTS:
- Best CV Score: {best_score:.4f}
- Final Validation Accuracy: {final_val_acc:.4f}
- Training Epochs: {training_hist['epochs'] if training_hist else 'N/A'}

TEST SET RESULTS:
- Transformer Test Accuracy: {test_results['test_accuracy']:.4f}

CONFUSION MATRIX:
{test_results['confusion_matrix']}

OPTIMIZATION RESULTS:
- Search method: Grid search with 3-fold CV
- Trials completed: {len(search_results)}
- Best parameters: {best_params}

FILES GENERATED:
- G4_transformer_model.h5 (trained Transformer model)
- G4_experimental_results.json (complete results)
- G4_model_metadata.json (model metadata)
- test_predictions.csv (test set predictions)

NOTES:
- Data augmentation addresses class imbalance effectively
- Transformer attention mechanism learns spectral dependencies
- Minimal preprocessing preserves maximum information
- Self-attention captures long-range spectral relationships
"""
    
    with open('G4_performance_summary.txt', 'w') as f:
        f.write(summary)
    
    print("✓ Saved G4_experimental_results.json")
    print("✓ Saved G4_performance_summary.txt")
    print("✓ Saved test_predictions.csv")
    
    print(f"\n🎯 G4 TRANSFORMER TEST ACCURACY: {test_results['test_accuracy']:.4f}")
    print("✅ G4 experiment completed successfully!")

if __name__ == "__main__":
    main() 