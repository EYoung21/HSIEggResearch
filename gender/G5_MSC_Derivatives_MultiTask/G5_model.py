import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class MultiTaskDeepClassifier:
    """
    Multi-task deep learning classifier for simultaneous gender and mortality prediction
    Uses shared feature extraction with separate task-specific heads
    """
    
    def __init__(self, input_dim, random_state=42):
        """
        Initialize multi-task classifier
        
        Args:
            input_dim: Number of input features
            random_state: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.random_state = random_state
        self.model = None
        self.history = None
        self.best_params = {}
        
    def create_multi_task_architecture(self,
                                     hidden_layers=[512, 256, 128],
                                     dropout_rate=0.3,
                                     learning_rate=0.001,
                                     task_weight_gender=1.0,
                                     task_weight_mortality=1.0):
        """
        Create multi-task deep learning architecture
        
        Args:
            hidden_layers: List of hidden layer sizes for shared network
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            task_weight_gender: Weight for gender task loss
            task_weight_mortality: Weight for mortality task loss
            
        Returns:
            Compiled multi-task model
        """
        
        print(f"Creating multi-task architecture with shared layers: {hidden_layers}")
        
        # Input layer
        inputs = tf.keras.layers.Input(shape=(self.input_dim,), name='spectral_input')
        
        # Shared feature extraction layers
        x = inputs
        for i, units in enumerate(hidden_layers):
            x = tf.keras.layers.Dense(
                units, 
                activation='relu', 
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                name=f'shared_dense_{i+1}'
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f'shared_bn_{i+1}')(x)
            x = tf.keras.layers.Dropout(dropout_rate, name=f'shared_dropout_{i+1}')(x)
        
        # Shared representation
        shared_features = tf.keras.layers.Dense(
            64, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name='shared_representation'
        )(x)
        shared_features = tf.keras.layers.BatchNormalization(name='shared_representation_bn')(shared_features)
        shared_features = tf.keras.layers.Dropout(dropout_rate, name='shared_representation_dropout')(shared_features)
        
        # Task-specific heads
        
        # Gender prediction head
        gender_head = tf.keras.layers.Dense(
            32, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name='gender_head_dense'
        )(shared_features)
        gender_head = tf.keras.layers.BatchNormalization(name='gender_head_bn')(gender_head)
        gender_head = tf.keras.layers.Dropout(dropout_rate * 0.5, name='gender_head_dropout')(gender_head)
        gender_output = tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            name='gender_output'
        )(gender_head)
        
        # Mortality prediction head
        mortality_head = tf.keras.layers.Dense(
            32, 
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            name='mortality_head_dense'
        )(shared_features)
        mortality_head = tf.keras.layers.BatchNormalization(name='mortality_head_bn')(mortality_head)
        mortality_head = tf.keras.layers.Dropout(dropout_rate * 0.5, name='mortality_head_dropout')(mortality_head)
        mortality_output = tf.keras.layers.Dense(
            1, 
            activation='sigmoid', 
            name='mortality_output'
        )(mortality_head)
        
        # Create model
        model = tf.keras.Model(
            inputs=inputs, 
            outputs=[gender_output, mortality_output], 
            name='MultiTaskSpectralClassifier'
        )
        
        # Compile model with task-specific losses and weights
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss={
                'gender_output': 'binary_crossentropy',
                'mortality_output': 'binary_crossentropy'
            },
            loss_weights={
                'gender_output': task_weight_gender,
                'mortality_output': task_weight_mortality
            },
            metrics={
                'gender_output': ['accuracy'],
                'mortality_output': ['accuracy']
            }
        )
        
        return model
    
    def hyperparameter_search(self, X_train, y_gender_train, y_mortality_train, cv_folds=3, max_trials=10):
        """
        Hyperparameter search for multi-task architecture
        
        Args:
            X_train: Training features
            y_gender_train: Gender training labels
            y_mortality_train: Mortality training labels
            cv_folds: Number of CV folds
            max_trials: Maximum number of parameter combinations
            
        Returns:
            Best parameters and CV scores
        """
        print(f"Starting multi-task hyperparameter search with {max_trials} trials...")
        
        # Define parameter combinations for multi-task learning
        param_combinations = [
            # (hidden_layers, dropout_rate, learning_rate, task_weight_gender, task_weight_mortality)
            ([512, 256, 128], 0.3, 0.001, 1.0, 1.0),  # Balanced tasks
            ([256, 128, 64], 0.2, 0.001, 1.0, 1.0),   # Smaller network
            ([512, 256], 0.3, 0.001, 1.0, 1.0),       # Fewer layers
            ([512, 256, 128], 0.4, 0.0005, 1.0, 1.0), # Higher dropout
            ([256, 128, 64], 0.3, 0.002, 1.0, 1.0),   # Higher learning rate
            ([512, 256, 128], 0.3, 0.001, 1.2, 0.8),  # Gender priority
            ([512, 256, 128], 0.3, 0.001, 0.8, 1.2),  # Mortality priority
            ([384, 192, 96], 0.25, 0.001, 1.0, 1.0),  # Alternative sizes
            ([512, 256, 128, 64], 0.3, 0.001, 1.0, 1.0), # Deeper network
            ([256, 128], 0.2, 0.001, 1.0, 1.0),       # Simple network
        ]
        
        best_score = 0
        best_params = None
        results = []
        
        for trial, params in enumerate(param_combinations[:max_trials]):
            hidden_layers, dropout_rate, learning_rate, task_weight_gender, task_weight_mortality = params
            
            print(f"\nTrial {trial+1}/{max_trials}: layers={hidden_layers}, dropout={dropout_rate}, "
                  f"lr={learning_rate}, weights=({task_weight_gender:.1f}, {task_weight_mortality:.1f})")
            
            try:
                # Cross-validation
                cv_scores_gender = []
                cv_scores_mortality = []
                cv_scores_combined = []
                
                kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                
                for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_gender_train)):
                    X_train_fold = X_train[train_idx]
                    X_val_fold = X_train[val_idx]
                    y_gender_train_fold = y_gender_train[train_idx]
                    y_gender_val_fold = y_gender_train[val_idx]
                    y_mortality_train_fold = y_mortality_train[train_idx]
                    y_mortality_val_fold = y_mortality_train[val_idx]
                    
                    # Create and train model
                    model = self.create_multi_task_architecture(
                        hidden_layers=hidden_layers,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate,
                        task_weight_gender=task_weight_gender,
                        task_weight_mortality=task_weight_mortality
                    )
                    
                    # Early stopping
                    early_stop = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        verbose=0
                    )
                    
                    # Train model
                    model.fit(
                        X_train_fold, 
                        {
                            'gender_output': y_gender_train_fold,
                            'mortality_output': y_mortality_train_fold
                        },
                        validation_data=(
                            X_val_fold,
                            {
                                'gender_output': y_gender_val_fold,
                                'mortality_output': y_mortality_val_fold
                            }
                        ),
                        epochs=50,  # Moderate epochs for search
                        batch_size=32,
                        callbacks=[early_stop],
                        verbose=0
                    )
                    
                    # Evaluate on validation fold
                    predictions = model.predict(X_val_fold, verbose=0)
                    gender_pred = (predictions[0] > 0.5).astype(int).flatten()
                    mortality_pred = (predictions[1] > 0.5).astype(int).flatten()
                    
                    # Calculate fold scores
                    gender_acc = accuracy_score(y_gender_val_fold, gender_pred)
                    mortality_acc = accuracy_score(y_mortality_val_fold, mortality_pred)
                    combined_acc = (gender_acc + mortality_acc) / 2  # Average of both tasks
                    
                    cv_scores_gender.append(gender_acc)
                    cv_scores_mortality.append(mortality_acc)
                    cv_scores_combined.append(combined_acc)
                    
                    # Clean up
                    del model
                    tf.keras.backend.clear_session()
                
                # Calculate mean CV scores
                mean_gender = np.mean(cv_scores_gender)
                mean_mortality = np.mean(cv_scores_mortality)
                mean_combined = np.mean(cv_scores_combined)
                
                std_gender = np.std(cv_scores_gender)
                std_mortality = np.std(cv_scores_mortality)
                std_combined = np.std(cv_scores_combined)
                
                print(f"  Gender CV Score: {mean_gender:.4f} ± {std_gender:.4f}")
                print(f"  Mortality CV Score: {mean_mortality:.4f} ± {std_mortality:.4f}")
                print(f"  Combined CV Score: {mean_combined:.4f} ± {std_combined:.4f}")
                
                results.append({
                    'params': params,
                    'gender_cv_score': mean_gender,
                    'mortality_cv_score': mean_mortality,
                    'combined_cv_score': mean_combined,
                    'gender_cv_std': std_gender,
                    'mortality_cv_std': std_mortality,
                    'combined_cv_std': std_combined
                })
                
                # Update best parameters (based on combined score)
                if mean_combined > best_score:
                    best_score = mean_combined
                    best_params = {
                        'hidden_layers': hidden_layers,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate,
                        'task_weight_gender': task_weight_gender,
                        'task_weight_mortality': task_weight_mortality
                    }
                    print(f"  ✓ New best combined score!")
                
            except Exception as e:
                print(f"  Error in trial {trial+1}: {e}")
                continue
        
        self.best_params = best_params
        print(f"\n✓ Multi-task hyperparameter search completed!")
        print(f"✓ Best combined CV score: {best_score:.4f}")
        print(f"✓ Best parameters: {best_params}")
        
        return best_params, best_score, results
    
    def train_optimized_model(self, X_train, y_gender_train, y_mortality_train, 
                            X_val=None, y_gender_val=None, y_mortality_val=None, epochs=150):
        """
        Train multi-task model with optimized hyperparameters
        
        Args:
            X_train: Training features
            y_gender_train: Gender training labels
            y_mortality_train: Mortality training labels
            X_val: Validation features (optional)
            y_gender_val: Gender validation labels (optional)
            y_mortality_val: Mortality validation labels (optional)
            epochs: Maximum training epochs
            
        Returns:
            Trained model and training history
        """
        if not self.best_params:
            raise ValueError("No optimized parameters found. Run hyperparameter_search first.")
        
        print("Training optimized multi-task model...")
        
        # Create model with best parameters
        self.model = self.create_multi_task_architecture(**self.best_params)
        
        print("\nMulti-Task Model Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Prepare validation data
        if X_val is not None and y_gender_val is not None and y_mortality_val is not None:
            validation_data = (
                X_val,
                {
                    'gender_output': y_gender_val,
                    'mortality_output': y_mortality_val
                }
            )
        else:
            # Use 20% of training data for validation
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_gender_train_split, y_gender_val_split, y_mortality_train_split, y_mortality_val_split = train_test_split(
                X_train, y_gender_train, y_mortality_train, 
                test_size=0.2, random_state=self.random_state, 
                stratify=y_gender_train  # Stratify by gender for now
            )
            X_train, y_gender_train, y_mortality_train = X_train_split, y_gender_train_split, y_mortality_train_split
            validation_data = (
                X_val_split,
                {
                    'gender_output': y_gender_val_split,
                    'mortality_output': y_mortality_val_split
                }
            )
        
        # Train model
        self.history = self.model.fit(
            X_train,
            {
                'gender_output': y_gender_train,
                'mortality_output': y_mortality_train
            },
            validation_data=validation_data,
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✓ Multi-task training completed!")
        return self.model, self.history
    
    def evaluate_on_test(self, X_test, y_gender_test, y_mortality_test, 
                        gender_encoder, mortality_encoder):
        """
        Comprehensive evaluation on test set for both tasks
        
        Args:
            X_test: Test features
            y_gender_test: Gender test labels
            y_mortality_test: Mortality test labels
            gender_encoder: Gender label encoder
            mortality_encoder: Mortality label encoder
            
        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_optimized_model first.")
        
        print("\n" + "="*50)
        print("MULTI-TASK TEST SET EVALUATION")
        print("="*50)
        
        # Predictions
        predictions = self.model.predict(X_test, verbose=0)
        gender_pred_proba = predictions[0].flatten()
        mortality_pred_proba = predictions[1].flatten()
        
        gender_pred = (gender_pred_proba > 0.5).astype(int)
        mortality_pred = (mortality_pred_proba > 0.5).astype(int)
        
        # Test accuracies
        gender_accuracy = accuracy_score(y_gender_test, gender_pred)
        mortality_accuracy = accuracy_score(y_mortality_test, mortality_pred)
        combined_accuracy = (gender_accuracy + mortality_accuracy) / 2
        
        print(f"Multi-Task Test Results:")
        print(f"  Gender Task Accuracy: {gender_accuracy:.4f}")
        print(f"  Mortality Task Accuracy: {mortality_accuracy:.4f}")
        print(f"  Combined Accuracy: {combined_accuracy:.4f}")
        
        # Gender task evaluation
        print(f"\n=== GENDER TASK EVALUATION ===")
        gender_class_names = gender_encoder.classes_
        print(f"Gender Classification Report:")
        print(classification_report(y_gender_test, gender_pred, target_names=gender_class_names))
        
        print(f"Gender Confusion Matrix:")
        gender_cm = confusion_matrix(y_gender_test, gender_pred)
        print(gender_cm)
        
        # Mortality task evaluation
        print(f"\n=== MORTALITY TASK EVALUATION ===")
        mortality_class_names = mortality_encoder.classes_
        print(f"Mortality Classification Report:")
        print(classification_report(y_mortality_test, mortality_pred, target_names=mortality_class_names))
        
        print(f"Mortality Confusion Matrix:")
        mortality_cm = confusion_matrix(y_mortality_test, mortality_pred)
        print(mortality_cm)
        
        # Joint task analysis
        print(f"\n=== JOINT TASK ANALYSIS ===")
        print("Gender vs Mortality Prediction Accuracy:")
        for i, gender_class in enumerate(gender_class_names):
            for j, mortality_class in enumerate(mortality_class_names):
                # Find samples with this joint label
                joint_mask = (y_gender_test == i) & (y_mortality_test == j)
                if np.sum(joint_mask) > 0:
                    gender_acc_joint = accuracy_score(y_gender_test[joint_mask], gender_pred[joint_mask])
                    mortality_acc_joint = accuracy_score(y_mortality_test[joint_mask], mortality_pred[joint_mask])
                    print(f"  {gender_class}-{mortality_class}: {np.sum(joint_mask)} samples, "
                          f"Gender: {gender_acc_joint:.3f}, Mortality: {mortality_acc_joint:.3f}")
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'gender_true': y_gender_test,
            'gender_pred': gender_pred,
            'gender_prob': gender_pred_proba,
            'mortality_true': y_mortality_test,
            'mortality_pred': mortality_pred,
            'mortality_prob': mortality_pred_proba,
        })
        
        predictions_df.to_csv('test_predictions.csv', index=False)
        
        return {
            'gender_accuracy': gender_accuracy,
            'mortality_accuracy': mortality_accuracy,
            'combined_accuracy': combined_accuracy,
            'gender_predictions': gender_pred,
            'mortality_predictions': mortality_pred,
            'gender_probabilities': gender_pred_proba,
            'mortality_probabilities': mortality_pred_proba,
            'gender_confusion_matrix': gender_cm,
            'mortality_confusion_matrix': mortality_cm,
            'gender_classification_report': classification_report(y_gender_test, gender_pred, 
                                                                target_names=gender_class_names, output_dict=True),
            'mortality_classification_report': classification_report(y_mortality_test, mortality_pred, 
                                                                   target_names=mortality_class_names, output_dict=True)
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
            'gender_output_accuracy': self.history.history['gender_output_accuracy'],
            'mortality_output_accuracy': self.history.history['mortality_output_accuracy'],
            'val_loss': self.history.history['val_loss'],
            'val_gender_output_accuracy': self.history.history['val_gender_output_accuracy'],
            'val_mortality_output_accuracy': self.history.history['val_mortality_output_accuracy'],
            'epochs': len(self.history.history['loss'])
        }
    
    def save_model(self, filepath='G5_multitask_model.h5'):
        """
        Save trained multi-task model
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        self.model.save(filepath)
        
        # Save additional metadata
        metadata = {
            'best_params': self.best_params,
            'input_dim': self.input_dim,
            'training_history': self.get_training_history()
        }
        
        with open('G5_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ Model saved to {filepath}")
        print(f"✓ Metadata saved to G5_model_metadata.json")

def main():
    """
    Main training pipeline for G5 multi-task experiment
    """
    print("="*70)
    print("G5 MULTI-TASK MODEL: MSC + Derivatives + Multi-Task Learning")
    print("="*70)
    
    # Load processed data
    print("Loading processed multi-task data...")
    X_train = np.load('X_train_processed.npy')
    X_test = np.load('X_test_processed.npy')
    y_gender_train = np.load('y_gender_train.npy')
    y_gender_test = np.load('y_gender_test.npy')
    y_mortality_train = np.load('y_mortality_train.npy')
    y_mortality_test = np.load('y_mortality_test.npy')
    
    # Load encoders
    gender_encoder = joblib.load('gender_encoder.pkl')
    mortality_encoder = joblib.load('mortality_encoder.pkl')
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Gender classes: {gender_encoder.classes_}")
    print(f"✓ Mortality classes: {mortality_encoder.classes_}")
    
    # Initialize multi-task classifier
    classifier = MultiTaskDeepClassifier(input_dim=X_train.shape[1], random_state=42)
    
    # Hyperparameter optimization
    print("\n" + "="*50)
    print("MULTI-TASK HYPERPARAMETER OPTIMIZATION")
    print("="*50)
    
    best_params, best_score, search_results = classifier.hyperparameter_search(
        X_train, y_gender_train, y_mortality_train,
        cv_folds=3,
        max_trials=8  # Reduced for faster execution
    )
    
    # Train optimized model
    print("\n" + "="*50)
    print("TRAINING OPTIMIZED MULTI-TASK MODEL")
    print("="*50)
    
    model, history = classifier.train_optimized_model(
        X_train, y_gender_train, y_mortality_train, epochs=150
    )
    
    # Test set evaluation
    test_results = classifier.evaluate_on_test(
        X_test, y_gender_test, y_mortality_test, 
        gender_encoder, mortality_encoder
    )
    
    # Save model and results
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    classifier.save_model('G5_multitask_model.h5')
    
    # Save comprehensive results
    results = {
        'experiment': 'G5_MSC_Derivatives_MultiTask',
        'preprocessing': 'MSC + Multi-Scale SG Derivatives (1st, 2nd, 3rd order)',
        'algorithm': 'Multi-Task Deep Learning (Shared + Task-Specific Heads)',
        'tasks': ['Gender Classification', 'Mortality Prediction'],
        'hyperparameter_optimization': {
            'method': 'Grid search with cross-validation',
            'trials': len(search_results),
            'best_combined_cv_score': float(best_score)
        },
        'best_parameters': best_params,
        'test_results': {
            'gender_accuracy': float(test_results['gender_accuracy']),
            'mortality_accuracy': float(test_results['mortality_accuracy']),
            'combined_accuracy': float(test_results['combined_accuracy']),
            'gender_confusion_matrix': test_results['gender_confusion_matrix'].tolist(),
            'mortality_confusion_matrix': test_results['mortality_confusion_matrix'].tolist(),
        },
        'training_history': classifier.get_training_history(),
        'input_features': int(X_train.shape[1]),
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0])
    }
    
    with open('G5_experimental_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary report
    training_hist = classifier.get_training_history()
    final_val_gender_acc = training_hist['val_gender_output_accuracy'][-1] if training_hist else 0
    final_val_mortality_acc = training_hist['val_mortality_output_accuracy'][-1] if training_hist else 0
    
    summary = f"""
G5 EXPERIMENT SUMMARY: MSC + Multi-Scale Derivatives + Multi-Task Learning
==========================================================================

METHODOLOGY:
- Preprocessing: MSC + Multi-Scale Savitzky-Golay Derivatives (1st, 2nd, 3rd)
- Algorithm: Multi-Task Deep Learning with Shared Feature Extraction
- Tasks: Simultaneous Gender and Mortality Prediction
- Architecture: Shared layers + Task-specific heads
- Optimization: Grid search with cross-validation

DATASET:
- Training samples: {X_train.shape[0]}
- Test samples: {X_test.shape[0]}
- Input features: {X_train.shape[1]} (4x derivative scales)
- Task 1 (Gender): {', '.join(gender_encoder.classes_)}
- Task 2 (Mortality): {', '.join(mortality_encoder.classes_)}

MULTI-TASK ARCHITECTURE:
- Shared layers: {best_params.get('hidden_layers', 'N/A')}
- Dropout rate: {best_params.get('dropout_rate', 'N/A')}
- Learning rate: {best_params.get('learning_rate', 'N/A')}
- Task weights: Gender={best_params.get('task_weight_gender', 'N/A')}, Mortality={best_params.get('task_weight_mortality', 'N/A')}

TRAINING RESULTS:
- Best Combined CV Score: {best_score:.4f}
- Final Val Gender Accuracy: {final_val_gender_acc:.4f}
- Final Val Mortality Accuracy: {final_val_mortality_acc:.4f}
- Training Epochs: {training_hist['epochs'] if training_hist else 'N/A'}

TEST SET RESULTS:
- Gender Task Accuracy: {test_results['gender_accuracy']:.4f}
- Mortality Task Accuracy: {test_results['mortality_accuracy']:.4f}
- Combined Task Accuracy: {test_results['combined_accuracy']:.4f}

CONFUSION MATRICES:
Gender Task:
{test_results['gender_confusion_matrix']}

Mortality Task:
{test_results['mortality_confusion_matrix']}

OPTIMIZATION RESULTS:
- Search method: Grid search with 3-fold CV
- Trials completed: {len(search_results)}
- Best parameters: {best_params}

FILES GENERATED:
- G5_multitask_model.h5 (trained multi-task model)
- G5_experimental_results.json (complete results)
- G5_model_metadata.json (model metadata)
- test_predictions.csv (test set predictions for both tasks)

NOTES:
- Multi-task learning leverages shared spectral features
- Simultaneous prediction improves feature utilization
- Task-specific heads capture unique patterns for each task
- Proven MSC + derivatives preprocessing from G1
"""
    
    with open('G5_performance_summary.txt', 'w') as f:
        f.write(summary)
    
    print("✓ Saved G5_experimental_results.json")
    print("✓ Saved G5_performance_summary.txt")
    print("✓ Saved test_predictions.csv")
    
    print(f"\n🎯 G5 MULTI-TASK TEST RESULTS:")
    print(f"   Gender Task: {test_results['gender_accuracy']:.4f}")
    print(f"   Mortality Task: {test_results['mortality_accuracy']:.4f}")
    print(f"   Combined: {test_results['combined_accuracy']:.4f}")
    print("✅ G5 experiment completed successfully!")

if __name__ == "__main__":
    main() 