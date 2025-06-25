import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class TransferLearningClassifier:
    """
    Advanced transfer learning classifier for gender prediction
    Combines pre-trained feature extractors with domain adaptation
    """
    
    def __init__(self, input_dim, random_state=42):
        """
        Initialize transfer learning classifier
        
        Args:
            input_dim: Number of input features
            random_state: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        
    def create_autoencoder(self, encoding_dim=32):
        """
        Create autoencoder for feature learning
        """
        print(f"Creating autoencoder (encoding_dim={encoding_dim})...")
        
        # Encoder
        input_layer = tf.keras.layers.Input(shape=(self.input_dim,))
        encoded = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
        encoded = tf.keras.layers.BatchNormalization()(encoded)
        encoded = tf.keras.layers.Dropout(0.2)(encoded)
        encoded = tf.keras.layers.Dense(encoding_dim, activation='relu', name='encoded')(encoded)
        
        # Decoder
        decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
        decoded = tf.keras.layers.BatchNormalization()(decoded)
        decoded = tf.keras.layers.Dropout(0.2)(decoded)
        decoded = tf.keras.layers.Dense(self.input_dim, activation='linear')(decoded)
        
        autoencoder = tf.keras.Model(input_layer, decoded)
        encoder = tf.keras.Model(input_layer, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder, encoder
    
    def pretrain_autoencoder(self, X_train, epochs=80):
        """
        Pre-train autoencoder
        """
        print("Pre-training autoencoder...")
        
        autoencoder, encoder = self.create_autoencoder()
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        
        history = autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        self.models['autoencoder'] = autoencoder
        self.models['encoder'] = encoder
        
        print("✓ Autoencoder pre-training completed!")
        return encoder, history
    
    def create_transfer_classifier(self, encoder):
        """
        Create transfer learning classifier
        """
        print("Creating transfer classifier...")
        
        input_layer = tf.keras.layers.Input(shape=(self.input_dim,))
        encoded_features = encoder(input_layer)
        
        # Classification head
        x = tf.keras.layers.Dense(64, activation='relu')(encoded_features)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(input_layer, output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fine_tune_model(self, model, encoder, X_train, y_train, X_val, y_val):
        """
        Fine-tune the transfer model
        """
        print("Fine-tuning transfer model...")
        
        # Phase 1: Frozen encoder
        for layer in encoder.layers:
            layer.trainable = False
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True
        )
        
        history1 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Phase 2: Fine-tune all
        for layer in encoder.layers:
            layer.trainable = True
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        combined_history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
        }
        
        print("✓ Fine-tuning completed!")
        return model, combined_history
    
    def train_ensemble_models(self, X_train, y_train, encoder):
        """
        Train ensemble with encoded features
        """
        print("Training ensemble models...")
        
        encoded_features = encoder.predict(X_train, verbose=0)
        
        # Individual models
        rf = RandomForestClassifier(n_estimators=200, random_state=self.random_state)
        svm = SVC(probability=True, random_state=self.random_state)
        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        # Train models
        rf.fit(encoded_features, y_train)
        svm.fit(encoded_features, y_train)
        lr.fit(encoded_features, y_train)
        
        # Ensemble
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('svm', svm), ('lr', lr)],
            voting='soft'
        )
        ensemble.fit(encoded_features, y_train)
        
        self.models.update({
            'random_forest': rf,
            'svm': svm,
            'logistic_regression': lr,
            'ensemble': ensemble
        })
        
        print("✓ Ensemble training completed!")
        return {'rf': rf, 'svm': svm, 'lr': lr, 'ensemble': ensemble}
    
    def evaluate_models(self, X_test, y_test, label_encoder):
        """
        Evaluate all models
        """
        print("\n" + "="*50)
        print("TRANSFER LEARNING EVALUATION")
        print("="*50)
        
        results = {}
        encoder = self.models['encoder']
        
        # Deep transfer model
        if 'transfer_deep' in self.models:
            print("\n=== DEEP TRANSFER MODEL ===")
            deep_model = self.models['transfer_deep']
            deep_pred_proba = deep_model.predict(X_test, verbose=0).flatten()
            deep_pred = (deep_pred_proba > 0.5).astype(int)
            deep_accuracy = accuracy_score(y_test, deep_pred)
            
            print(f"Deep Transfer Accuracy: {deep_accuracy:.4f}")
            print(classification_report(y_test, deep_pred, target_names=label_encoder.classes_))
            print(confusion_matrix(y_test, deep_pred))
            
            results['deep_transfer'] = {
                'accuracy': deep_accuracy,
                'predictions': deep_pred,
                'probabilities': deep_pred_proba
            }
        
        # Ensemble models
        encoded_test = encoder.predict(X_test, verbose=0)
        
        for model_name in ['random_forest', 'svm', 'logistic_regression', 'ensemble']:
            if model_name in self.models:
                print(f"\n=== {model_name.upper().replace('_', ' ')} ===")
                model = self.models[model_name]
                pred = model.predict(encoded_test)
                pred_proba = model.predict_proba(encoded_test)[:, 1]
                accuracy = accuracy_score(y_test, pred)
                
                print(f"{model_name.title()} Accuracy: {accuracy:.4f}")
                print(classification_report(y_test, pred, target_names=label_encoder.classes_))
                print(confusion_matrix(y_test, pred))
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'predictions': pred,
                    'probabilities': pred_proba
                }
        
        # Best model
        best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_model]['accuracy']
        
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"🎯 BEST ACCURACY: {best_accuracy:.4f}")
        
        return results, best_model, best_accuracy

def main():
    """
    Main training pipeline for G6 transfer learning experiment
    """
    print("="*70)
    print("G6 TRANSFER LEARNING: SNV + Optimized + Transfer Learning")
    print("="*70)
    
    # Load processed data
    print("Loading processed transfer learning data...")
    X_train = np.load('X_train_processed.npy')
    X_test = np.load('X_test_processed.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    # Load encoders
    label_encoder = joblib.load('label_encoder.pkl')
    
    print(f"✓ Training set: {X_train.shape}")
    print(f"✓ Test set: {X_test.shape}")
    print(f"✓ Classes: {label_encoder.classes_}")
    
    # Initialize transfer learning classifier
    classifier = TransferLearningClassifier(input_dim=X_train.shape[1], random_state=42)
    
    # Step 1: Pre-train autoencoder
    print("\n" + "="*50)
    print("AUTOENCODER PRE-TRAINING")
    print("="*50)
    
    encoder, ae_history = classifier.pretrain_autoencoder(X_train)
    
    # Step 2: Deep transfer learning
    print("\n" + "="*50)
    print("DEEP TRANSFER LEARNING")
    print("="*50)
    
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    transfer_model = classifier.create_transfer_classifier(encoder)
    fine_tuned_model, ft_history = classifier.fine_tune_model(
        transfer_model, encoder, X_train_split, y_train_split, X_val_split, y_val_split
    )
    classifier.models['transfer_deep'] = fine_tuned_model
    
    # Step 3: Ensemble models
    print("\n" + "="*50)
    print("ENSEMBLE TRANSFER LEARNING")
    print("="*50)
    
    ensemble_models = classifier.train_ensemble_models(X_train, y_train, encoder)
    
    # Step 4: Evaluation
    results, best_model, best_accuracy = classifier.evaluate_models(X_test, y_test, label_encoder)
    
    # Step 5: Save results
    print("\n" + "="*50)
    print("SAVING RESULTS")
    print("="*50)
    
    # Save models
    classifier.models['autoencoder'].save('G6_autoencoder.h5')
    classifier.models['encoder'].save('G6_encoder.h5')
    classifier.models['transfer_deep'].save('G6_transfer_classifier.h5')
    
    classical_models = {name: classifier.models[name] for name in 
                       ['random_forest', 'svm', 'logistic_regression', 'ensemble']}
    joblib.dump(classical_models, 'G6_classical_models.pkl')
    
    # Results
    experiment_results = {
        'experiment': 'G6_SNV_Optimized_Transfer',
        'preprocessing': 'SNV + Optimized + Transfer Learning',
        'algorithm': 'Transfer Learning (Autoencoder + Deep/Ensemble)',
        'best_model': best_model,
        'best_accuracy': float(best_accuracy),
        'input_features': int(X_train.shape[1]),
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'model_results': {}
    }
    
    for model_name, result in results.items():
        experiment_results['model_results'][model_name] = {
            'accuracy': float(result['accuracy']),
            'predictions': result['predictions'].tolist(),
            'probabilities': result['probabilities'].tolist()
        }
    
    with open('G6_experimental_results.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    # Summary
    summary = f"""
G6 EXPERIMENT SUMMARY: SNV + Optimized + Transfer Learning
==========================================================

METHODOLOGY:
- Preprocessing: SNV + Optimized Feature Selection + Transfer Learning Scaling
- Algorithm: Transfer Learning with Autoencoder Pre-training
- Models: Deep Transfer + Ensemble (RF, SVM, LR)

DATASET:
- Training samples: {X_train.shape[0]}
- Test samples: {X_test.shape[0]}
- Features: {X_train.shape[1]} optimized transfer learning features
- Classes: {', '.join(label_encoder.classes_)}

MODEL RESULTS:
"""
    
    for model_name, result in results.items():
        summary += f"- {model_name.replace('_', ' ').title()}: {result['accuracy']:.4f}\n"
    
    summary += f"""
BEST MODEL: {best_model.replace('_', ' ').title()}
BEST ACCURACY: {best_accuracy:.4f}

TRANSFER LEARNING PIPELINE:
1. Autoencoder pre-training for feature learning
2. Transfer classifier with fine-tuning
3. Ensemble models on encoded features
4. Multi-model evaluation and selection

FILES GENERATED:
- G6_autoencoder.h5 (pre-trained autoencoder)
- G6_encoder.h5 (feature extractor)
- G6_transfer_classifier.h5 (transfer classifier)
- G6_classical_models.pkl (ensemble models)
- G6_experimental_results.json (complete results)
"""
    
    with open('G6_performance_summary.txt', 'w') as f:
        f.write(summary)
    
    print("✓ Saved results and models")
    print(f"\n🎯 G6 RESULTS: {best_model} - {best_accuracy:.4f}")
    print("✅ G6 experiment completed!")

if __name__ == "__main__":
    main() 