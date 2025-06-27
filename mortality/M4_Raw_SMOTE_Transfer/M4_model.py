"""
M4 Model: Raw + SMOTE + Transfer Learning for Mortality Classification
Transfer learning approach using pre-trained features and domain adaptation
"""

import numpy as np
import pandas as pd
import time
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

class TransferLearningAdapter(BaseEstimator, ClassifierMixin):
    """Transfer learning adapter for domain adaptation"""
    
    def __init__(self, base_model, feature_extractor=None, adaptation_layers=None, random_state=42):
        self.base_model = base_model
        self.feature_extractor = feature_extractor
        self.adaptation_layers = adaptation_layers or []
        self.random_state = random_state
        
        # Fitted components
        self.is_fitted = False
        self.feature_scaler = StandardScaler()
        
    def extract_transferable_features(self, X):
        """Extract features that can be transferred across domains"""
        if self.feature_extractor is not None:
            return self.feature_extractor.transform(X)
        return X
    
    def fit(self, X, y):
        """Fit the transfer learning model"""
        # Extract transferable features
        X_features = self.extract_transferable_features(X)
        
        # Scale features for transfer
        X_scaled = self.feature_scaler.fit_transform(X_features)
        
        # Apply adaptation layers if specified
        X_adapted = X_scaled
        for layer in self.adaptation_layers:
            X_adapted = layer.fit_transform(X_adapted, y)
        
        # Fit base model
        self.base_model.fit(X_adapted, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Predict using transfer learning model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract and transform features
        X_features = self.extract_transferable_features(X)
        X_scaled = self.feature_scaler.transform(X_features)
        
        # Apply adaptation layers
        X_adapted = X_scaled
        for layer in self.adaptation_layers:
            X_adapted = layer.transform(X_adapted)
        
        return self.base_model.predict(X_adapted)
    
    def predict_proba(self, X):
        """Predict probabilities using transfer learning model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract and transform features
        X_features = self.extract_transferable_features(X)
        X_scaled = self.feature_scaler.transform(X_features)
        
        # Apply adaptation layers
        X_adapted = X_scaled
        for layer in self.adaptation_layers:
            X_adapted = layer.transform(X_adapted)
        
        if hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X_adapted)
        else:
            # For models without predict_proba, return decision function as probability
            decisions = self.base_model.decision_function(X_adapted)
            # Convert to probabilities using sigmoid
            probs = 1 / (1 + np.exp(-decisions))
            return np.column_stack([1 - probs, probs])

class MortalityTransferModel:
    """M4 Model: Raw + SMOTE + Transfer Learning for mortality classification"""
    
    def __init__(self, transfer_strategy='feature_extraction', random_state=42):
        self.transfer_strategy = transfer_strategy
        self.random_state = random_state
        
        # Models and components
        self.pre_trained_extractor = None
        self.transfer_models = {}
        self.best_model = None
        
        # Training metadata
        self.training_time = 0
        self.cv_scores = {}
        self.model_performances = {}
        
        # Class balancing
        self.smote_sampler = SMOTEENN(
            smote=SMOTE(random_state=random_state, k_neighbors=3),
            enn=EditedNearestNeighbours(n_neighbors=3),
            random_state=random_state
        )
    
    def create_feature_extractor(self, X_train, n_components=50):
        """Create a feature extractor for transfer learning"""
        print(f"Creating feature extractor with {n_components} components...")
        
        # Use PCA as a simple feature extractor
        extractor = PCA(n_components=min(n_components, X_train.shape[1]), random_state=self.random_state)
        extractor.fit(X_train)
        
        explained_variance = np.sum(extractor.explained_variance_ratio_)
        print(f"✓ Feature extractor created: {explained_variance:.3f} variance explained")
        
        return extractor
    
    def create_transfer_models(self):
        """Create different transfer learning models"""
        print("Creating transfer learning models...")
        
        # Model 1: Neural Network with transfer learning
        mlp_base = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        self.transfer_models['neural_network'] = TransferLearningAdapter(
            base_model=mlp_base,
            feature_extractor=self.pre_trained_extractor,
            random_state=self.random_state
        )
        
        # Model 2: SVM with transfer learning
        svm_base = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=self.random_state
        )
        
        self.transfer_models['svm_transfer'] = TransferLearningAdapter(
            base_model=svm_base,
            feature_extractor=self.pre_trained_extractor,
            random_state=self.random_state
        )
        
        # Model 3: Random Forest with transfer learning
        rf_base = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.transfer_models['random_forest'] = TransferLearningAdapter(
            base_model=rf_base,
            feature_extractor=self.pre_trained_extractor,
            random_state=self.random_state
        )
        
        # Model 4: Deep adaptation model
        deep_mlp_base = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30
        )
        
        # Add PCA adaptation layer for deep model
        adaptation_pca = PCA(n_components=20, random_state=self.random_state)
        
        self.transfer_models['deep_adaptation'] = TransferLearningAdapter(
            base_model=deep_mlp_base,
            feature_extractor=self.pre_trained_extractor,
            adaptation_layers=[adaptation_pca],
            random_state=self.random_state
        )
        
        print(f"✓ Created {len(self.transfer_models)} transfer learning models")
    
    def train_mortality_transfer_models(self, X_train, y_train):
        """Train mortality classification with transfer learning"""
        print("\n" + "="*60)
        print("TRAINING M4: RAW + SMOTE + TRANSFER LEARNING")
        print("="*60)
        
        start_time = time.time()
        
        # Apply SMOTE-ENN for class balancing
        print("Applying SMOTE-ENN for class balancing...")
        print(f"Original distribution: {np.bincount(y_train)}")
        
        try:
            X_resampled, y_resampled = self.smote_sampler.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"Warning: SMOTE-ENN failed, using original data: {e}")
            X_resampled, y_resampled = X_train, y_train
        
        print(f"Resampled distribution: {np.bincount(y_resampled)}")
        print(f"✓ Dataset resampled: {X_resampled.shape[0]} samples")
        
        # Create feature extractor (pre-training step)
        print("\nCreating transfer learning feature extractor...")
        self.pre_trained_extractor = self.create_feature_extractor(X_resampled, n_components=50)
        
        # Create transfer learning models
        self.create_transfer_models()
        
        # Train and evaluate each model
        print("\nTraining transfer learning models...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for model_name, model in self.transfer_models.items():
            print(f"\nTraining {model_name}...")
            
            # Train model
            model.fit(X_resampled, y_resampled)
            
            # Cross-validation
            try:
                cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='accuracy')
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                
                self.cv_scores[f'{model_name}_cv_mean'] = cv_mean
                self.cv_scores[f'{model_name}_cv_std'] = cv_std
                
                print(f"  ✓ {model_name} CV: {cv_mean:.4f} ± {cv_std:.4f}")
                
            except Exception as e:
                print(f"  ⚠ CV failed for {model_name}: {e}")
                self.cv_scores[f'{model_name}_cv_mean'] = 0.5
                self.cv_scores[f'{model_name}_cv_std'] = 0.0
        
        self.training_time = time.time() - start_time
        
        print(f"\n✓ Transfer learning training completed in {self.training_time:.2f} seconds")
        
        return {
            'training_time': self.training_time,
            'cv_scores': self.cv_scores,
            'resampled_shape': X_resampled.shape,
            'models_trained': list(self.transfer_models.keys())
        }
    
    def evaluate_mortality_prediction(self, X_test, y_test, label_encoder):
        """Evaluate mortality prediction with transfer learning"""
        print("\n" + "="*60)
        print("EVALUATING TRANSFER LEARNING MODELS")
        print("="*60)
        
        results = {}
        best_accuracy = 0
        best_model_name = None
        
        for model_name, model in self.transfer_models.items():
            print(f"\nEvaluating {model_name}...")
            
            try:
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, predictions)
                auc = roc_auc_score(y_test, probabilities)
                cm = confusion_matrix(y_test, predictions)
                cr = classification_report(
                    y_test, predictions, target_names=label_encoder.classes_, output_dict=True
                )
                
                results[model_name] = {
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist(),
                    'accuracy': accuracy,
                    'auc': auc,
                    'confusion_matrix': cm.tolist(),
                    'classification_report': cr
                }
                
                self.model_performances[model_name] = {
                    'accuracy': accuracy,
                    'auc': auc
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  AUC: {auc:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = model_name
                    self.best_model = model
                
            except Exception as e:
                print(f"  ❌ Evaluation failed for {model_name}: {e}")
                results[model_name] = {
                    'predictions': [],
                    'probabilities': [],
                    'accuracy': 0.0,
                    'auc': 0.5,
                    'confusion_matrix': [[0, 0], [0, 0]],
                    'classification_report': {}
                }
        
        # Add best model information
        results['best_model'] = {
            'name': best_model_name,
            'accuracy': best_accuracy,
            'auc': results[best_model_name]['auc'] if best_model_name else 0.5
        }
        
        print(f"\nTransfer Learning Results:")
        print(f"  Best Model: {best_model_name}")
        print(f"  Best Accuracy: {best_accuracy:.4f}")
        if best_model_name:
            print(f"  Best AUC: {results[best_model_name]['auc']:.4f}")
        
        return results
    
    def analyze_transfer_features(self, preprocessing_info, top_k=15):
        """Analyze transfer learning feature importance"""
        print(f"\nAnalyzing transfer learning features...")
        
        feature_analysis = {
            'transfer_strategy': self.transfer_strategy,
            'feature_extractor_components': self.pre_trained_extractor.n_components_ if self.pre_trained_extractor else 0,
            'explained_variance': np.sum(self.pre_trained_extractor.explained_variance_ratio_) if self.pre_trained_extractor else 0,
            'model_performances': self.model_performances,
            'total_features': preprocessing_info.get('n_enhanced_features', 0)
        }
        
        if self.pre_trained_extractor is not None:
            # Get most important transferred components
            components = self.pre_trained_extractor.components_
            component_importance = np.sum(np.abs(components), axis=1)
            
            top_components = np.argsort(component_importance)[-top_k:][::-1]
            
            feature_analysis['top_transfer_components'] = [
                {
                    'component': f'Transfer_Component_{i+1}',
                    'importance': float(component_importance[i]),
                    'explained_variance': float(self.pre_trained_extractor.explained_variance_ratio_[i])
                }
                for i in top_components
            ]
        
        print(f"✓ Transfer learning analysis completed")
        return feature_analysis
    
    def save_model_and_results(self, results, feature_analysis, preprocessing_info):
        """Save transfer learning models and results"""
        print("\nSaving M4 transfer learning models and results...")
        
        experiment_results = {
            'experiment_name': 'M4_Raw_SMOTE_Transfer',
            'methodology': {
                'preprocessing': 'Raw spectra with minimal processing',
                'models': 'Transfer Learning (Neural Network, SVM, Random Forest, Deep Adaptation)',
                'class_balancing': 'SMOTE-ENN',
                'transfer_strategy': self.transfer_strategy,
                'feature_extraction': 'PCA-based transfer learning'
            },
            'data_info': {
                'n_samples': preprocessing_info.get('n_samples', 0),
                'n_features': preprocessing_info.get('n_enhanced_features', 0),
                'wavelength_range': preprocessing_info.get('wavelength_range', 'Unknown')
            },
            'training_results': {
                'training_time': self.training_time,
                'cv_scores': self.cv_scores,
                'transfer_components': self.pre_trained_extractor.n_components_ if self.pre_trained_extractor else 0
            },
            'evaluation_results': results,
            'feature_analysis': feature_analysis,
            'preprocessing_stats': preprocessing_info
        }
        
        with open('M4_experimental_results.json', 'w') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(np, 'int64') and isinstance(obj, np.int64):
                    return int(obj)
                elif hasattr(np, 'int32') and isinstance(obj, np.int32):
                    return int(obj)
                elif hasattr(np, 'float64') and isinstance(obj, np.float64):
                    return float(obj)
                elif hasattr(np, 'float32') and isinstance(obj, np.float32):
                    return float(obj)
                else:
                    return str(obj)
            
            json.dump(experiment_results, f, indent=2, default=convert_numpy)
        
        # Performance summary
        best_model = results['best_model']['name']
        best_accuracy = results['best_model']['accuracy']
        best_auc = results['best_model']['auc']
        
        explained_var = np.sum(self.pre_trained_extractor.explained_variance_ratio_) if self.pre_trained_extractor else 0
        
        summary_text = f"""
M4 MORTALITY CLASSIFICATION - TRANSFER LEARNING PERFORMANCE SUMMARY
=================================================================

METHODOLOGY:
- Preprocessing: Raw spectra with minimal processing
- Models: Transfer Learning (Neural Network, SVM, Random Forest, Deep Adaptation)
- Class Balancing: SMOTE-ENN
- Transfer Strategy: {self.transfer_strategy}
- Feature Extraction: PCA-based ({self.pre_trained_extractor.n_components_ if self.pre_trained_extractor else 0} components)

DATASET:
- Samples: {preprocessing_info.get('n_samples', 'Unknown')}
- Features: {preprocessing_info.get('n_enhanced_features', 'Unknown')} enhanced features
- Wavelength Range: {preprocessing_info.get('wavelength_range', 'Unknown')}

TRAINING PERFORMANCE:
- Training Time: {self.training_time:.2f} seconds
- Transfer Components: {self.pre_trained_extractor.n_components_ if self.pre_trained_extractor else 0}
- Explained Variance: {explained_var:.3f}

CROSS-VALIDATION SCORES:"""

        for model_name in self.transfer_models.keys():
            cv_mean = self.cv_scores.get(f'{model_name}_cv_mean', 0)
            cv_std = self.cv_scores.get(f'{model_name}_cv_std', 0)
            summary_text += f"\n- {model_name.replace('_', ' ').title()}: {cv_mean:.4f} ± {cv_std:.4f}"

        summary_text += f"""

TEST PERFORMANCE:
- Best Model: {best_model.replace('_', ' ').title() if best_model else 'None'}
- Test Accuracy: {best_accuracy:.4f}
- Test AUC: {best_auc:.4f}

INDIVIDUAL MODEL RESULTS:"""

        for model_name in self.transfer_models.keys():
            if model_name in results:
                acc = results[model_name]['accuracy']
                auc = results[model_name]['auc']
                summary_text += f"\n- {model_name.replace('_', ' ').title()}: {acc:.4f} (AUC: {auc:.4f})"

        summary_text += f"""

TRANSFER LEARNING INSIGHTS:
- Feature Extraction: PCA-based transfer learning
- Domain Adaptation: Neural network and ensemble approaches
- Raw Data Processing: Minimal preprocessing preserves spectral information
- Class Balancing: SMOTE-ENN handles mortality class imbalance

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open('M4_performance_summary.txt', 'w') as f:
            f.write(summary_text)
        
        print("✓ Results saved:")
        print("  - M4_experimental_results.json")
        print("  - M4_performance_summary.txt")

def main():
    """Example usage"""
    print("M4 Transfer Learning Model for Mortality Classification")
    print("Usage: Import and use with preprocessed raw spectral data")
    
    print("\nTransfer Learning Configuration:")
    print("- Feature extraction with PCA")
    print("- Multiple transfer learning models")
    print("- Domain adaptation techniques")
    print("- SMOTE-ENN class balancing")

if __name__ == "__main__":
    main() 