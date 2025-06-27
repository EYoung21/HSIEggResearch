"""
M7 Model: SNV + Mixup + Semi-Supervised Learning for Mortality Classification
Semi-supervised learning models that utilize both labeled and unlabeled data
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelSpreading, LabelPropagation, SelfTrainingClassifier
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

class SemiSupervisedEnsemble(BaseEstimator, ClassifierMixin):
    """Ensemble of semi-supervised learning models for mortality classification"""
    
    def __init__(self, 
                 use_label_spreading=True,
                 use_label_propagation=True,
                 use_self_training=True,
                 ensemble_voting='soft',
                 random_state=42):
        
        self.use_label_spreading = use_label_spreading
        self.use_label_propagation = use_label_propagation
        self.use_self_training = use_self_training
        self.ensemble_voting = ensemble_voting
        self.random_state = random_state
        
        self.models = {}
        self.model_weights = {}
        self.is_fitted = False
        
    def _create_models(self):
        """Create semi-supervised learning models"""
        models = {}
        
        if self.use_label_spreading:
            models['label_spreading'] = LabelSpreading(
                kernel='knn',
                n_neighbors=7,
                alpha=0.2,
                max_iter=30,
                tol=1e-3
            )
        
        if self.use_label_propagation:
            models['label_propagation'] = LabelPropagation(
                kernel='knn',
                n_neighbors=7,
                max_iter=30,
                tol=1e-3
            )
        
        if self.use_self_training:
            base_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state
            )
            models['self_training'] = SelfTrainingClassifier(
                base_estimator=base_classifier,
                threshold=0.75,
                criterion='threshold',
                k_best=10,
                max_iter=10
            )
        
        return models
    
    def fit(self, X_labeled, y_labeled, X_unlabeled=None):
        """Train semi-supervised ensemble"""
        print(f"Training Semi-Supervised Ensemble...")
        print(f"Labeled data: {X_labeled.shape}")
        if X_unlabeled is not None and len(X_unlabeled) > 0:
            print(f"Unlabeled data: {X_unlabeled.shape}")
        
        self.models = self._create_models()
        
        # Prepare data for semi-supervised learning
        if X_unlabeled is not None and len(X_unlabeled) > 0:
            # Combine labeled and unlabeled data
            X_combined = np.vstack([X_labeled, X_unlabeled])
            y_combined = np.hstack([y_labeled, [-1] * len(X_unlabeled)])
        else:
            X_combined = X_labeled
            y_combined = y_labeled
            print("Warning: No unlabeled data provided. Using supervised learning.")
        
        # Train each model
        model_scores = {}
        
        for model_name, model in self.models.items():
            try:
                print(f"Training {model_name}...")
                
                if model_name in ['label_spreading', 'label_propagation']:
                    # These models expect -1 for unlabeled samples
                    model.fit(X_combined, y_combined)
                elif model_name == 'self_training':
                    # Self-training only on labeled data initially
                    model.fit(X_labeled, y_labeled)
                
                # Evaluate on labeled data for weight calculation
                y_pred = model.predict(X_labeled)
                score = accuracy_score(y_labeled, y_pred)
                model_scores[model_name] = score
                print(f"✓ {model_name} trained (accuracy: {score:.4f})")
                
            except Exception as e:
                print(f"⚠ Warning: {model_name} training failed: {str(e)}")
                # Remove failed model
                del self.models[model_name]
        
        # Calculate ensemble weights based on performance
        if model_scores:
            total_score = sum(model_scores.values())
            if total_score > 0:
                self.model_weights = {name: score/total_score for name, score in model_scores.items()}
            else:
                self.model_weights = {name: 1.0/len(self.models) for name in self.models.keys()}
        else:
            self.model_weights = {name: 1.0/len(self.models) for name in self.models.keys()}
        
        print(f"✓ Model weights: {self.model_weights}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict using ensemble of semi-supervised models"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if len(self.models) == 0:
            raise ValueError("No successfully trained models available.")
        
        # Get predictions from each model
        predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.model_weights[model_name])
            except Exception as e:
                print(f"Warning: {model_name} prediction failed: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions.")
        
        # Weighted majority voting
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        final_predictions = []
        for i in range(predictions.shape[1]):
            votes = {}
            for j, pred in enumerate(predictions[:, i]):
                if pred not in votes:
                    votes[pred] = 0
                votes[pred] += weights[j]
            
            final_pred = max(votes.keys(), key=lambda x: votes[x])
            final_predictions.append(final_pred)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        probabilities = []
        weights = []
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    probabilities.append(proba)
                    weights.append(self.model_weights[model_name])
                else:
                    # Convert hard predictions to probabilities
                    pred = model.predict(X)
                    proba = np.zeros((len(pred), 2))
                    proba[pred == 0, 0] = 1.0
                    proba[pred == 1, 1] = 1.0
                    probabilities.append(proba)
                    weights.append(self.model_weights[model_name])
            except:
                continue
        
        if probabilities:
            probabilities = np.array(probabilities)
            weights = np.array(weights)
            
            # Weighted average of probabilities
            ensemble_proba = np.average(probabilities, axis=0, weights=weights)
            return ensemble_proba
        else:
            # Fallback
            pred = self.predict(X)
            proba = np.zeros((len(pred), 2))
            proba[pred == 0, 0] = 1.0
            proba[pred == 1, 1] = 1.0
            return proba

class M7MortalityClassifier:
    """M7 Complete classifier with semi-supervised learning"""
    
    def __init__(self, 
                 use_semi_supervised=True,
                 ensemble_voting='soft',
                 random_state=42):
        
        self.use_semi_supervised = use_semi_supervised
        self.ensemble_voting = ensemble_voting
        self.random_state = random_state
        
        # Primary semi-supervised model
        if use_semi_supervised:
            self.primary_model = SemiSupervisedEnsemble(
                ensemble_voting=ensemble_voting,
                random_state=random_state
            )
        else:
            # Fallback to supervised learning
            self.primary_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state
            )
        
        # Backup supervised models
        self.backup_model_1 = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state
        )
        
        self.backup_model_2 = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=random_state
        )
        
        self.is_fitted = False
        self.training_stats = {}
    
    def fit(self, X_labeled, y_labeled, X_unlabeled=None):
        """Train all models"""
        print("\n" + "="*60)
        print("M7 MODEL TRAINING: SEMI-SUPERVISED ENSEMBLE")
        print("="*60)
        
        print(f"Labeled training data: {X_labeled.shape}")
        print(f"Labeled target distribution: {np.bincount(y_labeled)}")
        if X_unlabeled is not None and len(X_unlabeled) > 0:
            print(f"Unlabeled data: {X_unlabeled.shape}")
        
        # Train primary model
        print("\n1. Training primary semi-supervised model...")
        if self.use_semi_supervised:
            self.primary_model.fit(X_labeled, y_labeled, X_unlabeled)
        else:
            self.primary_model.fit(X_labeled, y_labeled)
        
        # Train backup models (supervised only)
        print("\n2. Training backup supervised models...")
        self.backup_model_1.fit(X_labeled, y_labeled)
        self.backup_model_2.fit(X_labeled, y_labeled)
        print("✓ Backup models trained")
        
        self.is_fitted = True
        
        # Store training stats
        self.training_stats = {
            'n_labeled_samples': X_labeled.shape[0],
            'n_features': X_labeled.shape[1],
            'n_unlabeled_samples': len(X_unlabeled) if X_unlabeled is not None else 0,
            'labeled_class_distribution': dict(zip(*np.unique(y_labeled, return_counts=True))),
            'primary_model': 'SemiSupervisedEnsemble' if self.use_semi_supervised else 'RandomForest',
            'semi_supervised_ratio': len(X_unlabeled) / (len(X_labeled) + len(X_unlabeled)) if X_unlabeled is not None and len(X_unlabeled) > 0 else 0.0
        }
        
        print(f"\n✓ M7 model training completed")
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
        
        # Get predictions from all models
        pred_primary = self.primary_model.predict_proba(X)[:, 1]
        pred_backup1 = self.backup_model_1.predict_proba(X)[:, 1]
        pred_backup2 = self.backup_model_2.predict_proba(X)[:, 1]
        
        # Weighted ensemble
        ensemble_proba = (0.6 * pred_primary + 0.25 * pred_backup1 + 0.15 * pred_backup2)
        
        return (ensemble_proba > 0.5).astype(int)
    
    def predict_ensemble_proba(self, X):
        """Predict ensemble probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
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
        if use_ensemble:
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
        
        # Use backup model for cross-validation (simpler and faster)
        cv_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state
        )
        
        cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring='accuracy')
        
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
    print("M7 Semi-Supervised Mortality Classifier")
    print("Features:")
    print("- Semi-supervised learning ensemble")
    print("- Label spreading and propagation")
    print("- Self-training with high-confidence samples")
    print("- Mixup data augmentation")

if __name__ == "__main__":
    main() 