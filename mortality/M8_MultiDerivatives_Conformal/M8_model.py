"""
M8 Model: Multi-Derivatives + Conformal Prediction for Mortality Classification
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

class ConformalPredictor:
    """Conformal prediction wrapper for uncertainty quantification"""
    
    def __init__(self, base_estimator, alpha=0.1):
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile = None
        self.is_fitted = False
        
    def fit(self, X_train, y_train, X_cal, y_cal):
        """Fit the base estimator and calibrate conformal scores"""
        print(f"Training conformal predictor with alpha={self.alpha}")
        
        # Train base estimator
        self.base_estimator.fit(X_train, y_train)
        
        # Get calibration scores
        if hasattr(self.base_estimator, 'predict_proba'):
            cal_probs = self.base_estimator.predict_proba(X_cal)
            # For binary classification, use 1 - probability of true class
            self.calibration_scores = []
            for i, true_label in enumerate(y_cal):
                if cal_probs.shape[1] > true_label:
                    score = 1 - cal_probs[i, true_label]
                else:
                    score = 1 - cal_probs[i, -1]  # Fallback
                self.calibration_scores.append(score)
        else:
            # For classifiers without predict_proba, use binary score
            cal_preds = self.base_estimator.predict(X_cal)
            self.calibration_scores = [0 if pred == true else 1 for pred, true in zip(cal_preds, y_cal)]
        
        self.calibration_scores = np.array(self.calibration_scores)
        
        # Calculate quantile for conformal prediction
        n_cal = len(self.calibration_scores)
        self.quantile = np.quantile(self.calibration_scores, 
                                   (n_cal + 1) * (1 - self.alpha) / n_cal)
        
        print(f"✓ Conformal predictor calibrated with quantile: {self.quantile:.4f}")
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict with base estimator"""
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted")
        return self.base_estimator.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities with base estimator"""
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted")
        if hasattr(self.base_estimator, 'predict_proba'):
            return self.base_estimator.predict_proba(X)
        else:
            # Convert hard predictions to probabilities
            pred = self.predict(X)
            proba = np.zeros((len(pred), 2))
            proba[pred == 0, 0] = 1.0
            proba[pred == 1, 1] = 1.0
            return proba
    
    def predict_with_confidence(self, X):
        """Predict with conformal confidence sets"""
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted")
        
        if hasattr(self.base_estimator, 'predict_proba'):
            probs = self.base_estimator.predict_proba(X)
            predictions = []
            confidences = []
            
            for prob_vector in probs:
                # For each class, check if 1 - prob <= quantile
                confident_classes = []
                class_confidences = []
                
                for class_idx, prob in enumerate(prob_vector):
                    nonconformity_score = 1 - prob
                    if nonconformity_score <= self.quantile:
                        confident_classes.append(class_idx)
                        class_confidences.append(prob)
                
                if len(confident_classes) == 0:
                    # If no class is confident, predict the most likely
                    confident_classes = [np.argmax(prob_vector)]
                    class_confidences = [np.max(prob_vector)]
                
                predictions.append(confident_classes)
                confidences.append(class_confidences)
            
            return predictions, confidences
        else:
            # For non-probabilistic classifiers
            preds = self.base_estimator.predict(X)
            predictions = [[pred] for pred in preds]
            confidences = [[1.0] for _ in preds]
            return predictions, confidences

class M8MortalityClassifier:
    """M8 Complete classifier with multi-derivatives and conformal prediction"""
    
    def __init__(self, 
                 conformal_alpha=0.1,
                 use_ensemble=True,
                 random_state=42):
        
        self.conformal_alpha = conformal_alpha
        self.use_ensemble = use_ensemble
        self.random_state = random_state
        
        # Create base models for ensemble
        self.base_models = self._create_base_models()
        
        # Conformal predictors
        self.conformal_models = {}
        
        # Ensemble model
        self.ensemble_model = None
        
        self.is_fitted = False
        self.training_stats = {}
        
    def _create_base_models(self):
        """Create diverse base models for ensemble"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            )
        }
        
        return models
    
    def fit(self, X_train, X_cal, y_train, y_cal):
        """Train all models with conformal calibration"""
        print("\n" + "="*60)
        print("M8 MODEL TRAINING: MULTI-DERIVATIVES + CONFORMAL")
        print("="*60)
        
        print(f"Training data: {X_train.shape}")
        print(f"Calibration data: {X_cal.shape}")
        print(f"Target distribution: {np.bincount(y_train.astype(int))}")
        
        # Train individual conformal models
        print("\n1. Training individual conformal models...")
        model_scores = {}
        
        for model_name, model in self.base_models.items():
            try:
                print(f"Training {model_name}...")
                
                # Create conformal predictor
                conformal_model = ConformalPredictor(
                    base_estimator=model,
                    alpha=self.conformal_alpha
                )
                
                # Fit with conformal calibration
                conformal_model.fit(X_train, y_train, X_cal, y_cal)
                
                # Evaluate on calibration set
                y_pred_cal = conformal_model.predict(X_cal)
                score = accuracy_score(y_cal, y_pred_cal)
                model_scores[model_name] = score
                
                # Store conformal model
                self.conformal_models[model_name] = conformal_model
                
                print(f"✓ {model_name} trained (accuracy: {score:.4f})")
                
            except Exception as e:
                print(f"⚠ Warning: {model_name} training failed: {str(e)}")
                continue
        
        # Train ensemble model
        if self.use_ensemble and len(self.conformal_models) > 1:
            print("\n2. Training ensemble model...")
            self.ensemble_model = VotingConformalClassifier(
                list(self.conformal_models.values()),
                alpha=self.conformal_alpha
            )
            self.ensemble_model.fit(X_train, y_train, X_cal, y_cal)
            print("✓ Ensemble model trained")
        
        self.is_fitted = True
        
        # Store training stats
        self.training_stats = {
            'n_train_samples': X_train.shape[0],
            'n_cal_samples': X_cal.shape[0],
            'n_features': X_train.shape[1],
            'n_models': len(self.conformal_models),
            'model_names': list(self.conformal_models.keys()),
            'model_scores': model_scores,
            'conformal_alpha': self.conformal_alpha,
            'ensemble_used': self.use_ensemble and len(self.conformal_models) > 1,
            'train_class_distribution': dict(zip(*np.unique(y_train.astype(int), return_counts=True)))
        }
        
        print(f"\n✓ M8 model training completed")
        print(f"  - Trained models: {len(self.conformal_models)}")
        print(f"  - Ensemble: {'Yes' if self.training_stats['ensemble_used'] else 'No'}")
        
        return self
    
    def predict(self, X, use_ensemble=True):
        """Predict using best performing model or ensemble"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if use_ensemble and self.ensemble_model is not None:
            return self.ensemble_model.predict(X)
        else:
            # Use best individual model
            best_model_name = max(self.training_stats['model_scores'].keys(),
                                key=lambda x: self.training_stats['model_scores'][x])
            return self.conformal_models[best_model_name].predict(X)
    
    def predict_proba(self, X, use_ensemble=True):
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if use_ensemble and self.ensemble_model is not None:
            return self.ensemble_model.predict_proba(X)
        else:
            # Use best individual model
            best_model_name = max(self.training_stats['model_scores'].keys(),
                                key=lambda x: self.training_stats['model_scores'][x])
            return self.conformal_models[best_model_name].predict_proba(X)
    
    def predict_with_uncertainty(self, X, use_ensemble=True):
        """Predict with conformal uncertainty quantification"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        if use_ensemble and self.ensemble_model is not None:
            return self.ensemble_model.predict_with_confidence(X)
        else:
            # Use best individual model
            best_model_name = max(self.training_stats['model_scores'].keys(),
                                key=lambda x: self.training_stats['model_scores'][x])
            return self.conformal_models[best_model_name].predict_with_confidence(X)
    
    def get_uncertainty_metrics(self, X, y_true=None):
        """Calculate uncertainty quantification metrics"""
        predictions, confidences = self.predict_with_uncertainty(X)
        
        # Calculate metrics
        single_predictions = []
        uncertainty_scores = []
        set_sizes = []
        
        for pred_set, conf_set in zip(predictions, confidences):
            if len(pred_set) == 1:
                single_predictions.append(pred_set[0])
                uncertainty_scores.append(1 - max(conf_set))
                set_sizes.append(1)
            else:
                # For multiple predictions, use the most confident one
                best_idx = np.argmax(conf_set)
                single_predictions.append(pred_set[best_idx])
                uncertainty_scores.append(1 - conf_set[best_idx])
                set_sizes.append(len(pred_set))
        
        metrics = {
            'predictions': single_predictions,
            'uncertainty_scores': uncertainty_scores,
            'prediction_set_sizes': set_sizes,
            'average_set_size': np.mean(set_sizes),
            'coverage_rate': np.mean([1 for size in set_sizes if size > 0]),
            'uncertain_predictions': np.sum([1 for size in set_sizes if size > 1])
        }
        
        if y_true is not None:
            metrics['accuracy'] = accuracy_score(y_true, single_predictions)
            # Check if true labels are in prediction sets
            coverage_correct = 0
            for i, true_label in enumerate(y_true):
                if true_label in predictions[i]:
                    coverage_correct += 1
            metrics['empirical_coverage'] = coverage_correct / len(y_true)
        
        return metrics
    
    def evaluate_model(self, X, y, use_ensemble=True):
        """Comprehensive model evaluation with uncertainty quantification"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Standard predictions
        y_pred = self.predict(X, use_ensemble=use_ensemble)
        y_proba = self.predict_proba(X, use_ensemble=use_ensemble)[:, 1]
        
        # Uncertainty metrics
        uncertainty_metrics = self.get_uncertainty_metrics(X, y.astype(int))
        
        # Standard metrics
        metrics = {
            'model_name': 'Ensemble' if use_ensemble and self.ensemble_model is not None else 'Best_Individual',
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=1),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=1),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=1),
            'auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        # Add uncertainty metrics
        metrics.update({
            'uncertainty_' + k: v for k, v in uncertainty_metrics.items() 
            if k not in ['predictions']
        })
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        print(f"Performing {cv}-fold cross-validation...")
        
        # Use a single model for CV (faster)
        cv_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
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

class VotingConformalClassifier:
    """Ensemble voting classifier with conformal prediction"""
    
    def __init__(self, conformal_models, alpha=0.1):
        self.conformal_models = conformal_models
        self.alpha = alpha
        self.is_fitted = False
        
    def fit(self, X_train, y_train, X_cal, y_cal):
        """Fit is already done for individual models"""
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict using majority voting"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        predictions = []
        for model in self.conformal_models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Majority voting
        ensemble_pred = []
        for i in range(predictions.shape[1]):
            votes = np.bincount(predictions[:, i].astype(int))
            ensemble_pred.append(np.argmax(votes))
        
        return np.array(ensemble_pred)
    
    def predict_proba(self, X):
        """Predict probabilities using averaging"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        probabilities = []
        for model in self.conformal_models:
            proba = model.predict_proba(X)
            probabilities.append(proba)
        
        # Average probabilities
        ensemble_proba = np.mean(probabilities, axis=0)
        return ensemble_proba
    
    def predict_with_confidence(self, X):
        """Predict with ensemble conformal confidence"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        all_predictions = []
        all_confidences = []
        
        # Get predictions from all models
        for model in self.conformal_models:
            preds, confs = model.predict_with_confidence(X)
            all_predictions.append(preds)
            all_confidences.append(confs)
        
        # Combine predictions using union of prediction sets
        ensemble_predictions = []
        ensemble_confidences = []
        
        for i in range(len(X)):
            combined_set = set()
            combined_conf = {}
            
            for j, (pred_list, conf_list) in enumerate(zip(all_predictions, all_confidences)):
                for pred, conf in zip(pred_list[i], conf_list[i]):
                    combined_set.add(pred)
                    if pred not in combined_conf:
                        combined_conf[pred] = []
                    combined_conf[pred].append(conf)
            
            # Average confidences for each class
            final_preds = list(combined_set)
            final_confs = [np.mean(combined_conf[pred]) for pred in final_preds]
            
            ensemble_predictions.append(final_preds)
            ensemble_confidences.append(final_confs)
        
        return ensemble_predictions, ensemble_confidences

def main():
    """Example usage"""
    print("M8 Multi-Derivatives Conformal Mortality Classifier")
    print("Features:")
    print("- Multiple derivative transformations")
    print("- Conformal prediction for uncertainty quantification")
    print("- Ensemble of diverse base models")

if __name__ == "__main__":
    main() 