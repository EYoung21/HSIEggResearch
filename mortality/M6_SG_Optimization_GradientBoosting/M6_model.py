"""
M6 Model: SG + Optimization + Gradient Boosting for Mortality Classification
Advanced gradient boosting with extensive hyperparameter optimization
"""

import numpy as np
import pandas as pd
import time
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

class AdvancedGradientBoostingOptimizer:
    """M6 Model: Advanced Gradient Boosting with optimization for mortality classification"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Gradient boosting models
        self.sklearn_gb = None
        self.xgb_model = None
        self.lgb_model = None
        self.best_model = None
        self.best_model_name = None
        
        # Training metadata
        self.training_time = 0
        self.optimization_time = 0
        self.cv_scores = {}
        self.best_params = {}
        
        # Class balancing
        self.smote_enn = SMOTEENN(
            smote=SMOTE(random_state=random_state, k_neighbors=3),
            enn=EditedNearestNeighbours(n_neighbors=3),
            random_state=random_state
        )
    
    def optimize_sklearn_gradient_boosting(self, X_resampled, y_resampled):
        """Optimize Scikit-learn Gradient Boosting with extensive search"""
        print("Optimizing Scikit-learn Gradient Boosting...")
        
        # First: Coarse grid search
        print("  Step 1: Setting up coarse parameter grid...")
        coarse_param_distributions = {
            'n_estimators': [50, 100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
        print(f"  Parameter combinations to test: {len(coarse_param_distributions['n_estimators']) * len(coarse_param_distributions['learning_rate']) * len(coarse_param_distributions['max_depth'])}")
        
        gb = GradientBoostingClassifier(random_state=self.random_state)
        
        # Coarse search
        print("  Step 2: Starting coarse RandomizedSearchCV (50 iterations, 3-fold CV)...")
        import time
        coarse_start = time.time()
        
        coarse_search = RandomizedSearchCV(
            gb, coarse_param_distributions, n_iter=50, cv=3,
            scoring='roc_auc', random_state=self.random_state, n_jobs=-1, verbose=1
        )
        
        print("  Running coarse search - this may take several minutes...")
        coarse_search.fit(X_resampled, y_resampled)
        coarse_time = time.time() - coarse_start
        print(f"  Coarse search completed in {coarse_time:.1f} seconds")
        
        # Fine-tune around best parameters
        print(f"  Step 3: Fine-tuning around best parameters...")
        best_params = coarse_search.best_params_
        print(f"  Best coarse params: {best_params}")
        print(f"  Coarse best score: {coarse_search.best_score_:.4f}")
        
        fine_param_grid = {
            'n_estimators': [max(50, best_params['n_estimators'] - 50),
                           best_params['n_estimators'],
                           best_params['n_estimators'] + 50],
            'learning_rate': [max(0.01, best_params['learning_rate'] - 0.02),
                            best_params['learning_rate'],
                            min(0.3, best_params['learning_rate'] + 0.02)],
            'max_depth': [max(3, best_params['max_depth'] - 1),
                         best_params['max_depth'],
                         min(15, best_params['max_depth'] + 1)],
            'min_samples_split': [best_params['min_samples_split']],
            'min_samples_leaf': [best_params['min_samples_leaf']],
            'subsample': [best_params['subsample']],
            'max_features': [best_params['max_features']]
        }
        
        total_combinations = (len(fine_param_grid['n_estimators']) * 
                            len(fine_param_grid['learning_rate']) * 
                            len(fine_param_grid['max_depth']))
        print(f"  Fine grid combinations: {total_combinations}")
        
        # Fine search
        print("  Step 4: Starting fine GridSearchCV (5-fold CV)...")
        fine_start = time.time()
        
        fine_search = GridSearchCV(
            gb, fine_param_grid, cv=5,
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        print("  Running fine search...")
        fine_search.fit(X_resampled, y_resampled)
        fine_time = time.time() - fine_start
        print(f"  Fine search completed in {fine_time:.1f} seconds")
        
        self.best_params['sklearn_gb'] = fine_search.best_params_
        
        print(f"✓ Sklearn GB Best Score: {fine_search.best_score_:.4f}")
        print(f"✓ Total optimization time: {(coarse_time + fine_time):.1f} seconds")
        return fine_search.best_estimator_
    
    def optimize_xgboost(self, X_resampled, y_resampled):
        """Optimize XGBoost with advanced parameter search"""
        print("Optimizing XGBoost...")
        print("  Setting up XGBoost parameter distributions...")
        
        # XGBoost parameter space
        param_distributions = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 12),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 2),
            'reg_lambda': uniform(1, 4),
            'min_child_weight': randint(1, 10),
            'gamma': uniform(0, 0.5)
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            tree_method='hist'  # Faster training
        )
        
        print("  Starting XGBoost RandomizedSearchCV (75 iterations, 3-fold CV)...")
        xgb_start = time.time()
        
        random_search = RandomizedSearchCV(
            xgb_model, param_distributions, n_iter=75, cv=3,
            scoring='roc_auc', random_state=self.random_state, n_jobs=-1, verbose=1
        )
        
        print("  Running XGBoost optimization...")
        random_search.fit(X_resampled, y_resampled)
        xgb_time = time.time() - xgb_start
        print(f"  XGBoost optimization completed in {xgb_time:.1f} seconds")
        
        self.best_params['xgboost'] = random_search.best_params_
        
        print(f"✓ XGBoost Best Score: {random_search.best_score_:.4f}")
        return random_search.best_estimator_
    
    def optimize_lightgbm(self, X_resampled, y_resampled):
        """Optimize LightGBM with parameter optimization"""
        print("Optimizing LightGBM...")
        print("  Setting up LightGBM parameter distributions...")
        
        param_distributions = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 15),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 2),
            'reg_lambda': uniform(1, 4),
            'num_leaves': randint(20, 150),
            'min_child_samples': randint(10, 50),
            'min_split_gain': uniform(0, 0.1)
        }
        
        lgb_model = lgb.LGBMClassifier(
            random_state=self.random_state,
            objective='binary',
            verbosity=-1
        )
        
        print("  Starting LightGBM RandomizedSearchCV (75 iterations, 3-fold CV)...")
        lgb_start = time.time()
        
        random_search = RandomizedSearchCV(
            lgb_model, param_distributions, n_iter=75, cv=3,
            scoring='roc_auc', random_state=self.random_state, n_jobs=-1, verbose=1
        )
        
        print("  Running LightGBM optimization...")
        random_search.fit(X_resampled, y_resampled)
        lgb_time = time.time() - lgb_start
        print(f"  LightGBM optimization completed in {lgb_time:.1f} seconds")
        
        self.best_params['lightgbm'] = random_search.best_params_
        
        print(f"✓ LightGBM Best Score: {random_search.best_score_:.4f}")
        return random_search.best_estimator_
    
    def select_best_model(self, X_resampled, y_resampled):
        """Select the best gradient boosting model through cross-validation"""
        print("\nSelecting best gradient boosting model...")
        
        models = {
            'sklearn_gb': self.sklearn_gb,
            'xgboost': self.xgb_model,
            'lightgbm': self.lgb_model
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        best_score = 0
        best_model = None
        best_name = None
        
        for name, model in models.items():
            if model is not None:
                cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='accuracy')
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                self.cv_scores[name] = {
                    'mean': mean_score,
                    'std': std_score,
                    'scores': cv_scores.tolist()
                }
                
                print(f"{name}: {mean_score:.4f} ± {std_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_name = name
        
        self.best_model = best_model
        self.best_model_name = best_name
        
        print(f"\nBest model: {best_name} (CV: {best_score:.4f})")
        return best_model, best_name
    
    def train_mortality_gradient_boosting(self, X_train, y_train):
        """Train M6: Advanced Gradient Boosting for mortality classification"""
        print("\n" + "="*60)
        print("TRAINING M6: SG + OPTIMIZATION + GRADIENT BOOSTING")
        print("="*60)
        
        start_time = time.time()
        
        # Apply SMOTE-ENN for class balancing
        print("Applying SMOTE-ENN for class balancing...")
        print(f"Original distribution: {np.bincount(y_train)}")
        
        try:
            X_resampled, y_resampled = self.smote_enn.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"Warning: SMOTE-ENN failed, using original data: {e}")
            X_resampled, y_resampled = X_train, y_train
        print(f"Resampled distribution: {np.bincount(y_resampled)}")
        print(f"Dataset resampled: {X_resampled.shape[0]} samples")
        
        # Optimize individual gradient boosting models
        optimization_start = time.time()
        
        print("\nOptimizing gradient boosting models...")
        self.sklearn_gb = self.optimize_sklearn_gradient_boosting(X_resampled, y_resampled)
        self.xgb_model = self.optimize_xgboost(X_resampled, y_resampled)
        self.lgb_model = self.optimize_lightgbm(X_resampled, y_resampled)
        
        self.optimization_time = time.time() - optimization_start
        
        # Select best model
        self.best_model, self.best_model_name = self.select_best_model(X_resampled, y_resampled)
        
        # Final training on best model
        print(f"\nFinal training with {self.best_model_name}...")
        self.best_model.fit(X_resampled, y_resampled)
        
        self.training_time = time.time() - start_time
        
        print(f"\nTraining completed!")
        print(f"Total training time: {self.training_time:.2f} seconds")
        print(f"Optimization time: {self.optimization_time:.2f} seconds")
        
        return self.best_model
    
    def evaluate_mortality_prediction(self, X_test, y_test, label_encoder):
        """Evaluate the best gradient boosting model on test set"""
        print("\n" + "="*60)
        print("M6 EVALUATION: GRADIENT BOOSTING PERFORMANCE")
        print("="*60)
        
        if self.best_model is None:
            print("Error: No model trained yet!")
            return None
        
        # Test set predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        # Confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, 
                                     target_names=label_encoder.classes_,
                                     output_dict=True)
        
        print(f"Best Model: {self.best_model_name}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
        # Feature importance analysis
        feature_importance = None
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = self.best_model.feature_importances_
            
            # Top 10 most important features
            top_indices = np.argsort(feature_importance)[-10:][::-1]
            print(f"\nTop 10 Most Important Features:")
            for i, idx in enumerate(top_indices):
                print(f"{i+1:2d}. Feature {idx:3d}: {feature_importance[idx]:.6f}")
        
        results = {
            'best_model': self.best_model_name,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'cv_scores': self.cv_scores,
            'best_params': self.best_params,
            'training_time': self.training_time,
            'optimization_time': self.optimization_time,
            'feature_importance': feature_importance.tolist() if feature_importance is not None else None
        }
        
        return results
    
    def save_model_and_results(self, results, preprocessing_info):
        """Save M6 model and results"""
        print("\nSaving M6 model and results...")
        
        # Save the best model
        import joblib
        joblib.dump(self.best_model, f'M6_{self.best_model_name}_model.pkl')
        
        # Save all optimized models
        models_dict = {
            'sklearn_gb': self.sklearn_gb,
            'xgboost': self.xgb_model,
            'lightgbm': self.lgb_model,
            'best_model_name': self.best_model_name
        }
        joblib.dump(models_dict, 'M6_all_gradient_boosting_models.pkl')
        
        # Save SMOTE-ENN
        joblib.dump(self.smote_enn, 'M6_smote_enn.pkl')
        
        # Combine results with preprocessing info
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        complete_results = {
            'experiment': 'M6_SG_Optimization_GradientBoosting',
            'preprocessing': preprocessing_info,
            'model_results': convert_numpy(results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save experimental results
        with open('M6_experimental_results.json', 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        # Create human-readable summary
        summary = f"""
M6 EXPERIMENT SUMMARY: SG + Optimization + Gradient Boosting
==========================================================

METHODOLOGY:
- Preprocessing: Optimized Savitzky-Golay + Morphological features
- Algorithm: Advanced Gradient Boosting (Sklearn/XGBoost/LightGBM)
- Optimization: Extensive hyperparameter search
- Features: {preprocessing_info['n_final_features']} optimized features

BEST MODEL: {self.best_model_name}

PERFORMANCE:
- Test Accuracy: {results['test_accuracy']:.4f}
- Test AUC: {results['test_auc']:.4f}
- Training Time: {results['training_time']:.2f} seconds
- Optimization Time: {results['optimization_time']:.2f} seconds

CROSS-VALIDATION SCORES:
"""
        for model_name, scores in results['cv_scores'].items():
            summary += f"- {model_name}: {scores['mean']:.4f} ± {scores['std']:.4f}\n"
        
        summary += f"""
OPTIMIZATION DETAILS:
- Sklearn GB params: {results['best_params'].get('sklearn_gb', 'N/A')}
- XGBoost params: {results['best_params'].get('xgboost', 'N/A')}
- LightGBM params: {results['best_params'].get('lightgbm', 'N/A')}

FILES GENERATED:
- M6_{self.best_model_name}_model.pkl (best model)
- M6_all_gradient_boosting_models.pkl (all models)
- M6_experimental_results.json (complete results)
- M6_performance_summary.txt (this summary)

NOTES:
- Extensive optimization across 3 gradient boosting algorithms
- Morphological features enhance spectral information
- {self.best_model_name} selected as best performing model
- Class balancing with SMOTE-ENN addresses mortality imbalance
"""
        
        with open('M6_performance_summary.txt', 'w') as f:
            f.write(summary)
        
        print("M6 model and results saved!")
        print(f"- Best model: M6_{self.best_model_name}_model.pkl")
        print(f"- Results: M6_experimental_results.json")
        print(f"- Summary: M6_performance_summary.txt")
        
        return complete_results

def main():
    """Main function for M6 gradient boosting model"""
    print("Starting M6 Model: SG + Optimization + Gradient Boosting")
    
    # Load preprocessed data
    try:
        X_train = np.load('X_train_processed.npy')
        X_test = np.load('X_test_processed.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        
        import joblib
        label_encoder = joblib.load('label_encoder.pkl')
        
        import json
        with open('preprocessing_info.json', 'r') as f:
            preprocessing_info = json.load(f)
        
        print(f"Loaded preprocessed data:")
        print(f"  Training: {X_train.shape}")
        print(f"  Test: {X_test.shape}")
        
    except FileNotFoundError as e:
        print(f"Error: Preprocessed data not found: {e}")
        print("Please run M6_preprocessing.py first!")
        return
    
    # Initialize and train model
    optimizer = AdvancedGradientBoostingOptimizer(random_state=42)
    
    # Train model
    best_model = optimizer.train_mortality_gradient_boosting(X_train, y_train)
    
    # Evaluate model
    results = optimizer.evaluate_mortality_prediction(X_test, y_test, label_encoder)
    
    # Save model and results
    complete_results = optimizer.save_model_and_results(results, preprocessing_info)
    
    print(f"\nM6 Gradient Boosting experiment completed!")
    print(f"Best model: {results['best_model']}")
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    main() 