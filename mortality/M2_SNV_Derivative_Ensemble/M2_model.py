"""
M2 Model: SNV + Derivatives + Ensemble for Mortality Classification
Advanced ensemble learning with multiple algorithms for mortality prediction
"""

import numpy as np
import pandas as pd
import time
import json
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class MortalityEnsemble:
    """M2 Model: SNV + Derivatives + Ensemble for mortality classification"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Individual models
        self.rf_model = None
        self.svm_model = None
        self.xgb_model = None
        self.ensemble_model = None
        
        # Training metadata
        self.training_time = 0
        self.feature_importance = {}
        self.cv_scores = {}
        self.best_params = {}
        
        # Class balancing
        self.smote_enn = SMOTEENN(
            smote=SMOTE(random_state=random_state, k_neighbors=3),
            enn=EditedNearestNeighbours(n_neighbors=3),
            random_state=random_state
        )
    
    def optimize_random_forest(self, X_resampled, y_resampled):
        """Optimize Random Forest hyperparameters"""
        print("Optimizing Random Forest...")
        
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            rf, param_distributions, n_iter=20, cv=3,
            scoring='roc_auc', random_state=self.random_state, n_jobs=-1
        )
        
        random_search.fit(X_resampled, y_resampled)
        self.best_params['random_forest'] = random_search.best_params_
        
        print(f"✓ RF Best Score: {random_search.best_score_:.4f}")
        return random_search.best_estimator_
    
    def optimize_svm(self, X_resampled, y_resampled):
        """Optimize SVM hyperparameters"""
        print("Optimizing SVM...")
        
        param_distributions = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        svm = SVC(random_state=self.random_state, probability=True)
        
        random_search = RandomizedSearchCV(
            svm, param_distributions, n_iter=15, cv=3,
            scoring='roc_auc', random_state=self.random_state, n_jobs=-1
        )
        
        random_search.fit(X_resampled, y_resampled)
        self.best_params['svm'] = random_search.best_params_
        
        print(f"✓ SVM Best Score: {random_search.best_score_:.4f}")
        return random_search.best_estimator_
    
    def optimize_xgboost(self, X_resampled, y_resampled):
        """Optimize XGBoost hyperparameters"""
        print("Optimizing XGBoost...")
        
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        random_search = RandomizedSearchCV(
            xgb_model, param_distributions, n_iter=20, cv=3,
            scoring='roc_auc', random_state=self.random_state, n_jobs=-1
        )
        
        random_search.fit(X_resampled, y_resampled)
        self.best_params['xgboost'] = random_search.best_params_
        
        print(f"✓ XGB Best Score: {random_search.best_score_:.4f}")
        return random_search.best_estimator_
    
    def create_ensemble(self, rf_model, svm_model, xgb_model):
        """Create voting ensemble"""
        print("Creating voting ensemble...")
        
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('svm', svm_model), 
                ('xgb', xgb_model)
            ],
            voting='soft'  # Use predicted probabilities
        )
        
        print("✓ Ensemble created with RF + SVM + XGBoost")
        return ensemble
    
    def train_mortality_ensemble(self, X_train, y_train):
        """Train mortality classification ensemble"""
        print("\n" + "="*60)
        print("TRAINING M2: SNV + DERIVATIVES + ENSEMBLE")
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
        print(f"✓ Dataset resampled: {X_resampled.shape[0]} samples")
        
        # Optimize individual models
        print("\nOptimizing individual models...")
        self.rf_model = self.optimize_random_forest(X_resampled, y_resampled)
        self.svm_model = self.optimize_svm(X_resampled, y_resampled)
        self.xgb_model = self.optimize_xgboost(X_resampled, y_resampled)
        
        # Create and train ensemble
        print("\nTraining ensemble model...")
        self.ensemble_model = self.create_ensemble(self.rf_model, self.svm_model, self.xgb_model)
        self.ensemble_model.fit(X_resampled, y_resampled)
        
        # Cross-validation evaluation
        print("\nPerforming cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Individual model CV
        rf_cv_scores = cross_val_score(self.rf_model, X_resampled, y_resampled, cv=cv, scoring='accuracy')
        svm_cv_scores = cross_val_score(self.svm_model, X_resampled, y_resampled, cv=cv, scoring='accuracy')
        xgb_cv_scores = cross_val_score(self.xgb_model, X_resampled, y_resampled, cv=cv, scoring='accuracy')
        
        # Ensemble CV
        ensemble_cv_scores = cross_val_score(self.ensemble_model, X_resampled, y_resampled, cv=cv, scoring='accuracy')
        
        self.cv_scores = {
            'rf_cv_mean': np.mean(rf_cv_scores),
            'rf_cv_std': np.std(rf_cv_scores),
            'svm_cv_mean': np.mean(svm_cv_scores),
            'svm_cv_std': np.std(svm_cv_scores),
            'xgb_cv_mean': np.mean(xgb_cv_scores),
            'xgb_cv_std': np.std(xgb_cv_scores),
            'ensemble_cv_mean': np.mean(ensemble_cv_scores),
            'ensemble_cv_std': np.std(ensemble_cv_scores)
        }
        
        # Feature importance from ensemble components
        if hasattr(self.rf_model, 'feature_importances_'):
            self.feature_importance['random_forest'] = self.rf_model.feature_importances_
        if hasattr(self.xgb_model, 'feature_importances_'):
            self.feature_importance['xgboost'] = self.xgb_model.feature_importances_
        
        self.training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {self.training_time:.2f} seconds")
        print(f"✓ RF CV: {self.cv_scores['rf_cv_mean']:.4f} ± {self.cv_scores['rf_cv_std']:.4f}")
        print(f"✓ SVM CV: {self.cv_scores['svm_cv_mean']:.4f} ± {self.cv_scores['svm_cv_std']:.4f}")
        print(f"✓ XGB CV: {self.cv_scores['xgb_cv_mean']:.4f} ± {self.cv_scores['xgb_cv_std']:.4f}")
        print(f"✓ Ensemble CV: {self.cv_scores['ensemble_cv_mean']:.4f} ± {self.cv_scores['ensemble_cv_std']:.4f}")
        
        return {
            'training_time': self.training_time,
            'cv_scores': self.cv_scores,
            'best_params': self.best_params,
            'resampled_shape': X_resampled.shape
        }
    
    def evaluate_mortality_prediction(self, X_test, y_test, label_encoder):
        """Evaluate mortality prediction ensemble"""
        print("\n" + "="*60)
        print("EVALUATING MORTALITY ENSEMBLE")
        print("="*60)
        
        results = {}
        
        # Evaluate individual models
        models = {
            'Random Forest': self.rf_model,
            'SVM': self.svm_model,
            'XGBoost': self.xgb_model,
            'Ensemble': self.ensemble_model
        }
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, predictions)
            auc = roc_auc_score(y_test, probabilities)
            
            results[name.lower().replace(' ', '_')] = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'accuracy': accuracy,
                'auc': auc,
                'confusion_matrix': confusion_matrix(y_test, predictions),
                'classification_report': classification_report(
                    y_test, predictions, target_names=label_encoder.classes_, output_dict=True
                )
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  AUC: {auc:.4f}")
        
        # Determine best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        results['best_model'] = {
            'name': best_model_name,
            'accuracy': best_accuracy,
            'auc': results[best_model_name]['auc']
        }
        
        print(f"\nMortality Ensemble Results:")
        print(f"  Best Model: {best_model_name.replace('_', ' ').title()}")
        print(f"  Best Accuracy: {best_accuracy:.4f}")
        print(f"  Best AUC: {results[best_model_name]['auc']:.4f}")
        
        return results
    
    def analyze_feature_importance(self, preprocessing_info, top_k=15):
        """Analyze feature importance from ensemble"""
        print(f"\nAnalyzing top {top_k} features...")
        
        # Combine feature importance from RF and XGB
        rf_importance = self.feature_importance.get('random_forest', np.array([]))
        xgb_importance = self.feature_importance.get('xgboost', np.array([]))
        
        if len(rf_importance) > 0 and len(xgb_importance) > 0:
            # Average importance from both models
            combined_importance = (rf_importance + xgb_importance) / 2
        elif len(rf_importance) > 0:
            combined_importance = rf_importance
        elif len(xgb_importance) > 0:
            combined_importance = xgb_importance
        else:
            print("No feature importance available")
            return {}
        
        feature_names = preprocessing_info.get('feature_names', [f'Feature_{i}' for i in range(len(combined_importance))])
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': combined_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_k} Most Important Features:")
        print("-" * 50)
        for idx, row in importance_df.head(top_k).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        # Analyze by feature type
        feature_types = {
            'SNV': importance_df[importance_df['feature'].str.contains('SNV')]['importance'].sum(),
            '1st_Derivative': importance_df[importance_df['feature'].str.contains('1stDeriv')]['importance'].sum(),
            '2nd_Derivative': importance_df[importance_df['feature'].str.contains('2ndDeriv')]['importance'].sum(),
            'Ratios': importance_df[importance_df['feature'].str.contains('Ratio')]['importance'].sum()
        }
        
        print(f"\nImportance by Feature Type:")
        for feature_type, importance in feature_types.items():
            print(f"  {feature_type}: {importance:.4f}")
        
        return {
            'feature_importance': importance_df.head(top_k).to_dict('records'),
            'feature_type_importance': feature_types,
            'total_features': len(feature_names)
        }
    
    def save_model_and_results(self, results, feature_analysis, preprocessing_info):
        """Save models and results"""
        print("\nSaving M2 mortality models and results...")
        
        experiment_results = {
            'experiment_name': 'M2_SNV_Derivative_Ensemble',
            'methodology': {
                'preprocessing': 'SNV + 1st/2nd Derivatives + Spectral Ratios',
                'models': 'Random Forest + SVM + XGBoost Ensemble',
                'class_balancing': 'SMOTE-ENN',
                'optimization': 'RandomizedSearchCV'
            },
            'data_info': {
                'n_samples': preprocessing_info.get('n_samples', 0),
                'n_features': preprocessing_info.get('n_enhanced_features', 0),
                'wavelength_range': preprocessing_info.get('wavelength_range', 'Unknown')
            },
            'training_results': {
                'training_time': self.training_time,
                'cv_scores': self.cv_scores,
                'best_parameters': self.best_params
            },
            'evaluation_results': results,
            'feature_analysis': feature_analysis,
            'preprocessing_stats': preprocessing_info
        }
        
        with open('M2_experimental_results.json', 'w') as f:
            # Convert numpy arrays and complex objects to JSON-serializable format
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                else:
                    return str(obj)
            
            json.dump(experiment_results, f, indent=2, default=convert_numpy)
        
        # Performance summary
        best_model = results['best_model']['name'].replace('_', ' ').title()
        best_accuracy = results['best_model']['accuracy']
        best_auc = results['best_model']['auc']
        
        summary_text = f"""
M2 MORTALITY CLASSIFICATION - PERFORMANCE SUMMARY
================================================

METHODOLOGY:
- Preprocessing: SNV + 1st/2nd Derivatives + Spectral Ratios
- Models: Random Forest + SVM + XGBoost Ensemble
- Class Balancing: SMOTE-ENN
- Optimization: RandomizedSearchCV with 3-fold CV

DATASET:
- Samples: {preprocessing_info.get('n_samples', 'Unknown')}
- Features: {preprocessing_info.get('n_enhanced_features', 'Unknown')} (enhanced from {preprocessing_info.get('n_wavelengths', 'Unknown')} wavelengths)
- Wavelength Range: {preprocessing_info.get('wavelength_range', 'Unknown')}

TRAINING PERFORMANCE:
- Training Time: {self.training_time:.2f} seconds
- Random Forest CV: {self.cv_scores['rf_cv_mean']:.4f} ± {self.cv_scores['rf_cv_std']:.4f}
- SVM CV: {self.cv_scores['svm_cv_mean']:.4f} ± {self.cv_scores['svm_cv_std']:.4f}
- XGBoost CV: {self.cv_scores['xgb_cv_mean']:.4f} ± {self.cv_scores['xgb_cv_std']:.4f}
- Ensemble CV: {self.cv_scores['ensemble_cv_mean']:.4f} ± {self.cv_scores['ensemble_cv_std']:.4f}

TEST PERFORMANCE:
- Best Model: {best_model}
- Test Accuracy: {best_accuracy:.4f}
- Test AUC: {best_auc:.4f}

INDIVIDUAL MODEL RESULTS:
- Random Forest: {results['random_forest']['accuracy']:.4f} (AUC: {results['random_forest']['auc']:.4f})
- SVM: {results['svm']['accuracy']:.4f} (AUC: {results['svm']['auc']:.4f})
- XGBoost: {results['xgboost']['accuracy']:.4f} (AUC: {results['xgboost']['auc']:.4f})
- Ensemble: {results['ensemble']['accuracy']:.4f} (AUC: {results['ensemble']['auc']:.4f})

FEATURE IMPORTANCE INSIGHTS:
- Most important feature type: {max(feature_analysis['feature_type_importance'], key=feature_analysis['feature_type_importance'].get)}
- Total features analyzed: {feature_analysis['total_features']}

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open('M2_performance_summary.txt', 'w') as f:
            f.write(summary_text)
        
        print("✓ Results saved:")
        print("  - M2_experimental_results.json")
        print("  - M2_performance_summary.txt")

def main():
    """Example usage"""
    print("M2 Mortality Ensemble Model")
    print("Usage: Import and use with preprocessed data")
    
    # Example parameters
    print("\nModel Configuration:")
    print("- Random Forest: n_estimators=200, optimized hyperparameters")
    print("- SVM: RBF kernel, optimized C and gamma")
    print("- XGBoost: gradient boosting, optimized hyperparameters")
    print("- Ensemble: Soft voting classifier")
    print("- Class Balancing: SMOTE-ENN")

if __name__ == "__main__":
    main() 