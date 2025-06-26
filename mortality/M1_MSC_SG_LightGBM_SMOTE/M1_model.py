import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import joblib
import json
import time
import warnings
warnings.filterwarnings('ignore')

class MortalityLightGBMSMOTE:
    """M1 Model: LightGBM + SMOTE-ENN for Mortality Classification"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # SMOTE-ENN for class balancing
        self.smote_enn = SMOTEENN(
            smote=SMOTE(random_state=random_state, k_neighbors=3),
            enn=EditedNearestNeighbours(n_neighbors=3),
            random_state=random_state
        )
        
        # LightGBM parameters
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        
        self.lgb_model = None
        self.ensemble_model = None
        self.training_time = 0
        self.feature_importance = None
        self.cv_scores = []
        
    def optimize_lgb_parameters(self, X_resampled, y_resampled):
        """Optimize LightGBM hyperparameters"""
        print("Optimizing LightGBM hyperparameters...")
        
        from sklearn.model_selection import RandomizedSearchCV
        
        param_distributions = {
            'num_leaves': [15, 31, 63, 127],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'feature_fraction': [0.7, 0.8, 0.9, 1.0],
            'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
            'min_child_samples': [10, 20, 30, 50],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
        
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        random_search = RandomizedSearchCV(
            lgb_model, param_distributions, n_iter=20, cv=3,
            scoring='roc_auc', random_state=self.random_state, n_jobs=-1
        )
        
        random_search.fit(X_resampled, y_resampled)
        
        print(f"✓ Best CV score: {random_search.best_score_:.4f}")
        print(f"✓ Best parameters: {random_search.best_params_}")
        
        self.lgb_params.update(random_search.best_params_)
        return random_search.best_params_
    
    def create_ensemble_model(self, X_resampled, y_resampled):
        """Create ensemble model with multiple LightGBM variants"""
        print("Creating LightGBM ensemble...")
        
        # Base model
        lgb_base = lgb.LGBMClassifier(**self.lgb_params, n_estimators=200)
        
        # Fast variant
        lgb_fast_params = self.lgb_params.copy()
        lgb_fast_params.update({
            'learning_rate': 0.1,
            'n_estimators': 100,
            'num_leaves': 15
        })
        lgb_fast = lgb.LGBMClassifier(**lgb_fast_params)
        
        # Deep variant
        lgb_deep_params = self.lgb_params.copy()
        lgb_deep_params.update({
            'learning_rate': 0.02,
            'n_estimators': 500,
            'num_leaves': 63,
            'min_child_samples': 10
        })
        lgb_deep = lgb.LGBMClassifier(**lgb_deep_params)
        
        ensemble = VotingClassifier(
            estimators=[
                ('lgb_base', lgb_base),
                ('lgb_fast', lgb_fast),
                ('lgb_deep', lgb_deep)
            ],
            voting='soft'
        )
        
        print("✓ Ensemble created with 3 LightGBM variants")
        return ensemble
    
    def train_mortality_model(self, X_train, y_train):
        """Train mortality classification model"""
        print("\n" + "="*60)
        print("TRAINING M1: LIGHTGBM + SMOTE FOR MORTALITY")
        print("="*60)
        
        start_time = time.time()
        
        # Apply SMOTE-ENN
        print("Applying SMOTE-ENN for class balancing...")
        print(f"Original distribution: {np.bincount(y_train)}")
        
        X_resampled, y_resampled = self.smote_enn.fit_resample(X_train, y_train)
        
        print(f"Resampled distribution: {np.bincount(y_resampled)}")
        print(f"✓ Dataset resampled: {X_resampled.shape[0]} samples")
        
        # Optimize hyperparameters
        best_params = self.optimize_lgb_parameters(X_resampled, y_resampled)
        
        # Train main model
        print("\nTraining main LightGBM model...")
        self.lgb_model = lgb.LGBMClassifier(**self.lgb_params, n_estimators=300)
        self.lgb_model.fit(X_resampled, y_resampled)
        
        # Train ensemble
        self.ensemble_model = self.create_ensemble_model(X_resampled, y_resampled)
        self.ensemble_model.fit(X_resampled, y_resampled)
        
        # Cross-validation
        print("\nPerforming cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        lgb_cv_scores = cross_val_score(
            self.lgb_model, X_resampled, y_resampled, cv=cv, scoring='accuracy'
        )
        
        ensemble_cv_scores = cross_val_score(
            self.ensemble_model, X_resampled, y_resampled, cv=cv, scoring='accuracy'
        )
        
        self.cv_scores = {
            'lgb_cv_mean': np.mean(lgb_cv_scores),
            'lgb_cv_std': np.std(lgb_cv_scores),
            'ensemble_cv_mean': np.mean(ensemble_cv_scores),
            'ensemble_cv_std': np.std(ensemble_cv_scores)
        }
        
        self.feature_importance = self.lgb_model.feature_importances_
        self.training_time = time.time() - start_time
        
        print(f"\n✓ Training completed in {self.training_time:.2f} seconds")
        print(f"✓ LightGBM CV: {self.cv_scores['lgb_cv_mean']:.4f} ± {self.cv_scores['lgb_cv_std']:.4f}")
        print(f"✓ Ensemble CV: {self.cv_scores['ensemble_cv_mean']:.4f} ± {self.cv_scores['ensemble_cv_std']:.4f}")
        
        return {
            'lgb_model': self.lgb_model,
            'ensemble_model': self.ensemble_model,
            'cv_scores': self.cv_scores,
            'training_time': self.training_time,
            'best_params': best_params,
            'resampled_size': X_resampled.shape[0]
        }
    
    def evaluate_mortality_prediction(self, X_test, y_test, label_encoder):
        """Evaluate mortality prediction models"""
        print("\n" + "="*60)
        print("EVALUATING MORTALITY PREDICTION MODELS")
        print("="*60)
        
        results = {}
        
        # Evaluate LightGBM
        print("Evaluating LightGBM model...")
        lgb_predictions = self.lgb_model.predict(X_test)
        lgb_probabilities = self.lgb_model.predict_proba(X_test)[:, 1]
        
        lgb_accuracy = accuracy_score(y_test, lgb_predictions)
        lgb_auc = roc_auc_score(y_test, lgb_probabilities)
        
        results['lgb'] = {
            'accuracy': lgb_accuracy,
            'auc': lgb_auc,
            'predictions': lgb_predictions,
            'probabilities': lgb_probabilities,
            'confusion_matrix': confusion_matrix(y_test, lgb_predictions),
            'classification_report': classification_report(
                y_test, lgb_predictions, target_names=label_encoder.classes_, output_dict=True
            )
        }
        
        # Evaluate Ensemble
        print("Evaluating Ensemble model...")
        ensemble_predictions = self.ensemble_model.predict(X_test)
        ensemble_probabilities = self.ensemble_model.predict_proba(X_test)[:, 1]
        
        ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
        ensemble_auc = roc_auc_score(y_test, ensemble_probabilities)
        
        results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'auc': ensemble_auc,
            'predictions': ensemble_predictions,
            'probabilities': ensemble_probabilities,
            'confusion_matrix': confusion_matrix(y_test, ensemble_predictions),
            'classification_report': classification_report(
                y_test, ensemble_predictions, target_names=label_encoder.classes_, output_dict=True
            )
        }
        
        # Best model
        if ensemble_accuracy > lgb_accuracy:
            best_model = 'ensemble'
            best_accuracy = ensemble_accuracy
            best_auc = ensemble_auc
        else:
            best_model = 'lgb'
            best_accuracy = lgb_accuracy
            best_auc = lgb_auc
        
        results['best_model'] = {
            'name': best_model,
            'accuracy': best_accuracy,
            'auc': best_auc
        }
        
        print(f"\nMortality Prediction Results:")
        print(f"  LightGBM - Accuracy: {lgb_accuracy:.4f}, AUC: {lgb_auc:.4f}")
        print(f"  Ensemble - Accuracy: {ensemble_accuracy:.4f}, AUC: {ensemble_auc:.4f}")
        print(f"  Best Model: {best_model} (Accuracy: {best_accuracy:.4f})")
        
        return results
    
    def analyze_feature_importance(self, preprocessing_info, top_k=20):
        """Analyze feature importance"""
        print(f"\nAnalyzing top {top_k} features...")
        
        importance_scores = self.feature_importance
        n_selected = preprocessing_info['n_selected_features']
        
        feature_names = []
        for i in range(n_selected):
            feature_names.append(f'StandardScale_Feature_{i+1}')
        for i in range(n_selected):
            feature_names.append(f'MinMaxScale_Feature_{i+1}')
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_k} Most Important Features:")
        print("-" * 50)
        for idx, row in importance_df.head(top_k).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        standard_importance = importance_df[importance_df['feature'].str.contains('StandardScale')]['importance'].sum()
        minmax_importance = importance_df[importance_df['feature'].str.contains('MinMaxScale')]['importance'].sum()
        
        print(f"\nImportance by Scaling Method:")
        print(f"  Standard Scaled: {standard_importance:.4f}")
        print(f"  MinMax Scaled: {minmax_importance:.4f}")
        
        return {
            'importance_scores': importance_scores,
            'feature_names': feature_names,
            'top_features': importance_df.head(top_k).to_dict('records'),
            'scaling_analysis': {
                'standard_total': standard_importance,
                'minmax_total': minmax_importance
            }
        }
    
    def save_model_and_results(self, results, feature_analysis, preprocessing_info):
        """Save models and results"""
        print("\nSaving M1 mortality models and results...")
        
        # Save models
        joblib.dump(self.lgb_model, 'M1_lightgbm_model.pkl')
        joblib.dump(self.ensemble_model, 'M1_ensemble_model.pkl')
        joblib.dump(self.smote_enn, 'M1_smote_enn.pkl')
        
        # Save feature importance
        feature_importance_df = pd.DataFrame({
            'feature': feature_analysis['feature_names'],
            'importance': feature_analysis['importance_scores']
        }).sort_values('importance', ascending=False)
        
        feature_importance_df.to_csv('feature_importance.csv', index=False)
        
        # Comprehensive results
        experiment_results = {
            'experiment_id': 'M1_MSC_SG_LightGBM_SMOTE',
            'methodology': {
                'preprocessing': 'MSC + Savitzky-Golay filtering',
                'feature_selection': preprocessing_info['feature_selection_method'],
                'class_balancing': 'SMOTE-ENN',
                'algorithm': 'LightGBM + Ensemble',
                'total_features': preprocessing_info['n_final_features']
            },
            'performance': {
                'best_model': results['best_model']['name'],
                'best_accuracy': float(results['best_model']['accuracy']),
                'best_auc': float(results['best_model']['auc']),
                'lgb_accuracy': float(results['lgb']['accuracy']),
                'lgb_auc': float(results['lgb']['auc']),
                'ensemble_accuracy': float(results['ensemble']['accuracy']),
                'ensemble_auc': float(results['ensemble']['auc']),
                'cv_scores': self.cv_scores,
                'training_time': self.training_time
            },
            'feature_analysis': {
                'top_10_features': feature_analysis['top_features'][:10],
                'scaling_importance': feature_analysis['scaling_analysis']
            },
            'preprocessing_details': preprocessing_info,
            'model_parameters': self.lgb_params
        }
        
        with open('M1_experimental_results.json', 'w') as f:
            json.dump(experiment_results, f, indent=2, default=str)
        
        summary_text = f"""
M1 MORTALITY CLASSIFICATION - PERFORMANCE SUMMARY
============================================================
Methodology: MSC + SG + LightGBM + SMOTE-ENN

RESULTS:
- Best Model: {results['best_model']['name']}
- Best Accuracy: {results['best_model']['accuracy']:.4f} ({results['best_model']['accuracy']*100:.2f}%)
- Best AUC: {results['best_model']['auc']:.4f}
- LightGBM Accuracy: {results['lgb']['accuracy']:.4f}
- Ensemble Accuracy: {results['ensemble']['accuracy']:.4f}
- Cross-Validation: {self.cv_scores['lgb_cv_mean']:.4f} ± {self.cv_scores['lgb_cv_std']:.4f}

PREPROCESSING:
- Original Features: {preprocessing_info['n_original_features']}
- Selected Features: {preprocessing_info['n_selected_features']}
- Final Features: {preprocessing_info['n_final_features']}
- Feature Selection: {preprocessing_info['feature_selection_method']}
- Class Balancing: SMOTE-ENN

TRAINING:
- Training Time: {self.training_time:.2f} seconds
- Algorithm: LightGBM + Ensemble Voting
"""
        
        with open('M1_performance_summary.txt', 'w') as f:
            f.write(summary_text)
        
        print("✓ Models and results saved:")
        print("  - M1_lightgbm_model.pkl")
        print("  - M1_ensemble_model.pkl") 
        print("  - M1_experimental_results.json")
        print("  - M1_performance_summary.txt")
        print("  - feature_importance.csv")

def main():
    """Main M1 model training and evaluation pipeline"""
    print("="*80)
    print("M1 MODEL: LIGHTGBM + SMOTE FOR MORTALITY CLASSIFICATION")
    print("="*80)
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load('X_train_processed.npy')
    X_test = np.load('X_test_processed.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    label_encoder = joblib.load('label_encoder.pkl')
    
    with open('preprocessing_info.json', 'r') as f:
        preprocessing_info = json.load(f)
    
    print(f"✓ Data loaded:")
    print(f"  - Training: {X_train.shape}")
    print(f"  - Test: {X_test.shape}")
    print(f"  - Classes: {label_encoder.classes_}")
    
    # Train model
    model = MortalityLightGBMSMOTE(random_state=42)
    training_results = model.train_mortality_model(X_train, y_train)
    
    # Evaluate model
    evaluation_results = model.evaluate_mortality_prediction(X_test, y_test, label_encoder)
    
    # Feature importance analysis
    feature_analysis = model.analyze_feature_importance(preprocessing_info, top_k=20)
    
    # Save everything
    model.save_model_and_results(evaluation_results, feature_analysis, preprocessing_info)
    
    print("\n" + "="*80)
    print("M1 EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Best Model: {evaluation_results['best_model']['name']}")
    print(f"Final Accuracy: {evaluation_results['best_model']['accuracy']:.4f}")
    print(f"Final AUC: {evaluation_results['best_model']['auc']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
