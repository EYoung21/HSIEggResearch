import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class AdvancedVotingEnsemble:
    """Advanced voting ensemble classifier"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = {}
        self.ensemble_models = {}
        self.feature_set_performance = {}
        
    def create_base_models(self):
        """Create diverse base models"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            ),
            'svm_rbf': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                random_state=self.random_state,
                max_iter=1000
            ),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
        }
        return models
    
    def train_models_on_feature_set(self, X_train, y_train, feature_set_name):
        """Train all base models on a specific feature set"""
        print(f"\nTraining models on {feature_set_name} features...")
        
        base_models = self.create_base_models()
        trained_models = {}
        performance_scores = {}
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for model_name, model in base_models.items():
            print(f"  Training {model_name}...")
            try:
                model.fit(X_train, y_train)
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                mean_cv_score = np.mean(cv_scores)
                std_cv_score = np.std(cv_scores)
                
                print(f"    CV Score: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
                
                trained_models[model_name] = model
                performance_scores[model_name] = {
                    'cv_mean': mean_cv_score,
                    'cv_std': std_cv_score,
                    'cv_scores': cv_scores.tolist()
                }
                
            except Exception as e:
                print(f"    Failed to train {model_name}: {e}")
        
        print(f"✓ Successfully trained {len(trained_models)} models on {feature_set_name}")
        return trained_models, performance_scores
    
    def create_voting_ensembles(self, feature_set_models, feature_set_name):
        """Create voting ensemble strategies"""
        print(f"Creating voting ensembles for {feature_set_name}...")
        
        ensembles = {}
        
        if len(feature_set_models) >= 3:
            all_models_list = [(name, model) for name, model in feature_set_models.items()]
            
            hard_voting = VotingClassifier(
                estimators=all_models_list,
                voting='hard'
            )
            
            soft_voting = VotingClassifier(
                estimators=all_models_list,
                voting='soft'
            )
            
            ensembles[f'{feature_set_name}_hard_voting'] = hard_voting
            ensembles[f'{feature_set_name}_soft_voting'] = soft_voting
        
        print(f"✓ Created {len(ensembles)} voting ensembles for {feature_set_name}")
        return ensembles
    
    def train_all_ensembles(self, feature_sets_data, y_train):
        """Train ensembles on all feature sets"""
        print("\n" + "="*60)
        print("TRAINING ADVANCED VOTING ENSEMBLES")
        print("="*60)
        
        all_trained_models = {}
        all_ensembles = {}
        
        for feature_set_name, X_train in feature_sets_data.items():
            print(f"\n--- Processing {feature_set_name} ---")
            
            trained_models, performance = self.train_models_on_feature_set(
                X_train, y_train, feature_set_name
            )
            
            all_trained_models[feature_set_name] = trained_models
            self.feature_set_performance[feature_set_name] = performance
            
            ensembles = self.create_voting_ensembles(trained_models, feature_set_name)
            
            for ensemble_name, ensemble_model in ensembles.items():
                try:
                    ensemble_model.fit(X_train, y_train)
                    all_ensembles[ensemble_name] = ensemble_model
                    print(f"  ✓ Trained {ensemble_name}")
                except Exception as e:
                    print(f"  ✗ Failed to train {ensemble_name}: {e}")
        
        self.base_models = all_trained_models
        self.ensemble_models = all_ensembles
        
        print(f"\n✓ Training completed:")
        print(f"  - Feature sets: {len(feature_sets_data)}")
        print(f"  - Base models: {sum(len(models) for models in all_trained_models.values())}")
        print(f"  - Ensemble models: {len(all_ensembles)}")
        
        return all_trained_models, all_ensembles
    
    def evaluate_all_models(self, feature_sets_test, y_test, label_encoder):
        """Evaluate all models and ensembles"""
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS AND ENSEMBLES")
        print("="*60)
        
        results = {}
        
        # Evaluate base models
        print("\n--- Base Models ---")
        for feature_set_name, models in self.base_models.items():
            X_test = feature_sets_test[feature_set_name]
            
            for model_name, model in models.items():
                full_name = f"{feature_set_name}_{model_name}"
                
                try:
                    predictions = model.predict(X_test)
                    probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else predictions
                    accuracy = accuracy_score(y_test, predictions)
                    
                    results[full_name] = {
                        'type': 'base_model',
                        'feature_set': feature_set_name,
                        'algorithm': model_name,
                        'accuracy': accuracy,
                        'predictions': predictions,
                        'probabilities': probabilities
                    }
                    
                    print(f"  {full_name}: {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"  {full_name}: Failed - {e}")
        
        # Evaluate ensemble models
        print("\n--- Ensemble Models ---")
        for ensemble_name, ensemble_model in self.ensemble_models.items():
            # Find corresponding feature set
            feature_set_name = None
            for fs_name in feature_sets_test.keys():
                if ensemble_name.startswith(fs_name):
                    feature_set_name = fs_name
                    break
            
            if feature_set_name:
                X_test = feature_sets_test[feature_set_name]
                
                try:
                    predictions = ensemble_model.predict(X_test)
                    probabilities = ensemble_model.predict_proba(X_test)[:, 1] if hasattr(ensemble_model, 'predict_proba') else predictions
                    accuracy = accuracy_score(y_test, predictions)
                    
                    results[ensemble_name] = {
                        'type': 'ensemble',
                        'feature_set': feature_set_name,
                        'algorithm': ensemble_name,
                        'accuracy': accuracy,
                        'predictions': predictions,
                        'probabilities': probabilities
                    }
                    
                    print(f"  {ensemble_name}: {accuracy:.4f}")
                    
                except Exception as e:
                    print(f"  {ensemble_name}: Failed - {e}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        print(f"\n🏆 BEST MODEL: {best_model_name}")
        print(f"🎯 BEST ACCURACY: {best_accuracy:.4f}")
        
        # Detailed evaluation of best model
        print(f"\n=== BEST MODEL DETAILS ===")
        best_result = results[best_model_name]
        print(f"Type: {best_result['type']}")
        print(f"Feature Set: {best_result['feature_set']}")
        print(f"Algorithm: {best_result['algorithm']}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, best_result['predictions'], target_names=label_encoder.classes_))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, best_result['predictions']))
        
        return results, best_model_name, best_accuracy

def load_feature_sets():
    """Load all preprocessed feature sets"""
    print("Loading ensemble feature sets...")
    
    feature_set_names = ['snv_sg', 'wavelet', 'combined', 'snv_minmax', 'snv_robust']
    
    X_train_dict = {}
    X_test_dict = {}
    
    for name in feature_set_names:
        try:
            X_train = np.load(f'X_train_{name}.npy')
            X_test = np.load(f'X_test_{name}.npy')
            
            X_train_dict[name] = X_train
            X_test_dict[name] = X_test
            
            print(f"✓ Loaded {name}: train {X_train.shape}, test {X_test.shape}")
            
        except FileNotFoundError:
            print(f"✗ Failed to load {name}")
    
    return X_train_dict, X_test_dict

def main():
    """Main voting ensemble pipeline"""
    print("="*70)
    print("G7 VOTING ENSEMBLE: Wavelet + SNV + Advanced Voting")
    print("="*70)
    
    # Load feature sets
    X_train_dict, X_test_dict = load_feature_sets()
    
    # Load labels
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    label_encoder = joblib.load('label_encoder.pkl')
    
    print(f"\nDataset summary:")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Feature sets: {len(X_train_dict)}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Initialize voting ensemble
    ensemble = AdvancedVotingEnsemble(random_state=42)
    
    # Train all ensembles
    trained_models, ensemble_models = ensemble.train_all_ensembles(X_train_dict, y_train)
    
    # Evaluate all models
    results, best_model, best_accuracy = ensemble.evaluate_all_models(
        X_test_dict, y_test, label_encoder
    )
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save ensemble models
    ensemble_save_data = {
        'base_models': trained_models,
        'ensemble_models': ensemble_models,
        'feature_set_performance': ensemble.feature_set_performance
    }
    
    joblib.dump(ensemble_save_data, 'G7_voting_ensemble_models.pkl')
    
    # Save experimental results
    experiment_results = {
        'experiment': 'G7_Wavelet_SNV_Voting',
        'preprocessing': 'Wavelet + SNV + Multi-path ensemble',
        'algorithm': 'Advanced Voting Ensemble',
        'best_model': best_model,
        'best_accuracy': float(best_accuracy),
        'feature_sets': len(X_train_dict),
        'total_models': len(results),
        'training_samples': int(len(y_train)),
        'test_samples': int(len(y_test)),
        'model_results': {}
    }
    
    # Add model results
    for model_name, result in results.items():
        experiment_results['model_results'][model_name] = {
            'type': result['type'],
            'feature_set': result['feature_set'],
            'accuracy': float(result['accuracy']),
            'predictions': result['predictions'].tolist(),
            'probabilities': result['probabilities'].tolist()
        }
    
    with open('G7_experimental_results.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    # Create performance summary
    base_models = {k: v for k, v in results.items() if v['type'] == 'base_model'}
    ensembles = {k: v for k, v in results.items() if v['type'] == 'ensemble'}
    
    summary = f"""G7 EXPERIMENT SUMMARY: Wavelet + SNV + Voting Ensemble
=======================================================

METHODOLOGY:
- Preprocessing: Wavelet decomposition + SNV normalization + Multi-path features
- Algorithm: Advanced Voting Ensemble
- Feature Sets: {len(X_train_dict)} different preprocessing approaches
- Models: {len(results)} total models (base + ensemble)

DATASET:
- Training samples: {len(y_train)}
- Test samples: {len(y_test)}
- Classes: {', '.join(label_encoder.classes_)}

BEST MODEL: {best_model}
BEST ACCURACY: {best_accuracy:.4f}

Top 5 Base Models:
"""
    
    for model_name, result in sorted(base_models.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:5]:
        summary += f"- {model_name}: {result['accuracy']:.4f}\n"
    
    summary += f"""
Ensemble Models ({len(ensembles)}):
"""
    
    for model_name, result in sorted(ensembles.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        summary += f"- {model_name}: {result['accuracy']:.4f}\n"
    
    summary += f"""
FILES GENERATED:
- G7_voting_ensemble_models.pkl (all trained models)
- G7_experimental_results.json (complete results)
- G7_performance_summary.txt (this summary)
"""
    
    with open('G7_performance_summary.txt', 'w') as f:
        f.write(summary)
    
    print("✓ Saved G7_voting_ensemble_models.pkl")
    print("✓ Saved G7_experimental_results.json")
    print("✓ Saved G7_performance_summary.txt")
    
    print(f"\n🎯 G7 RESULTS: {best_model} - {best_accuracy:.4f}")
    print("✅ G7 voting ensemble experiment completed!")

if __name__ == "__main__":
    main() 