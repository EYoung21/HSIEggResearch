import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import warnings
warnings.filterwarnings('ignore')

# For Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    print("Warning: scikit-optimize not available. Using default hyperparameters.")
    BAYESIAN_OPT_AVAILABLE = False

class LightGBM_BayesOpt:
    """
    LightGBM Classifier with Bayesian Optimization for HSI gender classification
    """
    
    def __init__(self, use_bayesian_opt=True, n_calls=50, cv_folds=5, random_state=42):
        """
        Initialize LightGBM with Bayesian optimization
        
        Args:
            use_bayesian_opt: Whether to use Bayesian optimization for hyperparameters
            n_calls: Number of optimization calls
            cv_folds: Number of CV folds for optimization
            random_state: Random state for reproducibility
        """
        self.use_bayesian_opt = use_bayesian_opt and BAYESIAN_OPT_AVAILABLE
        self.n_calls = n_calls
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params = None
        self.model = None
        self.feature_importance = None
        
        # Default hyperparameters
        self.default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state
        }
        
        # Search space for Bayesian optimization
        self.search_space = [
            Integer(20, 100, name='num_leaves'),
            Real(0.01, 0.3, name='learning_rate', prior='log-uniform'),
            Real(0.6, 1.0, name='feature_fraction'),
            Real(0.6, 1.0, name='bagging_fraction'),
            Integer(1, 10, name='bagging_freq'),
            Real(0.01, 10.0, name='lambda_l1', prior='log-uniform'),
            Real(0.01, 10.0, name='lambda_l2', prior='log-uniform'),
            Integer(10, 50, name='min_data_in_leaf'),
            Real(0.1, 1.0, name='min_gain_to_split')
        ]
    
    def objective_function(self, params):
        """
        Objective function for Bayesian optimization
        
        Args:
            params: List of hyperparameter values
            
        Returns:
            Negative cross-validation accuracy (for minimization)
        """
        # Unpack parameters
        param_dict = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': params[0],
            'learning_rate': params[1],
            'feature_fraction': params[2],
            'bagging_fraction': params[3],
            'bagging_freq': params[4],
            'lambda_l1': params[5],
            'lambda_l2': params[6],
            'min_data_in_leaf': params[7],
            'min_gain_to_split': params[8],
            'verbose': -1,
            'random_state': self.random_state
        }
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        scores = []
        
        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            X_tr, X_val = self.X_train[train_idx], self.X_train[val_idx]
            y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
            
            # Train model
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                param_dict,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
            )
            
            # Predict and score
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            y_pred_binary = (y_pred > 0.5).astype(int)
            scores.append(accuracy_score(y_val, y_pred_binary))
        
        # Return negative mean accuracy (for minimization)
        return -np.mean(scores)
    
    def optimize_hyperparameters(self, X_train, y_train):
        """
        Optimize hyperparameters using Bayesian optimization
        """
        print("Starting Bayesian optimization...")
        print(f"Search space: {len(self.search_space)} hyperparameters")
        print(f"Optimization calls: {self.n_calls}")
        print(f"Cross-validation folds: {self.cv_folds}")
        
        # Store training data for objective function
        self.X_train = X_train
        self.y_train = y_train
        
        # Define objective with named parameters
        @use_named_args(self.search_space)
        def objective(**params):
            param_values = [params[dim.name] for dim in self.search_space]
            return self.objective_function(param_values)
        
        # Perform optimization
        result = gp_minimize(
            func=objective,
            dimensions=self.search_space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            acq_func='EI',  # Expected Improvement
            n_initial_points=10
        )
        
        # Extract best parameters
        best_params = dict(zip([dim.name for dim in self.search_space], result.x))
        
        # Combine with fixed parameters
        self.best_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': self.random_state,
            **best_params
        }
        
        print(f"✓ Optimization complete!")
        print(f"Best CV accuracy: {-result.fun:.4f}")
        print(f"Best parameters: {best_params}")
        
        return self.best_params
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print("\n" + "="*50)
        print("TRAINING LIGHTGBM MODEL")
        print("="*50)
        
        # Optimize hyperparameters if requested
        if self.use_bayesian_opt:
            params = self.optimize_hyperparameters(X_train, y_train)
        else:
            params = self.default_params
            print("Using default hyperparameters")
        
        # Prepare datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Train final model
        print("\nTraining final model...")
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)]
        )
        
        # Store feature importance
        self.feature_importance = self.model.feature_importance(importance_type='gain')
        
        print("✓ Model training complete!")
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return y_pred, y_pred_proba
    
    def evaluate(self, X_test, y_test, label_encoder=None):
        """
        Evaluate model performance
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Make predictions
        y_pred, y_pred_proba = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test samples: {len(y_test)}")
        
        # Classification report
        if label_encoder is not None:
            target_names = label_encoder.classes_
        else:
            target_names = ['Class 0', 'Class 1']
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def get_feature_importance(self, wavelengths=None, top_n=20):
        """
        Get feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained or feature importance not available.")
        
        # Create feature importance dataframe
        if wavelengths is not None:
            feature_names = [f"wavelength_{w}" for w in wavelengths]
        else:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(importance_df.head(top_n))
        
        return importance_df
    
    def save_model(self, filepath):
        """
        Save trained model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Save LightGBM model
        self.model.save_model(filepath)
        
        # Save additional info
        model_info = {
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'use_bayesian_opt': self.use_bayesian_opt
        }
        
        joblib.dump(model_info, filepath.replace('.txt', '_info.pkl'))
        print(f"✓ Model saved to {filepath}")

def load_processed_data():
    """
    Load preprocessed data from G1 preprocessing step
    """
    print("Loading preprocessed data...")
    
    X_train = np.load('X_train_processed.npy')
    X_test = np.load('X_test_processed.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    
    # Load metadata
    wavelengths_df = pd.read_csv('wavelengths.csv')
    wavelengths = wavelengths_df['wavelength'].tolist()
    
    label_encoder = joblib.load('label_encoder.pkl')
    
    print(f"✓ Training data: {X_train.shape}")
    print(f"✓ Test data: {X_test.shape}")
    print(f"✓ Classes: {label_encoder.classes_}")
    print(f"✓ Wavelengths: {len(wavelengths)}")
    
    return X_train, X_test, y_train, y_test, wavelengths, label_encoder

def main():
    """
    Main training and evaluation pipeline
    """
    print("="*60)
    print("G1 EXPERIMENT: LightGBM with Bayesian Optimization")
    print("="*60)
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test, wavelengths, le = load_processed_data()
    
    # Initialize and train model
    lgb_model = LightGBM_BayesOpt(
        use_bayesian_opt=BAYESIAN_OPT_AVAILABLE,
        n_calls=30,  # Reduce for faster testing
        cv_folds=5,
        random_state=42
    )
    
    # Train model
    lgb_model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    results = lgb_model.evaluate(X_test, y_test, le)
    
    # Show feature importance
    importance_df = lgb_model.get_feature_importance(wavelengths, top_n=20)
    
    # Save model and results
    lgb_model.save_model('lightgbm_model.txt')
    
    # Save results
    np.save('test_predictions.npy', results['predictions'])
    np.save('test_probabilities.npy', results['probabilities'])
    importance_df.to_csv('feature_importance.csv', index=False)
    
    print("\n" + "="*60)
    print("G1 EXPERIMENT COMPLETE!")
    print("="*60)
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")
    print("Files saved:")
    print("- lightgbm_model.txt (trained model)")
    print("- lightgbm_model_info.pkl (model metadata)")
    print("- test_predictions.npy (predictions)")
    print("- test_probabilities.npy (prediction probabilities)")
    print("- feature_importance.csv (wavelength importance)")

if __name__ == "__main__":
    main() 