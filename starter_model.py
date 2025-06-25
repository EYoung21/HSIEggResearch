import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_processed_data(dataset_type='gender'):
    """
    Load processed data for modeling
    """
    print(f"Loading {dataset_type} dataset...")
    
    X_train = np.load(f'data/processed/{dataset_type}_X_train.npy')
    X_test = np.load(f'data/processed/{dataset_type}_X_test.npy')
    y_train = np.load(f'data/processed/{dataset_type}_y_train.npy')
    y_test = np.load(f'data/processed/{dataset_type}_y_test.npy')
    
    # Load label encoder to get class names
    le = joblib.load(f'data/processed/{dataset_type}_label_encoder.pkl')
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Classes: {le.classes_}")
    
    return X_train, X_test, y_train, y_test, le

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest classifier
    """
    print("\nTraining Random Forest...")
    
    # Initialize and train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Random Forest Accuracy: {accuracy:.3f}")
    
    return rf, y_pred, accuracy

def train_svm(X_train, y_train, X_test, y_test):
    """
    Train SVM classifier
    """
    print("\nTraining SVM...")
    
    # Initialize and train model
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"SVM Accuracy: {accuracy:.3f}")
    
    return svm, y_pred, accuracy

def evaluate_model(y_test, y_pred, le, model_name):
    """
    Print detailed evaluation metrics
    """
    print(f"\n{model_name} Results:")
    print("="*40)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

def cross_validate_model(model, X_train, y_train, cv=5):
    """
    Perform cross-validation
    """
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

def save_model(model, model_name, dataset_type):
    """
    Save trained model
    """
    import os
    os.makedirs('models', exist_ok=True)
    
    filename = f'models/{dataset_type}_{model_name.lower().replace(" ", "_")}.pkl'
    joblib.dump(model, filename)
    print(f"Model saved: {filename}")

def run_baseline_models(dataset_type):
    """
    Run baseline models for specified dataset
    """
    print(f"\n{'='*60}")
    print(f"BASELINE MODELS - {dataset_type.upper()} CLASSIFICATION")
    print(f"{'='*60}")
    
    # Load data
    X_train, X_test, y_train, y_test, le = load_processed_data(dataset_type)
    
    # Check for NaN values
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        print("Warning: NaN values detected, removing affected samples...")
        
        # Remove NaN samples from training set
        train_mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        
        # Remove NaN samples from test set
        test_mask = ~np.isnan(X_test).any(axis=1)
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        
        print(f"After cleaning - Training: {X_train.shape}, Test: {X_test.shape}")
    
    # Train Random Forest
    rf_model, rf_pred, rf_accuracy = train_random_forest(X_train, y_train, X_test, y_test)
    evaluate_model(y_test, rf_pred, le, "Random Forest")
    cross_validate_model(rf_model, X_train, y_train)
    save_model(rf_model, "random_forest", dataset_type)
    
    # Train SVM
    svm_model, svm_pred, svm_accuracy = train_svm(X_train, y_train, X_test, y_test)
    evaluate_model(y_test, svm_pred, le, "SVM")
    cross_validate_model(svm_model, X_train, y_train)
    save_model(svm_model, "svm", dataset_type)
    
    # Summary
    print(f"\n{dataset_type.upper()} CLASSIFICATION SUMMARY:")
    print(f"Random Forest: {rf_accuracy:.3f}")
    print(f"SVM: {svm_accuracy:.3f}")
    
    return rf_accuracy, svm_accuracy

def main():
    """
    Main modeling pipeline
    """
    print("HSI Egg Classification - Baseline Models")
    print("="*50)
    
    results = {}
    
    # Run gender classification
    try:
        rf_acc, svm_acc = run_baseline_models('gender')
        results['gender'] = {'rf': rf_acc, 'svm': svm_acc}
    except Exception as e:
        print(f"Error with gender classification: {e}")
    
    # Run mortality classification
    try:
        rf_acc, svm_acc = run_baseline_models('mortality')
        results['mortality'] = {'rf': rf_acc, 'svm': svm_acc}
    except Exception as e:
        print(f"Error with mortality classification: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("BASELINE RESULTS SUMMARY")
    print("="*60)
    
    for dataset, scores in results.items():
        print(f"{dataset.capitalize()} Classification:")
        print(f"  Random Forest: {scores['rf']:.3f}")
        print(f"  SVM: {scores['svm']:.3f}")
    
    print("\nBaseline models completed!")

if __name__ == "__main__":
    main() 