"""
M6 Experiment Runner: SG + Optimization + Gradient Boosting
Complete pipeline for M6 mortality classification experiment
"""

import sys
import time
import traceback

def run_m6_experiment():
    """Run complete M6 experiment pipeline"""
    print("="*80)
    print("STARTING M6 EXPERIMENT: SG + OPTIMIZATION + GRADIENT BOOSTING")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Step 1: Preprocessing
        print("\nStep 1: Running M6 Preprocessing...")
        print("-" * 50)
        
        from M6_preprocessing import main as preprocessing_main
        X_train, X_test, y_train, y_test, preprocessing_info = preprocessing_main()
        
        print(f"\nPreprocessing completed!")
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Step 2: Model Training and Evaluation
        print("\nStep 2: Running M6 Model Training...")
        print("-" * 50)
        
        from M6_model import main as model_main
        results = model_main()
        
        print(f"\nModel training completed!")
        print(f"Best model: {results['best_model']}")
        print(f"Test accuracy: {results['test_accuracy']:.4f}")
        
        # Final summary
        total_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("M6 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Total experiment time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Best model: {results['best_model']}")
        print(f"Final test accuracy: {results['test_accuracy']:.4f}")
        print(f"Final test AUC: {results['test_auc']:.4f}")
        
        print(f"\nGenerated files:")
        print(f"- X_train_processed.npy, X_test_processed.npy (preprocessed data)")
        print(f"- M6_{results['best_model']}_model.pkl (best model)")
        print(f"- M6_experimental_results.json (complete results)")
        print(f"- M6_performance_summary.txt (human-readable summary)")
        
        return results
        
    except Exception as e:
        print(f"\nERROR in M6 experiment: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_m6_experiment()
    
    if results is not None:
        print(f"\nM6 experiment completed with {results['test_accuracy']:.2%} accuracy!")
    else:
        print("\nM6 experiment failed. Check error messages above.")
        sys.exit(1) 