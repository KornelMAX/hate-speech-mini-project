# scripts/train_svm.py
import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ResultsManager, EfficiencyTracker, compute_all_metrics


def main():
    # Configuration
    MODEL_NAME = "svm"
    SEED = 42
    np.random.seed(SEED)

    # Initialize managers
    results_mgr = ResultsManager(MODEL_NAME)
    efficiency_tracker = EfficiencyTracker(MODEL_NAME)

    print("\n" + "="*70)
    print("Training: Linear SVM with Character N-grams (calibrated)")
    print("="*70 + "\n")

    # ========================================================================
    # Load Pre-split Data
    # ========================================================================
    print("[1/8] Loading pre-split data...")

    train_df = pd.read_csv('data/train_split.csv')
    val_df = pd.read_csv('data/val_split.csv')
    test_df = pd.read_csv('data/test_split.csv')

    X_train, y_train = train_df['text'], train_df['label']
    X_val, y_val = val_df['text'], val_df['label']
    X_test, y_test = test_df['text'], test_df['label']

    print(f"      Train: {len(X_train):,} samples")
    print(f"      Val:   {len(X_val):,} samples")
    print(f"      Test:  {len(X_test):,} samples")

    # ========================================================================
    # Vectorization
    # ========================================================================
    print("\n[2/8] Creating character n-gram vectorizer...")

    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        max_features=20000,
        lowercase=True,
        sublinear_tf=True
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    print(f"      Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"      Feature matrix shape: {X_train_vec.shape}")

    # ========================================================================
    # Hyperparameter Tuning
    # ========================================================================
    print("\n[3/8] Hyperparameter tuning (5-fold CV)...")

    # Be nice to liblinear: more iters and dual='auto' lets sklearn choose based on shape
    svm = LinearSVC(
        class_weight='balanced',
        max_iter=10000,
        dual='auto',        # prefer primal when n_samples > n_features
        random_state=SEED
    )

    param_grid = {'C': [0.1, 1, 10, 100]}

    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_vec, y_train)

    print(f"      Best C: {grid_search.best_params_['C']}")
    print(f"      Best CV F1: {grid_search.best_score_:.4f}")

    # ========================================================================
    # Final Training
    # ========================================================================
    print("\n[4/8] Training final model with best parameters...")

    final_model = LinearSVC(
        C=grid_search.best_params_['C'],
        class_weight='balanced',
        max_iter=10000,
        dual='auto',
        random_state=SEED
    )

    efficiency_tracker.start_training()
    final_model.fit(X_train_vec, y_train)
    train_metrics = efficiency_tracker.end_training()
    print(f"      Training completed in {train_metrics['train_time_minutes']:.2f} minutes")

    # ========================================================================
    # Calibration on validation split (no leakage)
    # ========================================================================
    print("\n[4.5/8] Calibrating probabilities on validation split...")

    # scikit-learn >= 1.6 deprecates cv='prefit' in favor of FrozenEstimator
    use_frozen = False
    try:
        from sklearn.frozen import FrozenEstimator  # added in 1.6
        use_frozen = True
    except Exception:
        use_frozen = False

    def fit_calibrator(est, X_cal, y_cal, method):
        if use_frozen:
            # New API: wrap the already-fit estimator
            cal = CalibratedClassifierCV(FrozenEstimator(est), method=method)
            cal.fit(X_cal, y_cal)
            return cal
        else:
            # Older API: 'estimator' and cv='prefit'
            cal = CalibratedClassifierCV(estimator=est, method=method, cv='prefit')
            cal.fit(X_cal, y_cal)
            return cal

    cal_sigmoid  = fit_calibrator(final_model, X_val_vec, y_val, method='sigmoid')
    cal_isotonic = fit_calibrator(final_model, X_val_vec, y_val, method='isotonic')

    proba_sig = cal_sigmoid.predict_proba(X_val_vec)[:, 1]
    proba_iso = cal_isotonic.predict_proba(X_val_vec)[:, 1]

    brier_sig = brier_score_loss(y_val, proba_sig)
    brier_iso = brier_score_loss(y_val, proba_iso)

    if brier_sig <= brier_iso:
        calibrator = cal_sigmoid
        chosen_method = 'sigmoid'
        best_brier = brier_sig
    else:
        calibrator = cal_isotonic
        chosen_method = 'isotonic'
        best_brier = brier_iso

    print(f"      Chosen calibration: {chosen_method} (val Brier={best_brier:.4f})")

    # ========================================================================
    # Test Set Evaluation (with calibrated probabilities)
    # ========================================================================
    print("\n[5/8] Evaluating on test set (calibrated)...")

    y_proba = calibrator.predict_proba(X_test_vec)[:, 1]   # true probabilities
    y_pred  = (y_proba >= 0.5).astype(int)

    metrics = compute_all_metrics(y_test.values, y_pred, y_proba)

    print("\nTest Set Metrics (calibrated):")
    print(f"  Macro F1:     {metrics['f1_macro']:.4f}")
    print(f"  Hate F1:      {metrics['f1_hate']:.4f}")
    print(f"  PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"  Brier:        {metrics.get('brier_score', float('nan')):.4f}")
    print(f"  MCC:          {metrics['mcc']:.4f}")

    # ========================================================================
    # Efficiency Measurement
    # ========================================================================
    print("\n[6/8] Measuring efficiency...")

    def predict_fn(model, texts):
        if isinstance(texts, str):
            texts = [texts]
        vec = vectorizer.transform(texts)
        return model.predict(vec)  # CalibratedClassifierCV supports predict()

    latency_metrics = efficiency_tracker.measure_inference_latency(
        calibrator, predict_fn, X_test, num_samples=100
    )

    throughput_metrics = efficiency_tracker.measure_throughput(
        calibrator, predict_fn, X_test, batch_size=1000
    )

    efficiency_metrics = {
        **train_metrics,
        **latency_metrics,
        **throughput_metrics,
        'num_parameters': X_train_vec.shape[1],  # number of features
        'parameters_millions': X_train_vec.shape[1] / 1e6,
    }

    print(f"  Latency (p50):  {latency_metrics['latency_p50_ms']:.2f} ms")
    print(f"  Latency (p95):  {latency_metrics['latency_p95_ms']:.2f} ms")
    print(f"  Throughput:     {throughput_metrics['throughput_samples_per_sec']:.2f} samples/sec")

    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n[7/8] Saving results...")

    results_mgr.save_metrics(metrics)
    results_mgr.save_efficiency(efficiency_metrics)
    results_mgr.save_predictions(y_test.values, y_pred, y_proba)

    training_log = {
        'model_name': 'Linear SVM (calibrated)',
        'vectorizer': 'TF-IDF char n-grams (3-5)',
        'seed': SEED,
        'hyperparameters': {
            'C': grid_search.best_params_['C'],
            'class_weight': 'balanced',
            'max_features': 20000,
            'max_iter': 10000,
            'dual': 'auto',
        },
        'cv_best_f1': grid_search.best_score_,
        'calibration_method': chosen_method,
    }
    results_mgr.save_training_log(training_log)

    # Save calibrated model + vectorizer
    import joblib
    model_path = f"models/{MODEL_NAME}_calibrated.pkl"
    vectorizer_path = f"models/{MODEL_NAME}_vectorizer.pkl"

    os.makedirs("models", exist_ok=True)
    joblib.dump(calibrator, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"      Calibrated model saved: {model_path}")
    print(f"      Vectorizer saved: {vectorizer_path}")

    # ========================================================================
    # Reliability Diagram
    # ========================================================================
    print("\n[7.5/8] Saving reliability diagram...")
    prob_true, prob_pred = calibration_curve(y_test.values, y_proba, n_bins=10, strategy='quantile')
    plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1],'--', label='Perfectly calibrated')
    plt.plot(prob_pred, prob_true, marker='o', label=f'SVM ({chosen_method})')
    plt.xlabel('Predicted probability')
    plt.ylabel('Empirical frequency')
    plt.title('Reliability diagram (test)')
    plt.legend()
    plot_path = f"results/{MODEL_NAME}/reliability_test.png"
    os.makedirs(f"results/{MODEL_NAME}", exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight', dpi=160)
    print(f"      Reliability diagram saved: {plot_path}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n[8/8] Training Summary")
    print("="*70)
    print(f"Model:             Linear SVM (calibrated)")
    print(f"Macro F1:          {metrics['f1_macro']:.4f}")
    print(f"Hate F1:           {metrics['f1_hate']:.4f}")
    print(f"Training Time:     {train_metrics['train_time_minutes']:.2f} minutes")
    print(f"Features:          {X_train_vec.shape[1]:,}")
    print(f"\nResults saved to:  results/{MODEL_NAME}/")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
