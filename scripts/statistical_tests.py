# scripts/statistical_tests.py
"""
Statistical comparison with ROC-AUC and effect sizes
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import roc_auc_score
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def load_predictions(model_name):
    """Load predictions for a model"""
    pred_file = Path(f"results/{model_name}/predictions.csv")
    if not pred_file.exists():
        return None
    df = pd.read_csv(pred_file)
    return df['y_true'].values, df['y_pred'].values, df.get('y_proba_class_1', None)

def cohens_h(n01, n10):
    """
    Calculate Cohen's h effect size for McNemar test
    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    where p1 = n01/(n01+n10), p2 = n10/(n01+n10)
    """
    total = n01 + n10
    if total == 0:
        return 0.0
    
    p1 = n01 / total
    p2 = n10 / total
    
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    return h

def interpret_cohens_h(h):
    """Interpret Cohen's h effect size"""
    h_abs = abs(h)
    if h_abs < 0.2:
        return "negligible"
    elif h_abs < 0.5:
        return "small"
    elif h_abs < 0.8:
        return "medium"
    else:
        return "large"

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Perform McNemar's test with effect size
    """
    n00 = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    n01 = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    n10 = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    n11 = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))
    
    table = np.array([[n00, n01], [n10, n11]])
    
    if n01 + n10 == 0:
        return 0.0, 1.0, table, 0.0, 0.0, "negligible"
    
    result = mcnemar(table, exact=False, correction=True)
    
    odds_ratio = n01 / n10 if n10 > 0 else float('inf')
    effect_size = cohens_h(n01, n10)
    interpretation = interpret_cohens_h(effect_size)
    
    return result.statistic, result.pvalue, table, odds_ratio, effect_size, interpretation

def bootstrap_ci(y_true, y_pred, metric_fn, n_iterations=1000, alpha=0.05):
    """Compute bootstrap confidence interval"""
    n = len(y_true)
    metrics = []
    
    rng = np.random.RandomState(42)
    for _ in range(n_iterations):
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        metrics.append(metric_fn(y_true_boot, y_pred_boot))
    
    metrics = np.array(metrics)
    lower = np.percentile(metrics, (alpha/2) * 100)
    upper = np.percentile(metrics, (1 - alpha/2) * 100)
    
    return lower, upper

def f1_macro(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='macro')

def main():
    print("="*70)
    print("Statistical Testing with ROC-AUC and Effect Sizes")
    print("="*70)
    
    primary_models = [
    "svm",
    "bertweet_off_the_shelf",
    "bertweet_finetuned_weighted",
    "tw_roberta_hate_off_the_shelf",
    "tw_roberta_hate_finetuned_weighted",
    ]
    
    print("\nPrimary models:")
    for m in primary_models:
        print(f"  • {m}")
    
    # Load predictions
    print("\nLoading predictions...")
    models_data = {}
    for model in primary_models:
        result = load_predictions(model)
        if result is not None:
            y_true, y_pred, y_proba = result
            models_data[model] = {
                'y_true': y_true, 
                'y_pred': y_pred,
                'y_proba': y_proba.values if y_proba is not None else None
            }
            print(f"  ✓ {model}")
        else:
            print(f"  ✗ {model}: not found")
    
    if len(models_data) < 2:
        print("\nERROR: Need at least 2 models!")
        return
    
    # Verify same test set
    y_true_ref = list(models_data.values())[0]['y_true']
    for model, data in models_data.items():
        if not np.array_equal(y_true_ref, data['y_true']):
            print(f"\nWARNING: {model} has different test set!")
            return
    
    print("\n✓ All models use same test set")
    
    # Compute ROC-AUC
    print("\n" + "="*70)
    print("ROC-AUC Scores")
    print("="*70)
    
    roc_results = []
    for model, data in models_data.items():
        if data['y_proba'] is not None:
            roc_auc = roc_auc_score(data['y_true'], data['y_proba'])
            print(f"{model:40s}: {roc_auc:.4f}")
            roc_results.append({'Model': model, 'ROC_AUC': roc_auc})
        else:
            print(f"{model:40s}: N/A (no probabilities)")
    
    # McNemar tests with effect sizes
    print("\n" + "="*70)
    print("McNemar's Test with Effect Sizes")
    print("="*70)
    
    model_names = list(models_data.keys())
    n_comparisons = len(list(combinations(model_names, 2)))
    alpha = 0.05
    bonferroni_alpha = alpha / n_comparisons
    
    print(f"\nComparisons: {n_comparisons}")
    print(f"Bonferroni α: {bonferroni_alpha:.4f}")
    
    test_results = []
    
    for model1, model2 in combinations(model_names, 2):
        y_pred1 = models_data[model1]['y_pred']
        y_pred2 = models_data[model2]['y_pred']
        
        chi2, pvalue, table, odds_ratio, effect_size, interpretation = mcnemar_test(
            y_true_ref, y_pred1, y_pred2
        )
        
        significant = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < bonferroni_alpha else "ns"
        
        print(f"\n{model1} vs {model2}")
        print(f"  Chi² = {chi2:.4f}, p = {pvalue:.4f} {significant}")
        print(f"  Cohen's h = {effect_size:.4f} ({interpretation})")
        print(f"  Only {model1[:20]:20s} correct: {table[0,1]}")
        print(f"  Only {model2[:20]:20s} correct: {table[1,0]}")
        
        test_results.append({
            'Model_1': model1,
            'Model_2': model2,
            'Chi2': chi2,
            'p_value': pvalue,
            'Significant': significant,
            'Cohens_h': effect_size,
            'Effect_Size_Interpretation': interpretation,
            'Only_Model1_Correct': table[0,1],
            'Only_Model2_Correct': table[1,0],
        })
    
    # Bootstrap CIs
    print("\n" + "="*70)
    print("Bootstrap 95% Confidence Intervals")
    print("="*70)
    
    ci_results = []
    for model in model_names:
        y_pred = models_data[model]['y_pred']
        f1 = f1_macro(y_true_ref, y_pred)
        lower, upper = bootstrap_ci(y_true_ref, y_pred, f1_macro, n_iterations=1000)
        
        print(f"\n{model}")
        print(f"  F1: {f1:.4f} [{lower:.4f}, {upper:.4f}]")
        
        ci_results.append({
            'Model': model,
            'F1_Macro': f1,
            'CI_Lower': lower,
            'CI_Upper': upper,
            'CI_Width': upper - lower
        })
    
    # Save results
    os.makedirs("outputs/tables", exist_ok=True)
    
    pd.DataFrame(test_results).to_csv("outputs/tables/mcnemar_tests.csv", index=False)
    pd.DataFrame(ci_results).to_csv("outputs/tables/bootstrap_ci.csv", index=False)
    if roc_results:
        pd.DataFrame(roc_results).to_csv("outputs/tables/roc_auc.csv", index=False)
    
    print("\n" + "="*70)
    print("Statistical Testing Complete")
    print("="*70)
    print("\nFiles saved:")
    print("  • outputs/tables/mcnemar_tests.csv")
    print("  • outputs/tables/bootstrap_ci.csv")
    print("  • outputs/tables/roc_auc.csv")

if __name__ == "__main__":
    main()