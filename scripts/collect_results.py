# scripts/collect_results.py
import os, json
import pandas as pd
from pathlib import Path

PRIMARY = [
    "svm",
    "bertweet_off_the_shelf",
    "bertweet_finetuned_weighted",
    "tw_roberta_hate_off_the_shelf",
    "tw_roberta_hate_finetuned_weighted",
]

def main():
    print("="*70)
    print("Collecting Results (with efficiency ratios)")
    print("="*70)

    results_dir = Path("results")
    models_found = []

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        model_name = d.name
        metrics_file = d / "metrics.json"
        efficiency_file = d / "efficiency.json"
        preds_file = d / "predictions.csv"

        if not (metrics_file.exists() and efficiency_file.exists() and preds_file.exists()):
            print(f"✗ {model_name} (missing files)")
            continue

        with metrics_file.open() as f:
            metrics = json.load(f)
        with efficiency_file.open() as f:
            efficiency = json.load(f)

        models_found.append({"name": model_name, "metrics": metrics, "eff": efficiency})
        print(f"✓ {model_name}")

    if not models_found:
        print("No complete results found.")
        return

    summary_rows = []
    for item in models_found:
        name = item["name"]
        metrics = item["metrics"]
        eff = item["eff"]

        time_min = eff.get("train_time_minutes", eff.get("inference_time_minutes", 0.0))
        params_m = eff.get("parameters_millions", 0.0)
        f1_macro = metrics.get("f1_macro", 0.0)

        summary_rows.append({
            "Model": name,
            "F1_Macro": f1_macro,
            "F1_Hate": metrics.get("f1_hate", 0.0),
            "Precision_Hate": metrics.get("precision_hate", 0.0),
            "Recall_Hate": metrics.get("recall_hate", 0.0),
            "PR_AUC": metrics.get("pr_auc", None),
            "ROC_AUC": metrics.get("roc_auc", None),     
            "MCC": metrics.get("mcc", 0.0),
            "Brier": metrics.get("brier_score", None),   
            "Time_Minutes": time_min,
            "Latency_p50_ms": eff.get("latency_p50_ms", None),
            "Latency_p95_ms": eff.get("latency_p95_ms", None),
            "Throughput": eff.get("throughput_samples_per_sec", None),
            "Parameters_M": params_m,
            "F1_per_Minute": (f1_macro / time_min) if time_min else None,
            "F1_per_10M_Params": (f1_macro / params_m * 10) if params_m else None,
        })

    os.makedirs("outputs/tables", exist_ok=True)
    df = pd.DataFrame(summary_rows)
    df.to_csv("outputs/tables/all_models_summary.csv", index=False)
    print("\nSaved: outputs/tables/all_models_summary.csv")

    print("\nPrimary set presence:")
    for m in PRIMARY:
        print(("  ✓ " if any(x["name"] == m for x in models_found) else "  ✗ ") + m)

if __name__ == "__main__":
    main()
