# scripts/ablation_class_weights.py
import json, pandas as pd, numpy as np
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar

PAIRS = [
    ("bertweet", "bertweet_finetuned_weighted"),
    ("tw_roberta_hate_finetuned", "tw_roberta_hate_finetuned_weighted"),  
]

def load_preds(name):
    p = Path(f"results/{name}/predictions.csv")
    m = Path(f"results/{name}/metrics.json")
    if not p.exists() or not m.exists():
        raise FileNotFoundError(f"Missing results for {name}")
    df = pd.read_csv(p)
    with m.open() as f: metrics = json.load(f)
    return df["y_true"].values, df["y_pred"].values, metrics

def mcnemar_pair(y, y1, y2):
    n01 = ((y1 == y) & (y2 != y)).sum()
    n10 = ((y1 != y) & (y2 == y)).sum()
    table = [[0, n01],[n10, 0]]
    res = mcnemar(table, exact=False, correction=True)
    return float(res.statistic), float(res.pvalue), int(n01), int(n10)

def row_for(pair):
    base, imp = pair
    y, yb, Mb = load_preds(base)
    _, yi, Mi = load_preds(imp)
    def get(k): return Mi.get(k, np.nan) - Mb.get(k, np.nan)
    chi2, p, only_base, only_impr = mcnemar_pair(y, yb, yi)
    return {
        "Pair": f"{base} -> {imp}",
        "ΔF1_Macro": get("f1_macro"),
        "ΔF1_Hate": get("f1_hate"),
        "ΔPR_AUC": get("pr_auc"),
        "ΔMCC": get("mcc"),
        "McNemar_Chi2": chi2,
        "McNemar_p": p,
        "Only_base_correct": only_base,
        "Only_improved_correct": only_impr,
    }

def main():
    rows = [row_for(p) for p in PAIRS]
    out = Path("outputs/tables"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"ablation_class_weights.csv", index=False)
    print(f"Saved: {out/'ablation_class_weights.csv'}")

if __name__ == "__main__":
    main()
