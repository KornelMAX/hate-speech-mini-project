import os, json, re
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, brier_score_loss

PRIMARY = [
    "svm",
    "bertweet_off_the_shelf",
    "bertweet_finetuned_weighted",
    "tw_roberta_hate_off_the_shelf",
    "tw_roberta_hate_finetuned_weighted",
]

def load_model_data(model):
    base = Path(f"results/{model}")
    pred = base / "predictions.csv"
    met  = base / "metrics.json"
    if not (pred.exists() and met.exists()): return None
    df = pd.read_csv(pred)
    with met.open() as f: metrics = json.load(f)
    return {
        "name": model,
        "y_true": df["y_true"].values,
        "y_pred": df["y_pred"].values,
        "y_proba": df["y_proba_class_1"].values if "y_proba_class_1" in df else None,
        "metrics": metrics,
        "pred_df": df,
    }

def per_model_summary(name, y_true, y_pred, y_proba=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    prec0 = tn/(tn+fn) if (tn+fn)>0 else 0.0
    rec0  = tn/(tn+fp) if (tn+fp)>0 else 0.0
    f10   = 2*prec0*rec0/(prec0+rec0) if (prec0+rec0)>0 else 0.0
    prec1 = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec1  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f11   = 2*prec1*rec1/(prec1+rec1) if (prec1+rec1)>0 else 0.0
    out = {
        "model": name,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fp_rate": float(fp/(tn+fp)) if (tn+fp)>0 else 0.0,
        "fn_rate": float(fn/(tp+fn)) if (tp+fn)>0 else 0.0,
        "precision_nonhate": float(prec0), "recall_nonhate": float(rec0), "f1_nonhate": float(f10),
        "precision_hate": float(prec1),   "recall_hate": float(rec1),   "f1_hate": float(f11),
    }
    if y_proba is not None:
        out["brier_score"] = float(brier_score_loss(y_true, y_proba))  # proper calibration score
    return out

def add_slices(df_text, y_true, y_pred, y_proba):
    # minimal, informative slices for Twitter
    text = df_text.fillna("")
    lens = text.str.len()
    has_at   = text.str.contains(r"@\w")
    has_hash = text.str.contains(r"#\w")
    has_url  = text.str.contains(r"https?://")
    bins = pd.cut(lens, bins=[0,50,100,200,10000], labels=["<=50","51-100","101-200",">200"])
    slices = pd.DataFrame({
        "len_bin": bins,
        "has_at": has_at.astype(int),
        "has_hash": has_hash.astype(int),
        "has_url": has_url.astype(int),
        "y_true": y_true, "y_pred": y_pred
    })
    # simple per-slice error rates
    agg = (slices.assign(err=(slices.y_true!=slices.y_pred).astype(int))
           .groupby(["len_bin", "has_at", "has_hash", "has_url"], observed=True)["err"].mean()
           .reset_index().rename(columns={"err":"error_rate"}))
    return agg

def dump_hard_cases(model_name, test_texts, y_true, y_pred, y_proba, out_dir):
    df = pd.DataFrame({
        "text": test_texts, "y_true": y_true, "y_pred": y_pred,
        "proba": y_proba if y_proba is not None else np.nan
    })
    fp = df[(df.y_true==0) & (df.y_pred==1)]
    fn = df[(df.y_true==1) & (df.y_pred==0)]
    # rank by confidence if available
    if "proba" in fp: fp = fp.assign(conf=lambda d: d["proba"]).sort_values("conf", ascending=False)
    if "proba" in fn: fn = fn.assign(conf=lambda d: 1-d["proba"]).sort_values("conf", ascending=False)
    fp.head(50).to_csv(out_dir / f"{model_name}_top50_fp.csv", index=False)  # moderation angle
    fn.head(50).to_csv(out_dir / f"{model_name}_top50_fn.csv", index=False)  # recall angle

def main():
    print("="*70); print("Comprehensive Error Analysis"); print("="*70)
    test = pd.read_csv("data/test_split.csv")
    test_texts = test["text"]

    models_data = {}
    for m in PRIMARY:
        d = load_model_data(m)
        if d: models_data[m] = d; print("✓ Loaded", m)
        else: print("✗ Missing", m)
    if len(models_data) < 2:
        print("Need at least two models"); return

    # per-model summaries
    summaries = []
    outdir = Path("outputs/tables"); outdir.mkdir(parents=True, exist_ok=True)
    for name, d in models_data.items():
        s = per_model_summary(name, d["y_true"], d["y_pred"], d["y_proba"])
        summaries.append(s)
        dump_hard_cases(name, test_texts.values, d["y_true"], d["y_pred"], d["y_proba"], outdir)

        # slice analysis
        slice_tbl = add_slices(test_texts, d["y_true"], d["y_pred"], d["y_proba"])
        slice_tbl.to_csv(outdir / f"{name}_slice_errors.csv", index=False)

    pd.DataFrame(summaries).to_csv(outdir / "error_analysis.csv", index=False)

    # cross-model agreement / “hard for everyone”
    names = list(models_data.keys())
    agree = None; all_wrong = None; y_ref = None
    for name in names:
        d = models_data[name]
        y_ref = d["y_true"] if y_ref is None else y_ref
        corr = (d["y_true"] == d["y_pred"])
        agree = corr if agree is None else (agree & corr)
        all_wrong = (~corr) if all_wrong is None else (all_wrong & ~corr)
    pd.DataFrame({
        "all_models_correct": [int(agree.sum())],
        "all_models_wrong":   [int(all_wrong.sum())],
    }).to_csv(outdir / "cross_model_agreement.csv", index=False)

    print("Saved tables in outputs/tables")
    print("="*70)

if __name__ == "__main__":
    main()
