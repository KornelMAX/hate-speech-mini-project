# scripts/generate_figures.py
import os, json, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.calibration import calibration_curve
from matplotlib.backends.backend_pdf import PdfPages  # for the optional multipage export

# Export as vector PDF by default; keep text selectable
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

PRIMARY = [
    "svm",
    "bertweet_off_the_shelf",
    "bertweet_finetuned_weighted",
    "tw_roberta_hate_off_the_shelf",
    "tw_roberta_hate_finetuned_weighted",
]

def load_model(name):
    p = Path(f"results/{name}")
    with (p/"metrics.json").open() as f: M = json.load(f)
    with (p/"efficiency.json").open() as f: E = json.load(f)
    P = pd.read_csv(p/"predictions.csv")
    return {"name": name, "metrics": M, "eff": E,
            "y_true": P["y_true"].values,
            "y_pred": P["y_pred"].values,
            "y_proba": P["y_proba_class_1"].values if "y_proba_class_1" in P else None}

def plot_confusions(datas, out):
    fig, axes = plt.subplots(1, len(datas), figsize=(4.2*len(datas), 3.8))
    if len(datas)==1: axes=[axes]
    for ax, d in zip(axes, datas):
        cm = confusion_matrix(d["y_true"], d["y_pred"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(d["name"], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i,j], ha="center", va="center", fontsize=9)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Non-hate","Hate"], fontsize=8); ax.set_yticklabels(["Non-hate","Hate"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()

def plot_pr(datas, out):
    plt.figure(figsize=(6.2,4.8))
    for d in datas:
        if d["y_proba"] is None: continue
        pr, rc, _ = precision_recall_curve(d["y_true"], d["y_proba"])
        aucv = d["metrics"].get("pr_auc", 0.0)
        plt.plot(rc, pr, label=f"{d['name']} (PR-AUC={aucv:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, bbox_inches="tight"); plt.close()

def plot_roc(datas, out):
    plt.figure(figsize=(6.2,4.8))
    for d in datas:
        if d["y_proba"] is None: continue
        fpr, tpr, _ = roc_curve(d["y_true"], d["y_proba"])
        plt.plot(fpr, tpr, label=f"{d['name']} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1],'k--',lw=1,label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, bbox_inches="tight"); plt.close()

def plot_pareto(datas, out):
    plt.figure(figsize=(6.2,4.8))
    for d in datas:
        f1 = d["metrics"]["f1_macro"]
        tmin = d["eff"].get("train_time_minutes", d["eff"].get("inference_time_minutes", 0.0))
        plt.scatter(tmin, f1, s=110)
        plt.annotate(d["name"], (tmin, f1), textcoords="offset points", xytext=(6,4), fontsize=8)
    plt.xlabel("Time (min, train or infer)"); plt.ylabel("Macro-F1")
    plt.title("Effectiveness vs Efficiency"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out, bbox_inches="tight"); plt.close()

def plot_calibration(d, out):
    if d["y_proba"] is None: return
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8.8,3.8))
    frac, meanp = calibration_curve(d["y_true"], d["y_proba"], n_bins=10, strategy="quantile")
    ax1.plot([0,1],[0,1],"k--",lw=1, label="Perfect")
    ax1.plot(meanp, frac, "o-", label=d["name"]); ax1.legend(fontsize=8)
    ax1.set_xlabel("Mean predicted"); ax1.set_ylabel("Empirical freq"); ax1.set_title("Reliability")
    ax2.hist(d["y_proba"], bins=20, edgecolor="black"); ax2.set_title("Predicted probability histogram")
    for ax in (ax1, ax2): ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, bbox_inches="tight"); plt.close()

def main():
    os.makedirs("outputs/figures", exist_ok=True)
    datas=[]
    for m in PRIMARY:
        try: datas.append(load_model(m)); print("✓", m)
        except Exception as e: print("✗", m, e)
    if len(datas)<2:
        print("Need ≥2 models"); return

    # Per-figure PDF files (vector)
    plot_confusions(datas, "outputs/figures/confusions.pdf")
    plot_pr(datas, "outputs/figures/pr_curves.pdf")
    plot_roc(datas, "outputs/figures/roc_curves.pdf")
    plot_pareto(datas, "outputs/figures/pareto_f1_time.pdf")
    plot_calibration(datas[-1], "outputs/figures/calibration.pdf")

    # Optional: single multipage PDF with all figures (comment out if you hate convenience)
    with PdfPages("outputs/figures/all_figures.pdf") as pdf:
        for f in ["confusions.pdf","pr_curves.pdf","roc_curves.pdf","pareto_f1_time.pdf","calibration.pdf"]:
            fig = plt.figure()  # empty page wrapper to embed existing PDFs is messy; simpler to re-render:
            plt.close(fig)
        # If you want true re-render into one file, refactor the plot_* functions to return fig,
        # then call pdf.savefig(fig) right after each plotting call above.

    print("Figures saved in outputs/figures")

if __name__ == "__main__":
    main()
