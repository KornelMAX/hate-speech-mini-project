# scripts/eval_twroberta_hate.py
import os, sys, re, time, math
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ResultsManager, EfficiencyTracker, compute_all_metrics

def normalize_tweet(t):
    # Toggle this on/off as you like. Matches common TweetEval-style normalization.
    t = re.sub(r"@\w+", "@USER", str(t))
    t = re.sub(r"http\S+", "HTTPURL", t)
    return t

def batched_predict_proba(texts, tokenizer, model, device, max_length=128, batch_size=128):
    model.eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p[:, 1])  # class-1 probability (will remap if needed below)
    return np.concatenate(probs)

def main():
    MODEL_NAME_SHORT = "tw_roberta_hate_latest_eval"
    MODEL_NAME_FULL  = "cardiffnlp/twitter-roberta-base-hate-latest"
    MAX_LENGTH = 128
    USE_NORMALIZATION = True

    # Device
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    test_df = pd.read_csv("data/test_split.csv")
    texts = test_df["text"].astype(str).tolist()
    if USE_NORMALIZATION:
        texts = [normalize_tweet(t) for t in texts]
    y_true = test_df["label"].values

    # Load model + tokenizer
    print(f"Loading {MODEL_NAME_FULL} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FULL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_FULL)
    model.to(device)

    # Label mapping safety: ensure index 1 corresponds to HATE
    id2label = getattr(model.config, "id2label", None)
    idx_hate = 1
    if id2label:
        # Try to find which index carries the HATE class
        for k, v in id2label.items():
            if str(v).upper() in ["HATE", "LABEL_1", "LABEL1", "HATE_SPEECH", "HATE-OT"]:
                idx_hate = int(k); break

    # Efficiency
    results_mgr = ResultsManager(MODEL_NAME_SHORT)
    eff = EfficiencyTracker(MODEL_NAME_SHORT)

    # Predict probs
    print("Evaluating zero-shot on test set...")
    start = time.time()
    y_proba_full = batched_predict_proba(texts, tokenizer, model, device, MAX_LENGTH, batch_size=128)
    eval_time = time.time() - start

    # If hate is not index 1 in the model, remap from computed index
    # (We computed p[:,1] above; if idx_hate != 1, recompute with proper slicing)
    if id2label and idx_hate != 1:
        # recompute with correct index
        print("Remapping class probability to the true HATE index found in id2label...")
        def batched_prob_idx(texts, idx):
            out = []
            for i in range(0, len(texts), 128):
                batch = texts[i:i+128]
                enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    logits = model(**enc).logits
                    p = torch.softmax(logits, dim=1).cpu().numpy()
                    out.append(p[:, idx])
            return np.concatenate(out)
        y_proba_full = batched_prob_idx(texts, idx_hate)

    y_pred = (y_proba_full >= 0.5).astype(int)

    # Metrics
    metrics = compute_all_metrics(y_true, y_pred, y_proba_full)
    print("\nZero-shot Test Metrics (twitter-roberta-base-hate-latest):")
    print(f"  Macro F1: {metrics['f1_macro']:.4f} | Hate F1: {metrics['f1_hate']:.4f} | PR-AUC: {metrics.get('pr_auc', np.nan):.4f}")

    # Efficiency: no training, just inference latency/throughput
    def predict_fn(model_obj, samples):
        # classification for latency. You can also time probabilities if you want.
        enc = tokenizer(samples if isinstance(samples, list) else [samples],
                        padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model_obj(**enc).logits
            return torch.argmax(logits, dim=1).cpu().numpy()

    latency = eff.measure_inference_latency(model, predict_fn, texts, num_samples=100)
    throughput = eff.measure_throughput(model, predict_fn, texts, batch_size=1000)
    efficiency = {
        'train_time_seconds': 0.0, 'train_time_minutes': 0.0,  # zero-shot, no training
        **latency, **throughput,
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'parameters_millions': sum(p.numel() for p in model.parameters())/1e6,
        'eval_time_seconds_fullset': float(eval_time)
    }

    # Save
    results_mgr.save_metrics(metrics)
    results_mgr.save_efficiency(efficiency)
    results_mgr.save_predictions(y_true, y_pred, y_proba_full)
    results_mgr.save_training_log({
        'model_name': MODEL_NAME_FULL,
        'mode': 'zero-shot-eval',
        'max_length': MAX_LENGTH,
        'normalization': USE_NORMALIZATION,
        'device': device
    })

    print("\nDone. Results in results/tw_roberta_hate_latest_eval/")

if __name__ == "__main__":
    main()
