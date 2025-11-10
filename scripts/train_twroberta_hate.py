# scripts/train_twroberta_hate.py
import os, sys, re, time
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ResultsManager, EfficiencyTracker, compute_all_metrics

def normalize_tweet(t):
    t = re.sub(r"@\w+", "@USER", str(t))
    t = re.sub(r"http\S+", "HTTPURL", t)
    return t

def main():
    MODEL_NAME_SHORT = "tw_roberta_hate_latest_ft"
    MODEL_NAME_FULL  = "cardiffnlp/twitter-roberta-base-hate-latest"
    SEED = 42
    MAX_LENGTH = 128
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 64
    LEARNING_RATE = 1e-5   # smaller, since it’s already fine-tuned
    NUM_EPOCHS = 2         # short adaptation to avoid overfitting

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    results_mgr = ResultsManager(MODEL_NAME_SHORT)
    efficiency = EfficiencyTracker(MODEL_NAME_SHORT)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n== Fine-tuning {MODEL_NAME_FULL} on your splits ==\nDevice: {device}")

    # Load splits
    train_df = pd.read_csv('data/train_split.csv')
    val_df   = pd.read_csv('data/val_split.csv')
    test_df  = pd.read_csv('data/test_split.csv')

    # Optional normalization for tweets
    train_df['text'] = train_df['text'].astype(str).apply(normalize_tweet)
    val_df['text']   = val_df['text'].astype(str).apply(normalize_tweet)
    test_df['text']  = test_df['text'].astype(str).apply(normalize_tweet)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FULL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_FULL, num_labels=2)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params/1e6:.1f}M")

    def tok(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=MAX_LENGTH)

    dtrain = Dataset.from_pandas(train_df[['text','label']]).map(tok, batched=True)
    dval   = Dataset.from_pandas(val_df[['text','label']]).map(tok, batched=True)
    dtest  = Dataset.from_pandas(test_df[['text','label']]).map(tok, batched=True)

    def compute_metrics_for_trainer(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probas = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        m = compute_all_metrics(labels, preds, probas)
        return {'f1_macro': m['f1_macro'], 'f1_hate': m['f1_hate']}

    args = TrainingArguments(
        output_dir=f"results/{MODEL_NAME_SHORT}/checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=f"results/{MODEL_NAME_SHORT}/logs",
        logging_steps=100,
        report_to="none",
        seed=SEED,
        fp16=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dtrain,
        eval_dataset=dval,
        compute_metrics=compute_metrics_for_trainer,
    )

    # Fine-tune
    print("Fine-tuning...")
    efficiency.start_training()
    trainer.train()
    train_times = efficiency.end_training()
    print(f"Fine-tune time: {train_times['train_time_minutes']:.2f} minutes")

    # Test
    print("Evaluating on test...")
    preds = trainer.predict(dtest)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_proba = torch.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()
    y_true = test_df['label'].values
    metrics = compute_all_metrics(y_true, y_pred, y_proba)

    print("\nTest Metrics (fine-tuned twitter-roberta-base-hate-latest):")
    print(f"  Macro F1: {metrics['f1_macro']:.4f} | Hate F1: {metrics['f1_hate']:.4f} | PR-AUC: {metrics.get('pr_auc', np.nan):.4f}")

    # Efficiency: latency/throughput
    def predict_fn(model_obj, samples):
        enc = tokenizer(samples if isinstance(samples, list) else [samples],
                        padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model_obj(**enc).logits
            return torch.argmax(logits, dim=1).cpu().numpy()

    latency = efficiency.measure_inference_latency(model, predict_fn, test_df['text'].tolist(), num_samples=100)
    throughput = efficiency.measure_throughput(model, predict_fn, test_df['text'].tolist(), batch_size=1000)
    eff = {**train_times, **latency, **throughput,
           'num_parameters': num_params, 'parameters_millions': num_params/1e6}

    # Save
    results_mgr.save_metrics(metrics)
    results_mgr.save_efficiency(eff)
    results_mgr.save_predictions(y_true, y_pred, y_proba)
    results_mgr.save_training_log({
        'model_name': MODEL_NAME_FULL,
        'seed': SEED,
        'hyperparameters': {
            'max_length': MAX_LENGTH,
            'train_batch_size': TRAIN_BATCH_SIZE,
            'eval_batch_size': EVAL_BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'weight_decay': 0.01,
            'warmup_steps': 100,
        },
        'mode': 'fine-tune',
        'device': device
    })

    # Save model
    save_dir = f"models/{MODEL_NAME_SHORT}_finetuned"
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved fine-tuned model to {save_dir}")

if __name__ == "__main__":
    main()
