import os, sys, re, time
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# project utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ResultsManager, EfficiencyTracker, compute_all_metrics

def normalize_tweet(t: str) -> str:
    t = re.sub(r"@\w+", "@USER", str(t))
    t = re.sub(r"http\S+", "HTTPURL", t)
    return t

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    # accept **kwargs to be compatible with newer Trainer (num_items_in_batch, etc.)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            if self.class_weights.device != logits.device:
                self.class_weights = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def main():
    MODEL_NAME_SHORT = "bertweet_improved"
    MODEL_NAME_FULL  = "vinai/bertweet-base"
    SEED = 42
    MAX_LENGTH = 128
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE  = 64
    LEARNING_RATE    = 3e-5   # standard BERT fine-tune band
    NUM_EPOCHS       = 4

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    results_mgr = ResultsManager(MODEL_NAME_SHORT)
    tracker = EfficiencyTracker(MODEL_NAME_SHORT)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n== Fine-tuning {MODEL_NAME_FULL} (improved) ==\nDevice: {device}")

    # Data
    train_df = pd.read_csv("data/train_split.csv")
    val_df   = pd.read_csv("data/val_split.csv")
    test_df  = pd.read_csv("data/test_split.csv")

    for df in [train_df, val_df, test_df]:
        df["text"] = df["text"].astype(str).apply(normalize_tweet)

    # class weights from train labels
    counts = train_df["label"].value_counts().sort_index()
    n = counts.sum()
    n_classes = len(counts)
    class_weights = (n / (n_classes * counts)).values.astype(np.float32)
    print(f"Class weights: {class_weights}")

    # BERTweet tokenizer with tweet normalization
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FULL, use_fast=False, normalization=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_FULL, num_labels=2)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params/1e6:.1f}M")

    def tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    dtrain = Dataset.from_pandas(train_df[["text","label"]]).map(tok, batched=True)
    dval   = Dataset.from_pandas(val_df[["text","label"]]).map(tok, batched=True)
    dtest  = Dataset.from_pandas(test_df[["text","label"]]).map(tok, batched=True)

    dtrain = dtrain.rename_column("label","labels")
    dval   = dval.rename_column("label","labels")
    dtest  = dtest.rename_column("label","labels")

    def compute_metrics_for_trainer(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probas = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        m = compute_all_metrics(labels, preds, probas)
        return {"f1_macro": m["f1_macro"], "f1_hate": m["f1_hate"]}

    args = TrainingArguments(
        output_dir=f"results/{MODEL_NAME_SHORT}/checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        eval_strategy="epoch",          # new name; evaluation_strategy is deprecated
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=f"results/{MODEL_NAME_SHORT}/logs",
        logging_steps=100,
        report_to="none",
        seed=SEED,
        fp16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,    # silence MPS warning
        max_grad_norm=1.0,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=dtrain,
        eval_dataset=dval,
        compute_metrics=compute_metrics_for_trainer,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Fine-tuning...")
    tracker.start_training()
    trainer.train()
    train_times = tracker.end_training()
    print(f"Fine-tune time: {train_times['train_time_minutes']:.2f} minutes")

    print("Evaluating on test...")
    preds = trainer.predict(dtest)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_proba = torch.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()
    y_true = test_df["label"].values
    metrics = compute_all_metrics(y_true, y_pred, y_proba)

    # Efficiency: latency/throughput
    def predict_fn(model_obj, samples):
        enc = tokenizer(samples if isinstance(samples, list) else [samples],
                        padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model_obj(**enc).logits
            return torch.argmax(logits, dim=1).cpu().numpy()

    latency = tracker.measure_inference_latency(model, predict_fn, test_df["text"].tolist(), num_samples=100)
    throughput = tracker.measure_throughput(model, predict_fn, test_df["text"].tolist(), batch_size=1000)
    eff = {**train_times, **latency, **throughput,
           "num_parameters": num_params, "parameters_millions": num_params/1e6}

    # Save
    results_mgr.save_metrics(metrics)
    results_mgr.save_efficiency(eff)
    results_mgr.save_predictions(y_true, y_pred, y_proba)
    results_mgr.save_training_log({
        "model_name": MODEL_NAME_FULL,
        "seed": SEED,
        "hyperparameters": {
            "max_length": MAX_LENGTH,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "class_weights": class_weights.tolist(),
        },
        "mode": "fine-tune",
        "device": device
    })

    save_dir = f"models/{MODEL_NAME_SHORT}_finetuned"
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved fine-tuned model to {save_dir}")

if __name__ == "__main__":
    main()
