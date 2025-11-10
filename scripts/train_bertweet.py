# scripts/train_bertweet.py
import os
import sys
import pandas as pd
import numpy as np
import time
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ResultsManager, EfficiencyTracker, compute_all_metrics


def main():
    # Configuration
    MODEL_NAME_SHORT = "bertweet"
    MODEL_NAME_FULL = "vinai/bertweet-base"
    SEED = 42
    MAX_LENGTH = 128
    TRAIN_BATCH_SIZE = 32
    EVAL_BATCH_SIZE = 64
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 4
    
    # Set seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Initialize managers
    results_mgr = ResultsManager(MODEL_NAME_SHORT)
    efficiency_tracker = EfficiencyTracker(MODEL_NAME_SHORT)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("Using CPU")
    
    print("\n" + "="*70)
    print(f"Training: {MODEL_NAME_FULL}")
    print("="*70 + "\n")
    
    # ========================================================================
    # Load Pre-split Data
    # ========================================================================
    print("[1/10] Loading pre-split data...")
    
    if not os.path.exists('data/train_split.csv'):
        print("ERROR: Split files not found!")
        print("Run 'python scripts/prepare_data.py' first to create splits.")
        return
    
    train_df = pd.read_csv('data/train_split.csv')
    val_df = pd.read_csv('data/val_split.csv')
    test_df = pd.read_csv('data/test_split.csv')
    
    X_train, y_train = train_df['text'], train_df['label']
    X_val, y_val = val_df['text'], val_df['label']
    X_test, y_test = test_df['text'], test_df['label']
    
    print(f"       Train: {len(X_train):,} samples")
    print(f"       Val:   {len(X_val):,} samples")
    print(f"       Test:  {len(X_test):,} samples")
    
    # ========================================================================
    # Load Model and Tokenizer
    # ========================================================================
    print(f"\n[2/10] Loading {MODEL_NAME_FULL}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_FULL, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME_FULL,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"       Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # ========================================================================
    # Tokenization
    # ========================================================================
    print("\n[3/10] Tokenizing datasets...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    train_dataset = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
    val_dataset = Dataset.from_dict({'text': X_val.tolist(), 'label': y_val.tolist()})
    test_dataset = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    print("       Tokenization complete")
    
    # ========================================================================
    # Define Metrics for Trainer
    # ========================================================================
    def compute_metrics_for_trainer(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        probas = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        
        metrics = compute_all_metrics(labels, predictions, probas)
        return {
            'f1_macro': metrics['f1_macro'],
            'f1_hate': metrics['f1_hate'],
        }
    
    # ========================================================================
    # Training Configuration
    # ========================================================================
    print("\n[4/10] Configuring training...")
    
    training_args = TrainingArguments(
        output_dir=f'results/{MODEL_NAME_SHORT}/checkpoints',
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
        
        logging_dir=f'results/{MODEL_NAME_SHORT}/logs',
        logging_steps=100,
        logging_first_step=True,
        report_to="none",
        
        seed=SEED,
        fp16=False,
        dataloader_num_workers=0,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_for_trainer,
    )
    
    print(f"       Epochs: {NUM_EPOCHS}")
    print(f"       Learning rate: {LEARNING_RATE}")
    print(f"       Batch size: {TRAIN_BATCH_SIZE}")
    
    # ========================================================================
    # Training
    # ========================================================================
    print("\n[5/10] Training model...")
    print("-" * 70)
    
    efficiency_tracker.start_training()
    trainer.train()
    train_metrics = efficiency_tracker.end_training()
    
    print("-" * 70)
    print(f"Training completed in {train_metrics['train_time_minutes']:.2f} minutes")
    
    # ========================================================================
    # Test Set Evaluation
    # ========================================================================
    print("\n[6/10] Evaluating on test set...")
    
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
    
    # Compute all metrics
    metrics = compute_all_metrics(y_test.values, y_pred, y_proba)
    
    print("\nTest Set Metrics:")
    print(f"  Macro F1:     {metrics['f1_macro']:.4f}")
    print(f"  Hate F1:      {metrics['f1_hate']:.4f}")
    print(f"  PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"  MCC:          {metrics['mcc']:.4f}")
    
    # ========================================================================
    # Efficiency Measurement
    # ========================================================================
    print("\n[7/10] Measuring efficiency...")
    
    # Helper function for predictions
    def predict_fn(model, texts):
        if isinstance(texts, list):
            texts = texts
        elif hasattr(texts, 'tolist'):
            texts = texts.tolist()
        else:
            texts = [texts]
        
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        with torch.no_grad():
            outputs = model(**encodings)
        return outputs.logits.cpu().numpy()
    
    # Latency
    latency_metrics = efficiency_tracker.measure_inference_latency(
        model, predict_fn, X_test, num_samples=100
    )
    
    # Throughput
    throughput_metrics = efficiency_tracker.measure_throughput(
        model, predict_fn, X_test, batch_size=1000
    )
    
    # Combine efficiency metrics
    efficiency_metrics = {
        **train_metrics,
        **latency_metrics,
        **throughput_metrics,
        'num_parameters': num_params,
        'parameters_millions': num_params / 1e6,
    }
    
    print(f"  Latency (p50):  {latency_metrics['latency_p50_ms']:.2f} ms")
    print(f"  Latency (p95):  {latency_metrics['latency_p95_ms']:.2f} ms")
    print(f"  Throughput:     {throughput_metrics['throughput_samples_per_sec']:.2f} samples/sec")
    
    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n[8/10] Saving results...")
    
    results_mgr.save_metrics(metrics)
    results_mgr.save_efficiency(efficiency_metrics)
    results_mgr.save_predictions(y_test.values, y_pred, y_proba)
    
    # Save training log
    training_log = {
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
        'data_split': '70/15/15',
        'device': device,
    }
    results_mgr.save_training_log(training_log)
    
    # ========================================================================
    # Save Model
    # ========================================================================
    print("\n[9/10] Saving trained model...")
    
    model_save_path = f"models/{MODEL_NAME_SHORT}_finetuned"
    os.makedirs(model_save_path, exist_ok=True)
    
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"       Model saved to: {model_save_path}/")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n[10/10] Training Summary")
    print("="*70)
    print(f"Model:             {MODEL_NAME_FULL}")
    print(f"Macro F1:          {metrics['f1_macro']:.4f}")
    print(f"Hate F1:           {metrics['f1_hate']:.4f}")
    print(f"Training Time:     {train_metrics['train_time_minutes']:.2f} minutes")
    print(f"Parameters:        {num_params/1e6:.1f}M")
    print(f"\nResults saved to:  results/{MODEL_NAME_SHORT}/")
    print("="*70)
    print()


if __name__ == "__main__":
    main()