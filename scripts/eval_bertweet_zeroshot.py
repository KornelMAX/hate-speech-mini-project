# scripts/eval_bertweet_zeroshot.py
"""
Evaluate BERTweet zero-shot (no fine-tuning) to show power of fine-tuning
Uses sentiment analysis as proxy for hate detection
"""
import os
import sys
import pandas as pd
import numpy as np
import time
from transformers import pipeline
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ResultsManager, EfficiencyTracker, compute_all_metrics


def main():
    MODEL_NAME = "bertweet_zeroshot"
    MODEL_PATH = "finiteautomata/bertweet-base-sentiment-analysis"  # Pretrained sentiment model
    
    results_mgr = ResultsManager(MODEL_NAME)
    efficiency_tracker = EfficiencyTracker(MODEL_NAME)
    
    print("="*70)
    print("BERTweet Zero-Shot Evaluation (No Fine-tuning)")
    print("="*70)
    print("\nUsing pretrained sentiment model as proxy for hate detection")
    print("Negative sentiment → Hate | Positive/Neutral → Non-hate\n")
    
    # Load test data
    print("[1/6] Loading test data...")
    test_df = pd.read_csv('data/test_split.csv')
    X_test = test_df['text'].tolist()
    y_test = test_df['label'].values
    
    print(f"      Test samples: {len(X_test)}")
    
    # Load pipeline
    print("\n[2/6] Loading pretrained BERTweet (sentiment)...")
    device = 0 if torch.cuda.is_available() or torch.backends.mps.is_available() else -1
    
    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,
        device=device,
        truncation=True,
        max_length=128
    )
    
    print(f"      Device: {'GPU' if device >= 0 else 'CPU'}")
    
    # Inference with timing
    print("\n[3/6] Running zero-shot inference...")
    efficiency_tracker.start_training()  # Using this to time inference
    
    predictions = []
    for text in X_test:
        try:
            result = classifier(text)[0]
            # Map sentiment to hate: NEG → hate (1), POS/NEU → non-hate (0)
            pred = 1 if result['label'] == 'NEG' else 0
            score = result['score'] if result['label'] == 'NEG' else (1 - result['score'])
            predictions.append({'pred': pred, 'score': score})
        except:
            # If processing fails, predict non-hate
            predictions.append({'pred': 0, 'score': 0.5})
    
    inference_time = efficiency_tracker.end_training()
    
    y_pred = np.array([p['pred'] for p in predictions])
    y_proba = np.array([p['score'] for p in predictions])
    
    print(f"      Completed in {inference_time['train_time_minutes']:.2f} minutes")
    
    # Compute metrics
    print("\n[4/6] Computing metrics...")
    metrics = compute_all_metrics(y_test, y_pred, y_proba)
    
    print(f"\n      F1 Macro:  {metrics['f1_macro']:.4f}")
    print(f"      F1 Hate:   {metrics['f1_hate']:.4f}")
    print(f"      PR-AUC:    {metrics['pr_auc']:.4f}")
    
    # Efficiency metrics
    print("\n[5/6] Computing efficiency metrics...")
    
    def predict_fn(model, texts):
        if isinstance(texts, str):
            texts = [texts]
        return [classifier(t)[0] for t in texts]
    
    latency_metrics = efficiency_tracker.measure_inference_latency(
        classifier, predict_fn, X_test, num_samples=100
    )
    
    throughput_metrics = efficiency_tracker.measure_throughput(
        classifier, predict_fn, X_test, batch_size=1000
    )
    
    efficiency_metrics = {
        'inference_time_minutes': inference_time['train_time_minutes'],
        **latency_metrics,
        **throughput_metrics,
        'num_parameters': 135e6,  # BERTweet-base
        'parameters_millions': 135.0,
    }
    
    print(f"      Latency (p50): {latency_metrics['latency_p50_ms']:.2f} ms")
    print(f"      Throughput:    {throughput_metrics['throughput_samples_per_sec']:.2f} samples/sec")
    
    # Save results
    print("\n[6/6] Saving results...")
    results_mgr.save_metrics(metrics)
    results_mgr.save_efficiency(efficiency_metrics)
    results_mgr.save_predictions(y_test, y_pred, y_proba)
    
    training_log = {
        'model_name': 'BERTweet Zero-Shot',
        'model_path': MODEL_PATH,
        'approach': 'Sentiment analysis as proxy (NEG → hate)',
        'fine_tuning': False,
    }
    results_mgr.save_training_log(training_log)
    
    print("\n" + "="*70)
    print("Zero-Shot Evaluation Complete")
    print("="*70)
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"F1 Hate:  {metrics['f1_hate']:.4f}")
    print(f"\nCompare to fine-tuned BERTweet to see fine-tuning impact!")
    print("="*70)


if __name__ == "__main__":
    main()