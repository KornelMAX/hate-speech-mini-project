# scripts/prepare_data.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

def main():
    SEED = 42
    np.random.seed(SEED)
    
    print("="*70)
    print("Data Preparation for Multi-Model Comparison")
    print("="*70)
    
    # Load data
    print("\n[1/4] Loading original data...")
    df = pd.read_csv('data/train.csv')
    text_col = 'tweet' if 'tweet' in df.columns else 'text'
    
    print(f"      Total samples: {len(df):,}")
    print(f"      Columns: {df.columns.tolist()}")
    
    # Class distribution
    class_counts = df['label'].value_counts()
    print(f"\n[2/4] Class distribution:")
    print(f"      Class 0 (non-hate): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.1f}%)")
    print(f"      Class 1 (hate):     {class_counts[1]:,} ({class_counts[1]/len(df)*100:.1f}%)")
    print(f"      Imbalance ratio: {class_counts[0]/class_counts[1]:.2f}:1")
    
    # Create splits
    print("\n[3/4] Creating stratified splits (70/15/15)...")
    X = df[text_col]
    y = df['label']
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    
    # Second split: 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    
    # Create dataframes with reset index
    train_df = pd.DataFrame({'text': X_train.values, 'label': y_train.values})
    val_df = pd.DataFrame({'text': X_val.values, 'label': y_val.values})
    test_df = pd.DataFrame({'text': X_test.values, 'label': y_test.values})
    
    print(f"      Train: {len(train_df):,} samples")
    print(f"      Val:   {len(val_df):,} samples")
    print(f"      Test:  {len(test_df):,} samples")
    
    # Save splits
    print("\n[4/4] Saving splits to CSV files...")
    train_df.to_csv('data/train_split.csv', index=False)
    val_df.to_csv('data/val_split.csv', index=False)
    test_df.to_csv('data/test_split.csv', index=False)
    
    # Calculate and save metadata
    metadata = {
        'seed': SEED,
        'split_ratio': '70/15/15',
        'total_samples': len(df),
        'train': {
            'total': len(train_df),
            'class_0': int((train_df['label']==0).sum()),
            'class_1': int((train_df['label']==1).sum()),
        },
        'val': {
            'total': len(val_df),
            'class_0': int((val_df['label']==0).sum()),
            'class_1': int((val_df['label']==1).sum()),
        },
        'test': {
            'total': len(test_df),
            'class_0': int((test_df['label']==0).sum()),
            'class_1': int((test_df['label']==1).sum()),
        }
    }
    
    with open('data/split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("Data Preparation Complete")
    print("="*70)
    print("\nFiles created:")
    print("  • data/train_split.csv")
    print("  • data/val_split.csv")
    print("  • data/test_split.csv")
    print("  • data/split_metadata.json")
    
    print("\nSplit Details:")
    for split_name, split_data in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        c0 = (split_data['label']==0).sum()
        c1 = (split_data['label']==1).sum()
        print(f"  {split_name:5s}: {len(split_data):5,} samples (Class 0: {c0:5,}, Class 1: {c1:4,})")
    
    print("\n  IMPORTANT:")
    print("  All model training scripts MUST load these split files.")
    print("  This ensures fair comparison across models.")
    print("="*70)

if __name__ == "__main__":
    main()