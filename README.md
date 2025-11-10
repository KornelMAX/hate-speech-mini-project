# Twitter Hate-Speech Mini-Project

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# If dataset not included, download from Kaggle (see below) then:
python scripts/prepare_data.py
# Train models
python scripts/train_svm.py
python scripts/train_bertweet_improved.py
python scripts/train_twroberta_hate_improved.py
# Collect + analyze
python scripts/collect_results.py
python scripts/statistical_tests.py
python scripts/error_analysis.py
python scripts/ablation_class_weights.py
python scripts/generate_figures.py
