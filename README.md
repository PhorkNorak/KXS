# KhmerXScore — Automated Scoring of Khmer Short-Answer Questions

## Workflow

```
Step 1: Prepare data (run ONCE)
    python scripts/prepare_data.py
    
    Reads:   data/dataset.csv (ORIGINAL — never modified)
    Creates: data/processed/
               ├── full_clean.csv      (1184 cleaned + normalized)
               ├── train.csv           (828 samples, 70%)
               ├── val.csv             (178 samples, 15%)
               ├── test.csv            (178 samples, 15%)
               └── data_summary.json   (statistics)

Step 2: Run experiments
    python scripts/run_all.py                    # All 15 models
    python scripts/run_all.py --tier 1           # Classical baselines
    python scripts/run_all.py --tier 5           # Transformers
    python scripts/run_all.py --tier 6 \
        --openai_key KEY --anthropic_key KEY --google_key KEY
```

## 15 Models across 6 Tiers

| Tier | Models | Text Input |
|------|--------|-----------|
| 1-2 | BoW+Cosine, TF-IDF+Cosine, TF-IDF+SVR | Segmented |
| 2 | FastText+Cosine | Segmented |
| 3 | BiLSTM+Attention (×3 seeds) | Segmented |
| 4 | Sentence-MiniLM+Cosine | Raw |
| 5 | mBERT Simple/Dual, XLM-R, PrahokBART, GTE (×3 seeds) | Raw/Seg |
| 6 | GPT-4 zero/few, Claude zero, Gemini zero | Raw |

## Evaluation Metrics
- Agreement: QWK, Accuracy, Exact Agreement, Adjacent Agreement
- Correlation: Pearson, Spearman
- Error: RMSE, MAE
- Classification: Weighted F1, Precision, Recall
- Visual: Confusion Matrix, Scatter Plot, Distribution Plot, Loss Curves
- Breakdown: Per-subject, Per-score-level
