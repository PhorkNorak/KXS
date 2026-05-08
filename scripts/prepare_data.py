#!/usr/bin/env python3
"""
Step 1: Data Preparation
=========================
Reads the ORIGINAL dataset.csv (never modified)
Outputs clean, normalized, split files:
  data/processed/full_clean.csv        — cleaned + normalized (all 1184)
  data/processed/train.csv             — 70% (828 samples)
  data/processed/val.csv               — 15% (178 samples)
  data/processed/test.csv              — 15% (178 samples)
  data/processed/data_summary.json     — statistics and metadata

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --seed 42
"""
import os, sys, json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from preprocessing.pipeline import KhmerPreprocessor


def prepare(seed=42):
    print("=" * 60)
    print("  KhmerXScore — Data Preparation")
    print("=" * 60)

    # ===== 1. READ ORIGINAL (never modify this file) =====
    raw_path = "data/dataset.csv"
    print(f"\n1. Reading original: {raw_path}")
    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    print(f"   Raw rows: {len(df)}")
    print(f"   Raw columns: {list(df.columns)}")

    # ===== 2. CLEAN =====
    print("\n2. Cleaning...")

    # Strip column name whitespace
    df.columns = df.columns.str.strip()

    # Strip subject name trailing spaces
    df["Subject"] = df["Subject"].str.strip()

    # Drop rows with NaN answers
    n_before = len(df)
    df = df.dropna(subset=["Answer"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    print(f"   Dropped {n_dropped} rows with NaN answers")
    print(f"   Remaining: {len(df)} rows")

    # ===== 3. NORMALIZE SCORES =====
    print("\n3. Normalizing scores...")
    df["score_ratio"] = (df["Student Score"] / df["Max Score"]).clip(0.0, 1.0)
    df["score_discrete"] = (df["score_ratio"] * 4).round().clip(0, 4).astype(int)

    print(f"   Score ratio: mean={df['score_ratio'].mean():.3f}, std={df['score_ratio'].std():.3f}")
    print(f"   Discrete distribution:")
    for s in range(5):
        n = (df["score_discrete"] == s).sum()
        pct = n / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"     Score {s}: {n:>4} ({pct:>5.1f}%) {bar}")

    # ===== 4. PREPROCESS TEXT =====
    print("\n4. Preprocessing text...")

    # Raw (for transformers — no segmentation)
    pp_raw = KhmerPreprocessor(segment=False)
    df["answer_clean"] = df["Answer"].apply(pp_raw)
    df["reference_clean"] = df["Reference"].apply(pp_raw)
    print("   ✓ Raw cleaned text (answer_clean, reference_clean)")

    # Segmented (for ML/DL baselines + PrahokBART)
    try:
        pp_seg = KhmerPreprocessor(segment=True)
        df["answer_seg"] = df["Answer"].apply(pp_seg)
        df["reference_seg"] = df["Reference"].apply(pp_seg)
        print("   ✓ Segmented text using khmernltk (answer_seg, reference_seg)")
    except Exception:
        df["answer_seg"] = df["answer_clean"]
        df["reference_seg"] = df["reference_clean"]
        print("   ⚠ khmernltk not available — using raw text as fallback for segmented")

    # ===== 5. COMPUTE TEXT STATISTICS =====
    print("\n5. Computing text statistics...")
    df["answer_len_chars"] = df["answer_clean"].str.len()
    df["reference_len_chars"] = df["reference_clean"].str.len()
    df["answer_len_words"] = df["answer_seg"].str.split().str.len()
    df["reference_len_words"] = df["reference_seg"].str.split().str.len()

    print(f"   Answer length:    mean={df['answer_len_chars'].mean():.0f} chars, "
          f"{df['answer_len_words'].mean():.0f} words")
    print(f"   Reference length: mean={df['reference_len_chars'].mean():.0f} chars, "
          f"{df['reference_len_words'].mean():.0f} words")

    # ===== 6. SPLIT =====
    print(f"\n6. Splitting (seed={seed}, stratified by score_discrete)...")
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df["score_discrete"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["score_discrete"]
    )
    print(f"   Train: {len(train_df)} ({len(train_df)/len(df)*100:.0f}%)")
    print(f"   Val:   {len(val_df)} ({len(val_df)/len(df)*100:.0f}%)")
    print(f"   Test:  {len(test_df)} ({len(test_df)/len(df)*100:.0f}%)")

    # Verify stratification
    print(f"\n   Score distribution per split:")
    print(f"   {'Score':>6} {'Train':>7} {'Val':>7} {'Test':>7}")
    print(f"   {'-'*30}")
    for s in range(5):
        tr_n = (train_df["score_discrete"] == s).sum()
        vl_n = (val_df["score_discrete"] == s).sum()
        te_n = (test_df["score_discrete"] == s).sum()
        print(f"   {s:>6} {tr_n:>7} {vl_n:>7} {te_n:>7}")

    # ===== 7. SAVE =====
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    # Columns to save
    cols = [
        "SchoolID", "ClassID", "Subject", "StudentID", "QuestionID",
        "Question", "Reference", "Answer",
        "Student Score", "Max Score", "Year",
        "score_ratio", "score_discrete",
        "answer_clean", "reference_clean",
        "answer_seg", "reference_seg",
        "answer_len_chars", "reference_len_chars",
        "answer_len_words", "reference_len_words",
    ]

    full_path = f"{out_dir}/full_clean.csv"
    train_path = f"{out_dir}/train.csv"
    val_path = f"{out_dir}/val.csv"
    test_path = f"{out_dir}/test.csv"

    df[cols].to_csv(full_path, index=False, encoding="utf-8-sig")
    train_df[cols].to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df[cols].to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df[cols].to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"\n7. Saved:")
    print(f"   {full_path} ({len(df)} rows)")
    print(f"   {train_path} ({len(train_df)} rows)")
    print(f"   {val_path} ({len(val_df)} rows)")
    print(f"   {test_path} ({len(test_df)} rows)")

    # ===== 8. SAVE SUMMARY =====
    summary = {
        "original_file": raw_path,
        "total_samples": len(df),
        "dropped_nan": n_dropped,
        "seed": seed,
        "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "subjects": df["Subject"].value_counts().to_dict(),
        "schools": df["SchoolID"].value_counts().to_dict(),
        "unique_questions": int(df["QuestionID"].nunique()),
        "unique_students": int(df["StudentID"].nunique()),
        "max_score_values": sorted(df["Max Score"].unique().tolist()),
        "score_ratio": {
            "mean": round(float(df["score_ratio"].mean()), 4),
            "std": round(float(df["score_ratio"].std()), 4),
            "min": round(float(df["score_ratio"].min()), 4),
            "max": round(float(df["score_ratio"].max()), 4),
        },
        "score_discrete_distribution": {
            str(k): int(v) for k, v in df["score_discrete"].value_counts().sort_index().items()
        },
        "text_stats": {
            "answer_chars_mean": round(float(df["answer_len_chars"].mean()), 1),
            "answer_words_mean": round(float(df["answer_len_words"].mean()), 1),
            "reference_chars_mean": round(float(df["reference_len_chars"].mean()), 1),
            "reference_words_mean": round(float(df["reference_len_words"].mean()), 1),
        },
        "per_subject_scores": {
            subj: {
                "n": int(sub["score_ratio"].count()),
                "mean": round(float(sub["score_ratio"].mean()), 4),
                "std": round(float(sub["score_ratio"].std()), 4),
            }
            for subj in df["Subject"].unique()
            for sub in [df[df["Subject"] == subj]]
        },
    }

    summary_path = f"{out_dir}/data_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"   {summary_path}")

    print(f"\n{'=' * 60}")
    print(f"  DATA PREPARATION COMPLETE")
    print(f"  Original dataset.csv is UNTOUCHED")
    print(f"  All processed files in data/processed/")
    print(f"{'=' * 60}")

    return df, train_df, val_df, test_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.seed)
