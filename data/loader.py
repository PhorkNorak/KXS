"""
KhmerSAG Dataset Loader
========================
Reads PRE-PROCESSED split files from data/processed/
Original dataset.csv is never touched by training code.

Usage:
    # First run: python scripts/prepare_data.py
    # Then in training code:
    from data.loader import load_splits, KhmerSAGDataset, KhmerSAGDualDataset
    train_df, val_df, test_df = load_splits()
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class KhmerSAGDataset(Dataset):
    """Simple architecture: [CLS] answer [SEP] reference [SEP]"""
    def __init__(self, df, tokenizer, max_len=256, text_col="answer_clean", ref_col="reference_clean"):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.ml = max_len
        self.text_col = text_col
        self.ref_col = ref_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        enc = self.tok(str(r[self.text_col]), str(r[self.ref_col]),
                       max_length=self.ml, padding="max_length", truncation=True,
                       return_tensors="pt")
        out = {"input_ids": enc["input_ids"].squeeze(0),
               "attention_mask": enc["attention_mask"].squeeze(0),
               "score": torch.tensor(float(r["score_ratio"]), dtype=torch.float)}
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return out


class KhmerSAGDualDataset(Dataset):
    """Dual architecture: encode answer and reference separately"""
    def __init__(self, df, tokenizer, max_len=256, text_col="answer_clean", ref_col="reference_clean"):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.ml = max_len
        self.text_col = text_col
        self.ref_col = ref_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        a_enc = self.tok(str(r[self.text_col]), max_length=self.ml, padding="max_length",
                         truncation=True, return_tensors="pt")
        r_enc = self.tok(str(r[self.ref_col]), max_length=self.ml, padding="max_length",
                         truncation=True, return_tensors="pt")
        out = {"ans_input_ids": a_enc["input_ids"].squeeze(0),
               "ans_attention_mask": a_enc["attention_mask"].squeeze(0),
               "ref_input_ids": r_enc["input_ids"].squeeze(0),
               "ref_attention_mask": r_enc["attention_mask"].squeeze(0),
               "score": torch.tensor(float(r["score_ratio"]), dtype=torch.float)}
        if "token_type_ids" in a_enc:
            out["ans_token_type_ids"] = a_enc["token_type_ids"].squeeze(0)
            out["ref_token_type_ids"] = r_enc["token_type_ids"].squeeze(0)
        return out


def load_splits(processed_dir="data/processed"):
    """
    Load pre-processed train/val/test CSVs.
    Run scripts/prepare_data.py first!
    """
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    # Check if processed files exist
    if not os.path.exists(train_path):
        print("ERROR: Processed data not found!")
        print("Run this first:  python scripts/prepare_data.py")
        print(f"Expected files in: {processed_dir}/")
        raise FileNotFoundError(f"{train_path} not found. Run prepare_data.py first.")

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    val_df = pd.read_csv(val_path, encoding="utf-8-sig")
    test_df = pd.read_csv(test_path, encoding="utf-8-sig")

    print(f"Loaded splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Columns: {list(train_df.columns)}")

    return train_df, val_df, test_df


def load_full(processed_dir="data/processed"):
    """Load the full cleaned dataset."""
    path = os.path.join(processed_dir, "full_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    return pd.read_csv(path, encoding="utf-8-sig")
