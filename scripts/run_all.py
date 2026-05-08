#!/usr/bin/env python3
"""
KhmerXScore — Run All 15 Models
=================================
Usage:
    python scripts/run_all.py                        # All tiers
    python scripts/run_all.py --tier 1               # Classical only
    python scripts/run_all.py --tier 5               # Transformers only
    python scripts/run_all.py --tier 6 --openai_key KEY --anthropic_key KEY --google_key KEY  # NOT YET — requires API keys
"""
import os, sys, json, argparse, time, random
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from data.loader import load_splits, KhmerSAGDataset, KhmerSAGDualDataset
from evaluation.metrics import compute_metrics, print_metrics, aggregate_seeds, benchmark_latency


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======== TIER 1-2 ========
def run_tier1(train_df, test_df):
    from models.tier1_classical import BoWCosine, TFIDFCosine, TFIDFSVR
    results = {}
    tr_a = train_df["answer_seg"].tolist()
    tr_r = train_df["reference_seg"].tolist()
    tr_s = train_df["score_ratio"].values
    te_a = test_df["answer_seg"].tolist()
    te_r = test_df["reference_seg"].tolist()
    te_s = test_df["score_ratio"].values

    for Cls in [BoWCosine, TFIDFCosine, TFIDFSVR]:
        m = Cls()
        print(f"\n--- {m.name} ---")
        t0 = time.perf_counter()
        m.fit(tr_a, tr_r, tr_s)
        train_time = round(time.perf_counter() - t0, 3)
        preds = m.predict(te_a, te_r)
        metrics = compute_metrics(te_s, preds)
        timing = benchmark_latency(m.predict, te_a, te_r)
        metrics["train_time_sec"] = train_time
        metrics.update(timing)
        print_metrics(metrics, m.name)
        results[m.name] = metrics
    return results


# ======== TIER 2 ========
def run_tier2(train_df, test_df):
    from models.tier2_fasttext import FastTextCosine
    m = FastTextCosine()
    print(f"\n--- {m.name} ---")
    t0 = time.perf_counter()
    m.fit()
    train_time = round(time.perf_counter() - t0, 3)
    if m.model is None:
        print("FastText not available, skipping")
        return {}
    te_a = test_df["answer_seg"].tolist()
    te_r = test_df["reference_seg"].tolist()
    preds = m.predict(te_a, te_r)
    metrics = compute_metrics(test_df["score_ratio"].values, preds)
    timing = benchmark_latency(m.predict, te_a, te_r)
    metrics["train_time_sec"] = train_time
    metrics.update(timing)
    print_metrics(metrics, m.name)
    return {m.name: metrics}


# ======== TIER 3 ========
def run_tier3(train_df, val_df, test_df, device, seeds):
    from models.tier3_bilstm import train_bilstm
    all_results = []
    for seed in seeds:
        print(f"\n--- BiLSTM+Attention (seed={seed}) ---")
        seed_everything(seed)
        preds, labels, infer_time_sec = train_bilstm(train_df, val_df, test_df, device, seed)
        m = compute_metrics(labels, preds)
        m["inference_time_sec"] = infer_time_sec
        m["throughput_per_sec"] = round(len(labels) / infer_time_sec, 1)
        m["latency_mean_ms"] = round(infer_time_sec / len(labels) * 1000, 3)
        print_metrics(m, f"BiLSTM+Attn seed={seed}")
        all_results.append(m)
    return {"BiLSTM+Attention": aggregate_seeds(all_results)}


# ======== TIER 4 ========
def run_tier4(test_df):
    from models.tier4_sentence import SentenceMiniLMCosine
    m = SentenceMiniLMCosine()
    print(f"\n--- {m.name} ---")
    t0 = time.perf_counter()
    m.fit()
    train_time = round(time.perf_counter() - t0, 3)
    te_a = test_df["answer_clean"].tolist()
    te_r = test_df["reference_clean"].tolist()
    preds = m.predict(te_a, te_r)
    metrics = compute_metrics(test_df["score_ratio"].values, preds)
    timing = benchmark_latency(m.predict, te_a, te_r)
    metrics["train_time_sec"] = train_time
    metrics.update(timing)
    print_metrics(metrics, m.name)
    return {m.name: metrics}


# ======== TIER 5 ========
def run_tier5(train_df, val_df, test_df, device, seeds):
    from models.tier5_transformers import TransformerSimple, TransformerDual, train_transformer, predict_transformer
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    configs = [
        ("bert-base-multilingual-uncased", "simple", 6, False),
        ("bert-base-multilingual-uncased", "dual", 6, False),
        ("xlm-roberta-base", "simple", 6, False),
        # ("nict-astrec-att/prahokbart_base", "simple", 6, True),  # Uncomment when available
        ("Alibaba-NLP/gte-multilingual-base", "simple", 6, False),
    ]

    results = {}
    for model_name, arch, freeze, use_seg in configs:
        short = model_name.split("/")[-1]
        key = f"{arch.title()}({short})"
        print(f"\n{'='*50}\n  {key}\n{'='*50}")

        seed_results = []
        for seed in seeds:
            seed_everything(seed)
            print(f"\n  --- seed={seed} ---")

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

                # Choose text column
                ans_col = "answer_seg" if use_seg else "answer_clean"
                ref_col = "reference_seg" if use_seg else "reference_clean"

                # Temp df with correct columns
                tr = train_df.copy()
                tr["answer_clean"] = tr[ans_col]
                tr["reference_clean"] = tr[ref_col]
                vl = val_df.copy()
                vl["answer_clean"] = vl[ans_col]
                vl["reference_clean"] = vl[ref_col]
                te = test_df.copy()
                te["answer_clean"] = te[ans_col]
                te["reference_clean"] = te[ref_col]

                if arch == "simple":
                    model = TransformerSimple(model_name, freeze_layers=freeze).to(device)
                    tr_ld = DataLoader(KhmerSAGDataset(tr, tokenizer), batch_size=16, shuffle=True)
                    vl_ld = DataLoader(KhmerSAGDataset(vl, tokenizer), batch_size=32)
                    te_ld = DataLoader(KhmerSAGDataset(te, tokenizer), batch_size=32)
                else:
                    model = TransformerDual(model_name, freeze_layers=freeze).to(device)
                    tr_ld = DataLoader(KhmerSAGDualDataset(tr, tokenizer), batch_size=16, shuffle=True)
                    vl_ld = DataLoader(KhmerSAGDualDataset(vl, tokenizer), batch_size=32)
                    te_ld = DataLoader(KhmerSAGDualDataset(te, tokenizer), batch_size=32)

                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"    Trainable params: {trainable:,}")

                model = train_transformer(model, tr_ld, vl_ld, device)
                t_infer = time.perf_counter()
                preds, labels = predict_transformer(model, te_ld, device)
                infer_time_sec = round(time.perf_counter() - t_infer, 4)
                m = compute_metrics(labels, preds)
                m["inference_time_sec"] = infer_time_sec
                m["throughput_per_sec"] = round(len(labels) / infer_time_sec, 1)
                m["latency_mean_ms"] = round(infer_time_sec / len(labels) * 1000, 3)
                print_metrics(m, f"{key} seed={seed}")
                seed_results.append(m)

                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        if seed_results:
            results[key] = aggregate_seeds(seed_results)

    return results


# ======== TIER 6 ========
def run_tier6(test_df, train_df, openai_key=None, anthropic_key=None, google_key=None):
    from models.tier6_llm import GPT4Scorer, ClaudeScorer, GeminiScorer
    results = {}
    te_a = test_df["answer_clean"].tolist()
    te_r = test_df["reference_clean"].tolist()
    te_s = test_df["score_ratio"].values

    # Few-shot examples
    examples = []
    for sl in [0, 2, 4]:
        sub = train_df[train_df["score_discrete"] == sl]
        if len(sub) > 0:
            row = sub.iloc[0]
            examples.append({"ref": row["reference_clean"][:200], "ans": row["answer_clean"][:200],
                             "score": round(float(row["score_ratio"]), 2)})

    scorers = []
    if openai_key:
        scorers.append(GPT4Scorer(openai_key, mode="zero"))
        scorers.append(GPT4Scorer(openai_key, mode="few", examples=examples))
    if anthropic_key:
        scorers.append(ClaudeScorer(anthropic_key))
    if google_key:
        scorers.append(GeminiScorer(google_key))

    for scorer in scorers:
        print(f"\n--- {scorer.name} ---")
        try:
            latencies_ms = []
            preds = []
            for a, r in zip(te_a, te_r):
                t0 = time.perf_counter()
                p = scorer.predict([a], [r])
                latencies_ms.append((time.perf_counter() - t0) * 1000)
                preds.append(p[0])
            import numpy as _np
            lat = _np.array(latencies_ms)
            metrics = compute_metrics(te_s, _np.array(preds))
            metrics["latency_mean_ms"]    = round(float(lat.mean()), 3)
            metrics["latency_p50_ms"]     = round(float(_np.percentile(lat, 50)), 3)
            metrics["latency_p95_ms"]     = round(float(_np.percentile(lat, 95)), 3)
            metrics["latency_p99_ms"]     = round(float(_np.percentile(lat, 99)), 3)
            metrics["throughput_per_sec"] = round(float(len(te_a) / (lat.sum() / 1000)), 1)
            print_metrics(metrics, scorer.name)
            results[scorer.name] = metrics
        except Exception as e:
            print(f"  ERROR: {e}")
    return results


# ======== MAIN ========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    # Data is always loaded from data/processed/ (run prepare_data.py first)
    parser.add_argument("--openai_key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--anthropic_key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--google_key", default=os.environ.get("GOOGLE_API_KEY"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\n" + "=" * 60 + "\n  LOADING DATA\n" + "=" * 60)
    train_df, val_df, test_df = load_splits("data/processed")

    all_results = {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.tier is None or args.tier == 1:
        print("\n" + "=" * 60 + "\n  TIER 1-2: CLASSICAL BASELINES\n" + "=" * 60)
        all_results.update(run_tier1(train_df, test_df))

    if args.tier is None or args.tier == 2:
        print("\n" + "=" * 60 + "\n  TIER 2: FASTTEXT\n" + "=" * 60)
        all_results.update(run_tier2(train_df, test_df))

    if args.tier is None or args.tier == 3:
        print("\n" + "=" * 60 + "\n  TIER 3: BiLSTM+ATTENTION\n" + "=" * 60)
        all_results.update(run_tier3(train_df, val_df, test_df, device, args.seeds))

    if args.tier is None or args.tier == 4:
        print("\n" + "=" * 60 + "\n  TIER 4: SENTENCE EMBEDDINGS\n" + "=" * 60)
        all_results.update(run_tier4(test_df))

    if args.tier is None or args.tier == 5:
        print("\n" + "=" * 60 + "\n  TIER 5: TRANSFORMERS\n" + "=" * 60)
        all_results.update(run_tier5(train_df, val_df, test_df, device, args.seeds))

    # NOTE: Tier 6 (LLMs) disabled until API keys are configured.
    # Uncomment when ready:
    # if args.tier is None or args.tier == 6:
    #     print("\n" + "=" * 60 + "\n  TIER 6: LLMs\n" + "=" * 60)
    #     all_results.update(run_tier6(test_df, train_df, args.openai_key,
    #                                   args.anthropic_key, args.google_key))

    # Save
    os.makedirs("results", exist_ok=True)
    out = f"results/results_{ts}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<35} {'QWK':>8} {'Pearson':>8} {'RMSE':>8} {'MAE':>8} {'Lat(ms)':>9} {'Samples/s':>10}")
    print("-" * 92)
    for name, m in sorted(all_results.items()):
        if "error" in m:
            continue
        std_str = f"±{m.get('qwk_std', 0):.2f}" if "qwk_std" in m else "      "
        lat_str = f"{m['latency_mean_ms']:>9.2f}" if "latency_mean_ms" in m else f"{'N/A':>9}"
        tps_str = f"{m['throughput_per_sec']:>10.1f}" if "throughput_per_sec" in m else f"{'N/A':>10}"
        print(f"{name:<35} {m['qwk']:>7.4f}{std_str} {m['pearson']:>8.4f} {m['rmse']:>8.4f} {m['mae']:>8.4f} {lat_str} {tps_str}")
    print("=" * 92)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
