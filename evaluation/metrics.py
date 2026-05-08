"""
KhmerXScore Evaluation Metrics — Complete
==========================================
Agreement:    QWK, Accuracy, Exact Agreement, Adjacent Agreement
Correlation:  Pearson, Spearman
Error:        RMSE, MAE
Classification: Weighted F1, Precision, Recall
Visual:       Confusion Matrix, Loss Curves, Scatter Plot, Distribution
Breakdown:    Per-subject, Per-score-level
"""
import time
import numpy as np
import os
from scipy import stats
from sklearn.metrics import (
    cohen_kappa_score, mean_squared_error, mean_absolute_error,
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report
)


def to_discrete(y, scale=4):
    """Convert continuous 0-1 ratio to discrete 0-4 score."""
    return np.clip(np.round(np.array(y, dtype=float) * scale), 0, scale).astype(int)


def compute_metrics(y_true, y_pred, scale=4):
    """
    Compute ALL evaluation metrics.
    Inputs: continuous 0-1 ratios.
    Returns: dict with all metrics.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), 0, 1)
    n = len(y_true)
    if n < 3:
        return {"qwk": 0, "pearson": 0, "spearman": 0, "rmse": 0, "mae": 0,
                "accuracy": 0, "f1_weighted": 0, "exact_agreement": 0, "n": n}

    # Discrete versions
    yt_d = to_discrete(y_true, scale)
    yp_d = to_discrete(y_pred, scale)

    # --- Agreement Metrics ---
    try:
        qwk = float(cohen_kappa_score(yt_d, yp_d, weights="quadratic"))
    except:
        qwk = 0.0

    accuracy = float(accuracy_score(yt_d, yp_d))
    exact_agreement = float(np.mean(yt_d == yp_d))
    adjacent_agreement = float(np.mean(np.abs(yt_d - yp_d) <= 1))

    # --- Correlation Metrics ---
    try:
        pearson = float(stats.pearsonr(y_true, y_pred)[0])
    except:
        pearson = 0.0
    try:
        spearman = float(stats.spearmanr(y_true, y_pred)[0])
    except:
        spearman = 0.0

    # --- Error Metrics ---
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    # --- Classification Metrics (weighted for imbalanced classes) ---
    labels = list(range(scale + 1))
    try:
        f1_w = float(f1_score(yt_d, yp_d, average="weighted", labels=labels, zero_division=0))
    except:
        f1_w = 0.0
    try:
        prec_w = float(precision_score(yt_d, yp_d, average="weighted", labels=labels, zero_division=0))
    except:
        prec_w = 0.0
    try:
        rec_w = float(recall_score(yt_d, yp_d, average="weighted", labels=labels, zero_division=0))
    except:
        rec_w = 0.0

    return {
        # Agreement
        "qwk": qwk,
        "accuracy": accuracy,
        "exact_agreement": exact_agreement,
        "adjacent_agreement": adjacent_agreement,
        # Correlation
        "pearson": pearson,
        "spearman": spearman,
        # Error
        "rmse": rmse,
        "mae": mae,
        # Classification
        "f1_weighted": f1_w,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        # Meta
        "n": n,
    }


def compute_confusion_matrix(y_true, y_pred, scale=4):
    """Compute confusion matrix from continuous predictions."""
    yt_d = to_discrete(y_true, scale)
    yp_d = to_discrete(y_pred, scale)
    labels = list(range(scale + 1))
    cm = confusion_matrix(yt_d, yp_d, labels=labels)
    return cm


def compute_per_score_level(y_true, y_pred, scale=4):
    """Break down metrics by score level."""
    yt_d = to_discrete(y_true, scale)
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), 0, 1)

    results = {}
    for level in range(scale + 1):
        mask = yt_d == level
        n = int(mask.sum())
        if n == 0:
            results[level] = {"n": 0, "mae": 0, "accuracy": 0}
            continue

        level_true = y_true[mask]
        level_pred = y_pred[mask]
        level_pred_d = to_discrete(level_pred, scale)

        results[level] = {
            "n": n,
            "mae": float(mean_absolute_error(level_true, level_pred)),
            "rmse": float(np.sqrt(mean_squared_error(level_true, level_pred))),
            "accuracy": float(np.mean(level_pred_d == level)),
            "mean_pred": float(level_pred.mean()),
            "mean_true": float(level_true.mean()),
        }
    return results


def compute_per_subject(df, pred_col="prediction", scale=4):
    """Compute full metrics per subject."""
    results = {}
    for subj in sorted(df["Subject"].unique()):
        sub = df[df["Subject"] == subj]
        if len(sub) < 3:
            continue
        results[subj] = compute_metrics(sub["score_ratio"].values, sub[pred_col].values, scale)
    return results


def benchmark_latency(predict_fn, answers, references, warmup=5):
    """
    Per-sample inference latency with warmup runs.

    predict_fn: callable([answer], [reference]) -> array
    Warmup discards cold-start overhead (model loading, kernel caching).
    Returns mean, P50, P95, P99 in ms and throughput in samples/sec.
    """
    n = len(answers)
    for i in range(min(warmup, n)):
        predict_fn([answers[i]], [references[i]])

    latencies_ms = []
    for a, r in zip(answers, references):
        t0 = time.perf_counter()
        predict_fn([a], [r])
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies_ms)
    total_sec = lat.sum() / 1000
    return {
        "latency_mean_ms":    round(float(lat.mean()), 3),
        "latency_p50_ms":     round(float(np.percentile(lat, 50)), 3),
        "latency_p95_ms":     round(float(np.percentile(lat, 95)), 3),
        "latency_p99_ms":     round(float(np.percentile(lat, 99)), 3),
        "throughput_per_sec": round(float(n / total_sec), 1),
    }


def aggregate_seeds(results_list):
    """Average metrics across multiple seeds. Reports mean ± std."""
    keys = [k for k in results_list[0] if k not in ("n",)]
    avg = {}
    for k in keys:
        vals = [r[k] for r in results_list]
        avg[k] = float(np.mean(vals))
        avg[k + "_std"] = float(np.std(vals))
    avg["n"] = results_list[0]["n"]
    avg["n_seeds"] = len(results_list)
    return avg


# ============================
# PRINTING
# ============================

def print_metrics(m, title=""):
    """Pretty-print all metrics."""
    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  Agreement:")
    print(f"    QWK:                {m['qwk']:.4f}", end="")
    if "qwk_std" in m: print(f"  ± {m['qwk_std']:.4f}", end="")
    print()
    print(f"    Accuracy:           {m['accuracy']:.4f}")
    print(f"    Exact Agreement:    {m['exact_agreement']:.2%}")
    print(f"    Adjacent Agr (±1):  {m['adjacent_agreement']:.2%}")
    print(f"  Correlation:")
    print(f"    Pearson r:          {m['pearson']:.4f}")
    print(f"    Spearman ρ:         {m['spearman']:.4f}")
    print(f"  Error:")
    print(f"    RMSE:               {m['rmse']:.4f}")
    print(f"    MAE:                {m['mae']:.4f}")
    print(f"  Classification:")
    print(f"    F1 (weighted):      {m['f1_weighted']:.4f}")
    print(f"    Precision (wt):     {m['precision_weighted']:.4f}")
    print(f"    Recall (wt):        {m['recall_weighted']:.4f}")
    print(f"  N: {m['n']}", end="")
    if "n_seeds" in m: print(f"  (avg of {m['n_seeds']} seeds)", end="")
    print()
    if "train_time_sec" in m or "latency_mean_ms" in m or "inference_time_sec" in m:
        print(f"  Timing:")
        if "train_time_sec" in m:
            print(f"    Train time:         {m['train_time_sec']:.3f} s")
        if "latency_mean_ms" in m:
            print(f"    Latency mean:       {m['latency_mean_ms']:.3f} ms/sample")
            if "latency_p50_ms" in m:
                print(f"    Latency P50:        {m['latency_p50_ms']:.3f} ms/sample")
                print(f"    Latency P95:        {m['latency_p95_ms']:.3f} ms/sample")
                print(f"    Latency P99:        {m['latency_p99_ms']:.3f} ms/sample")
            print(f"    Throughput:         {m['throughput_per_sec']:.1f} samples/s")
        elif "inference_time_sec" in m:
            n_s = m.get("n", 1)
            print(f"    Inference time:     {m['inference_time_sec']:.3f} s (batch, {n_s} samples)")
            print(f"    Latency mean:       {m['inference_time_sec'] / n_s * 1000:.3f} ms/sample")
            print(f"    Throughput:         {n_s / m['inference_time_sec']:.1f} samples/s")
    print(f"{'=' * 60}")


def print_confusion_matrix(cm, title="Confusion Matrix"):
    """Print confusion matrix as formatted text."""
    n_classes = cm.shape[0]
    print(f"\n  {title}")
    print(f"  {'':>8}", end="")
    for j in range(n_classes):
        print(f" Pred={j:>2}", end="")
    print()
    print(f"  {'':>8}{'-' * (8 * n_classes)}")
    for i in range(n_classes):
        print(f"  True={i:>2} |", end="")
        for j in range(n_classes):
            val = cm[i, j]
            if i == j:
                print(f"  [{val:>3}]", end="")  # Highlight diagonal
            else:
                print(f"   {val:>3} ", end="")
        print()
    # Summary
    total = cm.sum()
    correct = np.trace(cm)
    print(f"  {'':>8}{'-' * (8 * n_classes)}")
    print(f"  Correct: {correct}/{total} ({correct/total:.1%})")


def print_per_score_level(psl, title="Per-Score-Level Analysis"):
    """Print per-score-level breakdown."""
    print(f"\n  {title}")
    print(f"  {'Score':>6} {'N':>5} {'MAE':>8} {'RMSE':>8} {'Acc':>8} {'MeanPred':>9} {'MeanTrue':>9}")
    print(f"  {'-' * 56}")
    for level in sorted(psl.keys()):
        m = psl[level]
        if m["n"] == 0:
            print(f"  {level:>6} {0:>5}    ---      ---      ---       ---       ---")
        else:
            print(f"  {level:>6} {m['n']:>5} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['accuracy']:>8.2%} {m['mean_pred']:>9.3f} {m['mean_true']:>9.3f}")


# ============================
# VISUALIZATION (save to file)
# ============================

def save_confusion_matrix_plot(cm, path, title="Confusion Matrix"):
    """Save confusion matrix as heatmap image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))
        # Normalize to percentages
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=range(cm.shape[1]), yticklabels=range(cm.shape[0]),
                    ax=ax, vmin=0, vmax=1)

        # Add raw counts as secondary annotation
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.75, f"(n={cm[i, j]})", ha="center", va="center",
                        fontsize=7, color="gray")

        ax.set_xlabel("Predicted Score", fontsize=12)
        ax.set_ylabel("Actual Score", fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
    except ImportError:
        print("  matplotlib/seaborn not available for plotting")


def save_loss_curves(train_losses, val_losses, path, title="Training History"):
    """Save training and validation loss curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, "b-o", markersize=3, label="Training Loss")
        ax.plot(epochs, val_losses, "r-o", markersize=3, label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
    except ImportError:
        print("  matplotlib not available")


def save_scatter_plot(y_true, y_pred, path, title="Predicted vs Actual"):
    """Save scatter plot of predicted vs actual scores."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors="navy", linewidths=0.5)
        ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual Score (ratio)", fontsize=12)
        ax.set_ylabel("Predicted Score (ratio)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add correlation text
        r = float(stats.pearsonr(y_true, y_pred)[0])
        ax.text(0.05, 0.92, f"Pearson r = {r:.3f}", transform=ax.transAxes, fontsize=11)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
    except ImportError:
        print("  matplotlib not available")


def save_distribution_plot(y_true, y_pred, path, title="Score Distribution"):
    """Save overlapping histogram of actual vs predicted discrete scores."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        yt_d = to_discrete(y_true)
        yp_d = to_discrete(y_pred)

        fig, ax = plt.subplots(figsize=(8, 5))
        scores = range(5)
        width = 0.35
        true_counts = [np.sum(yt_d == s) for s in scores]
        pred_counts = [np.sum(yp_d == s) for s in scores]

        x = np.array(scores)
        ax.bar(x - width / 2, true_counts, width, label="Actual", color="#4A90D9", alpha=0.8)
        ax.bar(x + width / 2, pred_counts, width, label="Predicted", color="#E74C3C", alpha=0.8)
        ax.set_xlabel("Discrete Score (0-4)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(scores)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
    except ImportError:
        print("  matplotlib not available")


def generate_full_report(y_true, y_pred, model_name, output_dir="results", df=None):
    """Generate complete evaluation report with all metrics and plots."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = model_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")

    # 1. All metrics
    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, model_name)

    # 2. Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    print_confusion_matrix(cm, f"{model_name} — Confusion Matrix")
    save_confusion_matrix_plot(cm, f"{output_dir}/{prefix}_confusion.png", f"{model_name} — Confusion Matrix")

    # 3. Per-score-level
    psl = compute_per_score_level(y_true, y_pred)
    print_per_score_level(psl, f"{model_name} — Per-Score Analysis")

    # 4. Scatter plot
    save_scatter_plot(np.array(y_true), np.array(y_pred),
                      f"{output_dir}/{prefix}_scatter.png", f"{model_name} — Predicted vs Actual")

    # 5. Distribution
    save_distribution_plot(np.array(y_true), np.array(y_pred),
                           f"{output_dir}/{prefix}_distribution.png", f"{model_name} — Score Distribution")

    # 6. Per-subject (if df provided)
    if df is not None:
        print(f"\n  Per-Subject Breakdown:")
        print(f"  {'Subject':<18} {'QWK':>8} {'Pearson':>8} {'F1':>8} {'RMSE':>8} {'N':>5}")
        print(f"  {'-' * 52}")
        ps = compute_per_subject(df)
        for subj, sm in ps.items():
            print(f"  {subj:<18} {sm['qwk']:>8.4f} {sm['pearson']:>8.4f} {sm['f1_weighted']:>8.4f} {sm['rmse']:>8.4f} {sm['n']:>5}")

    return metrics
