"""
AspectStreamSA Ablation Study
==============================
Runs 4 ablations using cached checkpoints — no redownloading, no API calls.

Usage:
    python ablation.py --output_dir results

Ablations:
    A1: Remove domain lexicon (plain VADER, no streaming-specific words)
    A2: Random routing (same 42.5/57.5 split but randomly assigned)
    A3: No aspect windowing (classify full review text instead of aspect sentence)
    A4: No LLM (route everything to VADER regardless of confidence)
"""

import os
import sys
import pickle
import argparse
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score
from tabulate import tabulate

# ── load helpers from main script ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from aspectstreamsa import (
    vader_classify, DOMAIN_LEXICON, ASPECT_TAXONOMY,
    detect_sarcasm_signals
)


def load_cache(output_dir, name):
    path = os.path.join(output_dir, ".cache", f"{name}.pkl")
    if not os.path.exists(path):
        print(f"ERROR: Cache file not found: {path}")
        print("Make sure you have run the main pipeline first.")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


def metrics(gt, preds, name):
    """Binary metrics, neutral mapped to negative."""
    y_true, y_pred = [], []
    for g, p in zip(gt, preds):
        if g in ["positive", "negative"]:
            y_true.append(g)
            y_pred.append(p if p in ["positive", "negative"] else "negative")

    macro_f1  = f1_score(y_true, y_pred, labels=["negative","positive"], average="macro",    zero_division=0)
    precision = precision_score(y_true, y_pred, labels=["negative","positive"], average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, labels=["negative","positive"], average="macro",    zero_division=0)
    accuracy  = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)

    # Coverage-F1: exclude neutral
    pairs = [(g, p) for g, p in zip(gt, preds) if p != "neutral"]
    if pairs:
        gt2, p2 = zip(*pairs)
        cov_f1 = f1_score(list(gt2), list(p2), labels=["negative","positive"], average="macro", zero_division=0)
        coverage = len(pairs) / len(gt) * 100
    else:
        cov_f1, coverage = 0.0, 0.0

    return {
        "name": name,
        "binary_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "coverage_f1": cov_f1,
        "coverage_pct": coverage,
        "n": len(y_true)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Loading cached checkpoints...")
    aspects_df       = load_cache(args.output_dir, "aspects_df")
    roberta_preds, roberta_confs, _ = load_cache(args.output_dir, "roberta_results")
    vader_doc_preds, vader_doc_scores, vader_asp_preds, vader_asp_scores = load_cache(args.output_dir, "vader_results")
    stream_preds, routing_decisions  = load_cache(args.output_dir, "routing_results")

    gt = aspects_df["ground_truth"].tolist()
    rows_list = list(aspects_df.iterrows())
    tau = 0.85

    print(f"Loaded {len(gt)} aspect-sentiment pairs\n")

    results = []

    # ── Baseline: full AspectStreamSA ──────────────────────────────────────
    results.append(metrics(gt, stream_preds, "AspectStreamSA (full)"))

    # ── A1: No domain lexicon ──────────────────────────────────────────────
    print("Running A1: No domain lexicon...")
    import nltk; nltk.download("vader_lexicon", quiet=True)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    plain_analyzer = SentimentIntensityAnalyzer()  # no custom lexicon

    a1_preds = []
    for i, route in enumerate(routing_decisions):
        _, row = rows_list[i]
        if route == "vader":
            compound = plain_analyzer.polarity_scores(row["context"])["compound"]
            if compound <= -0.15:
                label = "negative"
            elif compound >= 0.20:
                label = "positive"
            else:
                label = "neutral"
            a1_preds.append(label)
        else:
            # LLM predictions unchanged
            a1_preds.append(stream_preds[i])

    results.append(metrics(gt, a1_preds, "A1: No domain lexicon"))

    # ── A2: Random routing (same split ratio, random assignment) ───────────
    print("Running A2: Random routing...")
    vader_count = sum(1 for r in routing_decisions if r == "vader")
    llm_count   = len(routing_decisions) - vader_count
    vader_ratio = vader_count / len(routing_decisions)

    # Random route mask with same proportions
    random_routes = np.random.choice(
        ["vader", "llm"], size=len(routing_decisions),
        p=[vader_ratio, 1 - vader_ratio]
    )

    # Augmented VADER for VADER-routed
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    aug_analyzer = SentimentIntensityAnalyzer()
    for term, score in DOMAIN_LEXICON.items():
        aug_analyzer.lexicon[term] = score

    a2_preds = []
    for i, route in enumerate(random_routes):
        _, row = rows_list[i]
        if route == "vader":
            compound = aug_analyzer.polarity_scores(row["context"])["compound"]
            label = "negative" if compound <= -0.15 else "positive" if compound >= 0.20 else "neutral"
            a2_preds.append(label)
        else:
            a2_preds.append(stream_preds[i])

    results.append(metrics(gt, a2_preds, "A2: Random routing"))

    # ── A3: No aspect windowing (full review text for VADER) ───────────────
    print("Running A3: No aspect windowing...")
    a3_preds = []
    for i, route in enumerate(routing_decisions):
        _, row = rows_list[i]
        if route == "vader":
            # Use FULL review text instead of aspect context window
            compound = aug_analyzer.polarity_scores(row["full_text"])["compound"]
            label = "negative" if compound <= -0.15 else "positive" if compound >= 0.20 else "neutral"
            a3_preds.append(label)
        else:
            a3_preds.append(stream_preds[i])

    results.append(metrics(gt, a3_preds, "A3: No aspect windowing"))

    # ── A4: No LLM (everything goes to VADER) ─────────────────────────────
    print("Running A4: No LLM routing...")
    a4_preds = []
    for i, (_, row) in enumerate(rows_list):
        compound = aug_analyzer.polarity_scores(row["context"])["compound"]
        label = "negative" if compound <= -0.15 else "positive" if compound >= 0.20 else "neutral"
        a4_preds.append(label)

    results.append(metrics(gt, a4_preds, "A4: No LLM (VADER-all)"))

    # ── A5: No confidence gating (all goes to LLM) ────────────────────────
    # We already have the LLM predictions from stream_preds for LLM-routed items.
    # For VADER-routed items we use RoBERTa prediction as proxy since we can't re-call the API.
    print("Running A5: No confidence gating (LLM for all — approximated)...")
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    a5_preds = []
    for i, route in enumerate(routing_decisions):
        if route == "llm":
            a5_preds.append(stream_preds[i])
        else:
            # Proxy: use RoBERTa prediction (best available without re-calling API)
            a5_preds.append(roberta_preds[i])

    results.append(metrics(gt, a5_preds, "A5: No routing (LLM-proxy-all)"))

    # ── Print results ──────────────────────────────────────────────────────
    rows = [[
        r["name"],
        f"{r['binary_f1']:.3f}",
        f"{r['precision']:.3f}",
        f"{r['recall']:.3f}",
        f"{r['accuracy']:.3f}",
        f"{r['coverage_f1']:.3f} ({r['coverage_pct']:.1f}%)"
    ] for r in results]

    headers = ["Method", "Binary F1", "Precision", "Recall", "Accuracy", "Coverage-F1"]
    table = tabulate(rows, headers=headers, tablefmt="grid")

    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(table)

    # ── Delta analysis ─────────────────────────────────────────────────────
    baseline_cov = results[0]["coverage_f1"]
    baseline_bin = results[0]["binary_f1"]
    print("\nDelta from full AspectStreamSA (Coverage-F1):")
    for r in results[1:]:
        delta = r["coverage_f1"] - baseline_cov
        sign  = "+" if delta >= 0 else ""
        print(f"  {r['name']}: {sign}{delta:.3f}")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, "ablation_results.txt")
    with open(out_path, "w") as f:
        f.write("ABLATION STUDY RESULTS\n")
        f.write("="*80 + "\n")
        f.write(table)
        f.write("\n\nDelta from full AspectStreamSA (Coverage-F1):\n")
        for r in results[1:]:
            delta = r["coverage_f1"] - baseline_cov
            sign  = "+" if delta >= 0 else ""
            f.write(f"  {r['name']}: {sign}{delta:.3f}\n")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
