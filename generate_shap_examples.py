#!/usr/bin/env python3
"""
generate_shap_examples.py

Standalone SHAP explainability run for the AspectRoute revision (Reviewer 3's
request for a worked, annotated attribution example).

IMPORTANT: This does NOT re-run the pipeline and makes NO API calls.
It loads the cached aspects from your existing results/.cache/ and computes
SHAP attributions using the local RoBERTa sentiment model only.

Usage (from the folder that contains your `results/` directory):
    python generate_shap_examples.py
    python generate_shap_examples.py --results_dir results --n_samples 20

Output:
    results/shap_examples_revision.txt
"""

import argparse
import os
import pickle

import numpy as np


def load_cache(results_dir, name):
    path = os.path.join(results_dir, ".cache", name + ".pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results",
                    help="Directory containing .cache/ (default: results)")
    ap.add_argument("--n_samples", type=int, default=20,
                    help="Number of contexts to explain (default: 20)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for sampling (default: 42, matches paper)")
    ap.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    help="HuggingFace model used as the routing encoder")
    args = ap.parse_args()

    import torch
    import shap
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print("Loading cached aspects (no API calls, no re-run)...")
    aspects_df = load_cache(args.results_dir, "aspects_df")
    stream_preds, routing = load_cache(args.results_dir, "routing_results")
    print(f"  {len(aspects_df)} aspect mentions loaded")

    print(f"Loading encoder: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    device = "cpu"

    def predict_proba(texts):
        out = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512, padding=True).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(**inputs).logits, dim=-1)
            out.append(probs.cpu().numpy()[0])
        return np.array(out)

    # Sample VADER-routed and LLM-routed items separately so the write-up can
    # show one worked example from each path (this is what the reviewer asked for).
    routing = np.array(routing)
    rng = np.random.default_rng(args.seed)

    picks = []
    for path in ["vader", "llm"]:
        idx = np.where(routing == path)[0]
        # prefer non-neutral committed predictions -- more informative examples
        preds = np.array(stream_preds, dtype=object)
        committed = idx[np.isin(preds[idx], ["positive", "negative"])]
        pool = committed if len(committed) > 0 else idx
        chosen = rng.choice(pool, size=min(args.n_samples // 2, len(pool)), replace=False)
        picks.extend([(int(i), path) for i in chosen])

    texts = [aspects_df.iloc[i]["context"] for i, _ in picks]

    print(f"Computing SHAP attributions for {len(texts)} contexts (CPU, a few minutes)...")
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_proba, masker,
                               output_names=["negative", "neutral", "positive"])
    shap_values = explainer(texts)

    out_path = os.path.join(args.results_dir, "shap_examples_revision.txt")
    with open(out_path, "w") as f:
        f.write("SHAP ATTRIBUTION EXAMPLES (AspectRoute revision)\n")
        f.write("=" * 70 + "\n")
        f.write("Computed from cached aspect contexts using the local RoBERTa encoder.\n")
        f.write("No LLM API calls were made; predictions below are the ones reported\n")
        f.write("in the paper (loaded from cache).\n\n")

        for k in range(len(shap_values)):
            i, path = picks[k]
            row = aspects_df.iloc[i]
            sv = shap_values[k]
            pred_class = int(np.argmax(sv.base_values + sv.values.sum(axis=0)))
            importance = sv.values[:, pred_class]
            tokens = sv.data
            order = np.argsort(np.abs(importance))[::-1][:8]

            f.write("-" * 70 + "\n")
            f.write(f"Example {k+1}  [routing path: {path.upper()}]\n")
            f.write(f"  Aspect category : {row['aspect']}\n")
            f.write(f"  Matched keyword : {row['keyword']}\n")
            f.write(f"  IMDB label      : {row['ground_truth']}\n")
            f.write(f"  System output   : {stream_preds[i]}\n")
            f.write(f"  Context         : {row['context'][:300]}\n")
            f.write(f"  Encoder class   : {['negative','neutral','positive'][pred_class]}\n")
            f.write("  Top attributed tokens (token, SHAP value):\n")
            for j in order:
                if j < len(tokens):
                    tok = str(tokens[j]).strip()
                    if tok:
                        f.write(f"      {tok:<20} {importance[j]:+.4f}\n")
            f.write("\n")

    print(f"\nDone. Wrote {out_path}")
    print("Send me that file and I'll turn the clearest case into the worked")
    print("example for the revised manuscript.")


if __name__ == "__main__":
    main()
