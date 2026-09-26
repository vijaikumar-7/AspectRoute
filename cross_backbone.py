#!/usr/bin/env python3
"""
cross_backbone.py

Cross-backbone comparison for AspectRoute (Reviewer 3's request, and evidence
for Reviewer 2 that the contribution is the routing policy rather than one
specific LLM).

DESIGN
------
Routing is decided by RoBERTa encoder confidence and the sarcasm heuristic.
Neither depends on which LLM sits behind the escalated path, so the routing
decisions are IDENTICAL across backbones. Only the classifier applied to
escalated items changes.

We therefore take a random sample of the escalated (LLM-routed) items, run a
second backbone over exactly those items, and compare it against the cached
gpt-4o-mini predictions for the same items. This is a paired comparison on
identical inputs, which is what makes the result interpretable.

Sampling (rather than all 23,763) keeps the run inside a 10,000 RPD limit.
4,000 items gives roughly +/-1.5% precision on agreement rate, which is ample.

USAGE
-----
    export OPENAI_API_KEY="sk-..."
    python3 cross_backbone.py --dry_run
    python3 cross_backbone.py --model gpt-4.1-nano --n 4000

Pick --model to suit the argument you want to make:
    a CHEAPER model  -> shows routing still works if you economise further
    a STRONGER model -> shows diminishing returns from spending more
Any chat-completions model your key can access will work.

OUTPUT (in results/)
    backbone_<model>.jsonl        raw per-item results (resumable)
    backbone_comparison.txt       agreement, accuracy, cost comparison
"""

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# Per-1M-token pricing. Add entries as needed; unknown models fall back to
# measured-token reporting without a dollar figure.
PRICING = {
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4o-mini":            (0.15, 0.60),
    "gpt-4.1-nano":           (0.10, 0.40),
    "gpt-4.1-mini":           (0.40, 1.60),
    "gpt-4o":                 (2.50, 10.00),
}
BASE_MODEL = "gpt-4o-mini-2024-07-18"   # the cached backbone we compare against


def load_cache(results_dir, name):
    with open(os.path.join(results_dir, ".cache", name + ".pkl"), "rb") as f:
        return pickle.load(f)


def build_prompt(text, aspect):
    """Identical to the original run. Do not modify -- comparability depends on it."""
    return (
        f'Analyze the sentiment of the following movie review excerpt '
        f'specifically regarding the "{aspect}" aspect.\n\n'
        f'Review excerpt: "{text}"\n\n'
        f'Classify the sentiment toward {aspect} as exactly one of: '
        f'positive, negative, neutral\n\n'
        f'Consider sarcasm, implicit sentiment, and context carefully.\n'
        f'Respond with ONLY a JSON object: {{"sentiment": '
        f'"positive/negative/neutral", "confidence": 0.0-1.0, '
        f'"reasoning": "brief explanation"}}'
    )


def classify_one(client, model, idx, text, aspect, max_retries=6):
    delay = 5.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": build_prompt(text, aspect)}],
                temperature=0.1,
                max_tokens=150,
            )
            raw = resp.choices[0].message.content
            try:
                parsed = json.loads(raw)
                sent = parsed.get("sentiment", "neutral")
            except Exception:
                sent = "neutral"
            if sent not in ("positive", "negative", "neutral"):
                sent = "neutral"
            return idx, sent, resp.usage.prompt_tokens, resp.usage.completion_tokens, None
        except Exception as e:
            msg = str(e)
            transient = ("429" in msg or "rate limit" in msg.lower()
                         or "timeout" in msg.lower() or "500" in msg
                         or "502" in msg or "503" in msg)
            if transient and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
                continue
            return idx, None, 0, 0, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--model", default="gpt-4.1-nano",
                    help="Second backbone to evaluate")
    ap.add_argument("--n", type=int, default=4000,
                    help="Number of escalated items to sample (default 4000)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    print("Loading cache (no calls yet)...")
    aspects_df = load_cache(args.results_dir, "aspects_df")
    stream_preds, routing = load_cache(args.results_dir, "routing_results")
    routing = np.array(routing)
    base_preds = np.array(stream_preds, dtype=object)
    gt = np.array(aspects_df["ground_truth"].tolist(), dtype=object)

    escalated = np.where(routing == "llm")[0]
    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(escalated, size=min(args.n, len(escalated)), replace=False))

    print(f"  escalated items available : {len(escalated)}")
    print(f"  sampling                  : {len(sample)}")
    print(f"  second backbone           : {args.model}")

    out_path = os.path.join(args.results_dir, f"backbone_{args.model.replace('/','_')}.jsonl")
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["idx"]] = r
                except Exception:
                    pass
        print(f"  resuming                  : {len(done)} already done")

    pending = [int(i) for i in sample if int(i) not in done]
    print(f"  pending this run          : {len(pending)}")

    if args.dry_run:
        words = np.array([len(aspects_df.iloc[i]["context"].split()) for i in pending])
        est_in = ((words + 49) * 1.35).sum()
        est_out = len(pending) * 50
        if args.model in PRICING:
            pin, pout = PRICING[args.model]
            print(f"\n  ESTIMATED cost: ${est_in/1e6*pin + est_out/1e6*pout:.3f}")
        print("\nDry run complete. Re-run without --dry_run.")
        return

    if pending:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("\nERROR: export OPENAI_API_KEY first.")
            sys.exit(1)
        import openai
        client = openai.OpenAI(api_key=key)

        print(f"\nRunning {len(pending)} calls with {args.workers} workers...")
        t0, errors = time.time(), 0
        fh = open(out_path, "a")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(classify_one, client, args.model, int(i),
                              aspects_df.iloc[i]["context"],
                              aspects_df.iloc[i]["aspect"]): int(i) for i in pending}
            for n, fut in enumerate(as_completed(futs), 1):
                idx, sent, ti, to, err = fut.result()
                if err:
                    errors += 1
                    if errors <= 3:
                        print(f"  [error] idx={idx}: {err[:110]}")
                else:
                    rec = {"idx": idx, "sentiment": sent, "tok_in": ti, "tok_out": to}
                    fh.write(json.dumps(rec) + "\n")
                    done[idx] = rec
                if n % 250 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"  {n}/{len(pending)}  ({n/el:.1f}/s, {errors} errors)")
        fh.close()
        print(f"\nCompleted in {(time.time()-t0)/60:.1f} min, {errors} errors")

    # ---------------- analysis ----------------
    idxs = np.array([i for i in sample if int(i) in done])
    if len(idxs) == 0:
        print("No results to analyse.")
        return
    new = np.array([done[int(i)]["sentiment"] for i in idxs], dtype=object)
    old = base_preds[idxs]
    g = gt[idxs]

    agree = (new == old).mean()

    def commit_acc(p):
        m = np.isin(p, ["positive", "negative"])
        return ((g[m] == p[m]).mean() if m.sum() else 0.0), m.mean(), m.sum()

    a_old, cov_old, n_old = commit_acc(old)
    a_new, cov_new, n_new = commit_acc(new)

    ti = sum(done[int(i)]["tok_in"] for i in idxs)
    to = sum(done[int(i)]["tok_out"] for i in idxs)
    avg_in, avg_out = ti / len(idxs), to / len(idxs)

    def cost_per_1k(model, n_calls=23763, n_rev=9916):
        if model not in PRICING:
            return None
        pin, pout = PRICING[model]
        return ((avg_in * n_calls) / 1e6 * pin + (avg_out * n_calls) / 1e6 * pout) / n_rev * 1000

    c_new = cost_per_1k(args.model)
    c_old = cost_per_1k(BASE_MODEL)

    lines = []
    lines.append("CROSS-BACKBONE COMPARISON (AspectRoute revision)")
    lines.append("=" * 68)
    lines.append(f"Paired sample of escalated (LLM-routed) items: n = {len(idxs)}")
    lines.append("Routing decisions are identical across backbones by construction;")
    lines.append("only the classifier applied to escalated items differs.")
    lines.append("")
    lines.append(f"Backbone A (cached) : {BASE_MODEL}")
    lines.append(f"Backbone B (new)    : {args.model}")
    lines.append("")
    lines.append(f"Label agreement between backbones : {agree*100:.1f}%")
    lines.append("")
    lines.append(f"{'Backbone':<28}{'Commit-acc':<13}{'Coverage':<12}{'n committed'}")
    lines.append(f"{BASE_MODEL:<28}{a_old*100:>6.1f}%{'':<6}{cov_old*100:>6.1f}%{'':<5}{n_old}")
    lines.append(f"{args.model:<28}{a_new*100:>6.1f}%{'':<6}{cov_new*100:>6.1f}%{'':<5}{n_new}")
    lines.append("")
    lines.append(f"Measured tokens/call (backbone B): {avg_in:.1f} in, {avg_out:.1f} out")
    if c_new is not None and c_old is not None:
        lines.append(f"Projected AspectRoute cost per 1K reviews (57.5% escalation):")
        lines.append(f"   with {BASE_MODEL}: ${c_old:.3f}")
        lines.append(f"   with {args.model}: ${c_new:.3f}")
    lines.append("")
    # Data-driven summary -- deliberately makes no interpretive claim.
    both = np.isin(old, ["positive","negative"]) & np.isin(new, ["positive","negative"])
    lines.append("DECOMPOSITION (no interpretation is asserted)")
    lines.append(f"  Items where both backbones committed : {both.sum()}")
    if both.sum():
        lines.append(f"  Agreement among those                : {(old[both]==new[both]).mean()*100:.1f}%")
    lines.append(f"  Coverage difference (B - A)          : {(cov_new-cov_old)*100:+.1f} pts")
    lines.append("  Low overall agreement may reflect abstention differences rather")
    lines.append("  than label disagreement; inspect the committed-agreement figure.")

    report = "\n".join(lines)
    with open(os.path.join(args.results_dir, "backbone_comparison.txt"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print("\nWrote results/backbone_comparison.txt")


if __name__ == "__main__":
    main()
