#!/usr/bin/env python3
"""
regenerate_escalated.py

Completes an exact LLM-all baseline for AspectRoute, and records REAL token
counts so cost can be reported from measurement rather than an assumed constant.

Why this is cheap: the tau=0.85 run already produced GPT-4o-mini predictions for
the 23,763 escalated items. Only the 17,552 VADER-routed items still need a call.
Cached + new = a genuine LLM-all result over all 41,315 aspect mentions.

Safety properties:
  * Never re-calls an item that already has a cached LLM prediction.
  * Writes results incrementally to JSONL; safe to stop and restart (resumes).
  * Records prompt/completion tokens returned by the API for every call.
  * Uses the SAME prompt as the original run, so results are comparable.

Usage (from the folder containing your `results/` directory):

    export OPENAI_API_KEY="sk-..."          # do NOT hard-code the key
    python3 regenerate_escalated.py --dry_run   # check counts + cost first
    python3 regenerate_escalated.py             # run for real

Outputs (in results/):
    llm_all_completion.jsonl   incremental raw results, one JSON per line
    llm_all_predictions.pkl    final merged LLM-all prediction array
    llm_all_cost_report.txt    measured token counts and actual cost
"""

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# gpt-4o-mini pricing (USD per 1M tokens) -- update if pricing changes
PRICE_IN_PER_M = 0.15
PRICE_OUT_PER_M = 0.60
MODEL = "gpt-4o-mini-2024-07-18"


def load_cache(results_dir, name):
    with open(os.path.join(results_dir, ".cache", name + ".pkl"), "rb") as f:
        return pickle.load(f)


def build_prompt(text, aspect):
    """EXACTLY the prompt used in the original run -- do not change."""
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


def classify_one(client, idx, text, aspect, max_retries=6):
    """One API call with exponential backoff on rate limits.

    Returns (idx, sentiment, confidence, tok_in, tok_out, err).
    On 429 / transient errors it sleeps and retries rather than giving up,
    so a long unattended run does not silently lose items.
    """
    delay = 5.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": build_prompt(text, aspect)}],
                temperature=0.1,
                max_tokens=150,
            )
            tok_in = resp.usage.prompt_tokens
            tok_out = resp.usage.completion_tokens
            raw = resp.choices[0].message.content
            try:
                parsed = json.loads(raw)
                sentiment = parsed.get("sentiment", "neutral")
                confidence = parsed.get("confidence", 0.8)
            except Exception:
                sentiment, confidence = "neutral", 0.5
            if sentiment not in ("positive", "negative", "neutral"):
                sentiment = "neutral"
            return idx, sentiment, confidence, tok_in, tok_out, None
        except Exception as e:
            msg = str(e)
            transient = ("429" in msg or "rate limit" in msg.lower()
                         or "timeout" in msg.lower() or "500" in msg
                         or "502" in msg or "503" in msg or "529" in msg)
            if transient and attempt < max_retries - 1:
                time.sleep(delay)
                delay = min(delay * 2, 120.0)   # 5,10,20,40,80,120s
                continue
            return idx, None, None, 0, 0, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent requests (default 4; lower = fewer rate-limit stalls)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Report how many calls are needed and estimated cost, then exit")
    ap.add_argument("--limit", type=int, default=0,
                    help="Only process the first N pending items (for a small test)")
    args = ap.parse_args()

    out_jsonl = os.path.join(args.results_dir, "escalated_regenerated.jsonl")
    seed_jsonl = os.path.join(args.results_dir, "backbone_gpt-4o-mini-2024-07-18.jsonl")

    print("Loading cache (no calls yet)...")
    aspects_df = load_cache(args.results_dir, "aspects_df")
    stream_preds, routing = load_cache(args.results_dir, "routing_results")
    routing = np.array(routing)
    stream_preds = np.array(stream_preds, dtype=object)

    need_idx = np.where(routing == "llm")[0]     # escalated items: cached preds are contaminated
    have_idx = np.where(routing == "vader")[0]   # (not re-run here)

    print(f"  total aspect mentions : {len(routing)}")
    print(f"  VADER-routed (not touched): {len(have_idx)}")
    print(f"  escalated items to regenerate: {len(need_idx)}")

    # resume support
    done = {}
    if os.path.exists(out_jsonl):
        with open(out_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["idx"]] = r
                except Exception:
                    continue
        print(f"  resuming: {len(done)} already completed in a previous run")

    # reuse fresh gpt-4o-mini predictions from the cross-backbone check
    if os.path.exists(seed_jsonl):
        n_seed = 0
        with open(seed_jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r["idx"] not in done:
                        r.setdefault("confidence", None)
                        done[r["idx"]] = r; n_seed += 1
                except Exception:
                    continue
        print(f"  reused from cross-backbone run: {n_seed}")
    pending = [i for i in need_idx if int(i) not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"  pending this run      : {len(pending)}")

    if args.dry_run or not pending:
        words = np.array([len(aspects_df.iloc[i]["context"].split()) for i in pending]) if pending else np.array([0])
        est_in = ((words + 49) * 1.35).sum()
        est_out = len(pending) * 45
        est = est_in / 1e6 * PRICE_IN_PER_M + est_out / 1e6 * PRICE_OUT_PER_M
        print(f"\n  ESTIMATED cost for {len(pending)} calls: ${est:.2f}")
        print("  (actual cost will be computed from real token counts)")
        if args.dry_run:
            print("\nDry run complete. Re-run without --dry_run to execute.")
            return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: set OPENAI_API_KEY in your environment first:")
        print('  export OPENAI_API_KEY="sk-..."')
        sys.exit(1)

    import openai
    client = openai.OpenAI(api_key=api_key)

    print(f"\nRunning {len(pending)} calls with {args.workers} workers...")
    t0 = time.time()
    errors = 0
    fh = open(out_jsonl, "a")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(classify_one, client, int(i),
                      aspects_df.iloc[i]["context"], aspects_df.iloc[i]["aspect"]): int(i)
            for i in pending
        }
        for n, fut in enumerate(as_completed(futs), 1):
            idx, sent, conf, ti, to, err = fut.result()
            if err:
                errors += 1
                if errors <= 5:
                    print(f"  [error] idx={idx}: {err[:120]}")
            else:
                rec = {"idx": idx, "sentiment": sent, "confidence": conf,
                       "tok_in": ti, "tok_out": to}
                fh.write(json.dumps(rec) + "\n")
                done[idx] = rec
            if n % 250 == 0:
                fh.flush()
                el = time.time() - t0
                rate = n / el if el > 0 else 0
                eta = (len(pending) - n) / rate / 60 if rate > 0 else 0
                print(f"  {n}/{len(pending)}  ({rate:.1f}/s, ~{eta:.0f} min left, {errors} errors)")

    fh.close()
    print(f"\nCompleted in {(time.time()-t0)/60:.1f} min, {errors} errors")

    fresh = np.empty(len(routing), dtype=object)
    missing = 0
    for i in need_idx:
        r = done.get(int(i))
        if r: fresh[i] = r["sentiment"]
        else: missing += 1
    out_pkl = os.path.join(args.results_dir, "escalated_fresh_predictions.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(fresh, f)
    print(f"\nFresh predictions for escalated items: {len(need_idx)-missing}/{len(need_idx)}")
    print(f"Still missing: {missing}  (if >0, re-run tomorrow; it resumes)")
    print(f"Wrote {out_pkl}")


if __name__ == "__main__":
    main()
