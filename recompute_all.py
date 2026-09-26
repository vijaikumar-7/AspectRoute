#!/usr/bin/env python3
"""
recompute_all.py

Regenerates every IMDB result in the AspectRoute revision from cached
predictions. Makes NO API calls and takes about a minute.

WHY THIS EXISTS
---------------
The original pipeline silently recorded failed API calls as "neutral"
predictions (the LLM classifier caught every exception and returned
"neutral", 0.5). Under rate limiting this contaminated the escalated-item
predictions: the cached run commits on 6.7% of escalated items, while a
fresh run of the SAME model on the SAME items commits on 62.2%, with 100%
agreement wherever both commit.

All escalated-item predictions were therefore regenerated with proper error
handling. This script recomputes every reported figure from the clean data
and prints the contamination evidence alongside it.

INPUTS (all under --results_dir)
    .cache/aspects_df.pkl              aspect mentions, contexts, gold labels
    .cache/routing_results.pkl         (original predictions, routing decisions)
    .cache/roberta_results.pkl         encoder predictions + confidences
    .cache/vader_results.pkl           VADER predictions
    escalated_fresh_predictions.pkl    regenerated escalated predictions
    llm_all_predictions.pkl            LLM predictions for non-escalated items

USAGE
    python3 recompute_all.py
    python3 recompute_all.py --results_dir results --bootstrap 2000

OUTPUT
    results/corrected_results.txt   (and the same report on stdout)
"""

import argparse
import os
import pickle
from collections import Counter, defaultdict
from math import erfc, sqrt

import numpy as np

# Measured from the regeneration run (real API usage counters), used for cost.
TOK_IN, TOK_OUT = 132.3, 52.2
PRICE_IN, PRICE_OUT = 0.15, 0.60      # USD per 1M tokens, gpt-4o-mini
POS, NEG, NEU = "positive", "negative", "neutral"
COMMITTED = (POS, NEG)

SARCASM_PATTERNS = [
    "oh great", "yeah right", "sure thing", "how wonderful", "what a surprise",
    "obviously", "clearly", "of course", "wow just wow", "thanks for nothing",
    "so original", "never seen that before", "how creative", "shocking",
    "who would have guessed", "certainly", "apparently", "supposedly",
    "allegedly", "as if",
]


# ----------------------------------------------------------------- metrics
def coverage_f1(gold, pred):
    """Macro-F1 over committed (non-neutral) predictions, plus coverage."""
    m = np.isin(pred, COMMITTED)
    if m.sum() == 0:
        return 0.0, 0.0
    g, p, f1s = gold[m], pred[m], []
    for c in (NEG, POS):
        tp = np.sum((p == c) & (g == c))
        fp = np.sum((p == c) & (g != c))
        fn = np.sum((p != c) & (g == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s)), float(m.mean())


def precision_recall(gold, pred):
    m = np.isin(pred, COMMITTED)
    g, p, P, R = gold[m], pred[m], [], []
    for c in (NEG, POS):
        tp = np.sum((p == c) & (g == c))
        fp = np.sum((p == c) & (g != c))
        fn = np.sum((p != c) & (g == c))
        P.append(tp / (tp + fp) if tp + fp else 0.0)
        R.append(tp / (tp + fn) if tp + fn else 0.0)
    return float(np.mean(P)), float(np.mean(R))


def binary_f1(gold, pred):
    """Neutral mapped to negative, as in the paper's secondary metric."""
    pb = np.where(pred == POS, POS, NEG)
    f1s = []
    for c in (NEG, POS):
        tp = np.sum((pb == c) & (gold == c))
        fp = np.sum((pb == c) & (gold != c))
        fn = np.sum((pb != c) & (gold == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def committed_accuracy(gold, pred):
    m = np.isin(pred, COMMITTED)
    return float((gold[m] == pred[m]).mean()) if m.sum() else 0.0


def review_bootstrap_ci(gold, pred, groups, n_boot, seed=42):
    """Resample whole reviews, preserving within-review correlation."""
    ids = list(groups)
    arrays = {r: np.asarray(v) for r, v in groups.items()}
    R = len(ids)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, R, R)
        idx = np.concatenate([arrays[ids[c]] for c in chosen])
        stats[b] = coverage_f1(gold[idx], pred[idx])[0]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def mcnemar_committed(gold, a, b):
    """Paired test on items where BOTH systems commit (coverage-matched)."""
    m = np.isin(a, COMMITTED) & np.isin(b, COMMITTED)
    ca, cb = (a[m] == gold[m]), (b[m] == gold[m])
    x, y = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
    chi = (abs(x - y) - 1) ** 2 / (x + y) if (x + y) else 0.0
    return int(m.sum()), x, y, float(erfc(sqrt(chi) / sqrt(2)))


def cost_per_1k(n_llm_calls, n_reviews):
    usd = (TOK_IN * n_llm_calls) / 1e6 * PRICE_IN + \
          (TOK_OUT * n_llm_calls) / 1e6 * PRICE_OUT
    return usd / n_reviews * 1000


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    R = args.results_dir
    out = []

    def say(line=""):
        print(line)
        out.append(line)

    def cache(name):
        with open(os.path.join(R, ".cache", name + ".pkl"), "rb") as f:
            return pickle.load(f)

    def load(name):
        with open(os.path.join(R, name + ".pkl"), "rb") as f:
            return np.array(pickle.load(f), dtype=object)

    aspects = cache("aspects_df")
    cached_preds, routing = cache("routing_results")
    roberta_preds, roberta_conf, _ = cache("roberta_results")
    vader = cache("vader_results")

    gold = np.array(aspects["ground_truth"].tolist(), dtype=object)
    aspect_cat = np.array(aspects["aspect"].tolist())
    review_id = np.array(aspects["review_idx"].tolist())
    contexts = aspects["context"].tolist()

    cached_preds = np.array(cached_preds, dtype=object)
    routing = np.array(routing)
    roberta = np.array(roberta_preds, dtype=object)
    conf = np.array(roberta_conf)
    vader_aspect = np.array(vader[2], dtype=object)

    fresh_escalated = load("escalated_fresh_predictions")   # escalated items
    llm_non_escalated = load("llm_all_predictions")         # everything else

    escalated = routing == "llm"
    n_reviews = int(len(set(review_id.tolist())))
    sarcasm = np.array([any(p in t.lower() for p in SARCASM_PATTERNS)
                        for t in contexts])

    # Configurations, all built from clean predictions -----------------
    cfg = {
        "VADER-only": vader_aspect,
        "Twitter-RoBERTa (zero-shot)": roberta,
        "AspectRoute (VADER cheap path)": np.where(escalated, fresh_escalated, vader_aspect),
        "AspectRoute (RoBERTa cheap path)": np.where(escalated, fresh_escalated, roberta),
        "LLM-all": np.where(escalated, fresh_escalated, llm_non_escalated),
    }
    llm_calls = {
        "VADER-only": 0,
        "Twitter-RoBERTa (zero-shot)": 0,
        "AspectRoute (VADER cheap path)": int(escalated.sum()),
        "AspectRoute (RoBERTa cheap path)": int(escalated.sum()),
        "LLM-all": len(routing),
    }
    headline = cfg["AspectRoute (RoBERTa cheap path)"]

    groups = defaultdict(list)
    for i, r in enumerate(review_id):
        groups[r].append(i)

    say("=" * 92)
    say("ASPECTROUTE - CORRECTED RESULTS (regenerated from clean predictions)")
    say("=" * 92)
    say(f"Aspect mentions: {len(gold):,}   Reviews: {n_reviews:,}   "
        f"Escalated: {escalated.sum():,} ({100*escalated.mean():.1f}%)")

    # 0. Contamination evidence ---------------------------------------
    say()
    say("0. CONTAMINATION EVIDENCE (why results were regenerated)")
    say("-" * 92)
    old_commit = np.isin(cached_preds[escalated], COMMITTED).mean()
    new_commit = np.isin(fresh_escalated[escalated], COMMITTED).mean()
    both = np.isin(cached_preds[escalated], COMMITTED) & \
           np.isin(fresh_escalated[escalated], COMMITTED)
    agree = (cached_preds[escalated][both] == fresh_escalated[escalated][both]).mean()
    say(f"  Commit rate on escalated items, original cached run : {100*old_commit:5.1f}%")
    say(f"  Commit rate on escalated items, regenerated run     : {100*new_commit:5.1f}%")
    say(f"  Agreement where both runs committed                 : {100*agree:5.1f}%  (n={both.sum()})")
    say("  Same model, same prompt, same items. Perfect agreement on committed")
    say("  labels indicates the difference is failed calls silently stored as")
    say("  'neutral', not a change in model behaviour.")

    # 1. Main table ----------------------------------------------------
    say()
    say("1. OVERALL RESULTS")
    say("-" * 92)
    say(f"{'Configuration':<34}{'Cov-F1':<8}{'95% CI':<18}"
        f"{'Cover%':<9}{'CommitAcc':<11}{'BinF1':<8}{'Cost/1K'}")
    for name, pred in cfg.items():
        f1, cov = coverage_f1(gold, pred)
        lo, hi = review_bootstrap_ci(gold, pred, groups, args.bootstrap)
        say(f"{name:<34}{f1:<8.3f}[{lo:.3f}, {hi:.3f}]   "
            f"{100*cov:<9.1f}{100*committed_accuracy(gold, pred):<11.1f}"
            f"{binary_f1(gold, pred):<8.3f}${cost_per_1k(llm_calls[name], n_reviews):.3f}")
    say("  CIs use review-level bootstrap (whole reviews resampled).")

    # 2. Significance --------------------------------------------------
    say()
    say("2. PAIRED SIGNIFICANCE (McNemar, items where both systems commit)")
    say("-" * 92)
    for label, other in [
        ("vs LLM-all", cfg["LLM-all"]),
        ("vs Twitter-RoBERTa (zero-shot)", cfg["Twitter-RoBERTa (zero-shot)"]),
        ("vs AspectRoute (VADER cheap path)", cfg["AspectRoute (VADER cheap path)"]),
    ]:
        n, x, y, p = mcnemar_committed(gold, headline, other)
        say(f"  RoBERTa-cheap {label:<36} n={n:<6} b={x:<5} c={y:<5} "
            f"p={p:.2e}  {'significant' if p < 0.05 else 'n.s.'}")
    say("  b = headline correct & other wrong; c = the reverse.")

    # 3. Per-aspect ----------------------------------------------------
    say()
    say("3. PER-ASPECT PERFORMANCE (headline configuration)")
    say("-" * 92)
    say(f"{'Aspect':<22}{'Cov-F1':<9}{'Prec':<9}{'Rec':<9}{'%LLM':<8}{'N'}")
    wsum = tot = 0.0
    for a in ["narrative", "performance", "visual", "audio",
              "emotional_impact", "cultural_relevance", "technical"]:
        m = aspect_cat == a
        f1, _ = coverage_f1(gold[m], headline[m])
        pr, rc = precision_recall(gold[m], headline[m])
        say(f"{a.replace('_',' ').title():<22}{f1:<9.3f}{pr:<9.3f}{rc:<9.3f}"
            f"{100*escalated[m].mean():<8.0f}{m.sum():,}")
        wsum += f1 * m.sum()
        tot += m.sum()
    pooled, _ = coverage_f1(gold, headline)
    say(f"  N-weighted mean {wsum/tot:.3f} vs pooled {pooled:.3f} "
        f"(macro-F1 is nonlinear, so these need not match exactly)")

    # 4. IASD ----------------------------------------------------------
    say()
    say("4. INTER-ASPECT SENTIMENT DIVERGENCE")
    say("-" * 92)
    multi = divergent = 0
    patterns = Counter()
    for r, idx in groups.items():
        if len(idx) < 2:
            continue
        multi += 1
        labels = {aspect_cat[i]: headline[i] for i in idx
                  if headline[i] in COMMITTED}
        vals = set(labels.values())
        if POS in vals and NEG in vals:
            divergent += 1
            for pa in sorted(a for a, l in labels.items() if l == POS):
                for na in sorted(a for a, l in labels.items() if l == NEG):
                    patterns[(pa, na)] += 1
    say(f"  Multi-aspect reviews: {multi:,}   Divergent: {divergent:,}   "
        f"IASD = {100*divergent/multi:.1f}%")
    for (pa, na), c in patterns.most_common(5):
        say(f"    +{pa:<20} / -{na:<20} {c:>5}  ({100*c/divergent:.1f}% of divergent)")

    # 5. Threshold sensitivity ----------------------------------------
    say()
    say("5. THRESHOLD SENSITIVITY (headline configuration)")
    say("-" * 92)
    say(f"{'tau':<8}{'Cov-F1':<10}{'Coverage%':<12}{'%LLM':<9}{'Cost/1K'}")
    for tau in (0.70, 0.75, 0.80, 0.85):
        route = (conf < tau) | sarcasm
        f1, cov = coverage_f1(gold, np.where(route, fresh_escalated, roberta))
        say(f"{tau:<8.2f}{f1:<10.3f}{100*cov:<12.1f}{100*route.mean():<9.1f}"
            f"${cost_per_1k(int(route.sum()), n_reviews):.3f}")
    say("  Exact, not interpolated: every item escalated at tau <= 0.85 has a")
    say("  regenerated LLM prediction. tau > 0.85 would require further calls.")

    # 6. Ablations -----------------------------------------------------
    say()
    say("6. ABLATIONS")
    say("-" * 92)
    base, _ = coverage_f1(gold, headline)
    rng = np.random.default_rng(42)
    random_route = rng.permutation(len(escalated)) < escalated.sum()
    rows = [
        ("AspectRoute (RoBERTa cheap path, full)", headline),
        ("A2: random routing at same 57.5% split", np.where(random_route, fresh_escalated, roberta)),
        ("A4: no LLM (RoBERTa only)", roberta),
        ("A4b: no LLM (VADER only)", vader_aspect),
        ("A5: LLM-all (measured, not a proxy)", cfg["LLM-all"]),
        ("VADER cheap path variant", cfg["AspectRoute (VADER cheap path)"]),
    ]
    say(f"{'Configuration':<44}{'Cov-F1':<10}{'Delta'}")
    for name, pred in rows:
        f1, _ = coverage_f1(gold, pred)
        say(f"{name:<44}{f1:<10.3f}{'--' if pred is headline else f'{f1-base:+.3f}'}")
    say("  A1 (no domain lexicon) and A3 (no aspect windowing) require")
    say("  re-extraction rather than re-prediction and are not recomputable here.")

    # 7. Routing signal decomposition ----------------------------------
    say()
    say("7. ROUTING SIGNAL DECOMPOSITION")
    say("-" * 92)
    low = conf < 0.85
    n_esc = escalated.sum()
    say(f"  Sarcasm patterns match {sarcasm.sum():,} of {len(sarcasm):,} contexts "
        f"({100*sarcasm.mean():.1f}%)")
    say(f"    low confidence only      : {int((low & ~sarcasm & escalated).sum()):>6,} "
        f"({100*(low & ~sarcasm & escalated).sum()/n_esc:4.1f}% of escalations)")
    say(f"    both triggers            : {int((low & sarcasm & escalated).sum()):>6,} "
        f"({100*(low & sarcasm & escalated).sum()/n_esc:4.1f}%)")
    say(f"    sarcasm pattern only     : {int((~low & sarcasm & escalated).sum()):>6,} "
        f"({100*(~low & sarcasm & escalated).sum()/n_esc:4.1f}%)")
    never = [p for p in SARCASM_PATTERNS
             if not any(p in t.lower() for t in contexts)]
    say(f"  Patterns that never matched ({len(never)} of {len(SARCASM_PATTERNS)}): "
        f"{', '.join(never)}")

    path = os.path.join(R, "corrected_results.txt")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
