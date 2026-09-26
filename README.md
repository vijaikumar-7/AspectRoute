# AspectRoute

Code, data and verification harness for **"AspectRoute: Cost-Aware
Confidence-Gated Escalation to LLMs for Aspect-Based Sentiment Analysis of
Entertainment Reviews"** (submitted to the *Journal of Information Science and
Engineering*, manuscript 260570).

AspectRoute decides, per aspect mention, whether large language model (LLM)
inference is warranted. Aspect terms are extracted by a keyword pipeline with
contextual windowing; each aspect–context pair is escalated to an LLM only when a
RoBERTa encoder is insufficiently confident, and all remaining pairs are classified
locally at no API cost.

---

## Reproducing the reported results

Every figure in the paper can be regenerated from cached predictions **without
making any API calls**:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 recompute_all.py
```

This takes about a minute and writes `results/corrected_results.txt`, which
contains the overall results table, paired significance tests, the per-aspect
breakdown, IASD, the threshold-sensitivity analysis, the ablations, and the
routing-signal decomposition — that is, Tables 1–5 and Sections 5.1–5.9 of the
paper.

---

## A correction to the original evaluation

**Please read this section before drawing conclusions from the cached
predictions in `results/.cache/`.**

The original pipeline contained a fault. In `aspectstreamsa.py`, the LLM
classification function caught every exception and returned a neutral label:

```python
except Exception as e:
    return "neutral", 0.5
```

Under API rate limiting, failed calls were therefore recorded as neutral
predictions rather than as errors. The effect was substantial: on escalated
items, the original run commits to a non-neutral label 6.7% of the time, whereas
re-running the same model with the same prompt over the same items commits 62.2%
of the time, with 99.7% agreement wherever both runs committed (n = 1,460). The
near-perfect agreement on committed labels indicates that the difference is failed
calls, not model variability.

All 23,763 escalated predictions were regenerated with corrected error handling
(zero failures, zero missing items), and every IMDB result in the paper was
recomputed from them. The SemEval evaluation was unaffected: it required far fewer
calls and its escalated items committed at 80–88%.

**`aspectstreamsa.py` is published as it ran, with the fault intact and a header comment added to flag it**, so
that this account can be verified independently. `regenerate_escalated.py`
contains the corrected error handling (exponential backoff on rate limits, no
silent fallback), and `recompute_all.py` prints the contamination evidence
alongside the corrected results.

---

## Repository contents

### Pipeline

| File | Purpose |
|---|---|
| `aspectstreamsa.py` | Original IMDB pipeline. **Contains the fault described above; published unmodified.** |
| `semeval_eval.py` | SemEval-2014 Task 4 evaluation. Use `--test_only` for the held-out test split reported in the paper. |
| `ablation.py` | Ablation configurations. |

### Revision scripts

| File | Purpose | API calls |
|---|---|---|
| `recompute_all.py` | Regenerates every reported figure from cache. | none |
| `regenerate_escalated.py` | Re-runs escalated items with corrected error handling. | ~23.8k |
| `complete_llm_all.py` | Completes the LLM-all baseline over non-escalated items. | ~17.5k |
| `cross_backbone.py` | Paired comparison of a second LLM backbone on identical items. | configurable |
| `generate_shap_examples.py` | Token-level attribution examples. | none |

Scripts that make API calls read the key from the `OPENAI_API_KEY` environment
variable, write results incrementally to JSONL, and resume safely if interrupted.
None writes a fallback label on failure.

### Data

| Path | Contents |
|---|---|
| `results/.cache/*.pkl` | Original cached predictions, including the contaminated escalated predictions. Retained for verification. |
| `results/escalated_fresh_predictions.pkl` | Regenerated escalated predictions (23,763 items). |
| `results/llm_all_predictions.pkl` | LLM predictions for non-escalated items, forming the LLM-all baseline. |
| `results/corrected_results.txt` | Output of `recompute_all.py`. |
| `results/shap_examples_revision.txt` | Attribution examples, including the failure case discussed in Section 5.7. |

---

## Headline results

On 41,315 aspect mentions from 9,916 IMDB reviews:

| Configuration | Coverage-F1 | 95% CI | Coverage | Cost / 1K |
|---|---|---|---|---|
| VADER only | 0.652 | [.645, .658] | 79.8% | $0.000 |
| Twitter-RoBERTa (zero-shot) | 0.771 | [.764, .777] | 74.6% | $0.000 |
| AspectRoute (lexical local path) | 0.786 | [.780, .792] | 72.2% | $0.123 |
| **AspectRoute (encoder local path)** | **0.823** | **[.818, .828]** | **74.4%** | **$0.123** |
| Escalate everything | 0.826 | [.820, .831] | 67.0% | $0.213 |

Escalating the same 57.5% of mentions **at random** yields 0.774, so the benefit
comes from *which* mentions are escalated rather than from LLM use alone.

Costs are computed from measured token counts (mean 132.3 input, 52.2 output
tokens per call) at GPT-4o-mini pricing, not from an assumed constant.

---

## Known limitations

These are stated in the paper and repeated here so that users of the code are not
misled:

- The aspect taxonomy and the three hand-built lexicons were constructed without a
  formal coding protocol, independent coders, or inter-annotator agreement.
- The sarcasm component is a substring matcher, not a sarcasm detector. Five of its
  twenty patterns never match on this corpus, and 98.1% of matches come from
  generic discourse markers ordinarily used sincerely. Its precision and recall are
  unmeasured, as the corpus provides no sarcasm ground truth.
- The escalation threshold was selected on the same IMDB sample used for
  evaluation, not a held-out split.
- IMDB provides document-level labels, so the IMDB evaluation cannot fully validate
  aspect-level predictions. The SemEval evaluation is included to address this.

## Citation

Citation details will be added if the paper is accepted.

---

## Supplementary material

Three items referenced in the paper are held here rather than in the manuscript,
to keep the submission within the journal's page allowance. See
[SUPPLEMENTARY.md](SUPPLEMENTARY.md) for:

- **Per-aspect performance** — Coverage-F1, precision, recall and escalation rate
  for each of the seven aspect categories.
- **Pipeline architecture figure** — `architecture.png`, with TikZ source in
  `architecture.tex`.
- **Annotation protocol** — an executable protocol for constructing the
  aspect-level benchmark proposed in Section 7, including corpus, schema,
  procedure and Fleiss' κ target.
- **LLM prompt template** — the full zero-shot prompt used for all escalated
  mentions.

## Licence

MIT. See [LICENSE](LICENSE).
