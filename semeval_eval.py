"""
SemEval-2014 Task 4 Evaluation for AspectStreamSA
===================================================
Runs the pipeline on the SemEval-2014 ABSA benchmark (restaurants + laptops).
Unlike IMDB, SemEval has ASPECT-LEVEL sentiment labels (positive/negative/neutral)
so no evaluation mismatch — this is the proper 3-class benchmark.

Usage:
    python semeval_eval.py --openai_key sk-YOUR_KEY
    python semeval_eval.py                           # uses local fallback, no API

Outputs:
    results/semeval_overall.txt      — 3-class metrics vs all baselines
    results/semeval_per_aspect.txt   — per SemEval aspect category breakdown
    results/semeval_confusion.txt    — confusion matrices
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from aspectstreamsa import (
    setup_vader, vader_classify, DOMAIN_LEXICON,
    detect_sarcasm_signals, setup_roberta, setup_local_llm,
    llm_classify_openai, llm_classify_anthropic, llm_classify_local
)


# ── SemEval-2014 aspect category mapping ─────────────────────────────────
# Maps SemEval aspect categories to human-readable labels for reporting
SEMEVAL_ASPECT_MAP = {
    # Restaurants
    "FOOD":       "Food Quality",
    "SERVICE":    "Service",
    "AMBIENCE":   "Ambience",
    "PRICE":      "Price",
    "ANECDOTES/MISCELLANEOUS": "Miscellaneous",
    # Laptops
    "BATTERY":    "Battery",
    "DISPLAY":    "Display",
    "KEYBOARD":   "Keyboard",
    "MOUSE":      "Mouse",
    "MOTHERBOARD":"Motherboard",
    "CPU":        "CPU",
    "MEMORY":     "Memory",
    "HARD DISC":  "Storage",
    "GRAPHICS":   "Graphics",
    "SOFTWARE":   "Software",
    "OS":         "OS",
    "WARRANTY":   "Warranty",
    "SHIPPING":   "Shipping",
    "SUPPORT":    "Support",
    "COMPANY":    "Company",
    "LAPTOP":     "Laptop (general)",
    "PORTS":      "Ports",
    "FANS/COOLING":"Cooling",
    "OPTICAL DRIVES": "Optical Drives",
    "POWER SUPPLY": "Power Supply",
}

POLARITY_MAP = {"positive": "positive", "negative": "negative",
                "neutral": "neutral", "conflict": "neutral"}


# ── Data loading ──────────────────────────────────────────────────────────

def load_from_huggingface(domain="restaurant"):
    """Try to load SemEval-2014 from HuggingFace datasets."""
    from datasets import load_dataset

    # Most reliable HuggingFace source for SemEval-2014 ABSA
    candidates = [
        ("tommasobonomo/sem_eval_2014_aspect_sentiment", domain),
        ("Nayan09/semeval-2014-task4", None),
    ]

    for dataset_name, config in candidates:
        try:
            print(f"   Trying HuggingFace: {dataset_name}...")
            ds = load_dataset(dataset_name, config) if config else load_dataset(dataset_name)
            return ds, dataset_name
        except Exception:
            continue
    return None, None


def load_from_xml(xml_path):
    """Parse SemEval-2014 XML format."""
    records = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for sentence in root.findall(".//sentence"):
        text_el = sentence.find("text")
        if text_el is None or text_el.text is None:
            continue
        text = text_el.text.strip()

        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is None:
            continue

        for term in aspect_terms.findall("aspectTerm"):
            polarity = term.get("polarity", "neutral")
            if polarity == "conflict":
                polarity = "neutral"  # treat conflict as neutral
            aspect_word = term.get("term", "")

            # Extract context: the sentence containing the aspect
            records.append({
                "text": text,
                "aspect_term": aspect_word,
                "aspect_category": "general",
                "polarity": polarity,
                "context": text  # full sentence as context (sentences are short in SemEval)
            })

    return pd.DataFrame(records)


def download_semeval_xml(domain="restaurants"):
    """Download SemEval-2014 XML from GitHub mirror."""
    import urllib.request

    urls = {
        "restaurants": [
            "https://raw.githubusercontent.com/ThomasK427/aspect-based-sentiment-analysis-data/master/SemEval-2014-Task-4/Restaurants_Train_v2.xml",
            "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/Restaurants_Train.xml",
        ],
        "laptops": [
            "https://raw.githubusercontent.com/ThomasK427/aspect-based-sentiment-analysis-data/master/SemEval-2014-Task-4/Laptop_Train_v2.xml",
            "https://raw.githubusercontent.com/songyouwei/ABSA-PyTorch/master/datasets/semeval14/Laptop_Train.xml",
        ]
    }

    for url in urls.get(domain, []):
        try:
            print(f"   Downloading: {url}")
            tmp_path = f"/tmp/semeval_{domain}.xml"
            urllib.request.urlretrieve(url, tmp_path)
            df = load_from_xml(tmp_path)
            if len(df) > 0:
                print(f"   Loaded {len(df)} aspect mentions from XML")
                return df
        except Exception as e:
            print(f"   Failed: {e}")
            continue
    return None


def load_from_txt(txt_path):
    """
    Parse ABSADatasets .txt format:
    sentence####[([aspect_word_indices], [opinion_word_indices], 'SENTIMENT')]
    """
    import ast
    records = []
    pol_map = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}

    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "####" not in line:
                continue
            try:
                sentence, triplets_str = line.split("####", 1)
                words = sentence.strip().split()
                triplets = ast.literal_eval(triplets_str.strip())

                for triplet in triplets:
                    asp_indices, op_indices, sentiment = triplet
                    # Extract aspect term from word indices
                    asp_words = [words[i] for i in asp_indices if i < len(words)]
                    aspect_term = " ".join(asp_words)
                    polarity = pol_map.get(sentiment, "neutral")

                    records.append({
                        "text": sentence.strip(),
                        "aspect_term": aspect_term,
                        "aspect_category": "general",
                        "polarity": polarity,
                        "context": sentence.strip()
                    })
            except Exception:
                continue

    return pd.DataFrame(records)


def load_semeval(domain="restaurants"):
    """Load SemEval-2014 from the cloned ABSADatasets repo."""
    print(f"\n[1] Loading SemEval-2014 Task 4 ({domain})...")

    # Primary: use the cloned ABSADatasets repo txt files
    txt_paths = {
        "restaurants": [
            "ABSADatasets/datasets/aste_datasets/400.SemEval/402.Restaurant14/train.txt",
            "ABSADatasets/datasets/aste_datasets/400.SemEval/402.Restaurant14/test.txt",
        ],
        "laptops": [
            "ABSADatasets/datasets/aste_datasets/400.SemEval/401.Laptop14/train.txt",
            "ABSADatasets/datasets/aste_datasets/400.SemEval/401.Laptop14/test.txt",
        ],
        "both": [
            "ABSADatasets/datasets/aste_datasets/400.SemEval/402.Restaurant14/train.txt",
            "ABSADatasets/datasets/aste_datasets/400.SemEval/402.Restaurant14/test.txt",
            "ABSADatasets/datasets/aste_datasets/400.SemEval/401.Laptop14/train.txt",
            "ABSADatasets/datasets/aste_datasets/400.SemEval/401.Laptop14/test.txt",
        ]
    }

    all_records = []
    for path in txt_paths.get(domain, txt_paths["restaurants"]):
        if os.path.exists(path):
            df_part = load_from_txt(path)
            if len(df_part) > 0:
                all_records.append(df_part)
                print(f"   Loaded {len(df_part)} records from {path.split('/')[-1]}")
        else:
            print(f"   Not found: {path}")

    if all_records:
        df = pd.concat(all_records, ignore_index=True)
        print(f"   Total: {len(df)} aspect mentions")
        return df

    # Fallback: check for local XML files
    for fname in [f"{domain}.xml", "Restaurants_Train.xml", "Laptop_Train.xml"]:
        if os.path.exists(fname):
            print(f"   Using local XML: {fname}")
            return load_from_xml(fname)

    print("\n" + "="*60)
    print("ERROR: ABSADatasets not found.")
    print("Run this first:")
    print("   git clone https://github.com/yangheng95/ABSADatasets.git --depth 1")
    print("="*60)
    sys.exit(1)
    print(f"\n[1] Loading SemEval-2014 Task 4 ({domain})...")

    # Strategy 1: pyabsa (bundles SemEval-2014, most reliable)
    try:
        print("   Trying pyabsa...")
        from pyabsa import DatasetItem
        from pyabsa.utils.data_utils.dataset_manager import DatasetItem as DI
        import pyabsa
        # pyabsa stores downloaded data in ~/.pyabsa/
        dataset_path = os.path.expanduser(
            f"~/.pyabsa/datasets/SemEval-{'Restaurants' if domain=='restaurants' else 'Laptops'}"
        )
        if not os.path.exists(dataset_path):
            print("   Downloading SemEval via pyabsa (one-time)...")
            from pyabsa import download_all_available_datasets
            download_all_available_datasets()
        # Find XML files in pyabsa cache
        for root, dirs, files in os.walk(os.path.expanduser("~/.pyabsa")):
            for f in files:
                if f.endswith(".xml") and domain[:3].lower() in f.lower():
                    df = load_from_xml(os.path.join(root, f))
                    if len(df) > 0:
                        print(f"   Loaded {len(df)} records via pyabsa")
                        return df
    except Exception as e:
        print(f"   pyabsa failed: {e}")

    # Strategy 2: Working GitHub mirrors (multiple fallbacks)
    url_sets = {
        "restaurants": [
            "https://raw.githubusercontent.com/NUSTM/ABSA-BERT-pair/master/data/semeval-2014/restaurant/train.xml",
            "https://raw.githubusercontent.com/NUSTM/ABSA-BERT-pair/master/data/semeval-2014/restaurant/test.xml",
            "https://raw.githubusercontent.com/madrugado/semeval-2016-task-5/master/data/EN_REST_SB1_TEST.xml.gold",
        ],
        "laptops": [
            "https://raw.githubusercontent.com/NUSTM/ABSA-BERT-pair/master/data/semeval-2014/laptop/train.xml",
            "https://raw.githubusercontent.com/NUSTM/ABSA-BERT-pair/master/data/semeval-2014/laptop/test.xml",
        ]
    }

    import urllib.request
    all_records = []
    for url in url_sets.get(domain, []):
        try:
            print(f"   Trying: {url.split('/')[-1]}")
            tmp = f"/tmp/se14_{domain}_{url.split('/')[-1]}"
            urllib.request.urlretrieve(url, tmp)
            # Check not a 404 page
            with open(tmp) as f:
                first = f.read(50)
            if "404" in first or "Not Found" in first:
                continue
            df_part = load_from_xml(tmp)
            if len(df_part) > 0:
                all_records.append(df_part)
                print(f"   Got {len(df_part)} records")
        except Exception as e:
            print(f"   Failed: {e}")

    if all_records:
        df = pd.concat(all_records, ignore_index=True)
        print(f"   Total: {len(df)} aspect mentions")
        return df

    # Strategy 3: HuggingFace
    for ds_name, config in [
        ("HamidRezaAttar/SemEval-2014-task-4-Restaurant-Reviews", None),
        ("Nayan09/semeval-2014-task4", None),
        ("DataCompass/SemEval-2014-ABSA", None),
    ]:
        try:
            from datasets import load_dataset
            print(f"   Trying HuggingFace: {ds_name}...")
            ds = load_dataset(ds_name, config) if config else load_dataset(ds_name)
            records = []
            for split in ds.keys():
                for row in ds[split]:
                    p = POLARITY_MAP.get(str(row.get("polarity",
                        row.get("sentiment","neutral"))).lower(), "neutral")
                    text = str(row.get("text", row.get("sentence",
                        row.get("review",""))))
                    records.append({
                        "text": text,
                        "aspect_term": str(row.get("term",
                            row.get("aspect_term", row.get("aspect","aspect")))),
                        "aspect_category": str(row.get("aspect",
                            row.get("category","general"))),
                        "polarity": p,
                        "context": text
                    })
            if records:
                df = pd.DataFrame(records)
                print(f"   Loaded {len(df)} records")
                return df
        except Exception as e:
            print(f"   Failed: {e}")

    # Strategy 4: local XML file
    for fname in [f"{domain}.xml", f"{domain}_train.xml",
                  "Restaurants_Train.xml", "Laptop_Train.xml"]:
        if os.path.exists(fname):
            print(f"   Found local file: {fname}")
            return load_from_xml(fname)

    # All failed
    print("\n" + "="*60)
    print("All auto-download methods failed.")
    print("Manual fix (2 minutes):\n")
    print("Step 1 — install pyabsa and let it download for you:")
    print("   pip install pyabsa")
    print("   python -c \"from pyabsa import download_all_available_datasets; download_all_available_datasets()\"")
    print("   python semeval_eval.py --domain restaurants")
    print()
    print("OR Step 2 — download file manually:")
    print("   Open: https://github.com/NUSTM/ABSA-BERT-pair/tree/master/data/semeval-2014/restaurant")
    print("   Download train.xml → rename to restaurants.xml")
    print(f"   Place in: ~/Desktop/paper/AspectStreamSA_Code/")
    print("   Re-run: python semeval_eval.py --domain restaurants")
    print("="*60)
    sys.exit(1)


# ── Classification methods ────────────────────────────────────────────────

def run_vader_semeval(df, analyzer, use_context=True):
    """VADER on SemEval aspect contexts."""
    preds, scores = [], []
    for _, row in df.iterrows():
        text = row["context"] if use_context else row["text"]
        label, score = vader_classify(text, analyzer)
        preds.append(label)
        scores.append(score)
    return preds, scores


def run_roberta_semeval(df, model, tokenizer, label_map, device="cpu"):
    """RoBERTa on SemEval aspect contexts."""
    import torch
    preds, confs, all_probs = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="   RoBERTa"):
        text = row["context"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, predicted = torch.max(probs, dim=-1)

        preds.append(label_map[predicted.item()])
        confs.append(confidence.item())
        all_probs.append(probs.cpu().numpy()[0])

    return preds, confs, all_probs


def run_routing_semeval(df, roberta_preds, roberta_confs,
                        vader_analyzer, llm_func, tau=0.85):
    """Confidence-gated routing on SemEval data."""
    print(f"\n[5] Confidence-gated routing (tau={tau})...")

    routes = []
    for i, (_, row) in enumerate(df.iterrows()):
        has_sarcasm, _ = detect_sarcasm_signals(row["context"])
        routes.append("llm" if (roberta_confs[i] < tau or has_sarcasm) else "vader")

    llm_idx   = [i for i, r in enumerate(routes) if r == "llm"]
    vader_idx  = [i for i, r in enumerate(routes) if r == "vader"]
    rows_list  = list(df.iterrows())

    print(f"   VADER: {len(vader_idx)} ({len(vader_idx)/len(routes)*100:.1f}%)")
    print(f"   LLM:   {len(llm_idx)}   ({len(llm_idx)/len(routes)*100:.1f}%)")

    final_preds = [None] * len(df)

    # VADER pass
    for i in vader_idx:
        _, row = rows_list[i]
        label, _ = vader_classify(row["context"], vader_analyzer)
        final_preds[i] = label

    # LLM pass — parallel
    def call_llm(i):
        _, row = rows_list[i]
        label, _ = llm_func(row["context"], row["aspect_term"])
        return i, label

    if llm_idx:
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(call_llm, i): i for i in llm_idx}
            for future in tqdm(as_completed(futures), total=len(llm_idx),
                               desc="   LLM calls"):
                i, label = future.result()
                final_preds[i] = label

    final_preds = [p if p else "neutral" for p in final_preds]
    return final_preds, routes


# ── Metrics ───────────────────────────────────────────────────────────────

def semeval_metrics(gt, preds, name, three_class=True):
    """
    Proper 3-class metrics since SemEval has true aspect-level labels.
    No neutral mapping needed — this is the clean evaluation.
    """
    labels = ["negative", "neutral", "positive"]

    macro_f1  = f1_score(gt, preds, labels=labels, average="macro",    zero_division=0)
    w_f1      = f1_score(gt, preds, labels=labels, average="weighted", zero_division=0)
    precision = precision_score(gt, preds, labels=labels, average="macro", zero_division=0)
    recall    = recall_score(gt, preds, labels=labels, average="macro",    zero_division=0)
    accuracy  = sum(a == b for a, b in zip(gt, preds)) / len(gt)

    return {
        "name": name,
        "macro_f1": macro_f1,
        "weighted_f1": w_f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "n": len(gt)
    }


def per_category_metrics(df, preds, name):
    """F1 per SemEval aspect category."""
    df = df.copy()
    df["pred"] = preds
    rows = []
    for cat, group in df.groupby("aspect_category"):
        gt2   = group["polarity"].tolist()
        pred2 = group["pred"].tolist()
        f1 = f1_score(gt2, pred2, labels=["negative","neutral","positive"],
                      average="macro", zero_division=0)
        display = SEMEVAL_ASPECT_MAP.get(cat.upper(), cat)
        rows.append([display, f"{f1:.3f}", len(group)])
    rows.sort(key=lambda x: -float(x[1]))
    return rows


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["restaurants","laptops","both"],
                        default="restaurants",
                        help="SemEval-2014 domain (default: restaurants)")
    parser.add_argument("--openai_key",    type=str, default=None)
    parser.add_argument("--anthropic_key", type=str, default=None)
    parser.add_argument("--tau",           type=float, default=0.85)
    parser.add_argument("--output_dir",    type=str, default="results")
    parser.add_argument("--use_gpu",       action="store_true")
    parser.add_argument("--test_only", action="store_true",
                        help="Evaluate on the SemEval TEST split only "
                             "(for like-for-like comparison vs fine-tuned RoBERTa).")
    args = parser.parse_args()

    import torch
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    domains = ["restaurants", "laptops"] if args.domain == "both" else [args.domain]
    all_results = {}

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"DOMAIN: {domain.upper()}")
        print(f"{'='*60}")

        # Load data
        if args.test_only:
            from semeval_eval import load_from_txt
            _base = "ABSADatasets/datasets/aste_datasets/400.SemEval"
            _testpath = {
                "restaurants": f"{_base}/402.Restaurant14/test.txt",
                "laptops":     f"{_base}/401.Laptop14/test.txt",
            }[domain]
            print(f"\n[1] Loading SemEval-2014 TEST split only ({domain})...")
            df = load_from_txt(_testpath)
            print(f"   Loaded {len(df)} test records from {_testpath.split('/')[-1]}")
        else:
            df = load_semeval(domain)
        df = df[df["polarity"].isin(["positive","negative","neutral"])].reset_index(drop=True)
        print(f"   Final: {len(df)} aspect mentions")
        print(f"   Label dist: {dict(df['polarity'].value_counts())}")

        gt = df["polarity"].tolist()

        # Setup models
        vader_analyzer = setup_vader()
        print("\n[2] Running VADER baselines...")
        vader_doc_preds,  _  = run_vader_semeval(df, vader_analyzer, use_context=False)
        vader_asp_preds,  _  = run_vader_semeval(df, vader_analyzer, use_context=True)

        print("\n[3] Running RoBERTa...")
        roberta_model, roberta_tok, roberta_labels = setup_roberta(device)
        roberta_preds, roberta_confs, _ = run_roberta_semeval(
            df, roberta_model, roberta_tok, roberta_labels, device
        )

        # LLM function
        if args.openai_key:
            print("\n   Using OpenAI API")
            llm_func = lambda t, a: llm_classify_openai(t, a, args.openai_key)
        elif args.anthropic_key:
            print("\n   Using Anthropic API")
            llm_func = lambda t, a: llm_classify_anthropic(t, a, args.anthropic_key)
        else:
            print("\n   Using local fallback model (no API key)")
            llm_model, llm_tok, llm_labels = setup_local_llm(device)
            llm_func = lambda t, a: llm_classify_local(t, a, llm_model, llm_tok, llm_labels, device)

        print("\n[4] Running AspectStreamSA routing...")
        stream_preds, routing_decisions = run_routing_semeval(
            df, roberta_preds, roberta_confs,
            vader_analyzer, llm_func, tau=args.tau
        )

        # Compute all metrics (TRUE 3-CLASS — no mapping needed)
        print("\n[6] Computing metrics (true 3-class)...")
        results = {
            "VADER-Only":    semeval_metrics(gt, vader_doc_preds, "VADER-Only"),
            "VADER-Aspect":  semeval_metrics(gt, vader_asp_preds, "VADER-Aspect"),
            "RoBERTa-FT":    semeval_metrics(gt, roberta_preds,   "RoBERTa-FT"),
            "AspectStreamSA":semeval_metrics(gt, stream_preds,    "AspectStreamSA"),
        }
        all_results[domain] = (results, df, stream_preds, routing_decisions)

        # dump predictions for the stats harness
        import pickle as _pkl
        if "aspect" in df.columns:
            _aspects = df["aspect"].tolist()
        elif "category" in df.columns:
            _aspects = df["category"].tolist()
        else:
            _aspects = [None] * len(df)
        _dump = {}
        _dump["domain"] = domain
        _dump["gt"] = list(gt)
        _dump["vader_doc_preds"] = list(vader_doc_preds)
        _dump["vader_asp_preds"] = list(vader_asp_preds)
        _dump["roberta_preds"] = list(roberta_preds)
        _dump["roberta_confs"] = [float(x) for x in roberta_confs]
        _dump["stream_preds"] = list(stream_preds)
        _dump["routing_decisions"] = list(routing_decisions)
        _dump["aspects"] = _aspects
        _suffix = "_test" if args.test_only else ""
        _dpath = os.path.join(args.output_dir, "semeval_" + domain + _suffix + "_preds.pkl")
        with open(_dpath, "wb") as _f:
            _pkl.dump(_dump, _f)
        print("   [stats] dumped predictions to " + _dpath)

        # Print table
        rows = [[r["name"], f"{r['macro_f1']:.3f}", f"{r['weighted_f1']:.3f}",
                 f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                 f"{r['accuracy']:.3f}"] for r in results.values()]
        headers = ["Method", "Macro-F1", "W-F1", "Precision", "Recall", "Accuracy"]
        table = tabulate(rows, headers=headers, tablefmt="grid")

        print(f"\nSemEval-2014 {domain.title()} Results (TRUE 3-CLASS)")
        print("="*70)
        print(table)

        # Per-category
        cat_rows = per_category_metrics(df, stream_preds, "AspectStreamSA")
        cat_table = tabulate(cat_rows, headers=["Aspect Category", "Macro-F1", "N"],
                             tablefmt="grid")
        print(f"\nPer-Category F1 (AspectStreamSA):")
        print(cat_table)

        # Classification report
        print(f"\nDetailed classification report (AspectStreamSA):")
        print(classification_report(gt, stream_preds,
              labels=["negative","neutral","positive"], zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(gt, stream_preds,
                              labels=["negative","neutral","positive"])
        print("Confusion matrix (rows=true, cols=pred):")
        print(f"             neg   neu   pos")
        for label, row_data in zip(["negative","neutral","positive"], cm):
            print(f"  {label:10s}  {row_data[0]:4d}  {row_data[1]:4d}  {row_data[2]:4d}")

        # Save
        out_path = os.path.join(args.output_dir, f"semeval_{domain}_results.txt")
        with open(out_path, "w") as f:
            f.write(f"SemEval-2014 Task 4 — {domain.title()} (True 3-Class Evaluation)\n")
            f.write("="*70 + "\n")
            f.write(table + "\n\n")
            f.write("Per-Category F1 (AspectStreamSA):\n")
            f.write(cat_table + "\n\n")
            f.write("Classification Report (AspectStreamSA):\n")
            f.write(classification_report(gt, stream_preds,
                    labels=["negative","neutral","positive"], zero_division=0))
            f.write("\nRouting: VADER=%.1f%%  LLM=%.1f%%\n" % (
                sum(1 for r in routing_decisions if r=="vader") / len(routing_decisions) * 100,
                sum(1 for r in routing_decisions if r=="llm")   / len(routing_decisions) * 100
            ))

        print(f"\nSaved to {out_path}")

    print(f"\n{'='*60}")
    print("All done. Paste results into paper Table 2 (SemEval evaluation).")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()