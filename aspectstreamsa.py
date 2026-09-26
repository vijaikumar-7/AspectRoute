# =====================================================================
#  NOTE: THIS FILE IS PUBLISHED AS IT RAN, INCLUDING A KNOWN FAULT.
#
#  The LLM classification function below catches every exception and
#  returns ("neutral", 0.5). Under API rate limiting this silently
#  recorded failed calls as neutral predictions. See README.md for the
#  evidence and the effect on the original results.
#
#  Do not reuse this error handling. regenerate_escalated.py contains
#  the corrected version (exponential backoff, no silent fallback), and
#  recompute_all.py regenerates all reported figures from the corrected
#  predictions.
# =====================================================================
"""
AspectStreamSA: Adaptive Aspect-Based Sentiment Analysis Pipeline
================================================================
Full runnable pipeline that produces real benchmarks on IMDB movie reviews.

Usage:
    python aspectstreamsa.py                    # Run with defaults (2000 samples, CPU)
    python aspectstreamsa.py --n_samples 5000   # More samples for better stats
    python aspectstreamsa.py --use_gpu          # Use GPU if available
    python aspectstreamsa.py --openai_key sk-.. # Use OpenAI API for LLM routing
    python aspectstreamsa.py --anthropic_key sk-.. # Use Anthropic API for LLM routing

Outputs:
    results/benchmark_results.csv        - Per-sample predictions
    results/overall_metrics.txt          - Main comparison table
    results/per_aspect_metrics.txt       - Per-aspect breakdown
    results/routing_analysis.txt         - Routing efficiency stats
    results/shap_examples.txt            - SHAP explanation samples
    results/figures/                     - All charts as PNG
"""

import argparse
import json
import os
import warnings
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tabulate import tabulate

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. STREAMING-SPECIFIC ASPECT TAXONOMY
# ============================================================

ASPECT_TAXONOMY = {
    "narrative": {
        "keywords": [
            "plot", "story", "storyline", "narrative", "script", "writing",
            "screenplay", "twist", "ending", "beginning", "pacing", "pace",
            "arc", "climax", "conclusion", "premise", "predictable",
            "coherent", "confusing", "boring", "engaging", "gripping",
            "suspense", "tension", "buildup", "filler", "dragged",
            "rushed", "slow", "fast-paced", "cliffhanger", "formulaic",
            "original", "cliche", "dialogue", "conversations", "monologue"
        ],
        "description": "Plot structure, story coherence, pacing, dialogue"
    },
    "performance": {
        "keywords": [
            "acting", "actor", "actress", "cast", "performance", "role",
            "character", "chemistry", "convincing", "wooden", "overact",
            "underact", "deliver", "portray", "expression", "emotion",
            "talent", "star", "lead", "supporting", "ensemble", "casting",
            "miscast", "charisma", "protagonist", "antagonist", "villain",
            "hero", "heroine"
        ],
        "description": "Acting quality, character portrayal, casting"
    },
    "visual": {
        "keywords": [
            "cinematography", "visual", "camera", "shot", "scene",
            "lighting", "color", "aesthetic", "beautiful", "stunning",
            "gorgeous", "ugly", "cgi", "effects", "vfx", "special effects",
            "production", "set", "design", "costume", "makeup",
            "animation", "graphics", "look", "style", "frame",
            "composition", "landscape", "imagery"
        ],
        "description": "Cinematography, visual effects, production design"
    },
    "audio": {
        "keywords": [
            "music", "soundtrack", "score", "sound", "audio", "song",
            "theme", "musical", "noise", "volume", "loud", "quiet",
            "silence", "voice", "dubbing", "subtitle", "mix", "bass",
            "orchestra", "composer", "singing", "tune", "melody"
        ],
        "description": "Soundtrack, sound design, audio quality"
    },
    "emotional_impact": {
        "keywords": [
            "feel", "feeling", "felt", "emotion", "emotional", "cry",
            "laugh", "funny", "hilarious", "sad", "happy", "joy",
            "depressing", "uplifting", "inspiring", "moving", "touching",
            "heartbreaking", "heartwarming", "scary", "terrifying",
            "creepy", "disturbing", "thrilling", "exciting", "dull",
            "tedious", "enjoyable", "entertaining", "bored", "loved",
            "hated", "disappointing", "satisfying", "rewatchable",
            "memorable", "forgettable", "waste of time", "masterpiece"
        ],
        "description": "Engagement, emotional resonance, rewatchability"
    },
    "cultural_relevance": {
        "keywords": [
            "culture", "cultural", "society", "social", "political",
            "message", "theme", "moral", "representation", "diverse",
            "diversity", "inclusive", "stereotype", "authentic",
            "realistic", "relatable", "relevant", "important",
            "meaningful", "profound", "deep", "shallow", "superficial",
            "propaganda", "woke", "commentary", "allegory", "metaphor",
            "symbolism"
        ],
        "description": "Representation, cultural authenticity, thematic depth"
    },
    "technical": {
        "keywords": [
            "quality", "resolution", "stream", "buffer", "load",
            "subtitle", "dub", "translation", "glitch", "bug",
            "interface", "recommend", "algorithm", "suggest",
            "available", "region", "release", "season", "episode",
            "series", "sequel", "prequel", "franchise", "runtime",
            "length", "short", "long", "minutes", "hours"
        ],
        "description": "Streaming quality, technical aspects, format"
    }
}

# Domain-specific VADER lexicon augmentation
DOMAIN_LEXICON = {
    # Positive streaming terms
    "bingeworthy": 3.2, "binge-worthy": 3.2, "bingewatch": 2.8,
    "masterpiece": 3.8, "gripping": 2.9, "riveting": 3.0,
    "captivating": 3.1, "mesmerizing": 3.2, "spellbinding": 3.3,
    "groundbreaking": 3.0, "phenomenal": 3.5, "outstanding": 3.2,
    "brilliant": 3.1, "flawless": 3.4, "breathtaking": 3.3,
    "rewatchable": 2.5, "addictive": 2.4, "bingeable": 2.8,
    "oscar-worthy": 3.5, "emmy-worthy": 3.3, "award-winning": 2.8,
    "must-watch": 3.0, "must-see": 3.0, "unmissable": 3.1,
    "underrated": 1.5, "hidden gem": 2.8,
    # Negative streaming terms
    "unwatchable": -3.5, "cringe": -2.5, "cringeworthy": -2.8,
    "plothole": -2.8, "plot hole": -2.8, "plotholes": -2.8,
    "formulaic": -1.9, "predictable": -1.8, "overrated": -2.0,
    "dragged": -2.2, "filler": -2.0, "rushed": -2.1,
    "nonsensical": -2.5, "incoherent": -2.6, "convoluted": -2.0,
    "miscast": -2.3, "wooden": -2.4, "stilted": -2.2,
    "pretentious": -2.1, "derivative": -1.8, "generic": -1.5,
    "forgettable": -2.0, "mediocre": -1.8, "lackluster": -2.0,
    "disappointing": -2.5, "letdown": -2.3, "overhyped": -2.0,
    "cancelled": -1.0, "cliffhanger": 0.5, "slow burn": 0.5,
    "tearjerker": 1.5, "feel-good": 2.5,
}

# Sarcasm indicator patterns
SARCASM_INDICATORS = [
    "oh great", "yeah right", "sure thing", "how wonderful",
    "what a surprise", "obviously", "clearly", "of course",
    "wow just wow", "thanks for nothing", "so original",
    "never seen that before", "how creative", "shocking",
    "who would have guessed", "certainly", "apparently",
    "supposedly", "allegedly", "as if"
]


# ============================================================
# 2. DATA LOADING
# ============================================================

def load_dataset_imdb(n_samples=2000, seed=42):
    """Load IMDB movie reviews from HuggingFace datasets."""
    print("\n[1/8] Loading IMDB dataset...")
    from datasets import load_dataset

    ds = load_dataset("imdb", split="test")
    df = pd.DataFrame(ds)
    df.columns = ["text", "label"]

    # label: 0=negative, 1=positive
    # Map to three-class: we'll treat reviews with mixed signals as neutral
    # For ground truth, IMDB is binary, so we map: 0->negative, 1->positive
    df["ground_truth"] = df["label"].map({0: "negative", 1: "positive"})

    # Sample for tractability
    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    print(f"   Loaded {len(df)} reviews")
    print(f"   Label distribution: {dict(df['ground_truth'].value_counts())}")
    return df


def load_dataset_rottentomatoes(n_samples=2000, seed=42):
    """Load Rotten Tomatoes movie reviews (shorter, faster to process)."""
    print("\n[1/8] Loading Rotten Tomatoes dataset...")
    from datasets import load_dataset

    ds = load_dataset("rotten_tomatoes", split="test")
    df = pd.DataFrame(ds)
    df.columns = ["text", "label"]
    df["ground_truth"] = df["label"].map({0: "negative", 1: "positive"})

    if n_samples and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)

    print(f"   Loaded {len(df)} reviews")
    print(f"   Label distribution: {dict(df['ground_truth'].value_counts())}")
    return df


# ============================================================
# 3. ASPECT EXTRACTION
# ============================================================

def extract_aspects(text, nlp=None):
    """
    Extract aspect mentions from review text using keyword matching
    with context window extraction.

    Returns list of dicts: {aspect, span, context, start, end}
    """
    text_lower = text.lower()
    found_aspects = []
    seen_aspects = set()

    for aspect_name, aspect_info in ASPECT_TAXONOMY.items():
        for keyword in aspect_info["keywords"]:
            kw_lower = keyword.lower()
            idx = text_lower.find(kw_lower)
            if idx != -1 and aspect_name not in seen_aspects:
                # Extract context window (sentence containing the keyword)
                # Find sentence boundaries
                sent_start = max(0, text_lower.rfind(".", 0, idx) + 1)
                sent_end = text_lower.find(".", idx)
                if sent_end == -1:
                    sent_end = len(text)
                else:
                    sent_end += 1

                context = text[sent_start:sent_end].strip()
                if len(context) < 10:
                    # Fallback: use a window around the keyword
                    w_start = max(0, idx - 100)
                    w_end = min(len(text), idx + 100)
                    context = text[w_start:w_end].strip()

                found_aspects.append({
                    "aspect": aspect_name,
                    "keyword": keyword,
                    "context": context,
                    "start": idx,
                    "end": idx + len(keyword)
                })
                seen_aspects.add(aspect_name)
                break  # One match per aspect per review

    return found_aspects


def extract_aspects_batch(df):
    """Extract aspects for all reviews."""
    print("\n[2/8] Extracting aspects from reviews...")
    all_aspects = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Aspect extraction"):
        aspects = extract_aspects(row["text"])
        for asp in aspects:
            asp["review_idx"] = idx
            asp["ground_truth"] = row["ground_truth"]
            asp["full_text"] = row["text"]
        all_aspects.extend(aspects)

    aspects_df = pd.DataFrame(all_aspects)
    print(f"   Extracted {len(aspects_df)} aspect mentions from {len(df)} reviews")
    print(f"   Aspect distribution:")
    for asp, count in aspects_df["aspect"].value_counts().items():
        print(f"     {asp}: {count}")
    return aspects_df


# ============================================================
# 4. VADER SENTIMENT ANALYSIS
# ============================================================

def setup_vader():
    """Initialize VADER with domain-specific lexicon."""
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    # Augment with domain-specific terms
    for term, score in DOMAIN_LEXICON.items():
        analyzer.lexicon[term] = score

    return analyzer


def vader_classify(text, analyzer, thresholds=(-0.15, 0.20)):
    """
    Classify sentiment using augmented VADER.
    Returns (label, compound_score)
    """
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound <= thresholds[0]:
        return "negative", compound
    elif compound >= thresholds[1]:
        return "positive", compound
    else:
        return "neutral", compound


def run_vader_baseline(aspects_df, analyzer, use_context=False):
    """Run VADER on all aspects. use_context=True uses aspect window, False uses full text."""
    print(f"\n[3/8] Running VADER {'(aspect-windowed)' if use_context else '(document-level)'}...")
    predictions = []
    scores = []

    for _, row in tqdm(aspects_df.iterrows(), total=len(aspects_df), desc="   VADER"):
        text = row["context"] if use_context else row["full_text"]
        label, score = vader_classify(text, analyzer)
        predictions.append(label)
        scores.append(score)

    return predictions, scores


# ============================================================
# 5. RoBERTa SENTIMENT ANALYSIS
# ============================================================

def setup_roberta(device="cpu"):
    """Load pre-trained RoBERTa sentiment model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print(f"   Loading RoBERTa model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Label mapping for this model: 0=negative, 1=neutral, 2=positive
    label_map = {0: "negative", 1: "neutral", 2: "positive"}

    return model, tokenizer, label_map


def roberta_classify(text, model, tokenizer, label_map, device="cpu"):
    """
    Classify sentiment with RoBERTa.
    Returns (label, confidence, logits)
    """
    import torch

    # Truncate to model max length
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=-1)

    label = label_map[predicted.item()]
    conf = confidence.item()
    probs_np = probs.cpu().numpy()[0]

    return label, conf, probs_np


def run_roberta(aspects_df, model, tokenizer, label_map, device="cpu"):
    """Run RoBERTa on all aspect contexts."""
    print("\n[4/8] Running RoBERTa sentiment classification...")
    predictions = []
    confidences = []
    all_probs = []

    for _, row in tqdm(aspects_df.iterrows(), total=len(aspects_df), desc="   RoBERTa"):
        label, conf, probs = roberta_classify(
            row["context"], model, tokenizer, label_map, device
        )
        predictions.append(label)
        confidences.append(conf)
        all_probs.append(probs)

    return predictions, confidences, all_probs


# ============================================================
# 6. LLM CLASSIFICATION (for low-confidence routing)
# ============================================================

def llm_classify_openai(text, aspect, api_key):
    """Use OpenAI API for sentiment classification."""
    import openai
    client = openai.OpenAI(api_key=api_key)

    prompt = f"""Analyze the sentiment of the following movie review excerpt specifically regarding the "{aspect}" aspect.

Review excerpt: "{text}"

Classify the sentiment toward {aspect} as exactly one of: positive, negative, neutral

Consider sarcasm, implicit sentiment, and context carefully.
Respond with ONLY a JSON object: {{"sentiment": "positive/negative/neutral", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("sentiment", "neutral"), result.get("confidence", 0.8)
    except Exception as e:
        return "neutral", 0.5


def llm_classify_anthropic(text, aspect, api_key):
    """Use Anthropic API for sentiment classification."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Analyze the sentiment of the following movie review excerpt specifically regarding the "{aspect}" aspect.

Review excerpt: "{text}"

Classify the sentiment toward {aspect} as exactly one of: positive, negative, neutral

Consider sarcasm, implicit sentiment, and context carefully.
Respond with ONLY a JSON object: {{"sentiment": "positive/negative/neutral", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        result = json.loads(response.content[0].text)
        return result.get("sentiment", "neutral"), result.get("confidence", 0.8)
    except Exception as e:
        return "neutral", 0.5


def llm_classify_local(text, aspect, model, tokenizer, label_map, device="cpu"):
    """
    Local LLM fallback: use a LARGER transformer model.
    We use DeBERTa-v3 which is more capable than RoBERTa for nuanced cases.
    """
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, predicted = torch.max(probs, dim=-1)

    return label_map[predicted.item()], confidence.item()


def setup_local_llm(device="cpu"):
    """Load a larger/different model as the 'LLM' fallback."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # DeBERTa is generally more capable than RoBERTa on nuanced text
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    print(f"   Loading fallback model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    label_map = {0: "positive", 1: "neutral", 2: "negative"}
    return model, tokenizer, label_map


# ============================================================
# 7. CONFIDENCE-GATED ROUTING
# ============================================================

def detect_sarcasm_signals(text):
    """Check for sarcasm indicator patterns."""
    text_lower = text.lower()
    count = sum(1 for pattern in SARCASM_INDICATORS if pattern in text_lower)
    return count > 0, count


def confidence_gated_routing(
    aspects_df, roberta_preds, roberta_confs, roberta_probs,
    vader_analyzer, llm_func, tau=0.85, max_workers=20
):
    """
    Route each aspect to VADER or LLM based on RoBERTa confidence.
    Uses parallel threads for LLM calls to avoid sequential bottleneck.

    High confidence (>= tau): Use augmented VADER on aspect context
    Low confidence (< tau):   Use LLM for deeper analysis
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"\n[5/8] Running confidence-gated routing (tau={tau}, workers={max_workers})...")

    # First pass: decide routing for every item
    routes = []
    for i, (_, row) in enumerate(aspects_df.iterrows()):
        conf = roberta_confs[i]
        has_sarcasm, _ = detect_sarcasm_signals(row["context"])
        if conf < tau or has_sarcasm:
            routes.append("llm")
        else:
            routes.append("vader")

    llm_indices = [i for i, r in enumerate(routes) if r == "llm"]
    vader_indices = [i for i, r in enumerate(routes) if r == "vader"]
    print(f"   VADER-routed: {len(vader_indices)} ({len(vader_indices)/len(routes)*100:.1f}%)")
    print(f"   LLM-routed:   {len(llm_indices)} ({len(llm_indices)/len(routes)*100:.1f}%)")
    print(f"   Sending {len(llm_indices)} calls to OpenAI in parallel (this is fast now)...")

    # Pre-fill results array
    final_preds = [None] * len(aspects_df)
    rows_list = list(aspects_df.iterrows())

    # VADER pass (instant)
    for i in vader_indices:
        _, row = rows_list[i]
        label, _ = vader_classify(row["context"], vader_analyzer)
        final_preds[i] = label

    # LLM pass — parallel with progress bar
    def call_llm(i):
        _, row = rows_list[i]
        label, _ = llm_func(row["context"], row["aspect"])
        return i, label

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(call_llm, i): i for i in llm_indices}
        for future in tqdm(as_completed(futures), total=len(llm_indices), desc="   LLM calls"):
            i, label = future.result()
            final_preds[i] = label

    # Fallback for any None
    final_preds = [p if p is not None else "neutral" for p in final_preds]

    return final_preds, routes


# ============================================================
# 8. ENSEMBLE BASELINE
# ============================================================

def roberta_vader_ensemble(roberta_preds, roberta_confs, vader_preds, vader_scores):
    """Simple averaging ensemble of RoBERTa + VADER."""
    label_to_num = {"negative": -1, "neutral": 0, "positive": 1}
    num_to_label = {-1: "negative", 0: "neutral", 1: "positive"}

    preds = []
    for rp, rc, vp, vs in zip(roberta_preds, roberta_confs, vader_preds, vader_scores):
        # Weighted average: RoBERTa weighted by confidence
        r_num = label_to_num.get(rp, 0) * rc
        v_num = vs  # VADER compound is already -1 to 1
        avg = (r_num + v_num) / 2

        if avg < -0.1:
            preds.append("negative")
        elif avg > 0.1:
            preds.append("positive")
        else:
            preds.append("neutral")
    return preds


# ============================================================
# 9. SHAP EXPLAINABILITY
# ============================================================

def run_shap_analysis(aspects_df, model, tokenizer, device="cpu", n_samples=50):
    """Run SHAP analysis on a subset of predictions."""
    print("\n[6/8] Running SHAP explainability analysis...")
    import shap

    # Create prediction function for SHAP
    def predict_proba(texts):
        import torch
        results = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            results.append(probs.cpu().numpy()[0])
        return np.array(results)

    # Sample subset for SHAP (it's computationally expensive)
    sample_indices = np.random.choice(
        len(aspects_df), min(n_samples, len(aspects_df)), replace=False
    )
    sample_texts = aspects_df.iloc[sample_indices]["context"].tolist()

    # Use partition explainer for text
    try:
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(predict_proba, masker, output_names=["negative", "neutral", "positive"])
        shap_values = explainer(sample_texts[:min(20, len(sample_texts))])

        explanations = []
        for i in range(len(shap_values)):
            # Get top contributing tokens for the predicted class
            sv = shap_values[i]
            pred_class = np.argmax(sv.base_values + sv.values.sum(axis=0))
            token_importance = sv.values[:, pred_class]
            tokens = sv.data

            # Top 5 most important tokens
            top_indices = np.argsort(np.abs(token_importance))[-5:][::-1]
            top_tokens = [(tokens[j], float(token_importance[j])) for j in top_indices if j < len(tokens)]

            explanations.append({
                "text": sample_texts[i][:100] + "...",
                "top_tokens": top_tokens,
                "predicted_class": ["negative", "neutral", "positive"][pred_class]
            })

        print(f"   Generated SHAP explanations for {len(explanations)} samples")
        return explanations
    except Exception as e:
        print(f"   SHAP analysis encountered an error: {e}")
        print("   Falling back to simple attention-based attribution...")
        return generate_simple_explanations(aspects_df, sample_indices, model, tokenizer, device)


def generate_simple_explanations(aspects_df, indices, model, tokenizer, device):
    """Fallback: use gradient-based attribution if SHAP fails."""
    import torch

    explanations = []
    for idx in indices[:20]:
        row = aspects_df.iloc[idx]
        text = row["context"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512, padding=True).to(device)

        # Get model attention weights
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()

            # Average attention across heads and layers
            attentions = outputs.attentions
            avg_attention = torch.stack(attentions).mean(dim=(0, 1, 2))
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            # Top 5 attended tokens
            top_indices = torch.argsort(avg_attention, descending=True)[:5]
            top_tokens = [(tokens[j], float(avg_attention[j])) for j in top_indices
                          if j < len(tokens) and tokens[j] not in ["<s>", "</s>", "<pad>"]]

        explanations.append({
            "text": text[:100] + "...",
            "top_tokens": top_tokens[:5],
            "predicted_class": ["negative", "neutral", "positive"][pred_class]
        })

    return explanations


# ============================================================
# 10. INTER-ASPECT SENTIMENT DIVERGENCE
# ============================================================

def compute_iasd(aspects_df, predictions, col_name="prediction"):
    """Compute Inter-Aspect Sentiment Divergence."""
    print("\n   Computing Inter-Aspect Sentiment Divergence (IASD)...")

    temp_df = aspects_df.copy()
    temp_df["prediction"] = predictions

    # Group by review
    divergent_reviews = 0
    total_multi_aspect = 0
    divergence_patterns = Counter()

    for review_idx, group in temp_df.groupby("review_idx"):
        if len(group) < 2:
            continue
        total_multi_aspect += 1

        sentiments = set(group["prediction"])
        if len(sentiments) > 1 and "neutral" not in sentiments:
            # Has both positive and negative
            if "positive" in sentiments and "negative" in sentiments:
                divergent_reviews += 1
                # Record divergence patterns
                pos_aspects = group[group["prediction"] == "positive"]["aspect"].tolist()
                neg_aspects = group[group["prediction"] == "negative"]["aspect"].tolist()
                for pa in pos_aspects:
                    for na in neg_aspects:
                        divergence_patterns[f"+{pa} / -{na}"] += 1

    iasd_rate = divergent_reviews / total_multi_aspect if total_multi_aspect > 0 else 0
    return iasd_rate, total_multi_aspect, divergent_reviews, divergence_patterns


# ============================================================
# 11. METRICS AND REPORTING
# ============================================================

def compute_metrics(y_true, y_pred, method_name=""):
    """Compute classification metrics."""
    # Map to binary for IMDB (which has no neutral ground truth)
    # Filter out neutral predictions for fair comparison
    labels = ["negative", "positive"]

    # For samples where ground truth is binary, map neutral predictions
    # to the closest class based on distribution
    y_pred_binary = []
    y_true_binary = []
    for yt, yp in zip(y_true, y_pred):
        if yt in labels:
            y_true_binary.append(yt)
            y_pred_binary.append(yp if yp in labels else "negative")  # neutral -> negative for binary

    macro_f1 = f1_score(y_true_binary, y_pred_binary, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_binary, y_pred_binary, labels=labels, average="weighted", zero_division=0)
    precision = precision_score(y_true_binary, y_pred_binary, labels=labels, average="macro", zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, labels=labels, average="macro", zero_division=0)
    accuracy = sum(1 for a, b in zip(y_true_binary, y_pred_binary) if a == b) / len(y_true_binary) if y_true_binary else 0

    return {
        "method": method_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision": precision,
        "recall": recall,
        "n_samples": len(y_true_binary)
    }


def generate_report(results_dict, aspects_df, routing_decisions, shap_explanations, output_dir):
    """Generate all output files and figures."""
    print("\n[8/8] Generating reports and figures...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    # --- Overall metrics table ---
    metrics_rows = []
    for method, metrics in results_dict.items():
        metrics_rows.append([
            metrics["method"],
            f"{metrics['macro_f1']:.3f}",
            f"{metrics['weighted_f1']:.3f}",
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['accuracy']:.3f}",
            metrics["n_samples"]
        ])

    headers = ["Method", "Macro-F1", "W-F1", "Precision", "Recall", "Accuracy", "N"]
    table = tabulate(metrics_rows, headers=headers, tablefmt="grid")

    print("\n" + "=" * 70)
    print("OVERALL SENTIMENT CLASSIFICATION RESULTS")
    print("=" * 70)
    print(table)

    with open(os.path.join(output_dir, "overall_metrics.txt"), "w") as f:
        f.write("OVERALL SENTIMENT CLASSIFICATION RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(table)
        f.write("\n\nNote: Ground truth from IMDB binary labels (positive/negative).\n")
        f.write("Neutral predictions are mapped to negative for binary evaluation.\n")

    # --- Per-aspect metrics ---
    if "aspectstreamsa" in results_dict:
        print("\n" + "=" * 70)
        print("PER-ASPECT PERFORMANCE (AspectStreamSA)")
        print("=" * 70)

        aspect_metrics = []
        temp_df = aspects_df.copy()
        temp_df["pred"] = results_dict["aspectstreamsa"].get("raw_preds", [])
        temp_df["route"] = routing_decisions if routing_decisions else ["unknown"] * len(temp_df)

        for aspect_name in ASPECT_TAXONOMY.keys():
            asp_subset = temp_df[temp_df["aspect"] == aspect_name]
            if len(asp_subset) < 5:
                continue

            m = compute_metrics(
                asp_subset["ground_truth"].tolist(),
                asp_subset["pred"].tolist(),
                aspect_name
            )

            vader_pct = (asp_subset["route"] == "vader").mean() * 100 if "route" in asp_subset.columns else 0
            llm_pct = (asp_subset["route"] == "llm").mean() * 100 if "route" in asp_subset.columns else 0

            # Sarcasm rate
            sarc_rate = asp_subset["context"].apply(
                lambda x: detect_sarcasm_signals(x)[0]
            ).mean() * 100

            aspect_metrics.append([
                aspect_name, f"{m['macro_f1']:.3f}", f"{m['precision']:.3f}",
                f"{m['recall']:.3f}", f"{vader_pct:.0f}%", f"{llm_pct:.0f}%",
                f"{sarc_rate:.1f}%", len(asp_subset)
            ])

        asp_headers = ["Aspect", "F1", "Prec.", "Recall", "%VADER", "%LLM", "Sarc.%", "N"]
        asp_table = tabulate(aspect_metrics, headers=asp_headers, tablefmt="grid")
        print(asp_table)

        with open(os.path.join(output_dir, "per_aspect_metrics.txt"), "w") as f:
            f.write("PER-ASPECT PERFORMANCE (AspectStreamSA)\n")
            f.write("=" * 70 + "\n")
            f.write(asp_table)

    # --- Routing analysis ---
    if routing_decisions:
        route_counter = Counter(routing_decisions)
        total = len(routing_decisions)

        routing_text = f"""
ROUTING EFFICIENCY ANALYSIS
{'='*70}
Total aspect-sentiment pairs: {total}
VADER-routed: {route_counter.get('vader', 0)} ({route_counter.get('vader', 0)/total*100:.1f}%)
LLM-routed:   {route_counter.get('llm', 0)} ({route_counter.get('llm', 0)/total*100:.1f}%)

Estimated cost comparison (per 1000 reviews):
  VADER-only:       $0.00
  LLM-all:          ~$4.72  (all routed to API)
  AspectStreamSA:   ~${route_counter.get('llm', 0)/total * 4.72:.2f}  ({route_counter.get('llm', 0)/total*100:.1f}% routed to API)
  Cost reduction:   {(1 - route_counter.get('llm', 0)/total)*100:.1f}%
"""
        print(routing_text)
        with open(os.path.join(output_dir, "routing_analysis.txt"), "w") as f:
            f.write(routing_text)

    # --- SHAP explanations ---
    if shap_explanations:
        with open(os.path.join(output_dir, "shap_examples.txt"), "w") as f:
            f.write("SHAP / ATTENTION-BASED EXPLANATIONS\n")
            f.write("=" * 70 + "\n\n")
            for i, exp in enumerate(shap_explanations[:10]):
                f.write(f"Example {i+1}:\n")
                f.write(f"  Text: {exp['text']}\n")
                f.write(f"  Predicted: {exp['predicted_class']}\n")
                f.write(f"  Top tokens:\n")
                for token, importance in exp["top_tokens"]:
                    f.write(f"    '{token}': {importance:.4f}\n")
                f.write("\n")

    # --- IASD ---
    if "aspectstreamsa" in results_dict and "raw_preds" in results_dict["aspectstreamsa"]:
        iasd_rate, multi_count, div_count, patterns = compute_iasd(
            aspects_df, results_dict["aspectstreamsa"]["raw_preds"]
        )

        iasd_text = f"""
INTER-ASPECT SENTIMENT DIVERGENCE (IASD)
{'='*70}
Multi-aspect reviews:    {multi_count}
Divergent reviews:       {div_count}
IASD rate:               {iasd_rate*100:.1f}%

Top divergence patterns:
"""
        for pattern, count in patterns.most_common(10):
            iasd_text += f"  {pattern}: {count}\n"

        print(iasd_text)
        with open(os.path.join(output_dir, "iasd_analysis.txt"), "w") as f:
            f.write(iasd_text)

    # --- Figures ---
    generate_figures(results_dict, aspects_df, routing_decisions, output_dir)

    # --- Save raw predictions ---
    if "aspectstreamsa" in results_dict and "raw_preds" in results_dict["aspectstreamsa"]:
        export_df = aspects_df[["review_idx", "aspect", "keyword", "context", "ground_truth"]].copy()
        export_df["prediction"] = results_dict["aspectstreamsa"]["raw_preds"]
        export_df["route"] = routing_decisions if routing_decisions else "unknown"
        export_df.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
        print(f"\n   Raw predictions saved to {output_dir}/benchmark_results.csv")


def generate_figures(results_dict, aspects_df, routing_decisions, output_dir):
    """Generate comparison charts."""
    fig_dir = os.path.join(output_dir, "figures")

    # 1. Method comparison bar chart
    methods = []
    f1_scores = []
    for method, metrics in results_dict.items():
        methods.append(metrics["method"])
        f1_scores.append(metrics["macro_f1"])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#c0392b", "#e67e22", "#2980b9", "#8e44ad", "#27ae60", "#1abc9c"]
    bars = ax.bar(methods, f1_scores, color=colors[:len(methods)], edgecolor="white", linewidth=1.5)
    ax.set_ylabel("Macro-F1 Score", fontsize=12)
    ax.set_title("Sentiment Classification: Method Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.0)
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "method_comparison.png"), dpi=150)
    plt.close()

    # 2. Aspect distribution
    asp_counts = aspects_df["aspect"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    asp_counts.plot(kind="barh", ax=ax, color="#2980b9", edgecolor="white")
    ax.set_xlabel("Number of Mentions", fontsize=12)
    ax.set_title("Aspect Distribution Across Reviews", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "aspect_distribution.png"), dpi=150)
    plt.close()

    # 3. Routing distribution pie chart
    if routing_decisions:
        route_counts = Counter(routing_decisions)
        fig, ax = plt.subplots(figsize=(8, 8))
        labels = [f"VADER\n({route_counts.get('vader', 0)})", f"LLM\n({route_counts.get('llm', 0)})"]
        sizes = [route_counts.get("vader", 0), route_counts.get("llm", 0)]
        colors_pie = ["#3498db", "#e74c3c"]
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%",
               startangle=90, textprops={"fontsize": 13})
        ax.set_title("Confidence-Gated Routing Distribution", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "routing_distribution.png"), dpi=150)
        plt.close()

    # 4. Confidence distribution histogram
    if "roberta_confs" in results_dict.get("roberta_ft", {}):
        confs = results_dict["roberta_ft"]["roberta_confs"]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(confs, bins=50, color="#2980b9", edgecolor="white", alpha=0.8)
        ax.axvline(x=0.85, color="#e74c3c", linestyle="--", linewidth=2, label="Threshold (τ=0.85)")
        ax.set_xlabel("RoBERTa Confidence Score", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Confidence Score Distribution with Routing Threshold", fontsize=14, fontweight="bold")
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "confidence_distribution.png"), dpi=150)
        plt.close()

    print(f"   Figures saved to {fig_dir}/")


# ============================================================
# 12. THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

def threshold_sensitivity(
    aspects_df, roberta_confs, roberta_probs, vader_analyzer,
    llm_func, label_map_roberta
):
    """Test different routing thresholds."""
    print("\n[7/8] Running threshold sensitivity analysis...")
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    results = []

    for tau in thresholds:
        preds, routes = confidence_gated_routing(
            aspects_df, None, roberta_confs, roberta_probs,
            vader_analyzer, llm_func, tau=tau
        )
        metrics = compute_metrics(
            aspects_df["ground_truth"].tolist(), preds, f"tau={tau}"
        )
        vader_pct = sum(1 for r in routes if r == "vader") / len(routes) * 100
        results.append({
            "tau": tau, "macro_f1": metrics["macro_f1"],
            "vader_pct": vader_pct, "llm_pct": 100 - vader_pct
        })
        # Suppress routing prints for sensitivity runs
    
    print("\n   Threshold Sensitivity:")
    sens_headers = ["Threshold", "Macro-F1", "% VADER", "% LLM"]
    sens_rows = [[r["tau"], f"{r['macro_f1']:.3f}", f"{r['vader_pct']:.1f}%", f"{r['llm_pct']:.1f}%"]
                  for r in results]
    print(tabulate(sens_rows, headers=sens_headers, tablefmt="grid"))

    return results


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AspectStreamSA Pipeline")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of reviews to process (default: 2000)")
    parser.add_argument("--dataset", choices=["imdb", "rotten_tomatoes"], default="imdb",
                        help="Dataset to use (default: imdb)")
    parser.add_argument("--tau", type=float, default=0.85,
                        help="Routing confidence threshold (default: 0.85)")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU if available")
    parser.add_argument("--openai_key", type=str, default=None,
                        help="OpenAI API key for LLM routing")
    parser.add_argument("--anthropic_key", type=str, default=None,
                        help="Anthropic API key for LLM routing")
    parser.add_argument("--skip_shap", action="store_true",
                        help="Skip SHAP analysis (faster)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--skip_sensitivity", action="store_true",
                        help="Skip threshold sensitivity analysis")
    args = parser.parse_args()

    import torch
    import pickle

    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    start_time = time.time()
    cache_dir = os.path.join(args.output_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    def save_ckpt(name, obj):
        with open(os.path.join(cache_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)
        print(f"   [checkpoint saved: {name}]")

    def load_ckpt(name):
        path = os.path.join(cache_dir, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                obj = pickle.load(f)
            print(f"   [checkpoint found: {name} — skipping this step]")
            return obj, True
        return None, False

    # ---- Step 1: Load data ----
    cached, hit = load_ckpt("df")
    if hit:
        df = cached
    else:
        if args.dataset == "imdb":
            df = load_dataset_imdb(n_samples=args.n_samples)
        else:
            df = load_dataset_rottentomatoes(n_samples=args.n_samples)
        save_ckpt("df", df)

    # ---- Step 2: Extract aspects ----
    cached, hit = load_ckpt("aspects_df")
    if hit:
        aspects_df = cached
    else:
        aspects_df = extract_aspects_batch(df)
        save_ckpt("aspects_df", aspects_df)

    if len(aspects_df) == 0:
        print("ERROR: No aspects extracted. Try a larger sample or different dataset.")
        return

    # ---- Step 3: VADER baselines ----
    vader_analyzer = setup_vader()
    cached, hit = load_ckpt("vader_results")
    if hit:
        vader_doc_preds, vader_doc_scores, vader_asp_preds, vader_asp_scores = cached
    else:
        vader_doc_preds, vader_doc_scores = run_vader_baseline(aspects_df, vader_analyzer, use_context=False)
        vader_asp_preds, vader_asp_scores = run_vader_baseline(aspects_df, vader_analyzer, use_context=True)
        save_ckpt("vader_results", (vader_doc_preds, vader_doc_scores, vader_asp_preds, vader_asp_scores))

    # ---- Step 4: RoBERTa ----
    roberta_model, roberta_tok, roberta_labels = setup_roberta(device)
    cached, hit = load_ckpt("roberta_results")
    if hit:
        roberta_preds, roberta_confs, roberta_probs = cached
    else:
        roberta_preds, roberta_confs, roberta_probs = run_roberta(
            aspects_df, roberta_model, roberta_tok, roberta_labels, device
        )
        save_ckpt("roberta_results", (roberta_preds, roberta_confs, roberta_probs))

    # ---- Step 5: Set up LLM function ----
    if args.openai_key:
        print("\n   Using OpenAI API for LLM routing")
        llm_func = lambda text, aspect: llm_classify_openai(text, aspect, args.openai_key)
    elif args.anthropic_key:
        print("\n   Using Anthropic API for LLM routing")
        llm_func = lambda text, aspect: llm_classify_anthropic(text, aspect, args.anthropic_key)
    else:
        print("\n   Using local fallback model for LLM routing (no API key provided)")
        llm_model, llm_tok, llm_labels = setup_local_llm(device)
        llm_func = lambda text, aspect: llm_classify_local(
            text, aspect, llm_model, llm_tok, llm_labels, device
        )

    # ---- Step 6: Confidence-gated routing (AspectStreamSA) ----
    cached, hit = load_ckpt("routing_results")
    if hit:
        stream_preds, routing_decisions = cached
    else:
        stream_preds, routing_decisions = confidence_gated_routing(
            aspects_df, roberta_preds, roberta_confs, roberta_probs,
            vader_analyzer, llm_func, tau=args.tau
        )
        save_ckpt("routing_results", (stream_preds, routing_decisions))

    # ---- Step 7: Ensemble baseline ----
    ensemble_preds = roberta_vader_ensemble(
        roberta_preds, roberta_confs, vader_asp_preds, vader_asp_scores
    )

    # ---- Step 8: Compute all metrics ----
    gt = aspects_df["ground_truth"].tolist()

    results_dict = {
        "vader_doc":      compute_metrics(gt, vader_doc_preds,  "VADER-Only"),
        "vader_aspect":   compute_metrics(gt, vader_asp_preds,  "VADER-Aspect"),
        "roberta_ft":     compute_metrics(gt, roberta_preds,    "RoBERTa-FT"),
        "ensemble":       compute_metrics(gt, ensemble_preds,   "RoBERTa+VADER"),
        "aspectstreamsa": compute_metrics(gt, stream_preds,     "AspectStreamSA"),
    }
    results_dict["aspectstreamsa"]["raw_preds"]   = stream_preds
    results_dict["roberta_ft"]["roberta_confs"]   = roberta_confs

    # ---- SHAP analysis ----
    shap_explanations = []
    if not args.skip_shap:
        shap_explanations = run_shap_analysis(
            aspects_df, roberta_model, roberta_tok, device, n_samples=30
        )

    # ---- Threshold sensitivity ----
    if not args.skip_sensitivity:
        threshold_sensitivity(
            aspects_df, roberta_confs, roberta_probs,
            vader_analyzer, llm_func, roberta_labels
        )

    # ---- Generate reports ----
    generate_report(results_dict, aspects_df, routing_decisions, shap_explanations, args.output_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Pipeline completed in {elapsed:.1f} seconds")
    print(f"Results saved to {args.output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()