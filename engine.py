"""Model loading, text cleaning and explanation logic for CineSense AI.

This module deliberately has no Streamlit imports so it can be tested on its own.
The cleaning pipeline mirrors the training notebook exactly (main.ipynb), because
the saved TF-IDF vectorizer expects text that was cleaned the same way.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = APP_DIR / "tfidf_vectorizer.pkl"

# NLTK's English stop-word list, used only if NLTK's data cannot be downloaded.
_FALLBACK_STOPWORDS = frozenset(
    """i me my myself we our ours ourselves you you're you've you'll you'd your yours
    yourself yourselves he him his himself she she's her hers herself it it's its itself
    they them their theirs themselves what which who whom this that that'll these those am
    is are was were be been being have has had having do does did doing a an the and but if
    or because as until while of at by for with about against between into through during
    before after above below to from up down in out on off over under again further then
    once here there when where why how all any both each few more most other some such no
    nor not only own same so than too very s t can will just don don't should should've now
    d ll m o re ve y ain aren aren't couldn couldn't didn didn't doesn doesn't hadn hadn't
    hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan
    shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't""".split()
)


@dataclass(frozen=True)
class TextTools:
    stop_words: frozenset
    lemmatize: Callable[[str], str]
    source: str  # "nltk" or "fallback"


@dataclass(frozen=True)
class Analysis:
    sentiment: str                      # "positive" | "negative"
    confidence: float                   # probability of the predicted class
    probabilities: dict                 # {"negative": p, "positive": p}
    word_scores: dict                   # {vocab term: signed contribution toward positive}
    top_positive: list                  # [(word, score), ...]
    top_negative: list                  # [(word, score), ...]


def load_artifacts():
    """Load the trained classifier and TF-IDF vectorizer, with friendly errors."""
    missing = [p.name for p in (MODEL_PATH, VECTORIZER_PATH) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing)
            + f". Keep them in the same folder as app.py ({APP_DIR})."
        )
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    labels = {str(c) for c in getattr(model, "classes_", [])}
    if labels != {"negative", "positive"}:
        raise ValueError(f"Unexpected model classes: {sorted(labels)}")
    return model, tfidf


def _nltk_tools():
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stops = frozenset(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("tests")  # forces WordNet to load; raises LookupError if absent
    return TextTools(stops, lemmatizer.lemmatize, "nltk")


def load_text_tools() -> TextTools:
    """Use NLTK when available (downloading data only if needed); never crash offline."""
    try:
        return _nltk_tools()
    except ImportError:
        pass
    except LookupError:
        try:
            import nltk

            for resource in ("stopwords", "wordnet", "omw-1.4"):
                nltk.download(resource, quiet=True)
            return _nltk_tools()
        except Exception:
            pass
    except Exception:
        pass
    return TextTools(_FALLBACK_STOPWORDS, lambda word: word, "fallback")


def clean_review(text: str, tools: TextTools) -> str:
    """Same steps as clean_text() in the training notebook."""
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = [w for w in text.split() if w not in tools.stop_words]
    return " ".join(tools.lemmatize(w) for w in words)


def confidence_label(confidence: float) -> str:
    if confidence >= 0.90:
        return "Very high confidence"
    if confidence >= 0.75:
        return "High confidence"
    if confidence >= 0.60:
        return "Moderate confidence"
    return "Low confidence · mixed signals"


def analyze(review: str, model, tfidf, tools: TextTools, top_n: int = 6):
    """Return an Analysis, or None if nothing is left after cleaning."""
    cleaned = clean_review(review, tools)
    if not cleaned:
        return None

    vector = tfidf.transform([cleaned])
    labels = [str(c) for c in model.classes_]
    proba = model.predict_proba(vector)[0]
    probabilities = {label: float(p) for label, p in zip(labels, proba)}
    best = int(np.argmax(proba))

    scores: dict = {}
    try:  # per-word explanation (linear models only)
        coef = np.asarray(model.coef_)[0]
        sign = 1.0 if labels.index("positive") == 1 else -1.0
        names = tfidf.get_feature_names_out()
        row = vector.tocsr()
        for j, value in zip(row.indices, row.data):
            scores[str(names[j])] = float(value * coef[j] * sign)
    except Exception:
        scores = {}

    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    top_negative = [kv for kv in ranked if kv[1] < 0][:top_n]
    top_positive = [kv for kv in reversed(ranked) if kv[1] > 0][:top_n]

    return Analysis(
        sentiment=labels[best],
        confidence=float(proba[best]),
        probabilities=probabilities,
        word_scores=scores,
        top_positive=top_positive,
        top_negative=top_negative,
    )


def highlight_html(review: str, scores: dict, tools: TextTools) -> str:
    """Return the review as safe HTML with influential words highlighted."""
    text = re.sub(r"<br\s*/?>", "\n", review, flags=re.IGNORECASE)
    peak = max((abs(v) for v in scores.values()), default=0.0) or 1.0

    pieces, cursor = [], 0
    for match in re.finditer(r"[A-Za-z][A-Za-z'’]*", text):
        pieces.append(html.escape(text[cursor:match.start()]))
        raw = match.group()
        norm = re.sub(r"[^a-z]", "", raw.lower())
        score = 0.0
        if norm and norm not in tools.stop_words:
            score = scores.get(tools.lemmatize(norm), 0.0)
        if score:
            alpha = 0.14 + 0.46 * min(1.0, abs(score) / peak)
            rgb = "16,185,129" if score > 0 else "244,63,94"
            pieces.append(
                f'<span class="hl" style="background:rgba({rgb},{alpha:.2f})">'
                f"{html.escape(raw)}</span>"
            )
        else:
            pieces.append(html.escape(raw))
        cursor = match.end()
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces).replace("\r", "").replace("\n", "<br>")
