"""CineSense AI — a movie-review sentiment analyzer (Streamlit).

Run with:  python -m streamlit run app.py
"""
from __future__ import annotations

import html
from datetime import datetime

import streamlit as st

import engine

st.set_page_config(
    page_title="CineSense AI · Movie Review Analyzer",
    page_icon="🎬",
    layout="wide",
)

SAMPLES = {
    "😍 Loved it": (
        "A charming, beautifully acted film with a story that stayed with me long "
        "after the credits. The soundtrack was wonderful and the ending was perfect."
    ),
    "💔 Hated it": (
        "The plot was dull, the characters felt flat, and the ending was "
        "disappointing. Two hours of my life wasted on a boring, predictable mess."
    ),
    "🤔 Mixed": (
        "The cinematography was gorgeous and the lead performance was strong, but "
        "the script dragged and the final act felt rushed and confusing."
    ),
}

CSS = """
<style>
.block-container{max-width:1180px;padding-top:2rem;padding-bottom:4rem}
div.hero{padding:1.7rem 1.9rem;border-radius:22px;margin-bottom:1.4rem;
  background:linear-gradient(135deg,rgba(99,102,241,.18),rgba(236,72,153,.12) 60%,rgba(16,185,129,.10));
  border:1px solid rgba(128,128,128,.2)}
div.hero .kicker{font-size:.74rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#6366f1}
div.hero h1{margin:.25rem 0 .35rem!important;padding:0!important;font-size:2.5rem!important;line-height:1.1!important;font-weight:800!important}
div.hero p{margin:0;opacity:.78;font-size:1.03rem;max-width:660px}
.pill{display:inline-block;padding:.18rem .65rem;border-radius:999px;font-size:.74rem;font-weight:600;
  border:1px solid rgba(128,128,128,.32);margin:.8rem .4rem 0 0;opacity:.88}
.result{display:flex;gap:1rem;align-items:center;padding:1.15rem 1.35rem;border-radius:18px;
  border:1px solid rgba(128,128,128,.2);margin:.3rem 0 .9rem}
.result.pos{background:linear-gradient(135deg,rgba(16,185,129,.18),rgba(16,185,129,.04));border-color:rgba(16,185,129,.5)}
.result.neg{background:linear-gradient(135deg,rgba(244,63,94,.18),rgba(244,63,94,.04));border-color:rgba(244,63,94,.5)}
.result .emoji{font-size:2.5rem;line-height:1}
.result .verdict{font-size:1.55rem;font-weight:800;line-height:1.1}
.result .sub{opacity:.75;font-size:.9rem;margin-top:.15rem}
.meter{display:flex;height:11px;border-radius:999px;overflow:hidden;background:rgba(128,128,128,.22)}
.meter .p{background:#10b981}.meter .n{background:#f43f5e}
.meter-labels{display:flex;justify-content:space-between;font-size:.8rem;opacity:.75;margin:.4rem 0 1.1rem}
.section-title{font-weight:700;font-size:.95rem;margin:1rem 0 .45rem}
.review-view{padding:1rem 1.15rem;border-radius:14px;border:1px solid rgba(128,128,128,.22);
  background:rgba(128,128,128,.06);line-height:1.85;font-size:.98rem}
.hl{padding:.08rem .3rem;border-radius:6px}
.legend{font-size:.78rem;opacity:.65;margin:.5rem 0 0}
.chips{display:flex;flex-wrap:wrap;gap:.4rem;margin:.1rem 0 .4rem}
.chip{padding:.2rem .7rem;border-radius:999px;font-size:.82rem;font-weight:600;border:1px solid}
.chip.pos{background:rgba(16,185,129,.14);border-color:rgba(16,185,129,.5)}
.chip.neg{background:rgba(244,63,94,.14);border-color:rgba(244,63,94,.5)}
.hist{display:flex;gap:.65rem;align-items:flex-start;padding:.6rem 0;border-bottom:1px solid rgba(128,128,128,.16);font-size:.88rem}
.hist:last-child{border-bottom:none}
.dot{width:10px;height:10px;border-radius:50%;margin-top:.4rem;flex:none}
.dot.pos{background:#10b981}.dot.neg{background:#f43f5e}
.hist .t{flex:1;opacity:.92}.hist .c{opacity:.6;font-variant-numeric:tabular-nums;white-space:nowrap}
.empty{opacity:.6;font-size:.9rem;padding:.4rem 0}
</style>
"""


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading model…")
def get_artifacts():
    return engine.load_artifacts()


@st.cache_resource(show_spinner="Preparing language tools…")
def get_tools():
    return engine.load_text_tools()


try:
    model, tfidf = get_artifacts()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()
except Exception as err:  # e.g. scikit-learn version mismatch when unpickling
    st.error(
        f"Could not load the model: {err}\n\n"
        "Install the exact versions from requirements.txt "
        "(`python -m pip install -r requirements.txt`)."
    )
    st.stop()

tools = get_tools()


# ---------------------------------------------------------------- state
def _init_state():
    st.session_state.setdefault("review_text", "")
    st.session_state.setdefault("last", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("counts", {"positive": 0, "negative": 0})


def use_sample(text: str):
    st.session_state.review_text = text


def clear_review():
    st.session_state.review_text = ""
    st.session_state.last = None


def clear_history():
    st.session_state.history = []
    st.session_state.counts = {"positive": 0, "negative": 0}
    st.session_state.last = None


_init_state()
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def chips(items, kind):
    if not items:
        return '<div class="empty">None found in this review.</div>'
    body = "".join(f'<span class="chip {kind}">{html.escape(w)}</span>' for w, _ in items)
    return f'<div class="chips">{body}</div>'


def render_result(last):
    a = last["analysis"]
    is_pos = a.sentiment == "positive"
    kind = "pos" if is_pos else "neg"
    p_pos = a.probabilities.get("positive", 0.0)
    p_neg = a.probabilities.get("negative", 0.0)

    st.markdown(
        f'<div class="result {kind}"><div class="emoji">{"😍" if is_pos else "💔"}</div>'
        f'<div><div class="verdict">{"Positive" if is_pos else "Negative"} review</div>'
        f'<div class="sub">{a.confidence:.1%} · {engine.confidence_label(a.confidence)}</div></div></div>'
        f'<div class="meter" role="img" aria-label="Negative {p_neg:.0%}, positive {p_pos:.0%}">'
        f'<div class="n" style="width:{p_neg:.2%}"></div><div class="p" style="width:{p_pos:.2%}"></div></div>'
        f'<div class="meter-labels"><span>Negative {p_neg:.1%}</span><span>Positive {p_pos:.1%}</span></div>',
        unsafe_allow_html=True,
    )

    if a.word_scores:
        view = engine.highlight_html(last["text"], a.word_scores, tools)
        st.markdown(
            '<div class="section-title">What shaped this result</div>'
            f'<div class="review-view">{view}</div>'
            '<div class="legend">Green words push toward positive, red toward negative; '
            "stronger color means stronger influence. Common filler words are ignored.</div>",
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)
        left.markdown(
            '<div class="section-title">Top positive signals</div>' + chips(a.top_positive, "pos"),
            unsafe_allow_html=True,
        )
        right.markdown(
            '<div class="section-title">Top negative signals</div>' + chips(a.top_negative, "neg"),
            unsafe_allow_html=True,
        )


def render_history(history):
    if not history:
        st.markdown('<div class="empty">Your analyzed reviews will appear here.</div>', unsafe_allow_html=True)
        return
    rows = []
    for item in history[:8]:
        snippet = item["text"].replace("\n", " ")
        snippet = html.escape(snippet if len(snippet) <= 90 else snippet[:87] + "…")
        kind = "pos" if item["sentiment"] == "positive" else "neg"
        rows.append(
            f'<div class="hist"><span class="dot {kind}"></span><span class="t">{snippet}</span>'
            f'<span class="c">{item["confidence"]:.0%} · {item["time"]}</span></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


# ---------------------------------------------------------------- header
st.markdown(
    '<div class="hero"><div class="kicker">Movie Review Lab · NLP</div>'
    "<h1>CineSense AI</h1>"
    "<p>Paste any movie review and see whether it reads positive or negative — "
    "and exactly which words tipped the scale.</p>"
    f'<span class="pill">{html.escape(type(model).__name__)}</span>'
    '<span class="pill">TF-IDF · 5,000 features</span>'
    '<span class="pill">≈ 88.6% test accuracy</span></div>',
    unsafe_allow_html=True,
)

if tools.source == "fallback":
    st.warning(
        "Offline mode: NLTK language data could not be downloaded, so a built-in "
        "stop-word list is used and word lemmatization is skipped. Results may "
        "differ slightly. Connect to the internet once and restart to fix this."
    )

# ---------------------------------------------------------------- layout
analyze_col, stats_col = st.columns([1.55, 1], gap="large")

with analyze_col:
    st.subheader("Analyze a review")
    st.caption("Try a sample, or write your own.")

    sample_cols = st.columns(len(SAMPLES))
    for col, (label, text) in zip(sample_cols, SAMPLES.items()):
        col.button(label, key=f"sample_{label}", on_click=use_sample, args=(text,), width="stretch")

    with st.form("review_form", border=False):
        review = st.text_area(
            "Your review",
            key="review_text",
            height=190,
            max_chars=5000,
            placeholder="Write or paste a movie review here…",
            label_visibility="collapsed",
        )
        go_col, clear_col = st.columns([3, 1])
        submitted = go_col.form_submit_button("Analyze sentiment", type="primary", width="stretch")
        clear_col.form_submit_button("Clear", on_click=clear_review, width="stretch")

    if submitted:
        st.session_state.last = None
        text = review.strip()
        if not text:
            st.warning("Write or paste a review first.")
        else:
            analysis = engine.analyze(text, model, tfidf, tools)
            if analysis is None:
                st.warning("That text has no usable words left after cleaning. Try a longer review.")
            else:
                st.session_state.last = {"analysis": analysis, "text": text}
                counts = dict(st.session_state.counts)
                counts[analysis.sentiment] = counts.get(analysis.sentiment, 0) + 1
                st.session_state.counts = counts
                st.session_state.history = [
                    {
                        "text": text,
                        "sentiment": analysis.sentiment,
                        "confidence": analysis.confidence,
                        "time": datetime.now().strftime("%H:%M"),
                    },
                    *st.session_state.history,
                ][:20]

    if st.session_state.last:
        render_result(st.session_state.last)

with stats_col:
    st.subheader("Session insights")
    counts = st.session_state.counts
    pos, neg = counts["positive"], counts["negative"]
    total = pos + neg

    m1, m2, m3 = st.columns(3)
    m1.metric("Analyzed", total, border=True)
    m2.metric("Positive", pos, border=True)
    m3.metric("Negative", neg, border=True)

    pos_share = pos / total if total else 0.0
    neg_share = neg / total if total else 0.0
    st.markdown(
        '<div class="section-title">Sentiment split</div>'
        f'<div class="meter" role="img" aria-label="Positive {pos_share:.0%}, negative {neg_share:.0%}">'
        f'<div class="p" style="width:{pos_share:.2%}"></div>'
        f'<div class="n" style="width:{neg_share:.2%}"></div></div>'
        f'<div class="meter-labels"><span>Positive {pos_share:.0%}</span><span>Negative {neg_share:.0%}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Recent reviews</div>', unsafe_allow_html=True)
    render_history(st.session_state.history)
    if st.session_state.history:
        st.button("Clear history", on_click=clear_history, width="stretch")
    st.caption("History lives only in this browser session. Nothing is saved or uploaded.")

with st.expander("About this model"):
    st.markdown(
        "- **Data:** 50,000 labeled IMDB reviews (balanced), 80/20 train/test split.\n"
        "- **Pipeline:** lowercase → strip HTML and symbols → remove stop words → "
        "lemmatize → TF-IDF (5,000 features) → Logistic Regression.\n"
        "- **Test accuracy:** Logistic Regression 88.6% · SVM 87.8% · Naive Bayes 85.3%.\n"
        "- **Limits:** the model reads individual words, not context. Stop-word removal "
        "also drops negations like “not”, so short phrases such as “not good” can be "
        "misjudged. Sarcasm and mixed reviews are hard for it too."
    )
