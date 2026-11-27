# CineSense AI — Movie Review Sentiment Analyzer

An NLP and machine learning application that analyzes IMDB movie reviews and classifies
audience opinions as positive or negative using TF-IDF and machine learning algorithms.
Paste a review to get a sentiment, confidence score, and the words that influenced the
result. Built with Streamlit, TF-IDF, and Logistic Regression (~88.6% test accuracy on
50,000 IMDB reviews).

## Run it (Windows)

1. Install **Python 3.12** from https://www.python.org/downloads/ — tick **"Add python.exe to PATH"**.
2. Unzip this folder anywhere. **Delete any old `.venv` folder** inside it.
3. Double-click **`run.bat`**. The first run installs everything (a few minutes), then your
   browser opens at http://localhost:8501.

Manual alternative (Command Prompt inside this folder):

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Run it (macOS / Linux)

```bash
chmod +x run.sh && ./run.sh
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` / `py` not recognised | Reinstall Python and tick "Add python.exe to PATH". |
| Streamlit asks for an email | Just press Enter. |
| Port 8501 already in use | `python -m streamlit run app.py --server.port 8502` |
| Yellow "Offline mode" banner | NLTK language data couldn't download. Connect to the internet once and restart. |
| Model fails to load | Use the exact versions in `requirements.txt` (the model was saved with scikit-learn 1.9.1). |

## Files

- `app.py` – the web page (UI)
- `engine.py` – cleaning, prediction and word-level explanations
- `sentiment_model.pkl`, `tfidf_vectorizer.pkl` – the trained model (keep next to `app.py`)
- `main.ipynb` – training notebook (needs `IMDB Dataset.csv` in this folder, plus
  `python -m pip install pandas matplotlib seaborn jupyter`)
