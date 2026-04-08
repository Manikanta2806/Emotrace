# EmoTrace — Enhanced Emotion–Cause Extraction
### Implicit Cause Quality Analysis Pipeline

> A research + production NLP system with two independently comparable pipelines.

---

## Project Structure

```
emotion_project/
├── backend/
│   ├── base_model.py        ← Base pipeline (7 steps)
│   ├── enhanced_model.py    ← Enhanced pipeline (+ Quality Analysis)
│   ├── comparison.py        ← Comparison system + metrics
│   └── main.py              ← FastAPI server
├── frontend/
│   └── index.html           ← Full UI (works standalone + with API)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
cd emotion_project
pip install -r requirements.txt

# Optional: TextBlob data (for sentiment alignment)
python -m textblob.download_corpora
```

### 2. Start FastAPI backend

```bash
cd backend
uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 3. Open the frontend

```bash
# Simply open in browser:
open frontend/index.html
# Or serve it:
python -m http.server 3000 --directory frontend
```

---

## API Reference

### POST `/analyze/base`
Run the **Base Model Pipeline**.

```json
{
  "text": "I studied hard but failed the exam and feel sad",
  "ground_truth_emotion": "sadness",   // optional
  "ground_truth_cause": "failed the exam"  // optional
}
```

**Response:**
```json
{
  "pipeline": "base",
  "emotion": "sadness",
  "cause": "I studied hard but failed the exam and feel sad",
  "implicit_causes": [
    {"text": "The person experienced a loss...", "score": 0.5210}
  ],
  "explanation": "Emotion 'sadness' detected in: ...",
  "causal_score": 0.4213
}
```

---

### POST `/analyze/enhanced`
Run the **Enhanced Model Pipeline** (with quality analysis).

Same request format. Response includes quality-classified implicit causes.

---

### POST `/compare`
Run **both pipelines** and get full comparison.

**Response:**
```json
{
  "base": { ... },
  "enhanced": { ... },
  "base_metrics":     {"accuracy": 0.39, "precision": 0.41, "recall": 0.41, "f1_score": 0.41},
  "enhanced_metrics": {"accuracy": 0.43, "precision": 0.46, "recall": 0.45, "f1_score": 0.45},
  "qualitative":  { "agreement_level": "partial", "delta": 0.02 },
  "errors":       ["..."],
  "winner":       "enhanced",
  "delta_f1":     0.0458
}
```

---

## Architecture Overview

### Base Model Pipeline (7 Steps)

| Step | Module | Description |
|------|--------|-------------|
| 1 | `InputProcessor` | RECCON dict / raw text / utterance list → `List[Utterance]` |
| 2 | `EmotionDetector` | Keyword-based emotion annotation (7 classes) |
| 3 | `PairGenerator` | (cause, emotion) pairs with temporal ordering |
| 4 | `ImplicitCauseGenerator` | Template-based hypotheses per emotion |
| 5 | `SemanticScorer` | TF-IDF cosine similarity (→ Sentence-BERT when available) |
| 6 | `ImplicitCauseSelector` | Best implicit cause by combined score |
| 7 | `CausalReasoner` | Temporal + semantic threshold → is_causal + explanation |

### Enhanced Model Pipeline (adds Quality Analysis)

Steps 1–5 identical to Base. Then:

| Step | Module | Description |
|------|--------|-------------|
| 6 | `ImplicitCauseQualityAnalyzer` | 3-dimensional quality scoring |
| 7 | Quality Filtering | Strong (≥0.35) vs Weak classification |
| 8 | `EnhancedCausalReasoner` | Uses ONLY strong implicit causes |

**Quality Dimensions:**

| Dimension | Weight | Method |
|-----------|--------|--------|
| Semantic Relevance | 0.45 | Sentence-BERT similarity to emotion utterance |
| Specificity | 0.30 | Word count, numbers, named entities |
| Sentiment Alignment | 0.25 | TextBlob polarity vs emotion polarity map |

---

## Example Output

**Input:** `"I studied hard but failed the exam and feel sad"`

**Base Model:**
```
Emotion: SADNESS
Cause:   "I studied hard but failed the exam and feel sad"
Causal Score: 0.4213
Best Implicit: "The person experienced a loss or failure that caused sadness."
```

**Enhanced Model:**
```
Emotion: SADNESS
Cause:   "I studied hard but failed the exam and feel sad"
Causal Score: 0.4891
Strong Causes: 2 | Weak (filtered): 4
Best Implicit: "An important goal was not achieved, leading to sadness."
  Quality → Class: STRONG | Semantic: 0.521 | Specificity: 0.612 | Sentiment: 1.000 | Score: 0.641
```

---

## Extending the System

### Add a Pretrained Emotion Classifier
Replace `EmotionDetector._detect_emotion()` with a HuggingFace model:
```python
from transformers import pipeline
clf = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
result = clf(text)[0]
return result["label"].lower()
```

### Enable Sentence-BERT
Install and it auto-enables:
```bash
pip install sentence-transformers
```
`SemanticScorer` auto-detects and upgrades from TF-IDF to `all-MiniLM-L6-v2`.

### Supabase Integration
```python
from supabase import create_client
sb = create_client(SUPABASE_URL, SUPABASE_KEY)
sb.table("analyses").insert({"input": text, "emotion": emotion, "cause": cause}).execute()
```

---

## Keyboard Shortcuts (Frontend)
- `Ctrl+Enter` / `Cmd+Enter` — Run analysis or comparison

---

## Dependencies
- `fastapi` + `uvicorn` — Backend API
- `sentence-transformers` — Semantic scoring (optional, auto-detected)
- `textblob` — Sentiment alignment (optional, auto-detected)
- `transformers` — For emotion model extension
- `supabase` — Database integration (optional)
