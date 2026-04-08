"""
FastAPI Backend — EmoTrace
Endpoints: POST /analyze/base | /analyze/enhanced | /compare
All inference via real HuggingFace models (no API keys required).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from base_model     import BaseModelPipeline
from enhanced_model import EnhancedModelPipeline
from comparison     import ComparisonSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EmoTrace — Emotion-Cause Extraction API",
    description="HuggingFace-powered dual-pipeline NLP system",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True,
)

# Initialize pipelines at startup (models load on first call)
base_pipe = BaseModelPipeline()
enh_pipe  = EnhancedModelPipeline()
cmp_sys   = ComparisonSystem()


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

class ConversationInput(BaseModel):
    text: str = Field(..., description="Raw text or Speaker: utterance format")
    ground_truth_emotion: Optional[str] = None
    ground_truth_cause:   Optional[str] = None

class ICResponse(BaseModel):
    text: str
    score: float
    quality_class: Optional[str] = None
    quality_score: Optional[float] = None
    semantic_relevance: Optional[float] = None
    specificity_score: Optional[float] = None
    sentiment_score: Optional[float] = None

class AnalysisResponse(BaseModel):
    pipeline: str
    emotion: str
    emotion_score: float
    cause: str
    implicit_causes: List[ICResponse]
    explanation: str
    causal_score: float

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    note: str

class CompareResponse(BaseModel):
    base:             AnalysisResponse
    enhanced:         AnalysisResponse
    base_metrics:     MetricsResponse
    enhanced_metrics: MetricsResponse
    qualitative:      Dict[str, Any]
    errors:           List[str]
    winner:           str
    delta_f1:         float


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _format_ics_base(output) -> List[ICResponse]:
    ics = []
    for ic in (output.implicit_causes or [])[:6]:
        ics.append(ICResponse(
            text=ic.text,
            score=round(ic.score, 4),
        ))
    return ics

def _format_ics_enhanced(output) -> List[ICResponse]:
    ics = []
    for ic in (output.implicit_causes or [])[:6]:
        ics.append(ICResponse(
            text=ic.text,
            score=round(ic.base_score, 4),
            quality_class=ic.quality_class,
            quality_score=round(ic.quality_score, 4),
            semantic_relevance=round(ic.semantic_relevance, 4),
            specificity_score=round(ic.specificity_score, 4),
            sentiment_score=round(ic.sentiment_score, 4),
        ))
    return ics

def _to_analysis(output, pipeline_name: str) -> AnalysisResponse:
    is_enhanced = pipeline_name == "enhanced"
    ics = _format_ics_enhanced(output) if is_enhanced else _format_ics_base(output)
    cs  = output.best_pair.causal_score if output.best_pair else 0.0
    return AnalysisResponse(
        pipeline=pipeline_name,
        emotion=output.emotion,
        emotion_score=round(output.emotion_score, 4),
        cause=output.cause,
        implicit_causes=ics,
        explanation=output.explanation,
        causal_score=round(cs, 4),
    )


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "running", "version": "2.0.0", "models": [
        "j-hartmann/emotion-english-distilroberta-base",
        "google/flan-t5-large",
        "sentence-transformers/all-MiniLM-L6-v2",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ]}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze/base", response_model=AnalysisResponse)
def analyze_base(payload: ConversationInput):
    try:
        out = base_pipe.run(payload.text)
        return _to_analysis(out, "base")
    except Exception as e:
        logger.exception("/analyze/base failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/enhanced", response_model=AnalysisResponse)
def analyze_enhanced(payload: ConversationInput):
    try:
        out = enh_pipe.run(payload.text)
        return _to_analysis(out, "enhanced")
    except Exception as e:
        logger.exception("/analyze/enhanced failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare", response_model=CompareResponse)
def compare(payload: ConversationInput):
    try:
        result = cmp_sys.compare(
            payload.text,
            gt_emotion=payload.ground_truth_emotion,
            gt_cause=payload.ground_truth_cause,
        )
        return CompareResponse(
            base=_to_analysis(result.base_output, "base"),
            enhanced=_to_analysis(result.enhanced_output, "enhanced"),
            base_metrics=MetricsResponse(**vars(result.base_metrics)),
            enhanced_metrics=MetricsResponse(**vars(result.enhanced_metrics)),
            qualitative=result.qualitative,
            errors=result.errors,
            winner=result.winner,
            delta_f1=result.delta_f1,
        )
    except Exception as e:
        logger.exception("/compare failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
