"""
ENHANCED MODEL PIPELINE — HuggingFace Powered
Adds Implicit Cause Quality Analysis on top of Base Pipeline.

Extra HuggingFace models used:
  Sentiment Analysis : cardiffnlp/twitter-roberta-base-sentiment-latest
  (Semantic + SBERT already loaded by base_model via ModelRegistry)

Quality Analysis dimensions:
  1. Semantic Relevance  — SBERT cosine similarity to emotion utterance
  2. Specificity         — Flan-T5 judges whether cause is specific enough
  3. Sentiment Alignment — RoBERTa sentiment vs expected emotion polarity
"""

import torch
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from transformers import pipeline as hf_pipeline

from base_model import (
    Utterance, ImplicitCause, CausalPair, BaseModelOutput,
    InputProcessor, EmotionDetector, PairGenerator,
    ImplicitCauseGenerator, SemanticScorer,
    ImplicitCauseSelector, CausalReasoner, ModelRegistry,
    DEVICE, DEVICE_NAME,
)


# ─────────────────────────────────────────────
# SENTIMENT MODEL (singleton)
# ─────────────────────────────────────────────

class SentimentRegistry:
    _pipe = None

    @classmethod
    def sentiment_clf(cls):
        if cls._pipe is None:
            print("[ModelRegistry] Loading: cardiffnlp/twitter-roberta-base-sentiment-latest")
            cls._pipe = hf_pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=DEVICE,
                truncation=True,
                max_length=512,
            )
        return cls._pipe


# ─────────────────────────────────────────────
# ENHANCED DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class QualifiedImplicitCause:
    text: str
    base_score: float = 0.0          # SBERT combined score from base pipeline
    semantic_relevance: float = 0.0  # SBERT sim to emotion utterance
    specificity_score: float = 0.0   # Flan-T5 specificity judgment (0–1)
    sentiment_score: float = 0.0     # RoBERTa sentiment alignment
    quality_score: float = 0.0       # Weighted aggregate
    quality_class: str = "weak"      # "strong" | "weak"

@dataclass
class EnhancedCausalPair:
    cause_utt: Utterance
    emotion_utt: Utterance
    all_causes: List[QualifiedImplicitCause] = field(default_factory=list)
    strong_causes: List[QualifiedImplicitCause] = field(default_factory=list)
    weak_causes: List[QualifiedImplicitCause] = field(default_factory=list)
    best_implicit: Optional[QualifiedImplicitCause] = None
    causal_score: float = 0.0
    is_causal: bool = False
    reasoning: str = ""

@dataclass
class EnhancedModelOutput:
    utterances: List[Utterance]
    emotion_utterances: List[Utterance]
    causal_pairs: List[EnhancedCausalPair]
    best_pair: Optional[EnhancedCausalPair]
    emotion: str                           # fine-grained label
    emotion_score: float
    emotion_family: str = "neutral"
    emotion_intensity: float = 0.0
    valence: float = 0.0
    secondary_emotions: Optional[list] = None
    cause: str = "Undetermined"
    implicit_causes: List[QualifiedImplicitCause] = field(default_factory=list)
    explanation: str = ""
    quality_report: Dict = field(default_factory=dict)
    pipeline: str = "enhanced"


# ─────────────────────────────────────────────
# IMPLICIT CAUSE QUALITY ANALYZER  (NEW MODULE)
# ─────────────────────────────────────────────

class ImplicitCauseQualityAnalyzer:
    """
    Evaluates every implicit cause on 3 real model-based dimensions.

    Dimension 1 — Semantic Relevance (weight 0.40)
        SBERT cosine similarity between the implicit cause
        and the emotion utterance. Measures topical alignment.

    Dimension 2 — Specificity (weight 0.35)
        Flan-T5 answers: "Is this explanation specific and detailed? yes/no"
        A vague cause like "something happened" scores low.
        A specific cause like "failing the exam despite hard work" scores high.

    Dimension 3 — Sentiment Alignment (weight 0.25)
        cardiffnlp/twitter-roberta-base-sentiment-latest returns
        Positive / Neutral / Negative.
        We compare the sentiment direction of the implicit cause
        to the expected polarity of the detected emotion.
        Mismatch → weak signal. Alignment → strong signal.

    Classification:
        quality_score >= STRONG_THRESHOLD → "strong"
        quality_score <  STRONG_THRESHOLD → "weak"
    """

    STRONG_THRESHOLD = 0.42

    EMOTION_POLARITY = {
        # Ekman families
        "sadness": "negative", "anger": "negative", "fear": "negative",
        "disgust": "negative", "joy": "positive", "surprise": "neutral",
        "neutral": "neutral",
        # Joy sub-emotions
        "excitement": "positive", "pride": "positive", "relief": "positive",
        "gratitude": "positive", "contentment": "positive", "elation": "positive",
        # Sadness sub-emotions
        "grief": "negative", "disappointment": "negative", "loneliness": "negative",
        "despair": "negative", "nostalgia": "negative",
        # Anger sub-emotions
        "frustration": "negative", "irritation": "negative", "resentment": "negative",
        "contempt": "negative", "envy": "negative",
        # Fear sub-emotions
        "anxiety": "negative", "worry": "negative", "dread": "negative",
        "shame": "negative", "guilt": "negative", "embarrassment": "negative",
        # Surprise sub-emotions
        "shock": "neutral", "confusion": "neutral", "awe": "positive",
    }

    def __init__(self, scorer: SemanticScorer):
        self.scorer = scorer

    # ── Dimension 1 ──────────────────────────────────────────────────────────
    def _semantic_relevance(self, ic_text: str, emotion_utt: Utterance) -> float:
        return self.scorer.similarity(ic_text, emotion_utt.text)

    # ── Dimension 2 ──────────────────────────────────────────────────────────
    def _specificity(self, ic_text: str) -> float:
        """
        Ask Flan-T5: is this a specific, concrete explanation?
        Returns 1.0 (yes), 0.5 (unclear), 0.0 (no).
        """
        prompt = (
            f"Sentence: \"{ic_text}\"\n"
            f"Is this sentence a specific, concrete, and detailed explanation "
            f"(not vague or generic)? Answer only: yes or no."
        )
        answer = ModelRegistry.t5_generate(prompt, max_new_tokens=5, num_beams=2).lower()
        if answer.startswith("yes"):
            return 1.0
        if answer.startswith("no"):
            return 0.0
        return 0.5

    # ── Dimension 3 ──────────────────────────────────────────────────────────
    def _sentiment_alignment(self, ic_text: str, emotion: str) -> float:
        """
        Use RoBERTa to predict sentiment of the implicit cause.
        Compare against expected polarity for the detected emotion.
        """
        expected = self.EMOTION_POLARITY.get(emotion.lower(), "neutral")
        clf = SentimentRegistry.sentiment_clf()
        result = clf(ic_text[:512])[0]
        label  = result["label"].lower()   # "positive", "negative", "neutral"

        # Map model labels (model uses "positive"/"negative"/"neutral")
        if expected == "neutral":
            return 0.5
        if expected == "negative" and "negative" in label:
            return 1.0
        if expected == "positive" and "positive" in label:
            return 1.0
        if "neutral" in label:
            return 0.4
        return 0.0

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def analyze(self, ic: ImplicitCause, emotion_utt: Utterance) -> QualifiedImplicitCause:
        sem  = self._semantic_relevance(ic.text, emotion_utt)
        spec = self._specificity(ic.text)
        sent = self._sentiment_alignment(ic.text, emotion_utt.emotion or "neutral")

        quality = round(0.40 * sem + 0.35 * spec + 0.25 * sent, 4)
        label   = "strong" if quality >= self.STRONG_THRESHOLD else "weak"

        return QualifiedImplicitCause(
            text=ic.text,
            base_score=ic.score,
            semantic_relevance=sem,
            specificity_score=spec,
            sentiment_score=sent,
            quality_score=quality,
            quality_class=label,
        )

    def analyze_all(
        self, ics: List[ImplicitCause], emotion_utt: Utterance
    ) -> List[QualifiedImplicitCause]:
        return [self.analyze(ic, emotion_utt) for ic in ics]


# ─────────────────────────────────────────────
# ENHANCED CAUSAL REASONER
# ─────────────────────────────────────────────

class EnhancedCausalReasoner:
    """
    Same structure as base CausalReasoner but:
    - Uses ONLY strong implicit causes for reasoning.
    - Falls back to weak causes with a score penalty if none are strong.
    - Quality score is factored into the final causal score.
    """

    THRESHOLD = 0.20

    def __init__(self, scorer: SemanticScorer):
        self.scorer = scorer

    def _llm_judgment(self, cause_text, emotion_text, implicit_cause, emotion) -> tuple:
        prompt = (
            f"Cause: \"{cause_text}\"\n"
            f"Emotion statement: \"{emotion_text}\" ({emotion})\n"
            f"Implicit reason: \"{implicit_cause}\"\n"
            f"Does the cause explain the emotion given the implicit reason? "
            f"Answer only: yes or no."
        )
        answer = ModelRegistry.t5_generate(prompt, max_new_tokens=5, num_beams=2)
        return answer.lower().startswith("yes"), answer

    def reason(self, pair: EnhancedCausalPair) -> EnhancedCausalPair:
        active = pair.strong_causes if pair.strong_causes else pair.weak_causes
        penalty = 0.0 if pair.strong_causes else 0.05

        if not active:
            pair.is_causal = False
            pair.reasoning = "No implicit causes available."
            return pair

        best = max(active, key=lambda ic: ic.quality_score)
        pair.best_implicit = best

        temporal_ok    = pair.cause_utt.index <= pair.emotion_utt.index
        direct_sim     = self.scorer.similarity(pair.cause_utt.text, pair.emotion_utt.text)
        cause_impl_sim = self.scorer.similarity(pair.cause_utt.text, best.text)

        llm_causal, llm_ans = False, "skipped"
        if temporal_ok:
            llm_causal, llm_ans = self._llm_judgment(
                pair.cause_utt.text, pair.emotion_utt.text,
                best.text, pair.emotion_utt.emotion or "unknown",
            )

        llm_bonus = 0.20 if llm_causal else 0.0
        pair.causal_score = round(
            max(
                0.30 * direct_sim +
                0.25 * cause_impl_sim +
                0.20 * best.quality_score +
                0.05 * best.semantic_relevance +
                llm_bonus - penalty,
                0.0,
            ), 4
        )
        pair.is_causal = temporal_ok and (llm_causal or pair.causal_score >= self.THRESHOLD)

        source = "strong" if pair.strong_causes else "weak (fallback)"
        pair.reasoning = (
            f"Temporal: {'✓' if temporal_ok else '✗'} | "
            f"Direct sim: {direct_sim:.3f} | "
            f"Cause↔Implicit: {cause_impl_sim:.3f} | "
            f"IC quality ({source}): {best.quality_score:.3f} | "
            f"LLM verdict: {llm_ans} | "
            f"Causal score: {pair.causal_score:.3f}"
        )
        return pair


# ─────────────────────────────────────────────
# ENHANCED PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

class EnhancedModelPipeline:
    """
    Full LLM-powered Enhanced Pipeline.
    Steps 1–5 identical to Base.
    Adds: Quality Analysis → Strong/Weak filter → Enhanced reasoning.
    """

    def __init__(self):
        self.processor        = InputProcessor()
        self.detector         = EmotionDetector()
        self.pair_gen         = PairGenerator()
        self.ic_gen           = ImplicitCauseGenerator()
        self.scorer           = SemanticScorer()
        self.ic_selector      = ImplicitCauseSelector()
        self.quality_analyzer = ImplicitCauseQualityAnalyzer(self.scorer)
        self.reasoner         = EnhancedCausalReasoner(self.scorer)

    def run(self, input_data) -> EnhancedModelOutput:
        # Steps 1–2
        utterances   = self.processor.process(input_data)
        utterances   = self.detector.detect(utterances)
        emotion_utts = self.detector.get_emotion_utterances(utterances)
        if not emotion_utts:
            emotion_utts = utterances[-1:]
        print(f"[Enhanced] Emotion: {emotion_utts[0].emotion} ({emotion_utts[0].emotion_score:.3f})")

        # Step 3
        raw_pairs = self.pair_gen.generate(utterances, emotion_utts)
        enhanced_pairs: List[EnhancedCausalPair] = [
            EnhancedCausalPair(cause_utt=rp.cause_utt, emotion_utt=rp.emotion_utt)
            for rp in raw_pairs
        ]

        quality_report = {"total": 0, "strong": 0, "weak": 0}

        # Steps 4–7 (enhanced)
        for pair in enhanced_pairs:
            raw_ics  = self.ic_gen.generate(pair.emotion_utt, pair.cause_utt.text)
            scored   = self.scorer.score_implicit_causes(pair.emotion_utt, pair.cause_utt, raw_ics)

            # Quality Analysis (NEW)
            qualified           = self.quality_analyzer.analyze_all(scored, pair.emotion_utt)
            pair.all_causes     = qualified
            pair.strong_causes  = sorted(
                [ic for ic in qualified if ic.quality_class == "strong"],
                key=lambda ic: ic.quality_score, reverse=True
            )
            pair.weak_causes    = sorted(
                [ic for ic in qualified if ic.quality_class == "weak"],
                key=lambda ic: ic.quality_score, reverse=True
            )
            quality_report["total"]  += len(qualified)
            quality_report["strong"] += len(pair.strong_causes)
            quality_report["weak"]   += len(pair.weak_causes)

            # Enhanced causal reasoning (uses only strong causes)
            pair = self.reasoner.reason(pair)

        # Best pair
        causal    = [p for p in enhanced_pairs if p.is_causal]
        best_pair = (
            max(causal, key=lambda p: p.causal_score) if causal
            else max(enhanced_pairs, key=lambda p: p.causal_score) if enhanced_pairs
            else None
        )

        quality_report["strong_ratio"] = round(
            quality_report["strong"] / max(quality_report["total"], 1), 4
        )

        emotion            = emotion_utts[0].emotion            if emotion_utts else "neutral"
        emotion_score      = emotion_utts[0].emotion_score      if emotion_utts else 0.0
        emotion_family     = emotion_utts[0].emotion_family     if emotion_utts else "neutral"
        emotion_intensity  = emotion_utts[0].emotion_intensity  if emotion_utts else 0.0
        valence            = emotion_utts[0].valence            if emotion_utts else 0.0
        secondary_emotions = emotion_utts[0].secondary_emotions if emotion_utts else []
        cause              = best_pair.cause_utt.text           if best_pair    else "Undetermined"
        all_ics            = best_pair.all_causes               if best_pair    else []

        return EnhancedModelOutput(
            utterances=utterances,
            emotion_utterances=emotion_utts,
            causal_pairs=enhanced_pairs,
            best_pair=best_pair,
            emotion=emotion,
            emotion_score=emotion_score,
            emotion_family=emotion_family,
            emotion_intensity=emotion_intensity,
            valence=valence,
            secondary_emotions=secondary_emotions,
            cause=cause,
            implicit_causes=all_ics,
            explanation=self._explain(best_pair, emotion, emotion_family),
            quality_report=quality_report,
            pipeline="enhanced",
        )
    def _explain(self, pair: Optional[EnhancedCausalPair], emotion: str, family: str = "") -> str:
        if not pair:
            return (
                f"A {emotion} emotion was detected in the conversation, "
                f"but no preceding utterance could be confirmed as its cause."
            )
        best = pair.best_implicit
        cau  = pair.cause_utt.text

        family_note = f" (a form of {family})" if family and family != emotion else ""
        explanation = (
            f"The conversation expresses {emotion}{family_note}. "
            f"The most likely cause is: \"{cau}\". "
        )
        if best:
            explanation += f"The underlying reason is: {best.text} "
            quality_tag = "high-confidence" if best.quality_class == "strong" else "inferred"
            explanation += (
                f"(Quality: {quality_tag}, "
                f"causal confidence: {pair.causal_score:.2f})"
            )
        return explanation
