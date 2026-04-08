"""
COMPARISON MODULE
Runs Base and Enhanced pipelines on the same input.
Computes Accuracy, Precision, Recall, F1.
Produces qualitative analysis and error report.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from base_model     import BaseModelOutput
from enhanced_model import EnhancedModelOutput


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Metrics:
    accuracy:  float = 0.0
    precision: float = 0.0
    recall:    float = 0.0
    f1_score:  float = 0.0
    note: str = ""

@dataclass
class ComparisonResult:
    base_output:      BaseModelOutput
    enhanced_output:  EnhancedModelOutput
    base_metrics:     Metrics
    enhanced_metrics: Metrics
    qualitative:      Dict
    errors:           List[str]
    winner:           str
    delta_f1:         float


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

class MetricsCalculator:

    def compute(self, output, gt_emotion: Optional[str], gt_cause: Optional[str]) -> Metrics:
        if gt_emotion and gt_cause:
            return self._with_gt(output, gt_emotion, gt_cause)
        return self._heuristic(output)

    def _with_gt(self, output, gt_emotion: str, gt_cause: str) -> Metrics:
        em_match = float(
            output.emotion.lower() in gt_emotion.lower() or
            gt_emotion.lower() in output.emotion.lower()
        )
        pred_tok = set(output.cause.lower().split())
        gt_tok   = set(gt_cause.lower().split())
        overlap  = pred_tok & gt_tok
        prec     = len(overlap) / max(len(pred_tok), 1)
        rec      = len(overlap) / max(len(gt_tok),   1)
        f1       = 2 * prec * rec / max(prec + rec, 1e-9)
        acc      = (em_match + f1) / 2
        return Metrics(
            accuracy=round(acc,  4),
            precision=round(prec, 4),
            recall=round(rec,    4),
            f1_score=round(f1,   4),
            note="Evaluated against ground truth labels"
        )

    def _heuristic(self, output) -> Metrics:
        score = 0.0
        if output.emotion and output.emotion != "neutral":
            score += 0.25
        # Emotion confidence from HF model
        score += 0.25 * min(getattr(output, "emotion_score", 0.5), 1.0)
        if output.cause and "Undetermined" not in output.cause:
            score += 0.20
        if output.best_pair and output.best_pair.causal_score > 0:
            score += 0.30 * min(output.best_pair.causal_score / 0.60, 1.0)
        score = round(score, 4)
        return Metrics(
            accuracy=score, precision=score,
            recall=score,   f1_score=score,
            note="Heuristic self-evaluation (no ground truth)"
        )


# ─────────────────────────────────────────────
# QUALITATIVE COMPARISON
# ─────────────────────────────────────────────

def qualitative_compare(base: BaseModelOutput, enhanced: EnhancedModelOutput) -> Dict:
    same_emotion = base.emotion == enhanced.emotion
    same_cause   = base.cause   == enhanced.cause
    agreement    = "full" if (same_emotion and same_cause) else (
                   "partial" if same_emotion else "diverged")

    base_score = base.best_pair.causal_score     if base.best_pair     else 0.0
    enh_score  = enhanced.best_pair.causal_score if enhanced.best_pair else 0.0
    qr         = enhanced.quality_report

    return {
        "agreement_level": agreement,
        "emotion_agreement": same_emotion,
        "cause_agreement":   same_cause,
        "base": {
            "emotion": base.emotion,
            "emotion_confidence": base.emotion_score,
            "cause":   base.cause,
            "num_implicit_causes": len(base.implicit_causes),
            "causal_score": round(base_score, 4),
        },
        "enhanced": {
            "emotion": enhanced.emotion,
            "emotion_confidence": enhanced.emotion_score,
            "cause":   enhanced.cause,
            "num_implicit_causes": len(enhanced.implicit_causes),
            "causal_score": round(enh_score, 4),
            "quality_report": qr,
        },
        "delta_causal_score": round(enh_score - base_score, 4),
    }


def collect_errors(base: BaseModelOutput, enhanced: EnhancedModelOutput) -> List[str]:
    errors = []
    if base.emotion == "neutral":
        errors.append("Base: No non-neutral emotion detected.")
    if enhanced.emotion == "neutral":
        errors.append("Enhanced: No non-neutral emotion detected.")
    if "Undetermined" in base.cause:
        errors.append("Base: Causal utterance could not be determined.")
    if "Undetermined" in enhanced.cause:
        errors.append("Enhanced: Causal utterance could not be determined.")
    if base.best_pair and base.best_pair.causal_score < 0.15:
        errors.append(f"Base: Low causal confidence ({base.best_pair.causal_score:.3f}).")
    if enhanced.best_pair and enhanced.best_pair.causal_score < 0.15:
        errors.append(f"Enhanced: Low causal confidence ({enhanced.best_pair.causal_score:.3f}).")
    strong_ratio = enhanced.quality_report.get("strong_ratio", 1.0)
    if strong_ratio < 0.20:
        errors.append(f"Enhanced: Very few strong ICs ({strong_ratio*100:.0f}%). Causes may be vague.")
    return errors or ["No significant errors detected."]


# ─────────────────────────────────────────────
# COMPARISON ORCHESTRATOR
# ─────────────────────────────────────────────

class ComparisonSystem:

    def __init__(self):
        from base_model     import BaseModelPipeline
        from enhanced_model import EnhancedModelPipeline
        self.base     = BaseModelPipeline()
        self.enhanced = EnhancedModelPipeline()
        self.metrics  = MetricsCalculator()

    def compare(
        self,
        input_data,
        gt_emotion: Optional[str] = None,
        gt_cause:   Optional[str] = None,
    ) -> ComparisonResult:
        base_out = self.base.run(input_data)
        enh_out  = self.enhanced.run(input_data)

        bm = self.metrics.compute(base_out, gt_emotion, gt_cause)
        em = self.metrics.compute(enh_out,  gt_emotion, gt_cause)

        qual   = qualitative_compare(base_out, enh_out)
        errors = collect_errors(base_out, enh_out)

        delta  = round(em.f1_score - bm.f1_score, 4)
        winner = "enhanced" if delta > 0.01 else ("base" if delta < -0.01 else "tie")

        return ComparisonResult(
            base_output=base_out,
            enhanced_output=enh_out,
            base_metrics=bm,
            enhanced_metrics=em,
            qualitative=qual,
            errors=errors,
            winner=winner,
            delta_f1=delta,
        )
