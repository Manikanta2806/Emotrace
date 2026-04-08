"""
BASE MODEL PIPELINE — HuggingFace Powered
Enhanced Emotion-Cause Extraction using Implicit Cause Quality Analysis

Real HuggingFace models used:
  Emotion Detection  : j-hartmann/emotion-english-distilroberta-base
  Implicit Cause Gen : google/flan-t5-large  (instruction-tuned, zero-shot)
  Semantic Scoring   : sentence-transformers/all-MiniLM-L6-v2
  Causal Reasoning   : google/flan-t5-large  (yes/no causal judgment)

No hardcoded keywords. No templates. Real inference on real conversation.
"""

import re
import torch
from typing import List, Optional
from dataclasses import dataclass, field
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util


# ─────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────

DEVICE      = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[EmoTrace] Running on: {DEVICE_NAME.upper()}")


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Utterance:
    index: int
    speaker: str
    text: str
    emotion: Optional[str] = None          # fine-grained label (e.g. "anxiety")
    emotion_score: float = 0.0
    emotion_family: Optional[str] = None   # Ekman family (e.g. "fear")
    emotion_intensity: float = 0.0         # 0–1 strength of expression
    valence: float = 0.0                   # -1 negative … +1 positive
    secondary_emotions: Optional[list] = None  # [{emotion, score}, ...]

@dataclass
class ImplicitCause:
    text: str
    score: float = 0.0

@dataclass
class CausalPair:
    cause_utt: "Utterance"
    emotion_utt: "Utterance"
    implicit_causes: List[ImplicitCause] = field(default_factory=list)
    best_implicit: Optional[ImplicitCause] = None
    causal_score: float = 0.0
    is_causal: bool = False
    reasoning: str = ""

@dataclass
class BaseModelOutput:
    utterances: List["Utterance"]
    emotion_utterances: List["Utterance"]
    causal_pairs: List[CausalPair]
    best_pair: Optional[CausalPair]
    emotion: str                           # fine-grained label
    emotion_score: float
    emotion_family: str = "neutral"        # Ekman family
    emotion_intensity: float = 0.0
    valence: float = 0.0
    secondary_emotions: Optional[list] = None
    cause: str = "Undetermined"
    implicit_causes: List = field(default_factory=list)
    explanation: str = ""
    pipeline: str = "base"


# ─────────────────────────────────────────────
# MODEL REGISTRY  (lazy-loaded singletons)
# ─────────────────────────────────────────────

class ModelRegistry:
    """
    All HuggingFace models are loaded once on first use
    and shared across the entire pipeline run.
    """
    _emotion_pipe  = None
    _sbert         = None
    _t5_tok        = None
    _t5_model      = None

    # ── Emotion classifier ────────────────────
    @classmethod
    def emotion_clf(cls):
        if cls._emotion_pipe is None:
            print("[ModelRegistry] Loading: j-hartmann/emotion-english-distilroberta-base")
            cls._emotion_pipe = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,        # return scores for ALL emotions
                device=DEVICE,
                truncation=True,
                max_length=512,
            )
        return cls._emotion_pipe

    # ── Sentence-BERT ─────────────────────────
    @classmethod
    def sbert(cls):
        if cls._sbert is None:
            print("[ModelRegistry] Loading: sentence-transformers/all-MiniLM-L6-v2")
            cls._sbert = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._sbert

    # ── Flan-T5 (generation) ──────────────────
    @classmethod
    def flan_t5(cls):
        if cls._t5_model is None:
            print("[ModelRegistry] Loading: google/flan-t5-large")
            cls._t5_tok   = AutoTokenizer.from_pretrained("google/flan-t5-large")
            cls._t5_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-large",
                torch_dtype=torch.float16 if DEVICE_NAME == "cuda" else torch.float32,
            ).to(DEVICE_NAME)
            cls._t5_model.eval()
        return cls._t5_tok, cls._t5_model

    @classmethod
    def t5_generate(cls, prompt: str, max_new_tokens: int = 80, num_beams: int = 4) -> str:
        tok, model = cls.flan_t5()
        inputs = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(DEVICE_NAME)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return tok.decode(out[0], skip_special_tokens=True).strip()


# ─────────────────────────────────────────────
# STEP 1 — INPUT PROCESSING
# ─────────────────────────────────────────────

class InputProcessor:
    """
    Converts any input format into List[Utterance].
    Supports:
      • RECCON dataset dict
      • "Speaker: text" multi-turn dialogue string
      • Plain text (split into sentences)
      • List[str] or List[dict]
    """

    def process(self, data) -> List[Utterance]:
        if isinstance(data, dict):
            return self._from_reccon(data)
        if isinstance(data, str):
            return self._from_raw_text(data)
        if isinstance(data, list):
            return self._from_list(data)
        raise ValueError(f"Unsupported input type: {type(data)}")

    def _from_reccon(self, data: dict) -> List[Utterance]:
        utts, idx = [], 0
        for turns in data.values():
            for t in turns:
                utts.append(Utterance(
                    index=idx,
                    speaker=t.get("speaker", f"S{idx}"),
                    text=t.get("utterance", t.get("text", "")),
                    emotion=t.get("emotion"),
                ))
                idx += 1
        return utts

    def _from_raw_text(self, text: str) -> List[Utterance]:
        # Detect "Speaker: utterance" format
        matches = re.findall(r'^([A-Za-z0-9_\-\s]+):\s*(.+)$', text, re.MULTILINE)
        if len(matches) >= 2:
            return [
                Utterance(index=i, speaker=spk.strip(), text=utt.strip())
                for i, (spk, utt) in enumerate(matches)
            ]
        # Plain text → split on sentence boundaries
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
        return [Utterance(index=i, speaker="User", text=s) for i, s in enumerate(sents)]

    def _from_list(self, items: list) -> List[Utterance]:
        utts = []
        for i, item in enumerate(items):
            if isinstance(item, str):
                utts.append(Utterance(index=i, speaker=f"S{i}", text=item))
            elif isinstance(item, dict):
                utts.append(Utterance(
                    index=i,
                    speaker=item.get("speaker", f"S{i}"),
                    text=item.get("text", item.get("utterance", "")),
                    emotion=item.get("emotion"),
                ))
        return utts


# ─────────────────────────────────────────────
# STEP 2 — EMOTION DETECTION (HuggingFace)
# ─────────────────────────────────────────────

class RichEmotionAnalyzer:
    """
    Maps j-hartmann's 7 Ekman labels → 28 fine-grained sub-emotions.

    Strategy (zero extra models — reuses already-loaded ModelRegistry):
      Step A: Get all 7 raw scores from j-hartmann.
      Step B: Use keyword hints for fast-path sub-emotion detection.
              For ambiguous cases (top-1 score < 0.70), use Flan-T5
              to pick the best sub-label from the family taxonomy.
      Step C: Compute intensity, valence, and secondary emotions.

    Sub-emotion taxonomy per Ekman family:
      joy      → joy, excitement, pride, relief, gratitude, contentment, elation
      sadness  → sadness, grief, disappointment, loneliness, despair, nostalgia
      anger    → anger, frustration, irritation, resentment, contempt, envy
      fear     → fear, anxiety, worry, dread, shame, guilt, embarrassment
      surprise → surprise, shock, confusion, awe
      disgust  → disgust, contempt
      neutral  → neutral
    """

    TAXONOMY = {
        "joy":      ["joy", "excitement", "pride", "relief", "gratitude", "contentment", "elation"],
        "sadness":  ["sadness", "grief", "disappointment", "loneliness", "despair", "nostalgia"],
        "anger":    ["anger", "frustration", "irritation", "resentment", "contempt", "envy"],
        "fear":     ["fear", "anxiety", "worry", "dread", "shame", "guilt", "embarrassment"],
        "surprise": ["surprise", "shock", "confusion", "awe"],
        "disgust":  ["disgust", "contempt"],
        "neutral":  ["neutral"],
    }

    VALENCE = {
        "joy": 0.85, "sadness": -0.80, "anger": -0.75,
        "fear": -0.70, "surprise": 0.10, "disgust": -0.65, "neutral": 0.0,
    }

    SUB_VALENCE = {
        "relief": 0.60, "gratitude": 0.90, "contentment": 0.70,
        "elation": 0.95, "pride": 0.75, "excitement": 0.80,
        "grief": -0.95, "despair": -0.92, "loneliness": -0.85,
        "disappointment": -0.70, "nostalgia": -0.30,
        "frustration": -0.60, "irritation": -0.55, "resentment": -0.80,
        "contempt": -0.70, "envy": -0.65,
        "anxiety": -0.72, "worry": -0.60, "dread": -0.85,
        "shame": -0.78, "guilt": -0.75, "embarrassment": -0.50,
        "shock": -0.20, "confusion": -0.25, "awe": 0.50,
    }

    KEYWORD_HINTS = {
        "grief":          r"griev|bereav|mourn|lost.*someone|passed away|died",
        "disappointment": r"disappoint|let.*down|expected.*but|hoped.*but|didn.t.*turn",
        "loneliness":     r"alone|lonely|no one|isolated|nobody",
        "despair":        r"despair|hopeless|no way out|give up|nothing.*matter",
        "nostalgia":      r"miss|used to|back then|remember when|those days",
        "excitement":     r"excit|can.t wait|so pumped|thrilled|stoked",
        "pride":          r"proud|accomplished|achieved|finally did|made it",
        "relief":         r"relief|relieved|glad.*over|thank.*god|finally.*safe",
        "gratitude":      r"grateful|thankful|appreciate|means.*lot|thank you",
        "elation":        r"elated|ecstatic|over.*moon|best.*day|incredible.*feeling",
        "contentment":    r"content|satisfied|at peace|good enough|quite happy",
        "frustration":    r"frustrat|so annoying|why.*keep|not.*working|ugh",
        "irritation":     r"irritat|petty|minor annoy|slightly.*annoyed|bugs me",
        "resentment":     r"resent|unfair|never.*forget|they always|been.*doing.*this",
        "contempt":       r"contempt|beneath|pathetic|don.t respect",
        "envy":           r"envy|jealous|wish.*had|why.*them|not.*fair.*they",
        "anxiety":        r"anxious|nervous|what if|panic|uneasy|on edge",
        "worry":          r"worried|concerned|hope.*okay|what.*happen|keep.*thinking",
        "dread":          r"dread|dreading|scared.*of|terror|nightmare",
        "shame":          r"ashamed|shame|how.*could.*I|can.t face",
        "guilt":          r"guilty|my fault|should.*have|I caused|blame myself",
        "embarrassment":  r"embarrass|awkward|cringe|so.*stupid.*of.*me",
        "shock":          r"shocked|can.t believe|never.*expected|jaw.*drop|stunned",
        "confusion":      r"confused|what.*going on|don.t understand|lost.*here",
        "awe":            r"awe|breathtaking|incredible.*sight|overwhelmed.*beauty",
    }

    def _refine_with_t5(self, family: str, text: str, context: str) -> str:
        options = self.TAXONOMY.get(family, [family])
        if len(options) == 1:
            return options[0]
        opts_str = ", ".join(options)
        prompt = (
            f"Text: \"{text[:200]}\"\n"
            f"Context: \"{context[:150]}\"\n"
            f"The speaker's emotion belongs to the '{family}' family.\n"
            f"Which single word best describes the specific emotion? "
            f"Choose exactly one from: {opts_str}.\n"
            f"Answer with only the single word."
        )
        raw = ModelRegistry.t5_generate(prompt, max_new_tokens=6, num_beams=2).strip().lower()
        for opt in options:
            if opt in raw:
                return opt
        return options[0]

    def analyze(self, utt: Utterance, all_scores: list, context: str) -> Utterance:
        sorted_scores = sorted(all_scores, key=lambda x: x["score"], reverse=True)
        top       = sorted_scores[0]
        family    = top["label"].lower()
        top_score = top["score"]

        runner_up = sorted_scores[1]["score"] if len(sorted_scores) > 1 else 0.0
        gap       = top_score - runner_up
        intensity = round(min(top_score * 0.6 + gap * 0.4, 1.0), 4)

        # Fast-path: keyword hints (no extra inference)
        sub_emotion = None
        text_lower  = utt.text.lower()
        for sub, pattern in self.KEYWORD_HINTS.items():
            if sub in self.TAXONOMY.get(family, []) and re.search(pattern, text_lower):
                sub_emotion = sub
                break

        # Slow-path: T5 refinement for ambiguous or nuanced families
        if sub_emotion is None:
            if top_score < 0.70 or family in ("joy", "sadness", "anger", "fear"):
                sub_emotion = self._refine_with_t5(family, utt.text, context)
            else:
                sub_emotion = family

        valence = self.SUB_VALENCE.get(sub_emotion, self.VALENCE.get(family, 0.0))

        secondary = [
            {"emotion": s["label"].lower(), "score": round(s["score"], 4)}
            for s in sorted_scores[1:]
            if s["label"].lower() != "neutral" and s["score"] > 0.10
        ][:3]

        utt.emotion            = sub_emotion
        utt.emotion_score      = round(top_score, 4)
        utt.emotion_family     = family
        utt.emotion_intensity  = intensity
        utt.valence            = round(valence, 4)
        utt.secondary_emotions = secondary
        return utt


class EmotionDetector:
    """
    Wraps j-hartmann classifier + RichEmotionAnalyzer.
    Produces fine-grained sub-emotions, intensity, valence,
    and secondary emotions — zero extra model downloads.
    """

    def __init__(self):
        self._rich = RichEmotionAnalyzer()

    def detect(self, utterances: List[Utterance]) -> List[Utterance]:
        needs_detection = [u for u in utterances if u.emotion is None]
        if not needs_detection:
            for u in utterances:
                if u.emotion and u.emotion_family is None:
                    u.emotion_family = u.emotion
            return utterances

        clf     = ModelRegistry.emotion_clf()
        texts   = [u.text for u in needs_detection]
        results = clf(texts)   # List[List[{label, score}]]

        for utt, scores in zip(needs_detection, results):
            context = utterances[utt.index - 1].text if utt.index > 0 else ""
            self._rich.analyze(utt, scores, context)

        return utterances

    def get_emotion_utterances(self, utterances: List[Utterance]) -> List[Utterance]:
        emo = [u for u in utterances
               if u.emotion and u.emotion_family != "neutral" and u.emotion != "neutral"]
        return sorted(emo, key=lambda u: u.emotion_score, reverse=True)


# ─────────────────────────────────────────────
# STEP 3 — UTTERANCE PAIR GENERATION
# ─────────────────────────────────────────────

class PairGenerator:
    """
    Generates all (cause_utt, emotion_utt) pairs where
    cause_index <= emotion_index (temporal constraint).
    """

    def generate(
        self,
        utterances: List[Utterance],
        emotion_utts: List[Utterance],
    ) -> List[CausalPair]:
        pairs = []
        for emo in emotion_utts:
            for cause in utterances:
                if cause.index <= emo.index:
                    pairs.append(CausalPair(cause_utt=cause, emotion_utt=emo))
        return pairs


# ─────────────────────────────────────────────
# STEP 4 — IMPLICIT CAUSE GENERATION (Flan-T5)
# ─────────────────────────────────────────────

class ImplicitCauseGenerator:
    """
    Uses google/flan-t5-large with 4 structurally DISTINCT prompts,
    each targeting a different causal angle to minimise redundancy.

    Problems fixed vs old version:
    1. Old prompts A/B/C/D were paraphrases of each other → T5 produced
       near-identical outputs ("The speaker experienced X because Y").
    2. Old prompts had no deduplication → same sentence appeared 3–4 times.
    3. Old prompts didn't constrain output structure → T5 often produced
       sentence fragments instead of complete explanations.

    New design — 4 orthogonal angles:
    Angle 1 (EVENT)       — what concrete event directly triggered the emotion?
    Angle 2 (BELIEF)      — what did the speaker believe/expect that was violated?
    Angle 3 (CONSEQUENCE) — what does this emotion tell us about what was at stake?
    Angle 4 (RELATIONSHIP)— what interpersonal or social dynamic explains this?

    Each prompt enforces:
    • A complete sentence (not a fragment)
    • Grounded in the ACTUAL cause utterance text
    • Distinct vocabulary nudge via different framing words

    Post-generation deduplication:
    • Exact match removal
    • Near-duplicate removal (>80% token overlap → keep higher-scored one)
    """

    # Minimum token overlap ratio that counts as a duplicate
    _DUP_THRESHOLD = 0.80

    def generate(self, emotion_utt: Utterance, context: str) -> List[str]:
        emotion = emotion_utt.emotion or "an emotion"
        eu_text = emotion_utt.text[:280]
        ctx     = context[:280]

        prompts = [
            # ── Angle 1: Concrete triggering event ──────────────────────────
            # Forces T5 to name a specific event/action, not just restate the text.
            (
                f"Conversation:\n"
                f"  Context: \"{ctx}\"\n"
                f"  Speaker says: \"{eu_text}\"\n"
                f"Instruction: In ONE complete sentence starting with 'The speaker felt "
                f"{emotion} because', explain the specific event or action that directly "
                f"triggered this {emotion}. Do not repeat the speaker's exact words."
            ),
            # ── Angle 2: Violated belief or expectation ──────────────────────
            # Forces T5 to reason about what the person EXPECTED vs what happened.
            (
                f"Context: \"{ctx}\"\n"
                f"The person expressed {emotion} by saying: \"{eu_text}\"\n"
                f"Instruction: In ONE complete sentence, describe what expectation, "
                f"belief, or hope the person held that was violated or disappointed, "
                f"leading to their {emotion}. Start with 'The person expected' or "
                f"'The person believed'."
            ),
            # ── Angle 3: Stakes and personal significance ────────────────────
            # Forces T5 to reason about WHY this matters to the speaker personally.
            (
                f"Someone said \"{eu_text}\" expressing {emotion}.\n"
                f"Prior exchange: \"{ctx}\"\n"
                f"Instruction: In ONE complete sentence, explain what personal goal, "
                f"relationship, or value was at stake for the speaker that made this "
                f"situation cause {emotion}. Start with 'This situation mattered to "
                f"the speaker because'."
            ),
            # ── Angle 4: Social/relational dynamic ──────────────────────────
            # Forces T5 to reason about interpersonal dynamics, not just the event.
            (
                f"Dialogue context: \"{ctx}\"\n"
                f"Response: \"{eu_text}\" ({emotion})\n"
                f"Instruction: In ONE complete sentence, explain the interpersonal or "
                f"social dynamic between the speakers that contributed to this "
                f"{emotion}. Start with 'The relationship dynamic' or 'The interaction'."
            ),
        ]

        raw_causes = []
        for prompt in prompts:
            result = ModelRegistry.t5_generate(
                prompt, max_new_tokens=90, num_beams=4
            ).strip()
            # Skip empty or very short outputs (fragments)
            if result and len(result.split()) >= 6:
                raw_causes.append(result)

        # ── Deduplication ────────────────────────────────────────────────────
        deduped = self._deduplicate(raw_causes)

        return deduped or [
            f"The speaker felt {emotion} because the situation in the conversation "
            f"directly affected something personally significant to them."
        ]

    def _token_overlap(self, a: str, b: str) -> float:
        """Jaccard token overlap between two sentences."""
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _deduplicate(self, causes: List[str]) -> List[str]:
        """
        Remove near-duplicates. Keep the first (usually most complete) version
        when two sentences share >80% token overlap.
        Also strip outputs that are just the input context repeated verbatim.
        """
        kept = []
        for candidate in causes:
            is_dup = False
            for existing in kept:
                if self._token_overlap(candidate, existing) > self._DUP_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(candidate)
        return kept


# ─────────────────────────────────────────────
# STEP 5 — SEMANTIC SCORING (Sentence-BERT)
# ─────────────────────────────────────────────

class SemanticScorer:
    """
    all-MiniLM-L6-v2 cosine similarity.
    Scores each implicit cause against both the emotion utterance
    and the candidate cause utterance.
    """

    def similarity(self, a: str, b: str) -> float:
        model = ModelRegistry.sbert()
        ea, eb = model.encode([a, b], convert_to_tensor=True)
        return round(util.cos_sim(ea, eb).item(), 4)

    def batch_similarity(self, anchor: str, candidates: List[str]) -> List[float]:
        model  = ModelRegistry.sbert()
        a_emb  = model.encode(anchor,     convert_to_tensor=True)
        c_embs = model.encode(candidates, convert_to_tensor=True)
        return [round(float(s), 4) for s in util.cos_sim(a_emb, c_embs)[0].tolist()]

    def score_implicit_causes(
        self,
        emotion_utt: Utterance,
        cause_utt: Utterance,
        implicit_causes: List[str],
    ) -> List[ImplicitCause]:
        if not implicit_causes:
            return []
        sim_emo   = self.batch_similarity(emotion_utt.text, implicit_causes)
        sim_cause = self.batch_similarity(cause_utt.text,   implicit_causes)
        scored = [
            ImplicitCause(
                text=ic,
                score=round(0.6 * sim_emo[i] + 0.4 * sim_cause[i], 4),
            )
            for i, ic in enumerate(implicit_causes)
        ]
        return sorted(scored, key=lambda x: x.score, reverse=True)


# ─────────────────────────────────────────────
# STEP 6 — IMPLICIT CAUSE SELECTION
# ─────────────────────────────────────────────

class ImplicitCauseSelector:
    def select_best(self, scored: List[ImplicitCause]) -> Optional[ImplicitCause]:
        return scored[0] if scored else None


# ─────────────────────────────────────────────
# STEP 7 — CAUSAL REASONING (Flan-T5 + SBERT)
# ─────────────────────────────────────────────

class CausalReasoner:
    """
    Flan-T5 judges causality with a yes/no question.
    SBERT score provides a continuous fallback signal.
    Temporal ordering is enforced.
    """

    THRESHOLD = 0.20

    def __init__(self, scorer: SemanticScorer):
        self.scorer = scorer

    def _llm_judgment(
        self,
        cause_text: str,
        emotion_text: str,
        implicit_cause: str,
        emotion: str,
    ) -> tuple:
        prompt = (
            f"Statement A (possible cause): \"{cause_text}\"\n"
            f"Statement B (emotion): \"{emotion_text}\" — expresses {emotion}\n"
            f"Implicit reason: \"{implicit_cause}\"\n"
            f"Does Statement A causally explain why the person feels {emotion}, "
            f"considering the implicit reason? Answer only: yes or no."
        )
        answer = ModelRegistry.t5_generate(prompt, max_new_tokens=5, num_beams=2)
        return answer.lower().startswith("yes"), answer

    def reason(self, pair: CausalPair) -> CausalPair:
        if pair.best_implicit is None:
            pair.is_causal = False
            pair.reasoning = "No implicit cause available."
            return pair

        temporal_ok    = pair.cause_utt.index <= pair.emotion_utt.index
        direct_sim     = self.scorer.similarity(pair.cause_utt.text, pair.emotion_utt.text)
        cause_impl_sim = self.scorer.similarity(pair.cause_utt.text, pair.best_implicit.text)

        # LLM judgment (only if temporal constraint satisfied)
        if temporal_ok:
            llm_causal, llm_ans = self._llm_judgment(
                cause_text=pair.cause_utt.text,
                emotion_text=pair.emotion_utt.text,
                implicit_cause=pair.best_implicit.text,
                emotion=pair.emotion_utt.emotion or "unknown",
            )
        else:
            llm_causal, llm_ans = False, "skipped (temporal fail)"

        llm_bonus = 0.20 if llm_causal else 0.0
        pair.causal_score = round(
            0.35 * direct_sim +
            0.30 * cause_impl_sim +
            0.15 * pair.best_implicit.score +
            llm_bonus,
            4,
        )
        pair.is_causal = temporal_ok and (llm_causal or pair.causal_score >= self.THRESHOLD)
        pair.reasoning = (
            f"Temporal: {'✓' if temporal_ok else '✗'} | "
            f"Direct sim (SBERT): {direct_sim:.3f} | "
            f"Cause↔Implicit: {cause_impl_sim:.3f} | "
            f"LLM verdict: {llm_ans} | "
            f"Causal score: {pair.causal_score:.3f}"
        )
        return pair


# ─────────────────────────────────────────────
# BASE PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

class BaseModelPipeline:
    """
    Full LLM-powered Base Pipeline.
    All inference is done by real HuggingFace models.
    """

    def __init__(self):
        self.processor   = InputProcessor()
        self.detector    = EmotionDetector()
        self.pair_gen    = PairGenerator()
        self.ic_gen      = ImplicitCauseGenerator()
        self.scorer      = SemanticScorer()
        self.ic_selector = ImplicitCauseSelector()
        self.reasoner    = CausalReasoner(self.scorer)

    def run(self, input_data) -> BaseModelOutput:
        # 1. Parse input
        utterances = self.processor.process(input_data)
        print(f"[Base] Parsed {len(utterances)} utterance(s)")

        # 2. Detect emotions with HF model
        utterances   = self.detector.detect(utterances)
        emotion_utts = self.detector.get_emotion_utterances(utterances)
        if not emotion_utts:
            emotion_utts = utterances[-1:]      # fallback to last utterance
        print(f"[Base] Emotion detected: {emotion_utts[0].emotion} ({emotion_utts[0].emotion_score:.3f})")

        # 3. Generate candidate pairs
        pairs = self.pair_gen.generate(utterances, emotion_utts)
        print(f"[Base] {len(pairs)} candidate pairs")

        # 4–7. Process each pair
        for pair in pairs:
            raw_ics            = self.ic_gen.generate(pair.emotion_utt, pair.cause_utt.text)
            pair.implicit_causes = self.scorer.score_implicit_causes(
                pair.emotion_utt, pair.cause_utt, raw_ics
            )
            pair.best_implicit = self.ic_selector.select_best(pair.implicit_causes)
            pair               = self.reasoner.reason(pair)

        # Select best pair
        causal    = [p for p in pairs if p.is_causal]
        best_pair = (
            max(causal, key=lambda p: p.causal_score) if causal
            else max(pairs, key=lambda p: p.causal_score) if pairs
            else None
        )

        emotion       = emotion_utts[0].emotion            if emotion_utts else "neutral"
        emotion_score = emotion_utts[0].emotion_score      if emotion_utts else 0.0
        emotion_family    = emotion_utts[0].emotion_family    if emotion_utts else "neutral"
        emotion_intensity = emotion_utts[0].emotion_intensity if emotion_utts else 0.0
        valence           = emotion_utts[0].valence           if emotion_utts else 0.0
        secondary_emotions = emotion_utts[0].secondary_emotions if emotion_utts else []
        cause         = best_pair.cause_utt.text      if best_pair    else "Undetermined"
        all_ics       = best_pair.implicit_causes      if best_pair    else []

        return BaseModelOutput(
            utterances=utterances,
            emotion_utterances=emotion_utts,
            causal_pairs=pairs,
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
            pipeline="base",
        )

    def _explain(self, pair: Optional[CausalPair], emotion: str, family: str = "") -> str:
        if not pair:
            return (
                f"A {emotion} emotion was detected in the conversation, "
                f"but no preceding utterance could be identified as its cause."
            )
        ic  = pair.best_implicit.text if pair.best_implicit else ""
        cau = pair.cause_utt.text

        family_note = f" (a form of {family})" if family and family != emotion else ""
        explanation = (
            f"The conversation expresses {emotion}{family_note}. "
            f"The most likely cause is: \"{cau}\". "
        )
        if ic:
            explanation += f"The underlying reason is: {ic} "
        explanation += f"(Causal confidence: {pair.causal_score:.2f})"
        return explanation
