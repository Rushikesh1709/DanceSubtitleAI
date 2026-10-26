"""Turns movement intervals into readable, time-aligned subtitles (deterministic templates)."""
import math
import re
import textwrap
from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Optional, Sequence, Tuple

from .hand_gesture_service import load_vocabulary

MAX_TEXT_LENGTH = 500


@dataclass
class SubtitleSettings:
    max_chars_per_line: int = 42
    max_lines: int = 2
    min_duration: float = 1.5
    max_duration: float = 6.0
    min_gap: float = 0.1
    confidence_threshold: float = 0.5
    low_confidence_mode: str = "neutral"      # neutral | hide | show
    show_no_pose_subtitles: bool = True
    show_mudra_candidates: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "SubtitleSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in names})


@dataclass
class Segment:
    start: float
    end: float
    text: str
    category: Optional[str] = None
    confidence: Optional[float] = None   # a RELIABILITY score of the visual evidence, not a probability

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def duration(self) -> float:
        return self.end - self.start


# ----------------------------------------------------------------------------- wording
ARM_CHANGE = {"raised": "raises {arm}", "out": "extends {arm} outward", "bent": "bends {arm}",
              "lowered": "lowers {arm}", "mid": "brings {arm} to chest level"}
ARM_HOLD = {"raised": "keeps {arm} raised", "out": "holds {arm} extended outward", "bent": "keeps {arm} bent",
            "lowered": "keeps {arm} lowered", "mid": "holds {arm} near chest level"}
ARM_SUBJ_CHANGE = {"raised": "rises", "out": "extends outward", "bent": "bends",
                   "lowered": "lowers", "mid": "moves to chest level"}
ARM_SUBJ_HOLD = {"raised": "stays raised", "out": "stays extended outward", "bent": "stays bent",
                 "lowered": "stays lowered", "mid": "stays near chest level"}
ARM_STATIC = {"raised": "raised", "out": "extended outward", "bent": "bent",
              "lowered": "lowered", "mid": "held near chest level"}
LEG_CLAUSE = {"upright": "while standing upright", "wide_upright": "in a wide, upright stance",
              "slight": "with slightly bent knees", "wide_slight": "in a wide stance with slightly bent knees",
              "bent": "with the knees bent", "wide_bent": "in a wide, bent-knee stance",
              "deep": "in a low stance with deeply bent knees", "wide_deep": "in a wide, low stance"}
LEG_CLAUSE_SUBJ = dict(LEG_CLAUSE, upright="as the dancer stands upright")
LEG_STATIC = {"upright": "upright", "wide_upright": "wide, upright", "slight": "slightly bent-knee",
              "wide_slight": "wide, slightly bent-knee", "bent": "bent-knee", "wide_bent": "wide, bent-knee",
              "deep": "low, deep-bend", "wide_deep": "wide, low"}
EXPRESSION_TEXT = {"smiling": "a smile", "raised_eyebrows": "raised eyebrows", "open_mouth": "an open mouth"}

NEUTRAL_TEXT = "The dancer's movement is difficult to interpret here; manual review is suggested."
NO_POSE_TEXT = "The dancer is not clearly visible in this part of the video."
CONTINUE_TEXT = {"held_posture": "The dancer continues to hold this posture.",
                 "movement": "The dancer continues this movement."}


def _arm_phrase(iv, prev) -> Tuple[str, bool]:
    """Return (text, has_own_subject). If has_own_subject is False the text follows 'The dancer'."""
    pl = prev.arm_left if (prev is not None and prev.kind == "pose") else None
    pr = prev.arm_right if (prev is not None and prev.kind == "pose") else None
    left, right = iv.arm_left, iv.arm_right
    ch_l, ch_r = left != pl, right != pr
    if left and right and left == right:
        table = ARM_CHANGE if (ch_l or ch_r) else ARM_HOLD
        return table[left].format(arm="both arms"), False
    if left and not right:
        return (ARM_CHANGE if ch_l else ARM_HOLD)[left].format(arm="the left arm"), False
    if right and not left:
        return (ARM_CHANGE if ch_r else ARM_HOLD)[right].format(arm="the right arm"), False
    # two different arm states: put the arm that changed first
    first, second = ("right", "left") if (ch_r or not ch_l) else ("left", "right")
    state = {"left": left, "right": right}
    changed = {"left": ch_l, "right": ch_r}
    p1 = (ARM_SUBJ_CHANGE if changed[first] else ARM_SUBJ_HOLD)[state[first]]
    p2 = (ARM_SUBJ_CHANGE if changed[second] else ARM_SUBJ_HOLD)[state[second]]
    return f"The {first} arm {p1} while the {second} arm {p2}", True


def _arm_static(iv) -> str:
    left, right = iv.arm_left, iv.arm_right
    if left and right and left == right:
        return f"both arms {ARM_STATIC[left]}"
    parts = []
    if right:
        parts.append(f"the right arm {ARM_STATIC[right]}")
    if left:
        parts.append(f"the left arm {ARM_STATIC[left]}")
    return " and ".join(parts)


def _hand_sentences(iv, vocab: dict, show_candidates: bool) -> List[str]:
    shapes = vocab.get("shapes", {})
    out = []
    hands = getattr(iv, "hands", {}) or {}
    changed = getattr(iv, "hand_changed", {}) or {}

    def desc(shape):
        info = shapes.get(shape, {})
        text = info.get("description", shape.replace("_", " "))
        cands = info.get("candidate_mudras") or []
        if show_candidates and cands:
            text += f" (possible resemblance to {', '.join(cands)}; not verified)"
        return text

    l, r = hands.get("left"), hands.get("right")
    if l and r and l == r and not changed:
        out.append(f"Both hands hold {desc(l)}.")
    else:
        for side, shape in (("right", r), ("left", l)):
            if changed.get(side):
                out.append(f"The {side} hand changes between hand shapes.")
            elif shape:
                out.append(f"The {side} hand holds {desc(shape)}.")
    return out


def describe_interval(iv, prev, dance_form: Optional[str], s: SubtitleSettings,
                      vocab: Optional[dict] = None) -> Tuple[Optional[str], str]:
    """Return (text or None if hidden, category)."""
    if iv.kind == "no_pose":
        return (NO_POSE_TEXT, "no_pose") if s.show_no_pose_subtitles else (None, "no_pose")
    if iv.reliability < s.confidence_threshold and s.low_confidence_mode != "show":
        return (None, "uncertain") if s.low_confidence_mode == "hide" else (NEUTRAL_TEXT, "uncertain")

    vocab = vocab if vocab is not None else load_vocabulary()
    held = iv.motion == "still"
    leg = iv.leg
    araimandi = (dance_form or "").lower() == "bharatanatyam" and leg in ("wide_bent", "wide_deep")

    if held:
        category = "held_posture"
        if iv.arm_left in (None, "lowered", "mid") and iv.arm_right in (None, "lowered", "mid") \
                and leg in (None, "upright", "wide_upright", "slight"):
            body = "The dancer holds a balanced standing posture"
        else:
            body = f"The dancer holds a posture with {_arm_static(iv)}"
            if leg:
                adj = LEG_STATIC[leg]
                body += f" in {'an' if adj[0] in 'aeiou' else 'a'} {adj} stance"
    else:
        category = "movement"
        arms, own_subject = _arm_phrase(iv, prev)
        body = arms if own_subject else f"The dancer {arms}"
        if leg:
            clause = (LEG_CLAUSE_SUBJ if own_subject else LEG_CLAUSE)[leg]
            body += (", " if own_subject else " ") + clause
    if araimandi:
        body += " (resembling a half-sitting araimandi stance)"

    extras = []
    if iv.turning and iv.lateral:
        extras.append("turning and moving sideways")
    elif iv.turning:
        extras.append("turning the body")
    elif iv.lateral:
        extras.append("moving sideways")
    if not held and iv.motion == "fast":
        extras.append("with quick movement")
    elif not held and iv.motion == "slow":
        extras.append("with slow, controlled movement")
    if extras:
        body += ", " + " and ".join(extras)
    sentences = [body + "."]
    sentences += _hand_sentences(iv, vocab, s.show_mudra_candidates)
    expr = getattr(iv, "expression", None)
    if expr in EXPRESSION_TEXT:
        sentences.append(f"The facial expression appears to be {EXPRESSION_TEXT[expr]} (estimate).")
    if getattr(iv, "repeated", False):
        sentences.append("This sequence of postures repeats.")
    text = " ".join(sentences)
    if iv.reliability < s.confidence_threshold:          # only reachable with mode == "show"
        text += " (low reliability)"
    elif iv.reliability < s.confidence_threshold + 0.15:
        text += " (uncertain)"
    return text, category


# ----------------------------------------------------------------------------- segment generation
def generate_segments(intervals: Sequence, duration: float, dance_form: Optional[str] = None,
                      settings: Optional[SubtitleSettings] = None) -> List[Segment]:
    s = settings or SubtitleSettings()
    vocab = load_vocabulary()
    segs: List[Segment] = []
    prev = None
    for iv in intervals:
        text, category = describe_interval(iv, prev, dance_form, s, vocab)
        prev = iv if iv.kind == "pose" else None
        if text is None:
            continue
        segs.append(Segment(start=iv.start, end=min(iv.end, duration) if duration else iv.end, text=text,
                            category=category, confidence=round(float(iv.reliability), 3)))
    segs = _merge_equal(segs)
    segs = _enforce_min_duration(segs, s, duration)
    segs = _split_long(segs, s)
    segs = _enforce_gaps(segs, s)
    out = []
    for g in segs:
        if duration:
            g.end = min(g.end, duration)
        if g.end - g.start >= 0.05:
            g.start, g.end = round(g.start, 3), round(g.end, 3)
            out.append(g)
    return out


def _merge_equal(segs: List[Segment]) -> List[Segment]:
    out: List[Segment] = []
    for g in segs:
        if out and out[-1].text == g.text and g.start - out[-1].end < 0.5:
            a = out[-1]
            total = a.duration + g.duration
            if a.confidence is not None and g.confidence is not None and total > 0:
                a.confidence = round((a.confidence * a.duration + g.confidence * g.duration) / total, 3)
            a.end = g.end
        else:
            out.append(Segment(g.start, g.end, g.text, g.category, g.confidence))
    return out


def _enforce_min_duration(segs: List[Segment], s: SubtitleSettings, duration: float) -> List[Segment]:
    segs = list(segs)
    i = 0
    while i < len(segs):
        g = segs[i]
        if g.duration >= s.min_duration or len(segs) == 1:
            i += 1
            continue
        limit = (segs[i + 1].start - s.min_gap) if i + 1 < len(segs) else (duration or g.end + s.min_duration)
        g.end = max(g.end, min(g.start + s.min_duration, limit))   # extend into free time if there is any
        if g.duration >= s.min_duration:
            i += 1
            continue
        if i > 0:                                                  # otherwise absorb into the previous segment
            segs[i - 1].end = max(segs[i - 1].end, g.end)
            segs.pop(i)
        elif i + 1 < len(segs):                                    # first segment: absorb into the next one
            segs[i + 1].start = g.start
            segs.pop(i)
        else:
            i += 1
    return _merge_equal(segs)


def _split_long(segs: List[Segment], s: SubtitleSettings) -> List[Segment]:
    out: List[Segment] = []
    for g in segs:
        if g.duration <= s.max_duration + 1e-6:
            out.append(g)
            continue
        n = math.ceil(g.duration / s.max_duration)
        step = g.duration / n
        for k in range(n):
            text = g.text if k == 0 else CONTINUE_TEXT.get(g.category, g.text)
            out.append(Segment(g.start + k * step, g.start + (k + 1) * step, text, g.category, g.confidence))
    return out


def _enforce_gaps(segs: List[Segment], s: SubtitleSettings) -> List[Segment]:
    for i in range(1, len(segs)):
        a, b = segs[i - 1], segs[i]
        if b.start < a.end + s.min_gap:
            new_end = b.start - s.min_gap
            if new_end - a.start >= 0.3:
                a.end = new_end
            else:
                b.start = a.end + s.min_gap
    return segs


# ----------------------------------------------------------------------------- validation & editing
def parse_timestamp(value) -> float:
    """Accepts seconds (12.5), 'MM:SS.mmm', 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm'."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        raise ValueError("empty timestamp")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid timestamp: {value!r}")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"invalid timestamp: {value!r}") from None
    total = 0.0
    for n in nums:
        if n < 0:
            raise ValueError("timestamps cannot be negative")
        total = total * 60 + n
    return total


def format_srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    sec, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def format_vtt_time(seconds: float) -> str:
    return format_srt_time(seconds).replace(",", ".")


def validate_segments(segs: Sequence[Segment], duration: Optional[float] = None) -> List[str]:
    errors = []
    prev_end = 0.0
    for i, g in enumerate(segs, start=1):
        if g.start < 0:
            errors.append(f"Subtitle {i}: start time cannot be negative.")
        if g.end <= g.start:
            errors.append(f"Subtitle {i}: end time must be after the start time.")
        if not g.text or not g.text.strip():
            errors.append(f"Subtitle {i}: text cannot be empty.")
        elif len(g.text) > MAX_TEXT_LENGTH:
            errors.append(f"Subtitle {i}: text is longer than {MAX_TEXT_LENGTH} characters.")
        if i > 1 and g.start < prev_end - 1e-6:
            errors.append(f"Subtitle {i}: overlaps the previous subtitle.")
        if duration and g.end > duration + 0.5:
            errors.append(f"Subtitle {i}: ends after the end of the video ({duration:.1f}s).")
        prev_end = max(prev_end, g.end)
    return errors


def segments_from_payload(items) -> Tuple[List[Segment], List[str]]:
    """Convert JSON items (seconds or timestamp strings) to sorted Segments plus error messages."""
    segs, errors = [], []
    if not isinstance(items, list):
        return [], ["Subtitle data must be a list."]
    for i, it in enumerate(items, start=1):
        try:
            conf = it.get("confidence")
            segs.append(Segment(parse_timestamp(it["start"]), parse_timestamp(it["end"]),
                                str(it.get("text", "")).strip(), it.get("category") or None,
                                float(conf) if conf not in (None, "") else None))
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            errors.append(f"Subtitle {i}: {exc if isinstance(exc, ValueError) else 'missing or invalid fields'}.")
    segs.sort(key=lambda g: (g.start, g.end))
    return segs, errors


def merge_segments(segs: List[Segment], index: int) -> List[Segment]:
    if index < 0 or index + 1 >= len(segs):
        raise ValueError("There is no next subtitle to merge with.")
    a, b = segs[index], segs[index + 1]
    text = a.text if a.text == b.text else f"{a.text} {b.text}".strip()
    conf = None if (a.confidence is None or b.confidence is None) else round(min(a.confidence, b.confidence), 3)
    merged = Segment(a.start, max(a.end, b.end), text, a.category, conf)
    return segs[:index] + [merged] + segs[index + 2:]


def split_segment(segs: List[Segment], index: int, at: float) -> List[Segment]:
    if index < 0 or index >= len(segs):
        raise ValueError("Subtitle not found.")
    g = segs[index]
    if not (g.start + 0.3 <= at <= g.end - 0.3):
        raise ValueError("The split time must be inside the subtitle, at least 0.3 s from each edge.")
    words = g.text.split()
    if len(words) >= 2:
        target = (at - g.start) / g.duration * len(" ".join(words))
        best, pos, best_cut = None, 0, 1
        for k, w in enumerate(words[:-1], start=1):
            pos += len(w) + (1 if k > 1 else 0)
            if best is None or abs(pos - target) < best:
                best, best_cut = abs(pos - target), k
        first, second = " ".join(words[:best_cut]), " ".join(words[best_cut:])
    else:
        first = second = g.text
    left = Segment(g.start, round(at - 0.05, 3), first, g.category, g.confidence)
    right = Segment(round(at, 3), g.end, second, g.category, g.confidence)
    return segs[:index] + [left, right] + segs[index + 1:]


# ----------------------------------------------------------------------------- SRT / WebVTT
def wrap_text(text: str, max_chars: int = 42, max_lines: int = 2) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return [""]
    width = max(10, int(max_chars))
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text]
    while len(lines) > max(1, max_lines) and width < max_chars * 3:   # never drop words: widen instead
        width += 4
        lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return lines


def to_srt(segs: Sequence[Segment], settings: Optional[SubtitleSettings] = None) -> str:
    s = settings or SubtitleSettings()
    blocks = []
    for i, g in enumerate(segs, start=1):
        body = "\n".join(wrap_text(g.text, s.max_chars_per_line, s.max_lines))
        blocks.append(f"{i}\n{format_srt_time(g.start)} --> {format_srt_time(g.end)}\n{body}\n")
    return "\n".join(blocks)


def to_vtt(segs: Sequence[Segment], settings: Optional[SubtitleSettings] = None) -> str:
    s = settings or SubtitleSettings()
    blocks = ["WEBVTT\n"]
    for i, g in enumerate(segs, start=1):
        lines = wrap_text(g.text, s.max_chars_per_line, s.max_lines)
        body = "\n".join(l.replace("&", "&amp;").replace("<", "&lt;").replace("-->", "->") for l in lines)
        blocks.append(f"{i}\n{format_vtt_time(g.start)} --> {format_vtt_time(g.end)}\n{body}\n")
    return "\n".join(blocks)
