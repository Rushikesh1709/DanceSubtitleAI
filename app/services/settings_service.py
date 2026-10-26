"""Validation and persistence of application settings."""
import json
from typing import Dict, Tuple

from sqlalchemy import select

from ..models import ApplicationSetting

FLOAT_RANGES = {
    "sampling_fps": (0.5, 15.0), "sensitivity": (0.5, 2.0), "min_duration": (0.5, 10.0),
    "max_duration": (2.0, 30.0), "min_gap": (0.0, 2.0), "confidence_threshold": (0.0, 1.0),
}
INT_RANGES = {"max_chars_per_line": (20, 80), "max_lines": (1, 4), "model_complexity": (0, 2)}
BOOL_KEYS = ("enable_hands", "enable_face", "show_no_pose_subtitles", "show_mudra_candidates")
CHOICES = {"low_confidence_mode": ("neutral", "hide", "show"), "output_format": ("srt", "vtt")}


def _to_bool(v) -> bool:
    return str(v).lower() in ("1", "true", "on", "yes")


def validate_settings(data: dict, defaults: dict) -> Tuple[dict, list]:
    """Return (clean settings, errors). Missing keys keep their value from `defaults`."""
    clean, errors = dict(defaults), []
    for k, (lo, hi) in FLOAT_RANGES.items():
        if k in data and data[k] not in (None, ""):
            try:
                v = float(data[k])
            except (TypeError, ValueError):
                errors.append(f"{k.replace('_', ' ')} must be a number.")
                continue
            if not lo <= v <= hi:
                errors.append(f"{k.replace('_', ' ')} must be between {lo} and {hi}.")
            else:
                clean[k] = v
    for k, (lo, hi) in INT_RANGES.items():
        if k in data and data[k] not in (None, ""):
            try:
                v = int(float(data[k]))
            except (TypeError, ValueError):
                errors.append(f"{k.replace('_', ' ')} must be a whole number.")
                continue
            if not lo <= v <= hi:
                errors.append(f"{k.replace('_', ' ')} must be between {lo} and {hi}.")
            else:
                clean[k] = v
    for k in BOOL_KEYS:
        if k in data:
            clean[k] = _to_bool(data[k])
    for k, allowed in CHOICES.items():
        if k in data and data[k] not in (None, ""):
            if data[k] in allowed:
                clean[k] = data[k]
            else:
                errors.append(f"{k.replace('_', ' ')} must be one of: {', '.join(allowed)}.")
    if clean["max_duration"] < clean["min_duration"]:
        errors.append("Maximum subtitle duration must not be smaller than the minimum duration.")
    return clean, errors


def load_settings(session, defaults: dict) -> Dict[str, object]:
    stored = {}
    for row in session.execute(select(ApplicationSetting)).scalars():
        try:
            stored[row.key] = json.loads(row.value_json)
        except ValueError:
            continue
    merged = dict(defaults)
    merged.update({k: v for k, v in stored.items() if k in defaults})
    return merged


def save_settings(session, settings: dict) -> None:
    for key, value in settings.items():
        row = session.get(ApplicationSetting, key)
        if row is None:
            session.add(ApplicationSetting(key=key, value_json=json.dumps(value)))
        else:
            row.value_json = json.dumps(value)
    session.commit()
