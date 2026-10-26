"""Small shared helpers for temporal smoothing and run-length grouping."""
from collections import Counter
from typing import Any, List, Optional, Sequence


def mode_filter(labels: Sequence[Optional[Any]], window: int) -> List[Optional[Any]]:
    """Sliding-window majority vote. `None` values are ignored; ties keep the original label.

    A window of N samples (N odd, >=1) removes rapid label flicker. If every label in a window
    is None the result for that position stays None (a genuine gap is not invented).
    """
    n = len(labels)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    half = window // 2
    out: List[Optional[Any]] = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        counts = Counter(l for l in labels[lo:hi] if l is not None)
        if not counts:
            out.append(None)
            continue
        best = max(counts.values())
        winners = [k for k, v in counts.items() if v == best]
        if labels[i] in winners:
            out.append(labels[i])
        else:
            out.append(winners[0])
    return out


def sample_edges(times: Sequence[float], end_time: Optional[float] = None) -> List[float]:
    """Return N+1 edges: sample i covers [edges[i], edges[i+1])."""
    if len(times) == 0:
        return []
    ts = [float(t) for t in times]
    if len(ts) > 1:
        diffs = sorted(b - a for a, b in zip(ts[:-1], ts[1:]) if b > a)
        dt = diffs[len(diffs) // 2] if diffs else 0.2
    else:
        dt = 0.2
    last = ts[-1] + dt
    if end_time is not None:
        last = max(ts[-1] + 1e-3, min(last, end_time))
    return ts + [last]


def group_runs(labels: Sequence[Optional[Any]]):
    """Group consecutive equal labels -> list of (label, first_index, last_index)."""
    runs = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((labels[start], start, i - 1))
            start = i
    return runs if labels else []
