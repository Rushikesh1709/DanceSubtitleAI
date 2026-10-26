"""Synthetic data builders shared by the tests (no real model is involved)."""
import cv2
import numpy as np


def make_pose(arm_l="lowered", arm_r="lowered", knees="straight", cx=0.5, vis=0.95, yaw_scale=1.0):
    """A (33, 4) MediaPipe-style landmark array (normalised, aspect ratio 1) with a chosen posture."""
    p = np.zeros((33, 4), dtype=np.float32)
    p[:, 3] = vis

    def S(i, x, y):
        p[i, 0], p[i, 1] = x, y

    sh_y, hip_y, hw = 0.30, 0.55, 0.09 * yaw_scale
    S(11, cx - hw, sh_y); S(12, cx + hw, sh_y); S(23, cx - 0.07, hip_y); S(24, cx + 0.07, hip_y)
    S(0, cx, 0.2); S(7, cx - 0.03, 0.2); S(8, cx + 0.03, 0.2)

    def arm(side, state):
        sx = cx - hw if side == "l" else cx + hw
        d = -1 if side == "l" else 1
        el, wr = (13, 15) if side == "l" else (14, 16)
        if state == "raised":
            S(el, sx + d * 0.04, sh_y - 0.08); S(wr, sx + d * 0.05, sh_y - 0.18)
        elif state == "out":
            S(el, sx + d * 0.12, sh_y + 0.01); S(wr, sx + d * 0.25, sh_y + 0.02)
        elif state == "lowered":
            S(el, sx + d * 0.01, sh_y + 0.14); S(wr, sx + d * 0.01, sh_y + 0.28)
        elif state == "bent":
            S(el, sx + d * 0.08, sh_y + 0.08); S(wr, sx + d * 0.02, sh_y + 0.02)

    arm("l", arm_l); arm("r", arm_r)
    if knees == "straight":
        S(25, cx - 0.07, 0.70); S(26, cx + 0.07, 0.70); S(27, cx - 0.07, 0.85); S(28, cx + 0.07, 0.85)
    else:  # wide stance with bent knees
        S(25, cx - 0.16, 0.68); S(26, cx + 0.16, 0.68); S(27, cx - 0.11, 0.82); S(28, cx + 0.11, 0.82)
    return p


def make_sequence(blocks, dt=0.2):
    """blocks: list of (n_samples, kwargs for make_pose | None for 'no detection'). -> times, poses"""
    times, poses, t = [], [], 0.0
    for n, kw in blocks:
        for _ in range(n):
            times.append(round(t, 3))
            poses.append(None if kw is None else make_pose(**kw))
            t += dt
    return times, poses


def make_hand(extended=(), thumb=False, spread=False):
    """21 hand landmarks (x, y) of a hand pointing up; `extended` lists straight fingers."""
    lm = np.zeros((21, 2), dtype=np.float32)
    lm[0] = (0.5, 0.9)
    mcp = {"index": (0.44, 0.65), "middle": (0.50, 0.63), "ring": (0.56, 0.65), "pinky": (0.62, 0.68)}
    ids = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    fan = {"index": -0.08, "middle": -0.02, "ring": 0.04, "pinky": 0.10}
    for name, base in ids.items():
        mx, my = mcp[name]
        dx = fan[name] if spread else 0.0
        lm[base] = (mx, my)
        if name in extended:
            lm[base + 1] = (mx + dx * 0.3, my - 0.07)
            lm[base + 2] = (mx + dx * 0.6, my - 0.12)
            lm[base + 3] = (mx + dx, my - 0.17)
        else:
            lm[base + 1] = (mx, my - 0.06)
            lm[base + 2] = (mx, my - 0.03)
            lm[base + 3] = (mx, my)
    lm[1], lm[2], lm[3] = (0.46, 0.85), (0.41, 0.78), (0.37, 0.72)
    lm[4] = (0.33, 0.67) if thumb else (0.50, 0.74)
    if not thumb:
        lm[3] = (0.45, 0.77)
    return lm


def make_video(path, n_frames=50, fps=10, size=(160, 120)):
    """Write a small synthetic video whose frames differ (a moving square). Returns the path."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "OpenCV cannot write mp4v videos on this machine"
    for i in range(n_frames):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        x = int((i / max(1, n_frames - 1)) * (w - 30))
        cv2.rectangle(frame, (x, 40), (x + 30, 80), (0, 200, 255), -1)
        writer.write(frame)
    writer.release()
    return str(path)
