"""Pixel -> metre calibration from a scale bar, plus floor-boundary (wall) detection (OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np

from . import config as C


def _foreground(img_bgr):
    """Binary mask of 'ink' (walls, scale bar).  Handles dark-on-light and light-on-dark plans.
    Threshold = midpoint between the background level and the extreme ink level (robust to
    brightness change / JPEG), after an edge-preserving 3x3 median filter."""
    if img_bgr is None or not hasattr(img_bgr, "shape") or img_bgr.size == 0:
        raise ValueError("Input image is None or empty")
    gray = cv2.medianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 3)
    bg = float(np.median(gray))
    if bg > 127:
        ink = float(np.percentile(gray, 1.0))
        return (gray < (bg + ink) / 2.0).astype(np.uint8)
    ink = float(np.percentile(gray, 99.5))
    return (gray > (bg + ink) / 2.0).astype(np.uint8)


def find_floor_rect(img_bgr, wall_px=C.WALL_PX):
    """Return (x0, y0, x1, y1) of the wall centre-line (px) or None.  Largest outer contour."""
    fg = _foreground(img_bgr)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, area = None, 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > area:
            best, area = (x, y, w, h), w * h
    if best is None or area < 0.25 * img_bgr.shape[0] * img_bgr.shape[1]:
        return None
    x, y, w, h = best
    half = float(np.ceil(wall_px / 2.0))     # cv2 draws a t-px line about (t+2)-px wide on screen
    return (x + half, y + half, x + w - 1 - half, y + h - 1 - half)


def find_scale_bar(img_bgr, floor_rect=None, tick_px=C.SCALE_BAR_TICK_PX):
    """Measure the horizontal scale-bar length in pixels (below the floor outline)."""
    H = img_bgr.shape[0]
    y_start = int(floor_rect[3]) + 4 if floor_rect is not None else int(0.85 * H)
    fg = _foreground(img_bgr)[y_start:, :]
    if fg.size == 0:
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if w >= 15 and h <= 14 and w / max(h, 1) >= 3:
            if best is None or w > best[2]:
                best = (x, y, w, h)
    if best is None:
        return None
    return float(best[2] - (2 * np.ceil(tick_px / 2.0) + 1))   # remove the end-tick width (rendered, inclusive)


def calibrate(img_bgr, scale_bar_m=C.SCALE_BAR_M, manual_px_per_m=None):
    """Return dict(px_per_m, floor_rect, floor_w_m, floor_h_m, warnings)."""
    warn = []
    fr = find_floor_rect(img_bgr)
    if fr is None:
        warn.append("floor outline not found - using image margins as the floor boundary")
        fr = (C.MARGIN_L, C.MARGIN_T, img_bgr.shape[1] - C.MARGIN_R, img_bgr.shape[0] - C.MARGIN_B)
    if manual_px_per_m:
        s = float(manual_px_per_m)
    else:
        L = find_scale_bar(img_bgr, fr)
        if L is None:
            warn.append("scale bar not found - supply px/m manually")
            s = None
        else:
            s = L / scale_bar_m
    out = dict(
        px_per_m=s,
        floor_rect=fr,
        floor_w_m=((fr[2] - fr[0]) / s if s else None),
        floor_h_m=((fr[3] - fr[1]) / s if s else None),
        warnings=warn,
    )
    return out
