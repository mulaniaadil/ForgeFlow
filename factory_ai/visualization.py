"""Matplotlib / OpenCV helpers for layouts, detections and results."""
from __future__ import annotations

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from . import config as C  # noqa: E402
from .layout_optimizer import center, path_distances, trace_path  # noqa: E402


def plot_layout(p, layout, title="", show_flows=True, show_paths=0, ax=None, save=None, halo=False):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.set_xlim(0, p.floor_w); ax.set_ylim(0, p.floor_h); ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, p.floor_w + 1, 2)); ax.set_yticks(np.arange(0, p.floor_h + 1, 2))
    ax.grid(True, alpha=0.2)
    ax.add_patch(Rectangle((0, 0), p.floor_w, p.floor_h, fill=False, lw=2.5, ec="black"))
    for o in p.obstacles:
        ax.add_patch(Rectangle(o[:2], o[2], o[3], fc="#555", ec="black"))
        if halo:
            c = p.clearance
            ax.add_patch(Rectangle((o[0] - c, o[1] - c), o[2] + 2 * c, o[3] + 2 * c, fill=False, ls=":", ec="#999"))
    for n, r in p.fixed.items():
        ax.add_patch(Rectangle(r[:2], r[2], r[3], fc=C.COLORS.get(n, "#2E7D32"), ec="black", alpha=0.9))
        ax.text(r[0] + r[2] / 2, r[1] + r[3] / 2, n[:3].upper(), ha="center", va="center", color="white", fontsize=7, weight="bold")
    for n, r in layout.items():
        ax.add_patch(Rectangle(r[:2], r[2], r[3], fc=C.COLORS.get(n, "#888"), ec="black", alpha=0.88))
        if halo:
            c = p.clearance / 2
            ax.add_patch(Rectangle((r[0] - c, r[1] - c), r[2] + 2 * c, r[3] + 2 * c, fill=False, ls=":", ec="#777", lw=0.7))
        ax.text(r[0] + r[2] / 2, r[1] + r[3] / 2, n, ha="center", va="center", color="white", fontsize=8, weight="bold")
    if show_flows:
        mx = max(p.flows.values()) if p.flows else 1
        for (a, b), f in p.flows.items():
            ra = layout.get(a) or p.fixed.get(a); rb = layout.get(b) or p.fixed.get(b)
            if ra is None or rb is None:
                continue
            (x1, y1), (x2, y2) = center(ra), center(rb)
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=9,
                                         lw=0.4 + 3.0 * f / mx, color="#222", alpha=0.45, connectionstyle="arc3,rad=0.08"))
    if show_paths:
        dist, access, maps, free = path_distances(p, layout)
        top = sorted(p.flows.items(), key=lambda kv: -kv[1])[:show_paths]
        for (a, b), f in top:
            if (a, b) in dist and a in maps:
                cells = trace_path(p, maps, access, a, b)
                if cells:
                    xs = [(cx + 0.5) * p.grid for cx, cy in cells]; ys = [(cy + 0.5) * p.grid for cx, cy in cells]
                    ax.plot(xs, ys, color="#d62728", lw=1.6, alpha=0.8)
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors="#555")
    if own:
        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=150, bbox_inches="tight")
        return fig
    return ax


def draw_detections(img_bgr, dets, scale=1):
    """Draw detection boxes + labels.  `scale` > 1 upsamples first so labels stay legible on small plans."""
    out = img_bgr.copy()
    if scale != 1:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for d in dets:
        x0, y0, x1, y1 = [int(round(v * scale)) for v in d["box"]]
        hex_col = C.COLORS.get(d["cls"], "#888888")
        col = tuple(int(hex_col.lstrip("#")[i:i + 2], 16) for i in (4, 2, 0))
        if scale > 1:
            fs, label = 0.3 * scale, f'{d["cls"]} {d["score"]:.0%}'
            cv2.rectangle(out, (x0, y0), (x1, y1), col, max(1, int(scale)))
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ty = y0 - 3 if y0 - th - 4 >= 0 else y0 + th + 3
            cv2.rectangle(out, (x0, ty - th - 2), (x0 + tw + 4, ty + base - 1), col, -1)
            cv2.putText(out, label, (x0 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
            cv2.putText(out, f'{d["cls"][:3]} {d["score"]:.2f}', (x0, max(y0 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)
    return out


def side_by_side(p, before, after, c_before, c_after, save=None, unit="unit-load-m / shift"):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2))
    plot_layout(p, before, f"Before  (path cost {c_before:,.0f} {unit})", ax=axs[0])
    if c_before and np.isfinite(c_before) and c_before > 0:
        diff = 100 * (1 - c_after / c_before)
        pct = f"-{diff:.1f}%" if diff >= 0 else f"+{-diff:.1f}%"
    else:
        pct = "N/A"
    plot_layout(p, after, f"After  (path cost {c_after:,.0f} {unit};  {pct})", ax=axs[1], show_paths=3)
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig
