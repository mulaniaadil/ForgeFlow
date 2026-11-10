"""Synthetic factory floor-plan generator with COCO-style annotations.

Every image is a top-down plan with: thick outer wall, grid, one of each of 6 machines
(random size / rotation / position), Receiving + Dispatch dock markers, 0-3 columns,
unlabelled distractor pallets and aisle markings, and a 5 m scale bar.  Three visual styles
(colour, grey-hatch, blueprint) are used for in-distribution data; an extra "ood" split
adds a re-coloured palette, noise, blur and JPEG artefacts to probe robustness.

IMPORTANT: this is *synthetic* data.  It validates the pipeline and gives measurable
detector metrics, but it is not a substitute for real annotated factory plans.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .layout_optimizer import Problem, random_feasible_layout
from .process_flow import build_flows

CODES = {"Cutting": "CUT", "Drilling": "DRL", "Assembly": "ASM", "Inspection": "INS",
         "Packaging": "PKG", "Storage": "STO", "Receiving": "IN", "Dispatch": "OUT"}
HATCH = {"Cutting": "/", "Drilling": "\\", "Assembly": "+", "Inspection": ".",
         "Packaging": "-", "Storage": "|"}


def _bgr(hexcolor):
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


STYLES = {
    "color": dict(bg=(247, 247, 247), grid=(226, 226, 226), wall=(40, 40, 40), outline=(30, 30, 30),
                  text=(255, 255, 255), hatch=(255, 255, 255), fill=True),
    "gray": dict(bg=(240, 240, 240), grid=(220, 220, 220), wall=(50, 50, 50), outline=(20, 20, 20),
                 text=(10, 10, 10), hatch=(60, 60, 60), fill=False),
    "blueprint": dict(bg=(110, 55, 20), grid=(135, 80, 40), wall=(245, 245, 245), outline=(250, 250, 250),
                      text=(255, 255, 255), hatch=(235, 235, 235), fill=False),
}


def _hatch(canvas, x0, y0, x1, y1, kind, color, sp=6):
    if x1 - x0 < 3 or y1 - y0 < 3:
        return
    sub = canvas[y0:y1, x0:x1].copy()
    h, w = sub.shape[:2]
    if kind == "/":
        for k in range(-h, w, sp):
            cv2.line(sub, (k, h), (k + h, 0), color, 1)
    elif kind == "\\":
        for k in range(-h, w, sp):
            cv2.line(sub, (k, 0), (k + h, h), color, 1)
    elif kind == "+":
        for k in range(sp // 2, w, sp):
            cv2.line(sub, (k, 0), (k, h), color, 1)
        for k in range(sp // 2, h, sp):
            cv2.line(sub, (0, k), (w, k), color, 1)
    elif kind == "-":
        for k in range(sp // 2, h, sp):
            cv2.line(sub, (0, k), (w, k), color, 1)
    elif kind == "|":
        for k in range(sp // 2, w, sp):
            cv2.line(sub, (k, 0), (k, h), color, 1)
    elif kind == ".":
        for yy in range(sp // 2, h, sp):
            for xx in range(sp // 2, w, sp):
                cv2.circle(sub, (xx, yy), 1, color, -1)
    canvas[y0:y1, x0:x1] = sub


def plan_transform(W, H):
    s = min((C.IMG_W - C.MARGIN_L - C.MARGIN_R) / W, (C.IMG_H - C.MARGIN_T - C.MARGIN_B) / H)
    fx0 = (C.IMG_W - W * s) / 2.0
    fy0 = float(C.MARGIN_T)
    fy1 = fy0 + H * s
    return s, fx0, fy0, fy1


def rect_to_px(r, s, fx0, fy1):
    x, y, w, h = r
    return (int(round(fx0 + x * s)), int(round(fy1 - (y + h) * s)),
            int(round(fx0 + (x + w) * s)), int(round(fy1 - y * s)))


def make_scene(rng: random.Random):
    """Random floor, docks, columns and a feasible 'current' machine layout (metres)."""
    W = rng.choice([26, 28, 30, 32, 34, 36])
    H = rng.choice([18, 20, 22, 24])
    dw, dl = C.DOCK_SIZE
    # Receiving on left or top wall, Dispatch on right or bottom wall
    if rng.random() < 0.5:
        rec = (0.0, rng.randint(2, int((H - dl) / 0.5) - 2) * 0.5 + 2.0, dw, dl)
    else:
        rec = (rng.randint(2, int((W - dl) / 0.5) - 2) * 0.5 + 1.0, H - dw, dl, dw)
    if rng.random() < 0.5:
        dis = (W - dw, rng.randint(2, int((H - dl) / 0.5) - 2) * 0.5 + 0.5, dw, dl)
    else:
        dis = (rng.randint(2, int((W - dl) / 0.5) - 2) * 0.5 + 1.0, 0.0, dl, dw)
    cols = []
    for _ in range(rng.randint(0, 3)):
        for _ in range(100):
            c = (rng.randint(6, int((W - 6) / 0.5)) * 0.5, rng.randint(6, int((H - 6) / 0.5)) * 0.5,
                 C.COLUMN_SIZE, C.COLUMN_SIZE)
            if all(abs(c[0] - o[0]) > 3 or abs(c[1] - o[1]) > 3 for o in cols):
                cols.append(c)
                break
    machines = {}
    for n, (w, h) in C.NOMINAL_SIZE.items():
        f = rng.uniform(0.85, 1.2)
        machines[n] = (round(w * f * 2) / 2, round(h * f * 2) / 2)
    prob = Problem(W, H, machines, build_flows(), fixed={"Receiving": rec, "Dispatch": dis},
                   obstacles=cols, clearance=1.0, wall_margin=0.5)
    for _ in range(50):
        try:
            layout = random_feasible_layout(prob, rng, max_restarts=30, tries=200)
            return prob, layout
        except RuntimeError:
            continue
    raise RuntimeError("scene generation failed")


def render(prob: Problem, layout, style_name: str, rng: random.Random, ood=False):
    st = dict(STYLES[style_name])
    W, H = prob.floor_w, prob.floor_h
    s, fx0, fy0, fy1 = plan_transform(W, H)
    canvas = np.full((C.IMG_H, C.IMG_W, 3), st["bg"], np.uint8)
    # grid every 2 m (ood: different spacing)
    gm = rng.choice([2, 2.5, 3]) if ood else 2
    xm = 0.0
    while xm <= W + 1e-6:
        px = int(round(fx0 + xm * s))
        cv2.line(canvas, (px, int(fy0)), (px, int(round(fy1))), st["grid"], 1)
        xm += gm
    ym = 0.0
    while ym <= H + 1e-6:
        py = int(round(fy1 - ym * s))
        cv2.line(canvas, (int(round(fx0)), py), (int(round(fx0 + W * s)), py), st["grid"], 1)
        ym += gm

    palette = {n: _bgr(c) for n, c in C.COLORS.items()}
    if ood:   # shuffled palette -> colour is no longer a reliable cue
        names = list(C.MACHINES)
        cols = [palette[n] for n in names]
        rng.shuffle(cols)
        palette.update(dict(zip(names, cols)))

    # distractors: pallets + aisle markings (not annotated)
    occupied = list(layout.values()) + list(prob.obstacles) + list(prob.fixed.values())
    for _ in range(rng.randint(2, 4)):
        for _ in range(30):
            w, h = rng.choice([(1.0, 1.2), (1.2, 1.0), (0.8, 0.8)])
            r = (rng.uniform(1, W - 2), rng.uniform(1, H - 2), w, h)
            if all(r[0] + r[2] + 0.3 < o[0] or o[0] + o[2] + 0.3 < r[0] or
                   r[1] + r[3] + 0.3 < o[1] or o[1] + o[3] + 0.3 < r[1] for o in occupied):
                x0, y0, x1, y1 = rect_to_px(r, s, fx0, fy1)
                col = (175, 175, 175) if style_name != "blueprint" else (170, 120, 70)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), col, -1)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), st["grid"] if style_name == "blueprint" else (120, 120, 120), 1)
                occupied.append(r)
                break
    for _ in range(rng.randint(1, 2)):          # dashed aisle centre-line
        yline = rng.uniform(2, H - 2)
        py = int(round(fy1 - yline * s))
        for xx in range(int(fx0) + 4, int(fx0 + W * s) - 4, 12):
            cv2.line(canvas, (xx, py), (xx + 6, py), (60, 190, 230) if style_name != "gray" else (150, 150, 150), 1)

    boxes = []   # (class, (x0,y0,x1,y1))

    def draw_obj(name, r, fillcol, code, hatch):
        x0, y0, x1, y1 = rect_to_px(r, s, fx0, fy1)
        if st["fill"] or name in C.DOCKS:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), fillcol, -1)
        elif style_name == "gray":
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (205, 205, 205), -1)
        if hatch:
            _hatch(canvas, x0, y0, x1, y1, hatch, st["hatch"] if name not in C.DOCKS else (255, 255, 255))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), st["outline"], 1)
        if code:
            fs = 0.32
            (tw, th), _ = cv2.getTextSize(code, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            tx, ty = (x0 + x1 - tw) // 2, (y0 + y1 + th) // 2
            tcol = st["text"]
            if style_name == "color" and hatch:
                cv2.rectangle(canvas, (tx - 1, ty - th - 1), (tx + tw + 1, ty + 2), fillcol, -1)
            elif style_name == "gray":
                cv2.rectangle(canvas, (tx - 1, ty - th - 1), (tx + tw + 1, ty + 2), (240, 240, 240), -1)
            cv2.putText(canvas, code, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, tcol, 1, cv2.LINE_AA)
        boxes.append((name, (x0, y0, x1, y1)))

    for n, r in layout.items():
        draw_obj(n, r, palette[n], CODES[n], HATCH[n])
    for n, r in prob.fixed.items():
        draw_obj(n, r, palette[n], CODES[n], None)
    for r in prob.obstacles:
        x0, y0, x1, y1 = rect_to_px(r, s, fx0, fy1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (70, 70, 70) if style_name != "blueprint" else (200, 200, 200), -1)
        cv2.line(canvas, (x0, y0), (x1, y1), (230, 230, 230) if style_name != "blueprint" else (110, 55, 20), 1)
        cv2.line(canvas, (x0, y1), (x1, y0), (230, 230, 230) if style_name != "blueprint" else (110, 55, 20), 1)
        boxes.append((C.OBSTACLE, (x0, y0, x1, y1)))

    # outer wall (centre-line = floor boundary)
    cv2.rectangle(canvas, (int(round(fx0)), int(round(fy0))), (int(round(fx0 + W * s)), int(round(fy1))),
                  st["wall"], C.WALL_PX)
    # scale bar
    L = int(round(C.SCALE_BAR_M * s))
    bx0, by = 14, C.IMG_H - 12
    cv2.line(canvas, (bx0, by), (bx0 + L, by), st["wall"], 2)
    cv2.line(canvas, (bx0, by - 4), (bx0, by + 4), st["wall"], C.SCALE_BAR_TICK_PX)
    cv2.line(canvas, (bx0 + L, by - 4), (bx0 + L, by + 4), st["wall"], C.SCALE_BAR_TICK_PX)
    cv2.putText(canvas, f"{int(C.SCALE_BAR_M)} m", (bx0 + L + 6, by + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                st["wall"], 1, cv2.LINE_AA)

    if ood:
        f = rng.uniform(0.8, 1.15)
        canvas = np.clip(canvas.astype(np.float32) * f, 0, 255).astype(np.uint8)
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    sigma = rng.uniform(5, 12) if ood else (rng.uniform(0, 4) if rng.random() < 0.5 else 0)
    if sigma > 0:
        noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, sigma, canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if ood:
        ok, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 45])
        canvas = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    meta = dict(floor_w=W, floor_h=H, scale_px_per_m=s, floor_px=[fx0, fy0, fx0 + W * s, fy1],
                scale_bar_m=C.SCALE_BAR_M, scale_bar_px=L,
                layout={n: list(r) for n, r in layout.items()},
                fixed={n: list(r) for n, r in prob.fixed.items()},
                obstacles=[list(o) for o in prob.obstacles],
                machines={n: list(v) for n, v in prob.machines.items()})
    return canvas, boxes, meta


def generate_dataset(out_dir, n_id=700, n_ood=100, seed=0, split=(0.70, 0.15, 0.15)):
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    n_tr, n_va = round(n_id * split[0]), round(n_id * split[1])
    split_of = lambda i: "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    coco = {k: dict(images=[], annotations=[], categories=[{"id": i, "name": n} for n, i in C.CLASS_TO_ID.items()])
            for k in ["train", "val", "test", "ood"]}
    metas = {k: {} for k in coco}
    ann_id = 1
    styles = list(STYLES)
    order = list(range(n_id))
    rng.shuffle(order)   # split by image: indices are shuffled, then assigned to splits
    jobs = [(split_of(k), styles[order[k] % 3], False) for k in range(n_id)]
    jobs += [("ood", rng.choice(styles), True) for _ in range(n_ood)]
    for idx, (sp, style, ood) in enumerate(jobs):
        prob, layout = make_scene(rng)
        img, boxes, meta = render(prob, layout, style, rng, ood=ood)
        fn = f"{sp}_{idx:04d}.png"
        cv2.imwrite(str(out / "images" / fn), img)
        iid = idx + 1
        coco[sp]["images"].append(dict(id=iid, file_name=fn, width=C.IMG_W, height=C.IMG_H, style=style, ood=ood))
        for name, (x0, y0, x1, y1) in boxes:
            coco[sp]["annotations"].append(dict(id=ann_id, image_id=iid, category_id=C.CLASS_TO_ID[name],
                                                bbox=[x0, y0, x1 - x0, y1 - y0], area=(x1 - x0) * (y1 - y0), iscrowd=0))
            ann_id += 1
        meta["style"] = style
        metas[sp][fn] = meta
    for sp in coco:
        (out / f"{sp}.json").write_text(json.dumps(coco[sp]))
        (out / f"{sp}_meta.json").write_text(json.dumps(metas[sp]))
    return {k: len(v["images"]) for k, v in coco.items()}
