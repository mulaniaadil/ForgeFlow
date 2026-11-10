"""End-to-end pipeline: floor-plan image -> detections -> metres -> editable layout -> optimisation."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import config as C
from .calibration import calibrate
from .detector import predict
from .layout_optimizer import Problem, optimize
from .process_flow import build_flows


@dataclass
class Scene:
    floor_w: float
    floor_h: float
    layout: dict                 # machine -> (x, y, w, h) metres
    fixed: dict                  # dock -> (x, y, w, h)
    obstacles: list              # [(x, y, w, h)]
    px_per_m: float
    detections: list = field(default_factory=list)   # raw kept detections (px)
    warnings: list = field(default_factory=list)


def _nms_agnostic(dets, thr=0.6):
    from .detector import iou_matrix
    dets = sorted(dets, key=lambda d: -d["score"])
    keep = []
    for d in dets:
        if all(iou_matrix(np.array([d["box"]]), np.array([k["box"]]))[0, 0] < thr for k in keep):
            keep.append(d)
    return keep


def detect_scene(img_bgr, model, scale_bar_m=C.SCALE_BAR_M, score_thr=0.3, manual_px_per_m=None, device="cpu"):
    cal = calibrate(img_bgr, scale_bar_m, manual_px_per_m)
    warn = list(cal["warnings"])
    if not cal["px_per_m"]:
        raise ValueError("; ".join(warn) or "calibration failed")
    s = cal["px_per_m"]
    fx0, fy0, fx1, fy1 = cal["floor_rect"]
    pred = predict(model, [cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)], device, score_thr=score_thr)[0]
    dets = [dict(cls=C.ID_TO_CLASS[int(l)], score=float(sc), box=[float(v) for v in b])
            for b, l, sc in zip(pred["boxes"], pred["labels"], pred["scores"])]
    dets = _nms_agnostic(dets)

    def to_m(b):
        x0, y0, x1, y1 = b
        return (round(max((x0 - fx0) / s, 0.0), 2), round(max((fy1 - y1) / s, 0.0), 2),
                round((x1 - x0) / s, 2), round((y1 - y0) / s, 2))

    layout, fixed, obstacles, kept = {}, {}, [], []
    for name in C.MACHINES + C.DOCKS:
        c = [d for d in dets if d["cls"] == name]
        if not c:
            warn.append(f"{name} not detected")
            continue
        d = max(c, key=lambda d: d["score"])
        kept.append(d)
        (layout if name in C.MACHINES else fixed)[name] = to_m(d["box"])
    for d in dets:
        if d["cls"] == C.OBSTACLE:
            obstacles.append(to_m(d["box"]))
            kept.append(d)
    return Scene(floor_w=round((fx1 - fx0) / s, 2), floor_h=round((fy1 - fy0) / s, 2), layout=layout,
                 fixed=fixed, obstacles=obstacles, px_per_m=s, detections=kept, warnings=warn)


def make_problem(scene: Scene, flows=None, clearance=1.0, detection_margin=0.3, wall_margin=0.5, grid=0.5, allow_rotation=True):
    avail = set(scene.layout) | set(scene.fixed)
    flows = build_flows(available=avail) if flows is None else {k: v for k, v in flows.items()
                                                                if k[0] in avail and k[1] in avail}
    machines = {n: (r[2], r[3]) for n, r in scene.layout.items()}
    # detection_margin pads clearance to absorb ~0.2 m box/calibration error from the detector
    return Problem(scene.floor_w, scene.floor_h, machines, flows, dict(scene.fixed), list(scene.obstacles),
                   clearance=clearance + detection_margin, wall_margin=wall_margin, grid=grid, allow_rotation=allow_rotation)


def run_pipeline(img_bgr, model, scale_bar_m=C.SCALE_BAR_M, score_thr=0.3, manual_px_per_m=None,
                 flows=None, clearance=1.0, n_seeds=6, iters=25000, refine_iters=200, device="cpu"):
    scene = detect_scene(img_bgr, model, scale_bar_m, score_thr, manual_px_per_m, device)
    prob = make_problem(scene, flows, clearance)
    res = optimize(prob, scene.layout, n_seeds=n_seeds, iters=iters, refine_iters=refine_iters)
    return scene, prob, res
