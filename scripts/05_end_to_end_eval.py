"""End-to-end test: image -> detect -> calibrate -> layout -> optimise, vs ground-truth layout.
Writes outputs/end_to_end_eval.{json,csv} and outputs/end_to_end_example.png"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # relative paths (data/, outputs/) resolve from the project root
from factory_ai import config as C  # noqa: E402
from factory_ai.detector import load_model  # noqa: E402
from factory_ai.layout_optimizer import Problem, cost_path, optimize, violations  # noqa: E402
from factory_ai.pipeline import detect_scene, make_problem  # noqa: E402
from factory_ai.process_flow import build_flows  # noqa: E402
from factory_ai.visualization import draw_detections, side_by_side  # noqa: E402

Path("outputs").mkdir(parents=True, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
model = load_model("outputs/detector.pt")
rows, example = [], None
for sp in ["test", "ood"]:
    meta = json.loads(Path(f"data/{sp}_meta.json").read_text())
    for fn in list(meta)[:N]:
        m = meta[fn]
        img = cv2.imread(f"data/images/{fn}")
        row = dict(split=sp, image=fn, ok=False)
        try:
            sc = detect_scene(img, model)
        except Exception as e:  # calibration failure
            row["error"] = str(e); rows.append(row); continue
        complete = len(sc.layout) == 6 and len(sc.fixed) == 2
        row.update(complete=complete, n_cols_true=len(m["obstacles"]), n_cols_det=len(sc.obstacles))
        if not complete:
            rows.append(row); continue
        gt = {n: tuple(r) for n, r in m["layout"].items()}
        ce = [np.hypot(*(np.array([sc.layout[n][0] + sc.layout[n][2] / 2, sc.layout[n][1] + sc.layout[n][3] / 2])
                         - np.array([gt[n][0] + gt[n][2] / 2, gt[n][1] + gt[n][3] / 2]))) for n in gt]
        row["centre_err_m"] = float(np.mean(ce))
        row["size_err_m"] = float(np.mean([abs(sc.layout[n][2] - gt[n][2]) + abs(sc.layout[n][3] - gt[n][3]) for n in gt]) / 2)
        Pgt = Problem(m["floor_w"], m["floor_h"], {n: tuple(v) for n, v in m["machines"].items()}, build_flows(),
                      {k: tuple(v) for k, v in m["fixed"].items()}, [tuple(o) for o in m["obstacles"]])
        Pdt = make_problem(sc)
        c_gt = cost_path(Pgt, gt); c_det = cost_path(Pdt, sc.layout)   # initial cost, true vs estimated
        row["initial_cost_true"], row["initial_cost_est"] = c_gt, c_det
        row["initial_cost_err_%"] = 100 * abs(c_det - c_gt) / c_gt
        r_gt = optimize(Pgt, gt, n_seeds=3, iters=15000, refine_iters=60)
        r_dt = optimize(Pdt, sc.layout, n_seeds=3, iters=15000, refine_iters=60)
        # evaluate the layout optimised from DETECTIONS on the TRUE problem (is it good in reality?)
        feas_true = not violations(Pgt, r_dt["final_layout"])
        row.update(opt_cost_gt_pipeline=r_gt["final_cost_path"], opt_cost_det_pipeline_on_true=cost_path(Pgt, r_dt["final_layout"]),
                   opt_feasible_on_true=feas_true, reduction_gt_pct=100 * (1 - r_gt["final_cost_path"] / c_gt),
                   reduction_det_pct=100 * (1 - cost_path(Pgt, r_dt["final_layout"]) / c_gt), ok=True)
        rows.append(row)
        if example is None and sp == "test":
            example = (img, sc, Pdt, r_dt)
df = pd.DataFrame(rows); df.to_csv("outputs/end_to_end_eval.csv", index=False)
summ = {}
for sp, g in df.groupby("split"):
    ok = g[g.ok == True]
    summ[sp] = dict(n=len(g), complete_detection=float(g.get("complete", pd.Series(dtype=bool)).fillna(False).mean()),
                    centre_err_m=float(ok.centre_err_m.mean()) if len(ok) else None,
                    initial_cost_err_pct=float(ok["initial_cost_err_%"].mean()) if len(ok) else None,
                    reduction_true_layout_pct=float(ok.reduction_gt_pct.mean()) if len(ok) else None,
                    reduction_from_detections_pct=float(ok.reduction_det_pct.mean()) if len(ok) else None,
                    feasible_on_true_frac=float(ok.opt_feasible_on_true.mean()) if len(ok) else None)
Path("outputs/end_to_end_eval.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))
if example:
    img, sc, P, r = example
    cv2.imwrite("outputs/end_to_end_detections.png", cv2.resize(draw_detections(img, sc.detections), None, fx=2, fy=2))
    side_by_side(P, sc.layout, r["final_layout"], r["initial_cost_path"], r["final_cost_path"], save="outputs/end_to_end_example.png")
