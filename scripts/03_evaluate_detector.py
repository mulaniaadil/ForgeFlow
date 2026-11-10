"""Evaluate the trained detector on val / test / ood splits (+ calibration accuracy) -> outputs/detector_eval.json"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # relative paths (data/, outputs/) resolve from the project root
from factory_ai import config as C  # noqa: E402
from factory_ai.calibration import calibrate  # noqa: E402
from factory_ai.detector import PlanDataset, evaluate, load_model, predict  # noqa: E402
from factory_ai.visualization import draw_detections  # noqa: E402

data, ckpt = "data", "outputs/detector.pt"
Path("outputs").mkdir(parents=True, exist_ok=True)
model = load_model(ckpt)
res = {}
for sp in ["val", "test", "ood"]:
    ev = evaluate(model, PlanDataset(data, sp), score_thr=0.5)
    res[sp] = ev
    print(sp, {k: round(v, 3) for k, v in ev.items() if isinstance(v, float)})
# calibration accuracy
cal = {}
for sp in ["test", "ood"]:
    meta = json.loads(Path(f"{data}/{sp}_meta.json").read_text())
    e, ew, eh = [], [], []
    for fn, m in meta.items():
        c = calibrate(cv2.imread(f"{data}/images/{fn}"))
        if c.get("px_per_m") is not None and c.get("floor_w_m") is not None:
            e.append(abs(c["px_per_m"] - m["scale_px_per_m"]) / m["scale_px_per_m"] * 100)
            ew.append(abs(c["floor_w_m"] - m["floor_w"])); eh.append(abs(c["floor_h_m"] - m["floor_h"]))
    cal[sp] = dict(scale_err_pct_mean=float(np.mean(e)) if e else 0.0, scale_err_pct_max=float(np.max(e)) if e else 0.0,
                   floor_w_err_m_mean=float(np.mean(ew)) if ew else 0.0, floor_h_err_m_mean=float(np.mean(eh)) if eh else 0.0, n=len(meta))
res["calibration"] = cal
Path("outputs/detector_eval.json").write_text(json.dumps(res, indent=1))
# qualitative figure: 2 test + 2 ood images
rows = []
for sp in ["test", "ood"]:
    meta = json.loads(Path(f"{data}/{sp}_meta.json").read_text())
    for fn in list(meta)[:2]:
        img = cv2.imread(f"{data}/images/{fn}")
        pr = predict(model, [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)], score_thr=0.5)[0]
        dets = [dict(cls=C.ID_TO_CLASS[int(l)], score=float(s), box=b.tolist()) for b, l, s in zip(pr["boxes"], pr["labels"], pr["scores"])]
        rows.append(cv2.resize(draw_detections(img, dets), None, fx=2, fy=2))
cv2.imwrite("outputs/detections_examples.png", np.vstack([np.hstack(rows[:2]), np.hstack(rows[2:])]))
