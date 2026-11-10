"""Optimiser experiments on the documented demo scenario -> outputs/optimizer_*.{json,csv,png}"""
import json
import random
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # relative paths (data/, outputs/) resolve from the project root
from factory_ai import config as C  # noqa: E402
from factory_ai.layout_optimizer import (Problem, cost_manhattan, cost_path, is_feasible, optimize,  # noqa: E402
                                         random_feasible_layout, refine_with_path, simulated_annealing)
from factory_ai.process_flow import build_flows, flow_table  # noqa: E402
from factory_ai.visualization import plot_layout, side_by_side  # noqa: E402

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
SEEDS, ITERS, REF = 10, 30000, 300

# Phase-1 floor + layout, extended with docks, one column, aisle clearance
init = {"Cutting": (2, 12, 4, 3), "Drilling": (9, 12, 3, 3), "Assembly": (17, 12, 5, 3),
        "Inspection": (20, 5, 3, 2.5), "Packaging": (11, 5, 4, 3), "Storage": (2, 3, 4, 4)}
flows = build_flows()
flow_table(flows).to_csv(OUT / "flow_matrix.csv", index=False)


def prob(**kw):
    d = dict(floor_w=30, floor_h=20, machines=dict(C.NOMINAL_SIZE), flows=flows,
             fixed={"Receiving": (0, 17, 1.5, 3), "Dispatch": (28.5, 1, 1.5, 3)},
             obstacles=[(14, 9, 1.5, 1.5)], clearance=1.0)
    d.update(kw)
    return Problem(**d)


P = prob()
assert is_feasible(P, init), "initial layout must be feasible"
t = time.time()
R = optimize(P, init, n_seeds=SEEDS, iters=ITERS, refine_iters=REF)
print("optimize time", round(time.time() - t, 1), "s")

# random-feasible baseline
rng = random.Random(1)
rand = [random_feasible_layout(P, rng) for _ in range(200)]
rm = np.array([cost_manhattan(P, l) for l in rand]); rp = np.array([cost_path(P, l) for l in rand[:100]])

sa_m = np.array([r["cost_manhattan"] for r in R["runs"]]); sa_p = np.array([r["cost_path"] for r in R["runs"]])
rows = [
    ("Initial (Phase-1 layout)", R["initial_cost_manhattan"], R["initial_cost_path"]),
    ("Random feasible (mean of 200 / 100)", rm.mean(), rp.mean()),
    (f"SA only, mean of {SEEDS} seeds", sa_m.mean(), sa_p.mean()),
    ("SA only, best seed (by path cost)", R["best_sa_cost_manhattan"], R["best_sa_cost_path"]),
    ("SA + path refinement (final)", R["final_cost_manhattan"], R["final_cost_path"]),
]
tab = pd.DataFrame(rows, columns=["Layout", "Manhattan_cost", "ShortestPath_cost"])
tab["Path_reduction_vs_initial_%"] = 100 * (1 - tab["ShortestPath_cost"] / R["initial_cost_path"])
tab.round(1).to_csv(OUT / "optimizer_summary.csv", index=False)
print(tab.round(1).to_string(index=False))
print("SA path cost across seeds: mean %.0f std %.0f min %.0f max %.0f" % (sa_p.mean(), sa_p.std(), sa_p.min(), sa_p.max()))
print("SA manhattan across seeds: mean %.0f std %.1f" % (sa_m.mean(), sa_m.std()))

# ablation: clearance / rotation (5 seeds, SA + no refinement, report path cost of best)
abl = []
for name, kw in [("clearance 0.5 m", dict(clearance=0.5)), ("clearance 1.0 m (default)", {}),
                 ("clearance 1.5 m", dict(clearance=1.5)), ("no rotation", dict(allow_rotation=False)),
                 ("no column obstacle", dict(obstacles=[]))]:
    Pa = prob(**kw)
    init_a = init if is_feasible(Pa, init) else None
    Ra = optimize(Pa, init_a, n_seeds=5, iters=20000, refine_iters=0)
    abl.append(dict(Setting=name, best_manhattan=Ra["best_sa_cost_manhattan"], best_path=Ra["best_sa_cost_path"],
                    initial_feasible=is_feasible(Pa, init)))
abl = pd.DataFrame(abl); abl.round(1).to_csv(OUT / "optimizer_ablation.csv", index=False)
print(abl.round(1).to_string(index=False))

# figures
side_by_side(P, init, R["final_layout"], R["initial_cost_path"], R["final_cost_path"], save=OUT / "before_after_layout.png")
fig, ax = plt.subplots(figsize=(6.5, 4))
for r in R["runs"]:
    ax.plot(np.arange(len(r["history"])) * 200, r["history"], alpha=0.7)
ax.set_xlabel("SA iteration"); ax.set_ylabel("best Manhattan cost"); ax.set_title("SA convergence (10 seeds)")
ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT / "sa_convergence.png", dpi=150)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(rm, bins=25, color="#bbb", label="random feasible layouts"); ax.axvline(R["initial_cost_manhattan"], c="r", label="initial")
ax.axvline(sa_m.mean(), c="g", label="SA (mean)"); ax.set_xlabel("Manhattan cost"); ax.legend(); ax.set_title("Optimiser vs random layouts")
fig.tight_layout(); fig.savefig(OUT / "optimizer_vs_random.png", dpi=150)

res = dict(seeds=SEEDS, iters=ITERS, refine_iters=REF, initial_manhattan=R["initial_cost_manhattan"],
           initial_path=R["initial_cost_path"], final_manhattan=R["final_cost_manhattan"], final_path=R["final_cost_path"],
           sa_manhattan_mean=float(sa_m.mean()), sa_manhattan_std=float(sa_m.std()), sa_path_mean=float(sa_p.mean()),
           sa_path_std=float(sa_p.std()), random_manhattan_mean=float(rm.mean()), random_manhattan_std=float(rm.std()),
           random_path_mean=float(rp.mean()), final_feasible=R["final_feasible"],
           final_layout={k: list(v) for k, v in R["final_layout"].items()})
(OUT / "optimizer_results.json").write_text(json.dumps(res, indent=1))
