"""Generate the synthetic dataset:  python scripts/01_generate_data.py --out data --n 700"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # relative paths (data/, outputs/) resolve from the project root
from factory_ai.synthetic_data import generate_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data")
ap.add_argument("--n", type=int, default=700, help="in-distribution images (split 70/15/15 by image)")
ap.add_argument("--n_ood", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
print(generate_dataset(a.out, a.n, a.n_ood, a.seed))
