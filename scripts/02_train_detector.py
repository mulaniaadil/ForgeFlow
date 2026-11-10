"""Train the detector:  python scripts/02_train_detector.py --epochs 16
In Colab (internet + GPU) add --pretrained to fine-tune COCO-pretrained weights."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # relative paths (data/, outputs/) resolve from the project root
from factory_ai.detector import train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="outputs/detector.pt")
ap.add_argument("--epochs", type=int, default=16)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--pretrained", action="store_true")
ap.add_argument("--max_minutes", type=float, default=None)
a = ap.parse_args()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
hist = train(a.data, a.out, a.epochs, a.batch, a.lr, a.pretrained, max_minutes=a.max_minutes)
Path(a.out).with_name("training_history.json").write_text(json.dumps(hist, indent=1))
