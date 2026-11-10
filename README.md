# Factory Layout Studio
**Using computer vision to design a factory with optimal placement of equipment** — Work System Design course project.

Turn a floor-plan image into an editable digital layout, then find the equipment arrangement that minimises
material-handling distance while keeping aisle clearances.

```
floor-plan image → PyTorch detector (machines, docks, columns) → OpenCV scale-bar calibration (px → m)
→ editable layout → simulated-annealing optimiser (clearance, obstacles, rotation, shortest aisle path)
→ improved layout + relocation plan
```

Stack: Python, PyTorch/TorchVision (Faster R-CNN MobileNetV3-FPN), OpenCV, NumPy, pandas, Matplotlib, Streamlit.

## Key results
| | |
|---|---|
| Detector, held-out test plans | mAP@0.5 **0.950**, precision 0.904, recall 0.916 |
| Detector, stress-test plans (different look) | mAP@0.5 0.319 — main limitation |
| Scale calibration error | 0.67 % mean |
| Optimiser, 30 × 20 m reference plant | 14,155 → 6,453 load-m/shift (**−54.4 %**), all constraints met |
| End to end, 40 test plans | all objects found in 67.5 %; 65.9 % saving from detected vs 69.0 % from true layouts |

Source: `outputs/*.json` / `*.csv`. Full write-up: `presentation/Factory_Layout_Studio_Final_Report.docx`.

## Quick start
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
A trained model (`outputs/detector.pt`) and sample plans (`data/images/`) are included, so the app works immediately.
Without the model the app still runs; only the image-import page is disabled.

## Using the app
| Page | What you do |
|---|---|
| **Dashboard** | Overview: floor area, equipment, constraint check, latest saving. |
| **Import floor plan** | Upload a plan or pick a sample; review detections; click **Use this layout**. Scale and detection-confidence settings are on this page. |
| **Layout editor** | Edit every object in metres (x, y = lower-left corner), add missed equipment, delete false detections, lock machines in place, set floor size. **Save changes** to apply. |
| **Material flow** | Set product volumes (unit loads per shift); view the from-to matrix and busiest links. |
| **Optimize & export** | Choose clearance, position tolerance and search effort (Quick / Balanced / Thorough); compare before/after; download the relocation plan (CSV), drawing (PNG) or project (JSON). |

Results are marked out of date whenever the layout or volumes change.

## Reproduce the experiments
```bash
python scripts/01_generate_data.py               # 800 synthetic plans + COCO labels (split by image)
python scripts/02_train_detector.py --epochs 16  # CPU ~30 min; GPU/Colab: add --pretrained
python scripts/03_evaluate_detector.py           # mAP / precision / recall, calibration error
python scripts/04_optimizer_experiments.py       # optimiser tables + figures
python scripts/05_end_to_end_eval.py 40          # image → layout → optimisation vs ground truth
python -m pytest -q tests                        # optimiser smoke tests
```
- Scripts can be run from any directory; paths resolve from the project root.
- Training writes to `outputs/detector.partial.pt` and only replaces `outputs/detector.pt` when it finishes,
  so an interrupted run never breaks the app.
- On 8 GB RAM machines use `--batch 4`, close other heavy apps, and make sure the drive has a few GB free.
- Colab: upload the project zip and open `colab_run_all.ipynb`.

## Project structure
| Path | Contents |
|---|---|
| `app.py` | Streamlit application |
| `factory_ai/` | Library: `config`, `synthetic_data`, `detector`, `calibration`, `process_flow`, `layout_optimizer`, `pipeline`, `visualization` |
| `scripts/` | Numbered experiment scripts (01–05) |
| `tests/` | Optimiser smoke tests |
| `data/` | Generated plan images and COCO annotations (train / val / test / ood) |
| `outputs/` | Model checkpoint, metrics (JSON/CSV) and figures |
| `presentation/` | Final report, presentation slides (script in speaker notes), presenter guide with Q&A |
| `assets/`, `.streamlit/` | App logo and theme |

## Limitations
- The detector is trained and tested on **synthetic** plans and does not generalise to visually different plans
  (stress-test mAP@0.5 0.32); it has not been tested on real factory drawings.
- One machine per type; flow volumes are an assumed scenario.
- Clearance is a geometric gap, not a safety-code check (no fire exits, aisle widths or forklift turning radii).
- Simulated annealing is a heuristic: good solutions, no proof of global optimality.
