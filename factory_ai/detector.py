"""PyTorch object detector (TorchVision Faster R-CNN MobileNetV3-FPN): dataset, model, train, eval."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                          fasterrcnn_mobilenet_v3_large_320_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from . import config as C


# ----------------------------------------------------------------------------- data
class PlanDataset(torch.utils.data.Dataset):
    """COCO-style json + image folder.  `train=True` enables light photometric/flip augmentation."""

    def __init__(self, root, split, train=False):
        self.root = Path(root)
        d = json.loads((self.root / f"{split}.json").read_text())
        self.images = d["images"]
        self.by_img = {}
        for a in d["annotations"]:
            self.by_img.setdefault(a["image_id"], []).append(a)
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        info = self.images[i]
        img = cv2.cvtColor(cv2.imread(str(self.root / "images" / info["file_name"])), cv2.COLOR_BGR2RGB)
        boxes, labels = [], []
        for a in self.by_img.get(info["id"], []):
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        boxes = np.array(boxes, np.float32).reshape(-1, 4)
        labels = np.array(labels, np.int64)
        if self.train:
            img, boxes = _augment(img, boxes)
        t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return t, dict(boxes=torch.from_numpy(boxes), labels=torch.from_numpy(labels), image_id=info["id"])


def _augment(img, boxes):
    h, w = img.shape[:2]
    if random.random() < 0.5:                       # horizontal flip
        img = img[:, ::-1].copy()
        boxes = boxes.copy()
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
    if random.random() < 0.5:                       # vertical flip
        img = img[::-1].copy()
        boxes = boxes.copy()
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
    f = np.float32
    img = img.astype(f)
    img = img * random.uniform(0.8, 1.2) + random.uniform(-15, 15)          # brightness / contrast
    if random.random() < 0.3:                                               # channel swap (colour-cue robustness)
        img = img[:, :, np.random.permutation(3)]
    if random.random() < 0.3:
        img = img + np.random.normal(0, random.uniform(2, 8), img.shape)    # noise
    if random.random() < 0.15:
        img = 255 - img                                                      # inversion (blueprint-like)
    return np.clip(img, 0, 255).astype(np.uint8), boxes


def collate(b):
    return tuple(zip(*b))


# ----------------------------------------------------------------------------- model
def build_model(pretrained=False):
    """pretrained=True loads COCO weights (needs internet; use in Colab) then replaces the head."""
    kw = dict(min_size=C.IMG_H, max_size=C.IMG_W, box_detections_per_img=40,
              rpn_batch_size_per_image=128, box_batch_size_per_image=128)
    if pretrained:
        m = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT, **kw)
        inf = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(inf, C.NUM_DET_CLASSES)
        return m
    return fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None,
                                                 num_classes=C.NUM_DET_CLASSES, **kw)


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, imgs_rgb, device="cpu", score_thr=0.05):
    """imgs_rgb: list of HxWx3 uint8 RGB arrays -> list of dicts(boxes, labels, scores)."""
    if not imgs_rgb:
        return []
    model.eval()
    ts = [torch.from_numpy(i.astype(np.float32) / 255.0).permute(2, 0, 1).to(device) for i in imgs_rgb]
    out = model(ts)
    res = []
    for o in out:
        k = o["scores"] >= score_thr
        res.append(dict(boxes=o["boxes"][k].cpu().numpy(), labels=o["labels"][k].cpu().numpy(),
                        scores=o["scores"][k].cpu().numpy()))
    return res


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(pretrained=False)
    m.load_state_dict(ck["model"])
    m.to(device).eval()
    return m


# ----------------------------------------------------------------------------- evaluation
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def evaluate(model, dataset, device="cpu", score_thr=0.5, iou_thr=0.5, batch=16):
    """COCO mAP (torchmetrics) + per-class precision / recall / F1 at (score_thr, iou_thr)."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True, backend="pycocotools")
    tp = {c: 0 for c in C.CLASSES}; fp = {c: 0 for c in C.CLASSES}; fn = {c: 0 for c in C.CLASSES}
    for s in range(0, len(dataset), batch):
        items = [dataset[i] for i in range(s, min(s + batch, len(dataset)))]
        imgs = [(t.permute(1, 2, 0).numpy() * 255).astype(np.uint8) for t, _ in items]
        preds = predict(model, imgs, device, score_thr=0.01)
        metric.update([dict(boxes=torch.tensor(p["boxes"]).reshape(-1, 4), scores=torch.tensor(p["scores"]),
                            labels=torch.tensor(p["labels"], dtype=torch.int64)) for p in preds],
                      [dict(boxes=t["boxes"], labels=t["labels"]) for _, t in items])
        for p, (_, t) in zip(preds, items):
            k = p["scores"] >= score_thr
            pb, pl, ps = p["boxes"][k], p["labels"][k], p["scores"][k]
            for cid, cname in C.ID_TO_CLASS.items():
                g = t["boxes"].numpy()[t["labels"].numpy() == cid]
                d = pb[pl == cid]; sc = ps[pl == cid]
                d = d[np.argsort(-sc)]
                ious = iou_matrix(d, g)
                used = set()
                for i in range(len(d)):
                    j = int(np.argmax(ious[i])) if len(g) else -1
                    if j >= 0 and ious[i, j] >= iou_thr and j not in used:
                        used.add(j); tp[cname] += 1
                    else:
                        fp[cname] += 1
                fn[cname] += len(g) - len(used)
    r = metric.compute()
    per = {}
    for cname in C.CLASSES:
        P = tp[cname] / max(tp[cname] + fp[cname], 1); R = tp[cname] / max(tp[cname] + fn[cname], 1)
        per[cname] = dict(precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-9), tp=tp[cname], fp=fp[cname], fn=fn[cname])
    T = sum(tp.values()); Fp = sum(fp.values()); Fn = sum(fn.values())
    P = T / max(T + Fp, 1); R = T / max(T + Fn, 1)
    cm = r.get("map_per_class")
    ids = r.get("classes")
    ap_cls = {}
    if cm is not None and ids is not None and cm.ndim > 0:
        for cid, v in zip(ids.tolist(), cm.tolist()):
            if int(cid) in C.ID_TO_CLASS:
                ap_cls[C.ID_TO_CLASS[int(cid)]] = v
    return dict(mAP=float(r["map"]), mAP50=float(r["map_50"]), mAP75=float(r["map_75"]),
                precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-9),
                per_class=per, ap_per_class=ap_cls, score_thr=score_thr, iou_thr=iou_thr, n_images=len(dataset))


# ----------------------------------------------------------------------------- training
def train(data_root, out_path, epochs=15, batch=8, lr=1e-3, pretrained=False, device=None, seed=0,
         eval_every=3, log=print, max_minutes=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if device == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))
    tr = PlanDataset(data_root, "train", train=True)
    va = PlanDataset(data_root, "val", train=False)
    dl = torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, collate_fn=collate, num_workers=0)
    model = build_model(pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr * (0.3 if pretrained else 1.0), weight_decay=1e-4)
    total = max(1, epochs * len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=total, pct_start=0.1)
    # Write checkpoints to a side file; promote to out_path only when training finishes, so an
    # interrupted run (e.g. out-of-memory) never replaces a good deployed model.
    out_path = Path(out_path)
    tmp_path = out_path.with_name(out_path.stem + ".partial" + out_path.suffix)
    hist = []
    best, t0 = -1.0, time.time()
    ep = 0
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for imgs, tg in dl:
            imgs = [i.to(device) for i in imgs]
            tg = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items() if k != "image_id"} for t in tg]
            loss = sum(model(imgs, tg).values())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step(); sched.step()
            run += loss.item()
        rec = dict(epoch=ep, train_loss=run / len(dl), minutes=(time.time() - t0) / 60)
        if ep % eval_every == 0 or ep == epochs:
            ev = evaluate(model, va, device)
            rec.update(val_mAP=ev["mAP"], val_mAP50=ev["mAP50"], val_precision=ev["precision"], val_recall=ev["recall"])
            if ev["mAP50"] > best:                  # model selection on VALIDATION only
                best = ev["mAP50"]
                torch.save(dict(model=model.state_dict(), epoch=ep, val_mAP50=best), tmp_path)
        hist.append(rec)
        log(json.dumps(rec))
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            log("time budget reached, stopping early")
            break
    if not tmp_path.exists():
        torch.save(dict(model=model.state_dict(), epoch=ep, val_mAP50=best), tmp_path)
    tmp_path.replace(out_path)
    return hist
