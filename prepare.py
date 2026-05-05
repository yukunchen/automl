"""Frozen utilities for the autoresearch-cv detection loop. DO NOT MODIFY.

Contract (program.md):
- COCO data loading
- Teacher model loading (RetinaNet ResNet50 FPN v2)
- mAP evaluator (pycocotools)
- AI Hub submit + parse helper
- Hard-constraint checker
- Standard summary printer

Usage:
    python prepare.py --download         # ~1.2GB: val2017 + annotations
    python prepare.py --download --full  # ~19GB: also train2017
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

CACHE_DIR = Path(os.environ.get("AUTORESEARCH_CV_CACHE", "~/.cache/autoresearch-cv")).expanduser()
COCO_DIR = CACHE_DIR / "coco"
TEACHER_PATH = CACHE_DIR / "teacher.pt"

# Hard constraints (program.md). Out of scope for the agent to change.
HARD_CONSTRAINTS = {
    "latency_ms": 30.0,
    "size_mb": 10.0,
    "peak_mem_mb": 200.0,
}

# AI Hub target — Snapdragon 8 Gen 1 (SM8450).
AIHUB_DEVICE_NAME = "Samsung Galaxy S22 5G"
AIHUB_TARGET_RUNTIME = "qnn_context_binary"


# ----------------------------------------------------------------------------
# Device + seeding
# ----------------------------------------------------------------------------

def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed: int = 0) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# COCO download
# ----------------------------------------------------------------------------

_COCO_URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def _unzip(zip_path: Path, dest_dir: Path, marker_subdir: str) -> None:
    if (dest_dir / marker_subdir).exists():
        return
    print(f"unzipping {zip_path} -> {dest_dir}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)


def download_coco(splits: Iterable[str] = ("val2017", "annotations")) -> None:
    """Idempotent."""
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    zips_dir = COCO_DIR / "_zips"
    zips_dir.mkdir(exist_ok=True)
    for split in splits:
        url = _COCO_URLS[split]
        _download(url, zips_dir / Path(url).name)
    for split in splits:
        zpath = zips_dir / Path(_COCO_URLS[split]).name
        marker = "annotations" if split == "annotations" else split
        _unzip(zpath, COCO_DIR, marker)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class CocoDetectionTV(Dataset):
    """torchvision CocoDetection wrapper that emits {boxes,labels} dicts in
    xyxy format at original image scale, ready for torchvision detection models.

    Returns: (image_tensor[3,H,W] in [0,1], target_dict, image_id)
    """

    def __init__(self, img_dir: Path, ann_file: Path):
        from torchvision.datasets import CocoDetection
        self._inner = CocoDetection(str(img_dir), str(ann_file))
        self.coco = self._inner.coco
        self.image_ids = list(self._inner.ids)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int):
        img, anns = self._inner[idx]
        image_id = self._inner.ids[idx]
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels_t = (torch.tensor(labels, dtype=torch.int64) if labels
                    else torch.zeros((0,), dtype=torch.int64))
        img_t = torchvision.transforms.functional.to_tensor(img)
        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([image_id])}
        return img_t, target, image_id


def coco_collate(batch):
    imgs, targets, ids = zip(*batch)
    return list(imgs), list(targets), list(ids)


def load_coco_val() -> CocoDetectionTV:
    return CocoDetectionTV(COCO_DIR / "val2017", COCO_DIR / "annotations/instances_val2017.json")


def load_coco_train() -> CocoDetectionTV:
    train_dir = COCO_DIR / "train2017"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"{train_dir} not found. Run `python prepare.py --download --full` first."
        )
    return CocoDetectionTV(train_dir, COCO_DIR / "annotations/instances_train2017.json")


# ----------------------------------------------------------------------------
# Teacher
# ----------------------------------------------------------------------------

def load_teacher() -> torch.nn.Module:
    """RetinaNet ResNet50 FPN v2, COCO-pretrained (~41.5 mAP). Eval, frozen."""
    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    model = retinanet_resnet50_fpn_v2(weights=weights, score_thresh=0.05).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ----------------------------------------------------------------------------
# mAP evaluator
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_map(model: torch.nn.Module, dataset: CocoDetectionTV,
                 batch_size: int = 4, max_images: int | None = None,
                 dev: torch.device | None = None) -> tuple[float, float]:
    """COCO mAP via pycocotools. Returns (mAP@[.5:.95], mAP@.5)."""
    from pycocotools.cocoeval import COCOeval
    dev = dev or device()
    model = model.to(dev).eval()
    indices = list(range(len(dataset)))
    if max_images is not None:
        indices = indices[:max_images]
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=coco_collate)

    results = []
    seen_image_ids = []
    for imgs, _targets, ids in loader:
        imgs_dev = [im.to(dev) for im in imgs]
        preds = model(imgs_dev)
        for img_id, p in zip(ids, preds):
            seen_image_ids.append(int(img_id))
            boxes = p["boxes"].cpu().numpy()
            scores = p["scores"].cpu().numpy()
            labels = p["labels"].cpu().numpy()
            for b, s, lbl in zip(boxes, scores, labels):
                x1, y1, x2, y2 = b.tolist()
                results.append({
                    "image_id": int(img_id),
                    "category_id": int(lbl),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(s),
                })

    if not results:
        return 0.0, 0.0
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        pred_path = f.name
    try:
        coco_dt = dataset.coco.loadRes(pred_path)
        ev = COCOeval(dataset.coco, coco_dt, "bbox")
        ev.params.imgIds = seen_image_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        return float(ev.stats[0]), float(ev.stats[1])
    finally:
        os.unlink(pred_path)


# ----------------------------------------------------------------------------
# AI Hub helper
# ----------------------------------------------------------------------------

def submit_aihub_profile(traced_model, input_shape: tuple[int, ...],
                          quantize: bool = False, calibration_data=None,
                          device_name: str = AIHUB_DEVICE_NAME,
                          target_runtime: str = AIHUB_TARGET_RUNTIME,
                          extra_options: str = "") -> dict:
    """Compile + profile on a real Snapdragon device. Returns metrics dict.

    Keys: latency_ms_p50, latency_ms_p99, peak_mem_mb, size_mb,
          npu_layers, cpu_fallback_ops, gpu_layers, qnn_export_ok,
          compile_url, profile_url, target_model.
    """
    import qai_hub as hub

    options = f"--target_runtime {target_runtime}"
    if quantize:
        options += " --quantize_full_type w8a8"
    if extra_options:
        options += " " + extra_options

    dev = hub.Device(device_name)
    metrics: dict = {"qnn_export_ok": False}

    compile_kwargs = dict(model=traced_model, device=dev,
                          input_specs={"image": input_shape}, options=options)
    if quantize and calibration_data is not None:
        compile_kwargs["calibration_data"] = calibration_data

    cjob = hub.submit_compile_job(**compile_kwargs)
    metrics["compile_url"] = cjob.url
    target_model = cjob.get_target_model()
    if target_model is None:
        return metrics
    metrics["qnn_export_ok"] = True

    pjob = hub.submit_profile_job(model=target_model, device=dev)
    metrics["profile_url"] = pjob.url
    profile = pjob.download_profile()
    if profile is None:
        return metrics

    summary = profile.get("execution_summary", {})
    detail = profile.get("execution_detail", [])

    metrics["latency_ms_p50"] = summary.get("estimated_inference_time", 0) / 1000.0
    metrics["latency_ms_p99"] = metrics["latency_ms_p50"]  # AI Hub returns single estimate
    peak = summary.get("inference_memory_peak_range", [0, 0])
    metrics["peak_mem_mb"] = peak[1] / (1024 * 1024)
    metrics["size_mb"] = summary.get("compiled_model_size_bytes", 0) / 1e6
    metrics["npu_layers"] = sum(1 for l in detail if l.get("compute_unit") == "NPU")
    metrics["cpu_fallback_ops"] = sum(1 for l in detail if l.get("compute_unit") == "CPU")
    metrics["gpu_layers"] = sum(1 for l in detail if l.get("compute_unit") == "GPU")
    metrics["target_model"] = target_model
    return metrics


# ----------------------------------------------------------------------------
# Constraint checker
# ----------------------------------------------------------------------------

def check_constraints(metrics: dict) -> tuple[float, str]:
    """Return (score, reason). score = quantized_map if all hard constraints pass, else -inf."""
    if not metrics.get("qnn_export_ok", False):
        return float("-inf"), "qnn_export_failed"
    lat = metrics.get("latency_ms_p50", math.inf)
    if lat > HARD_CONSTRAINTS["latency_ms"]:
        return float("-inf"), f"latency {lat:.1f}ms > {HARD_CONSTRAINTS['latency_ms']}ms"
    sz = metrics.get("size_mb", math.inf)
    if sz > HARD_CONSTRAINTS["size_mb"]:
        return float("-inf"), f"size {sz:.1f}MB > {HARD_CONSTRAINTS['size_mb']}MB"
    mem = metrics.get("peak_mem_mb", math.inf)
    if mem > HARD_CONSTRAINTS["peak_mem_mb"]:
        return float("-inf"), f"peak_mem {mem:.0f}MB > {HARD_CONSTRAINTS['peak_mem_mb']}MB"
    cpu_fb = metrics.get("cpu_fallback_ops", 0)
    if cpu_fb > 0:
        return float("-inf"), f"{cpu_fb} CPU fallback ops"
    return float(metrics.get("quantized_map", 0.0)), "ok"


# ----------------------------------------------------------------------------
# Summary printer
# ----------------------------------------------------------------------------

_SUMMARY_KEYS = [
    "fp32_map", "fp32_map50",
    "quantized_map",
    "latency_ms_p50", "latency_ms_p99",
    "size_mb", "peak_mem_mb",
    "qnn_export_ok", "cpu_fallback_ops",
    "score",
    "training_seconds", "deploy_seconds",
]

def print_summary(metrics: dict) -> None:
    print("---")
    for k in _SUMMARY_KEYS:
        v = metrics.get(k, "n/a")
        if isinstance(v, bool):
            print(f"{k+':':<20}{'true' if v else 'false'}")
        elif isinstance(v, float):
            print(f"{k+':':<20}-inf" if v == float("-inf") else f"{k+':':<20}{v:.4f}")
        else:
            print(f"{k+':':<20}{v}")
    print("---")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="Download COCO val + annotations.")
    ap.add_argument("--full", action="store_true", help="Also download COCO train2017 (~18GB).")
    ap.add_argument("--cache-teacher", action="store_true", help="Force-cache teacher weights.")
    args = ap.parse_args()

    if args.download:
        splits = ["val2017", "annotations"]
        if args.full:
            splits.append("train2017")
        download_coco(splits)
        print(f"COCO ready at {COCO_DIR}")
    if args.cache_teacher:
        m = load_teacher()
        TEACHER_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(m.state_dict(), TEACHER_PATH)
        print(f"teacher cached at {TEACHER_PATH}")
    if not (args.download or args.cache_teacher):
        print("Nothing to do. Try --download.")


if __name__ == "__main__":
    main()
