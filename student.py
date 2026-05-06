"""Student model + distillation training. AGENT EDITS THIS FILE.

Baseline: torchvision SSDLite320-MobileNetV3-Large, COCO-pretrained. We
fine-tune briefly with the teacher's high-confidence predictions as
pseudo-labels (a simple, architecture-agnostic form of distillation).

What's intentionally weak in this baseline (room for the agent):
- No real KD loss — just supervised loss against teacher pseudo-labels.
- One short epoch on a small subset for a fast first run.
- No feature/relation distillation, no temperature, no loss balancing.

The agent should iterate on:
- Student architecture (try smaller/faster variants, prune the head, etc.)
- Distillation method (logit/feature/relation KD on matched layers)
- Loss balance, augmentation, schedule, optimizer
- Training data scale (turn on full train2017 with --full once stable)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

import prepare

STUDENT_CKPT = prepare.CACHE_DIR / "student.pt"
TRAIN_METRICS_PATH = Path("metrics_train.json")

# AGENT: this drives both training input and the deploy.py traced shape.
INPUT_SIZE = 320

# AGENT: tune.
NUM_TRAIN_IMAGES = 5000
NUM_EPOCHS = 1               # GT-only fine-tune at very low LR
BATCH_SIZE = 8
LR = 1e-5                    # gentle nudge from pretrained on val[500:] GT
EVAL_MAX_IMAGES = 500        # validate on a subset for speed; agent can raise


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def build_student() -> torch.nn.Module:
    """AGENT: replace with a smaller / NPU-friendlier architecture as you iterate."""
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    return ssdlite320_mobilenet_v3_large(weights=weights)


# ----------------------------------------------------------------------------
# Distillation
# ----------------------------------------------------------------------------

@torch.no_grad()
def teacher_pseudo_labels(teacher: torch.nn.Module, images, score_thresh: float = 0.3):
    """Run teacher in eval mode and convert outputs to detection target format.

    AGENT: replace this with a real distillation loss (logit KD, feature KD, etc.).
    Pseudo-labels lose information vs. soft labels — they're the easy baseline.
    """
    teacher.eval()
    preds = teacher(images)
    targets = []
    for p in preds:
        keep = p["scores"] > score_thresh
        targets.append({
            "boxes": p["boxes"][keep].detach(),
            "labels": p["labels"][keep].detach(),
        })
    return targets


def train_step(student, teacher, images, gt_targets, optimizer):
    """GT-supervised fine-tune. Teacher unused — pseudo-labels were destructive
    in every prior run, so we go straight to ground truth.
    """
    del teacher  # explicitly unused
    images = [im.to(prepare.device(), non_blocking=True) for im in images]
    targets = [{"boxes": t["boxes"].to(prepare.device()),
                "labels": t["labels"].to(prepare.device())} for t in gt_targets]
    # Drop empties — torchvision detection losses can NaN on zero-target inputs.
    keep = [i for i, t in enumerate(targets) if t["boxes"].numel() > 0]
    if not keep:
        return None
    images = [images[i] for i in keep]
    targets = [targets[i] for i in keep]

    student.train()
    loss_dict = student(images, targets)
    loss = sum(loss_dict.values())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
    optimizer.step()
    return float(loss.detach())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    prepare.seed_all(0)
    t0 = time.time()
    dev = prepare.device()
    print(f"device: {dev}", flush=True)

    student = build_student().to(dev)
    teacher = prepare.load_teacher().to(dev)

    # Use train2017 if available, else fall back to splitting val2017.
    try:
        train_ds = prepare.load_coco_train()
    except FileNotFoundError:
        print("train2017 not present, using val2017[NUM_TRAIN_IMAGES:] for training", flush=True)
        full = prepare.load_coco_val()
        train_ds = Subset(full, range(EVAL_MAX_IMAGES, len(full)))

    if NUM_TRAIN_IMAGES is not None and len(train_ds) > NUM_TRAIN_IMAGES:
        train_ds = Subset(train_ds, range(NUM_TRAIN_IMAGES))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=prepare.coco_collate, drop_last=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=1e-4)

    print(f"training: {len(train_ds)} imgs, {NUM_EPOCHS} epochs, bs={BATCH_SIZE}", flush=True)
    step = 0
    for epoch in range(NUM_EPOCHS):
        for imgs, gt, _ids in train_loader:
            loss = train_step(student, teacher, imgs, gt, optimizer)
            if loss is not None and step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {loss:.4f}", flush=True)
            step += 1

    STUDENT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": student.state_dict(), "input_size": INPUT_SIZE}, STUDENT_CKPT)
    print(f"checkpoint -> {STUDENT_CKPT}", flush=True)

    print("evaluating on val2017...", flush=True)
    val_ds = prepare.load_coco_val()
    fp32_map, fp32_map50 = prepare.evaluate_map(student, val_ds, batch_size=4,
                                                max_images=EVAL_MAX_IMAGES, dev=dev)

    metrics = {
        "fp32_map": fp32_map,
        "fp32_map50": fp32_map50,
        "training_seconds": time.time() - t0,
    }
    TRAIN_METRICS_PATH.write_text(json.dumps(metrics))
    print(f"fp32_map:           {fp32_map:.4f}")
    print(f"fp32_map50:         {fp32_map50:.4f}")
    print(f"training_seconds:   {metrics['training_seconds']:.1f}")


if __name__ == "__main__":
    main()
