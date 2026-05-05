"""Quantize, export to QNN, profile + on-device mAP. AGENT EDITS THIS FILE.

Pipeline:
  1. Trace student (backbone + head only — post-processing stays on host).
  2. Submit to AI Hub: compile with INT8 quantization, profile on real Galaxy S22.
  3. Submit an on-device inference job for a small val subset.
  4. Run host-side anchor decode + NMS on the on-device outputs.
  5. Compute COCO mAP — this is the true `quantized_map`.

Agent levers:
- Quantization scheme (`quantize`, `extra_options` to submit_aihub_profile)
- Calibration data (pass it to submit_aihub_profile when ready)
- Held-FP16 layers via AI Hub options
- ON_DEVICE_EVAL_IMAGES — bigger = more accurate quantized_map but costs AI Hub quota
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import prepare
import student

DEPLOY_METRICS_PATH = Path("metrics_deploy.json")
ON_DEVICE_EVAL_IMAGES = 200  # AGENT: bump up if AI Hub budget allows


# ----------------------------------------------------------------------------
# Trace
# ----------------------------------------------------------------------------

class _BackbonePlusHead(torch.nn.Module):
    """The portion of SSD that runs on-device. Returns (bbox_regression, cls_logits)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        features = self.model.backbone(image)
        head = self.model.head(list(features.values()))
        return head["bbox_regression"], head["cls_logits"]


def build_host_student() -> torch.nn.Module:
    s = student.build_student()
    payload = torch.load(student.STUDENT_CKPT, map_location="cpu", weights_only=False)
    s.load_state_dict(payload["state_dict"] if "state_dict" in payload else payload)
    return s.eval()


def trace_for_npu(host_model: torch.nn.Module, input_size: int):
    wrapped = _BackbonePlusHead(host_model).eval()
    example = torch.rand(1, 3, input_size, input_size)
    return torch.jit.trace(wrapped, example, strict=False), (1, 3, input_size, input_size)


# ----------------------------------------------------------------------------
# On-device mAP
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_on_device_map(host_model: torch.nn.Module, target_model,
                           input_size: int, max_images: int) -> tuple[float, float, dict]:
    """Run on-device inference for a val subset, decode on host, compute mAP."""
    val_ds = prepare.load_coco_val()
    n = min(max_images, len(val_ds))
    print(f"on-device eval: preparing {n} val images...", flush=True)

    # Build tensors at the on-device input resolution + remember original sizes.
    images_resized = []
    image_ids = []
    original_sizes = []  # (h, w)
    for i in range(n):
        img, _target, img_id = val_ds[i]
        _, h, w = img.shape
        original_sizes.append((h, w))
        image_ids.append(int(img_id))
        resized = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(input_size, input_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        images_resized.append(resized.numpy().astype(np.float32))

    # AI Hub expects {input_name: list[np.ndarray]}, where each ndarray is one sample
    # with shape matching the compiled input spec (1,3,H,W).
    inputs = {"image": [im[np.newaxis, ...] for im in images_resized]}

    print(f"on-device eval: submitting inference job ({n} images)...", flush=True)
    outputs = prepare.submit_aihub_inference(target_model, inputs)

    # outputs is a dict mapping output names -> list of ndarrays (one per input).
    # _BackbonePlusHead returns (bbox_regression, cls_logits) so AI Hub names them
    # 'output_0' and 'output_1' (or similar). Pull by order.
    out_names = list(outputs.keys())
    bbox_outs = outputs[out_names[0]]
    cls_outs = outputs[out_names[1]]

    # Pre-compute anchors once (input is fixed-size so anchors don't change).
    print("on-device eval: decoding...", flush=True)
    image_list_dummy = host_model.transform(
        [torch.zeros(3, input_size, input_size)], None
    )[0]
    feats_dummy = list(host_model.backbone(image_list_dummy.tensors).values())
    anchors_one = host_model.anchor_generator(image_list_dummy, feats_dummy)[0]

    results = []
    for img_id, (h, w), bbox_np, cls_np in zip(image_ids, original_sizes, bbox_outs, cls_outs):
        bbox_t = torch.from_numpy(bbox_np)
        cls_t = torch.from_numpy(cls_np)
        head_outputs = {"bbox_regression": bbox_t, "cls_logits": cls_t}
        # postprocess in resized coordinate space
        detections = host_model.postprocess_detections(
            head_outputs, [anchors_one], [(input_size, input_size)],
        )[0]
        # rescale boxes back to original image size (x first, y second)
        sx = w / input_size
        sy = h / input_size
        boxes = detections["boxes"].numpy()
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        scores = detections["scores"].numpy()
        labels = detections["labels"].numpy()
        for b, s, lbl in zip(boxes, scores, labels):
            x1, y1, x2, y2 = b.tolist()
            results.append({
                "image_id": img_id,
                "category_id": int(lbl),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(s),
            })

    if not results:
        return 0.0, 0.0, {"on_device_predictions": 0}

    import tempfile
    from pycocotools.cocoeval import COCOeval
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        pred_path = f.name
    try:
        coco_dt = val_ds.coco.loadRes(pred_path)
        ev = COCOeval(val_ds.coco, coco_dt, "bbox")
        ev.params.imgIds = image_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        return float(ev.stats[0]), float(ev.stats[1]), {
            "on_device_predictions": len(results),
            "on_device_eval_images": n,
        }
    finally:
        Path(pred_path).unlink(missing_ok=True)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    train_metrics = json.loads(student.TRAIN_METRICS_PATH.read_text())

    print("loading + tracing student...", flush=True)
    host_model = build_host_student()
    traced, input_shape = trace_for_npu(host_model, student.INPUT_SIZE)

    print("submitting compile + profile (INT8)...", flush=True)
    hub_metrics = prepare.submit_aihub_profile(
        traced, input_shape,
        quantize=True, calibration_data=None, extra_options="",
    )

    if not hub_metrics.get("qnn_export_ok"):
        prepare.print_summary({**train_metrics, **hub_metrics, "score": float("-inf"),
                               "deploy_seconds": time.time() - t0})
        print("constraint_status:  qnn_export_failed")
        return

    target_model = hub_metrics["target_model"]
    print("running on-device mAP eval...", flush=True)
    q_map, q_map50, eval_meta = evaluate_on_device_map(
        host_model, target_model, student.INPUT_SIZE, ON_DEVICE_EVAL_IMAGES,
    )

    metrics = {
        **train_metrics,
        "quantized_map": q_map,
        "quantized_map50": q_map50,
        "latency_ms_p50": hub_metrics.get("latency_ms_p50"),
        "latency_ms_p99": hub_metrics.get("latency_ms_p99"),
        "size_mb": hub_metrics.get("size_mb"),
        "peak_mem_mb": hub_metrics.get("peak_mem_mb"),
        "qnn_export_ok": hub_metrics.get("qnn_export_ok", False),
        "cpu_fallback_ops": hub_metrics.get("cpu_fallback_ops", 0),
        "deploy_seconds": time.time() - t0,
        **eval_meta,
    }
    score, reason = prepare.check_constraints(metrics)
    metrics["score"] = score
    metrics["constraint_status"] = reason

    DEPLOY_METRICS_PATH.write_text(json.dumps({
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float, str, bool, list, dict, type(None)))
    }, default=str))

    prepare.print_summary(metrics)
    print(f"constraint_status:  {reason}")
    print(f"on_device_eval:     {eval_meta}")
    if "profile_url" in hub_metrics:
        print(f"profile_url:        {hub_metrics['profile_url']}")


if __name__ == "__main__":
    main()
