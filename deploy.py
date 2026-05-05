"""Quantize, export to QNN, profile on real Snapdragon device. AGENT EDITS.

Baseline: take the student checkpoint from student.py, trace to TorchScript,
submit to AI Hub with PTQ INT8 (w8a8), profile on Samsung Galaxy S22 (SD8 Gen 1),
and report the program.md-mandated summary.

Known shortcuts in this baseline (room for the agent):
- quantized_map is approximated as fp32_map. Replace with on-device inference
  (hub.submit_inference_job) + offline COCO eval to get the real number.
- No calibration data passed — uses the AI Hub default calibration. Pass a
  representative calibration_data dict for tighter quantization.
- No mixed precision — entire model is INT8. Some layers (regression head)
  often benefit from being held at FP16; try `extra_options="--quantize_io"`
  variants and per-layer overrides.
- No graph-fusion tweaks. Try AI Hub's compile flags for further latency.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

import prepare
import student

DEPLOY_METRICS_PATH = Path("metrics_deploy.json")


def trace_student(ckpt_path: Path, input_size: int):
    """Load student weights and trace the *forward path used at inference*.

    torchvision detection models return list-of-dicts in eval mode and
    don't trace cleanly out of the box. We wrap to expose a tensor-in,
    tensors-out function suitable for QNN.
    """
    s = student.build_student()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    s.load_state_dict(payload["state_dict"] if "state_dict" in payload else payload)
    s.eval()

    # AGENT: this wrapper is the part of the model that actually runs on-device.
    # The post-processing (anchor decoding, NMS) often lives outside QNN. Keep
    # this lean — only the dense compute (backbone + head) needs to be on NPU.
    class BackbonePlusHead(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image):
            # SSD's transform expects a list; we approximate with a single-image batch.
            features = self.model.backbone(image)
            head_outputs = self.model.head(list(features.values()))
            # head_outputs: dict with 'bbox_regression' and 'cls_logits'
            return head_outputs["bbox_regression"], head_outputs["cls_logits"]

    wrapped = BackbonePlusHead(s).eval()
    example = torch.rand(1, 3, input_size, input_size)
    traced = torch.jit.trace(wrapped, example, strict=False)
    return traced, (1, 3, input_size, input_size)


def main() -> None:
    t0 = time.time()

    train_metrics = json.loads(student.TRAIN_METRICS_PATH.read_text())
    fp32_map = train_metrics["fp32_map"]
    fp32_map50 = train_metrics["fp32_map50"]

    print("tracing student...", flush=True)
    traced, input_shape = trace_student(student.STUDENT_CKPT, student.INPUT_SIZE)

    print("submitting to AI Hub (compile + profile, INT8)...", flush=True)
    # AGENT: tune quantize, extra_options, calibration_data.
    hub_metrics = prepare.submit_aihub_profile(
        traced, input_shape,
        quantize=True,
        calibration_data=None,
        extra_options="",
    )

    # AGENT: TODO — replace with real on-device mAP.
    # Path:
    #   1. hub.submit_inference_job(model=hub_metrics["target_model"], inputs=...)
    #   2. download outputs, run NMS + COCO eval offline
    quantized_map = fp32_map  # proxy; honest replacement is the agent's job
    quantized_map_note = "approx=fp32 (TODO: on-device eval)"

    metrics = {
        **train_metrics,
        "quantized_map": quantized_map,
        "latency_ms_p50": hub_metrics.get("latency_ms_p50"),
        "latency_ms_p99": hub_metrics.get("latency_ms_p99"),
        "size_mb": hub_metrics.get("size_mb"),
        "peak_mem_mb": hub_metrics.get("peak_mem_mb"),
        "qnn_export_ok": hub_metrics.get("qnn_export_ok", False),
        "cpu_fallback_ops": hub_metrics.get("cpu_fallback_ops", 0),
        "deploy_seconds": time.time() - t0,
    }
    score, reason = prepare.check_constraints(metrics)
    metrics["score"] = score
    metrics["constraint_status"] = reason

    DEPLOY_METRICS_PATH.write_text(json.dumps({
        k: v for k, v in metrics.items() if not isinstance(v, object) or isinstance(v, (int, float, str, bool, list, dict))
    }, default=str))

    prepare.print_summary(metrics)
    print(f"constraint_status:  {reason}")
    print(f"quantized_map_note: {quantized_map_note}")
    if "profile_url" in hub_metrics:
        print(f"profile_url:        {hub_metrics['profile_url']}")


if __name__ == "__main__":
    main()
