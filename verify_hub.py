"""
Sanity check: submit a torchvision MobileNetV3-Small to Qualcomm AI Hub,
profile it on a Snapdragon 8 Gen 1 device, and report whether the round trip
works for our autoresearch loop.

Goal of this script:
1. Confirm AI Hub auth works.
2. Confirm SD8 Gen 1 is reachable.
3. Confirm we can: TorchScript-trace -> upload -> compile -> profile -> read latency.
4. Surface CPU-fallback ops, since those are what the autoresearch loop will gate on.
"""

import torch
import torchvision.models as tvm
import qai_hub as hub

DEVICE_NAME = "Samsung Galaxy S22 5G"  # SD8 Gen 1 (sm8450)
INPUT_SHAPE = (1, 3, 224, 224)


def main():
    print(f"qai_hub version: {hub.__version__}")

    model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT).eval()
    example = torch.rand(INPUT_SHAPE)
    traced = torch.jit.trace(model, example)
    print(f"traced model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    device = hub.Device(name=DEVICE_NAME)
    print(f"target device: {device.name}")

    print("submitting compile job (TorchScript -> QNN)...")
    compile_job = hub.submit_compile_job(
        model=traced,
        device=device,
        input_specs={"image": INPUT_SHAPE},
        options="--target_runtime qnn_context_binary",
    )
    print(f"  compile job: {compile_job.job_id}")
    target_model = compile_job.get_target_model()
    if target_model is None:
        print("  COMPILE FAILED — see job page")
        print(f"  {compile_job.url}")
        return 1
    print(f"  compiled model id: {target_model.model_id}")

    print("submitting profile job (run on real device)...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    print(f"  profile job: {profile_job.job_id}")
    profile = profile_job.download_profile()
    if profile is None:
        print("  PROFILE FAILED")
        print(f"  {profile_job.url}")
        return 1

    exec_detail = profile["execution_detail"]
    summary = profile["execution_summary"]

    inference_us = summary.get("estimated_inference_time", 0)
    peak_mem = summary.get("inference_memory_peak_range", [0, 0])
    npu_layers = sum(1 for l in exec_detail if l.get("compute_unit") == "NPU")
    cpu_layers = sum(1 for l in exec_detail if l.get("compute_unit") == "CPU")
    gpu_layers = sum(1 for l in exec_detail if l.get("compute_unit") == "GPU")

    print()
    print("---")
    print(f"device:              {DEVICE_NAME}")
    print(f"latency_ms_p50:      {inference_us / 1000:.2f}")
    print(f"peak_mem_mb:         {peak_mem[1] / (1024 * 1024):.1f}")
    print(f"size_mb:             {summary.get('compiled_model_size_bytes', 0) / 1e6:.2f}")
    print(f"npu_layers:          {npu_layers}")
    print(f"cpu_fallback_ops:    {cpu_layers}")
    print(f"gpu_layers:          {gpu_layers}")
    print(f"profile_url:         {profile_job.url}")
    print("---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
