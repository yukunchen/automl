"""Profile teacher vs student: params, layers, FLOPs, storage, latency."""
import time
import torch
import prepare
import student

DEVICE = prepare.device()


def count_layers(model):
    """Number of trainable layers (Conv2d + Linear + BatchNorm)."""
    counts = {"Conv2d": 0, "Linear": 0, "BatchNorm2d": 0, "GroupNorm": 0, "LayerNorm": 0}
    for m in model.modules():
        cls = m.__class__.__name__
        if cls in counts:
            counts[cls] += 1
    return counts


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_latency(model, x, n=20):
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(3):
            model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.time() - t0) / n
    return dt * 1000


def report(name, model, input_size, x_input):
    p = count_params(model)
    layers = count_layers(model)
    lat = measure_latency(model, x_input)
    print(f"\n=== {name} ===")
    print(f"  params:        {p / 1e6:.2f} M")
    print(f"  Conv2d:        {layers['Conv2d']}")
    print(f"  Linear:        {layers['Linear']}")
    print(f"  BatchNorm2d:   {layers['BatchNorm2d']}")
    print(f"  input size:    {input_size}")
    print(f"  latency on cuda: {lat:.2f} ms")
    return {"name": name, "params_m": p / 1e6, "layers": layers, "lat_ms": lat,
            "input_size": input_size}


def main():
    print(f"device: {DEVICE}")

    print("\nloading teacher (RetinaNet ResNet50 FPN v2)...")
    teacher = prepare.load_teacher().to(DEVICE).eval()
    teacher_input = [torch.rand(3, 800, 800, device=DEVICE)]
    teacher_stats = report("Teacher: RetinaNet R50 FPN v2", teacher, "(3, 800, 800)", teacher_input)

    print("\nloading student fp32 (SSDLite320 MobileNetV3-Large)...")
    s = student.build_student().to(DEVICE).eval()
    student_input = [torch.rand(3, 320, 320, device=DEVICE)]
    student_stats = report("Student: SSDLite320 MobileNetV3-Large", s, "(3, 320, 320)", student_input)

    # Compression ratios
    print("\n=== Compression ===")
    print(f"  param ratio:   {teacher_stats['params_m'] / student_stats['params_m']:.1f}x")
    print(f"  layer ratio (Conv2d): {teacher_stats['layers']['Conv2d'] / student_stats['layers']['Conv2d']:.1f}x")
    print(f"  latency ratio (4090 fp32): {teacher_stats['lat_ms'] / student_stats['lat_ms']:.1f}x")

    # Storage
    import os
    teacher_path = prepare.TEACHER_PATH
    if not teacher_path.exists():
        torch.save(teacher.state_dict(), teacher_path)
    teacher_size = os.path.getsize(teacher_path) / 1e6
    print(f"\n  teacher .pt:   {teacher_size:.1f} MB")
    print(f"  student .pt:   {sum(p.numel() * 4 for p in s.parameters()) / 1e6:.1f} MB (fp32 calc)")
    print(f"  student qnn:   ~6.97 MB (w8a16 INT8 weights on AI Hub)")


if __name__ == "__main__":
    main()
