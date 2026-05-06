"""Visualize detection quality across teacher / student-fp32 / student-quantized.

Picks N diverse COCO val images, runs each model, draws annotated boxes,
composes a grid PNG. Tells the deployment-chain story:
    GPU teacher  ->  mobile student fp32  ->  on-device w8a16 quantized
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import prepare
import student
import deploy

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

OUT_DIR = Path("/tmp/autoresearch_viz")
OUT_DIR.mkdir(exist_ok=True)
SCORE_THRESHOLD = 0.4   # only draw confident boxes
INPUT_SIZE = student.INPUT_SIZE
DEVICE = prepare.device()

# 8 deliberately varied COCO val image ids: street, indoor, animals, sports, crowd
IMAGE_IDS = [
    37777,    # baseball game
    397133,   # zebra herd
    252219,   # dining table
    87038,    # crowded street
    174482,   # baseball pitcher
    403385,   # dog and frisbee
    61471,    # bird
    472375,   # surfer
]

# ----------------------------------------------------------------------------
# Color palette (one per category, from a fixed pastel set)
# ----------------------------------------------------------------------------
CATEGORY_COLORS = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9",
] * 10

def color_for(label_id: int) -> str:
    return CATEGORY_COLORS[int(label_id) % len(CATEGORY_COLORS)]


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------

def load_font(size: int = 14) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_detections(img: Image.Image, boxes_xyxy, scores, labels,
                    cat_id_to_name: dict, threshold: float = SCORE_THRESHOLD) -> Image.Image:
    out = img.copy().convert("RGB")
    drw = ImageDraw.Draw(out)
    font = load_font(max(12, img.height // 35))
    n = 0
    for box, sc, lbl in sorted(zip(boxes_xyxy, scores, labels),
                                key=lambda t: -t[1]):
        if sc < threshold:
            continue
        n += 1
        if n > 20: break
        x1, y1, x2, y2 = [float(v) for v in box]
        c = color_for(int(lbl))
        drw.rectangle([x1, y1, x2, y2], outline=c, width=3)
        name = cat_id_to_name.get(int(lbl), str(int(lbl)))
        text = f"{name} {sc:.2f}"
        tb = drw.textbbox((x1, y1), text, font=font)
        pad = 2
        drw.rectangle([tb[0]-pad, tb[1]-pad, tb[2]+pad, tb[3]+pad], fill=c)
        drw.text((x1, y1), text, fill="white", font=font)
    return out


# ----------------------------------------------------------------------------
# Detection runners
# ----------------------------------------------------------------------------

@torch.no_grad()
def run_host_model(model, pil_img):
    """Run a host-side detection model. Returns (boxes_xyxy, scores, labels)."""
    import torchvision.transforms.functional as F
    img_t = F.to_tensor(pil_img).to(DEVICE)
    pred = model([img_t])[0]
    return (pred["boxes"].cpu().numpy(),
            pred["scores"].cpu().numpy(),
            pred["labels"].cpu().numpy())


def prepare_input_for_npu(pil_img):
    """Resize to (INPUT_SIZE, INPUT_SIZE), to-tensor in [0,1], add batch dim, return numpy."""
    import torchvision.transforms.functional as F
    img_t = F.to_tensor(pil_img.resize((INPUT_SIZE, INPUT_SIZE)))
    return img_t.unsqueeze(0).numpy().astype(np.float32)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    val = prepare.load_coco_val()
    coco = val.coco
    cat_id_to_name = {c["id"]: c["name"] for c in coco.cats.values()}

    # Resolve image_ids to dataset indices
    id_to_idx = {img_id: i for i, img_id in enumerate(val.image_ids)}
    use_ids = [iid for iid in IMAGE_IDS if iid in id_to_idx]
    print(f"using {len(use_ids)} val images: {use_ids}", flush=True)

    # Load models
    print("loading teacher (RetinaNet)...", flush=True)
    teacher = prepare.load_teacher().to(DEVICE).eval()
    print("loading student fp32 (pretrained SSDLite)...", flush=True)
    student_host = student.build_student().to(DEVICE).eval()

    # 1) Run teacher + student fp32 on host
    teacher_results = []
    student_fp32_results = []
    pil_imgs = []
    for img_id in use_ids:
        idx = id_to_idx[img_id]
        img_t, _, _ = val[idx]
        pil = Image.fromarray((img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        pil_imgs.append(pil)
        teacher_results.append(run_host_model(teacher, pil))
        student_fp32_results.append(run_host_model(student_host, pil))
        print(f"  {img_id}: teacher={len(teacher_results[-1][0])}, "
              f"student_fp32={len(student_fp32_results[-1][0])}", flush=True)

    # 2) Build the on-device quantized model and run inference
    print("tracing student for NPU...", flush=True)
    # Need a fresh host model in eval mode for tracing (deploy._BackbonePlusHead wraps it)
    host_for_trace = student.build_student().eval()
    traced, input_shape = deploy.trace_for_npu(host_for_trace, INPUT_SIZE)

    print("compiling on AI Hub (w8a16, 256-image calib)...", flush=True)
    # Build calibration set (last 256 of val)
    n_total = len(val)
    calib = []
    for i in range(n_total - 256, n_total):
        img_t, _, _ = val[i]
        calib.append(prepare_input_for_npu(
            Image.fromarray((img_t.permute(1,2,0).numpy()*255).astype(np.uint8))))
    calibration_data = {"image": calib}

    hub_metrics = prepare.submit_aihub_profile(
        traced, input_shape,
        quantize=True, calibration_data=calibration_data,
        extra_options="--quantize_full_type w8a16",
    )
    target_model = hub_metrics["target_model"]
    print(f"  qnn_export_ok={hub_metrics.get('qnn_export_ok')}, "
          f"latency={hub_metrics.get('latency_ms_p50'):.2f}ms", flush=True)

    print(f"submitting inference job for {len(pil_imgs)} images...", flush=True)
    quant_inputs = {"image": [prepare_input_for_npu(p) for p in pil_imgs]}
    outputs = prepare.submit_aihub_inference(target_model, quant_inputs)
    out_names = list(outputs.keys())
    bbox_outs = outputs[out_names[0]]
    cls_outs = outputs[out_names[1]]

    # 3) Decode quantized outputs through host post-processing
    print("decoding quantized outputs...", flush=True)
    image_list_dummy = student_host.transform([torch.zeros(3, INPUT_SIZE, INPUT_SIZE)], None)[0]
    feats_dummy = list(student_host.backbone(image_list_dummy.tensors.to(DEVICE)).values())
    anchors_one = student_host.anchor_generator(
        image_list_dummy, [f.cpu() for f in feats_dummy])[0]

    quant_results = []
    for pil, bbox_np, cls_np in zip(pil_imgs, bbox_outs, cls_outs):
        bbox_t = torch.from_numpy(bbox_np)
        cls_t = torch.from_numpy(cls_np)
        head_outputs = {"bbox_regression": bbox_t, "cls_logits": cls_t}
        det = student_host.cpu().postprocess_detections(
            head_outputs, [anchors_one], [(INPUT_SIZE, INPUT_SIZE)])[0]
        # Rescale to original image
        boxes = det["boxes"].numpy().copy()
        sx = pil.width / INPUT_SIZE
        sy = pil.height / INPUT_SIZE
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        quant_results.append((boxes,
                              det["scores"].numpy(),
                              det["labels"].numpy()))
        # Move student back to device for next iter (we did .cpu() above)
        student_host.to(DEVICE)

    # 4) Annotate and save per-image triptych PNGs
    print("composing visualizations...", flush=True)
    titles = [
        "Teacher (RetinaNet R50, fp32 GPU)",
        "Student (SSDLite, fp32 host)",
        "Student (SSDLite, w8a16 on-device)",
    ]
    rows = []
    target_w = 480
    for i, (img_id, pil) in enumerate(zip(use_ids, pil_imgs)):
        triptych = []
        for results, title in zip(
            [teacher_results[i], student_fp32_results[i], quant_results[i]],
            titles,
        ):
            ann = draw_detections(pil, *results, cat_id_to_name)
            ann.thumbnail((target_w, target_w * pil.height // pil.width))
            triptych.append((title, ann))
        rows.append((img_id, triptych))

    # Compose grid
    img_w = max(t[1].width for r in rows for t in r[1])
    img_h = max(t[1].height for r in rows for t in r[1])
    title_h = 30
    cell_w, cell_h = img_w + 8, img_h + title_h + 8
    grid_w = cell_w * 3
    grid_h = cell_h * len(rows)
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    drw = ImageDraw.Draw(grid)
    title_font = load_font(16)

    for row_i, (img_id, triptych) in enumerate(rows):
        for col_i, (title, ann) in enumerate(triptych):
            x = col_i * cell_w + 4
            y = row_i * cell_h + 4
            drw.rectangle([x, y, x + img_w, y + title_h - 2], fill="#222")
            drw.text((x + 6, y + 5), f"{title}  [img {img_id}]",
                     fill="white", font=title_font)
            grid.paste(ann, (x, y + title_h))

    out_path = OUT_DIR / "comparison_grid.png"
    grid.save(out_path, dpi=(120, 120))
    print(f"\nsaved: {out_path}", flush=True)
    # Also save individual triptychs for closer inspection
    for img_id, triptych in rows:
        for col_i, (title, ann) in enumerate(triptych):
            ann.save(OUT_DIR / f"{img_id}_{col_i}.png")
    print(f"individual annotated images: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
