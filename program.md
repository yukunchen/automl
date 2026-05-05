# autoresearch-cv (detection · PC→mobile distillation)

This is an experiment to have an LLM agent autonomously distill a PC-grade
object detector into a mobile-deployable student, targeting Snapdragon 8
Gen 1 class hardware.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g.
   `cv-may5`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current
   master.
3. **Read the in-scope files**:
   - `README.md` — repo context.
   - `prepare.py` — frozen: dataset loader, teacher model loader, mAP
     evaluator, on-device benchmark client. Do not modify.
   - `student.py` — the student architecture, distillation loss, training
     loop. **You modify this.**
   - `deploy.py` — quantization, ONNX→QNN export, on-device benchmark
     submission. **You modify this.**
4. **Verify resources exist**:
   - `~/.cache/autoresearch-cv/coco/` (or chosen dataset) populated.
   - Teacher checkpoint at `~/.cache/autoresearch-cv/teacher.pt`.
   - `QAI_HUB_API_TOKEN` env var set (Qualcomm AI Hub for remote real-device
     benchmark on SD8 Gen 1). If not, tell the human.
5. **Initialize results.tsv**: create with header row only.
6. **Confirm and go**.

Once you get confirmation, kick off experimentation.

## The problem

- **Teacher**: a PC-grade detector running on RTX 4090 (e.g. DINO / YOLOv8-X
  / RT-DETR). Frozen. Defines the upper bound and provides soft labels +
  feature maps.
- **Student**: must run on Snapdragon 8 Gen 1 class NPU (Hexagon).
- **Dataset**: COCO val2017 (or the dataset specified in `prepare.py`) —
  evaluated with standard mAP@[.5:.95].

**Hard constraints (any violation → score = -inf, discard the run):**

| Constraint | Budget |
|---|---|
| Single-image latency on SD8 Gen 1 NPU | ≤ 30 ms |
| Quantized model size | ≤ 10 MB |
| Peak runtime memory | ≤ 200 MB |
| Must export to QNN successfully | required |
| Must benchmark green on AI Hub | required |

**Soft objective (the score):**

```
score = mAP_quantized   if all hard constraints pass
        -inf            otherwise
```

We use the **quantized, on-device** mAP as the metric — not the FP32 PyTorch
mAP. This is the only number that matters for shipping. Quantization gap
gets attributed to the experiment that introduced it.

## Experimentation

Each experiment runs end-to-end via:

```
bash run_experiment.sh "<short description of what changed>"
```

This handles venv switching (`.venv-train` for training, `.venv-deploy` for
AI Hub), writes everything to `run.log`, and prints a `RESULTS_ROW` line you
append to `results.tsv` (after deciding keep/discard/crash).

`deploy.py` submits the model to **Qualcomm AI Hub** which runs it on a real
SD8 Gen 1 device and returns latency + per-layer profile. No physical
hardware needed locally.

**What you CAN do:**
- Modify `student.py` — student architecture (backbone, neck, head),
  distillation method (logit KD, feature KD, attention transfer, relation
  KD, on-policy), loss weights, temperature, optimizer, schedule, augment.
- Modify `deploy.py` — quantization scheme (PTQ vs QAT, INT8/INT4,
  per-channel/per-tensor, mixed precision per layer), calibration set size,
  operator fusion config, QNN graph optimization flags.

**What you CANNOT do:**
- Modify `prepare.py`. Frozen. Contains: data loader, teacher loader, mAP
  evaluator, AI Hub client, hard-constraint checker.
- Add new pip dependencies.
- Change the metric or the hard-constraint budgets.
- Use synthetic test data or fake-quantize when reporting a score — only
  real on-device numbers count.

## Mobile-friendly architecture rules

The Hexagon NPU is fast on a narrow set of ops. Stay inside the lane:

**Prefer**: depthwise-separable conv, 3×3 / 1×1 conv, ReLU/ReLU6, BN (fold
into conv), residual add, static shapes, NHWC layout, INT8.

**Avoid**: GroupNorm/LayerNorm in conv stages (slow), large kernels (>5),
dilated conv with large rates, dynamic shapes, fancy attention with large
softmax, ops AI Hub flags as "CPU fallback" (kills latency).

If a layer falls back to CPU on the device profile, treat it as a bug and
fix it.

## Output format

`student.py` prints a training summary; `deploy.py` prints the deployment
summary. The combined log will contain:

```
---
fp32_map:           0.428
fp32_map50:         0.612
quantized_map:      0.401
latency_ms_p50:     22.4
latency_ms_p99:     27.1
size_mb:            8.7
peak_mem_mb:        174
qnn_export_ok:      true
cpu_fallback_ops:   0
score:              0.401
training_seconds:   3120
deploy_seconds:     180
---
```

Extract with:
```
grep -E "^(score|quantized_map|latency_ms_p50|size_mb|peak_mem_mb|qnn_export_ok|cpu_fallback_ops):" run.log
```

If `score: -inf`, identify which constraint failed from the other lines.

## Logging results

Append to `results.tsv` (tab-separated). Header:

```
commit	score	fp32_map	q_map	lat_ms	size_mb	mem_mb	qnn_ok	cpu_fb	status	description
```

1. git commit (short, 7 chars)
2. score (quantized_map if pass, else -inf)
3. fp32 mAP (PyTorch, pre-quant)
4. quantized mAP (on-device)
5. p50 latency ms
6. size MB
7. peak memory MB
8. qnn export ok (true/false)
9. cpu fallback op count
10. status: `keep`, `discard`, `crash`
11. short description

Example:

```
commit	score	fp32_map	q_map	lat_ms	size_mb	mem_mb	qnn_ok	cpu_fb	status	description
a1b2c3d	0.358	0.401	0.358	24.1	8.2	168	true	0	keep	baseline: MobileNetV3-S backbone + FCOS head, logit KD T=4
b2c3d4e	-inf	0.412	0.000	-1.0	9.1	0	false	-	discard	add MobileViT block — QNN export fails on attention softmax shape
c3d4e5f	0.371	0.408	0.371	26.8	8.9	172	true	0	keep	+ feature KD on neck P3/P4/P5
d4e5f6g	-inf	0.395	0.342	34.2	8.4	165	true	2	discard	wider neck — latency budget blown, 2 ops on CPU
e5f6g7h	0.379	0.405	0.379	23.0	7.1	160	true	0	keep	switch PTQ → QAT, recover 0.8 mAP
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/cv-may5`).

LOOP FOREVER:

1. Check git state.
2. Form a hypothesis. Modify `student.py` and/or `deploy.py`.
3. `git commit -am "<short description>"`.
4. `bash run_experiment.sh "<short description>"` — runs train+deploy,
   produces `run.log` and a `RESULTS_ROW` line on stdout.
5. If the script printed a `crash` row, `tail -n 80 run.log`, decide:
   fix-and-rerun if trivial, else log `crash` in `results.tsv` and revert.
6. Read the `RESULTS_ROW` line. Compare quantized_map (column 4) to the
   current best in `results.tsv`.
7. Edit the `RESULTS_ROW` to set status: `keep` if improved (and no
   constraint violation), `discard` if not, `crash` if failed.
8. Append the edited row to `results.tsv`.
9. If `keep`, advance the branch (the commit stays). Else
   `git reset --hard HEAD~1` back to the last keep commit.

**Important**: the metric you optimize is **quantized_map under all hard
constraints**, not fp32 mAP. A change that lifts fp32 mAP by 2 points but
loses 3 points to quantization is a regression. Stay honest.

## Iteration strategy hints

Two coupled axes. Don't tune both at once — you'll never know which knob
moved the number.

**Outer loop (architecture + distillation, slow, hours-per-experiment):**
- Backbone choice (MobileNetV3-S/L, EfficientNet-Lite, MobileOne, RepVGG-A0)
- Neck (PAN, BiFPN-lite, simple FPN)
- Head (FCOS-lite, NanoDet-style, YOLOX-tiny head)
- Distillation: logit KD → + feature KD on neck → + relation KD → on-policy
- Loss balance, temperature, schedule

**Inner loop (quantization & deployment, fast, minutes-per-experiment):**
Given a fixed trained checkpoint, agent iterates on `deploy.py` only:
- PTQ calibration set size, mix
- Per-channel vs per-tensor weight quant
- Activation quant scheme
- Layers held at FP16 (typically detection head's regression branch)
- QNN graph fusion flags

Inner loop is cheap — exploit it. After a successful outer-loop training,
spend 10-20 inner-loop iterations squeezing quantized_map before moving on.

## Failure modes to watch

1. **CPU fallback ops** — single unsupported op kills latency. `cpu_fb > 0`
   means investigate immediately.
2. **Quantization collapse** — fp32 mAP fine, quantized mAP near zero. Usually
   activation outliers in detection head. Try QAT or hold that layer FP16.
3. **Memory spike during NMS** — pre-NMS box count explodes on hard images.
   Cap top-K before NMS.
4. **Static shape violation** — anchor-free heads with variable output count
   need padding to fixed shape for QNN.
5. **Train/eval skew** — augmentation pipeline differs between PC train and
   on-device eval. Match preprocessing exactly.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask whether to continue.
The human may be asleep. Run until manually stopped. If you run out of
ideas: re-read recent `keep` rows for what's working, read teacher feature
maps to see what student is missing, try combining two near-misses, try
something more radical (different student family entirely). The loop runs
until interrupted, period.

A typical outer-loop experiment takes 30-90 min, an inner-loop one takes
5-15 min, so expect ~10-30 experiments per overnight run.
