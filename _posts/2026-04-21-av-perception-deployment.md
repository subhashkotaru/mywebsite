---
title: "Deploying Perception Models for Autonomous Vehicles"
date: 2026-04-21
description: "A concrete end-to-end guide to deploying a BEV perception model from PyTorch to TensorRT — covering ONNX export, quantisation, parity validation, HIL testing, and the continuous fleet-data retraining loop. Written for ML deployment roles at AV companies like Kodiak, Waymo, and Cruise."
tags: [autonomous-vehicles, perception, tensorrt, deployment, inference, cuda]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">What This Role Is Really Optimising</a></li>
    <li><a href="#stage-a">Stage A — Problem Framing</a></li>
    <li><a href="#stage-b">Stage B — Data Pipeline & Distributed Training</a>
      <ul class="post-toc-sublist">
        <li><a href="#data-curation">Data Curation & Scenario Slices</a></li>
        <li><a href="#training-infra">Distributed Training Infrastructure</a></li>
        <li><a href="#training-metrics">What to Measure During Training</a></li>
      </ul>
    </li>
    <li><a href="#stage-c">Stage C — Offline Evaluation & Acceptance Criteria</a></li>
    <li><a href="#stage-d">Stage D — PyTorch → ONNX Export</a>
      <ul class="post-toc-sublist">
        <li><a href="#export-steps">Export Steps</a></li>
        <li><a href="#export-failures">Common Export Failures</a></li>
        <li><a href="#onnx-parity">ONNXRuntime Parity Check</a></li>
      </ul>
    </li>
    <li><a href="#stage-e">Stage E — TensorRT Engine Build</a>
      <ul class="post-toc-sublist">
        <li><a href="#fp16-int8">FP16 vs INT8</a></li>
        <li><a href="#calibration">INT8 Calibration</a></li>
        <li><a href="#trt-knobs">TensorRT Knobs</a></li>
      </ul>
    </li>
    <li><a href="#stage-f">Stage F — Custom CUDA Kernels & TRT Plugins</a></li>
    <li><a href="#stage-g">Stage G — Numerical Parity Testing</a></li>
    <li><a href="#stage-h">Stage H — End-to-End Stack Integration & Profiling</a>
      <ul class="post-toc-sublist">
        <li><a href="#pipeline-bottlenecks">Pipeline Bottlenecks</a></li>
        <li><a href="#profiling-tools">Profiling Tools</a></li>
      </ul>
    </li>
    <li><a href="#stage-i">Stage I — HIL and Log Replay Validation</a></li>
    <li><a href="#stage-j">Stage J — Rollout & Continuous Learning Loop</a></li>
    <li><a href="#cross-functional">Cross-Functional Interfaces</a>
      <ul class="post-toc-sublist">
        <li><a href="#xf-perception">Working with Perception</a></li>
        <li><a href="#xf-planning">Working with Planning</a></li>
        <li><a href="#xf-simulation">Working with Simulation</a></li>
        <li><a href="#xf-infra">Working with Autonomy Infrastructure</a></li>
      </ul>
    </li>
    <li><a href="#ml-infra">Scalable ML Infrastructure & Model CI/CD</a>
      <ul class="post-toc-sublist">
        <li><a href="#model-cicd">Model CI/CD Pipeline</a></li>
        <li><a href="#artifact-registry">Artifact Registry & Versioning</a></li>
        <li><a href="#deployment-gates">Deployment Gates as Code</a></li>
        <li><a href="#train-profiling">Train-Time Profiling Deep Dive</a></li>
      </ul>
    </li>
    <li><a href="#topic-tree">Topic Tree: What to Study</a>
      <ul class="post-toc-sublist">
        <li><a href="#root-perception">Autonomy Perception Basics</a></li>
        <li><a href="#root-deployment">Deployment Fundamentals</a></li>
        <li><a href="#root-gpu">GPU & System Optimisation</a></li>
        <li><a href="#root-training">Distributed Training</a></li>
        <li><a href="#root-validation">Validation & Parity</a></li>
        <li><a href="#root-safety">Safety & Robustness Mindset</a></li>
      </ul>
    </li>
    <li><a href="#interview-prep">Interview Prep: Question Buckets</a></li>
  </ul>
</nav>

---

## What This Role Is Really Optimising
{: #overview}

Most ML candidates optimise for accuracy. Strong deployment engineers optimise for four things simultaneously:

<div class="post-flow post-flow--compare" role="group" aria-label="Four deployment axes">
  <div class="post-flow__col">
    <p class="post-flow__col-label">What Most Know</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">A. Accuracy — mAP, IoU, recall curves</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">What Autonomy Needs</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">A. Accuracy — does perception improve?</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">B. Latency — can it run within the frame budget (50ms at 20Hz)?</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">C. Fidelity — does the deployed runtime match the offline model exactly?</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">D. Safety robustness — does it still work on rare, high-stakes edge cases?</span></li>
    </ol>
  </div>
</div>

The concrete case study we will follow: deploying a **camera-based BEV perception model** for a class-8 highway trucking platform (Kodiak-style). The model takes multi-camera frames + calibration and outputs 3D boxes, classes, velocities, lane boundaries, and drivable area — all consumed by the tracker, predictor, and planner.

---

## Stage A — Problem Framing
{: #stage-a}

Before writing a single line of training or deployment code, answer these questions:

**What are the safety-critical object classes?**

| Class | Why critical | Failure mode |
|---|---|---|
| Stopped truck | Can appear suddenly over a hill at highway speed | False negative → collision |
| Pedestrian / cyclist | Unpredictable trajectory | Any miss is unacceptable |
| Work zone cones | Define drivable corridor | Miss → drives over cones, activates safety driver |
| Debris / tire fragments | Unexpected on highway | FP on road markings wastes interventions |
| Cut-in vehicle | 3-second window to react | Late detection → uncomfortable or unsafe brake |

**What is the latency budget?**

At 20Hz, each frame takes 50ms. The perception model gets a fraction of that:

```
Total frame budget:     50ms
Sensor sync + ingest:   ~3ms
Preprocessing (CPU):    ~5ms
Model inference:        ≤ 25ms  ← TensorRT target
Postprocessing + NMS:   ~5ms
Output serialisation:   ~2ms
Remaining margin:       ~10ms
```

If your TensorRT engine takes 35ms, you must cut 10ms before deploying — through quantisation, pruning, or architectural changes.

**What is the expected downstream consumption?**

```
Perception outputs:
  ├── 3D boxes + class + velocity     → Multi-Object Tracker
  ├── Lane boundaries                 → Lane module → Planner
  ├── Drivable area mask              → Occupancy grid → Planner
  └── BEV feature map (optional)      → Prediction model (end-to-end)
```

Small regressions in perception can cascade. If score calibration shifts after INT8 quantisation, the tracker's assignment threshold may misfire — objects start getting created and destroyed spuriously, producing velocity jitter that makes the planner brake erratically. **Validate downstream impact, not just box-level metrics.**

**Interview question:** Before deploying a new perception model to the fleet, what acceptance criteria would you define?

> **Answer:** I would define criteria at four levels. (1) **Task metrics**: no regression on critical-class recall (pedestrian, stopped truck) beyond 0.5pp; overall mAP must match or exceed baseline. These are measured on held-out scenario slices, not just aggregate test set. (2) **Latency**: TensorRT engine P95 latency ≤ 25ms on target hardware; full pipeline P95 ≤ 45ms including preprocessing and postprocessing. (3) **Parity gates**: max absolute error between PyTorch and TRT outputs ≤ 1e-2 on 1000 validation frames; no class where recall drops > 1pp after conversion. (4) **Downstream**: tracker stability (track fragmentation rate, ID switch rate) must not regress on the log replay suite; planner comfort metric (jerk, unnecessary decels) must not worsen on the HIL suite. If any gate fails, the deployment is blocked. The gates are not negotiable — they exist because small perception errors become large safety errors downstream.

---

## Stage B — Data Pipeline & Distributed Training
{: #stage-b}

### Data Curation & Scenario Slices
{: #data-curation}

Fleet logs are the primary data source. Raw logs are not a training set — they need structured curation:

<div class="post-flow" role="group" aria-label="Data pipeline stages">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Raw fleet logs — continuous sensor streams, unstructured, petabytes</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Scenario mining — tag segments: cut-in, highway merge, work zone, night, rain, glare</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Annotation — 3D box labels, lane boundaries, drivable area (lidar-assisted auto-label + QA)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Hard negative mining — pull frames where model failed on previous version</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Slice construction — build balanced train/val sets per scenario</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Augmentation pipeline — camera noise, lighting jitter, synthetic rain/fog, virtual cut-ins</span></li>
  </ol>
</div>

**Scenario slice design:**

```python
SCENARIO_SLICES = {
    # Safety-critical — always represented, recall tracked separately
    "stopped_truck":     {"min_samples": 5000, "eval_weight": 3.0},
    "pedestrian_cross":  {"min_samples": 3000, "eval_weight": 3.0},
    "work_zone_cones":   {"min_samples": 4000, "eval_weight": 2.0},

    # Distribution coverage
    "nighttime":         {"min_samples": 8000, "eval_weight": 1.5},
    "rain_glare":        {"min_samples": 3000, "eval_weight": 1.5},
    "highway_merge":     {"min_samples": 6000, "eval_weight": 1.0},
    "long_range_50m":    {"min_samples": 4000, "eval_weight": 1.5},  # trucks beyond 50m
    "partial_occlusion": {"min_samples": 5000, "eval_weight": 1.5},
}
```

**Auto-labelling pipeline:** lidar provides ground truth 3D boxes. A lidar-trained teacher model auto-labels camera frames. Human QA samples ~5% of auto-labels, particularly for rare classes and edge cases. This 10–100× reduces annotation cost vs pure human labelling.

**Interview question:** Your model's overall mAP is excellent, but safety review flags that pedestrian recall in rain dropped 4pp after the last training run. How do you fix this without hurting the rest of the model?

> **Answer:** First I'd confirm the regression is real — run on the rain-pedestrian slice before and after the change that caused it (new data, augmentation change, architecture change). Once confirmed: (1) **Check if rain-pedestrian samples are underrepresented** — if the new training data has fewer rain logs, the model forgot. Fix: re-balance training with explicit rain-pedestrian minimum. (2) **Check augmentation** — if we added a new augmentation that distorts pedestrian silhouettes in ways not seen at test time (e.g., aggressive blur at night), the model learns the wrong distribution. Fix: turn off or soften the augmentation. (3) **Add rain-pedestrian hard examples to training** and increase their sampling weight. (4) **Add rain-pedestrian recall to the primary eval dashboard** so future regressions are caught immediately. The fix should be targeted — I don't want to retrain from scratch or change the full data recipe. Surgical sample re-weighting + slice tracking is usually enough.

---

### Distributed Training Infrastructure
{: #training-infra}

```python
# DDP training loop skeleton for BEV perception
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

def train(rank: int, world_size: int, config: dict):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = BEVPerceptionModel(config).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Mixed precision: FP16 forward/backward, FP32 master weights
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # DistributedSampler ensures no overlap between GPUs
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=config["batch_per_gpu"],
                        sampler=sampler, num_workers=8, pin_memory=True,
                        persistent_workers=True)   # keep workers alive across epochs

    for epoch in range(config["epochs"]):
        sampler.set_epoch(epoch)  # reshuffle per epoch for DDP
        for batch in loader:
            images = batch["images"].to(rank, non_blocking=True)   # H2D async
            targets = batch["targets"]

            with autocast():   # FP16 forward pass
                outputs = model(images)
                loss = compute_loss(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stability
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
```

**When to use FSDP instead of DDP:**

| Scenario | Recommendation |
|---|---|
| Model fits on one GPU (e.g., < 40GB) | DDP — simpler, lower overhead |
| Model doesn't fit on one GPU | FSDP — shard parameters, grads, optimizer state |
| Very large backbone (ViT-H, dual encoder) | FSDP with `FULL_SHARD` + activation checkpointing |
| Multi-node with NVLink within node | DDP within node, FSDP across nodes |

**Common training bottlenecks and fixes:**

| Bottleneck | How to detect | Fix |
|---|---|---|
| CPU preprocessing | `nvtx` / profiler shows long DataLoader gaps | Increase `num_workers`, use DALI/NVJPEG for decoding |
| Small batch size → low GPU utilisation | SM utilisation < 60% in `nvidia-smi` | Gradient accumulation, larger batch per GPU |
| AllReduce latency | `dist.all_reduce` time > 30% of step time | Gradient compression (PowerSGD), reduce synchronisation frequency |
| Storage I/O | Worker processes stalled on disk reads | Pre-cache to tmpfs/SSD, shuffle files across nodes, WebDataset sharding |
| Expensive augmentations | Profile per-augmentation time | Offload heavy augmentations to GPU (Kornia), cache preprocessed tensors |

---

### What to Measure During Training
{: #training-metrics}

Never track only aggregate mAP. Build a per-slice eval dashboard that runs every N steps:

```python
EVAL_METRICS = {
    # Aggregate
    "mAP@0.5":              {"threshold": 0.5},
    "mAP@0.5:0.95":         {"threshold": "coco"},

    # Per class — tracked separately
    "AP_pedestrian":        {},
    "AP_cyclist":           {},
    "AP_truck_stopped":     {},
    "AP_cone":              {},

    # Long-range
    "recall@50m_truck":     {"distance_range": (40, 60)},
    "recall@30m_ped":       {"distance_range": (20, 40)},

    # Safety: recall at low FP budget
    "recall_at_0.1FPpF_ped": {"FP_per_frame_budget": 0.1},  # rare but critical

    # Temporal quality
    "box_jitter":            {},   # mean std of box position over 5 consecutive frames
    "score_jitter":          {},   # variance in confidence for same object across frames

    # Lane / drivable area
    "lane_F1@0.5m":          {},
    "drivable_IoU":          {},
}
```

**Why temporal jitter matters:** A box that jumps ±0.5m laterally per frame causes the tracker to either drift or re-initialise — producing velocity jitter that propagates to the planner as false accelerations. Low jitter is a proxy for temporal model consistency, independent of absolute accuracy.

---

## Stage C — Offline Evaluation & Acceptance Criteria
{: #stage-c}

Before any export or deployment work, define your acceptance gates for the **PyTorch model**:

```python
ACCEPTANCE_GATES = {
    # No deployment proceeds unless all gates pass

    "mAP@0.5":                  {"min": 0.62,  "baseline": 0.60},  # +2pp floor
    "recall_pedestrian@0.5":    {"min": 0.85,  "baseline": 0.84},  # safety-critical
    "recall_stopped_truck@0.5": {"min": 0.88,  "baseline": 0.87},  # safety-critical
    "recall_cone@0.5":          {"min": 0.80,  "baseline": 0.79},

    # Slices
    "AP_nighttime":             {"min": 0.55,  "no_regression": -0.01},
    "AP_rain":                  {"min": 0.53,  "no_regression": -0.01},
    "AP_long_range_50m":        {"min": 0.45,  "no_regression": -0.02},

    # Calibration: post-NMS score calibration should match baseline
    "calibration_ECE":          {"max": 0.05},  # Expected Calibration Error < 5%

    # Latency (PyTorch, A100, batch=1)
    "inference_ms_p95":         {"max": 60},    # TRT will be ~2x faster → 30ms target
}
```

**Calibration check:** expected calibration error (ECE) measures whether predicted confidence ≈ actual frequency. If ECE is high after training, INT8 quantisation will make it worse — and downstream score thresholds will be wrong.

```python
def expected_calibration_error(confidences, labels, n_bins=15):
    """
    ECE: average gap between predicted confidence and actual accuracy, binned.
    A well-calibrated model: confidence 0.8 → correct 80% of the time.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = labels[mask].float().mean()
        ece += mask.mean() * abs(avg_conf - avg_acc)
    return ece
```

---

## Stage D — PyTorch → ONNX Export
{: #stage-d}

### Export Steps
{: #export-steps}

```python
import torch
import torch.onnx

def export_to_onnx(model, onnx_path: str, input_shape: tuple):
    model.eval()

    # 1. Create representative dummy input (matches runtime input exactly)
    dummy = torch.randn(input_shape).cuda()  # e.g. (1, 6, 3, 900, 1600) for 6 cameras

    # 2. Freeze model — disable dropout, batchnorm running stats fixed
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            opset_version=17,           # use latest stable opset
            input_names=["images"],
            output_names=["boxes", "scores", "labels", "lane_pts"],
            dynamic_axes={
                # Only make batch dynamic if needed — static shapes are faster
                # "images": {0: "batch_size"},
            },
            do_constant_folding=True,   # fold constant subgraphs
            export_params=True,
            verbose=False,
        )

    # 3. Verify the ONNX graph is valid
    import onnx
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print(f"ONNX model exported: {onnx_path}")
    print(f"Graph inputs:  {[n.name for n in model_onnx.graph.input]}")
    print(f"Graph outputs: {[n.name for n in model_onnx.graph.output]}")
```

**Before exporting — pre-export checklist:**

```python
# Items to handle BEFORE torch.onnx.export
PREEXPORT_CHECKLIST = [
    "Remove .training mode checks (model.eval() and no if self.training: branches)",
    "Replace F.interpolate with explicit mode='bilinear', align_corners=False",
    "Replace custom autograd.Function with ONNX-compatible equivalent",
    "Bake in camera intrinsics / calibration as constants, not dynamic inputs",
    "Replace Python list comprehensions in graph-critical paths with tensor ops",
    "Move any Python-only postprocessing (NMS, decode) OUT of the exported model",
    "Replace torch.where with ONNX-compatible scatter/gather if control-flow dependent",
]
```

### Common Export Failures
{: #export-failures}

| Failure | Symptom | Fix |
|---|---|---|
| **Custom autograd op** | `RuntimeError: No ONNX export function for type CustomOp` | Register an ONNX symbolic for the op, or replace with ONNX-compatible equivalent |
| **Dynamic control flow** | `torch.jit.export` traceing fails; graph looks wrong | Replace `if tensor.item() > threshold:` with `torch.where(...)` |
| **Resize/interpolation mismatch** | Outputs differ between PT and ORT due to different resize math | Explicitly pass `align_corners=False`, compare results at known scale factors |
| **Variable-length NMS** | ONNX graph can't represent dynamic-length outputs | Move NMS to postprocessing AFTER the ONNX model; export only backbone + head |
| **Unsupported opset ops** | `AttributeError: ONNX opset 13 does not support...` | Upgrade opset version or write a custom op symbolic |
| **Shape inference fails** | onnx.checker passes but tensorrt build fails later | Use `onnx-simplifier` (`onnxsim`) to canonicalise the graph |
| **Normalisation drift** | Pixel mean/std baked in differently in PT vs ONNX | Explicitly add normalise op to the graph; don't rely on model `__init__` defaults |

```bash
# Simplify ONNX graph (recommended before TRT build)
pip install onnxsim
python -m onnxsim model.onnx model_simplified.onnx

# Inspect the ONNX graph visually
pip install netron
netron model_simplified.onnx   # opens browser viewer
```

### ONNXRuntime Parity Check
{: #onnx-parity}

```python
import onnxruntime as ort
import numpy as np
import torch

def check_onnx_parity(model_pt, onnx_path: str, inputs: dict, tol: float = 1e-4):
    """
    Compare PyTorch model output vs ONNXRuntime output on the same input.
    Run on 100+ frames. Log failures — don't just assert.
    """
    model_pt.eval()

    # PyTorch forward pass
    with torch.no_grad():
        pt_outputs = model_pt(**{k: v.cuda() for k, v in inputs.items()})

    # ONNXRuntime forward pass
    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
    ort_outputs = sess.run(None, ort_inputs)

    # Compare
    failures = []
    for name, pt_out, ort_out in zip(["boxes","scores","labels"], pt_outputs, ort_outputs):
        pt_np = pt_out.cpu().numpy()
        max_abs_err = np.abs(pt_np - ort_out).max()
        mean_abs_err = np.abs(pt_np - ort_out).mean()
        if max_abs_err > tol:
            failures.append({
                "output": name,
                "max_abs_err": max_abs_err,
                "mean_abs_err": mean_abs_err,
            })
        print(f"  {name}: max_err={max_abs_err:.2e}, mean_err={mean_abs_err:.2e}")

    if failures:
        print("PARITY FAILURES — do not proceed to TensorRT:")
        for f in failures:
            print(f"  {f}")
    else:
        print("ONNX parity check PASSED")
    return failures
```

**Interview question:** You export a model to ONNX and the outputs differ from PyTorch by up to 0.03 in confidence scores, but the box coordinates match exactly. Is this acceptable? What do you do?

> **Answer:** It depends on the magnitude and distribution of the difference, not just the number. First: is 0.03 a uniform shift or is it score-dependent? A uniform shift means downstream thresholds need recalibration but the ranking is preserved — potentially acceptable. A score-dependent shift (models scores near 0.5 differently than scores near 0.9) changes which detections survive NMS and which don't — not acceptable without understanding why. Second: check if it affects recall on critical classes. Compute recall@50Hz-threshold on the validation set with both PT and ONNX scores — if they agree within 0.5pp on pedestrian and stopped-truck recall, the delta is probably tolerable. Third: root-cause the difference anyway. Common cause: `torch.sigmoid` in PT vs `onnx::Sigmoid` with different float rounding. Run a bisection on the layer outputs to find where the divergence starts. Fix it rather than accepting it — a model deployed to a truck should have no unexplained numerical surprises.

---

## Stage E — TensorRT Engine Build
{: #stage-e}

### FP16 vs INT8
{: #fp16-int8}

Start with FP16. Only move to INT8 if latency still misses the budget after FP16.

```python
import tensorrt as trt

def build_trt_engine(onnx_path: str, engine_path: str, precision: str = "fp16"):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        parser.parse(f.read())

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))  # 4GB

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # Must provide a calibrator (see below)
        config.int8_calibrator = build_int8_calibrator(calibration_data)

    # For static input shapes: set optimisation profile
    profile = builder.create_optimization_profile()
    # min, opt, max shapes (all same for static deployment)
    profile.set_shape("images", (1,6,3,900,1600), (1,6,3,900,1600), (1,6,3,900,1600))
    config.add_optimization_profile(profile)

    serialised = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialised)
    print(f"TensorRT engine saved: {engine_path}")
```

### INT8 Calibration
{: #calibration}

INT8 maps 32-bit activations to 8-bit integers using per-layer scale factors. The scale factors are computed from a **calibration dataset** — a small representative sample of real inputs.

```python
class PerceptionCalibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibrator using entropy calibration (recommended for CNNs).
    
    Entropy calibration: choose the scale that minimises KL divergence
    between the full-precision activation distribution and the quantised one.
    Better than MinMax for activations with long tails (detection scores).
    """

    def __init__(self, calib_frames: list, cache_file: str = "calib_cache.bin"):
        super().__init__()
        self.frames = calib_frames      # 500–1000 diverse frames
        self.index = 0
        self.cache_file = cache_file
        # Allocate pinned memory for fast H2D transfer
        self.device_input = cuda.mem_alloc(calib_frames[0].nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index].ravel()
        cuda.memcpy_htod(self.device_input, frame)
        self.index += 1
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
```

**Critical: calibration dataset design.** The calibration set must cover the full **input distribution**. A calibration set of only daytime highway frames will produce poor INT8 accuracy at night — nighttime activations have a very different distribution, and the scales computed from daytime don't transfer.

```python
CALIBRATION_DATASET_COMPOSITION = {
    "daytime_clear":   0.30,
    "nighttime":       0.20,
    "rain_glare":      0.15,
    "work_zone":       0.15,
    "long_range":      0.10,
    "stopped_vehicle": 0.10,
}
```

### TensorRT Knobs
{: #trt-knobs}

| Knob | What it does | Recommendation |
|---|---|---|
| **Precision** | FP32 / FP16 / INT8 | Start FP16; INT8 only if latency gate fails |
| **Static shapes** | Fix batch and spatial dims | Always for AV deployment — eliminates shape inference overhead |
| **Workspace size** | Memory TRT can use for tactic selection | 4–8GB during build; set lower for runtime if memory-constrained |
| **Tactic selection** | TRT benchmarks multiple CUDA kernels and picks the fastest | Let it run fully (can take hours); cache the timing DB |
| **Layer precision overrides** | Force individual layers to FP16 even in INT8 mode | Keep detection heads and score sigmoid in FP16; quantise backbone |
| **Kernel fusion** | TRT automatically fuses Conv+BN+ReLU into one kernel | Enabled by default; verify with Nsight that fusions happened |

**Per-layer precision strategy (INT8 deployment):**

```python
# Force sensitive layers to stay in FP16 even when building INT8 engine
# Detection head scores are highly sensitive to quantisation noise
FORCE_FP16_LAYERS = [
    "detection_head.cls_logits",    # classification scores
    "detection_head.iou_pred",      # IoU branch
    "lane_head.heatmap",            # lane confidence heatmap
    "sigmoid_*",                    # all sigmoid activations
]

for layer_name in FORCE_FP16_LAYERS:
    layer = network.get_layer(layer_name)
    if layer:
        layer.precision = trt.DataType.HALF
        layer.set_output_type(0, trt.DataType.HALF)
```

**Interview question:** You build a TensorRT FP16 engine and test it. Pedestrian recall drops by 3pp compared to ONNX, but truck recall is unchanged. What is happening and how do you fix it?

> **Answer:** Pedestrian detections are smaller in the feature map than trucks — they occupy fewer pixels and produce weaker activations. FP16 has limited dynamic range compared to FP32. Weak activations from small objects sit close to the FP16 representable minimum and suffer more rounding error than the strong activations from large objects. The fix: (1) identify which layers produce the problematic rounding by comparing per-layer activation distributions in FP32 vs FP16 using hooks. (2) Force the layers responsible for small-object feature extraction (early detection head, small-stride feature pyramid level) to stay in FP32 or use `--precisionConstraints prefer` to let TRT choose FP32 where it matters. (3) Alternatively, add a pedestrian-specific calibration loss term in QAT (quantisation-aware training) so the model learns to make pedestrian activations more robust to rounding. The broader lesson: INT8/FP16 regressions are not uniform across classes — always break down your eval by class and distance range after every precision change.

---

## Stage F — Custom CUDA Kernels & TRT Plugins
{: #stage-f}

When standard ops are insufficient, you write TRT plugins or custom CUDA kernels:

**When to write a custom kernel:**
- NMS (non-maximum suppression) with non-standard IoU computation
- Deformable convolution (used in some BEV perception backbones)
- BEV pooling / voxel scatter (lifting 2D features to 3D BEV space)
- Custom attention with camera-geometry positional encoding
- Fused decode of anchor-free detection head (decode + clip + filter in one pass)

**TRT plugin skeleton:**

```cpp
// TensorRT plugin for custom BEV scatter (2D → BEV grid)
class BEVScatterPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    BEVScatterPlugin(int bev_h, int bev_w, float x_min, float x_max,
                     float y_min, float y_max)
        : bev_h_(bev_h), bev_w_(bev_w), x_min_(x_min), x_max_(x_max),
          y_min_(y_min), y_max_(y_max) {}

    // TRT calls this to get the output shape given the input shapes
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override
    {
        // output: [batch, C, bev_h, bev_w]
        nvinfer1::DimsExprs output;
        output.nbDims = 4;
        output.d[0] = inputs[0].d[0];          // batch
        output.d[1] = inputs[0].d[1];          // channels
        output.d[2] = exprBuilder.constant(bev_h_);
        output.d[3] = exprBuilder.constant(bev_w_);
        return output;
    }

    // The actual CUDA kernel is called here
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) noexcept override
    {
        bev_scatter_cuda(
            static_cast<const float*>(inputs[0]),    // 2D features: (N, C, H, W)
            static_cast<const float*>(inputs[1]),    // 3D points: (N, 3) xyz
            static_cast<float*>(outputs[0]),         // BEV grid: (N, C, bev_h, bev_w)
            inputDesc[0].dims, bev_h_, bev_w_,
            x_min_, x_max_, y_min_, y_max_,
            stream
        );
        return 0;
    }

private:
    int bev_h_, bev_w_;
    float x_min_, x_max_, y_min_, y_max_;
};
```

**Memory layout considerations for custom kernels:**

```cuda
// Bad: strided access across channels — cache unfriendly
// feature[batch][channel][h][w] in NCHW: accessing all channels for one pixel
// requires jumping channel_stride = H*W floats between reads

// Better: NHWC layout for feature aggregation
// accessing all channels for one pixel is contiguous memory → coalesced access
// TRT supports NHWC layout natively; request it in the plugin shape contract
```

**Overlapping compute and memory transfers with CUDA streams:**

```python
# Create separate streams for preprocessing, inference, postprocessing
preprocess_stream = torch.cuda.Stream()
infer_stream      = torch.cuda.Stream()
postprocess_stream = torch.cuda.Stream()

# Frame N+1 preprocessing runs in parallel with frame N inference
with torch.cuda.stream(preprocess_stream):
    images_n1 = preprocess(raw_frames[n+1])

# Synchronise: inference needs preprocessed input
infer_stream.wait_stream(preprocess_stream)

with torch.cuda.stream(infer_stream):
    outputs_n = trt_engine.execute_async_v3(images_n)

# Postprocessing for frame N-1 runs in parallel with inference for frame N
postprocess_stream.wait_stream(infer_stream)
with torch.cuda.stream(postprocess_stream):
    detections_n_minus1 = postprocess(outputs_n_minus1)
```

---

## Stage G — Numerical Parity Testing
{: #stage-g}

Parity testing is the gate between "model converted" and "model deployable". Run at three levels:

<div class="post-flow" role="group" aria-label="Parity testing levels">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Input parity — same bytes: crop/resize/normalise/camera ordering/timestamp alignment</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Intermediate parity — feature maps at backbone output and neck/FPN output</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Output parity — box xyzwlh, class logits, lane heatmap, occupancy grid</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Task-metric parity — recall/AP on 500-frame parity set: PT vs ONNX vs TRT</span></li>
  </ol>
</div>

```python
PARITY_TOLERANCES = {
    # Runtime     Output type               Max absolute err    Acceptable recall drop
    "onnx":   {"boxes_xyz":      1e-4,    "class_scores":  1e-4,   "recall_delta": 0.002},
    "trt_fp16": {"boxes_xyz":    5e-3,    "class_scores":  5e-3,   "recall_delta": 0.005},
    "trt_int8": {"boxes_xyz":    2e-2,    "class_scores":  1e-2,   "recall_delta": 0.010},
}

def run_parity_suite(pt_model, onnx_session, trt_engine, val_frames):
    results = {"onnx": [], "trt_fp16": [], "trt_int8": []}
    for frame in val_frames:
        pt_out  = run_pytorch(pt_model, frame)
        ort_out = run_onnxruntime(onnx_session, frame)
        trt_out = run_tensorrt(trt_engine, frame)

        for runtime, out in [("onnx", ort_out), ("trt_fp16", trt_out)]:
            for key in ["boxes_xyz", "class_scores"]:
                err = np.abs(pt_out[key] - out[key]).max()
                results[runtime].append({"key": key, "max_err": err})

    # Flag any frame where error exceeds tolerance
    for runtime, tol in PARITY_TOLERANCES.items():
        violations = [r for r in results[runtime]
                      if r["max_err"] > tol[r["key"]]]
        if violations:
            print(f"PARITY VIOLATION in {runtime}: {len(violations)} frames")
            for v in violations[:5]:
                print(f"  {v}")
```

**Example parity failures and their causes:**

| Failure | What you see | Root cause |
|---|---|---|
| Small distant vehicles disappear in INT8 | Score drops below threshold post-quantisation | Weak activations rounded to zero; fix: FP16 for detection head |
| Score calibration shifts | All scores ±0.05 in TRT vs PT | Sigmoid computed differently in TRT; fix: layer precision override |
| Lane boundaries noisier at night | Lane heatmap more noisy in FP16 | Low-light features have smaller activation values; fix: keep lane head FP32 |
| Temporal instability increases | Box jitter higher in TRT | Batch norm statistics not properly folded at export; fix: re-export with `torch.jit.trace` and explicit BN folding |

---

## Stage H — End-to-End Stack Integration & Profiling
{: #stage-h}

### Pipeline Bottlenecks
{: #pipeline-bottlenecks}

A model that benchmarks at 20ms in isolation often takes 45ms in the full autonomy pipeline. Reasons:

```
Isolated benchmark:  [inference 20ms]

Full pipeline:
[sensor sync 2ms] → [CPU decode 4ms] → [H2D copy 3ms] → [inference 20ms]
→ [D2H copy 2ms] → [NMS CPU 5ms] → [serialise to tracker 4ms]
= 40ms, not 20ms

Plus: contention with concurrently running models (radar, lidar segmentation)
      can add another 5–10ms of latency
```

**Common full-pipeline bottlenecks:**

| Bottleneck | Detection | Fix |
|---|---|---|
| **CPU preprocessing stall** | GPU idle while CPU decodes/resizes images | Move preprocessing to GPU (CUDA kernels, DALI), use pinned memory + async H2D |
| **Synchronous H2D copies** | `cudaMemcpy` (blocking) before inference | Use `cudaMemcpyAsync` + streams; overlap with previous frame postprocessing |
| **NMS on CPU** | GPU finishes inference, then waits for CPU NMS to finish | Move NMS to GPU (CUDA NMS plugin or TRT plugin) |
| **Model contention** | Multiple models fighting for GPU SM time | Schedule models to different CUDA streams; use CUDA MPS for better sharing |
| **Memory pressure** | Frame drops under load | Profile `nvidia-smi` memory; reduce model precision or batch size |

### Profiling Tools
{: #profiling-tools}

**PyTorch Profiler — training and Python-level inference:**

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
) as prof:
    for batch in val_loader:
        model(batch["images"].cuda())

# Print the most expensive CPU and CUDA ops
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

**Nsight Systems — full timeline view:**

```bash
# Profile the full inference pipeline including host-device copies and kernel gaps
nsys profile \
    --trace cuda,nvtx,osrt \
    --output profile_inference \
    python run_inference.py

# Open report in Nsight Systems GUI
nsys-ui profile_inference.nsys-rep
```

In the Nsight Systems timeline, look for:
- **Gaps between CUDA kernels** → CPU is stalling the pipeline; add NVTX annotations to find which Python code is responsible
- **Host-device copies during inference** → should be zero if async setup is correct
- **Multiple streams** → confirm preprocessing and inference are truly overlapped

**Nsight Compute — kernel-level GPU utilisation:**

```bash
# Profile a single kernel in depth (use after Nsight Systems identifies the slow kernel)
ncu \
    --metrics sm__throughput.avg_pct_of_peak_sustained_elapsed,\
              l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
              sm__warps_active.avg_pct_of_peak_sustained_active \
    python run_inference.py
```

**CUDA events — lightweight in-process timing:**

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event   = torch.cuda.Event(enable_timing=True)

start_event.record()
outputs = trt_engine(inputs)
end_event.record()

torch.cuda.synchronize()   # wait for GPU to finish
elapsed_ms = start_event.elapsed_time(end_event)
print(f"TRT inference: {elapsed_ms:.2f}ms")
```

**Interview question:** Your isolated TRT benchmark says 22ms. In the full stack, P95 latency is 47ms. Walk through how you would diagnose where the 25ms of overhead is.

> **Answer:** I'd instrument the pipeline with CUDA events at each stage boundary — sensor sync, H2D copy start/end, inference start/end, D2H copy start/end, NMS start/end, output write. First pass: print median and P95 for each stage across 1000 frames. This immediately shows which stage has the budget — e.g., H2D copy is 8ms instead of 2ms. If H2D is the culprit: check if it's using `cudaMemcpyAsync` with pinned memory (should be) vs `cudaMemcpy` (blocks host thread). If preprocessing is the culprit: use Nsight Systems to see if the CPU decode thread is actually running in parallel with the GPU or serialised. If the inference itself is longer in-stack than in isolation: there's GPU contention with another model — use `nvidia-smi dmon` or DCGM to observe SM utilisation over time and see if another workload is running concurrently. After identifying the bottleneck, fix it in isolation (e.g., move NMS to GPU), re-benchmark, and confirm end-to-end P95 drops accordingly. Never guess where time is going — measure first.

---

## Stage I — HIL and Log Replay Validation
{: #stage-i}

HIL (hardware-in-the-loop) is the closest you can get to real-world validation without putting the truck on the road.

**Validation levels:**

<div class="post-flow" role="group" aria-label="HIL validation levels">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Tensor-level parity — run same log frames through onboard hardware; compare output tensors to reference</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Task-level metrics — mAP, recall, lane F1 on log replay; compare to offline PyTorch baseline</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Scenario behaviour — specific scenario suites: cone detection in work zones, stopped trucks, cut-ins</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Downstream planner impact — run tracker + predictor + planner on HIL outputs; check comfort/safety metrics</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Gate review — all metrics must pass acceptance criteria before fleet deployment is approved</span></li>
  </ol>
</div>

**Log replay test suite design:**

```python
HIL_TEST_SUITE = {
    # Each test: a set of log segments + expected behaviour

    "stopped_truck_over_hill": {
        "description": "Truck appears from behind a curve at 60mph relative closing speed",
        "segments": ["log_id_0041", "log_id_0088", "log_id_0201"],
        "assertions": [
            {"type": "recall", "class": "truck", "distance": (0, 80), "min_recall": 0.92},
            {"type": "latency_to_detect", "class": "truck", "max_ms": 200},  # 4 frames at 20Hz
        ],
    },

    "work_zone_cones": {
        "description": "Narrow merge point with cone-defined corridor",
        "segments": ["log_id_0315", "log_id_0316", "log_id_0317"],
        "assertions": [
            {"type": "recall", "class": "cone", "min_recall": 0.88},
            {"type": "no_false_negatives_in_corridor"},  # any miss means no drivable path
        ],
    },

    "night_pedestrian": {
        "description": "Pedestrian crossing highway off-ramp at night",
        "segments": ["log_id_0502", "log_id_0503"],
        "assertions": [
            {"type": "recall", "class": "pedestrian", "min_recall": 0.90},
        ],
    },

    "guardrail_fp_budget": {
        "description": "Long straight highway section with continuous guardrails",
        "segments": ["log_id_0600", "log_id_0601", "log_id_0602"],
        "assertions": [
            # False positives near guardrails must stay low — they spook the planner
            {"type": "fp_per_frame", "class": "vehicle", "region": "guardrail_zone", "max": 0.05},
        ],
    },
}
```

**What to look for beyond box metrics:**

```
Tracker stability:
  - ID switches per kilometer: does the tracker lose and re-acquire objects?
  - Track age distribution: are tracks being terminated prematurely?

Planner comfort:
  - Unnecessary decelerations (score drop above threshold triggers planner brake)
  - Lateral jitter (noisy lane boundary → steering wobble)
  - Intervention rate on simulated scenarios

Temporal consistency:
  - 3D box position variance per track ID over 1 second
  - Score variance for the same object over consecutive frames
```

**Interview question:** Log replay shows the new model is better on aggregate mAP (+2pp) but the safety team flags that it has more "object blinking" — tracks appearing and disappearing for 1–2 frames. How do you decide whether to deploy?

> **Answer:** Blinking is a tracker symptom — it means per-frame detection confidence is oscillating around the tracker's assignment threshold. Even if mAP improved, blinking is a downstream regression. I would: (1) measure the blink rate (ID switches/km) on the full log replay suite and compare to baseline — quantify the regression. (2) Root-cause: is it a score calibration shift (the model scores objects lower than the baseline, causing detections to intermittently fall below the tracker's threshold), or is it genuine temporal instability in the model outputs? (3) If score calibration: re-calibrate the deployment thresholds using Platt scaling or temperature scaling on the validation set. (4) If temporal instability: add box jitter metric to the training eval, add a temporal consistency loss term in training (penalise large frame-to-frame changes in box position for the same GT object), retrain. Do not deploy until blink rate matches or beats baseline — the planner behaviour regression is a safety concern even if offline metrics improved.

---

## Stage J — Rollout & Continuous Learning Loop
{: #stage-j}

Deployment is not the end of the loop — it is the beginning of data collection for the next model version.

<div class="post-flow" role="group" aria-label="Continuous learning loop">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Fleet deployment — new model on a canary fleet (10% of trucks)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Telemetry collection — log model outputs, interventions, near-misses, unusual scenarios</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Failure mining — cluster low-confidence detections, near-miss events, human takeovers by cause</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Hard example curation — pull log segments around failures, annotate, add to training</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Retrain — new model trained on expanded dataset including hard examples</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Regression gates — all acceptance criteria must pass before new version proceeds to HIL</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">HIL validation → canary rollout → full fleet — only if all gates pass</span></li>
  </ol>
</div>

**Canary rollout strategy:**

```python
ROLLOUT_STAGES = [
    {"fleet_fraction": 0.05,  "duration_days": 3,  "abort_if": "any_safety_critical_regression"},
    {"fleet_fraction": 0.15,  "duration_days": 5,  "abort_if": "intervention_rate_increases_2pct"},
    {"fleet_fraction": 0.40,  "duration_days": 7,  "abort_if": "any_regression_in_slice_metrics"},
    {"fleet_fraction": 1.00,  "duration_days": 0,  "abort_if": "same"},
]
```

**Failure mining pipeline:**

```python
def mine_hard_examples(fleet_logs, model_version: str):
    """
    Identify frames where the deployed model was uncertain or wrong.
    These are candidates for annotation and training data inclusion.
    """
    hard_examples = []
    for log in fleet_logs:
        for frame in log.frames:
            detections = frame.model_outputs[model_version]

            # Low-confidence detections that survived NMS (model is uncertain)
            uncertain = [d for d in detections if 0.3 < d.score < 0.6]

            # Frames near human takeover events (within 5 seconds)
            if frame.time_to_takeover < 5.0:
                hard_examples.append({"log_id": log.id, "frame_id": frame.id,
                                       "reason": "near_takeover"})

            # Retrospective: GT boxes (from offline lidar) not matched by model
            unmatched_gt = [g for g in frame.gt_boxes
                            if not any(iou(g, d) > 0.5 for d in detections)]
            if unmatched_gt:
                hard_examples.append({"log_id": log.id, "frame_id": frame.id,
                                       "reason": "false_negative",
                                       "classes": [g.cls for g in unmatched_gt]})

    return hard_examples
```

---

## Cross-Functional Interfaces
{: #cross-functional}

A deployment engineer is not an isolated systems person — they sit at the intersection of every team that touches the model's lifecycle. Understanding what each team needs from you, and what friction points arise, is as important as the technical skills.

<div class="post-flow" role="group" aria-label="Cross-functional touchpoints">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Perception — defines model architecture and training; you make it deployable</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Planning — consumes model outputs; defines what "good enough" means end-to-end</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Simulation — generates synthetic data and validates in closed-loop; needs model parity</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Autonomy Infrastructure — runs the fleet, manages hardware, operates the deployment pipeline</span></li>
  </ol>
</div>

---

### Working with Perception
{: #xf-perception}

**Their world:** Perception researchers iterate on model architecture, loss functions, and data pipelines. They measure success in mAP, IoU, recall curves — offline, on GPUs, in FP32, often with large batch sizes that would never exist on a truck.

**Your interface:**
- You receive a model checkpoint and a specification of its inputs/outputs. Your job is to make that checkpoint run on the target hardware within the latency budget, without losing quality.
- **Common friction:** the model was designed without deployment constraints in mind — dynamic shapes, custom ops, Python-level postprocessing baked into the forward pass. You often have to push back and request architectural changes *before* training completes, not after.

**What to ask before they start training:**
```
□ Are all ops in the ONNX opset you're targeting? (check against opset 17 op coverage)
□ Is NMS/postprocessing inside or outside the exported graph?
□ Are input shapes static or dynamic? (static = faster TRT build + inference)
□ Are there any autograd.Function custom ops? (need ONNX symbolics)
□ Does the model use F.interpolate? If so, with fixed align_corners?
□ What is the expected batch size and input resolution on the truck?
□ Is BN in eval mode at export time? (training mode BN → wrong inference behaviour)
```

**Negotiating architectural changes:** if the perception team uses deformable attention (common in BEV perception models like BEVFormer), you may need to write a custom TRT plugin to support it. The conversation is: "this op is not supported natively in TRT; here are three options: (A) we write a plugin — 2 weeks effort, (B) we replace with a TRT-compatible approximation — small quality delta, (C) we switch to a DETR-style head that TRT handles natively." Present trade-offs, let the team decide, document the decision.

**Handling "but it works in PyTorch" feedback:**
When parity check shows divergence, perception engineers often attribute it to TRT being "wrong." The correct response is: trace the divergence layer-by-layer. Write a script that compares intermediate feature maps between PyTorch and TRT at every major block (backbone, neck, head). The divergence will appear at a specific layer — and it will have a specific cause (FP16 overflow, unsupported op fallback, BN fold error). Never say "TRT is wrong" — instead say "here is the exact layer where outputs diverge, here is the cause, here is the fix."

---

### Working with Planning
{: #xf-planning}

**Their world:** Planning consumes structured outputs from perception — 3D boxes with velocities, lane boundaries, drivable area, predicted trajectories. They build motion planners and trajectory optimisers that assume the perception outputs are accurate, low-jitter, and consistent across frames.

**Why planning cares about deployment quality:**
- Small regressions in score calibration → planner's priority assignment breaks
- Increased box jitter → velocity estimates become noisy → planner makes uncomfortable maneuvers
- False positives near guardrails → planner plans over-conservatively (ghost objects blocking the path)
- Detection latency increase → planning has less time to react → safety margin shrinks

**Your interface:**
- Establish a **perception-planning contract**: a formal spec of the output format, coordinate frame, timestamp convention, confidence range, and latency guarantee.
- When you change the model (e.g., switch to INT8), run the full planner on the new model's outputs on the HIL suite and measure planner-level metrics, not just box-level.

```python
# Perception-Planning contract (interface spec)
PERCEPTION_OUTPUT_SPEC = {
    "coordinate_frame":   "ego_vehicle_at_camera_timestamp",
    "box_format":         "x_center, y_center, z_center, l, w, h, yaw",  # in meters
    "velocity_frame":     "ego_vehicle",     # velocity in m/s, ego-relative
    "confidence_range":   (0.0, 1.0),        # NOT logits — must be post-sigmoid
    "timestamp_convention": "middle_of_exposure",
    "max_output_latency_ms": 5,              # from inference complete to planning callback
    "nms_applied":        True,              # NMS done in postprocessing, not model
    "max_detections":     200,               # cap on number of boxes per frame
}
```

**What planning needs you to verify before deployment:**
1. Score calibration is preserved (ECE matches baseline)
2. Box jitter at P90 ≤ baseline (0.1m lateral, 0.3m longitudinal)
3. Velocity estimation error ≤ 0.5 m/s for objects with at least 5 frames of track history
4. Output latency (inference → planning callback) stays within contract

**Interview question:** Planning files a bug: "after deploying the new model, the truck is braking unnecessarily at highway on-ramps." How do you investigate?

> **Answer:** This is a planner symptom that could have multiple perception causes. I'd start by running the log replay of the on-ramp segment through both the old model and the new model and diffing the outputs. Hypotheses to check in order of likelihood: (1) **False positive ghost objects on the on-ramp** — new model creates a spurious vehicle detection in the merge zone that the planner reacts to. Check FP rate at on-ramp segments in the val set. (2) **Score calibration shift** — the new model scores vehicles higher in ambiguous merge scenarios, raising their perceived threat level. Check score distribution on merge scenarios specifically — not aggregate. (3) **Increased box jitter near merge zone** — noisy lane boundary prediction makes the drivable corridor narrower, causing the planner to brake to give more margin. Check lane boundary jitter on the affected log segments. (4) **Latency regression** — if inference is now 5ms slower, planning has 5ms less time to plan, and the safety margin shrinks, triggering a conservative decel. Check end-to-end pipeline P95 latency. Fix depends on root cause — but I'd never just roll back without identifying which of these is the actual driver.

---

### Working with Simulation
{: #xf-simulation}

**Their world:** Simulation runs closed-loop tests — the truck's software stack (including perception, prediction, planning) in a simulated environment, either with synthetic sensor data or with replayed real-world logs. They use simulation to test scenarios that are rare or unsafe to test on the road.

**Why simulation needs model parity:**
- If the simulation model is PyTorch FP32 but the truck runs TRT INT8, simulation results don't predict onboard behaviour — simulation is validating a different model than the one deployed.
- The simulation team needs to know: "when we run the deployed model in simulation, will we see the same outputs as the truck?"

**Your interface with simulation:**
1. **Export the simulation model from the same TRT engine as the truck** (or from ONNXRuntime with the same preprocessing). The goal is no discrepancy between simulation and production.
2. **Provide a deterministic inference wrapper** — given the same input bytes, always produces the same output (no non-determinism from model parallelism, no session-to-session variation).
3. **Expose a Python inference API** that simulation can call, backed by the same TRT engine:

```python
class OnboardModelSimWrapper:
    """
    Drop-in wrapper for simulation: same interface as the onboard C++ inference,
    backed by the production TRT engine (or ONNXRuntime for CPU-only sim nodes).
    """

    def __init__(self, engine_path: str, use_trt: bool = True):
        if use_trt:
            self.backend = TRTInferenceBackend(engine_path)
        else:
            # ONNXRuntime fallback for simulation nodes without GPU
            self.backend = ONNXRuntimeBackend(engine_path.replace(".engine", ".onnx"))

    def infer(self, camera_frames: dict, calibration: dict) -> dict:
        """
        Exact same call signature as the onboard C++ node.
        Returns: {"boxes": ..., "scores": ..., "labels": ..., "lanes": ...}
        """
        preprocessed = self._preprocess(camera_frames, calibration)
        raw_outputs = self.backend.run(preprocessed)
        return self._postprocess(raw_outputs)

    def _preprocess(self, frames, calibration):
        # IDENTICAL preprocessing as the onboard C++ pipeline
        # Any drift here breaks simulation-onboard parity
        return preprocess_for_inference(frames, calibration,
                                        **PREPROCESSING_SPEC)  # shared spec
```

4. **Track simulation-onboard parity as a metric**: run the same log through simulation and onboard; compare output tensors. Report parity degradation to the simulation team whenever a new model is released.

**Common failure: synthetic-to-real gap in simulation.** If simulation generates synthetic camera frames (rendered from a 3D world model), the appearance distribution can differ from real cameras (lighting model, lens distortion, sensor noise). The deployed model trained on real data may perform differently on synthetic frames. Work with the simulation team to add domain randomisation in the renderer and validate that the gap is within an acceptable bound before using simulation results to gate deployment.

---

### Working with Autonomy Infrastructure
{: #xf-infra}

**Their world:** Autonomy infrastructure manages the fleet of trucks, the onboard compute hardware, the OTA (over-the-air) software update system, the data collection pipeline, and the CI/CD infrastructure for deploying software to vehicles.

**Your interface:**

| Task | What you provide | What infra provides |
|---|---|---|
| **Model deployment** | TRT engine binary + preprocessing spec + output contract | OTA packaging, signing, rollout orchestration |
| **Hardware targeting** | Latency benchmarks per hardware SKU (Orin, Drive AGX) | Target hardware specs, CUDA/TRT version constraints |
| **Data collection** | Log segment tagging spec (which scenarios to keep) | Petabyte-scale storage, log ingest pipeline, tagging API |
| **CI/CD integration** | Acceptance test suite, quality gates, parity test scripts | Test runner infrastructure, result dashboards, alert routing |
| **Versioning** | Model artifact + engine binary + config hash | Artifact registry, version → truck mapping, rollback tooling |

**Hardware targeting — why it matters:** Kodiak trucks may run on different compute modules across the fleet (older Drive AGX Pegasus vs newer Orin). The TRT engine is hardware-specific — an Orin engine will not run on Pegasus. You must maintain separate engines per hardware SKU, and the parity and latency gates must pass on all supported hardware versions.

```bash
# Build engines for all supported hardware SKUs
for SKU in orin_32gb pegasus_16gb; do
    trtexec \
        --onnx=model_simplified.onnx \
        --saveEngine=model_${SKU}.engine \
        --fp16 \
        --shapes=images:1x6x3x900x1600 \
        --workspace=4096 \
        --exportTiming=timing_${SKU}.json \
        --verbose
    echo "Engine for ${SKU}: $(du -sh model_${SKU}.engine)"
done
```

**OTA deployment constraints:** when deploying a new TRT engine OTA to a truck that may be on the road:
- The engine swap must be **atomic** — never a state where part of the old model and part of the new model are in memory
- The truck must be able to **fall back** to the previous engine if the new one fails health checks on startup
- The engine binary must be **cryptographically signed** to prevent tampering

**Interview question:** Infra tells you that the new Orin hardware has 40% more tensor core throughput than the old AGX. Your model already hits the latency budget on AGX. Should you just ship the same engine, or do something different for Orin?

> **Answer:** Don't just ship the same engine. TRT engines are hardware-specific — the AGX engine is not valid on Orin at all (different CUDA compute capability, different tensor core instructions). You need to rebuild the engine on Orin. But more importantly, if Orin has 40% more throughput, it means you have headroom — you can use that headroom to improve quality rather than just run faster. Options: (1) Enable higher-precision layers (FP32 instead of FP16 for sensitive heads) within the same latency budget. (2) Run a slightly larger model or higher-resolution input that was previously latency-constrained. (3) Increase `ef_construction` in any ANN-style search step. The right framing: the latency budget on each hardware SKU defines a quality envelope — use the full envelope on each SKU to maximize perception quality.

---

## Scalable ML Infrastructure & Model CI/CD
{: #ml-infra}

The JD explicitly asks for this: *"scalable AI infrastructure that supports continuous learning and deployment across Kodiak's fleet."* This is not just about one model — it is about building the machinery that makes deploying any model fast, reliable, and safe.

---

### Model CI/CD Pipeline
{: #model-cicd}

Every model version goes through the same automated pipeline. No human makes a deployment decision — they only set the gates and approve exceptions.

<div class="post-flow" role="group" aria-label="Model CI/CD pipeline">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">1. Training completes — checkpoint registered in artifact registry with metadata hash</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">2. Offline evaluation — mAP/recall/slice metrics computed against frozen eval set; gate check</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">3. Export pipeline — ONNX export → onnxsim → ONNX parity check (gate: max err < 1e-4)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">4. TRT build — per-SKU engine build; latency benchmark; latency gate check</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">5. Parity suite — PyTorch vs ORT vs TRT on 1000 frames; per-class recall gate check</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">6. HIL validation — scenario suite on HIL rig; planner-impact metrics; gate check</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">7. Canary deploy — 5% fleet; monitor intervention rate and blink rate for 72 hours</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">8. Full fleet deploy — OTA rollout; automated rollback if any gate violation detected</span></li>
  </ol>
</div>

**Gate failures are blocking, not advisory.** If step 2 fails (offline eval regression), the pipeline stops. The model is not exported. A failing gate generates a JIRA ticket assigned to the model owner, with the gate diff attached. No manual override — any exception requires sign-off from two leads and a documented justification.

**Pipeline implementation (simplified):**

```python
# Model CI/CD orchestration using Prefect/Airflow-style DAG
class ModelDeploymentPipeline:

    def run(self, checkpoint_id: str) -> DeploymentResult:
        artifact = self.registry.get(checkpoint_id)

        # Stage 1: Offline evaluation
        eval_result = self.run_offline_eval(artifact)
        if not eval_result.passes_gates(ACCEPTANCE_GATES):
            return DeploymentResult.blocked("offline_eval", eval_result.diff())

        # Stage 2: Export
        onnx_path = self.export_to_onnx(artifact)
        onnx_parity = self.check_onnx_parity(artifact, onnx_path)
        if not onnx_parity.passes(max_abs_err=1e-4):
            return DeploymentResult.blocked("onnx_parity", onnx_parity.failures())

        # Stage 3: TRT build per SKU
        engines = {}
        for sku in SUPPORTED_HARDWARE_SKUS:
            engine_path = self.build_trt_engine(onnx_path, sku)
            latency = self.benchmark_engine(engine_path, sku)
            if latency.p95_ms > LATENCY_BUDGET_MS[sku]:
                return DeploymentResult.blocked(f"latency_{sku}", latency)
            engines[sku] = engine_path

        # Stage 4: Full parity suite
        parity = self.run_parity_suite(artifact, engines)
        if not parity.passes_recall_gates():
            return DeploymentResult.blocked("trt_parity", parity.regressions())

        # Stage 5: HIL (async, runs on HIL rig — may take 2–4 hours)
        hil_result = self.run_hil_suite(engines)
        if not hil_result.passes():
            return DeploymentResult.blocked("hil", hil_result.failures())

        # Stage 6: Register for canary rollout
        release = self.registry.create_release(checkpoint_id, engines, eval_result)
        self.rollout_manager.start_canary(release, fraction=0.05, duration_hours=72)

        return DeploymentResult.success(release)
```

---

### Artifact Registry & Versioning
{: #artifact-registry}

Every artifact — checkpoint, ONNX file, TRT engine, calibration cache, evaluation result — is versioned and immutable.

```python
# Artifact schema: everything needed to reproduce a deployment
ARTIFACT_SCHEMA = {
    "checkpoint_id":      "sha256 of model weights",
    "training_config":    "hash of training config YAML",
    "dataset_version":    "hash of training dataset manifest",
    "onnx_hash":          "sha256 of ONNX file",
    "engines": {
        "orin_32gb":      "sha256 of TRT engine binary",
        "pegasus_16gb":   "sha256 of TRT engine binary",
    },
    "calibration_cache":  "sha256 of INT8 calibration cache (if INT8)",
    "eval_results":       "pointer to eval result JSON in object store",
    "parity_results":     "pointer to parity test JSON in object store",
    "preprocessing_spec": "hash of preprocessing config",  # CRITICAL: ties preprocessing to model
    "output_contract":    "hash of output contract spec",
    "created_at":         "ISO 8601 timestamp",
    "deployed_to_fleet":  "list of truck IDs currently running this model version",
}
```

**Why hash the preprocessing spec?** The model and the preprocessing are co-dependent. If you deploy a new model but use the old preprocessing (even by mistake), you get silent performance degradation — no error, just wrong results. Tying the preprocessing hash to the model artifact means any mismatch is caught at deploy time.

**Truck-to-version mapping:** every truck in the fleet tracks which model version it is running. This enables:
- Per-truck rollback: if one truck shows anomalous behaviour, roll back just that truck
- Fleet-wide version distribution dashboard: what fraction of trucks are on each version
- Cohort analysis: compare intervention rates between trucks on v1.2 vs v1.3

---

### Deployment Gates as Code
{: #deployment-gates}

Gates are code, not spreadsheets. They live in the same repo as the model code, versioned with it.

```python
# gates.py — checked into model repo, reviewed in PRs like any code change
from dataclasses import dataclass
from typing import Optional

@dataclass
class QualityGate:
    metric: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    max_regression_vs_baseline: Optional[float] = None  # max allowed drop vs current fleet model

DEPLOYMENT_GATES_V2 = [
    # Aggregate quality
    QualityGate("mAP_50",                    min_value=0.62),
    QualityGate("mAP_50",                    max_regression_vs_baseline=-0.005),  # no > 0.5pp drop

    # Safety-critical recall — stricter
    QualityGate("recall_pedestrian",         min_value=0.85, max_regression_vs_baseline=-0.003),
    QualityGate("recall_stopped_truck",      min_value=0.88, max_regression_vs_baseline=-0.003),
    QualityGate("recall_cone",               min_value=0.80, max_regression_vs_baseline=-0.005),

    # Scenario slices
    QualityGate("AP_nighttime",              max_regression_vs_baseline=-0.010),
    QualityGate("AP_rain",                   max_regression_vs_baseline=-0.010),

    # Calibration
    QualityGate("calibration_ECE",           max_value=0.05),

    # Parity
    QualityGate("trt_fp16_max_abs_err",      max_value=0.005),
    QualityGate("trt_fp16_recall_delta_ped", max_value=0.005),   # absolute delta

    # Latency
    QualityGate("trt_latency_p95_orin_ms",   max_value=25.0),
    QualityGate("pipeline_latency_p95_ms",   max_value=45.0),

    # Downstream (planner-impact)
    QualityGate("tracker_id_switch_per_km",  max_regression_vs_baseline=-0.05),
    QualityGate("unnecessary_decel_per_km",  max_regression_vs_baseline=-0.02),
]

def evaluate_gates(results: dict, baseline_results: dict) -> list[dict]:
    """Evaluate all gates, return list of violations with details."""
    violations = []
    for gate in DEPLOYMENT_GATES_V2:
        value = results.get(gate.metric)
        baseline = baseline_results.get(gate.metric)
        if value is None:
            violations.append({"gate": gate.metric, "reason": "metric_missing"})
            continue
        if gate.min_value is not None and value < gate.min_value:
            violations.append({"gate": gate.metric, "value": value,
                                "min_required": gate.min_value})
        if gate.max_value is not None and value > gate.max_value:
            violations.append({"gate": gate.metric, "value": value,
                                "max_allowed": gate.max_value})
        if gate.max_regression_vs_baseline is not None and baseline is not None:
            delta = value - baseline
            if delta < gate.max_regression_vs_baseline:
                violations.append({"gate": gate.metric, "value": value,
                                    "baseline": baseline, "delta": delta,
                                    "max_allowed_delta": gate.max_regression_vs_baseline})
    return violations
```

**Gates as PRs:** when the perception team proposes a new model that would fail an existing gate (e.g., due to architecture trade-off), the gate must be updated in the same PR as the model change. This forces an explicit, reviewed decision — not a silent threshold change.

---

### Train-Time Profiling Deep Dive
{: #train-profiling}

The JD specifically calls out *"profiling tooling to improve train time."* Here is the systematic workflow:

**Step 1: Establish MFU (Model FLOP Utilisation)**

MFU tells you how much of the theoretical hardware throughput you're using. A healthy training job should hit 40–60% MFU on A100s with a well-tuned setup.

```python
def compute_mfu(
    model_params: int,
    seq_len: int,
    batch_size: int,
    step_time_sec: float,
    gpu_flops_per_sec: float = 312e12,   # A100 FP16 tensor core peak
) -> float:
    """
    MFU = actual throughput / theoretical peak
    For a transformer with 'model_params' parameters:
    FLOPs per forward+backward ≈ 6 × params × seq_len × batch_size
    (6 = 2 forward + 4 backward, approximate)
    """
    flops_per_step = 6 * model_params * seq_len * batch_size
    actual_flops_per_sec = flops_per_step / step_time_sec
    return actual_flops_per_sec / gpu_flops_per_sec

# If MFU < 30%: serious inefficiency — likely DataLoader or communication bound
# If MFU 30–50%: acceptable for most CV models (memory-bandwidth bottlenecks)
# If MFU > 50%: well-optimised
```

**Step 2: Identify the bottleneck tier**

```bash
# While training, watch GPU utilisation in real time
watch -n 1 nvidia-smi dmon -s u

# Also watch CPU util and I/O wait
htop   # look for: are DataLoader workers CPU-maxed? is there I/O wait?

# Network utilisation (for multi-node training — is AllReduce saturating IB?)
iftop -i ib0   # or: ibstat, nload on InfiniBand interface
```

**Step 3: Profile with PyTorch Profiler + Chrome trace**

```python
# Detailed per-op profile with NCCL (AllReduce) visibility
from torch.profiler import profile, ProfilerActivity, schedule

# Wait 5 steps, warm up 2, then profile 3 steps
prof_schedule = schedule(wait=5, warmup=2, active=3)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=tensorboard_trace_handler("./runs/train_profile"),
    record_shapes=True,
    profile_memory=True,
    with_stack=False,   # True = slower but shows Python call stacks
) as prof:
    for step, batch in enumerate(loader):
        train_step(model, optimizer, batch)
        prof.step()
        if step >= 10:
            break
```

Open `tensorboard --logdir runs/train_profile` and look at the **Trace View**:
- Each row is a CUDA stream or CPU thread
- Look for **gaps in the GPU trace** (GPU idle = CPU is stalling it)
- Look for `nccl:all_reduce` taking more than 20% of step time (communication-bound)
- Look for `DataLoader_0` thread showing long blocking reads (storage I/O bound)

**Step 4: Bottleneck-specific fixes**

```
BOTTLENECK: DataLoader is slow (visible as gap between GPU steps in trace)
─────────────────────────────────────────────────────────
Symptom: PyTorch profiler shows long gaps labelled "DataLoader worker"
Fix 1: increase num_workers (rule: 4 × GPU count)
Fix 2: use pin_memory=True + non_blocking=True on .to(device)
Fix 3: switch from JPEG file-per-sample to WebDataset (tar shards) —
        eliminates random-access disk seeks; sequential reads = 10× faster
Fix 4: cache preprocessed tensors (resize + normalize already done) to SSD
Fix 5: for camera frames: use NVJPEG (GPU JPEG decoder) instead of PIL

BOTTLENECK: NCCL AllReduce takes > 20% of step time
─────────────────────────────────────────────────────────
Symptom: nccl:all_reduce shows in profiler trace, NCCL timeline long
Fix 1: gradient compression (PowerSGD, 1-bit Adam) — reduces comm volume 10–50×
Fix 2: reduce AllReduce frequency: accumulate N gradients before syncing
Fix 3: overlap gradient comm with backward pass:
       DDP does this automatically if find_unused_parameters=False
       Verify with profiler: backward and all_reduce should overlap
Fix 4: check InfiniBand config — NCCL_IB_GID_INDEX, NCCL_SOCKET_IFNAME
       Misconfigured IB can cut bandwidth 5×

BOTTLENECK: Low GPU utilisation despite fast DataLoader (compute wasteful)
─────────────────────────────────────────────────────────
Symptom: high SM util but low MFU; PyTorch profiler shows many small kernels
Fix 1: torch.compile (PyTorch 2.0+) — fuses ops, eliminates Python overhead
       model = torch.compile(model, mode="max-autotune")
Fix 2: increase batch size — amortises kernel launch overhead
Fix 3: reduce Python overhead in training loop (avoid Python list comprehensions
       on GPU tensors, avoid .item() calls mid-loop, avoid unnecessary .cpu() calls)
Fix 4: use channels_last memory format for CNN models:
       model = model.to(memory_format=torch.channels_last)
       images = images.to(memory_format=torch.channels_last)
       → improves cache utilisation for conv layers
```

**Step 5: NCCL communication trace (multi-node)**

```bash
# Enable NCCL debug output to see communication pattern
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL \
    torchrun --nproc_per_node=8 train.py 2>&1 | grep -E "NCCL|ring|tree"

# Profile NCCL specifically with nsys
nsys profile \
    --trace cuda,nvtx,nccl \
    --output nccl_profile \
    torchrun --nproc_per_node=8 train.py
```

In the Nsight Systems timeline with NCCL tracing, verify:
- AllReduce is running **during** the backward pass (overlapped), not **after** (sequential)
- The ring-allreduce pattern shows all GPUs contributing simultaneously
- No GPU is significantly behind others at the AllReduce barrier (load imbalance)

**torch.compile — when it helps and when it doesn't:**

```python
# Best for: transformer backbones, attention, FFNs — lots of elementwise ops
# Worst for: models with Python-side control flow (if/else on tensor values),
#            dynamic shapes that change between steps

# Modes:
model = torch.compile(model, mode="default")       # ~10% speedup, fast compile
model = torch.compile(model, mode="reduce-overhead") # ~20% speedup, more aggressive
model = torch.compile(model, mode="max-autotune")    # best throughput, slow compile (minutes)

# Check what torch.compile is actually doing
import torch._dynamo
torch._dynamo.explain(model)(dummy_input)  # shows which parts were captured vs not
```

**Summary: train-time profiling workflow**

```
1. Measure MFU → sets the ceiling on what's achievable
2. Watch GPU/CPU/I/O util → identifies which tier the bottleneck is in
3. PyTorch profiler trace → pinpoints the specific op/phase
4. Fix the worst bottleneck
5. Re-measure MFU → confirm improvement
6. Repeat until MFU is within 10% of target or the bottleneck is unavoidable
```

> **Interview question:** Your training job on 64 A100s is running at 22% MFU. Your expected MFU for this model size on this hardware is 45%. Walk through how you would find and fix the bottleneck.
>
> **Answer:** 22% vs 45% expected MFU means roughly half the compute is being wasted. Step 1: check `nvidia-smi dmon` on all 64 GPUs simultaneously — are all GPUs showing similar utilisation, or are some 90% and others 10%? If uneven, you have load imbalance (some GPUs have heavier batches than others — fix with dynamic batching or better dataset sharding). Step 2: if all GPUs are uniformly low, run PyTorch Profiler for 5 steps. Look at the Trace View in TensorBoard. If you see long gaps between CUDA kernels labelled `DataLoader_X`, the bottleneck is data loading — fix with more workers, WebDataset sharding, or NVJPEG decoding. Step 3: if the DataLoader is fast but you see `nccl:all_reduce` taking 40% of the step time and it is *not* overlapping with the backward pass, the bottleneck is communication. Check that DDP has `find_unused_parameters=False` (enables gradient bucketing + overlap). Check NCCL config for IB. Step 4: if the GPU is running full but MFU is still low, check for excessive small-kernel launches — run `nsys profile` and look for many short (<10μs) kernels. Apply `torch.compile(mode="max-autotune")` to fuse these. In my experience, 22% → 45% is almost always DataLoader + NCCL overlap being fixed together.

---

## Topic Tree: What to Study
{: #topic-tree}

### Root 1 — Autonomy Perception Basics
{: #root-perception}

| Topic | What to know | Depth |
|---|---|---|
| **Object detection** | Anchor-based (FCOS, CenterPoint) vs anchor-free; NMS variants | Deep |
| **BEV perception** | LSS (Lift-Splat-Shoot), BEVDet, BEVFormer — lifting 2D → BEV | Deep |
| **Segmentation** | Semantic (drivable area, lane) vs instance; BEV seg | Medium |
| **Multi-object tracking** | Hungarian matching, Kalman filter, byte tracker | Medium |
| **Temporal modelling** | 3D convolutions, ConvLSTM, deformable attention over time | Medium |
| **Sensor fusion** | Camera + LiDAR fusion at feature level vs late fusion | Medium |
| **Occupancy prediction** | Voxel grid occupancy vs BEV occupancy, 4D occupancy | Medium |
| **Prediction / forecasting** | Trajectory prediction, multimodal futures, MTR, Wayformer | Light |

**BEV perception key concept — LSS (Lift-Splat-Shoot):**

The core challenge: map 2D image features (pixels) to 3D BEV space using only camera intrinsics (no LiDAR).

```
1. LIFT: for each 2D pixel, predict a depth distribution D ∈ R^d
   Feature map (H×W×C) × depth bins (D) → frustum features (H×W×D×C)

2. SPLAT: for each frustum point, compute 3D position using camera extrinsics
   Project frustum points into BEV grid using known camera poses
   Scatter/pool features into BEV grid cells

3. SHOOT: run BEV backbone on the pooled BEV features
   Output: BEV detection heatmaps, lane heatmaps, drivable area
```

### Root 2 — Deployment Fundamentals
{: #root-deployment}

**PTQ vs QAT:**

| | PTQ (Post-Training Quantisation) | QAT (Quantisation-Aware Training) |
|---|---|---|
| **How** | Calibrate scale factors after training | Simulate quantisation during training |
| **Cost** | Fast (hours) | Slow (full retrain or fine-tune) |
| **Quality** | Good for most layers; struggles with outlier activations | Better quality, especially for sensitive layers |
| **When to use** | Start here; if recall drop > 1pp, move to QAT | Safety-critical classes, tight latency AND quality |

**Pruning: when it actually helps latency:**

Unstructured sparsity (setting individual weights to zero) does **not** improve latency on dense GPU hardware unless sparsity > 90%+ and hardware supports sparse matrix kernels (Ampere `SpMMA`). Structured pruning (removing entire channels, heads, or blocks) always helps because it reduces the shape of the computation.

```python
# Structured channel pruning: remove channels with lowest L1 norm
import torch.nn.utils.prune as prune

for module_name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        # Prune 20% of output channels with smallest L1 weight norm
        prune.ln_structured(module, name="weight", amount=0.20, n=1, dim=0)
        # Remove the pruning reparametrisation (make it permanent)
        prune.remove(module, "weight")
```

### Root 3 — GPU & System Optimisation
{: #root-gpu}

**Roofline analysis for inference:**

At batch=1 decode, the model is almost certainly **memory-bandwidth-bound**, not compute-bound. The roofline:

```
A100 peak FLOPs:     312 TFLOP/s (FP16)
A100 memory BW:      2 TB/s
Roofline crossover:  312e12 / 2e12 = 156 FLOP/byte

Model FP16 at batch=1:
  Weight bytes per forward:  param_count × 2 bytes
  FLOPs per forward:         2 × param_count × seq_len (approx)
  Arithmetic intensity:      ≈ seq_len FLOP/byte

For batch=1, short input: intensity << 156 → memory-bandwidth-bound
Fix: larger batch, quantisation (fewer bytes), model compression
```

**CUDA stream concurrency model:**

```
Default stream (stream 0): all ops serialised
Named streams: ops on different streams can overlap if no data dependency

Producer-consumer pattern:
  stream_A: preprocess frame N+1
  stream_B: infer on frame N       (waits for N+1 preprocess on stream_A)
  stream_C: postprocess frame N-1  (waits for N infer on stream_B)
  → maximum throughput: all three stages execute in parallel
```

### Root 4 — Distributed Training
{: #root-training}

**Communication complexity of AllReduce variants:**

| Algorithm | Latency | Bandwidth | When to use |
|---|---|---|---|
| **Ring AllReduce** | O(p) | O(1) optimal | Standard DDP; N GPUs on a ring |
| **Tree AllReduce** | O(log p) | O(log p) bandwidth | Many nodes, high latency network |
| **Recursive Halving/Doubling** | O(log p) | O(1) | MPI default; good for all-pairs |
| **Hierarchical** | O(log p) + local O(1) | Optimal | NVLink within node + IB across nodes |

**ZeRO Stage 3** shards parameters across data-parallel workers — each GPU only holds 1/N of the parameters, gradients, and optimizer states. The communication cost is the same as DDP (one AllReduce per step) but memory per GPU scales as 1/N, enabling models otherwise too large for DDP.

### Root 5 — Validation & Parity
{: #root-validation}

**Offline vs deployed parity checklist:**

```
Preprocessing parity:
  □ Same image crop region (verify with pixel-level diff on raw inputs)
  □ Same resize algorithm and align_corners setting
  □ Same normalisation (mean/std, applied in same order)
  □ Same camera ordering if multi-camera model
  □ Same timestamp alignment if temporal model

Numerical parity:
  □ Run 1000 validation frames through all three runtimes
  □ Log max/mean absolute error per output tensor per runtime
  □ Check recall/AP delta per class per runtime (not just aggregate)
  □ Verify no "catastrophic" frames (outlier errors >> average)

Calibration parity:
  □ ECE before and after conversion
  □ Score distribution histograms (are all scores shifted uniformly or unevenly?)
```

### Root 6 — Safety & Robustness Mindset
{: #root-safety}

**The safety evaluation mindset:** for safety-critical classes, recall matters more than precision. A false negative (missing a pedestrian) is far worse than a false positive (spurious box). Evaluate at multiple operating points:

```python
# Plot recall vs false-positives-per-frame curve (FROC curve)
# for each safety-critical class

def froc_curve(detections, gt_boxes, class_id, fp_per_frame_range):
    recalls = []
    for fp_budget in fp_per_frame_range:
        # Find score threshold that achieves exactly fp_budget FP/frame
        threshold = binary_search_threshold(detections, fp_budget)
        recall = compute_recall_at_threshold(detections, gt_boxes, class_id, threshold)
        recalls.append(recall)
    return recalls

# Key operating points:
# recall @ 0.1 FP/frame: very tight precision — safety-critical scenario
# recall @ 1.0 FP/frame: loose precision — max recall for background monitoring
```

---

## Interview Prep: Question Buckets
{: #interview-prep}

**The model answer structure:** the strongest answers follow this flow: *identify constraints → propose staged approach → name specific tools/metrics at each stage → identify failure modes and how to catch them early*.

---

**Q: How would you deploy a BEV perception model from PyTorch to an onboard truck?**

> Staged: (1) Freeze model and freeze preprocessing assumptions. (2) Export to ONNX with static shapes, opset 17, `do_constant_folding=True`. Verify with onnxsim + ONNXRuntime parity check (max abs error < 1e-4). (3) Build TensorRT FP16 engine with static input shapes. Run the parity suite: per-class recall must not drop > 0.5pp. (4) Profile in the full pipeline — not isolated — using CUDA events at each stage boundary. Find and fix any CPU preprocessing stalls, sync H2D copies, or CPU NMS bottlenecks. (5) Run HIL validation on the safety-critical scenario suite (stopped truck, work zone, night pedestrian). Gate: all HIL assertions must pass. (6) Canary rollout to 5% of fleet, monitoring blink rate, intervention rate, and per-slice recall. Expand rollout only if canary metrics are clean.

---

**Q: What can break during ONNX/TensorRT conversion?**

> Five categories: (1) **Unsupported ops** — custom autograd functions, ops added after ONNX opset, Python-level control flow. Fix: register ONNX symbolics or rewrite the op. (2) **Shape issues** — dynamic shapes cause TRT to build multiple tactics, performance cliff vs static shapes; shape inference failures break TRT build. Fix: export with static shapes; use `onnxsim`. (3) **Precision regression** — FP16 rounds small activations to zero, hurting small-object recall; INT8 calibration on non-representative data. Fix: per-layer precision overrides; representative calibration dataset. (4) **Preprocessing mismatch** — different `align_corners`, different normalisation. Fix: bake preprocessing into the graph or verify with pixel-level diff. (5) **Postprocessing** — NMS with variable-length outputs can't be expressed in static ONNX. Fix: move NMS out of the exported model.

---

**Q: How would you validate that FP16/INT8 did not hurt safety-critical behaviour?**

> Three levels: (1) **Box-level**: per-class recall comparison between PyTorch baseline and converted model on a 1000-frame parity set; flag any class with > 0.5pp drop. (2) **Calibration**: compare score ECE and score distribution histograms — a systematic shift means downstream thresholds need recalibration. (3) **Downstream**: run the full tracker + planner on both model outputs on the log replay suite; check ID switch rate, blink rate, unnecessary deceleration count. If (3) shows regression even if (1) and (2) pass, the model is not deployable. The safety-critical answer is always "validate end-to-end, not just the model in isolation."

---

**Q: Why can a model benchmark fast alone but be slow in the full autonomy stack?**

> Three reasons: (1) **Pipeline serialisation** — in isolation, you measure only inference. In the stack, the preprocessing stage must complete before inference can start. If preprocessing is synchronous CPU code, it adds 5–10ms of sequential latency that doesn't appear in isolated benchmarks. Fix: async H2D copies, GPU preprocessing, CUDA streams. (2) **GPU contention** — other models (lidar segmentation, radar processing) are running concurrently on the same GPU. Each model's kernels compete for SM time. Fix: measure with all stack components running simultaneously. (3) **Memory pressure** — when all models are running, the combined working set may exceed L2 cache capacity, causing more HBM bandwidth pressure. A model that is compute-bound in isolation becomes bandwidth-bound under memory pressure. Profiling with `ncu --metrics sm__throughput` in full-stack conditions reveals this.

---

**Q: How would you profile and optimise train time for a large perception model?**

> Step 1: establish baseline step time and GPU utilisation with `nvidia-smi dmon -s u`. If GPU utilisation < 70%, you are CPU-bound. Step 2: use PyTorch profiler with `ProfilerActivity.CPU` and `ProfilerActivity.CUDA` to find the top CPU and CUDA time consumers. Common culprits in order of frequency: (a) DataLoader I/O — fix with more workers, WebDataset, SSD caching; (b) CPU augmentations — fix with GPU augmentations via Kornia/DALI; (c) Host-device copies of annotations — fix with pinned memory and `non_blocking=True`; (d) Gradient sync (AllReduce) — if > 20% of step time, gradient accumulation or ZeRO to reduce sync frequency. Step 3: if compute-bound, profile with Nsight Compute to find the slow kernel. If MFU < 40%, there is likely kernel launch overhead — fuse ops with `torch.compile` or write a Triton kernel.

---

**Q: How would you identify edge cases from real-world fleet logs?**

> Three strategies: (1) **Prospective mining**: tag scenarios at log collection time using automated classifiers (detect rain from camera stats, detect construction zones from HD map overlap, detect cut-ins from ego-relative kinematics). Build a scenario index. (2) **Retrospective mining**: after inference, find frames where model confidence was low (0.3–0.6) for any class — these are uncertainty hotspots. Also find frames where a GT-annotated object (from offline lidar) was missed by the model (false negatives). Both signal hard cases. (3) **Intervention-driven mining**: any human safety driver intervention is a high-value signal. Pull the 10 seconds before the intervention event and run through an automated classifier to determine if perception was involved. These frames become the highest-priority annotation queue.

---

**Q: What does parity between offboard and onboard models actually mean, and why is it hard?**

> Parity means: for the same physical scene, the onboard model (TensorRT, INT8) produces outputs that are numerically close enough to the offline model (PyTorch, FP32) that downstream behaviour is identical. It is hard because: (1) Floating-point operations are not associative — reordering ops (as TRT does for kernel fusion) changes rounding. (2) Lower precision (FP16/INT8) introduces quantisation error that is not uniformly distributed across the output — it is worse for small activations, long-tail classes, and boundary cases. (3) Preprocessing is easy to drift — a one-pixel difference in the crop region, a different rounding mode for the resize, or a different pixel normalisation order all produce different feature maps from the same raw image. The way to manage it is: lock the preprocessing specification and test it with exact byte-level comparison; test the model numerically on a held-out parity set; and test downstream task metrics (recall, tracker stability) rather than just raw tensor error.

---

**Q: How would you design a continuous deployment loop for autonomy models?**

> The loop has five components: (1) **Instrumented deployment**: all production inferences log confidence distributions, frame timestamps, and any human intervention events. (2) **Failure mining pipeline**: automated jobs cluster low-confidence frames and near-intervention events by scenario type. (3) **Annotation pipeline**: mined frames enter an annotation queue; auto-labelled with lidar teacher, human QA for rare classes. New hard examples are added to the training dataset. (4) **Gated retrain**: retrained model must pass all acceptance gates (aggregate metrics, per-slice recall, parity gates, HIL suite) before entering rollout. No human in the loop for each gate — automated CI/CD for model deployment. (5) **Canary rollout**: deploy to 5% of fleet, monitor real-world metrics (intervention rate, blink rate) for 3 days, then expand. Any regression triggers automatic rollback. The key discipline: the gates are not negotiable. A model that improves mAP but introduces a recall regression on stopped trucks does not deploy, no matter how good the aggregate numbers look.
