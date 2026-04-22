---
title: "CNNs, VLMs, Diffusion, Video & World Models"
date: 2026-04-22
description: "A ground-up treatment of visual deep learning — convolutional networks, vision-language models, diffusion generative models, video models, and world models — with loss functions, training details, and a full VLM evaluation section."
tags: [cnn, vlm, diffusion, video-models, world-models, multimodal, generative-ai]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#cnns">Convolutional Neural Networks</a>
      <ul class="post-toc-sublist">
        <li><a href="#conv-operation">The Convolution Operation</a></li>
        <li><a href="#cnn-building-blocks">Building Blocks: Pooling, BN, Residual, Dilated, Transposed</a></li>
        <li><a href="#cnn-architectures">Key Backbone Architectures</a></li>
        <li><a href="#object-detection">Object Detection: R-CNN → YOLO → DETR</a></li>
        <li><a href="#segmentation">Segmentation: FCN, U-Net, DeepLab, Panoptic</a></li>
        <li><a href="#vae">Variational Autoencoder (VAE)</a></li>
        <li><a href="#cnn-losses">Loss Functions</a></li>
        <li><a href="#cnn-intuition">Intuition Checks: Hard Questions</a></li>
        <li><a href="#vision-transformers">From CNNs to Vision Transformers</a></li>
      </ul>
    </li>
    <li><a href="#vlms">Vision-Language Models</a>
      <ul class="post-toc-sublist">
        <li><a href="#vlm-evolution">Evolution: Captioning → CLIP → LLM-VLMs</a></li>
        <li><a href="#vlm-architecture">Architecture: Encoder, Projector, LLM</a></li>
        <li><a href="#vlm-training">Training Stages & Objectives</a></li>
        <li><a href="#vlm-losses">VLM Loss Functions</a></li>
        <li><a href="#vlm-models">Major Models</a></li>
        <li><a href="#vlm-evaluation">Evaluation</a></li>
        <li><a href="#vlm-failure">Failure Modes</a></li>
      </ul>
    </li>
    <li><a href="#diffusion">Diffusion Models</a>
      <ul class="post-toc-sublist">
        <li><a href="#diffusion-intuition">Intuition</a></li>
        <li><a href="#forward-process">Forward Process</a></li>
        <li><a href="#reverse-process">Reverse Process & Training</a></li>
        <li><a href="#diffusion-losses">Diffusion Loss Functions</a></li>
        <li><a href="#conditioning">Conditioning & Classifier-Free Guidance</a></li>
        <li><a href="#latent-diffusion">Latent Diffusion & DiT</a></li>
        <li><a href="#samplers">Samplers & Speed</a></li>
        <li><a href="#flow-matching">Flow Matching</a></li>
        <li><a href="#diffusion-models-list">Major Models</a></li>
      </ul>
    </li>
    <li><a href="#video-models">Video Models</a>
      <ul class="post-toc-sublist">
        <li><a href="#video-challenges">Challenges of Video</a></li>
        <li><a href="#video-architectures">Architectures: 3D CNN → Transformer → Diffusion</a></li>
        <li><a href="#video-losses">Video Loss Functions</a></li>
        <li><a href="#video-models-list">Major Models</a></li>
      </ul>
    </li>
    <li><a href="#world-models">World Models</a>
      <ul class="post-toc-sublist">
        <li><a href="#world-model-what">What a World Model Is</a></li>
        <li><a href="#world-model-components">State, Dynamics, Reward</a></li>
        <li><a href="#world-model-losses">World Model Loss Functions</a></li>
        <li><a href="#world-model-lineages">Lineages: Compact Latent → Video → LLM-Based</a></li>
        <li><a href="#world-model-list">Major Models</a></li>
        <li><a href="#world-model-failure">Failure Modes</a></li>
      </ul>
    </li>
    <li><a href="#multimodal-eval">Multimodal Model Evaluation</a>
      <ul class="post-toc-sublist">
        <li><a href="#vlm-benchmarks">VLM Benchmarks</a></li>
        <li><a href="#generation-eval">Generation Evaluation</a></li>
        <li><a href="#video-eval">Video Model Evaluation</a></li>
        <li><a href="#world-model-eval">World Model Evaluation</a></li>
        <li><a href="#eval-pitfalls">Evaluation Pitfalls</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## Convolutional Neural Networks {#cnns}

CNNs are the foundation of modern computer vision. Understanding them is not optional — ViT patch embeddings, VLM visual encoders, diffusion U-Nets, and video models all trace their lineage here. This section goes deeper than most treatments: detection pipelines, segmentation architectures, the VAE, intuition challenges, and then the transition to transformers.

### The Convolution Operation {#conv-operation}

A 2D convolution slides a learnable filter $W \in \mathbb{R}^{k \times k \times C_{in}}$ over an input feature map $X \in \mathbb{R}^{H \times W \times C_{in}}$:

$$
Y_{i,j,c_{out}} = \sigma\!\left(\sum_{u=0}^{k-1}\sum_{v=0}^{k-1}\sum_{c_{in}} W_{u,v,c_{in},c_{out}}\, X_{i+u,\, j+v,\, c_{in}} + b_{c_{out}}\right)
$$

Output spatial size with padding $p$ and stride $s$:

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1
$$

**Key inductive biases:**

| Bias | What it says | Why it helps |
|---|---|---|
| **Local receptive fields** | Each neuron sees only a small patch | Natural images have local structure |
| **Weight sharing** | Same filter applied everywhere | Translation equivariance; fewer parameters |
| **Hierarchical stacking** | Deeper layers have larger effective receptive fields | Edges → textures → parts → objects |

**Parameter count**: a conv layer with $C_{in}$ input channels, $C_{out}$ output channels, kernel $k \times k$ has $k^2 \cdot C_{in} \cdot C_{out} + C_{out}$ parameters — independent of image size.

**Depthwise separable convolution** ([MobileNet](https://arxiv.org/abs/1704.04861), EfficientNet):

$$
\text{DW} + \text{PW:} \quad k^2 \cdot C_{in} + C_{in} \cdot C_{out} \quad \text{vs standard} \quad k^2 \cdot C_{in} \cdot C_{out}
$$

Reduction factor $\approx \frac{1}{C_{out}} + \frac{1}{k^2}$. For $k=3$, $C_{out}=256$: ~8–9× fewer parameters.

> **Counterintuitive**: convolution as we write it in deep learning is technically *cross-correlation* — the filter is not flipped before sliding. True mathematical convolution flips the kernel. The distinction doesn't matter for learning because the network learns whatever filter achieves the task, flipped or not. The name "convolution" stuck from signal processing.

> **Counterintuitive**: a $7\times7$ conv and three stacked $3\times3$ convs have the same receptive field, but the three $3\times3$ convs have $3 \times 9C^2 = 27C^2$ parameters vs $49C^2$ for the single $7\times7$ — plus two extra non-linearities. This is why VGG and everything after uses $3\times3$ almost exclusively.

---

### CNN Building Blocks {#cnn-building-blocks}

#### Pooling

Max pooling: $y_{i,j} = \max_{u,v \in \text{window}} x_{i+u, j+v}$. Reduces spatial size, provides local translation invariance. Global Average Pooling (GAP) collapses $H \times W$ to a single vector before the classifier — introduces a powerful inductive bias that each channel map corresponds to a class-specific detector.

> **Counterintuitive — GAP vs Flatten**: Flattening after the last conv layer and using a dense layer has $H \cdot W \cdot C \cdot \text{classes}$ parameters and forces a fixed input resolution. GAP has $C \cdot \text{classes}$ parameters and works on any input size. GAP usually generalises better because it forces each channel to be a spatial detector, not a spatially-specific lookup.

#### Batch Normalisation

$$
\hat{x}^{(k)} = \frac{x^{(k)} - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}, \qquad y^{(k)} = \gamma \hat{x}^{(k)} + \beta
$$

BN normalises over the batch dimension per channel. Accelerates training, acts as regularisation (the noise from batch statistics), and reduces sensitivity to weight initialisation. At inference, uses exponential moving average statistics accumulated during training — **not** the current batch stats.

> **Counterintuitive**: BN is often said to reduce "internal covariate shift," but that explanation is [disputed](https://arxiv.org/abs/1805.11604). The empirical reason it works may be that it smooths the loss landscape, making gradient steps more reliable — not that it keeps layer input distributions fixed.

> **Counterintuitive**: BN behaves very differently at train vs inference time. A common bug: forgetting to call `model.eval()` before inference, leaving the model using noisy batch statistics instead of learned running stats. This silently degrades performance.

Other normalisation variants used when batch size is small (e.g., object detection, video):

| Norm | Normalises over | Good for |
|---|---|---|
| **BatchNorm** | Batch + spatial (per channel) | Large-batch classification |
| **LayerNorm** | All channels + spatial (per sample) | Transformers, NLP |
| **InstanceNorm** | Spatial only (per sample, per channel) | Style transfer |
| **GroupNorm** | Groups of channels (per sample) | Small-batch detection/video |

#### Residual (Skip) Connection

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

The residual branch only needs to learn the *delta* from the identity. Crucially, gradients flow directly through the skip connection: $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right)$. The `1` ensures a gradient highway regardless of how small $\frac{\partial \mathcal{F}}{\partial \mathbf{x}}$ is. [ResNet](https://arxiv.org/abs/1512.03385) showed 152-layer networks could be trained stably — previously impossible.

**Bottleneck block**: $1\times1$ conv (reduce channels) → $3\times3$ conv → $1\times1$ conv (expand channels). Reduces FLOPs while maintaining representational capacity.

> **Counterintuitive**: ResNet-110 (with residuals) outperforms ResNet-110 without residuals by a large margin — but also outperforms a plain 20-layer network. The issue is not vanishing gradients in the backward pass (gradient clipping fixes that); it's that without residuals, *deeper networks simply fail to learn the identity mapping*, so they perform worse than shallower ones. Residuals make identity the default, and the network deviates only when useful.

#### Dilated (Atrous) Convolution

Insert gaps between filter elements, controlled by dilation rate $d$:

$$
Y_{i,j} = \sum_{u,v} W_{u,v} \cdot X_{i + d \cdot u,\, j + d \cdot v}
$$

With $d=2$, a $3\times3$ filter covers a $5\times5$ area. With $d=4$, a $13\times13$ area — all with the same 9 parameters. Used in semantic segmentation ([DeepLab](https://arxiv.org/abs/1606.00915)) to expand receptive field without striding, preserving spatial resolution for dense prediction.

Stacking dilations $(1, 2, 4, 8, \ldots)$ gives exponentially growing receptive fields; the [WaveNet](https://arxiv.org/abs/1609.03499) architecture uses this for audio.

#### Transposed Convolution (Deconvolution)

Used to upsample feature maps — the spatial inverse of a strided convolution.

$$
H_{out} = (H_{in} - 1) \cdot s - 2p + k
$$

Inserts $s-1$ zeros between input pixels, then convolves. Learnable upsampling — unlike bilinear interpolation. Used in decoder paths of U-Net, segmentation heads, and VAE decoders. Suffers from **checkerboard artifacts** when $k$ is not divisible by $s$; fix: use bilinear upsample + $1\times1$ conv instead.

---

### Key CNN Architectures {#cnn-architectures}

| Model | Year | Paper | Key Innovation |
|---|---|---|---|
| **AlexNet** | 2012 | [link](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | Deep CNN on GPU; ReLU; dropout; data augmentation |
| **VGGNet** | 2014 | [link](https://arxiv.org/abs/1409.1556) | Very deep with $3\times3$ only; showed depth matters |
| **GoogLeNet / Inception** | 2014 | [link](https://arxiv.org/abs/1409.4842) | Inception module: parallel multi-scale convolutions |
| **ResNet** | 2015 | [link](https://arxiv.org/abs/1512.03385) | Residual connections; enabled 100+ layer nets |
| **DenseNet** | 2017 | [link](https://arxiv.org/abs/1608.06993) | Dense connections: every layer feeds all subsequent |
| **MobileNet** | 2017 | [link](https://arxiv.org/abs/1704.04861) | Depthwise separable convs; mobile-first |
| **EfficientNet** | 2019 | [link](https://arxiv.org/abs/1905.11946) | Compound scaling of depth/width/resolution |
| **ConvNeXt** | 2022 | [link](https://arxiv.org/abs/2201.03545) | Modernised ResNet matching ViT (depthwise 7×7, LN, GELU) |

**EfficientNet compound scaling**: scale depth $d = \alpha^\phi$, width $w = \beta^\phi$, resolution $r = \gamma^\phi$ subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$. The constraint keeps total FLOPs doubling per scaling step.

**DenseNet** connections: layer $l$ receives feature maps from all preceding layers $0, 1, \ldots, l-1$. Total parameters are fewer than ResNet despite more connections, because each layer can be narrow (32 channels) since it sees the entire history. The downside: GPU memory scales quadratically with depth.

---

### Object Detection {#object-detection}

Detection is harder than classification: you need to simultaneously *find* objects (localisation) and *name* them (classification). The field evolved through three broad eras.

#### Two-Stage Detectors: R-CNN Family

**R-CNN (2014)** — [paper](https://arxiv.org/abs/1311.2524)

1. Run selective search to propose ~2000 region proposals per image
2. Warp each proposal to fixed size, run CNN independently → 2000 forward passes per image
3. SVM classifier on CNN features; separate bounding box regressor

Accurate but absurdly slow (~47 seconds per image). The CNN runs 2000 times — once per region.

**Fast R-CNN (2015)** — [paper](https://arxiv.org/abs/1504.08083)

Key insight: run the CNN *once* on the whole image, then extract region features from the resulting feature map.

1. Compute feature map $F$ for entire image with a CNN backbone
2. For each region proposal: **RoI Pooling** — crop the corresponding region from $F$, pool to fixed $7\times7$ regardless of proposal size
3. Shared FC layers → class probabilities + box offsets

Still uses selective search (slow), but the CNN runs once. ~210ms/image. RoI Pooling quantises proposal coordinates to feature map grid — loses sub-pixel precision.

**Faster R-CNN (2015)** — [paper](https://arxiv.org/abs/1506.01497)

Eliminates selective search entirely. Adds a **Region Proposal Network (RPN)** that shares the same CNN backbone:

```
Image → CNN Backbone → Feature Map
                            ├─→ RPN: predicts objectness + box offsets
                            │         for each anchor at each spatial location
                            └─→ RoI Pooling → Classification head
```

**Anchors**: pre-defined boxes of multiple scales and aspect ratios at every spatial location. The RPN predicts for each anchor: (1) is there an object here? (2) what are the dx, dy, dw, dh offsets?

At each spatial location with $k$ anchors, the RPN outputs $2k$ objectness scores and $4k$ box offsets. Proposals with high objectness scores are passed to the detection head.

**RPN Loss:**

$$
\mathcal{L}_{RPN} = \frac{1}{N_{cls}}\sum_i \mathcal{L}_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}}\sum_i p_i^* \mathcal{L}_{reg}(t_i, t_i^*)
$$

where $p_i^*=1$ for positive anchors (IoU > 0.7 with any GT box), $p_i^*=0$ for negative (IoU < 0.3). The regression loss is Smooth L1 over the 4 offsets.

**Anchor assignment rule**: Positive if IoU > 0.7 with any GT; negative if IoU < 0.3 with all GT; ignored otherwise (0.3–0.7). The 1:3 positive:negative sampling ratio prevents the loss from being dominated by easy negatives.

**FPN — Feature Pyramid Network** ([paper](https://arxiv.org/abs/1612.03144)): multi-scale feature maps from a single forward pass. Bottom-up CNN produces feature maps at strides 4, 8, 16, 32. FPN adds a top-down pathway with lateral connections:

$$
M_k = \text{upsample}(M_{k+1}) + \text{conv}_{1\times1}(C_k)
$$

Each level predicts objects at a specific scale range. Small objects use high-resolution early layers (stride 4); large objects use semantically rich late layers (stride 32). This unified multi-scale detection without needing image pyramids.

**Mask R-CNN** ([paper](https://arxiv.org/abs/1703.06870)): extends Faster R-CNN with a parallel mask branch. For each proposal, predicts a $28\times28$ binary mask per class. Introduces **RoI Align** to fix the quantisation error of RoI Pooling: uses bilinear interpolation at non-integer feature map coordinates instead of rounding. This small change substantially improves mask quality.

$$
\mathcal{L}_{total} = \mathcal{L}_{cls} + \mathcal{L}_{box} + \mathcal{L}_{mask}
$$

Mask loss is binary cross-entropy applied only to the GT class channel — not averaged over all classes, preventing class competition.

> **Counterintuitive — why two stages?** Two-stage detectors are slower but more accurate. The RPN generates a small set of high-quality proposals, the detection head sees only those. Single-stage detectors (YOLO, SSD) must handle all anchors in a single pass — the imbalance between foreground and background anchors is extreme (1:1000+), which is why focal loss was invented.

#### Single-Stage Detectors: YOLO Family

Single-stage detectors skip the region proposal step entirely and predict boxes and classes directly from the feature map in one forward pass — much faster.

**YOLO v1 (2016)** — [paper](https://arxiv.org/abs/1506.02640)

Divide image into $S \times S$ grid (e.g., 7×7). Each cell predicts $B$ bounding boxes and $C$ class probabilities:

- Each box: $(x, y, w, h, \text{conf})$ — 5 values
- Each cell: $C$ class scores
- Total output: $S \times S \times (5B + C)$

Box confidence: $P(\text{Object}) \times \text{IoU}_{pred}^{truth}$.

Limitation: each cell predicts only one class, so can't detect multiple objects of different classes in same cell. Also: relative coordinates to cell make regression harder.

**SSD (2016)** — [paper](https://arxiv.org/abs/1512.02325): multi-scale single-stage. Predicts from feature maps at multiple scales (8×8, 4×4, 2×2, 1×1), placing anchors at each scale. No RPN, but gets multi-scale benefits. Still suffers from class imbalance.

**RetinaNet + Focal Loss (2017)** — [paper](https://arxiv.org/abs/1708.02002)

The reason single-stage detectors lagged: the massive class imbalance. For a $640\times640$ image with $\sim$100K anchors and maybe 10 GT boxes, ~99,990 anchors are background. Even if each background anchor contributes a small loss, they overwhelm the foreground signal.

**Focal Loss** down-weights easy examples:

$$
\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

$(1-p_t)^\gamma$ is the **modulating factor**. When $p_t$ is high (easy example, correctly classified with high confidence), $(1-p_t)^\gamma \approx 0$ — nearly zero weight. Hard examples where $p_t$ is low get weight $\approx 1$. Standard $\gamma=2$: an example with $p_t=0.9$ gets downweighted by $(0.1)^2 = 0.01\times$. $\alpha_t$ handles class frequency imbalance separately.

This single change made single-stage RetinaNet match Faster R-CNN accuracy at faster speed.

**YOLO v3 (2018)** — [paper](https://arxiv.org/abs/1804.02767): multi-scale predictions at 3 feature map scales (like FPN), 3 anchor sizes per scale. Darknet-53 backbone. Strong accuracy/speed trade-off; becomes the standard for real-time detection.

**YOLO v5 / v8 / v9 / v10**: incremental improvements — CSP connections, anchor-free heads, decoupled heads (separate classification + regression), BN in backbone. v8/v10 are anchor-free and dominate practical use.

**FCOS (2019)** — [paper](https://arxiv.org/abs/1904.01355): fully anchor-free. Each spatial location in the feature map predicts: (l, t, r, b) — distances to four box sides, plus centerness score. No anchor hyperparameters to tune.

$$
\text{centerness} = \sqrt{\frac{\min(l, r)}{\max(l, r)} \cdot \frac{\min(t, b)}{\max(t, b)}}
$$

Centerness suppresses low-quality boxes far from the object centre.

#### Transformer-Based Detection: DETR

**DETR (2020)** — [paper](https://arxiv.org/abs/2005.12872): removes all hand-engineered components — no anchors, no NMS, no RPN.

```
Image → CNN backbone → Flattened feature map
    → Transformer encoder (self-attention across spatial positions)
    → Transformer decoder (N learned object queries cross-attend to encoder output)
    → N box predictions (one per query)
```

N fixed object queries (typically 100) are learned embeddings. Each learns to attend to a region and decode a box + class. After training, most queries specialise to specific scales/regions.

**Hungarian Matching** — the core training trick. DETR predicts exactly $N$ boxes; GT has $M \leq N$ objects. How do you assign predictions to GT without NMS?

Find the optimal bijective assignment $\hat{\sigma}$ between predictions $\{\hat{y}_i\}$ and GT $\{y_j\}$ (padded with $\emptyset$) that minimises the total matching cost:

$$
\hat{\sigma} = \argmin_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{match}(y_i, \hat{y}_{\sigma(i)})
$$

$$
\mathcal{L}_{match}(y_i, \hat{y}_j) = -\mathbf{1}_{c_i \neq \emptyset}\hat{p}_{\sigma(i)}(c_i) + \mathbf{1}_{c_i \neq \emptyset}\mathcal{L}_{box}(b_i, \hat{b}_{\sigma(i)})
$$

The Hungarian algorithm (Kuhn-Munkres) solves this assignment in $O(N^3)$. Once matched, compute the actual detection loss only on matched pairs; unmatched predictions get a "no object" label.

**Why Hungarian matching eliminates NMS**: without a 1:1 assignment, multiple predictions compete for the same GT box → need NMS to suppress duplicates. With Hungarian matching, each GT is assigned to exactly one prediction — no duplicates by construction.

**DETR weakness**: slow to converge (~500 epochs vs ~12 for Faster R-CNN). Attention needs many iterations to learn to ignore background. **Deformable DETR** ([paper](https://arxiv.org/abs/2010.04159)) fixes this with deformable attention — each query attends to a small set of sampled key positions rather than the full map — converging in ~10× fewer epochs.

#### Non-Maximum Suppression (NMS)

Most detectors predict multiple overlapping boxes for the same object. NMS removes duplicates:

1. Sort boxes by confidence score descending
2. Take the highest-confidence box; mark as kept
3. Remove all boxes with IoU > threshold (typically 0.5) with the kept box
4. Repeat until no boxes remain

**Soft NMS** ([paper](https://arxiv.org/abs/1704.04503)): instead of hard removal, decay the score of overlapping boxes:

$$
s_i = s_i \cdot e^{-\text{IoU}(M, b_i)^2 / \sigma}
$$

Better for crowded scenes where true overlapping objects exist.

> **Counterintuitive — NMS is not differentiable**. The whole detection pipeline (including NMS) cannot be trained end-to-end through NMS because it's a greedy sorting procedure. This motivated DETR's Hungarian matching, which is also not differentiable but is applied *before* the loss rather than *after* — so gradients flow through the predicted scores.

#### Detection Metrics

**IoU (Intersection over Union)**:

$$
\text{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

**Precision-Recall curve**: at a given IoU threshold (e.g., 0.5), a prediction is TP if IoU with GT > threshold, FP otherwise. Vary the confidence threshold to get the PR curve.

**AP (Average Precision)**: area under the PR curve. **mAP**: mean AP across classes. **COCO metric** uses mAP averaged over IoU thresholds 0.5:0.05:0.95 — much harder than the single-threshold VOC metric.

$$
\text{AP} = \int_0^1 p(r)\, dr \approx \sum_k (r_k - r_{k-1}) p_k
$$

---

### Segmentation {#segmentation}

Segmentation assigns a label to every pixel (semantic) or every object instance (instance) or both (panoptic).

| Task | Output | Example use |
|---|---|---|
| **Semantic** | Per-pixel class label | Road/sky/car segmentation |
| **Instance** | Per-pixel instance ID (same class → different IDs) | Counting individual cells |
| **Panoptic** | Semantic for background + instance for objects | Autonomous driving scene understanding |

#### Fully Convolutional Networks (FCN)

[FCN (2015)](https://arxiv.org/abs/1411.4038): replace classification head (FC layers) with $1\times1$ convolutions. Output a spatial score map, upsample to input resolution with transposed convolution. First end-to-end trainable dense prediction network.

Problem: strided pooling discards spatial information. Final feature map is $1/32$ of input. Upsampling 32× in one step is too coarse. Solution: skip connections from earlier layers (FCN-16s, FCN-8s) fuse fine spatial detail with deep semantic context.

#### U-Net

[U-Net (2015)](https://arxiv.org/abs/1505.04597) — originally for biomedical image segmentation; now the backbone of diffusion models, Stable Diffusion included.

Architecture: symmetric encoder-decoder with **skip connections at every resolution level**.

```
Input (572×572)
  ↓ conv×2, MaxPool          Encoder (contracting path)
[64]  → 284×284
  ↓ conv×2, MaxPool
[128] → 140×140
  ↓ conv×2, MaxPool
[256] → 68×68
  ↓ conv×2, MaxPool
[512] → 32×32
  ↓ conv×2             (bottleneck)
[1024] → 28×28
  ↑ TransposedConv + concat with encoder skip     Decoder (expanding path)
[512] → 52×52
  ↑ TransposedConv + concat
[256] → 100×100
  ↑ TransposedConv + concat
[128] → 196×196
  ↑ TransposedConv + concat
[64]  → 388×388
  → 1×1 conv → segmentation map
```

The skip connections concatenate full-resolution encoder feature maps with the upsampled decoder features. This gives the decoder access to both: (1) spatial detail from early layers, (2) semantic context from the bottleneck.

> **Why U-Net is everywhere in diffusion models**: diffusion denoisers need to reconstruct fine-grained pixel detail (like segmentation decoders) while also understanding global semantics at multiple scales. The U-Net architecture, with its multi-scale skip connections, is a natural fit. The timestep embedding is injected at each resolution level, giving the network explicit knowledge of how noisy the input is.

**Loss**: standard is pixel-wise cross-entropy + Dice loss:

$$
\mathcal{L}_{seg} = \mathcal{L}_{CE} + \mathcal{L}_{Dice} = -\sum_{i} y_i \log \hat{p}_i + \left(1 - \frac{2\sum_i p_i g_i}{\sum_i p_i^2 + \sum_i g_i^2}\right)
$$

Dice loss is particularly important for medical imaging where foreground pixels are rare (<5% of image).

#### DeepLab Family

[DeepLabv3+](https://arxiv.org/abs/1802.02611): combines dilated convolutions to maintain resolution + ASPP (Atrous Spatial Pyramid Pooling) for multi-scale context + decoder for boundary refinement.

**ASPP**: apply parallel dilated convolutions with rates $r \in \{6, 12, 18\}$ plus global average pooling. Concatenate all, fuse with $1\times1$ conv. Captures objects at multiple scales without resolution loss.

#### Panoptic Segmentation

[Panoptic FPN (2019)](https://arxiv.org/abs/1901.02446): extends Mask R-CNN (instance) with a semantic segmentation head on the same FPN features. Two heads, one backbone.

**Panoptic Quality (PQ)**:

$$
\text{PQ} = \underbrace{\frac{\sum_{(p,g)\in TP} \text{IoU}(p,g)}{|TP|}}_{\text{Segmentation Quality (SQ)}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{Recognition Quality (RQ)}}
$$

---

### Variational Autoencoder (VAE) {#vae}

The VAE is not strictly a CNN architecture, but it uses CNNs as encoder/decoder and is the component that lets diffusion models work in latent space rather than pixels. Understanding it is essential.

**Standard Autoencoder**: encoder $q(z \mid x)$ maps input to a latent code $z$; decoder $p(x \mid z)$ reconstructs. Problem: the latent space has no structure — you can't sample from it. Nearby points in latent space may decode to completely different images.

**VAE** ([Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)): instead of encoding to a point, encode to a *distribution* — specifically $q_\phi(z \mid x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x) \cdot I)$.

The encoder outputs two vectors: $\boldsymbol{\mu}$ and $\log \boldsymbol{\sigma}^2$. Sample via **reparametrisation trick** (makes the sample differentiable):

$$
z = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
$$

Without this trick: $z \sim \mathcal{N}(\mu, \sigma^2)$ is a stochastic node — gradients can't flow through sampling. The reparametrisation moves the randomness to $\boldsymbol{\epsilon}$, which is not a function of parameters, so $\frac{\partial z}{\partial \mu}$ and $\frac{\partial z}{\partial \sigma}$ are well-defined.

**ELBO — Evidence Lower Bound** (the VAE loss):

$$
\mathcal{L}_{VAE} = \underbrace{-\mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right]}_{\text{Reconstruction loss}} + \underbrace{D_{KL}\!\left(q_\phi(z \mid x) \,\|\, p(z)\right)}_{\text{Regularisation}}
$$

The KL term with a standard Normal prior $p(z) = \mathcal{N}(0, I)$ has a closed form:

$$
D_{KL} = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
$$

**What each term does**:
- **Reconstruction**: decoder must be able to reconstruct $x$ from $z$. This forces $z$ to be informative.
- **KL regularisation**: forces the posterior $q(z \mid x)$ to stay close to the prior $\mathcal{N}(0,I)$. This gives the latent space structure — interpolation between latents produces coherent outputs, and sampling from $\mathcal{N}(0,I)$ generates valid images.

> **Counterintuitive — posterior collapse**: if the decoder is too powerful (e.g., a large autoregressive model), it can reconstruct $x$ from $z$ without actually using $z$. The KL term then drives $q(z \mid x) \to p(z)$ — the posterior collapses to the prior. The model ignores the latent code entirely. Fix: KL annealing (start training with $\beta=0$, gradually increase), or $\beta$-VAE (downweight KL), or weak decoders.

> **Counterintuitive — ELBO maximisation ≠ exact marginal likelihood maximisation**: the ELBO is a lower bound on $\log p(x)$. Maximising the ELBO tightens the gap $D_{KL}(q_\phi(z \mid x) \,\|\, p_\theta(z \mid x))$ — the approximate posterior approaches the true posterior. But the ELBO and true $\log p(x)$ are only equal when $q_\phi = p_\theta$, which requires infinite capacity in the approximate posterior.

**VQ-VAE** ([van den Oord 2017](https://arxiv.org/abs/1711.00937)): replaces the continuous latent with a discrete codebook. The encoder output is quantised to the nearest codebook vector. Used in DALL-E (original), enabling autoregressive image generation over a finite token vocabulary. The quantisation step is not differentiable; fixed with a **straight-through estimator** — copy gradients from decoder input to encoder output as if the quantisation didn't happen.

**Stable Diffusion's VAE**: 4-channel latent at 1/8 spatial resolution. Trained with a combination of reconstruction loss, perceptual loss (VGG features), and adversarial loss from a patch discriminator. The diffusion model then operates entirely in this 4-channel latent space — 64× fewer pixels per step.

---

### CNN Loss Functions {#cnn-losses}

**Classification — Cross-Entropy:**

$$
\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \log \hat{p}_c
$$

For binary classification (sigmoid output): $\mathcal{L}_{BCE} = -[y \log \hat{p} + (1-y)\log(1-\hat{p})]$

**Focal Loss (RetinaNet)** — for extreme class imbalance in detection:

$$
\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

$\gamma=2$, $\alpha_t=0.25$ for background class. Easy negatives ($p_t=0.99$) contribute $(0.01)^2 \approx 0.0001\times$ their normal weight.

**Detection — Smooth L1 / Huber** for bounding box regression:

$$
\mathcal{L}_{loc}(x) = \begin{cases} 0.5 x^2 & |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}
$$

**GIoU Loss** — directly optimises IoU rather than coordinate distance:

$$
\mathcal{L}_{GIoU} = 1 - \text{IoU} + \frac{|\text{enclosing box} \setminus (A \cup B)|}{|\text{enclosing box}|}
$$

When boxes don't overlap, IoU=0 for any configuration — gradient is zero. GIoU's second term is still non-zero, providing a gradient that moves predictions toward GT.

**DIoU / CIoU** ([paper](https://arxiv.org/abs/1911.08287)): DIoU adds a penalty on the distance between box centres; CIoU further adds an aspect ratio consistency term:

$$
\mathcal{L}_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v, \quad v = \frac{4}{\pi^2}\left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2
$$

**Segmentation — Dice Loss** (handles class imbalance):

$$
\mathcal{L}_{Dice} = 1 - \frac{2\sum_i p_i g_i}{\sum_i p_i^2 + \sum_i g_i^2}
$$

Combined: $\mathcal{L}_{seg} = \mathcal{L}_{CE} + \mathcal{L}_{Dice}$. Cross-entropy penalises per-pixel mistakes; Dice penalises mismatch in overall mask shape. Complementary.

**Contrastive Pretraining — NT-Xent / InfoNCE:**

$$
\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}
$$

Used in [SimCLR](https://arxiv.org/abs/2002.05709), [MoCo](https://arxiv.org/abs/1911.05722) for self-supervised visual pretraining.

---

### Intuition Check: Hard Questions {#cnn-intuition}

These are questions where the right answer is not obvious — worth sitting with before reading the answer.

**Q: Why does increasing depth sometimes make training accuracy *worse* before residuals?**

Not a vanishing gradient problem — gradient clipping handles that. The issue is that without residuals, a deeper network must learn to approximate the identity mapping when more depth is not beneficial. But MLPs are bad at learning exact identity maps from scratch through composition. The residual reformulation $\mathcal{F}(x) = H(x) - x$ makes identity the default (when $\mathcal{F} \to 0$) and deviation the exception. A plain 56-layer network genuinely underperforms a 20-layer one on the training set — it's not an overfitting issue.

**Q: Translation equivariance vs translation invariance — which does a CNN have, and why does it matter?**

CNNs are translation **equivariant** (not invariant): if you shift the input, the output feature map shifts by the same amount. Max pooling introduces partial **invariance** (insensitivity to small shifts within the pooling window). Global Average Pooling introduces full translation invariance at the cost of spatial information.

This matters for detection: you *want* equivariance in the backbone — the feature map should shift when the object shifts, so the RPN can find the object at its new location. You *want* invariance in the classifier — it shouldn't matter exactly where in the RoI-pooled window the object lands.

**Q: Why doesn't a larger receptive field always help?**

Theoretical receptive field grows quickly with depth, but the **effective receptive field** ([paper](https://arxiv.org/abs/1701.04128)) — the region actually influencing output — is much smaller and has a Gaussian shape (central pixels matter far more than peripheral ones). A 101-layer ResNet may have a theoretical receptive field covering the entire image, but in practice only a central patch of ~200×200 pixels meaningfully influences any given neuron. This limits performance on tasks requiring global context — motivating attention mechanisms.

**Q: Why does BN help at training time but sometimes hurt at test time with small batches?**

BN at training time uses the current mini-batch statistics (mean, variance). At test time, it uses exponential moving averages from training. If the test distribution differs from training (covariate shift) OR if the batch size at test time is 1 (so the moving average was computed over a different distribution), BN can degrade. Solutions: Group Norm (doesn't use batch statistics), or calibrate BN statistics on the target data.

**Q: If you double the number of filters in every layer of a CNN, how much does compute increase?**

Roughly 4×. Compute in a conv layer scales as $k^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W$. Doubling both $C_{in}$ and $C_{out}$ → 4× compute. This is why EfficientNet scales width conservatively: doubling width costs 4× compute but accuracy gains are sublinear.

**Q: What breaks when you apply a CNN trained on ImageNet to satellite imagery?**

Several things: (1) ImageNet images are taken from human-height perspective — textures and shape statistics differ from top-down aerial views; (2) scale — a "car" in a satellite image may be 5 pixels wide; (3) colour channels — satellite sensors often have more than 3 channels (NIR, SWIR), but ImageNet-pretrained CNNs expect RGB. Solutions: domain-adaptive finetuning, use early layer features only (more general), or train from scratch if data permits.

---

### From CNNs to Vision Transformers {#vision-transformers}

CNNs dominated vision until [ViT (2021)](https://arxiv.org/abs/2010.11929) showed that a pure transformer on image patches could match or exceed CNNs when pretrained at sufficient scale.

**ViT** flattens an image into $N = HW/P^2$ non-overlapping patches of size $P \times P$, linearly projects each to a $d$-dimensional embedding, prepends a `[CLS]` token, adds 2D positional embeddings, and applies a standard transformer encoder:

$$
\mathbf{z}_0 = [\mathbf{x}_{cls};\; \mathbf{x}_p^1 E;\; \ldots;\; \mathbf{x}_p^N E] + \mathbf{E}_{pos}
$$

**What ViT lacks vs CNNs:**
- No translation equivariance (must learn it from data)
- No explicit local inductive bias → needs large-scale data (JFT-300M) or strong augmentation ([DeiT](https://arxiv.org/abs/2012.12877) uses distillation to train ViT on ImageNet-only)
- Quadratic attention in spatial resolution → need hierarchical variants (Swin)

> **Counterintuitive**: ViT learns positional encodings that resemble spatial distance — nearby patches have similar positional embeddings even though the model was given no such prior. The model discovers locality from data alone. But it also learns global relationships that CNNs cannot — heads that directly attend from patch $i$ to patch $j$ regardless of distance.

**[Swin Transformer](https://arxiv.org/abs/2103.14030)** adds two things ViT lacks:
1. **Hierarchical feature maps** via patch merging (4→8→16→32 stride)
2. **Local window attention** (8×8 windows) with shifted windows between layers for cross-window communication at $O(N)$ cost instead of $O(N^2)$

Swin is the preferred backbone for detection/segmentation where multi-scale features are needed.

**[DINOv2](https://arxiv.org/abs/2304.07193)** distills self-supervised ViT features (no labels, trained on curated LVD-142M) with self-distillation (student-teacher). The resulting features generalise to depth estimation, segmentation, classification, and VLM visual encoders — often without any finetuning.

**CNN vs ViT in practice (2026):**

| Property | CNN (ResNet/ConvNeXt) | ViT (plain/Swin) |
|---|---|---|
| Data efficiency | High (strong inductive bias) | Low (needs JFT/LVD scale) |
| Global context | Limited by receptive field | Full from layer 1 |
| Multi-scale features | Native (FPN-ready) | Needs Swin or FPN wrapper |
| Translation equivariance | Built-in | Learned |
| Inference on small images | Efficient | Inefficient (patch overhead) |
| Scaling behaviour | Good but plateaus | Excellent — scales with data+compute |
| VLM visual encoder | Rare (CLIP ViT dominates) | Standard |

---

## Vision-Language Models {#vlms}

### Evolution: Captioning → CLIP → LLM-VLMs {#vlm-evolution}

**Generation 1 — CNN + RNN Captioning (2014–2018)**

Show & Tell: ResNet feature → LSTM decoder generating caption token-by-token. Cross-entropy loss over word sequences. Weak at grounding; fluent but hallucination-prone.

**Generation 2 — Attention over Regions (2018–2021)**

Bottom-Up Top-Down: Faster R-CNN extracts object-region features; attention mechanism in decoder decides which region to look at for each word. Enabled Visual Question Answering (VQA).

**Generation 3 — CLIP and Contrastive Pretraining (2021)**

[CLIP](https://arxiv.org/abs/2103.00020) trained a vision encoder (ViT or ResNet) and a text encoder (transformer) jointly on 400M image-text pairs from the internet using contrastive loss. At inference, classify by computing similarity of image embedding to all class-name embeddings — no finetuning needed.

Key insight: language supervision scales better than fixed label sets, and contrastive objectives force the representations to be semantically aligned.

**Generation 4 — LLM-backbone VLMs (2022–present)**

Attach a visual encoder to a frozen or finetuned LLM. The LLM provides language understanding, generation, and instruction following. The key question: **how to connect vision to language?**

---

### Architecture: Encoder, Projector, LLM {#vlm-architecture}

```
[Image]
   │
   ▼
┌──────────────┐
│ Vision Encoder│  (ViT-L, ViT-H, SigLIP, DINOv2, ...)
│ → patch tokens│  shape: (N_patches, D_vision)
└──────────────┘
   │
   ▼
┌──────────────┐
│   Projector  │  bridges vision ↔ language embedding spaces
└──────────────┘
   │
   ▼
┌──────────────────────────────────────┐
│            Language Model            │  (LLaMA, Mistral, Qwen, ...)
│ [IMG_TOK_1, ..., IMG_TOK_K, text...] │
└──────────────────────────────────────┘
   │
   ▼
[Generated text response]
```

**Projector types:**

| Projector | How it works | Used in |
|---|---|---|
| **Linear** | Single $W \in \mathbb{R}^{D_v \times D_l}$ | CLIP zero-shot |
| **MLP (2-layer)** | Linear → GELU → Linear | LLaVA-1.5 |
| **Q-Former** | Learnable query tokens cross-attend to patch tokens | InstructBLIP, BLIP-2 |
| **Perceiver Resampler** | Fixed set of learned latents cross-attend; compresses to fixed length | Flamingo, Idefics |
| **C-Abstractor** | CNN-style downsampling of patch grid | MobileVLM |

Q-Former and Perceiver Resampler compress $N_{patches}$ (often 256–1024) to a fixed small set (32–64 tokens), saving LLM context. MLP projectors keep all patches but are faster to train.

**High-resolution handling:**

A 336px ViT produces $24 \times 24 = 576$ patch tokens. For 1080px inputs that's 8,100 tokens — too many for typical context windows. Solutions:

- **Dynamic resolution** (LLaVA-HD, InternVL): tile the image into crops, encode each crop separately, concatenate
- **Token compression** (LLaVA-OneVision, Qwen-VL): average-pool spatial neighbours before the LLM
- **Native resolution ViT** (NaViT, SigLIP2): pack variable-size images into fixed compute budget using fractional positional embeddings

---

### Training Stages & Objectives {#vlm-training}

Most VLMs follow a 2–3 stage recipe:

**Stage 1 — Vision-language alignment (projector pretraining)**

Freeze vision encoder and LLM. Train only the projector on image-caption pairs. Objective: the LLM's language modelling loss on the caption, conditioned on image tokens. Fast: typically millions of pairs, hours on 8 GPUs.

**Stage 2 — Visual instruction tuning**

Unfreeze projector; optionally unfreeze LLM (or use LoRA). Train on instruction-following datasets: VQA pairs, OCR, charts, code, multi-turn conversation. The model learns to follow natural-language instructions about images.

**Stage 3 (optional) — RLHF / DPO / RLAIF**

Preference alignment on multimodal outputs. RLHF with human preference pairs on image+response; or DPO directly. Reduces hallucination, improves instruction adherence.

---

### VLM Loss Functions {#vlm-losses}

**Contrastive loss (CLIP training):**

For a batch of $N$ image-text pairs, with image embeddings $\mathbf{i}_k$ and text embeddings $\mathbf{t}_k$ (both L2-normalised), and learnable temperature $\tau$:

$$
\mathcal{L}_{CLIP} = -\frac{1}{2N}\sum_{k=1}^{N}\left[\log\frac{e^{\mathbf{i}_k \cdot \mathbf{t}_k / \tau}}{\sum_j e^{\mathbf{i}_k \cdot \mathbf{t}_j / \tau}} + \log\frac{e^{\mathbf{t}_k \cdot \mathbf{i}_k / \tau}}{\sum_j e^{\mathbf{t}_k \cdot \mathbf{i}_j / \tau}}\right]
$$

Two terms: image-to-text and text-to-image. At convergence, the diagonal of the $N \times N$ similarity matrix should be maximised.

**SigLIP — sigmoid contrastive (no softmax denominator):**

$$
\mathcal{L}_{SigLIP} = -\frac{1}{N^2}\sum_{i,j} \log \sigma\!\left(z_{ij}\cdot y_{ij} - b\right)
$$

where $z_{ij} = \mathbf{i}_i \cdot \mathbf{t}_j / \tau$, $y_{ij} = +1$ if $i=j$ else $-1$, and $b$ is a learnable bias. No softmax normalisation over the batch → scales better to very large batches; better zero-shot performance.

**Autoregressive LM loss (instruction tuning):**

$$
\mathcal{L}_{LM} = -\sum_{t=1}^{T} \log p_\theta(x_t \mid \mathbf{v}, x_{<t})
$$

where $\mathbf{v}$ are the visual tokens prepended to the context. Only the text response tokens are in the loss (not the instruction tokens — they are "assistant" side). Identical to causal LM pretraining, but conditioned on a visual prefix.

**ITM — Image-Text Matching (used in BLIP):**

Binary cross-entropy over a [CLS] embedding predicting whether an image and text are paired. Negative mining: sample hard negatives (most similar un-paired sample in the batch).

$$
\mathcal{L}_{ITM} = -\mathbb{E}_{(I,T)}\left[y \log p_{match} + (1-y)\log(1-p_{match})\right]
$$

**ITC + ITM + LM combination (BLIP-2):**

$$
\mathcal{L}_{BLIP-2} = \mathcal{L}_{ITC} + \mathcal{L}_{ITM} + \mathcal{L}_{LM}
$$

All three objectives operate on the Q-Former simultaneously. ITC aligns at the global level, ITM at the fine-grained level, LM trains generation.

**Grounding losses (object detection / region VLMs):**

For grounding text to bounding boxes (GLIP, Grounding DINO):

$$
\mathcal{L}_{ground} = \mathcal{L}_{cls} + \lambda_1 \mathcal{L}_{L1} + \lambda_2 \mathcal{L}_{GIoU}
$$

where $\mathcal{L}_{cls}$ is the language-image contrastive alignment at region level.

---

### Major VLM Models {#vlm-models}

| Model | Org | Visual Encoder | Projector | LLM | Notable |
|---|---|---|---|---|---|
| **Flamingo** | DeepMind | NFNet | Perceiver Resampler | Chinchilla | Few-shot with interleaved image/text |
| **BLIP-2** | Salesforce | EVA-CLIP | Q-Former | Vicuna / FlanT5 | Modular; frozen backbone |
| **LLaVA-1.5** | UW | CLIP ViT-L | 2-layer MLP | LLaMA-2 / Vicuna | Simple; strong on benchmarks |
| **InstructBLIP** | Salesforce | EVA-CLIP | Q-Former | Vicuna | Instruction-tuned BLIP |
| **Idefics2** | HuggingFace | SigLIP | Perceiver | Mistral-7B | Open; high-res |
| **InternVL2** | Shanghai AI Lab | InternViT | MLP | InternLM2 | Tiled high-res; strong OCR |
| **Qwen-VL** | Alibaba | ViT | Cross-attention | Qwen-7B | Strong Chinese; dynamic resolution |
| **GPT-4V / 4o** | OpenAI | Unknown | Unknown | GPT-4 | Frontier; multimodal reasoning |
| **Gemini 1.5** | Google | Unknown | Unknown | Gemini | 1M context; video native |
| **Claude 3.5** | Anthropic | Unknown | Unknown | Claude 3.5 | Document understanding |
| **LLaVA-OneVision** | UW | SigLIP | MLP + pool | Qwen2-7B | Best open single/multi-image |
| **Molmo** | AI2 | CLIP | Unknown | OLMo | Pointing, grounding |

---

### VLM Evaluation {#vlm-evaluation}

#### Understanding & Reasoning Benchmarks

| Benchmark | Task | Metric | What it tests |
|---|---|---|---|
| **VQA v2** | Open-ended Q&A on images | Accuracy | Basic visual understanding |
| **GQA** | Compositional scene graph Q&A | Accuracy | Spatial, relational reasoning |
| **MMMU** | Multi-discipline (14 subjects) college-level Q&A | Accuracy | Expert multimodal reasoning |
| **MMBench** | 2,974 structured choice Q&A | Accuracy (CircularEval) | Coverage across perception types |
| **MMStar** | Hard visually-grounded questions (no text-only solvable) | Accuracy | True vision dependency |
| **MathVista** | Math problems with figures | Accuracy | Visual math reasoning |
| **AI2D** | Science diagram Q&A | Accuracy | Diagram understanding |
| **ChartQA** | Chart Q&A (human + augmented) | Relaxed accuracy | Chart / data visualisation |
| **DocVQA** | Document image Q&A | ANLS (normalised Levenshtein) | OCR + layout understanding |
| **TextVQA** | Scene text reading + Q&A | Accuracy | OCR in the wild |
| **InfoVQA** | Infographic comprehension | ANLS | Visual information layout |
| **OCRBench** | 29 OCR subtasks | Accuracy | Comprehensive OCR |

#### Hallucination Benchmarks

| Benchmark | What it measures | Format |
|---|---|---|
| **POPE** | Object hallucination — does model say absent objects exist? | Yes/No Q&A; F1 |
| **MMHAL-BENCH** | Response-level hallucination on 96 images | GPT-4 scoring 0–6 |
| **HallusionBench** | Visual illusion and hallucination robustness | Accuracy |
| **FaithScore** | Atomic fact faithfulness in long descriptions | Automatic sub-claim verification |

**POPE** is the canonical lightweight hallucination test. Three sampling strategies — random, popular, adversarial — probe whether the model confidently asserts objects not present.

#### Grounding & Referring Benchmarks

| Benchmark | Task | Metric |
|---|---|---|
| **RefCOCO / RefCOCO+ / RefCOCOg** | Referring expression comprehension | Acc@0.5 IoU |
| **Flickr30K Entities** | Phrase grounding | Recall@K |
| **PointQA** | Point-based object location | Accuracy |

#### Video Understanding

| Benchmark | Task | Metric |
|---|---|---|
| **EgoSchema** | 5-choice Q&A on 3-min egocentric clips | Accuracy |
| **Video-MME** | Multi-choice Q&A, short/medium/long video | Accuracy |
| **MVBench** | 20 temporal tasks | Accuracy |

#### Evaluation Methodology

**CircularEval**: run each multiple-choice question with all option rotations; model must be consistent. Removes option-position bias — models that learned "answer is always B" fail.

**NormLevenshtein (ANLS)**: for OCR-heavy tasks where exact string match is too strict:

$$
\text{ANLS}(a, \hat{a}) = 1 - \frac{\text{EditDistance}(a, \hat{a})}{\max(|a|, |\hat{a}|)}
$$

**GPT-4-as-judge** for open-ended generation: compare model output to reference on a 1–10 scale. Correlates well with human judgement but has self-consistency bias (GPT-4 favours GPT-4 outputs).

**Metric pitfalls:**
- VQA accuracy rewards short answers; models learn to hedge or refuse to get partial credit
- Many benchmarks are saturated — frontier models score >90%. Need harder evals (MMMU-Pro, OlympiadBench)
- Data contamination: WebCrawled images may contain benchmark images in model pretraining data

#### Practical Evaluation Suite

For a new VLM, run in this order:
1. **POPE** — hallucination sanity check (cheap, fast)
2. **MMBench / MMStar** — general capability coverage
3. **TextVQA / DocVQA** — OCR capability
4. **ChartQA** — data comprehension
5. **MMMU** — expert reasoning
6. **RefCOCO** — grounding (if your use case involves localisation)
7. **Video-MME** — if video is in scope

---

### VLM Failure Modes {#vlm-failure}

| Failure | Cause | Mitigation |
|---|---|---|
| **Object hallucination** | Language model over-relies on priors; weak vision signal | POPE testing; RLHF on hallucination; contrastive decoding |
| **Counting errors** | Patch tokens don't explicitly encode count | High-res tiling; counting-specific data |
| **Fine-grained spatial reasoning** | ViT patches lose spatial precision for "left of / right of" | Grounding pretraining; coordinate prediction training |
| **Long-context visual degradation** | Many image tokens push text far from image in context | Efficient attention; visual token compression |
| **OCR on degraded images** | Low-res ViT; no specialised OCR pretraining | High-res encoder; synth OCR data |
| **Multi-image inconsistency** | No cross-image attention in naive architecture | Interleaved training; Flamingo-style architecture |

---

## Diffusion Models {#diffusion}

### Intuition {#diffusion-intuition}

Diffusion models learn to reverse a process that gradually destroys structure. Train on: "given a partially-noisy image, what was the original?" At generation time, start from pure noise and iteratively denoise.

The key insight is that learning the full distribution $p(\mathbf{x})$ is hard, but learning $p(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ — the reverse of a single small noising step — is tractable.

---

### Forward Process {#forward-process}

The forward process adds Gaussian noise over $T$ steps (typically $T = 1000$), following a variance schedule $\beta_1, \ldots, \beta_T$:

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\; \beta_t \mathbf{I}\right)
$$

Because Gaussian noise composes nicely, we can sample $\mathbf{x}_t$ at any step $t$ directly from $\mathbf{x}_0$ in closed form. Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:

$$
q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1 - \bar{\alpha}_t)\mathbf{I}\right)
$$

Or equivalently: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$.

At $t=0$: original image. At $t=T$: approximately $\mathcal{N}(0, \mathbf{I})$.

**Variance schedules:**
- **Linear** (DDPM): $\beta_t$ increases linearly from $10^{-4}$ to $0.02$
- **Cosine** (improved DDPM): $\bar{\alpha}_t = \cos^2\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$ — avoids too-sharp noise increase at start; better for small images
- **EDM (Karras 2022)**: parametrises directly in noise level $\sigma$, separate from network preconditioning

---

### Reverse Process & Training {#reverse-process}

The model $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ learns to predict the noise $\boldsymbol{\epsilon}$ added to $\mathbf{x}_0$. The reverse step:

$$
p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\; \sigma_t^2 \mathbf{I}\right)
$$

The predicted mean is:

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)
$$

Training samples random $t$ and random $\boldsymbol{\epsilon}$, then trains the network to predict $\boldsymbol{\epsilon}$ from the noisy image.

---

### Diffusion Loss Functions {#diffusion-losses}

**Simple (noise prediction) loss — DDPM:**

$$
\mathcal{L}_{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},\, t)\right\|^2\right]
$$

This is just an L2 loss on noise prediction. Remarkably simple given the probabilistic motivation.

**VLB (variational lower bound) — full derivation target:**

The full objective maximises the evidence lower bound on $\log p(\mathbf{x}_0)$:

$$
\mathcal{L}_{VLB} = \mathbb{E}_q\!\left[-\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1) + \sum_{t>1} D_{KL}\!\left(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\right)\right]
$$

The KL terms compare the forward posterior (which is tractable given $\mathbf{x}_0$) with the learned reverse. DDPM showed that $\mathcal{L}_{simple}$ (ignoring weighting) works better in practice despite not being the full VLB.

**x0-prediction vs ε-prediction vs v-prediction:**

Three equivalent parametrisations — predict the original image, predict the noise, or predict the "velocity" $\mathbf{v} = \sqrt{\bar{\alpha}_t}\boldsymbol{\epsilon} - \sqrt{1-\bar{\alpha}_t}\mathbf{x}_0$. v-prediction is more stable at high noise levels and is used in Stable Diffusion 2+, Imagen.

**Perceptual loss (for high-frequency detail):**

Some diffusion decoders add a perceptual term (VGG feature distance) and adversarial loss from a discriminator:

$$
\mathcal{L}_{dec} = \mathcal{L}_{rec} + \lambda_p \mathcal{L}_{perceptual} + \lambda_{adv} \mathcal{L}_{GAN}
$$

This is used in the VAE decoder of Stable Diffusion — the diffusion model operates in latent space, and the VAE decoder maps back to pixels.

---

### Conditioning & Classifier-Free Guidance {#conditioning}

**Classifier guidance (original, Dhariwal 2021):** train a noisy-image classifier $p_\phi(y \mid \mathbf{x}_t)$, then shift the score:

$$
\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t) - s \cdot \sigma_t \nabla_{\mathbf{x}_t} \log p_\phi(y \mid \mathbf{x}_t)
$$

Requires a separate classifier trained on noisy data. Awkward.

**Classifier-Free Guidance (CFG, Ho 2022):** train a single model that can be conditional or unconditional by randomly dropping the condition during training ($10$–$20$\% of steps, replace $c$ with $\emptyset$). At inference, extrapolate:

$$
\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \emptyset) + w\bigl(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \emptyset)\bigr)
$$

Guidance scale $w$ controls the trade-off: higher $w$ → more condition-faithful but less diverse. $w=1$ is unconditional; $w=7.5$ is a common default for text-to-image.

Text conditioning: encode the text prompt with CLIP text encoder or T5; cross-attend in the U-Net or DiT.

---

### Latent Diffusion & DiT {#latent-diffusion}

**Latent Diffusion (Stable Diffusion):** instead of denoising in pixel space ($512^2 \times 3$), first compress with a VAE to a latent $\mathbf{z} \in \mathbb{R}^{64 \times 64 \times 4}$ (8× spatial compression). Run the diffusion model in this latent space. Only decode to pixels at the end.

Benefits: 64× fewer pixels to process per denoising step → fast training + inference. The VAE encoder/decoder are pretrained separately with perceptual + adversarial loss.

**U-Net backbone (original SD):** Encoder downsampling path + Decoder upsampling path with skip connections. Each resolution level has ResNet blocks + cross-attention (for text conditioning).

**Diffusion Transformer (DiT, Peebles 2023):** replace U-Net with a ViT. Patch the latent, apply transformer blocks with adaptive layer norm (adaLN) to inject timestep and class conditioning:

$$
[\gamma, \beta] = \text{MLP}(t_{emb} + c_{emb}), \qquad \text{adaLN}(x) = \gamma \cdot \text{LN}(x) + \beta
$$

DiT scales better with compute than U-Net. Sora, SD3, Flux all use transformer-based diffusion backbones.

**MM-DiT (SD3 / Flux):** separate streams for image and text tokens, coupled by bidirectional cross-attention — gives text tokens full access to image context and vice versa throughout the denoising process.

---

### Samplers & Speed {#samplers}

| Sampler | Steps needed | Key idea |
|---|---|---|
| **DDPM** | 1000 | Full Markov chain; stochastic |
| **DDIM** | 20–50 | Non-Markovian; deterministic; same quality as DDPM |
| **DPM-Solver++** | 15–25 | Higher-order ODE solver for diffusion |
| **PNDM** | 20 | Pseudo-numerical methods |
| **LCM (Consistency)** | 4–8 | Consistency distillation; learn to map any point to $\mathbf{x}_0$ in one step |
| **SDXL-Turbo / SDXL-Lightning** | 1–4 | Adversarial distillation; GAN-trained from SD |

**DDIM** key insight: the forward process can be made deterministic — same seed → same image. Also enables latent space interpolation.

---

### Flow Matching {#flow-matching}

Flow Matching (Lipman 2022, used in SD3, Flux, Stable Video) replaces the noisy Markov chain with a simple **straight-line path** from noise $\mathbf{x}_1 \sim \mathcal{N}(0,I)$ to data $\mathbf{x}_0$:

$$
\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1, \qquad t \in [0, 1]
$$

The target velocity (the "flow") is trivially:

$$
\mathbf{u}_t = \mathbf{x}_1 - \mathbf{x}_0
$$

Loss: train the network $v_\theta(\mathbf{x}_t, t)$ to predict this velocity:

$$
\mathcal{L}_{FM} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1}\!\left[\left\|v_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0)\right\|^2\right]
$$

Advantages over DDPM: straight trajectories → fewer ODE steps needed; simpler training objective; better high-frequency detail. Converges faster.

---

### Major Diffusion Models {#diffusion-models-list}

| Model | Year | Backbone | Key advance |
|---|---|---|---|
| **DDPM** | 2020 | U-Net | First practical score-matching framework |
| **DALL-E 2** | 2022 | U-Net | CLIP embeddings as condition |
| **Imagen** | 2022 | U-Net | T5 text encoder; cascaded super-resolution |
| **Stable Diffusion 1.x** | 2022 | Latent U-Net | Open; latent diffusion |
| **SDXL** | 2023 | Latent U-Net | Larger; multi-aspect; 2-stage refinement |
| **eDiff-I** | 2023 | U-Net | Ensemble of specialist models per timestep |
| **DiT** | 2023 | ViT | Transformer backbone; scales better |
| **SD3** | 2024 | MM-DiT | Flow matching; text/image dual stream |
| **Flux** | 2024 | MM-DiT | Black Forest Labs; best open text-to-image |
| **DALL-E 3** | 2023 | Unknown | Recaptioned training data; very prompt-faithful |

---

## Video Models {#video-models}

### Challenges of Video {#video-challenges}

Video = images + time. Three extra challenges vs images:

1. **Temporal consistency**: frames must be coherent; objects can't flicker or teleport
2. **Motion modelling**: the model must represent not just appearance but how things move
3. **Compute**: a 10-second 24fps video has 240 frames. Naive ViT on each frame at 576 tokens/frame = 138,240 tokens. Infeasible.

---

### Architectures: 3D CNN → Transformer → Diffusion {#video-architectures}

**3D CNN (C3D, I3D, SlowFast):**

Extend 2D conv to 3D: kernel $k_t \times k_h \times k_w$. Captures local spatiotemporal patterns. SlowFast uses two pathways: slow (low frame rate, high spatial detail) + fast (high frame rate, low spatial detail). Efficient but limited temporal range.

**Video Transformers (TimeSformer, ViViT):**

Apply ViT to video. Two attention decompositions:
- **Factorised attention (TimeSformer)**: spatial attention then temporal attention per block — $O(N_s^2 + N_t^2)$ instead of $O((N_s N_t)^2)$
- **Tubelet embedding (ViViT)**: extract spatiotemporal tubes instead of frame-by-frame patches

**Video Diffusion:**

Extend latent diffusion to video. Inflate 2D U-Net: replace 2D convs with 3D (or + temporal conv layers); add temporal attention layers. Key designs:

- **Temporal U-Net (Video LDM)**: spatial and temporal layers interleaved
- **Joint space-time attention (Sora)**: full 3D attention over space and time; patch video into spacetime tokens
- **Causal video VAE**: encode video into spacetime latent; first frame exact, subsequent frames conditioned on previous → enables long video generation by streaming

**Sora's key architectural insight:** treat a video as a sequence of *spacetime patches* (just as ViT treats images as spatial patches). A DiT operating on these spacetime tokens can handle variable-length, variable-resolution video within a single unified architecture.

---

### Video Loss Functions {#video-losses}

**Frame-level reconstruction:**

$$
\mathcal{L}_{rec} = \frac{1}{T}\sum_{t=1}^{T} \|\hat{\mathbf{x}}_t - \mathbf{x}_t\|^2
$$

**Temporal consistency loss:**

Penalise inter-frame variation not present in ground truth:

$$
\mathcal{L}_{temp} = \sum_{t=1}^{T-1}\|\hat{\mathbf{x}}_{t+1} - \hat{\mathbf{x}}_t - (\mathbf{x}_{t+1} - \mathbf{x}_t)\|^2
$$

**Optical flow warping loss:**

Compute optical flow $\mathbf{F}_{t \to t+1}$ from ground truth; warp predicted frame; compare:

$$
\mathcal{L}_{warp} = \left\|\hat{\mathbf{x}}_{t+1} - \mathcal{W}(\hat{\mathbf{x}}_t, \mathbf{F}_{t \to t+1})\right\|^2
$$

**Video diffusion noise loss:**

Same as image diffusion but applied to a video latent $\mathbf{z}_{1:T}$:

$$
\mathcal{L}_{video} = \mathbb{E}_{t,\boldsymbol{\epsilon}}\!\left[\left\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_{1:T}^{(t)}, t, c)\right\|^2\right]
$$

**Contrastive video-text (CLIP4Clip, VideoCLIP):**

Same InfoNCE as image-text CLIP, but video frames are aggregated (mean pool or attention) before similarity.

---

### Major Video Models {#video-models-list}

| Model | Org | Architecture | Notable |
|---|---|---|---|
| **CogVideo** | Tsinghua | Transformer | First open text-to-video |
| **Make-A-Video** | Meta | Inflated U-Net | Image model → video via temporal layers |
| **Imagen Video** | Google | Cascaded diffusion | Hierarchical spatial + temporal SR |
| **Gen-2** | Runway | Latent diffusion | Commercial; image-to-video |
| **Stable Video Diffusion** | Stability AI | Temporal U-Net | Open; image-to-video; flow matching |
| **Sora** | OpenAI | Spacetime DiT | Long, high-fidelity, variable-length video |
| **Wan** | Alibaba | DiT | Best open text-to-video (2025) |
| **Mochi-1** | Genmo | DiT | High motion quality; open |
| **CogVideoX** | Tsinghua | DiT + Expert Transformer | Strong open model |
| **Kling** | Kuaishou | Unknown | High physical realism |

---

## World Models {#world-models}

### What a World Model Is {#world-model-what}

A world model is a model that answers: **"Given the current state and an action I take, what comes next?"**

A discriminative perception model (VLM, classifier) describes what the world *is*. A world model describes how the world *evolves*. This distinction is critical for planning and control:

- A VLM can describe a chess board but not simulate a game
- A world model can simulate moves and evaluate positions for lookahead

Formally, a world model learns:

$$
p(s_{t+1}, r_t \mid s_t, a_t)
$$

where $s_t$ is the state, $a_t$ is the action, $r_t$ is the reward, and $s_{t+1}$ is the next state.

---

### State, Dynamics, Reward {#world-model-components}

| Component | What it models | Parametrisation |
|---|---|---|
| **Encoder** $e_\phi$ | Raw obs $o_t$ → compact state $s_t$ | CNN, ViT |
| **Dynamics model** $f_\theta$ | $(s_t, a_t) \to s_{t+1}$ | MLP, RNN, Transformer |
| **Reward predictor** $r_\psi$ | $s_t \to \hat{r}_t$ | MLP |
| **Value function** $V_\xi$ | $s_t \to \hat{V}(s_t)$ | MLP |
| **Policy** $\pi_\omega$ | $s_t \to a_t$ | MLP, actor-critic |
| **Decoder** $d_\zeta$ (optional) | $s_t \to \hat{o}_t$ | CNN transpose, diffusion |

Having a decoder is not required — compact latent models (Dreamer) don't reconstruct every frame. The decoder loss helps learn good representations but the policy operates in latent space.

---

### World Model Loss Functions {#world-model-losses}

**Observation reconstruction loss:**

$$
\mathcal{L}_{rec} = -\mathbb{E}\!\left[\log p_\zeta(o_t \mid s_t)\right]
$$

For images: pixel-level MSE or perceptual loss. For discrete observations: cross-entropy.

**Dynamics loss (RSSM in Dreamer):**

The Recurrent State Space Model (RSSM) maintains a deterministic recurrent state $h_t$ and a stochastic state $z_t$:

$$
h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1}), \qquad z_t \sim q_\phi(z_t \mid h_t, o_t)
$$

Prior (predicted) distribution: $p_\theta(z_t \mid h_t)$. Posterior (from observation): $q_\phi(z_t \mid h_t, o_t)$.

KL divergence between posterior and prior:

$$
\mathcal{L}_{dyn} = \mathbb{E}\!\left[D_{KL}\!\left(q_\phi(z_t \mid h_t, o_t) \,\|\, p_\theta(z_t \mid h_t)\right)\right]
$$

This term forces the dynamics model to predict what comes next from latents alone.

**KL balancing (DreamerV2):** separately scale terms inside the KL:

$$
\mathcal{L}_{KL} = \alpha \cdot D_{KL}[\text{sg}(q) \| p] + (1-\alpha) \cdot D_{KL}[q \| \text{sg}(p)]
$$

Stop-gradient (`sg`) on one side at a time — 80% weight on pushing $p$ toward $q$, 20% on pushing $q$ toward $p$. Prevents posterior collapse.

**Reward prediction loss:**

$$
\mathcal{L}_{rew} = -\mathbb{E}\!\left[\log p_\psi(r_t \mid s_t)\right]
$$

**Actor-Critic loss (in imagination):**

Dreamer generates imagined rollouts from the world model (no real env interaction), then trains actor and critic on these imagined trajectories:

$$
\mathcal{L}_{actor} = -\mathbb{E}_\pi\!\left[\sum_\tau \lambda^\tau V_\xi(s_{t+\tau})\right]
$$

$$
\mathcal{L}_{critic} = \mathbb{E}\!\left[\left(V_\xi(s_t) - \text{sg}(\hat{V}_t^\lambda)\right)^2\right]
$$

where $\hat{V}_t^\lambda$ is the $\lambda$-return (mixture of n-step returns).

**Video prediction loss (for large-scale world models like Genie, GameNGen):**

These treat world models as video generation problems. Loss = diffusion noise loss or next-frame cross-entropy:

$$
\mathcal{L}_{next-frame} = -\sum_t \log p_\theta(f_{t+1} \mid f_{\leq t}, a_{\leq t})
$$

---

### Lineages: Compact Latent → Video → LLM-Based {#world-model-lineages}

**Lineage 1 — Compact Latent Dynamics (RL-focused)**

Goal: efficient RL by planning in a small latent space. Minimise real-environment steps.

- **RSSM / PlaNet** (2019): latent dynamics model; plan with CEM in latent space
- **Dreamer** (2020): actor-critic trained on imagined rollouts; pixel-perfect Atari
- **DreamerV2** (2021): discrete latent states; matches DQN on Atari with 200× fewer env steps
- **DreamerV3** (2023): single set of hyperparameters across all domains (Atari, DMC, Minecraft); scales to 100B+ imagination steps

**Lineage 2 — Video World Models (physics + simulation)**

Goal: generate photorealistic video of a simulated world that responds to actions.

- **GameNGen** (Google, 2024): a diffusion model (SD) finetuned to generate DOOM gameplay frames conditioned on action history. Plays at 20fps. First neural "game engine"
- **Genie** (Google DeepMind, 2024): learns interactive environments from unlabelled internet video. Infers latent actions; generates next frame from any action. No action labels needed
- **Oasis** (Decart, 2024): real-time Minecraft world model; 360p at 20fps; no game engine

**Lineage 3 — Language Model World Models**

Goal: model world states as token sequences; planning = generation.

- **UniSim** (2023): uses diffusion to simulate diverse real-world observations for embodied agents
- **GROOT / VPT** (OpenAI): supervised imitation on Minecraft video → agent policies
- **GAIA-1** (Wayve, 2023): generative world model for autonomous driving; generates future video from ego-actions and text commands
- **DriveDreamer** / **DriveX**: autoregressive video diffusion for driving simulation

---

### Major World Models {#world-model-list}

| Model | Domain | Core mechanism |
|---|---|---|
| **DreamerV3** | RL (general) | RSSM + actor-critic in imagination |
| **IRIS** | Atari | Transformer world model + MuZero-style planning |
| **TWM** | Atari | Transformer dynamics; offline RL |
| **GameNGen** | Video games (DOOM) | Diffusion conditioned on action history |
| **Genie** | Internet video | Latent action inference + video generation |
| **Oasis** | Minecraft | Real-time video diffusion world model |
| **GAIA-1** | Autonomous driving | Autoregressive video generation from ego-actions |
| **UniSim** | Embodied AI | Diffusion-based real-world action simulator |
| **Robotic Transformer 2** | Robotics | VLM policy; fine-grained control |

---

### World Model Failure Modes {#world-model-failure}

| Failure | Cause | Mitigation |
|---|---|---|
| **Compounding error** | Small per-step dynamics error grows exponentially over rollout horizon | Short imagination horizons; ensemble disagreement as uncertainty signal |
| **Spurious spurious correlations** | Model learns background statistics instead of causal structure | Causal intervention training; data diversity |
| **Reward hallucination** | Imagined rewards don't match true env rewards | Pessimistic value estimates; model uncertainty penalty |
| **Out-of-distribution latents** | Agent exploits model errors (model hacking) | Curiosity/novelty penalties; conservative planning |
| **No physical common sense** | Compact latent models don't understand gravity, rigid bodies | Video-based models with physics supervision |

---

## Multimodal Model Evaluation {#multimodal-eval}

### VLM Benchmarks {#vlm-benchmarks}

(Covered in detail in the [VLM Evaluation section above](#vlm-evaluation).)

Quick reference table of top benchmarks by capability:

| Capability | Primary Benchmark | Secondary |
|---|---|---|
| General understanding | MMBench, MMStar | MME |
| Expert reasoning | MMMU, MathVista | ScienceQA |
| OCR / documents | DocVQA, OCRBench | TextVQA |
| Charts / data | ChartQA, InfoVQA | FigureQA |
| Hallucination | POPE, HallusionBench | MMHAL |
| Grounding | RefCOCO, Flickr30K | PointQA |
| Video | Video-MME, EgoSchema | MVBench |

---

### Generation Evaluation {#generation-eval}

**FID — Fréchet Inception Distance:**

$$
\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

Computes distance between feature distributions of real and generated images in InceptionV3 feature space. Lower is better. Sensitive to both quality and diversity. Requires large sample sizes (≥10K) for stable estimates.

**Inception Score (IS):**

$$
\text{IS} = \exp\!\left(\mathbb{E}_{\mathbf{x}}\!\left[D_{KL}(p(y \mid \mathbf{x}) \,\|\, p(y))\right]\right)
$$

High IS = generated images are recognisable (high $p(y \mid \mathbf{x})$) and diverse (high entropy $p(y)$). Doesn't compare to real data — FID is preferred.

**CLIP Score — text-image alignment:**

$$
\text{CLIP Score} = \mathbb{E}\!\left[\cos(\mathbf{e}_{image}, \mathbf{e}_{text})\right] \times 100
$$

Measures how well generated images match their text prompt. Range roughly 0–35; higher is better. Used alongside FID: FID measures realism, CLIP score measures faithfulness.

**DINO-based diversity:** compute pairwise cosine distance between DINOv2 features of generated images. Low variance = mode collapse.

**Human evaluation dimensions for image generation:**

| Dimension | Question |
|---|---|
| Prompt faithfulness | Does it depict what was asked? |
| Photorealism | Could it be a real photograph? |
| Aesthetic quality | Is it visually pleasing? |
| Coherence | Are objects physically consistent? |
| Text rendering | If text was requested, is it legible? |

---

### Video Model Evaluation {#video-eval}

| Metric | Measures | Formula / Method |
|---|---|---|
| **FVD** (Fréchet Video Distance) | Video quality + temporal coherence | Same as FID but on I3D spatiotemporal features |
| **PSNR** | Pixel reconstruction fidelity | $10\log_{10}(\text{MAX}^2 / \text{MSE})$ |
| **SSIM** | Structural similarity | Luminance × contrast × structure |
| **LPIPS** | Perceptual distance | VGG/AlexNet feature MSE |
| **CLIPSIM** | Text-video alignment | Mean CLIP cosine over frames |
| **Action Accuracy** | World model fidelity | Does the generated frame match intended action? |
| **EgoSchema / Video-MME** | Video understanding | Accuracy on temporal Q&A |

**Temporal consistency**: compute optical flow between consecutive generated frames; plot flow magnitude variance — high variance = flickering.

---

### World Model Evaluation {#world-model-eval}

World models are evaluated differently depending on lineage:

**Compact latent (RL) models:**

- **Downstream task return**: train a policy using the world model; measure real-environment cumulative reward
- **Data efficiency**: return as a function of real-environment steps (not imagination steps)
- **Open-loop prediction quality**: MSE / SSIM of predicted observation sequences (doesn't capture compounding error)
- **Model hacking frequency**: how often does the agent find world-model exploits?

**Video world models:**

- **Action controllability**: given an action, does the generated video reflect it? (human or classifier evaluation)
- **FVD on held-out trajectories**
- **Physical plausibility score**: human rating on gravity, collision, occlusion

**LLM-based world models:**

- **Playability**: can a human play the simulated game and find it coherent? (GameNGen)
- **Downstream policy performance**: does training on simulated data transfer to real environments?

---

### Evaluation Pitfalls {#eval-pitfalls}

| Pitfall | Detail |
|---|---|
| **FID not robust to number of samples** | Computing FID on 1K samples vs 50K gives very different numbers — always report sample count |
| **CLIP score doesn't catch spatial errors** | "A red ball to the left of a blue cube" — CLIP can't verify spatial relationships reliably |
| **Benchmark saturation** | POPE F1 > 90% for most frontier models; no longer discriminates |
| **Human eval is expensive and noisy** | Inter-rater agreement is often low; always report Krippendorff's α or Fleiss κ |
| **Metric-gaming** | Training on FID-like signals leads to images that fool InceptionV3 but look wrong to humans |
| **Missing diversity in VQA** | Yes/no questions can be answered 65% correctly by always saying "yes" (VQAv2 prior) |
| **Video FVD ignores text alignment** | A perfectly looping video scores well on FVD but is useless for text-conditioned generation |
