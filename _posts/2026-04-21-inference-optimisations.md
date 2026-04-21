---
title: "Inference Optimisations"
date: 2026-04-21
description: "Techniques for making LLM inference fast and memory-efficient — quantisation, KV cache management, batching, speculative decoding, and hardware-aware kernel design, with the reasoning behind every design decision."
tags: [ml-systems, inference, llm, optimisation]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#roofline">Roofline Analysis</a>
      <ul class="post-toc-sublist">
        <li><a href="#hardware-ceilings">Hardware Ceilings</a></li>
        <li><a href="#arithmetic-intensity">Arithmetic Intensity</a></li>
        <li><a href="#roofline-model">The Roofline Model</a></li>
        <li><a href="#transformer-accounting">Transformer Layer Accounting</a></li>
        <li><a href="#glue-ops">Glue Ops: The Hidden Bandwidth Tax</a></li>
        <li><a href="#three-bottlenecks">Three Bottlenecks, Three Levers</a></li>
        <li><a href="#mfu">MFU and Profiling Workflow</a></li>
      </ul>
    </li>
    <li><a href="#arch-bottlenecks">Architectures to Break Bottlenecks</a>
      <ul class="post-toc-sublist">
        <li><a href="#sparsity-taxonomy">Four Kinds of Sparsity</a></li>
        <li><a href="#moe-ffn">MoE — Breaking the FFN Bottleneck</a></li>
        <li><a href="#mla-kv">MLA — Breaking the KV Bottleneck</a></li>
        <li><a href="#sparse-attention">Sparse Attention — Breaking L² at Long Context</a></li>
        <li><a href="#hybrid-stacks">Hybrid Stacks</a></li>
        <li><a href="#arch-summary">Putting It Together</a></li>
      </ul>
    </li>
    <li><a href="#quantisation">Quantisation</a>
      <ul class="post-toc-sublist">
        <li><a href="#weight-quant">Weight Quantisation</a></li>
        <li><a href="#activation-quant">Activation Quantisation</a></li>
        <li><a href="#kv-quant">KV Cache Quantisation</a></li>
      </ul>
    </li>
    <li><a href="#kv-cache">KV Cache Management</a>
      <ul class="post-toc-sublist">
        <li><a href="#pagedattention">PagedAttention</a></li>
        <li><a href="#prefix-caching">Prefix Caching</a></li>
        <li><a href="#kv-eviction">KV Eviction</a></li>
      </ul>
    </li>
    <li><a href="#batching">Batching Strategies</a>
      <ul class="post-toc-sublist">
        <li><a href="#continuous-batching">Continuous Batching</a></li>
        <li><a href="#chunked-prefill">Chunked Prefill</a></li>
      </ul>
    </li>
    <li><a href="#speculative">Speculative Decoding</a>
      <ul class="post-toc-sublist">
        <li><a href="#draft-verify">Draft-Verify</a></li>
        <li><a href="#model-free">Model-Free Methods</a></li>
      </ul>
    </li>
    <li><a href="#kernel-opt">Kernel-Level Optimisations</a>
      <ul class="post-toc-sublist">
        <li><a href="#flashattention">FlashAttention</a></li>
        <li><a href="#flash-decoding">Flash-Decoding</a></li>
        <li><a href="#kernel-fusion">Kernel Fusion</a></li>
      </ul>
    </li>
    <li><a href="#disaggregation">Prefill-Decode Disaggregation</a></li>
    <li><a href="#model-compression">Model Compression</a>
      <ul class="post-toc-sublist">
        <li><a href="#speculative-decoding-depth">Speculative Decoding: Depth</a></li>
        <li><a href="#pruning-depth">Pruning</a></li>
        <li><a href="#minitron">Minitron: Prune then Distill</a></li>
        <li><a href="#elastic-models">Elastic Models (Flextron/LlamaFlex)</a></li>
        <li><a href="#nas-lana-puzzle">NAS: LANA and Puzzle</a></li>
      </ul>
    </li>
    <li><a href="#serving-frameworks">Serving Frameworks & System Architecture</a>
      <ul class="post-toc-sublist">
        <li><a href="#serving-system-layers">Serving System Layers</a></li>
        <li><a href="#scheduler-design">Scheduler Design</a></li>
        <li><a href="#framework-landscape">Framework Landscape</a></li>
        <li><a href="#pd-disaggregation-systems">PD Disaggregation in Systems</a></li>
        <li><a href="#ep-long-context">Expert Parallelism & Long Context</a></li>
        <li><a href="#observability-ops">Observability & Operations</a></li>
      </ul>
    </li>
    <li><a href="#vllm-internals">vLLM Internals</a>
      <ul class="post-toc-sublist">
        <li><a href="#cpu-overheads">Minimising CPU Overheads</a></li>
        <li><a href="#cuda-graphs">CUDA Graphs & Piecewise Execution</a></li>
        <li><a href="#parallelism-vllm">Parallelism in vLLM</a></li>
        <li><a href="#hybrid-memory">Hybrid Memory Allocator</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## Overview
{: #overview}

LLM inference is fundamentally different from training. The forward pass through a 70B model must complete in tens of milliseconds to meet latency SLOs, yet the model's 140 GB of weights must be loaded from HBM on every token generated. The bottleneck is almost never arithmetic — it is **memory bandwidth**.

**Why the bottleneck is bandwidth, not compute.** A single A100 GPU has ~312 TFLOPS of BF16 compute and ~2 TB/s HBM bandwidth. Generating one token from a 70B model requires loading 140 GB of weights. At 2 TB/s that takes 70 ms — 14 tokens/second maximum, and the GPU's tensor cores are barely exercised. With 4× A100s (combined ~8 TB/s bandwidth), this becomes 17.5 ms/token ceiling. To keep the GPU's tensor cores usefully busy during decode, you'd need to run thousands of tokens of compute for every weight byte loaded — which only happens with very large batch sizes.

**The arithmetic intensity threshold.** Arithmetic intensity = FLOPs / bytes of memory traffic. For each linear layer `y = Wx` with `W ∈ ℝ^(d×d)`:
- Memory traffic: `d²` bytes to load W (dominant), `d` bytes for x and y
- Compute: `2d²` FLOPs (one multiply + one add per element of W)
- Arithmetic intensity = `2d²` / `d²` = 2 FLOP/byte (for a single token)

The A100's **roofline** (where compute and bandwidth are equally saturated) is ~312 TFLOPS / 2 TB/s = 156 FLOP/byte. Decode at batch size 1 runs at intensity 2 — 78× below the roofline. You're using 1.3% of compute capacity. Only at batch size 78 do you hit the roofline and become compute-bound.

This analysis drives the entire optimisation landscape: reduce bytes per weight (quantisation), reuse loaded weights across more tokens (larger batches), reduce the number of weight reads (speculative decoding), and cut memory traffic within a forward pass (FlashAttention, kernel fusion).

<div class="post-flow post-flow--horizontal" role="group" aria-label="Three inference bottlenecks">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Memory bandwidth — loading weights every token</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">KV cache memory — grows linearly with context × batch</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Sequential dependency — tokens generated one at a time</span></li>
  </ol>
</div>

> **Interview question:** You have a 7B model running on a single A100 (80GB, 2TB/s HBM bandwidth). At what batch size does inference become compute-bound rather than memory-bound, and why does this matter for latency vs throughput?
>
> *A 7B model has ~7B × 2 bytes = 14GB of weights (BF16). Each forward pass loads all weights once: 14GB at 2TB/s = 7ms theoretical minimum per step. The A100 has 312 TFLOPS BF16. One decode step at batch size B requires ~2 × 7B × B = 14B × B FLOPs. Time to compute: 14B × B / 312T = B × 45µs. Memory time: 7ms constant. You hit compute-bound when compute time > memory time: B × 45µs > 7ms → B > 155. So at batch size ~155, inference transitions from memory-bound (latency ∝ constant, throughput ∝ B) to compute-bound (latency ∝ B, throughput ∝ constant). This matters: for a single-user chatbot, B=1, you're deep in memory-bound regime — quantisation helps enormously. For a high-throughput API serving many users, you want B ≥ 155 to amortise weight loads — at that point quantisation helps less (you're compute-bound), and batching strategy matters more.*

---

## Roofline Analysis
{: #roofline}

The roofline model is the single most important mental tool for performance engineering on modern accelerators. It gives you a **principled lower bound on runtime** and tells you exactly which resource is your limiting factor — compute or memory bandwidth — before you run a single profiler. Without this framework, performance work is guesswork.

### Hardware Ceilings
{: #hardware-ceilings}

Every accelerator has two fundamental peak capabilities:

| Resource | Symbol | Unit | H100 SXM example |
|---|---|---|---|
| Peak compute throughput | `C_peak` | FLOPs/s | ~1 PFLOPs/s (BF16 tensor cores) |
| Peak HBM bandwidth | `BW_HBM` | bytes/s | ~3.35 TB/s |

Any kernel that performs `FLOPs_op` math and moves `bytes_op` data faces two unavoidable time lower bounds:

```
T_math ≥ FLOPs_op / C_peak        (compute floor)
T_mem  ≥ bytes_op  / BW_HBM       (bandwidth floor)

⟹  T_op ≥ max(T_math, T_mem)
```

This is not an approximation — it's a physical ceiling. No kernel optimisation can beat the hardware. The practical question is: which bound is the active constraint for your operation?

**Why bytes moved means HBM.** Registers and L1 are ~256 KB/SM; L2 is ~50 MB GPU-wide. LLM weight matrices and KV caches are tens to hundreds of GB — they live in HBM and must be streamed through the memory hierarchy. Any kernel that cannot reuse data in registers or L1 will be limited by HBM bandwidth. This is why we focus on HBM traffic exclusively.

---

### Arithmetic Intensity
{: #arithmetic-intensity}

**Arithmetic intensity (AI)** is the key summary statistic that predicts which ceiling you hit:

```
AI = FLOPs_op / bytes_op     [FLOPs/byte]
```

- **High AI** → lots of compute reuse per byte loaded → more likely compute-bound
- **Low AI** → streaming data with little reuse → bandwidth-bound

The **critical intensity** (the "knee" of the roofline) is:

```
AI* = C_peak / BW_HBM
```

For H100: `AI* ≈ 1 PFLOPs/s ÷ 3.35 TB/s ≈ 300 FLOPs/byte`.

- If `AI < AI*`: bandwidth-bound — best achievable perf ≈ `BW_HBM × AI`
- If `AI > AI*`: compute-bound — best achievable perf ≈ `C_peak`

**Linear layer example.** For `Y = XW` with `X ∈ ℝ^(B_tok × D)`, `W ∈ ℝ^(D × F)`:

```
FLOPs    ≈ 2·B_tok·D·F
bytes    ≳ b·(B_tok·D  +  D·F  +  B_tok·F)
            read X      read W    write Y
```

Arithmetic intensity:
```
AI_linear ≈ 2·B_tok·D·F / [b·(B_tok·D + D·F + B_tok·F)]
```

Two regimes dominate:

| Regime | Condition | AI simplifies to | BF16 result |
|---|---|---|---|
| Small batch (weights dominate bytes) | `B_tok ≪ D` | `2·B_tok / b` | `AI ≈ B_tok` |
| Large batch (activations dominate bytes) | `B_tok ≫ D` | `2·D·F / [b·(D+F)]` | Saturates at `~D` |

The critical insight: at batch size 1 decode, `AI ≈ 1 FLOPs/byte` in BF16 — over **300× below the knee**. The GPU's tensor cores are delivering ~0.3% of their rated throughput. This is why every inference optimisation ultimately comes back to reducing bytes or increasing batch.

> **Interview question:** A 7B model runs at 14 tokens/second on one A100 at batch=1. Your team proposes INT4 quantisation. A colleague says "just increase batch size to 128." Which is right, and what does roofline analysis tell you?
>
> *Both help, but for different reasons. At batch=1 BF16: AI ≈ 1 FLOP/byte, far below AI\*=156 FLOP/byte — bandwidth-bound. INT4 halves bytes (AI doubles to ~2 still bandwidth-bound but now 2× faster, directly reducing TTFT latency). At batch=128 BF16: AI ≈ 128 FLOP/byte ≈ AI\*, approaching compute-bound. Beyond this, more batch doesn't reduce per-token memory time — it just increases parallelism. Optimal: apply INT4 first (cuts latency at all batch sizes) then batch as high as memory allows (increases throughput). The two techniques are complementary: INT4 moves the knee from B=155 to B=78, giving compute-bound throughput at half the batch size.*

---

### The Roofline Model
{: #roofline-model}

```
Performance (FLOPs/s)
      │
      │          Compute roof ──────────────── C_peak
      │         /
      │        / (slope = BW_HBM)
      │       /         ← bandwidth-limited region
      │      /  knee (AI*)
      │     /
      │    /
      └────────────────────────────────────────
                  Arithmetic Intensity (FLOPs/byte)
```

**How to use it:**

1. Compute `AI* = C_peak / BW_HBM` (hardware constant)
2. Compute `AI_op = FLOPs_op / bytes_op` (algorithm + shapes)
3. Compare:
   - `AI_op < AI*` → bandwidth-bound; fix: reduce bytes or increase reuse
   - `AI_op > AI*` → compute-bound; fix: better kernels, operator fusion, more parallelism
   - Both far from peak → overhead-bound: kernel launch overhead, poor fusion, stalled pipeline

**The decision rule is shape-dependent.** For the same kernel, increasing `B_tok` (adding more tokens per batch) increases weight reuse, moving the operator rightward on the roofline. This is why large-batch training is compute-efficient while single-token decode is bandwidth-bound.

---

### Transformer Layer Accounting
{: #transformer-accounting}

A single decoder block has two dominant compute components — attention projections and the MLP — plus "glue" ops (norms, residuals, softmax). Let's count FLOPs and bytes for each using the notation: `B` = batch, `L` = sequence length, `D` = model width, `N` = query heads, `H` = head dim (`D = N·H`), `N_KV` = KV heads, `F` = MLP hidden dim (typically `F ≈ 4D`), `b` = bytes per element (2 for BF16).

**Attention projections (Q, K, V, O — four GEMMs of shape `(BL × D)(D × D)`):**

```
FLOPs_proj     ≈ 8·B·L·D²           (4 matmuls × 2·BL·D²)
bytes_proj     ≳ b·(8·B·L·D + 4·D²) (activations + weights)
AI_proj        ≈ 2·B·L / b           (when B·L ≪ D)
```

The `4D²` weight term dominates at small batch — weight reads, not activations, are the bottleneck. Increasing `B·L` increases reuse of the `4D²` weights, pushing AI upward.

**Attention quadratic (`QK^T` and `PV` — the `L²` terms):**

Naïve implementation materialises `S ∈ ℝ^{B×N×L×L}` and `P = softmax(S)` in HBM:

```
FLOPs_quad     ≈ 4·B·L²·D           (2 matmuls × 2·B·L²·D)
bytes_quad     ≳ b·(4·B·L·D + 4·B·N·L²)   ← the L² traffic term
AI_quad_naive  ≈ L·H / [b·(H + L)]
               ≈ H/b   when L ≫ H           (e.g., H=128: AI ≈ 64 for BF16)
```

**The devastating implication:** naive quadratic attention has AI ≈ H/b regardless of sequence length or batch size. Doubling L doubles both FLOPs *and* bytes together, keeping AI flat. The operator is stuck under the memory roof with no shape knob to fix it — this is why FlashAttention was a breakthrough.

**FlashAttention (tiled, no S/P materialisation):**

```
bytes_quad_flash ≳ b·(4·B·L·D)      (only Q, K, V, A — no S or P)
AI_quad_flash    ≈ 4·B·L²·D / (4·b·B·L·D)  =  L/b
```

For BF16 and L=4096: AI ≈ 2048 FLOPs/byte — far above AI\*=300. FlashAttention moves quadratic attention from deeply bandwidth-bound to compute-bound by fusing the S and P ops and keeping them in shared memory.

**MLP (gated SwiGLU, three matmuls: up, gate, down):**

```
FLOPs_MLP  ≈ 6·B·L·D·F  ≈ 24·B·L·D²  (when F ≈ 4D)
bytes_MLP  ≳ b·(3·B·L·D + 6·B·L·F + 3·D·F)
AI_MLP     ≈ 2·B·L/b    (small batch)
           ≈ 8D/(9b)    (large batch, F≈4D)
```

MLP dominates **parameter count**: with F≈4D, `P_MLP ≈ 12D²` vs `P_attn ≈ 4D²` per layer. At large batch, MLP AI scales with D — wider models become more compute-efficient per layer.

**Per-layer summary:**

| Component | Params | Forward FLOPs | Bytes moved (naive) | AI (small batch) |
|---|---|---|---|---|
| Attn projections | 4D² | 8·BL·D² | b·(8BLD + 4D²) | ≈ BL |
| Attn quadratic | 0 | 4·BL²·D | b·(4BLD + 4·BNL²) | ≈ H/b (capped) |
| MLP (gated) | 3DF | 6·BL·DF | b·(3BLD + 6BLF + 3DF) | ≈ BL |
| Glue ops | — | O(BLD) | O(b·BLD) | << 1 |

**When does L² attention dominate over the linear MLP?**

```
FLOPs_L²   / FLOPs_linear  =  4·B·L²·D  /  32·B·L·D²  =  L / (8D)
```

L² becomes dominant only when `L ≳ 8D`. For D=7000 (typical 7B model), that's L ≳ 56,000 tokens. At standard context lengths (4K–32K), the **MLP dominates compute** — attention dominates memory bandwidth.

---

### Glue Ops: The Hidden Bandwidth Tax
{: #glue-ops}

LayerNorm, residual adds, and softmax look cheap — they're simple elementwise or reduction ops. But they are **archetypal bandwidth-bound operations** with AI so low that no hardware can make them compute-bound. They are invisible in naive FLOPs counts but real in wall-clock time.

**Residual add** (`Y = X + R`, `X, R, Y ∈ ℝ^{B×L×D}`):

```
FLOPs_add  ≈ B·L·D          (one add per element)
bytes_add  ≳ 3·b·B·L·D      (read X, read R, write Y)
AI_add     ≈ 1/3b            →  BF16: AI ≈ 0.17 FLOPs/byte
```

AI of 0.17 is essentially zero on any modern GPU. Residual adds are pure bandwidth cost — if they appear as significant time in your profiler, it means the surrounding kernels were not fused.

**LayerNorm / RMSNorm** (operates over D features per token):

```
bytes_LN   ≳ κ·b·B·L·D      where κ ≥ 2 (multiple passes for mean, variance, normalize)
AI_LN      ≈ 1/κb            (small constant regardless of shapes)
```

LayerNorm is an example of κ > 1 streaming — every naïve implementation reads the input multiple times. RMSNorm (no mean computation) has κ=1 minimum, which is one reason it's faster: fewer passes reduce the HBM traffic multiplier.

**Softmax** (attention, applied over L scores per query):

```
FLOPs_softmax ≈ c·L     (max, exp, sum, divide — small constant c)
bytes_softmax ≳ 2·b·L   (read scores, write probs)
AI_softmax    ≈ c/(2b)  (constant, independent of L)
```

Scaling L does not improve softmax AI — FLOPs and bytes both grow as L, leaving AI flat. Standalone softmax is always under the memory roof.

**Why fusion matters:** if LayerNorm → attention scores → softmax → output projection are fused into a single kernel, the intermediate results stay in registers or shared memory — no HBM round-trips. Effective AI for the fused kernel is much higher than any individual op. This is precisely what FlashAttention exploits: fusing the score computation, softmax, and value aggregation eliminates the `4·BNL²` bytes of intermediate attention matrix traffic.

> **Interview question:** You profile a 30B model and find 40% of inference time is spent in LayerNorm and residual adds, not attention or MLP. What does this tell you and what do you do?
>
> *This is a classic overhead-bound signature: the "big" ops (matmuls, attention) are fast, but the "glue" ops between them dominate because they are unfused — each writes an intermediate result to HBM, and the next op reads it back. AI ≈ 0.1–0.3 FLOPs/byte means these ops sit far left on the roofline, capped by memory bandwidth. Fix: operator fusion. Merge LayerNorm into the preceding matmul's output write (compute norm in the same pass that writes activations), and merge residual adds into attention output accumulation. PyTorch torch.compile or custom Triton kernels handle this. After fusion, these ops disappear from the profile — their bytes are amortised into the surrounding matmul's HBM traffic, which already has much higher AI.*

---

### Three Bottlenecks, Three Levers
{: #three-bottlenecks}

Combining the per-layer accounting, three headline bottlenecks emerge for dense Transformers. Which one dominates depends on model width D, context length L, and whether you're training or serving.

**Bottleneck 1 — MLP/FFN (compute + parameters):**

MLP dominates parameter count (`12D²` vs `4D²` for attention projections) and, at moderate context, dominates FLOPs. Every token pays for every parameter with no conditional computation.

| Symptom | Intervention |
|---|---|
| High GPU memory usage from weights | Quantisation (INT4/INT8 weights) |
| MFU limited by weight reads at small batch | Increase batch to improve weight reuse |
| Too expensive to serve large models | Mixture of Experts — activate only K of N experts per token |

**Bottleneck 2 — Attention KV bandwidth (inference at long context):**

During decode, the KV cache for all past tokens must be streamed from HBM on every step. KV cache grows as:

```
KV_bytes = n_layers × N_KV × H × L_ctx × b
```

For a 70B model (96 layers, N_KV=8 GQA, H=128, L=32K, BF16): 96 × 8 × 128 × 32768 × 2 ≈ **6.4 GB per request**. At batch=32, that's 204 GB just for KV cache — exceeding an 80GB A100 entirely.

| Symptom | Intervention |
|---|---|
| KV cache OOM at moderate batch/context | PagedAttention (eliminate fragmentation), GQA/MQA (reduce N_KV) |
| Slow decode at long context | KV quantisation (INT8/INT4 KV), eviction policies (H2O, StreamingLLM) |
| Redundant prefix recomputation | Prefix caching (RadixAttention) |

**Bottleneck 3 — L² attention (very long context):**

The quadratic attention matmuls (`4·BL²·D` FLOPs) start dominating when `L ≳ 8D`. For 7B models this is L > 56K; for 70B (D≈8192) this is L > 65K. At 128K+ context windows, L² attention becomes the primary compute cost.

| Symptom | Intervention |
|---|---|
| Compute time grows quadratically with context | FlashAttention (eliminates L² memory traffic) |
| Still too slow at very long context | Sparse attention (attend to local window + global tokens), linear attention hybrids |
| Flash helps but context still limited | Sliding window attention, hierarchical attention |

**Summary table:**

| Bottleneck | Active when | Primary symptom | Fix |
|---|---|---|---|
| MLP/FFN bandwidth | Small batch, any L | Low MFU, weight-read-bound | Quantisation, larger batch, MoE |
| KV cache pressure | Long context, large batch | OOM or slow decode | GQA, KV quant, PagedAttention |
| L² attention compute | L ≳ 8D (e.g., L > 56K for 7B) | Compute grows quadratically | FlashAttention, sparse/hybrid attention |

> **Interview question:** Training FLOPs = `6 × P × N_tok`. You want to halve training cost. An engineer proposes halving the model size P. Another proposes halving the training tokens N_tok. Roofline analysis says they're equivalent in FLOPs — are they equivalent in practice?
>
> *No — they have very different downstream effects. Halving P changes the model's AI: a smaller model has fewer weight bytes per forward pass, so at the same batch size it reaches the bandwidth-bound threshold at a lower batch size. Critically, a smaller model has lower quality ceiling — Chinchilla scaling laws show that both P and N_tok matter for the loss. Halving N_tok may leave significant capability on the table because the model never sees enough data to converge (the "undertrained" regime). Halving P with more tokens is often preferable for inference because a smaller model is cheaper to serve: lower memory, higher throughput per GPU. The FLOPs are the same but the deployed cost is very different. This is the "inference-optimised training" argument — train a smaller model for longer rather than a larger model for fewer tokens.*

---

### MFU and the Profiling Workflow
{: #mfu}

**Model FLOPs Utilisation (MFU)** is the standard metric for training and serving efficiency:

```
MFU = achieved_FLOPs_per_second / C_peak
```

MFU is a **symptom metric**, not a diagnosis. Low MFU could mean:
- Bandwidth-bound (AI < AI\*): improve batching, quantise, fuse kernels
- Overhead-bound: too many small kernels, poor operator fusion, kernel launch latency
- Communication-bound: AllReduce or KV transfer dominates step time

**Disciplined profiling workflow:**

```
1. Measure end-to-end: step time, TTFT, TBT, tokens/sec
2. Measure per-kernel: time, achieved FLOPs/s, achieved BW
3. Compute AI for the slow kernel (FLOPs / bytes)
4. Compare AI to AI* → classify as compute/bandwidth/overhead-bound
5. Form a hypothesis: "MLP matmuls are BW-bound because batch=8 < crossover ~155"
6. Apply the matching lever: increase batch, quantise, fuse
7. Re-measure — verify the hypothesis
```

If both achieved FLOPs/s *and* achieved BW are far from their respective ceilings, the bottleneck is overhead-bound: kernel launch overhead, poor scheduling, or stalled pipeline. The fix is fusion (fewer, larger kernels) or better occupancy.

**Training estimate:** Given training FLOPs ≈ `6·P·N_tok`, wall-clock training time is:

```
T_train ≈ 6·P·N_tok / (G × u × C_peak)
```

where G = number of GPUs and u = MFU achieved. Halving MFU doubles training time — which at millions-of-dollar scale is catastrophic. This is why kernel quality and distributed training efficiency are first-order concerns, not micro-optimisations.

> **Interview question:** A training run achieves 38% MFU on 512 H100s. Profiling shows attention is 22% of step time and AllReduce is 41%. Where do you invest engineering effort?
>
> *AllReduce at 41% of step time dominates — this is communication-bound, not compute or bandwidth-bound. Fix: (1) overlap AllReduce with backward computation using gradient bucketing (PyTorch DDP does this; verify it's actually overlapping by checking timeline), (2) switch from ring AllReduce to hierarchical tree AllReduce if nodes are connected by slower inter-node links (NVLink within node, InfiniBand across), (3) enable ZeRO Stage 1/2 (ReduceScatter + AllGather instead of full AllReduce; same total bytes but pipeline-able with backward), (4) use gradient compression (FP8 grads, top-K sparsification) with care for convergence impact. Attention at 22% of step time is secondary — FlashAttention is likely already in use; further gains require long-context specific work. After fixing communication, re-profile: the MFU ceiling is now the compute/bandwidth limit, not the communication limit.*

---

## Architectures to Break Bottlenecks
{: #arch-bottlenecks}

The previous section diagnosed three bottlenecks from first principles — MLP compute/params, KV cache bandwidth, and L² attention — and suggested generic fixes (quantisation, GQA, FlashAttention). This section goes one level deeper: **architectural interventions** that restructure the model itself to change the dominant FLOPs or bytes term at the source.

The framing is the same roofline equation:

```
T_total ≳ max( FLOPs / C_peak,  bytes_moved / BW_HBM )
```

Modern large models are not simply "bigger Transformers." They are **selective** about where they spend compute and bandwidth — using three classes of structural change:

- **Compute sparsity** (MoE) — reduce FLOPs by only activating a subset of parameters per token
- **State compression** (MLA, GQA) — reduce bytes by caching smaller representations
- **Long-context mechanisms** (sparse attention, hybrids) — reduce the effective L² term by attending to fewer past tokens per layer, or by eliminating attention entirely from most layers

---

### Four Kinds of Sparsity
{: #sparsity-taxonomy}

"Sparsity" is overloaded. Knowing which kind you're using tells you exactly what changes — FLOPs, bytes, cache layout, or communication pattern.

| Kind | What it means | What changes |
|---|---|---|
| **Parameter sparsity** (MoE) | Many params exist, but only a subset activated per token | Active FLOPs ↓, total params ↑ |
| **State compression/sharing** (GQA/MLA) | Store less per token, or share across heads | KV bytes ↓, sometimes reconstruction compute ↑ |
| **Computation sparsity** (DSA/NSA) | Each query attends to a structured subset of past tokens | L² FLOPs → L·k_sel, irregular memory access ↑ |
| **Architectural sparsity** (hybrids) | Most layers are O(L) constant-state; only a few are attention | KV cache size ∝ n_attn ≪ n_layers, overall complexity ↓ |

Each maps cleanly onto one of the three bottlenecks. Choosing the wrong mechanism for your bottleneck wastes engineering effort.

---

### MoE — Breaking the FFN Bottleneck
{: #moe-ffn}

**The problem.** In a dense Transformer with F ≈ 4D, each layer has:

```
P_FFN,dense ≈ 12D²    (params per layer)
FLOPs_FFN   ≈ 24·B·L·D²   (per layer forward)
```

Every token activates every parameter — "total capacity" and "per-token cost" are locked together. You cannot add capacity without adding per-token compute.

**MoE breaks this coupling.** Replace the single dense FFN with `N_r` routed experts plus `N_s` shared experts, each with hidden dimension `F_e`. Only the top-`K_r` routed experts activate per token:

```
P_FFN,total       ≈ (N_s + N_r) · 3D·F_e     (all parameters)
P_FFN,active/token ≈ (N_s + K_r) · 3D·F_e     (per token)
```

For DeepSeek-V3 (D=7168, F_e=2048, N_s=1, N_r=256, K_r=8):

```
P_total       ≈ 257 × 3 × 7168 × 2048 ≈ 11.3B params/layer
P_active/token ≈   9 × 3 × 7168 × 2048 ≈ 0.40B params/token
```

Each token activates ~3.5% of the FFN parameters. The model gains access to 28× more FFN capacity at essentially the same per-token compute.

**The forward pass:**

<div class="post-flow" role="group" aria-label="MoE forward pass">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Router scores all N_r experts: s_t = W_router · u_t ∈ ℝ^N_r</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Top-K_r selection → S_t (set of selected expert indices)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Dispatch: send token activations to ranks hosting selected experts (AllToAll)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Expert compute: each expert runs a standard gated FFN on its token batch</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Combine: weighted sum of expert outputs + shared expert outputs → h'_t</span></li>
  </ol>
</div>

**Two router variants:**

*Classic softmax router:*
```
s_t = W_router · u_t ∈ ℝ^N_r
p_{i,t} = exp(s_{i,t}) / Σ_j exp(s_{j,t})
S_t = Top-K_r(p_t)
```

*Sigmoid-affinity router (DeepSeek style):*
```
s_{i,t} = σ(u_t^T · e_i)       (e_i = learned expert embedding)
g_{i,t} = s_{i,t} / Σ_{j∈S_t} s_{j,t}    (normalised within selected set)
```

**Load balancing is a throughput bug, not a training detail.** If the router collapses onto a few popular experts, those experts become hot spots. In expert-parallel serving, one overloaded rank stalls all others — step time becomes `max_i(n_i)` rather than `mean_i(n_i)`. A routing bias controller fixes this without a large auxiliary loss:

```
# After each training step:
if expert i is overloaded:  b_i ← b_i − γ
if expert i is underloaded: b_i ← b_i + γ

# Selection uses bias-adjusted scores:
i selected for token t  ⟺  s_{i,t} + b_i ∈ Top-K_r(t)
```

**AllToAll communication cost.** Under expert parallelism, dispatching token activations across GPUs costs:

```
MoE_comm ≈ 2 × B_tok × D × b × K_r bytes per layer
```

For D=7168, b=2 (BF16), K_r=8: `2 × D × b × K_r ≈ 224 KB per token per layer`. This is why **LatentMoE** routes in a smaller latent dimension ℓ < D — the routed payload shrinks to `2·B_tok·ℓ·b·K_r`, reducing AllToAll volume by `D/ℓ`.

**Kimi K2 vs DeepSeek-V3 — scaling total capacity under fixed activated compute:**

| | DeepSeek-V3 | Kimi K2 |
|---|---|---|
| Total params | 671B | 1.04T |
| Activated params/token | 37B | 32B |
| Routed experts (N_r) | 256 | 384 |
| Active experts/token (K_r) | 8 | 8 |

K_r stays fixed; N_r grows. More total expert capacity with the same per-token compute — the MoE scaling axis.

> **Interview question:** An MoE model achieves better benchmark scores than a dense model of the same activated parameter count, but serving latency is 40% higher. What are the likely causes and how do you fix them?
>
> *Three sources of latency tax: (1) **AllToAll communication** — dispatching token activations to expert ranks adds a synchronisation barrier per MoE layer. Fix: topology-aware expert placement (co-locate popular expert pairs), LatentMoE to shrink payload, or expert parallelism within NVLink domain only. (2) **Load imbalance** — if the router hasn't converged its bias controller, some experts receive 3–4× average tokens, stalling the barrier. Fix: check expert load distribution in profiler; tune γ or add auxiliary balance loss. (3) **Small expert GEMMs** — each expert's sub-batch is smaller than the full batch, so GroupGEMM efficiency drops. Fix: larger K_r (more tokens per expert per step), or increase global batch size. The fundamental check: measure AllToAll time vs expert GEMM time in profiles — if AllToAll > 30% of MoE layer time, communication is the bottleneck; if GEMM is slow, it's a batching/occupancy issue.*

---

### MLA — Breaking the KV Bottleneck
{: #mla-kv}

**Why decode is KV-bandwidth-bound.** During autoregressive generation, the query for the new token is tiny (one vector), but attention must scan the entire KV cache — which grows as:

```
KV bytes (MHA) ≈ 2 × L × n_layers × N_KV × H × b
```

For a 671B model (D=7168, n_layers=61) with full MHA at L=128K in BF16:
```
≈ 2 × 128,000 × 61 × 7168 × 2 ≈ 208 GiB per sequence
```

That's 208 GB of memory *and* 208 GB of bandwidth consumed per decode step, per sequence. With batch=16 you'd need 3.3 TB of HBM just for KV caches.

**Two cleanly different reduction strategies:**

| Strategy | Mechanism | What shrinks | What's preserved |
|---|---|---|---|
| **GQA/MQA** (head-space sharing) | Reduce N_KV (fewer KV heads) | KV bytes ∝ N_KV/N | Head-space representation unchanged |
| **MLA** (latent-space caching) | Cache a compressed latent, reconstruct heads on demand | Cache dc+d_hR ≪ N_KV×H | Full head-space quality (via reconstruction) |

**GQA/MQA — head-space sharing:**

Query heads stay at N. KV heads reduce to N_KV ≪ N (each KV head shared across a group of query heads). KV cache bytes drop by factor N_KV/N:

```
KV bytes (GQA) ≈ 2 × L × n_layers × N_KV × H × b
```

MQA is the extreme: N_KV=1, one KV head shared across all N query heads. GQA with N_KV=8 (used in Llama-3, Mistral) is the practical sweet spot — ~16× KV reduction at negligible quality loss.

**MLA — latent-space caching:**

Instead of caching K and V in head-space, MLA caches a compressed **content latent** c_t ∈ ℝ^d_c and a small **RoPE component** r_t ∈ ℝ^d_hR per token per layer. At attention time, head-space K and V are reconstructed:

```
Compression (cache this):
  c_t = W_c · u_t ∈ ℝ^d_c        (down-project token state)
  r_t = RoPE(W_r · u_t) ∈ ℝ^d_hR  (position-aware part)

Reconstruction (at attention):
  k^C_{t,i} = W^↑_{K,i} · c_t    (per-head up-project)
  v_{t,i}   = W^↑_{V,i} · c_t
  k_{t,i}   = [k^C_{t,i} ; r_t]  (append RoPE key)
```

Cache bytes per token per layer:
```
KV bytes (MLA) ≈ (d_c + d_hR) × b per token per layer
```

For DeepSeek-V3 (d_c=512, d_hR=64, n_layers=61) at L=128K:
```
≈ 128,000 × 61 × 576 × 2 ≈ 8.4 GiB per sequence
```

That's **25× less** than the MHA baseline (208 GiB → 8.4 GiB).

**The trade-off:** MLA adds reconstruction GEMMs (up-projections) at every attention step. In practice this is a **net win** because decode is bandwidth-bound — cutting bytes by 25× reduces the dominant bottleneck more than the reconstruction compute increases it. The analogy to FlashAttention is exact: spend a little more compute to move far fewer bytes.

**Why MLA decouples dimensions.** Standard attention assumes D = N·H. MLA breaks this — d_c and d_hR are separate hyperparameters chosen to balance cache bytes, reconstruction compute, and model quality. Substituting D = N·H into MLA cache math gives wrong answers.

**KV reduction comparison:**

| Mechanism | Cache bytes per token/layer | Reduction vs MHA |
|---|---|---|
| MHA (baseline) | N_KV × H × b = D × b | 1× |
| GQA (N_KV=8, N=64) | 8 × H × b | ~8–16× |
| MQA (N_KV=1) | H × b | N× |
| MLA (d_c=512, d_hR=64) | 576 × b | ~25× (at D=7168) |

> **Interview question:** Your 70B model with GQA (N_KV=8) still runs out of KV cache memory at long context. A colleague proposes switching to MLA. What is the performance tradeoff and under what conditions is MLA strictly better than GQA?
>
> *MLA strictly reduces cache bytes more aggressively — (d_c + d_hR) can be tuned independently of N_KV, and at D=7168 it gives ~25× compression vs ~8× for GQA. But MLA adds reconstruction GEMMs at every decode step: for each cached token, two up-projections (W^↑_K and W^↑_V) must run to recover head-space K and V. The win condition: **bandwidth savings must outweigh reconstruction compute**. Since decode is bandwidth-bound (AI ≈ 1 FLOP/byte at batch=1), reducing cache bytes directly reduces the memory-bound time. The reconstruction GEMMs have higher arithmetic intensity (they reuse the weight matrices across the batch) and land closer to the compute roof. MLA is strictly better when: (1) context is long enough that KV bandwidth dominates step time, (2) kernel support exists to fuse reconstruction into the attention kernel (otherwise you add kernel launch overhead), (3) quantisation of the latent cache (INT8) is applied, further cutting bytes. MLA is worse if: reconstruction can't be fused, batch is tiny (reconstruction overhead visible), or the model is short-context where KV isn't the bottleneck.*

---

### Sparse Attention — Breaking L² at Long Context
{: #sparse-attention}

**Why FlashAttention alone isn't enough.** FlashAttention eliminates the `4·BNL²·b` bytes of intermediate attention matrix traffic and moves quadratic attention toward compute-bound. But the **FLOPs themselves** still scale as `4·BL²·D`. At L ~ 10⁵–10⁶:

```
FLOPs_L² = 4·B·L²·D  dominates when L ≳ 8D
```

For D=7168: L ≳ 57,000 tokens. At L=1M, L² attention is ~17.5× more expensive than all MLP layers combined. FlashAttention is necessary but not sufficient — you must change the algorithm.

**The options: make the set of keys smaller, or stop doing attention.**

**DSA — Deep Sparse Attention (retrieval-inside-attention):**

DSA uses a cheap "lightning indexer" to score all past tokens, then runs full attention on only the top-k_sel selected tokens:

```
Step 1 — Indexer scoring (cheap, runs over all L past tokens):
  For each indexer head j:
    m_{t,s,j} = ReLU(⟨q^I_{t,j}, k^I_s⟩)   (non-negative match score)
  Mix across heads with query-dependent weights w_t:
    I_{t,s} = Σ_j w_{t,j} · m_{t,s,j}

Step 2 — Top-k selection:
  S_t = Top-k_sel({I_{t,s} : s < t})

Step 3 — Sparse attention (full cost, only k_sel entries):
  u_t ← Attn(h_t, {c_s | s ∈ S_t})
```

L² complexity becomes `2·B·L·k_sel·D` — linear in L for fixed k_sel. The indexer itself still scores all L tokens but it's very cheap (ReLU dot products, not softmax).

**DSA's systems tax:** irregular memory access patterns (the k_sel selected tokens are non-contiguous), a separate indexer KV buffer, and batching complications (prefill and decode have different indexer behavior). Under GQA, if different query heads in a group select different tokens, the system must fetch the union — erasing some of the savings. Common fix: run DSA in an MQA-like mode where selection is shared across heads.

**NSA — Native Sparse Attention (hierarchical paths):**

Simple sparse selection risks losing context if the indexer misses relevant tokens. NSA uses three parallel branches with a learned gating to capture different receptive fields:

<div class="post-flow post-flow--horizontal" role="group" aria-label="NSA three branches">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Sliding window — recent local context (exact)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Token selection — important tokens/blocks (sparse)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Token compression — coarse summaries (global context)</span></li>
  </ol>
</div>

The branch outputs are combined with a learned gate. This gives both local detail and global context without paying full L². The tradeoff: more moving parts — window attention + sparse selection + compression kernels + gating — each with its own tuning knobs.

**DSA vs NSA — when to use which:**

| | DSA | NSA |
|---|---|---|
| **Use when** | Relevance is very sparse; retrieval-like access | Need both local detail and global context |
| **Watch out for** | Recall risk if top-k misses; irregular gather overhead | Tuning instability; more kernel surface area |
| **Key knob** | k_sel, MQA-compatible group-consistent selection | Window size w, selection budget k_sel, compression rate, gating stability |

> **Interview question:** You switch from full attention to DSA (top-k_sel=256, L=32K, D=4096). Theoretically this reduces attention FLOPs by 32K/256 = 125×. In practice you observe only a 3× speedup. Why?
>
> *Several reasons. (1) **Indexer is not free** — the indexer still scores all L tokens, running `2·B·L·H_I·D` dot products. At L=32K with H_I=4 heads, this can be a significant fraction of the "saved" attention FLOPs. (2) **Irregular memory access** — the k_sel=256 selected KV entries are non-contiguous in memory. Gathering them breaks the sequential HBM streaming pattern that makes dense FlashAttention cache-friendly. The effective bandwidth for gather-access is much lower than sequential bandwidth. (3) **Overhead ops** — top-k selection kernel, index construction, and dispatch add latency not present in the dense path. (4) **Prefill vs decode** — DSA primarily helps decode; prefill still needs to score all tokens for the indexer. At L=32K with a long prompt, prefill dominates wall time and sees no speedup. (5) **Insufficient batch to amortise** — at batch=1, memory bandwidth is the bottleneck regardless; sparse access just changes which bytes are loaded, not how many total bytes are loaded per token (the k_sel KV entries still need to be fetched).*

---

### Hybrid Stacks
{: #hybrid-stacks}

**The most aggressive long-context strategy:** stop doing full attention in most layers entirely.

Hybrid stacks interleave attention layers with **linear-time constant-state sequence layers** (SSM / Mamba-style blocks). These have O(L) compute and a fixed-size recurrent state — no KV cache growth with context.

**Nemotron 3 Nano example:** 52 total layers, only **n_attn = 6** are attention layers. The remaining 46 are Mamba-2 blocks.

Two direct wins:

| | Full attention stack | Hybrid stack |
|---|---|---|
| Compute per layer | O(L²) for attention layers | O(L) for Mamba, O(L²) for n_attn layers |
| KV cache size | Grows with all n_layers | Grows with only n_attn layers |
| Long-context scaling | Quadratic | Near-linear |

For Nemotron 3 Nano: KV cache is `6/52 ≈ 11.5%` the size of a full attention stack. At L=1M tokens, this is the difference between an impractical 200+ GB KV cache and a manageable ~23 GB one.

**The catch:** too few attention layers weakens global routing — SSM layers process context through a fixed-size recurrent state, so long-range dependencies that don't fit in that state are lost. The few remaining attention layers must compensate by providing global context that SSMs can't. The key design question: **where to place the attention layers** in the stack (early, late, uniformly spaced) and how many to keep.

**Serving implications:** hybrid stacks require runtime support for *two* types of state simultaneously — KV cache for the attention layers and SSM recurrent state for the Mamba layers. These have different memory layouts, different quantisation sensitivities, and different kernel requirements. A serving engine that handles attention-only models cannot directly serve hybrid models without modification.

---

### Putting It Together
{: #arch-summary}

Modern frontier models (DeepSeek-V3, Kimi K2, Nemotron 3) combine all three architectural interventions because each targets a different dominant term:

<div class="post-flow" role="group" aria-label="Three-axis selectivity">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">MoE — select which FFN parameters activate per token (decouple capacity from per-token compute)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">MLA — select what decode state to cache per token (compress KV bytes aggressively)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">DSA/NSA/Hybrids — select which past tokens each query attends to (break L² scaling)</span></li>
  </ol>
</div>

**The unifying principle is selectivity** — spend compute and bandwidth only where it matters.

**One bottleneck → one knob:**

| Bottleneck | Root cause | Architectural fix | Key parameter |
|---|---|---|---|
| FFN compute/params | Every token activates all FFN weights | MoE (top-K_r routing) | N_r, K_r, F_e |
| KV cache / decode BW | Full head-space KV grows with L and batch | GQA/MQA (N_KV↓), MLA (d_c + d_hR↓) | N_KV or d_c, d_hR |
| Long-context L² | QK^T scales as L² in FLOPs and bytes | DSA/NSA (k_sel ≪ L), hybrids (n_attn ≪ n_layers) | k_sel or n_attn |

**Systems stack requirements** — the architecture only wins if the serving system supports it:

- **MoE**: high-bandwidth AllToAll for expert parallelism, robust load balancing, good GPU topology mapping
- **LatentMoE**: routing in latent dimension ℓ < D reduces AllToAll volume by D/ℓ
- **MLA**: kernel support for latent cache formats + fused reconstruction; quantisation of latent cache
- **DSA/NSA**: fast top-k selection kernels, cache layouts compatible with continuous batching, group-consistent selection under GQA
- **Hybrids**: runtime support for mixed state (KV cache + SSM state), heterogeneous kernel scheduling

> **Interview question:** You're asked to design a 300B-parameter model that must serve at L=128K context with p99 TTFT < 2s and decode > 20 tokens/second on 8× H100s. Walk through the architectural choices.
>
> *Start from the constraints and work backwards. (1) **Parameter budget vs served size**: 300B at BF16 = 600 GB. 8× H100 = 640 GB HBM total — tight, no room for large KV cache. Use INT8 weights (300 GB) to free ~300 GB for KV + activations. (2) **KV cache pressure**: at L=128K and batch=32, MHA KV ≈ 2 × 128K × n_layers × D × b. With MLA (d_c=512, d_hR=64), KV ≈ 128K × n_layers × 576 × 2. For 96 layers: ~14 GB per sequence × 32 batch = 450 GB — still too large. Use MLA + INT8 KV quantisation: ~7 GB per sequence, 225 GB at batch=32. Feasible. (3) **FFN efficiency**: 300B model with MoE (256 experts, top-8) activates ~37B params/token — same per-token cost as a 37B dense model, but 300B total capacity. This raises quality without raising latency. (4) **TTFT at 128K**: prefill with FlashAttention handles L² efficiently. The L² crossover is L ≳ 8D ≈ 65K for D=8192, so at 128K we're in the L²-dominated regime. Add DSA (top-1024 selection after indexer) to cut prefill attention cost. (5) **Decode throughput**: with MLA + INT8, bandwidth per decode step is dominated by weight reads (~300 GB / 8 GPUs = 37.5 GB per GPU at batch=1). At 8 × 3.35 TB/s = 26.8 TB/s aggregate bandwidth, weight streaming alone allows ~715 tokens/s — well above the 20 token/s target even at batch=1. The bottleneck at batch=32 will be KV bandwidth. Architecture summary: MoE (256/8) + MLA (d_c=512) + DSA (k_sel=1024) + INT8 weights + INT8 KV cache.*

---

## Quantisation
{: #quantisation}

Quantisation represents weights (and optionally activations) in fewer bits, reducing both memory footprint and bytes-per-token loaded.

### Weight Quantisation
{: #weight-quant}

The simplest approach: quantise model weights, leave activations in FP16 at runtime. Weights are stored as low-bit integers; before each matrix multiply, they are dequantised back to FP16 in-register, and the matmul runs in FP16.

**Block-wise quantisation.** Naive per-tensor quantisation sets a single scale for the entire weight matrix: `scale = max(|W|) / (2^(b-1) - 1)`. A single large outlier drives the scale high, wasting most of the bit range on near-zero values. Block-wise quantisation divides the weight matrix into small blocks (typically 128 elements), each with its own scale:

```
For each block of 128 weights:
  scale = max(|W_block|) / (2^(b-1) - 1)
  W_int = round(W / scale)

At matmul time:
  W_fp16 = W_int × scale   (dequantise in-register)
```

Outliers in one block only affect that block's scale, not the entire tensor. Overhead: one FP32 scale per 128 weights = 32/128 = 0.25 bits/param.

**INT8 (W8A16):** weights stored as int8, dequantised to FP16 before matmul. Memory: 1 byte/param vs 2 in FP16 → 2× memory reduction, ~1.8× throughput gain in the bandwidth-bound regime. Accuracy loss is negligible for most tasks — 8 bits provides 256 quantisation levels, more than enough for smooth weight distributions.

**INT4 (W4A16):** 4 bits/param → 4× memory reduction. Requires groupwise quantisation (smaller group size of 64–128) because 4 bits (16 levels) is insufficient for the full weight range without local scaling. Typical accuracy degradation: <1% on standard benchmarks for 7B+ models. Sub-2% degradation at 4 bits is remarkable — it suggests the weight matrices have low effective rank (most information is in a few dominant directions), so coarser quantisation of minor directions causes minimal loss.

**GPTQ** minimises quantisation error using a second-order approach. For each row of a weight matrix, it quantises one column at a time, then updates the remaining unquantised columns to compensate for the error introduced. The update uses the inverse Hessian of the layer's output with respect to its weights (approximated via a calibration dataset):

```
For column i:
  w_q[i] = quantise(w[i])
  error = w[i] - w_q[i]
  w[i+1:] -= error × H_inv[i, i+1:] / H_inv[i,i]   # error propagation
```

This is the Optimal Brain Compression (OBC) framework. It's expensive to run (O(d³) per layer for the Hessian inversion), but produces significantly better quantised models than round-to-nearest at the same bit width.

**AWQ (Activation-Aware Weight Quantisation)** makes a key observation: not all weight channels are equally important. Channels corresponding to large activation magnitudes cause disproportionate output error when quantised poorly, because the error is amplified by the large activation. AWQ identifies these "salient" channels and protects them via per-channel scaling before quantisation:

```
For each output channel j, find scale s_j that minimises:
  ‖ quantise(W · diag(s)) · (X / s) - WX ‖²

where s scales up important channels (so they get finer quantisation bins)
and X/s compensates by scaling activations down
```

AWQ finds these scales analytically from activation statistics, without per-column optimisation. Faster than GPTQ at comparable or better accuracy — now the dominant 4-bit quantisation method.

**FP8:** H100 and later GPUs support native FP8 (E4M3 and E5M2 formats) tensor core operations at 2× the throughput of BF16 tensor cores. Unlike integer formats, FP8 retains the floating-point dynamic range — no need for block scales or dequantisation within the matmul kernel. E4M3 (4 exponent bits, 3 mantissa bits) is used for weights and activations; E5M2 (5 exponent bits) for gradients (larger range needed). FP8 inference with per-tensor or per-channel scales is now standard for H100 deployments.

> **Interview question:** W4A16 quantisation consistently achieves less than 1% accuracy degradation on 7B+ models but much larger degradation on 1B models. Why does model size affect quantisation robustness?
>
> *Two mechanisms. First, larger models have more parameters per layer and thus more redundancy — quantisation error in one weight can be compensated by others in the same layer. A 7B model's attention weight matrix has 4096×4096 = 16M entries; even if 4-bit quantisation introduces noise in 10% of them, the remaining 14.4M entries provide the signal. A 1B model's matrices are smaller and each entry carries proportionally more weight in the final output. Second, larger models learn more distributed representations — the weight matrices have lower effective rank (most of the information is concentrated in fewer singular directions). When you quantise, you're adding noise proportional to the weight scale, which primarily corrupts small singular values. Large models have many small singular values (distributed, low-rank structure), so quantisation noise falls mostly on directions that contribute little to the output. Small models have less distributed representations — every direction matters more. This is also why post-training quantisation works better on over-parameterised models.*

### Activation Quantisation
{: #activation-quant}

**W8A8** enables fully integer matrix multiplications — the matmul itself runs in INT8, giving 2–4× throughput vs FP16 matmuls on hardware with INT8 tensor core support (A100, H100). The challenge: LLM activations have **outliers**. LLaMA-family models exhibit a small fraction of activation channels (~0.1% of dimensions) with magnitudes 100–1000× larger than typical values. These outliers arise from specific attention patterns and FFN activations learned during pretraining and are persistent across inputs.

With naive INT8 quantisation, the scale is set by the outlier range, wasting most quantisation bins on the common near-zero values. The result is catastrophic accuracy loss.

**LLM.int8()** handles outliers by decomposing the matmul: identify outlier channels (those exceeding a threshold — empirically ~6.0 in absolute value), extract them, compute those columns in FP16, compute the remainder in INT8, and sum the results. The decomposition is per-batch, not pre-determined. Accuracy is preserved with ~1–5% throughput overhead for the split computation.

**SmoothQuant** takes the insight that activations are hard to quantise but weights are easy (weights are fixed, you can apply arbitrary transformations offline). It migrates the quantisation difficulty from activations to weights via a mathematically equivalent channel-wise scaling:

```
Y = X W = (X · diag(s)⁻¹) · (diag(s) · W) = X_smooth · W_smooth
```

Choose `s_j = max(|X_j|)^α / max(|W_j|)^(1-α)` where α ∈ [0,1] controls how much difficulty to push to weights vs activations. At α=0.5, both are equally smoothed. X_smooth is computed at runtime (cheap division); W_smooth is pre-computed offline (fold into weights). Enables accurate W8A8 without outlier handling overhead at inference time.

> **Interview question:** Why do LLM activations have persistent outliers in specific channels, and why don't the weights have the same problem?
>
> *Activation outliers arise from the interaction between attention patterns and layer normalisation. In transformer layers, certain neurons learn to act as "feature detectors" for globally important patterns — for example, a neuron that activates strongly whenever the model is in a "counting" mode or "code generation" mode. These neurons must activate with large magnitude to communicate their signal clearly through the residual stream, because they're competing with thousands of other dimensions. The channels where outliers occur are consistent across inputs because they represent stable learned features of the model. Weights don't have this problem because they're fixed after training and optimised to have a specific distribution (controlled by initialisation and weight decay). Weight distributions are approximately normal by construction. Activation distributions, however, emerge from the data distribution passing through the model — and the data has structured patterns that create structured activation outliers. This is also why activation quantisation is harder: the outlier channels are not known in advance (they're input-dependent in magnitude, even if the channels themselves are consistent), and any fixed scale set on the entire tensor will either waste bits on non-outlier values or clip the outliers.*

### KV Cache Quantisation
{: #kv-quant}

The KV cache grows as `batch_size × seq_len × n_layers × 2 × n_heads × d_head` bytes. For a 70B model with 96 layers, 8 heads, d_head=128, batch size 32, and context 32k: 32 × 32768 × 96 × 2 × 8 × 128 × 2 bytes = **~8TB**. Even at 4k context, this is ~1TB. The KV cache dominates GPU memory at scale.

**KV INT8:** per-channel quantisation of keys and values before caching, dequantise before attention computation. Keys and values are relatively smooth (they pass through layer norm before being produced), so INT8 is accurate with minimal impact on generation quality. Reduces KV memory by 2×.

**KV FP8:** similar approach using E4M3 format, reducing by 4× vs FP16. Accuracy is slightly worse than INT8 for keys (which have higher dynamic range), but acceptable for most tasks.

**KV INT4:** more aggressive — per-channel or per-head scaling before INT4 quantisation. 4× memory reduction vs FP16 at the cost of slightly more dequantisation overhead. Best suited for very long contexts where the memory saving is critical.

**Why KV quantisation is safer than weight quantisation.** KV tensors are attention-averaged: the output of an attention head is `softmax(QKᵀ/√d) · V`. Even if individual K or V elements have quantisation error, these errors partially average out across the heads and the sequence dimension. Weight quantisation errors, by contrast, compound multiplicatively through the layer stack.

> **Interview question:** You are designing a KV cache quantisation scheme. Should you quantise keys and values the same way, or differently? Why?
>
> *Differently — keys and values have different roles in attention and different statistical properties. Keys are compared against queries via dot products: `score = QKᵀ/√d`. The dot product is sensitive to the relative magnitudes and directions of K and Q. Quantisation error in K is amplified by the softmax (errors in attention logits get exponentially amplified near the argmax of the distribution). So keys need higher precision or finer quantisation. Values are linearly weighted by attention scores: `output = softmax(·) · V`. Linear weighting is more robust to quantisation noise than the exponential softmax. Additionally, values tend to have smoother distributions than keys in practice. Empirically, using FP8 or INT8 for keys and INT4 for values achieves the best memory-quality tradeoff — the key precision matters more for attention pattern accuracy, while value precision matters less for output quality.*

---

## KV Cache Management
{: #kv-cache}

The KV cache stores attention keys and values for all previous tokens so decoding can attend to them without recomputation. Naively managed, it wastes GPU memory through fragmentation and discards reusable prefixes on request completion.

### PagedAttention
{: #pagedattention}

**The fragmentation problem.** Traditional serving pre-allocates a contiguous block of GPU memory for each request's maximum possible KV cache (prompt length + max output length). This causes two types of waste:
- **Internal fragmentation**: the request generates fewer tokens than the maximum — unused memory within the reserved block
- **External fragmentation**: free blocks scattered between allocated blocks are too small to serve new requests

In practice, pre-allocation wastes 60–80% of GPU KV memory. This directly limits the batch size — fewer concurrent requests means lower throughput.

**PagedAttention** applies OS virtual memory concepts. Physical GPU memory is divided into fixed-size **KV blocks** (typically 16–32 tokens per block). Each request holds a **block table** — a mapping from logical block index to physical block number — rather than a contiguous reservation. Blocks are allocated on demand as the sequence grows, one block at a time:

<div class="post-flow" role="group" aria-label="PagedAttention allocation">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Physical memory divided into fixed KV blocks (16–32 tokens/block)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each request holds a block table: logical → physical block mapping</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">New blocks allocated on demand as sequence grows — no pre-reservation</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Blocks freed immediately on completion — no wasted reservation, <1 block of internal fragmentation</span></li>
  </ol>
</div>

Maximum internal fragmentation is now `block_size - 1` tokens per request. External fragmentation is eliminated — any free block can serve any request regardless of its physical location. Effective KV memory utilisation jumps from 20–40% (naive) to >95%.

**Attention over non-contiguous blocks.** The attention kernel must gather K/V from physically non-contiguous blocks using the block table. This requires a custom attention kernel that follows the indirection through the block table — standard batched GEMM won't work. The kernel is slightly less cache-friendly than contiguous attention, but the throughput gain from higher memory utilisation dominates.

**Copy-on-write for beam search.** When a request forks into B beams, all beams share the prompt's KV blocks. Blocks are reference-counted: when beam `i` writes a new token's KV into a shared block, it first copies the block and decrements the shared reference count. Only when a beam uniquely owns a block can it write to it in-place. This avoids duplicating the prompt's KV cache B times — critical when B=4–8 and the prompt is long.

> **Interview question:** PagedAttention eliminates external fragmentation by using non-contiguous physical blocks. But GPU memory isn't like CPU memory — non-contiguous access patterns have real costs. Quantify the trade-off.
>
> *GPU memory access is most efficient when threads in a warp access consecutive memory addresses (coalesced access). With PagedAttention, K/V blocks for the same request are physically scattered. When the attention kernel fetches a block for position range [i*block_size, (i+1)*block_size], it loads from a contiguous physical location — good coalescing within a block. The problem is between blocks: block i+1 may be at a completely different physical address, requiring a new memory transaction with potentially cold cache. For block_size=16 tokens with d_head=128 in FP16, each block is 16 × 128 × 2 = 4KB. Modern GPU L2 caches are ~40MB, so with many concurrent requests the blocks from one request compete with other requests' blocks. In practice, vLLM benchmarks show PagedAttention is 5–10% slower than contiguous attention for the same KV data — but this is dominated by the 2–4× throughput increase from fitting more requests in memory. The net effect is always positive.*

### Prefix Caching
{: #prefix-caching}

Many requests share a common prefix: a system prompt, a fixed few-shot preamble, or a shared conversation history. Without caching, each new request recomputes KV for the entire prefix from scratch — O(L²) attention for a prefix of length L, repeated for every request.

**RadixAttention** (SGLang) maintains a global LRU cache of KV blocks organised as a **radix tree** (compact prefix trie keyed by token sequence). On request arrival:

1. Hash the incoming token sequence prefix
2. Walk the radix tree from root — each node represents a cached block matching a token sub-sequence
3. Longest-prefix match: find the deepest tree node matching the request's prefix
4. Load matched KV blocks from cache (O(1) per block); only the unmatched suffix requires fresh prefill

```
Example tree:
  [system prompt tokens] → [example A] → [user "what is 2+2?"]
                         → [example B] → [user "write code for..."]
```

When a cached prefix is evicted (LRU policy), its physical KV blocks are freed back to the pool. Tree nodes are shared reference-counted — eviction only happens when no active request references the node.

**Where prefix caching excels:**
- **Multi-turn chat**: each user turn extends the previous conversation — the full history KV is cached
- **Batch inference over fixed system prompts**: 10,000 requests sharing a 2,000-token system prompt save 95% of their prefill compute
- **Agent tool loops**: tool descriptions and few-shot examples reused across hundreds of tool calls

> **Interview question:** RadixAttention uses a radix tree keyed by token sequences. What happens when two requests share a prefix but differ in a single token somewhere in the middle — does prefix caching help at all?
>
> *Only for the portion before the differing token. The radix tree stores exact token-sequence prefixes. If request A is [S, A, B, C, D] and request B is [S, A, X, D, E], they share the prefix [S, A] — only those two tokens' KV are reusable. The divergence at X means request B must recompute from X onward, even if D and E happen to be the same tokens as in request A. This matters for designs like few-shot prompting: if you randomly shuffle the few-shot examples between requests, the shared prefix length drops to 0 even though the same examples appear in each request. Fix: always put shared content before variable content in the prompt. For RAG, put the system prompt first (always shared), then retrieved documents (variable), then the question. This maximises the cached prefix length. Some systems go further and canonicalise the prompt by sorting retrieved chunks by their hash, making the prefix deterministic across requests with the same retrieved set.*

### KV Eviction
{: #kv-eviction}

When context exceeds the KV cache budget, entries must be evicted. The question is which tokens to drop.

**Why not just truncate old tokens?** The "sliding window" approach — drop tokens beyond a fixed recency window — is simple but destroys long-range dependencies. A question about something mentioned 10,000 tokens ago gets a wrong answer because the relevant KV was evicted.

**H₂O (Heavy Hitter Oracle)** uses attention scores as an importance signal. Tokens that receive high cumulative attention across recent decoding steps are "heavy hitters" — the model consistently routes attention to them. H₂O maintains a running estimate of cumulative attention per token and evicts the lowest-scoring entries when the budget is exceeded:

```
importance[i] += attention_score[current_step, i]   # update for each decode step
when budget exceeded: evict argmin(importance)
```

H₂O also observes the **attention sink** phenomenon: the first 1–4 tokens always receive disproportionately high attention regardless of their content (the model routes "background" attention to the beginning). These sink tokens are always protected from eviction.

**StreamingLLM** makes the attention sink insight its core mechanism: always keep sink tokens (first 4) + a sliding window of recent tokens (last W). This enables **infinite-length streaming generation** at fixed KV memory — never more than W+4 tokens in the cache. The cost: all mid-context information beyond the sliding window is lost. Accurate only when the task doesn't require long-range dependencies (streaming summarisation of a live transcript, for example).

**SnapKV** clusters key vectors to identify which tokens encode similar information, then evicts one representative from each cluster — avoiding redundant KV entries. Works well when long contexts have repeated or paraphrased information (documents with repetitive structure).

> **Interview question:** H₂O uses cumulative attention scores to decide which KV entries to evict. But attention scores change every decode step — a token ignored early might become critical later. Is cumulative attention a good proxy for future importance?
>
> *It's a reasonable heuristic but has real failure modes. Cumulative attention weights early steps heavily — if a token received attention in steps 1–10 but is irrelevant for steps 11–100, it won't be evicted despite being useless going forward. Conversely, a token that becomes relevant at step 50 (e.g. a name mentioned early in a document, referred back to 50 steps into generation) has a low cumulative score and may have been evicted before it's needed. The deeper issue: importance is task-dependent in a way that attention scores can't fully capture. A sentence defining a term may receive low attention during "general reading" but high attention when a follow-up question references that term. Better alternatives: (1) look-ahead importance estimation — run a lightweight "probe" to predict which tokens will be attended to in future steps; (2) learned eviction policies — train a small model to predict per-token importance given the task context; (3) task-aware retention — for retrieval-heavy tasks, protect tokens that appear in the query. H₂O is the best simple heuristic but not the optimal solution.*

---

## Batching Strategies
{: #batching}

### Continuous Batching
{: #continuous-batching}

**The static batching problem.** A batch of 32 requests starts decoding together. Request 1 generates 10 tokens and finishes; requests 2–32 continue generating up to 200 tokens. GPU slot 1 sits idle for 190 decode steps while waiting for the batch to complete before the next batch starts. With heterogeneous output lengths (typical in production — some responses are one sentence, some are paragraphs), static batching achieves 20–30% GPU utilisation.

**Continuous batching** (also called iteration-level scheduling) operates at the individual decode step level:

<div class="post-flow post-flow--compare" role="group" aria-label="Static vs continuous batching">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Static Batching</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">All requests in batch run to completion before next batch</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Completed requests leave idle GPU slots</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Throughput collapses with output length variance</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Continuous Batching ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">After every decode step, check for completed requests (EOS token emitted)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Evict completed requests immediately; admit new waiting requests</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Batch size stays near maximum — GPU always saturated</span></li>
    </ol>
  </div>
</div>

**Why it took so long to be standard.** Static batching is the natural extension of training batch processing — take a batch, run it, take the next batch. Continuous batching requires a different mental model: the GPU is always running, and the "batch" is a dynamic pool that the CPU scheduler continuously adjusts. The scheduler must handle heterogeneous sequence lengths within a batch (different requests are at different decode steps), which complicates memory management (solved by PagedAttention) and requires the attention kernel to handle variable-length sequences efficiently.

**Overlapped CPU-GPU scheduling.** The scheduler has non-trivial CPU work each step: check which requests emitted EOS, free their KV blocks, run the admissions queue, select new requests, allocate KV blocks. Without pipelining, this CPU work happens serially after the GPU finishes and before the next GPU step starts. In high-throughput systems, this scheduling overhead can consume >50% of wall-clock time. The fix: run the CPU scheduler for step `t` concurrently with the GPU executing step `t+1` — the GPU result from step `t` is queued while the GPU starts step `t+1`, and the CPU processes step `t`'s results in parallel.

> **Interview question:** With continuous batching, the batch at each step has requests at different sequence lengths. How does the attention kernel handle this efficiently?
>
> *Each request in a decode batch has a different KV cache length — request A is at step 50, request B is at step 200. Standard batched matmul requires rectangular input tensors, so you'd need to pad all sequences to the maximum length. Padding tokens waste compute on the attention softmax and matmul. Solutions: (1) Paged KV with custom attention kernel — each request has a block table; the kernel uses per-request sequence length metadata to determine how many blocks to load, no padding needed. (2) FlashAttention with variable-length support — flash_attn_varlen_func takes a cu_seqlens array (cumulative sequence lengths) and processes all requests packed into a single tensor without padding. This is the dominant approach in vLLM and SGLang. (3) Separate prefill and decode batches — prefill requests (variable length, many tokens) are processed separately from decode requests (one new token each), each with their own batching strategy. Chunked prefill interleaves these.*

### Chunked Prefill
{: #chunked-prefill}

**The prefill-decode interference problem.** A 10,000-token prefill request takes O(10000²) attention operations — potentially hundreds of milliseconds. While this prefill runs, all decode-phase requests in the batch are blocked. Their per-token latency (TBT — time between tokens) spikes, violating latency SLOs for interactive requests.

**Chunked prefill** breaks the prefill into fixed-size chunks (e.g. 512 tokens) and interleaves them with decode steps:

```
Iter 1: prefill_chunk(tokens 0–511) + decode_batch(all current decode requests)
Iter 2: prefill_chunk(tokens 512–1023) + decode_batch
Iter 3: prefill_chunk(tokens 1024–1535) + decode_batch
...
Iter k: final prefill chunk (request ready for decode) + decode_batch
```

Each iteration has bounded duration — no single step dominates. Decode TBT stays stable. Total prefill latency (TTFT — time to first token) increases slightly because it's now spread over multiple iterations, but this is often acceptable: interactive applications care more about TBT than TTFT.

**Why mixing prefill and decode improves GPU utilisation.** Prefill is compute-bound (many tokens processed in parallel → high arithmetic intensity). Decode is memory-bound (one token → low arithmetic intensity). On modern GPUs with separate tensor core and memory controller resources, compute-bound and memory-bound work can partially overlap. A mixed prefill+decode iteration uses both resources simultaneously — the tensor cores serve the prefill while memory bandwidth loads KV for the decode requests. This can push overall GPU utilisation above what either pure prefill or pure decode achieves alone.

> **Interview question:** What is the optimal chunk size for chunked prefill, and what factors affect this choice?
>
> *The optimal chunk size trades off three things: (1) Decode TBT: larger chunks mean each iteration is slower (prefill compute dominates), increasing TBT for decode requests. Smaller chunks → better TBT. (2) Prefill TTFT: smaller chunks mean more iterations to complete the prefill, increasing TTFT. (3) GPU utilisation: very small chunks (e.g. 32 tokens) have high kernel launch overhead and low compute intensity — the GPU spends too much time on overhead. Very large chunks (e.g. 4096 tokens) push GPU into fully compute-bound prefill with no bandwidth left for decode. The sweet spot is typically 512–2048 tokens, tuned by profiling the specific GPU and workload mix. For latency-sensitive APIs: smaller chunks (256–512) to minimise TBT spikes. For throughput-oriented batch inference: larger chunks (1024–4096) to maximise prefill speed. Some systems (Sarathi-Serve) make chunk size adaptive per request — large chunks for batch-mode requests, small for interactive.*

---

## Speculative Decoding
{: #speculative}

**The core insight.** LLM inference is memory-bandwidth-bound: the GPU loads 140 GB of weights to produce a single token while using ~2% of compute capacity. If we can verify `γ` draft tokens in a single LLM forward pass, we get `γ` tokens per memory load instead of 1 — a theoretical `γ×` speedup. The challenge: verification must be lossless (identical output distribution to sampling from the LLM alone).

### Draft-Verify
{: #draft-verify}

A **small speculative model (SSM)** generates `γ` draft tokens autoregressively. The SSM has far fewer parameters — 100–500M vs 7B+ for the main model — so its generation is fast. The LLM then runs one forward pass over `[prompt + γ draft tokens]`, producing output distributions at each draft position in parallel.

**Speculative sampling** verifies each draft token `x̃ᵢ` against the LLM's distribution `p(·|context)`:

```
Draft SSM generates: x̃₁, x̃₂, ..., x̃_γ

LLM forward pass produces: q(·|x<1), q(·|x<2), ..., q(·|x<γ), q(·|x<γ+1)
SSM had produced:          p(·|x<1), p(·|x<2), ..., p(·|x<γ)

For each position i:
  r = uniform(0, 1)
  if r < q(x̃ᵢ|·) / p(x̃ᵢ|·):
    accept x̃ᵢ      # keep the draft token
  else:
    sample from norm(max(0, q - p))   # resample from corrected distribution
    stop (all subsequent drafts discarded)
```

**Why this is lossless.** The acceptance probability `min(1, q/p)` and the rejection resampling `norm(max(0, q-p))` together produce a marginal distribution identical to `q`. This is a standard result from rejection sampling theory — the speculative sampling paper proves it formally. The key: we never just naively accept the draft token; we always test it against the LLM's distribution.

**When is speculation beneficial?** The speedup factor is `E[accepted_tokens + 1] / (1 + γ × SSM_cost / LLM_cost)`. Useful when: the SSM acceptance rate is high (SSM and LLM agree often), the SSM is much smaller than the LLM (low cost ratio), and the task is easy enough that a small model drafts well. Speculation hurts (or is neutral) when: the task requires knowledge the SSM lacks, output is highly random (temperature >> 1 → low acceptance), or the SSM is too expensive relative to LLM.

**SpecInfer** extends to **tree-based speculation**: instead of a single draft sequence, multiple SSMs generate different draft sequences, merged into a token tree. Each path from root to leaf is a possible continuation. The LLM verifies the entire tree in one forward pass using **tree attention** — a causal mask shaped like the tree topology, so each node attends only to its ancestors in the tree. All paths are verified simultaneously; the deepest accepted prefix across all paths is the output. This can accept up to `max_tree_depth` tokens per LLM call.

**EAGLE** tightens the draft model design: it reuses the main LLM's embedding matrix and LM head (no separate model to load), adding only a single shallow attention layer trained to predict the LLM's next feature (hidden state), not the token. EAGLE then samples `K` candidate tokens per position and builds a dynamic tree based on accumulated path likelihood, rejecting low-probability paths before verification.

**Medusa** adds multiple auxiliary decoding heads directly to the LLM — no separate model at all. Head 0 predicts the next token (standard), head 1 predicts the token 2 steps ahead, head 2 predicts 3 steps ahead, etc. All heads run in parallel on the current hidden state. Verified suffixes extend the output. Medusa trades some accuracy in multi-step prediction for zero model-loading overhead.

> **Interview question:** Speculative decoding's speedup depends on the draft acceptance rate. What determines acceptance rate, and how would you choose an SSM for a given LLM?
>
> *Acceptance rate is the probability that the SSM's draft token matches what the LLM would have sampled. It depends on: (1) Distributional alignment: the SSM must have similar "opinions" about which tokens are likely given a context. A 7B model using a 68M model as its draft works well for common text because both learned from the same distribution — disagreement happens mainly on rare or nuanced choices. (2) Task difficulty: on easy, predictable text (boilerplate, simple factual answers), even a tiny SSM accepts well. On creative or knowledge-intensive tasks, small models diverge. (3) Sampling temperature: at temperature 0 (greedy), acceptance means exact token match — high bar. At temperature 1.0, acceptance is probabilistic — easier to achieve because the reference distribution q is more spread out. (4) SSM size: larger SSMs accept better but are more expensive to run. The sweet spot is typically an SSM 10–50× smaller than the main model. For choosing an SSM: prefer models from the same family (same tokenizer, similar pretraining) — LLaMA-7B + LLaMA-68M works better than LLaMA-7B + GPT-2. Measure acceptance rate on your actual workload distribution — don't benchmark on generic text if you're serving coding queries.*

### Model-Free Methods
{: #model-free}

**Prompt lookup decoding** searches the input prompt for the current context suffix — if the last `n` tokens appear verbatim in the prompt, copy the `k` tokens that followed them in the prompt as draft tokens. No separate model, zero GPU memory overhead. Works surprisingly well for tasks where the output echoes the input: summarisation (repeating key phrases), RAG (copying retrieved sentences), code editing (reproducing unchanged code blocks).

**SuffixDecoding** generalises to a two-tier suffix tree:
- **Request-level tree**: indexes tokens generated so far in the current request — useful for repeated patterns within a single generation
- **Global tree**: indexes outputs across all prior requests — reuses patterns from similar past queries

Draft candidates are assembled by querying both trees, scored by accumulated unigram likelihood, and verified by the LLM in one pass. Especially effective for constrained outputs (JSON templates, code with repeated structures).

**Lookahead decoding** requires no external model or index — it runs two branches per decode step in the same LLM forward pass:
- **Lookahead branch**: generates tokens at positions ahead of the current sequence head (speculative lookahead)
- **Verification branch**: checks n-grams collected from previous lookahead steps against the current sequence

Both branches are batched into a single LLM call. Verified n-grams are accepted, extending the output without additional LLM calls. The algorithm is complex but zero-overhead — no extra models, no index maintenance.

> **Interview question:** Prompt lookup decoding requires no extra model, but it only works when the output copies parts of the input. For a general-purpose chat API, when would you use it and when not?
>
> *Use it when: input and output have high lexical overlap — summarisation ("The report shows that X" → output "The report indicates X"), code completion with context (copying function signatures, variable names from the prompt), document Q&A (answering by quoting the document). Don't use it when: output is creative, knowledge-generated, or reasoning-heavy — the answer to "What is the capital of France?" doesn't appear in the question. For a general chat API, prompt lookup is a good always-on optimisation with zero cost when it fails (draft rejected, standard decode continues) and meaningful gains (2–3× speedup on high-overlap queries) when it works. You can run it in parallel with a small SSM and use whichever produces the better draft first — this is the "speculative decoding ensemble" approach.*

---

## Kernel-Level Optimisations
{: #kernel-opt}

### FlashAttention
{: #flashattention}

**Standard attention materialises the full N×N score matrix.** For a single attention head at sequence length N=32,768 in FP16: the score matrix is `32768² × 2 bytes = 2.1 GB` for one head. A 96-layer model with 8 heads needs `96 × 8 × 2.1 GB = 1.6 TB` of HBM bandwidth per forward pass just for attention score matrices — and that memory doesn't exist on any single GPU.

The root cause: computing `softmax(QKᵀ)` requires the full row of `QKᵀ` to normalise (softmax denominator sums over all N key positions). Standard implementations write the full N×N matrix to HBM, read it back for softmax, write the softmax output, read it back for multiplication with V. 4 HBM round-trips of O(N²) data.

**FlashAttention** computes attention without materialising the N×N matrix, by using the **online softmax** trick and tiling over the sequence dimension:

```
Split Q into tiles of size B_r; split K, V into tiles of size B_c

For each Q tile:
  For each K, V tile:
    Load Q_tile, K_tile, V_tile into SRAM
    Compute S = Q_tile × K_tile^T  (partial scores, in SRAM)
    Update running (m, l, O) using online softmax:
      m_new = max(m_old, rowmax(S))
      l_new = exp(m_old - m_new) × l_old + rowsum(exp(S - m_new))
      O_new = diag(exp(m_old - m_new)) × O_old + exp(S - m_new) × V_tile
  Write final O = O / l to HBM  (only once per Q tile)
```

The **online softmax** identity allows correct normalisation without storing the full row: after processing all K/V tiles for a Q tile, rescale the accumulated output by the final denominator `l`. The running maximum `m` prevents numerical overflow in the exponentials.

| | Standard Attention | FlashAttention |
|---|---|---|
| HBM reads/writes | O(N²) — full score matrix | O(N) — only Q, K, V, O |
| HBM traffic (N=32k, 1 head) | ~40 GB | ~4 GB |
| Runtime (A100) | 41.7 ms | 7.3 ms |
| Memory for backward | O(N²) stored | O(N) — recompute from stored softmax stats |

**FlashAttention-2** improves warp-level parallelism: splits the Q dimension across warps rather than the K/V dimension, reducing the fraction of time warps spend waiting for shared memory. ~2× throughput improvement.

**FlashAttention-3** (H100-specific) adds asynchronous TMA (Tensor Memory Accelerator) for pipelined data loading, and Warpgroup MMA for WGMMA instructions. Overlaps data loading with compute — while one warp group computes on loaded data, another loads the next tile. Approaches 75% of H100 peak BF16 throughput.

> **Interview question:** FlashAttention's backward pass "recomputes" the attention matrix rather than storing it. Why is this faster than storing the O(N²) attention matrix despite the extra computation?
>
> *The backward pass needs the attention weights `P = softmax(QKᵀ/√d)` to compute gradients for Q, K, V. Standard attention stores P to HBM during the forward pass (O(N²) bytes) and reads it back during backward (O(N²) bandwidth). FlashAttention stores only the softmax statistics (m, l) — O(N) bytes — and recomputes P from Q, K, m, l on the fly during the backward pass. The recomputation cost is exactly one attention computation worth of FLOPs, which is much cheaper than the O(N²) HBM traffic to store and reload P. The key: GPUs are compute-rich but bandwidth-starved. At N=32k, storing and reloading P costs 2 × N² × 2 bytes = 4GB of HBM traffic per head. Recomputing P costs 2N² FLOPs per head. At A100 roofline (compute/bandwidth = 156 FLOP/byte), recomputation is ~12.5× cheaper than the equivalent bandwidth cost. This is the general principle of "rematerialisation" — trade cheap compute for expensive memory.*

### Flash-Decoding
{: #flash-decoding}

**The single-query bottleneck.** FlashAttention parallelises across the query dimension — it's fast when there are many query tokens (prefill phase). During autoregressive decode, there is exactly one new query token. All the K/V sequence of length N must be processed serially by a single kernel. At N=32k tokens, this is a slow sequential scan.

**Flash-Decoding** parallelises the K/V dimension instead. Split the K/V sequence into `P` chunks; assign each chunk to a separate CUDA thread block:

```
Thread block i processes K[i*chunk_size : (i+1)*chunk_size], V[...]

Output partial: Oᵢ, lseᵢ (log-sum-exp for local softmax normalisation)

Final reduction across P partial outputs:
  lse_global = log(Σ exp(lseᵢ))
  O = Σ Oᵢ × exp(lseᵢ - lse_global)   # correct global softmax weighting
```

The reduction is valid because softmax over K/V positions is associative in log-space — the same mathematical identity as online softmax. Each thread block runs independently and writes its partial result to a small intermediate buffer; a final lightweight kernel performs the reduction.

At N=32k and P=8 splits: 8 thread blocks run in parallel vs 1, each handling 4k tokens. On an A100 with 108 SMs, this gives ~8× better SM utilisation during the decode attention step. Benchmarks show 8× faster decode attention at 32k context.

> **Interview question:** Flash-Decoding adds a reduction step across P partial outputs. Doesn't this reduction negate the parallelism benefit?
>
> *No, because the reduction is O(P) work — trivially small. P is typically 8–64, so the reduction is 8–64 additions. The main kernel work is O(N/P) per thread block, parallelised across P blocks — this dominates. The reduction is a negligible constant. More precisely: without Flash-Decoding, one thread block processes all N K/V pairs sequentially — O(N) serial steps. With Flash-Decoding, P thread blocks each process N/P pairs in parallel, then a reduction in O(P) steps. Total steps: O(N/P + P). Minimised at P = √N: O(2√N). For N=32768, this is ~360 steps vs 32768 without — a 90× speedup in the theoretical model. In practice, GPU parallelism is bounded by SM count and memory bandwidth, so the actual speedup is 8–10× at N=32k, but the principle holds.*

### Kernel Fusion
{: #kernel-fusion}

**The kernel launch overhead problem.** Each GPU kernel call has fixed overhead: ~5–10µs of CPU-GPU synchronisation, plus memory traffic to write intermediates to HBM and read them back for the next operation. A transformer block has 20–30 separate operations (matrix multiplications, normalisations, activations, residuals) each of which, unfused, writes its output to HBM and passes it to the next kernel.

**Operator fusion** merges adjacent operations into a single kernel, keeping intermediates in registers or shared memory and never writing them to HBM.

| Unfused sequence | Fused kernel | HBM traffic saved |
|---|---|---|
| RMSNorm → linear | FusedRMSNormLinear | Intermediate norm output (~d floats per token) |
| Linear → SiLU → Linear (SwiGLU) | FusedSwiGLU | SiLU activation result |
| RoPE → QKV split → attention | FusedRoPEAttention | RoPE output tensor |
| Linear → residual → RMSNorm | FusedLinearResNorm | Linear output tensor |
| INT4 load → dequantise → matmul | FusedDequantMatmul | Decoded FP16 weight tensor |

**Quantised kernel fusion** is the most impactful for W4A16 inference. Without fusion: load INT4 weights from HBM → write FP16 decoded weights to HBM → load FP16 for matmul. With fusion: load INT4 weights from HBM → dequantise in registers → matmul directly. The saved HBM write+read is the same size as the original weight matrix — a 2× reduction in weight traffic (already loaded once for INT4; avoid the extra FP16 round-trip).

**Triton** enables custom fused kernels without writing raw CUDA: express the tile structure, specify load/store patterns, and Triton's compiler handles warp scheduling, register allocation, and memory coalescing. This dramatically lowers the barrier to writing production-quality fused kernels — vLLM, SGLang, and most modern inference stacks implement their hot-path kernels in Triton.

> **Interview question:** You profile a 7B model and find the bottleneck is not the attention or FFN matmuls but the LayerNorm and residual add operations. How would you address this, and why does it happen?
>
> *This is a classic memory-bandwidth bottleneck on elementwise operations. LayerNorm requires: (1) load x from HBM, (2) compute mean and variance (reduction), (3) normalise, (4) scale and shift with γ, β, (5) write output to HBM. The residual add: (6) load residual from HBM, (7) add, (8) write. Each step that touches HBM costs bandwidth. For a 7B model with d=4096, each token's LayerNorm output is 4096 × 2 bytes = 8KB. At sequence length 2048, batch 32: 32 × 2048 × 8KB × 2 (read + write) = 1GB per LayerNorm. With 96 layers × 2 LayerNorms = 192 LayerNorms = 192GB of HBM traffic just for normalisation. The fix: fuse LayerNorm with the subsequent linear layer — load x once, compute the norm in shared memory, multiply by the weight matrix without writing the normalised activations to HBM. Similarly, fuse the residual add with the LayerNorm that follows it. This eliminates ~192GB of HBM traffic per forward pass. It happens at 7B because 7B models are deeply memory-bandwidth-bound at small batch sizes — the elementwise ops, though low in FLOP count, have very low arithmetic intensity (1–2 FLOP/byte) and saturate the memory bus.*

---

## Prefill-Decode Disaggregation
{: #disaggregation}

**Why co-location is suboptimal.** Prefill and decode have fundamentally different resource profiles:

| | Prefill | Decode |
|---|---|---|
| Compute pattern | All prompt tokens in parallel → high arithmetic intensity | One token at a time → low arithmetic intensity |
| Bottleneck | Compute (tensor cores are saturated) | Memory bandwidth (loading weights per token) |
| Sensitivity | TTFT (time to first token) — moderately latency-sensitive | TBT (time between tokens) — highly latency-sensitive |
| Optimal batch | Longer prompts fill the tensor cores | Larger decode batch amortises weight loads |

Running both on the same GPUs forces compromises: decode batches are interrupted by expensive prefill phases (spiking TBT), and prefill cannot fully saturate tensor cores because the GPU must also keep the decode batch moving. Neither phase is optimally served.

**Disaggregation** assigns prefill and decode to separate GPU pools:

<div class="post-flow post-flow--compare" role="group" aria-label="Prefill vs decode pools">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Prefill Pool</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Runs only prefill — tensor cores always busy</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Batches many long prompts together for compute efficiency</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Transfers finished KV cache to decode pool via NVLink / RDMA</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Decode Pool</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Runs only decode — no prefill interruptions, stable TBT</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Maximises decode batch size — amortises weight loads across many requests</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Can use different GPU types — memory-bandwidth-optimised (e.g. HBM3)</span></li>
    </ol>
  </div>
</div>

**The KV transfer bottleneck.** After prefill completes, the KV cache (prompt_len × n_layers × 2 × n_heads × d_head bytes) must be transferred from the prefill GPU to the decode GPU. For a 1000-token prompt in a 70B model: 1000 × 96 × 2 × 8 × 128 × 2 bytes = ~393MB. Over NVLink (112 GB/s): 3.5ms. Over PCIe between nodes (32 GB/s): 12ms. Over RDMA (100 GbE, ~12 GB/s): 33ms. This transfer latency adds directly to TTFT and can dominate if network bandwidth is insufficient.

**DistServe** pipelines the KV transfer: as the prefill GPU generates KV entries layer by layer, it streams each layer's KV to the decode GPU before the next layer is computed. The decode GPU can start decoding as soon as the final layer's KV arrives — overlapping transfer with prefill computation. This hides most of the transfer latency.

**Splitwise** takes a more radical approach: prefill GPUs are GPU-rich (high compute), decode GPUs are memory-bandwidth-rich. Different hardware SKUs optimised for each phase. The routing system assigns each incoming request to a prefill GPU, executes prefill, transfers KV, hands off to a decode GPU pool. Reported 2–3× throughput improvement over co-located serving.

**When disaggregation isn't worth it.** The KV transfer adds engineering complexity and TTFT latency. For short prompts (< 200 tokens), prefill is cheap and the transfer overhead dominates. For applications that are latency-sensitive on TTFT rather than TBT (search, where the first response matters most), disaggregation may hurt. Best suited for: long-context RAG applications, document processing, and chat where users tolerate slightly higher TTFT for dramatically better TBT.

> **Interview question:** Your disaggregated serving system is meeting TBT SLOs but TTFT has spiked. The prefill pool has high GPU utilisation but the decode pool is underutilised. What's happening and how do you fix it?
>
> *Classic prefill bottleneck: the prefill pool is saturated — it's processing requests as fast as it can, but requests are queuing in front of it. Meanwhile the decode pool is starving for new requests to work on. The decode pool has capacity but nothing to decode because the prefill pool can't hand off fast enough. Diagnoses and fixes: (1) Scale up prefill pool — add more GPUs. Since prefill is compute-bound, this directly increases throughput linearly. (2) Improve prefill batching — are long and short prompts batched together? Short prompts waste tensor core compute because the batch isn't large enough to saturate the matmuls. Bucket by length, batch similar-length prompts together. (3) Check KV transfer bandwidth — if the prefill pool finishes but is waiting to transfer KV, you have a network bottleneck. Upgrade to NVLink within node or RDMA across nodes. (4) Enable chunked prefill — break very long prompts into chunks so other requests don't wait for one huge prefill to finish. (5) Increase prefill batch size — prefill benefits from large batches (compute-bound regime), so packing more prompts per prefill call increases throughput per GPU.*

---

## Model Compression
{: #model-compression}

Quantisation reduces the bit-width of existing weights. Model compression goes further: it **physically removes parameters**, changing the architecture itself. The goal is the same — reduce memory footprint and improve tokens/second — but the mechanism differs fundamentally. Compression also unlocks a "train large, deploy small" workflow: train one high-capacity model, then compress it into a family of smaller variants without paying the full training cost each time.

**The core economic argument.** Training a 13B model from scratch on 15T tokens costs roughly $4.2M at $1.5/hr A100 rates. A separate 7B, 5B, 3B, and 1B would collectively cost ~$12M. Compression techniques (prune + distill on 100–400B tokens) reduce that total to ~$4.4M — a 2.7× reduction — while achieving higher accuracy than training each model from scratch because the compressed models inherit the larger model's learned representations.

### Speculative Decoding: Deeper Mechanics
{: #speculative-decoding-depth}

The speculative decoding section above covers the core mechanic. Here we go deeper on the **lossless guarantee**, the **speedup equation**, and the **drafter complexity ladder**.

**Why lossless rejection sampling works.** Let `q(x)` be the draft model's probability and `p(x)` the target model's probability for a given token. The acceptance rule:

- Accept with probability `min(1, p(x)/q(x))`
- If rejected: resample from `p'(x) ∝ max(0, p(x) − q(x))`

This guarantees the accepted sequence follows the target distribution `p` exactly. When `p(x) ≥ q(x)`, the token is always accepted (the target agrees or is more confident). When `p(x) < q(x)`, the draft was overconfident — acceptance is partial, and the residual distribution `p' = max(0, p-q)` normalised covers the probability mass that `q` "stole" from lower-confidence regions. The result: the final token is always drawn from `p`, never from `q`. This is a strict mathematical guarantee, not an approximation.

**Expected speedup formula.** Let `τ` = acceptance rate per drafted token, `K` = number of drafted tokens per round, `c` = cost of drafting K tokens relative to one target step:

```
S = (1 − τ^(K+1)) / ((1 − τ)(cK + 1))
```

At `τ = 0.8`, `K = 5`, `c = 0.1`: `S = (1 − 0.8^6) / (0.2 × 1.5) ≈ 2.7×`. The drafter must be cheap (low `c`) and well-aligned (high `τ`) — a large, misaligned drafter increases `c` without proportionally improving `τ`, destroying the speedup.

**Drafter complexity ladder** (cheapest → most efficient acceptance rate):

| Method | Where drafts live | Overhead | Typical τ |
|---|---|---|---|
| n-gram | Token trie from context | ~0 | 0.4–0.6 |
| Medusa | Extra prediction heads on target | Small (no separate model) | 0.6–0.75 |
| EAGLE | Drafts in hidden-state space | Moderate (extra AR step in feature space) | 0.75–0.85 |
| MTP (Multi-Token Prediction) | Target trained with MTP head | Built into training | 0.8–0.85 |
| External drafter | Separate small model | Highest | Up to 0.9 |

**EAGLE** avoids the token-space alignment problem: instead of predicting discrete tokens, it drafts the next **hidden state** autoregressively, then decodes tokens from that state. This works because the feature space is smoother than the token space — small distribution mismatches in logits correspond to large mismatches in token probabilities, but similar hidden states often produce similar distributions.

**MTP in production.** DeepSeek-R1 trains with MTP heads — the model produces not just the next token but the next `K` tokens in parallel during training. At inference, these heads serve as zero-cost drafters. Nemotron-3-Super drafts 3 and accepts 2.5 on average; Qwen3.5 drafts 8 and accepts 5.5.

> **Interview question:** You're deploying a speculative decoding system with EAGLE. The EAGLE drafter uses the target model's hidden states as input. Why does this improve acceptance rate over a separate small model drafter, and what's the failure mode?
>
> *The root cause of low acceptance rates is distribution mismatch: the drafter model sees different activations (because it has a different architecture and was trained differently) and develops a subtly different distribution. EAGLE sidesteps this by running directly in the target model's hidden-state space — the draft network sees the same representations the target would use, so its predictions are better calibrated to the target's distribution. The failure mode: the EAGLE drafting step is an additional autoregressive pass over a shallow network that consumes the target's hidden states. If the target model's hidden states change distribution (e.g. due to quantisation or fine-tuning), EAGLE must be retrained. More importantly, EAGLE's draft step adds latency — if the target model is fast enough (large batch, prefill-heavy workload) that the drafter's overhead exceeds the saving from accepted tokens, speculative decoding with EAGLE is net-negative. EAGLE works best for small-concurrency or single-user scenarios where memory bandwidth is the binding constraint per request.*

### Pruning
{: #pruning-depth}

**What pruning does.** Quantisation shrinks the representation of existing weights. Pruning **removes weights entirely** — creating a smaller dense matrix (structured pruning) or a sparse matrix (unstructured/semi-structured pruning).

**Three pruning granularities:**

<div class="post-flow" role="list" aria-label="Pruning granularity ladder">
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted"><strong>Unstructured</strong> — individual weight zeroing. Maximum flexibility, but sparse matrices don't map well to GPU tensor cores. Requires specialized hardware (e.g. NVIDIA A100 sparse cores) to realise speedups.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>2:4 semi-structured (N:M sparsity)</strong> — exactly 2 of every 4 consecutive weights are zeroed. NVIDIA Ampere+ GPUs have dedicated 2:4 sparse tensor core support: 2× throughput on 50% sparse weights, no software overhead. The pattern constraint limits accuracy vs unstructured, but hardware support makes it immediately deployable.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--green"><strong>Structured</strong> — remove entire attention heads, MLP neurons, or transformer layers. Result is a smaller dense matrix — immediate speedup on all hardware with no sparsity support. The coarsest granularity but the most universally deployable.</span></div>
</div>

**Importance estimation.** The core challenge: which parameters can be removed with minimal accuracy loss? Common signals:

- **Magnitude**: weight magnitude as a proxy for importance. Fast but ignores interactions.
- **Activation magnitude**: for each attention head or MLP neuron, measure the mean activation magnitude over a calibration corpus. Heads with near-zero activations contribute little to outputs.
- **Gradient-based (Hessian)**: parameters that, when perturbed, cause large loss increases are important. The Optimal Brain Compression (OBC) framework uses second-order information: `importance(w_i) ∝ w_i² / [H⁻¹]_{ii}`. Expensive (O(d³) Hessian inversion) but more accurate.

**Depth vs width pruning tradeoff.** Empirical finding from LLaMA-3-8B experiments:

| Strategy | Speed | Accuracy | Reasoning impact |
|---|---|---|---|
| **Depth pruning** (remove layers) | Faster (fewer sequential ops) | Lower | Severe (GSM8k: 16.8 vs 41.2 after removal of 25% layers) |
| **Width pruning** (reduce heads/neurons) | Slightly slower | Higher | Milder |

Depth pruning is faster at inference because it reduces the sequential depth of the network — no pipeline waiting. But reasoning tasks depend critically on depth: multi-step logical reasoning requires many transformer layers to "think". Width pruning removes less-used dimensions within each layer, preserving the full sequential depth.

> **Interview question:** You're instructed to prune a 13B reasoning model to 7B. Would you prefer depth or width pruning, and how would you measure success?
>
> *For a reasoning model, default to width pruning. Reasoning tasks (math, multi-step logic) are the most sensitive to depth reduction — each layer performs one step of "processing", and removing layers collapses the chain of reasoning. Width pruning within each layer removes less-used attention heads and MLP dimensions while preserving the full 32/40/96-layer depth. Measurement: evaluate on GSM8K (math reasoning), MATH, and HumanEval before and after. Compare to a from-scratch 7B baseline — the pruned model should match or exceed it on these benchmarks. Also measure latency per token at batch size 1 (the memory-bandwidth regime) and batch size 32 (the compute regime). If the 7B pruned model underperforms the 7B baseline on reasoning by more than ~5%, consider (1) increasing distillation tokens, (2) using iterative pruning (prune to 10B, recover, prune to 8B, recover, prune to 7B), or (3) switching to width+depth hybrid if the latency target demands it.*

### Minitron: Prune Then Distill
{: #minitron}

**The key insight.** Pruning alone leaves a degraded model. Training from scratch produces a well-optimised model but costs 15–25T tokens. Minitron combines both: prune aggressively, then recover with distillation on just 100–400B tokens. The compressed model inherits the large model's learned representations as a starting point, so recovery is fast.

**Five-step Minitron methodology:**

<div class="post-flow" role="list" aria-label="Minitron pipeline">
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">1. Train a large model (e.g. Nemotron-4-15B, LLaMA-3.1-8B)</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">2. Rank parameters by activation-magnitude importance on a calibration corpus (Wikipedia subset). Score each attention head, MLP neuron, and layer.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">3. Neural Architecture Search (NAS) via Integer Linear Programming: given a target parameter budget (±5%), enumerate feasible architectures across depth, width, heads, and MLP ratios. Select the best via lightweight fine-tuning.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">4. Prune: remove least-important heads, neurons, and layers to reach the target architecture.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">5. Distill: fine-tune the pruned student against the large teacher using KL divergence on soft logits. If original training data isn't available (e.g. Mistral-NeMo-12B), first "teacher-correct" the teacher model with ~100B tokens before distillation.</span></div>
</div>

**KL divergence as the distillation loss.** Hard labels (one-hot) throw away the teacher's confidence information. Soft labels — the full teacher logit distribution — encode which wrong answers are "almost right". KL divergence `KL(p_teacher || q_student) = Σ p(x) log(p(x)/q(x))` minimises the student's deviation from the teacher's full distribution, not just its top-1 prediction. This transfers "dark knowledge": the teacher assigns 0.3% probability to Class B and 0.1% to Class C, which tells the student how semantically related the classes are.

**Results.** Pruning a 13B model to 10B costs $0.06M vs $3.2M from scratch — a 53× reduction. The pruned+distilled 10B model achieves +18% MMLU improvement over a same-size from-scratch baseline. Producing the full model family (13B, 10B, 7B, 5B, 3B) costs $4.4M total vs $12M from scratch — a 2.7× saving.

**When original training data isn't available.** Proprietary models (Mistral, Llama) don't release training data. Minitron's solution: teacher correction — fine-tune the teacher on a general corpus for ~100B tokens to "re-warm" its output distribution, then distill. This removes the distribution shift between the teacher's original training data and the available distillation corpus.

> **Interview question:** Minitron uses KL divergence to train the student against the teacher's soft labels. Why not just use cross-entropy on the teacher's top-1 predictions?
>
> *Top-1 cross-entropy discards the teacher's probability mass on non-top tokens — exactly the information that teaches generalisation. Consider classifying a sentence: the teacher assigns 70% to "positive", 20% to "neutral", 10% to "negative". Top-1 training sees only "positive" — the student learns nothing about the relationship between the classes. KL divergence `Σ p log(p/q)` uses all three probabilities, forcing the student to match the teacher's uncertainty and inter-class similarity. This is "dark knowledge" — the teacher's soft predictions encode more information per example than a hard label. Practical consequence: KL distillation typically requires 2–3× fewer training tokens to reach the same accuracy as hard-label training, because each example carries more signal. The tradeoff: KL divergence requires storing teacher logits (or running teacher inference online), which doubles memory or adds latency. For a very large teacher, offline logit generation is preferred: generate teacher logits once, store, then distill offline.*

### Elastic Models: Flextron and LlamaFlex
{: #elastic-models}

**The problem with the pruning-per-model approach.** Even with Minitron's 53× cost reduction, you still run a separate pruning+distillation pass for each target size. If deployment requirements change (new hardware, different latency budget, unexpected load), you start over. Elastic models solve this differently: train **one model that contains many sizes**.

**Flextron's core idea.** Starting from a pretrained model (e.g. LLaMA-3-8B):

1. **Permute** attention heads, MLP channels, and hidden dimensions by importance — most important first.
2. **Elastic continued training** (using ~5% of original training tokens): train a router jointly with the model. The router selects a sub-architecture for each forward pass given a latency or parameter constraint.
3. At inference: **slice** the model at the desired budget. Because weights are importance-sorted, the top-k heads/channels are always the most important subset.

The router adjusts `n_layers`, `hidden_size`, `n_heads`, and `MLP_width` dynamically. LlamaFlex (v2, ICLR 2025) extends this to a layer-level MoE: each layer has a nested expert structure where each expert includes all smaller experts. A per-layer router selects optimal width, enabling zero-shot model sampling from 1.75B to 7B from a single 8B checkpoint — close to Minitron static pruning quality without separate pruning+distillation runs.

**Cost comparison:**

| Approach | Training tokens | Result |
|---|---|---|
| From scratch (per size) | 15T × n_sizes | n separate models |
| Minitron (per size) | 100–400B × n_sizes | n separate models |
| Flextron / LlamaFlex | 60B total (5% of 15T) | One model → any size |

**Nemotron-Elastic on Nemotron-Nano-v2.** Applied to hybrid Mamba-Transformer architectures: compress MoE expert width, hidden dim, and top-k routing. Constant deployment cost — the same physical model serves different size requests by routing to different subsets of experts. Enables "adjust inference cost per token" dynamically: long reasoning chains can use the full model; simple queries route to the compact subnetwork.

> **Interview question:** Flextron uses importance-sorted weight ordering so that slicing the top-k weights gives the best k-width subnetwork. Why does sorting by importance enable this, and what happens to the training procedure?
>
> *Standard weight ordering is arbitrary — weight 1 and weight 512 in a layer have no ordering relationship. Sorting by importance makes the ordering meaningful: the top-k weights are always the highest-importance subset. This "nesting" property means any width-k slice (k ≤ K_max) is a valid, well-structured subnetwork — you're always keeping the most important parameters and discarding the least important ones. The training procedure must adapt: during elastic training, each forward pass randomly samples a width from the allowed set {k_min, ..., k_max} (or is selected by the router). The gradient update applies to the selected subnetwork. This is equivalent to training all subnetworks simultaneously with shared weights — each smaller subnetwork's gradient flows through only the parameters it uses, so larger widths get updated every step while smaller ones get updated as their width is sampled. The ordering sort must be done before elastic training begins, because once elastic training starts, the gradient updates will tend to reinforce the initial importance ordering — if you sort mid-training, the early-trained "random" ordering will confuse the router.*

### Neural Architecture Search: LANA and Puzzle
{: #nas-lana-puzzle}

**Why NAS matters post-training.** Even after choosing a pruning ratio, there are thousands of valid architectures at a given parameter count — different numbers of layers, heads, MLP ratios, attention types. LANA and Puzzle are two approaches to efficiently searching this space **after a model is trained**, rather than during training.

**LANA (Latency Aware Network Acceleration).** Given a trained model and a target constraint (latency, memory, or cost), LANA:

1. Identifies "training-friendly" operations in the trained model (standard GELU, full multi-head attention)
2. Swaps them for hardware-friendly alternatives (ReLU, GQA, Flash Attention, SWA) that satisfy the constraint
3. Uses ILP (Integer Linear Programming) to find the swap configuration in under 10 minutes
4. Quick fine-tuning (5–30% of full training) to recover accuracy

The key insight: the trained model's weights are "close" to the optimal weights for the new architecture, so only a short recovery fine-tune is needed. Block-level distillation (MSE on per-block outputs) provides strong supervision without cross-layer gradient propagation — cheap because each block's distillation is independent.

**Puzzle: NAS as a Knapsack Problem.** Puzzle scales NAS to 70B+ models by decomposing the search into:

1. **Score each block variant independently** (replace just one block at a time, measure MMLU/KL divergence). This done-once scoring is the expensive step, but only runs once per parent model.
2. **Relax the global score** as a sum of per-block scores — a simplification that makes the search tractable.
3. **Mixed Integer Programming** solves for the best combination of block variants under memory and runtime constraints — in seconds, not hours.
4. **Healing**: adjacent block combinations may be incompatible (error accumulation). Run short KD on multiple candidate architectures, pick the best, then run full KD.

Applied to GPT-OSS 120B → 88B: 1.22–2.82× throughput improvement (depending on GPU count) with 100%+ accuracy retention. Techniques used: heterogeneous MoE expert pruning (8–128 experts per layer, independently configured), selective replacement of full-context attention with window attention (8 of 18 global attention layers), FP8 KV-cache quantisation.

**Architecture insight from Puzzle experiments.** Not all transformer blocks are equally compressible. Attention layers near the beginning and end of the network are harder to prune (they set up representations and collect them). Middle layers show more redundancy — adjacent layers often learn similar transformations. MLP layers are generally more compressible than attention because attention's role in routing information is harder to replicate with fewer heads.

> **Interview question:** Puzzle relaxes the global model score as a sum of per-block scores to make search tractable. Why is this relaxation an approximation rather than exact, and how does the "healing" step compensate?
>
> *The relaxation is an approximation because transformer blocks are not independent: the output of block `i` is the input to block `i+1`. If you replace block `i` with a compressed variant, its output distribution shifts — which means block `i+1` sees different inputs than when you scored it in isolation. The per-block score was measured by replacing only one block at a time (keeping all others at full capacity), so it doesn't account for these cross-block interactions. When you assemble many compressed blocks together, the individual approximation errors compound (error accumulation). The healing step compensates: after selecting the best architecture via ILP, run a short KD pass on multiple candidate architectures to let the blocks adapt to each other's compressed outputs. The KD loss operates end-to-end, so gradients flow across block boundaries — the healing pass effectively fine-tunes inter-block compatibility. Healing is cheap because (1) only short KD is needed (the weights are already close to optimal), and (2) running it on multiple candidates in parallel and selecting the best finds a good solution without an expensive long search. Without healing, the ILP solution would be architecturally sound but practically suboptimal due to accumulated distribution shift.*

---

## Serving Frameworks & System Architecture
{: #serving-frameworks}

LLM inference is not "a GPU running a model." It is a **distributed system** comprising an API layer, a smart router, a scheduler, a runtime executor, a KV memory manager, distributed communication, and an observability stack. Each layer has its own design tradeoffs, and performance problems are usually caused by the wrong layer being blamed.

**The workload has changed.** Early LLM serving was single-turn chat on dense models. Today we serve:
- Reasoning models with long, bursty completions
- Agentic workflows revisiting large tool traces across turns
- Long-context assistants with 32K–1M token contexts
- Multimodal inputs (image, audio) that change batching semantics
- MoE models requiring expert dispatch and all-to-all collectives

These workloads make **memory movement, KV reuse, queueing policy, and routing** matter as much as raw kernel throughput.

### Serving System Layers
{: #serving-system-layers}

**Per-request latency decomposition:**

```
TTFT = t_queue + t_tokenize + t_prefill + t_sample + t_first_byte
ITL  ≈ t_decode_step + t_sample + t_stream_flush
```

TTFT is dominated by prefill (compute-bound). ITL is dominated by decode (memory-bandwidth-bound). Optimising one can worsen the other — chunked prefill trades prefill throughput for better ITL stability.

**The KV memory accounting equation:**

```
KV_bytes ≈ T × 2 × L × H_kv × d_h × b
```

where T = context length, L = layers, H_kv = KV heads (not query heads with GQA/MQA), d_h = head dim, b = bytes/element. For a model with L=80, H_kv=8, d_h=128, b=2 bytes: **320 KiB per token**. 32 concurrent requests at 8K context = 80 GiB of KV state before model weights. This is why KV management is a first-order design variable for long-context workloads.

**Subsystem map:**

<div class="post-flow" role="list" aria-label="Serving engine subsystems">
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Frontend / API</strong> — OpenAI-compatible REST, validation, tokenization, request-state tracking. Rust rewrite in vLLM V1 eliminates Python overhead on the critical path.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Router / Gateway</strong> — LLM-aware: routes by cache affinity (which worker already has this prefix?), phase balance (prefill-heavy vs. decode-heavy), pool type (PD disaggregation), and request shape (long vs. short).</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Scheduler</strong> — per-step decisions: which requests to admit, token budget split between prefill and decode, whether to chunk a long prefill, whether speculative draft tokens fit, whether to preempt under pressure.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>KV Memory Manager</strong> — paged allocation, prefix reuse tracking, eviction, host-offload, cross-engine KV transfer for PD disaggregation. No longer a local allocator — part of a distributed cache hierarchy: HBM → host DRAM → SSD → remote KV service.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Executor / Runtime</strong> — launches the forward pass: attention kernels, fused MLP/norm, sampling, CUDA graph capture/replay, TP/EP/PP collectives, paged KV metadata for scatter/gather.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--green"><strong>Observability</strong> — Prometheus metrics: queue depth, running requests, cache hit rate, TTFT/ITL histograms, offload/reload rates, all-to-all latency. Without telemetry, tuning is superstition.</span></div>
</div>

> **Interview question:** A traditional HTTP load balancer routes LLM requests round-robin. What specifically does it miss, and what does an LLM-aware router need to know?
>
> *A round-robin balancer sees only replica count and request rate — it has no knowledge of what's inside each inference engine. It misses: (1) Cache affinity: if Worker A already has the KV blocks for a system prompt that appears in 90% of requests, routing those requests to Worker B wastes the cache hit. An LLM-aware router sends requests with matching prefixes to workers that have those prefix KV blocks warm. (2) Phase balance: a worker mid-way through 10 long decode sequences has very different available capacity than one just finishing 10 prefills — even if both have identical CPU load. (3) Pool heterogeneity: in PD-disaggregated serving, prefill workers and decode workers have entirely different resource profiles; routing a prefill to a decode-only worker is catastrophic. (4) Request shape: a 100K-token context request needs a worker with a large available KV budget, not just any available worker. An LLM-aware router needs to query each worker's current KV usage, running sequences, cache contents (or at least prefix hashes), and phase state. This is why modern stacks (Dynamo's Smart Router, SGLang's gateway, vLLM's router) are separate components purpose-built for LLM traffic.*

### Scheduler Design
{: #scheduler-design}

The scheduler makes per-engine-step decisions. Its choices encode the product's latency vs. throughput tradeoff:

**Six per-step decisions:**
1. Which waiting requests to admit (admission control)
2. Token budget split: how much to decode vs. prefill
3. Whether to chunk a long prefill (chunked prefill)
4. Whether speculative draft tokens fit the token/page budget
5. Whether to reload offloaded KV for a request about to be served
6. Which active requests to preempt under memory pressure

**Chunked prefill tradeoff.** A 32K-token prompt would dominate an engine step, blocking all decoding requests. Chunked prefill splits the prompt into N-token chunks: the scheduler interleaves prefill chunks with decode steps. vLLM V1 enables this by default, prioritising decode and using remaining budget for prefill chunks.

```
Chunk too aggressively → scheduler overhead dominates, prefill throughput falls
Chunk too little     → one giant prefill destroys tail ITL for all concurrent requests
```

**p99 tail latency killers** (almost never caused by slow kernels):
- Long prompt admitted at the wrong time
- Cache miss on a large prefix when demand spikes
- MoE expert imbalance — one expert gets 8K tokens while another gets 1K
- Speculative decoding overconsumes page budget, starving new admits
- Transport retries in PD mode under congestion
- Host-side grammar/tool logic stalling the decode stream

> **Interview question:** A scheduler optimised for median latency performs terribly at p99. Why, and what scheduling change would fix it?
>
> *Median-optimised schedulers tend to use greedy admission: admit requests as fast as possible, keep GPU utilisation high. This works when all requests are similar length — but in production, 5% of requests have 10× longer prompts or 10× longer outputs. A single long-prompt request admitted at the wrong time monopolises the engine for many steps, causing all concurrent requests to stall (decode starvation). The p99 spike is the waiting time for that long request to clear. Fixes: (1) Chunked prefill — break the long prompt into chunks, let decode proceed between chunks. This caps the per-step latency impact of any one request. (2) Priority scheduling — classify requests by expected cost (prompt length is known at admission; output length can be estimated) and deprioritise very long requests during high-load periods. (3) Preemption — if a large request is monopolising the KV budget, pause it (checkpoint its KV to host), serve other requests, then resume. (4) Separate queues by SLO class — interactive chat in one queue (decode-optimised, small chunks), batch processing in another (throughput-optimised). The key insight: p99 is a tail problem caused by heterogeneous request shapes, and the fix is controlling how long any one shape can block others — not faster average execution.*

### Framework Landscape
{: #framework-landscape}

| Framework | Think of it as | Core strengths | Best fit | Watch-outs |
|---|---|---|---|---|
| **vLLM** | General open engine | Broad model support, PagedAttention, community ecosystem, strong scheduler | Research-to-production, many model families | May not peak on a single NVIDIA path |
| **SGLang** | Frontier-traffic engine | RadixAttention, PD disaggregation as first-class, large-scale EP, HiCache, cache-aware gateway | MoE/reasoning at scale, frontier deployments | Evolving; large operational surface |
| **TensorRT-LLM** | NVIDIA-specialised runtime | Peak NVIDIA kernel stack, overlap scheduler, FP8/FP4 support, AutoDeploy graph transformation | NVIDIA-only production, max throughput on Hopper/Blackwell | Less portable; some feature combinations in beta |
| **NVIDIA Dynamo** | Distributed control plane | GPU resource planner, smart KV-aware router, low-latency KV transfer, multi-tier KV manager | Multi-node reasoning model serving | Needs a runtime engine underneath; operational complexity |
| **LMCache** | KV tier / extension | Cross-query KV reuse, host-offload persistence, cross-engine KV transfer, composable with vLLM/SGLang | Cache-heavy workloads, long-context RAG, PD systems | Only helps if workload has reusable KV |

**How these compose.** Modern production stacks layer these components:

<div class="post-flow" role="list" aria-label="Serving stack composition">
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Clients → Gateway / Smart Router (cache-aware, PD-aware routing)</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Prefill pool: vLLM / SGLang / TRT-LLM (compute-heavy, throughput-oriented)</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Decode pool: vLLM / SGLang / TRT-LLM (bandwidth-heavy, latency-sensitive)</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">KV tier: HBM → host DRAM → remote/SSD (LMCache / HiCache / Dynamo KV Manager)</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Control plane: autoscaling, resource planning, canary analysis (Dynamo / Ray Serve)</span></div>
</div>

**SGLang distinctive features:**
- **RadixAttention**: a radix-tree-indexed KV cache enabling O(prefix_length) lookup for cache hits. Requests sharing prefixes share KV blocks — key for agentic workloads where tool descriptions are repeated across thousands of calls.
- **HiCache**: hierarchical KV caching beyond GPU memory. Hot pages in HBM, warm pages spilled to host DRAM, cold reusable history in remote or disk-backed stores. Reported 6× throughput improvement and large TTFT reductions for long-session workloads.
- **Large-scale EP**: expert parallelism at the scale where all-to-all becomes the binding constraint. SGLang pairs EP with PD disaggregation to keep expert dispatch efficient during decode.

**TensorRT-LLM AutoDeploy.** Takes a model in ordinary PyTorch/HuggingFace form and automatically extracts a computation graph, applies sharding, quantisation, KV integration, layer fusion, and CUDA-graph-friendly rewriting. Reduces the engineering tax from "research model" to "tuned deployment" — no manual graph surgery required.

> **Interview question:** You're building a serving stack for a large MoE reasoning model (Kimi K2.5 scale, 1T parameters). What parallelism strategy would you use, and why?
>
> *At 1T parameters, no single GPU or even single node fits the model. You need a combination of parallelism types: Expert parallelism (EP) for the MoE layers — distribute different experts to different GPUs. EP is preferable to tensor parallelism for MoE because it has smaller communication volume: tokens are routed to experts on specific GPUs (all-to-all), which is cheaper than TP's all-reduce on every linear layer. Attention layers need separate handling — EP only applies to MoE FFN layers; attention still needs DP or TP. For high-throughput inference, use DP-attention (different requests' attention on different GPUs) paired with EP for FFN. Data parallelism (DP) at the replica level for independent scaling. PD disaggregation: separate prefill and decode pools. For a reasoning model with long outputs, prefill is cheap relative to decode — disaggregation lets you tune each pool independently. KV transfer between pools via NIXL or UCX. The Kimi K2.5 case study confirms this: high-throughput mode uses P:10×DEP-4-GPUs, D:1×DEP-24-GPUs — much more decode capacity, reflecting the long reasoning outputs. Low-latency mode (non-PD, TP=4) gives lower overhead for interactive use cases.*

### Prefill-Decode Disaggregation in Production
{: #pd-disaggregation-systems}

The disaggregation section earlier covered the fundamental tradeoff. Here we go deeper on **when it helps, what it costs, and how the transport layer works**.

**The PD benefit inequality:**

```
PD helps if:
  Δt_interference_avoided > t_KV_transfer + t_orchestration + t_placement_mistakes
```

PD is attractive when: prompts are long and variable, decode latency matters (interactive chat, voice, agents), concurrency is high enough for pool specialisation to pay off, and interconnect is fast enough that KV movement doesn't dominate.

PD is not worth it for: tiny requests, low concurrency, or weak KV transfer paths.

**KV transport backends.** Once you split prefill and decode, the transport layer is load-bearing:

| Backend | Use case | Notes |
|---|---|---|
| NIXL | vLLM PD, SGLang PD, TRT-LLM | NVIDIA's optimised KV transfer; NVLink within node, RDMA across nodes |
| UCX | TRT-LLM, MPI-backed setups | General-purpose HPC transport |
| Mooncake | SGLang | Mooncake cluster transfer protocol |
| Host-copy fallback | Low-bandwidth environments | Last resort; serialises KV through CPU |

**What matters in production:** setup cost per transfer (sessions and handshakes), reliability under load (retries kill p99), compatibility with chunked prefill (can you start decoding before prefill finishes?), and KV layout compatibility between the prefill and decode engines (must match page size, quantisation, head format).

### Expert Parallelism & Long Context
{: #ep-long-context}

**Expert parallelism (EP) for sparse MoE.** Dense model serving shards weight matrices across GPUs with all-reduce. MoE serving requires a different pattern: route each token to its assigned experts (all-to-all dispatch), compute on the assigned experts locally, then all-to-all reduce back. The serving questions:

1. **Where do experts live?** With 64 experts and 8 GPUs, assign 8 experts per GPU. But expert load is skewed — popular experts get more tokens, starving others.
2. **Expert Load Balancing (EPLB):** replicate hot experts across multiple GPUs. If Expert 0 receives 8K tokens vs. Expert 2 receiving 1K tokens, replicate Expert 0 so each GPU serves ~4K tokens. vLLM implements EPLB for MoE inference.
3. **EP + DP attention:** EP shards the FFN experts; attention heads still need a parallelism strategy. DP-attention runs each request's attention on one GPU (no cross-GPU communication for attention) while EP handles the MoE dispatch.

**Hierarchical KV for long context.** GPU HBM alone cannot hold KV for long-session workloads:

```
Hot pages    → HBM         (lowest latency, highest cost)
Warm pages   → host DRAM   (2-5× cheaper, ~5-10× slower fetch)
Cold history → remote/SSD  (cheapest, high rehydrate latency)
```

The challenge: rehydrate latency can destroy TTFT if a cold-cache request needs 1GB of KV fetched from SSD before decoding can begin. HiCache and LMCache solve this with prefetch policies (predict which KV will be needed before the request arrives based on prefix hashes), tiered eviction (keep recently-accessed and high-hit-rate prefixes warm), and pipelining (overlap KV rehydration with prefill computation).

> **Interview question:** Your long-context serving system uses hierarchical KV caching (HBM → host → remote). A user submits a 200K-token document that they've submitted before. Walk through what happens and where the performance bottleneck likely is.
>
> *Step 1: The router looks up the request prefix hash against the cache index. If the 200K-token document has been processed before, its KV blocks exist somewhere in the hierarchy. Step 2: The engine determines which KV blocks are hot (HBM — fast), warm (host DRAM — moderate), or cold (remote/SSD — slow). For a document processed yesterday, most blocks are likely cold. Step 3: Before prefill begins, the engine must rehydrate cold KV blocks — fetching from remote/SSD into HBM. For 200K tokens at 320 KiB/token = 64GB of KV. From SSD at ~3 GB/s: ~21 seconds. This is the bottleneck — TTFT is dominated by rehydration, not prefill computation. Fixes: (1) Prefetch: if the system can predict which documents will be requested (e.g. from a session ID or document hash seen in the API gateway), start rehydrating KV before the full request arrives. (2) Keep-warm policy: for high-value documents (frequently accessed), keep their KV blocks in host DRAM even when not actively serving requests. (3) Hierarchical prefetch: while rehydrating from SSD → host, simultaneously copy host blocks → HBM, so the pipeline stays busy. (4) KV quantisation: compress KV from FP16 to INT8 or FP8, halving the rehydration data volume. The real lesson: for long-context systems, KV economics (storage cost, rehydration latency, hit rate) are the product, not kernel throughput.*

### Observability & Operations
{: #observability-ops}

**Six metric categories for a production serving system:**

| Category | Example metrics | Why it matters |
|---|---|---|
| Admission / queueing | Waiting requests, rejected requests, queue age | Tells you the system is overloaded before the GPU graph even runs |
| Latency | TTFT, prefill time, decode time, TPOT, p50/p95/p99 histograms | Direct SLO health |
| KV memory | Cache usage %, reusable blocks, offload/reload rates | Whether long-context traffic is sustainable |
| Workload shape | Prompt length, output length, cache hit rate by prompt family | Separates product changes from engine regressions |
| Distributed health | All-to-all latency, transport errors, cross-engine KV transfer rate | Identifies TP/EP/PD bottlenecks |
| Business efficiency | Tokens/s/GPU, cost/request, cache-adjusted utilisation | Converts systems wins into deployment economics |

**Symptom → likely subsystem diagnostic table:**

| Symptom | Likely bottleneck | First questions |
|---|---|---|
| High TTFT, stable ITL | Prefill, tokenization, or cache miss | Are prompts longer? Did hit rate fall? Prefill pools saturated? |
| Good medians, awful p99 | Scheduler interference or transport stalls | Long prefills interrupting decode? TP/EP collectives spiking? |
| GPU OOM / low concurrency | KV fragmentation or bad eviction policy | Resident context exploding? Block size or reuse policy wrong? |
| Speculation gives no benefit | Poor acceptance or scheduler pressure | What is acceptance rate? Draft tokens eating page budget? |
| PD regresses performance | Transfer/routing overhead | KV transfer fast enough? Requests placed with cache affinity? |

**Three deployment recipes:**

1. **Small/medium model, straightforward traffic**: unified vLLM or SGLang pool with continuous batching, prefix caching, chunked prefill, and good metrics. Start here.
2. **Long-context interactive assistant**: chunked prefill, strong prefix caching, possibly a remote KV tier (LMCache/HiCache), careful p99 monitoring. Add complexity only if the workload demands it.
3. **Large reasoning/MoE service at scale**: EP + DP-attention, PD disaggregation, cache-aware router, separate prefill/decode pools, rigorous transport/KV accounting.

**Benchmark-and-tune runbook:**
1. Start from one engine with realistic traffic traces (not synthetic 512→512 benchmarks)
2. Measure TTFT, ITL, queue depth, active KV usage, tokens/s/GPU
3. Enable cheap wins first: continuous batching, prefix caching, chunked prefill, CUDA graphs/overlap
4. For MoE: evaluate EP backends and expert locality
5. If long prompts dominate: evaluate PD disaggregation and remote/host KV strategies
6. Re-benchmark with the exact feature combination you intend to ship — feature interactions matter

> **Interview question:** Your benchmark shows 3× higher throughput than production. What could explain the gap, and how do you close it?
>
> *The classic benchmark-vs-production gap almost always comes from workload mismatch: (1) Prompt/output length distribution: benchmarks often use fixed-length requests (e.g., 512 prompt, 512 output). Production has a heavy tail — 5% of requests are 10× longer and dominate scheduling. (2) Cache reuse: benchmarks run cold or with 100% cache hit rate. Production has partial reuse that's hard to model; your benchmark might not exercise the cache-miss code path at all. (3) Feature combinations: benchmarked without guided decoding, tool-call processing, or structured output constraints, which add host-side latency on every token in production. (4) Traffic burst patterns: benchmarks send requests at constant rate; production has diurnal patterns with 3× peak-to-trough variation, and queue buildup during bursts changes scheduler behaviour fundamentally. (5) Concurrency: the benchmark may run fewer concurrent requests than production, keeping the scheduler in a different regime (prefill-dominant vs. decode-dominant). Fixes: (1) Replay production traffic traces (with length distributions and inter-arrival times). (2) Warm the cache to a realistic hit rate before measuring. (3) Enable all production features during benchmarking. (4) Measure p50/p95/p99, not just throughput — production SLOs are usually on p95 TTFT and p95 ITL. (5) Run at realistic concurrency levels across the full traffic range (low-load, median-load, peak-load).*

---

## vLLM Internals
{: #vllm-internals}

vLLM is the most widely adopted open-source inference engine. Understanding its internals reveals the engineering choices that make production LLM serving possible, and the tradeoffs that shape every subsequent framework design.

**vLLM's four optimisation pillars:**
1. Minimising CPU overheads
2. Efficient GPU kernels
3. Model parallelism
4. Efficient memory management & caching

### Minimising CPU Overheads
{: #cpu-overheads}

**Why CPU overhead is uniquely damaging for LLM decode.** Training steps take 100ms–1s; 1ms of CPU overhead is < 1% overhead. Decode steps take 5–10ms (at batch size 1); 1ms of CPU overhead is 10–20% overhead. At 100–200 tok/s, you have 5–10ms per token — barely enough time for a Python function call.

**Four CPU overhead reduction techniques:**

**1. API Server: Python → Rust.** The HTTP server itself was a Python bottleneck. vLLM V1 rewrites the API server in Rust, eliminating GIL contention and Python-level JSON parsing overhead for every incoming request.

**2. Async scheduling.** The naive approach (synchronous scheduling) runs: schedule → prepare → execute → schedule → prepare → execute. Each phase is sequential, keeping the GPU idle during schedule+prepare. Async scheduling overlaps: while the GPU executes step N, the CPU is simultaneously scheduling and preparing step N+1. "Schedule and prepare the next batch one step ahead" — zero GPU synchronisation stalls.

**3. GPU-native input preparation (MRV2).** vLLM's batching logic involves complex bookkeeping: paged attention page tables, sequence masks, sampling parameters, token offsets. Previously implemented as many small PyTorch ops on CPU. MRV2 replaces this with custom Triton kernels that run on GPU — the input preparation happens in the same GPU stream as the forward pass, eliminating CPU→GPU data transfer and Python overhead.

**4. CUDA Graphs.** Python/PyTorch overhead (dispatching each `torch.nn.Linear`, `torch.nn.GELU`) can account for up to 50% of overall latency at small batch sizes. CUDA Graphs capture the entire sequence of GPU operations and replay them with a single CPU call — eliminating per-op dispatch overhead.

**Async scheduling + spec decoding.** GPU-native input preparation makes async scheduling compatible with speculative decoding. The rejection sampling kernel runs on GPU and produces the accepted token set; the input preparation kernel (also on GPU) can immediately read those results to prepare the next step's batch. With CPU-based input preparation, you'd need to synchronise to CPU after rejection sampling — a sync round-trip that breaks pipelining.

> **Interview question:** vLLM's async scheduler prepares step N+1 while step N is executing. What happens if step N produces a result that changes what step N+1 should look like — e.g., a request finishes early or a speculative token is rejected?
>
> *This is exactly the challenge that GPU-native input preparation solves. In synchronous scheduling, every step: (1) GPU executes, (2) CPU reads results, (3) CPU decides what's next, (4) CPU prepares inputs, (5) GPU executes next step. This is safe but slow. Async scheduling prepares step N+1 before step N completes — which works only if step N+1 can be prepared without knowing step N's outcome. For most decode steps this is fine: batch composition changes slowly. But for speculative decoding, rejection sampling decides which drafted tokens to keep — and this directly changes what tokens are "next" for step N+1. The solution: GPU-native input preparation. The rejection sampling kernel and the input preparation kernel both run on GPU in the same CUDA stream. The input prep kernel reads the rejection mask directly from GPU memory without any CPU round-trip. vLLM schedules both kernels in sequence with no synchronisation gap — the CPU isn't involved between them, so the async pipeline is unbroken. The result: async scheduling is compatible with speculative decoding without paying a CPU sync penalty.*

### CUDA Graphs & Piecewise Execution
{: #cuda-graphs}

**Full CUDA graph.** Capture the entire model's forward pass into one graph. Pros: minimal CPU overhead — one call replays everything. Cons: requires static shapes (batch size, sequence lengths must be predetermined), and no CPU operations can occur during model execution.

**The dynamism problem.** LLM inference has three sources of dynamism that break full CUDA graphs:
1. **Dynamic scheduling**: arbitrary mix of prefill and decode requests in the same batch
2. **Kernel heuristics**: some kernels (e.g. Cascade Attention) make runtime decisions based on shapes that can't be predetermined
3. **CPU offloading**: some operations require CPU involvement during model execution

**Piecewise CUDA Graphs (vLLM's solution).** Split the model into "token-wise operations" (stateless, static-shape) and "attention" (dynamic, stateful). Run token-wise operations under CUDA graphs (fast), run attention in PyTorch eager mode (flexible). Use `torch.compile` to automatically split at attention boundaries.

```
Token-wise ops (MLP, LayerNorm, RoPE) → CUDA graph N   → matmul/gelu kernels captured
Attention                             → PyTorch Eager  → dynamic paged attention
Token-wise ops (post-attention MLP)   → CUDA graph N+1 → matmul/gelu kernels captured
```

**Performance.** Compared to full PyTorch eager: 639% faster. Compared to full CUDA graph: only 27% slower at batch size 1, and effectively 0–2% slower at batch size ≥ 8. The flexibility of dynamic attention is nearly free once the batch is large enough to amortise the eager overhead.

> **Interview question:** Why is the attention operation specifically excluded from CUDA graph capture, while MLP and LayerNorm are captured?
>
> *CUDA graph capture requires static shapes and no CPU operations. Attention fails both requirements in a serving engine: (1) Shape dynamism: paged attention accesses KV blocks scattered across non-contiguous memory. The page table (which maps token positions to physical KV block addresses) changes every step as requests enter, exit, and advance. A captured graph would have the old page table baked in — but the actual page table is different each step. (2) Feature interactions: attention implementations make runtime heuristics: whether to use Flash Attention vs. standard attention depends on sequence length; cascade attention for shared prefixes needs runtime branching. These can't be predetermined at capture time. (3) CPU involvement for KV management: deciding which KV blocks to fetch (for requests with evicted KV that need reloading) requires CPU decisions that occur between steps. MLP and LayerNorm, by contrast, have fixed input shapes (batch × d_model), fixed weight matrices, and no per-step state — they're naturally static and graph-capturable. The piecewise design captures exactly those parts that are safe to capture, leaving flexible PyTorch eager execution only for the parts that need it.*

### Parallelism in vLLM
{: #parallelism-vllm}

vLLM supports five parallelism strategies with distinct communication, memory, and latency tradeoffs:

| Strategy | Mechanism | Pros | Cons |
|---|---|---|---|
| **Data parallelism (DP)** | Replicated engines; load-balancer routes requests | No inter-engine communication | No memory saving; no latency reduction per request |
| **Tensor parallelism (TP)** | Partition weight matrices; all-reduce between GPUs | Reduces per-request latency; KV cache partitioned | Heavy communication — needs NVLink; limited by KV head count |
| **Expert parallelism (EP)** | Distribute MoE experts across GPUs | Friendly for GPU kernels; smaller communication than TP | Only for MoE layers; attention needs separate parallelism; load imbalance |
| **Context parallelism (CP)** | Extend DP to subsequences of one long sequence | Parallelises the sequence dimension; balances KV across shards | No weight memory saving; only one GPU generates per request at a time |
| **Pipeline parallelism (PP)** | Distribute layers across GPUs; pipeline execution | Lowest communication overhead | Increased latency; load imbalance between stages |

**Mixed parallelism for MoE at scale.** No single strategy suffices for a 1T-parameter MoE model. vLLM uses:
- TP for attention layers (within a node, NVLink)
- EP for MoE FFN layers (all-to-all with DP-attention)
- DP for request-level scaling (replicas)
- PD disaggregation as an orthogonal dimension

**Expert Load Balancing (EPLB).** MoE routing is inherently unbalanced: popular topics route to popular experts. GPU 0 might process 8K tokens through Expert 0 while GPU 1 processes only 1K tokens through Expert 2 — GPU 1 sits idle 87% of the time. EPLB replicates hot experts:

```
Before EPLB: GPU 0 → Expert 0 (8K tokens), Expert 2 (1K tokens) → 9K tokens total, GPU 1 → Expert 1 (6K tokens), Expert 3 (1K tokens) → 7K tokens total
After EPLB:  GPU 0 → Expert 0 (4K), Expert 1 (3K), Expert 3 (1K) → 8K tokens total, GPU 1 → Expert 2 (1K), Expert 0 (4K), Expert 1 (3K), Expert 3 (1K) → 9K tokens total (balanced)
```

Replication doubles Expert 0's memory cost but nearly halves the load imbalance — a worthwhile trade when Expert 0 is a bottleneck.

> **Interview question:** You're serving a dense attention model (not MoE) across 8 GPUs. Would you prefer tensor parallelism or pipeline parallelism, and how does the workload characteristics affect your answer?
>
> *For a dense attention model at serving time, tensor parallelism (TP) is almost always preferable to pipeline parallelism (PP) when NVLink is available. TP reduces per-request latency: all 8 GPUs work on every token's forward pass in parallel, so the time per step is divided by 8 (minus communication overhead). PP keeps each layer on one GPU, so the forward pass is sequential across 8 GPUs — you get pipelining of different requests across stages, but each individual request's latency is dominated by the sum of stage latencies plus pipeline bubbles. TP is clearly better for interactive latency-sensitive workloads. PP only wins in two cases: (1) slow interconnect (100GbE RDMA, not NVLink) — PP's lower communication volume means PP's all-reduces are cheaper than TP's; (2) extremely large models where TP's memory reduction is insufficient and you need both to fit the model. Workload nuance: if you're serving at very high throughput with large batch sizes, PP can achieve better hardware utilisation (pipeline keeps all stages busy). But for interactive chat, small batch sizes, or agentic workloads with variable request rates, TP's per-request latency advantage dominates.*

### Hybrid Memory Allocator
{: #hybrid-memory}

**The problem: hybrid architectures have heterogeneous KV.** Modern LLMs mix multiple attention types to handle long context efficiently:
- **Full attention** (standard transformer layers) — unbounded context, expensive KV growth
- **Sliding window attention** (SWA) — fixed KV window per layer, bounded memory
- **Linear attention / Mamba / DeltaNet** — recurrent state rather than materialised KV

GPT-OSS, Qwen 3.5, Nemotron-H, and others use these mixtures. A naive static partitioning approach (pre-allocate fixed HBM regions for each layer type) causes severe fragmentation: full attention layers may be underutilised while linear-attention layers waste pre-allocated space.

**vLLM's solution: dynamic partitioning via a Hybrid Memory Allocator.** All layer types share a single unified memory pool. Each layer type gets its own allocator with appropriate granularity:
- **Full attention allocator**: allocates by sequence (unpredictable length, paged allocation)
- **Gated DeltaNet / linear attention allocator**: allocates fixed-size state blocks (1024 tokens, predictable)

The shared pool adjusts block sizes dynamically based on the layer type's access patterns. Result: 0–2% memory waste for all open-source models tested, vs. 30%+ fragmentation with static partitioning.

**Why block size matters.** PagedAttention uses fixed block sizes (e.g., 16 tokens per block). For full attention with variable-length sequences, blocks are allocated on demand — fine. For linear attention with fixed-size states, the allocator can use larger, aligned blocks that match the state size exactly, reducing metadata overhead. Mixing both types in one pool requires the allocator to handle multiple block granularities without fragmentation.

> **Interview question:** A Mamba-Transformer hybrid model alternates between full attention layers (materialised KV) and Mamba layers (recurrent SSM state). How does a memory allocator handle these two fundamentally different memory access patterns in one pool?
>
> *Full attention KV and Mamba SSM state have opposite memory profiles: KV grows linearly with sequence length (unbounded, variable) while Mamba state is fixed-size per sequence (constant regardless of length). A naive approach: pre-allocate fixed regions for each type. This fails because: a 100K-token request needs massive KV space but constant Mamba state; a short request needs little KV but still the same Mamba state. Static partitioning either wastes KV space for short requests or wastes Mamba space for long requests. The solution (vLLM's hybrid allocator): one shared pool, two allocator policies. The full-attention allocator uses paged allocation: KV blocks of size (block_size × n_heads × d_head × 2 bytes) are allocated on demand and returned when the sequence ends — same as standard PagedAttention. The Mamba allocator uses fixed-size allocation: state size = n_ssm_states × d_model per sequence, allocated once at sequence start, freed at sequence end. Both allocators carve from the same physical pool. The pool manager tracks which pages are used by which allocator and merges freed pages back into the common pool. This eliminates the fragmentation of static partitioning and allows the pool to dynamically skew toward whichever type is currently dominant in the workload. The key insight: the abstraction layer is "pages from a pool," not "regions for each type" — the pool doesn't know or care about layer type, only page size and residency.*
