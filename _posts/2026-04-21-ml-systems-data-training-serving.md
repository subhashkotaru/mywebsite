---
title: "ML Systems: Training, and Serving"
date: 2026-04-21
description: "Machine learning systems — from automatic differentiation to training at scale and production serving."
tags: [ml-systems, machine-learning, engineering]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview of ML Systems</a></li>
    <li><a href="#autodiff">Automatic Differentiation</a>
      <ul class="post-toc-sublist">
        <li><a href="#diff-methods">Differentiation Methods</a></li>
        <li><a href="#forward-ad">Forward Mode AD</a></li>
        <li><a href="#reverse-ad">Reverse Mode AD</a></li>
        <li><a href="#extending-graph">Extending the Computational Graph</a></li>
      </ul>
    </li>
    <li><a href="#hardware-accel">Hardware Kernel Acceleration</a>
      <ul class="post-toc-sublist">
        <li><a href="#accel-techniques">General Techniques</a></li>
        <li><a href="#matmul">Case Study: Matrix Multiplication</a></li>
        <li><a href="#memory-hierarchy">Memory Hierarchy & Tiling</a></li>
        <li><a href="#reuse">Memory Load Reuse</a></li>
        <li><a href="#swizzle">Swizzle Layouts</a></li>
        <li><a href="#tensor-cores">Tensor Cores</a></li>
        <li><a href="#tma">Tensor Memory Accelerator (TMA)</a></li>
      </ul>
    </li>
    <li><a href="#transformers">Transformers & Attention</a>
      <ul class="post-toc-sublist">
        <li><a href="#self-attention">Self-Attention</a></li>
        <li><a href="#multi-head">Multi-Head Attention</a></li>
        <li><a href="#flashattention">FlashAttention</a></li>
        <li><a href="#kv-cache">KV Cache & Autoregressive Decoding</a></li>
        <li><a href="#flash-decoding">Flash-Decoding</a></li>
      </ul>
    </li>
    <li><a href="#parallelisation">Parallelisation Part 1 — Data Parallelism & ZeRO</a>
      <ul class="post-toc-sublist">
        <li><a href="#data-parallel">Data Parallelism</a></li>
        <li><a href="#allreduce">AllReduce Algorithms</a></li>
        <li><a href="#memory-breakdown">Memory Breakdown</a></li>
        <li><a href="#zero">ZeRO: Zero Redundancy Optimizer</a></li>
      </ul>
    </li>
    <li><a href="#parallelisation-2">Parallelisation Part 2 — Model & Pipeline</a>
      <ul class="post-toc-sublist">
        <li><a href="#tensor-parallelism">Tensor Model Parallelism</a></li>
        <li><a href="#pipeline-parallelism">Pipeline Model Parallelism</a></li>
        <li><a href="#3d-parallelism">3D Parallelism</a></li>
      </ul>
    </li>
    <li><a href="#memory-optimisations">Memory Optimisations</a>
      <ul class="post-toc-sublist">
        <li><a href="#activation-checkpointing">Activation Checkpointing</a></li>
        <li><a href="#mixed-precision">Mixed Precision Training</a></li>
        <li><a href="#fsdp">Fully Sharded Data Parallelism (FSDP)</a></li>
      </ul>
    </li>
    <li><a href="#ml-compilation">ML Compilation</a>
      <ul class="post-toc-sublist">
        <li><a href="#relax">Multi-Level Abstraction: Relax</a></li>
        <li><a href="#symbolic-shapes">First-Class Symbolic Shapes</a></li>
        <li><a href="#op-fusion">Operator Fusion Across Levels</a></li>
        <li><a href="#structured-gen">Structured Output Generation</a></li>
        <li><a href="#mlc-engine">Universal Deployment: MLC Engine</a></li>
      </ul>
    </li>
    <li><a href="#moe">Mixture of Experts</a>
      <ul class="post-toc-sublist">
        <li><a href="#moe-layer">MoE Layer</a></li>
        <li><a href="#moe-compute">Efficient MoE Computation</a></li>
      </ul>
    </li>
    <li><a href="#llm-finetuning">LLM Fine-tuning</a>
      <ul class="post-toc-sublist">
        <li><a href="#prompt-tuning">Prompt & Prefix Tuning</a></li>
        <li><a href="#adapter-tuning">Adapter Tuning</a></li>
        <li><a href="#lora">LoRA</a></li>
        <li><a href="#qlora">QLoRA</a></li>
        <li><a href="#side-tuning">Side Tuning</a></li>
      </ul>
    </li>
    <li><a href="#llm-serving">LLM Serving</a>
      <ul class="post-toc-sublist">
        <li><a href="#continuous-batching">Continuous Batching</a></li>
        <li><a href="#pagedattention">PagedAttention</a></li>
        <li><a href="#radixattention">RadixAttention</a></li>
        <li><a href="#speculative-decoding">Speculative Decoding</a></li>
        <li><a href="#model-free-speculation">Model-Free Speculation</a></li>
      </ul>
    </li>
  </ul>
</nav>

## Overview of Machine Learning Systems
{: #overview}

<div class="post-flow" role="group" aria-label="ML system stack, flowing top to bottom">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar">Automatic Differentiation</span></li>
    <li class="post-flow__step"><span class="post-flow__bar">Graph-Level Optimization</span></li>
    <li class="post-flow__step"><span class="post-flow__bar">Parallelization</span></li>
    <li class="post-flow__step"><span class="post-flow__bar">Kernel Generation</span></li>
    <li class="post-flow__step"><span class="post-flow__bar">Memory Optimization</span></li>
  </ol>
</div>

A modern ML system is more than model weights and a training loop. It is a stack of interacting components, each solving a distinct efficiency or correctness problem. The five core concerns that cut across data, training, and serving are:

- **Automatic Differentiation** — computing gradients through arbitrary computation graphs without hand-derived math. Frameworks like PyTorch and JAX make this transparent, but understanding the chain rule at the graph level helps you debug NaN gradients and avoid memory blowups.
- **Graph-Level Optimization** — compilers (XLA, TorchInductor, TVM) rewrite your computation graph before execution: fusing operations, eliminating dead code, and choosing layouts. Getting this right can be the difference between 2× and 10× throughput.
- **Parallelization** — spreading work across devices and nodes. Data parallelism replicates the model; tensor/pipeline parallelism splits it. The tradeoff is communication overhead vs. memory per device, and the sweet spot changes with model size and hardware topology.
- **Kernel Generation** — translating high-level ops into hardware-specific code (CUDA, HIP, Metal). Tools like Triton let teams write custom kernels in Python; compilers auto-generate them. Poor kernels leave most of the GPU sitting idle.
- **Memory Optimization** — fitting large models and batches into limited HBM. Techniques include activation checkpointing (recompute instead of store), mixed precision (fp16/bf16), optimizer state sharding, and offloading to CPU. Memory is almost always the bottleneck before compute is.

---

## Automatic Differentiation
{: #autodiff}

Training any ML model requires computing gradients of a loss function with respect to millions of parameters. Doing this by hand is infeasible — automatic differentiation (AD) turns the computation graph itself into a gradient machine.

### Differentiation Methods
{: #diff-methods}

<div class="post-flow post-flow--horizontal" role="group" aria-label="Three approaches to differentiation">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Numerical</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Symbolic</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue post-flow__bar--accent">Automatic (AD)</span></li>
  </ol>
</div>

| Method | Idea | Limitation |
|---|---|---|
| **Numerical** | Finite differences: `(f(θ+ε) - f(θ-ε)) / 2ε` | Slow — one pass per parameter; numerical error |
| **Symbolic** | Apply sum/product/chain rules to expressions | Expression blowup; redundant sub-expression recomputation |
| **Automatic (AD)** | Decompose into primitives, propagate derivatives through the graph | Best of both — exact, efficient, composable |

Numerical differentiation is still valuable as a **gradient checker** in unit tests — verify that your AD implementation matches the finite-difference estimate.

### Forward Mode AD
{: #forward-ad}

Forward mode defines a *tangent* `v̇ᵢ = ∂vᵢ/∂x₁` for every intermediate value and propagates it left-to-right through the computation graph alongside the primal value.

<div class="post-flow" role="group" aria-label="Forward AD pass">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Primal forward pass — compute all vᵢ</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Propagate tangent v̇ᵢ in topological order</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Output tangent = ∂y/∂x₁</span></li>
  </ol>
</div>

**Example** — for `y = ln(x₁) + x₁x₂ − sin(x₂)` with `(x₁, x₂) = (2, 5)`:

```
v̇₃ = v̇₁/v₁ = 1/2 = 0.5        (ln node)
v̇₄ = v̇₁·v₂ + v̇₂·v₁ = 5       (multiply node, v̇₂=0)
v̇₆ = v̇₃ + v̇₄ = 5.5            (add node)
∂y/∂x₁ = v̇₇ = 5.5
```

**Limitation**: for `f: ℝⁿ → ℝ`, you need **n separate forward passes** — one per input. Useless for neural networks where `n` is in the millions.

### Reverse Mode AD
{: #reverse-ad}

Reverse mode defines an *adjoint* `v̄ᵢ = ∂y/∂vᵢ` and propagates it **right-to-left** — a single backward pass yields gradients with respect to **all** inputs at once.

<div class="post-flow" role="group" aria-label="Reverse AD pass">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Forward pass — compute and store all vᵢ</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Seed output adjoint: v̄_out = 1</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Traverse nodes in reverse topological order</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">At each node: compute partial adjoints → accumulate into inputs</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Read off ∂y/∂xᵢ = v̄ᵢ for all inputs</span></li>
  </ol>
</div>

**Key rule** for a node with multiple downstream consumers:

> `v̄ᵢ = Σⱼ ∈ next(i)  v̄ⱼ · (∂vⱼ/∂vᵢ)`

Partial adjoints from each path are computed separately and **summed**. In matrix form for a `matmul` layer `Z = XW`:

```
X̄ = Z̄ Wᵀ      # gradient w.r.t. input
W̄ = Xᵀ Z̄      # gradient w.r.t. weights
```

**Reverse AD pseudocode**:

```python
def gradient(out):
    node_to_grad = {out: [1]}
    for i in reverse_topo_order(out):
        v_bar_i = sum(node_to_grad[i])          # accumulate partial adjoints
        for k in inputs(i):
            v_k_to_i = v_bar_i * d(vᵢ)/d(vₖ)  # propagate to each input
            node_to_grad[k].append(v_k_to_i)
    return node_to_grad[input]
```

### Extending the Computational Graph
{: #extending-graph}

<div class="post-flow post-flow--compare" role="group" aria-label="Backprop vs extended graph AD">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Backprop (classic)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Forward graph only</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Backward ops reuse forward nodes in-place</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Caffe, CUDA-convnet era</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Extended Graph AD (modern)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">New adjoint nodes appended to the graph</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Gradient is itself a differentiable computation</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Enables grad-of-grad (second derivatives)</span></li>
    </ol>
  </div>
</div>

Because the backward pass is just more graph nodes, you can run AD **again** on the gradient to get second-order derivatives — the foundation for Hessian-based optimisers and meta-learning. PyTorch's `autograd` and JAX's `grad` both use this approach.

---

## Hardware Kernel Acceleration
{: #hardware-accel}

High-level ML ops like `matmul` or `conv2d` ultimately land on bare metal — CUDA cores, tensor cores, AVX registers. The gap between a naive implementation and a tuned kernel can be **10–100×**. This section covers the techniques that close that gap.

### General Techniques
{: #accel-techniques}

<div class="post-flow post-flow--horizontal" role="group" aria-label="General acceleration techniques">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Vectorization</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Data Layout & Strides</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Parallelization</span></li>
  </ol>
</div>

**Vectorization** — modern CPUs/GPUs process multiple values in a single instruction (SIMD). Loading `float4` instead of `float` gives a 4× throughput gain with the same number of instructions. Memory must be 128-bit aligned.

```c
float4 a = load_float4(A + i*4);
float4 b = load_float4(B + i*4);
store_float4(C + i*4, add_float4(a, b));
```

**Data layout & strides** — how a matrix is stored in memory matters as much as the compute itself.

| Layout | Formula | Notes |
|---|---|---|
| Row-major | `A[i,j] → data[i·cols + j]` | Default in C/NumPy |
| Column-major | `A[i,j] → data[j·rows + i]` | Default in Fortran/MATLAB |
| Strided | `A[i,j] → data[i·s₀ + j·s₁]` | Zero-copy slice/transpose/broadcast |

Strides enable **zero-copy transforms** (transpose = swap strides; broadcast = stride 0) but break contiguous memory access, making vectorization harder. Compact the array before feeding it to a kernel when you need full bandwidth.

**Parallelization** — `#pragma omp parallel for` or CUDA thread blocks spread independent iterations across cores. The key requirement: iterations must not share mutable state.

---

### Case Study: Matrix Multiplication
{: #matmul}

Matrix multiplication `C = A · Bᵀ` is the single most important kernel in ML — every linear layer, attention score, and gradient update bottlenecks here.

**Naive implementation — O(n³), memory-bound:**

```c
for (int i = 0; i < n; i++)
  for (int j = 0; j < n; j++) {
    C[i][j] = 0;
    for (int k = 0; k < n; k++)
      C[i][j] += A[i][k] * B[j][k];
  }
```

Every inner-loop iteration reads `A[i][k]` and `B[j][k]` fresh from DRAM. Total DRAM load: **2 · n³** elements at ~200 ns latency per cache miss.

---

### Memory Hierarchy & Tiling
{: #memory-hierarchy}

<div class="post-flow" role="group" aria-label="Memory hierarchy from slowest to fastest">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">DRAM — ~200 ns · huge capacity</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">L2 Cache — ~14 ns · medium capacity</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">L1 Cache — ~7 ns · small (32–64 KB)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Registers — ~0.5 ns · tiny (few KB)</span></li>
  </ol>
</div>

The trick is to **keep data in registers or L1 as long as possible** and amortise the expensive DRAM fetch over many reuse operations.

**Register tiling** — instead of computing one scalar `C[i][j]`, compute a `v1×v2` tile at a time, loading a `v1×v3` block of A and `v2×v3` block of B into registers:

```c
register float c[v1][v2] = 0;
for (int k = 0; k < n/v3; k++) {
  register float a[v1][v3] = A[i][k];  // loaded once, reused v2 times
  register float b[v2][v3] = B[j][k];  // loaded once, reused v1 times
  c += dot(a, b.T);
}
```

| | Naive | Register tiled |
|---|---|---|
| A DRAM loads | n³ | n³ / v2 |
| B DRAM loads | n³ | n³ / v1 |
| Register footprint | 3 floats | v1·v3 + v2·v3 + v1·v2 |

**Cache-line tiling** — before register tiling, load `b1` rows of A and `b2` rows of B into L1 cache, so the inner register-tiled loop sees L1 latency instead of DRAM latency:

```c
for (int i = 0; i < n/b1; i++) {
  l1cache float a[b1][n] = A[i];       // one DRAM fetch for b1 rows
  for (int j = 0; j < n/b2; j++) {
    l1cache float b[b2][n] = B[j];
    C[i][j] = dot(a, b.T);             // fully in L1, apply register tiling here
  }
}
```

Constraint: `b1·n + b2·n < L1 cache size`. Combined load cost:

```
dramspeed × (n² + n³/b1)   +   l1speed × (n³/v2 + n³/v1)
```

---

### Memory Load Reuse
{: #reuse}

The central insight of all these optimisations is **reuse ratio** — how many times a loaded value is used before it is evicted.

<div class="post-flow post-flow--compare" role="group" aria-label="Reuse in matmul vs convolution">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Matrix Multiplication</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">A[i,k] is independent of j</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Tile j by v → A reused v times</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">DRAM load: n³/v instead of n³</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Convolution</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Input[b,k,y+ry,x+rx] shared across output channels</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Tile output channel → reuse input patch</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Same principle, higher-dimensional loop nest</span></li>
    </ol>
  </div>
</div>

Identifying **which index a value is independent of**, then tiling along that index to manufacture reuse, is the universal template for writing fast kernels — whether on CPU with AVX, GPU with CUDA shared memory, or via compiler frameworks like TVM and Triton.

---

### Swizzle Layouts
{: #swizzle}

Tiling gets data into shared memory — but shared memory has its own access hazard: **bank conflicts**. GPU shared memory is divided into 32 banks, each 4 bytes wide. When multiple threads in a warp access addresses that fall in the same bank, those accesses are serialised, destroying the bandwidth advantage of shared memory.

The naive fix is padding: add one extra column to shift each row's starting address so no two rows alias the same bank. But padding wastes memory and breaks tile alignment.

**Swizzling** solves this without waste: instead of storing row `i` at a linear offset `i·cols`, remap the physical address using a XOR of the row and column indices:

```
physical_col = logical_col XOR (logical_row >> k)
```

This spreads each row's elements across different banks, so threads reading an entire column access one element per bank — conflict-free. Critically, the hardware applies the remapping implicitly; the programmer just configures a swizzle mode (e.g. `SWIZZLE_128B`) and passes pre-swizzled addresses. No manual index arithmetic needed at the call site.

**SWIZZLE_128B** is the standard choice for FP16 GEMM: it guarantees conflict-free access to an 8×8 tile (8 rows, 8 columns, 128 bytes per row) — exactly the granularity tensor cores consume. Using a mismatched swizzle mode or none at all can silently halve shared-memory bandwidth.

<div class="post-flow post-flow--compare" role="group" aria-label="Padding vs swizzling for bank conflicts">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Padding</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Add one extra column per row</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Wastes memory and breaks tile alignment</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Simple but inefficient at scale</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Swizzle ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">XOR row and column indices to remap addresses</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">No wasted memory, tile alignment preserved</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Hardware applies implicitly — configure once</span></li>
    </ol>
  </div>
</div>

---

### Tensor Cores
{: #tensor-cores}

Every modern NVIDIA GPU (Volta onwards) ships with **tensor cores** — specialised execution units that compute a small matrix multiply-accumulate (MMA) in a single instruction rather than looping over scalar FMAs.

A single tensor core instruction computes `D = A · B + C` where A, B, C, D are small fixed-size matrices (e.g. 16×8×16 for FP16 on Ampere). The hardware fuses the multiply and add into one pipeline stage with no intermediate rounding, which is both faster and more numerically stable than the equivalent sequence of scalar ops.

<div class="post-flow" role="group" aria-label="Tensor core operation">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Load A tile (16×16, FP16) and B tile (16×8, FP16) from shared memory</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Issue warp-level MMA instruction: D = A·B + C</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Result accumulates in FP32 registers — no round-trip to shared memory</span></li>
  </ol>
</div>

The catch is that tensor cores are extremely demanding about memory layout. The A and B tiles must arrive in shared memory in exactly the arrangement the MMA instruction expects — which is why the swizzle mode must match the MMA tile size. A misaligned layout forces the warp to spend extra instructions shuffling data before the MMA can fire, negating most of the speedup.

In practice, getting tensor core utilisation above 80% requires coordinating three things together: the right swizzle for conflict-free shared-memory access, the right tile sizes to keep the pipeline full, and enough in-flight warps to hide the MMA latency.

---

### Tensor Memory Accelerator (TMA)
{: #tma}

Even with tiling and tensor cores, the bottleneck often shifts to **data movement** — copying tiles from global memory (HBM) into shared memory fast enough to keep the compute pipeline fed. On Hopper GPUs, NVIDIA introduced the **Tensor Memory Accelerator (TMA)** to offload this work entirely from the warp.

With a normal `cp.async` copy, each warp issues its own memory requests and stalls on them. TMA replaces this with a single descriptor-driven transfer: the programmer pre-describes the tile shape, stride, and swizzle mode in a descriptor object, then issues one instruction that launches an asynchronous DMA engine to move the whole tile.

<div class="post-flow" role="group" aria-label="TMA vs manual async copy">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Describe tile geometry once: shape, stride, swizzle mode → TMA descriptor</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Issue one TMA load instruction per tile — warp does not block</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Hardware DMA engine transfers tile from global → shared memory</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Warp waits on a barrier, then computes — copy and compute fully overlapped</span></li>
  </ol>
</div>

TMA also handles multi-dimensional tiling natively: a 3D TMA descriptor can iterate over a batch × row × column tile in one instruction, which would otherwise require a nested loop of `cp.async` calls. And because the swizzle mode is baked into the descriptor, TMA guarantees the tile lands in shared memory already in the layout tensor cores expect — no post-copy shuffle needed.

The result is that on Hopper, a well-written GEMM kernel can overlap global→shared copies for the next tile with tensor core computation on the current tile at near-full bandwidth, pushing arithmetic intensity close to the hardware roofline.

---

## Transformers & Attention
{: #transformers}

Attention is the mechanism that lets a model weight how much each position in the input should influence each output position. Transformers make it the **primary** computation, replacing recurrence entirely — which is why they parallelise so well on GPUs.

### Self-Attention
{: #self-attention}

Given a sequence of hidden states, self-attention computes a **weighted combination** of all positions for each query position:

> `A(Q, K, V) = softmax(QKᵀ / √d) · V`

- **Q** (query), **K** (key), **V** (value) are all linear projections of the same input — hence *self*-attention
- The `L×L` score matrix `S = QKᵀ` captures pairwise relevance between every token pair
- Dividing by `√d` prevents dot products from growing so large that softmax saturates

<div class="post-flow" role="group" aria-label="Self-attention forward pass">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Input X → project to Q, K, V  (3 matmuls)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">S = QKᵀ / √d  — L×L score matrix</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">A = softmax(S)  — normalise per row</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">O = A · V  — weighted sum of values</span></li>
  </ol>
</div>

**Compute challenge**: the intermediate `L×L` attention matrix is `O(N²)` in sequence length. At `N=32768` with fp16, that's 2 GB just for one head of one layer — and it must be written to and read back from HBM.

### Multi-Head Attention
{: #multi-head}

Instead of a single attention function, run **H parallel heads** with independent projections then concatenate:

> `Z = MultiHead(Q, K, V) = Concat(Z₀, …, Z_{H-1}) · Wₒ`
> where `Zᵢ = A(QWᵢQ, KWᵢK, VWᵢV)`

<div class="post-flow post-flow--horizontal" role="group" aria-label="Multi-head attention benefits">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">More parallelism — heads independent</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Smaller d per head — cheaper per matmul</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue post-flow__bar--accent">Richer representations — different subspaces</span></li>
  </ol>
</div>

Each head attends in a different **subspace** of the embedding, so the model can simultaneously track syntax, semantics, and coreference in parallel.

### FlashAttention
{: #flashattention}

**Problem** — naive attention materialises the full `N×N` matrix in HBM at each layer:

| | Standard Attention | FlashAttention |
|---|---|---|
| Global memory access | 40.3 GB | 4.4 GB |
| Runtime | 41.7 ms | 7.3 ms |
| Memory scaling | O(N²) | O(N) |

**Key idea**: compute attention in **tiles** so the `N×N` score matrix never fully lands in HBM — it lives transiently in shared memory.

<div class="post-flow" role="group" aria-label="FlashAttention algorithm">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Load Q/K/V block-by-block from HBM → shared memory</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Compute partial softmax on-chip using online softmax scaling</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Accumulate output O in registers; scale by running normaliser</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Write only the final O back to HBM — no intermediate A stored</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Backward: recompute A from stored softmax norms (size N, not N²)</span></li>
  </ol>
</div>

**Online softmax** makes tiling safe: when combining two blocks of scores `x⁽¹⁾` and `x⁽²⁾`, the merged softmax can be computed from the per-block max and sum without ever seeing all scores at once.

**Parallelism strategy:**
- **Thread-block level** — assign different attention **heads** to different blocks (16–64 heads → 16–64 blocks). Then assign different **query** segments to additional blocks. Keys/values cannot be split across blocks because softmax requires seeing all keys for a given query.
- **Warp level** — split across queries (not K/V), so warps never need to communicate to reduce partial softmax results.

Result: **2–4× faster** wall time and **10–20× less memory** vs standard attention, with identical numerical output.

### KV Cache & Autoregressive Decoding
{: #kv-cache}

LLM inference generates tokens one at a time — each new token attends to **all previous tokens**:

<div class="post-flow post-flow--compare" role="group" aria-label="Pre-filling vs decoding phases">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Pre-filling (iteration 0)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">All input tokens processed in parallel</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Full attention matrix — compute-bound</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">FlashAttention applies here</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Decoding (iterations 1…T)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Single new token query each step</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Attend to all past K/V — memory-bound</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">K/V cache avoids recomputing past keys/values</span></li>
    </ol>
  </div>
</div>

**KV cache**: store K and V tensors from all past tokens so each decoding step only computes the new token's Q and reads cached K/V. Tradeoff: memory grows linearly with sequence length and batch size — a 70B model with a 32k context and batch size 32 can consume hundreds of GB just for the KV cache.

**FlashAttention in decoding** doesn't apply directly: with a single query there is no parallelism across queries, and the kernel must scan all keys/values **sequentially** — inefficient when the context is long.

### Flash-Decoding
{: #flash-decoding}

**Problem**: standard FlashAttention scans K/V sequentially for a single query token. On long contexts (e.g. 32k tokens) the kernel is bottlenecked by memory bandwidth, not compute.

<div class="post-flow" role="group" aria-label="Flash-Decoding algorithm">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Split K/V into small chunks across thread blocks</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each block computes partial attention output with FlashAttention</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Reduce partial outputs across all splits (attention is associative)</span></li>
  </ol>
</div>

The key insight is that the attention output `O = softmax(QKᵀ)V` is **associative over K/V splits** — partial outputs with their softmax normalisation factors can be merged exactly, just like two blocks in online softmax. This lets the decoding kernel parallelise across the K/V dimension instead of processing it sequentially.

Result: **up to 8× faster** than prior work on long-context decoding, with no change in numerical output.

---

## Parallelisation
{: #parallelisation}

A single GPU tops out at ~80 GB of HBM today. GPT-3 needs ~2800 GB just for weights. The only path forward is spreading training across many devices — and doing so without proportionally exploding communication cost.

### Data Parallelism
{: #data-parallel}

The simplest approach: **replicate the full model on every GPU**, partition the training data across them, compute independent gradients, then aggregate.

<div class="post-flow" role="group" aria-label="Data parallel training loop">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Partition dataset into N mini-batches, one per GPU</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each GPU runs forward + backward independently → local gradients</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Aggregate gradients across all GPUs (AllReduce)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Each GPU applies identical weight update → all stay in sync</span></li>
  </ol>
</div>

The weight update rule distributes perfectly because the full-batch gradient is the **average** of per-sample gradients — so splitting samples across GPUs and averaging the resulting gradients is mathematically equivalent to the single-GPU case.

**Parameter Server** (early approach): workers push gradients to a central server; server accumulates and broadcasts updated weights. Problem: the server is a **bandwidth bottleneck** — all N workers funnel traffic through one node. Doesn't scale past ~tens of workers.

### AllReduce Algorithms
{: #allreduce}

AllReduce replaces the centralised server with **peer-to-peer collective communication** across all workers.

<div class="post-flow post-flow--compare" role="group" aria-label="AllReduce topologies">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Naïve AllReduce</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Each worker sends gradients to all others</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Total comm: N(N−1)·M</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Same bottleneck as param server</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Ring AllReduce ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Workers in a ring; M params split into N slices</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Each worker sends M/N per step × 2N steps</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Total comm: 2·M (independent of N!)</span></li>
    </ol>
  </div>
</div>

**Ring AllReduce** in two phases:

1. **Aggregation** — each worker sends its slice (M/N params) to its right neighbour; after N steps every worker holds the fully-aggregated version of *its own* slice.
2. **Broadcast** — each worker sends its aggregated slice to the right; after N more steps every worker has all slices.

Total: `2M` parameters communicated per worker, **regardless of N**. This is why NCCL (NVIDIA Collective Communications Library) uses Ring AllReduce as its default.

| Method | Total communication | Bandwidth per worker | Scalability |
|---|---|---|---|
| Parameter Server | 2·N·M | M·N / bw | Poor |
| Naïve AllReduce | N²·M | N·M / bw | Poor |
| **Ring AllReduce** | **2·N·M** | **2M / bw** | **Excellent** |
| Tree AllReduce | 2·N·M | 2·log(N)·M / bw | Good |
| Butterfly AllReduce | N·M·log(N) | M·log(N) / bw | Moderate |

Ring and Tree both achieve `2·N·M` total bytes but Ring spreads the load evenly across all workers at `2M` per worker — Tree starves leaf nodes while the root becomes a bottleneck.

### Memory Breakdown
{: #memory-breakdown}

Data parallelism requires **each GPU to hold a full model copy**. For Adam-based training with mixed precision, per-parameter memory is:

<div class="post-flow" role="group" aria-label="Mixed-precision memory per parameter">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">FP16 parameters — 2 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">FP16 gradients — 2 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">FP32 master weights — 4 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">FP32 Adam momentum — 4 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">FP32 Adam variance — 4 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Total: ~20 bytes/param  →  1B params = 20 GB/GPU</span></li>
  </ol>
</div>

That doesn't include activations or the input batch. By the time you add those, even an 80 GB A100 can't fit GPT-3 (175B params = 2800 GB) on a **single GPU**, let alone a rack.

| Model | Params | Memory footprint |
|---|---|---|
| BERT-Large | 0.32B | 5 GB |
| GPT-2 | 1.5B | 24 GB |
| Turing NLG 17.2B | 17.2B | 275 GB |
| GPT-3 | 175B | 2800 GB |

### ZeRO: Zero Redundancy Optimizer
{: #zero}

ZeRO (from Microsoft DeepSpeed) keeps the data-parallel communication structure but **eliminates the state redundancy** — instead of every GPU holding identical copies of all optimiser state, that state is partitioned across GPUs.

<div class="post-flow post-flow--compare" role="group" aria-label="ZeRO stages">
  <div class="post-flow__col">
    <p class="post-flow__col-label">What gets partitioned</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Stage 1 — Optimizer states (FP32 weights, momentum, variance)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Stage 2 — + Gradients (FP16)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Stage 3 — + Parameters (FP16)</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Memory reduction (N GPUs)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Stage 1 — 4× savings (optimizer state ÷ N)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Stage 2 — 8× savings (+ gradients ÷ N)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Stage 3 — N× savings (all state ÷ N)</span></li>
    </ol>
  </div>
</div>

**Stage 1 training loop (most commonly deployed):**

<div class="post-flow" role="group" aria-label="ZeRO Stage 1 training iteration">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Forward pass — all GPUs run full model on their data shard</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Backward pass — each GPU computes FP16 gradients for all params</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">AllReduce — average FP16 gradients across all GPUs</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each GPU updates only its shard of FP32 optimizer state (Adam)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">AllGather — broadcast updated FP16 weights so all GPUs are in sync</span></li>
  </ol>
</div>

**Stage 2** adds **ReduceScatter** instead of AllReduce for gradients: after backprop of each layer, the gradient is immediately reduced and only the owning GPU keeps it — so gradient memory footprint shrinks by N× before the next layer's backward even starts.

**Stage 3** partitions parameters too. During the forward pass, each GPU **broadcasts its own parameter shard** to all others, then discards received parameters right after use. During backward, parameters are re-broadcast as needed. This achieves full N× memory reduction but adds communication on both forward and backward — the tradeoff is network bandwidth vs. HBM capacity.

> **Turing NLG 17.2B** (2021) was trained with ZeRO Stage 1 + Megatron tensor parallelism. Stage 1 alone dropped per-GPU optimizer memory from 275 GB down to ~34 GB on 8 GPUs.

---

## Parallelisation Part 2 — Model & Pipeline
{: #parallelisation-2}

Data parallelism hits a wall when a single model replica no longer fits in GPU memory. The solution is to **split the model itself** — either within a layer (tensor parallelism) or across layers (pipeline parallelism).

### Tensor Model Parallelism
{: #tensor-parallelism}

**Tensor parallelism** partitions the weight matrix of a single layer across devices, so each GPU only stores and computes a *slice* of that layer. No layer needs to fit on one GPU.

For a linear layer `Y = X × W`, there are two natural splits:

<div class="post-flow post-flow--compare" role="group" aria-label="Two ways to partition a linear layer">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Partition output columns</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">GPU 1 holds W₁ → computes Y₁ = X × W₁</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">GPU 2 holds W₂ → computes Y₂ = X × W₂</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">No sync during forward; AllGather outputs after</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Comm cost: O(B × C_in) on backward</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Partition input rows (reduce output)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">GPU 1 holds W₁, receives X₁ → partial Y₁</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">GPU 2 holds W₂, receives X₂ → partial Y₂</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">AllReduce: Y = Y₁ + Y₂ after forward</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Comm cost: O(B × C_out) on forward</span></li>
    </ol>
  </div>
</div>

**Communication cost comparison** — the right strategy depends on whether parameters or activations dominate:

| Strategy | Forward comm | Backward comm | Grad sync |
|---|---|---|---|
| Data parallelism | 0 | 0 | O(C_out × C_in) |
| Tensor MP — partition output | 0 | O(B × C_in) | 0 |
| Tensor MP — reduce output | O(B × C_out) | 0 | 0 |

**Megatron-LM** applies tensor parallelism to Transformers by combining both strategies within a single layer:

- **FC layers**: partition output (`Y = GeLU(X × A)`) then reduce output (`Z = Dropout(Y × B)`) — only one AllReduce per transformer sub-layer
- **Multi-head attention**: each GPU owns a subset of attention heads (independent by construction), then reduces after the output projection `W_o`

This approach scaled GPT-style models to **512 GPUs** by combining tensor and data parallelism.

**CNNs** follow a different split: conv layers (90–95% compute, 5% params, large activations) use **data parallelism**, while fully-connected layers (5–10% compute, 95% params, small activations) use **tensor model parallelism** — matching the parallelism strategy to the bottleneck.

### Pipeline Model Parallelism
{: #pipeline-parallelism}

**Pipeline parallelism** assigns consecutive *layers* (a "stage") to each device. With a naïve schedule only one device is active at a time — GPU utilisation collapses.

The fix is **micro-batching**: split the mini-batch into `m` micro-batches and pipeline them through the `p` stages.

<div class="post-flow" role="group" aria-label="Pipeline parallelism bubble analysis">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Split mini-batch into m micro-batches</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Stage i starts forward on micro-batch k+1 while stage i+1 runs micro-batch k</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Pipeline fill/drain creates idle "bubble" at start and end</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Bubble fraction = (p−1) / m  →  shrinks as m grows</span></li>
  </ol>
</div>

> `Bubble fraction = (p − 1) / m`  
> With `p = 8` stages and `m = 32` micro-batches, only **22%** of cycles are wasted.

**GPipe schedule** (flush-based): run all `m` forward passes, then all `m` backward passes. Simple, but keeps activations for all in-flight micro-batches in memory simultaneously → memory scales with `m × p`.

**1F1B schedule** (one-forward-one-backward): in steady state, interleave one forward and one backward pass. Limits in-flight micro-batches to `p` instead of `m` → **4× less memory** at the same bubble fraction.

**Interleaved 1F1B**: divide each stage further into `v` sub-stages. Each device handles `v` non-contiguous chunks of layers.

<div class="post-flow post-flow--compare" role="group" aria-label="Pipeline schedule comparison">
  <div class="post-flow__col">
    <p class="post-flow__col-label">1F1B</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Bubble = (p−1)/m</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">In-flight micro-batches = p</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Communication per step = 1 send/recv</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Interleaved 1F1B ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Bubble = (p−1)/(v·m)  →  v× smaller</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">In-flight micro-batches = p (unchanged)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Communication per step = v sends/recvs</span></li>
    </ol>
  </div>
</div>

Interleaved 1F1B reduces the bubble by `v×` at the cost of `v×` more pipeline communication — a worthwhile tradeoff for large `p`.

### 3D Parallelism
{: #3d-parallelism}

No single strategy dominates. Real production systems combine all three:

<div class="post-flow post-flow--horizontal" role="group" aria-label="3D parallelism dimensions">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Data parallel — replicate across groups of GPUs</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Tensor parallel — split layers within each GPU group</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Pipeline parallel — split stages across GPU groups</span></li>
  </ol>
</div>

| Dimension | Pros | Cons |
|---|---|---|
| Data parallel | Massively scalable, no forward/backward comm | Full model replica must fit on one GPU |
| Tensor model parallel | Supports huge individual layers | Limited scalability; each layer split across only a few GPUs |
| Pipeline model parallel | Supports very deep models, large batch training | Pipeline bubble; must transfer activations between stages |

> **DeepSpeed 3D parallelism** (used by Megatron-DeepSpeed) layers all three: tensor parallelism within a node (fast NVLink), pipeline parallelism across nodes (slower inter-node links), and data parallelism + ZeRO across pipeline replicas. This strategy trained **Megatron-Turing NLG 530B** on 4480 A100 GPUs.

---

## Memory Optimisations
{: #memory-optimisations}

Three distinct memory consumers eat GPU HBM during training: **model weights**, **optimizer states**, and **intermediate activations**. The previous section (ZeRO / FSDP) attacked weights and optimizer state. This section attacks activations and numerical precision.

<div class="post-flow post-flow--horizontal" role="group" aria-label="Three sources of GPU memory pressure">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Model weights</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Optimizer states (Adam: 3× weight size)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Activations (O(N) for N-layer net)</span></li>
  </ol>
</div>

At inference, you only need two buffers to pass data forward — no need to keep any intermediate activations. Training is different: every intermediate value from the forward pass must be kept alive until its corresponding backward pass consumes it.

### Activation Checkpointing
{: #activation-checkpointing}

**The problem**: a naïve N-layer network stores all N intermediate activations simultaneously → O(N) memory.

**The key insight**: you can trade compute for memory by **recomputing** activations on demand during backprop instead of storing them.

<div class="post-flow" role="group" aria-label="Checkpointing strategy">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Forward pass — save only "checkpoint" nodes every K layers; discard the rest</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Backward pass — when a missing activation is needed, recompute it from the nearest checkpoint</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Memory cost = O(N/K) checkpoints + O(K) recompute buffer</span></li>
  </ol>
</div>

**Optimal checkpoint spacing**: minimise `N/K + K` by setting `K = √N`:

> `Memory cost = O(N/√N) + O(√N) = O(√N)`

A 1000-layer network that would need 1000 activation buffers now needs only ~32 — a **31× reduction** at the cost of one extra forward pass per segment during backward.

| Strategy | Memory | Extra compute |
|---|---|---|
| No checkpointing | O(N) | 0 |
| Checkpoint every K layers | O(N/K + K) | ~1× forward per segment |
| Optimal K = √N | **O(√N)** | ~1× total forward re-run |

In PyTorch this is `torch.utils.checkpoint.checkpoint()`. In practice it is applied at the transformer-block granularity — checkpoint at each block boundary, recompute within-block activations during backward.

### Mixed Precision Training
{: #mixed-precision}

Modern GPUs have dedicated FP16/BF16 tensor cores that are **2–8× faster** than FP32. Mixed precision exploits this while preserving accuracy.

<div class="post-flow post-flow--compare" role="group" aria-label="FP16 vs BF16 tradeoffs">
  <div class="post-flow__col">
    <p class="post-flow__col-label">float16 (FP16)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">5-bit exponent, 10-bit fraction</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">More fraction bits → higher precision</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Smaller range → prone to overflow/underflow</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Needs loss scaling to avoid vanishing gradients</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">bfloat16 (BF16)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">8-bit exponent, 7-bit fraction</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Same dynamic range as FP32</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">No overflow issues — preferred for training</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Less precise fraction → may lose fine detail</span></li>
    </ol>
  </div>
</div>

**Mixed precision** means different layers use different precisions depending on sensitivity:

- **Matmuls and convolutions** — run in FP16/BF16 on tensor cores (fast path)
- **Softmax, layer norm, loss** — accumulate in FP32 (these involve summing many values; FP16 causes catastrophic cancellation)
- **Master weights and optimizer state** — always kept in FP32 for stable gradient updates

```
linear (FP16 weights)
    ↓  FP16 activations
softmax
    ↓  FP32 accumulation (overflow-safe)
loss
    ↓  FP32 gradients
optimizer update  ←  FP32 master weights
    ↓  cast back to FP16
next iteration
```

**Loss scaling** (FP16 only): gradients in FP16 can underflow to zero for small values. Multiply the loss by a large scalar before backward, then divide the resulting gradients back before the weight update. BF16 avoids this entirely due to its wider exponent range — which is why BF16 has become the default for LLM training on A100/H100.

### Fully Sharded Data Parallelism (FSDP)
{: #fsdp}

FSDP is PyTorch's native implementation of **ZeRO Stage 3** — it shards parameters, gradients, *and* optimizer states across all data-parallel ranks.

The communication primitives that make it work:

<div class="post-flow post-flow--compare" role="group" aria-label="AllReduce decomposed into primitives">
  <div class="post-flow__col">
    <p class="post-flow__col-label">ReduceScatter</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each GPU sends its gradients around the ring</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each GPU accumulates one shard</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Result: GPU i holds fully-reduced shard i</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">AllGather</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Each GPU broadcasts its shard</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">All GPUs collect every shard</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Result: all GPUs have full tensor</span></li>
    </ol>
  </div>
</div>

> **AllReduce = ReduceScatter + AllGather** — the ring-based AllReduce you saw in Part 1 is just these two operations back-to-back.

**FSDP forward pass per layer:**

<div class="post-flow" role="group" aria-label="FSDP per-layer execution">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">AllGather — reconstruct full layer weights from shards on all GPUs</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Compute forward pass locally with full weights</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Discard the gathered weights immediately — only keep activation + shard</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Backward: AllGather weights again for gradient computation</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">ReduceScatter — each GPU accumulates only its own gradient shard</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Each GPU updates its own weight shard with local optimizer state</span></li>
  </ol>
</div>

Peak memory per GPU drops from `20M` bytes (full Adam) to roughly `20M/N + activation overhead` — a near-linear improvement with the number of GPUs.

**FSDP vs ZeRO comparison:**

| | ZeRO Stage 1 | ZeRO Stage 2 | ZeRO Stage 3 / FSDP |
|---|---|---|---|
| Optimizer states sharded | ✓ | ✓ | ✓ |
| Gradients sharded | — | ✓ | ✓ |
| Parameters sharded | — | — | ✓ |
| Extra comm vs baseline | +AllGather weights | +ReduceScatter grads | +AllGather ×2 per layer |
| Memory reduction (N GPUs) | ~4× | ~8× | ~N× |

FSDP is available in `torch.distributed.fsdp` and is the standard choice for fine-tuning or training models that exceed single-GPU memory. Combined with activation checkpointing and BF16 mixed precision, it enables training 70B+ parameter models on commodity GPU clusters.

---

## ML Compilation
{: #ml-compilation}

Writing a model in PyTorch gets you a working implementation. Getting that model to run efficiently across cloud GPUs, mobile SoCs, web browsers, and embedded devices is a different problem — one that ML compilers exist to solve. Rather than hand-tuning for each target, an ML compiler takes a model description and applies a pipeline of transformations to produce optimised code for any backend.

The challenge is that modern ML models are not static: sequence lengths vary, KV caches grow, quantised weights need on-the-fly dequantisation, and operator fusion opportunities span multiple abstraction levels (graph, tensor program, library call). A compiler that can only handle fixed shapes or single-level IRs misses most of the optimisation potential.

### Multi-Level Abstraction: Relax
{: #relax}

Traditional ML compilers operate at one level — either a high-level computation graph (like XLA's HLO) or a low-level tensor program (like TVM's TIR). Optimisations at the graph level (op fusion, layout rewriting) cannot see inside individual kernels; optimisations at the kernel level cannot see the surrounding graph. **Relax** addresses this with a **cross-level IR** that simultaneously encapsulates all three:

<div class="post-flow post-flow--horizontal" role="group" aria-label="Three levels in Relax">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Computational graph — high-level op sequences, data flow</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Tensor programs (TIR) — loop nests, buffer accesses, schedules</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Library calls — cuBLAS, CUTLASS, vendor-optimised kernels</span></li>
  </ol>
</div>

A Relax `IRModule` is a collection of typed functions. Graph-level functions call into tensor programs via `call_tir` and into external libraries via `call_dps_library` — all within the same module, with shapes tracked symbolically across all call boundaries:

```python
def main(x: Tensor(("n", 128), "f16"), w: Tensor((128, 256), "f16")):
    n = sym_var()
    lv0 = call_tir(mm, [x, w], Tensor((n, 256), "f16"))   # custom kernel
    lv1 = relu(lv0)
    lv2 = call_dps_library("cutlass.rms_norm", [lv1], Tensor((n, 256), "f16"))
    ...
```

Because the entire program is one module, a single compiler pass can reason about data flow from the graph down into loop bodies and back — enabling cross-level optimisations that neither graph compilers nor kernel compilers alone can perform.

### First-Class Symbolic Shapes
{: #symbolic-shapes}

Dynamic shapes are the norm in LLM inference: batch size varies, sequence length grows token by token, and KV cache dimensions change per request. Most compilers represent unknown dimensions as opaque `?` placeholders, which breaks any analysis that depends on shape relationships.

Relax uses **symbolic shape variables** — named integer expressions — instead. A function annotated with `Tensor(("n", 2, 2), "f32")` propagates the symbol `n` through every downstream operation:

```python
# Typical approach: loses shape info
def any_shape_fn(x: Tensor((?, 2, 2), "f32")):
    lv0: Tensor((?, 4), "f32") = reshape(x, ...)   # ? is opaque

# Relax: symbolic shapes preserved
def symbolic_shape_fn(x: Tensor(("n", 2, 2), "f32")):
    n, m = sym_var(), sym_var()
    lv0: Tensor((n, 4), "f32") = reshape(x, shape(n, 4))   # n * 4 known
    lv1: Tensor((n * 4,), "f32") = flatten(lv0)
    lv2 = match_cast(unique(lv1), Tensor((m,), "f32"))      # m introduced for unknown result
    lv3: Tensor((m,), "f32") = exp(lv2)
```

Shape deduction propagates across function boundaries: if `subfn` takes `Shape(["n", "m"])` and returns `Tensor(("n*m",))`, a call `subfn(shape(n, 4))` statically resolves to output shape `(n*4,)`. Shapes that cannot be determined statically (e.g. the output of `unique`) introduce a fresh symbol at a `match_cast` boundary — preserving as much information as possible without requiring full shape specialisation.

### Operator Fusion Across Levels
{: #op-fusion}

The canonical example of cross-level optimisation is **fusing quantised weight decoding into a matmul**. Naively, this is two operations: `decode_q4` (dequantise 4-bit weights to fp16) followed by `mm` (matmul). Without fusion, the decoded fp16 weight matrix must be written to and read from HBM — doubling memory traffic.

Relax fuses this in three compiler passes:

<div class="post-flow" role="group" aria-label="Three-pass fusion for quantised matmul">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Pass 1 — AnalyzePattern: classify each TIR function as Injective, OutputEwiseFusible, etc.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Pass 2 — FuseOps: merge compatible adjacent ops at the graph level into a subgraph function</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Pass 3 — FuseTIR: inline the constituent TIR loop bodies into one fused kernel, eliminating the intermediate buffer</span></li>
  </ol>
</div>

After FuseTIR, `decode_q4` and `mm` become a single kernel that dequantises each weight element on-the-fly inside the matmul loop — the decoded value is computed in a register and consumed immediately, never written to HBM:

```c
// fused_decode_q4_mm: one kernel, no intermediate W buffer
for k, j in grid(128, 256):
    W_reg = ((data[k, j//8] >> (k%8*4)) & 15 - 7) * scale[k // 32]
for i, j, k in grid(n, 256, 128):
    Y[i, j] += X[i, k] * W_reg   // W stays in registers
```

This pattern generalises: any injective producer (elementwise decode, transpose, scale) followed by a reduction consumer (matmul, conv) can be fused the same way, with the compiler identifying the opportunity automatically from compute patterns.

### Structured Output Generation
{: #structured-gen}

Agentic LLM applications increasingly require outputs that conform to a schema — JSON for tool calls, SQL for NL2QL, typed structs for API integration. Constraint decoding enforces this by masking the vocabulary at each decoding step to only tokens valid under the current grammar state.

The challenge is speed: the LLM vocabulary is 128k+ tokens, grammar state is tracked by a pushdown automaton whose stack can be unbounded (infinite possible states), and the GPU decodes faster than a naive CPU mask generator can keep up.

**XGrammar** solves this with a key observation: for any grammar state, more than 99% of tokens can have their validity pre-computed offline by inspecting only the top of the parsing stack — they are **context-independent**. The remaining `<1%` context-dependent tokens are checked at runtime. At each decoding step:

<div class="post-flow" role="group" aria-label="XGrammar mask generation">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Retrieve pre-computed token mask for current grammar state — O(1) lookup</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Check context-dependent outlier tokens at runtime — tiny set, fast</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Apply combined mask to LLM logits before sampling</span></li>
  </ol>
</div>

Result: XGrammar achieves **170× faster** JSON schema mask generation and **260× faster** context-free grammar mask generation vs prior systems, at near-zero throughput overhead — making structured generation practical even for high-throughput serving.

### Universal Deployment: MLC Engine
{: #mlc-engine}

The Relax IR compiles to a universal runtime that targets CUDA, Vulkan, WebGPU, and Metal from the same model program. This is the basis of **MLC Engine** (Machine Learning Compilation Engine), which deploys the same Llama/Phi/Mistral model family across:

| Target | Backend | Notes |
|---|---|---|
| Cloud GPU server | CUDA + TensorCore + TMA | OpenAI-compatible API server |
| Desktop / laptop | Vulkan / Metal | Native app, no cloud required |
| iOS / Android | Metal / OpenCL | On-device, 4-bit quantised |
| Web browser | WebGPU | Runs in-browser via WebLLM |
| Embedded (Orange Pi, SteamDeck) | Vulkan | Sub-$100 hardware |

The same compilation pipeline — symbolic shape propagation, cross-level op fusion, quantisation-aware kernel generation — applies to all targets. Model developers write once; the compiler handles backend-specific code generation and memory planning per deployment context.

---

## Mixture of Experts
{: #moe}

Standard transformer blocks apply the same feed-forward network to every token in every layer — a dense computation where all weights participate regardless of input. **Mixture of Experts (MoE)** replaces the FFN with a collection of specialised subnetworks (experts), routing each token to only a small subset of them. The result: far more total parameters for the same per-token compute cost.

### MoE Layer
{: #moe-layer}

A standard FFN computes `H = LayerNorm(ReLU(ZW₁)W₂ + Z)` where `W₁ ∈ ℝⁿˣᵐ`. Increasing the feature size scales compute quadratically and mixes all information through a single bottleneck. MoE replaces this with N independent FFN experts and a learned gating network that selects K of them per token:

```
G = Softmax(X Wᴳ)                        # gating scores over N experts
I = TopK(G, k=2)                          # select top-2 experts
s₀ = G[i₀] / (G[i₀] + G[i₁])            # normalised weights
Y = s₀ · FFNᵢ₀(X) + s₁ · FFNᵢ₁(X)      # weighted expert outputs
```

**Mixtral-8×7B** is the canonical example: 8 experts, 2 activated per token. Every token uses 2 of 8 experts, so the active parameter count stays the same as a dense 7B model while total parameters are 8×7B = 56B. The model gains capacity to specialise without paying the full compute cost.

<div class="post-flow post-flow--compare" role="group" aria-label="Dense FFN vs MoE layer">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Dense FFN</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">All weights active for every token</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Compute scales quadratically with feature size</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">No specialisation — everything mixed in one network</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">MoE Layer ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Only K of N experts active per token</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Same per-token compute, N× more parameters</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Experts specialise on different input distributions</span></li>
    </ol>
  </div>
</div>

In a Transformer, the MoE layer simply replaces every FFN block — self-attention stays dense, experts handle the position-wise feed-forward computation.

### Efficient MoE Computation
{: #moe-compute}

Naively executing MoE means running a separate matmul per expert, losing the batching efficiency that makes GPUs fast. The solution is to **permute tokens by expert assignment and batch all expert computations together**.

**Token permutation via prefix sum**: given a batch of B tokens each routed to K experts, build a selection mask of shape `[B, N]` and compute its column-wise prefix sum (cumulative sum). The prefix sum values give each token's position in the sorted expert order — a step that parallelises perfectly on GPU via a scan kernel.

**Grouped GEMM (GroupGemm)**: after permutation, all tokens assigned to expert 0 are contiguous in memory, followed by expert 1's tokens, and so on. This is stored in CSR (compressed sparse row) format: a `data` array of token activations and an `indptr` array marking where each expert's chunk starts. A single `GroupGemm` kernel then dispatches one GEMM per expert in parallel:

```
Z = GroupGemm(data, indptr, weights)
# data: permuted token activations
# indptr: [0, n₀, n₀+n₁, ..., B·K]
# weights: [W₁, W₂, ..., Wₙ]
```

<div class="post-flow" role="group" aria-label="Batched MoE computation pipeline">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Gating network scores all N experts for each token</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">TopK selection → build selection mask [B × N]</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Prefix sum on mask columns → permutation indices</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Permute tokens → contiguous groups per expert</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">GroupGemm: one GEMM per expert, batched in one kernel</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Un-permute outputs → weighted sum per token</span></li>
  </ol>
</div>

The key efficiency gain: in the single-batch setting, only K of N experts are loaded and computed. In a batched setting, different tokens route to different experts, so across a large batch all experts are likely active — but each individual expert's sub-batch is smaller than the full batch, making GroupGemm the right primitive instead of a single large GEMM.

---

## LLM Fine-tuning
{: #llm-finetuning}

A pretrained base model is a general-purpose predictor. Fine-tuning adapts it to a downstream task — but full fine-tuning of a 175B model requires the same hardware as pretraining (80 A100s, 1 TB per checkpoint) and is impractical for most use cases. **Parameter-efficient fine-tuning (PEFT)** methods reduce trainable parameters by orders of magnitude while retaining most of the accuracy gain.

### Prompt & Prefix Tuning
{: #prompt-tuning}

The lightest-weight approach: don't touch the model weights at all. **Prompt engineering** manually crafts text prefixes that steer the model via in-context learning — chain-of-thought, few-shot examples, role descriptions. No gradient required. The limitation is that discrete tokens are not differentiable; you cannot optimise the prompt directly on a labelled dataset.

**Prefix tuning** makes the prompt continuous and trainable. A sequence of **virtual tokens** — vectors with learnable parameters — is prepended to the input at every layer. The frozen LLM attends to these prefix vectors exactly as if they were real token embeddings, but the prefix parameters are updated by backpropagation through the task loss:

<div class="post-flow post-flow--compare" role="group" aria-label="Prompt engineering vs prefix tuning">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Prompt Engineering</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Discrete text tokens prepended to input</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">No gradient — relies on in-context learning</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Hard to optimise systematically</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Prefix Tuning ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Continuous virtual token vectors prepended at each layer</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">LLM frozen; only prefix parameters trained</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Directly optimised on labelled dataset via backprop</span></li>
    </ol>
  </div>
</div>

### Adapter Tuning
{: #adapter-tuning}

**Adapters** insert small trainable modules between the frozen layers of the LLM. Each adapter is a bottleneck: a down-projection to a low-dimensional space, a non-linearity, and an up-projection back. Initialised near-identity so training is stable. Only adapter weights are updated; the base model is never touched.

Adapters achieve comparable accuracy to full fine-tuning with far fewer trainable parameters. The drawback: they add new layers, which means extra memory for intermediate activations during backprop and a small inference latency overhead.

### LoRA
{: #lora}

**LoRA (Low-Rank Adaptation)** avoids adding new layers by reparameterising the weight update instead. For a weight matrix `W ∈ ℝᵈˣᵈ`, the update `ΔW` during fine-tuning is hypothesised to lie in a low-rank subspace. LoRA factors it as:

```
ΔW = B × A,   B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵈ,   r ≪ d
```

During training, `W` is frozen and only `A` and `B` are updated. `A` is initialised from a Gaussian; `B` is initialised to zero so `ΔW = 0` at the start of training. At inference, the adapted weight is folded in: `W' = W + BA`. **No inference latency overhead** — the merged weight is the same shape as the original.

<div class="post-flow post-flow--compare" role="group" aria-label="Full fine-tuning vs LoRA">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Full Fine-tuning</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">ΔW has d² trainable parameters</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Full Adam state: 3× weight memory</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">80 A100s for 175B GPT-3</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">LoRA ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">ΔW = BA has 2dr parameters — r× cheaper</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Adam state only for A and B — tiny</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">No inference overhead — fold BA into W at deploy time</span></li>
    </ol>
  </div>
</div>

LoRA is typically applied to the query and value projection matrices in attention, and to the MLP layers. Variants replace the matrix product with a Hadamard product (LoHa) for higher rank at the same parameter count, or a Kronecker product (LoKr) to preserve the original matrix's rank structure.

### QLoRA
{: #qlora}

LoRA reduces trainable parameters but the frozen base model weights still consume memory in fp16. **QLoRA** quantises the frozen weights to 4 bits, cutting base model memory by 4×, while keeping the LoRA adapter weights and activations in fp16 for stable training.

Naive 4-bit quantisation wastes bins when the weight distribution has large outliers. QLoRA addresses this with two levels of quantisation:

1. **Block-wise quantisation** — divide weights into blocks of size B (e.g. 64); each block has its own fp32 scaling constant. Internal fragmentation: 32/B extra bits per parameter.
2. **Double quantisation** — quantise the fp32 scaling constants themselves with a second 8-bit quantisation (block size 256). This reduces the overhead from 0.5 bits/param to ~0.127 bits/param.

```
Layer: 4-bit frozen weights (base model)
     + 16-bit LoRA A, B matrices
     + 16-bit activations
→ memory: ~5 bits/param total vs 16 bits for fp16 base + LoRA
```

QLoRA achieves on-par accuracy with full fp16 fine-tuning, enabling 70B-scale models to be fine-tuned on a single GPU.

### Side Tuning
{: #side-tuning}

Both adapters and LoRA still require backpropagating through the frozen base model to update adapter/LoRA weights — which means storing all intermediate activations in memory. For a 70B model this is hundreds of GB regardless of how few parameters are being trained.

**Side tuning** eliminates this by introducing a **smaller parallel side network** that runs alongside the frozen base model. Information flows from base to side (via downsampled residual connections) but never from side back to base. Backpropagation only traverses the side network:

<div class="post-flow" role="group" aria-label="Side tuning information flow">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Base LLM: 4-bit quantised, frozen, forward-only</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each base layer's output downsampled → injected into side network</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Side network: small trainable transformer, runs in fp16</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Backprop stays entirely within side network — no base activations stored</span></li>
  </ol>
</div>

**Quantized Side Tuning (QST)** combines 4-bit double quantisation of the base model with a side network: intermediate activations are ~4 GB for a 70B model vs 197 GB for full fine-tuning and 78 GB for QLoRA, while matching QLoRA's output quality across benchmarks.

| Method | Weights | Optimizer states | Intermediate activations | Total (70B) |
|---|---|---|---|---|
| Full fine-tuning | 140 GB | 420 GB | 197 GB | 757 GB |
| QLoRA | 36 GB | 13 GB | 197 GB | 246 GB |
| QST | 36 GB | 4 GB | 69 GB | 109 GB |

---

## LLM Serving
{: #llm-serving}

Training a large model is one problem; serving it to thousands of concurrent users is another. The core tension in LLM serving is that autoregressive decoding is inherently sequential — each token depends on the one before — while a production system must saturate expensive GPU hardware with many simultaneous requests. This section covers three key system-level techniques for closing that gap, followed by speculative decoding as a complementary algorithmic approach.

**The serving problem in numbers**: serving GPT-3 (175B parameters) in fp16 requires at least ten A100-40GB GPUs. Generating 256 tokens takes ~20 seconds. A single request's KV cache takes ~3 GB of GPU memory. Without careful system design, the GPU sits idle between requests and memory fragmentation limits batch size.

---

### Continuous Batching
{: #continuous-batching}

The naive approach to batching LLM requests is **static batching**: collect a fixed set of requests, run them together until all finish, then start the next batch. This breaks down because requests finish at different times — a short request completing at iteration 10 leaves its GPU slot idle until the longest request in the batch finishes at iteration 100.

<div class="post-flow post-flow--compare" role="group" aria-label="Static vs continuous batching">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Static Batching</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Fixed batch runs to completion</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Short requests leave idle GPU slots</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">New requests wait for entire batch to finish</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Continuous Batching ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Completed requests evicted mid-batch</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">New requests inserted immediately into free slots</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">GPU stays saturated at all times</span></li>
    </ol>
  </div>
</div>

**Continuous batching** (introduced in ORCA, OSDI'22) operates at the iteration level rather than the request level. After every decoding step, the scheduler checks which requests have emitted `<EOS>` and evicts them, then admits waiting requests from the pool to fill the freed slots — all before the next GPU iteration begins. The batch composition changes every step; the GPU never sees an idle slot.

**Step-by-step example** with a max batch size of 3:

<div class="post-flow" role="group" aria-label="Continuous batching step by step">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Iter 1 — batch: [R1, R2]; R3 waits in pool</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Iter 2 — batch: [R1, R2, R3]; R2 emits EOS → evicted; R4, R5 arrive in pool</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Iter 3 — R2's slot filled by R4; batch: [R1, R3, R4]</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">GPU never idles — R5 admitted as soon as any slot opens</span></li>
  </ol>
</div>

The CPU scheduler runs the following loop every iteration: receive new requests, process results from the previous GPU step, check stop conditions, run prefix matching and request reordering, and allocate memory for the next batch. An unoptimised scheduler can consume more than 50% of wall time doing this on CPU — a problem addressed by **overlapped scheduling**, where the CPU processes the results of iteration `t` while the GPU runs iteration `t+1` concurrently.

---

### PagedAttention
{: #pagedattention}

Continuous batching fixes GPU utilisation. The next bottleneck is memory: the KV cache for each request grows token by token during decoding, and naive implementations pre-allocate a contiguous block of GPU memory sized to the request's *maximum possible output length*. This wastes memory in two ways:

- **Internal fragmentation** — the actual output is almost always shorter than the maximum; the tail of the block is never used
- **External fragmentation** — requests with different max lengths leave gaps between allocations that are too small to use for any other request

In practice, only 20–40% of allocated KV cache memory holds actual token states.

**PagedAttention** (vLLM) borrows the OS virtual memory abstraction and applies it to KV cache. Physical GPU memory is divided into fixed-size **KV blocks** (e.g. 4 tokens per block). Each request gets a **block table** — a mapping from logical block index to physical block number — rather than a contiguous reservation.

<div class="post-flow" role="group" aria-label="PagedAttention memory layout">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Physical memory divided into fixed KV blocks (e.g. 4 tokens each)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each request assigned a block table: logical block → physical block</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">New blocks allocated on demand as the sequence grows</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Blocks freed immediately when a request finishes — no wasted reservation</span></li>
  </ol>
</div>

**Example**: request A with prompt "Alan Turing is a computer scientist" fills logical blocks 0–2 (3 blocks × 4 tokens = 12 tokens for an 8-token prompt, with block 2 partially filled). As decoding produces "and mathematician renowned…", new physical blocks are allocated on demand and appended to the block table. The last block is the only source of internal fragmentation — wasted tokens per request is strictly less than the block size.

Attention computation with a block table works because **attention is associative and commutative**: the kernel fetches non-contiguous KV blocks using the block table and applies attention on the fly. Block layout is invisible to the attention math.

**Memory efficiency**: PagedAttention eliminates external fragmentation entirely and caps internal fragmentation at `block_size − 1` tokens per request, raising effective KV cache utilisation from 20–40% to near 100%.

---

### RadixAttention
{: #radixattention}

PagedAttention handles one request's KV cache efficiently. But across requests, there is often massive redundancy: many requests share a common system prompt, few-shot examples, or conversation prefix. Standard serving systems discard the KV cache when a request finishes and recompute it from scratch for every new request — even if the prefix is identical.

**RadixAttention** (SGLang) maintains a global **LRU cache of KV blocks** organised as a **radix tree** (a compact prefix tree). The key of each tree node is a token sequence; the value is the cached KV block for that sequence. When a new request arrives, the scheduler performs a longest-prefix match against the tree. Any matched prefix is loaded directly from cache, skipping the prefill computation for those tokens entirely.

<div class="post-flow" role="group" aria-label="RadixAttention prefix sharing">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Request arrives → longest-prefix match in radix tree</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Matched prefix: load KV blocks from cache, skip prefill</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Unmatched suffix: run prefill, extend the tree with new KV blocks</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">On eviction: LRU policy removes least-recently-used leaf nodes first</span></li>
  </ol>
</div>

This is especially effective for multi-turn chat (each turn shares the full conversation history), batch inference over a shared system prompt, and few-shot prompting where the examples are constant. The radix tree structure ensures that the shared prefix is stored once regardless of how many requests use it.

---

### Speculative Decoding
{: #speculative-decoding}

All techniques so far improve *throughput* — more requests per second. **Speculative decoding** attacks a different problem: reducing *latency* for individual requests by exploiting the fact that LLM inference is memory-bandwidth-bound, not compute-bound.

The bottleneck: at decode time the GPU must load all model weights from HBM on every step just to produce one token. Compute utilisation for a 70B model on 4 A100s with 4K context is only ~2%, while memory bandwidth is near saturation. The GPU is doing almost no arithmetic relative to its capability.

**Key insight**: if we can propose multiple draft tokens cheaply and verify them all in one LLM forward pass, we produce multiple tokens per LLM call while keeping the output distribution identical to autoregressive decoding.

**Model-based speculative decoding** uses a small speculative model (SSM) to draft, and the large LLM to verify:

<div class="post-flow" role="group" aria-label="Speculative decoding workflow">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">SSM autoregressively generates γ draft tokens — fast, cheap</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">LLM runs one forward pass over [prompt + γ draft tokens] in parallel</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">LLM output at position i verified against SSM draft token at position i</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">All matching prefix tokens accepted; first mismatch corrected from LLM distribution</span></li>
  </ol>
</div>

**Acceptance criterion** (speculative sampling): given SSM probability `p(x)` and LLM probability `q(x)` for draft token `x`:
- If `p(x) ≤ q(x)`: accept `x` unconditionally — the SSM was conservative, LLM agrees
- If `p(x) > q(x)`: accept `x` with probability `q(x)/p(x)` — the SSM was overconfident
- On rejection: sample from the normalised residual `norm(max(0, q(x) − p(x)))` to correct the distribution

This guarantees the joint output distribution is identical to sampling from the LLM alone — speculative decoding is lossless.

**SpecInfer** extends this with **tree-based speculation**: instead of a single linear sequence of γ draft tokens, multiple SSMs each generate a sequence, and their outputs are merged into a token tree. The LLM then verifies the entire tree in one pass using **tree attention** — a topology-aware causal mask that runs tree-structured attention in a single GPU kernel without redundant computation.

<div class="post-flow" role="group" aria-label="SpecInfer tree-based speculation">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Multiple SSMs each produce a draft token sequence</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Token tree merge: deduplicate shared prefixes into a compact tree</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">LLM verifies all tree nodes in one forward pass with tree attention</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Accepted path: longest verified prefix from the tree — up to 2.4× speedup</span></li>
  </ol>
</div>

Tree attention linearises the token tree depth-first and applies a topology-aware causal mask: token `tᵢ` attends to `tⱼ` iff `tⱼ` is an ancestor of `tᵢ` in the tree. This replaces the usual left-to-right causal mask and allows all tree nodes to be decoded in one kernel call.

**EAGLE** is a more recent variant that tightens the draft model design. Instead of a fully separate SSM, the draft model reuses the LLM's embedding and language model head layers and adds only a small trainable attention layer on top. During each draft step, EAGLE expands `K` candidate nodes and then rerankds them by accumulated likelihood before selecting which subtree to verify — a dynamic tree structure that adapts to the prompt.

---

### Model-Free Speculation
{: #model-free-speculation}

Model-based approaches require maintaining a separate SSM. Model-free speculation generates draft tokens from patterns in the model's own previous outputs — no extra model parameters needed.

**Prompt lookup decoding** is the simplest: search the input prompt for the current decoding suffix, and if found, directly copy the next few tokens from the prompt as draft tokens. This works well for tasks with heavy prompt-to-output repetition (summarisation, code editing, RAG).

**SuffixDecoding** generalises this with a two-tier suffix tree:

<div class="post-flow post-flow--compare" role="group" aria-label="SuffixDecoding two-tier tree">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Request Tree</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Indexes tokens generated so far for this request</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Captures request-specific repetition patterns</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Global Tree</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Indexes outputs across all previous requests</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Captures cross-request repetition — shared domain vocabulary</span></li>
    </ol>
  </div>
</div>

At each step, the current decoding suffix is looked up in both trees. Candidate continuations are expanded, scored by accumulated path likelihood, and assembled into a speculation tree that the LLM verifies in one pass.

**Lookahead decoding** takes a different angle: run two branches in parallel at each step — a *lookahead branch* that generates one token at each position `i, i+1, i+2, …` ahead of the current position, and a *verification branch* that checks n-grams assembled from a pool of previously seen lookahead tokens. Newly confirmed n-grams are added to the pool. Both branches are batched into a single LLM forward pass, and any verified n-gram that extends the current output is accepted — breaking the sequential dependency without a separate draft model.

<div class="post-next-link">
  <span class="post-next-label">Read next</span>
  <a href="{{ '/blogs/cuda-programming-gpu-architecture' | relative_url }}" class="post-next-title">CUDA Programming &amp; GPU Architecture →</a>
  <p class="post-next-desc">Threads, warps, shared memory, and the two-level tiling strategy that maps high-level kernels onto Streaming Multiprocessors.</p>
</div>
