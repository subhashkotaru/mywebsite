---
title: "ML Systems: Training, and Serving"
date: 2026-04-21
description: "Machine learning systems — from automatic differentiation to training at scale and production serving."
tags: [ml-systems, machine-learning, engineering]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#scalable-ai">Scalable AI: The End-to-End Engineering Discipline</a>
      <ul class="post-toc-sublist">
        <li><a href="#lifecycle">The AI Lifecycle (Stages 0–6)</a></li>
        <li><a href="#ai-stack">The Modern AI Stack</a></li>
        <li><a href="#two-views">Model View vs Systems View</a></li>
        <li><a href="#training-vs-inference">Training Frameworks vs Inference Engines</a></li>
        <li><a href="#scaling-walls">Scaling Walls</a></li>
      </ul>
    </li>
    <li><a href="#overview">Overview of ML(Model) Systems</a></li>
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
    <li><a href="#parallelism-advanced">Parallelism: Communication, Context &amp; Experts</a>
      <ul class="post-toc-sublist">
        <li><a href="#comm-model">The α–β Communication Cost Model</a></li>
        <li><a href="#topology-mapping">Topology Mapping</a></li>
        <li><a href="#dp-fsdp-hsdp">DP → ZeRO → FSDP → HSDP</a></li>
        <li><a href="#tp-advanced">Tensor Parallelism In Depth</a></li>
        <li><a href="#pp-advanced">Pipeline Parallelism In Depth</a></li>
        <li><a href="#cp-sp">Context &amp; Sequence Parallelism (CP/SP)</a></li>
        <li><a href="#ep">Expert Parallelism (EP)</a></li>
        <li><a href="#5d-mesh">5D Mesh Composition</a></li>
        <li><a href="#parallelism-recipes">Training vs Serving Recipes</a></li>
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

## Scalable AI: The End-to-End Engineering Discipline
{: #scalable-ai}

> *"In 2026, 'a training run' is not a system. Large-scale AI is end-to-end engineering, and quality, cost, and reliability need co-design across stages."*  
> — UC Berkeley Scalable AI, Spring 2026

Most ML courses treat model training as the core problem and serve the model as an afterthought. Real production AI is the opposite: the training run is one stage in a **seven-stage lifecycle**, and decisions made at Stage 0 (target definition) cascade all the way to Stage 5 (application reliability). Upstream mistakes — a tokenizer with the wrong vocab size, an architecture with too many KV heads, data contamination in the eval set — compound silently until they become expensive downstream failures.

This section frames the **two maps** you need to reason about any large-scale AI system:

1. **The Lifecycle** (time axis) — what happens at each stage and what artifact each stage produces
2. **The Stack** (layer axis) — what software and hardware layers support each workload, and which layer is your current bottleneck

<div class="post-flow post-flow--horizontal" role="group" aria-label="Two coordinate systems for large-scale AI">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Lifecycle map — time: Stages 0 → 6</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Stack map — layers: workloads → frameworks → compute substrate</span></li>
  </ol>
</div>

---

### The AI Lifecycle (Stages 0–6)
{: #lifecycle}

Every stage has a **goal**, a set of **decisions**, and a **concrete artifact**. If you can't name the artifact, you're not done with the stage.

<div class="post-flow" role="group" aria-label="AI lifecycle stages 0 to 6">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Stage 0 — Targets: define success criteria before burning GPU-hours</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Stage 1 — Architecture: choose a model family that fits the constraint envelope</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Stage 2 — Pre-training: large-scale self-supervised learning on curated data</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Stage 3 — Post-training: SFT + preference optimisation → controllable, useful model</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Stage 4 — Inference: serve under real traffic within SLOs and cost budgets</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Stage 5 — Applications: context engineering, tool use, reliability, guardrails</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Stage 6 — Research: tighten the next lifecycle with credible experimental evidence</span></li>
  </ol>
</div>

**Stage 0 — Targets**

Before any training, get painfully concrete:

- **Target distribution**: what do you want to be good at — chat, code, math, tool use?
- **Quality definition**: specific benchmarks, human evals, app-level metrics (not "generally capable")
- **Cost envelope**: training GPU-hours, inference SLOs (p50/p99 latency, tokens/$ budget)

The recurring failure mode: "we will measure that later." Without a measurement, capability silently regresses. Every post-training decision and every inference optimisation traces back to the quality bar set here.

**Stage 1 — Architecture**

Architecture decisions have a long tail: they set training dynamics *and* the economics of serving for the model's entire lifespan. Key choices:

| Axis | Choice | Serving consequence |
|---|---|---|
| Attention | Full vs. GQA vs. MLA | KV cache size per request |
| FFN | Dense vs. MoE | Expert routing overhead; memory spill |
| Context | Full vs. sparse/linear attention | Long-context latency and memory |
| Normalisation | Pre-norm vs. post-norm | Stability at depth; fine-tuning sensitivity |

Artifact: a randomly-initialised model definition (architecture config + weight shapes).

**Stage 2 — Pre-training**

Industrial-scale self-supervised learning is three coupled problems:

1. **Data** — acquisition, deduplication, quality filtering, decontamination from eval sets (NeMo Curator)
2. **Training** — loss schedule, numerical stability, throughput engineering (NeMo AutoModel / Megatron)
3. **Systems** — distributed strategy, memory management, fault tolerance

Artifact: a base model checkpoint + training telemetry (loss curves, stability signals, throughput).

**Stage 3 — Post-training**

Converts "capable" into "controllable and useful":

- **SFT** — instruction following, format discipline, tool call schemas
- **Preference optimisation** (RLHF/DPO/GRPO) — helpfulness, safety, task success
- Evaluation shifts from perplexity to **behaviour-centric**: task success rate, refusal accuracy, schema validity

Artifact: a serving-ready post-trained checkpoint.

**Stage 4 — Efficient Inference**

The four questions every inference engineer asks:

1. What is the cost per token?
2. What is p50/p99 latency per request?
3. How do we batch without breaking user experience (chunked prefill, continuous batching)?
4. What happens when context length grows (KV cache pressure, disaggregation)?

Tools: compilation/graph capture (Dynamo), quantisation (TRT-LLM), decoding engines ([vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang)).

Artifact: a production serving configuration, often with quantised variants per SLO tier.

**Stage 5 — Applications**

Where weights meet real users. The stack below the model determines reliability:

- **Context engineering**: retrieval, reranking, memory, compression
- **Tool use**: function-calling loops, agents, planners, verifiers (NeMo Agent Toolkit)
- **Reliability**: schema-constrained outputs, retries, validation, guardrails (NeMo Guardrails)
- **Safety**: adversarial evaluation (Garak), content moderation hooks, incident response

A single weak link upstream — data contamination, misaligned fine-tuning, aggressive quantisation — can make weights that look great in isolation fail continuously in production.

**Stage 6 — Research**

Once the stack is understood, research questions sharpen:

- Which bottleneck is **fundamental** vs. contingent on current hardware?
- Which architectural change reduces total cost without breaking quality?
- What training objective unlocks better reasoning, tool use, or robustness?

Artifact: evidence (positive *or* negative) that tightens the next lifecycle iteration. A clean negative result with a diagnosis teaches more than a noisy positive.

---

### The Modern AI Stack
{: #ai-stack}

The lifecycle runs on a **layered stack**. When something is slow, expensive, or unreliable, the question is: *which layer is the bottleneck?*

| Layer | What it does | Example tooling |
|---|---|---|
| **AI Workloads** | Data pipelines, training, eval, serving, safety monitoring | NeMo Curator, NeMo Data Designer |
| **Frameworks & Engines** | Differentiable compute graphs, distributed training, serving engines | PyTorch/JAX, Megatron, FSDP, DeepSpeed, vLLM, SGLang |
| **Distributed Compute** | Task scheduling, fault tolerance across nodes | Ray, Spark, custom distributed services |
| **Orchestration** | Container and cluster management | Kubernetes, SLURM, VM-based deployments |
| **Compute Substrate** | Physical GPUs, networking (NVLink, InfiniBand), storage, cloud | H100 nodes, NVLink3, RDMA over IB |

**Why layers matter**: a kernel tuning effort (Frameworks layer) can be bottlenecked by interconnect bandwidth (Compute Substrate layer). A scheduling improvement (Orchestration layer) only helps if the Frameworks layer isn't blocking on synchronisation. Diagnosing at the wrong layer wastes time.

---

### Model View vs Systems View
{: #two-views}

High-performing teams hold **two complementary reasoning modes** simultaneously and switch between them deliberately:

<div class="post-flow post-flow--compare" role="group" aria-label="Model view vs systems view">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Model View</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Optimise for quality: loss, benchmarks, behaviour</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Choose architecture for scaling on the target distribution</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Curate data for specific capabilities, not just "more tokens"</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Risk: great weights that can't hit latency/cost budgets</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Systems View</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Start from serving constraints: SLOs, budget, memory ceilings</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Design architecture + deployment that satisfies those constraints</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Train within that envelope; measure relentlessly</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Risk: efficient model that can't learn enough to be useful</span></li>
    </ol>
  </div>
</div>

The trap of the pure **Model View**: you train a model with excellent benchmark numbers, then discover at Stage 4 that the KV cache is 12 GB per request — too large to serve at reasonable batch sizes — because the architecture was never constrained by serving memory. Fixing this requires retraining.

The trap of the pure **Systems View**: you optimise aggressively for throughput (aggressive quantisation, very short context window) and ship a model that is fast but too restricted to be useful — the quality bar from Stage 0 was never met.

The correct approach: start with the **serving constraint** (latency budget, memory per GPU, $/1M tokens), work backwards to architecture constraints (KV heads, context length, quantisation target), then optimise quality within that envelope.

**Interview question**: A model passes all benchmarks in evaluation, but after deployment latency SLOs are consistently violated. Walk through how you would diagnose whether the bottleneck is compute, memory bandwidth, KV cache size, or scheduling.

> **Answer**: start with profiling, not guessing. (1) Is GPU compute saturated? — check SM utilisation; if low, the kernel is memory-bandwidth-bound. (2) Is HBM bandwidth saturated? — roofline analysis: at batch=1 decode, arithmetic intensity ≈ 2 FLOP/byte vs. A100's 156 roofline, so decode is almost always bandwidth-bound. Fix: larger batch, quantisation to reduce weight bytes. (3) Is KV cache pressure causing eviction or swap? — vLLM metrics: block utilisation, eviction rate. Fix: PagedAttention, smaller KV (GQA, MLA), aggressive KV quantisation. (4) Is the scheduler introducing latency? — measure queuing time vs. compute time per request; overlapped scheduling or disaggregated prefill/decode may help.

---

### Training Frameworks vs Inference Engines
{: #training-vs-inference}

The same weights run on fundamentally different software stacks for training and inference. Confusing the two causes misdiagnosed performance problems.

| | Training Frameworks | Inference Engines |
|---|---|---|
| **Examples** | NeMo AutoModel, Megatron, DeepSpeed, FSDP | Dynamo, TensorRT-LLM, vLLM, SGLang |
| **Primary goal** | Throughput — eat tokens fast | Latency and concurrency |
| **Key concern** | Numerical stability under backprop | Continuous batching + request scheduling |
| **Batching** | Large static mini-batches; gradient sync | Dynamic batches; variable sequence lengths |
| **Memory** | Activation checkpointing, optimizer state sharding | KV cache management; fragmentation avoidance |
| **Fault model** | Resume from checkpoint on node failure | Zero-downtime deploys; request retries |

**Why this matters**: a training engineer who optimises their kernel for training throughput may ship a kernel that is fast in the training regime (large batch, full matrix compute) but memory-bandwidth-bound in the serving regime (batch=1, memory-bound decode). The roofline position is completely different.

**Interview question**: You are told "our model is slow at inference." What is the first thing you measure, and why?

> **Answer**: time-to-first-token (TTFT) vs. time-between-tokens (TBT). They diagnose different bottlenecks. High TTFT → prefill is slow (compute-bound, long prompt, or chunked prefill chunk too small). High TBT → decode is slow (memory bandwidth, KV cache pressure, small batch). Once you know which phase is slow, profile the GPU at that phase: SM utilisation, HBM bandwidth utilisation, and KV block eviction rate. Diagnosis drives the fix (chunked prefill scheduling, quantisation, PagedAttention, speculative decoding, etc.).

---

### Scaling Walls
{: #scaling-walls}

Every large-scale AI system eventually hits one of five **scaling walls**. The skill is diagnosing *which one* from measurements, not intuition.

| Wall | What it limits | Typical symptom | Fix direction |
|---|---|---|---|
| **Compute (FLOPs)** | Training throughput, prefill speed | GPU SM utilisation saturated | Better kernels, tensor parallelism, operator fusion |
| **Memory capacity** | Model size, KV cache batch size | OOM errors, small max batch | ZeRO/FSDP, quantisation, GQA/MLA, KV offloading |
| **Memory bandwidth** | Decode throughput per GPU | Low SM util, high HBM util | Larger batch to amortise weight loads, quantisation |
| **Communication** | Scaling across GPUs/nodes | AllReduce dominates step time | Gradient compression, topology-aware parallelism, overlap |
| **Data** | Training quality, convergence speed | Loss plateau despite more compute | Better curation, deduplication, synthetic data |

The wall you're hitting changes with scale. A single-GPU fine-tuning job is memory-capacity-bound (can't fit the model). A 512-GPU pretraining run is often communication-bound (AllReduce latency dominates). A high-throughput serving system is memory-bandwidth-bound (weight loads per decode step exceed compute).

**Common misconception**: "our GPU utilisation is only 40%, so we're compute-bound." GPU SM utilisation being low means you are *not* compute-bound — you are probably memory-bandwidth-bound (the GPU is waiting for data) or communication-bound (it's stalled on a collective). True compute-bound systems have SM utilisation near 100%.

---

## Overview of Machine Learning(Model) Systems
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

**[FlashAttention](https://arxiv.org/abs/2205.14135) in decoding** doesn't apply directly: with a single query there is no parallelism across queries, and the kernel must scan all keys/values **sequentially** — inefficient when the context is long.

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

## Parallelism: Communication, Context & Experts
{: #parallelism-advanced}

The previous sections introduced the three main parallelism axes. This section adds the **communication cost model** that lets you predict when collectives dominate, the **topology mapping rules** that decide which strategies go where, two newer parallelism dimensions (**CP/SP** for long context, **EP** for MoE), and practical recipes for assembling these into a **5D mesh** for both training and serving.

The extended runtime bound with communication as a third roof:

```
T_total ≳ max( T_compute,  T_HBM,  T_comm )
                              ↑
                       α + β × bytes
```

Parallelism shifts pressure among these three terms and decides what sits on the critical path.

---

### The α–β Communication Cost Model
{: #comm-model}

Every distributed collective operation pays two costs:

```
T_comm(n bytes) ≈ α + β · n

α = startup latency    (kernel launch, routing setup, synchronisation)
β = 1 / bandwidth      (time per byte; set by the bottleneck link)
```

| Message regime | Dominant term | Fix |
|---|---|---|
| Small (n ≪ α/β) | α — latency-bound | Batch/fuse messages; reduce collective count |
| Large (n ≫ α/β) | β·n — bandwidth-bound | Compress, quantise, or shard data |

**Why this matters for decode.** During autoregressive generation the token batch is tiny — TP collectives carry `B_tok · D` bytes where B_tok=1. At BF16 with D=8192 that's just 16 KB per collective. On NVLink (α ≈ 2–5 µs) the latency term already dominates. Adding more TP ranks multiplies α without reducing bytes — TPOT grows linearly with TP group size for small batches.

**Ring AllReduce cost.** For p ranks and a tensor of n bytes, ring AllReduce (ReduceScatter + AllGather) costs:

```
T_ring(n) ≈ 2(p−1)·α  +  2·(p−1)/p · β·n
```

For large n the bandwidth term dominates and approaches 2βn regardless of p — Ring AllReduce achieves near-optimal bandwidth scaling. For small n, the 2(p−1)α startup term dominates — bigger groups hurt latency.

**Gradient bucketisation** exploits this: instead of AllReducing each parameter tensor individually (paying α per tensor), pack gradients into 25–100 MiB buckets. Fewer, larger messages amortise the α cost while the bucket launches are overlapped with ongoing backprop.

**The eight collective primitives:**

| Primitive | What it does | Used by |
|---|---|---|
| **AllReduce** | Sum across ranks; everyone gets full result | DP gradient sync |
| **ReduceScatter** | Reduce then shard output (each rank keeps one slice) | ZeRO-2/3, FSDP backward |
| **AllGather** | Gather shards so everyone gets full tensor | ZeRO-3/FSDP forward, TP |
| **AllToAll** | Many-to-many "transpose" of shards | MoE dispatch/combine (EP) |
| **Broadcast** | Root sends to all | Checkpoint distribution |
| **Gather/Scatter** | Rooted: build or distribute full tensor | Initialisation |
| **SendRecv** | Point-to-point | Pipeline stage boundaries (PP) |

Key: if you can name the primitive a strategy uses, you can predict payload size, topology sensitivity, and whether α or β dominates.

> **Interview question:** A training run on 64 GPUs achieves 60% MFU. Profiling shows 35% of step time is AllReduce. Gradient bucketing is already enabled. What are the likely remaining causes and fixes?
>
> *With bucketing already enabled the problem is either (1) **not enough overlap** — backprop for later layers finishes too quickly, so the bucket AllReduce for early layers can't start until backprop completes. Fix: reduce bucket size slightly so buckets launch earlier; or use ZeRO-2 (ReduceScatter per bucket instead of AllReduce, same bytes but pipelined differently). (2) **Inter-node bandwidth bottleneck** — 64 GPUs likely spans multiple nodes; if all DP AllReduces cross InfiniBand the β term dominates for large models. Fix: HSDP — reduce within node on NVLink first, then reduce the partial result across nodes. (3) **Gradient accumulation not used** — if global batch is fixed and DP=64, each GPU's local batch is tiny; accumulate gradients over A micro-steps to reduce AllReduce frequency by A×. (4) **Too many DP ranks, too few TP** — if one layer's weight matrix is 8GB and TP=1, the backward GEMM is memory-bound and fast, leaving a long communication window uncovered. Try TP=2 or 4 to slow down individual GEMMs and make overlap more effective.*

---

### Topology Mapping
{: #topology-mapping}

Not all GPU pairs have the same latency and bandwidth. The same collective on NVLink vs InfiniBand can differ by 10–50× in bandwidth and 5–10× in latency. Topology mapping assigns process groups to physical hardware so the **hottest collectives use the fastest links**.

**Typical cluster hierarchy:**

| Scope | Link | Bandwidth | Latency |
|---|---|---|---|
| Within a node (GPU–GPU via NVSwitch) | NVLink | ~900 GB/s aggregate | ~1–2 µs |
| Within a node (GPU–GPU via PCIe) | PCIe | ~64 GB/s | ~5–10 µs |
| Across nodes | InfiniBand HDR/NDR | ~25–50 GB/s per NIC | ~1–5 µs + routing |
| Across nodes | Ethernet | ~12.5–50 GB/s | ~5–20 µs |

**Hot vs cold collectives:**

| Collective | Frequency | Hotness | Placement rule |
|---|---|---|---|
| TP per-layer AllReduce/AllGather | Every layer, forward + backward | 🔥 Hot | Within node (NVLink) |
| EP AllToAll dispatch/combine | Every MoE layer | 🔥 Hot | Within node if possible |
| CP attention exchange | Every attention layer | 🔥 Hot | Within node; cross-node carefully |
| PP SendRecv at stage boundaries | Every microbatch × boundary | Medium | Can span nodes (point-to-point) |
| DP gradient AllReduce | Once per optimizer step | ❄ Cold | Can span nodes; overlap with backprop |

**General rule:** Frequent-per-layer collectives (TP, EP, CP) must stay inside the fastest fabric available. Infrequent-per-step collectives (DP) can afford slower links because they overlap with compute. PP is point-to-point — it can span nodes but introduces serialisation latency, which hurts serving more than training.

---

### DP → ZeRO → FSDP → HSDP
{: #dp-fsdp-hsdp}

These form a progression of increasingly aggressive memory sharding, each adding more communication in exchange for less memory per GPU.

**DDP (baseline).** Full model replicated on every GPU. AllReduce gradients once per step — `P · b_g` bytes total, overlapped with backprop via gradient bucketing.

```
Memory per GPU ≈ P · (b_θ + b_g + b_m,v + b_master) ≈ 16 bytes/param (AdamW mixed precision)
```

DDP is the simplest and fastest when the model fits. It breaks only on capacity (OOM).

**ZeRO stages** shard the replicated state, cutting memory by N_dp without changing compute:

| Stage | What's sharded | Memory reduction | New communication |
|---|---|---|---|
| ZeRO-1 | Optimizer state (m, v) | ~4× | None beyond DDP |
| ZeRO-2 | + Gradients | ~8× | ReduceScatter grads during backward |
| ZeRO-3 / FSDP | + Parameters | ~N_dp× | AllGather params (fwd+bwd) + ReduceScatter grads |

**FSDP per-layer loop:**

<div class="post-flow" role="group" aria-label="FSDP per-layer execution">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">AllGather: reconstruct full layer weights from shards on all ranks</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Compute forward pass with full weights; optionally reshard immediately</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Backward: AllGather again; compute local gradients</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">ReduceScatter: each GPU accumulates only its gradient shard</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Optimizer: update local weight shard using local optimizer state shard</span></li>
  </ol>
</div>

FSDP moves communication from "once per step" (DDP) to "many times per step" (per layer). This is fine in training when prefetch overlapping is enabled — AllGather for layer ℓ+1 runs while layer ℓ is computing. It is dangerous for serving latency: the per-layer AllGather sits on the decode critical path.

**Three FSDP performance knobs:**
1. **Sharding strategy**: ZeRO-2-like (shard gradients + optimizer only) vs ZeRO-3-like (shard parameters too)
2. **Prefetch and overlap**: AllGather for the next layer must be launched while the current layer runs — if not overlapped it becomes visible latency
3. **Wrapping granularity**: too-small units → too many collectives (α tax); too-large units → memory spikes and poor concurrency

**HSDP (Hierarchical Sharding Data Parallelism)** applies sharding in two levels matched to topology:
- Inner group: shard+reduce within node on NVLink (fast, low α)
- Outer group: reduce aggregated results across nodes on InfiniBand (slower, but fewer operations)

Decision rubric: DDP if it fits → FSDP if OOM → HSDP if multi-node with FSDP showing slow cross-node AllGathers.

---

### Tensor Parallelism In Depth
{: #tp-advanced}

TP shards individual weight matrices across N_tp ranks. Every linear layer `Y = XW` has two natural split axes:

**Column-parallel (shard output features, F/p per rank):**
- Each rank holds `W_i` (columns F/p to F/p+1)
- Local GEMM: `Y_i = X · W_i` — X is replicated, Y is sharded
- No AllReduce during forward; optional AllGather if next op needs full Y
- Backward: ReduceScatter on X gradient

**Row-parallel (shard input features, D/p per rank):**
- Each rank holds `W_i` (rows D/p to D/p+1) and matching input shard `X_i`
- Local GEMM: `Ỹ_i = X_i · W_i` — partial results; AllReduce to sum → full Y
- AllReduce payload: `B_tok · D · b` (smaller than the intermediate F-sized activations)

**Megatron strategy — synchronise where the tensor is small:**

```
Expand:   D → F (≈4D):  Column-parallel  →  Y stays sharded at size B_tok × F
Contract: F → D:        Row-parallel     →  AllReduce on smaller B_tok × D tensor
```

Keep the biggest intermediate activations sharded; put the unavoidable AllReduce at the narrowest boundary tensor.

**Inside a decoder block:**
- `W_QKV`: column-parallel (shard heads; head independence makes this clean)
- `W_O`: row-parallel (reduce at residual, size D)
- `W_up, W_gate`: column-parallel (D→F; big intermediate stays sharded)
- `W_down`: row-parallel (F→D; reduce at residual)

Result: **two AllReduces per transformer block** — one at attention output, one at MLP output.

**Training vs serving TP size trade-off:**

| View | Effect of larger N_tp |
|---|---|
| Training | Reduces per-rank weight memory; helps fit large microbatches; collectives often tolerable on NVLink |
| Serving (decode) | More participants in per-layer collectives; for small B_tok, α dominates → TPOT grows |

Rule: use the smallest N_tp that fits weights per shard while meeting latency targets. Keep TP within one NVSwitch domain.

**TP failure modes:**

| Symptom | Likely cause | Fix |
|---|---|---|
| TPOT grows linearly with N_tp | α-dominated collectives on decode (small messages) | Keep TP within node; reduce N_tp; add replicas |
| Low throughput despite TP | Microbatches too small; kernels inefficient | Increase B_tok via DP or accumulation |
| Cross-node TP is terrible | Per-layer collectives crossing slow InfiniBand | Move TP intra-node; use PP across nodes |
| OOM in activations | Replicated tensors or AllGather too costly | Enable SP/CP; use ReduceScatter to keep activations sharded |

---

### Pipeline Parallelism In Depth
{: #pp-advanced}

PP assigns consecutive transformer blocks to ordered "stages." Each stage lives on one (or a few) GPUs. Activations are passed forward via SendRecv at stage boundaries; activation gradients pass backward similarly.

**The bubble problem.** With N_pp stages and naïve execution only one stage is active at a time — (N_pp − 1)/N_pp of cycles are wasted. Microbatching fills the pipeline:

```
Bubble fraction ≈ (N_pp − 1) / (M + N_pp − 1)
```

With N_pp=8, M=32 microbatches: bubble fraction ≈ 18%.

**Schedule comparison:**

| Schedule | Bubble fraction | Activation memory | Comm per step |
|---|---|---|---|
| GPipe (fill-drain) | (N_pp−1)/M | O(M × microbatch) — stores all | 1 SendRecv/boundary |
| 1F1B | (N_pp−1)/M | O(N_pp × microbatch) — much less | 1 SendRecv/boundary |
| Interleaved 1F1B | (N_pp−1)/(v·M) | O(N_pp × microbatch) | v× more SendRecvs |

1F1B interleaves one forward and one backward in steady state — same bubble as GPipe but ~4× less activation memory at equal microbatch count. Interleaved 1F1B divides each stage into v sub-stages, reducing the bubble by v× at the cost of v× more point-to-point messages.

**PP boundary payload:**
```
bytes_send ≈ B_tok,µ · D · b_a   (per boundary, per microbatch)
```

Always cut at transformer-block boundaries where activations are `B_tok × D`. Cutting inside an MLP (where intermediate size is `B_tok × F ≈ 4·B_tok·D`) quadruples boundary traffic.

**Stage balance is as important as bubble reduction.** If one stage has 20% more compute than others, it becomes the straggler and everyone waits regardless of schedule. Profile per-stage compute time and rebalance layer assignments if needed.

**PP checklist:**
1. **Fit**: pick N_pp so stage-local weights + activations fit
2. **Cut**: at transformer-block boundaries (B_tok × D payload)
3. **Balance**: profile stages; avoid expensive layers clustering in one stage
4. **Fill**: choose (B_µ, M) and schedule (GPipe/1F1B/interleaved) to reduce bubbles

**PP in serving.** PP introduces stage-to-stage serialisation on the decode critical path — every token must pass through all N_pp stages sequentially. This directly increases TPOT. Use PP in serving only when required for capacity, and compensate with replica-level parallelism and aggressive batching.

> **Interview question:** You are deploying a 530B model on 128 H100s (8 per node, 16 nodes). Describe the parallelism strategy, the topology mapping, and what failure modes to watch.
>
> *Strategy: 3D parallelism. TP=8 within each node (NVLink; handles per-layer collectives at full NVSwitch bandwidth). PP=8 across 8 consecutive nodes (1F1B schedule; each stage = 2 nodes × 8 GPUs = 16 GPUs of TP). DP=2 across the remaining 2 pipeline replicas (AllReduce gradients once per step, overlapped with backprop). Total: TP×PP×DP = 8×8×2 = 128 GPUs. Memory per GPU: 530B params × 2 bytes/param BF16 = 1060 GB ÷ 128 GPUs ≈ 8.3 GB/GPU for weights alone — leaves room for optimizer state and activations with ZeRO-1. Topology mapping: TP collectives stay within NVSwitch domain (< 1 µs α); PP SendRecv crosses nodes via InfiniBand (acceptable for point-to-point; ≈ 2–5 µs). DP AllReduce crosses nodes but fires only once per step and overlaps with backprop. Failure modes to watch: (1) PP stage imbalance — if some stages have more compute (e.g., first/last have embedding layers), they become stragglers. Fix: rebalance layer assignments, profile per-stage time. (2) Pipeline bubble at M=16 microbatches — bubble ≈ (8−1)/(16+8−1) ≈ 30%. Increase M or switch to interleaved 1F1B. (3) DP AllReduce bandwidth — 530B params × 2 bytes = 1060 GB of gradients; ring AllReduce ≈ 2×1060 GB / (16 nodes × 25 GB/s IB NIC) ≈ 5.3 seconds if sequential. Must overlap with backprop using gradient bucketing; verify overlap in profiler.*

---

### Context & Sequence Parallelism (CP/SP)
{: #cp-sp}

At very long sequence lengths (L = 100K+), two problems arise even with TP and PP:
1. **Activations** scale with `B · L` — a single layer's activation tensor may exceed GPU memory
2. **Attention FLOPs** scale as `L²` — the batch cannot grow to amortise the cost because memory is already full

The solution: shard the **sequence dimension** across ranks.

**SP vs CP (Megatron terminology):**

| | Sequence Parallelism (SP) | Context Parallelism (CP) |
|---|---|---|
| Scope | Targeted: shard selected activations along L to reduce redundancy in TP composition | Full: shard all inputs and activations along L |
| What's embarrassingly parallel | Linear layers, MLPs, norms — all token-wise ops split cleanly | Same — most ops are token-wise |
| What's hard | Attention requires cross-rank K/V access | Attention requires cross-rank K/V access |
| Mental model | A trick to make TP cheaper; reduces activation replication | Full sequence-dimension model parallelism |

**CP attention exchange — two patterns:**

For CP group size N_cp, each rank owns a block of L/N_cp tokens.

*Pattern A: Ring exchange (blockwise attention)*
```
1. Keep local Q on each rank
2. Circulate K/V blocks around the ring (N_cp − 1 steps)
3. Accumulate partial attention outputs with online softmax scaling
```
Bandwidth-efficient; exposes α per ring step. Total bytes per attention layer:
```
bytes_CP ≈ 2·B·L·K·H·b · (1 − 1/N_cp)    [near full KV exchange for large N_cp]
```

*Pattern B: AllGather K/V*
```
AllGather K/V so each rank has the full sequence and can compute attention locally.
Memory cost: full K/V per rank — often infeasible at extreme L.
```

**Training vs serving CP:**
- **Training**: CP introduced because B can't grow (activation memory full); shard L instead to keep GPUs busy
- **Serving**: CP introduced because KV cache and/or prefill at extreme L don't fit on one device; different prefill and decode groups may use different CP configurations

**The caveat:** CP is kernel- and layout-sensitive. The ring exchange pattern must match the attention kernel's tile structure and the KV cache format. An incompatible layout adds index shuffling overhead that erases the bandwidth benefit.

> **Interview question:** You need to run prefill for a 128K-token prompt on a model where the KV cache at that length would exceed single-GPU memory even with GQA. You have 4 GPUs. How do you structure the parallelism?
>
> *Use CP=4 for prefill. Shard the 128K prompt into 4 blocks of 32K tokens, one per GPU. Linear layers and MLPs are embarrassingly parallel — each GPU processes its own 32K tokens independently. Attention requires K/V exchange: use ring pattern (3 hops for 4 GPUs) to circulate K/V blocks while accumulating partial attention outputs with online softmax. Each GPU holds K/V for its own 32K tokens plus one incoming block at a time — memory stays proportional to L/4 = 32K. Communication cost per attention layer: ≈ 2·B·32K·K·H·b·3 per rank (3 ring steps). After prefill completes, KV cache for the full 128K is distributed across 4 GPUs (each holds its shard). For decode: continue with CP=4 (ring attention on the cached K/V), or disaggregate — transfer the sharded KV to a dedicated decode group (possibly with smaller CP). Watch for: ring step latency at decode (batch=1 makes each step α-dominated); consider reducing CP for decode and using TP instead within the decode group.*

---

### Expert Parallelism (EP)
{: #ep}

MoE increases total parameters without increasing activated compute per token — but it forces a new distributed pattern: tokens must be physically routed to the ranks that host their selected experts.

**Dense vs MoE dispatch:**
- Dense FFN: every token touches all FFN weights via one large GEMM — embarrassingly local
- MoE FFN: each token touches K_r of N_r experts, which may live on different ranks — requires AllToAll

**MoE workflow under EP:**

<div class="post-flow" role="group" aria-label="Expert parallelism forward pass">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Router: compute top-K_r expert IDs per token (local, cheap)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Dispatch AllToAll: send token activations to expert-owning ranks</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Expert compute: each rank runs grouped GEMMs on its assigned token batch</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Combine AllToAll: return expert outputs; weighted sum with routing gates</span></li>
  </ol>
</div>

**AllToAll communication volume:**
```
bytes_MoE_comm ≈ 2 · B_tok · D · b · K_r    (dispatch + combine)
```

For D=7168, b=2 (BF16), K_r=8: ≈ 229 KB per token per MoE layer. AllToAll is many-to-many — every rank sends to every other rank — making it more contention-prone than ring AllReduce and extremely topology-sensitive.

**Load balancing — the straggler problem.**

If the router routes more tokens to expert 0 than expert 1, rank 0 takes longer, and the AllToAll combine must wait for the slowest rank:

```
utilisation ≈ E[tokens per rank] / max(tokens per rank)
```

Poor balance → low utilisation → step time set by the slowest expert rank.

**Capacity factor** bounds the worst case: each expert is allocated `capacity = ⌈cf · B_tok / N_experts⌉` slots. Tokens beyond capacity are either dropped (quality risk) or rerouted (extra compute). MoE throughput is set by worst-case load, not average.

**Bias controller** for online load balancing (DeepSeek style):
```
# After each training step:
if expert i overloaded:   b_i ← b_i − γ
if expert i underloaded:  b_i ← b_i + γ

# Selection: i ∈ S_t  iff  s_{i,t} + b_i ∈ Top-K_r(t)
```

Treats expert routing as a feedback control problem — pushes traffic away from hot experts without needing a large auxiliary loss coefficient.

**EP in serving — decode danger zone.** During decode, B_tok is tiny (often 1–8 tokens per request). AllToAll dispatch over N_ep ranks becomes α-dominated — startup latency, not bandwidth, sets the AllToAll cost. Cross-node EP is often catastrophic for TPOT: a single layer-level AllToAll over InfiniBand adds 5–20 µs of latency that compounds across all MoE layers.

Practical rule: keep EP within one NVSwitch domain when possible. If N_ep must be large, reduce it and add replicas instead.

**EP failure modes:**

| Symptom | Likely cause | Fix |
|---|---|---|
| AllToAll dominates MoE layer time | Cross-node EP; α-dominated at small batch | Keep EP within node; reduce N_ep; LatentMoE (smaller payload) |
| Expert GEMMs slow despite EP | Expert sub-batches too small for efficient GroupGEMM | Increase global batch; higher K_r; larger capacity factor |
| Straggler on one EP rank | Router imbalance; capacity overflow | Tune bias controller γ; add auxiliary balance loss |
| OOM with large N_r | All expert weights on few ranks | Increase N_ep; distribute experts more evenly |

---

### 5D Mesh Composition
{: #5d-mesh}

Real production training and serving combine all five parallelism dimensions into a process mesh:

```
N_GPUs ≈ N_dp × N_tp × N_pp × N_cp × N_ep
```

ZeRO/FSDP are policies *inside* the DP dimension (what is replicated vs sharded), not a separate mesh axis. Offloading can be viewed as a sixth knob when HBM capacity is the hard limit.

**The complete parallelism cheat sheet:**

| Dimension | Shards what | Primary win | Dominant collective | Where it goes |
|---|---|---|---|---|
| **DP** | Batch / requests | Throughput scale-out | AllReduce (train) / none (serve) | Across nodes |
| **FSDP/ZeRO** | Training state (θ, ∇θ, m,v) | Memory capacity | AllGather + ReduceScatter | Within/across nodes |
| **TP** | Per-layer weights/heads | Fit big layers; keep GEMMs large | AllReduce/AllGather per layer | **Within node** |
| **PP** | Depth (layers) | Fit deep models; scale to more nodes | SendRecv (point-to-point) | Across nodes |
| **CP/SP** | Sequence dimension | Extreme context (L > memory) | Attention ring exchange | **Within node first** |
| **EP** | Expert parameters (MoE) | Capacity at fixed active compute | AllToAll | **Within node** |

**Topology mapping rule:** Frequent-per-layer collectives (TP, EP, CP) must use the fastest available fabric. Put them within one NVSwitch domain. Infrequent-per-step collectives (DP, FSDP) can span nodes — their cost is amortised over a full optimizer step and overlapped with backprop. PP is point-to-point and can span nodes, but adds serialisation latency.

**Typical recipe for 8-GPU-per-node clusters:**

```
Within node (NVLink):   TP=8 (and EP if MoE; CP if long context)
Across nodes (IB):      PP if depth requires it
Across node groups:     DP / FSDP with hierarchical (HSDP) collectives
```

**Hot vs cold classification:**

| Collective | Hot? | Reason |
|---|---|---|
| TP AllReduce in decode | 🔥🔥 | Fires every layer, token-by-token; α dominates |
| EP AllToAll in decode | 🔥🔥 | Same; cross-node is catastrophic |
| CP ring in prefill | 🔥 | Fires every attention layer; bandwidth-heavy but amortisable |
| PP SendRecv in decode | 🔥 | On critical decode path; adds serialisation |
| DP AllReduce in training | ❄ | Once per step; overlapped with backprop |

---

### Training vs Serving Recipes
{: #parallelism-recipes}

Training and serving use the same parallelism words but optimise for opposite objectives. A strategy that maximises training throughput can actively hurt serving latency.

**Training recipe (maximise tokens/sec):**

<div class="post-flow" role="group" aria-label="Training parallelism configuration">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Capacity: choose sharding (FSDP/TP/PP/CP) so weights + grads + optimizer + activations all fit</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Throughput: make B_tok large enough for efficient GEMMs — avoid tiny microbatches</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Topology: map TP/EP/CP on fast intra-node fabrics; push DP/PP outward</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Validate: profile overlap, stage balance (PP), router imbalance (EP)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Iterate: adjust mesh sizes, wrapping granularity, bucket sizes, micro-batch count</span></li>
  </ol>
</div>

**Serving recipe (minimise TTFT and TPOT, maximise throughput):**

<div class="post-flow" role="group" aria-label="Serving parallelism configuration">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Fit: weights + KV cache + runtime buffers (TP/PP, quantisation, selective offload)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Meet SLOs: minimise critical-path collectives on decode path (TPOT)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Separate prefill vs decode: different CP groups, batching strategies, hardware pools</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Scale throughput: add replicas (serving "DP") with good load balancing and KV-cache-aware routing</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Watch tail latency: avoid cross-node TP/EP on decode; keep groups compact</span></li>
  </ol>
</div>

**Key differences training → serving:**

| Concern | Training | Serving |
|---|---|---|
| Unit of work | Optimizer step (fwd + bwd + sync + update) | Request (prefill + decode) |
| Memory dominates | θ + ∇θ + m,v + saved activations | θ + KV cache (+ runtime buffers; no optimizer) |
| Batch size | Large (efficiency) | Variable; decode is often B=1–32 |
| AllReduce timing | Once per step; overlapped with backprop | N/A (inference) |
| FSDP in serving | Often a latency risk — per-layer AllGather on decode path | Prefer TP over FSDP for serving |
| PP in serving | Tolerable bubble in training | Adds serialisation to decode critical path |

> **Interview question:** You have a 70B model to deploy for interactive chat (p99 TPOT < 50ms) and a separate batch-inference workload (throughput > 5000 tokens/sec). Can you use the same parallelism configuration for both? What changes?
>
> *No — optimal parallelism is objective-dependent. For **interactive chat** (TPOT < 50ms): optimise for decode latency. Use TP=8 within one node (NVLink; each decode step pays 2 AllReduces/layer but α ≈ 1–2 µs on NVLink, tolerable). Avoid PP (adds stage serialisation to decode path). No FSDP (per-layer AllGather on critical path). Fit the 70B model × BF16 = 140 GB on 8× H100 (80 GB each = 640 GB total — fits with room for KV). For **batch inference** (5000 tokens/sec): optimise for throughput. Use TP=8 + PP=2 to serve on 16 GPUs per replica. Large batch sizes (B=64–256) make TP collectives bandwidth-bound rather than α-dominated — large messages benefit from NVLink bandwidth. PP bubble is acceptable because you're throughput-not-latency constrained. Use continuous batching to saturate both stages. Deploy multiple replicas for linear throughput scaling. The same TP=8 configuration works for both, but PP should be added only for the batch workload. The interactive deployment should use more, smaller replicas (each TP=8, no PP) with a load balancer — this gives better tail latency and horizontal throughput scaling than a single TP+PP configuration.*

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

**[LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation)** avoids adding new layers by reparameterising the weight update instead. For a weight matrix `W ∈ ℝᵈˣᵈ`, the update `ΔW` during fine-tuning is hypothesised to lie in a low-rank subspace. LoRA factors it as:

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

LoRA reduces trainable parameters but the frozen base model weights still consume memory in fp16. **[QLoRA](https://arxiv.org/abs/2305.14314)** quantises the frozen weights to 4 bits, cutting base model memory by 4×, while keeping the LoRA adapter weights and activations in fp16 for stable training.

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

**[PagedAttention](https://arxiv.org/abs/2309.06180)** (vLLM) borrows the OS virtual memory abstraction and applies it to KV cache. Physical GPU memory is divided into fixed-size **KV blocks** (e.g. 4 tokens per block). Each request gets a **block table** — a mapping from logical block index to physical block number — rather than a contiguous reservation.

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

**[RadixAttention](https://arxiv.org/abs/2312.07104)** (SGLang) maintains a global **LRU cache of KV blocks** organised as a **radix tree** (a compact prefix tree). The key of each tree node is a token sequence; the value is the cached KV block for that sequence. When a new request arrives, the scheduler performs a longest-prefix match against the tree. Any matched prefix is loaded directly from cache, skipping the prefill computation for those tokens entirely.

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

All techniques so far improve *throughput* — more requests per second. **[Speculative decoding](https://arxiv.org/abs/2211.17192)** attacks a different problem: reducing *latency* for individual requests by exploiting the fact that LLM inference is memory-bandwidth-bound, not compute-bound.

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

