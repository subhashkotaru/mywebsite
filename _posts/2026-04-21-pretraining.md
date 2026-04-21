---
title: "Pretraining"
date: 2026-04-21
description: "How large language models are pretrained — data pipelines, tokenisation, training objectives, scaling laws, and the systems that make it feasible."
tags: [ml-systems, pretraining, llm, scaling]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#data">Data Pipeline</a>
      <ul class="post-toc-sublist">
        <li><a href="#data-sources">Sources & Scale</a></li>
        <li><a href="#data-quality">Quality Filtering</a></li>
        <li><a href="#tokenisation">Tokenisation</a></li>
      </ul>
    </li>
    <li><a href="#objective">Training Objective</a>
      <ul class="post-toc-sublist">
        <li><a href="#clm">Causal Language Modelling</a></li>
        <li><a href="#mlm">Masked Language Modelling</a></li>
      </ul>
    </li>
    <li><a href="#architecture">Model Architecture</a>
      <ul class="post-toc-sublist">
        <li><a href="#transformer-block">Transformer Block</a></li>
        <li><a href="#positional-encoding">Positional Encoding</a></li>
        <li><a href="#normalisation">Normalisation</a></li>
      </ul>
    </li>
    <li><a href="#optimiser">Optimiser & Training Stability</a>
      <ul class="post-toc-sublist">
        <li><a href="#adam">AdamW</a></li>
        <li><a href="#lr-schedule">Learning Rate Schedule</a></li>
        <li><a href="#grad-clipping">Gradient Clipping</a></li>
      </ul>
    </li>
    <li><a href="#scaling-laws">Scaling Laws</a></li>
    <li><a href="#systems">Systems for Pretraining at Scale</a></li>
  </ul>
</nav>

---

## Overview
{: #overview}

**Pretraining** is the first and most compute-intensive phase of building a large language model. The model is initialised randomly and trained on a massive corpus of text — typically trillions of tokens — to predict the next token (or masked tokens) in a sequence. No labels are required beyond the text itself; the supervision signal comes from the data.

The result is a **foundation model**: a general-purpose representation of language that has absorbed factual knowledge, grammatical structure, reasoning patterns, and stylistic variation from the training corpus. Downstream fine-tuning then specialises this foundation at a fraction of the pretraining cost.

<div class="post-flow" role="group" aria-label="Pretraining pipeline">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Raw text corpus — web, books, code, academic papers</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Data pipeline — filter, deduplicate, tokenise</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Model training — billions of gradient steps across thousands of GPUs</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Foundation model checkpoint — ready for fine-tuning</span></li>
  </ol>
</div>

What makes pretraining expensive: GPT-3 (175B parameters) required ~3.14 × 10²³ FLOPs to train — roughly 355 GPU-years on V100s. The cost is dominated by the sheer volume of matrix multiplications in the transformer's attention and FFN layers, performed over 300B tokens of text.

---

## Data Pipeline
{: #data}

### Sources & Scale
{: #data-sources}

Modern LLMs are trained on diverse, multi-source corpora assembled from:

| Source | Examples | Characteristics |
|---|---|---|
| Web crawl | Common Crawl, C4 | Massive scale, noisy quality |
| Books | BookCorpus, Gutenberg | Long-range coherence, formal writing |
| Code | GitHub, StackOverflow | Structured, reasoning-dense |
| Academic | ArXiv, PubMed | Technical depth, precise language |
| Curated | Wikipedia, news | High quality, broad coverage |

LLaMA-3.1 was trained on 15 trillion tokens; GPT-3 on 300B. The raw Common Crawl snapshot (monthly web crawl) runs to petabytes — a substantial engineering effort is required just to ingest and process it.

**Domain mixing**: the ratio of training tokens from each source matters. Code improves reasoning; books improve coherence; web text improves coverage. Llama-2 used 89.7% web + books, 8% code, 2.5% Wikipedia. Getting this mix right is as important as data volume.

### Quality Filtering
{: #data-quality}

Raw web crawl data contains spam, duplicate pages, low-quality boilerplate, and toxic content. A multi-stage filtering pipeline is applied:

<div class="post-flow" role="group" aria-label="Data quality filtering pipeline">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Language detection — keep target languages, discard others</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Heuristic filters — min/max length, punctuation ratio, repeated n-grams</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Deduplication — exact and near-duplicate removal (MinHash, suffix arrays)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Quality scoring — classifier trained on curated vs web text</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Toxicity filtering — remove harmful content with classifier or keyword lists</span></li>
  </ol>
</div>

**Deduplication** deserves emphasis: the web contains enormous amounts of near-duplicate text (mirrored articles, scraped pages). Training on duplicates causes the model to memorise specific passages rather than learn general patterns — and degrades benchmark performance measurably. MinHash LSH finds near-duplicates at scale in sub-quadratic time.

### Tokenisation
{: #tokenisation}

Text is converted to integers before feeding to the model. The tokeniser defines the vocabulary of atomic units the model works with.

**Byte-Pair Encoding (BPE)** is the dominant approach. It starts with individual bytes (256 symbols) and iteratively merges the most frequent adjacent pair into a new token:

```
Iteration 0: ["l", "o", "w", "e", "r"]
Iteration 1: ["lo", "w", "e", "r"]      # "l"+"o" merged (most frequent)
Iteration 2: ["lo", "we", "r"]          # "w"+"e" merged
...
Final vocab: 32k–128k tokens
```

BPE finds a good compression of the training corpus — common words become single tokens, rare words decompose into subword pieces. GPT-4 uses `cl100k_base` with 100,277 tokens; Llama-3 uses 128k tokens.

**Tokenisation effects on model behaviour**: tokenisation is not linguistically neutral. Numbers are often split into individual digits (`"2024"` → `["20", "24"]`), making arithmetic hard. Non-English scripts with larger character sets get tokenised at fewer characters per token, inflating their effective sequence length and raising inference cost.

---

## Training Objective
{: #objective}

### Causal Language Modelling
{: #clm}

**Causal language modelling (CLM)** — also called next-token prediction — is the pretraining objective for decoder-only models (GPT, Llama, Mistral). Given a sequence `x₁, x₂, ..., xₜ`, the model is trained to minimise the negative log-likelihood of each token given its prefix:

```
L = -Σₜ log P(xₜ | x₁, ..., xₜ₋₁)
```

The training data is just text — no labels needed. Every token in the sequence is simultaneously a training input and a prediction target (shifted by one). This makes CLM extremely data-efficient: a 1024-token sequence yields 1024 prediction targets.

**Causal masking**: the attention mask ensures position `i` can only attend to positions `≤ i`. This triangular mask is implemented efficiently in FlashAttention and is what makes decoder-only models autoregressive — each position is predicted without seeing future context.

### Masked Language Modelling
{: #mlm}

**Masked language modelling (MLM)** — used by BERT and encoder-only models — randomly masks 15% of input tokens and trains the model to predict the masked values. Unlike CLM, the model sees the full bidirectional context around each mask:

```
Input:  "The cat [MASK] on the mat"
Target:        "sat"
```

MLM produces better representations for classification and extraction tasks (where seeing full context matters) but cannot generate text autoregressively — the model needs to see the complete sequence to fill in masks. For generative tasks, CLM is preferred.

---

## Model Architecture
{: #architecture}

All modern LLMs are built on the **transformer** architecture. The specific design choices made during pretraining become fixed — they cannot be changed without retraining from scratch.

### Transformer Block
{: #transformer-block}

Each transformer block applies two sub-layers with residual connections:

```
h = h + SelfAttention(LayerNorm(h))
h = h + FFN(LayerNorm(h))
```

**Self-attention** computes pairwise token relevance: `A(Q,K,V) = softmax(QKᵀ/√d)·V`. In multi-head attention, this is run H times in parallel with different learned projections, then concatenated. The `O(L²)` cost in sequence length L is the main scaling bottleneck.

**Feed-forward network**: each FFN is two linear layers with a non-linearity:
- GPT-style: `FFN(x) = ReLU(xW₁)W₂`
- Modern (LLaMA, Mistral): `FFN(x) = (SiLU(xW₁) ⊙ xW_gate)W₂` — SwiGLU gating, 8/3× wider hidden dim

The FFN contains ~2/3 of a transformer's parameters and is where MoE layers are inserted in sparse models.

### Positional Encoding
{: #positional-encoding}

Transformers have no built-in notion of sequence order — positional encodings provide it.

<div class="post-flow post-flow--compare" role="group" aria-label="Absolute vs rotary positional encoding">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Absolute (sinusoidal / learned)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Fixed vector added to each token embedding</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Position is absolute — generalises poorly beyond training length</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Used in original Transformer, BERT, GPT-2</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">RoPE (Rotary) ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Rotates Q, K vectors by position-dependent angle</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Dot product Q·K encodes relative position automatically</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Extrapolates to longer contexts — used in LLaMA, Mistral, GPT-4</span></li>
    </ol>
  </div>
</div>

**RoPE** applies a rotation matrix `R(θ·m)` to query and key vectors at position `m`. Because `R(θ·m)ᵀ R(θ·n) = R(θ·(m−n))`, the attention score automatically encodes the *relative* distance `m−n` rather than absolute positions. This gives better generalisation beyond the training context length and is now the standard for decoder-only LLMs.

### Normalisation
{: #normalisation}

Normalisation stabilises training by preventing activations from exploding or vanishing. The two choices are:

**LayerNorm** (original transformer): normalise across the feature dimension per token. Applied post-attention and post-FFN in BERT-style models.

**RMSNorm** (modern LLMs): normalise by root mean square only, dropping the mean-centering step:

```
RMSNorm(x) = x / RMS(x) · γ,   RMS(x) = √(1/d · Σ xᵢ²)
```

RMSNorm is ~15% faster than LayerNorm (no mean computation) and works as well empirically. LLaMA, Mistral, and most recent models use **pre-RMSNorm** (applied before attention/FFN, not after) — more stable for deep models.

---

## Optimiser & Training Stability
{: #optimiser}

### AdamW
{: #adam}

LLM pretraining universally uses **AdamW** — Adam with decoupled weight decay. For each parameter `θ`:

```
m = β₁·m + (1−β₁)·g          # first moment (momentum)
v = β₂·v + (1−β₂)·g²         # second moment (variance)
m̂ = m / (1−β₁ᵗ)             # bias correction
v̂ = v / (1−β₂ᵗ)
θ = θ − η · m̂/(√v̂ + ε) − η·λ·θ   # update + weight decay
```

Typical hyperparameters: `β₁=0.9, β₂=0.95, ε=1e-8, λ=0.1`. The `β₂=0.95` (vs Adam's default 0.999) makes the variance estimate adapt faster — important for the non-stationary gradient distributions seen in LLM training.

**Memory cost**: AdamW stores `m` and `v` in fp32, doubling the memory footprint vs storing weights alone. For a 70B model: 70B × 8 bytes = 560 GB just for optimizer state. ZeRO Stage 1 partitions this across GPUs; FSDP partitions weights, gradients, and optimizer state together.

### Learning Rate Schedule
{: #lr-schedule}

LLM training uses a **warmup + cosine decay** schedule:

<div class="post-flow" role="group" aria-label="Learning rate schedule phases">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Linear warmup — ramp from 0 to η_max over ~2000 steps</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Cosine decay — anneal from η_max to η_min over remaining steps</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Final LR: η_min ≈ η_max / 10 (e.g. 1e-5 → 1e-6)</span></li>
  </ol>
</div>

Warmup prevents early catastrophic updates before the optimizer has accumulated reliable gradient statistics. Cosine decay provides smooth annealing that avoids the sharp loss spike of step-decay schedules. Some recent runs (MiniCPM, Llama-3) use a **multi-stage** schedule: train most steps at high LR, then apply a rapid cooldown on high-quality data — improving final token efficiency.

### Gradient Clipping
{: #grad-clipping}

Gradient explosions in deep transformers can destabilise training irreversibly. **Global gradient norm clipping** rescales the entire gradient vector if its L2 norm exceeds a threshold:

```
if ‖g‖ > max_norm:
    g = g × max_norm / ‖g‖
```

`max_norm = 1.0` is the standard. This is computed across all parameters jointly — clipping per-layer is less effective because it distorts the relative gradient magnitudes between layers.

**Loss spikes** still occur even with clipping — sudden jumps of 0.1–0.5 in training loss followed by recovery are common in long runs. The common response is to roll back to the checkpoint before the spike and resume with a temporarily reduced learning rate.

---

## Scaling Laws
{: #scaling-laws}

Scaling laws (Kaplan et al. 2020; Hoffmann et al. 2022 — "Chinchilla") describe how model performance varies with compute budget, model size, and training tokens. The key finding from Chinchilla:

> **For a fixed compute budget `C` (in FLOPs), the optimal model size `N` and training tokens `D` satisfy `N ∝ D ∝ √C` — model size and token count should scale equally.**

This overturned the prior practice of training very large models on relatively few tokens. GPT-3 (175B params, 300B tokens) is undertrained by Chinchilla standards — a 70B model trained on 1.4T tokens achieves comparable loss at 4× less inference cost.

**Compute-optimal frontier**:

| Model | Params | Tokens | Compute (FLOPs) | Chinchilla-optimal? |
|---|---|---|---|---|
| GPT-3 | 175B | 300B | 3.1 × 10²³ | No — undertrained |
| Chinchilla | 70B | 1.4T | 5.8 × 10²³ | Yes |
| LLaMA-3.1 | 405B | 15T | — | Over-trained (inference budget) |

**Inference-adjusted scaling**: Chinchilla optimises for loss at training time. In practice, inference is often the bottleneck — serving a 70B model is cheaper than 175B, so it is worth training the smaller model on more tokens even past the compute-optimal point. LLaMA and Mistral explicitly target this: train smaller models for longer to maximise accuracy-per-inference-dollar.

**Emergent capabilities**: scaling laws predict average loss smoothly, but individual capabilities emerge non-smoothly at certain scale thresholds — arithmetic, multi-step reasoning, and in-context learning appear at model sizes that cannot be predicted from smaller-scale extrapolation alone.

---

## Systems for Pretraining at Scale
{: #systems}

Pretraining a frontier model requires coordinating thousands of GPUs for months. The core systems challenges:

<div class="post-flow post-flow--horizontal" role="group" aria-label="Three parallelism dimensions">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Data parallelism — replicate model, shard data</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Tensor parallelism — shard layers within a node</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Pipeline parallelism — shard layers across nodes</span></li>
  </ol>
</div>

**Data parallelism + ZeRO**: replicate the model across N GPUs, each processing a different mini-batch. ZeRO Stage 3 shards parameters, gradients, and optimizer state across all replicas — memory per GPU drops from `20M` to `20M/N` bytes (where `M` is parameter count). AllReduce communication cost is `2M` bytes regardless of N with ring AllReduce.

**Tensor parallelism (Megatron-LM)**: split each weight matrix across GPUs within a node, exploiting fast NVLink. Each GPU computes a column slice of each linear layer; one AllReduce per transformer sub-layer synchronises partial results. Scales to 8–16 GPUs per node.

**Pipeline parallelism**: assign consecutive transformer layers to different nodes. Micro-batching and 1F1B scheduling keep the pipeline busy — bubble fraction `(p−1)/m` where `p` is pipeline stages and `m` is micro-batches. Communication is the activation tensor between stages, much smaller than AllReduce.

**Fault tolerance**: at 10,000-GPU scale, hardware failures during a multi-week training run are near-certain. Checkpointing every 1–4 hours to distributed storage allows recovery. Some systems use **asynchronous checkpointing** (write to CPU memory, flush to storage in background) to avoid the GPU idle time during checkpoint writes.

**Numerical stability**: bf16 is the default for LLM pretraining — same dynamic range as fp32 avoids overflow, simpler than fp16 loss scaling. Master weights and optimizer state stay in fp32. Loss spikes are monitored with automated alerting; anomalous gradient norms trigger checkpoint rollback.
