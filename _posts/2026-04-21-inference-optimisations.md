---
title: "Inference Optimisations"
date: 2026-04-21
description: "Techniques for making LLM inference fast and memory-efficient — quantisation, KV cache management, batching, speculative decoding, and hardware-aware kernel design."
tags: [ml-systems, inference, llm, optimisation]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
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
  </ul>
</nav>

---

## Overview
{: #overview}

LLM inference is fundamentally different from training. The forward pass through a 70B model must complete in tens of milliseconds to meet latency SLOs, yet the model's 140 GB of weights must be loaded from HBM on every token generated. The bottleneck is almost never arithmetic — it is **memory bandwidth**.

To see why: generating one token from Llama-2-70B on 4× A100s (combined ~8 TB/s HBM bandwidth) requires loading 140 GB of weights. At 8 TB/s, that takes 17.5 ms/token — 57 tokens/second maximum, independent of how fast the matrix multiplications run. Adding more compute does not help; only reducing the bytes that must be read per token does.

<div class="post-flow post-flow--horizontal" role="group" aria-label="Three inference bottlenecks">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Memory bandwidth — loading weights every token</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">KV cache memory — grows linearly with context × batch</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Sequential dependency — tokens generated one at a time</span></li>
  </ol>
</div>

The optimisation landscape maps directly onto these three bottlenecks: **quantisation** reduces bytes-per-weight; **KV cache management** controls memory growth; **speculative decoding** breaks sequential dependency. Kernel-level work (FlashAttention, fused operators) cuts memory traffic within a single forward pass. The sections below cover each in depth.

---

## Quantisation
{: #quantisation}

Quantisation represents weights (and optionally activations) in fewer bits, reducing both the memory footprint and the bytes-to-load per token.

### Weight Quantisation
{: #weight-quant}

The simplest approach: quantise only the model weights, leaving activations in fp16 at runtime. For a weight tensor `W`, block-wise quantisation divides it into contiguous blocks, computes a scale per block, and stores quantised integers:

```
Quantise:   W_int = round(W / scale),   scale = max(|W_block|) / (2^(b-1) - 1)
Dequantise: W_fp16 = W_int × scale
```

**INT8 weight-only quantisation (W8A16)**: weights stored as int8, dequantised to fp16 before the matmul. Memory per parameter: 1 byte vs 2 bytes in fp16 → 2× memory reduction, ~1.8× throughput gain (bandwidth-bound regime). Accuracy loss is minimal for most tasks at 8 bits.

**INT4 (W4A16)**: 4 bits/param → 4× memory reduction. Requires groupwise quantisation (separate scale per group of 128 weights) to preserve accuracy. Used in GPTQ, AWQ, and QLoRA. Typical accuracy degradation: <1% on standard benchmarks for 7B+ models.

**GPTQ** minimises quantisation error by solving a second-order optimisation per layer: given a small calibration dataset, it quantises weights one column at a time and updates remaining columns to compensate for the error introduced. This is expensive to run once but produces higher-quality quantised models than round-to-nearest.

**AWQ (Activation-aware Weight Quantisation)** observes that not all weight channels are equally important — channels corresponding to large activation magnitudes cause more quantisation error. AWQ scales those channels up before quantisation (and scales activations down correspondingly) to protect them:

```
# Per-channel scaling before quantisation
s = argmin Σ ‖(W·diag(s))_quant × X/s - WX‖²
```

AWQ achieves better accuracy than GPTQ at the same bit-width with faster quantisation (no per-column optimisation needed).

**FP8**: NVIDIA H100 and later GPUs support native FP8 (E4M3 and E5M2 formats) tensor core operations. FP8 inference runs at 2× the throughput of FP16 tensor cores without the quantisation overhead of integer formats — FP8 keeps the floating-point dynamic range and simply uses fewer mantissa bits. This is now the default for H100 inference at scale.

### Activation Quantisation
{: #activation-quant}

**W8A8 (INT8 weights + INT8 activations)** enables fully integer matrix multiplications — faster than W8A16 because the matmul itself runs in int8 rather than fp16. The challenge is that LLM activations have **outliers**: a small fraction of channels have magnitudes 100× larger than typical values, making naive quantisation inaccurate.

**LLM.int8()** handles this by splitting the matmul: identify outlier channels (>0.1% of values exceeding a threshold), compute those columns in fp16, and compute the remainder in int8. The two results are summed. This preserves accuracy with minimal throughput overhead.

**SmoothQuant** takes a different approach: migrate quantisation difficulty from activations to weights. For each channel, apply a per-channel scale `s` to activations and `1/s` to the corresponding weight column — mathematically equivalent but easier to quantise because activations are smoothed:

```
Y = (X·diag(s)⁻¹) × (diag(s)·W) = X_smooth × W_smooth
```

`s` is chosen to equalise the quantisation ranges of X and W per channel, enabling accurate W8A8 without outlier handling overhead.

### KV Cache Quantisation
{: #kv-quant}

The KV cache at long context grows to dominate GPU memory. For a 70B model with batch size 32 and context 32k, the KV cache alone can exceed 1 TB. Quantising KV cache values to int8 or fp8 halves or quarters this footprint with minimal quality degradation — keys and values are summed over attention heads, averaging out quantisation noise.

**KV int4** is more aggressive: per-channel scaling of keys and values before int4 quantisation, with dequantisation at attention computation time. This reduces KV memory by 4× at the cost of a dequantisation step per attention call.

---

## KV Cache Management
{: #kv-cache}

The KV cache stores attention keys and values for all previous tokens so decoding can attend to them without recomputation. Naively managed, it wastes GPU memory through fragmentation and is discarded on request completion even when future requests could reuse it.

### PagedAttention
{: #pagedattention}

**PagedAttention** (vLLM) applies OS virtual memory concepts to KV cache. Physical GPU memory is divided into fixed-size **KV blocks** (typically 16–32 tokens per block). Each request's KV cache is represented as a **block table** — a mapping from logical block indices to physical block numbers — rather than a contiguous reservation.

<div class="post-flow" role="group" aria-label="PagedAttention allocation">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Physical memory split into fixed KV blocks (e.g. 16 tokens/block)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each request holds a block table: logical → physical block mapping</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Blocks allocated on demand as sequence grows</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Blocks freed immediately on request completion — no wasted reservation</span></li>
  </ol>
</div>

Benefits: eliminates external fragmentation entirely; caps internal fragmentation at `block_size − 1` tokens per request. Raises effective KV memory utilisation from 20–40% (naive) to >95%. Attention computation fetches non-contiguous blocks via the block table — valid because attention is commutative over K/V positions.

**Copy-on-write for beam search**: when a request forks into multiple beams, blocks in the shared prefix are reference-counted and copied only when a beam diverges. This avoids duplicating the shared KV prefix for each beam.

### Prefix Caching
{: #prefix-caching}

Many requests share a common prefix — a system prompt, few-shot examples, or a conversation history. Standard serving discards KV cache on request completion and recomputes the prefix from scratch for every new request.

**RadixAttention** (SGLang) maintains a global LRU cache of KV blocks organised as a **radix tree** (compact prefix tree). On request arrival, a longest-prefix match is performed against the tree. Matched prefix blocks are loaded from cache; only the unmatched suffix requires fresh prefill computation.

```
Tree structure:
  [system prompt] → [example 1] → [user turn 1] → ...
                 → [example 2] → [user turn 1] → ...
```

When a cached prefix is evicted (LRU), its physical KV blocks are freed. This is especially effective for multi-turn chat (full conversation history shared), batch inference over a fixed system prompt, and agent frameworks where tool descriptions are reused across thousands of calls.

### KV Eviction
{: #kv-eviction}

For very long contexts that exceed the KV cache budget, **selective eviction** drops KV entries for tokens deemed least important. Attention score is a natural importance signal: tokens that receive low attention across recent decoding steps are unlikely to be needed again.

**H₂O (Heavy Hitter Oracle)** tracks a running estimate of cumulative attention scores per token and evicts the lowest-scoring entries when the cache is full. A small set of "heavy hitter" tokens (those that consistently receive high attention) are protected from eviction.

**StreamingLLM** takes a different approach for infinite-context streaming: always keep the first few tokens (they accumulate high attention regardless of content — the "attention sink" phenomenon) plus a sliding window of recent tokens. This enables unbounded streaming generation at fixed KV memory cost at the price of losing mid-context information.

---

## Batching Strategies
{: #batching}

### Continuous Batching
{: #continuous-batching}

**Static batching** holds a fixed set of requests until all complete before starting the next batch. Requests with short outputs leave GPU slots idle while long requests finish — GPU utilisation collapses for heterogeneous workloads.

**Continuous batching** (Orca) operates at the iteration level. After every decoding step, the scheduler evicts completed requests (those that emitted `<EOS>`) and admits new requests from the waiting pool — immediately, without waiting for the rest of the batch:

<div class="post-flow post-flow--compare" role="group" aria-label="Static vs continuous batching">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Static Batching</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Batch runs to completion before next batch starts</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Short requests leave idle slots</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Throughput degrades with output length variance</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Continuous Batching ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Completed requests replaced mid-batch</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">GPU always saturated</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">2–10× higher throughput on real workloads</span></li>
    </ol>
  </div>
</div>

**Overlapped scheduling**: the CPU scheduler (which checks stop conditions, runs prefix matching, allocates KV blocks) runs concurrently with the GPU executing the previous iteration. The scheduler processes iteration `t` results while the GPU runs iteration `t+1`, hiding the scheduling overhead that otherwise consumes >50% of wall time in naive implementations.

### Chunked Prefill
{: #chunked-prefill}

Long-context requests have expensive prefill phases (processing the entire prompt) that block the GPU from serving decode-phase requests. **Chunked prefill** breaks the prefill into fixed-size chunks and interleaves them with decode steps:

```
Iteration 1: prefill chunk 0 (512 tokens) + decode batch
Iteration 2: prefill chunk 1 (512 tokens) + decode batch
Iteration 3: prefill chunk 2 (512 tokens) + decode batch
...
```

This bounds prefill latency impact on decoding throughput at the cost of slightly higher prefill latency (spread over multiple iterations). It also improves GPU utilisation by keeping both prefill and decode work in the same kernel call — prefill is compute-bound; decode is memory-bound; mixing them on modern GPUs with sufficient parallelism can improve overall hardware efficiency.

---

## Speculative Decoding
{: #speculative}

Speculative decoding exploits the observation that LLM inference is memory-bandwidth-bound: the GPU loads 140 GB of weights to produce a single token while using only ~2% of its compute capacity. If multiple draft tokens can be **verified in one LLM forward pass**, we get multiple tokens per memory load.

### Draft-Verify
{: #draft-verify}

A **small speculative model (SSM)** generates `γ` draft tokens autoregressively — fast, because the SSM has far fewer parameters. The LLM then runs one forward pass over `[prompt + γ draft tokens]`, producing output distributions at each draft position simultaneously.

**Speculative sampling** verifies each draft token `xᵢ` against the LLM's distribution:
- If `p_SSM(xᵢ) ≤ p_LLM(xᵢ)`: accept unconditionally
- If `p_SSM(xᵢ) > p_LLM(xᵢ)`: accept with probability `p_LLM(xᵢ) / p_SSM(xᵢ)`
- On rejection: sample from `norm(max(0, p_LLM − p_SSM))` and stop

This guarantees the output distribution is **identical** to sampling from the LLM alone — speculative decoding is lossless.

**SpecInfer** extends this with **tree-based speculation**: multiple SSMs each generate a draft sequence, merged into a token tree. The LLM verifies the entire tree in one forward pass using tree attention — a topology-aware causal mask that runs all tree branches in a single kernel. Result: 1.3–2.4× speedup over standard inference.

**EAGLE** tightens the SSM design: the draft model reuses the LLM's embedding and language model head, adding only a small trainable attention layer. It expands `K` candidate tokens per step and rerankds them by accumulated path likelihood before verification — a dynamic tree that adapts per prompt.

**Medusa** adds multiple auxiliary decoding heads to the LLM itself — no separate model. Each head predicts a token at offset 1, 2, 3, ... from the current position. The main head and Medusa heads run in parallel; verified suffixes extend the output by multiple tokens per step.

### Model-Free Methods
{: #model-free}

**Prompt lookup decoding**: search the input prompt for the current context suffix; if found, copy the following tokens as draft. Requires no extra model — works by exploiting repetition between prompt and output (common in summarisation, RAG, code editing).

**SuffixDecoding** generalises with a two-tier suffix tree: a **request tree** indexes tokens generated so far in the current request; a **global tree** indexes outputs across all prior requests. Draft candidates are assembled from both trees, scored by accumulated likelihood, and verified by the LLM in one pass.

**Lookahead decoding** runs two branches in parallel each step:
- *Lookahead branch*: generates tokens at positions ahead of the current head
- *Verification branch*: checks n-grams from a pool of previously generated lookahead tokens

Both branches are batched into a single LLM forward pass. Verified n-grams that extend the output are accepted, breaking the sequential dependency without any draft model.

---

## Kernel-Level Optimisations
{: #kernel-opt}

### FlashAttention
{: #flashattention}

Standard attention materialises the full `N×N` score matrix in HBM before computing the output — an `O(N²)` memory cost that becomes 2+ GB for a single head at context length 32k.

**FlashAttention** avoids this by computing attention in tiles that fit in on-chip SRAM, never writing the full score matrix to HBM:

<div class="post-flow" role="group" aria-label="FlashAttention tiling">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Load Q, K, V block-by-block from HBM → on-chip SRAM</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Compute partial softmax using online normalisation (no full row needed)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Accumulate output O in registers; rescale running normaliser</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Write only final O to HBM — no intermediate N×N matrix</span></li>
  </ol>
</div>

| | Standard Attention | FlashAttention |
|---|---|---|
| HBM reads/writes | 40.3 GB | 4.4 GB |
| Runtime (A100) | 41.7 ms | 7.3 ms |
| Memory scaling | O(N²) | O(N) |

The backward pass recomputes the attention matrix from stored softmax statistics (size O(N), not O(N²)) rather than materialising and storing it. FlashAttention-2 improves warp-level parallelism; FlashAttention-3 adds async TMA pipelining for H100.

### Flash-Decoding
{: #flash-decoding}

FlashAttention parallelises across the query dimension. During autoregressive decoding with a single query token, there is no query parallelism — the kernel must scan all `N` keys and values sequentially. At long contexts (32k+ tokens), this is the new bottleneck.

**Flash-Decoding** parallelises across the K/V dimension instead: split the K/V sequence into chunks, assign each chunk to a separate thread block, compute partial attention outputs with local softmax normalisation, then reduce the partial outputs across all splits:

<div class="post-flow" role="group" aria-label="Flash-Decoding parallelism">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Split K/V into P chunks — one thread block per chunk</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each block computes partial output Oᵢ with local softmax normaliser (lseᵢ)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Reduce: O = Σ Oᵢ · exp(lseᵢ − lse_global) / Σ exp(lseᵢ − lse_global)</span></li>
  </ol>
</div>

The reduction is valid because attention is associative over K/V splits — the same identity used in online softmax. Result: up to 8× faster decoding at 32k+ context compared to standard FlashAttention.

### Kernel Fusion
{: #kernel-fusion}

Each separate GPU kernel launch has overhead: memory reads and writes to/from HBM, kernel launch latency. **Operator fusion** merges adjacent operations into a single kernel, keeping intermediate values in registers or shared memory.

Common fusions in LLM inference:

| Unfused | Fused | Benefit |
|---|---|---|
| RMSNorm + linear | FusedRMSNormLinear | Eliminates intermediate HBM write |
| Linear + activation | FusedLinearSiLU | Activation computed in-register |
| Rotary embedding + QKV split | FusedRoPEQKV | One pass over input activations |
| Attention + residual + norm | FusedAttentionResNorm | Avoids storing attention output to HBM |

**Quantised kernel fusion**: fuse weight dequantisation into the matmul — dequantise each weight element in-register inside the matmul loop rather than writing a decoded fp16 weight matrix to HBM first (as covered in the ML Compilation section). This is the dominant technique in W4A16 inference kernels (GPTQ, AWQ runtimes).

**Triton** enables custom fused kernels in Python without writing CUDA directly: express the tile loop structure, and Triton handles memory coalescing, register allocation, and warp scheduling. Most modern inference frameworks (vLLM, SGLang) implement their custom kernels in Triton.

---

## Prefill-Decode Disaggregation
{: #disaggregation}

**Prefill** (processing the input prompt) and **decode** (generating output tokens one by one) have fundamentally different resource profiles:

| | Prefill | Decode |
|---|---|---|
| Compute pattern | Processes all prompt tokens in parallel | One token at a time |
| Bottleneck | Compute (matmuls over long sequences) | Memory bandwidth (load weights per token) |
| Latency sensitivity | Moderately sensitive (TTFT) | Highly sensitive (TBT) |
| Batch efficiency | Benefits from long sequences | Benefits from large batch size |

Running both on the same hardware forces compromises. **Prefill-decode disaggregation** assigns them to separate GPU pools:

<div class="post-flow post-flow--compare" role="group" aria-label="Prefill vs decode GPU pools">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Prefill GPUs</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Optimised for high compute utilisation</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Large batches of long prompts</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Transfer KV cache to decode pool after prefill</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Decode GPUs</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Optimised for memory bandwidth</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Large decode batch — amortise weight loads</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">No prefill interruptions — stable decode latency</span></li>
    </ol>
  </div>
</div>

The KV cache transfer between prefill and decode pools is the key engineering challenge — it requires high-bandwidth inter-GPU links (NVLink within a node, RDMA across nodes) and careful scheduling to avoid stalling the decode pool while waiting for KV transfers.

**DistServe** and **Splitwise** are production systems that implement disaggregation. They report 2–3× improvements in goodput (requests served per second meeting latency SLOs) over co-located serving by allowing each pool to be independently scaled and optimised. The tradeoff is operational complexity: two fleets to manage, KV transfer infrastructure to maintain, and routing logic to balance load across the split.
