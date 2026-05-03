---
title: "LLM Inference Systems"
date: 2026-04-30
display_order: 9
description: "How LLM inference actually works — prefill vs decode, KV cache memory mechanics, continuous batching, the throughput-latency tradeoff, and a deep dive into vLLM internals and serving framework tradeoffs."
tags: [ml-systems, inference, llm]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#prefill-decode">Prefill vs Decode</a>
      <ul class="post-toc-sublist">
        <li><a href="#what-happens">What Happens in Each Phase</a></li>
        <li><a href="#compute-vs-memory">Compute-Bound vs Memory-Bandwidth-Bound</a></li>
        <li><a href="#ttft-tpot">TTFT vs TPOT</a></li>
        <li><a href="#why-it-matters">Why the Split Matters for Systems</a></li>
      </ul>
    </li>
    <li><a href="#kv-cache">KV Cache</a>
      <ul class="post-toc-sublist">
        <li><a href="#what-is-kv">What Gets Cached and Why</a></li>
        <li><a href="#kv-growth">Memory Growth</a></li>
        <li><a href="#kv-fragmentation">Fragmentation and PagedAttention</a></li>
        <li><a href="#prefix-caching">Prefix Caching</a></li>
      </ul>
    </li>
    <li><a href="#continuous-batching">Continuous Batching</a>
      <ul class="post-toc-sublist">
        <li><a href="#static-batching-problem">Why Static Batching Fails</a></li>
        <li><a href="#iteration-level">Iteration-Level Scheduling</a></li>
        <li><a href="#chunked-prefill">Chunked Prefill</a></li>
      </ul>
    </li>
    <li><a href="#throughput-latency">Throughput vs Latency</a>
      <ul class="post-toc-sublist">
        <li><a href="#the-tradeoff">The Core Tradeoff</a></li>
        <li><a href="#interactive-vs-batch">Interactive vs Batch Workloads</a></li>
        <li><a href="#pd-disaggregation">Prefill-Decode Disaggregation</a></li>
      </ul>
    </li>
    <li><a href="#vllm-internals">vLLM Internals</a>
      <ul class="post-toc-sublist">
        <li><a href="#vllm-pagedattn">PagedAttention Implementation</a></li>
        <li><a href="#vllm-scheduler">The Scheduler: Queues and Preemption</a></li>
        <li><a href="#vllm-batching">Batching and CUDA Graphs</a></li>
        <li><a href="#vllm-cpu-overhead">CPU Overhead and How It's Hidden</a></li>
      </ul>
    </li>
    <li><a href="#framework-comparison">Serving Frameworks: vLLM vs TensorRT-LLM vs SGLang</a>
      <ul class="post-toc-sublist">
        <li><a href="#framework-philosophy">Core Philosophy</a></li>
        <li><a href="#kv-cache-diff">KV Cache Management Differences</a></li>
        <li><a href="#when-to-use">When to Use Which</a></li>
      </ul>
    </li>
    <li><a href="#inference-slow">Why Is Inference Slow?</a>
      <ul class="post-toc-sublist">
        <li><a href="#slow-bandwidth">Memory Bandwidth Bottleneck</a></li>
        <li><a href="#slow-kv">KV Cache Pressure</a></li>
        <li><a href="#slow-sequential">Sequential Decode Dependency</a></li>
        <li><a href="#slow-cpu">CPU Scheduling Overhead</a></li>
      </ul>
    </li>
    <li><a href="#kubernetes">Kubernetes for LLM Serving</a>
      <ul class="post-toc-sublist">
        <li><a href="#k8s-primitives">Pods, Services, Deployments</a></li>
        <li><a href="#gpu-scheduling">GPU Scheduling</a></li>
        <li><a href="#autoscaling">Autoscaling: HPA vs KEDA</a></li>
        <li><a href="#rolling-canary">Rolling Updates and Canary</a></li>
        <li><a href="#cold-start">Cold Start Problem</a></li>
        <li><a href="#deploy-vllm">Deploying vLLM on Kubernetes</a></li>
      </ul>
    </li>
    <li><a href="#envoy">L7 Load Balancing and Envoy</a>
      <ul class="post-toc-sublist">
        <li><a href="#l4-vs-l7">L4 vs L7</a></li>
        <li><a href="#envoy-arch">Envoy Architecture</a></li>
        <li><a href="#routing">Routing, Retries, Timeouts</a></li>
        <li><a href="#circuit-breaking">Circuit Breaking</a></li>
        <li><a href="#rate-limiting">Rate Limiting</a></li>
        <li><a href="#llm-routing">LLM-Specific Routing</a></li>
        <li><a href="#prevent-overload">Preventing Overload</a></li>
      </ul>
    </li>
    <li><a href="#perf-optimisation">Performance Optimisation: Reducing Cost Per Token</a>
      <ul class="post-toc-sublist">
        <li><a href="#cost-decomposition">Cost Per Token: The Decomposition</a></li>
        <li><a href="#roofline-diagnostic">Roofline Thinking as Diagnostic</a></li>
        <li><a href="#cost-playbook">The Cost-Per-Token Reduction Playbook</a></li>
        <li><a href="#batching-lever">Batching: The First Lever</a></li>
        <li><a href="#kv-reuse">KV Cache Reuse</a></li>
        <li><a href="#speculative-decoding-perf">Speculative Decoding</a></li>
        <li><a href="#gpu-utilisation">GPU Utilisation and Memory Bandwidth</a></li>
        <li><a href="#profiling">Profiling: Nsight Conceptually</a></li>
        <li><a href="#cost-synthesis">Synthesis: The "Reduce Cost Per Token" Answer</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## Prefill vs Decode
{: #prefill-decode}

Every LLM generation request has two distinct phases with radically different hardware profiles. Understanding the split is the foundation for every inference systems decision.

### What Happens in Each Phase
{: #what-happens}

**Prefill** processes all input tokens simultaneously. The model runs one forward pass over the entire prompt, computing Key and Value tensors for every input position in parallel. The output of prefill is: (1) the first generated token, and (2) the KV cache — the stored K, V tensors for every layer and every input position, ready for decode to reuse.

**Decode** generates output tokens one at a time. At each step, the model processes only the single newly generated token, but it must attend over the full KV cache of all previous tokens to compute the next one. Each decode step produces one token and extends the KV cache by one position. This repeats until an EOS token is emitted or the max length is reached.

```
Prefill (parallel over all L input tokens):
  Input: [token_1, token_2, ..., token_L]
  Output: KV_cache[all layers][all positions] + first output token

Decode (sequential, one token per step):
  Step 1: new_token_1 → attend over KV_cache[L+0 positions] → new_token_2
  Step 2: new_token_2 → attend over KV_cache[L+1 positions] → new_token_3
  ...
  Step N: new_token_N → EOS
```

This is autoregressive generation: each token depends on all previous tokens, so you cannot parallelise across output tokens. The sequential dependency is fundamental to the transformer architecture.

### Compute-Bound vs Memory-Bandwidth-Bound
{: #compute-vs-memory}

The two phases hit completely different hardware limits.

**Prefill is compute-bound.** With L input tokens processed in parallel, the attention Q/K/V projections are large matrix multiplications: shape `[L × D] × [D × D]`. With L in the thousands, arithmetic intensity (FLOPs per byte of memory traffic) is high — the GPU's tensor cores stay busy. MFU during prefill can reach 50–70% on modern hardware.

**Decode is memory-bandwidth-bound.** Each decode step processes exactly one new token. The projection is `[1 × D] × [D × D]` — a matrix-vector product, not a matrix-matrix product. For a 70B model in BF16:

```
Weight bytes per decode step ≈ 140 GB  (must stream all weights from HBM)
FLOPs per decode step ≈ 2 × 70B = 140 GFLOPs

Arithmetic intensity = 140 GFLOPs / 140 GB = 1 FLOP/byte

A100 roofline crossover ≈ 156 FLOP/byte
→ decode at batch=1 runs at 1/156 = 0.6% of compute capacity
```

The GPU is spending essentially all its time moving weights from HBM to registers, not doing arithmetic. Adding more FLOPS to the GPU wouldn't help. The only way to improve throughput is to reduce bytes (quantisation) or increase reuse (larger batch size).

At batch size B, arithmetic intensity becomes approximately `B FLOP/byte` for the weight-dominated decode step. The crossover from memory-bound to compute-bound is around batch size 156 on an A100 — meaning most production decode runs are memory-bound.

<div class="post-flow post-flow--compare" role="group" aria-label="Prefill vs Decode hardware profile">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Prefill</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Processes L tokens in parallel → matrix-matrix multiply</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Compute-bound — tensor cores saturated</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">MFU: 50–70% achievable</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Bottleneck: raw FLOPS (H100 > A100 helps)</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Decode</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Processes 1 token per step → matrix-vector multiply</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Memory-bandwidth-bound — HBM saturated</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">MFU: 1–5% at batch=1</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Bottleneck: HBM bandwidth (more GB/s helps)</span></li>
    </ol>
  </div>
</div>

### TTFT vs TPOT
{: #ttft-tpot}

The two phases map directly onto the two latency metrics that define user experience.

**TTFT (Time to First Token)** = prefill latency. The user submits a request and waits with a blank screen until the first character appears. TTFT is dominated by:
- Prompt length (O(L²) attention for prefill)
- Current GPU queue depth (if other requests are being prefilled ahead of you)
- Whether the GPU is running chunked prefill or pure prefill

**TPOT (Time Per Output Token)** = decode latency per step, also called inter-token latency (ITL). The user sees tokens stream in. TPOT is dominated by:
- Weight streaming time from HBM (≈ `2 × P × bytes_per_param / bandwidth`)
- Batch size (larger batch → same weight bytes shared across more tokens → higher throughput but same latency per step)
- KV cache access time for long contexts

**The key asymmetry:** TTFT can be seconds for long prompts even on fast hardware. TPOT is typically 30–100ms per token regardless of prompt length (it's bandwidth-bound on weights, not on prompt length). A 10K-token prompt might take 2 seconds to prefill but then decode at 20 tokens/second.

```
Typical production targets:
  TTFT P50 < 500ms,  P99 < 2s
  TPOT P50 < 50ms,   P99 < 100ms   (= 10–20 tokens/second visible to user)
```

> **Interview question:** A user complains that sometimes the first token takes 5 seconds, but other times it's under 200ms. Decoding speed is always fine. What are the likely causes?
>
> *The symptom — variable TTFT but stable TPOT — points directly at prefill-phase interference. The fast case: the request has a short prompt and goes straight to prefill. The slow case is one of: (1) **Prefill-decode contention** — a long-prompt request is being prefilled on the same GPU, occupying it for hundreds of milliseconds and blocking your request from starting prefill. (2) **Queue backup** — during a traffic spike, requests queue waiting for the prefill slot; decode requests run fine because they already hold KV cache. (3) **Cold start** — a pod is starting up under scale-out and model loading adds to the latency. Fix (1) and (2) with chunked prefill (caps single-step prefill time) or prefill-decode disaggregation (separate GPU pools).*

### Why the Split Matters for Systems
{: #why-it-matters}

The prefill/decode asymmetry creates the central tension in LLM serving system design:

- **A single GPU running both phases** means a long prefill blocks all currently-decoding requests on that GPU, creating TPOT spikes for other users.
- **Routing decisions depend on phase**: a new request in prefill needs a compute-rich machine; a request in decode needs high memory bandwidth.
- **Scaling is different for each**: if TTFT is bad, you need more prefill capacity (or chunked prefill); if TPOT is bad, you need more memory bandwidth or smaller batch sizes.
- **Disaggregation** — running prefill and decode on separate GPU pools — is the architectural response to this tension. Covered in the [throughput vs latency](#throughput-latency) section.

---

## KV Cache
{: #kv-cache}

### What Gets Cached and Why
{: #what-is-kv}

In the transformer attention mechanism, every token produces three vectors: Query (Q), Key (K), and Value (V) via learned linear projections.

```
Q = W_Q · x    K = W_K · x    V = W_V · x

Attention(Q, K, V) = softmax(Q · Kᵀ / √d_head) · V
```

During autoregressive decode, the new token at position `t` needs to attend to every past token's K and V:

```
output_t = softmax(Q_t · [K_1, K_2, ..., K_t]ᵀ / √d_head) · [V_1, V_2, ..., V_t]
```

Without a cache, you'd recompute K and V for all past tokens at every decode step — `O(t²)` total work across all decode steps. The KV cache stores the K and V tensors as they are computed, so each new decode step only needs to compute K, V for the one new token and append it to the cache.

**The trade: compute for memory.** KV caching converts `O(t²)` redundant compute into `O(t)` memory. This is almost always the right trade — GPU memory is available and recomputation is wasteful — but the memory cost becomes the dominant constraint at scale.

### Memory Growth
{: #kv-growth}

KV cache size grows with every dimension of the problem:

```
KV_bytes = 2 × n_layers × n_kv_heads × d_head × seq_len × batch_size × bytes_per_element
           ↑             ↑             ↑          ↑          ↑            ↑
        K and V      per-layer     head dim    context    concurrent    2 for FP16
                     cache        (e.g. 128)   length     requests
```

**Concrete example — Llama-3 8B (MHA, FP16):**
- 32 layers, 32 KV heads, d_head = 128, FP16 (2 bytes)
- Per token per layer: 2 × 32 × 128 × 2 = **16,384 bytes = 16 KB**
- At 4K context (one request): 16 KB × 32 layers × 4096 tokens = **2 GB**
- At batch=32: **64 GB** — exceeds a single 80GB A100 when combined with model weights

**Concrete example — Llama-3 70B (GQA, FP16):**
- 80 layers, **8 KV heads** (GQA reduces from 64 query heads to 8), d_head = 128, FP16
- Per token per layer: 2 × 8 × 128 × 2 = **4,096 bytes = 4 KB**
- At 32K context (one request): 4 KB × 80 layers × 32768 tokens = **10 GB**
- At batch=8: **80 GB** — fills the entire A100 with nothing left for weights

This is why **Grouped Query Attention (GQA)** was introduced: by sharing KV heads across groups of query heads (`n_kv_heads ≪ n_q_heads`), the KV cache shrinks proportionally. Llama-3 70B's GQA reduces KV cache 8× vs full MHA at the same model quality.

KV cache growth with context length is **linear** for a fixed batch (not quadratic — that would be attention FLOPs for prefill, not the cache itself). But because both context length and batch size multiply together, the product grows fast.

> **Interview question:** You have an 80GB A100. A 13B model takes 26GB of weights in FP16. How many concurrent requests can you serve at 4K context?
>
> *Available for KV cache: 80 − 26 = 54 GB. Llama-2 13B has 40 layers, 40 KV heads, d_head=128, FP16. KV per token per layer = 2 × 40 × 128 × 2 = 20,480 bytes. Per request at 4K context = 20,480 × 40 × 4096 = 3.2 GB. Max concurrent requests = 54 GB / 3.2 GB ≈ 16. In practice fewer — you need headroom for activations, overhead, and incomplete blocks. A reasonable answer is 10–12. This is why KV quantisation (INT8 halves the KV footprint to 1.6 GB/request → 33 concurrent) and GQA matter so much for serving economics.*

### Fragmentation and PagedAttention
{: #kv-fragmentation}

The naive approach to KV memory management pre-allocates a contiguous GPU memory block for each request's full KV cache at the start of the request:

```
Request A: reserve space for max_output_len = 2048 tokens → allocate 3.2 GB upfront
Request B: reserve 3.2 GB upfront
Request C: reserve 3.2 GB upfront
```

This creates two types of waste:

**Internal fragmentation:** Request A generates only 200 tokens before emitting EOS. The remaining 1848 positions worth of reserved memory sits idle but is not available to other requests.

**External fragmentation:** As requests complete at different times, the freed blocks are scattered across GPU memory. When a new request arrives that needs 3.2 GB contiguous, there may be 20 GB of total free memory but no single 3.2 GB contiguous region available.

In practice, pre-allocation wastes 60–80% of GPU KV memory. This directly caps the achievable batch size — and batch size is the primary lever for throughput.

**[PagedAttention](https://arxiv.org/abs/2309.06180)** applies OS virtual memory paging to KV storage. Physical GPU memory is divided into fixed-size **KV blocks** (typically 16–32 tokens per block). Each request maintains a **block table** — a logical-to-physical mapping — rather than a contiguous reservation. Blocks are allocated on demand, one at a time, as the sequence grows:

```
Before (contiguous, pre-allocated):
  GPU memory: [AAAA......][BBBB......][CCCC......][free free free]
                          ↑ wasted    ↑ wasted    ↑ can't use (fragmented)

After (PagedAttention, block-level):
  GPU memory: [A0][B0][C0][A1][free][B1][C1][A2][free][free][B2]
              ↑ any request can use any free block
```

Maximum internal fragmentation: `block_size − 1` tokens per request (at most one partially-filled block). External fragmentation: eliminated — any free block serves any request regardless of physical location. KV memory utilisation jumps from 20–40% to over 95%.

The cost: the attention kernel must follow the block table indirection to gather K/V values from non-contiguous physical locations. This requires a custom attention kernel and is slightly less cache-friendly than dense contiguous access. In practice, the throughput gain from fitting 2–4× more requests in memory dominates the access pattern overhead.

**Copy-on-write for beam search.** Multiple beams from the same request share the prompt's KV blocks. Blocks are reference-counted. When beam `i` needs to write a new token into a shared block, it first copies the block to a new physical location (copy-on-write), then writes. The shared blocks remain intact for the other beams.

> **Interview question:** PagedAttention solves fragmentation but requires a custom attention kernel. A colleague proposes instead to just run the standard kernel on padded contiguous tensors, filling unused slots with zeros. What's wrong with this?
>
> *Three problems. First, padding wastes compute: a batch of 32 requests at various sequence lengths (from 50 to 2000 tokens) padded to the max of 2000 means short requests consume 40× more attention FLOPs than they need. Second, you still haven't solved the allocation problem — you still need to pre-allocate contiguous buffers per request, restoring all the fragmentation waste. The padding only helps the kernel, not the allocator. Third, padding introduces a correctness issue in causal attention: you must ensure that real tokens don't attend to padding positions and padding positions don't contribute to softmax. This requires carefully constructed attention masks — manageable but error-prone. PagedAttention solves the allocation problem at the root; the "padded contiguous" approach avoids writing a custom kernel but leaves the memory fragmentation unsolved.*

### Prefix Caching
{: #prefix-caching}

Many requests share an identical prefix: the same system prompt for every user in a chatbot deployment, the same few-shot examples for every document in a batch job, the accumulated conversation history in a multi-turn dialogue.

Without prefix caching, each request re-runs prefill over the shared prefix — wasting O(L²) attention FLOPs on tokens whose KV tensors were already computed for the previous request.

**[RadixAttention](https://arxiv.org/abs/2312.07104)** (SGLang) organises the KV block pool as a **radix tree** keyed by token sequences. On request arrival, the system walks the tree to find the longest matching prefix already in cache. Only the unmatched suffix requires fresh prefill:

```
Tree (after several requests with shared system prompt):

[system prompt: 1024 tokens] ──► [user A turn 1] ──► [user A turn 2]
                             └──► [user B turn 1]
                             └──► [user C turn 1] ──► [user C turn 2]
```

A new request from user A in turn 3 finds its entire history (system prompt + turns 1 and 2) already cached. Only the new user message needs prefill. The KV for 3000 tokens is free.

Eviction is LRU at the block level with reference counting — blocks held by active requests are never evicted. Tree nodes are only evicted when all descendants are also evictable (you can't evict a parent node while a child is in use).

**Where prefix caching excels:**
- System prompts shared by all users (1024-token system prompt with 100 concurrent users → 100 prefills saved per turn)
- Batch jobs over a fixed document or context (RAG retrieval shared across multiple queries)
- Multi-turn chat (each turn adds incrementally; prior history is free)
- Agent loops with repeated tool descriptions

> **Interview question:** Your chatbot has a 2048-token system prompt. 1000 concurrent users each send a 100-token message. Without prefix caching vs with it, what's the prefill compute difference?
>
> *Without caching: each request prefills 2048 + 100 = 2148 tokens. Total: 1000 × 2148 = 2.148M tokens of prefill. With prefix caching (system prompt warm): each request prefills only the 100-token user message. Total: 1000 × 100 = 100K tokens. Speedup: 21.5× reduction in prefill compute. The catch: the first request after the system prompt changes (or after cache eviction) pays full prefill cost. And the prefix cache occupies KV memory that could otherwise serve more decode slots — there's a cache size vs decode capacity tradeoff. In practice, a 2048-token system prompt at 32 KV bytes/token = 64 KB, so 100 users' worth is only 6.4 MB — negligible. For 100K-token contexts, the tradeoff becomes real.*

---

## Continuous Batching
{: #continuous-batching}

### Why Static Batching Fails
{: #static-batching-problem}

Static batching is the training-era approach applied to inference: collect N requests, process them as a batch, wait for all to complete, collect the next N.

The problem is LLM output lengths are wildly variable. A factual query generates 20 tokens. A code generation request generates 800. In a batch of 32 requests:

```
Step 0:   [req1 req2 req3 ... req32]   — all 32 decoding, GPU fully used
Step 50:  [req1 req2 req3 ... req32]   — most still decoding
Step 100: [         req12 ... req32]   — 11 requests completed, 21 idle slots
Step 200: [                   req32]   — 31 idle slots, 1 request finishing
...
Step 800: req32 finishes → next batch starts
```

GPU utilisation collapses as the batch drains. With realistic output-length distributions (median 50 tokens, mean 200, max 2000), static batching achieves 20–35% GPU utilisation.

The wasted time cannot be filled: new requests can't start until the entire current batch completes, because static batching doesn't have the machinery to add requests mid-batch. The new requests queue and wait.

### Iteration-Level Scheduling
{: #iteration-level}

**Continuous batching** (ORCA, 2022) makes a conceptual shift: schedule at the level of individual decode iterations, not request batches.

After every decode step, the scheduler runs:

<div class="post-flow" role="group" aria-label="Continuous batching scheduler loop">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">GPU executes one decode step on the current batch</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Scheduler checks outputs: which requests emitted EOS?</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Completed requests evicted: KV blocks freed, slot opened</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Waiting requests admitted to fill open slots (prefill runs inline)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Next decode step runs immediately on the updated batch</span></li>
  </ol>
</div>

The batch is now a **dynamic pool** that the CPU scheduler continuously adjusts. A request finishing at step 100 opens a slot that a waiting request fills at step 101. The GPU is never idle waiting for the slowest request.

```
Step 100: [req1 ✓  req2 req3 ... req32]   — req1 finishes, slot opens
Step 101: [req33   req2 req3 ... req32]   — req33 admitted, prefill + decode mixed
Step 102: [req33   req2 req3 ... req32]   — all decoding at full batch
```

This requires PagedAttention: the different requests in the batch are at wildly different decode positions (req2 at step 100, req33 just starting), so their KV caches have different sizes. Contiguous pre-allocation would require knowing output lengths in advance; PagedAttention allocates blocks on demand.

**Throughput impact.** Static batching at 30% GPU utilisation with continuous batching at 80–85% on the same hardware. vLLM's paper reports 23× throughput improvement over naive serving (no batching) and roughly 2× over static batching with comparable batch sizes.

**The CPU scheduling overhead problem.** After every GPU decode step, the scheduler has non-trivial CPU work: scan completed requests, free KV blocks, score waiting requests by priority/SLO, allocate KV blocks, construct the next batch. Without pipelining, this CPU overhead adds latency between GPU steps.

The fix: **overlapped scheduling**. While the GPU executes step `t+1`, the CPU processes the results of step `t` and prepares the batch for step `t+2`. This pipelines CPU scheduling with GPU execution, hiding the scheduling overhead entirely in steady state.

> **Interview question:** With continuous batching, you have 32 requests at different decode positions. How does the attention kernel handle attending over different KV cache lengths without wasting compute on padding?
>
> *Padding to max length wastes FLOPs proportional to the longest request in the batch — a 2000-token request in a batch where everyone else is at 50 tokens wastes 40× more attention compute per step than necessary. The production solution is a variable-length attention kernel. FlashAttention supports this via `flash_attn_varlen_func` which takes a `cu_seqlens` array (cumulative sequence lengths: [0, 50, 100, ..., 2000]) and packs all requests into a single flattened input tensor, processing each request's attention over exactly its KV length with no padding. Under PagedAttention, the kernel additionally follows each request's block table to gather K/V from non-contiguous physical locations. This combination — variable-length + paged — is what vLLM and SGLang use in production. The kernel is more complex than standard batched attention but eliminates both the compute waste from padding and the memory waste from fragmentation.*

### Chunked Prefill
{: #chunked-prefill}

Continuous batching solves the static-batching throughput problem but creates a new one: **prefill-decode interference**.

When a new long-prompt request enters the batch, its prefill runs on the GPU. A 10,000-token prompt requires O(10,000²) attention operations for prefill — hundreds of milliseconds. During this time, all currently-decoding requests in the batch are stalled. Their TPOT spikes, violating latency SLOs for interactive requests.

**Chunked prefill** (Sarathi, 2023) breaks the prefill into fixed-size chunks and interleaves them with decode:

```
iter 1: [prefill_chunk(tokens 0–511)]    + [decode all current requests]
iter 2: [prefill_chunk(tokens 512–1023)] + [decode all current requests]
iter 3: [prefill_chunk(tokens 1024–1535)]+ [decode all current requests]
...
iter k: [final prefill chunk, req ready for decode] + [decode]
```

Every iteration has bounded duration. Decode TPOT stays stable because no single prefill dominates a full iteration. The cost: TTFT for the new request is higher because prefill is now spread across multiple iterations instead of one. This is often the right tradeoff — interactive users care more about smooth streaming (TPOT) than the first-token wait (TTFT).

**Bonus: better GPU utilisation.** Prefill is compute-bound (high arithmetic intensity); decode is memory-bandwidth-bound (low arithmetic intensity). On a GPU, these operations use different hardware resources — tensor cores vs HBM controller. A mixed prefill+decode iteration can exploit both simultaneously, achieving higher total GPU utilisation than either pure prefill or pure decode alone.

The optimal chunk size trades off TTFT (larger chunks → faster per-request prefill) against TPOT stability (smaller chunks → shorter GPU stall per decode step). Typical range: 512–2048 tokens. Latency-sensitive APIs use smaller chunks; batch-mode jobs use larger.

---

## Throughput vs Latency
{: #throughput-latency}

### The Core Tradeoff
{: #the-tradeoff}

The fundamental tension in LLM serving: every action that increases throughput increases latency, and vice versa.

**Batching and throughput.** During decode, the GPU loads model weights from HBM once per step regardless of batch size. At batch=1, those weight bytes generate 1 token. At batch=32, they generate 32 tokens. Throughput (tokens/sec) scales roughly linearly with batch size up to the memory-bound/compute-bound crossover. This is the "free lunch" of batching — the weight load is amortised.

**Batching and latency.** A larger batch means more requests competing for the same KV memory and the same decode slot. A new request arriving to a full batch waits in queue. The P99 latency grows as the batch stays full longer. Additionally, larger batches can slightly increase per-step time (more KV cache to stream, more output to scatter), adding to TPOT.

**The queuing theory view.** At low load, requests arrive, find an empty slot, and start immediately — latency is close to the minimum (just the GPU compute time). As load increases, the batch fills up and requests queue. Little's Law: `L = λW` where L = mean requests in system, λ = arrival rate, W = mean wait time. As utilisation approaches 100%, queueing time grows unboundedly — the classic M/M/1 queueing blow-up.

```
                    latency
                       │
                       │                              /
                       │                           /
                  P99  │                        /
                       │                    /
                  P50  │_______________/
                       │
                       └────────────────────────────── load (% of capacity)
                                     ↑
                               ~70% is where P99 starts climbing fast
```

The practical implication: you cannot safely run at 95% utilisation. At 70% utilisation, queue depth is manageable. At 90%, the tail latency blows up. LLM serving systems typically target 60–75% sustained utilisation with burst capacity headroom.

**The batch size knob.** In practice you don't directly control batch size — it emerges from the traffic rate and the continuous batching scheduler. What you control:
- `max_batch_size`: cap on concurrent requests per GPU worker (sets the throughput ceiling and memory limit)
- Autoscaling: add GPU workers when queue depth builds up
- Prefill chunk size: controls how much prefill competes with decode per step

| Workload | Optimise for | Set max_batch_size | Prefill chunk |
|---|---|---|---|
| Interactive chat | TTFT, TPOT | Small (4–16) | Small (256–512) |
| Coding assistant | TPOT (smooth streaming) | Medium (8–32) | Medium (512–1024) |
| Batch document processing | Throughput | Large (32–128) | Large (2048–8192) |

> **Interview question:** Your P50 TTFT is 300ms and your P99 is 4 seconds. P50 TPOT is fine. What does this distribution tell you and what do you fix?
>
> *P99 TTFT >> P50 TTFT with healthy TPOT means: most requests see fast prefill (short prompts, low contention), but a tail of requests wait a long time before their first token. The first token wait is queue time plus prefill time. Root causes: (1) Long-prompt requests — some users send 10K-token prompts that take seconds to prefill, blocking others from starting theirs. Diagnosis: plot TTFT vs prompt length — if the tail correlates with long prompts, this is the cause. Fix: chunked prefill (cap per-step prefill at 512 tokens). (2) Batch saturation — traffic spikes fill the batch, and P99 requests queue for tens of seconds waiting for a slot. Diagnosis: plot queue depth over time — if spikes correlate with TTFT P99 spikes, this is the cause. Fix: scale out, or increase `max_batch_size` if KV memory allows. (3) Prefill-decode contention — a full batch of decoding requests leaves no room for new prefills. Fix: reserve some batch slots for prefill-only, or use priority scheduling (preempt low-priority decode to admit high-priority prefill).*

### Interactive vs Batch Workloads
{: #interactive-vs-batch}

The right operating point depends entirely on what you're optimising.

**Interactive workloads** (user-facing chat, coding assistants, search augmentation):
- Users notice TTFT above ~1 second and TPOT above ~80ms/token
- Optimise for: P95 TTFT < 1s, P99 TPOT < 80ms
- Accept: lower throughput — run at 50–65% utilisation to keep tail latency low
- Batch strategy: continuous batching with small max batch size, chunked prefill with small chunk sizes
- Scaling: keep warm replicas to avoid cold-start latency on scale-up

**Batch workloads** (offline document processing, data generation, evals):
- No human waiting — latency of individual requests doesn't matter, only total job time
- Optimise for: tokens per second per dollar
- Accept: high TTFT and variable TPOT on individual requests
- Batch strategy: large max batch size, large prefill chunks, fill GPU to 85–90% utilisation
- Scaling: scale down aggressively when queue drains (use spot/preemptible GPUs)

**Mixed deployments** — serving interactive and batch traffic on the same fleet — require priority queuing: interactive requests preempt batch requests for KV slots. Batch requests run on leftover capacity. This maximises GPU utilisation (batch fills idle time) without compromising interactive latency SLOs.

### Prefill-Decode Disaggregation
{: #pd-disaggregation}

The hardware mismatch between prefill (compute-bound) and decode (memory-bandwidth-bound) suggests a structural solution: run them on separate, specialised GPU pools.

**Disaggregated serving** routes each request through two stages on two different machines:

```
New request
     │
     ▼
┌────────────────┐    KV transfer    ┌────────────────┐
│  Prefill Pool  │ ─────────────────►│  Decode Pool   │
│  (H100 × 4)   │   RDMA / NVLink   │  (A100 × 8)   │
│                │                   │                │
│  Compute-dense │                   │  BW-optimised  │
│  Fast prefill  │                   │  Steady TPOT   │
└────────────────┘                   └────────────────┘
```

After prefill completes, the KV cache is transferred to a decode worker over RDMA (RoCE/InfiniBand) or NVLink. The transfer adds latency (typically 10–50ms for a 4K-token KV cache), but enables:

1. **No prefill-decode interference** — decode TPOT is never stalled by a long prefill because they run on different machines
2. **Hardware specialisation** — prefill pools can use H100s (best FLOPS); decode pools can use A100s (best HBM bandwidth per dollar)
3. **Independent scaling** — if TTFT is bad, scale out the prefill pool; if TPOT is bad, scale out the decode pool
4. **Better GPU utilisation** — each pool operates in its optimal regime (compute-bound prefill, BW-bound decode) rather than both being suboptimal on the same GPU

The cost: network bandwidth for KV transfer (a 13B model, 32 layers, 4K context, FP16 = ~0.5 GB per request) and the complexity of coordinating two pools with a scheduler.

Disaggregation makes economic sense when:
- The fleet is large enough that dedicated pools don't leave hardware idle
- Traffic has enough long-prompt requests that prefill-decode interference is measurable
- The KV transfer latency is small relative to total response time

For small deployments, chunked prefill achieves most of the benefit at far lower complexity.

> **Interview question:** You're running a disaggregated prefill-decode system. A request with a 50,000-token prompt arrives. Walk through exactly what happens and identify where the bottlenecks are.
>
> *The request goes to a prefill worker. Prefill is O(L²) attention — 50K² = 2.5B attention operations per layer, across 32 layers. On an H100 at 1 PFLOPS BF16, and ~4 × 50K² × 32 = ~3.2 TFLOPs total prefill compute: ~3.2 seconds just for attention. The prefill worker is compute-bound and fully utilised during this time. After prefill: KV cache for 50K tokens. At 32 layers, 32 heads, d_head=128, FP16: 2 × 32 × 32 × 128 × 50000 × 2 ≈ 13 GB. This must transfer to the decode worker over RDMA. At 200 GB/s InfiniBand: ~65ms transfer time. Decode worker receives the 13 GB KV cache and starts generating. First decode step must stream the 13 GB KV cache from HBM plus model weights — this step is slow. TPOT for first few tokens will be high (13 GB KV + model weights streaming), then settles as no new KV tokens are added much. The bottlenecks: (1) prefill is ~3s for a 50K prompt — unavoidable compute cost, (2) KV transfer is ~65ms — fast but adds to TTFT, (3) first decode steps have high TPOT due to long-context KV streaming — mitigate with KV quantisation (INT8 halves the 13 GB to 6.5 GB).*

---

## vLLM Internals
{: #vllm-internals}

vLLM is the de facto reference implementation for production LLM serving. Understanding its internals gives you the vocabulary to reason about any serving system.

### PagedAttention Implementation
{: #vllm-pagedattn}

The physical GPU memory for KV storage is divided into a **global block pool** — a flat array of fixed-size blocks, each holding KV tensors for `block_size` tokens (default: 16) across all layers and heads. Every block in the pool is identical in size and can serve any request.

Each active request holds a **block table**: a per-request array mapping logical block indices (0, 1, 2, ...) to physical block IDs in the pool. When the sequence grows past a block boundary, the allocator pops a free block from the pool and appends its ID to the block table.

```
Global block pool (80GB GPU, e.g. 4096 blocks of 16 tokens):
  [block 0: free] [block 1: req_A, logical 0] [block 2: req_B, logical 0]
  [block 3: req_A, logical 1] [block 4: free] [block 5: req_C, logical 0] ...

Request A block table:
  logical 0 → physical 1
  logical 1 → physical 3
  logical 2 → physical 7   (just allocated)
```

The PagedAttention CUDA kernel uses this block table at runtime. Instead of a single pointer to a contiguous KV buffer, the kernel receives the block table array and the `block_size`. For each query vector, it iterates over logical blocks, dereferences the physical block address, and loads the K/V tensors for that chunk of positions. The online softmax accumulation (the "flash attention" trick) handles the partial sums correctly across non-contiguous blocks.

**Reference counting for sharing.** When two sequences share a prompt prefix (parallel sampling, beam search), they can point to the same physical blocks for the shared region. Each physical block carries a reference count. When a sequence needs to modify a shared block (decode extends into it), the allocator checks the ref count:
- ref_count == 1: the sequence owns the block, write in-place
- ref_count > 1: copy-on-write — allocate a new block, copy the contents, decrement the old block's ref count, update the block table to point to the new block

This means two beams of a beam search request share the entire prompt KV (potentially gigabytes) without duplication, until their outputs diverge.

**Memory pressure and eviction.** When the free block pool runs out, vLLM must evict something. Two options:
- **Recompute** (default): free all blocks for the lowest-priority request, discard its KV entirely. On re-scheduling, it re-prefills from scratch. Zero memory overhead for eviction, but wastes compute on re-prefill.
- **Swap to CPU**: copy the request's KV blocks to CPU DRAM, free the GPU blocks. On re-scheduling, copy back. Avoids re-prefill at the cost of CPU memory and PCIe bandwidth. Used for beam search where re-prefill is expensive.

> **Interview question:** vLLM's block size is typically 16 tokens. Why not 1 token (maximum flexibility) or 1024 tokens (fewer table entries)?
>
> *Block size trades off three things. Too small (1 token): the block table becomes enormous — at 32K context, 32K entries per request, each requiring a GPU memory dereference. The attention kernel must do 32K non-coalesced memory reads. Also, per-block metadata overhead (ref counts, free list pointers) dominates. Too large (1024 tokens): internal fragmentation can waste up to 1023 tokens per request (≈ 95% waste for short requests). Also, copy-on-write becomes expensive — sharing breaks at a coarse granularity, requiring large copies. Block size 16 balances: the block table fits in registers or shared memory for typical contexts, fragmentation is at most 15 tokens/request (<1%), and copy-on-write copies are small (16 × n_layers × n_kv_heads × d_head × 2 bytes). vLLM exposes this as `--block-size` and experimentation shows 16–32 optimal for most models.*

### The Scheduler: Queues and Preemption
{: #vllm-scheduler}

The vLLM scheduler maintains three queues and runs once per GPU step:

<div class="post-flow" role="group" aria-label="vLLM scheduler queues">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Waiting</strong> — requests that have arrived but not yet started prefill. Ordered by arrival time (FCFS default) or priority.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Running</strong> — requests whose prefill has started or completed; they hold KV blocks in GPU memory and participate in every decode step.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Swapped</strong> — requests preempted off GPU (their KV blocks moved to CPU DRAM). Awaiting free blocks to resume.</span></li>
  </ol>
</div>

**Per-step scheduling loop:**

```
1. For each running request: compute how many new blocks it needs this step
2. If total blocks available ≥ total needed: proceed
3. If not enough blocks:
     a. Move swapped requests back to waiting (can't help if no blocks)
     b. Preempt lowest-priority running requests until enough blocks freed
        → policy: recompute (free blocks, send to waiting) or swap (copy to CPU)
4. Promote waiting requests to running if:
     a. Block pool can cover their first chunk
     b. Total batched tokens < max_num_batched_tokens (the token budget)
5. Allocate blocks for newly admitted requests
6. Build the batch: running (decode) + newly admitted (prefill, chunked)
7. Submit to GPU
```

**Token budget.** The key constraint is `max_num_batched_tokens` — the maximum number of tokens processed in a single GPU step. Decode requests each contribute 1 token per step. Prefill chunks contribute `chunk_size` tokens. The scheduler fills the budget: decode requests first (higher priority), then prefill chunks until the budget is exhausted. A waiting request whose full prefill doesn't fit in the remaining budget is chunked: the first `remaining_budget` tokens are scheduled this step, the rest next step.

**FCFS vs priority scheduling.** Default FCFS is fair but suboptimal for mixed workloads (interactive + batch). Priority scheduling assigns numeric priorities at submission time — interactive requests get priority=10, batch jobs get priority=1. The preemption logic targets the lowest-priority running request first, protecting interactive latency.

> **Interview question:** vLLM preempts by recompute by default rather than swapping. Under what conditions would you switch to swap-based preemption?
>
> *Recompute is preferred when: the preempted request has a short prompt (cheap to re-prefill), the PCIe bandwidth between GPU and CPU is a bottleneck (H100 PCIe: 64 GB/s), or CPU DRAM is limited. Swap is preferred when: the preempted request has a very long prompt (thousands of tokens) that would take seconds to re-prefill, and the KV cache for those tokens is large enough that re-prefill cost outweighs the copy cost. Calculation: re-prefill cost ≈ `O(L²)` attention FLOPs; swap cost ≈ `2 × KV_bytes / PCIe_bandwidth`. For a 4K-token prompt on Llama-3 8B (KV ≈ 2 GB), swap time ≈ 2 GB / 64 GB/s ≈ 31ms. Re-prefill time ≈ compute-bound prefill at ~100ms. Swap wins. For a 100-token prompt (KV ≈ 50 MB), swap time ≈ 0.8ms but re-prefill ≈ <1ms — recompute wins. vLLM uses recompute for most requests because typical prompts are short; beam search explicitly requests swap because all beams share a long prompt that would be expensive to re-prefill.*

### Batching and CUDA Graphs
{: #vllm-batching}

**Flattened batch representation.** At each step, vLLM constructs a single flat tensor containing all token inputs for the step: decode tokens (one per running request) and prefill chunks (variable length). Position IDs and attention metadata (sequence lengths, block tables) accompany the tensor. The model's forward pass runs once over this flattened input — no per-request loops at the Python level.

**Why CUDA Graphs.** Python-based LLM inference has high GPU kernel launch overhead. Each operator in a PyTorch forward pass (matmul, RMSNorm, softmax, etc.) is a separate CUDA kernel launch — hundreds of launches per forward pass, each with ~5–10µs overhead. At batch=1 with a fast model (Llama-3 8B), total decode step time might be 5–10ms, of which 30–50% is kernel launch overhead.

CUDA Graphs solve this: capture the entire forward pass as a graph once, then replay the graph on subsequent steps with a single CUDA API call. The kernel launches are pre-compiled into the graph; at replay time, only the tensor data changes (new tokens, updated KV pointers), not the launch sequence.

**Piecewise CUDA Graphs** (vLLM's approach). Naively, a single CUDA graph would need to be re-captured every time the batch size or sequence length changes. vLLM uses piecewise graphs:
- A set of graphs is pre-captured for each common decode batch size (1, 2, 4, 8, 16, ..., max_batch_size)
- At each decode step, select the captured graph matching the current batch size (padding to the next size if needed)
- For prefill steps (variable length, not graph-captured): fall back to eager mode

The overhead is small padding waste; the benefit is eliminating Python kernel launch overhead for the dominant decode path.

> **Interview question:** CUDA Graphs require the graph to be re-captured when batch size changes. How does vLLM serve a batch that's growing (new requests arriving) without re-capturing on every step?
>
> *vLLM pre-captures graphs for a set of discrete batch sizes: [1, 2, 4, 8, 16, 32, 64, 128, ...]. When the actual batch size is, say, 20, vLLM pads to the next captured size (32) with dummy tokens that don't affect real outputs (masked in attention, ignored in outputs). The graph for batch=32 is replayed. Padding waste is at most 2× but usually far less because batches tend to stay near their maximum. The continuous batching scheduler tries to keep the batch size near `max_batch_size` by aggressively admitting new requests, so the batch rarely changes size by more than a few requests per step. The real scenario where this breaks down is cold start: when the first request arrives, the batch is 1; CUDA graph for batch=1 runs. Second request arrives, batch=2; graph for batch=2 runs. Etc. But this is fast — graph selection is a pointer lookup, and the padding overhead at small batch is trivial compared to the single-request decode time.*

### CPU Overhead and How It's Hidden
{: #vllm-cpu-overhead}

A profiling study of Llama-3 8B on a single H100 broke down wall-clock time:

```
GPU compute:      38%
HTTP/API server:  33%
Scheduling:       29%
```

Only 38% of time was actual GPU computation. The other 62% was CPU work — the Python API server blocking under the GIL and the scheduler running synchronously.

**vLLM v0.6 fixes (2024):**

1. **Process separation via ZMQ.** The HTTP API server and the inference engine run in separate processes, connected by ZeroMQ sockets. The API server handles tokenisation, request queuing, and response streaming; the engine handles scheduling and GPU execution. They no longer compete for the Python GIL.

2. **Multi-step scheduling.** Instead of alternating GPU-step → CPU-schedule → GPU-step, vLLM schedules `N` future steps at once (default N=10 for pure decode batches). The GPU executes these `N` steps continuously while the CPU processes the outputs of the completed steps in parallel. The CPU scheduler runs once every N steps instead of every step, reducing scheduling overhead by N×.

3. **Asynchronous output processing.** Token detokenisation and streaming response updates happen in a thread pool while the GPU runs the next step. Previously these ran synchronously in the hot path, adding ~1ms per step.

Result: Llama-3 70B throughput improved 28% purely from these CPU-side changes, with no changes to the model or GPU kernels.

---

## Serving Frameworks: vLLM vs TensorRT-LLM vs SGLang
{: #framework-comparison}

### Core Philosophy
{: #framework-philosophy}

The three dominant open-source serving frameworks solve the same problem from different angles:

<div class="post-flow post-flow--compare" role="group" aria-label="Framework philosophy comparison">
  <div class="post-flow__col">
    <p class="post-flow__col-label">vLLM</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Dynamic memory management via PagedAttention</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">No compilation — runs any model immediately</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Focus: high throughput multi-user serving</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Widest model support, best docs, largest community</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">TensorRT-LLM</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Compile model to fused GPU kernels via TensorRT</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">NVIDIA-optimised: best raw throughput on NVIDIA hardware</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Focus: maximum throughput for a fixed production model</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">~28 min compile time, NVIDIA-only, smaller community</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">SGLang</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">RadixAttention: token-level KV prefix caching</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Simple scheduler (~4K LOC vs vLLM's larger codebase)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Focus: workloads with shared prefixes (RAG, multi-turn, agents)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Also strong at structured output generation</span></li>
    </ol>
  </div>
</div>

**TensorRT-LLM's compilation advantage.** TRT-LLM converts the model to ONNX, then TensorRT fuses multiple ops into single kernels (e.g., LayerNorm + MatMul + GELU becomes one kernel with no intermediate HBM writes), auto-tunes kernel variants for the target GPU, and quantises to FP8 with hardware-native support. The compilation takes 15–30 minutes but produces kernels that are 8–15% faster than vLLM's dynamic kernels. For a stable production model (one model version, high traffic), this is worth the operational complexity.

**Performance comparison (H100, Llama-3 70B, vary-length workload):**

| Framework | Throughput (tok/s) | TTFT P50 | Cold start |
|---|---|---|---|
| vLLM | ~12,500 | ~120ms | ~60s |
| SGLang | ~16,200 | ~112ms | ~58s |
| TensorRT-LLM | ~18,500 | ~105ms | ~28 min |

SGLang's throughput advantage over vLLM comes from its prefix caching efficiency. TRT-LLM's advantage is compiled kernels. These numbers are workload-dependent — on zero-prefix-overlap workloads, SGLang's advantage over vLLM shrinks significantly.

### KV Cache Management Differences
{: #kv-cache-diff}

**vLLM: block-level hashing.** KV blocks are 16 tokens each. A block is "prefix-cacheable" if all 16 tokens are identical across requests. vLLM hashes each block's token sequence and maintains a hash table of reusable cached blocks. Only full-block matches reuse cache; a shared prefix of 17 tokens with a new single-token divergence at position 18 reuses only block 0 (tokens 0–15), not the partial block 1.

**SGLang RadixAttention: token-level radix tree.** The cache is a radix tree where each edge represents a token sequence. Any shared prefix, regardless of block alignment, is found and reused. A shared prefix of 1024 tokens saves 1024 tokens' worth of prefill compute, not just `floor(1024/16) × 16 = 1024` — same in this case, but for odd-length prefixes, SGLang wins. More importantly, RadixAttention handles dynamic prefix growth naturally: as a multi-turn conversation grows, the tree extends rather than invalidating cached blocks.

**The tradeoff:** RadixAttention's LRU cache occupies GPU memory that vLLM could use for live KV. When prefix overlap is high (e.g., all requests share a 4K system prompt), SGLang wins significantly. When prefix overlap is low (diverse prompts), the cache sits unused and wastes memory, hurting effective batch size.

### When to Use Which
{: #when-to-use}

| Situation | Recommendation | Reason |
|---|---|---|
| Getting started, any model, fast iteration | **vLLM** | No compilation, widest support, best docs |
| High shared-prefix traffic (RAG, chatbots with fixed system prompt, agents) | **SGLang** | RadixAttention gives large throughput gains |
| Single stable model in production, NVIDIA hardware, max throughput | **TensorRT-LLM** | Compiled kernels win, 28-min compile amortises |
| Multi-turn dialogue heavy workload | **SGLang** | Conversation history caching is native |
| Research / experimentation | **vLLM** | Easy to modify, active community |
| Structured output (JSON schemas, constrained generation) | **SGLang** | Built-in structured generation support |

> **Interview question:** You're deploying a customer support chatbot with a 2048-token system prompt and 50K daily active users averaging 5 turns each. Which framework do you use and why?
>
> *SGLang. The workload has a very high prefix cache hit rate: every one of the 250K daily requests shares the exact same 2048-token system prompt, and multi-turn users share their accumulated conversation history. With RadixAttention, the system prompt KV is computed once and served from cache for every subsequent request. The cache hit ratio on the system prompt alone is (250K - 1) / 250K ≈ 100%. On average, turns 2–5 additionally share the conversation history. At 50 tokens/turn average, turn 3 has ~100 tokens of shared history on top of the 2048 system prompt — 2148 tokens cached, 50 tokens prefilled. This 97.7% prefill savings translates directly to higher throughput (fewer GPU cycles per request) and lower TTFT. vLLM's block-level caching would also capture the system prompt hit (it's exactly 128 blocks of 16), but wouldn't handle the variable-length conversation history as efficiently. TRT-LLM's kernel advantages don't matter here — the bottleneck is prefill compute, which prefix caching eliminates, not kernel speed.*

---

## Why Is Inference Slow?
{: #inference-slow}

Three questions that come up in every serving interview. Concrete, bottleneck-first answers.

### Memory Bandwidth Bottleneck
{: #slow-bandwidth}

**The core problem:** during decode, the GPU must load all model weights from HBM on every token generated. For a 70B model in BF16, that's 140 GB per decode step. At A100 HBM bandwidth (2 TB/s), the minimum time to stream the weights is `140 GB / 2 TB/s = 70ms`. That's the bandwidth floor for a single token — 14 tokens/second maximum from weight bandwidth alone, before any compute or KV cache is considered.

Arithmetic intensity during decode at batch=1 ≈ 1 FLOP/byte. The A100 roofline crossover is 156 FLOP/byte. Decode runs at 0.6% of theoretical compute capacity. The GPU's tensor cores are idle.

**How to improve throughput (amortise the bandwidth):**

| Technique | Mechanism | Gain |
|---|---|---|
| Larger batch size | Same weights loaded, more tokens produced → throughput ∝ batch | Up to compute crossover (~B=156 on A100) |
| INT4 weight quantisation | 4× fewer bytes to load → 4× faster weight streaming | ~2–3× TPOT improvement at small batch |
| Speculative decoding | `γ` tokens verified per weight load instead of 1 | Up to `γ×` theoretical speedup (acceptance rate limited) |
| Better hardware | H100 HBM3: 3.35 TB/s vs A100's 2 TB/s → 1.7× decode throughput | Directly proportional to bandwidth ratio |

The bandwidth wall is why INT4 quantisation matters so much for serving: it directly cuts the dominant cost. For a single user chatbot (batch=1), INT4 is essentially free compute — you're just loading 4× fewer bytes at the same bandwidth ceiling.

### KV Cache Pressure
{: #slow-kv}

**The problem at long context.** As context grows, the KV cache for the sequence grows. Each decode step must stream not just model weights but also the entire KV cache accumulated so far. At 4K context on Llama-3 8B (32 layers, 32 KV heads, d_head=128, FP16):

```
KV cache size = 2 × 32 × 32 × 128 × 4096 × 2 bytes = 2 GB
Weight size = ~16 GB

Step time ∝ (16 GB weights + 2 GB KV) / 2 TB/s ≈ 9ms
```

At 32K context, KV grows to 16 GB — equal to the weights. At 128K context, KV is 64 GB and dominates:

```
Step time ∝ (16 GB + 64 GB) / 2 TB/s ≈ 40ms  → 25 tokens/second ceiling
```

**How to handle long context:**

| Technique | Mechanism | Tradeoff |
|---|---|---|
| KV quantisation (INT8) | Halve KV bytes | Slight accuracy loss on long-range dependencies |
| GQA (fewer KV heads) | n_kv_heads=8 vs 32: 4× smaller KV | Baked into architecture, not runtime config |
| KV eviction (H₂O, StreamingLLM) | Drop low-importance KV entries | Accuracy risk on evicted tokens |
| Prefix caching | Don't recompute shared prefix → don't re-stream it | Only helps for shared prefixes |
| Sliding window attention | Only keep last W tokens in cache | Loses long-range context entirely |

**The batch size vs context tradeoff.** A fixed GPU memory budget must be split between model weights, activations, and KV cache. Longer contexts leave less room for KV from other requests, shrinking the effective batch size. This is the "context-throughput curve": throughput is roughly constant at short contexts, then falls as context length grows and batch size shrinks. At 128K context on a single A100 with a 13B model, you may be forced to batch=1.

### Sequential Decode Dependency
{: #slow-sequential}

**Why you cannot parallelise output generation.** Token `t+1` depends on token `t` — this is fundamental to the autoregressive formulation. You cannot generate token 5 without knowing tokens 1–4. No matter how many GPUs you have, a single sequence's decode is sequential.

**This means latency (TPOT) has a hard floor.** For a single request, TPOT ≈ `(weights + KV) / HBM_bandwidth` — you cannot beat this for a single sequence, regardless of parallelism.

**Parallelism helps throughput, not single-request latency.** Tensor parallelism (TP) across multiple GPUs reduces weight streaming time per GPU (each GPU holds `1/TP` of the weights) but adds communication overhead (AllReduce per layer). TP=4 on an H100 NVLink cluster reduces per-GPU weight bytes 4× but adds ~1ms AllReduce per layer — the latency gain is often small for TP > 4.

**The speculative decoding escape hatch.** The only way to generate multiple tokens per LLM forward pass. A small draft model generates `γ` candidate tokens; the LLM verifies all `γ` in one forward pass. If all accepted, you produced `γ` tokens at the cost of one weight load. Acceptance rate of 0.8 with `γ=4` gives an expected 3.2 tokens per LLM call → 3.2× TPOT improvement. The cost: running the draft model (small) and the overhead of verification logic.

### CPU Scheduling Overhead
{: #slow-cpu}

**How big is the problem?** Profiled on Llama-3 8B (H100): 38% GPU compute, 33% HTTP server, 29% scheduling. The GPU is idle 62% of the time.

**Root causes:**
- Python GIL: the API server and engine in the same process compete for the lock. The API server blocks the engine during request parsing, tokenisation, and response streaming.
- Synchronous scheduling: after each GPU step, Python runs the scheduler (check completions, free blocks, admit new requests, allocate blocks, build next batch) before the GPU can start the next step.
- Output processing: detokenisation and streaming updates happen in the hot path.

**The fix: overlap and separate.**

```
Without pipelining:
  GPU[step t] → CPU[schedule step t+1] → GPU[step t+1] → CPU[schedule t+2] ...
  GPU idle during CPU scheduling

With multi-step scheduling + process separation:
  GPU[step t] → GPU[step t+1] → GPU[step t+2] → ...
  CPU[process t results] → CPU[process t+1 results] → ...  (parallel)
```

Multi-step scheduling schedules `N` steps ahead. The GPU runs continuously; the CPU processes outputs asynchronously. Process separation (API server ↔ engine over ZMQ) eliminates GIL contention entirely.

The 28% throughput improvement on Llama-3 70B from vLLM v0.6 is entirely from these CPU-side changes — no model changes, no new kernels.

> **Interview question:** A senior engineer says "just use a bigger GPU." Walk them through why H100 doesn't solve the inference slowness problem end-to-end.
>
> *H100 vs A100 gains break down by bottleneck. (1) Compute: H100 BF16 tensor cores are ~5× faster than A100 (1 PFLOPS vs 312 TFLOPS). This helps prefill (compute-bound) enormously. (2) Memory bandwidth: H100 HBM3 is 3.35 TB/s vs A100's 2 TB/s — only 1.7× more. Since decode is memory-bandwidth-bound, decode latency improves by at most 1.7×, not 5×. So upgrading to H100 makes prefill dramatically faster but decode only 1.7× faster. (3) KV cache pressure: H100 has 80GB HBM (same as A100 80GB variant). Same batch size limit at long context. (4) CPU overhead: H100 does nothing for Python GIL contention, scheduling latency, or HTTP server overhead — still 62% idle time until you fix the software stack. (5) Cost: H100 is 3–5× more expensive than A100 per hour. For a decode-heavy workload, you're paying 4× more for 1.7× decode throughput improvement. The real levers for decode throughput are: quantisation (cut weight bytes), larger batch (amortise weights), CPU pipeline (eliminate scheduling idle time), and speculative decoding (multiple tokens per weight load). H100 is the right upgrade for prefill-heavy workloads — not decode-heavy ones.*

---

## Kubernetes for LLM Serving
{: #kubernetes}

### Pods, Services, Deployments
{: #k8s-primitives}

**Pod** — the smallest schedulable unit. For LLM serving, one pod typically wraps one vLLM process along with its GPU allocation. The pod spec declares the container image, resource requests/limits, health probes, and volume mounts. Everything else in Kubernetes is about managing groups of pods.

**Why GPU requests must equal limits.** CPU and memory are time-sliced — Kubernetes can overcommit them. GPUs are not. If `requests.nvidia.com/gpu < limits.nvidia.com/gpu`, the scheduler sees the request value and places the pod on a node that has that many free GPU slots. At runtime the container can try to access more GPUs than the node reserved for it, causing failures or silently sharing with another container. Always set `requests == limits` for GPU:

```yaml
resources:
  requests:
    nvidia.com/gpu: "2"
    memory: "160Gi"
    cpu: "16"
  limits:
    nvidia.com/gpu: "2"   # must match requests
    memory: "160Gi"
    cpu: "16"
```

**Service** — a stable DNS name and virtual IP in front of a dynamic set of pods. When pods restart and get new IPs, the Service endpoint stays constant. Three types matter for LLM serving:

| Type | DNS / IP | Use case |
|---|---|---|
| `ClusterIP` | Stable VIP, kube-proxy load-balances | Standard: any pod in cluster reaches `vllm-svc:8000` |
| `Headless` (`clusterIP: None`) | DNS returns individual pod IPs | Client-side routing to specific pods for KV cache affinity |
| `LoadBalancer` | Provisions cloud LB with external IP | Expose serving endpoint outside the cluster |

For KV cache affinity (sending a user's requests to the same pod that has their context cached), use a headless Service so the routing layer can resolve and target individual pod IPs rather than a shared VIP.

**Deployment** — declares the desired state of a replicated set of pods and manages rolling updates. The key fields:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-70b
  namespace: llm-serving
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1          # allow 1 extra pod during update (3+1=4 total)
      maxUnavailable: 0    # never drop below 3 ready pods
  selector:
    matchLabels:
      app: vllm-server
      model: llama-70b
  template:
    metadata:
      labels:
        app: vllm-server
        model: llama-70b
    spec:
      # ... pod spec
```

`maxUnavailable: 0` is non-negotiable for GPU serving. If it were 1, Kubernetes could terminate an old pod before the new one is ready — during that window, GPU serving capacity drops below the required replicas. With large models that take 5–10 minutes to load, that window is long.

### GPU Scheduling
{: #gpu-scheduling}

The **NVIDIA Device Plugin** runs as a DaemonSet on every GPU node. It advertises `nvidia.com/gpu` as an extended resource in node status and handles the device injection when a pod is scheduled. Without it, `nvidia.com/gpu` requests are unrecognised and pods are never scheduled.

**Node affinity — target specific GPU SKUs.** The device plugin labels nodes with GPU model information. Use these labels to pin pods to the right hardware:

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: nvidia.com/gpu.product
          operator: In
          values:
          - A100-SXM4-80GB
          - H100-SXM4-80GB
```

`required` means the pod won't schedule anywhere else. Use `preferred` (with a weight) when you have a fallback GPU tier you'll accept.

**Taints and tolerations — dedicate GPU nodes.** Without taints, any pod (including CPU-only workloads) can land on expensive GPU nodes. Taint the GPU nodes so only GPU-tolerating pods schedule there:

```bash
kubectl taint nodes gpu-node-1 nvidia-gpu=true:NoSchedule
```

```yaml
tolerations:
- key: nvidia-gpu
  operator: Equal
  value: "true"
  effect: NoSchedule
```

**Topology spread — distribute across failure domains.** A three-replica deployment with all pods on the same node fails entirely when that node goes down. Spread across nodes and availability zones:

```yaml
topologySpreadConstraints:
- maxSkew: 1
  topologyKey: topology.kubernetes.io/zone
  whenUnsatisfiable: DoNotSchedule
  labelSelector:
    matchLabels:
      app: vllm-server
- maxSkew: 1
  topologyKey: kubernetes.io/hostname
  whenUnsatisfiable: ScheduleAnyway
```

`maxSkew: 1` means at most 1 more pod per zone than any other zone. `DoNotSchedule` enforces strictly; `ScheduleAnyway` is best-effort.

**Pod anti-affinity — never co-locate replicas.** For tensor-parallel workloads, you want all shards of one replica on the same node (for NVLink bandwidth) but each replica on a different node (for fault isolation):

```yaml
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchLabels:
          app: vllm-server
      topologyKey: kubernetes.io/hostname   # no two replicas on same node
```

> **Interview question:** You have 3 vLLM replicas and 3 GPU nodes in 2 AZs (2 nodes in us-east-1a, 1 node in us-east-1b). You apply `maxSkew:1, topologyKey:zone, whenUnsatisfiable:DoNotSchedule`. What happens when you try to schedule the third replica?
>
> *Zone us-east-1a has 2 replicas, zone us-east-1b has 1 replica. Skew = 2 − 1 = 1, which satisfies `maxSkew:1`. The third replica is already scheduled. The issue arises if you try to add a 4th: us-east-1a would have 3, us-east-1b would have 1, skew = 2 — violates the constraint. The 4th pod would remain Pending with a `SchedulingGated` or `Unschedulable` event. Fix: use `whenUnsatisfiable:ScheduleAnyway` for a best-effort spread when you have uneven zone capacity, or pre-balance your GPU node fleet across zones.*

### Autoscaling: HPA vs KEDA
{: #autoscaling}

**Why standard HPA fails for LLM workloads.** HPA with CPU or memory metrics is wrong because:
- GPU inference is GPU-bound, not CPU-bound. CPU utilisation on a vLLM pod may sit at 20–30% even when the GPU is fully saturated and users are queuing.
- Memory is pinned to the KV cache and model weights from pod start — it doesn't grow with load. A saturated pod and an idle pod have identical memory usage.

The right signal is what users actually experience: **request queue depth** and **KV cache fill level**.

**KEDA (Kubernetes Event-Driven Autoscaling)** extends HPA with arbitrary external triggers. It reads Prometheus metrics and scales Deployments directly:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
  namespace: llm-serving
spec:
  scaleTargetRef:
    name: vllm-llama-70b
  minReplicaCount: 2        # never go below 2 — cold start risk
  maxReplicaCount: 10
  cooldownPeriod: 300       # wait 5min before scaling down (avoid thrash)
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      query: |
        sum(vllm:num_requests_waiting{model="llama-70b"})
      threshold: "10"       # scale up if >10 requests waiting
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      query: |
        avg(vllm:gpu_cache_usage_perc{model="llama-70b"})
      threshold: "85"       # scale up if average GPU KV cache >85% full
```

vLLM exposes these on its `/metrics` Prometheus endpoint:

```
vllm:num_requests_waiting      — requests in scheduler queue
vllm:num_requests_running      — active requests on GPU
vllm:gpu_cache_usage_perc      — % of KV cache memory used
vllm:e2e_request_latency_seconds_bucket  — latency histogram
```

**Scale-down aggressiveness.** GPU nodes take 5–15 minutes to provision and GPU pods take 2–10 minutes to load models. Scale down too fast → cold start latency on the next traffic spike. `cooldownPeriod: 300` keeps newly-scaled pods alive for 5 minutes before scaling back down.

### Rolling Updates and Canary
{: #rolling-canary}

**Rolling update flow with `maxUnavailable:0`.** Kubernetes creates one new pod (`maxSurge:1`), waits for it to pass its `startupProbe` and `readinessProbe`, then terminates one old pod. This repeats until all replicas are on the new version. The new pod must be fully Ready before an old pod is removed — for a 70B model loading in 8 minutes, each step of the rollout takes at least 8 minutes. A 3-replica deployment takes ~24 minutes to fully roll.

**PodDisruptionBudget** prevents node drain or cluster autoscaler from destroying replicas during maintenance, on top of the rolling update protection:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: vllm-pdb
  namespace: llm-serving
spec:
  minAvailable: 2           # at least 2 pods must be available at all times
  selector:
    matchLabels:
      app: vllm-server
```

**Canary deployment.** Run two Deployments in parallel with different version labels, then split traffic at the routing layer:

```yaml
# Stable Deployment — 3 replicas, old image
metadata:
  name: vllm-llama-stable
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: vllm-server
        version: stable

---
# Canary Deployment — 1 replica, new image
metadata:
  name: vllm-llama-canary
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: vllm-server
        version: canary
```

Split traffic at the Gateway API layer:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: vllm-canary-split
spec:
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /v1
    backendRefs:
    - name: vllm-llama-stable
      port: 8000
      weight: 95
    - name: vllm-llama-canary
      port: 8000
      weight: 5
```

Monitor error rates and latency per version (`by (version)` in Prometheus), gradually shift weight to canary, then replace stable once metrics are clean.

### Cold Start Problem
{: #cold-start}

The cold start problem for LLMs is worse than for any other workload. Model weights must be loaded from storage into GPU HBM before the first token can be generated. Loading times on typical NFS-backed PVCs:

| Model | Weights | Load time (NFS ~90 MB/s) |
|---|---|---|
| 7B (FP16) | 14 GB | ~2.5 min |
| 13B (FP16) | 26 GB | ~5 min |
| 70B (FP16) | 140 GB | ~25 min |
| 70B (INT4) | 35 GB | ~6 min |

During this time, the pod is alive but cannot serve requests. Kubernetes must not route traffic to it, and must not restart it thinking it's hung.

**The three-probe pattern:**

```yaml
# startupProbe: gates traffic until model is loaded
# failureThreshold × periodSeconds = maximum startup budget
startupProbe:
  httpGet:
    path: /health
    port: 8000
  failureThreshold: 60    # 60 × 10s = 10 minutes budget
  periodSeconds: 10
  timeoutSeconds: 5

# readinessProbe: removes pod from Service if it becomes unhealthy after load
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  periodSeconds: 5
  failureThreshold: 3     # 3 consecutive failures → NotReady
  timeoutSeconds: 3

# livenessProbe: restarts pod only if it's truly stuck (not just slow)
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 300   # don't check for 5 minutes
  periodSeconds: 30
  failureThreshold: 2
```

The `startupProbe` is the key piece. Until it succeeds, the `livenessProbe` is suppressed — Kubernetes won't restart the pod for being "slow". Once the startup probe succeeds, the readiness probe takes over for continuous health monitoring.

**Mitigation strategies:**

<div class="post-flow" role="group" aria-label="Cold start mitigation strategies">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>minReplicas ≥ 2</strong> — never scale to zero. Always keep warm replicas. The cheapest mitigation: idle GPU cost vs 25-minute cold start on every traffic resumption.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Local NVMe cache</strong> — if the GPU node has local NVMe, copy weights from NFS to local disk on first load. Subsequent pods on the same node load from NVMe (~1–2 GB/s) instead of NFS (~90 MB/s): 70B loads in ~2 minutes instead of 25.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Quantised weights</strong> — INT4 weights are 4× smaller. 70B INT4 = 35 GB → ~6 min on NFS vs 25 min for FP16. Start-up time directly proportional to bytes loaded.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Init container pre-fetch</strong> — run an init container that pulls weights to a shared volume before the vLLM container starts. Init containers run sequentially; the main container only starts once init succeeds.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted"><strong>Node affinity to warm nodes</strong> — label nodes that have the model weights already cached locally; prefer scheduling new pods there.</span></li>
  </ol>
</div>

**GPU utilisation accounting.** Each warm idle replica consumes GPU memory (model weights pinned in HBM) but has near-zero compute utilisation between requests. For a 70B FP16 model, 140 GB of HBM is permanently occupied per replica. On an 80GB A100, this requires 2 GPUs per replica just for weights. Scale-down decisions must account for the cost of future cold starts, not just current idle cost.

### Deploying vLLM on Kubernetes
{: #deploy-vllm}

The full production answer to "how do you deploy vLLM on Kubernetes?" — every component and why it's there:

```yaml
# 1. Namespace isolation
apiVersion: v1
kind: Namespace
metadata:
  name: llm-serving

---
# 2. Model weights PVC (ReadWriteMany — multiple pods can mount)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llama-70b-weights
  namespace: llm-serving
spec:
  accessModes: [ReadWriteMany]
  storageClassName: fast-nfs
  resources:
    requests:
      storage: 150Gi

---
# 3. PodDisruptionBudget — prevent involuntary evictions
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: vllm-pdb
  namespace: llm-serving
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: vllm-server

---
# 4. Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-70b
  namespace: llm-serving
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: vllm-server
      model: llama-70b
  template:
    metadata:
      labels:
        app: vllm-server
        model: llama-70b
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      terminationGracePeriodSeconds: 120
      tolerations:
      - key: nvidia-gpu
        operator: Equal
        value: "true"
        effect: NoSchedule
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                - A100-SXM4-80GB
                - H100-SXM4-80GB
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app: vllm-server
            topologyKey: kubernetes.io/hostname
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: vllm-server
      initContainers:
      - name: wait-for-weights
        image: busybox:latest
        command: [sh, -c, "until [ -f /models/config.json ]; do sleep 5; done"]
        volumeMounts:
        - name: weights
          mountPath: /models
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.6.0
        args:
        - --model=/models
        - --tensor-parallel-size=2
        - --gpu-memory-utilization=0.92
        - --max-model-len=8192
        - --port=8000
        resources:
          requests:
            nvidia.com/gpu: "2"
            memory: "160Gi"
            cpu: "16"
          limits:
            nvidia.com/gpu: "2"
            memory: "160Gi"
            cpu: "16"
        ports:
        - containerPort: 8000
        startupProbe:
          httpGet:
            path: /health
            port: 8000
          failureThreshold: 60    # 10 minutes budget
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          periodSeconds: 5
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 300
          periodSeconds: 30
          failureThreshold: 2
        lifecycle:
          preStop:
            exec:
              command: [/bin/sh, -c, sleep 15]   # drain in-flight requests
        volumeMounts:
        - name: weights
          mountPath: /models
      volumes:
      - name: weights
        persistentVolumeClaim:
          claimName: llama-70b-weights

---
# 5. Service
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama-svc
  namespace: llm-serving
spec:
  selector:
    app: vllm-server
    model: llama-70b
  ports:
  - port: 8000
    targetPort: 8000

---
# 6. KEDA autoscaler on queue depth
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-scaler
  namespace: llm-serving
spec:
  scaleTargetRef:
    name: vllm-llama-70b
  minReplicaCount: 2
  maxReplicaCount: 10
  cooldownPeriod: 300
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring:9090
      query: sum(vllm:num_requests_waiting{model="llama-70b"})
      threshold: "10"
```

> **Interview question:** You deploy this and a rollout gets stuck — one pod is Pending for 20 minutes. Walk through your diagnosis.
>
> *`kubectl describe pod <stuck-pod>` is the first tool. Look at the Events section at the bottom. Common causes: (1) `0/3 nodes are available: 3 Insufficient nvidia.com/gpu` — the cluster doesn't have enough free GPU slots. Check `kubectl describe nodes | grep nvidia.com/gpu` to see Allocatable vs Requests. If all GPUs are in use, the pod waits until one frees up (rolling update stalls). Fix: increase maxSurge or add a node. (2) `MatchNodeSelector failed` — nodeAffinity is too restrictive and no matching node exists. Check that gpu.product label is spelled correctly. (3) `PVC not bound` — the PVC is still Pending because the StorageClass can't provision on the target zone. Check `kubectl get pvc`. (4) `TopologySpread cannot satisfy constraints` — not enough nodes in the required zones. For the rollout stuck specifically: check if the new pod's startupProbe is failing. If the pod is Running but not Ready, `kubectl logs <pod>` often shows the model loading error (OOM during weight loading, CUDA driver version mismatch, missing HuggingFace token).*

---

## L7 Load Balancing and Envoy
{: #envoy}

### L4 vs L7
{: #l4-vs-l7}

Every proxy operates at a layer of the network stack. The distinction between L4 and L7 determines what it can see — and what routing decisions it can make.

**L4 (Transport layer)** works with TCP and UDP connections. It sees: source IP, destination IP, port numbers, connection state. It cannot see HTTP headers, request bodies, or URL paths — those are payload bytes, opaque to an L4 proxy. L4 load balancers distribute connections (not requests) round-robin or by least-connections. AWS NLB and kube-proxy operate at L4.

**L7 (Application layer)** parses the HTTP protocol. It sees: method, path, headers, request body. It can route on any of these fields. Envoy, NGINX, and AWS ALB operate at L7.

**Why L7 is required for LLM serving:**

```
POST /v1/chat/completions HTTP/1.1
Host: api.example.com
X-Model: llama-70b              ← L4 never sees this
Content-Length: 1820
Authorization: Bearer sk-...

{"model": "llama-70b", "messages": [...], "max_tokens": 500}
```

An L4 proxy sees `api.example.com:443` and forwards the TCP bytes to one of several backends. It cannot distinguish a request for `llama-70b` from one for `mistral-7b` — those models run on different GPU pods and must go to different clusters. Only an L7 proxy can read `X-Model: llama-70b` from the header and route accordingly.

Streaming (Server-Sent Events, chunked transfer) also requires L7 awareness. The proxy must maintain the long-lived HTTP connection, forward chunked response bodies token by token without buffering, and handle the semantic end of the stream (final chunk). An L4 proxy just forwards bytes — it can do this, but you lose the ability to apply per-request policies (timeouts, retries, rate limits) because there are no request boundaries to attach policy to.

| | L4 | L7 |
|---|---|---|
| Routing granularity | Per connection | Per request |
| Can read headers | No | Yes |
| Model-based routing | No | Yes |
| Per-request timeout | No | Yes |
| Circuit breaking | Connection-level | Request-level |
| Rate limiting | Connection rate | Request rate, token rate |
| Latency overhead | Sub-millisecond | 1–5ms (HTTP parsing) |

For LLM serving, the 1–5ms L7 overhead is irrelevant against generation times of seconds to minutes.

### Envoy Architecture
{: #envoy-arch}

Envoy is a high-performance L7 proxy written in C++. Its internal request pipeline:

```
Downstream (client)
       │
  Listener                  binds to 0.0.0.0:443
       │
  Filter Chain              TLS termination, connection-level filters
       │
  HTTP Connection Manager   parses HTTP/1.1, HTTP/2, gRPC
       │
  HTTP Filters              (ordered pipeline):
    ├── JWT auth             validate API key
    ├── Rate limiter         count requests / tokens
    ├── Ext processor        call external service for custom logic
    └── Router               match request to cluster
       │
  Cluster                   named pool of upstream endpoints
       │
  Load Balancer             round-robin / least-requests / ring-hash
       │
  Endpoint                  individual pod IP:port
       │
Upstream (vLLM pod)
```

**xDS — dynamic configuration without restart.** All of Envoy's config (listeners, routes, clusters, endpoints) can be updated at runtime via gRPC push from a control plane. This is the `xDS` protocol family:

| xDS API | Manages | Example update |
|---|---|---|
| LDS | Listeners | "listen on a new port" |
| RDS | Route tables | "add route for new model" |
| CDS | Cluster definitions | "new backend pool for llama3-8b" |
| EDS | Endpoints in clusters | "pod IP 10.0.0.5 joined cluster" |

When a new vLLM pod passes its readiness probe, Kubernetes updates the Service endpoints. Envoy's control plane (e.g. Envoy Gateway) detects this and pushes an EDS update: the new pod IP is added to the cluster. Envoy starts routing to it within seconds — no restart, no config reload.

### Routing, Retries, Timeouts
{: #routing}

**Header-based routing by model:**

```yaml
# Virtual host with multiple route rules
virtualHosts:
- name: llm-api
  domains: ["api.example.com"]
  routes:
  - name: route-llama-70b
    match:
      prefix: "/v1/"
      headers:
      - name: x-model
        stringMatch:
          exact: llama-70b
    route:
      cluster: cluster-llama-70b
      timeout: 300s
  - name: route-llama-8b
    match:
      prefix: "/v1/"
      headers:
      - name: x-model
        stringMatch:
          exact: llama-8b
    route:
      cluster: cluster-llama-8b
      timeout: 120s
  - name: route-embeddings
    match:
      prefix: "/v1/embeddings"
    route:
      cluster: cluster-embeddings
      timeout: 30s
```

**Canary via weighted clusters:**

```yaml
route:
  weightedClusters:
    clusters:
    - name: cluster-llama-70b-stable
      weight: 95
    - name: cluster-llama-70b-canary
      weight: 5
  timeout: 300s
```

**Timeout configuration for LLMs.** Default HTTP timeouts in most proxies are 30–60 seconds — correct for web APIs, catastrophic for LLMs. A 500-token response at 20 tokens/second takes 25 seconds. A 2000-token response at 10 tokens/second takes 200 seconds.

```yaml
route:
  cluster: cluster-llama-70b
  timeout: 300s              # total response time budget
  retryPolicy:
    numRetries: 2
    perTryTimeout: 150s      # each attempt gets 150s before retry
    retryOn: "5xx,reset,connect-failure"
```

**When NOT to retry.** Streaming responses must not be retried after the response has started. If the client has received tokens 1–50 and the connection drops, retrying would restart generation from token 1 — the client already consumed tokens 1–50 and would see a duplicate or inconsistent stream. Only retry on pre-response errors (connection failures, 5xx before first byte):

```yaml
retryPolicy:
  retryOn: "reset,connect-failure,503"   # not "5xx" (which catches mid-stream failures)
  numRetries: 2
  perTryTimeout: 300s                    # don't rely on retries for timeouts
```

**Retry budget** prevents retry storms. If 100 requests all time out simultaneously and each retries 3 times, you send 400 requests to an already-overloaded backend. Limit retries to a fraction of active requests:

```yaml
retryPolicy:
  retryOn: "reset,connect-failure,503"
  numRetries: 2
  retryHostPredicate:
  - name: envoy.retry_host_predicates.previous_hosts   # don't retry on same host
```

### Circuit Breaking
{: #circuit-breaking}

Circuit breaking limits the number of requests a cluster will accept, preventing overloaded GPU pods from being buried under a backlog they can never drain.

**Threshold-based circuit breaker (max queue depth):**

```yaml
circuitBreakers:
  thresholds:
  - priority: DEFAULT
    maxConnections: 100          # concurrent TCP connections per cluster
    maxPendingRequests: 200      # requests queued waiting for a connection slot
    maxRequests: 500             # total concurrent active requests
    maxRetries: 20
    trackRemaining: true
```

When `maxPendingRequests` is hit, Envoy returns `503 Service Unavailable` immediately instead of queuing further. The client gets a fast failure rather than a slow timeout.

**Outlier detection (endpoint-level ejection):**

```yaml
outlierDetection:
  consecutive5xx: 5              # 5 consecutive 5xx → eject the endpoint
  interval: 10s                  # evaluation window
  baseEjectionTime: 60s          # ejected for at least 60 seconds
  maxEjectionPercent: 50         # never eject more than 50% of endpoints
  enforcingConsecutive5xx: 100   # enforce 100% of the time
```

When a vLLM pod is overloaded, it returns 503. After 5 consecutive 503s, Envoy stops sending traffic to that endpoint for 60 seconds. The pod can drain its current queue without receiving new requests. After `baseEjectionTime`, the endpoint is re-admitted and tested with a small fraction of traffic.

**Why `maxEjectionPercent: 50` matters.** If all pods are struggling, you don't want to eject all of them and have Envoy return 503 to every client. Capping at 50% ensures at least half the fleet stays in rotation even under broad degradation.

> **Interview question:** Your GPU cluster gets a traffic spike — all 4 vLLM pods start returning 503. The circuit breaker ejects all 4 (maxEjectionPercent wasn't set). Now Envoy has no healthy endpoints and returns 503 to every new request. KEDA scales up, but the new pod takes 8 minutes to load. How do you prevent this from happening?
>
> *Three fixes. First, set `maxEjectionPercent: 50` — at least 2 pods stay in rotation even when overloaded, returning 503 slowly rather than Envoy returning 503 instantly with no backend. Second, fix the root cause: vLLM's `/health` endpoint should return 503 only when truly unable to accept requests (KV cache full, OOM risk), not just when queue is long. A long queue is normal under load; the pod should still receive requests and process them. Third, implement readiness-based backpressure: vLLM returns 503 on `/health` when queue > threshold → pod removed from Service → Envoy routes to other pods. This is different from outlier detection ejection — the pod signals its own unavailability cleanly, and re-signals availability when its queue drains. The KEDA scale-up is the right long-term response; the circuit breaker is a short-term safety valve, not a replacement for scaling.*

### Rate Limiting
{: #rate-limiting}

**Local rate limiting (per Envoy process, no coordination):**

```yaml
httpFilters:
- name: envoy.filters.http.local_ratelimit
  typedConfig:
    tokenBucket:
      maxTokens: 500           # burst: allow 500 requests at once
      tokensPerFill: 100       # sustained: 100 req/s refill
      fillInterval: 1s
    filterEnabled:
      defaultValue:
        numerator: 100
        denominator: HUNDRED
```

Local rate limiting is fast (in-process, no network round-trip) but imprecise under multiple Envoy replicas — each instance has its own token bucket, so total throughput is `rate × number_of_Envoy_instances`.

**Global rate limiting (Redis-backed, coordinated):**

```yaml
httpFilters:
- name: envoy.filters.http.ratelimit
  typedConfig:
    domain: llm_api
    rateLimitService:
      grpcService:
        envoyGrpc:
          clusterName: rate_limit_service
        timeout: 50ms           # fail open if RLS is slow
    enableXRatelimitHeaders: true
```

The rate limit service (e.g. Envoy's open-source `ratelimit` or a custom service) is a gRPC service backed by Redis. All Envoy instances share the same counter. Rules are defined per (user_id, model) tuple, allowing per-user quotas.

**Token-based rate limiting — the LLM-specific problem.** HTTP request count is a bad signal for LLM cost. One request with `max_tokens: 1` costs trivially; one with `max_tokens: 2000` consumes 2000 GPU compute steps. The right unit is output tokens.

Implementation via `ext_proc` (External Processing filter): Envoy calls an external gRPC service with the request headers and body. The service checks the user's token quota against a Redis counter, blocks the request if over-quota, and tags it for post-response accounting. After vLLM returns the response, the `usage.completion_tokens` field is extracted and credited against the user's quota in Redis.

### LLM-Specific Routing
{: #llm-routing}

**Route by model via header.** The standard pattern: clients include `X-Model: <model-name>` in every request. Envoy matches on this header and routes to the correct cluster (different GPU pool per model):

```
X-Model: llama-70b      → cluster-llama-70b    (A100 × 2 per pod)
X-Model: llama-8b       → cluster-llama-8b     (A100 × 1 per pod)
X-Model: llama-3.2-1b   → cluster-llama-1b     (CPU or T4)
```

**Route by context length.** Clients that know their context window can hint via a header. Long-context requests need pods configured with large KV cache budgets; short-context requests can use smaller, faster pods:

```yaml
routes:
- match:
    prefix: "/v1/"
    headers:
    - name: x-context-tokens
      rangeMatch:
        start: 0
        end: 4096
  route:
    cluster: cluster-llama-8k       # standard context pool
- match:
    prefix: "/v1/"
    headers:
    - name: x-context-tokens
      rangeMatch:
        start: 4096
        end: 131072
  route:
    cluster: cluster-llama-128k     # long-context pool, tuned KV budget
```

**Route by GPU load via ext_proc.** The most sophisticated pattern: before routing, query an LLM-aware load balancer service that knows each pod's current queue depth and KV cache usage. The service returns the IP of the least-loaded pod; Envoy sets an upstream override header. This is the architecture that enables true LLM-aware routing — Envoy handles L7 protocol semantics, the external service handles GPU-aware placement:

```
Client request
    │
    ▼
Envoy (ext_proc filter)
    │  headers + metadata
    ▼
LLM Load Balancer Service
    │  queries Prometheus for all pod metrics:
    │  vllm:num_requests_waiting, vllm:gpu_cache_usage_perc
    │  returns: {"route_to_pod": "10.0.1.42:8000"}
    ▼
Envoy sets upstream header
    │
    ▼
vLLM pod at 10.0.1.42
```

**Consistent hashing for KV cache affinity.** When prefix caching is enabled, requests sharing a system prompt should go to the same pod that has that prefix cached. Hash the system prompt (or user session ID) to a stable pod:

```yaml
cluster:
  name: cluster-llama-70b
  lbPolicy: RING_HASH
  ringHashLbConfig:
    minimumRingSize: 1024

route:
  cluster: cluster-llama-70b
  hashPolicy:
  - header:
      headerName: x-session-id    # same session → same pod → KV cache hit
```

### Preventing Overload
{: #prevent-overload}

The full answer to "how do you prevent overload?" is a layered defence — each layer catches what the one above misses:

<div class="post-flow" role="group" aria-label="Overload prevention layers">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Rate limiter (admission)</strong> — reject requests before they reach the GPU fleet. Token bucket enforces sustained rate limits. Global rate limiting enforces per-user quotas. Requests that would exceed budget get 429 immediately.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Circuit breaker (overload detection)</strong> — `maxPendingRequests` caps the queue depth at Envoy. When the queue is full, new requests get 503 without ever touching the GPU. Prevents the queue from growing to infinite depth under a traffic spike.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Readiness backpressure (self-reporting)</strong> — vLLM's `/health` endpoint returns 503 when KV cache is critically full or GPU OOM is imminent. Kubernetes removes the pod from Service endpoints. Envoy stops routing to it. The pod drains its existing work without receiving more.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Outlier detection (unhealthy endpoint ejection)</strong> — if a pod returns consecutive 5xx errors (perhaps it's OOM'd), Envoy ejects it automatically without waiting for the readiness probe cycle.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green"><strong>KEDA autoscaling (capacity expansion)</strong> — when queue depth exceeds the threshold, KEDA triggers scale-out. New pods are admitted to the cluster after their startupProbe succeeds. This is the long-term response; the layers above buy time.</span></li>
  </ol>
</div>

```
Request arrives
    │
[Rate limiter] ──── over quota? ──► 429 Too Many Requests
    │
[Circuit breaker] ── queue full? ──► 503 Service Unavailable
    │
[Route to healthy endpoint]
    │
[vLLM /health check] ── KV OOM risk? ──► pod → NotReady → rerouted
    │
[vLLM processes request]
    │
[KEDA watches queue] ── queue > 10? ──► scale up Deployment
```

**The architecture in one sentence — the line you should be able to say in any interview:**

> **Envoy handles L7 routing: it reads the model header, enforces rate limits, applies circuit breaking, and selects the right backend cluster. The LLM-aware scheduler — whether that's KEDA reacting to queue depth, an ext_proc service querying pod metrics, or a custom inference gateway — decides GPU placement: which pod, which node, and when to add capacity. They operate at different layers and don't communicate directly.**

The complementary stack for OpenShift/enterprise deployments uses the same primitives: OpenShift's built-in Ingress Controller (Envoy-backed HAProxy) for L7 routing, OpenShift's machine autoscaler for node provisioning, and KEDA for workload-driven pod scaling — the architecture is identical, only the operator names change.

> **Interview question:** A client sends 1000 requests/second to your LLM API. You have 4 vLLM pods, each handling 20 req/s max. Walk through what happens end-to-end when the spike hits.
>
> *The fleet capacity is 4 × 20 = 80 req/s. At 1000 req/s, 920 req/s is excess. Layer-by-layer: (1) Rate limiter: if configured at 80 req/s sustained, 920 req/s gets 429 immediately at the Envoy layer — zero GPU impact. This is the ideal case. (2) If no rate limiter: requests reach the circuit breaker. `maxPendingRequests: 200` means Envoy queues 200 and starts returning 503 for everything beyond. Pods receive requests up to their capacity and start building up queue. (3) vLLM's KV cache fills as the queue grows. When utilisation hits the threshold, `/health` returns 503 → pods go NotReady → Envoy ejects them one by one. Now clients all get 503. (4) KEDA detects `vllm:num_requests_waiting > 10`. Scales Deployment from 4 to 10 replicas. But cold start takes 8 minutes. For those 8 minutes, only 4 pods serve, still at 80 req/s capacity. (5) After 8 minutes: 10 pods online, 200 req/s capacity. Still 5× under the spike. KEDA triggers another scale-up. (6) At full scale (maxReplicas=20): 400 req/s. Still 2.5× under 1000 req/s — you hit the cluster's GPU ceiling. The surplus requests get 503 or 429. The lesson: rate limiting at the gateway is the only way to protect the fleet from spikes that exceed max capacity. Autoscaling helps with gradual growth; it cannot absorb instantaneous 12× spikes.*

---

## Performance Optimisation: Reducing Cost Per Token
{: #perf-optimisation}

"Reduce cost per token" is the canonical LLM serving interview question. It has a deterministic answer structure: measure where the cost comes from, identify which bottleneck dominates, apply the matching lever. The roofline model is the diagnostic backbone; everything else is a specific intervention on a specific term.

### Cost Per Token: The Decomposition
{: #cost-decomposition}

GPU cost per token = GPU time per token × GPU cost per second. GPU time per token has three components:

```
T_per_token = T_weights + T_KV + T_overhead

T_weights  = model_bytes / (HBM_bandwidth × batch_size)   [memory-bound, amortised by batch]
T_KV       = KV_cache_bytes / HBM_bandwidth                [grows linearly with context]
T_overhead = kernel_launch + scheduling + CPU sync         [fixed per step]
```

At **batch=1, short context**: `T_weights` dominates. The GPU loads 140 GB of weights and produces 1 token. Everything else is negligible.

At **batch=128, short context**: `T_weights` is amortised 128×. The bottleneck shifts to compute (you approach the roofline crossover) or CPU scheduling overhead.

At **any batch, long context** (32K+ tokens): `T_KV` grows to match or exceed `T_weights`. At 128K context on a 70B model, KV is larger than the weights themselves.

The intervention depends on which term dominates. You cannot fix `T_KV` by increasing batch size.

### Roofline Thinking as Diagnostic
{: #roofline-diagnostic}

Every performance problem in LLM inference maps to one of three positions on the roofline:

```
Achieved performance (FLOP/s)
      │
      │              Compute roof ─────────────── C_peak
      │             /
      │            /
      │           /  (slope = HBM_bandwidth)
      │          /
      │    ●    / ← bandwidth-bound kernel (most decode ops)
      │        *   ← roofline knee (AI = C_peak / BW)
      │            ●  ← compute-bound kernel (prefill, large batch matmul)
      └──────────────────────────────────────────
                    Arithmetic Intensity (FLOP/byte)
```

**Position 1: bandwidth-bound (left of knee).** Decode weight matmul at batch=1 has AI ≈ 1 FLOP/byte. A100 knee is at AI* ≈ 156 FLOP/byte. You're using 0.6% of compute. The fix is anything that **reduces bytes or increases reuse**: quantisation, larger batch, speculative decoding.

**Position 2: compute-bound (right of knee).** Prefill with a 4K prompt is compute-bound — high AI from processing many tokens in parallel. Adding HBM bandwidth doesn't help; you need more FLOPS (H100 > A100) or better kernel efficiency.

**Position 3: overhead-bound (below both ceilings).** Both compute and bandwidth utilisation are low. The bottleneck is CPU scheduling, kernel launch overhead, or idle time between steps. Fix: CUDA Graphs, multi-step scheduling, operator fusion.

**The diagnostic question for any slow kernel:** measure `achieved_FLOP/s` and `achieved_BW`. If `achieved_BW ≈ peak_BW` and `achieved_FLOP/s << peak_FLOP/s`, you're bandwidth-bound. If `achieved_FLOP/s ≈ peak_FLOP/s` and `achieved_BW << peak_BW`, you're compute-bound. If both are far below ceiling, overhead-bound.

> **Interview question:** Your decode throughput is 800 tokens/second on an A100 (2 TB/s HBM, 312 TFLOPS BF16). The 7B model is in FP16 (14 GB). What is the theoretical maximum, and what does the gap tell you?
>
> *Theoretical decode maximum (memory-bound floor): loading 14 GB weights at 2 TB/s takes 7ms per step → 143 tokens/second per stream at batch=1. At batch=8: 8 streams, each costs 7ms → 8/0.007 = 1,143 tokens/second. At batch=128: 128/0.007 = 18,286 tokens/second ceiling. You're achieving 800 tokens/second. If batch=8: ceiling is 1,143, you're at 70% — reasonable, probably overhead or KV bandwidth eating the rest. If batch=128: ceiling is 18K, you're at 4.4% — something is very wrong. Either batch is much lower than 128 in practice, or there's a massive CPU scheduling bottleneck, or KV bandwidth is dominating. Diagnosis: profile one step with `torch.cuda.synchronize()` timing around the forward pass only. If forward pass alone is close to the bandwidth-bound estimate, the overhead is elsewhere (HTTP, scheduling). If the forward pass itself is slow, it's a kernel issue.*

### The Cost-Per-Token Reduction Playbook
{: #cost-playbook}

An ordered decision framework. Apply in sequence — each step should be measured before moving to the next:

<div class="post-flow" role="group" aria-label="Cost reduction sequence">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Step 1 — Measure.</strong> Get baseline: tokens/sec, GPU utilisation (compute + memory BW separately), batch size distribution, queue depth. Don't optimise blind.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Step 2 — Maximise batch size.</strong> The single highest-leverage knob. Same weight bytes, more tokens produced. Check if you're memory-limited (KV cache full) or queue-limited (not enough traffic). Fix: continuous batching, increase max_batch_size, add KV quantisation to free memory for more slots.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Step 3 — Quantise weights.</strong> INT4 cuts weight bytes 4×, shifting the memory-bound floor 4× lower. At the same batch size, you get 4× more decode throughput in the pure bandwidth-bound regime. AWQ/GPTQ for accuracy. FP8 on H100 for full-precision throughput.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Step 4 — Enable prefix caching.</strong> If your workload has shared prefixes (system prompts, shared context), prefix caching eliminates the prefill cost for those tokens. A 2K-token system prompt shared by 1M requests saves 2B tokens of prefill compute.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Step 5 — Add speculative decoding.</strong> Multiple tokens per LLM forward pass. With acceptance rate ~0.8 and γ=4: ~3.2 tokens per weight load instead of 1. Best ROI when the draft model is cheap (same family, 10-50× smaller). Model-free methods (prompt lookup) are zero-cost when they hit.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green"><strong>Step 6 — Fix CPU overhead.</strong> If profiling shows GPU idle time between steps: enable CUDA Graphs, multi-step scheduling (vLLM v0.6+), async output processing. 28% throughput gain reported on Llama-3 70B from software changes alone.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green"><strong>Step 7 — Quantise KV cache.</strong> At long context, T_KV dominates. KV INT8 halves KV bytes → halves KV streaming time → doubles effective max context at same throughput. KV FP8 on H100 is near-lossless.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted"><strong>Step 8 — Disaggregate prefill/decode.</strong> If TTFT is hurting throughput (prefill blocking decode slots), split into separate pools. Hardware specialisation: H100s for prefill (compute-bound), A100s for decode (BW-bound). Only worth the complexity at fleet scale.</span></li>
  </ol>
</div>

Each step has a **different mechanism** and a **different bottleneck** it addresses. Applying step 5 (speculative decoding) when the real bottleneck is CPU overhead (step 6) will show no improvement. Measure after each step.

### Batching: The First Lever
{: #batching-lever}

Batching is the most important single variable. The memory bandwidth cost of a decode step is:

```
T_step = (weight_bytes + KV_bytes) / HBM_bandwidth
```

`weight_bytes` is fixed regardless of batch size. Every additional request in the batch produces one more output token at near-zero marginal cost until you hit the compute-bound crossover. At batch=B, you produce B tokens for the same `weight_bytes` load:

```
tokens_per_second ≈ B / T_step  (memory-bound regime, B < B_crossover)
cost_per_token    ≈ T_step / B  (shrinks linearly with batch)
```

Doubling batch size halves cost per token — until you hit the crossover or run out of KV memory.

**What limits batch size in practice:**

| Constraint | How it manifests | Fix |
|---|---|---|
| KV cache memory full | OOM, requests queue | KV quantisation, PagedAttention, smaller context |
| Not enough requests (low traffic) | Batch stays at 1–4 | Accept it; keep minReplicas low |
| Prefill blocks decode slots | New requests can't join batch | Chunked prefill |
| CPU scheduling too slow | GPU idle between steps | CUDA Graphs, multi-step scheduling |

**Continuous batching** keeps the batch near maximum at all times by filling empty slots as soon as requests complete. Without it, static batching wastes 60–80% of potential throughput as finished requests leave slots idle until the full batch completes.

### KV Cache Reuse
{: #kv-reuse}

KV cache reuse is prefix caching applied systematically. The cost saving is direct: reused KV tokens require zero prefill compute.

**Where reuse is high-value:**

```
Scenario                        Reuse ratio         Savings
─────────────────────────────────────────────────────────────
System prompt (2K tokens)       (N-1)/N per request  ~100% at scale
Multi-turn chat (avg 5 turns)   history/total ≈ 80%  ~80% of prefill
RAG with fixed retrieved docs   ~60-90%              large
One-off diverse queries         ~0%                  none
```

**Implementation.** vLLM's block-level prefix caching hashes 16-token blocks; SGLang's RadixAttention uses a radix tree for token-level matching. For high-reuse workloads (fixed system prompt + many users), SGLang's approach is strictly better — it reuses the exact system prompt KV across every request with a single tree lookup.

**The routing dependency.** Prefix caching only works if the request that needs a cached prefix is routed to the pod that has it cached. This requires consistent-hash routing at the load balancer keyed on the system prompt hash or session ID. A round-robin router delivers 0% cache hit rate even with caching enabled.

**KV transfer as a caching layer.** In disaggregated serving, the KV cache computed on the prefill worker can be stored in a shared distributed KV store (e.g., Redis, Infinispan, custom RDMA-backed store). Decode workers pull from it on demand. This decouples prefix cache lifetime from individual pod lifetime and enables cross-pod reuse without sticky routing.

### Speculative Decoding
{: #speculative-decoding-perf}

Speculative decoding breaks the one-token-per-weight-load constraint by using a cheap draft model to propose multiple tokens, verified in one LLM forward pass.

**The math.** With draft length γ and acceptance rate α:
```
E[accepted tokens per LLM call] = (1 - α^(γ+1)) / (1 - α)

At α=0.8, γ=4:  E = (1 - 0.8^5) / (1 - 0.8) = (1 - 0.328) / 0.2 = 3.36
```
You get 3.36 tokens per LLM forward pass on average. The LLM weight load is amortised 3.36× → cost per token drops 3.36×.

**When speculation works:**
- Task is predictable (factual Q&A, code completion in established patterns, translation)
- Draft model is from the same family as the LLM (same tokeniser, similar pretraining)
- Temperature is low (greedy/near-greedy sampling → higher acceptance rate)
- Batch size is small (at large batch, the LLM is already compute-bound; extra tokens don't reduce cost proportionally)

**When it doesn't:**
- Creative/open-ended generation (LLM distribution diverges from draft)
- High temperature (random outputs, draft guesses poorly)
- Already at large batch (cost is compute-bound, not weight-bound; extra tokens add compute)
- Draft model is too expensive (if SSM is 1/5 the cost of LLM, total overhead is 1 + 1/5 per speculation cycle)

**Model-free speculative decoding.** Prompt lookup decoding searches the input context for the last N tokens and copies the following K tokens as draft candidates. Zero GPU cost, zero memory overhead, zero accuracy risk (LLM verifies everything). Works wherever output echoes input: summarisation, RAG answers, code editing. Run it always — it's free when it misses.

**EAGLE / Medusa.** EAGLE adds a single shallow attention layer that predicts the LLM's next hidden state, builds a token tree, and verifies with the LLM. Medusa adds multiple auxiliary decoding heads to the LLM, each predicting further-ahead tokens. Both share the main LLM's embedding/LM-head weights, so they add minimal memory. EAGLE achieves 2.5–3.5× speedup on coding and instruction-following tasks.

> **Interview question:** You have a Llama-3 70B (BF16) serving API on 2× A100 (80GB each, 2 TB/s each). Batch size stays around 4 due to low traffic. Speculative decoding with a 7B draft model achieves α=0.8, γ=4. Estimate the cost-per-token improvement, and identify what else you should do first.
>
> *First, check whether speculative decoding is even the right lever. At batch=4 BF16 on 2×A100: arithmetic intensity ≈ 4 FLOP/byte, far below the crossover (~156). You're firmly bandwidth-bound. The weight streaming cost: 140 GB / (2 × 2 TB/s) = 35ms per step → 4 tokens per step → ~114 tokens/second. With speculative decoding at α=0.8, γ=4: E[tokens per LLM call] = 3.36. Throughput: 3.36 × 1000/35 ≈ 96 tokens/second — wait, that's worse. Why? The 7B draft model also costs time. 7B at batch=4 BF16 on 2×A100: weight streaming = 14 GB / 4 TB/s = 3.5ms per draft step × 4 draft steps = 14ms. Total per LLM call: 14ms (draft) + 35ms (verify) = 49ms for 3.36 tokens → 68 tokens/second. Speculation actually hurts here because the draft overhead isn't negligible at small batch. The right first step is quantisation: INT4 reduces 70B to 35 GB → 35/4000 = 8.75ms per step → 4/0.00875 = 457 tokens/second. That's a 4× improvement from quantisation alone with zero accuracy discussion needed. After INT4, the crossover drops to batch≈78, you're still bandwidth-bound at batch=4, but throughput improved 4×. Now consider speculative decoding: at INT4, LLM call costs 8.75ms; 7B INT4 draft costs 0.875ms × 4 = 3.5ms; total 12.25ms for 3.36 tokens → 274 tokens/second. Net improvement over INT4 alone: 274/457 = 60% — worse again because draft cost is proportionally high. Conclusion: for batch=4, quantisation first (4× gain), then accept the limitation. Speculative decoding adds value only if you can use a much smaller draft (1B, not 7B) or if traffic grows enough to batch=16+.*

### GPU Utilisation and Memory Bandwidth
{: #gpu-utilisation}

**GPU utilisation is the wrong primary metric.** "GPU utilisation at 95%" sounds good but is ambiguous — it measures whether the GPU is *doing something*, not whether it's doing something *useful*. A GPU running memory-copy operations (data movement, KV cache reads) shows high utilisation but near-zero compute utilisation.

**The right metrics:**

| Metric | What it measures | Tool |
|---|---|---|
| `sm_active_cycles / total_cycles` | Fraction of time SMs are active | Nsight Systems |
| `achieved_bandwidth / peak_bandwidth` | Memory bandwidth utilisation | Nsight Compute |
| `achieved_FLOPS / peak_FLOPS` | Compute utilisation | Nsight Compute |
| MFU = achieved_FLOPS / C_peak | How much of peak FLOPS is useful work | Manual calculation |
| `vllm:gpu_cache_usage_perc` | KV cache fill level | Prometheus |
| Tokens/second/GPU | The actual business metric | vLLM /metrics |

**The memory bandwidth utilisation number you want.** For a decode-dominated workload (bandwidth-bound), you want `achieved_BW / peak_BW ≈ 80–95%`. If it's 30%, you have either a scheduling/batching problem (GPU is idle between steps) or an overhead problem (CPU preparing batches too slowly).

**Compute utilisation intentionally low.** In the bandwidth-bound regime, compute utilisation during decode is 1–5% by design — the GPU is waiting for HBM, not waiting for tensor cores. This is normal, not a problem to fix. Trying to increase compute utilisation in this regime (by adding useless FLOPs) would be wrong.

### Profiling: Nsight Conceptually
{: #profiling}

You don't need to memorise Nsight hotkeys. You need to know what to look for and what the findings mean.

**Nsight Systems** — the timeline view. Shows CPU and GPU activity on a shared timeline. Key questions:

```
Is the GPU idle between kernel launches?
→ Large gaps between GPU kernels = CPU scheduling overhead
→ Fix: CUDA Graphs (eliminates per-kernel Python launch overhead),
        multi-step scheduling (schedule GPU ahead)

Are CPU and GPU running in parallel?
→ If CPU is busy while GPU is idle = CPU is the bottleneck
→ If GPU is busy while CPU is idle = good overlap

How long is each phase?
→ Long prefill blocks = chunked prefill needed
→ Long CPU scheduling = overlap with multi-step scheduling
```

**Nsight Compute** — the kernel-level view. Shows per-kernel memory bandwidth utilisation, compute utilisation, achieved FLOPS, arithmetic intensity. Key questions:

```
For the dominant kernel (usually the attention or MLP matmul):

1. What is achieved_bandwidth vs peak_bandwidth?
   → >80%: good, you're bandwidth-bound and saturating it
   → <40%: either low arithmetic intensity (small batch) or
            irregular access patterns (paged attention gather)

2. What is achieved_FLOPS vs peak_FLOPS?
   → >70%: compute-bound, you're using the tensor cores well
   → <10%: bandwidth-bound (normal for decode), or overhead-bound

3. What is arithmetic intensity?
   → Compute AI from kernel shapes: FLOPs / bytes
   → Compare to hardware knee (156 for A100 BF16, 300 for H100 BF16)
   → Left of knee: memory-bound; right: compute-bound
```

**The five-step profiling workflow:**

<div class="post-flow" role="group" aria-label="Profiling workflow">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Measure end-to-end.</strong> tokens/sec, TTFT P50/P99, TPOT P50/P99. This tells you what's broken but not why.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Timeline with Nsight Systems.</strong> Find where time is going: is the GPU idle? Is CPU dominant? How long are prefill vs decode steps? Identify the phase that dominates.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Kernel-level with Nsight Compute.</strong> Profile the dominant kernel. Measure achieved BW and FLOPS. Compute arithmetic intensity. Classify: bandwidth-bound, compute-bound, or overhead-bound.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Form a hypothesis.</strong> "MLP matmuls are bandwidth-bound because batch=4 gives AI≈4, far below knee at 156." This should be falsifiable.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green"><strong>Apply the matching lever and re-measure.</strong> Increase batch → AI increases. Quantise → bytes decrease, AI increases. Enable CUDA Graphs → kernel launch gaps disappear. Verify the hypothesis held.</span></li>
  </ol>
</div>

**Common findings and their meaning:**

```
Finding                               Meaning                Fix
──────────────────────────────────────────────────────────────────────
GPU idle 40% of time (Nsight Sys.)    CPU scheduling overhead  CUDA Graphs, multi-step scheduling
Bandwidth at 95%, FLOPS at 2%         Bandwidth-bound          Quantisation, larger batch
FLOPS at 80%, bandwidth at 40%        Compute-bound            Better kernels (TensorRT-LLM), TP
Both below 30%                        Overhead-bound           Operator fusion, kernel consolidation
Attention kernel slow at long ctx     KV streaming bound       KV INT8, GQA architecture
Prefill dominates timeline            Long prompts             Chunked prefill, prefix caching
Huge gap between steps (Nsight Sys.)  Python kernel launch     CUDA Graphs
```

> **Interview question:** Nsight Compute shows your MLP matmul achieves 1.8 TB/s memory bandwidth on an A100 (peak: 2.0 TB/s) and 18 TFLOPS compute (peak: 312 TFLOPS). Arithmetic intensity is 2.1 FLOP/byte. Diagnose and prescribe.
>
> *This is clearly bandwidth-bound: 90% memory bandwidth utilisation, 5.8% compute utilisation, AI=2.1 far below the knee at 156. The 90% bandwidth utilisation means we're near the hardware ceiling for this operating point — you cannot squeeze much more throughput from the kernel itself. The relevant question is: why is AI so low? AI = 2 × batch × seq_len × d_model / weight_bytes. At batch=1: AI ≈ 2 × 1 × 1 × d_model / d_model² = 2/d_model ≈ 2 for d_model=1 (roughly). At batch=B: AI ≈ 2B. So AI=2.1 means batch ≈ 1. You're running batch=1 decode. The fix is not kernel tuning — you're already at 90% bandwidth utilisation, which is near-optimal for this AI. The fix is either (a) increase batch size to amortise weight loads across more tokens — if you can get to batch=16, AI≈32, throughput roughly 16× better; (b) quantise weights — INT4 cuts weight bytes 4× → AI × 4 at same batch, throughput 4× better while still bandwidth-bound; (c) both. Don't touch the kernel. The kernel is fine.*

### Synthesis: The "Reduce Cost Per Token" Answer
{: #cost-synthesis}

When asked "how do you reduce cost per token?" in an interview, the answer is a diagnosis loop, not a list of techniques. The structure:

**1. Measure the current operating point.**

```
cost_per_token = (GPU_cost_per_hour × GPUs) / tokens_per_hour

tokens_per_hour = tokens_per_second × 3600
```

Get tokens/sec from vLLM `/metrics`. Get GPU cost from your cloud provider. Now you have a number to beat.

**2. Identify which term in `T_per_token = T_weights + T_KV + T_overhead` dominates.**

- Run a decode step with `torch.cuda.synchronize()` timing. Compare against the theoretical minimum: `weight_bytes / HBM_bandwidth`. If actual ≈ theoretical: you're bandwidth-bound on weights. If actual >> theoretical: overhead.
- At long context: compare against `(weight_bytes + KV_bytes) / HBM_bandwidth`. If KV term dominates, it's the target.

**3. Apply the matching lever.**

| Dominant term | Root cause | Lever |
|---|---|---|
| `T_weights`, small batch | Low arithmetic intensity | Increase batch, quantise weights (INT4) |
| `T_weights`, large batch but still slow | At compute crossover | Better kernels (TRT-LLM), FP8, TP |
| `T_KV` | Long context, KV streaming | KV INT8/FP8, GQA, KV eviction |
| `T_overhead` | CPU scheduling, kernel launch | CUDA Graphs, multi-step scheduling, process separation |
| Prefill cost (TTFT-driven) | Long shared prompts | Prefix caching, chunked prefill |
| All tokens, even short | Sequential decode limit | Speculative decoding |

**4. Quantify before committing engineering time.**

```
Quantisation (INT4):
  weight_bytes: 140 GB → 35 GB
  T_weights: 70ms → 17.5ms per step at batch=1
  Throughput: 14 tokens/sec → 57 tokens/sec
  Cost/token improvement: 4×
  Engineering cost: run AWQ calibration + accuracy eval (1 day)

Speculative decoding (α=0.8, γ=4):
  Effective tokens per LLM call: 3.36
  Throughput improvement: 3.36×
  Cost/token improvement: ~2.5× (draft model overhead eats ~25%)
  Engineering cost: 1 week to integrate draft model + test
  Constraint: only works if acceptance rate holds for your workload

Prefix caching (2K system prompt, 1M requests/day):
  Saved prefill: 2K tokens × 1M = 2B tokens/day
  At 70B model: 2B tokens × 2×70B FLOPs ≈ 280 PFLOPS saved
  Engineering cost: enable in vLLM (one flag), fix routing to use consistent hash (1 day)
  → Highest ROI if you have shared prefixes
```

The best interventions are almost always: (1) quantisation first (large gain, low risk, fast), (2) prefix caching if workload has shared prefixes (potentially free compute), (3) batch size maximisation via continuous batching (operational, not engineering work), (4) speculative decoding if draft model acceptance is validated on your actual traffic.

Hardware upgrades (H100 vs A100) are justified when: prefill is the dominant cost (H100 is 5× faster on compute-bound prefill), or you need the H100's higher HBM3 bandwidth (1.7× vs A100) for extreme batch sizes, or FP8 native execution matters. For decode-heavy, low-batch workloads, H100 costs 3–5× more for 1.7× decode improvement — poor ROI compared to software-level optimisations.

---

## System Design
{: #system-design}

Three fully worked answers for the most common LLM system design questions. Each follows the same five-part structure: requirements → architecture → bottlenecks → scaling → optimisations. Speak in this order — interviewers are pattern-matching against it.

---

### Design an LLM Inference System
{: #design-inference}

#### 1. Requirements
{: #inf-requirements}

Start by scoping — never skip this.

**Functional:**
- Single model or multi-model? (assume multi-model: Llama-3 8B for latency-sensitive, 70B for quality-sensitive)
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`)
- Streaming responses (SSE)
- Context lengths up to 32K tokens

**Non-functional (ask for numbers):**
- Scale: 10K RPM (~167 req/sec)
- Latency SLOs: TTFT P95 < 1s, TPOT P99 < 80ms/token (≈ 12+ tokens/sec)
- Availability: 99.9% (< 8.7 hours/year downtime)
- Cost target: < $0.001 per 1K output tokens

**State your assumptions out loud:**
- Traffic is not uniform — 10× peak/mean ratio is realistic
- Output lengths vary: median 200 tokens, max 2048 tokens
- ~30% of requests share a common system prompt (prefix caching viable)

#### 2. Architecture
{: #inf-architecture}

Draw this diagram, name every box, explain the data flow in one sentence per layer.

```
                        Internet
                            │
                    ┌───────▼────────┐
                    │  API Gateway   │  Auth, TLS, rate limit
                    │  (Envoy L7)    │  route by X-Model header
                    └───────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Model Pool  │  │  Model Pool  │  │  Model Pool  │
    │  Llama-3 8B  │  │  Llama-3 70B │  │  Embeddings  │
    │  vLLM ×4    │  │  vLLM ×2    │  │  CPU/GPU ×2  │
    └──────┬───────┘  └──────┬───────┘  └──────────────┘
           │                 │
    ┌──────▼─────────────────▼──────┐
    │       Inference Router        │  KV-cache-aware LB
    │  (prefix-hash + least-loaded) │  continuous batching
    └───────────────────────────────┘
           │
    ┌──────▼──────┐
    │  GPU Nodes  │  Kubernetes worker nodes
    │  NVIDIA A100│  NVIDIA device plugin
    └─────────────┘
           │
    ┌──────▼──────────────────────┐
    │  Storage Layer              │
    │  Model weights PVC (NFS)    │
    │  Prompt/response cache      │
    └─────────────────────────────┘
```

**Layer-by-layer walkthrough:**

**API Gateway (Envoy).** Every request enters here. Envoy terminates TLS, validates JWT/API-key via ext_authz filter, enforces per-user rate limits (token bucket), and routes by the `X-Model` header to the right model pool. Timeout set to 300s. Streaming response buffering disabled (`x-envoy-upstream-rq-per-try-timeout-ms` not set on streaming routes).

**Inference Router.** A lightweight service sitting between Envoy and the vLLM pods. It implements KV-cache-aware load balancing: hash the system prompt prefix to a consistent backend (maximise cache hits), fall back to least-queue-depth when that backend is saturated. This is the component that makes prefix caching actually work — a pure round-robin Envoy cluster would give 0% cache hit rate.

**vLLM Engine Pods.** One Deployment per model. Each pod runs vLLM with PagedAttention, continuous batching, chunked prefill. Kubernetes Deployment with `maxUnavailable:0`, `startupProbe` with 10-min budget. Prometheus metrics exposed on `/metrics`.

**Kubernetes.** GPU nodes tainted `nvidia-gpu=true`, pods tolerate it. Node affinity on `nvidia.com/gpu.product` pins 8B pools to A100-40GB and 70B pools to A100-80GB. KEDA ScaledObject watches `vllm:num_requests_waiting` with threshold 10, `minReplicas:2` to avoid cold start.

**Storage.** Model weights on ReadWriteMany PVC. Init container copies weights to local NVMe on first start, subsequent starts load from local disk.

#### 3. Bottlenecks
{: #inf-bottlenecks}

Name the bottlenecks in order of when they'll hit as load grows:

**Bottleneck 1: KV cache memory.** As concurrent requests grow, KV cache fills GPU HBM. At 32K context × 32 concurrent requests on Llama-3 8B: KV ≈ 32 × 2 GB = 64 GB — exceeds a single 80GB A100 when combined with 16 GB of weights. Fix: KV INT8 (halves KV), GQA architecture (Llama-3 8B already has GQA), reduce max_model_len.

**Bottleneck 2: Memory bandwidth.** Decode is bandwidth-bound. At batch=16 BF16, AI ≈ 16 — still far below the A100 crossover at 156. Throughput scales with batch but hits the bandwidth ceiling before compute. Fix: quantise to INT4 (35 GB model → 4× more tokens per bandwidth dollar), increase batch.

**Bottleneck 3: Prefill latency at scale.** Long prompts block the GPU for hundreds of ms, spiking TPOT for other users. Fix: chunked prefill (cap per-iteration prefill at 512 tokens), prefix caching (skip prefill for shared system prompts entirely).

**Bottleneck 4: Cold start under scale-out.** KEDA triggers a new pod; it takes 5–8 minutes to load weights. During this window, the cluster remains undersized. Fix: `minReplicas:2`, local NVMe weight cache, INT4 weights (35 GB loads in ~2 min vs 140 GB in 8 min).

**Bottleneck 5: CPU scheduling overhead.** At high token rates, vLLM's Python scheduler adds latency between GPU steps. Fix: multi-step scheduling (schedule N steps ahead), CUDA Graphs, process separation (API server and engine in separate processes).

#### 4. Scaling
{: #inf-scaling}

**Horizontal scaling (more pods).** KEDA scales on `vllm:num_requests_waiting`. New pods start in ~5 min for 8B (NVMe) or ~2 min for INT4-quantised. `minReplicas:2` absorbs traffic spikes before scale-out completes.

**Vertical scaling (bigger GPU).** H100 → A100: prefill is 5× faster (compute-bound, H100 FLOPS advantage), decode is 1.7× faster (bandwidth-bound, HBM3 advantage). Use H100 when TTFT SLO is the binding constraint. Use A100 when TPOT/cost is the binding constraint.

**Multi-region.** Route by latency (GeoDNS or Anycast). Each region has independent Kubernetes cluster, separate Envoy fleet. No cross-region KV transfer — keep sessions pinned to a region. Global rate limit service (Redis) handles cross-region quota enforcement.

**Model parallelism.** 70B in FP16 = 140 GB, exceeds one A100. Tensor parallelism across 2 GPUs (each holds 70 GB): AllReduce on every layer, adds ~1ms/layer on NVLink. Pipeline parallelism across nodes adds pipeline bubble. For serving, TP=2 on NVLink is the sweet spot; TP=4+ adds too much communication overhead.

#### 5. Optimisations
{: #inf-optimisations}

In priority order:

| Optimisation | Mechanism | Expected gain | When to apply |
|---|---|---|---|
| INT4 weight quantisation | 4× fewer weight bytes | ~4× decode throughput | Always, first |
| Continuous batching | Fill slots as requests complete | 2–3× throughput vs static | Already in vLLM |
| Prefix caching + consistent-hash routing | Skip prefill for shared system prompts | Up to 10× for shared-prompt workloads | When >20% traffic shares prefix |
| KV INT8 | 2× more context / same memory | 2× concurrent long-context requests | When context >8K |
| Speculative decoding | 3–4 tokens per LLM call | 2–3× latency reduction | Low-batch, predictable outputs |
| CUDA Graphs + multi-step scheduling | Eliminate Python kernel launch overhead | 20–30% throughput | All deployments |
| Chunked prefill | Cap per-step prefill, smooth TPOT | Reduces TPOT P99 by 50%+ under mixed load | When TPOT P99 is the SLO miss |

> **"What's your single highest-ROI optimisation?"** INT4 quantisation. It directly cuts the dominant cost (weight bandwidth), is low-risk on 7B+ models (<1% accuracy loss), ships in one day with AWQ calibration, and enables 2–4× higher batch size (more KV memory freed by smaller weights). Every other optimisation adds value on top of this baseline.

---

### Design ChatGPT's Backend
{: #design-chatgpt}

This question is asking about multi-turn conversation at Internet scale — not just LLM inference but the full user-facing product. The interview signal is whether you understand stateful conversation management on top of stateless GPU serving.

#### 1. Requirements
{: #chat-requirements}

**Functional:**
- Multi-turn chat: each message appends to a conversation history
- Streaming token-by-token responses
- System prompt per assistant persona/product
- Conversation history stored and retrieved per session
- Multiple model tiers: fast (GPT-3.5-class), quality (GPT-4-class)

**Non-functional:**
- Scale: 100M daily active users, average 5 messages/day = 500M messages/day = ~5800 req/sec sustained, 30K req/sec peak
- TTFT: < 500ms P95
- TPOT: < 50ms P99 (feels real-time to users)
- History storage: 30-day retention, average 50 turns × 500 tokens = 25K tokens per conversation
- Availability: 99.95%

**Key insight to state explicitly:** The LLM itself is stateless — it processes the full conversation context on every request. Statefulness lives in the conversation history store, not the model. This separation is the key architectural decision.

#### 2. Architecture
{: #chat-architecture}

```
User (browser / app)
         │  HTTPS + SSE
         ▼
┌─────────────────────────────────────────────────────┐
│                  API Layer                          │
│  Load balancer → API servers (stateless, Node.js)   │
│  Auth, session management, usage metering           │
└──────────────┬──────────────────┬───────────────────┘
               │                  │
               ▼                  ▼
┌──────────────────┐   ┌─────────────────────────────┐
│  Conversation    │   │     LLM Serving Layer        │
│  History Store   │   │                              │
│  (Postgres +     │   │  Envoy Gateway               │
│   Redis cache)   │   │    │                         │
│                  │   │    ├── Inference Router       │
│  GET /history    │   │    │   (prefix-hash LB)       │
│  POST /append    │   │    │                         │
└──────────────────┘   │    ├── Model Pool A (fast)   │
                       │    │   vLLM 8B × N pods      │
                       │    │                         │
                       │    └── Model Pool B (quality)│
                       │        vLLM 70B × M pods     │
                       │                              │
                       │  Kubernetes + KEDA           │
                       └─────────────────────────────┘
```

**Request flow for a user message:**

```
1. User sends message → API server
2. API server authenticates, meters usage
3. Fetch conversation history from Redis (hot) or Postgres (cold)
4. Build full prompt: [system_prompt] + [history] + [new_message]
5. POST to Envoy with X-Model, X-Session-ID headers
6. Inference Router: hash(system_prompt) → consistent pod (prefix cache hit)
7. vLLM: system_prompt KV from cache, new turns prefilled, decode streams
8. API server streams SSE tokens to user as they arrive
9. On stream end: async write new turn to Postgres + Redis
```

**The conversation history problem.** Each turn, the full history grows by one turn (~200–500 tokens). Naive approach: pass entire history to the LLM every request. At 50 turns × 400 tokens = 20K tokens, prefill takes seconds and KV cache consumes 10–20 GB per active conversation. Three mitigations:

- **Prefix caching with turn-level granularity.** The first 48 turns are identical to the last request. Their KV is cached. Only the new turn (turn 49) requires fresh prefill. This is why the inference router must use consistent-hash by session ID — route all turns of a conversation to the same vLLM pod.
- **Context window management.** Beyond a max history length (e.g., last 8K tokens), truncate oldest turns. LLM attention is most useful on recent turns anyway.
- **Summarisation compression.** When history exceeds budget, call a cheap model to summarise turns 1–40 into a 500-token summary, replace them with the summary. History stays bounded. The summary is cached as a KV prefix.

**System prompt caching.** ChatGPT-style products have different assistants with different system prompts (DALL-E tool use, code interpreter persona, browsing persona). Each system prompt is fixed and shared by all users of that assistant. A 2K-token system prompt with 100M users → every request shares the same 2K-token prefix → 100% cache hit rate on that prefix. The KV for that prefix is computed once per pod and reused indefinitely.

#### 3. Bottlenecks
{: #chat-bottlenecks}

**Bottleneck 1: Conversation history I/O.** At 5800 req/sec, fetching 20K tokens of history from Postgres for each request is slow. Redis caches the last N turns in memory (fast lookup by session_id). Cold sessions (>30 min inactive) go to Postgres. Cache hit rate target: 85%+.

**Bottleneck 2: Context length growth kills throughput.** A 50-turn conversation at 400 tokens/turn = 20K tokens. At batch=8 with 20K-token contexts, KV cache = 8 × 10 GB = 80 GB — fills an A100 entirely, leaving no room for weights. Fix: GQA architecture (Llama-3 uses 8 KV heads instead of 32, 4× KV reduction), KV INT8, context truncation.

**Bottleneck 3: Inconsistent routing breaks prefix cache.** If session turn 5 goes to pod-A and turn 6 goes to pod-B, pod-B has none of the conversation KV cached. It must re-prefill the full 2K token history. At 5800 req/sec × 20K token history × 2×70B FLOPs: massive wasted prefill compute. Fix: session-affinity routing (consistent hash on session_id).

**Bottleneck 4: Long-tail conversations.** 99th percentile user has 200-turn conversations. Full context: 80K tokens. Prefill: tens of seconds. TTFT >> 500ms SLO. Fix: summarisation compression at turn 50, sliding window with global attention on summary tokens.

#### 4. Scaling
{: #chat-scaling}

**LLM serving tier:** Same KEDA-based autoscaling as the single-model case. Two pools (fast/quality) scale independently. Quality pool kept at higher `minReplicas` because cold start time (10+ min for 70B) is longer.

**History tier:** Postgres with read replicas for conversation retrieval (mostly reads). Redis cluster sharded by `session_id`. Redis TTL = 2 hours of inactivity; after that, reads fall through to Postgres.

**API tier:** Stateless Node.js servers behind a network load balancer. Session stickiness for streaming connections (the SSE connection must stay to the same API server until the response completes, but different turns of a conversation can go to different API servers — history comes from the store, not server memory).

**Geographic distribution:** Conversations are user-specific, but the LLM inference is stateless once you have the history. Use GeoDNS to route users to the nearest region. Conversation history replicated async to secondary region (eventual consistency acceptable — a user switching regions mid-conversation sees a brief lag, not corruption).

#### 5. Optimisations
{: #chat-optimisations}

**Session-affinity routing is the highest-ROI optimisation** specific to multi-turn chat. It's free to implement (one config change in the inference router) and eliminates redundant prefill for every request beyond turn 1. At 50 turns per conversation, you pay for 1 full prefill and 49 incremental prefills — rather than 50 full prefills.

**System prompt prefix caching** is the second-highest ROI. A fixed 2048-token system prompt shared by all users: pay for it once per pod, serve it to all. With 100 pods and 100M requests/day, you save 99M × 2048 token prefills per day.

**Summarisation compression** prevents the "long conversation death spiral" — as conversations grow, TTFT grows with them until SLOs are violated. Compress history at a fixed threshold (e.g., 8K tokens), keep a running summary. The model never sees a context longer than 10K tokens regardless of conversation length.

---

### Design a Multi-Tenant LLM Serving Platform
{: #design-multitenant}

This is the hardest of the three — it's about isolation, fairness, and economics across many tenants with different requirements, sharing the same GPU fleet.

#### 1. Requirements
{: #mt-requirements}

**Functional:**
- Multiple tenants (teams, business units, or external customers) share the same GPU fleet
- Per-tenant model selection (each tenant can use a different model or the same model)
- Per-tenant rate limits and cost quotas
- Tenant isolation: one tenant's traffic spike cannot degrade another's latency
- Custom system prompts and fine-tuned LoRA adapters per tenant

**Non-functional:**
- Tenants: 100 internal teams, each with different SLO tiers
- SLO tiers: Premium (TTFT < 500ms), Standard (TTFT < 2s), Batch (best effort)
- Budget isolation: tenant A exhausting quota must not affect tenant B
- GPU utilisation target: >70% across the fleet (shared infrastructure should be more efficient than dedicated)

**State explicitly:** Multi-tenancy is a fairness problem as much as a scaling problem. The hard part is not serving at scale — it's guaranteeing that a badly-behaved tenant (e.g., one submitting 10K-token prompts at 100 req/sec) does not cause tail latency spikes for other tenants.

#### 2. Architecture
{: #mt-architecture}

```
Tenant A API key          Tenant B API key         Tenant C API key
       │                         │                        │
       └─────────────────────────┴────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Envoy API Gateway      │
                    │                         │
                    │  JWT auth + tenant ID    │
                    │  Per-tenant rate limit   │
                    │  Token quota check       │
                    │  Route to priority queue │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐
    │  Premium Queue   │  │  Standard Queue  │  │  Batch Queue   │
    │  (always served) │  │  (fair-share)    │  │  (spare cap.)  │
    └────────┬─────────┘  └────────┬─────────┘  └───────┬────────┘
             └──────────────────────┴────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Inference Scheduler      │
                    │                              │
                    │  Priority + fairness policy  │
                    │  KV-cache-aware routing      │
                    │  LoRA adapter routing        │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  Base Model     │  │  Base Model     │  │  LoRA Adapter   │
    │  vLLM Pod       │  │  vLLM Pod       │  │  Serving Pod    │
    │  (base weights) │  │  (base weights) │  │  (base + A/B/C) │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
                    │
    ┌───────────────▼────────────────┐
    │     Tenant Quota Service       │
    │  Redis: token counters         │
    │  Postgres: usage history       │
    │  Billing: cost attribution     │
    └────────────────────────────────┘
```

**Priority queue model.** Three logical queues map to scheduling priorities in the inference scheduler. The scheduler always drains Premium requests first, then Standard, then Batch on spare capacity. Premium SLO tenants pay more but get guaranteed head-of-line. Batch tenants pay less and accept variable latency.

**LoRA multi-tenancy.** Fine-tuned LoRA adapters are per-tenant but share the same base model weights. vLLM's LoRA serving loads the base model once and swaps LoRA weight deltas per request. Multiple adapters can be active simultaneously (up to `max_loras` in GPU memory). Routing: if request has `X-Lora-Id` header, route to LoRA-enabled pods; otherwise route to base model pods.

**Quota service.** Every request passes through a quota check before admission. The quota service maintains per-tenant Redis counters: `tokens_used_today`, `tokens_used_this_minute`. Rate limiting enforces the burst rate; quota enforces the monthly budget. The quota service is called synchronously by the Envoy ext_authz filter — if over quota, 429 before the request touches a GPU.

#### 3. Bottlenecks
{: #mt-bottlenecks}

**Bottleneck 1: Head-of-line blocking across tenants.** Without priority queuing, a batch tenant submitting 10K-token prompts at 100 req/sec fills the GPU with long prefills, blocking Premium tenant requests that arrive after. Fix: strict priority queue — Premium requests preempt batch requests for prefill slots. In vLLM: `preemption_mode=recompute`, priority scheduling policy, premium requests have higher priority value.

**Bottleneck 2: Noisy-neighbour KV cache eviction.** At peak, batch tenant's large-context requests fill the KV cache, causing evictions of Premium tenant's KV. Premium tenant's next request must re-prefill. Fix: per-tenant KV cache quotas. Reserve N% of KV memory for Premium tier (vLLM doesn't natively support this yet — requires custom scheduler or separate Premium pods).

**Bottleneck 3: Quota service latency.** Every request calls the quota service synchronously. If Redis is slow (e.g., during a Redis failover), the quota check adds latency. Fix: local cache of quota state in Envoy (refresh every second), fail-open on Redis unavailability (accept requests, debit later).

**Bottleneck 4: LoRA adapter hot-swap overhead.** Loading a LoRA adapter into GPU memory takes milliseconds per request if the adapter is not cached. With 100 tenants × 1 adapter each and limited GPU memory for adapter cache, cache misses cause jitter. Fix: LRU adapter cache, pin frequently-used adapters (top-20% by traffic), route tenants to pods that have their adapter warm.

#### 4. Scaling
{: #mt-scaling}

**Fleet partitioning vs oversubscription.** Two models:
- **Partitioned:** each tenant tier has dedicated pods. Premium gets 20% of fleet, Standard gets 60%, Batch gets 20%. Strict isolation but low utilisation (Premium pods sit idle at 2am).
- **Oversubscribed with priority:** all pods serve all tiers, with scheduling priority. Premium always gets served first. Batch fills idle time. Higher fleet utilisation (70%+ vs 40% for partitioned) but requires careful priority implementation to maintain SLOs.

**The recommended answer:** hybrid. Premium tier gets a dedicated minimum (2 pods always reserved), the rest of the fleet is shared with priority scheduling. During a traffic spike, Premium requests immediately get the dedicated pods; Standard requests fill the shared pool; Batch requests wait or get 503.

**Per-tenant autoscaling signals.** KEDA can use per-tenant Prometheus metrics: `vllm:num_requests_waiting{tenant="premium"}`. Scale the Premium Deployment independently from the Batch Deployment. This gives per-SLO-tier autoscaling.

**LoRA at scale.** With 100 tenants and 100 adapters, adapter-to-pod affinity routing (consistent hash on `lora_id`) reduces adapter cache misses. A dedicated LoRA pod fleet (base model loaded, rotating through adapters) segregates adapter-serving from base-model-serving, preventing adapter hot-swap overhead from affecting base model latency.

#### 5. Optimisations
{: #mt-optimisations}

**Shared prefix caching across tenants.** If Tenant A and Tenant B both use the same base system prompt (e.g., "You are a helpful assistant"), their system prompt KV can be shared across pods. Cross-tenant prefix sharing requires that prefix caching is keyed on token content, not tenant ID — vLLM's block-level prefix caching already does this (hash of token sequence, not user identity).

**Batch request coalescing.** Batch-tier requests with the same system prompt and similar context lengths can be coalesced into a single large batched request. Instead of 100 batch requests each with a 1K-token system prompt prefilling separately, coalesce into one prefill pass over the system prompt, then fork the decode for 100 requests. Saves 99 system-prompt prefills.

**Cost attribution granularity.** For billing accuracy: attribute cost not by request count but by (input tokens × prefill cost rate) + (output tokens × decode cost rate). Prefill cost ∝ tokens² (O(L²) attention). Decode cost ∝ tokens (O(L) per step). Batch tenants typically have long inputs and short outputs; Premium tenants have short inputs and long outputs. A flat per-request price disadvantages one segment.

---

### The End-to-End Pipeline (What You Say in Any of These)
{: #e2e-pipeline}

When asked to walk through the full pipeline, this is the answer. Memorise the flow, fill in the details from context.

```
User request
    │
    ▼  [1] API Gateway (Envoy)
       - TLS termination
       - JWT / API key validation (ext_authz)
       - Per-tenant / per-user rate limiting (token bucket)
       - Token quota check (ext_proc → quota service)
       - Route by X-Model header → model cluster
       - Timeout: 300s, retry only on pre-stream 5xx
    │
    ▼  [2] Inference Router
       - Reads vllm:num_requests_waiting, gpu_cache_usage_perc per pod
       - Hash(system_prompt) → consistent pod (prefix cache affinity)
       - Fall back to least-loaded if consistent pod saturated
       - Sets X-Route-To-Pod header for Envoy upstream override
    │
    ▼  [3] vLLM Engine Pod
       - Scheduler: places request in priority queue (premium > standard > batch)
       - Prefill: check prefix cache → compute only uncached suffix
       - Continuous batching: request joins running batch at next iteration
       - Chunked prefill: long prompts split, interleaved with decode steps
       - Decode: autoregressive loop, streams tokens via SSE back through Envoy
       - KV blocks freed on EOS, returned to PagedAttention block pool
    │
    ▼  [4] Kubernetes Layer
       - Pod is on GPU node selected by: nodeAffinity (GPU model), 
         podAntiAffinity (spread across nodes), topologySpread (across AZs)
       - KEDA watches num_requests_waiting → scales Deployment replicas
       - PodDisruptionBudget: min 2 pods available during node maintenance
       - Rolling update: maxUnavailable:0, new pod must pass startupProbe
    │
    ▼  [5] Response
       - Tokens stream back through Envoy (SSE / chunked transfer)
       - Envoy records: request duration, upstream latency, token count (via ext_proc)
       - Quota service debits output tokens from tenant account
       - Metrics exported: TTFT, TPOT, e2e latency → Prometheus → Grafana
```

**The one-sentence version** for when you're asked to summarise:

> "Traffic enters Envoy for auth, rate limiting, and model routing. The inference router does KV-cache-aware load balancing to maximise prefix cache hits. vLLM handles continuous batching, PagedAttention, and chunked prefill on GPU. Kubernetes manages pod placement, autoscaling via KEDA on queue depth, and rolling updates with zero downtime. Metrics flow to Prometheus; alerts fire when TTFT P95 or queue depth exceed thresholds."

**What makes your answer stand out:**
- You say "prefix-hash routing" not just "load balancing" — shows you know prefix caching requires routing affinity
- You say "maxUnavailable:0 with startupProbe" not just "rolling update" — shows you know model loading takes minutes
- You say "KEDA on num_requests_waiting" not "HPA on CPU" — shows you know CPU is the wrong signal for GPU inference
- You say "chunked prefill to isolate TPOT from long prefills" not just "batching" — shows you know the prefill-decode interference problem
- You connect Envoy's circuit breaker to vLLM's readiness backpressure — shows you understand the overload prevention chain
