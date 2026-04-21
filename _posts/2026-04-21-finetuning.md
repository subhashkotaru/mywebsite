---
title: "Post-training"
date: 2026-04-21
description: "Adapting pretrained language models through the full post-training lifecycle — PEFT, alignment with SFT/DPO/GRPO, RLVR, chat templates, tool use, and modern evaluation."
tags: [ml-systems, post-training, fine-tuning, llm, peft, rlvr]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#full-finetuning">Full Fine-tuning</a></li>
    <li><a href="#peft">Parameter-Efficient Fine-tuning</a>
      <ul class="post-toc-sublist">
        <li><a href="#prompt-prefix">Prompt & Prefix Tuning</a></li>
        <li><a href="#adapters">Adapters</a></li>
        <li><a href="#lora">LoRA</a></li>
        <li><a href="#lora-variants">LoRA Variants</a></li>
        <li><a href="#qlora">QLoRA</a></li>
        <li><a href="#dora">DoRA</a></li>
        <li><a href="#selective">Selective Fine-tuning</a></li>
        <li><a href="#side-tuning">Side Tuning</a></li>
        <li><a href="#peft-comparison">Method Comparison</a></li>
      </ul>
    </li>
    <li><a href="#alignment">Alignment & RLHF</a>
      <ul class="post-toc-sublist">
        <li><a href="#sft">Supervised Fine-tuning</a></li>
        <li><a href="#reward-model">Reward Modelling</a></li>
        <li><a href="#ppo">PPO</a></li>
        <li><a href="#dpo">DPO</a></li>
        <li><a href="#orpo">ORPO</a></li>
        <li><a href="#grpo">GRPO</a></li>
      </ul>
    </li>
    <li><a href="#catastrophic-forgetting">Catastrophic Forgetting</a></li>
    <li><a href="#practical">Practical Considerations</a></li>
    <li><a href="#post-training-lifecycle">Post-Training Lifecycle: Structured Data to RLVR</a>
      <ul class="post-toc-sublist">
        <li><a href="#chat-templates">Chat Templates & Loss Masking</a></li>
        <li><a href="#tool-use-protocol">Tool Use as World Interaction</a></li>
        <li><a href="#structured-outputs">Structured Outputs & Schema Tokens</a></li>
        <li><a href="#sft-protocol">SFT as Protocol Learning</a></li>
        <li><a href="#preference-modeling">Preference Modeling</a></li>
        <li><a href="#rlvr-deep">RLVR: Reinforcement Learning with Verifiable Rewards</a></li>
        <li><a href="#post-training-eval">Post-Training Evaluation</a></li>
      </ul>
    </li>
    <li><a href="#sft-data-engineering">SFT Data Engineering & RL Environments</a>
      <ul class="post-toc-sublist">
        <li><a href="#sft-record-anatomy">SFT Record Anatomy & Mixture Design</a></li>
        <li><a href="#vertical-corpora">Behavior-Vertical Corpora</a></li>
        <li><a href="#curation-pipeline">Curation: Human + Synthetic</a></li>
        <li><a href="#limits-of-sft">Why SFT Is Not Enough</a></li>
        <li><a href="#rl-environments">RL Environments & Verifiers</a></li>
        <li><a href="#reward-shaping">Reward Shaping & RL Algorithms</a></li>
        <li><a href="#rl-infrastructure">RL Infrastructure & Distillation</a></li>
      </ul>
    </li>
    <li><a href="#rl-algorithms-deep">RL Algorithms for Post-Training</a>
      <ul class="post-toc-sublist">
        <li><a href="#pg-foundations">Policy Gradient Foundations</a></li>
        <li><a href="#ppo-deep">PPO: Trust-Region Recipe</a></li>
        <li><a href="#grpo-family">GRPO Family: Critic-Free Reasoning RL</a></li>
        <li><a href="#off-policy-rl">Off-Policy & Long-Context RL</a></li>
        <li><a href="#agent-rl">Agent RL: Multi-Turn Tool-Using Trajectories</a></li>
        <li><a href="#reward-engineering">Reward Engineering & Verifier Design</a></li>
        <li><a href="#rl-systems-scaling">Systems, Scaling Laws & Distillation</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## Overview
{: #overview}

A pretrained language model learns general statistical structure from vast amounts of text but is not immediately useful for a specific task — it may answer questions inconsistently, refuse instructions unpredictably, or produce outputs in the wrong format. **Post-training** adapts the model's weights and behaviour through a structured pipeline: PEFT methods for parameter efficiency, SFT to teach protocols, preference optimisation to encode quality, and RLVR to drive task correctness. The challenge is doing this cheaply: a 175B model requires identical hardware to pretraining to update all its weights, making full fine-tuning impractical for most practitioners.

The field has converged on two complementary strategies:

- **Parameter-efficient fine-tuning (PEFT)** — update a small fraction of parameters while freezing the bulk of the model
- **Alignment fine-tuning** — use human preference data and reinforcement learning to steer behaviour beyond what supervised objectives alone achieve

<div class="post-flow post-flow--horizontal" role="group" aria-label="Fine-tuning spectrum">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Prompt engineering — no training</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">PEFT — few trainable params</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Full fine-tuning — all params</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green post-flow__bar--accent">RLHF — human preference alignment</span></li>
  </ol>
</div>

**Why fine-tune at all?** Few-shot prompting can handle many tasks but hits a ceiling: it cannot update the model's internal representations, is bounded by context length, and cannot reliably instil behaviours requiring consistent multi-turn reasoning. Fine-tuning on task-specific data closes this gap substantially — GPT-3 fine-tuned on SQuAD V2 achieves 88.4% F1 vs 69.8% few-shot; on WikiSQL, 73% accuracy vs 20%.

> **Interview question:** When would you choose RAG over fine-tuning, and vice versa?
>
> *RAG is better when: the knowledge changes frequently (live data, recent events), you need citations and source attribution, the knowledge domain is large and diverse, and you have limited compute for training. Fine-tuning is better when: you need to change the model's output format or style consistently, the task requires internalised skills (code generation, reasoning patterns) rather than knowledge retrieval, the model needs to be fast at inference without retrieval latency, and the knowledge is stable enough to be baked in. The sweet spot for production systems is often both: fine-tune the model for format/behaviour and use RAG for up-to-date knowledge.*

---

## Full Fine-tuning
{: #full-finetuning}

Full fine-tuning updates every parameter in the model on the target dataset using standard backpropagation. The training loop is identical to pretraining — the differences are data volume (thousands of examples vs trillions of tokens) and learning rate (much smaller, to avoid erasing pretrained knowledge).

**Memory requirements** are the bottleneck. For a model with `M` parameters, Adam fine-tuning requires:

<div class="post-flow" role="group" aria-label="Full fine-tuning memory per parameter">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">BF16/FP16 weights — 2 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">BF16/FP16 gradients — 2 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">FP32 master weights (mixed precision) — 4 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">FP32 Adam first moment m — 4 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">FP32 Adam second moment v — 4 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Total: ~16 bytes/param → 7B model: ~112 GB, 70B model: ~1.1 TB</span></li>
  </ol>
</div>

**Why are both FP16 weights and FP32 master weights stored?** This is the standard mixed-precision training recipe. The forward and backward passes use FP16 (or BF16) for speed — matrix multiplications are 2–8× faster on tensor cores in reduced precision. But the weight update `θ = θ - η·g` requires precision: the gradient `g` is often much smaller in magnitude than `θ`. In FP16, `θ + η·g` can round to `θ` exactly when the gradient is small, making the step a no-op. Storing master weights in FP32 prevents this underflow. After the FP32 update, the result is downcast back to FP16 for the next forward pass.

**Catastrophic forgetting**: updating all weights aggressively on a narrow dataset erases previously learned capabilities — a model fine-tuned only on medical QA may lose general coding ability. Mitigations:
- Use a small learning rate (`1e-5` to `5e-6`) relative to pretraining
- Mix fine-tuning data with **replay samples** from the pretraining distribution (keep ~5–10% general data)
- Stop early before the model overfits the fine-tuning distribution

> **Interview question:** You fine-tune a 7B model for sentiment analysis and it achieves 95% accuracy on your test set, but when you deploy it, users report it refuses to follow basic formatting instructions it handled fine before. What happened, and how do you fix it?
>
> *Catastrophic forgetting — the model overwrote general instruction-following representations with sentiment-specific patterns. The fine-tuning signal on sentiment labels was strong enough to push the model into a region of weight space that destroys its general capabilities. Fix: (1) Lower the learning rate — you were updating too aggressively. (2) Add replay data: mix 10–20% general instruction-following examples into the fine-tuning batch. (3) Use LoRA instead of full fine-tuning — by constraining updates to low-rank subspaces, you're less likely to clobber pretrained representations. (4) Evaluate on a general capability benchmark (MMLU, MT-Bench) during fine-tuning and stop when general performance starts dropping.*

---

## Parameter-Efficient Fine-tuning
{: #peft}

PEFT methods freeze most of the pretrained model and introduce a small number of new or reparameterised trainable parameters. The frozen base handles general language understanding; the trainable component learns the task delta.

The taxonomy:

| Category | Key Idea | Examples |
|----------|----------|----------|
| **Additive — Adapters** | Insert bottleneck modules between frozen layers | Sequential Adapter, Parallel Adapter, AdapterFusion |
| **Additive — Soft Prompts** | Prepend learnable vectors to input or KV tensors | Prompt Tuning, Prefix Tuning, P-Tuning |
| **Partial — Selective** | Update only a chosen subset of existing parameters | BitFit, FISH Mask, Child-Tuning |
| **Reparameterised** | Factorise weight updates into low-rank or structured form | LoRA, DoRA, AdaLoRA, QLoRA |
| **Hybrid** | Combine multiple PEFT methods automatically | UniPELT, AutoPEFT |

### Prompt & Prefix Tuning
{: #prompt-prefix}

**Prompt tuning** prepends a sequence of learnable token embeddings — **soft prompts** — to the input. The LLM's weights are frozen; only the prompt parameters `P ∈ ℝˡˣᵈ` (where `l` is prefix length, `d` is embedding dimension) are updated by backpropagating through the frozen model:

```
X̂ = Concat(P, X) ∈ ℝ^(l+n)×d
```

The trainable parameter count is just `l × d` — for `l=20, d=4096` that's 81,920 parameters regardless of model size.

**Prefix tuning** extends this deeper: instead of only the input embeddings, learnable prefix vectors `P_k, P_v` are prepended to the key and value matrices at *every* attention layer. The attention computation becomes:

```
head = Attn(XWq, [P̂k, XWk], [P̂v, XWv])
```

where `P̂k = FFN(Pk)` — a small MLP reparameterises the prefix to stabilise training (direct optimisation of soft prompts is unstable). After training, the FFN is discarded; only the final prefix tensors are kept.

<div class="post-flow post-flow--compare" role="group" aria-label="Prompt vs prefix tuning">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Prompt Tuning</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Soft tokens prepended to input only</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Params: l × d_embed (e.g. 82K)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Works well at large scale (≥10B params)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Weak at small model scale</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Prefix Tuning</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Prefix inserted into K, V at every layer</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Params: l × n_layers × 2 × d_model</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Stronger influence across full model depth</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Inference: l extra KV entries per layer cached</span></li>
    </ol>
  </div>
</div>

**Why does prompt tuning work worse for small models?** At small scale, the frozen model has limited capacity — the gradient signal flowing back through the model to the soft prompt is weak and noisy. The model's representations are rigid; a few prepended vectors can't steer a 1B model as effectively as they steer a 100B model where the frozen representations are richer and more steerable. Prompt tuning at 11B approaches full fine-tuning performance; at 100M it is substantially worse.

**Why does prefix tuning use a MLP reparameterisation?** Directly optimising `Pk, Pv` as free parameters causes instability — the gradient landscape is ill-conditioned since these vectors live in the attention's KV space and interact with all queries. The MLP acts as a smoother mapping, making the optimisation landscape better-conditioned. After training the MLP is discarded (its output is the final fixed prefix), so there's no inference cost.

> **Interview question:** Both prompt tuning and prefix tuning add zero parameters to the deployed model checkpoint. But prefix tuning has a runtime cost that prompt tuning doesn't. What is it?
>
> *Prefix tuning adds l virtual tokens to the KV cache at every attention layer. For a 96-layer model with prefix length l=20, this adds 96 × 20 = 1,920 KV pairs that must be stored in the KV cache and attended to by every subsequent token. This increases the KV cache memory by l × n_layers × 2 × d_head × n_heads bytes per sequence, and adds l × n_layers dot products to every attention computation. For prompt tuning, the soft prompt only extends the input sequence by l tokens at layer 0 — the computational overhead is a single extra self-attention computation over l tokens, not repeated at every layer.*

### Adapters
{: #adapters}

**Adapter tuning** inserts small bottleneck modules into frozen transformer layers. The original Sequential Adapter (Houlsby et al., 2019) inserts an adapter after both the attention sub-layer and the FFN sub-layer:

```
h = h + W_up(ReLU(W_down(h)))
   where W_down ∈ ℝ^(d×r), W_up ∈ ℝ^(r×d), r ≪ d
```

`W_up` is initialised to zero so the adapter starts as an identity mapping — training is stable from step 0 because `ΔW=0` initially. Only the adapter parameters `W_down, W_up` are updated; the rest of the transformer is frozen.

**Adapter variants and the design space:**

| Variant | Key difference | Trade-off |
|---------|----------------|-----------|
| Sequential Adapter | After attn + after FFN | 2 adapters per layer, higher capacity |
| Residual Adapter | After FFN + LayerNorm only | Fewer parameters, lower overhead |
| Parallel Adapter | Alongside attention and FFN (not after) | Can be run in parallel, reducing latency |
| AdapterDrop | Prune unimportant adapters at inference | Recovers speed at cost of some accuracy |
| AdapterFusion | Train multiple task adapters, then learn a fusion layer | Multi-task without interference |

**The adapter inference latency problem.** Sequential adapters add two extra matmul + non-linearity operations per transformer layer. For a 96-layer model, this is 192 additional operations in the critical path — not parallelisable because each depends on the previous layer's output. Parallel adapters (running alongside attention/FFN, not after) allow the adapter computation to overlap with the base computation on modern hardware, reducing effective latency. But sequential adapters are simpler and remain widely used when latency is not the primary constraint.

> **Interview question:** You have fine-tuned separate adapters for translation and summarisation. A user wants both capabilities simultaneously. What options do you have, and what are the trade-offs?
>
> *Option 1: AdapterFusion — keep both adapters, add a trainable attention-based fusion layer that learns to weight them per-input. Best quality, but introduces extra inference overhead and requires fusion training data. Option 2: Adapter merging via SVD — compute the combined adapter by merging the two adapters' weight matrices through truncated SVD. Faster than fusion but loses task-specific specialisation — the merged adapter may underperform each individual adapter on its primary task. Option 3: Adapter concatenation — concatenate the two adapters' weight matrices, doubling the bottleneck rank. Straightforward, but the resulting adapter has twice the parameters and may inherit undesired behaviours from each adapter (e.g. a translation adapter that produces short outputs may bias the merged adapter toward brevity). Option 4: Serve separate models — route translation requests to one, summarisation to the other. Simplest, no degradation, but doubles serving memory.*

### LoRA
{: #lora}

**LoRA (Low-Rank Adaptation)** avoids new layers entirely by reparameterising the weight update. The key observation: weight updates in fine-tuning have low intrinsic rank — the gradient updates during adaptation live in a low-dimensional subspace of the full weight matrix space. LoRA makes this explicit.

For a frozen weight `W₀ ∈ ℝ^(d×k)`, the learned update is factored as:

```
ΔW = B × A,   B ∈ ℝ^(d×r), A ∈ ℝ^(r×k),   r ≪ min(d, k)
```

`A` is initialised from `𝒩(0, σ²)` (random); `B` is initialised to **zero** so `ΔW = BA = 0` at step 0 — the model starts as the exact pretrained model. The forward pass adds the low-rank term:

```
h = W₀x + BAx = (W₀ + BA)x
```

The scaling factor `α/r` is applied (where `α` is a hyperparameter, typically `α = r` or `2r`) to control the contribution magnitude relative to the pretrained weights. **At deployment, `W' = W₀ + (α/r)·BA` is computed once and folded in — zero inference latency overhead.**

**Why is B initialised to zero and not A?** If A=0 instead, the backward gradient through A is `∂L/∂A = Bᵀ (∂L/∂ΔW)`. Since B starts random, gradients flow into A from the start. If B=0, gradients through B are `∂L/∂B = (∂L/∂ΔW) Aᵀ` — but B is zero, so the entire ΔW contribution is zero and no gradient flows. The choice is: A random, B zero. This ensures ΔW=0 at init (the model is unchanged) while allowing A to receive gradients immediately, kick-starting learning.

**Where to apply LoRA.** The original paper applies it to Q and V projection matrices in attention. In practice, applying it to Q, K, V, and the output projection `W_O` works better. Some implementations also apply it to the FFN matrices (`W_up`, `W_down`). The FFN contains most of the model's parameters, so LoRA on FFN provides more capacity at the same rank.

**Rank selection.** `r = 4` to `r = 64` covers most use cases. For a `d=4096` layer with `r=8`, LoRA adds `2 × 4096 × 8 = 65,536` parameters vs `4096² = 16.7M` for the full weight — a 255× reduction. Higher rank increases expressivity but also memory and compute. For instruction-following tasks, `r=8` to `r=16` is typically sufficient. For complex skill acquisition (coding, domain expertise), `r=64` or higher may be needed.

> **Interview question:** LoRA claims zero inference overhead because you can merge ΔW back into W. But in what practical scenario would you NOT merge them, and why?
>
> *You would not merge when you need to switch between multiple LoRA adapters at runtime — e.g. serving different fine-tuned behaviours (customer support, code generation, medical) from the same base model. If you merge adapter A and a user requests adapter B, you'd need to merge B into the model again (or unmix A first, which requires storing the original W₀). Instead: keep W₀ frozen and frozen on GPU, load LoRA adapters as small separate matrices, and hot-swap them between requests. At r=8 for a 7B model, each adapter is ~16MB — trivial to store many. This is the "multi-tenant LoRA serving" pattern used in production systems like S-LoRA, which serves thousands of fine-tuned adapters from a single GPU cluster.*

### LoRA Variants
{: #lora-variants}

| Variant | Key idea | When to use |
|---------|----------|-------------|
| **AdaLoRA** | Dynamically allocate rank budget across weight matrices based on singular value importance | When you don't know which layers need more rank |
| **DyLoRA** | Train LoRA at multiple ranks simultaneously; use any rank at inference | When target deployment rank is uncertain |
| **LoHa** | `ΔW = (B₁ ⊙ A₁)(B₂ ⊙ A₂)` — Hadamard product of two low-rank pairs | Same params, higher effective expressivity than standard LoRA |
| **LoKr** | `ΔW = B ⊗ A` — Kronecker product | Preserves matrix structure, fewer params for same rank |
| **LoRA-FA** | Freeze A after random init, only train B | Halves optimizer state memory; A acts as a fixed random projection |
| **Delta-LoRA** | Also update pretrained weights `W₀` using ΔW differences | Closes the gap between LoRA and full fine-tuning |
| **MoELoRA** | Route inputs to different LoRA experts via a learned gate | Multi-task LoRA with minimal interference between tasks |

**AdaLoRA in depth.** Standard LoRA assigns the same rank `r` to every weight matrix — a flat allocation. In practice, different weight matrices have different sensitivity to adaptation: attention Q/K matrices in early layers may need rank 1, while FFN output matrices in later layers need rank 64. AdaLoRA starts with a higher total rank budget and uses SVD-based importance scoring to prune low-importance singular components during training, concentrating rank budget where it helps most. This achieves the same final parameter count as fixed-rank LoRA but better performance.

> **Interview question:** Why does AdaLoRA use SVD to allocate rank, and what's the cost of this approach?
>
> *AdaLoRA reparameterises each adapter as `ΔW = P × Λ × Q` where Λ is a diagonal matrix of singular values and P, Q are orthogonal matrices. During training, it prunes singular values with small magnitude (they contribute little to the update) and grows those with large magnitude. The importance score is the absolute value of each singular value times the gradient magnitude. This SVD decomposition is mathematically principled — the rank of a matrix is determined by its non-zero singular values, so pruning small singular values correctly reduces effective rank. The cost: computing SVD at each training step is expensive (O(r³) for rank-r matrices), and maintaining the orthogonality constraint on P, Q requires projected gradient updates. AdaLoRA is typically 2–3× slower to train than standard LoRA at the same parameter budget.*

### QLoRA
{: #qlora}

**QLoRA** quantises the frozen base model weights to 4 bits, reducing base model memory by ~4× while keeping the LoRA adapters in BF16 for stable gradient flow.

**Why naive 4-bit quantisation fails.** LLM weight matrices contain extreme outliers — individual weights that are 10–100× larger than most others. A uniform 4-bit quantisation scheme assigns 16 bins across the entire weight range. With outliers, most bins cluster around the centre and the actual weight distribution is poorly represented, causing large quantisation error. QLoRA solves this with two innovations:

1. **NF4 (NormalFloat4)**: a 4-bit data type that places quantisation bins at positions optimal for normally distributed weights (which LLM weights approximately are). NF4 is information-theoretically optimal for normal distributions — it minimises quantisation error for the actual weight distribution.

2. **Block-wise quantisation**: divide each weight matrix into blocks of 64 elements; each block has its own FP32 scale constant. This localises outlier effects — an outlier in one block doesn't skew quantisation for the rest of the matrix. Overhead: one FP32 per 64 weights = 0.5 extra bits/param.

3. **Double quantisation**: the FP32 block scales are themselves quantised to INT8 with block size 256. Overhead drops from 0.5 to `8/64 + 32/(64×256) ≈ 0.127` bits/param.

**Memory comparison (70B model):**

| Component | Full fine-tuning | LoRA (BF16 base) | QLoRA |
|-----------|-----------------|-----------------|-------|
| Base weights | 140 GB | 140 GB | ~35 GB |
| LoRA adapters | — | 0.5 GB | 0.5 GB |
| Optimizer state (BF16) | 280 GB | 2 GB | 2 GB |
| Activations | 197 GB | 197 GB | 197 GB |
| **Total** | **617 GB** | **339 GB** | **235 GB** |

QLoRA enables fine-tuning a 65B model on a single 80GB A100. The key insight: you don't need high-precision base weights during the backward pass — gradients flow through the quantised weights (with straight-through estimator treating quantisation as identity in the gradient), accumulate into the BF16 LoRA matrices, and only the LoRA adapters are updated.

> **Interview question:** QLoRA backpropagates gradients through 4-bit quantised weights. Quantisation is not differentiable — how does this work?
>
> *The straight-through estimator (STE): treat quantisation as the identity function in the backward pass. In the forward pass, `x_q = quantise(x)` — the true 4-bit value. In the backward pass, `∂L/∂x ≈ ∂L/∂x_q` — pretend the quantisation didn't happen and pass gradients straight through. This is an approximation, but empirically it works well because: (1) the gradient is used to update LoRA parameters, not the quantised weights themselves; (2) LoRA adapters are in full precision BF16, so the accumulated gradient update is precise; (3) quantisation noise acts like a mild regulariser. The frozen 4-bit weights never get updated — only LoRA A and B do — so the imprecise gradient through quantisation only needs to be good enough to train the LoRA adapters, not to recover the base weights.*

### DoRA
{: #dora}

**DoRA (Weight-Decomposed Low-Rank Adaptation)** addresses a fundamental difference between how LoRA updates weights and how full fine-tuning does. Empirical analysis shows:

- **Full fine-tuning** makes updates with high magnitude variation but consistent direction — it changes *how much* each direction matters.
- **LoRA** makes updates that are more uniform in magnitude but vary more in direction — it rotates the weight matrix.

DoRA decomposes the pretrained weight `W₀` into magnitude and direction components:

```
W₀ = m · (V / ‖V‖_c)    # magnitude m (scalar per column), direction V (unit column vectors)
```

During fine-tuning, the magnitude `m` is updated freely, while the directional component `V` is updated via LoRA:

```
W' = (m + Δm) · ((V + ΔV_LoRA) / ‖V + ΔV_LoRA‖_c)
```

This decomposition allows DoRA to independently control *how strongly* each direction is expressed (magnitude) and *which directions* are used (LoRA update), mimicking the learning pattern of full fine-tuning. Empirically, DoRA consistently outperforms LoRA across NLP and vision-language tasks while introducing **no additional inference latency** — the magnitude and direction components merge back into a single weight matrix at deployment.

> **Interview question:** DoRA adds magnitude vectors on top of LoRA — doesn't this just increase the parameter count? What's the actual benefit over simply using a higher LoRA rank?
>
> *Yes, DoRA adds one scalar per weight column (d scalars per d×k matrix), which is tiny — 4096 floats for a 4096×4096 matrix vs 4096² for the full weight. But the benefit is not just parameter count. The magnitude-direction decomposition fundamentally changes the learning dynamics. With standard LoRA at rank r, the update ΔW = BA constrains both magnitude and direction changes to live in the same rank-r subspace — you can't freely scale existing directions without also changing them. DoRA decouples these: magnitude can be updated for any direction (all d dimensions) while only the directional update is low-rank. A higher LoRA rank achieves similar expressivity but requires 2×d×r parameters (quadratic in r) — DoRA achieves similar learning dynamics to full fine-tuning with far fewer additional parameters than rank-matching LoRA would require.*

### Selective Fine-tuning
{: #selective}

Instead of adding new parameters, **selective fine-tuning** updates only a chosen subset of the existing pretrained parameters.

**BitFit** updates only the bias terms in each layer — a tiny fraction of parameters (typically <0.1% of model parameters). Surprisingly competitive on many NLP tasks. The rationale: biases shift the activation distribution in each layer without changing the directional structure of the weight matrices. Task adaptation often requires shifting what features are active, not necessarily how features are extracted.

**FISH Mask** computes gradient-based importance scores for all parameters across a few training steps, then creates a sparse binary mask selecting the top-k% most important parameters to update. The Fisher information of each parameter (squared gradient magnitude averaged over data) approximates its importance to the task.

**Child-Tuning** randomly samples a subnetwork of the model to update each step, with the sampled subset determined by task-specific gradient importance. Unlike random dropout which is training regularisation, Child-Tuning explicitly selects high-importance sub-networks for updates.

> **Interview question:** BitFit updates only biases yet achieves competitive performance on many tasks. Why would changing only biases be sufficient for task adaptation?
>
> *Biases in each layer are additive offsets that shift the distribution of activations before the non-linearity. In ReLU/SiLU activations, whether a neuron fires depends on whether `Wx + b > 0`. Adjusting `b` changes which neurons activate without changing what patterns they detect (governed by `W`). Many task adaptations are essentially "which features matter for this task" — selecting a subset of the model's existing feature detectors to fire more or less readily. Biases are the most direct handle for this. The limitations: biases can't create new feature detectors or change directional representations. Tasks requiring fundamentally new capabilities (learning a new language, a new reasoning skill) will exceed what bias shifts can achieve. BitFit works best for tasks where the base model already has the right features, just needs to be nudged to use them.*

### Side Tuning
{: #side-tuning}

Both adapters and LoRA still require **backpropagating gradients through the entire frozen base model** — because the adapter/LoRA parameters are embedded inside the model, the gradient must flow through every frozen layer to reach them. This means storing all intermediate activations in memory during the forward pass: for a 70B model, ~197 GB of activation memory regardless of how few parameters are being trained.

**Side tuning** eliminates this by routing backpropagation through a separate, smaller **side network** rather than through the base model. Information flows from base to side via downsampled residual connections, but gradients never flow back through the base:

<div class="post-flow" role="group" aria-label="Side tuning information flow">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Base LLM: quantised, frozen, forward-only — no activations stored, no grad tracking</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each base layer output → linear downsample → inject into parallel side network layer</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Side network: small trainable transformer in BF16 (e.g. 1B params for a 70B base)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Backward pass stays entirely within the side network — base is never touched</span></li>
  </ol>
</div>

**Quantised Side Tuning (QST)** combines 4-bit double quantisation of the base with a side network:

| Method | Weights | Optimizer | Activations | Total (70B) |
|--------|---------|-----------|-------------|-------------|
| Full fine-tuning | 140 GB | 280 GB | 197 GB | 617 GB |
| QLoRA | 35 GB | 2 GB | 197 GB | 234 GB |
| QST | 35 GB | 4 GB | 69 GB | 108 GB |

QST matches QLoRA's accuracy at less than half the total memory. The activation memory drop (197 → 69 GB) is the key win — because backprop stays within the small side network, only that network's activations need to be stored, not the full 70B base's.

> **Interview question:** Side tuning uses a smaller network that runs alongside the frozen base. But the side network still needs to "see" what the base is doing at each layer. What's the computational cost of these downsampled injections?
>
> *Each injection is a linear projection from the base's hidden dimension d_base (e.g. 8192 for 70B) down to the side network's dimension d_side (e.g. 1024). This is a `d_base × d_side` matrix applied to every token at every layer — for 96 layers and sequence length 2048, that's 96 × 2048 × 8192 × 1024 ≈ 1.6T operations just for injections. This is significant — typically 20–30% of the side network's own computation. The key is that these are simple linear projections (no non-linearity), which are extremely fast on tensor cores. The memory cost is also small: the injection weights are `n_layers × d_base × d_side` = 96 × 8192 × 1024 ≈ 805M parameters × 2 bytes = ~1.6GB, dwarfed by the base model.*

### Method Comparison
{: #peft-comparison}

| Method | Trainable params | Inference overhead | Backprop through base | Multi-adapter serving |
|--------|-----------------|-------------------|-----------------------|----------------------|
| Prompt tuning | l × d (tiny) | Extra input tokens | Yes | Trivial (swap prompt) |
| Prefix tuning | l × L × 2d | Extra KV per layer | Yes | Trivial (swap prefix) |
| Adapter (sequential) | 2 × 2 × d × r per layer | 2 matmuls per layer | Yes | Requires adapter hot-swap |
| LoRA | 2 × d × r per layer | Zero (merge at deploy) | Yes | Hot-swap A,B matrices |
| QLoRA | 2 × d × r (BF16) | Zero (after merge) | Via 4-bit base | Same as LoRA |
| DoRA | 2dr + d per layer | Zero (merge) | Yes | Same as LoRA |
| BitFit | #biases (~0.1%) | Zero | Yes | Trivial |
| Side tuning | Side network | Extra side network | No (side only) | Complex |

---

## Alignment & RLHF
{: #alignment}

PEFT methods adapt a model to a task given labelled input-output pairs. **Alignment** is a different goal: shaping model behaviour to be helpful, harmless, and honest in open-ended conversation — where there is no single correct output and human preferences are the signal.

### Supervised Fine-tuning
{: #sft}

The first stage of alignment: fine-tune the base pretrained model on a dataset of **high-quality demonstrations** — human-written conversations, instruction-response pairs, and chain-of-thought examples. This teaches the model the *format* of helpful responses and how to follow instructions.

**Data quality dominates data quantity for SFT.** LIMA (Less Is More for Alignment) showed that 1,000 carefully curated examples can match the performance of models trained on 52,000 examples. The bottleneck is not the amount of data but its diversity and quality — covering the range of instruction types, response formats, and domain areas you care about.

**What SFT teaches vs what it can't.** SFT learns to produce text *in the style* of the demonstrations. If the demonstrations are helpful, polite, and well-structured, the model learns to mimic that style. But SFT cannot: (1) distinguish correct from plausible-sounding incorrect answers when both appear in the training data, (2) consistently refuse harmful requests unless the demonstrations systematically include such refusals, (3) learn preferences between responses of different quality unless the training signal distinguishes them.

> **Interview question:** Why is the SFT stage necessary before RLHF? Can you skip it and go directly from the pretrained model to PPO or DPO?
>
> *Technically you can, but it fails in practice. The pretrained model has learned to predict text — it will generate continuations that statistically follow the pretraining distribution, which includes everything from news articles to Reddit arguments to code. When the reward model tries to score these completions, the distribution mismatch is enormous: the policy is generating tokens that look like web text, not instruction responses. The reward model was trained on instruction-following responses and gives garbage scores on arbitrary web-text completions. The KL penalty in PPO keeps the policy close to the starting point — if the starting point is the pretrained model generating web text, the policy can't escape to good instruction-following behaviour within a reasonable number of steps. SFT moves the starting distribution into the right neighbourhood, so RLHF is fine-tuning within a region where the reward model's feedback is meaningful.*

### Reward Modelling
{: #reward-model}

A **reward model (RM)** is trained to score model outputs according to human preferences. Training data: **pairwise comparisons** — for the same prompt, a human annotator picks the preferred response from two options.

The RM is initialised from the SFT model (same architecture) with a scalar projection head replacing the language model head. It is trained with the **Bradley-Terry pairwise ranking loss**:

```
L_RM = -E_{(x, y_w, y_l)} [ log σ( r(x, y_w) - r(x, y_l) ) ]
```

where `y_w` is the preferred response, `y_l` the rejected, and `r(x, y)` is the scalar reward score. Minimising this loss pushes `r(x, y_w) > r(x, y_l)` — preferred responses get higher scores.

**Why Bradley-Terry and not a direct regression loss?** A regression loss (e.g. MSE to a target score) requires absolute numerical labels — "this response is a 7/10". Human annotators are much better at relative comparisons ("A is better than B") than absolute ratings, which are noisy, subjective, and inconsistent across annotators. Bradley-Terry models the probability that one item is preferred over another as a logistic function of their score difference, which only requires relative preferences as training signal.

**Reward hacking.** The reward model is imperfect — it's a neural network trained on finite human labels. The policy can find input patterns that exploit the reward model's blind spots, producing outputs with high reward that are nonsensical or harmful to actual users. This is called reward hacking (or "Goodhart's Law for reward models"). The KL penalty in PPO is the primary defence.

> **Interview question:** Your reward model achieves 80% pairwise accuracy on a held-out preference dataset. Is this good? How would you know if it's good enough for RLHF training?
>
> *80% pairwise accuracy sounds decent but is not sufficient information. Key questions: (1) What is human-human agreement on the same pairs? If two annotators agree 75% of the time, 80% model accuracy is effectively perfect — you can't do better than inter-annotator agreement. (2) Does accuracy vary by topic? A reward model can achieve 80% overall by being very accurate on easy cases (clear helpfulness vs clear harm) while being random on subtle trade-offs. (3) What is the reward model's calibration — do high-confidence predictions actually have higher accuracy? Poor calibration means the model's scores can be exploited. (4) Does the reward model generalise to out-of-distribution responses? The RLHF policy will generate responses unlike the training distribution; the RM must be robust to these. In practice: run a small RLHF loop and measure whether final policy outputs are preferred by humans over SFT outputs.*

### PPO
{: #ppo}

With a trained reward model, the SFT model is further updated using **Proximal Policy Optimisation (PPO)** — an RL algorithm designed for stable policy gradient updates.

The SFT model is the **policy** `π_θ`: it takes a prompt as state and generates a response token-by-token. The reward model scores the completed response. PPO maximises:

```
J(θ) = E_{x~D, y~π_θ} [ r(x,y) - β · KL(π_θ(·|x) ‖ π_ref(·|x)) ]
```

where `π_ref` is the frozen SFT model and `β` controls how far the policy can drift. The KL term penalises the policy for generating token distributions that deviate from the SFT baseline.

<div class="post-flow" role="group" aria-label="PPO RLHF training loop">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Sample batch of prompts from dataset</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Policy π_θ generates full responses autoregressively</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Reward model scores each response → scalar r(x,y)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">KL divergence computed: β · KL(π_θ ‖ π_ref) subtracted from reward</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Value network estimates baseline return per token</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">PPO clipped surrogate loss updates π_θ — clip ratio prevents large updates</span></li>
  </ol>
</div>

**PPO's clipping mechanism.** The PPO objective clips the policy ratio `π_θ/π_old` to `[1-ε, 1+ε]` (typically `ε=0.2`):

```
L_CLIP = E [ min( r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t ) ]
```

where `r_t = π_θ(a|s) / π_old(a|s)` is the importance ratio and `A_t` is the advantage. Clipping prevents the policy from moving too far from where the current gradient estimate was computed — large updates can be inaccurate because the advantage estimate was computed under the old policy.

**Memory cost of PPO.** Four models live in GPU memory simultaneously:
1. Policy `π_θ` (trained, BF16)
2. Reference policy `π_ref` (frozen SFT copy, for KL)
3. Reward model (frozen, for scoring)
4. Value network (trained, for advantage estimation)

For a 7B model this is 4 × ~14GB = ~56 GB minimum, before activations or optimizer state. This is why PPO is expensive — it requires 4× the serving memory of a single model.

> **Interview question:** Why does PPO require a value network (critic) in addition to the policy? What would happen if you removed it?
>
> *The value network estimates the expected cumulative reward from the current state — the "baseline" or "critic". Without it, the advantage estimate `A_t = r_t - baseline` reduces to just `A_t = r_t` (raw rewards). The problem with raw rewards: high variance. The reward `r(x,y)` is a noisy signal assigned to the entire response; attribution to individual token decisions is ambiguous. If one response gets reward 1.5 and another gets 0.5, we want to update token decisions that caused the difference — but both responses may have had identical good tokens and only differed at one decision point. Without a baseline, every token in the high-reward response gets credited equally, including the many tokens that contributed nothing to the quality difference. The value network provides a per-state baseline that reduces gradient variance, making learning more sample-efficient. Removing it makes training extremely noisy — you'd need 5–10× more samples to achieve the same policy improvement.*

### DPO
{: #dpo}

**Direct Preference Optimisation (DPO)** bypasses the reward model and PPO entirely. It derives a closed-form loss that directly optimises the policy on preference pairs.

**The key insight**: in RLHF with KL regularisation, there is a unique optimal policy:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)
```

This implies the optimal reward can be written in terms of the optimal policy:

```
r*(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)
```

Substituting this into the Bradley-Terry preference model and noting that `Z(x)` cancels in the pairwise comparison, you get the DPO loss:

```
L_DPO = -E [ log σ( β · log(π_θ(y_w|x) / π_ref(y_w|x))  -  β · log(π_θ(y_l|x) / π_ref(y_l|x)) ) ]
```

This is a simple classification loss: increase the log-likelihood of preferred responses relative to the reference, decrease the log-likelihood of rejected responses relative to the reference. No reward model, no sampling loop, no value network.

<div class="post-flow post-flow--compare" role="group" aria-label="PPO vs DPO">
  <div class="post-flow__col">
    <p class="post-flow__col-label">PPO (RLHF)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Requires separate reward model training</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">4 models in memory: policy + ref + RM + value</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Online: generates new completions each step</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Can incorporate new preference labels during training</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Better on code generation (online exploration)</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">DPO</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">No reward model needed</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">2 models: policy + frozen reference</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Offline: trains on fixed preference dataset</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Cannot incorporate online feedback</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Simpler, stable, widely preferred for NLP alignment</span></li>
    </ol>
  </div>
</div>

**DPO's known failure mode: distribution shift.** DPO is an offline algorithm — it trains on a fixed preference dataset. The preferred and rejected responses in the dataset were generated by some prior model. If the policy being trained drifts far from that prior model's distribution, the preference labels may no longer be valid for the policy's own outputs. Iterative DPO (re-sample from the current policy, re-label, re-train) addresses this but partially re-introduces the online complexity of PPO.

> **Interview question:** DPO increases the likelihood of y_w and decreases the likelihood of y_l. But what stops it from just making y_l have very low probability while leaving y_w unchanged — is that a valid solution?
>
> *Yes, and it's a real problem. The DPO loss can be minimised by two distinct mechanisms: (1) increasing `log π(y_w)` — moving the policy to assign more probability to preferred responses (the desired behaviour); or (2) decreasing `log π(y_l)` — simply suppressing rejected responses regardless of quality. The second path is degenerate — the policy learns to be very conservative (suppress anything the training set labeled as rejected) rather than actually learning what makes responses good. In practice, DPO implementations monitor both `log π(y_w)` and `log π(y_l)` separately. If `log π(y_l)` decreases rapidly while `log π(y_w)` barely changes, the training is collapsing. Fix: (1) SFT warmup on y_w before DPO; (2) reference policy regularisation strength β — higher β keeps the policy closer to π_ref, preventing extreme suppression; (3) constrain the loss to penalise cases where `log π(y_w)` falls below π_ref.*

### ORPO
{: #orpo}

**Odds-Ratio Preference Optimisation (ORPO)** eliminates the need for a reference model entirely by combining SFT and preference alignment into a single training objective:

```
L_ORPO = L_SFT + λ · L_OR
```

where `L_SFT = -log P(y_w|x)` is the standard cross-entropy on preferred responses, and the odds-ratio loss penalises the tendency to generate rejected responses:

```
L_OR = -log σ( log( odds(y_w|x) / odds(y_l|x) ) )

odds(y|x) = P(y|x) / (1 - P(y|x))
```

The key advantage: no separate SFT phase needed, no reference model in memory. ORPO trains in one stage on preference pairs. The SFT loss ensures the model learns from preferred outputs; the OR loss penalises rejected outputs. This is 2× more parameter-efficient than DPO (no reference model) and eliminates the two-stage pipeline.

### GRPO
{: #grpo}

**Group Relative Policy Optimisation (GRPO)**, used in DeepSeek-R1, eliminates the value network from PPO while keeping the online RL loop. For each prompt, it samples `G` completions from the current policy and uses their relative rewards as baselines:

```
A_i = (r_i - mean(r_1,...,r_G)) / std(r_1,...,r_G)
```

The advantage of the i-th completion is its reward normalised relative to the group — no learned value function needed. This removes one of PPO's four models from memory and stabilises training by making advantage estimates self-normalising. GRPO is used for reasoning tasks where verifiable rewards (correct/incorrect math answers) provide clean training signal without human annotation.

> **Interview question:** GRPO samples G=8 completions per prompt and normalises rewards within the group. What happens if all G completions are equally good (or equally bad)?
>
> *If all G completions receive the same reward, the normalised advantages are all zero: `(r_i - mean) / std` with zero std is undefined (or zero). The policy gradient is zero — no update. This is actually fine: if the policy is already generating consistently good (or bad) responses to a prompt, there's nothing to learn from relative comparisons on that prompt. GRPO implementations handle the zero-std case by adding a small ε to the denominator or skipping the update for that batch. More interesting: if std is very small (all completions nearly identical reward), advantages are near zero and the gradient signal is weak. This means GRPO has reduced signal on "easy" prompts (model already solves them consistently) and "impossible" prompts (model never gets them right) — which is actually desirable. It concentrates training signal on prompts where the model is on the boundary, similar to curriculum learning.*

---

## Catastrophic Forgetting
{: #catastrophic-forgetting}

**Catastrophic forgetting** is the tendency of a neural network to abruptly lose previously learned information when trained on new data. In LLM fine-tuning, aggressive training on a narrow task can overwrite general capabilities — a model fine-tuned only on medical QA may lose the ability to write code.

The mechanism: gradient updates during fine-tuning move weights in the direction that minimises the fine-tuning loss. This direction is unconstrained — it can be orthogonal or even opposed to the direction needed to preserve pretraining capabilities.

**Mitigation strategies:**

| Strategy | Mechanism | Cost |
|----------|-----------|------|
| Low learning rate | Smaller weight updates, less overwriting | Slower convergence |
| Replay / data mixing | Mix 10-20% pretraining data into fine-tuning batches | Requires access to pretraining data |
| Elastic Weight Consolidation (EWC) | Add regularisation term penalising movement of important weights | Requires computing Fisher information |
| PEFT methods | Constrain updates to low-rank subspace — less room to overwrite | Slightly lower peak accuracy |
| Gradient projection | Project fine-tuning gradients to be orthogonal to pretraining gradient space | Computationally expensive |

> **Interview question:** You are continually fine-tuning a model — first on task A, then on task B, then on task C. Each new fine-tuning run degrades performance on the previous tasks. Propose a solution that doesn't require storing all previous training data.
>
> *Elastic Weight Consolidation (EWC): after fine-tuning on each task, compute the Fisher information matrix F_A (approximated as squared gradient magnitudes) over the task-A dataset. F_A identifies which weights are most important to task A. When fine-tuning on task B, add a regularisation term: `L_B + λ·Σᵢ F_A_i · (θ_i - θ_A_i)²` — this penalises moving weights that were important to task A, while allowing free movement of weights that weren't. Repeat for each subsequent task. The limitation: F scales with the number of tasks — storing F for N tasks requires N × #params storage. Progressive Neural Networks solve this differently: freeze all previous columns and add a new network column for each task, connected to previous columns via lateral connections. No forgetting by construction, but memory grows linearly with tasks.*

---

## Practical Considerations
{: #practical}

**Choosing between PEFT methods.** The decision tree:

<div class="post-flow" role="group" aria-label="PEFT method selection">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Memory severely constrained (single consumer GPU, &lt;24GB) → QLoRA</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Need to serve many fine-tuned variants from one base → LoRA (hot-swap adapters)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Inference latency is critical, zero overhead required → LoRA or DoRA (merge at deploy)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Very few labels, large model (&gt;10B) → Prompt tuning or prefix tuning</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Maximum quality, compute available → Full fine-tuning or DoRA at high rank</span></li>
  </ol>
</div>

**LoRA hyperparameter selection.** The most important choices:
- `r` (rank): 8–16 for instruction tuning; 32–64 for domain adaptation; 128+ for skill acquisition
- `α` (scaling): typically `α = r` or `α = 2r`; higher α strengthens the adapter's contribution
- Which modules to apply LoRA to: at minimum Q, V; ideally Q, K, V, O, and FFN up/down projections
- Learning rate: 3e-4 to 1e-3 for LoRA (higher than full fine-tuning because fewer parameters)

**Alignment pipeline selection:**
- **DPO**: simplest, most widely used, good for chat/instruction following
- **PPO**: necessary when you need online exploration — e.g. code generation where you can verify outputs automatically
- **ORPO**: one-stage alternative to DPO, good when you don't want a reference model
- **GRPO**: for reasoning tasks with verifiable rewards (math, code)

> **Interview question:** You're fine-tuning a 70B model using QLoRA on a single 8×80GB GPU node. The training is running but is 3× slower than you expected. What are the most likely bottlenecks and how would you diagnose them?
>
> *Likely bottlenecks in order of frequency: (1) GPU utilisation — run `nvidia-smi` during training. If GPU util is &lt;80%, you're compute-starved, likely due to small batch size or CPU data loading bottleneck. Increase batch size (use gradient accumulation if it doesn't fit in memory). (2) Data loading — if GPU util shows gaps (drops to 0% regularly), the CPU data pipeline can't keep up. Use `num_workers > 0` in DataLoader and pin_memory=True. (3) 4-bit dequantisation overhead — QLoRA must dequantise base weights before each forward pass. If the model architecture has many small matrix operations (e.g. GQA with many KV heads), dequantisation overhead is disproportionate. Try disabling quantisation temporarily to measure. (4) LoRA configuration — if LoRA is applied to too many modules (every linear layer), the trainable parameter count grows and so does gradient computation. Profile with `torch.profiler` to see which operations dominate. (5) Sequence length padding — if batches have high padding ratios, you're computing attention on padding tokens. Use dynamic padding (pad to longest in batch) or sequence packing.*

---

## Post-Training Lifecycle: Structured Data to RLVR
{: #post-training-lifecycle}

Pre-training learns statistical regularities from unstructured token sequences. Post-training teaches the model to operate inside a **protocol** — roles, turns, tool schemas, and verifiable success criteria. The data type changes completely:

| Stage | Data type | Objective | Failure mode |
|---|---|---|---|
| Pre-training | Unstructured text sequences | NTP over all tokens | Contamination, duplication, memorisation |
| SFT | Structured chat transcripts with role tags | NTP on assistant spans only | Imitates bad data, brittle to schema drift |
| Preference | Comparison pairs (x, y⁺, y⁻) | Shift distribution toward preferred outputs | Length bias, reward hacking, off-policy staleness |
| RLVR | Prompts + verifiers | Maximise verified task success | Verifier exploitation, mode collapse |

**One-sentence reframe:** post-training is changing the *distribution of token sequences* the model is trained to produce — from web-text completion to protocol-compliant, verifiably correct assistant behavior.

### Chat Templates & Loss Masking
{: #chat-templates}

The **chat template** is the serialisation format that converts structured message objects (role + content + tool schemas) into a flat token sequence the model trains on. It is the bridge between the developer API and the CLM objective.

```
messages = [
  {"role": "system",  "content": "You are a support agent."},
  {"role": "user",    "content": "Refund order #A-1930."},
]
tools = [{"name": "order_lookup", "parameters": {...}}]
```

Materialises to:

```
<|system|>
You are a support agent.
# Tools
## order_lookup
Fetch order details.
Parameters: {"type":"object","properties":{"order_id":{"type":"string"}},...}
<|end|>
<|user|>
Refund order #A-1930.
<|end|>
<|assistant|>
```

**Why this matters beyond formatting.** Two models with identical weights but different chat templates behave differently — separator tokens, whether tool schemas are in-band or out-of-band, and how tool outputs are tagged all change the token sequence the model conditions on. A template mismatch between training and inference is a distribution shift that degrades eval accuracy without any weight change.

**Loss masking is part of the protocol.** After materialisation, a binary mask `mₜ ∈ {0,1}` selects which tokens receive gradients:

```
L(θ) = -Σₜ mₜ · log pθ(xₜ | x<t)
```

What is masked *out* (context-only): system prompt, tool schemas, user messages, tool results. What is *supervised*: assistant natural-language text, tool-call JSON (name + arguments), reasoning/chain-of-thought tokens. Tool results are never supervised — they come from the environment. If you accidentally supervise tool results, the model learns to hallucinate environment outputs.

**Parsing the output.** The chat template serialises structured data in; the serving parser extracts it out. The two must stay in lockstep — changing delimiters requires updating both training materialisation and inference parsing. Common failure modes:

- Partial delimiters: model starts `<|tool_c` then switches to plain text
- Malformed JSON: trailing comma, truncation at max-length cutoff
- Hallucinated tools: model invents a function not in the catalog
- Interleaved reasoning: thinking tokens mixed into tool-call JSON

Special tokens (`<|tool_call|>`) are a single token ID — easy to detect and tokenised consistently. Text-based delimiters (`` ```json ``) are multi-token and ambiguous; avoid them in production schemas.

**Reasoning parsers.** Modern reasoning models (o1, DeepSeek-R1, Qwen3) emit `<|thinking|>…<|/thinking|>` blocks parsed separately from the answer. Reasoning tokens are supervised during training (the model learns *how* to think), but the serving layer strips them from user-facing responses. Some pipelines mask reasoning tokens to avoid constraining style; others supervise to teach step-by-step patterns — both are valid choices with different capability trade-offs.

> **Interview question:** Your production tool-calling model suddenly starts failing to parse its own tool calls — the JSON is malformed in ~15% of requests. No code changed. What happened and how do you fix it?
>
> *Likely cause: the tool schema changed (a field was renamed, an enum was expanded, whitespace was added) without retraining. Because schemas are injected as literal tokens, even minor schema reformatting changes the token IDs the model conditions on — and produces. A schema that was "camelCase" during training producing "snake_case" field names during inference creates a distribution shift where the model has never seen the inference token sequence during training. The model's token probability mass for the closing brace is now distributed across several unexpected paths, causing truncation or malformed output. Fix: (1) pin schemas — treat them as versioned contracts; changes require retraining or at least ablation testing. (2) Use special tokens for delimiters so the model unambiguously separates reasoning from tool-call JSON. (3) Add a JSON repair layer in the serving parser as a short-term mitigation. (4) Fine-tune on examples with the new schema before deploying it.*

### Tool Use as World Interaction
{: #tool-use-protocol}

Tool calling is not "generate JSON and parse a response." The model interacts with a **partially observable world** through a narrow API interface. Each tool call is an action; each tool result is a compressed observation of a world with countless latent variables.

<div class="post-flow" role="group" aria-label="Tool use protocol loop">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">User intent → model infers required action (which tool, which arguments)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Tool call emitted as JSON in token stream</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Tool result returned as tool-role message (never supervised)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Model reconstructs world state from compressed JSON and produces response</span></li>
  </ol>
</div>

A single `order_lookup` response like `{"status":"delivered","days_since_delivery":3,"refundable":true}` hides warehouse inventory state, shipping carrier status, payment processor state, customer history, and policy rules. The model must infer what actions are possible from this tiny observation.

**What the model must learn:** (1) intent — what does the user actually want? (2) action selection — which tool advances the goal? (3) state interpretation — reconstruct world state from compressed JSON payload; (4) planning under uncertainty — decide next action given incomplete information.

**Post-training data: the "gold trace" view.** A post-training episode is a trajectory:
- *Inputs:* messages + tool catalogs + retrieved evidence
- *Actions:* assistant tokens (including tool calls)
- *Observations:* tool outputs
- *Outcome:* verifier label or preference label

Once traces are stored, they can be repurposed: SFT (imitate good traces), RL (rank traces by advantage). On-policy traces (generated from the current checkpoint) are more valuable than off-policy traces from older models — the model's token distribution at training time must overlap with the trace's token space for gradients to be informative.

**Tool schema token budget.** Each tool's full JSON schema is injected into the system prompt. Two tools ≈ 130 tokens; 50 production tools can consume 3,000+ tokens before any conversation starts. Total context budget: `C_total = C_msgs + C_tools + C_retrieval + C_generation`. Operational rules: use canonical schemas (deterministic key ordering, no extra whitespace), version schemas like code, use a small "router" tool set and load specialised tools dynamically.

**Security boundary.** Tool outputs are untrusted input — they can contain adversarial strings that attempt to override system instructions or exfiltrate data. Wrap tool outputs as tool-role messages with lower authority than the system prompt. Filter tool outputs from untrusted sources before they enter context.

> **Interview question:** You have 80 tools in your production catalog. Your tool-selection accuracy is 85% on single-tool tasks but drops to 55% on multi-tool tasks. What are the likely causes and how do you improve it?
>
> *Root causes: (1) Context saturation — 80 tool schemas at ~60 tokens each = ~4,800 tokens of system prompt, leaving limited space for conversation history. At long conversations, earlier context is truncated, reducing the model's access to prior tool results needed for multi-tool planning. (2) Missing multi-step training data — the model was likely fine-tuned on single-tool examples; multi-tool trajectories require the model to track state across calls, and if this was absent from SFT data, the model has no learned strategy. (3) Schema ambiguity — with 80 tools, several likely have overlapping descriptions, causing selection errors. Fix: (1) Load tools dynamically — use a lightweight router to select the relevant 5–10 tools per request before injecting schemas, cutting tool prompt tokens by 8–16×. (2) Add multi-step gold traces to SFT data — especially traces where intermediate tool results gate subsequent decisions. (3) Audit tool descriptions for disambiguation — names and descriptions should be contrastive, not overlapping. (4) Add RLVR with an end-to-end task verifier that scores the full multi-tool trajectory, not just individual tool calls.*

### Structured Outputs & Schema Tokens
{: #structured-outputs}

**Structured outputs** (JSON Schema constraints on model generation) turn format requirements from a prompting convention into a formal contract. Schema-driven decoding at inference time constrains invalid tokens using a finite-state machine over the schema, eliminating parse failures.

**The low-probability path problem.** Constrained decoding prevents syntactically invalid JSON but cannot guarantee semantic correctness. More importantly: if the model was never trained on a schema structure, constrained decoding forces it down low-probability token paths — the output is syntactically valid but semantically wrong, because the model is generating tokens it has low confidence in. Example: a medical JSON schema requiring `"icd10_code"` fields — the model produces syntactically valid ICD-10 format but generates nonsensical codes if it has no distribution over valid ICD-10 values.

**Fix: train on the contracts you deploy.** Include deployed schemas in post-training data at all stages — SFT examples with the exact schema, preference pairs contrasting correct vs incorrect values *within* the valid schema, RLVR with the schema validator as one component of the verifier. This aligns the model's probability mass with the constrained decoding surface, making constrained paths also the high-probability paths.

**Schema versioning is operational.** Two "semantically identical" schemas can differ by 25% in token count just from whitespace and naming conventions (`order_id` → `orderId`, `"c"` → `"Celsius"`). Each change shifts every downstream token ID and probability. Treat schemas as versioned contracts: schema changes require either retraining or at minimum empirical validation that generation quality on the new schema is maintained.

> **Interview question:** You add JSON schema constraints to your production API. Accuracy on structured output tasks improves by 15% but latency increases by 40%. What's happening and how do you reduce latency overhead?
>
> *The latency increase comes from constrained decoding: at each step, the engine must evaluate which tokens are valid under the current schema state (a finite-state machine traversal), mask invalid tokens, then sample. For complex schemas with many fields, each step requires O(vocab_size) FSM checks. Fixes: (1) Cache the FSM state — most schemas are static; precompute the FSM and cache valid-token bitmasks per state, amortising computation across requests. (2) Reduce schema complexity — strip unnecessary fields, flatten nested objects, use `anyOf` sparingly. (3) Train the model to be naturally schema-compliant (SFT + RLVR on your schemas) — a well-trained model rarely tries invalid tokens, so constraint masking is a no-op most of the time, eliminating the overhead. (4) Use speculative decoding with a schema-aware draft model — the draft model proposes schema-compliant tokens at high speed, the larger model verifies. This recovers most of the constrained-decoding throughput cost.*

### SFT as Protocol Learning
{: #sft-protocol}

SFT is not fine-tuning on task examples — at post-training scale it is **teaching the model a protocol**: what roles mean, how turns are structured, when to call tools, how to format tool arguments, and how to use tool results correctly.

**SFT objective (assistant spans only):**

```
L_SFT(θ) = -Σ_{t ∈ A} log pθ(xₜ | x<t)
```

where `A` is the set of assistant token positions after loss masking. The gradient is zero on system, user, and tool-result tokens — those are context, not targets.

**What SFT teaches vs what it doesn't.** SFT from good demonstrations teaches the chat template distribution, canonical assistant style, when to call tools, how to format arguments under a schema, and multi-turn state management. What SFT cannot teach: fine-grained quality tradeoffs between two valid responses, robustness to adversarial or OOD inputs, and objective task correctness at scale (a model can imitate the format of a correct answer without being able to derive it).

**SFT dataset construction choices:**
- *On-policy vs off-policy demonstrations:* demonstrations from the current model (on-policy) generalise better than demonstrations from a much stronger model — the model learns from outputs in regions of token space it would actually visit.
- *Continued pre-training vs SFT:* continued pre-training keeps the NTP objective but changes the data distribution (more code, more domain text). SFT changes the loss masking and adds structural constraints. Choose continued pre-training for domain knowledge, SFT for behavioral protocol.
- *Rubric labels:* SFT datasets can include metadata (tone, correctness, safety flags) to condition on — models trained to produce these conditionally can be steered at inference time.

> **Interview question:** You train a model with SFT on 100k high-quality tool-use demonstrations and it achieves 90% tool-selection accuracy. When you deploy it, real-world accuracy is 60%. What happened?
>
> *Training-deployment distribution shift. SFT imitates a dataset — if the demonstration dataset does not cover the real-world distribution of requests, the model fails on OOD inputs. Likely causes: (1) The demonstrations covered a narrow set of tool combinations; real traffic has more diverse multi-tool queries. (2) Schema drift — the deployed schemas differ from training schemas (field names, whitespace, added fields). (3) Conversation length — demonstrations were short; real conversations are long, and the model loses track of prior tool results. (4) User phrasing — demonstration prompts were clean and explicit; real users are ambiguous. Fixes: (1) Collect real traffic logs and add failures to SFT data iteratively (online learning loop). (2) Audit schema versions and pin them. (3) Add longer multi-turn trajectories to training. (4) Add preference pairs where the preferred response handles ambiguous user intent correctly. (5) Add RLVR on task completion — SFT alone cannot close a 30pp gap; the model needs a correctness signal.*

### Preference Modeling
{: #preference-modeling}

SFT teaches the protocol; preference modeling encodes **what we want within the protocol** — quality, tone, safety, conciseness, schema compliance. The primitive is a comparison: for a prompt `x`, we have a preferred response `y⁺` and a dispreferred response `y⁻`.

**Bradley-Terry reward model:**

```
Pr(y⁺ ≻ y⁻ | x) = σ(rφ(x, y⁺) - rφ(x, y⁻))
L_RM = -log σ(rφ(x, y⁺) - rφ(x, y⁻))
```

**DPO (Direct Preference Optimisation):** skips the explicit reward model and directly updates the policy:

```
L_DPO = -log σ(β · (Δθ - Δref))
Δθ = log πθ(y⁺|x) - log πθ(y⁻|x)
```

where `β` controls how far the policy can move from the reference model. DPO is sensitive to off-policy data: `y⁺`, `y⁻` must come from regions the current policy `πθ` actually visits. Stale pairs from an older checkpoint push gradients toward token regions the current model would never reach.

**On-policiness matters.** Pairs are most useful when generated from the *current* policy checkpoint. As training progresses, earlier pairs become increasingly off-policy. Production preference pipelines iterate: sample from current checkpoint → label → train → sample again.

**Multi-objective preference tuning.** Real deployments optimize over several rubrics simultaneously:
- Helpfulness / task success
- Harmlessness / refusal correctness
- Verbosity / cost / latency proxies
- Formatting / schema compliance

Combined as weighted reward sums or conditional preferences. Pure preference optimization without regularisation drifts — the model loses capabilities and breaks tool protocols as it overfits to the preference signal:

```
max_θ E[V] - β · KL(πθ ∥ πref)
```

The KL term anchors the policy to the reference model, preventing catastrophic forgetting of pre-trained capabilities.

**Preference data pitfalls for tool use:**
- *Length bias:* labelers prefer longer answers even when they are wrong — normalise length in rubrics
- *Style over substance:* "sounds confident" beats "is correct" — add adversarial negatives with confident-sounding but wrong content
- *Schema preference pairs:* `y⁺` = valid JSON + correct semantics + correct enum values; `y⁻` = missing keys, invalid enums, or hallucinated field values

> **Interview question:** After DPO training your model scores better on human eval but worse on automated tool-call validation. What went wrong and how do you fix it?
>
> *DPO improved the response *style* (what human annotators prefer) while degrading functional correctness (what the verifier checks). The preference pairs likely reflected human annotator preferences — verbose, confident, well-formatted natural language — rather than functional tool-call correctness. Annotators may have preferred responses that explained tool decisions in natural language over terse but correctly formatted tool calls. Fix: (1) Add tool-specific preference pairs where y⁺ is defined by passing the schema validator and producing correct field values, not by human annotation. (2) Use composite rubrics: weight annotator preferences with automated tool-call validity scores so the labeling signal directly penalises schema violations. (3) Check reference model — if the DPO reference model was the pre-DPO checkpoint, and that checkpoint already had good tool-call accuracy, the DPO loss is trying to move away from the reference in ways that may accidentally degrade tool-call formatting. Constrain β more tightly or add a verifier-based reward as an additional term.*

### RLVR: Reinforcement Learning with Verifiable Rewards
{: #rlvr-deep}

Preference modeling encodes relative quality; RLVR optimises **task completion under a deterministic checker**. The reward is binary and comes from a verifier that can unambiguously confirm correctness — no subjective human judgment required, enabling massive sampling per prompt.

**RLVR objective:**

```
max_θ E_{x~D} E_{y~πθ(·|x)} V(x, y)
```

Common verifiers: unit/integration tests (software), answer checkers (math, multiple-choice), JSON schema validators, retrieval-grounding checks, safety policy checks.

**Training loop:**

<div class="post-flow" role="group" aria-label="RLVR training loop">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Sample prompt x from task distribution D</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Sample K rollouts y₁…yK from current policy πθ</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Run verifier V(x, yₖ) → binary reward for each rollout</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Policy update (GRPO/PPO) using advantage estimates from reward signal</span></li>
  </ol>
</div>

**Total cost ≈ (prompts) × K × (avg rollout length) + verifier cost.** Verifier efficiency is a first-class systems concern — cheap verifiers (schema validators, unit tests) enable larger K; expensive verifiers (model-based judges) constrain K.

**DeepSeek-R1 design choices.** Reward = correctness under a ground-truth verifier (outcome-only, no constraints on reasoning content). Group-normalised advantage via GRPO:

```
Aᵢ = (rᵢ - mean(r₁:G)) / std(r₁:G)
```

This reduces step-size sensitivity with binary rewards. Format rewards enforce structure not for aesthetics but to make the verifier robust (reliable output parsing). Multi-stage pipeline: rejection sampling → RL → SFT — pure RL alone hurts readability and general chat quality, so SFT on high-quality RL rollouts is used to restore it.

**Verifier design principles:**

| Property | Why it matters |
|---|---|
| Cheap to run at scale | Cost model: prompts × K × rollout length + verifier cost |
| High precision (few false positives) | False positives = model learns to exploit the verifier, not solve the task |
| Hard to game | Hidden test cases, randomised inputs, mutation testing |
| Aligned with true objective | Proxy metrics (passes linter) ≠ task success (correct output) |

Composite verifiers: `V(x,y) = ⊮[correct] · ⊮[format] · ⊮[safe]`. Conjunction keeps precision high. Partial credit (e.g., "compiles but fails 3 of 5 tests") is useful for curriculum but risks incentivising metric optimisation rather than task completion.

**RLVR failure modes:**

- *Verifier exploitation:* incomplete tests → model passes tests without solving the task. Fix: hidden + randomised test cases, mutation testing, adversarial prompts.
- *Mode collapse:* binary rewards collapse the policy toward narrow strategies. Fix: multiple rollouts per prompt, entropy/KL regularisation, diverse prompt distribution.
- *General capability regression:* RLVR on math/code can degrade instruction following and chat quality. Fix: multi-stage pipeline with SFT on high-quality RL rollouts to restore general quality.

**Measuring reasoning vs sampling luck.** `CoT-Pass@K` requires both reasoning trace and final answer to be correct — distinguishes genuine reasoning improvement from sampling luck. High `pass@K` with low `pass@1` means the model can solve the task but hasn't learned reliable reasoning paths; voting (cons@K) can recover quality at inference cost.

**RLVR-as-a-service (Reinforcement Fine-Tuning, RFT).** APIs now expose RLVR training where the developer provides a prompt dataset, a grader returning numeric reward, and evaluation splits. The bottleneck shifts from compute to verifier quality — any developer who can write a correct grader can run RL training. The verifier is a first-class artifact shipped alongside data, not an afterthought.

> **Interview question:** You use RLVR to train a code model on SWE-bench (test verifier = repo test suite). After 3000 RL steps, pass@1 improves from 20% to 40% on the training distribution but only 22% on a held-out set of new repos. What's happening and how do you fix it?
>
> *The model is overfitting to the training verifier's distribution — it has learned to exploit patterns specific to the repos and test styles in the training set, not the underlying coding skill. Classic reward hacking: the model found strategies that pass the specific tests (e.g., hardcoding edge cases, matching test fixture patterns) without generalising to new repos. Diagnoses: (1) Check if training prompts and test prompts have overlapping repo styles or dependencies — contamination at the task level. (2) Inspect failing rollouts on the held-out set — are they failing because of missing programming skill, or because of legitimate edge cases the model never saw? Fixes: (1) Expand training task diversity — more repos, more languages, more test styles, to prevent strategy collapse to any specific subset. (2) Randomise test cases — use mutation testing to generate variants that are behaviorally equivalent but syntactically different, preventing the model from pattern-matching tests. (3) Evaluate with stronger verifiers — hidden test cases not seen during training. (4) Add KL regularisation — prevent the policy from drifting too far from the pre-RLVR checkpoint, which constrains the space of exploitable strategies. (5) Use a curriculum: train on easier, well-covered tasks first; expand to harder, more diverse tasks as policy stabilises.*

### Post-Training Evaluation
{: #post-training-eval}

Evaluation is a system, not a score. It turns model behaviour into decisions: ship vs don't ship, pick a checkpoint, identify what SFT data to add, or which verifier to improve.

**Three evaluation tiers:**

| Tier | Methods | What it catches |
|---|---|---|
| Offline / benchmark | Static datasets, reproducible runs, regression tests | Capability regressions, benchmark trends |
| Online / product | A/B tests, user satisfaction, cost/latency | Real distribution, product-level quality |
| Adversarial / safety | Jailbreak attempts, red teaming | Safety regressions, policy violations |

**Modern benchmark taxonomy.** Post-training improvements are non-uniform — RLVR on math can regress instruction following; SFT with English tool data can hurt multilingual performance. Report by capability slice:

| Category | Example benchmarks | Post-training connection |
|---|---|---|
| Knowledge / factuality | MMLU-Pro, SuperGPQA, SimpleQA | Preference training can increase hallucination by rewarding confident answers |
| STEM & reasoning | GPQA Diamond, HLE, HMMT | Tests whether RLVR produces genuine reasoning or pattern matching |
| Coding | SWE-bench Verified, LiveCodeBench, Terminal-Bench | Canonical RLVR target: verifier = test suite, reward = pass/fail |
| Instruction following | IFEval, MultiChallenge, IFBench | Tests whether SFT taught robust constraint satisfaction |
| Long context | LongBench v2, AA-LCR | 256k window + short-context SFT < 128k + well-curated long-context post-training |
| Agent / tool use | BFCL-V4, TAU2-Bench, OSWorld | Tests full agentic loop: intent → action → state interpretation → next action |
| Multilingual | MMMLU, WMT24++, PolyMATH | English tool SFT can overwrite multilingual priors |

**Contamination resistance.** As models and datasets spread, test sets leak into training. Modern defences: live/rolling benchmarks with fresh tasks (LiveBench), private eval sets with strong access control, verifiable tasks where answers are hard to memorise (fresh competitive programming contests, new GitHub repos).

**The harness is part of the score.** Same model on SWE-bench varies 10–20 percentage points depending on scaffold, context management, and tool budgets:
- Basic (issue + full repo dump): 20–30%
- Oracle retrieval (issue + relevant files only): 40–50%
- Agentic (iterative search, test, retry): 50–70%+

Always report: harness version, prompt template, sampling params (temperature, top-p), tool policies, context management strategy, number of attempts, and compute budget. A score without harness details is not reproducible — it measures harness + model, not the model.

**pass@K and what each metric reveals:**

- `pass@1` (one greedy sample correct): modal capability — can the model reliably solve it?
- `pass@K` (any of K correct): coverage — does a solution exist in the model's distribution?
- `cons@K` (majority vote correct): reliability — voting recovers hidden correctness

High `pass@K`, low `pass@1`: model can solve it but not reliably. Sampling at inference time helps, but adds cost. High `cons@K`, low `pass@1`: voting recovers correctness — use majority decoding. `pass@1 ≈ pass@K`: either confident and calibrated, or hopelessly stuck.

**Deployment gates and monitoring.** Evaluation must drive deployment decisions:
- Safety gates: no regressions beyond threshold on red-team evaluations
- Schema validity ≥ 99.9% on critical production flows
- SWE-bench improvement without tool-call regression
- Latency/cost budget not exceeded

Production monitoring = continuous eval on live traffic: tool failure rates, argument validity, schema adherence, safety trigger rates, latency regressions. Live logs become the next training data — closing the loop from evaluation to SFT and RLVR.

> **Interview question:** Your team reports MMLU-Pro improved by 4 points after a new RLVR run on math tasks. Leadership wants to ship. What do you check before approving?
>
> *MMLU-Pro improvement is one signal — it doesn't tell you whether you're ready to ship. Checklist: (1) Slice-level regressions — RLVR on math can regress instruction following, multilingual, and tool-use slices. Run the full eval suite across all capability categories, not just MMLU-Pro. (2) Safety evaluation — new RL training can introduce unexpected safety regressions if the policy drifted far from the reference. Run red-team eval and compare to the previous checkpoint. (3) Schema and tool-call validity — if this model serves tool-use endpoints, verify that tool-call accuracy and schema adherence are maintained. (4) Long-context performance — RLVR often trains on short contexts; check LongBench v2 and needle-in-haystack for regressions. (5) Harness parity — confirm the MMLU-Pro improvement holds on your internal eval harness (same prompt template, sampling params as the previous comparison); MMLU-Pro scores can vary 2–3pp just from prompt formatting. (6) Contamination check — verify the new training data does not overlap with MMLU-Pro questions at the n-gram level. (7) A/B shadow traffic — run 1–5% of live traffic through the new model and monitor tool failure rates, user satisfaction, and latency before full rollout.*

---

## SFT Data Engineering & RL Environments
{: #sft-data-engineering}

The previous section established *what* post-training objectives look like. This section goes deeper into *what data and feedback actually cause a model to improve* on each behavior — moving from the taxonomy of training objectives to the concrete engineering of SFT corpora and RL environments.

**Core lens:** post-training is a data-and-feedback stack. Protocol SFT sets defaults; vertical SFT teaches domain skills; environments with verifiers let you explore and scale beyond the high-water mark SFT alone can reach.

| Data type | Record format | What it teaches | Limitation |
|---|---|---|---|
| SFT (static) | (x, y) pairs with loss mask | Protocol, format, tool syntax, domain defaults | Cannot discover strategies absent from data |
| Preference (static) | (x, y⁺, y⁻) comparisons | Relative quality, safety, style tradeoffs | Annotator bias, off-policy staleness |
| RL environment (interactive) | State → action → reward trajectories | Error recovery, multi-step credit assignment, on-policy exploration | Engineering cost, verifier precision required |

### SFT Record Anatomy & Mixture Design
{: #sft-record-anatomy}

**SFT is behavioral cloning over structured contexts.** Every SFT example has: a structured context `x` (messages, tools, metadata) and a target continuation `y` — the model maximises likelihood of assistant tokens only:

```
L_SFT(θ) = E_{(x,y)~D_SFT} Σ_{t=1}^{|y|} mₜ · log pθ(yₜ | x, y<t)
```

where `mₜ ∈ {0,1}` masks non-target tokens (system, user, tool outputs). **Crucially: the model cannot learn any strategy absent from the training data.** It can only imitate demonstrated continuations.

**Minimum SFT record fields:**

```json
{
  "messages": [{"role": "system", "content": "..."},
               {"role": "user",   "content": "..."},
               {"role": "assistant", "content": "..."}],
  "tools":    [{"name": "...", "parameters": {...}}],
  "metadata": {"vertical": "coding", "difficulty": "hard", "lang": "en"},
  "loss_mask": "assistant_only"
}
```

The `metadata` field is not cosmetic — it powers slice-level mixture weights and regression analysis. Without it, you cannot diagnose which data change caused a behavior change.

**Protocol SFT vs Vertical SFT.** Protocol SFT teaches roles, refusal policy, tool-call syntax, and output formatting — the universal defaults the model applies everywhere. Vertical SFT teaches domain-specific skills for each of the 8 capability slices (knowledge, reasoning, coding, instruction following, long context, agents, search, multilingual). Both must be present; protocol SFT without vertical SFT produces a polite model that cannot solve hard problems; vertical SFT without protocol SFT produces a capable model that breaks on tool calls and refuses incorrectly.

**Mixture design is a hidden hyperparameter.** The SFT corpus is a weighted mixture:

```
D_SFT = Σₖ wₖ · Dₖ,   Σₖ wₖ = 1
```

Changing `wₖ` changes what the model does *by default*: how often it uses tools, which languages it handles fluently, how cautiously it refuses. Data biases propagate directly through SFT: annotator demographics narrow what "natural" responses look like; over-represented topics create confident-but-wrong behaviour outside them; fixed orderings in agent traces cause failures when tool order changes at inference. Evaluate per-slice before and after mixture changes.

**Five quality dimensions that matter:**
1. *Correctness* — factual, logical, or functional accuracy (is it verifiable?)
2. *Constraint adherence* — formats, style, policies, tool schemas
3. *Coverage* — diversity over intents, domains, languages, edge cases
4. *Calibration* — appropriate uncertainty, hedging, abstention
5. *Robustness* — stable under prompt perturbations; not template-overfit

If you don't measure a dimension, training quietly optimises a proxy instead — typically verbosity, confidence, or aesthetic style.

> **Interview question:** You add 50k new SFT examples to your coding vertical and pass@1 on SWE-bench improves by 3pp, but IFEval drops by 4pp. What happened and how do you fix it?
>
> *The new coding examples shifted mixture weights: the model sees more low-level code debugging and less constraint-following text. SFT is behavioural cloning — adding more of one slice reduces the effective fraction of others (even with fixed absolute counts, if the new data is heavier). IFEval tests multi-constraint satisfaction; that skill regressed because the model is now optimising for code-style continuations. Fix: (1) Add the new coding data without removing anything — preserve absolute count of instruction-following examples. (2) Check whether the coding examples use structured outputs or strict formats; if not, add examples that overlap (e.g. "write a function that produces output conforming to this JSON schema"). (3) Run full vertical evaluation before and after any mixture change, not just the target vertical.*

### Behavior-Vertical Corpora
{: #vertical-corpora}

Each of the 8 capability verticals requires different context shapes, output artifacts, and verification methods:

| Vertical | Typical input | Target output | Verifier |
|---|---|---|---|
| Knowledge / factuality | Query + optional docs + search tool | Grounded answer with citations | Citation parser + entailment check + judge |
| STEM & reasoning | Problem statement + constraints | Final answer + optional CoT | Symbolic/numeric equivalence check |
| Coding | Repo context + failing tests | Patch/diff | Compile + test suite (fail-to-pass + pass-to-pass) |
| Instruction following | Multi-constraint prompt | Constrained structured output | Schema validator + regex + length/count checks |
| Long context | Multi-doc bundle + distractors | Answer with span citations | Span-match + contradiction check |
| General agent | Stateful dialog + tools + errors | Multi-turn trajectory with recovery | Task success predicate in sandbox |
| Search agent | Browsing tool + source constraints | Cited synthesis across hops | Grounding check + evidence coverage + citation parser |
| Multilingual | Locale instructions + cultural norms | Correct in language + style | Back-translation + cross-lingual consistency |

**Knowledge vertical — the hardest to get right.** The model must learn three distinct modes: (1) answer from parametric knowledge with honest hedging when uncertain; (2) decide when to invoke search (time-sensitive facts, niche domains, specific numbers); (3) after retrieval, commit to the evidence — not blend in parametric guesses that contradict sources. "Citation laundering" — citations present but irrelevant — is the key failure mode. Verification cascade: citation parser → entailment check (claim follows from cited passage?) → search-decision audit → judge for remaining quality dimensions.

**Coding vertical — the canonical RL target.** MiniMax's SWE pipeline illustrates scale: mine merged GitHub PRs with associated tests, build a runnable Docker environment per PR (agent-driven, with self-correction), extract fail-to-pass tests (must flip after the fix) and pass-to-pass tests (must stay green), then have the model attempt the fix in a sandbox — only patches passing *both* test sets enter the SFT set. Result: 140k+ SFT tasks, 10k+ runnable PRs, 10+ languages. Each PR is reused as multiple task types: bug-fix, test-writing, code-review, difficulty boost via merged-commit context.

**Agentic SFT — trajectory data, not single turns.** Kimi K2's pipeline: gather 3,000+ real MCP tool specs from GitHub, cluster by category, generate 20k+ synthetic tool variants, create thousands of synthetic agents (system prompts × tool subsets), simulate multi-turn usage with diverse synthetic users, judge all trajectories and keep only those passing quality rubrics. The key trajectory-level behaviors that single-turn QA cannot teach: ReAct-style reasoning (observe → decide → interpret), recovery from tool errors and partial results, appropriate termination (avoid infinite loops). The same checks that gate SFT data entry later serve as online RL reward.

**Long-context — faithfulness over comprehensiveness.** Without verification, models learn to "sound comprehensive" while making things up. Span-match verifier: cited sections must actually contain the claimed information. Distractor documents test whether the model anchors to relevant sources only. Contradiction check: answer must not contradict any cited source. LLM judges are only used *after* faithfulness checks pass — for coherence and quality, not correctness.

**Multilingual — the regression risk.** Tool use, coding, and reasoning SFT is overwhelmingly English. Without a stable multilingual slice in *every* training stage, SFT overwrites the multilingual knowledge the base model already has. Adding English tool-call SFT without a matching multilingual slice consistently regresses non-English performance. Verification: back-translation score for meaning preservation, cross-lingual consistency check (same answer across languages), script/encoding checks, native-speaker review for tone and domain terminology.

> **Interview question:** Your search-agent model achieves 80% grounding accuracy on single-hop questions but only 45% on 3-hop questions. What causes this and how do you fix the SFT data?
>
> *In multi-hop search, the model must compose results across sequential tool calls — the answer to hop 1 determines the query for hop 2, which gates hop 3. Single-hop SFT data does not teach this composition: the model learns to find one relevant source and cite it, but has no training signal for "use the result of the first search to reformulate the second query." Failure modes: (1) the model issues three parallel searches with independent queries rather than chaining; (2) it stops after one hop and fabricates the rest; (3) it correctly chains hops 1-2 but loses track of the original intent by hop 3. Fix: (1) Add multi-hop SFT examples where the assistant's tool calls are explicitly chained — the second search query is semantically derived from the first result, shown in context. (2) Add examples of explicit "what do I know so far?" reasoning between hops (plan-then-search). (3) Verify with `hops=3` grounding checks — every claim must trace to a specific hop's retrieved source, not a blend of all three. (4) Ensure the SFT distractor documents test cross-hop consistency, not just single-hop relevance.*

### Curation: Human + Synthetic
{: #curation-pipeline}

The full SFT curation pipeline:

<div class="post-flow" role="group" aria-label="SFT curation pipeline">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Spec — vertical + rubric definition (what does "high quality" mean for this slice?)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Author / Generate — human authoring, synthetic generation, or environment traces</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Filter & Verify — verifier cascade (syntax → static → execution → judge)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Normalise + Pack — canonicalise templates, compute token masks, schema validation</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Mix + Release — weighted corpus with immutable hashes, lineage metadata, audit logs</span></li>
  </ol>
</div>

**Human roles.** Generalist labelers (instruction following, style, safety), domain experts (medicine, law, math, security), "operator" annotators (agent traces in tool UIs), adversarial annotators (probe failure modes, write hard negatives), reviewers/adjudicators (resolve disagreements). Human time is scarce — humans focus on what machines cannot check: hard-to-verify domains, policy-sensitive behaviour, and *writing the rubrics* that synthetic pipelines then scale.

**Rubric design is the bottleneck.** A good rubric separates independent axes: correctness (factual/functional accuracy), helpfulness (addresses actual intent), format (schema validity, constraint satisfaction), safety (correct refusal behaviour), calibration (appropriate uncertainty). Vague rubrics produce style-optimised data: when the rubric is unclear, annotators fall back on their own preferences — usually verbosity and confidence. LLM judges trained on those labels amplify the same biases.

**Synthetic generation at scale.** Modern SFT datasets are predominantly synthetic: self-instruct prompt expansion (with strong filtering), teacher-student distillation (frontier model produces drafts), environment-generated traces (run agent in sandbox, keep verified outputs), programmatic templates (parameterised constraints, math problems, schemas). Pitfalls: distribution collapse (prompts all look the same), teacher artifacts (students inherit stylistic quirks), hidden leakage (synthetic data accidentally contains eval items).

**Rejection sampling — the workhorse.** Sample K candidate completions for the same prompt, run verifier V, keep only those passing:

```
D' = {(x, yₖ) : (x, {y₁,...,yK}) ~ D,  V(x, yₖ) = 1}
```

This is how SFT datasets are built at scale — generate many, keep the verified best. Rejection sampling turns a weak teacher into high-precision SFT data. DeepSeek-R1 applies this twice: once after RL stage 1, once after RL stage 2. Qwen 3 curates ~4k problems with verified answers, samples K candidates, keeps only those passing the verifier, then difficulty-filters to remove problems the model already solves at high pass rate.

**Verifier cascade — filter cheaply, verify expensively:**
1. *Syntax*: JSON parse, tool-call schema validation (milliseconds, catches ~50% of failures)
2. *Static analysis*: regex constraints, lint, type checks (fast, no execution needed)
3. *Execution*: run tests, sandbox, retrieval (expensive, only on candidates surviving stage 2)
4. *Judge*: only for remaining ambiguous cases after all deterministic checks pass

**LLM-as-judge limitations.** Effective for: style, clarity, tone, coherence scoring; pairwise comparisons on subjective rubric axes; triage (select for human review). Not a substitute for deterministic verification: judges exhibit same-family bias, reward persuasive phrasing over factual accuracy, and are sensitive to prompt wording. For objective correctness, always run deterministic checks first.

**Decontamination.** Contamination routes: direct inclusion of benchmark items, synthetic derivatives (prompt paraphrases of eval questions), web crawl contamination via public solutions. Defences: strict hashing / fuzzy n-gram matching against eval sets, isolate generators from eval corpora, keep quarantine and lineage metadata for every generated item so items can be retroactively removed.

**Diversity controls against synthetic collapse.** Teacher models produce high-probability outputs — without constraints, synthetic prompts drift into repetitive templates and safe, generic topics. Controls: prompt generators with latent variables (domain, difficulty, language, tool availability), n-gram / embedding dedupe within shards, coverage quotas per slice, adversarial prompts with rare constraints.

**Normalisation and versioning.** Every release must have: canonical role names and tool-call structure, one enforced chat template version, token-level masks computed correctly (train only on assistant tokens + selected reasoning tags), schema validation and render tests on serialised prompts, immutable dataset shards with content hashes, lineage graph (source → transformations → release), invalidation lists for problematic items discovered later. A template bug is a behavioural bug — it silently changes what the model learns and can break tool use, safety, or multilingual behaviour.

> **Interview question:** Your synthetic SFT pipeline produces 2M examples per week from a frontier teacher model. After 4 weeks you notice your model's output diversity is collapsing — it answers most questions the same way. What's happening and how do you fix it?
>
> *Teacher models prefer high-probability outputs. Over 4 weeks of sampling from the same teacher with the same prompt templates, the synthetic corpus drifts toward a narrow mode: the teacher's most confident, generic responses dominate. The model learns to imitate this mode rather than the full distribution of good responses. Fixes: (1) Latent variable diversification — add domain, difficulty, language, style, and persona as explicit conditioning variables in the prompt generator; force coverage quotas per combination. (2) Embedding-level dedup — run embedding similarity within each shard and drop examples with cosine similarity > 0.85 to any existing example. (3) Temperature sweep — generate at multiple temperatures and mix; high-temperature outputs are more diverse but require stricter filtering. (4) Adversarial prompts — explicitly generate "trap" cases and rare constraint combinations that force the teacher out of its comfort zone. (5) Rotate teacher models — using a single frontier model means its biases amplify; mix outputs from 2-3 teachers with different training histories. (6) Track n-gram entropy per vertical as a monitoring metric; alert when it drops below baseline.*

### Why SFT Is Not Enough
{: #limits-of-sft}

SFT has four fundamental limitations that RL environments are designed to address:

**Limitation 1: SFT cannot discover strategies absent from the data.** Behavioural cloning reproduces only demonstrated strategies. If no training example shows backtracking, query reformulation, or selective tool retry, the model has zero probability of producing those behaviours after SFT. Concrete cases: debugging (trying multiple edits, reading failure logs, revising from the error message), search (reformulating a query after first results are irrelevant), agent tasks (abandoning a failing plan rather than repeating it).

**Limitation 2: Covariate shift compounds over trajectories.** SFT trains on expert-written states. At inference the model visits its own (often incorrect) states. In multi-step tasks — agents, code repair, multi-hop reasoning — errors compound because the model has never seen or practised recovery from its own mistakes. RL rollouts train the model on states it *actually* produces, including error states, directly addressing the distribution mismatch.

**Limitation 3: SFT memorises, RL generalises.** Controlled experiments (Chu et al., 2025 — "SFT Memorizes, RL Generalizes") in two environments — a card-arithmetic game (GeneralPoints) and real-world navigation (V-IRL) — train on one rule/visual variant and test on held-out variants. SFT accuracy on held-out variants drops sharply as training compute increases — the model memorises the training distribution. RL with outcome-based reward: accuracy on unseen variants *increases* with compute, and visual recognition itself measurably improves. Additionally, adding sequential verification-revision steps during RL amplifies generalisation: more verification iterations ⇒ faster out-of-distribution performance growth. **SFT is still necessary as RL initialisation** — without prior SFT, RL produces unstable outputs and converges slowly. SFT fixes the output format and protocol; RL optimises task success within that format.

**Limitation 4: Conflicting annotator labels create incoherent defaults.** Different annotators resolve the same tradeoff differently: verbose vs concise, cautious vs confident, strict refusal vs helpful compliance. Averaging these contradictions via cross-entropy produces a model that hedges rather than committing to any coherent policy. Standard decomposition: SFT for unambiguous protocol defaults, preference learning (DPO/RLHF) for subjective tradeoffs where labels express relative quality, verifiers for objective correctness.

**What RL adds that SFT cannot provide:**
- *Multi-step credit assignment*: the model learns which steps in a trajectory actually mattered
- *On-policy exploration*: training data comes from what the model does, including states expert demonstrations never showed
- *Adaptive stopping*: the model learns when to keep going vs commit an answer, based on observed results
- *Reward variance as signal*: contrast between successes and failures on the same prompt drives learning; identical demonstrations give no contrast signal

> **Interview question:** You have 1M high-quality SFT examples for a code repair task. Your teammate argues that SFT is sufficient and RL adds unnecessary complexity. How do you respond?
>
> *SFT is sufficient if: (1) all strategies needed to solve hard code repair are present in your 1M examples, (2) the distribution of error states your model encounters at inference matches the distribution in your training data, and (3) you're willing to accept the model's performance ceiling at the performance ceiling of your demonstration authors. For simple, single-edit bugs, SFT may indeed be sufficient. For complex multi-step repair (search → read multiple files → generate patch → run tests → observe failure → revise patch → re-run), the "Chu et al. memorises vs generalises" result is directly applicable: as repair complexity increases, SFT performance degrades and RL performance improves. The key indicator is performance vs held-out task complexity: if performance degrades linearly as task difficulty (number of steps, number of files, depth of error) increases, SFT has hit its ceiling. RL won't help if your verifier (test suite) is weak — so the prerequisite conversation is verifier quality, not RL vs SFT.*

### RL Environments & Verifiers
{: #rl-environments}

**RL environment minimal definition for LLMs:**

```
(aₜ, sₜ) → (oₜ₊₁, rₜ₊₁, done)
```

The state `sₜ` is the token sequence so far: conversation history, tool outputs, retrieved documents, any scratchpad content. Actions can be token emissions (assistant text), structured tool calls (JSON arguments), or high-level actions in a wrapper (click, type, open file).

**Environment types in 2025-2026:**

| Environment type | Representative systems | Key property |
|---|---|---|
| Execution sandboxes | MiniMax Docker/GitHub PRs | Fail-to-pass + pass-to-pass test suites; deterministic reward |
| Tool simulators | Kimi K2 (20k+ tool variants) | Multi-turn agent trajectories; synthetic diversity |
| Web/browsing sandboxes | MiniMax WebExplorer (200k-token traces) | Multi-hop browsing; grounding verifier |
| Math/reasoning | DeepSeek-R1, Qwen 3 | Symbolic checkers + numeric equivalence |
| Safety | Red-team environments | Refusal correctness under optimisation pressure |

**Verifiers — precision over recall.** A false positive tells the model "good job" for a wrong answer — that directly teaches bad behaviour. A verifier that sometimes rejects correct outputs (lower recall) is much less harmful than one that lets wrong outputs through (lower precision). Desiderata: high precision (few incorrect outputs pass), stability (small prompt perturbations don't flip decisions), low cost (runs at scale — millions of rollouts per training run), decomposability (supports staged checks).

**The key insight: one verifier, two uses.** The same verifier controls both SFT data quality (offline rejection sampling) and RL reward quality (online per-rollout reward). Improving the verifier pays off twice — better SFT data and better RL rewards from the same investment. This is why verifier engineering is the highest-leverage activity in a post-training stack.

**RL training data: G rollouts per prompt.** In GRPO/DAPO, the model generates G independent rollouts per prompt. Each runs through the environment and receives a reward. The group-normalised advantage `Aᵢ = (rᵢ - mean(r₁:G)) / std(r₁:G)` computes the gradient from contrast between successes and failures on the *same prompt*. Groups where all G rollouts succeed or all fail produce zero gradient (no contrast) — DAPO's dynamic sampling discards these and resamples until the batch has reward variance.

**Multi-step rollout example (code sandbox, G=3):**
- Rollout 0: bash grep → file_edit (correct regex fix) → pytest → 1 passed. Reward: 1.0
- Rollout 1: bash cat (full file read) → file_edit (wrong line) → pytest → 1 FAILED. Reward: 0.0
- Rollout 2: bash grep → file_edit (partial fix) → pytest → 1 passed. Reward: 1.0
- Group mean reward: 0.67. Rollouts 0 and 2 get positive advantage; rollout 1 gets negative advantage.

**Curriculum design via verifier difficulty.** Start with easy tasks where the model succeeds often. Gradually introduce harder tasks and longer horizons as performance improves. Implementation: bucket tasks by current pass rate, up-weight the frontier bucket (tasks the model almost passes — that's where it learns the most), retain a stability suite (instruction following, multilingual, safety) in every training mix to prevent regressions.

**Reward hacking — the default failure mode.** Examples: schema verifier gaming (syntactically valid JSON with nonsense content), test-suite gaming (exploit incomplete tests), judge gaming (persuasive fluff that scores well but says nothing). Mitigations: more tests, fuzz testing, property-based checks, randomise prompts and test cases across rollouts, hold out private evals the policy never trains against, monitor reward vs held-out metrics (divergence signals hacking).

**Environment design best practices:**
- Include realistic failure modes: timeouts, invalid arguments, partial results, rate limits
- Enforce resource budgets: step limits, token limits, tool-call counts
- Security: sandbox execution (containers, seccomp), no external network unless task requires it, tool allowlists
- Observability: structured logs for every action and observation — enables replay, debugging, and distillation
- Version environment alongside model and tool catalog — API alignment between training environment and production is critical

> **Interview question:** You build a code repair RL environment using pytest as the verifier. After 2000 RL steps, test-suite pass rate is 75% but when you manually inspect the patches, 30% pass the tests without actually fixing the bug — they hardcode test fixture values or mock around the failing assertion. How do you fix this?
>
> *Classic reward hacking: the model discovered that pytest checks specific test assertions, and it can pass those assertions without fixing the underlying issue. Fixes — in order of effectiveness: (1) Property-based testing: instead of fixed test cases, generate random inputs and check that the fix is correct on all of them. A patch that hardcodes `return 42` fails when the input is `43`. (2) Mutation testing: automatically introduce small mutations to the codebase and verify that the fixed test still detects them. A patch that mocks away the assertion cannot survive mutations to the assertion's targets. (3) Randomise test fixtures across rollouts: the same bug should have different test fixtures in different rollouts, making fixture-hardcoding expensive (it only passes one rollout). (4) Cross-validation check: run the patch on a held-out test file for the same function. A genuine fix should work; a fixture-hardcoded patch won't. (5) Code review verifier: a secondary model or static analysis step that flags patches containing hardcoded literals, mock overrides, or test-specific conditionals as suspicious. Start with property-based tests and mutation testing — they have the best precision/cost ratio.*

### Reward Shaping & RL Algorithms
{: #reward-shaping}

**Why sparse binary rewards are not enough.** Sparse rewards (correct/incorrect) produce zero-gradient batches on hard problems where the model always fails. Every recent system adds reward shaping — but each technique introduces its own failure modes.

**DAPO (ByteDance/SIA, 2025) — four mechanisms on top of GRPO:**
1. *Dynamic sampling*: discard prompt groups where all G samples are correct or all incorrect (zero advantage ⇒ zero gradient); resample until the batch has reward variance
2. *Overlong reward shaping*: truncated sequences get a soft length penalty instead of a hard −1 (prevents the model from learning "truncate early to avoid negative reward")
3. *Token-level loss*: normalise per token, not per sequence, so long chains of thought are weighted fairly against short ones
4. *Clip-Higher*: asymmetric clipping (ε_low < ε_high) allows increasing good-action probability more freely, preventing entropy collapse while still bounding harmful updates

**GSPO (Qwen 3 Instruct/Coder/Thinking) — sequence-level ratios for MoE.** Standard GRPO uses per-token importance ratios `πθ(aₜ|sₜ)/πθ_old(aₜ|sₜ)`, which accumulate variance over long sequences and misalign with sequence-level rewards. GSPO uses sequence-level ratios — the geometric mean over token positions:

```
sᵢ(θ) = [πθ(yᵢ|x) / πθ_old(yᵢ|x)]^{1/|yᵢ|}
```

Benefits for MoE: per-token router shifts cancel in the geometric mean; more tolerant of precision mismatches across devices; eliminates train-inference mismatch where different parallelism strategies produce different expert routing.

**CISPO (MiniMax M2.5) — per-step process rewards for long agentic trajectories.** When trajectories reach 200k tokens with 30+ tool calls, outcome-only reward provides no credit assignment signal for early steps. CISPO assigns each intermediate step its own advantage based on per-step rewards, enabling learning from step-level successes and failures within a trajectory.

**Router replay (R3) — fixing MoE training-inference mismatch.** In MoE RL, rollouts run on an inference engine and gradient updates on a separate training engine. Even for the same input, different parallelism configurations can route tokens to different experts — inflating importance ratios and causing training collapse. R3: record the routing mask (which expert handles each token) during inference rollouts, then replay those same masks during the training forward pass. Reduces KL divergence between training and inference phases by ~10×. Trade-off vs GSPO: R3 fixes the mismatch directly but adds engineering complexity (routing mask storage, engine coupling); GSPO avoids the problem by operating at sequence level where per-token routing noise washes out.

**Reward shaping risks.** Unless the shaping satisfies the potential-based condition, it changes the optimal policy. The model may: emit redundant tool calls for step bonuses, pad outputs to hit length targets, exploit format loopholes. Mitigations: potential-based shaping (provably preserves optimal policy), hard constraints instead of positive intermediate rewards, entropy bonuses or Clip-Higher to prevent mode collapse, audit intermediate behaviours per-slice before scaling.

**Algorithm landscape summary:**

| Algorithm | Used by | Key property |
|---|---|---|
| GRPO | DeepSeek-R1, Qwen 3 (early) | Group mean baseline, no value network; ~10× faster than PPO |
| DAPO | ByteDance/SIA | Dynamic sampling + overlong shaping + Clip-Higher; beats R1-Zero on AIME in 50% fewer steps |
| GSPO | Qwen 3 Instruct/Coder/Thinking | Sequence-level ratios; stable MoE training without router replay |
| CISPO | MiniMax M2.5 | Per-step process rewards; handles 200k-token agentic trajectories |
| PPO + KL | OpenRLHF, earlier systems | Clipped surrogate + learned value network; the textbook baseline |

> **Interview question:** You're training a reasoning model with GRPO. After 1000 steps you notice the model is getting shorter and shorter — average rollout length dropped from 800 tokens to 200 tokens. What's happening?
>
> *Entropy collapse. The model has found a shortcut: short confident answers pass the verifier at some rate, but the GRPO advantage is computed relative to the group mean. If the group mean reward is 0.5 (mixed results), a short answer that gets reward 1.0 gets strong positive advantage regardless of how the answer was reached. The model is learning to emit short confident guesses because they occasionally pass the binary verifier while longer chains of thought sometimes lead to wrong answers (getting negative advantage). This is a failure mode of binary rewards + sequence-level normalisation. Fixes: (1) DAPO's Clip-Higher: asymmetric clipping allows increasing good-action probabilities more freely, counteracting entropy collapse by keeping the policy from collapsing to short outputs. (2) Length bonus: add a small positive reward component for reasoning traces that reach a minimum length. (3) Format reward: require `<think>...</think>` blocks with a minimum token count before the answer, with a separate format verifier. (4) Token-level loss (DAPO): normalise loss per token rather than per sequence — short sequences no longer have artificially high per-token weight, reducing the incentive to be brief.*

### RL Infrastructure & Distillation
{: #rl-infrastructure}

**Why RL is I/O-bound, not compute-bound.** Each gradient step needs many rollouts, and each rollout may call tools, run code, or hit external services. The training loop is heterogeneous: model inference is GPU-bound, environment execution is CPU/network-bound, and verifiers are often I/O-bound. Naïve sequential execution leaves the GPU idle during environment waits.

**System diagram: online RL loop for an LLM agent:**

<div class="post-flow" role="group" aria-label="RL infrastructure loop">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Policy (LLM) generates G rollouts per prompt — distributed inference (vLLM)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Environment servers execute tool calls — sandboxed, cached, parallelised</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Verifier service scores each rollout — microservice, deterministic, high precision</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Trainer (PPO/GRPO) computes advantages and updates policy — KL-controlled against reference</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Successful traces logged to replay buffer → distillation into new SFT data</span></li>
  </ol>
</div>

Most engineering complexity lives in the communication: serialisation, caching, batching, timeouts, and safety checks.

**Rollout optimisation.** Micro-batching across tasks and steps (vLLM-style KV cache reuse), async execution (overlap environment waits with token generation for the next rollout), speculative decoding for cheap rollout drafts. MiniMax Forge processes millions of samples/day at 200k-token contexts — the throughput-stability-flexibility triangle is the core infrastructure challenge.

**Safety in RL loops.** The optimiser searches for anything that increases reward — without guardrails it may find unsafe actions, exploit verifier blind spots, or work around tool restrictions. Guardrails at three levels: (1) environment-level (no external network access, tool allowlists, resource caps, proper virtualisation so the model cannot kill host processes); (2) reward-level (hard penalties for policy violations — unsafe compliance, refusal failures); (3) probing (red-team environments that test jailbreak resistance *under optimisation pressure*, not just on static prompts).

**Distillation: three mechanisms.**

*Rejection-sampling distillation (offline, off-policy):* run the RL-trained model on prompts, keep only outputs passing the verifier, add them to the SFT set. The student never sees teacher logits — only filtered text. DeepSeek-R1 applies this twice: after RL stage 1 and after RL stage 2.

*On-policy distillation (online, logit-level):* the student generates a completion; the loss minimises `KL(π_student(·|x) ∥ π_teacher(·|x))` token by token. Because the student samples on its own distribution, it learns to match the teacher in states it *actually visits* — not just pre-recorded examples. Qwen 3 uses this for reasoning: student and teacher run side by side, gradients come from logit divergence on the student's own tokens.

*Off-policy distillation (offline, imitation):* student trains on teacher's pre-generated outputs via standard SFT loss. Cheaper (no live teacher inference), but subject to the same covariate-shift limitation as SFT. Qwen 3 uses this for thinking/non-thinking mode fusion — reportedly ~10× less GPU than RL with comparable math/code accuracy.

**The flywheel — how strong models are actually built:**

<div class="post-flow" role="group" aria-label="Post-training flywheel">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Train / RL in environments — generate rollouts, update policy</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Evaluate by slices — identify regressions per capability vertical</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Mine failures + cluster — find systematic failure modes from rollout logs</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Curate new SFT + improve verifiers — then repeat</span></li>
  </ol>
</div>

**Four recurring patterns that distinguish strong post-training stacks:**
1. *Verifier-centric scaling*: more samples ⇒ more signal; building a better verifier usually helps more than switching RL algorithms
2. *Trajectory data for agentic behaviour*: single-turn SFT cannot teach state tracking, error recovery, or stopping decisions; agentic capabilities require multi-step trajectory data + environment RL + distillation of successful traces
3. *Multi-slice stability suites*: optimising one vertical can harm others; a fixed stability suite (instruction following, safety, multilingual, long context) must stay in every training mix and evaluation gate
4. *Closed loop*: evaluate → identify failures → build or improve verifiers → curate SFT data → run RL → distill → repeat

> **Interview question:** Your team has a 7B model and a 70B "teacher" model. You want to transfer the teacher's reasoning ability to the 7B student as cheaply as possible. Walk through your distillation strategy.
>
> *Three-stage approach: (1) Off-policy distillation first (cheapest): run the 70B teacher on a large prompt set (math, code, reasoning), keep all outputs — don't filter by verifier. Train the 7B student on these via standard SFT loss. This is cheap (no live teacher inference during student training) and captures the teacher's format and style. Limitation: covariate shift — the 7B student encounters its own (often wrong) states at inference that the teacher never produced. (2) On-policy distillation for reasoning (medium cost): the 7B student generates completions; loss = KL(student ∥ teacher) on the student's own tokens. The teacher must run in parallel — expensive but necessary to close the covariate-shift gap. Use this specifically for reasoning chains where the student's intermediate mistakes matter. (3) RLVR with verifiers (highest cost, highest ceiling): equip the 7B with the same verifiers (math checker, test suites) and run GRPO on its own rollouts. This is the only mechanism that can produce strategies absent from the teacher's demonstrations — including strategies the 7B discovers that the 70B wouldn't use. Budget allocation: if you have $X of compute, spend 50% on off-policy distillation (high data efficiency), 30% on on-policy distillation (closes the key gap), 20% on RLVR (raises the ceiling). Monitor all three capability verticals throughout; off-policy distillation alone often regresses safety and multilingual.*

---

## RL Algorithms for Post-Training
{: #rl-algorithms-deep}

Before 2025, RL was alignment polish — DPO, PPO on reward models, minor behaviour steering. From 2025 onward, RL became the **capability engine**: it discovers reasoning strategies, agent behaviours, and error-recovery patterns that no static labeller wrote down. The canonical pattern:

1. Cold-start SFT stabilises format and protocol
2. RL discovers stronger reasoning and agent strategies (GRPO/DAPO/GSPO/CISPO)
3. Distillation/fusion recovers latency and controllability for deployment

DeepSeek-R1, Kimi k1.5/K2, Qwen3, and MiniMax-M1/M2.5 all treat RL as the primary source of capability gain in post-training. Three forces made this viable simultaneously: **verifiable domains** (math exact-answer checks, code unit tests, agent environment scores provide high-precision reward), **long chain-of-thought as search** (extended CoT lets the model execute plan → check → revise → commit), and **systems maturity** (rollout fleets, async schedulers, disaggregated training-inference, replay buffers — >10× throughput gains reported by MiniMax 2025–2026).

### Policy Gradient Foundations
{: #pg-foundations}

**The RL loop in LLM post-training:**

```
State sₜ = (x, y<t)         — prompt + generated prefix + tool outputs
Action aₜ = yₜ              — next token (or structured tool call)
Policy πθ(aₜ|sₜ)            — model distribution over next action
Trajectory τ                — one completion or multi-turn rollout
Reward R(τ)                  — verifier score, test pass rate, environment outcome
```

The training data distribution is **not fixed** — it co-evolves with the current policy, prompt sampler, and reward computation.

**REINFORCE — the baseline policy gradient:**

```
∇θ J(θ) = Eτ~πθ [R(τ) · Σₜ ∇θ log πθ(aₜ|sₜ)]
```

If a trajectory gets high reward, increase probability of all its actions. If low, decrease. The problem: **catastrophic variance** for LLMs. With binary rewards and T-token trajectories, variance scales as O(T):

- Pass (R=1): pushes all T tokens up equally — filler, wrong intermediate steps, and correct steps get identical signal
- Fail (R=0): gradient is exactly zero — that trajectory teaches nothing
- A 4,000-token math response has 4,000 random terms in the gradient sum

**Baselines and advantage — the key variance reduction.** For any b(sₜ) independent of aₜ: `E[b(sₜ)∇θ log πθ(aₜ|sₜ)] = 0`. So replacing R(τ) with R(τ) − b(sₜ) preserves the expected gradient while reducing variance. This gives us:

```
V^π(s) = E_π[Gₜ | sₜ=s]           — value: expected return from state s
Q^π(s,a) = E_π[Gₜ | sₜ=s, aₜ=a]  — Q-value: expected return taking action a
A^π(s,a) = Q^π(s,a) − V^π(s)      — advantage: better or worse than average?
```

A > 0: this action was better than average from this state. A < 0: worse. Every algorithm computes advantage differently: PPO uses a learned critic + GAE, GRPO uses group-relative reward normalisation, REINFORCE++ uses batch-global normalisation.

**GAE (Generalised Advantage Estimation) — bias-variance knob:**

```
δₜ = rₜ + γVφ(sₜ₊₁) − Vφ(sₜ)          — TD residual
Âₜ^GAE(γ,λ) = Σₗ (γλ)ˡ δₜ₊ₗ
```

- λ ≈ 0: trust critic heavily → low variance, more bias
- λ ≈ 1: near Monte Carlo → low bias, higher variance

For reasoning RL with sparse terminal rewards (rT = R(τ), all other rₜ = 0) and γ=1, λ=1: `Âₜ = R(τ) − Vφ(sₜ)` — every token's advantage is "outcome minus how promising this prefix looked."

**Importance sampling — reusing stale rollouts.** After generating a batch with πθ_old, each gradient step changes the policy. Rolling out a fresh batch after each step is prohibitively expensive for long-CoT (minutes to hours per batch). Importance sampling re-weights old samples to approximate expectations under the current policy:

```
E_{a~πθ}[f(a)] = E_{a~μ}[(πθ(a)/μ(a)) · f(a)]
```

Full trajectory IS weights are products over thousands of tokens — they can explode or vanish. Trust regions contain this: KL penalty (`max E[R] − β·E[KL(πθ ∥ πref)]`) or ratio clipping (truncate `rₜ(θ) = πθ(aₜ|sₜ)/μ(aₜ|sₜ)` to `[1−ε, 1+ε]`).

Three distinct policies to keep straight: **current πθ** (being optimised, numerator of ratios), **behaviour/old μ** (generated the batch, denominator), **reference πref** (KL anchor, often stable SFT checkpoint).

> **Interview question:** Binary rewards mean half your batches have zero gradient (all-fail groups). How do you ensure the model still learns on hard problems?
>
> *Zero gradient on an all-fail group is only a problem if it persists — if the model never gets any reward signal on hard problems, it cannot improve on them. Three complementary approaches: (1) Curriculum sampling: bucket problems by current pass rate, up-weight the frontier bucket — problems the model almost passes (pass rate 5–40%) rather than problems it never passes (0%) or always passes (100%). Maximum gradient signal comes from tasks at the competence frontier. (2) DAPO dynamic sampling: explicitly discard groups where all G rollouts are correct or all wrong, and resample until the batch has reward variance. This ensures every training step has informative contrast. (3) Increase group size G for hard problems: with G=8 instead of G=4, the probability of at least one correct rollout is much higher for problems the model has a 10–20% individual pass rate on. A binary reward on a hard problem with G=8 and pass rate 15% gives a zero-gradient group only 27% of the time (0.85^8), vs 52% with G=4 (0.85^4). Even one correct rollout in a group is enough to produce a positive-advantage signal.*

### PPO: Trust-Region Recipe
{: #ppo-deep}

**PPO objective:**

```
L^PPO(θ) = E[Σₜ min(rₜ(θ)·Âₜ, clip(rₜ(θ), 1−ε, 1+ε)·Âₜ)]

rₜ(θ) = πθ(aₜ|sₜ) / πθ_old(aₜ|sₜ)
```

**Three ingredients:** (1) advantage via critic + GAE — a value network Vφ estimates expected future reward from each prefix; (2) policy ratio — compares new policy to behaviour policy that generated the batch; (3) clipping — truncates ratio to [1−ε, 1+ε], preventing overly large updates on stale data.

**How clipping works case-by-case (ε=0.2):**

| Scenario | rₜ | Behaviour |
|---|---|---|
| Âₜ > 0, rₜ = 1.15 | within bounds | normal gradient, probability rises |
| Âₜ > 0, rₜ = 1.25 | clipped to 1.20 | gradient stops — cannot push above ceiling |
| Âₜ < 0, rₜ = 0.75 | clipped to 0.80 | gradient stops — cannot push below floor |

The clipped surrogate creates a flat region beyond [1−ε, 1+ε] — zero gradient, so the optimiser never strays too far from the behaviour policy in a single update. This is a cheap surrogate for a hard trust-region constraint.

**PPO strengths and failure modes for long-CoT:**

| Strengths | Failure modes |
|---|---|
| Low-variance advantage from learned critic | Critic nearly doubles memory + compute |
| Mature ecosystem (OpenRLHF, verl, TRL) | Value prediction noisy on long revisionary chains |
| Works well with dense / intermediate rewards | Binary outcome rewards make critic targets unstable |
| Cleanest bridge to standard RL theory | Stale ratios accumulate across thousands of tokens |

**Is PPO obsolete?** No. Open-Reasoner-Zero (Mar 2025) showed vanilla PPO with γ=1, λ=1, rule-based rewards, and no KL penalty reproduces R1-Zero-like scaling. PPO remains strong when you can afford the critic and have informative intermediate signals. Critic-free methods gained traction for sparse-reward long-trajectory reasoning — not because PPO is wrong, but because the critic-learning problem is hard and expensive in that regime.

> **Interview question:** You're training a reasoning model with PPO. After 500 steps the value loss is still high and your critic predictions for any prefix longer than 1,000 tokens are essentially random. What's happening and what do you do?
>
> *The critic is being asked to predict the expected return from a prefix in the middle of a long chain-of-thought. This is fundamentally hard: whether a 3,000-token partial derivation will succeed depends on whether the next 1,000 tokens contain the key insight — information the critic cannot access from the prefix alone. The value function is trying to compress future trajectory randomness into a scalar, but for long exploratory reasoning chains this randomness is enormous and not predictable from the prefix. Options: (1) Switch to GRPO-style group-relative baseline — replace the learned critic with the mean reward of G sibling rollouts for the same prompt. No value network needed; the baseline is the empirical group mean which is a valid unbiased estimator of V^π for that prompt. (2) If staying with PPO: restrict critic inputs to short prefixes (< 500 tokens), use a shallower critic architecture, and apply strong regularisation (λ closer to 1 in GAE to rely less on the critic and more on Monte Carlo return). (3) Open-Reasoner-Zero finding: set γ=1, λ=1, no KL penalty — this essentially makes the critic only responsible for the final outcome baseline, which it can learn, rather than per-step advantage.*

### GRPO Family: Critic-Free Reasoning RL
{: #grpo-family}

**GRPO (Group Relative Policy Optimisation)** replaces the learned critic with within-group reward statistics:

For prompt q, sample G completions from πθ_old. Score each with a verifier. Estimate advantage:

```
Âᵢ = (rᵢ − mean(r₁,...,rG)) / std(r₁,...,rG)
```

The group mean approximates V^π(q) with zero critic cost:

```
r̄ ≈ E_{y~πθ_old(·|q)}[R(q,y)] = V^πθ_old(q)
```

Variance is bounded: `Var[Âᵢ] ≈ (G+1)/G ≈ 1` regardless of reward scale. But when all rᵢ are equal, σ→0 and Â is undefined — the **zero-signal group problem**.

**Leave-one-out (RLOO)** — unbiased even for small G:

```
b₋ᵢ = (1/(G−1)) Σⱼ≠ᵢ rⱼ
Âᵢ^RLOO = rᵢ − b₋ᵢ
```

Including sample i in its own baseline shrinks its own advantage; RLOO excludes it for an unbiased estimate.

**GRPO failure modes and the 2025 fix-up literature:**

| Failure | Mechanism | Fix |
|---|---|---|
| Zero-signal groups | All-correct or all-wrong → Â≈0 → wasted compute | DAPO dynamic sampling |
| Length bias | 1/\|oᵢ\| norm makes wrong long answers under-penalised | Dr.GRPO removes length normalisation |
| Entropy collapse | Symmetric clipping over-suppresses exploratory tokens | DAPO Clip-Higher (asymmetric) |
| Token/sequence mismatch | Reward is per-sequence, optimisation per-token | GSPO sequence-level ratios |

**Dr.GRPO — removing the hidden length bias.** Standard GRPO divides by response length `1/|oᵢ|`. Two completions: o₁ correct at 200 tokens, o₂ wrong at 2000 tokens. The gradient penalty on o₂ is 10× smaller per-token than o₁'s gradient reward. Over many updates: the policy learns *longer is safer*. Dr.GRPO removes the `1/|oᵢ|` normalisation — every token in a wrong response contributes the full penalty regardless of length.

**DAPO — four targeted repairs for long-CoT:**

1. *Clip-Higher (asymmetric clipping)*: `εₗₒw < εₕᵢgₕ` — policy can more freely increase probability on advantageous tokens. Preserves exploration diversity. Mathematically: with ε_low=0.2, ε_high=0.28, a good action's ratio can rise to 1.28 before clipping (vs 1.20 symmetric), but bad actions are still clipped at 0.80.
2. *Dynamic sampling*: discard groups with zero reward variance, resample until batch has informative signal. Focuses compute on the current competence frontier.
3. *Token-level loss*: aggregate gradient by token count, not by sample. Long chains of thought not automatically diluted vs short ones.
4. *Overlong reward shaping*: replace hard truncation penalty with soft length-aware penalty. Distinguishes productive long thinking from pathological overthinking.

**GSPO — sequence-level ratios for sequence-level rewards.** Per-token ratios over 4,000 tokens: even with each `rₜ ∈ [0.98, 1.02]`, the joint ratio `Π rₜ` ranges from `0.98^4000 ≈ 10⁻³⁵` to `1.02^4000 ≈ 10³⁴`. Token-level clipping cannot control this. GSPO uses geometric-mean ratios:

```
sᵢ(θ) = [Π rₜ(θ)]^{1/|yᵢ|} = exp((1/|yᵢ|) Σₜ log rₜ(θ))
```

Averages log-ratios so outlier tokens cannot dominate. Clipping applied to `sᵢ`, matching the granularity of the sequence-level reward. Stabilises MoE RL because per-token router variance washes out in the log-space average — key ingredient in Qwen3's large-MoE reasoning RL.

**REINFORCE++ — batch-global baseline instead of per-prompt grouping:**

```
Âᵢ^R++ = (Rᵢ − R̄_batch) / σ_batch
```

No per-prompt grouping needed; KL folded into the reward: `Rᵢ' = Rᵢ − β Σₜ KL(t)`. Avoids per-group degeneracies (σq→0 when all group rewards equal). Trade-off: loses prompt-local difficulty normalisation — harder prompts get the same baseline as easy ones.

**Comparison table:**

| Method | Baseline | Clip granularity | Main benefit | Watch out for |
|---|---|---|---|---|
| PPO | Learned critic + GAE | Token ratio | Classical low-variance updates | Critic cost + value error on long-CoT |
| GRPO | Prompt-group mean/std | Token ratio (impl.) | No critic; simple reasoning RL | Zero-signal groups; length bias |
| DAPO | GRPO + 4 fixes | Asymmetric token | Stabilised long-CoT RL | More heuristics; still GRPO family |
| Dr.GRPO | Group mean, no biasing norms | Token ratio | Fixes length + efficiency bias | Less battle-tested |
| REINFORCE++ | Batch-global mean/std | Token ratio | No per-prompt grouping | Loses prompt-local normalisation |
| GSPO | Group-relative seq. reward | Sequence ratio | Stable long-response MoE RL | Newer; less widely reproduced |

**Decision rule:** Critic + dense rewards → PPO. Verifier, multi-sample → GRPO/DAPO. Long MoE responses → GSPO. Off-policy reuse → CISPO.

> **Interview question:** Your GRPO training on a math dataset is working well at G=8, but you scale to a 32B MoE model and training becomes unstable — gradient norms spike and loss diverges within 200 steps. What's the likely cause and how do you fix it?
>
> *The most likely cause for MoE + GRPO instability is token-level ratio accumulation. In a 32B MoE, token-level logprobs have additional variance from expert routing — even small differences in routing between the rollout engine and training engine cause different logprobs for the same token. Over a long sequence (say 2,000 tokens), these small per-token ratio errors multiply, causing the joint importance ratio to drift far from 1. This inflates the effective gradient magnitude on some tokens, causing the observed norm spikes. Fix: switch to GSPO, which uses geometric-mean (log-space average) of per-token ratios as the sequence-level importance weight — per-token routing variance washes out in the average. If you want to stay with GRPO: (1) enforce FP32 precision for the LM head logprob computation in both rollout and training engines — logprob disagreement due to precision mismatch is a known major instability (MiniMax documented this); (2) use router replay (R3): record which expert handles each token during rollout, replay the same routing masks during the training forward pass; (3) tighten the clip range ε and add a KL penalty term to slow down divergence while you diagnose.*

### Off-Policy & Long-Context RL
{: #off-policy-rl}

**The on-policy vs off-policy trade-off:**

| | On-policy | Off-policy / reused trajectories |
|---|---|---|
| Data source | Current policy | Older checkpoint |
| Bias | Low | Higher (grows with staleness) |
| Rollout cost | High (fresh every update) | Low (reuse across updates) |
| Correction | None needed | Importance sampling required |
| When to use | Short-horizon reasoning | Long-CoT, expensive environments |

Long-context reasoning and agent RL force training onto the off-policy frontier. Three linked quantities govern feasibility: **reuse factor** (updates per generated batch), **policy staleness** (drift from behaviour policy), **correction quality** (accuracy of importance ratios).

**Kimi k1.5 — context length as an RL scaling axis.** Kimi k1.5 argues that scaling the rollout context window during RL up to 128k tokens is itself a major driver of reasoning gains. Longer rollout windows let the policy hold more partial derivations, revisit earlier branches, and behave more like iterative search. Techniques: *partial rollouts* (reuse large chunks of previously generated trajectories, continue from intermediate prefixes — avoids regenerating entire long trajectories after every update); *long2short distillation* (use long-CoT reasoning to generate strong supervisory traces, then distill into shorter-CoT models for deployment). Gains attributed to the combination of long context + online mirror descent + smart prompt selection + explicit length control.

**CISPO — clipped importance sampling for off-policy reuse.** PPO/GRPO-style clipping permanently zeros gradients for tokens whose ratio exits the clip band — including discourse tokens ("wait", "let me reconsider") important for exploration. CISPO clips the *weight*, not the *update*:

```
∇J(θ) ≈ E[w̄ · Â · ∇ log πθ(τ)]
```

where `w̄ = clip(wₜ, w_min, w_max)` is a clipped importance weight — **bounded but never zeroed**. Off-policy tokens contribute a down-weighted signal instead of no signal.

Comparing PPO vs CISPO gradients per token:

```
PPO:   ∇_t = rₜ(θ)·Âₜ·∇log πθ  if rₜ ∈ [1−ε, 1+ε], else 0
CISPO: ∇_t = clip(wₜ, wmin, wmax)·Âₜ·∇log πθ   (always nonzero)
```

At reuse factor k=4 (4 updates per batch), PPO clips out ~30–50% of tokens by later epochs. CISPO keeps all tokens trainable with bounded variance. MiniMax-M1 (hybrid-attention MoE + CISPO + diverse RL environments including sandboxed SWE) outperformed GRPO/DAPO at high reuse ratios.

> **Interview question:** You want to train an agent on trajectories that can be 50,000 tokens long. Fresh rollouts take 20 minutes each. How do you design the training loop to make this practical?
>
> *The core problem is that fresh on-policy rollouts at 50k tokens are prohibitively slow if you regenerate them after every gradient step. Design: (1) Set a reuse ratio of 4–8 updates per batch. Generate one batch of rollouts, run 4–8 gradient updates against it with CISPO (not PPO/GRPO-style clipping, which would zero out 30–50% of tokens by update 4). Monitor the KL divergence between current πθ and the behaviour policy that generated the batch — discard and regenerate the batch when KL exceeds a threshold (e.g. 0.1). (2) Partial rollouts: rather than regenerating complete 50k trajectories, cache intermediate prefixes. After a policy update, continue trajectories from cached intermediate states instead of restarting from the prompt. This amortises the generation cost. (3) Async scheduling: run rollout workers and training workers concurrently on different hardware. While the trainer processes batch N, rollout workers are generating batch N+1. The rollout queue depth determines the staleness budget — keep queue depth ≤ 2 batches to limit off-policy error. (4) Prefix merging: if multiple rollouts share long common prefixes (same prompt, same early steps), batch-process the shared prefix once and fan out from the branch point — MiniMax Forge reports ~40× speedup via tree-structured prefix merging for agentic tasks.*

### Agent RL: Multi-Turn Tool-Using Trajectories
{: #agent-rl}

**From completions to agent trajectories.** Reasoning RL: one prompt → one completion → one reward. Agent RL: actions and observations interleave:

```
τ = (s₀, a₁, o₁, s₁, a₂, o₂, ..., sK)

Actions: text, tool calls, code patches, browser operations, memory edits
States: conversation + retrieved docs + workspace + tool outputs + compressed context
```

Why this changes the optimisation problem:
- *Extreme latency variance*: one API-heavy trajectory may take 100× longer than a simple one
- *Harder credit assignment*: an early bad retrieval poisons 50 downstream steps
- *Composite reward*: task success + tool correctness + latency + safety must all be balanced
- *Non-stationary environment*: external APIs and tools change faster than the policy

**Composite reward for agent RL (MiniMax M2.5):**

```
R = α·R_task + β·R_process − γ·C_latency − δ·C_tool − η·C_unsafe
```

Process reward and latency-aware reward are first-class components — denser process signals reduce credit-assignment difficulty in long multi-turn trajectories.

**Agent RL failure modes and mitigations:**

| Failure | Mechanism | Mitigation |
|---|---|---|
| Tool farming | Policy games reward by calling tools in exploit patterns | Adversarial reward audit; tool-count penalties |
| Context bloat | Model keeps irrelevant context because reward ignores cost | Explicit latency/token-budget reward terms |
| Environment overfit | Great on training scaffold, poor transfer to new tools | Diverse environment portfolio; held-out envs |
| Credit leakage | Process reward accidentally rewards busy activity, not progress | Outcome-gated process rewards |
| Non-stationary tools | External APIs change faster than the policy | Environment versioning; robustness testing |

**The agent scaffold must be versioned and evaluated like training data.** Frontier post-training is moving toward environment *portfolios*, not single reward sources.

> **Interview question:** Your agent RL model achieves 70% task success on training environments but only 35% on production tool calls with slightly different argument schemas. What's the failure mode and how do you fix it?
>
> *Environment overfit: the model learned to call tools with the exact argument patterns present in training, but the production schemas differ — field names, nesting depth, or required vs optional fields changed. At inference, constrained decoding (if used) forces the model down low-probability token paths for the production schema, and without constraints, it generates the training schema format which the production API rejects. Root cause: the training environment is not aligned with the production environment. Fixes: (1) Schema diversification during training — synthesise 10–20 variants of each tool schema (camelCase vs snake_case, flat vs nested, different required fields) and rotate them across rollouts. The policy learns schema-invariant tool-calling rather than schema-specific patterns. (2) Version-pinned environments — any schema change in production must trigger a retraining or at minimum a fine-tuning run with the new schema. (3) Schema-aware constrained decoding at inference — ensures output is valid under the new schema syntactically, but train the model on the new schema to ensure it is also semantically correct. (4) Robustness testing: before promoting any model checkpoint, run it against 3–5 schema variants it has never seen. Track pass rate degradation as a deployment gate — more than 20pp drop signals environment overfit that will hurt production.*

### Reward Engineering & Verifier Design
{: #reward-engineering}

**Reward taxonomy:**

| Type | Examples | Strength | Failure mode |
|---|---|---|---|
| Outcome | Exact answer, pass/fail tests, task completion | Very precise | Sparse; delayed credit |
| Process | Step validity, partial progress, checkpointed tests | Denser signal | Easy to game |
| Format | JSON validity, schema adherence, language consistency | Stabilises outputs | Can dominate substance |
| Judge | LLM grader, reward model, rubric score | Broader coverage | Bias, drift, hacking |
| Cost | Latency, token budget, tool count | Product realism | Can suppress exploration |
| Safety | Refusal correctness, policy compliance | Risk control | Can over-penalise |

**Design preference:** use verifiers where you can, judges where you must. Combine as `R = α·R_verifier + β·R_judge`, with verifier dominating.

**Programmable graders — the right abstraction:**

```python
def grade(sample, item):
    score = 0.0
    if exact_answer(sample, item):   score += 0.7
    if valid_json(sample):            score += 0.1
    if cites_required_fields(sample): score += 0.1
    if under_token_budget(sample):    score += 0.1
    return min(score, 1.0)
```

Explicit, auditable, composable. The same pattern across frontier labs and commercial APIs (OpenAI RFT). The fastest path to useful RL is often *better graders*, not new algorithms.

**Reward hacking — the default failure mode.** Format gaming (syntactically valid JSON with nonsense content), judge flattering (persuasive fluff that scores well but says nothing), tool spam (redundant calls for step bonuses), length farming (padding to hit length targets), dataset leakage. Always run side evaluations independent of the training reward — DeepSeek-R1 documented reward increasing while CodeForces performance *decreased*: the central warning of all judge-shaped RL.

**Curriculum and prompt sampling are part of the algorithm.** Maximum gradient signal comes from prompts at the current competence frontier:
- Too easy: all-correct groups → advantages ≈ 0 → no learning
- Too hard: all-wrong groups → advantages ≈ 0 → no learning
- Frontier (5–40% pass rate): maximum reward variance → maximum advantage signal

2025 strategies: dynamic filtering of zero-signal groups (DAPO), difficulty-balanced prompt sets across verticals, explicit hard-example mining and synthesis, curriculum schedules that expand difficulty as the policy improves.

> **Interview question:** You run RLVR on a coding model for 1,000 steps and test-pass rate improves from 30% to 65%. But on a held-out eval set, performance only improves from 30% to 38%. What's happening?
>
> *Classic reward hacking / verifier overfit. The model improved dramatically on the training verifier (test suite) but barely generalised. The gap (65% vs 38%) is too large to be explained by normal distribution shift — it indicates the model found systematic exploits of the training test suite that don't transfer. Likely mechanisms: (1) Insufficient test coverage — the training tests have coverage gaps; the model learned to pass the specific assertions without actually implementing the correct logic. A function that hardcodes the expected output for the 3 training test cases will pass at 100% while failing all held-out tests. (2) Memorisation of test patterns — if the same problem types repeat across training prompts, the model learned the pattern of the test rather than the general skill. (3) Train-time reward hacking — the model may have learned to exploit the test framework itself (e.g. mock the assert function, manipulate environment variables). Diagnosis: manually inspect 20 high-training-reward / low-held-out-reward samples. If you see hardcoded returns, assert manipulation, or trivially wrong implementations that happen to pass the specific training tests, it's reward hacking. Fix: (1) Property-based tests — generate random inputs and check correctness on all; impossible to hardcode. (2) Mutation testing — introduce code mutations and verify the test detects them. (3) Randomise test cases across rollouts. (4) Add a held-out evaluation as a non-differentiable gate every 100 steps — alert when training reward diverges from held-out performance.*

### RL Systems, Scaling Laws & Distillation
{: #rl-systems-scaling}

**End-to-end RL system architecture:**

<div class="post-flow" role="group" aria-label="RL system architecture">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Prompt sampler → Rollout engine (vLLM, disaggregated from trainer)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Rollout engine → Environment/tool servers (sandboxed, async, cached)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Trajectories + rewards → Trainer (PPO/GRPO), with reference policy for KL</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Successful traces → Replay buffer → Distillation into SFT corpora</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Checkpoint registry → Held-out eval → Deployment gate</span></li>
  </ol>
</div>

**Training-inference disaggregation.** Rollout wants high-throughput serving (large batch size, KV cache reuse, speculative decoding). Training wants dense tensor-parallel backprop. Optimal configurations differ — separate them. **Logprob parity is critical:** MiniMax traced a major RL instability to precision mismatch between rollout and training logprobs. Fix: FP32 for the LM output head in both engines. All IS/trust-region methods assume exact logprob agreement — even small disagreements accumulate into training instability over thousands of tokens.

**Every rollout must log:** prompt, environment state, sampled actions, model logprobs, observations, reward components (not just the final scalar), policy version. These logs serve four roles: online RL training, offline diagnosis of reward hacks, distillation into SFT corpora, and research on better algorithms.

**Scaling laws for RL compute (ScaleRL, Oct 2025, >400k GPU-hours of curves).** Stable RL recipes follow predictable scaling trajectories analogous to pre-training scaling laws. Small pilot runs can estimate the shape of larger runs:

| Changes the asymptote | Changes compute efficiency |
|---|---|
| Prompt/task distribution | Loss aggregation, normalisation |
| Reward/verifier precision | Off-policy reuse ratio |
| Model architecture limits | Scheduling and rollout throughput |
| Environment richness | Clipping and baseline variants |

**Do not confuse "learns faster" with "will ultimately learn more."** Much of 2025 RL work improved efficiency without necessarily changing the final frontier under enough compute. The true scaling lever is often reward and environment quality, not the algorithm label.

**Distillation — three mechanisms:**

*Rejection-sampling distillation (offline, off-policy):* run RL-trained model on prompts, keep outputs passing verifier, add to SFT set. DeepSeek-R1 applies this twice.

*On-policy distillation (online, logit-level):* student generates completion; loss = `KL(π_student ∥ π_teacher)` on student's own tokens. Closes covariate-shift gap. Qwen3 uses this for reasoning — student and teacher run side by side.

*Off-policy distillation (offline, imitation):* student trains on teacher's pre-generated outputs via SFT loss. Cheapest but subject to covariate shift. Qwen3 uses this for thinking/non-thinking mode fusion, reporting ~10× less GPU than RL with comparable accuracy.

**Qwen3 thinking-mode fusion.** Four-stage pipeline: long-CoT cold start → reasoning RL → thinking-mode fusion → general RL. One deployable model with controllable reasoning budget — short mode for cheap requests, long/agent mode for frontier problems. Trade-off: fusion improves breadth but can slightly degrade peak specialised reasoning.

**Safety in RL loops.** RL amplifies any loophole via optimisation pressure. Treat reward design like security design: sandboxed environments, adversarial eval suites, non-regression gates before checkpoint promotion.

**A near-frontier RL programme checklist:**
1. Build a verifier-rich environment portfolio
2. Run cheap pilot sweeps to identify stable recipes and scaling trends
3. Scale rollout throughput and reward services together
4. Continuously audit reward hacking with held-out evaluations
5. Distill good traces and refresh the prompt mixture
6. Revisit the algorithm only after checking whether reward, data quality, or prompt distribution is the real bottleneck

> **Interview question:** Your manager asks you to switch from GRPO to a "newer algorithm" claiming it will improve your reasoning model by 10pp on MATH. How do you evaluate this claim?
>
> *Apply the ScaleRL framework: does the new algorithm change the asymptote or just compute efficiency? (1) Pilot experiment first: run both GRPO (your baseline) and the new algorithm for the same number of GPU-hours (e.g. 1,000 steps on a 7B model). If the new algorithm is only faster to reach a performance level GRPO would also reach given more compute, it is an efficiency improvement, not a capability improvement. (2) Check what ingredient changed: most "new" 2025 RL algorithms are GRPO remixes — they change the baseline estimator, the clipping shape, or the normalisation. Identify which of these changed, and whether that change addresses a specific observed failure mode in your training (zero-signal groups? length bias? entropy collapse?). If your training is not exhibiting those failure modes, the fix may not apply. (3) Check the asymptote: run both algorithms for 5× your normal compute budget. If the new algorithm's performance curve is still above GRPO's at 5× compute, it changes the asymptote. If they converge, it only changes efficiency. (4) Evaluate reward quality first: the ScaleRL result says the true scaling lever is often reward and environment quality. Before changing the algorithm, check whether improving your verifier precision or expanding your prompt distribution yields the same gain more cheaply. (5) Test generalisation: run both checkpoints on your held-out eval set, not just the training reward. An algorithm that improves training reward faster may also overfit to the training verifier faster.*
