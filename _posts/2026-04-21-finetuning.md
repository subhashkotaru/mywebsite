---
title: "Fine-tuning"
date: 2026-04-21
description: "Adapting pretrained language models to downstream tasks — from full fine-tuning to parameter-efficient methods, alignment, and RLHF."
tags: [ml-systems, fine-tuning, llm, peft]
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
        <li><a href="#lora">LoRA & Variants</a></li>
        <li><a href="#qlora">QLoRA</a></li>
        <li><a href="#side-tuning">Side Tuning</a></li>
      </ul>
    </li>
    <li><a href="#alignment">Alignment & RLHF</a>
      <ul class="post-toc-sublist">
        <li><a href="#sft">Supervised Fine-tuning</a></li>
        <li><a href="#reward-model">Reward Modelling</a></li>
        <li><a href="#ppo">PPO</a></li>
        <li><a href="#dpo">DPO</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## Overview
{: #overview}

A pretrained language model learns general statistical structure from vast amounts of text but is not immediately useful for a specific task — it may answer questions inconsistently, refuse instructions unpredictably, or produce outputs in the wrong format. **Fine-tuning** adapts the model's weights toward a target behaviour using a curated, labelled dataset. The challenge is doing this cheaply: a 175B model requires the same hardware as pretraining to update all its weights, making full fine-tuning impractical for most practitioners.

The field has converged on two complementary strategies:

- **Parameter-efficient fine-tuning (PEFT)** — update a small fraction of parameters (adapters, low-rank matrices, soft prompts) while keeping the bulk of the model frozen
- **Alignment fine-tuning** — use human preference data and reinforcement learning to steer model behaviour beyond what supervised fine-tuning alone achieves

<div class="post-flow post-flow--horizontal" role="group" aria-label="Fine-tuning spectrum">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Prompt engineering — no training</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">PEFT — few trainable params</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Full fine-tuning — all params</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green post-flow__bar--accent">RLHF — human preference alignment</span></li>
  </ol>
</div>

**Why fine-tune at all?** Few-shot prompting can handle many tasks but hits a ceiling: it cannot update the model's internal representations, is bounded by context length, and cannot reliably instil behaviours that require consistent multi-turn reasoning. Fine-tuning on task-specific data closes this gap substantially:

| Task | GPT-3 Few-shot | GPT-3 Fine-tuned |
|---|---|---|
| SQuAD V2 (F1) | 69.8% | 88.4% |
| RTE (Acc) | 69% | 85.4% |
| WikiSQL (Acc) | 20% | 73% |
| Spider (Acc) | 18% | 62% |

---

## Full Fine-tuning
{: #full-finetuning}

Full fine-tuning updates every parameter in the model on the target dataset using standard backpropagation. The training loop is identical to pretraining — the difference is data volume (thousands of examples rather than trillions of tokens) and learning rate (much smaller, to avoid catastrophic forgetting of pretrained knowledge).

**Memory requirements** are the bottleneck. For a model with `M` parameters, Adam fine-tuning requires:

<div class="post-flow" role="group" aria-label="Full fine-tuning memory per parameter">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">FP16 weights — 2 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">FP16 gradients — 2 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">FP32 master weights — 4 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">FP32 Adam momentum + variance — 8 bytes/param</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Intermediate activations — model/task dependent</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Total: ~16–20 bytes/param → 175B model needs ~2.8–3.5 TB</span></li>
  </ol>
</div>

For GPT-3 (175B), this demands ~80 A100-40GB GPUs and 1 TB of storage per checkpoint. Full fine-tuning is used when budget permits and maximum task accuracy is required — e.g. production instruction-tuning runs by labs with access to large GPU clusters.

**Catastrophic forgetting**: updating all weights aggressively on a narrow dataset erases previously learned capabilities. Mitigations include using a small learning rate (`1e-5` to `5e-6`), mixing fine-tuning data with replay samples from the pretraining distribution, and stopping early before the model overfits.

---

## Parameter-Efficient Fine-tuning
{: #peft}

PEFT methods freeze most of the pretrained model and introduce a small number of new or reparameterised trainable parameters. The frozen base handles general language understanding; the trainable component learns the task delta.

### Prompt & Prefix Tuning
{: #prompt-prefix}

**Prompt tuning** prepends a sequence of learnable token embeddings — **soft prompts** — to the input. The LLM's weights are frozen; only the soft prompt parameters are trained via backpropagation through the model. At inference, the soft prompt is prepended to every input automatically.

**Prefix tuning** extends this to every transformer layer: a learnable prefix is added to the key and value tensors of each attention layer, not just the input embeddings. The LLM attends to these prefix vectors at every layer, giving the prefix more influence over the model's internal representations.

<div class="post-flow post-flow--compare" role="group" aria-label="Prompt vs prefix tuning">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Prompt Tuning</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Soft tokens prepended to input only</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Trainable params: prefix_len × d_embed</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Works well for large models (≥10B)</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Prefix Tuning</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Prefix added to K, V at every layer</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Trainable params: prefix_len × n_layers × 2 × d_model</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Stronger task influence across depth</span></li>
    </ol>
  </div>
</div>

Both methods add zero inference latency beyond the extra KV entries in the prefix — no new layers, no weight merging needed.

### Adapters
{: #adapters}

**Adapter tuning** inserts small bottleneck modules between existing transformer layers. Each adapter applies a down-projection `W_down ∈ ℝᵈˣʳ`, a non-linearity, and an up-projection `W_up ∈ ℝʳˣᵈ` (where `r ≪ d`), with a residual connection bypassing it:

```
h = h + W_up(ReLU(W_down(h)))
```

Adapters are initialised near-identity (W_up initialised to zero) so training is stable from the start. During fine-tuning only the adapter weights are updated; during inference the adapter executes as an extra sub-layer in each transformer block.

**Tradeoff**: adapters add new layers, which increases inference latency slightly (extra matmuls per layer) and requires saving intermediate activations through the full frozen model during backprop — so memory savings on activations are limited even though weight and optimizer-state memory shrinks dramatically.

### LoRA & Variants
{: #lora}

**LoRA (Low-Rank Adaptation)** avoids new layers entirely by reparameterising the weight update. For a frozen weight `W ∈ ℝᵈˣᵈ`, the learned perturbation is factored as:

```
ΔW = B × A,   B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵈ,   r ≪ d
```

`A` is initialised from `𝒩(0, σ²)`; `B` is initialised to zero so `ΔW = 0` at step 0. The forward pass adds the low-rank term:

```
h = Wx + BAx = (W + BA)x
```

At deployment, `W' = W + BA` is computed once and folded in — **zero inference latency overhead**. LoRA is typically applied to the query, key, value, and output projection matrices in attention, and sometimes to the MLP layers.

**Rank selection**: `r = 4` to `r = 64` covers most use cases. Higher rank increases expressivity but also trainable parameter count (`2dr` per layer). For a `d=4096` layer with `r=8`, LoRA adds only 65,536 parameters vs 16.7M for the full weight.

**Variants**:

| Variant | Key idea | Benefit |
|---|---|---|
| LoHa | `ΔW = (B₁ ⊙ A₁)(B₂ ⊙ A₂)` — Hadamard product | Same params, higher effective rank |
| LoKr | `ΔW = B ⊗ A` — Kronecker product | Preserves matrix rank structure |
| DoRA | Decompose W into magnitude + direction; apply LoRA to direction | Closer to full fine-tuning dynamics |

### QLoRA
{: #qlora}

**QLoRA** quantises the frozen base model weights to 4 bits, reducing base model memory by 4× while keeping the LoRA adapters in fp16 for stable gradient flow.

Naive 4-bit quantisation wastes precision bins when weights have large outliers. QLoRA uses two levels:

1. **Block-wise quantisation** — divide weights into blocks of B=64; each block has its own fp32 scale constant. Overhead: 32/64 = 0.5 bits/param.
2. **Double quantisation** — quantise the fp32 scale constants themselves to int8 with block size 256. Overhead drops to `8/64 + 32/(64×256) ≈ 0.127` bits/param.

```
Layer memory breakdown (70B model):
  Base weights:  4-bit quantised    → ~35 GB
  LoRA A, B:     fp16               → ~0.5 GB
  Activations:   fp16               → ~78 GB
  Optimizer:     fp32 (LoRA only)   → ~4 GB
  Total QLoRA:   ~118 GB  vs  ~757 GB full fine-tuning
```

QLoRA achieves on-par accuracy with full fp16 fine-tuning, enabling 70B-scale models to be fine-tuned on a single 80 GB GPU.

### Side Tuning
{: #side-tuning}

Both adapters and LoRA require backpropagating gradients through the frozen base model — which means storing all intermediate activations in memory during the forward pass. For a 70B model this is ~197 GB of activations regardless of how few parameters are being trained.

**Side tuning** eliminates this by routing backpropagation through a separate, smaller **side network** rather than through the base model. Information flows from base to side via downsampled residual connections, but never in the reverse direction — so the base model requires only a forward pass:

<div class="post-flow" role="group" aria-label="Side tuning information flow">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Base LLM: 4-bit quantised, frozen, forward-only — no activations stored</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each base layer output → downsample → inject into side network</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Side network: small trainable transformer in fp16</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Backprop stays entirely within side network</span></li>
  </ol>
</div>

**Quantized Side Tuning (QST)** combines 4-bit double quantisation with a side network:

| Method | Weights | Optimizer states | Activations | Total (70B) |
|---|---|---|---|---|
| Full fine-tuning | 140 GB | 420 GB | 197 GB | 757 GB |
| QLoRA | 36 GB | 13 GB | 197 GB | 246 GB |
| QST | 36 GB | 4 GB | 69 GB | 109 GB |

QST matches QLoRA's accuracy at less than half the total memory, making 70B fine-tuning feasible on 2× 80 GB GPUs instead of 4.

---

## Alignment & RLHF
{: #alignment}

PEFT methods adapt a model to a task given labelled input-output pairs. **Alignment** is a different goal: shaping model behaviour to be helpful, harmless, and honest in open-ended conversation — where there is no single correct output and human preferences are the signal.

The standard alignment pipeline has three stages.

### Supervised Fine-tuning
{: #sft}

First, the base pretrained model is fine-tuned on a dataset of **high-quality demonstrations**: human-written conversations, instruction-response pairs, and chain-of-thought examples. This teaches the model the *format* of helpful responses — how to follow instructions, structure answers, and handle multi-turn dialogue.

SFT alone produces a model that mimics human-written text well but may still generate harmful content, be inconsistent across similar prompts, or optimise for "sounding good" rather than being correct. SFT is the entry point; RLHF is the correction pass.

### Reward Modelling
{: #reward-model}

A **reward model (RM)** is trained to score model outputs according to human preferences. The training data consists of **pairwise comparisons**: for the same prompt, a human annotator is shown two model responses and picks the preferred one.

The reward model is typically initialised from the SFT model (same architecture) with a scalar head added. It is trained with a Bradley-Terry pairwise ranking loss:

```
L = -E[log σ(r(x, y_w) - r(x, y_l))]
```

where `y_w` is the preferred response, `y_l` the rejected one, and `r(x, y)` is the scalar reward. The RM learns to assign higher scores to responses humans prefer — helpfulness, factual accuracy, appropriate tone — without those preferences being explicitly enumerated.

### PPO
{: #ppo}

With a trained reward model, the SFT model is further updated using **Proximal Policy Optimisation (PPO)** — a reinforcement learning algorithm. The SFT model is the *policy*: it takes a prompt as state and generates a response as action. The reward model scores the response.

<div class="post-flow" role="group" aria-label="PPO training loop for RLHF">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Sample a prompt from the dataset</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Policy (SFT model) generates a response autoregressively</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Reward model scores the response → scalar reward r</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">KL penalty: subtract β · KL(policy ‖ SFT) to prevent reward hacking</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">PPO updates policy weights to maximise r − β · KL</span></li>
  </ol>
</div>

The KL penalty is critical: without it the policy collapses into **reward hacking** — producing outputs that score highly on the reward model but are nonsensical to humans (the reward model, being imperfect, can be exploited). The KL term keeps the policy close to the SFT model.

PPO requires four models in memory simultaneously: the policy, a frozen reference policy (SFT copy for KL), the reward model, and a value network. This makes PPO expensive — typically requiring the same hardware as fine-tuning.

### DPO
{: #dpo}

**Direct Preference Optimisation (DPO)** bypasses the reward model and PPO entirely, fine-tuning the policy directly on preference pairs using a closed-form loss derived from the RLHF objective:

```
L_DPO = -E[log σ(β · log(π(y_w|x) / π_ref(y_w|x)) - β · log(π(y_l|x) / π_ref(y_l|x)))]
```

where `π` is the policy being trained, `π_ref` is the frozen SFT reference, and `β` controls how far the policy can move from the reference.

<div class="post-flow post-flow--compare" role="group" aria-label="PPO vs DPO">
  <div class="post-flow__col">
    <p class="post-flow__col-label">PPO (RLHF)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Requires separate reward model</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">4 models in memory simultaneously</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Complex training loop, reward hacking risk</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Strong performance on open-ended tasks</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">DPO ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">No reward model — trains directly on preferences</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">2 models in memory (policy + reference)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Simple supervised loss — stable training</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Less flexible for multi-step RL objectives</span></li>
    </ol>
  </div>
</div>

DPO is now the dominant alignment method for most fine-tuning practitioners: it achieves comparable or better performance than PPO on standard benchmarks at a fraction of the infrastructure cost. The tradeoff is that DPO cannot incorporate online feedback — it requires a fixed dataset of preference pairs, while PPO can actively sample and label new responses during training.
