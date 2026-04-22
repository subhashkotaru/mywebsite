---
title: "Pretraining"
date: 2026-04-21
description: "How large language models are pretrained — data pipelines, tokenisation, training objectives, architecture choices, scaling laws, and the reasoning behind every design decision."
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
    <li><a href="#init-loss-reg">Initialisation, Loss & Regularisation</a>
      <ul class="post-toc-sublist">
        <li><a href="#weight-init">Weight Initialisation (Xavier & He)</a></li>
        <li><a href="#loss-functions">Loss Functions</a></li>
        <li><a href="#dropout-reg">Dropout & Regularisation</a></li>
      </ul>
    </li>
    <li><a href="#optimiser">Optimiser & Training Stability</a>
      <ul class="post-toc-sublist">
        <li><a href="#adam">AdamW</a></li>
        <li><a href="#lr-schedule">Learning Rate Schedule</a></li>
        <li><a href="#grad-clipping">Gradient Clipping</a></li>
        <li><a href="#batch-size">Batch Size Selection</a></li>
      </ul>
    </li>
    <li><a href="#optimisers-deep">Optimisers: From SGD to Muon</a>
      <ul class="post-toc-sublist">
        <li><a href="#sgd-momentum">SGD & Momentum</a></li>
        <li><a href="#adam-adamw">Adam & AdamW</a></li>
        <li><a href="#lion">Lion</a></li>
        <li><a href="#shampoo">Shampoo</a></li>
        <li><a href="#muon">Muon</a></li>
        <li><a href="#mup">Maximal Update Parameterisation (muP)</a></li>
      </ul>
    </li>
    <li><a href="#token-sampling">Token Sampling & Decoding</a>
      <ul class="post-toc-sublist">
        <li><a href="#greedy-beam">Greedy & Beam Search</a></li>
        <li><a href="#temperature">Temperature Scaling</a></li>
        <li><a href="#topk-topp">Top-k & Top-p (Nucleus)</a></li>
        <li><a href="#minp">min-p Sampling</a></li>
        <li><a href="#sampling-tradeoffs">Tradeoffs & Practical Guide</a></li>
      </ul>
    </li>
    <li><a href="#scaling-laws">Scaling Laws</a></li>
    <li><a href="#systems">Systems for Pretraining at Scale</a></li>
    <li><a href="#tokens-day">Tokens/Day: The End-to-End Metric</a>
      <ul class="post-toc-sublist">
        <li><a href="#mfu-goodput">MFU vs Goodput</a></li>
        <li><a href="#step-timeline">Step-Time Decomposition</a></li>
        <li><a href="#mixed-precision-depth">Mixed Precision In Depth</a></li>
        <li><a href="#stability-dashboard">Training Stability Dashboard</a></li>
        <li><a href="#operational-kpis">Operational KPIs & MTBF</a></li>
        <li><a href="#corpus-engineering">Corpus Engineering</a></li>
        <li><a href="#synthetic-data">Synthetic & Instruction Data in Pre-Training</a></li>
      </ul>
    </li>
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

What makes pretraining expensive: GPT-3 (175B parameters) required ~3.14 × 10²³ FLOPs — roughly 355 GPU-years on V100s. The cost is dominated by matrix multiplications in attention and FFN layers, performed over 300B tokens. A rough rule of thumb: training a transformer costs approximately **6ND FLOPs**, where N is parameter count and D is training tokens (each parameter is touched twice per forward pass — multiply + accumulate — and once per backward pass × 2).

---

## Data Pipeline
{: #data}

### Sources & Scale
{: #data-sources}

Modern LLMs are trained on diverse, multi-source corpora:

| Source | Examples | Characteristics |
|---|---|---|
| Web crawl | Common Crawl, C4 | Massive scale, noisy quality |
| Books | BookCorpus, Gutenberg | Long-range coherence, formal writing |
| Code | GitHub, StackOverflow | Structured, reasoning-dense |
| Academic | ArXiv, PubMed | Technical depth, precise language |
| Curated | Wikipedia, news | High quality, broad coverage |

LLaMA-3.1 was trained on 15 trillion tokens; GPT-3 on 300B. The raw Common Crawl snapshot runs to petabytes — a substantial engineering effort is required just to ingest and process it.

**Domain mixing matters as much as volume.** Code improves multi-step reasoning (it trains the model to follow strict logical chains); books improve long-range coherence (paragraphs must stay on topic across thousands of tokens); web text provides breadth of knowledge. Llama-2 used roughly 89.7% web + books, 8% code, 2.5% Wikipedia. Setting this mix is a hyperparameter search — more code moves the model toward better reasoning but away from fluent prose generation.

**Why not just take more web data?** Web quality follows a power law: the top few percent of web pages are excellent; the long tail is spam, near-duplicate boilerplate, and machine-generated noise. Adding more low-quality web data past a certain point actively hurts — the model trains on noise instead of signal. This is why curated sources are upsampled: Wikipedia is repeated 3–4× in many training runs despite being a small fraction of raw crawl volume.

> **Interview question:** If you doubled the proportion of code in your training mix, what would you expect to improve and what might regress? What experiment would you run to check?
>
> *Code improves structured reasoning and instruction-following on technical tasks. You'd expect gains on GSM8k (math), HumanEval (code), and MATH benchmarks. What might regress: fluency on open-ended conversational tasks, performance on commonsense reasoning that relies on world knowledge encoded in natural prose. Experiment: train two small ablations (e.g. 1B parameters, 50B tokens) with different code ratios, evaluate on a broad eval suite covering both reasoning and knowledge tasks.*

### Quality Filtering
{: #data-quality}

Raw web crawl data contains spam, duplicates, low-quality boilerplate, and toxic content. A multi-stage filtering pipeline is applied:

<div class="post-flow" role="group" aria-label="Data quality filtering pipeline">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Language detection — keep target languages, discard others</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Heuristic filters — min/max length, punctuation ratio, repeated n-gram ratio</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Deduplication — exact and near-duplicate removal via [MinHash](https://arxiv.org/abs/2107.06499) LSH + suffix arrays</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Quality scoring — classifier trained on curated (Wikipedia/books) vs raw web text</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Toxicity filtering — harmful content removed with classifier or keyword lists</span></li>
  </ol>
</div>

**Why deduplication is not optional.** The web contains enormous amounts of near-duplicate text — mirrored articles, scraped pages, boilerplate headers. Training on duplicates causes two problems: (1) the model memorises specific passages verbatim rather than learning generalisable patterns; (2) duplicated documents effectively upweight certain facts and writing styles, biasing the model's learned distribution. MinHash LSH finds near-duplicates at scale in sub-quadratic time by hashing shingles (overlapping n-grams) and using locality-sensitive hashing to group similar documents without exhaustive pairwise comparison.

**Why a neural quality classifier?** Heuristic filters (min length, punctuation density) catch obvious garbage but miss subtly low-quality text — articles that are grammatical but factually wrong, coherent but content-free. Training a classifier to distinguish Wikipedia-quality prose from web crawl text generalises better and can be calibrated to trade precision against recall.

> **Interview question:** Your training data pipeline has a quality classifier that discards 80% of crawled data. Someone proposes lowering the threshold to keep more data, arguing "more data is always better." When are they right, and when are they wrong?
>
> *They're right if the discarded data is genuinely informative and the classifier has high false-negative rate. They're wrong if the discarded data is mostly noise — adding noisy tokens dilutes the training signal, increasing training cost without proportional quality gain. The right test: train two small ablations on the same compute budget, one with the tighter filter and one looser, and compare eval loss. If the tighter filter wins despite fewer tokens, noise is the problem.*

### Tokenisation
{: #tokenisation}

Text is converted to integers before feeding to the model. The tokeniser defines the vocabulary of atomic units the model works with.

**Byte-Pair Encoding (BPE)** is the dominant approach. It starts with individual bytes (256 symbols) and iteratively merges the most frequent adjacent pair into a new token:

```
Iteration 0: ["l", "o", "w", "e", "r"]
Iteration 1: ["lo", "w", "e", "r"]      # "l"+"o" merged (most frequent pair)
Iteration 2: ["lo", "we", "r"]          # "w"+"e" merged
...
Final vocab: 32k–128k tokens
```

BPE finds an efficient compression of the training corpus — common words become single tokens, rare words decompose into subword pieces. GPT-4 uses `cl100k_base` (100,277 tokens); LLaMA-3 uses 128k tokens.

**Why does vocabulary size matter?** Larger vocabulary → fewer tokens per document → shorter sequences → cheaper attention (O(L²)) → more documents fit in the same context window → less information is cut off. But larger vocabulary → bigger embedding matrix (embedding table is `V × d_model` — for V=128k, d=4096, that's 512M parameters just for embeddings) → more parameters to train for rare tokens with few training examples → sparse gradient updates for rare tokens.

**The sweet spot** in modern models is 32k–128k. Below ~32k you pay a high sequence-length cost for complex languages; above ~200k the embedding table becomes unwieldy and rare tokens are under-trained.

**Why BPE over word-level tokenisation?** Words fail on morphologically rich languages (Finnish, Turkish), rare words, and code. A word-level tokeniser with V=50k assigns UNK to anything outside the vocabulary. BPE handles arbitrary text gracefully: any byte sequence can be represented using the 256-byte fallback tokens.

**Tokenisation is not linguistically neutral.** Numbers are often split into individual digits (`"2024"` → `["20", "24"]`), making arithmetic hard because the model never sees "2024" as a unit. Non-English scripts with larger character sets (Chinese, Arabic) get fewer characters per token, inflating effective sequence length and raising inference cost relative to English. This is a known source of cross-lingual capability gap.

> **Interview question:** You're designing a tokeniser for a model that needs to do arithmetic. What changes would you make to standard BPE, and why?
>
> *The core problem is that numbers are split arbitrarily — "1024" might become ["10", "24"] or ["1", "024"] depending on frequency. Approaches: (1) always tokenise digits individually (1 digit = 1 token) so the model always sees consistent atomic units; (2) add explicit number tokens for 0–9999; (3) represent numbers in a canonical format (e.g. space-separated digits: "1 0 2 4"). The tradeoff is sequence length: individually-tokenised numbers are 4× longer, increasing attention cost. Models like Minerva and PaLM used digit-level tokenisation for math tasks.*

---

## Training Objective
{: #objective}

### Causal Language Modelling
{: #clm}

**Causal language modelling (CLM)** — next-token prediction — is the pretraining objective for decoder-only models (GPT, LLaMA, Mistral). Given a sequence `x₁, x₂, ..., xₜ`, the model minimises the negative log-likelihood of each token given its prefix:

```
L = -Σₜ log P(xₜ | x₁, ..., xₜ₋₁)
```

The training data is just text — no labels needed. Every token in the sequence is simultaneously a training input and a prediction target (shifted by one). This makes CLM extremely data-efficient: a 1024-token sequence yields 1024 prediction targets, all computed in a single forward pass.

**Causal masking** ensures position `i` can only attend to positions `≤ i`. This triangular mask is implemented efficiently in FlashAttention and is what makes decoder-only models autoregressive — each position is predicted without seeing future context.

**Why CLM produces general-purpose representations.** To predict the next token well, the model must compress a summary of all prior context into the residual stream at each position. Predicting `"the cat sat on the ___"` requires knowing that cat + sat implies a surface. Predicting code requires understanding scoping rules. Predicting scientific text requires knowing domain facts. The breadth of the corpus forces the model to learn general-purpose representations — because any narrow representation will fail on the vast majority of contexts.

> **Interview question:** CLM trains on every token position equally. Is this optimal? What would you change?
>
> *Uniform weighting is not optimal. Predicting common function words ("the", "a", "of") gives very little learning signal — these are nearly deterministic from context. Content words and low-frequency tokens carry more information per prediction. Techniques that address this: (1) token weighting — up-weight rare tokens in the loss; (2) curriculum learning — start with easier sequences, gradually increase difficulty; (3) ELECTRA-style discriminative objective — predict which tokens were replaced rather than generating them, giving a signal on every token including common ones. In practice, most production runs still use flat CLM because the engineering simplicity outweighs these gains at scale.*

### Masked Language Modelling
{: #mlm}

**Masked language modelling (MLM)** — used by [BERT](https://arxiv.org/abs/1810.04805) and encoder-only models — randomly masks 15% of input tokens and trains the model to predict the masked values. Unlike CLM, the model sees full bidirectional context around each mask:

```
Input:  "The cat [MASK] on the mat"
Target:        "sat"
```

MLM produces better representations for classification and extraction tasks (full context matters) but cannot generate text autoregressively — the model needs to see the complete sequence to fill in masks.

**The 15% masking rate is not arbitrary.** Too low and each training step provides sparse signal — most positions contribute zero loss. Too high and the input is so corrupted that there is insufficient context to predict masks reliably — the model learns to guess from statistics rather than context. 15% is empirically the crossover where context remains informative enough to support meaningful prediction.

> **Interview question:** Why can't you use an MLM-pretrained model (like BERT) directly for text generation?
>
> *BERT's attention is bidirectional — at inference time, it expects to see the full sequence to make predictions. There is no causal ordering enforced. If you tried to generate autoregressively, position t+1 would need to attend only to positions ≤ t, but BERT was never trained with that constraint. The result is that it has no notion of conditional generation — it learned to fill in gaps given surrounding context, not to extend a prefix. You would need to add a causal mask and fine-tune for generation, essentially converting it into a decoder, which largely throws away what MLM pretraining learned.*

---

## Model Architecture
{: #architecture}

All modern LLMs are built on the **[transformer](https://arxiv.org/abs/1706.03762)** architecture. The design choices made during pretraining become fixed — they cannot be changed without retraining from scratch. Every architectural decision is a trade-off between expressivity, memory, and compute.

### Transformer Block
{: #transformer-block}

Each transformer block applies two sub-layers with residual connections:

```
h = h + SelfAttention(RMSNorm(h))   # pre-norm: normalise before the sub-layer
h = h + FFN(RMSNorm(h))
```

**Self-attention** computes pairwise token relevance: `A(Q,K,V) = softmax(QKᵀ/√d)·V`. In multi-head attention, this is run H times in parallel with different learned projections, then concatenated. The `O(L²)` cost in sequence length L is the main scaling bottleneck for long contexts.

**Why scale by `√d`?** The raw dot product `QKᵀ` grows in magnitude as `d` increases — the components of Q and K each have variance 1, so their dot product has variance `d`. Without scaling, the softmax saturates into a near-one-hot distribution (all attention on one token) and gradients vanish. Dividing by `√d` keeps the dot products O(1) regardless of model dimension.

**Feed-forward network**: each FFN is two linear layers with a non-linearity. The FFN contains approximately 2/3 of a transformer's parameters and is where most factual knowledge is believed to be stored.

- GPT-style: `FFN(x) = ReLU(xW₁)W₂`, hidden dim = 4×d
- Modern (LLaMA, Mistral): `FFN(x) = (SiLU(xW₁) ⊙ xW_gate)W₂` — SwiGLU gating, hidden dim = 8/3×d

**Why SwiGLU over ReLU?** ReLU hard-zeros negative activations — this creates dead neurons and produces sparse activations that are hard for gradient flow. SwiGLU uses `SiLU(x) = x·σ(x)` (smooth, always nonzero gradient) combined with a gating mechanism `(SiLU(xW₁) ⊙ xW_gate)` that lets the network learn to selectively pass or suppress information. Empirically, SwiGLU consistently outperforms ReLU at matched parameter count, which is why every major model since PaLM uses it.

**Why is the hidden dim 8/3×d for SwiGLU vs 4×d for ReLU?** SwiGLU uses three weight matrices (W₁, W_gate, W₂) vs two for ReLU (W₁, W₂). To keep total FFN parameter count equal to the ReLU FFN (which has 2 × d × 4d = 8d² parameters), SwiGLU sets hidden dim to 8d/3 so that 3 × d × 8d/3 = 8d². In practice this is rounded to the nearest multiple of 64 for hardware efficiency.

> **Interview question:** Why do transformers use residual connections? What would happen if you removed them?
>
> *Residual connections `h = h + F(h)` create a direct gradient highway from the loss back to early layers: `∂L/∂h_early = ∂L/∂h_late + cross-terms`. Without residuals, gradients must flow multiplicatively through every layer's Jacobian. In a 96-layer model, the product of 96 Jacobians either explodes or vanishes depending on whether their singular values are above or below 1 — this is the deep network training problem. With residuals, the identity term ensures a minimum gradient of 1 regardless of depth. Additionally, residuals allow the model to learn "corrections" to the identity transformation — early layers pass information through largely unchanged while later layers refine it, which is a more natural learning dynamic.*

### Positional Encoding
{: #positional-encoding}

Transformers have no built-in notion of sequence order — positional encodings provide it.

<div class="post-flow post-flow--compare" role="group" aria-label="Absolute vs rotary positional encoding">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Absolute (sinusoidal / learned)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Fixed vector added to each token embedding at position m</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Attention score encodes absolute positions — model has to learn relative distance implicitly</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Generalises poorly beyond training context length</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Used in original Transformer, BERT, GPT-2</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">RoPE (Rotary) ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Rotates Q and K vectors by position-dependent angle θ·m</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Dot product Q·K encodes relative distance m−n automatically</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Better extrapolation to longer contexts via position interpolation</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Used in LLaMA, Mistral, GPT-4, Gemini</span></li>
    </ol>
  </div>
</div>

**How RoPE works mechanically.** RoPE applies a rotation matrix `R(θ·m)` to query and key vectors at position `m`. Because `R(θ·m)ᵀ R(θ·n) = R(θ·(m−n))`, the dot product between position-m query and position-n key automatically encodes the *relative* distance `m−n`. The base frequency `θ` controls how fast the rotation cycles — lower θ means slower variation, enabling longer-range attention patterns.

**Why relative > absolute for generalisation.** In language, what matters is typically the relationship between tokens, not their absolute positions in the document. "The dog that chased the cat ran away" — the model needs to know "chased" is 3 positions after "dog", not that "dog" is at position 4. Absolute encodings force the model to learn this difference implicitly; RoPE encodes it directly.

**RoPE and context extension.** When a model trained on 4K tokens sees a 100K-token input, it encounters rotational angles it has never seen during training. Position interpolation (YaRN, LongRoPE) rescales the angle — instead of position 50000 rotating at 50000·θ, it rotates at 50000·θ·(4000/100000), keeping it within the trained range. This is why RoPE-based models can be extended to longer contexts more easily than models with learned absolute embeddings.

> **Interview question:** A model trained with RoPE on 8K context is asked to do inference on a 32K input. What happens, and how would you fix it?
>
> *Without adjustment: positions 8001–32000 generate rotational angles outside the training distribution. The attention scores at those positions are unreliable — the model may either attend randomly to distant tokens or ignore them. The softmax may still work but produce meaningless distributions. Fix: position interpolation — rescale positions so 32K maps to the 0–8K angular range (divide each position index by 4). This keeps all rotational angles in-distribution. Better fix: fine-tune the model on longer sequences after interpolation — a few thousand steps are sufficient to adapt (LongLoRA does this efficiently with sparse attention during fine-tuning).*

### Normalisation
{: #normalisation}

Normalisation stabilises training by preventing activations from growing or shrinking uncontrollably as they pass through many layers.

**LayerNorm** (original transformer): normalise across the feature dimension per token. Computes both mean and variance:

```
LayerNorm(x) = (x − μ) / √(σ² + ε) · γ + β
```

where μ and σ² are computed over the d-dimensional feature vector for each token. Four statistics are needed per token: mean, variance, and two learned parameters (γ, β — scale and bias).

**RMSNorm** (modern LLMs): normalise by root mean square only, dropping the mean-centering step:

```
RMSNorm(x) = x / RMS(x) · γ,   RMS(x) = √(1/d · Σᵢ xᵢ²)
```

**Why RMSNorm is ~15% faster than LayerNorm.** LayerNorm requires computing the mean first, then the variance relative to that mean — two passes over the feature vector (or one pass with online Welford algorithm, but still more operations). RMSNorm computes only the second moment (sum of squares), halving the arithmetic. No β parameter either. The speedup is real and compounds across all layers in a 96-layer model.

**Why RMSNorm works as well empirically.** The mean-centering step in LayerNorm was motivated by the original BN paper's argument that removing mean shift reduces internal covariate shift. In practice, transformers with residual connections already constrain activations to stay near zero — the mean is naturally small. The variance (RMS) is the active constraint that prevents explosion. Dropping mean-centering loses little while gaining computation.

**Pre-norm vs post-norm.** The original transformer used post-norm: `h = LayerNorm(h + F(h))`. Almost all modern models use pre-norm: `h = h + F(LayerNorm(h))`.

Why pre-norm is more stable:

- **Post-norm** applies normalisation to the sum `h + F(h)`. At initialisation, `F(h)` is small (weights are tiny), so normalisation is effectively applied to `h` alone — the residual contribution disappears, preventing deep models from learning identity-like early-layer transformations. More critically, post-norm creates gradient pathways that amplify variance with depth.
- **Pre-norm** normalises the *input* to each sub-layer before it enters. Gradients flow through the residual pathway without passing through the normalisation op, giving stable gradient magnitudes regardless of depth.

The practical evidence: pre-norm models train stably at 96+ layers without learning rate warm-up hacks required for post-norm models of similar depth.

> **Interview question:** If you removed all normalisation from a transformer, what would happen during training, and at what depth does it become a problem?
>
> *Without normalisation, activations at layer l grow as a function of layer depth. Each linear layer has random initialisation with variance ~1/d (Xavier/He init). In the best case the singular values of each weight matrix are near 1, so activations neither explode nor vanish. In practice, the self-attention softmax creates spiky attention patterns that concentrate probability mass, leading to large activations in some heads. By layer 20–30, these compound multiplicatively and activations explode into NaN. In shallower models (6–12 layers) you might get away without normalisation with very careful initialisation, but it fails reliably at frontier scale. The deeper the model, the more critical normalisation becomes.*

---

## Initialisation, Loss & Regularisation
{: #init-loss-reg}

### Weight Initialisation: Xavier & He
{: #weight-init}

Weight initialisation sets the scale of parameters before any gradient updates. The goal: keep activation variance and gradient variance roughly constant across layers at the start of training — too small and gradients vanish, too large and they explode.

**The variance propagation problem.** In a linear layer $y = Wx$ where $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$, if each weight $w_{ij} \sim \mathcal{N}(0, \sigma^2)$ and inputs are zero-mean with variance $\text{Var}(x_i)$:

$$\text{Var}(y_j) = d_\text{in} \cdot \sigma^2 \cdot \text{Var}(x_i)$$

For variance to be preserved ($\text{Var}(y) = \text{Var}(x)$), we need $\sigma^2 = 1/d_\text{in}$.

**Xavier (Glorot) initialisation** ([Glorot & Bengio 2010](http://proceedings.mlr.press/v9/glorot10a.html)) — designed for linear activations and tanh/sigmoid:

$$\sigma^2 = \frac{2}{d_\text{in} + d_\text{out}}$$

or equivalently uniform $U\!\left[-\sqrt{\frac{6}{d_\text{in}+d_\text{out}}},\; \sqrt{\frac{6}{d_\text{in}+d_\text{out}}}\right]$

The $d_\text{in} + d_\text{out}$ denominator is a compromise: the forward pass wants $1/d_\text{in}$ (keep $\text{Var}(y) = \text{Var}(x)$) and the backward pass wants $1/d_\text{out}$ (keep gradient variance constant). Xavier averages the two.

**He (Kaiming) initialisation** ([He et al. 2015](https://arxiv.org/abs/1502.01852)) — designed for ReLU and its variants:

$$\sigma^2 = \frac{2}{d_\text{in}}$$

Why the factor of 2? ReLU zeros out negative activations, halving the effective variance. The factor of 2 compensates for this. Without it, activations shrink by $\sqrt{2}$ per layer — in a 50-layer network, that's $2^{-25}$ attenuation.

For Leaky ReLU with slope $\alpha$:

$$\sigma^2 = \frac{2}{(1 + \alpha^2)\, d_\text{in}}$$

**Comparison:**

| Initialisation | Formula | Designed for | Problem it solves |
|---|---|---|---|
| **Xavier (uniform)** | $U\!\left[-\sqrt{6/(d_\text{in}+d_\text{out})},\; \sqrt{6/(d_\text{in}+d_\text{out})}\right]$ | Tanh, sigmoid, linear | Forward + backward variance preservation |
| **Xavier (normal)** | $\mathcal{N}(0,\; 2/(d_\text{in}+d_\text{out}))$ | Tanh, sigmoid, linear | Same; normal version |
| **He (Kaiming)** | $\mathcal{N}(0,\; 2/d_\text{in})$ | ReLU, Leaky ReLU | Compensates for ReLU dead half |
| **LeCun** | $\mathcal{N}(0,\; 1/d_\text{in})$ | SELU | Requires specific activation + architecture |

**Bias initialisation:** almost always zeros — the bias does not affect variance propagation. Exception: output gate bias in LSTMs is sometimes initialised to 1 to encourage open gates early in training.

**In practice:** PyTorch default is Kaiming uniform for `nn.Linear`. For transformers, a common addition is scaling output projection weights by $1/\sqrt{2L}$ where $L$ is the number of layers — this keeps the residual stream variance stable at initialisation.

> **Interview question:** A 50-layer MLP using ReLU activations and Xavier initialisation has vanishing activations in the forward pass. Why, and how do you fix it?
>
> *Xavier was derived assuming linear (or symmetric) activations. ReLU zeros half the activation distribution — the effective fan-in that propagates signal is only $d_\text{in}/2$, not $d_\text{in}$. So Xavier's $\sigma^2 = 2/(d_\text{in}+d_\text{out})$ is too small by a factor of ~2. After 50 layers, activations decay by $(\approx 0.5)^{50/2} \approx 2^{-25}$, effectively zero. Fix: use He initialisation ($\sigma^2 = 2/d_\text{in}$). The factor of 2 exactly compensates for ReLU's expected variance halving.*

---

### Loss Functions
{: #loss-functions}

The loss function defines what the model is trained to optimise — a wrong choice leads to wrong gradients, poor convergence, or a model that optimises the wrong objective. Every loss has a probabilistic interpretation as a negative log-likelihood under some noise model.

**Autoregressive pretraining loss.** LLMs are trained with cross-entropy over the vocabulary at each position — the loss is the negative log-probability of the correct next token:

$$\mathcal{L}_\text{CLM} = -\frac{1}{T} \sum_{t=1}^T \log p_\theta(x_t \mid x_{<t})$$

This is the standard training objective for GPT-style models.

#### Classification Losses

**Cross-entropy (categorical).** For $M$ classes with one-hot target $y$ and softmax probabilities $p$:

$$\mathcal{L}_\text{CE} = -\sum_{c=1}^M y_c \log p_c = -\log p_{\text{true}}$$

The one-hot simplification means cross-entropy collapses to the negative log-probability of the correct class. Probabilistic basis: negative log-likelihood under the categorical distribution. Default choice for multi-class classification.

**Binary cross-entropy.** For binary targets $y \in \{0,1\}$ with sigmoid output $p$:

$$\mathcal{L}_\text{BCE} = -(y \log p + (1-y) \log(1-p))$$

Probabilistic basis: negative log-likelihood under the Bernoulli distribution. Used for binary classification and multi-label classification (independent sigmoid per class).

**Cross-entropy and KL divergence.** These are related by:

$$H(P, Q) = D_\text{KL}(P \| Q) + H(P)$$

When the true distribution $P$ is fixed (one-hot labels), $H(P)$ is constant, so minimising cross-entropy is equivalent to minimising KL divergence. In knowledge distillation where $P$ is a soft teacher distribution, you must use KL divergence directly since $H(P)$ changes.

**KL divergence:**

$$D_\text{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

Not symmetric: $D_\text{KL}(P\|Q) \neq D_\text{KL}(Q\|P)$. Forward KL ($P\|Q$, "mean-seeking") vs. reverse KL ($Q\|P$, "mode-seeking") have different behaviours — important in variational inference and distillation.

**Focal loss** ([Lin et al. 2017](https://arxiv.org/abs/1708.02002)):

$$\mathcal{L}_\text{focal} = -(1 - p_t)^\gamma \log p_t$$

where $p_t$ is the predicted probability for the true class and $\gamma \geq 0$ is the focusing parameter. At $\gamma=0$, reduces to standard cross-entropy. As $\gamma$ increases, the $(1-p_t)^\gamma$ factor downweights easy (well-classified) examples and focuses gradient on hard misclassified ones. Invented for object detection with extreme foreground/background imbalance (1:1000+). Use when class imbalance is severe.

**Hinge loss (SVM loss).** For binary labels $t \in \{-1,+1\}$ and classifier score $y$:

$$\mathcal{L}_\text{hinge} = \max(0,\; 1 - t \cdot y)$$

Zero loss when $t \cdot y \geq 1$ (correct prediction with sufficient margin). Non-differentiable at the hinge — use subgradient methods. Convex for linear classifiers. Less common in deep networks (cross-entropy preferred).

**Label smoothing.** Replaces hard one-hot targets with soft targets:

$$\tilde{y}_c = (1 - \epsilon)\, y_c + \frac{\epsilon}{M}$$

where $\epsilon$ is the smoothing factor (typically 0.1) and $M$ is the number of classes. Prevents the model from becoming overconfident (logits growing to ±∞). Used in nearly all modern LLM training runs.

#### Regression Losses

**Mean Squared Error (MSE / L2):**

$$\mathcal{L}_\text{MSE} = \frac{1}{m} \sum_{i=1}^m (y^{(i)} - \hat{y}^{(i)})^2$$

Probabilistic basis: maximum likelihood under Gaussian noise. Gradient grows linearly with error, making it sensitive to outliers. Use when Gaussian noise is a reasonable assumption and large errors should be penalised strongly.

**Why MSE is wrong for classification:** assumes Gaussian output noise (incorrect for discrete labels), creates non-convex objectives with sigmoid/softmax, and produces vanishing gradients for confident predictions. Always use cross-entropy for classification.

**Mean Absolute Error (MAE / L1):**

$$\mathcal{L}_\text{MAE} = \frac{1}{m} \sum_{i=1}^m |y^{(i)} - \hat{y}^{(i)}|$$

Probabilistic basis: maximum likelihood under Laplace noise. Constant gradient magnitude ($\pm 1$) makes it robust to outliers but can cause slower convergence near the optimum (gradient doesn't decrease as error decreases).

**Huber loss (Smooth L1):** hybrid that is quadratic for small errors and linear for large ones:

$$\mathcal{L}_\delta(a) = \begin{cases} \frac{1}{2} a^2 & |a| \leq \delta \\ \delta\left(|a| - \frac{1}{2}\delta\right) & |a| > \delta \end{cases}$$

Continuously differentiable at the transition point. Best of both worlds: MSE's smooth convergence near optimum + MAE's outlier robustness. $\delta$ is a hyperparameter — tune to the scale of expected residuals.

#### Metric Learning Losses

**Triplet loss.** Given anchor $a$, positive $p$ (same class), and negative $n$ (different class):

$$\mathcal{L}_\text{triplet} = \max\!\left(0,\; d(a, p) - d(a, n) + \text{margin}\right)$$

Enforces positives closer than negatives by at least `margin` (typically 1.0). Used in face recognition, person re-ID. Requires careful triplet mining — random negatives lead to trivial (zero-loss) training examples.

**InfoNCE / NT-Xent (contrastive).** For anchor $x_i$ and positive $x_i^+$ in a batch of $N$ pairs:

$$\mathcal{L}_\text{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \exp(\text{sim}(z_i, z_k)/\tau)}$$

$\tau$ is temperature (typical: 0.07). Larger batches provide more negatives — performance scales with batch size. Used in SimCLR, CLIP, and contrastive pretraining.

#### Loss Selection Guide

| Task | Loss | Activation | Notes |
|---|---|---|---|
| Multi-class classification | Cross-entropy | Softmax | Default |
| Binary classification | Binary cross-entropy | Sigmoid | |
| Multi-label classification | Binary cross-entropy | Sigmoid (per class) | Independent per class |
| Regression (Gaussian noise) | MSE | Linear | |
| Regression (outliers) | Huber or MAE | Linear | Tune $\delta$ |
| Class imbalance | Focal loss | Sigmoid/Softmax | $\gamma \in [0.5, 5]$ |
| Knowledge distillation | KL divergence | Softmax | Soft targets from teacher |
| Metric learning | Triplet or InfoNCE | L2-normalised embedding | |
| LLM pretraining | Cross-entropy | Softmax over vocab | One per position |

---

### Dropout & Regularisation
{: #dropout-reg}

Regularisation techniques constrain the model during training to improve generalisation. The core problem they address: a sufficiently large model can memorise the training set — regularisation makes memorisation harder and generalisation easier.

#### Weight Decay (L2 Regularisation)

Adds a penalty proportional to the squared magnitude of all weights to the loss:

$$\mathcal{L}_\text{reg} = \mathcal{L}_\text{task} + \frac{\lambda}{2} \sum_j w_j^2$$

Gradient update becomes: $w \leftarrow w(1 - \lambda\eta) - \eta \nabla_w \mathcal{L}_\text{task}$

The $(1 - \lambda\eta)$ factor shrinks weights toward zero each step — hence "weight decay." Equivalent to placing a Gaussian prior on weights ($w \sim \mathcal{N}(0, 1/\lambda)$) and computing the MAP estimate.

**L2 regularisation ≠ weight decay with Adam.** With SGD they are equivalent. With Adam, L2 adds to the gradient *before* adaptive scaling — large-gradient parameters get proportionally less regularisation. Weight decay applies the penalty *after* adaptive scaling, uniformly across all parameters. AdamW implements true weight decay; Adam + L2 does not.

#### Dropout

Dropout ([Srivastava et al. 2014](https://jmlr.org/papers/v15/srivastava14a.html)) randomly zeros out activations during training with probability $p$ (the *drop rate*):

**Standard dropout:**
- Training: each activation independently zeroed with probability $p$; surviving activations unchanged
- Inference: multiply all weights by $(1-p)$ to match expected activation magnitude

**Inverted dropout** (standard in practice):
- Training: zero with probability $p$; scale surviving activations by $1/(1-p)$
- Inference: no modification — weights are already in the right scale

Inverted dropout is preferred because inference is unchanged regardless of the training drop rate.

**Why dropout works:**
1. **Breaks co-adaptation:** neurons cannot rely on specific other neurons always being present — each must learn useful features independently
2. **Implicit ensemble:** with drop rate $p$ and $n$ neurons, there are $2^n$ possible sub-networks. Dropout approximates averaging over this ensemble
3. **Sparse representations:** encourages neurons to activate selectively rather than diffusely
4. **Training noise:** random masking acts as data augmentation at the representation level

**Architecture sizing rule:** if you plan to use dropout rate $p$ on a layer with $n$ units, size the layer to at least $n/(1-p)$ units. The effective capacity should match the task, not the pre-dropout size.

| Layer type | Typical drop rate |
|---|---|
| Input / visible layer | $p = 0.1$–$0.2$ |
| Hidden layers (MLP) | $p = 0.3$–$0.5$ |
| Attention dropout (transformers) | $p = 0.0$–$0.1$ |
| Residual dropout (transformers) | $p = 0.0$–$0.1$ |

Transformers at scale use very low or zero dropout — the massive dataset provides sufficient regularisation. Dropout rates of 0.1–0.3 are used in smaller-scale training runs.

**Max-norm regularisation.** Complement to dropout: clip the L2 norm of each neuron's incoming weight vector to a maximum $c$ (typical: $c = 3$–$4$). Prevents weights from exploding when activations are randomly dropped.

$$\|w_i\|_2 \leq c$$

#### Dropout Variants

**MC Dropout** (Gal & Ghahramani 2016): keep dropout active at inference time; run $K$ forward passes; treat the variance across outputs as an uncertainty estimate. Turns any dropout-trained network into a Bayesian approximation — cheap uncertainty quantification without separate ensemble models.

**DropConnect** (Wan et al. 2013): instead of dropping activations, randomly drop individual *weights* (connections). Finer-grained than dropout; not widely used in practice due to implementation complexity.

**Spatial Dropout** (for CNNs): instead of dropping individual activations, drop entire feature maps (channels). Preserves spatial coherence — if two adjacent pixels always co-occur in a feature map, standard dropout won't break their co-adaptation because they're rarely simultaneously dropped.

**Stochastic Depth / DropPath** ([Huang et al. 2016](https://arxiv.org/abs/1603.09382)): randomly skip entire residual blocks during training. Each layer survives with probability $p_l$, linearly decreasing from 1 at the first layer to $p_L$ at the last. At inference, scale each block's output by its survival probability.

$$h_l = \begin{cases} \text{identity}(x_l) & \text{with prob } 1 - p_l \\ x_l + F_l(x_l) & \text{with prob } p_l \end{cases}$$

Widely used in modern ViTs and ConvNeXt. Provides a strong regularisation effect and also speeds up training (skipped blocks cost no compute).

**Variational Dropout** ([Kingma et al. 2015](https://arxiv.org/abs/1506.02557)): learns the optimal drop rate per weight from data. Interprets dropout as a Gaussian multiplicative noise approximation; the per-weight drop rate becomes a learnable parameter. Can drive many weights to near-zero (effective pruning).

#### Regularisation Comparison

| Technique | Where applied | Main mechanism | When to prefer |
|---|---|---|---|
| **L2 / weight decay** | Parameters | Shrinks large weights | Almost always; use with AdamW |
| **Dropout** | Activations | Random unit removal | Small-to-medium datasets, MLPs |
| **Stochastic depth** | Residual blocks | Random layer skip | ViTs, deep ResNets |
| **Spatial dropout** | CNN feature maps | Channel-level drop | CNNs with spatially correlated features |
| **MC Dropout** | Activations (at inference) | Uncertainty estimation | When you need cheap uncertainty estimates |
| **Label smoothing** | Loss targets | Soft targets | Nearly all classification; prevents overconfidence |
| **Max-norm** | Weight norms | Clips weight magnitude | With high dropout rates |

> **Interview question:** A transformer model trained on a small dataset (50K examples) is overfitting badly. You try dropout 0.5 on attention and residual layers and training loss diverges. What went wrong and how do you fix it?
>
> *Dropout 0.5 is far too aggressive for transformer layers, where pre-norm residual connections depend on stable activation scaling. With 50% dropout on residual layers, approximately 75% of the residual stream is zeroed at any step (dropout on both branches independently). Training diverges because the effective learning signal becomes extremely noisy. Fix: (1) Lower dropout to 0.1 on residual and 0.0–0.05 on attention. (2) Add stochastic depth (DropPath) with survival rate ~0.9 — it's a more stable regulariser for transformers. (3) Add weight decay (1e-2 to 1e-1) via AdamW. (4) Use label smoothing $\epsilon = 0.1$. (5) If the dataset is genuinely tiny, consider data augmentation or synthetic data before throwing regularisation at the problem — regularisation reduces overfitting but doesn't add information.*

---

## Optimiser & Training Stability
{: #optimiser}

### AdamW
{: #adam}

LLM pretraining universally uses **[AdamW](https://arxiv.org/abs/1412.6980)** — Adam with decoupled weight decay. For each parameter `θ`:

```
m = β₁·m + (1−β₁)·g          # first moment: exponentially weighted gradient mean
v = β₂·v + (1−β₂)·g²         # second moment: exponentially weighted gradient variance
m̂ = m / (1−β₁ᵗ)             # bias correction (early steps: m is biased toward 0)
v̂ = v / (1−β₂ᵗ)
θ = θ − η · m̂/(√v̂ + ε) − η·λ·θ   # parameter update + decoupled weight decay
```

Typical hyperparameters: `β₁=0.9, β₂=0.95, ε=1e-8, λ=0.1`.

**Why `β₂=0.95` and not Adam's default 0.999?** The second moment `v` is an exponentially weighted average of squared gradients. With `β₂=0.999`, a single large gradient spike takes ~1000 steps to decay — the effective step size `m̂/√v̂` stays suppressed long after the spike. LLM training has highly non-stationary gradients (loss spikes, phase transitions as capabilities emerge). `β₂=0.95` makes `v` forget old gradient information 20× faster — the effective step size adapts within ~20 steps of a gradient change. The cost: slightly noisier step size, which is acceptable given the noise already present in mini-batch gradients.

**Why decoupled weight decay?** Original Adam applies weight decay as L2 regularisation in the gradient: `g = g + λθ`. This means weight decay is divided by `√v̂` before application — parameters with large gradient variance get less regularisation. Decoupled weight decay applies it directly: `θ = θ − η·λ·θ` after the Adam update. Every parameter decays at the same rate regardless of its gradient history. This is the correct implementation of L2 regularisation under adaptive optimisers.

**Memory cost.** AdamW stores `m` and `v` in fp32, adding 2× the parameter memory to optimizer state. For a 70B model: 70B × 16 bytes (weights fp16 + optimizer fp32) = well over 1TB. This is why ZeRO Stage 2/3 partitioning optimizer state across GPUs is essential — no single GPU can hold it.

> **Interview question:** You're training a 7B model and notice that some attention heads have extremely small gradients throughout training — effectively dead. What would cause this, and is it a problem?
>
> *Dead attention heads usually arise from attention collapse: a head learns to always attend to one token (e.g. the first token, often [BOS]) regardless of context. Once collapsed, the softmax output is near-one-hot, gradients through the softmax are near-zero, and the head stops learning. It's a local minimum. Is it a problem? Empirically, large models can afford dead heads — they have redundancy. But it's a waste of parameters. Causes: too-high learning rate early in training, insufficient normalisation before attention, poor initialisation. Fix: (1) lower initial learning rate; (2) head re-initialisation strategies; (3) attention entropy regularisation loss that penalises near-zero entropy attention patterns.*

### Learning Rate Schedule
{: #lr-schedule}

LLM training uses a **warmup + cosine decay** schedule:

<div class="post-flow" role="group" aria-label="Learning rate schedule phases">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Linear warmup — ramp from 0 to η_max over ~2000 steps</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Cosine decay — anneal from η_max to η_min over remaining training steps</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Final LR: η_min ≈ η_max / 10 (e.g. 3e-4 → 3e-5)</span></li>
  </ol>
</div>

**Why warmup?** At step 0, the optimizer's moment estimates `m` and `v` are both zero. Bias correction `m̂ = m/(1−β₁ᵗ)` partially compensates, but the variance estimate `v` is still noisy after only a few gradient samples — one large gradient spike causes `v̂` to be large, suppressing the update. More fundamentally, the model weights are random and the loss surface is steep and poorly conditioned — large updates at this stage can overshoot into regions from which recovery is slow. Warmup gradually increases the step size as the optimizer accumulates reliable gradient statistics.

**Why cosine and not linear decay?** Linear decay applies equal LR reduction per step throughout training. Cosine decay applies slow reduction early (when the model is still making large parameter changes), then accelerates reduction later (when the model is fine-tuning in a local basin). This matches the geometry of training loss: early steps are rapid descent, late steps are slow refinement.

**Multi-stage schedules** (MiniCPM, LLaMA-3): train most steps at high LR, then apply a rapid cosine cooldown on a small, high-quality subset of data. The intuition: the final cooldown phase locks in the "clean" data distribution — the model forgets the noisiest patterns from the bulk training and refines on curated text. This improves final quality without increasing total compute.

> **Interview question:** You observe that training loss stops decreasing after step 100k, but validation loss continues to decrease slowly. What's happening, and should you adjust your schedule?
>
> *Training loss plateau while validation loss decreases is a sign of overfitting on the training batch distribution — the model has memorised the most-seen patterns and stopped generalising. But if validation loss is still decreasing, the model isn't stuck; it's just getting small incremental improvements on held-out data. Action: check gradient norms — if they're declining, the model may need a higher LR or schedule refresh. Consider data shuffling to ensure the model sees new orderings. If you're near compute budget, the right call is often to decrease LR (trigger the cosine tail) to lock in the current solution. If you're mid-training, a cycle restart (increase LR briefly then decay) can escape local optima.*

### Gradient Clipping
{: #grad-clipping}

Gradient explosions in deep transformers can destabilise training irreversibly. **Global gradient norm clipping** rescales the entire gradient vector if its L2 norm exceeds a threshold:

```
g_norm = ‖g‖₂    # L2 norm across all parameters
if g_norm > max_norm:
    g = g × max_norm / g_norm
```

`max_norm = 1.0` is the standard. This is computed across all parameters jointly — clipping per-layer is less effective because it distorts the relative gradient magnitudes between layers. If layer 3 has 10× larger gradients than layer 10, that ratio is signal — clamping each layer separately destroys it. Global clipping preserves the relative structure while bounding the total update magnitude.

**Why do spikes happen despite clipping?** Loss spikes arise from a sudden large gradient, but clipping limits the *magnitude* not the *direction* of the gradient. A gradient pointing in a suddenly different direction (after a batch of unusual examples) is clipped in magnitude but still moves the model in the new direction. If the next batch reinforces this, the loss spikes. Recovery happens as gradient directions return to the running average direction encoded in the moment estimate `m`.

The common response to loss spikes: roll back to the checkpoint before the spike and resume with a temporarily reduced LR (0.5–0.9× the current value). Resume with the same or next checkpoint after 2000–5000 steps.

> **Interview question:** Why is global norm clipping preferred over per-parameter clipping or per-layer clipping?
>
> *Per-parameter clipping treats each parameter's gradient independently — it caps the update for each weight individually. This ignores the correlation structure: if an entire attention head has large gradients because it's learning an important new pattern, per-parameter clipping clips each weight in that head but doesn't reduce the aggregate update magnitude proportionally. The head's learning is distorted — some weights get clipped, others don't, breaking the consistent gradient signal the head needs to update coherently. Per-layer clipping is better but still distorts inter-layer relationships. Global norm clipping scales the entire gradient vector uniformly, preserving all relative magnitudes and ensuring the total update step size is bounded.*

### Batch Size Selection
{: #batch-size}

Batch size is one of the most consequential hyperparameters in pretraining, and the right choice depends on your compute and memory constraints.

**The gradient noise perspective.** With batch size B, the gradient is an average of B individual sample gradients. The variance of this estimate is `σ²/B` where `σ²` is the per-sample gradient variance. Larger B → lower gradient noise → more accurate gradient direction → can take larger LR steps without diverging. But: larger B → more tokens per step → fewer steps per epoch → fewer parameter updates for the same compute budget.

**The critical batch size.** Empirically, gradient noise follows a power law with dataset size. There exists a **critical batch size** `B_crit ≈ σ²/‖g̃‖²` (noise scale ÷ gradient signal) below which halving B roughly doubles the number of effective gradient steps (linear scaling of progress with compute); above it, doubling B adds little new information — you're averaging redundant gradients. For GPT-3, `B_crit ≈ 1–4M tokens` per step. Most frontier runs operate near this.

**Memory vs throughput trade-off.** Every GPU has fixed HBM. With B tokens per step and sequence length L:

- Activations: O(B · L · d) — dominate at large B × L
- KV cache during training: O(B · L · n_heads · d_head) per layer
- Each doubling of B doubles activation memory, requiring either gradient checkpointing or tensor parallelism

**Practical choices:**

| Regime | Batch size | Why |
|--------|-----------|-----|
| Early training (high loss, steep gradient) | Smaller batch (128k–512k tokens) | Gradients are large and informative — no need to average many samples |
| Late training (near convergence) | Larger batch (1M–4M tokens) | Loss landscape is flat; averaging more samples reduces noise, enabling small but precise steps |
| Memory-constrained | Gradient accumulation over micro-batches | Achieve large effective batch without fitting it in memory at once |

**Gradient accumulation.** If your GPU can only fit 4K tokens per step but you want an effective batch of 4M tokens, run 1000 micro-batch steps of 4K tokens each, accumulating gradients (not zeroing them), then apply one parameter update. The result is identical to a single step with 4M tokens — at the cost of 1000× more forward-backward passes per parameter update.

> **Interview question:** Consider no limit on compute or memory. How would this change your batch size selection?
>
> *With unlimited compute and memory, you can make B arbitrarily large. But there's a ceiling on usefulness: once B >> B_crit, each additional sample in the batch is nearly redundant — you're averaging gradients that all point in the same direction, so doubling B halves gradient noise but the noise was already negligible. The gradient direction is essentially the true gradient. At that point, more samples don't improve the update quality; they just cost more compute for the same parameter movement. The optimal strategy is to pick B ≈ B_crit, use the freed compute to train on more data instead (more steps), or increase model size. With unlimited resources, the binding constraint shifts entirely to the total token budget and the Chinchilla compute-optimal frontier — you'd scale N and D equally rather than pushing B further.*
>
> *A second effect: very large batches change the loss landscape dynamics. Small batches introduce noise that acts as implicit regularisation (escaping sharp local minima). Perfectly clean gradients at huge B can converge to sharper minima that generalise slightly worse. Some researchers find that adding a small amount of gradient noise back (noise injection) at huge batch sizes recovers this regularisation effect.*

---

## Scaling Laws
{: #scaling-laws}

[Scaling laws](https://arxiv.org/abs/2001.08361) (Kaplan et al. 2020; Hoffmann et al. 2022 — "[Chinchilla](https://arxiv.org/abs/2203.15556)") describe how model performance varies with compute budget, model size, and training tokens. The key empirical finding from Chinchilla:

> **For a fixed compute budget `C` (in FLOPs), the optimal model size `N` and training tokens `D` satisfy `N ∝ D` — model size and token count should scale equally.**

Training cost is approximately `C ≈ 6ND` FLOPs. Chinchilla showed that GPT-3 (175B params, 300B tokens) was significantly *undertrained* — a 70B model trained on 1.4T tokens achieves comparable loss at 4× less inference cost per token.

**Compute-optimal frontier:**

| Model | Params | Tokens | Chinchilla-optimal? |
|---|---|---|---|
| GPT-3 | 175B | 300B | No — undertrained by ~4× |
| Chinchilla | 70B | 1.4T | Yes |
| LLaMA-3.1 8B | 8B | 15T | Deliberately over-trained |
| LLaMA-3.1 405B | 405B | 15T | Over-trained (inference budget) |

**Why Chinchilla was wrong in practice.** Chinchilla minimises training loss. In deployment, inference cost dominates — you train a model once but run inference millions of times. A smaller model that's trained on more tokens achieves the same loss as the Chinchilla-optimal large model, but at a fraction of the inference compute per token. This is why LLaMA-3.1 8B on 15T tokens (nearly 2000× Chinchilla-optimal token count) is valuable — it achieves strong quality in a size that can run efficiently.

**Emergent capabilities** are not predicted by scaling laws. Average loss decreases smoothly and predictably. But individual capabilities — arithmetic, multi-step reasoning, in-context learning — emerge non-smoothly at certain scale thresholds. A model at 6B parameters may fail completely on 3-digit multiplication; at 13B it may suddenly succeed. This discontinuity is not captured by loss curves and is a major obstacle to predicting capability from small-scale experiments.

> **Interview question:** Scaling laws say that for fixed compute, you should split it evenly between model size and tokens. But LLaMA-3.1 8B is trained on 15T tokens (far more than Chinchilla-optimal). Is this a mistake?
>
> *Not a mistake — it's a deliberate choice driven by inference cost. Chinchilla optimises for minimal loss at the end of training, treating the training compute as the scarce resource. But in production, you care about loss per inference FLOP over the lifetime of the model. If you serve a model 10 billion times, a model that's 2× cheaper per forward pass saves enormously more compute than the compute spent extra-training it. A smaller model trained longer (more tokens) can match the loss of a larger model trained Chinchilla-optimally. The smaller model is then cheaper at every subsequent inference step. The right framing is: given a deployment compute budget (train + serve), not just a training budget, what N and D maximise quality per total FLOP? The answer shifts substantially toward smaller N and larger D than Chinchilla suggests.*

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

**Data parallelism + ZeRO**: replicate the model across N GPUs, each processing a different mini-batch. Standard data parallelism replicates all parameters — wasteful. ZeRO Stage 3 shards parameters, gradients, and optimizer state across all replicas. Memory per GPU drops from `20M` to `20M/N` bytes. AllReduce communication cost is `2M` bytes regardless of N with ring AllReduce (each GPU sends and receives once).

**Tensor parallelism (Megatron-LM)**: split each weight matrix across GPUs within a node, exploiting fast NVLink (up to 112 GB/s vs PCIe's 32 GB/s). Each GPU computes a column slice of each linear layer; one AllReduce per transformer sub-layer synchronises partial results. Works for up to 8–16 GPUs per node — beyond that, NVLink bandwidth is exhausted.

**Pipeline parallelism**: assign consecutive transformer layers to different nodes. Micro-batching and 1F1B (one forward, one backward) scheduling keep the pipeline busy — bubble fraction `(p−1)/m` where `p` is pipeline stages and `m` is micro-batches. Communication is just the activation tensor between adjacent stages (much smaller than an AllReduce).

**Fault tolerance**: at 10,000-GPU scale, hardware failures during a multi-week run are near-certain. Checkpointing every 1–4 hours to distributed storage allows recovery. Asynchronous checkpointing (write to CPU memory, flush to storage in background) avoids GPU idle time. Lost training steps from a failed run cost approximately `checkpoint_interval / training_duration` of total compute.

**Numerical stability**: bf16 is the default for LLM pretraining — same exponent range as fp32 avoids overflow without the manual loss scaling required by fp16. Master weights and optimizer state stay in fp32 (necessary for small gradient magnitudes to not underflow). Loss spikes are monitored with automated alerting; anomalous gradient norms trigger checkpoint rollback.

> **Interview question:** You're scaling from 256 to 4096 GPUs. Your data parallel AllReduce is now the bottleneck. What options do you have?
>
> *AllReduce cost with ring topology is 2M bytes independent of N (number of GPUs) — so going from 256 to 4096 GPUs doesn't increase AllReduce bytes. But it does increase the number of simultaneous communications, and the ring becomes longer, increasing latency proportionally to N. Options: (1) ZeRO — partition optimizer state and gradients across GPUs, replacing AllReduce with smaller Reduce-Scatter + AllGather; (2) Hierarchical AllReduce — perform AllReduce within a node (fast NVLink) then across nodes (slow IB), reducing cross-node traffic by N_per_node×; (3) Increase pipeline parallelism — fewer data-parallel replicas means smaller AllReduce; (4) Gradient compression — quantise gradients before communication at the cost of some training noise. In practice, most large runs use ZeRO + hierarchical comms + tensor parallelism within nodes.*

---

## Tokens/Day: The End-to-End Metric
{: #tokens-day}

The roofline model (MFU, arithmetic intensity) answers "how efficiently am I using the GPU while it is running?" But production pre-training has a second, outer loop: how much of wall-clock time is the GPU actually running, and what fraction of tokens consumed lead to useful learning? The correct north star is:

```
tokens/day (cluster) = G · goodput · (MFU · C_peak / F_tok) · 86400
```

where `G` is GPU count, `goodput ∈ [0,1]` is the fraction of wall-clock time doing useful work, `C_peak` is peak FLOPs/s, and `F_tok` is FLOPs per token. Using the dense-transformer rule of thumb `F_tok ≈ 6N_params` (forward ≈ 2N, backward ≈ 4N):

```
tokens/sec/GPU ≈ MFU · C_peak / (6 · N_params)
```

**Why the two-factor structure matters.** MFU and goodput are independent levers with different root causes. A run with 55% MFU and 90% goodput gives the same tokens/day as one with 99% MFU and 50% goodput — but they fail in completely different ways and require completely different fixes. Always decompose before acting.

| Factor | What it measures | What degrades it |
|---|---|---|
| MFU | Compute efficiency while running | Bad kernel shapes, poor operator fusion, comm not overlapped |
| Goodput | Fraction of time actually training | Failures/restarts, checkpoint I/O, dataloader stalls, eval overhead |

**Compute-optimal planning.** For fixed budget C FLOPs: `C ≈ 6 N_params · N_tok`, so `N_tok ≤ C / (6N_params)`. Making the model bigger while holding compute fixed means fewer tokens — training is a joint optimisation over architecture, data, and hardware.

### MFU vs Goodput
{: #mfu-goodput}

**MFU** (Model FLOPs Utilisation) measures how fast tokens are processed *while the GPU is computing*:

```
MFU = achieved_model_FLOPs/s / C_peak
```

MFU is a symptom, not a diagnosis. A low MFU could mean microbatch too small (low arithmetic intensity on weight GEMMs), attention kernels not FlashAttention-fused, poor pipeline bubble hiding, or long-tail ops (layernorm, softmax) serialising the timeline. Fixing MFU requires kernel profiling at the operator level.

**Goodput** measures how often the GPU is doing useful work at all:

```
goodput = useful_training_time / wall_clock_time
```

Sources of non-useful time: dataloader stalls (I/O or CPU tokenisation bottleneck), non-overlapped communication (stragglers holding up a barrier), checkpoint writes and eval passes, hardware failures and restarts, scheduler preemption. Fixing goodput requires run-level instrumentation — step time histograms, failure logs, checkpoint overhead measurements.

**Mental model.** MFU = how fast you drive. Goodput = how often you are driving. A 60% MFU / 95% goodput run beats a 90% MFU / 60% goodput run by nearly 2× in tokens/day.

> **Interview question:** Your tokens/day dropped 30% overnight. Nothing in the model code changed. How do you triage?
>
> *First decompose: `tokens/day ≈ throughput_run × goodput`. (1) Did throughput while running drop? → MFU regression: check whether a kernel shape changed (batch size, sequence length), whether a new op was introduced without fusion, whether comm overlap broke. (2) Did goodput drop? → check failure rate logs, checkpoint overhead, eval cadence, dataloader wait time. (3) Did neither drop but validation loss degraded? → silent data corruption. A data pipeline regression (new shard with junk, broken dedup) can poison training without affecting throughput. This is the hardest failure mode because it only shows up in model quality days or weeks later. Prevention: per-source PPL tracking and automatic data health dashboards.*

### Step-Time Decomposition
{: #step-timeline}

Every training step has a budget:

```
T_step ≈ T_data + T_fwd + T_bwd + T_opt + T_comm_non_overlapped + T_overhead
```

A hierarchy of observability matches causes to metrics:

| Level | Signals | Typical tools |
|---|---|---|
| Kernel | SM occupancy, mem BW, launch overhead | NVIDIA Nsight, PyTorch profiler |
| Operator | GEMM, attention, layernorm, comm times | torch.profiler traces |
| Step | Step time, tokens/sec, comm overlap fraction | training logger |
| Run | Goodput, MTBF, tokens/day, val loss | W&B / MLflow dashboards |

**Why overlap matters.** In a naïve pipeline, the GPU computes the backward pass, then waits for the AllReduce of gradients before the next forward. With gradient bucketing + async AllReduce, the gradient communication for early layers starts while the backward pass is still computing later layers — effectively hiding the AllReduce behind computation. This is the default in PyTorch DDP (25–100 MiB buckets) and why bucket size is a tunable knob.

**Sequence packing** turns padding into throughput. When documents are shorter than the context length L, naïve batching wastes compute on padding tokens. Packing concatenates multiple documents into one sequence (separated by `<EOS>`) with attention masks preventing cross-document leakage. Effect: higher arithmetic intensity, more uniform tensor shapes, fewer "wasted" tokens per step. Critical: if cross-document attention is not masked, the model learns spurious dependencies across document boundaries — a subtle training distribution bug.

> **Interview question:** Your step time profile shows 40% of each step is dataloader wait. Your GPU compute is sitting idle. What do you check first?
>
> *Dataloader stall means data is not ready when the GPU finishes the previous step. Root causes: (1) Preprocessing on CPU is slower than GPU compute — solution: increase dataloader workers, move tokenisation offline, prefetch more batches. (2) Storage I/O is the bottleneck — solution: copy dataset to local NVMe SSD, use memory-mapped files, switch to streaming-friendly formats (WebDataset, MosaicML StreamingDataset). (3) Data augmentation on the fly (e.g. packing, random masking) is CPU-bound — move to GPU or precompute. Quick check: `num_workers=0` vs `num_workers=8` step time comparison isolates whether CPU preprocessing or I/O is the culprit.*

### Mixed Precision In Depth
{: #mixed-precision-depth}

Tensor cores operate at peak throughput only at reduced precision. The standard recipe has evolved through three generations:

<div class="post-flow post-flow--compare" role="group" aria-label="Mixed precision generations">
  <div class="post-flow__col">
    <p class="post-flow__col-label">fp16 (Volta/Ampere)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">5-bit exponent — narrow dynamic range</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Gradient overflow common → requires dynamic loss scaling</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Master weights in fp32; update in fp16</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">bf16 (Ampere+) ✓ default</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">8-bit exponent — same range as fp32</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Overflow essentially eliminated; no loss scaling needed</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">10-bit mantissa (vs fp16's 10, fp32's 23) — lower precision but stable</span></li>
    </ol>
  </div>
</div>

**Why bf16 dominates.** Overflow was the main failure mode of fp16 — gradient values for large models routinely exceed fp16's max (~65504), producing NaN. bf16 shares fp32's 8-bit exponent, so its max is ~3.4×10³⁸ — overflow almost never occurs. The mantissa is only 7 bits (vs fp16's 10), giving lower precision per operation, but empirically this loss is negligible for LLM training because gradient noise already dominates numerical precision.

**FP8 on Hopper (H100).** Hopper Tensor Cores support native FP8 acceleration via NVIDIA's Transformer Engine (TE). Two formats exist:
- **E4M3** (4-bit exponent, 3-bit mantissa): highest precision, narrow range — used in the **forward pass** where activations are well-behaved.
- **E5M2** (5-bit exponent, 2-bit mantissa): wider range, less precision — used in the **backward pass** where gradients have higher dynamic range.

The default TE recipe is **HYBRID: E4M3 forward, E5M2 backward**. Per-tensor scaling: each tensor gets its own `scale = amax_history / FP8_max`, computed from a rolling maximum of recent absolute values (delayed scaling). At scale, TE reduces `amax` across a process group so all ranks use identical FP8 scales — prevents rank-to-rank precision drift.

**MXFP8 on Blackwell.** Classic FP8 uses one scale per tensor, which sometimes forces E5M2 even in the forward pass when a tensor has high dynamic range. MXFP8 assigns a separate scale to each **block of 32 values**, reducing dynamic-range stress so E4M3 can be used more broadly. Blackwell-era pre-training increasingly targets MXFP8 for better accuracy at the same throughput.

**NVFP4 (4-bit).** Blackwell 5th-gen Tensor Cores implement NVFP4 with micro-block scaling: groups of 16 FP4 values share a scale factor (tighter than 32-wide MXFP8 blocks), improving outlier robustness. Fully-NVFP4 pre-training is an active frontier — viable with careful recipe design but requires ablations against bf16 baselines.

**Key monitoring signals for any precision regime:** overflow/underflow counters per layer, NaN/Inf flags on activations and gradients, grad norm spikes correlated with precision changes, loss scaling factor (fp16 only — a declining scaler is an early warning of instability).

> **Interview question:** You switch from bf16 to FP8 training and see validation loss is 0.15 nats worse after 50B tokens. How do you debug whether it is a precision problem or a data/schedule problem?
>
> *Isolate precision as the variable: run a controlled ablation — identical config, identical data order, bf16 vs FP8, both stopped at 50B tokens. If the gap persists, it is precision. Then localise: (1) inspect TE's per-tensor amax histories — if any tensors are frequently saturating their scale, the quantisation error is high for that tensor; try per-layer FP8 disable. (2) Check if the problem is in the forward pass (E4M3) or backward (E5M2) — swap E5M2→E4M3 in the forward pass for a few steps and monitor grad norms. (3) Check if delayed scaling is stable — if amax history is noisy, the scale lags reality; reduce amax history window. If the ablation shows no gap, the issue is data or schedule, not precision.*

### Training Stability Dashboard
{: #stability-dashboard}

Training can fail in three qualitatively different ways, each requiring different instrumentation:

| Failure mode | Signature | Root cause examples |
|---|---|---|
| Hard failure | Job crashes (OOM, NCCL timeout, node death) | GPU failure, bad kernel shape, communication hang |
| Divergence | Loss increases and does not recover; grad norm explodes; NaNs appear | LR too high, precision overflow, bad batch statistics |
| Silent corruption | Model trains "normally" but val loss or downstream quality degrades | Data mixture bug, broken masking/packing, label shift, optimizer state desync |

**Silent corruption is the costliest failure mode** — it consumes millions of GPU-hours before detection, and the root cause may be hundreds of steps in the past. Prevention requires *invariants*: fixed validation sets checked frequently, per-source loss tracking, and data health metrics that run continuously.

**Minimal stability dashboard** — always log these:

*Optimization health:* training loss (smoothed + raw), validation loss / PPL on a fixed eval set, learning rate schedule, global gradient norm, per-tensor-group gradient norms (attention vs MLP), optimizer step magnitude `‖Δθ‖`.

*Numerics:* NaN/Inf flags on activations and gradients every step, overflow/underflow counters, grad scaler value (fp16 only).

*Data health:* token distribution stats (length, language ID), exact + near-dedup rate, document source mix fraction, top n-gram repetition indicators.

*Systems context:* step time, tokens/sec, dataloader wait, failure/restart events, checkpoint overhead.

**Divergence vs data spike — distinguishing signatures:**

- *Divergence*: loss increases monotonically, grad norms explode, NaNs appear, often correlated with LR changes or precision mode changes. Fix: lower LR, adjust warmup, enable grad clipping, switch to bf16, check optimizer hyperparams. Roll back to last good checkpoint.
- *Data spike*: isolated loss spikes that self-recover within 10–100 steps, often repeat at shard boundaries or when a particular data source enters the sampling pool. Fix: investigate offending documents, add quality filters, improve dedup, validate packing/masking boundaries.

**Gradient norm monitoring as the primary canary.**

```
‖g‖₂ = ‖∇_θ L‖₂    (global L2 norm across all parameters)
```

A stable run keeps grad norms in a characteristic range for its recipe. Watch for: sudden spikes (potential divergence), sudden drops to 0 (overflow/underflow or communication bug — gradients were silently zeroed), slow drift over long windows (recipe mismatch or data distribution shift). Log globally *and* per-layer to localize.

**Checkpointing as a stability component.** A checkpoint must contain: model weights θ, optimizer state (Adam m, v), RNG states (dropout, data order), scheduler state, and dataloader position or reproducible sampling seed. Missing any of these causes subtle divergence on resume — the scheduler jumps, the data order repeats, or dropout patterns change. Test resume explicitly before the production run.

> **Interview question:** Your 70B training run has been stable for 200B tokens. Suddenly validation loss starts increasing while training loss continues to decrease normally. What happened and what do you do?
>
> *Training loss decreasing while val loss increases is the signature of distribution shift, not divergence. The model is fitting the training distribution better but generalising worse — something changed in what it is seeing vs what it is evaluated on. Suspects, in order of likelihood: (1) data mixture shifted — a new data source entered rotation, the sampling weights changed, or a filter threshold drifted; check source mix logs and acceptance rates. (2) Benchmark contamination — eval data leaked into training; check dedup against eval set. (3) Label bug — masking or packing changed so the model is now being trained on tokens it should not predict. (4) Eval bug — eval code or checkpoint loading changed. Action: freeze data pipeline at current state, verify eval code, diff against last known-good config. If the issue is data, roll back the data mix (not the model weights) and monitor val loss recovery.*

### Operational KPIs & MTBF
{: #operational-kpis}

A multi-week pre-training run needs a scoreboard — a small set of numbers that summarise end-to-end health, decompose into actionable components, and allow day-over-day comparison.

| KPI | Definition | Why it matters |
|---|---|---|
| Throughput (TFLOPs/s/GPU) | Achieved compute rate while running | Kernel/operator efficiency |
| Tokens/day | Cluster-level training progress | End-to-end velocity |
| Goodput (%) | Useful training time / wall-clock time | Failure + overhead budget |
| Failures/day | Crash / preemption / timeout count | Infrastructure health |
| MTBF (GPU-hours) | `(G × wall-clock_hrs) / n_failures` | Scale-normalised resilience |
| Researcher interventions | Manual actions: restarts, param tweaks | True engineering cost |

**MTBF in GPU-hours matters because naive per-job MTBF hides scale.** On a 6,000-GPU run, a 1-in-10,000-GPU-hour failure rate means a job failure every `10,000/6,000 ≈ 1.7 hours`. At 1,000 GPUs the same hardware failure rate means a failure every 10 hours. GPU-hour normalisation makes the metric comparable across cluster sizes.

**Checkpoint interval from MTBF.** If job-level failures arrive every τ hours and you checkpoint every Δ hours:

```
expected rollback per failure ≈ Δ/2
expected wasted time per hour ≈ (1/τ) · (Δ/2)
```

Frequent checkpoints (small Δ) reduce rollback but increase checkpoint I/O overhead (goodput cost). Infrequent checkpoints (large Δ) maximise goodput but lose more work per failure. The optimum Δ balances these two costs — as a rule of thumb, set Δ so that checkpoint overhead is 1–5% of wall-clock. Use async checkpointing (write to CPU memory while GPU continues, flush to storage in background) to reduce GPU stall.

**KPI triage matrix** — when tokens/day drops, ask:

<div class="post-flow post-flow--compare" role="group" aria-label="KPI triage">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Throughput dropped, goodput stable</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Kernel shape changed (batch, seq len)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Comm overlap regression</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">New op without fusion</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Goodput dropped, throughput stable</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">More failures/restarts</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Checkpoint/eval overhead up</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Cluster preemption/scheduling</span></li>
    </ol>
  </div>
</div>

The worst case: tokens/day is stable but validation loss degrades. This is silent corruption — data mixture bug, leakage, or broken masking. The scoreboard **must** include model quality metrics alongside systems metrics; optimising one without the other produces garbage efficiently.

> **Interview question:** Your team hits 55% MFU on a 1,000-GPU run. Someone proposes spending two engineer-weeks to push MFU to 65%. Is this worth it?
>
> *Depends on where goodput sits. If goodput is 70% (frequent failures, slow checkpointing), the tokens/day is 0.55 × 0.70 = 38.5% of theoretical. Fixing goodput to 90% would yield 0.55 × 0.90 = 49.5% — a 28% improvement. Pushing MFU from 55% to 65% at 70% goodput gives 0.65 × 0.70 = 45.5% — a 18% improvement. So in this regime, goodput is the higher-leverage lever. On the other hand, if goodput is already 95%, the MFU improvement from 55% to 65% yields 0.65 × 0.95 = 61.8% vs 0.55 × 0.95 = 52.3% — a 18% improvement that is now the main remaining lever. The answer: measure both, identify which is the binding constraint, fix that first.*

### Corpus Engineering
{: #corpus-engineering}

Data curation quality is often as important as architecture changes for frontier models. A pre-training corpus is a pipeline — evaluate it with the same rigour as production code.

**Eight questions to audit any corpus:**

1. **Sources** — what raw inputs are included (web snapshots, books, code)?
2. **Extraction** — how is text extracted (HTML boilerplate removal, DOM heuristics)?
3. **Language ID** — how are languages detected and segmented (document-level + segment-level)?
4. **Quality filters** — heuristic rules, neural classifiers, perplexity filters?
5. **Dedup** — exact vs near-dup; within-source vs cross-source; at what granularity?
6. **Safety** — toxicity/PII filtering, policy constraints, allowlists/denylists?
7. **Mixing** — how are components weighted / sampled over time (static vs dynamic)?
8. **Provenance** — can you trace tokens back to sources for audits and retroactive removal?

**Common Crawl: gold and garbage.** CC is massive and diverse but requires engineering at every stage. HTML extraction (boilerplate removal, link density heuristics) must precede language ID. Language ID fails on short documents, code-heavy pages, and transliterated text. Quality filtering needs both heuristic rules (length thresholds, punctuation ratios, stopword density) and a neural classifier to catch subtly low-quality text.

**C4 vs [FineWeb](https://arxiv.org/abs/2406.17557) — the evolution of web curation:**

| Axis | C4 (T5-era) | FineWeb / FineWeb-Edu |
|---|---|---|
| Raw source | Single CC snapshot, English | 96 CC snapshots (2013+) |
| Selection philosophy | Heuristics + blocklists | Documented ablations; learned educational classifier |
| Optimised for | Clean enough + reproducible baselines | Quality-first at scale + targeted subsets |
| Known gotchas | Blocklists remove minority text; contamination risk | Scorer drift; subsets are objectives disguised as datasets |
| Best used for | Ablations, "web-cleaning 101" lessons | Strong open web component in modern mixtures |

**Deduplication is the highest-leverage step.** Duplicates increase memorisation risk, inflate effective data diversity less than raw count suggests, and cause loss spikes / training artifacts. Pipeline: (1) convert documents to shingles (5-grams), (2) compute MinHash signatures, (3) group near-duplicates via LSH (locality-sensitive hashing), (4) keep one representative per cluster. Engineering trade: aggressive dedup improves diversity but removes legitimate repeated structures (legal boilerplate, code templates) — tune thresholds by domain.

**Dedup must also protect evaluation integrity.** If training data contains near-duplicates of eval benchmarks, results are inflated — contamination. Mitigation: build a benchmark "do-not-train" set, dedup at the n-gram level against it, and track provenance so sources can be removed retroactively.

**Quality classifier design choices:**

- *Hard filtering* (discard below threshold) vs *soft weighting* (downsample proportional to score). Soft weighting yields smoother distributions and avoids catastrophic removal of rare but valuable text.
- *Perplexity filtering* using a reference LM as quality signal: high-PPL documents are gibberish or boilerplate — but also code, math, and low-resource languages. Use domain-specific reference models to avoid removing valuable technical content.
- *Acceptance rate* is the operational metric to monitor: large shifts usually indicate upstream extraction or filtering bugs, not genuine changes in data quality.

**Mixing weights as training hyperparameters.** Sample source `i` with probability `w_i`, tuned like any other hyperparameter. To prevent large sources from dominating, temperature sampling uses `p_i^α` for `α ∈ (0, 1)` — flattening the distribution toward uniform. Mixing weights should be monitored over time; silent drift due to source volume changes is a common silent corruption vector.

**Data curation KPIs to track continuously:**

| Category | Metrics |
|---|---|
| Coverage / diversity | Unique document count, domain distribution, language distribution |
| Quality / safety | Classifier score distributions, toxicity rate, PII hit rate, perplexity distribution by domain |
| Duplication | Exact dup rate, near-dup cluster size distribution, n-gram overlap with eval sets |
| Pipeline health | Acceptance rate drift, language proportion over time, source mix fraction |

> **Interview question:** You are building a pre-training corpus and your deduplication step removes 60% of the data. Someone argues that you are throwing away training signal. How do you respond, and how would you determine the right dedup threshold?
>
> *60% removal sounds aggressive but is typical for raw web crawls — the web is full of mirrors, quote farms, and template repetition. The question is whether the removed 60% is redundant or genuinely unique. Argument: once a document appears N times, the (N+1)th copy adds negligible new gradient signal but adds N times the memorisation pressure. The model allocates capacity to pattern-matching duplicates rather than learning generalisable representations. How to find the right threshold: (1) train small ablations (e.g. 1B params, 20B tokens) at varying dedup aggressiveness and compare val loss and downstream benchmark performance; (2) track the near-dup cluster size distribution — a long tail of 100+ copies per document is a sign that dedup threshold is too lenient; (3) check memorisation rate on held-out passages — aggressive dedup should reduce verbatim recall on training data.*

### Synthetic & Instruction Data in Pre-Training
{: #synthetic-data}

Modern pre-training pipelines no longer restrict themselves to organic web text. Two categories of non-organic data have become first-class components.

**Instruction data in pre-training** (2023–2026 trend). Rather than reserving instruction-response pairs for SFT/RLHF, many stacks mix them directly into pre-training:
- Reformat raw text into tasks (Q&A, extraction, summarisation) and train with the same CLM loss.
- Mix chat-formatted text and tool traces as another corpus component.
- "Instruction pre-training" pattern: synthesise hundreds of millions of instruction-response pairs, concatenate with raw corpora, train vanilla CLM (loss on all tokens including instructions).

Effect: models acquire format-following earlier in training, improving later instruction tuning efficiency. Risk: reduces diversity if instruction text dominates; creates contamination risk if instruction prompts overlap with eval benchmarks.

**Synthetic pre-training data** ([Phi](https://arxiv.org/abs/2306.11644) lineage, reasoning models). A strong generator model creates denser-than-web content:
- "Textbook" explanations with exercises (more supervision per token, fewer ads/boilerplate).
- Reasoning problems with verifiable answers (chain-of-thought traces).
- Code tasks with unit tests.
- Translations/paraphrases to densify rare distributions.

Why it can work: synthetic text can be more intentional than web text — more coverage of target skills per token. Key failure modes: (1) feedback loops — if synthetic data dominates and is low-diversity, the model overfits to generator artifacts; (2) over-specialisation — great on "school" tasks, worse on messy real-world text.

**Three guardrails for synthetic data in production:**

1. **Version + provenance** — record generator model/version, prompts, filtering thresholds. When a model regression appears, you need to trace it back to a specific synthetic data batch.
2. **Dedup + decontam** — dedup synthetic against raw sources *and* against eval benchmarks. Synthetic generators often inadvertently reproduce training data or benchmark solutions.
3. **Mixture controls** — cap synthetic fraction; run ablations at 5%, 20%, 50% and watch for capability regressions. Monitor synthetic fraction over time; silent increases from generator output volume changes are common.

**Monitoring metrics for synthetic/instruction data:** synthetic token fraction over time, diversity (repetition / n-gram entropy), per-source loss/PPL and spike detection, acceptance rate drift from quality filters, contamination overlap rates with eval.

> **Interview question:** You add 20% synthetic "textbook" data to your pre-training mix and see MMLU improve by 3 points but HumanEval (coding) regress by 5 points. Why might this happen and how do you fix it?
>
> *The synthetic textbooks are likely knowledge-dense (facts, concepts, Q&A) but light on code. Adding 20% knowledge-focused synthetic data effectively dilutes the code fraction of the training mix — if code was 10% before, it's now 8% of a larger corpus. MMLU improves because the model sees more structured knowledge in a test-friendly format. HumanEval regresses because code signal is diluted. Fix: (1) Keep the absolute code token count constant by adding synthetic textbooks as an additive component rather than a substitution — sample from {web, code, synthetic} with weights that preserve code's absolute contribution. (2) Add synthetic code specifically — generate code tasks, unit tests, and explanations to match the textbook component. (3) Run ablations at different mix ratios to find the Pareto-optimal point for both benchmarks.*

---

## Optimisers: From SGD to Muon
{: #optimisers-deep}

The choice of optimiser is not just an academic detail — it determines how fast a model converges, how much GPU memory is consumed per parameter, and whether hyperparameters transfer predictably as you scale. This section traces the arc from SGD to the newest matrix-aware optimisers that power frontier model training.

<div class="post-flow post-flow--horizontal" role="group" aria-label="Optimiser family tree">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">SGD / Momentum</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Adam / AdamW</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Lion</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Shampoo / Muon</span></li>
  </ol>
</div>

### SGD & Momentum
{: #sgd-momentum}

Vanilla stochastic gradient descent descends along the negative gradient at each mini-batch:

> `w_{t+1} = w_t − η · (1/|B|) Σ ∇L(w_t)`

**The core tension**: the learning rate η is shared across all parameter directions. Directions with large curvature (large singular values of the Hessian) demand a small η to stay stable; directions with small curvature (small singular values) make slow progress at that same small η. Every coordinate is bottlenecked by the worst-case direction.

**Momentum** smooths this by accumulating an exponential moving average of gradients:

```
m_{t+1} = β·m_t + (1−β)·∇L(w_t)
w_{t+1} = w_t − η·m_{t+1}
```

- Damps oscillations in high-curvature directions (momentum cancels opposite gradients)
- Accelerates in low-curvature directions where gradients point consistently
- Allows using a larger η without diverging

> **Think about it**: SGD with momentum is mathematically equivalent to GD with gradient noise. The noise comes from mini-batch sampling. What does this noise imply for the *minimum* SGD converges to compared to full-batch GD?

### Adam & AdamW
{: #adam-adamw}

**The SignGD idea**: instead of a step proportional to `∇L`, take a step proportional to `sign(∇L)` — equal-sized steps per coordinate regardless of gradient magnitude. This is equivalent to minimising the linear approximation of the loss under an **ℓ∞ norm constraint** (a "box" rather than a sphere in parameter space). Every coordinate moves at the same speed; there is no bottleneck from high-magnitude directions.

Adam smooths the sign idea with momentum on both the gradient and its square:

<div class="post-flow" role="group" aria-label="Adam update steps">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">First moment:  m_{t+1} = β₁·m_t + (1−β₁)·∇L(w_t)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Second moment: v_{t+1} = β₂·v_t + (1−β₂)·(∇L(w_t))²  (element-wise)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Bias correction: m̃ = m/(1−β₁ᵗ),  ṽ = v/(1−β₂ᵗ)  — undo zero-initialisation</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Update: w_{t+1} = w_t − η · m̃ / (√ṽ + ε)</span></li>
  </ol>
</div>

The division by `√v` is the adaptive normalisation — it approximates a per-coordinate step size that scales inversely with the recent gradient magnitude, mimicking SignGD. Near an optimum, signs flip and Adam oscillates; learning-rate decay solves this.

**AdamW** fixes a subtle coupling: if weight decay is added as `λw` inside the gradient, it gets absorbed into the adaptive normalisation `√v`, entangling regularisation with the learning rate. AdamW decouples them:

> `w_{t+1} = (1 − λ')·w_t − η · m̃ / (√ṽ + ε)`

`λ'` is a separate hyperparameter. **Important**: when using WSD (warmup-stable-decay) learning-rate schedules, `λ'` should be moved together with `η` to keep the effective regularisation constant.

**Memory cost**:

| Optimiser | State per parameter | Relative memory |
|---|---|---|
| SGD | 0 | 1× |
| SGD + momentum | momentum | 2× |
| Adam / AdamW | momentum + squared gradient | 3× |

> **Think about it**: Why does bias correction matter most at the very start of training? What would happen if you initialised m and v to their long-run expected values instead?

### Lion
{: #lion}

Lion (Evolved Sign Momentum, 2023) keeps the SignGD update structure but drops the second-moment buffer entirely:

```
c_t   = sign(β₁·m_t + (1−β₁)·∇L(w_t))   # sign of interpolated gradient
w_{t+1} = (1−λ')·w_t − η·c_t              # apply sign step + weight decay
m_{t+1} = β₁·m_t + (1−β₁)·∇L(w_t)       # update momentum after step
```

<div class="post-flow post-flow--compare" role="group" aria-label="Adam vs Lion">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Adam</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Stores m and v — 3× weight memory</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Step size ∝ m / √v (adaptive magnitude)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Works well with standard weight decay coupling</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Lion</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Stores only m — 2× weight memory</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Step size = constant (sign only)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Requires decoupled weight decay (different scale than Adam)</span></li>
    </ol>
  </div>
</div>

Because Lion takes equal-magnitude steps in every direction, the effective learning rate is much more uniform than Adam. This means the **weight decay coefficient must be re-tuned** when switching from Adam — the dynamics are fundamentally different.

> **Think about it**: Lion was discovered by program search (evolutionary algorithms), not human intuition. What does this tell us about the space of possible optimiser designs? Is there a principled reason to prefer sign-based updates for large models?

### Shampoo
{: #shampoo}

Momentum and Adam both operate **coordinate-wise** — they treat each scalar parameter independently. But neural network layers are matrices: the gradient `G = ∇_W L` is a matrix, and a "small" step in the Frobenius sense may still cause large changes in the function's output for typical inputs.

**The right notion of small**: constrain the update `ΔW` under the **spectral norm** (operator 2-norm), which measures the largest possible change in activations for a unit-norm input. This leads to the steepest descent under spectral norm:

> The solution to `min_{‖ΔW‖₂ ≤ η} ⟨G, ΔW⟩_F` is `ΔW* = −η·UV⊤`

where `G = UΣV⊤` is the SVD. `UV⊤` is the matrix "orthogonalisation" of G — it preserves the directions of the gradient but discards magnitudes, exactly as SignGD discards scalar magnitudes.

**Why not just compute SVD every step?** SVD is O(mn·min(m,n)) — prohibitively expensive for large weight matrices at every iteration.

**Shampoo's approximation**: notice that `(GGᵀ)^{-1/4} G (GᵀG)^{-1/4} = UV⊤` exactly. Instead of SVD, maintain running estimates of `L_t = Σ G_i Gᵢᵀ` (left factor) and `R_t = Σ Gᵢᵀ G_i` (right factor) as EMAs, then compute matrix fourth-roots using iterative GEMM-based methods (much faster on GPU hardware than full SVD).

<div class="post-flow" role="group" aria-label="Shampoo update">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Accumulate L_t = EMA(GGᵀ),  R_t = EMA(GᵀG)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Compute preconditioner: P = L_t^{-1/4} · G · R_t^{-1/4}  ≈ UV⊤</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Update: W_{t+1} = W_t − η·P</span></li>
  </ol>
</div>

Matrix fourth-roots are updated less frequently than the weight update itself — the expensive spectral computation is amortised over many steps.

> **Think about it**: Shampoo maintains separate left and right preconditioner matrices for each weight matrix. How does this memory cost compare to Adam? For a weight matrix of shape (d_out, d_in), what does Shampoo store?

### Muon
{: #muon}

Muon (Momentum Orthogonalized by Newton–Schulz, 2024) achieves the same UV⊤ update as Shampoo but replaces the expensive matrix root computation with **a handful of cheap polynomial iterations**.

**The geometry**: Shampoo's spectral norm constraint is `‖ΔW‖₂ ≤ η`. Muon replaces this with the **RMS-to-RMS induced norm** — which controls how much the layer output RMS changes for a typical input, matching the intuition behind Xavier/He initialisation:

> `‖A‖_{RMS→RMS} = √(d_in/d_out) · ‖A‖₂`

Under this constraint the optimal update is scaled semi-orthogonalisation:

> `ΔW* = −η · √(d_out/d_in) · UV⊤`

**Newton–Schulz orthogonalisation**: for any odd polynomial `p`, `p(G) = U·p(Σ)·Vᵀ` — it commutes with the SVD. The simple cubic `p(x) = 3x/2 − x³/2` maps any `x ∈ (0, √3)` toward 1, so repeatedly applying it to a Frobenius-normalised gradient drives all singular values toward 1 — producing `UV⊤` without ever computing singular vectors explicitly.

<div class="post-flow" role="group" aria-label="Muon full algorithm">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Momentum: B_t = μ·B_{t-1} + (1−μ)·∇_W L(W_t)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Normalise: B̂_t = B_t / ‖B_t‖_F  (puts singular values in (0,1])</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Orthogonalise: apply Newton–Schulz iteration to B̂_t → O_t ≈ UV⊤</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Update: W_{t+1} = W_t − η·√(d_out/d_in)·O_t</span></li>
  </ol>
</div>

**Memory**: only parameters + one momentum buffer — **2× parameter memory**, same as SGD+momentum, better than Adam's 3×.

**Key practical notes**:

- Non-matrix parameters (embeddings, biases, norms) are handled by AdamW — the two optimisers must be run together with coupled learning rates
- Exact orthogonality is not required; 5 Newton–Schulz steps is typically sufficient
- Higher-order polynomials converge faster but cost more per step; *PolarExpress* (2025) and *Turbo-Muon* (Dec 2025) push this frontier
- The critical batch size appears much larger for Muon than Adam, making it more compute-efficient at large scale
- Communication overhead in distributed training is an active engineering challenge (see *Dion*, *Muon-BP*)

> **Think about it**: Newton–Schulz orthogonalisation applies a polynomial to a matrix. Why does normalising by the Frobenius norm before applying the polynomial guarantee convergence? What would happen if a singular value were exactly 0?

### Maximal Update Parameterisation (muP)
{: #mup}

Even with a good optimiser, the **learning rate must be retuned** every time the model width changes. This is expensive — a full sweep at 70B scale costs millions of dollars. muP provides a principled scaling law that makes the optimal learning rate transfer across model sizes.

**The core question**: what learning rate η keeps the per-layer *activation change* bounded when width increases?

**Approach**: instead of constraining `‖ΔW‖₂` or `‖ΔW‖_F`, require the **RMS-to-RMS norm** of the update to stay below a constant γ:

> `‖ΔW‖_{RMS→RMS} ≤ γ`

For Adam/SignGD (where the update is approximately `η·sign(G)` and `G` is rank-1 at batch size 1), working through the geometry gives:

> **muP learning-rate scaling for Adam**: `η ∝ 1/d_in`

Wider layers get a proportionally smaller learning rate so activation changes stay constant regardless of width.

<div class="post-flow post-flow--compare" role="group" aria-label="muP learning rate scaling by optimizer">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Adam / SignGD</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Update ≈ η·sign(G) — uniform step size</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">‖sign(G)‖₂ = √(d_in · d_out)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">η ∝ 1/d_in  to keep RMS-to-RMS bounded</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">SGD</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Update = η·∇L — magnitude-proportional</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">‖∇L‖_F ∝ √(d_in/d_out)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">η ∝ d_out/d_in  (different scaling!)</span></li>
    </ol>
  </div>
</div>

**What muP buys in practice**:

1. **Hyperparameter transfer**: sweep learning rates on a small (e.g. 100M) model under muP scaling → the optimal η transfers directly to larger models with only minor refinement
2. **Width invariance**: activation RMS norms stay bounded as you widen layers — training is numerically stable at any width
3. **Correct initialisation**: standard Xavier/He sets weight variance ∝ `1/d_in`. muP analysis reveals a safer choice is variance ∝ `min(d_in, d_out) / d_in²` — bounds worst-case spectral norm growth

**The catch**: the scaling formula changes with the optimiser. Applying Adam's `1/d_in` rule to SGD gives the wrong answer. Without understanding *why* the math works, it's easy to misapply it across optimiser changes or architectural modifications (e.g. tied embeddings, attention temperature).

| Optimiser | LR scaling with width | Intuition |
|---|---|---|
| Adam / AdamW | η ∝ 1/d_in | Sign step magnitude grows as √(d_in·d_out); divide by d_in to cancel |
| SGD | η ∝ d_out/d_in | Gradient Frobenius norm ∝ √(d_in/d_out); must cancel differently |
| Muon | η ∝ √(d_out/d_in) | Built into the RMS-to-RMS update scaling |

> **Think about it**: muP makes the learning rate depend on layer shape, not just a global scalar. How does this interact with parameter sharing (e.g. an embedding matrix used both at the input and output of a transformer)? What would the correct scaling be?

---

## Token Sampling & Decoding
{: #token-sampling}

Once a model is trained, every generated token is the result of a decoding decision. Decoding strategy is a separate hyperparameter from training — the same model weights produce wildly different outputs depending on how you sample from the output distribution.

**Autoregressive factorisation.** LLMs generate text sequentially, each token conditioned on all previous ones:

$$p(x_{1:T}) = \prod_{t=1}^T p(x_t \mid x_{<t})$$

At each step $t$, the model outputs a logit vector $z \in \mathbb{R}^{|\mathcal{V}|}$ which is converted to a probability distribution via softmax:

$$q_i = \frac{e^{z_i}}{\sum_{j=1}^{|\mathcal{V}|} e^{z_j}}$$

The decoding method determines which token to select from this distribution.

### Greedy & Beam Search
{: #greedy-beam}

**Greedy decoding.** Select the highest-probability token at every step:

$$x_t = \arg\max_{i \in \mathcal{V}}\; p(x_t = i \mid x_{<t})$$

Complexity: $O(T \cdot |\mathcal{V}|)$. Fully deterministic and fast.

Failure modes: myopic — a locally optimal token can foreclose globally better continuations. Produces repetitive, degenerate output on open-ended tasks. Once a mistake is made, it propagates irreversibly.

**Exhaustive search.** The theoretically optimal approach — find the sequence maximising joint log-probability:

$$x_{1:T}^* = \arg\max_{x_{1:T} \in \mathcal{V}^T} \sum_{t=1}^{T} \log p(x_t \mid x_{<t})$$

Complexity: $O(|\mathcal{V}|^T)$ — exponentially intractable. Motivates beam search as a practical approximation.

**Beam search.** Maintains $k$ candidate hypotheses (beams) in parallel, always retaining the $k$ with highest cumulative log-probability:

$$\text{score}(h) = \sum_{i=1}^{t} \log p(x_i \mid x_{<i})$$

Steps per token: expand each of the $k$ beams by $|\mathcal{V}|$ candidates → score all $k \cdot |\mathcal{V}|$ → prune back to top $k$.

Complexity: $O(T \cdot k \cdot |\mathcal{V}|)$.

**Length normalisation** prevents bias toward short sequences (which accumulate fewer log-probability penalties):

$$\text{score}_\text{norm}(h) = \frac{1}{t^\alpha} \sum_{i=1}^{t} \log p(x_i \mid x_{<i}), \quad \alpha \in [0, 1]$$

$\alpha = 0$ is unnormalised beam search; $\alpha = 1$ is full length normalisation. Tune $\alpha$ on a dev set.

**Constrained beam search.** Extends beam search with explicit constraint satisfaction — forces inclusion of required tokens or phrases. Uses a *banking mechanism*: groups hypotheses by constraint satisfaction level and selects round-robin to ensure balanced progress across constraint levels.

$$\text{score}(h) = \sum_{i=1}^{t} \log p(x_i \mid x_{<i}) + \lambda \cdot f_\text{constraint}(h)$$

| Method | Deterministic | Diversity | Best for |
|---|---|---|---|
| **Greedy** | Yes | Very low | Short factual outputs, speed |
| **Beam search** | Yes | Low | Translation, summarisation, structured output |
| **Constrained beam** | Yes | Low | Outputs with mandatory phrases/tokens |

### Temperature Scaling
{: #temperature}

Temperature rescales logits before softmax — it doesn't change model parameters, only the sharpness of the output distribution:

$$q_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{|\mathcal{V}|} \exp(z_j / T)}$$

| Temperature range | Distribution shape | Typical use |
|---|---|---|
| $T \to 0^+$ | Delta function on argmax (= greedy) | Deterministic factual tasks |
| $T \in [0.1, 0.5]$ | Sharp, peaked | Code generation, factual Q&A |
| $T \in [0.6, 1.0]$ | Balanced | Dialogue, chat, instruction following |
| $T > 1.0$ | Flat, high entropy | Creative writing, brainstorming |

As $T \to 0^+$, the distribution collapses to greedy. As $T \to \infty$, all tokens become equally likely.

Temperature is a global rescaling — it amplifies or dampens all probability differences proportionally. It doesn't filter which tokens are eligible to be sampled.

### Top-k & Top-p (Nucleus) Sampling
{: #topk-topp}

**Top-k sampling.** Restrict sampling to the $k$ highest-probability tokens; renormalise over the restricted set:

$$\mathcal{V}_k = \text{TopK}\!\left(p(x_t \mid x_{<t}),\; k\right)$$

$$\tilde{p}(x_t = i) = \begin{cases} \dfrac{p(x_t=i)}{\sum_{j \in \mathcal{V}_k} p(x_t=j)} & i \in \mathcal{V}_k \\ 0 & \text{otherwise} \end{cases}$$

Edge cases: $k = 1$ → greedy; $k = |\mathcal{V}|$ → unrestricted sampling.

**Problem with fixed $k$.** When the model's distribution is peaked (high-confidence prediction), a fixed $k$ of e.g. 50 still allows 49 low-probability tokens to compete with the dominant one. When the distribution is flat (model is uncertain), $k = 50$ may cut off many equally plausible continuations.

**Top-p (nucleus) sampling** ([Holtzman et al. 2020](https://arxiv.org/abs/1904.09731)) fixes this by adapting the candidate set size to the distribution shape. Select the *smallest* set of tokens whose cumulative probability exceeds threshold $p$:

$$\mathcal{V}_p = \left\{i \;\Big|\; \sum_{j=1}^{i} p_j \geq p\right\} \quad \text{(tokens sorted by descending probability)}$$

$$\tilde{p}(x_t = i) = \begin{cases} \dfrac{p(x_t=i)}{\sum_{j \in \mathcal{V}_p} p(x_t=j)} & i \in \mathcal{V}_p \\ 0 & \text{otherwise} \end{cases}$$

Typical values: $p \in [0.7, 0.9]$ (commonly $p \approx 0.9$ for open-ended, $p \approx 0.75$ for balanced).

Behaviour:
- $p \to 0$ → approaches greedy (only top token)
- $p \to 1$ → approaches unrestricted sampling

When the model is confident, $\mathcal{V}_p$ is small (a few tokens cover $p$ of the mass). When the model is uncertain, $\mathcal{V}_p$ expands to include many plausible tokens. This adaptivity is the key advantage over top-k.

**Top-k vs. top-p in practice:**

| Property | Top-k | Top-p |
|---|---|---|
| Candidate set size | Fixed ($k$) | Adaptive (depends on distribution) |
| Handles peaked distributions | Poorly (includes too many low-prob tokens) | Well (small set) |
| Handles flat distributions | Poorly (cuts off valid tokens) | Well (large set) |
| Hyperparameter sensitivity | High | Lower |
| Standard in production | Widely used | Dominant for open-ended generation |

Many APIs offer both; apply top-k first then top-p (top-k limits the pool, top-p further trims it).

**Temperature vs. nucleus:** temperature reshapes the distribution *before* computing the nucleus; nucleus filters *after*. Most APIs treat them as composable: apply temperature first, then nucleus filter.

### min-p Sampling
{: #minp}

min-p filters tokens relative to the *maximum* token probability rather than an absolute threshold:

$$\tau = \text{min-p} \cdot p_\text{max}$$

$$\mathcal{V}_\text{min-p} = \left\{i \mid p(x_t = i \mid x_{<t}) \geq \tau \right\}$$

Algorithm:
1. Compute $p_\text{max}$ (the highest-probability token's probability)
2. Set threshold $\tau = \text{min-p} \times p_\text{max}$
3. Drop all tokens below $\tau$
4. Renormalise; sample from the remainder

**Adaptive behaviour:**
- When the model is confident ($p_\text{max}$ is high), $\tau$ is high → aggressive filtering → near-greedy
- When the model is uncertain ($p_\text{max}$ is low), $\tau$ is low → many tokens survive → high diversity

This is the opposite of the top-k failure mode: the threshold scales with model confidence, not with an externally fixed count.

Recommended: $\text{min-p} \in [0.05, 0.1]$ combined with $T > 1.0$. min-p is particularly effective at high temperatures — it lets temperature drive creativity while preventing the model from sampling incoherent tail tokens.

### Tradeoffs & Practical Guide
{: #sampling-tradeoffs}

| Method | Deterministic | Diversity | Coherence | Hyperparameters |
|---|---|---|---|---|
| **Greedy** | Yes | Very low | High | None |
| **Beam search** | Yes | Low | High | $k$, $\alpha$ |
| **Temperature** | No | Tunable | Degrades at $T > 1$ | $T$ |
| **Top-k** | No | Medium | Medium | $k$ |
| **Top-p (nucleus)** | No | High, adaptive | Good | $p$ |
| **min-p** | No | Adaptive | Good at high $T$ | min-p, $T$ |

**Decision guide:**

```
Structured output (translation, summarisation, code)?
    → Beam search (k=4–8, length normalisation α=0.6)

Factual Q&A, short answers?
    → Greedy or low-temperature (T=0.1–0.3) top-p (p=0.9)

Dialogue / instruction following?
    → Temperature T=0.7, top-p p=0.9

Creative writing / brainstorming?
    → Temperature T=1.0–1.3, min-p=0.05–0.1
      (min-p prevents incoherence at high T)

Diverse candidates for reranking?
    → High temperature + top-p, sample K=10–20, pick best
```

**Repetition penalty.** A common add-on: reduce the logit of any token that has already appeared in the context by a multiplicative factor $< 1$. Addresses greedy/beam search's tendency to repeat phrases. Can be too aggressive — penalising reasonable repeated words (e.g. "the").

**Common anti-patterns:**
- Using $T > 1$ without min-p or top-p: the model can sample extremely low-probability (incoherent) tokens
- Using fixed $k$ with no temperature: brittle across distribution shapes
- Setting $p$ close to 1.0 and $T$ close to 0: effectively greedy — nucleus adds no value
- Beam search for open-ended creative generation: produces generic, safe, repetitive outputs

> **Interview question:** A chatbot is producing repetitive, generic responses. The engineer increases top-k from 10 to 100. Responses become more diverse but sometimes incoherent. What's a better approach, and why?
>
> *The problem is that a fixed top-k of 10 was too restrictive when the model's distribution was flat (uncertain turns), and increasing it to 100 let in low-probability garbage tokens when the model was confident. The right fix is to switch to top-p (nucleus) sampling with $p \approx 0.9$: when the model is confident, the nucleus is small and quality is preserved; when uncertain, the nucleus expands naturally. If coherence is still a concern at higher diversity, combine top-p with a modest temperature ($T \approx 0.8$) and min-p ($\approx 0.05$) as a floor filter. The repetition issue is separate — add a repetition penalty (logit downweight of $0.9$–$0.95$ for already-seen tokens) rather than reaching for a blunt top-k change.*
