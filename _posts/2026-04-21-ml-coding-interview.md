---
title: "ML Coding Interview"
date: 2026-04-21
display_order: 14
description: "From-scratch Python implementations of the most common ML coding interview questions: attention, transformers, beam search, embeddings, RAG, and more."
tags: [ml, coding, interview, transformers, python]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#attention">Attention Mechanisms</a>
      <ul class="post-toc-sublist">
        <li><a href="#scaled-dot-product">Scaled Dot-Product Attention</a></li>
        <li><a href="#mha">Multi-Head Attention</a></li>
        <li><a href="#causal-mask">Causal Masking</a></li>
      </ul>
    </li>
    <li><a href="#transformer">Transformer Block</a>
      <ul class="post-toc-sublist">
        <li><a href="#layer-norm">LayerNorm &amp; RMSNorm</a></li>
        <li><a href="#ffn">Feed-Forward Network</a></li>
        <li><a href="#full-block">Full Transformer Block</a></li>
        <li><a href="#positional">Positional Encodings</a></li>
      </ul>
    </li>
    <li><a href="#decoding">Decoding Strategies</a>
      <ul class="post-toc-sublist">
        <li><a href="#greedy">Greedy &amp; Top-k / Top-p</a></li>
        <li><a href="#beam-search">Beam Search</a></li>
        <li><a href="#temperature">Temperature Scaling</a></li>
      </ul>
    </li>
    <li><a href="#embeddings">Embeddings &amp; Similarity</a>
      <ul class="post-toc-sublist">
        <li><a href="#cosine">Cosine Similarity &amp; Distance</a></li>
        <li><a href="#knn">k-NN Search</a></li>
        <li><a href="#bm25">BM25</a></li>
      </ul>
    </li>
    <li><a href="#rag">RAG Pipeline</a>
      <ul class="post-toc-sublist">
        <li><a href="#chunking">Chunking &amp; Indexing</a></li>
        <li><a href="#retrieval">Retrieval &amp; Reranking</a></li>
        <li><a href="#rag-full">Full RAG Loop</a></li>
      </ul>
    </li>
    <li><a href="#losses">Loss Functions</a>
      <ul class="post-toc-sublist">
        <li><a href="#cross-entropy">Cross-Entropy &amp; Label Smoothing</a></li>
        <li><a href="#contrastive">Contrastive / InfoNCE</a></li>
        <li><a href="#dpo-loss">DPO Loss</a></li>
      </ul>
    </li>
    <li><a href="#misc">Miscellaneous</a>
      <ul class="post-toc-sublist">
        <li><a href="#softmax">Numerically Stable Softmax</a></li>
        <li><a href="#topk">Top-k / Top-p Sampling</a></li>
        <li><a href="#rope">RoPE Embeddings</a></li>
        <li><a href="#kv-cache">KV Cache</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## Overview
{: #overview}

This page collects the most commonly asked ML coding questions in research engineer and ML engineer interviews. Every implementation is:

- **Pure NumPy or pure Python** — no PyTorch/JAX magic hiding the logic
- **Commented at the decision points** — not every line, just where the interview answer lives
- **Followed by interview Q&A** — what the interviewer actually wants to hear

All code assumes `import numpy as np` at the top. Shape comments use `(B, T, D)` notation — batch, sequence length, hidden dim.

---

## Attention Mechanisms
{: #attention}

### Scaled Dot-Product Attention
{: #scaled-dot-product}

The core operation of every transformer. Given queries $$Q$$, keys $$K$$, values $$V$$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

```python
import numpy as np

def softmax(x, axis=-1):
    # Subtract max for numerical stability before exp
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (B, T_q, d_k)
    K: (B, T_k, d_k)
    V: (B, T_k, d_v)
    mask: (B, T_q, T_k) bool — True positions are MASKED OUT (set to -inf)
    Returns: (B, T_q, d_v)
    """
    d_k = Q.shape[-1]
    # (B, T_q, T_k)
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    weights = softmax(scores, axis=-1)   # (B, T_q, T_k)
    return weights @ V                   # (B, T_q, d_v)
```

> **Interview question:** Why divide by $$\sqrt{d_k}$$?
>
> **Answer:** The dot product $$QK^\top$$ has variance proportional to $$d_k$$ when $$Q$$ and $$K$$ are unit-variance vectors (variance of a sum of $$d_k$$ independent unit-variance products is $$d_k$$). Without scaling, large $$d_k$$ pushes logits into the saturation region of softmax, causing near-zero gradients. Dividing by $$\sqrt{d_k}$$ restores unit variance.

---

### Multi-Head Attention
{: #mha}

Split the representation into $$h$$ heads, run attention in parallel, then concatenate and project.

```python
def multi_head_attention(X, W_q, W_k, W_v, W_o, num_heads, mask=None):
    """
    X:    (B, T, D)
    W_q, W_k, W_v: (D, D)   — full projection matrices
    W_o:  (D, D)             — output projection
    Returns: (B, T, D)
    """
    B, T, D = X.shape
    d_head = D // num_heads

    # Project to Q, K, V
    Q = X @ W_q   # (B, T, D)
    K = X @ W_k
    V = X @ W_v

    # Split into heads: (B, T, D) -> (B, h, T, d_head)
    def split_heads(x):
        return x.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)

    Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

    # Run attention per head — reshape to (B*h, T, d_head) for batched matmul
    Q = Q.reshape(B * num_heads, T, d_head)
    K = K.reshape(B * num_heads, T, d_head)
    V = V.reshape(B * num_heads, T, d_head)

    # Expand mask for all heads if provided
    if mask is not None:
        mask = np.repeat(mask, num_heads, axis=0)  # (B*h, T, T)

    out = scaled_dot_product_attention(Q, K, V, mask)  # (B*h, T, d_head)

    # Concatenate heads: (B*h, T, d_head) -> (B, T, D)
    out = out.reshape(B, num_heads, T, d_head)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)

    return out @ W_o   # (B, T, D)
```

> **Interview question:** What does each head learn?
>
> **Answer:** In theory, different heads can specialise on different types of relationships — one head might track syntactic dependencies, another coreference. In practice, analysis shows heads are often redundant and can be pruned with minimal loss. The key value is that the model has multiple independent "views" of the same sequence; gradient flow through each head is independent, which helps optimisation.

---

### Causal Masking
{: #causal-mask}

For autoregressive (decoder-only) models, position $$i$$ must not attend to positions $$j > i$$.

```python
def causal_mask(T):
    """
    Returns (1, T, T) bool mask — True = masked (upper triangle, excl. diagonal).
    Broadcast over batch dimension.
    """
    # np.triu with k=1 gives upper triangle excluding diagonal
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    return mask[np.newaxis, :, :]   # (1, T, T)

# Usage
T = 5
print(causal_mask(T).squeeze().astype(int))
# [[0 1 1 1 1]
#  [0 0 1 1 1]
#  [0 0 0 1 1]
#  [0 0 0 0 1]
#  [0 0 0 0 0]]
```

> **Interview question:** How does Flash Attention handle the causal mask efficiently?
>
> **Answer:** FlashAttention tiles $$Q$$, $$K$$, $$V$$ into SRAM blocks and computes attention block by block. For causal attention, any tile where all $$Q$$ positions are strictly less than all $$K$$ positions is fully masked and can be skipped entirely — saving roughly half the compute. The online softmax algorithm (track running max and sum) allows the normalisation to be computed without materialising the full $$T \times T$$ attention matrix.

---

## Transformer Block
{: #transformer}

### LayerNorm & RMSNorm
{: #layer-norm}

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    """
    x:     (B, T, D)
    gamma, beta: (D,)   — learned scale and shift
    Normalises over the last dimension (feature dim).
    """
    mean = x.mean(axis=-1, keepdims=True)       # (B, T, 1)
    var  = x.var(axis=-1, keepdims=True)        # (B, T, 1)
    x_norm = (x - mean) / np.sqrt(var + eps)   # (B, T, D)
    return gamma * x_norm + beta

def rms_norm(x, gamma, eps=1e-5):
    """
    RMSNorm: skip mean subtraction — only scale by RMS.
    Used in LLaMA, Mistral, and most modern LLMs.
    Cheaper: one less mean computation, and no beta parameter.
    """
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return gamma * (x / rms)
```

> **Interview question:** Why do modern LLMs prefer RMSNorm over LayerNorm?
>
> **Answer:** RMSNorm removes the mean-centering step, saving ~30% of the compute of LayerNorm. The hypothesis (backed by ablations in the LLaMA paper) is that the re-centring invariance of LayerNorm is not necessary — only the re-scaling invariance matters for training stability. RMSNorm also has no $$\beta$$ parameter, slightly reducing the parameter count.

---

### Feed-Forward Network
{: #ffn}

The standard two-layer FFN with SwiGLU (used in LLaMA/Mistral):

```python
def relu(x):
    return np.maximum(0, x)

def gelu(x):
    # Approximation used in GPT-2
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def silu(x):
    # SiLU / Swish: x * sigmoid(x)
    return x / (1 + np.exp(-x))

def ffn(x, W1, W2, activation=gelu):
    """Standard FFN: x -> Linear -> Activation -> Linear"""
    return activation(x @ W1) @ W2   # (B, T, D)

def swiglu_ffn(x, W_gate, W_up, W_down):
    """
    SwiGLU: gate * silu(up_proj), then down_proj.
    W_gate, W_up: (D, D_ff)
    W_down: (D_ff, D)
    ~1/3 larger D_ff than standard FFN for same param count.
    """
    gate = silu(x @ W_gate)   # (B, T, D_ff)
    up   = x @ W_up           # (B, T, D_ff)
    return (gate * up) @ W_down   # (B, T, D)
```

---

### Full Transformer Block
{: #full-block}

```python
def transformer_block(x, attn_params, ffn_params, num_heads, pre_norm=True):
    """
    Pre-norm (modern) vs post-norm (original "Attention is All You Need").
    x: (B, T, D)
    """
    W_q, W_k, W_v, W_o = attn_params
    W1, W2, gamma1, beta1, gamma2, beta2 = ffn_params

    if pre_norm:
        # Pre-norm: normalise BEFORE sublayer — more stable, used in GPT-2+
        x = x + multi_head_attention(
            layer_norm(x, gamma1, beta1), W_q, W_k, W_v, W_o, num_heads,
            mask=causal_mask(x.shape[1])
        )
        x = x + ffn(layer_norm(x, gamma2, beta2), W1, W2)
    else:
        # Post-norm: original transformer — normalise AFTER residual
        x = layer_norm(x + multi_head_attention(
            x, W_q, W_k, W_v, W_o, num_heads,
            mask=causal_mask(x.shape[1])
        ), gamma1, beta1)
        x = layer_norm(x + ffn(x, W1, W2), gamma2, beta2)

    return x
```

> **Interview question:** Why does pre-norm training converge more reliably?
>
> **Answer:** In post-norm, gradients must flow through the LayerNorm in the residual branch, and the norm can shrink gradients for early layers. Pre-norm ensures the residual stream is never passed through a norm before being added back — gradients flow directly through the identity residual path. This makes learning rate warmup less critical and enables training very deep transformers.

---

### Positional Encodings
{: #positional}

```python
def sinusoidal_pe(T, D):
    """
    Fixed sinusoidal positional encoding from "Attention is All You Need".
    Returns (T, D).
    """
    pe = np.zeros((T, D))
    pos = np.arange(T)[:, np.newaxis]          # (T, 1)
    div = np.exp(np.arange(0, D, 2) * (-np.log(10000.0) / D))  # (D/2,)

    pe[:, 0::2] = np.sin(pos * div)   # even dims
    pe[:, 1::2] = np.cos(pos * div)   # odd dims
    return pe   # (T, D)
```

---

## Decoding Strategies
{: #decoding}

### Greedy & Top-k / Top-p
{: #greedy}

```python
def greedy_decode(logits):
    """logits: (vocab_size,) — return argmax token."""
    return int(np.argmax(logits))

def top_k_sample(logits, k, temperature=1.0):
    """
    Keep only top-k logits, zero out the rest, then sample.
    logits: (vocab_size,)
    """
    logits = logits / temperature
    # Find the k-th largest value
    threshold = np.sort(logits)[-k]
    logits = np.where(logits >= threshold, logits, -1e9)
    probs = softmax(logits)
    return int(np.random.choice(len(probs), p=probs))

def top_p_sample(logits, p, temperature=1.0):
    """
    Nucleus sampling: keep smallest set of tokens whose cumulative prob >= p.
    logits: (vocab_size,)
    """
    logits = logits / temperature
    probs = softmax(logits)

    # Sort descending, compute cumulative sum
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)

    # Remove tokens once cumulative prob exceeds p
    # Keep at least 1 token (the most probable)
    cutoff = np.searchsorted(cumulative, p) + 1
    top_idx = sorted_idx[:cutoff]

    # Renormalise and sample
    top_probs = probs[top_idx]
    top_probs /= top_probs.sum()
    return int(np.random.choice(top_idx, p=top_probs))
```

> **Interview question:** What is the difference between top-k and top-p sampling?
>
> **Answer:** Top-k fixes the *number* of candidates regardless of how the probability mass is distributed. If the distribution is very peaked, k=50 includes many near-zero tokens; if it's flat, k=50 might miss most of the mass. Top-p (nucleus) adapts to the distribution shape: it includes only as many tokens as needed to cover p of the probability mass. A peaked distribution results in a small nucleus (1-5 tokens); a flat distribution results in a large nucleus. Top-p is generally preferred because it avoids both over-sampling from flat distributions and under-sampling from peaked ones.

---

### Beam Search
{: #beam-search}

```python
def beam_search(logprob_fn, initial_token, beam_width, max_len, eos_id):
    """
    logprob_fn(tokens) -> (vocab_size,) log-probabilities for next token.
    Returns the highest-scoring complete sequence.

    Each beam is (cumulative_log_prob, token_list).
    """
    beams = [(0.0, [initial_token])]
    completed = []

    for _ in range(max_len):
        candidates = []

        for score, tokens in beams:
            if tokens[-1] == eos_id:
                completed.append((score, tokens))
                continue

            log_probs = logprob_fn(tokens)   # (vocab_size,)

            # Expand: take top beam_width next tokens
            top_tokens = np.argsort(log_probs)[-beam_width:]
            for tok in top_tokens:
                new_score = score + log_probs[tok]
                candidates.append((new_score, tokens + [int(tok)]))

        if not candidates:
            break

        # Keep top beam_width beams by score
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

    # Return best completed sequence, or best beam if none completed
    all_seqs = completed + beams
    best_score, best_tokens = max(all_seqs, key=lambda x: x[0])
    return best_tokens
```

> **Interview question:** What are the failure modes of beam search for open-ended generation?
>
> **Answer:** Beam search maximises the joint probability of the sequence, but high-probability sequences tend to be generic and repetitive — the model has seen "safe" completions many times. This is the *text degeneration* problem described by Holtzman et al. (2020): beam search produces text that scores well but is dull. Additionally, beam search with length normalisation can still favour shorter sequences, and without it, it favours short ones even more. For open-ended generation (story writing, dialogue), sampling-based methods (top-p, temperature) produce more diverse and natural text. Beam search remains useful for constrained tasks: machine translation, summarisation, and any task where accuracy matters more than diversity.

> **Interview question:** How do you handle length normalisation in beam search?
>
> **Answer:** Divide the cumulative log-probability by $$T^\alpha$$ where $$T$$ is the sequence length and $$\alpha \in [0.6, 0.8]$$ is a tunable exponent. Without this, log-probabilities monotonically decrease as the sequence grows (each additional token multiplies by a probability $$\leq 1$$, adding a negative log), so shorter sequences always win. Length normalisation levels the playing field. $$\alpha = 1$$ is full normalisation (average log-prob); $$\alpha = 0$$ is no normalisation. The Google NMT paper found $$\alpha = 0.6$$ worked best empirically.

---

### Temperature Scaling
{: #temperature}

```python
def temperature_scale(logits, T):
    """
    Divide logits by T before softmax.
    T < 1: sharper distribution (more confident / greedy)
    T > 1: flatter distribution (more random / diverse)
    T -> 0: argmax (greedy)
    T -> inf: uniform distribution
    """
    return softmax(logits / T)

def calibration_temperature(logits_val, labels_val, T_range=np.linspace(0.1, 3.0, 100)):
    """
    Find T that minimises NLL on a validation set — post-hoc calibration.
    logits_val: (N, C)
    labels_val: (N,) int
    """
    best_T, best_nll = 1.0, float('inf')
    for T in T_range:
        probs = np.array([softmax(l / T) for l in logits_val])
        nll = -np.log(probs[np.arange(len(labels_val)), labels_val] + 1e-9).mean()
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T
```

---

## Embeddings & Similarity
{: #embeddings}

### Cosine Similarity & Distance
{: #cosine}

```python
def l2_normalize(x, eps=1e-9):
    """x: (..., D) — normalise along last dimension."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)

def cosine_similarity(a, b):
    """
    a: (D,) or (N, D)
    b: (D,) or (M, D)
    Returns scalar or (N, M) matrix of cosine similarities.
    """
    a = l2_normalize(a)
    b = l2_normalize(b)
    return a @ b.T   # dot product of unit vectors = cosine

def pairwise_cosine(A, B):
    """
    A: (N, D), B: (M, D)
    Returns (N, M) — all pairwise cosine similarities.
    """
    A = l2_normalize(A)
    B = l2_normalize(B)
    return A @ B.T

def cosine_distance(a, b):
    return 1 - cosine_similarity(a, b)
```

> **Interview question:** When would you use L2 distance over cosine similarity for embedding search?
>
> **Answer:** Cosine similarity is invariant to vector magnitude — it only measures directional alignment. This is appropriate when magnitude is an artifact (e.g., different document lengths produce different TF-IDF magnitudes, but you care about topic direction). L2 distance captures both direction and magnitude, which matters when magnitude is meaningful (e.g., dense retrieval models trained with contrastive loss often encode relevance partly in magnitude). For FAISS-based retrieval: if vectors are L2-normalised, cosine search is equivalent to MIPS (maximum inner product search), which FAISS's `IndexFlatIP` implements efficiently.

---

### k-NN Search
{: #knn}

```python
def exact_knn(query, index, k):
    """
    Brute-force exact k-NN by cosine similarity.
    query: (D,)
    index: (N, D)
    Returns top-k (indices, scores).
    """
    sims = pairwise_cosine(query[np.newaxis], index).squeeze()  # (N,)
    top_k_idx = np.argsort(sims)[-k:][::-1]
    return top_k_idx, sims[top_k_idx]

def euclidean_knn(query, index, k):
    """Exact k-NN by L2 distance."""
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    q_sq = (query ** 2).sum()
    i_sq = (index ** 2).sum(axis=1)   # (N,)
    dot  = index @ query              # (N,)
    dists = q_sq + i_sq - 2 * dot    # (N,)
    top_k_idx = np.argsort(dists)[:k]
    return top_k_idx, dists[top_k_idx]
```

---

### BM25
{: #bm25}

The standard sparse retrieval baseline — frequently asked as a "how would you implement keyword search" question.

```python
from collections import Counter
import math

class BM25:
    """
    BM25 scoring: TF-IDF variant with saturation and length normalisation.
    k1 controls TF saturation (typical: 1.2–2.0).
    b  controls length normalisation (typical: 0.75).
    """
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)

        # Tokenise
        self.tokenised = [doc.lower().split() for doc in corpus]
        self.avgdl = sum(len(d) for d in self.tokenised) / self.N

        # Document frequencies
        self.df = {}
        for doc in self.tokenised:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

        # IDF with smoothing (Robertson IDF)
        self.idf = {
            term: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.df.items()
        }

    def score(self, query, doc_idx):
        query_terms = query.lower().split()
        doc = self.tokenised[doc_idx]
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            f = tf.get(term, 0)
            # BM25 TF with saturation and length normalisation
            numerator   = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * numerator / denominator
        return score

    def retrieve(self, query, k=5):
        scores = [self.score(query, i) for i in range(self.N)]
        top_k = sorted(range(self.N), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, scores[i], self.corpus[i]) for i in top_k]
```

> **Interview question:** What are the failure modes of BM25 compared to dense retrieval?
>
> **Answer:** BM25 is purely lexical — it requires exact term overlap between query and document. It fails on synonymy (query: "car", document says "automobile"), paraphrase, and cross-lingual retrieval. It also has no notion of context: "bank" in a financial query and "bank" in a river query score identically. Dense retrieval (bi-encoders like DPR) maps queries and documents into a shared semantic space, handling synonymy and paraphrase. The trade-off: BM25 is exact, interpretable, and fast (inverted index); dense retrieval requires approximate nearest-neighbour indexing (FAISS/HNSW) and is slower to index. Hybrid retrieval (BM25 + dense re-ranking) is the standard production pattern.

---

## RAG Pipeline
{: #rag}

### Chunking & Indexing
{: #chunking}

```python
def chunk_text(text, chunk_size=256, overlap=32):
    """
    Split text into overlapping word-level chunks.
    overlap: number of words shared between consecutive chunks.
    """
    words = text.split()
    chunks = []
    stride = chunk_size - overlap
    for start in range(0, len(words), stride):
        chunk = ' '.join(words[start:start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks

def build_index(chunks, embed_fn):
    """
    embed_fn(list[str]) -> (N, D) numpy array.
    Returns L2-normalised embedding matrix for cosine search.
    """
    embeddings = embed_fn(chunks)           # (N, D)
    return l2_normalize(embeddings)         # (N, D)
```

---

### Retrieval & Reranking
{: #retrieval}

```python
def retrieve(query, index_embeddings, chunks, embed_fn, k=5):
    """
    Embed query, do cosine search against index, return top-k chunks.
    """
    q_emb = l2_normalize(embed_fn([query]))       # (1, D)
    sims  = (q_emb @ index_embeddings.T).squeeze()  # (N,)
    top_k = np.argsort(sims)[-k:][::-1]
    return [(chunks[i], float(sims[i])) for i in top_k]

def reciprocal_rank_fusion(rankings, k=60):
    """
    Fuse multiple ranked lists (e.g., BM25 + dense) via RRF.
    rankings: list of lists of doc IDs, ranked best-first.
    k: RRF constant (default 60, from Cormack et al.).
    Returns fused ranking as sorted list of (doc_id, score).
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

> **Interview question:** When does hybrid retrieval (BM25 + dense) outperform either alone?
>
> **Answer:** Hybrid retrieval is most beneficial when queries mix keyword specificity with semantic intent. A query like "BERT paper attention mechanism" benefits from BM25's exact match on "BERT" and "attention" while the dense model handles semantic similarity. RRF is a simple, parameter-free fusion that works surprisingly well — it avoids the score normalisation problem (BM25 and dense models produce scores on incompatible scales). Learned rerankers (cross-encoders) can further improve precision at the top of the list at the cost of latency.

---

### Full RAG Loop
{: #rag-full}

```python
def rag_pipeline(query, chunks, index_embeddings, embed_fn, generate_fn, k=5):
    """
    Full RAG: retrieve -> format prompt -> generate.

    embed_fn(list[str])   -> (N, D) embeddings
    generate_fn(prompt)   -> str response
    """
    # 1. Retrieve relevant chunks
    results = retrieve(query, index_embeddings, chunks, embed_fn, k=k)

    # 2. Format context
    context = '\n\n'.join([
        f"[{i+1}] {chunk}" for i, (chunk, score) in enumerate(results)
    ])

    # 3. Build prompt
    prompt = f"""Answer the question using only the provided context.

Context:
{context}

Question: {query}

Answer:"""

    # 4. Generate
    return generate_fn(prompt)
```

---

## Loss Functions
{: #losses}

### Cross-Entropy & Label Smoothing
{: #cross-entropy}

```python
def cross_entropy(logits, labels, eps=1e-9):
    """
    logits: (N, C)  — raw unnormalised scores
    labels: (N,)    — integer class indices
    """
    probs = softmax(logits)                          # (N, C)
    correct_probs = probs[np.arange(len(labels)), labels]
    return -np.log(correct_probs + eps).mean()

def label_smoothed_ce(logits, labels, smoothing=0.1, eps=1e-9):
    """
    Label smoothing: instead of one-hot targets, use
    (1 - smoothing) for the correct class and smoothing/(C-1) for others.
    Prevents the model from being overconfident.
    """
    N, C = logits.shape
    probs = softmax(logits)
    log_probs = np.log(probs + eps)

    # Smooth targets
    smooth_val = smoothing / (C - 1)
    targets = np.full((N, C), smooth_val)
    targets[np.arange(N), labels] = 1.0 - smoothing

    return -(targets * log_probs).sum(axis=-1).mean()
```

---

### Contrastive / InfoNCE
{: #contrastive}

The loss behind CLIP, SimCLR, and most embedding models.

```python
def infonce_loss(anchors, positives, temperature=0.07):
    """
    anchors:   (N, D) — e.g., query embeddings
    positives: (N, D) — e.g., matching document embeddings
    For each anchor, its positive is index i; all other N-1 are negatives.
    """
    anchors   = l2_normalize(anchors)
    positives = l2_normalize(positives)

    # (N, N) similarity matrix — diagonal is positive pairs
    logits = (anchors @ positives.T) / temperature

    # Labels: diagonal indices are the correct positives
    labels = np.arange(len(anchors))

    # Cross-entropy in both directions (symmetric), then average
    loss_a2p = cross_entropy(logits,   labels)
    loss_p2a = cross_entropy(logits.T, labels)
    return (loss_a2p + loss_p2a) / 2
```

> **Interview question:** Why does a lower temperature make InfoNCE harder to optimise?
>
> **Answer:** A lower temperature amplifies score differences — the softmax becomes more peaked. This means small similarity differences between positives and hard negatives produce large gradient signals, which is good for representation quality but makes training unstable early on when representations are random. Warmup schedules that start with higher temperature and anneal down (as in CLIP) are standard. Too low a temperature causes vanishing gradients for easy positives (the model is already confident) and exploding gradients for hard negatives.

---

### DPO Loss
{: #dpo-loss}

Direct Preference Optimisation — trains on (prompt, chosen, rejected) triples without a reward model.

```python
def dpo_loss(policy_logprobs_chosen, policy_logprobs_rejected,
             ref_logprobs_chosen,    ref_logprobs_rejected,
             beta=0.1):
    """
    All inputs are scalar log-probabilities of the full completion
    (sum of per-token log-probs).

    beta: KL penalty strength — higher = stay closer to reference policy.
    """
    # Log ratio of policy over reference for chosen and rejected
    log_ratio_chosen   = policy_logprobs_chosen   - ref_logprobs_chosen
    log_ratio_rejected = policy_logprobs_rejected - ref_logprobs_rejected

    # DPO objective: push chosen ratio up, rejected ratio down
    logits = beta * (log_ratio_chosen - log_ratio_rejected)
    loss = -np.log(1 / (1 + np.exp(-logits)))   # binary cross-entropy
    return loss
```

> **Interview question:** Why does DPO not need a reward model?
>
> **Answer:** DPO reparametrises the RLHF objective. It can be shown that the optimal policy under the KL-constrained reward maximisation is $$\pi^*(y \vert x) \propto \pi_\text{ref}(y \vert x) \exp(r(x,y)/\beta)$$. This implies the reward is a deterministic function of any policy and the reference: $$r(x,y) = \beta \log \frac{\pi(y \vert x)}{\pi_\text{ref}(y \vert x)} + \beta \log Z(x)$$. Since $$Z(x)$$ cancels in the Bradley-Terry preference model, we can directly substitute this implicit reward into the preference loss and optimise the policy directly — eliminating the reward model training step and the RL loop.

---

## Miscellaneous
{: #misc}

### Numerically Stable Softmax
{: #softmax}

```python
def softmax_stable(x):
    """
    Subtract max before exp to prevent overflow.
    Mathematically equivalent: softmax(x) = softmax(x - c) for any c.
    """
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def log_softmax(x):
    """
    Numerically stable log-softmax.
    Used when you need log-probabilities directly (avoids log(softmax(x)) instability).
    """
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))
```

> **Interview question:** Why is `log(softmax(x))` numerically unstable and how do you fix it?
>
> **Answer:** `softmax(x)` can produce values extremely close to 0 for non-maximum logits when logits are large, causing `log(softmax(x))` to return `-inf`. `log_softmax` avoids this by computing `x - max(x) - log(sum(exp(x - max(x))))` in one pass — the exponents are bounded in `[0, 1]` after the max subtraction, so the sum is well-conditioned. PyTorch's `F.cross_entropy` uses this internally via `nll_loss(log_softmax(logits), targets)`.

---

### Top-k / Top-p Filtering (Production Style)
{: #topk}

```python
def filter_logits(logits, top_k=0, top_p=0.0, temperature=1.0, min_tokens=1):
    """
    Apply temperature, then top-k and/or top-p filtering.
    logits: (vocab_size,)
    Returns filtered logits (not probabilities — caller applies softmax).
    """
    logits = logits / max(temperature, 1e-8)

    if top_k > 0:
        k = min(top_k, logits.size)
        kth_val = np.sort(logits)[-k]
        logits = np.where(logits < kth_val, -1e9, logits)

    if 0 < top_p < 1.0:
        sorted_idx = np.argsort(logits)[::-1]
        probs = softmax_stable(logits[sorted_idx])
        cumulative = np.cumsum(probs)
        # Remove tokens beyond the nucleus, keeping at least min_tokens
        cutoff_idx = max(np.searchsorted(cumulative, top_p) + 1, min_tokens)
        tokens_to_remove = sorted_idx[cutoff_idx:]
        logits[tokens_to_remove] = -1e9

    return logits
```

---

### RoPE (Rotary Positional Embeddings)
{: #rope}

Used in LLaMA, Mistral, GPT-NeoX — encodes position by rotating Q and K vectors.

```python
def precompute_rope_freqs(dim, max_seq_len, base=10000.0):
    """
    Precompute cos/sin tables for RoPE.
    dim: head dimension (must be even)
    Returns cos, sin each of shape (max_seq_len, dim//2).
    """
    half = dim // 2
    # theta_i = 1 / (base ^ (2i / dim)) for i in [0, half)
    theta = 1.0 / (base ** (np.arange(0, half) * 2 / dim))  # (half,)
    positions = np.arange(max_seq_len)[:, np.newaxis]        # (T, 1)
    freqs = positions * theta[np.newaxis, :]                 # (T, half)
    return np.cos(freqs), np.sin(freqs)                      # (T, half)

def apply_rope(x, cos, sin):
    """
    x:   (B, T, num_heads, head_dim)
    cos, sin: (T, head_dim//2)
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]

    # Rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    cos = cos[np.newaxis, :, np.newaxis, :]   # (1, T, 1, half)
    sin = sin[np.newaxis, :, np.newaxis, :]

    x_rotated = np.concatenate([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], axis=-1)
    return x_rotated
```

> **Interview question:** Why do RoPE embeddings extrapolate better than learned absolute positional embeddings?
>
> **Answer:** Learned absolute embeddings have no notion of relative position — position 512 has no geometric relationship to position 511. RoPE encodes relative position implicitly: the dot product of two RoPE-rotated vectors $$q_m$$ and $$k_n$$ depends only on $$(m - n)$$, not on $$m$$ and $$n$$ individually. This means the attention pattern for a token pair at relative distance $$d$$ is the same regardless of where in the sequence they appear. For extrapolation beyond the training context, techniques like YaRN and LongRoPE further scale the base frequency $$\theta$$ or interpolate positions, extending context windows from 4K to 128K+ without full retraining.

---

### KV Cache
{: #kv-cache}

```python
class KVCache:
    """
    Incremental KV cache for autoregressive decoding.
    Avoids recomputing K and V for all previous tokens at each step.
    """
    def __init__(self, num_layers, num_heads, max_seq_len, head_dim):
        self.k_cache = np.zeros((num_layers, num_heads, max_seq_len, head_dim))
        self.v_cache = np.zeros((num_layers, num_heads, max_seq_len, head_dim))
        self.cur_len = 0

    def update(self, layer_idx, k_new, v_new):
        """
        k_new, v_new: (num_heads, 1, head_dim) — single new token's K and V.
        Returns full K, V up to current position: (num_heads, cur_len, head_dim).
        """
        pos = self.cur_len
        self.k_cache[layer_idx, :, pos, :] = k_new[:, 0, :]
        self.v_cache[layer_idx, :, pos, :] = v_new[:, 0, :]

        k_full = self.k_cache[layer_idx, :, :pos + 1, :]
        v_full = self.v_cache[layer_idx, :, :pos + 1, :]
        return k_full, v_full

    def increment(self):
        self.cur_len += 1
```

> **Interview question:** What is the memory bottleneck of KV cache at large context lengths?
>
> **Answer:** KV cache size scales as $$2 \times \text{num\_layers} \times \text{num\_heads} \times T \times d_\text{head} \times \text{bytes\_per\_element}$$. For LLaMA-3 70B (80 layers, 8 KV heads, head dim 128, FP16): $$2 \times 80 \times 8 \times T \times 128 \times 2 = 327{,}680 \times T$$ bytes. At $$T = 128{,}000$$ tokens, that's ~40 GB — larger than the model weights for a single sequence. Techniques to reduce this include: **GQA** (grouped-query attention, reduces KV heads from 32 to 8), **MLA** (multi-head latent attention in DeepSeek, compresses KV into a low-rank latent), **quantised KV cache** (INT8/INT4), and **sliding window attention** (Mistral) which caps the effective KV length.
