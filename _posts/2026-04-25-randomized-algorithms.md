---
layout: post
title: "Randomized Algorithms for ML & Search"
date: 2026-04-25
display_order: 12
description: "Fingerprinting, hashing, ANN search (LSH, SimHash, HNSW, FAISS), Johnson-Lindenstrauss, and streaming algorithms — the mathematical machinery behind Microsoft-scale search and vector retrieval."
tags: [algorithms, search, ml-systems, ann, hashing, streaming]
math: true
---

Search at Microsoft scale — Bing, Azure Cognitive Search, GitHub Copilot's retrieval, Teams meeting search — means you have billions of documents, billions of query events per day, and a latency budget measured in tens of milliseconds. You cannot afford an exact linear scan. Randomized algorithms are the reason this is solvable at all: they trade a small, controllable probability of error for massive gains in time and space.

This post covers the mathematical core: fingerprinting, hashing theory, locality-sensitive hashing (LSH), the Johnson-Lindenstrauss lemma, practical ANN indices (HNSW, FAISS, DiskANN), and streaming algorithms for massive data streams. Every section is grounded in both the math and what this means when you're actually building search infrastructure.

---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#probability-tools">Probability Tools (the Foundation)</a></li>
    <li><a href="#fingerprinting">Fingerprinting</a></li>
    <li><a href="#hashing">Hashing Theory</a>
      <ul class="post-toc-sublist">
        <li><a href="#universal-hashing">Universal & Pairwise Independent Hashing</a></li>
        <li><a href="#minhash">MinHash & Jaccard Similarity</a></li>
      </ul>
    </li>
    <li><a href="#jl-lemma">Johnson-Lindenstrauss & Random Projection</a></li>
    <li><a href="#lsh">Locality-Sensitive Hashing (LSH)</a>
      <ul class="post-toc-sublist">
        <li><a href="#lsh-jaccard">LSH for Jaccard Similarity</a></li>
        <li><a href="#simhash">SimHash for Cosine Similarity</a></li>
        <li><a href="#lsh-theory">LSH-based ANN: Theory</a></li>
      </ul>
    </li>
    <li><a href="#ann-practice">ANN Search in Practice</a>
      <ul class="post-toc-sublist">
        <li><a href="#faiss">FAISS & Product Quantization</a></li>
        <li><a href="#hnsw">HNSW: Graph-Based Search</a></li>
        <li><a href="#diskann">DiskANN: Billion-Scale on Disk</a></li>
      </ul>
    </li>
    <li><a href="#streaming">Streaming Algorithms</a>
      <ul class="post-toc-sublist">
        <li><a href="#count-min">Count-Min Sketch</a></li>
        <li><a href="#distinct-elements">Distinct Elements: Flajolet-Martin & HyperLogLog</a></li>
        <li><a href="#bloom-filters">Bloom Filters</a></li>
      </ul>
    </li>
    <li><a href="#putting-together">Putting It Together: Search Pipeline</a></li>
  </ul>
</nav>

---

## Probability Tools (the Foundation) {#probability-tools}

Before the algorithms, three probability tools that appear everywhere.

**Markov's Inequality.** For any non-negative random variable $X$ and $t > 0$:

$$\Pr[X \geq t] \leq \frac{\mathbb{E}[X]}{t}$$

Simple but surprisingly powerful. If the average latency of your index is 10ms, the probability it exceeds 100ms is at most 10%. Markov needs only the expectation — no distribution assumption.

**Chebyshev's Inequality.** For any random variable $X$ with mean $\mu$ and variance $\sigma^2$, for any $k > 0$:

$$\Pr[|X - \mu| \geq k\sigma] \leq \frac{1}{k^2}$$

Equivalently: $\Pr[\lvert X - \mu \rvert \geq t] \leq \frac{\sigma^2}{t^2}$.

Chebyshev is two-sided (bounds both over and under the mean) and requires only the variance — not the full distribution. The cost: it gives polynomial tail bounds ($1/k^2$) rather than exponential. Good for "off the shelf" analysis; when you need tighter bounds, use Chernoff.

**Chernoff Bounds.** For independent random variables summing to $S$ with $\mathbb{E}[S] = \mu$, for $\epsilon < 1$:

$$\Pr[|S - \mu| \geq \epsilon \mu] \leq 2e^{-\epsilon^2 \mu / 3}$$

Exponential tails — probabilities shrink much faster than Chebyshev. This is why you can take a union bound over $n^2$ pairs and still get meaningful guarantees. The tradeoff: requires independence (or near-independence).

**Linearity of Variance (pairwise independence suffices):**

$$\text{Var}\!\left[\sum_{i=1}^k X_i\right] = \sum_{i=1}^k \text{Var}[X_i] \quad \text{when } X_1,\ldots,X_k \text{ are pairwise independent}$$

Mutual independence is not required — pairwise is enough. This fact underpins why CountMin Sketch and Flajolet-Martin work with cheap hash functions.

**Variance reduction by averaging.** If $X_1, \ldots, X_k$ are i.i.d. with mean $\mu$ and variance $\sigma^2$, then $\bar{X} = \frac{1}{k}\sum X_i$ has:

$$\mathbb{E}[\bar{X}] = \mu, \quad \text{Var}[\bar{X}] = \frac{\sigma^2}{k}$$

Standard deviation shrinks as $1/\sqrt{k}$. This is the universal trick in randomized algorithms: one estimator is noisy, $k$ estimators averaged together are $\sqrt{k}$ times less noisy.

---

## Fingerprinting {#fingerprinting}

**Problem:** You have two files $f_1, f_2$ — each could be gigabytes. Did they change? Sending both over the network to compare is too expensive. Can you compare compact "fingerprints" instead?

This comes up constantly in real systems:
- **Distributed storage**: did the replica diverge from the master?
- **Web crawling**: has this URL's content changed since we last indexed it?
- **Deduplication**: is this uploaded image already in our store? (Airbnb, Pinterest do this at scale)
- **Content-addressable storage**: Git, IPFS, Docker layers all use fingerprints as file identifiers.

**Rabin Fingerprinting (1981).** Interpret the file $f$ as a large integer (its bit string read as a binary number). Choose a random prime $p$ uniformly from $\{2, \ldots, tn \log(tn)\}$ where $n$ is the file length in bits and $t$ is a parameter controlling error probability. Define:

$$h(f) = f \bmod p$$

The fingerprint is just the remainder when divided by a random prime. It fits in $O(\log n + \log t)$ bits — for a 1MB file ($n \approx 8 \times 10^6$), the fingerprint is roughly 26 bits regardless of file size.

**Why this works — the key insight.** If $f_1 \neq f_2$, we want $h(f_1) \neq h(f_2)$, i.e., $p \nmid (f_1 - f_2)$. The integer $\lvert f_1 - f_2 \rvert$ is at most $2^n$, so it has at most $n$ distinct prime factors (since each prime is at least 2 and $2^n$ needs $n$ factors of 2 to reach that size). But our random prime is drawn from a pool of size roughly $tn \log(tn) / \ln(tn\log(tn)) \approx tn$ primes (by the Prime Number Theorem). So:

$$\Pr[h(f_1) = h(f_2) \mid f_1 \neq f_2] = \Pr[p \mid (f_1 - f_2)] \leq \frac{n}{\text{number of primes in pool}} \leq \frac{n}{tn} = \frac{1}{t}$$

Set $t = 10^{10}$ and your fingerprint is 96 bits — you're more likely to win the lottery than to get a collision. And the fingerprint fits in 12 bytes.

**Practical use at Microsoft scale.** Azure Blob Storage uses content hashing (MD5/SHA-256 at the block level) as fingerprints for deduplication and integrity checking. Bing's crawl pipeline fingerprints page content to detect when pages have changed between crawl cycles — a changed fingerprint triggers re-indexing; an identical fingerprint skips processing. Git uses SHA-1/SHA-256 fingerprints as object identifiers — a commit hash fingerprints the entire tree state.

```python
import hashlib
import random

def rabin_fingerprint(data: bytes, bits: int = 64) -> int:
    """
    Simplified Rabin fingerprint using a random large prime.
    Real Rabin uses polynomial arithmetic over GF(2^k).
    """
    # In practice, use a cryptographic hash truncated, or
    # polynomial hashing over GF(2^k) for streaming support
    return int(hashlib.md5(data).hexdigest(), 16) % (2 ** bits)

# Actual polynomial rolling hash (Rabin-Karp style)
# Supports sliding window — compute fingerprint of any substring in O(1)
class RollingHash:
    BASE = 257
    MOD = (1 << 61) - 1  # Mersenne prime — fast mod

    def __init__(self, window: int):
        self.window = window
        self.base_pow = pow(self.BASE, window, self.MOD)
        self.h = 0
        self.buf = []

    def update(self, byte: int) -> int:
        self.buf.append(byte)
        self.h = (self.h * self.BASE + byte) % self.MOD
        if len(self.buf) > self.window:
            old = self.buf.pop(0)
            self.h = (self.h - old * self.base_pow) % self.MOD
        return self.h
```

The rolling hash variant (Rabin-Karp) is what `rsync` uses to find matching blocks between two versions of a file — it slides a window and checks fingerprints, enabling delta-sync of changed blocks only.

---

## Hashing Theory {#hashing}

### Universal & Pairwise Independent Hashing {#universal-hashing}

A **uniformly random hash function** $h: U \to \{0, \ldots, m-1\}$ maps every key to a uniformly random slot, and all key-slot assignments are mutually independent. The problem: storing it requires a lookup table of size $\lvert U \rvert$ — for 8-character strings, that's more entries than atoms in the universe.

**Universal hash function** — a weaker, implementable guarantee:

> $h: U \to \{0, \ldots, m-1\}$ is **universal** if for any fixed $x \neq y \in U$:
> $$\Pr[h(x) = h(y)] \leq \frac{1}{m}$$

**Efficient construction:** Let $p$ be a prime with $\lvert U \rvert \leq p \leq 2\lvert U \rvert$. Pick $a \in \{1, \ldots, p-1\}$ and $b \in \{0, \ldots, p-1\}$ uniformly at random:

$$h_{a,b}(x) = \left[(ax + b) \bmod p\right] \bmod m$$

This takes only 2 integers to store (the seed $a, b$), yet gives the universal collision guarantee. This is what goes into CountMin sketch.

**Pairwise independent hash function** — slightly stronger:

> $h$ is **pairwise independent** if for any fixed $x \neq y$ and any $i, j \in \{0,\ldots,m-1\}$:
> $$\Pr[h(x) = i \text{ and } h(y) = j] = \frac{1}{m^2}$$

Same construction as universal hashing (without the restriction $a \neq 0$). Pairwise independence implies universal. Pairwise independence is exactly what you need for linearity of variance to hold — which means you can use cheap hash functions and still get correct variance analyses.

**Why not just use SHA-256?** Cryptographic hash functions (SHA, MD5) are designed for collision *resistance* — an adversary cannot find collisions. For randomized algorithms, you usually need a *random* function from a known distribution to reason probabilistically. SHA-256 is deterministic — you can't compute $\Pr[h(x) = h(y)]$ without knowing the input distribution. Use SHA-256 for security; use pairwise independent hash functions for algorithm analysis.

### MinHash & Jaccard Similarity {#minhash}

**Jaccard similarity** measures overlap between two sets:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Range $[0, 1]$. Fundamental for document similarity, near-duplicate detection, recommendation ("users who liked X also liked Y"), and set intersection estimation. For binary vectors $x, y \in \{0,1\}^d$, Jaccard is the fraction of positions where at least one is 1 that both are 1.

**Problem:** Computing exact Jaccard for two billion-element sets requires $O(\lvert A \rvert + \lvert B \rvert)$ time and space — linear in the set size. For web-scale near-duplicate detection, you'd compare every pair — that's $O(n^2)$ where $n$ is corpus size. You need sketches.

**MinHash** compresses each set to a small signature while preserving Jaccard:

> Let $h: U \to [0,1]$ be a uniformly random hash function. Define $c(A) = \min_{x \in A} h(x)$ — the minimum hash value in the set.
>
> **MinHash Lemma:** $\Pr[c(A) = c(B)] = J(A, B)$

**Why this is true** — beautiful argument. Consider any element in $A \cup B$. The element with the globally smallest hash value in $A \cup B$ (call it $x^*$) is equally likely to be any element in $A \cup B$ (since $h$ is uniformly random). So:

$$\Pr[c(A) = c(B)] = \Pr[x^* \in A \cap B] = \frac{|A \cap B|}{|A \cup B|} = J(A, B)$$

**From one hash to a sketch:** One MinHash value is a Bernoulli with mean $J(A,B)$ — high variance. Use $k$ independent hash functions $h_1, \ldots, h_k$, compute $c_j(A) = \min_{x \in A} h_j(x)$, estimate Jaccard as:

$$\hat{J}(A, B) = \frac{1}{k} \sum_{j=1}^k \mathbf{1}[c_j(A) = c_j(B)]$$

By variance reduction, with $k = O\!\left(\frac{\log(1/\delta)}{\epsilon^2}\right)$ hashes, this gives an $\epsilon$-accurate estimate with probability $1-\delta$.

```python
import hashlib
import struct
from typing import Set

def minhash_signature(items: Set[str], num_hashes: int = 128) -> list[int]:
    """
    Compute MinHash signature. Returns list of min hash values.
    Uses independent hash functions via (a*x + b) mod p.
    """
    p = (1 << 61) - 1  # large Mersenne prime
    import random
    rng = random.Random(42)
    params = [(rng.randint(1, p-1), rng.randint(0, p-1)) for _ in range(num_hashes)]
    
    sig = [float('inf')] * num_hashes
    
    for item in items:
        # Hash item to integer
        x = int(hashlib.md5(item.encode()).hexdigest(), 16) % p
        for j, (a, b) in enumerate(params):
            h = (a * x + b) % p
            if h < sig[j]:
                sig[j] = h
    
    return sig

def jaccard_estimate(sig_a: list, sig_b: list) -> float:
    """Estimate Jaccard from two MinHash signatures."""
    matches = sum(a == b for a, b in zip(sig_a, sig_b))
    return matches / len(sig_a)

# --- Near-duplicate detection at scale ---
# 1. Compute MinHash signatures for all documents
# 2. Documents with Jaccard estimate > 0.8 are near-duplicates
# 3. Band documents into hash buckets to avoid O(n^2) comparisons

def lsh_band(signature: list, num_bands: int, rows_per_band: int) -> list[int]:
    """
    Band the MinHash signature for LSH-based near-dup detection.
    Two docs collide in at least one band iff Jaccard > threshold.
    """
    buckets = []
    for b in range(num_bands):
        band = tuple(signature[b * rows_per_band:(b+1) * rows_per_band])
        bucket_id = hash(band)
        buckets.append(bucket_id)
    return buckets
```

**Microsoft/Bing use case:** Near-duplicate page detection in the web crawl. If two URLs serve nearly identical content (mirrored articles, scraped pages), only one should be indexed. Computing exact Jaccard between 50B pages is impossible. MinHash + banding (LSH) reduces this to: hash each page to a 256-int signature, band it into 32 bands of 8, find all pages that share any band — those are candidate near-duplicates. Total work is $O(n)$ for signatures + $O(\text{candidate pairs})$ for verification.

---

## Johnson-Lindenstrauss & Random Projection {#jl-lemma}

**Problem:** You have $n$ vectors in $\mathbb{R}^d$ (e.g., BERT embeddings with $d = 768$). Every pairwise distance computation costs $O(d)$. Can you compress to much lower dimension $k \ll d$ while preserving all pairwise distances?

**Johnson-Lindenstrauss Lemma (1984).** For any $n$ points $q_1, \ldots, q_n \in \mathbb{R}^d$ and $\epsilon \in (0,1)$, there exists a linear map $\Pi: \mathbb{R}^d \to \mathbb{R}^k$ with:

$$k = O\!\left(\frac{\log n}{\epsilon^2}\right)$$

such that for all pairs $i, j$:

$$(1 - \epsilon)\|q_i - q_j\|^2 \leq \|\Pi q_i - \Pi q_j\|^2 \leq (1 + \epsilon)\|q_i - q_j\|^2$$

The target dimension $k$ depends only on $\log n$ — **not on the original dimension $d$**. For $n = 10^9$ points and $\epsilon = 0.1$: $k \approx \frac{2 \ln 10^9}{0.01} \approx 4{,}150$. You can compress 768-dim BERT embeddings to ~4K dimensions and preserve all pairwise distances to within 10%.

**How $\Pi$ is constructed — remarkably simple.** Draw each entry $\Pi_{ij} \sim \frac{1}{\sqrt{k}} \mathcal{N}(0,1)$ independently. That's it. A random Gaussian matrix works — no data-dependence required (unlike PCA). Other valid choices: $\Pi_{ij} = \frac{1}{\sqrt{k}} \cdot \text{Uniform}(\{+1, -1\})$, or sparse random matrices for faster computation.

**Why it works — proof sketch.** Fix any vector $x \in \mathbb{R}^d$ with $\lVert x \rVert = 1$. The projection $\Pi x \in \mathbb{R}^k$ has each component $(\Pi x)_i = \langle \pi_i, x \rangle$ where $\pi_i$ is the $i$-th row of $\Pi$.

Since each entry of $\pi_i$ is $\mathcal{N}(0, 1/k)$:

$$\langle \pi_i, x \rangle = \sum_{j=1}^d \pi_{ij} x_j \sim \mathcal{N}\!\left(0, \frac{\|x\|^2}{k}\right) = \mathcal{N}\!\left(0, \frac{1}{k}\right)$$

(by stability of Gaussians: sum of independent Gaussians is Gaussian).

Therefore:

$$\|\Pi x\|^2 = \sum_{i=1}^k (\langle \pi_i, x \rangle)^2 = \sum_{i=1}^k \mathcal{N}(0, 1/k)^2$$

This is $\frac{1}{k}$ times a chi-squared distribution with $k$ degrees of freedom. The expected value is exactly 1 ($= \lVert x \rVert^2$). By the chi-squared concentration bound:

$$\Pr\!\left[\left|\|\Pi x\|^2 - \|x\|^2\right| \geq \epsilon \|x\|^2\right] \leq 2e^{-k\epsilon^2/4}$$

Setting $k = O(\log n / \epsilon^2)$ makes this probability $O(1/n^2)$. Union-bound over all $\binom{n}{2} < n^2$ pairs: the distortion guarantee holds for all pairs simultaneously.

**High-dimensional geometry is counterintuitive.** JL might seem impossible — how can you compress $d$ dimensions to $\log n$? The key insight: in high dimensions, there exist exponentially many ($\sim 2^d$) nearly-orthogonal unit vectors. The $n$ data points live in a much lower-dimensional subspace than $d$ suggests — JL finds that subspace randomly.

**Practical implications:**

```python
import numpy as np
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

# Compress 768-dim BERT embeddings to 256-dim
# Preserves cosine similarity to within ~5% for typical corpus sizes

embeddings = np.random.randn(100_000, 768)  # 100K BERT embeddings

# Gaussian JL
proj = GaussianRandomProjection(n_components=256, random_state=42)
compressed = proj.fit_transform(embeddings)  # (100K, 256)

# Sparse JL — 3x faster, same guarantees
sparse_proj = SparseRandomProjection(n_components=256, density=1/3)
compressed_sparse = sparse_proj.fit_transform(embeddings)

# Verify distance preservation
i, j = 0, 1
orig_dist = np.linalg.norm(embeddings[i] - embeddings[j])
proj_dist = np.linalg.norm(compressed[i] - compressed[j])
print(f"Original: {orig_dist:.3f}, Projected: {proj_dist:.3f}, ratio: {proj_dist/orig_dist:.3f}")
# Should be close to 1.0 (within epsilon)
```

**JL in practice for search:**
- Reduces memory: $n \times 768$ float32 = 300MB for 100K vectors → $n \times 256$ = 100MB
- Reduces compute: dot product is 3× cheaper in compressed space
- Does NOT reduce the $O(n)$ scan cost — for that, you need LSH or graph indices

---

## Locality-Sensitive Hashing (LSH) {#lsh}

**The core idea:** Design a hash function where the collision probability between two items is *high* if they're similar and *low* if they're dissimilar. Then build a hash table — colliding items are candidates for nearest neighbors, and you only need to check those.

> A family $\mathcal{H}$ of hash functions $h: \mathcal{X} \to \{0,\ldots,m-1\}$ is **locality-sensitive** for similarity $s(x,y)$ if:
> - $\Pr[h(x) = h(y)]$ is *higher* when $s(x,y)$ is higher (similar items likely collide)
> - $\Pr[h(x) = h(y)]$ is *lower* when $s(x,y)$ is lower (dissimilar items rarely collide)

This is the opposite of a cryptographic hash — you *want* similar inputs to collide.

### LSH for Jaccard Similarity {#lsh-jaccard}

MinHash is itself an LSH function:

$$\Pr[h_{\text{min}}(x) = h_{\text{min}}(y)] = J(x, y)$$

**Banding to amplify discrimination.** A single MinHash has probability $J(x,y)$ of collision — not sharp enough. If $J = 0.8$ and $J = 0.5$, collision probabilities are 0.8 and 0.5 — hard to tell apart. The **banding trick** creates a sharper threshold:

Concatenate $r$ MinHash values into a "band." Two vectors match the band if and only if all $r$ values agree:

$$\Pr[\text{band match}] = J(x,y)^r$$

Build $t$ independent bands. Two vectors collide in *at least one* band (and thus are candidates) with probability:

$$\Pr[\text{any band matches}] = 1 - (1 - J^r)^t$$

This is an **S-curve** in $J$ — increasing $r$ and $t$ makes the curve steeper:

```
J = 0.3:  1 - (1 - 0.3^5)^20 ≈ 1 - (1 - 0.00243)^20 ≈ 0.047  → rarely found
J = 0.5:  1 - (1 - 0.5^5)^20 ≈ 1 - (1 - 0.031)^20  ≈ 0.47   → sometimes found
J = 0.8:  1 - (1 - 0.8^5)^20 ≈ 1 - (1 - 0.328)^20  ≈ 0.9998 → almost always found
```

With $r=5, t=20$ bands (100 MinHash values total), items with $J > 0.8$ are almost always retrieved and items with $J < 0.3$ are almost always skipped. This is what makes **deduplication at web scale** efficient.

**Shazam example** (from Musco's lecture): 1 million audio clips. True matches have $J > 0.9$, near-matches have $J \in [0.5, 0.9]$, everything else $J < 0.5$. With $r=4, t=10$:
- True matches: $1-(1-0.9^4)^{10} = 1-(1-0.656)^{10} \approx 0.9998$ — almost never miss
- Near-matches: $\approx 0.20$ — check some, not all
- Non-matches: $\approx 0.004$ — almost never check

Total items checked per query: far fewer than 1 million, with near-perfect recall.

### SimHash for Cosine Similarity {#simhash}

For vectors in $\mathbb{R}^d$ (e.g., TF-IDF or embedding vectors), the natural similarity is cosine similarity $\cos(\theta(x,y)) = \frac{\langle x, y \rangle}{\lVert x \rVert \lVert y \rVert}$.

**SimHash** is an LSH function for cosine similarity. Draw a random Gaussian vector $g \in \mathbb{R}^d$ (each entry i.i.d. $\mathcal{N}(0,1)$). Define:

$$h(x) = \text{sign}(\langle g, x \rangle)$$

**Key result:**

$$\Pr[h(x) = h(y)] = 1 - \frac{\theta(x,y)}{\pi}$$

where $\theta(x,y) = \cos^{-1}(\cos\theta)$ is the angle between $x$ and $y$.

**Why:** $\text{sign}(\langle g, x\rangle) \neq \text{sign}(\langle g, y\rangle)$ when $g$ separates $x$ and $y$ with a hyperplane orthogonal to $g$. In 2D, this probability equals $\theta/\pi$ — the fraction of random directions that separate them. A rotation argument shows this holds in arbitrary dimensions.

This is the algorithm behind **Google's Web Search spam detection** and **near-duplicate web page detection**. Hash each document's TF-IDF vector with SimHash, store the hash, detect duplicates as documents with small Hamming distance between hashes.

```python
import numpy as np

def simhash(vector: np.ndarray, num_bits: int = 64) -> int:
    """
    Compute SimHash of a vector. Each bit corresponds to sign of a random projection.
    """
    rng = np.random.RandomState(42)
    projections = rng.randn(num_bits, len(vector))  # (num_bits, d)
    signs = np.sign(projections @ vector)            # (num_bits,)
    # Convert +1/-1 to bits
    bits = (signs > 0).astype(int)
    return int(''.join(map(str, bits)), 2)

def hamming_distance(h1: int, h2: int, num_bits: int = 64) -> int:
    return bin(h1 ^ h2).count('1')

def cosine_estimate_from_simhash(h1: int, h2: int, num_bits: int = 64) -> float:
    """Estimate cosine similarity from SimHash Hamming distance."""
    d = hamming_distance(h1, h2, num_bits)
    theta_over_pi = d / num_bits
    return np.cos(theta_over_pi * np.pi)

# Banded SimHash (r random projections per band)
def banded_simhash(vector: np.ndarray, r: int = 5, t: int = 20) -> list[int]:
    """Return t band hashes. Two items are candidates if any band matches."""
    rng = np.random.RandomState(0)
    projections = rng.randn(r * t, len(vector))
    signs = np.sign(projections @ vector)  # (r*t,)
    
    band_hashes = []
    for b in range(t):
        band_bits = tuple(signs[b*r:(b+1)*r])
        band_hashes.append(hash(band_bits))
    return band_hashes
```

**Banded SimHash** is what Bing uses for near-duplicate detection in the crawl pipeline: each page gets a 64-bit SimHash, pages with Hamming distance ≤ 3 are considered near-duplicates. The banding allows sub-linear candidate retrieval.

### LSH-Based ANN: Theory {#lsh-theory}

**Formal guarantee** (Indyk-Motwani 1998). Given a distance threshold $R$ and approximation ratio $C > 1$:

If there exists some database vector $q$ with $\lVert q - y \rVert \leq R$, return a vector $\tilde{q}$ with $\lVert \tilde{q} - y \rVert \leq C \cdot R$ in:
- **Time:** $\tilde{O}(n^{1/C^2})$ (sub-linear in $n$!)
- **Space:** $\tilde{O}(n^{1+1/C^2} + nd)$

For $C = 2$ (accept 2× approximate answer): query time is $O(n^{0.25})$ — much faster than linear scan. The catch: space grows as $n^{1.25}$, which is significant but manageable.

**Multi-level LSH for nearest-neighbor search.** Build hash tables for exponentially growing distance radii $R, 2R, 4R, \ldots$ Search from finest to coarsest until a candidate is found. Total number of levels: $O(\log(d_{\max}/d_{\min}))$ where $d_{\max}/d_{\min}$ is the "dynamic range" — the ratio of largest to smallest pairwise distance in the dataset.

---

## ANN Search in Practice {#ann-practice}

Theory gives you sub-linear query time with exponential space. In practice, you want to search a billion vectors in 10ms on a single machine with 64GB RAM. That requires engineering on top of the theory.

### FAISS & Product Quantization {#faiss}

[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is the most widely used ANN library. It combines dimensionality reduction (JL/PCA), quantization, and inverted index structures.

**Product Quantization (PQ).** Compress each $d$-dimensional vector into a short code by:
1. Split the $d$ dimensions into $M$ subspaces of $d/M$ dimensions each
2. For each subspace, train a $k$-means codebook with $K$ centroids
3. Encode each vector as $M$ indices, one per subspace

A 768-dim float32 vector (3072 bytes) becomes $M$ bytes with $M=8, K=256$. At query time, precompute distances from the query to all centroids in each subspace, then look up and sum those distances — $O(M \cdot K)$ precomputation + $O(M)$ per database vector.

**IVF (Inverted File Index).** Split the database into $n_{\text{cells}}$ Voronoi cells using k-means. At query time, probe only the nearest $n_{\text{probe}}$ cells (not all $n_{\text{cells}}$). This gives $O(n/n_{\text{cells}} \cdot n_{\text{probe}})$ candidates instead of $O(n)$.

```python
import faiss
import numpy as np

d = 768  # BERT embedding dimension
n = 1_000_000  # 1M vectors

# Generate synthetic data
data = np.random.randn(n, d).astype('float32')
faiss.normalize_L2(data)  # for cosine similarity

# ----- Option 1: Flat (exact, brute force) -----
index_flat = faiss.IndexFlatIP(d)  # Inner product = cosine after normalisation
index_flat.add(data)

# ----- Option 2: IVF + PQ (compressed, approximate) -----
nlist = 1000        # number of Voronoi cells
M = 16              # number of subspaces for PQ
nbits = 8           # bits per subspace → 256 centroids per subspace

quantizer = faiss.IndexFlatIP(d)
index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)

# Train on a representative sample
index_ivfpq.train(data[:100_000])
index_ivfpq.add(data)

# At query time
index_ivfpq.nprobe = 50  # probe 50/1000 cells — tune recall/speed tradeoff

queries = np.random.randn(100, d).astype('float32')
faiss.normalize_L2(queries)

k = 10
distances, indices = index_ivfpq.search(queries, k)

# ----- GPU acceleration -----
# FAISS has GPU implementations — 10-100x faster for brute force
# Useful when you have many queries per second
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
D, I = gpu_index.search(queries, k)
```

**At Bing/Azure scale:** Azure Cognitive Search uses a form of IVF-PQ under the hood for its vector search feature. The typical configuration: 1024-dim embeddings, 96 subspaces for PQ, 2048 Voronoi cells, probing 50 cells at query time. This achieves ~92% recall@10 at ~5ms query time for 100M vector corpora.

### HNSW: Graph-Based Search {#hnsw}

**Hierarchical Navigable Small World (HNSW)** is the dominant algorithm in production vector search (used by Qdrant, Weaviate, Pinecone, pgvector, Elasticsearch). It builds a multi-layer graph where short-range edges appear at low layers and long-range edges at high layers — a navigable small-world structure inspired by Milgram's six-degrees experiment.

**Structure:**
```
Layer 2 (sparse):  [q1] ——————————— [q5]
                     \                 /
Layer 1 (medium):   [q1]—[q2]  [q4]—[q5]
                          |    |
Layer 0 (dense):  [q1]-[q2]-[q3]-[q4]-[q5]-...
```

**Search (greedy routing):** Start at entry point on the top layer. Greedily move to the neighbor closest to the query. Descend to the next layer when no improvement can be made. Repeat until Layer 0. Collect the $k$ nearest neighbors at Layer 0.

**Insert:** Assign each new node a random layer $l \sim \lfloor -\ln(\text{Uniform}(0,1)) \cdot m_L \rfloor$ (exponentially distributed). Connect to the $M$ nearest neighbors at each layer from 0 to $l$.

```python
import hnswlib
import numpy as np

d = 768
n = 1_000_000

data = np.random.randn(n, d).astype('float32')
ids = np.arange(n)

# Build index
index = hnswlib.Index(space='cosine', dim=d)
index.init_index(
    max_elements=n,
    ef_construction=200,  # size of dynamic candidate list during construction
    M=16,                  # number of edges per node per layer
)

# Add vectors (can be done in batches)
index.add_items(data, ids, num_threads=8)

# Set ef for search — controls recall/speed tradeoff at query time
index.set_ef(50)  # must be >= k

# Query
query = np.random.randn(1, d).astype('float32')
labels, distances = index.knn_query(query, k=10)

# Save and load
index.save_index("hnsw_index.bin")
index.load_index("hnsw_index.bin", max_elements=n)
```

**HNSW vs LSH in practice:**

| | LSH | HNSW |
|---|---|---|
| Query time | $O(n^{1/C^2})$ worst case | $O(\log n)$ empirical |
| Build time | $O(n)$ | $O(n \log n)$ |
| Memory | $O(n^{1+1/C^2})$ | $O(n \cdot M)$ |
| Recall@10 | 80–90% | 95–99% |
| Updates | Easy | Easy (insert only; deletion harder) |
| Worst case guarantee | Yes | No (empirical) |

In practice HNSW almost always wins on recall-speed tradeoff. LSH retains value when you need hard theoretical guarantees or when the dataset is so large that HNSW's $O(nM)$ memory is prohibitive.

### DiskANN: Billion-Scale on Disk {#diskann}

[DiskANN](https://github.com/microsoft/DiskANN) (Microsoft Research, 2019) — the algorithm powering Azure's vector search at billion-scale. The key insight: HNSW keeps the entire graph in RAM (impossible for 1B × 768 float32 = 3TB). DiskANN builds a graph index that sits on NVMe SSD and uses a memory-resident compressed representation for fast routing.

**Algorithm:**
1. Compress vectors with PQ for fast in-memory distance estimates
2. Build a **Vamana graph** — a graph with bounded max-degree and good navigability — on the full-precision vectors on disk
3. At query time: use PQ distances for greedy routing (in RAM), fetch actual vectors from SSD only for the final candidates

**Why it works:** NVMe SSDs can do ~100K random 4KB reads/second at ~100μs latency. HNSW on 1B vectors needs ~50 hops per query, each potentially a cache miss → 5ms just in I/O. Vamana's graph has higher branching factor and shorter paths — fewer disk reads per query.

```python
# DiskANN Python interface (via diskannpy)
import diskannpy
import numpy as np

d = 768
n = 1_000_000_000  # 1B vectors

# Build index (offline, writes to disk)
diskannpy.build_disk_index(
    data="vectors.bin",           # raw float32 binary
    metric="mip",                 # maximum inner product
    index_directory="diskann_idx/",
    complexity=128,               # build-time beam width (higher = better quality)
    graph_degree=64,              # max edges per node
    search_memory_limit=8.0,      # GB of RAM for PQ compressed vectors
    index_memory_limit=0.2,       # GB for medoid + graph metadata
    num_threads=64,
)

# Query
index = diskannpy.DiskIndex(
    index_directory="diskann_idx/",
    num_threads=8,
    num_nodes_to_cache=500_000,   # cache hot nodes in RAM
)

queries = np.random.randn(1000, d).astype('float32')
k = 10
beam_width = 64  # larger = better recall, more I/O

neighbors, distances = index.search(queries, k, beam_width)
```

**Microsoft deployment numbers:** Azure Cognitive Search uses DiskANN to serve billion-scale vector search at <20ms p99 latency, ~96% recall@10, on commodity NVMe SSDs — without requiring the full index in RAM.

---

## Streaming Algorithms {#streaming}

**The streaming setting:** Data arrives as a sequence $x_1, x_2, \ldots, x_n$ (a "stream"). $n$ is massive — you cannot store all elements. You have $O(\text{polylog}(n))$ space, process each item once, and must answer queries approximately.

This matters for: real-time click counting, network traffic analysis, query frequency estimation, user behaviour analytics — any "analytics over massive event streams."

### Count-Min Sketch {#count-min}

**Problem:** A stream of items from universe $U$. For each item $x$, maintain an estimate of its frequency $f(x) = \lvert\{i : x_i = x\}\rvert$.

**Exact solution:** Hash map — $O(\lvert U \rvert)$ space in the worst case (all items distinct). For $\lvert U \rvert = 10^{12}$ (all possible IP address pairs), this is prohibitive.

**Count-Min Sketch** ([Cormode & Muthukrishnan 2003](https://dl.acm.org/doi/10.1145/762471.762473)):

Maintain a $t \times m$ table of counters $C[1..t][0..m-1]$, initialised to zero. Use $t$ pairwise-independent hash functions $h_1, \ldots, h_t: U \to \{0, \ldots, m-1\}$.

**Update:** for each stream item $x$, increment $C[j][h_j(x)]$ for all $j = 1, \ldots, t$.

**Query:** $\hat{f}(x) = \min_{j=1}^t C[j][h_j(x)]$

The min over rows eliminates most of the hash collision noise — any overcount in row $j$ comes from items that collide with $x$ under $h_j$, and it's unlikely all $t$ hash functions produce overcounters simultaneously.

**Guarantee:** With $m = e/\epsilon$ columns and $t = \ln(1/\delta)$ rows:

$$f(x) \leq \hat{f}(x) \leq f(x) + \epsilon \cdot \|f\|_1$$

with probability $\geq 1-\delta$, where $\lVert f \rVert_1 = n$ (total stream length). Space: $O\!\left(\frac{\log(1/\delta)}{\epsilon}\right)$ counters — independent of $\lvert U \rvert$ and $n$.

```python
import math
import hashlib

class CountMinSketch:
    def __init__(self, epsilon: float = 0.01, delta: float = 0.01):
        self.width = math.ceil(math.e / epsilon)     # columns
        self.depth = math.ceil(math.log(1 / delta))  # rows
        self.table = [[0] * self.width for _ in range(self.depth)]
        # Pairwise independent hash functions via (a*x + b) mod p
        import random
        rng = random.Random(42)
        p = (1 << 31) - 1  # prime
        self.params = [
            (rng.randint(1, p-1), rng.randint(0, p-1))
            for _ in range(self.depth)
        ]
        self.p = p

    def _hash(self, item: str, row: int) -> int:
        x = int(hashlib.md5(item.encode()).hexdigest(), 16) % self.p
        a, b = self.params[row]
        return (a * x + b) % self.p % self.width

    def update(self, item: str, count: int = 1):
        for row in range(self.depth):
            col = self._hash(item, row)
            self.table[row][col] += count

    def query(self, item: str) -> int:
        return min(
            self.table[row][self._hash(item, row)]
            for row in range(self.depth)
        )

    def heavy_hitters(self, threshold: int) -> list:
        """Items that appear > threshold times (not exact — requires tracking)."""
        # In practice, combine with a heap of candidates
        pass

# Usage: real-time query frequency estimation
sketch = CountMinSketch(epsilon=0.001, delta=0.001)

queries = ["microsoft", "azure", "bing", "azure", "microsoft", "microsoft"]
for q in queries:
    sketch.update(q)

print(sketch.query("microsoft"))  # ≈ 3
print(sketch.query("bing"))       # ≈ 1
```

**Applications at Microsoft scale:**
- **Bing query frequency:** Count-Min tracks query frequencies in real time for autocomplete ranking, without storing all $10^9$ unique queries. Space: $\sim$1MB for $\epsilon=0.001, \delta=0.01$.
- **Azure DDoS detection:** Count-Min tracks IP address hit rates per second. If any IP exceeds threshold, trigger mitigation.
- **Network traffic analysis:** Router logs — count packet frequencies per destination prefix.

**Heavy hitters (top-$k$ items):** Combine Count-Min with a min-heap of size $k$. After each update, if the estimate of $x$ exceeds the heap's minimum, insert $x$. This finds all items with frequency $> n/k$ in $O(\log k)$ per element.

### Distinct Elements: Flajolet-Martin & HyperLogLog {#distinct-elements}

**Problem:** Count the number of distinct elements in a stream. Exactly requires $O(D)$ space (store all seen items). For $D = 10^9$ unique users, this is 4GB just for IDs.

**Flajolet-Martin Algorithm:**

Choose a random hash function $h: U \to [0,1]$. Maintain $S = \min_{x \in \text{stream}} h(x)$ — the running minimum hash value.

**Key lemma:** $\mathbb{E}[S] = \frac{1}{D+1}$

**Why:** By symmetry, the smallest value out of $D$ uniform $[0,1]$ random variables has expected value $1/(D+1)$ — it's the first order statistic of $D$ uniforms. So the inverse of the minimum hash value estimates $D$:

$$\hat{D} = \frac{1}{S} - 1$$

**Variance reduction:** One estimator has too much noise ($\text{Var}[S] \approx \mu^2$, so relative std is 100%). Use $k$ independent hash functions, compute $k$ estimates, average them:

- $k$ estimators: $\text{Var}[\bar{S}] = \sigma^2/k$ → relative std $\approx 1/\sqrt{k}$
- By Chebyshev: $\Pr[\lvert \hat{D} - D \rvert \geq \epsilon D] \leq \delta$ with $k = O\!\left(\frac{1}{\epsilon^2 \delta}\right)$
- Space: $O\!\left(\frac{1}{\epsilon^2 \delta}\right)$ hash values

**HyperLogLog** (the production version):

Instead of storing the minimum hash value (a float requiring many bits), HyperLogLog stores the **maximum number of leading zeros** in the binary hash:

$$\text{For each distinct item } x: \quad m \leftarrow \max(m, \text{leadingZeros}(h(x)))$$

If there are $D$ distinct items, the expected maximum leading zeros is $\approx \log_2 D$. So $2^m$ estimates $D$. Using $b$ hash bits for bucketing:

$$\hat{D} = \alpha_m \cdot m^2 \cdot \left(\sum_{j=1}^m 2^{-M_j}\right)^{-1}$$

where $m = 2^b$ buckets and $\alpha_m$ is a bias correction constant.

**Space: $O(\log \log D + \log D)$ bits** — for $D = 10^{12}$, this is about 10 bits. HyperLogLog++ (Google) achieves 1.625% error with 1.5KB.

```python
import hashlib
import math

class HyperLogLog:
    def __init__(self, b: int = 10):
        """
        b: number of bits for bucketing (2^b buckets)
        Error rate ≈ 1.04 / sqrt(2^b)
        b=10 → 1024 buckets → ~3.2% error, ~1KB space
        b=14 → 16384 buckets → ~0.8% error, ~16KB space
        """
        self.b = b
        self.m = 1 << b  # number of registers
        self.registers = [0] * self.m
        # Bias correction constant
        if self.m >= 128:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)
        elif self.m == 64:
            self.alpha = 0.709
        elif self.m == 32:
            self.alpha = 0.697
        else:
            self.alpha = 0.673

    def _hash(self, item: str) -> int:
        return int(hashlib.sha256(item.encode()).hexdigest(), 16)

    def add(self, item: str):
        h = self._hash(item)
        # Use first b bits as bucket index
        bucket = h >> (128 - self.b)
        # Use remaining bits to count leading zeros
        remaining = h & ((1 << (128 - self.b)) - 1)
        # Count leading zeros + 1
        if remaining == 0:
            rho = 128 - self.b
        else:
            rho = (128 - self.b) - remaining.bit_length() + 1
        self.registers[bucket] = max(self.registers[bucket], rho)

    def count(self) -> int:
        Z = sum(2 ** (-r) for r in self.registers)
        estimate = self.alpha * self.m ** 2 / Z
        # Small range correction
        if estimate <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros > 0:
                estimate = self.m * math.log(self.m / zeros)
        return int(estimate)

    def merge(self, other: 'HyperLogLog') -> 'HyperLogLog':
        """Merge two HyperLogLog sketches — critical for distributed settings."""
        result = HyperLogLog(self.b)
        result.registers = [max(a, b) for a, b in zip(self.registers, other.registers)]
        return result

# Usage
hll = HyperLogLog(b=12)  # ~0.8MB, ~1.6% error

# Add 1M items
for i in range(1_000_000):
    hll.add(f"user_{i}")

print(f"Estimated distinct users: {hll.count():,}")  # ≈ 1,000,000

# Mergeability is the killer feature for distributed settings
# Count distinct users across 100 shards: just merge 100 HLLs
```

**HyperLogLog in production:** Google, Facebook, Twitter, Amazon Redshift, PostgreSQL all implement HyperLogLog. Use cases: distinct user counts for A/B tests, unique search query counts, distinct product views per SKU. The mergeability property is key — you can compute per-shard HLLs and merge them for global estimates without sharing raw data.

### Bloom Filters {#bloom-filters}

**Problem:** Given a set $S$, answer membership queries "is $x \in S$?" quickly and compactly. An exact hash set requires $O(\lvert S \rvert)$ space. Can you do better with a small false positive rate?

**Bloom Filter** ([Burton Howard Bloom, 1970](https://dl.acm.org/doi/10.1145/362686.362692)):

Maintain a bit array $B$ of $m$ bits, all initialised to 0. Use $k$ independent hash functions $h_1, \ldots, h_k: U \to \{0, \ldots, m-1\}$.

**Insert $x$:** Set $B[h_j(x)] = 1$ for all $j = 1, \ldots, k$.

**Query $x$:** Return "yes" iff $B[h_j(x)] = 1$ for all $j = 1, \ldots, k$.

**Guarantees:**
- **No false negatives:** if $x \in S$, all its bits are set → always returns "yes"
- **False positives possible:** if $x \notin S$, all $k$ bit positions might happen to be set by other elements

**False positive probability** with $n$ inserted elements:

$$\Pr[\text{false positive}] \approx \left(1 - e^{-kn/m}\right)^k$$

Optimal $k = (m/n) \ln 2$ gives:

$$\Pr[\text{FP}] \approx (0.6185)^{m/n}$$

For $m/n = 10$ bits per element: ~0.8% false positive rate. For $m/n = 15$ bits: ~0.02%.

```python
import math
import mmh3  # MurmurHash3 — fast non-cryptographic hash
from bitarray import bitarray

class BloomFilter:
    def __init__(self, capacity: int, false_positive_rate: float = 0.01):
        """
        capacity: expected number of elements
        false_positive_rate: acceptable false positive probability
        """
        self.capacity = capacity
        self.fp_rate = false_positive_rate
        # Optimal number of bits
        self.m = math.ceil(-capacity * math.log(false_positive_rate) / (math.log(2) ** 2))
        # Optimal number of hash functions
        self.k = math.ceil((self.m / capacity) * math.log(2))
        self.bits = bitarray(self.m)
        self.bits.setall(0)
        self.count = 0

    def add(self, item: str):
        for seed in range(self.k):
            pos = mmh3.hash(item, seed) % self.m
            self.bits[pos] = 1
        self.count += 1

    def __contains__(self, item: str) -> bool:
        return all(
            self.bits[mmh3.hash(item, seed) % self.m]
            for seed in range(self.k)
        )

    def __len__(self):
        return self.count

    @property
    def size_bytes(self) -> int:
        return self.m // 8

# Counting Bloom Filter — supports deletions
class CountingBloomFilter:
    def __init__(self, capacity: int, fp_rate: float = 0.01, bits_per_counter: int = 4):
        m = math.ceil(-capacity * math.log(fp_rate) / (math.log(2) ** 2))
        self.k = math.ceil((m / capacity) * math.log(2))
        self.max_count = (1 << bits_per_counter) - 1
        self.counters = [0] * m
        self.m = m

    def add(self, item: str):
        for seed in range(self.k):
            pos = mmh3.hash(item, seed) % self.m
            if self.counters[pos] < self.max_count:
                self.counters[pos] += 1

    def remove(self, item: str):
        for seed in range(self.k):
            pos = mmh3.hash(item, seed) % self.m
            if self.counters[pos] > 0:
                self.counters[pos] -= 1

    def __contains__(self, item: str) -> bool:
        return all(self.counters[mmh3.hash(item, seed) % self.m] > 0
                   for seed in range(self.k))
```

**Applications at Microsoft scale:**
- **Bing URL deduplication:** Before fetching a URL, check if it's already indexed. A Bloom filter of 100B indexed URLs takes ~200GB (2 bits/URL) — versus storing the full URL set which would need terabytes. False positives just mean occasionally skipping a valid re-crawl — acceptable.
- **Azure Cache-aside:** Before a database lookup, check the Bloom filter to skip lookups for keys that definitely don't exist. Eliminates "cache miss for non-existent key" round-trips.
- **Database query optimisation:** Cassandra, HBase, LevelDB all use Bloom filters to avoid disk reads for keys that don't exist in a SSTable.

---

## Putting It Together: Search Pipeline {#putting-together}

Modern neural search at Microsoft (Bing, Azure Cognitive Search, GitHub Copilot retrieval) layers these algorithms:

```
Query: "transformer attention mechanism"
         │
         ▼
┌─────────────────────────────────┐
│  Embedding (BERT/E5/Ada-002)    │  768-dim vector
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  JL Random Projection           │  768 → 256 dim (3x speedup)
└────────────────┬────────────────┘
                 │
         ┌───────┴───────┐
         ▼               ▼
┌────────────────┐ ┌─────────────────┐
│  HNSW Graph    │ │  IVF-PQ Index   │
│  (online)      │ │  (offline/disk) │
└────────┬───────┘ └────────┬────────┘
         └────────┬──────────┘
                  │ top-100 candidates
                  ▼
┌─────────────────────────────────┐
│  Exact re-ranking (full vectors)│  cosine similarity
└────────────────┬────────────────┘
                 │ top-10 results
                 ▼
┌─────────────────────────────────┐
│  Diversity / MMR filtering      │
└─────────────────────────────────┘
```

**Bloom filter:** Before embedding, check if query URL exists in Bloom-filtered crawl index. Skip embedding for obvious cache hits.

**Count-Min Sketch:** Track query frequencies in real time. Hot queries go to a warm cache; cold queries run the full pipeline.

**HyperLogLog:** Count distinct users per query cluster for relevance learning. Mergeable across data centres.

**MinHash + banding:** Deduplicate documents in the index. Two documents with Jaccard > 0.85 keep only one.

**SimHash:** Detect near-duplicate search results before displaying. If result 3 and result 7 have Hamming distance < 5 in their SimHash, suppress one.

**DiskANN:** At billion-scale, the HNSW graph moves to disk. Query routing stays in RAM via PQ-compressed vectors; final distance computation fetches from NVMe.

---

## Key Tradeoffs and Intuitions

| Algorithm | What it approximates | Space | Error guarantee | Merges? |
|---|---|---|---|---|
| Rabin Fingerprint | File identity | $O(\log n)$ bits | $1/t$ collision prob | N/A |
| MinHash | Jaccard similarity | $O(k)$ | $O(1/\sqrt{k})$ | Yes (union) |
| JL Projection | Euclidean distance | $O(k \cdot d)$ | $(1\pm\epsilon)$ multiplicative | Yes |
| SimHash | Cosine similarity | $O(\log n)$ bits | $\theta/\pi$ probability | Yes (OR) |
| Count-Min | Frequency | $O(\frac{\log(1/\delta)}{\epsilon})$ | $+\epsilon n$ additive | Yes (add) |
| Flajolet-Martin | Distinct count | $O(\frac{1}{\epsilon^2\delta})$ | $(1\pm\epsilon)$ multiplicative | Yes (min) |
| HyperLogLog | Distinct count | $O(\log\log D)$ | ~1.04/$\sqrt{m}$ | Yes (max) |
| Bloom Filter | Set membership | $O(n \cdot \frac{\log(1/\delta)}{\log 2})$ bits | $\delta$ false positive | No (standard) |

**The unifying theme:** These algorithms all trade a small, analytically bounded error for exponential savings in space and time. The error is not a bug — it's a design parameter you tune to your application's tolerance. A Bloom filter with 1% false positive rate is not "wrong" — it correctly trades 100× memory savings for 1% unnecessary work.

At Microsoft scale, these aren't academic exercises. The difference between $O(n)$ and $O(\sqrt{n})$ query time on a billion-vector index is the difference between 100ms and 1ms. The difference between $O(n)$ and $O(\log \log n)$ space for distinct element counting is the difference between terabytes and kilobytes. Randomized algorithms make large-scale search economically possible.
