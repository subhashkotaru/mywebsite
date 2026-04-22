---
title: "ML System Design Interview"
date: 2026-04-21
description: "A complete interview framework for ML system design — recommender systems, fraud detection, and agentic AI systems — with worked examples covering data pipelines, feature engineering, model selection, training, evaluation, serving, and failure handling."
tags: [ml-systems, system-design, interviews, machine-learning]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#framework">The Interview Framework</a>
      <ul class="post-toc-sublist">
        <li><a href="#framework-steps">The Eleven-Step Template</a></li>
        <li><a href="#time-management">Time Management</a></li>
        <li><a href="#what-interviewers-want">What Interviewers Are Evaluating</a></li>
      </ul>
    </li>
    <li><a href="#patterns">Common ML System Design Patterns</a>
      <ul class="post-toc-sublist">
        <li><a href="#two-tower">Two-Tower Architecture</a></li>
        <li><a href="#embedding-retrieval">Embedding Retrieval & ANN</a></li>
        <li><a href="#feature-stores">Feature Stores</a></li>
        <li><a href="#stream-vs-batch">Stream vs Batch Processing</a></li>
        <li><a href="#online-learning">Online Learning</a></li>
      </ul>
    </li>
    <li><a href="#recommender">Example 1: Recommender System (YouTube)</a>
      <ul class="post-toc-sublist">
        <li><a href="#rec-scoping">Problem Scoping</a></li>
        <li><a href="#rec-architecture">High-Level Architecture</a></li>
        <li><a href="#rec-data">Data Sources & Pipeline</a></li>
        <li><a href="#rec-features">Feature Engineering</a></li>
        <li><a href="#rec-model">Model Selection & Architecture</a></li>
        <li><a href="#rec-training">Training Pipeline</a></li>
        <li><a href="#rec-evaluation">Evaluation</a></li>
        <li><a href="#rec-serving">Serving & Inference</a></li>
        <li><a href="#rec-degradation">Graceful Degradation</a></li>
        <li><a href="#rec-failures">Failure Modes</a></li>
        <li><a href="#rec-monitoring">Monitoring</a></li>
      </ul>
    </li>
    <li><a href="#fraud">Example 2: Fraud Detection</a>
      <ul class="post-toc-sublist">
        <li><a href="#fraud-scoping">Problem Scoping</a></li>
        <li><a href="#fraud-architecture">High-Level Architecture</a></li>
        <li><a href="#fraud-data">Data Sources & Pipeline</a></li>
        <li><a href="#fraud-features">Feature Engineering</a></li>
        <li><a href="#fraud-model">Model Selection & Architecture</a></li>
        <li><a href="#fraud-training">Training Pipeline</a></li>
        <li><a href="#fraud-evaluation">Evaluation</a></li>
        <li><a href="#fraud-serving">Serving & Inference</a></li>
        <li><a href="#fraud-degradation">Graceful Degradation</a></li>
        <li><a href="#fraud-failures">Failure Modes</a></li>
        <li><a href="#fraud-monitoring">Monitoring</a></li>
      </ul>
    </li>
    <li><a href="#agent">Example 3: AI Coding Assistant (Agentic System)</a>
      <ul class="post-toc-sublist">
        <li><a href="#agent-scoping">Problem Scoping</a></li>
        <li><a href="#agent-architecture">High-Level Architecture</a></li>
        <li><a href="#agent-data">Data Sources & Pipeline</a></li>
        <li><a href="#agent-features">Feature Engineering & Context Management</a></li>
        <li><a href="#agent-model">Model Selection & Architecture</a></li>
        <li><a href="#agent-training">Training Pipeline</a></li>
        <li><a href="#agent-evaluation">Evaluation</a></li>
        <li><a href="#agent-serving">Serving & Inference</a></li>
        <li><a href="#agent-degradation">Graceful Degradation</a></li>
        <li><a href="#agent-failures">Failure Modes</a></li>
        <li><a href="#agent-monitoring">Monitoring</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## The Interview Framework
{: #framework}

ML system design interviews test whether you can translate a vague product requirement into a concrete, end-to-end machine learning system — one that works at scale, degrades gracefully, and can be monitored and improved over time. The interview is not about memorising architectures; it is about demonstrating structured thinking under ambiguity.

The single most important thing to do in the first two minutes: **ask clarifying questions**. The problem statement is always underspecified. What counts as a "good" recommendation? What is the acceptable latency? What is the training data situation? The answers reshape every subsequent decision. Interviewers who skip this and jump straight to modeling are penalised — not because questions are magic, but because the answers change the design.

### The Eleven-Step Template
{: #framework-steps}

Every ML system design answer can be structured around eleven steps in sequence. Each step feeds into the next. Skipping steps is the most common failure mode.

| Step | What to cover | Why it matters |
|---|---|---|
| 1. Problem scoping | Clarifying questions, constraints, scale, latency | Everything downstream depends on this |
| 2. High-level architecture | ASCII diagram, major components | Shows you can see the whole system |
| 3. Data sources & pipeline | Where data comes from, how it flows | Data is the most underrated part of ML |
| 4. Feature engineering | What features, how computed, embeddings | Often separates good from great answers |
| 5. Model selection | Architecture choice, why not others | Demonstrates ML breadth |
| 6. Training pipeline | Offline vs online, retraining cadence | Shows production awareness |
| 7. Evaluation | Offline metrics, online metrics, A/B testing | Shows you know how to measure success |
| 8. Serving & inference | Latency budget, caching, batching | Shows awareness of production constraints |
| 9. Graceful degradation | Fallbacks at each layer | Shows reliability thinking |
| 10. Failure modes | What breaks, detection, remediation | Shows operational maturity |
| 11. Monitoring | Metrics, alert thresholds, dashboards | Shows long-term thinking |

### Time Management
{: #time-management}

A typical ML system design interview is 45–60 minutes. The distribution of time that works:

| Phase | Time | What to do |
|---|---|---|
| Scoping & clarification | 5–8 min | Ask questions, state assumptions explicitly |
| High-level architecture | 5 min | Draw the diagram, name every box |
| Data & features | 10 min | Be specific — name the features, transformations |
| Model & training | 10 min | Equations if useful, justify the choice |
| Evaluation & serving | 8 min | Offline and online metrics, latency budget |
| Degradation, failures, monitoring | 7 min | At least one concrete failure per layer |
| Q&A / deep dives | remaining | Follow the interviewer's direction |

Spend too long on model architecture and you will not reach serving and monitoring — two sections that often separate passing from failing.

### What Interviewers Are Evaluating
{: #what-interviewers-want}

| Dimension | Positive signal | Negative signal |
|---|---|---|
| Problem framing | Clarifies before designing | Jumps to solution immediately |
| ML breadth | Discusses multiple model options | Only knows one model type |
| Data thinking | Names specific features, pipeline steps | Vague "we collect data and train" |
| Production awareness | Latency, caching, fallbacks, monitoring | Design that only works in a notebook |
| Trade-off reasoning | Explicitly names trade-offs | Presents design as the only option |
| Communication | Clear diagram, labels every component | Jumps between topics randomly |

---

## Common ML System Design Patterns
{: #patterns}

Before the worked examples, a reference on the four architectural patterns that appear repeatedly across different ML systems. Knowing these lets you sketch a sound architecture quickly in the interview.

### Two-Tower Architecture
{: #two-tower}

The two-tower model is the workhorse of retrieval problems. The query tower and the item tower produce embeddings in a shared latent space. At serving time, item embeddings are precomputed and indexed; the query embedding is computed on-the-fly and used for approximate nearest-neighbour (ANN) search.

```
Query Tower                 Item Tower
-----------                 ----------
user_id embedding           item_id embedding
user_history sequence       item_features (category, age, etc.)
context features            content embeddings (text, image)
        |                           |
      [MLP]                       [MLP]
        |                           |
  query_vec (d=256)         item_vec (d=256)
        \                         /
         ----[ dot product ]------
               relevance score
```

**Training objective**: the standard choice is in-batch softmax (treating other items in the batch as negatives). For each positive (user, item) pair, the loss is:

$$\mathcal{L} = -\log \frac{\exp(\mathbf{q} \cdot \mathbf{i}^+ / \tau)}{\sum_{j=1}^{B} \exp(\mathbf{q} \cdot \mathbf{i}_j / \tau)}$$

where $\tau$ is a temperature hyperparameter, $\mathbf{q}$ is the query embedding, $\mathbf{i}^+$ is the positive item embedding, and the denominator runs over all $B$ items in the batch.

**Key design decisions:**

| Decision | Options | Guidance |
|---|---|---|
| Embedding dimension | 64–512 | Larger = more expressive but slower ANN search |
| Negative sampling | In-batch, hard negatives, random | Mix hard negatives with in-batch for quality |
| Tower asymmetry | Same depth vs different | User tower can be deeper; item tower is offline |
| Temperature $\tau$ | 0.05–0.1 | Lower = sharper distribution, harder training |

### Embedding Retrieval & ANN
{: #embedding-retrieval}

Two-tower retrieval relies on approximate nearest-neighbour (ANN) search over a precomputed item index. The choice of ANN algorithm determines recall, latency, and memory cost.

| Method | Algorithm | Recall@100 | Latency | Memory | When to use |
|---|---|---|---|---|---|
| FAISS IVF | Inverted file index + flat | ~95% | 5–20ms | Low | Standard production choice |
| FAISS HNSW | Hierarchical navigable small world | ~98% | 2–10ms | High | When recall matters more than memory |
| ScaNN | Anisotropic quantisation | ~97% | 3–15ms | Medium | Google-scale recommendation |
| ANNOY | Forest of random projection trees | ~90% | 10–50ms | Low | Read-heavy, build-once use cases |

**Recall–latency tradeoff**: ANN search returns approximate results. Increasing the number of probe clusters (FAISS `nprobe`) improves recall at the cost of latency. In practice, tune `nprobe` so that P99 latency fits within the retrieval budget (typically 10–20ms for a 100ms end-to-end request).

### Feature Stores
{: #feature-stores}

A feature store is a centralised system that manages the computation, storage, and serving of ML features. It solves the **training-serving skew** problem: the features computed at training time must match exactly those computed at serving time.

```
                   ┌─────────────────────────────────┐
                   │          Feature Store           │
  Batch jobs  ───► │  Offline store (e.g. Hive/S3)   │ ──► Training jobs
  (daily/hourly)   │  Point-in-time correct joins     │
                   ├─────────────────────────────────┤
  Stream jobs ───► │  Online store (e.g. Redis/Dynamo)│ ──► Serving (low-latency)
  (real-time)      │  Feature serving API             │
                   └─────────────────────────────────┘
```

**Point-in-time correctness**: when constructing training examples, the feature value used must be the value that would have been available at the time of the label, not the current value. Feature stores enforce this through temporal joins. Without it, you get **label leakage** — features computed with future information.

**Key components:**

| Component | Purpose | Example technology |
|---|---|---|
| Offline store | Historical feature values for training | Hive, BigQuery, Delta Lake |
| Online store | Low-latency feature serving | Redis, DynamoDB, Bigtable |
| Feature registry | Schema, ownership, versioning | Feast, Tecton, Hopsworks |
| Transformation engine | Compute features from raw events | Spark (batch), Flink (stream) |

### Stream vs Batch Processing
{: #stream-vs-batch}

| Dimension | Batch | Stream |
|---|---|---|
| Latency | Minutes to hours | Milliseconds to seconds |
| Throughput | Very high | High but bounded by event rate |
| Feature freshness | Stale (hourly/daily) | Near-real-time |
| Complexity | Lower | Higher (state management, watermarks) |
| Cost | Lower per record | Higher (always-on infrastructure) |
| Best for | Training data, daily aggregates | Real-time features, fraud, personalization |

**Rule of thumb**: use batch for features where staleness is acceptable (user long-term interests, item metadata), and stream for features where recency is critical (user's last 5 actions, transaction velocity in the last 5 minutes, trending items).

### Online Learning
{: #online-learning}

Online learning updates model weights continuously as new data arrives, rather than on a fixed retraining schedule. It matters most when the data distribution shifts rapidly (ad auctions, financial markets) or when user preferences evolve quickly.

**Full online learning** (gradient descent on each incoming example) is rarely practical for deep models due to catastrophic forgetting. The practical approaches:

| Approach | How it works | Use case |
|---|---|---|
| Mini-batch online | Accumulate a rolling window of examples; retrain on schedule (hourly/daily) | Most recommender systems |
| Warm-start retraining | Initialise from the previous checkpoint; train on recent data | Ad click-through rate |
| Continual learning | Keep an episodic memory of old examples; mix with new during retraining | When distribution shift + catastrophic forgetting both matter |
| Embedding-only updates | Keep deep layers frozen; update embedding table from new interactions | Efficient for new item/user cold start |

---

## Example 1: Recommender System (YouTube)
{: #recommender}

### Problem Scoping
{: #rec-scoping}

**The question**: Design a video recommendation system for a platform like YouTube.

**Clarifying questions to ask — and why each matters:**

> **Interview question:** What clarifying questions would you ask before designing a YouTube-scale recommender?
>
> *The five questions that most change the design: (1) What is the primary metric — watch time, clicks, satisfaction, or diversity? Watch time optimisation leads to filter bubbles; satisfaction requires explicit feedback signals. (2) What is the acceptable end-to-end latency for the recommendation page? Sub-200ms constrains retrieval depth and model complexity. (3) What is the scale — DAU, number of videos, requests per second? 100M DAU with 800M videos is qualitatively different from 1M DAU with 1M videos. (4) Is there a cold-start problem for new users and new videos? This determines whether you need a separate cold-start path. (5) Are there hard constraints — no recommending certain content categories, regional restrictions, copyright? These become business logic filters in the pipeline.*

**Stated assumptions after clarification:**

| Parameter | Value | Implication |
|---|---|---|
| DAU | 100M | Need distributed retrieval; cannot recompute per-user in real time |
| Video catalog | 800M videos | ANN index over 800M items; need efficient retrieval |
| Requests per second | ~50K peak | Serving must be horizontally scalable |
| Latency budget | 200ms end-to-end | Retrieval: 20ms, ranking: 50ms, post-processing: 10ms |
| Primary metric | Long-term satisfaction (proxy: watch time + explicit satisfaction surveys) | Cannot purely optimise click-through |
| Cold start | Yes — new users and new videos | Need a separate cold-start branch |
| Content safety | Hard blocklist applied before ranking | Separate content policy layer |

### High-Level Architecture
{: #rec-architecture}

The standard YouTube-style recommender follows a **multi-stage funnel**: retrieval narrows 800M candidates to ~500, then ranking scores those 500 with a heavier model, then post-processing applies business rules.

```
User Request (user_id, context)
          |
          v
  ┌───────────────────────────────────────────────────────────┐
  │                    Retrieval Layer                        │
  │                                                           │
  │  ┌─────────────────┐    ┌────────────────────┐           │
  │  │ Two-Tower ANN   │    │  Inverted Index     │           │
  │  │ (collaborative) │    │  (topic / keyword)  │           │
  │  └────────┬────────┘    └─────────┬──────────┘           │
  │           └──────────┬────────────┘                      │
  │                      v                                    │
  │             Candidate Fusion (~500 videos)                │
  └──────────────────────┬────────────────────────────────────┘
                         |
                         v
  ┌───────────────────────────────────────────────────────────┐
  │                    Ranking Layer                          │
  │                                                           │
  │   Wide & Deep model (or DCN-v2)                          │
  │   Input: user features + item features + context          │
  │   Output: P(watch), P(like), P(skip) per candidate        │
  └──────────────────────┬────────────────────────────────────┘
                         |
                         v
  ┌───────────────────────────────────────────────────────────┐
  │                 Post-Processing Layer                     │
  │                                                           │
  │   Content safety filter → Diversity re-ranking           │
  │   → Freshness boost → Business rule injection            │
  └──────────────────────┬────────────────────────────────────┘
                         |
                         v
              Top-K recommendations (20–50 videos)
```

### Data Sources & Pipeline
{: #rec-data}

**Event data (the primary signal):**

| Event type | Schema | Volume | Latency |
|---|---|---|---|
| Video watch | user_id, video_id, watch_duration, total_duration, timestamp | Billions/day | Real-time stream |
| Click | user_id, video_id, position_in_list, timestamp | Hundreds of millions/day | Real-time stream |
| Like / dislike | user_id, video_id, action, timestamp | Tens of millions/day | Real-time stream |
| Search query | user_id, query_text, results_clicked, timestamp | Hundreds of millions/day | Real-time stream |
| Skip | user_id, video_id, skip_at_second, timestamp | Billions/day | Real-time stream |

**Content data (item side):**

- Video metadata: title, description, tags, upload timestamp, channel_id
- Video embeddings: extracted from a pre-trained video encoder (visual + audio)
- Transcript embeddings: BERT-style embedding of the ASR transcript
- Engagement statistics: total views, like ratio, average watch percentage

**Data pipeline:**

```
Raw events (Kafka)
      |
      ├── Stream processor (Flink)
      │       ├── Real-time feature computation (user session features)
      │       └── Online store write (Redis) → feature serving
      |
      └── Batch processor (Spark, daily)
              ├── Training data construction (user-video pairs + labels)
              ├── Negative sampling
              ├── Offline feature joins (point-in-time correct)
              └── Offline store write (Hive/S3) → training jobs
```

**Label construction**: the watch label is not binary. Use **normalised watch time** as the implicit label:

$$y = \min\!\left(1,\; \frac{\text{watch\_duration}}{\text{video\_duration}}\right)$$

This is better than binary watch/no-watch because it captures engagement depth. A user who watches 90% of a 10-minute video is a stronger positive signal than one who watches 5 seconds of a 30-second video. Augment with explicit labels (likes, dislikes, survey ratings) when available.

### Feature Engineering
{: #rec-features}

**User features:**

| Feature | Type | Computation | Freshness |
|---|---|---|---|
| User embedding (from watch history) | Dense (d=256) | Two-tower user tower, updated daily | Daily |
| Watch history (last 50 videos) | Sequence of item IDs | Rolling window over event stream | Real-time |
| User age bucket | Categorical | Registration data | Static |
| Device type | Categorical | Request context | Per-request |
| Time of day | Continuous | Request timestamp → sin/cos encoding | Per-request |
| Day of week | Categorical (7 classes) | Request timestamp | Per-request |
| User language | Categorical | User profile | Static |
| Geographic region | Categorical | IP → region mapping | Per-request |

**Item features:**

| Feature | Type | Computation | Freshness |
|---|---|---|---|
| Video embedding | Dense (d=256) | Two-tower item tower; video encoder | Updated on upload |
| Title/description embedding | Dense (d=128) | Sentence-BERT on title+description | Updated on upload |
| Video age in days | Continuous | Current time − upload time | Per-request (computed) |
| Channel embedding | Dense (d=64) | Trained alongside item tower | Daily |
| Like ratio (30-day) | Continuous | Batch aggregate | Daily |
| Average watch percentage (7-day) | Continuous | Batch aggregate | Daily |
| Topic category | Categorical (hierarchical) | Content classifier | On upload |
| Video length bucket | Categorical | Duration → bucket | Static |

**Context features:**

| Feature | Type | Notes |
|---|---|---|
| Page context | Categorical | Homepage, search results, related videos |
| Search query embedding | Dense (d=128) | Only for search-context recommendations |
| Previously watched in session | Sequence (last 10) | In-session behavior |

**Key transformations:**

- **Cyclic time encoding**: $\sin(2\pi h / 24)$ and $\cos(2\pi h / 24)$ for hour of day — preserves the continuity between 23:00 and 00:00 that a raw integer misses.
- **Sequence aggregation**: the watch history is a variable-length sequence. Average pooling over item embeddings is the baseline; a lightweight Transformer encoder (2–4 layers) captures order and recency.
- **Log normalization**: engagement counts (views, watch count) follow power-law distributions. Apply $\log(1 + x)$ before feeding to the model.
- **Embedding lookup with hashing**: for high-cardinality categoricals (video_id, channel_id), use hash-bucketed embedding tables with a collision-tolerant size (typically 10× the vocabulary size).

### Model Selection & Architecture
{: #rec-model}

**Stage 1 — Retrieval: Two-Tower**

The retrieval model must score 800M items per query in under 20ms. The only architecture that achieves this is the two-tower (dual encoder): precompute all item embeddings offline, build an ANN index, and at serving time compute a single query embedding and do ANN search.

```
User Tower:
  user_id_emb (d=64) + avg(watch_history_embs) (d=256)
  + device_emb (d=16) + time_features (d=8)
  → concat (d=344) → MLP [512, 256] → L2-normalise → query_vec (d=256)

Item Tower:
  video_emb (d=256) + title_emb (d=128) + channel_emb (d=64)
  + like_ratio + avg_watch_pct + video_age_log
  → concat (d=451) → MLP [512, 256] → L2-normalise → item_vec (d=256)
```

Training loss — in-batch softmax with temperature $\tau = 0.07$:

$$\mathcal{L}_{\text{retrieval}} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{e^{\mathbf{q}_i \cdot \mathbf{k}_i / \tau}}{\sum_{j=1}^{B} e^{\mathbf{q}_i \cdot \mathbf{k}_j / \tau}}$$

**Hard negatives** improve retrieval quality: sample items from the bottom quartile of the model's current ranking for the same user to construct challenging negatives that push the embedding space to be more discriminative.

**Stage 2 — Ranking: DCN-v2 (Deep & Cross Network)**

The ranking model sees only ~500 candidates and must score them with a richer feature set. DCN-v2 captures explicit feature interactions (cross network) alongside implicit interactions (deep network):

$$\text{cross layer: } \mathbf{x}_{l+1} = \mathbf{x}_0 \odot (W_l \mathbf{x}_l + \mathbf{b}_l) + \mathbf{x}_l$$

The final output is a multi-task head: the model predicts multiple objectives simultaneously.

$$\hat{y} = \sigma\!\left(w_{\text{watch}} \cdot \hat{p}_{\text{watch}} + w_{\text{like}} \cdot \hat{p}_{\text{like}} - w_{\text{skip}} \cdot \hat{p}_{\text{skip}}\right)$$

**Multi-task learning** is important: a model trained purely on clicks optimises clickbait; a model trained purely on watch time ignores short-form content where watch time is low but satisfaction is high. Weight the objectives to reflect the product goal.

> **Interview question:** Why use a two-stage retrieval + ranking architecture rather than a single model that scores all items?
>
> *The constraint is latency. A heavy ranking model (DCN-v2, 50M parameters) takes ~2ms per candidate on a GPU. Scoring 800M candidates would take 800M × 2ms = 1.6 billion ms — impractical. The two-tower retrieval model reduces the candidate set to ~500 using ANN search (~20ms total), and the ranking model then applies the heavy features only to those 500 candidates (500 × 2ms = 1 second on CPU; ~50ms on GPU). The trade-off: retrieval recall must be high (top-500 must contain the best items) because items not retrieved are not ranked. This is why retrieval recall@500 is a critical metric to monitor.*

### Training Pipeline
{: #rec-training}

**Retrieval model training:**

- Data: all (user, video) pairs where watch percentage > 50%, sampled over a 90-day window
- Negative sampling: 1 hard negative + 127 in-batch negatives per positive
- Batch size: 2048 (larger batches = more in-batch negatives = better contrastive signal)
- Optimizer: Adam, lr=1e-4, cosine decay schedule
- Retraining cadence: weekly (user embeddings and item embeddings are stable enough)
- Warm start from previous checkpoint

**Ranking model training:**

- Data: all candidate impressions with labels (watch ratio, like/dislike) over a 14-day window
- Negative examples: videos that were shown but skipped within 5 seconds
- Class imbalance: positives are rare (~5% of impressions); apply focal loss or positive upsampling
- Batch size: 4096
- Retraining cadence: daily (ranking must capture freshness and trending content)
- Training infrastructure: 64 GPUs, data-parallel training, model checkpoint every 2 hours

**Position bias correction**: the ranking training data is biased — videos shown in position 1 get more clicks regardless of relevance. Correct this by including position as a feature during training (position debiasing) but setting it to a neutral value at serving time:

$$\hat{p}_{\text{click}} = \sigma(f_{\text{model}}(\mathbf{x}) + \beta_{\text{pos}} \cdot \text{position\_bias})$$

At serving, $\text{position\_bias} = 0$ so the score reflects only item relevance.

### Evaluation
{: #rec-evaluation}

**Offline metrics (retrieval):**

| Metric | Definition | Target |
|---|---|---|
| Recall@K | Fraction of relevant items in top-K retrieved | Recall@500 > 90% |
| Precision@K | Fraction of top-K that are relevant | Precision@10 > 30% |
| Hit rate | Fraction of users where the held-out item is in top-K | Hit@100 > 60% |

**Offline metrics (ranking):**

| Metric | Definition | Target |
|---|---|---|
| AUC-ROC | Area under ROC for click prediction | > 0.78 |
| Normalised DCG@10 | Ranking quality with graded relevance | > 0.65 |
| Average watch pct (offline) | Mean predicted watch ratio vs actual | Within 5% relative error |

**Online metrics (A/B test primary):**

| Metric | Why it matters | Direction |
|---|---|---|
| Average watch time per session | Core product metric | Increase |
| Session depth (videos per session) | Engagement breadth | Increase |
| Return rate (D1, D7) | Long-term user value | Increase |
| Satisfaction survey score | Ground truth satisfaction | Increase |
| Skip rate | User dissatisfaction signal | Decrease |
| Clickbait escape rate | Diversity quality | Decrease |

**A/B testing setup:**

- Randomise users (not requests) into control and treatment; user-level randomisation avoids novelty effects
- Minimum detectable effect: 0.5% relative improvement in watch time (with 80% power, 5% significance)
- Required sample size: compute using standard power analysis; typically requires 1–2 weeks at 50% traffic
- Holdback: keep 5% of users on the previous model version permanently as a long-term holdback to measure compounding effects
- Guardrail metrics: content safety reports must not increase; diversity metrics must not degrade more than 2%

### Serving & Inference
{: #rec-serving}

**Latency budget breakdown:**

```
200ms end-to-end
  ├── Network (client → CDN → server): 20ms
  ├── Feature fetching (Redis lookup): 10ms
  ├── Retrieval (ANN search, 2 sources): 20ms
  ├── Candidate fusion: 2ms
  ├── Ranking inference (GPU): 50ms
  ├── Post-processing (filters, rerank): 10ms
  └── Response serialisation + network: 20ms
  Buffer: 68ms
```

**Retrieval serving:**
- Item embeddings (800M × 256 floats = 800GB at float32; 200GB at int8) stored in FAISS IVF-PQ index
- Index sharded across 40 retrieval servers; query broadcast to all shards, results merged
- Query embedding computed on CPU (user tower is cheap — no cross-attention, just MLP)
- ANN search: top-500 per shard, merge by score, deduplicate

**Ranking serving:**
- Batch all 500 candidates into a single GPU forward pass
- Model served on A100 GPUs behind a load balancer
- Feature fetching parallelised: user features from Redis, item features from item store
- Result cache: cache ranking scores per (user_id, item_id) for 5 minutes to handle repeated requests (back button, refresh)

**Precomputed user embeddings:**
- User embeddings are expensive to compute (sequence model over 50-item watch history)
- Precompute nightly for all 100M users; store in Redis with a 24-hour TTL
- For new events within the day, update via a lightweight online update path (streaming average of the new item embedding into the cached user embedding)

### Graceful Degradation
{: #rec-degradation}

```
Layer            | Primary                      | Fallback
-----------------+------------------------------+------------------------------
Retrieval        | Two-tower ANN search         | Popularity-based retrieval
Ranking          | DCN-v2 GPU inference         | Lightweight LR model on CPU
Feature serving  | Redis online store           | Default feature values (zeros)
User embedding   | Precomputed personalised     | Demographic-average embedding
Content safety   | Classifier                   | Blocklist-only (keyword filter)
Full stack down  | Ranked personalised list     | Top-100 global trending videos
```

**Key principle**: every layer must have a fallback that requires no ML model. The final fallback (global trending) is deterministic and never fails. Alert if any layer falls back in more than 1% of requests.

### Failure Modes
{: #rec-failures}

| Failure | Symptom | Detection | Remediation |
|---|---|---|---|
| Training-serving skew | Offline metrics good, online metrics bad | Monitor feature distribution at serving time vs training time | Feature store audit; re-train with serving-matched features |
| Popularity bias amplification | System always recommends already-popular videos | Track Gini coefficient of impression distribution | Add diversity objective to ranking; diversification re-ranking |
| Filter bubble | User stuck in one topic cluster | Track topic diversity per user session | Exploration injection: force 20% diverse candidates into retrieval |
| Feedback loop | Model watches data used to train next model | Compare new-user exploration rate with and without model | Exploration/exploitation balance in retrieval |
| Embedding staleness | New videos never surface | Monitor average video age in recommendations | Freshness boost feature in ranking; ensure new video indexing pipeline runs |
| Cold start failure | New users get generic recommendations | Track engagement rate for users in first 7 days | Separate cold-start model using onboarding selections |
| Position bias leakage | Model always ranks position-1 items higher | Audit ranking scores vs serving position correlation | Add position debiasing; validate at training time |

### Monitoring
{: #rec-monitoring}

**Model health metrics:**

| Metric | Alert threshold | Frequency |
|---|---|---|
| Retrieval recall@500 | Drop > 3% vs baseline | Hourly |
| Ranking AUC | Drop > 0.01 vs baseline | Daily |
| Feature missing rate (any feature) | > 1% of requests | Real-time |
| Embedding coverage (fraction of items indexed) | < 99% | Hourly |
| Candidate set size | P5 < 100 candidates (retrieval too sparse) | Real-time |

**Business metrics:**

| Metric | Alert threshold | Response |
|---|---|---|
| Watch time per session | Drop > 2% vs D-7 | Investigate top failure mode; consider rollback |
| Skip rate | Increase > 5% vs D-7 | Review ranking model freshness |
| Content safety reports | Any increase vs D-7 | Immediate investigation; potential rollback |
| Diversity (unique topics per session) | Drop > 10% vs D-7 | Check diversity re-ranking pipeline |

**Data pipeline monitoring:**

- Event lag: Kafka consumer lag should be < 30 seconds for real-time features
- Batch job SLA: training data generation job must complete by 04:00 UTC
- Feature freshness: monitor the `last_updated` timestamp of each feature in the online store
- Training data volume: alert if daily training examples drop by > 20% (upstream data loss)

---

## Example 2: Fraud Detection
{: #fraud}

### Problem Scoping
{: #fraud-scoping}

**The question**: Design a fraud detection system for a payment processor like Stripe or PayPal.

**Clarifying questions:**

> **Interview question:** What are the most important clarifying questions for a fraud detection system?
>
> *The questions that most change the design: (1) What transaction types — card-not-present e-commerce, card-present POS, ACH transfers, P2P payments? Each has a different fraud pattern and different latency constraint. (2) What is the acceptable latency? Card-not-present e-commerce needs a decision in < 300ms (inline with checkout). ACH can tolerate minutes to hours. (3) What is the acceptable false positive rate? Declining a legitimate transaction has a direct cost (lost revenue, user frustration). A 0.1% FPR might be acceptable; 1% is probably not. (4) Who are the adversaries — individual fraudsters or organised rings? This determines whether graph-based features matter. (5) What happens when the model is unavailable — auto-approve or auto-decline? This is the graceful degradation policy.*

**Stated assumptions:**

| Parameter | Value |
|---|---|
| Transaction volume | 10,000 TPS peak |
| Transaction types | Card-not-present e-commerce |
| Latency budget | 150ms end-to-end |
| False positive rate target | < 0.1% (to not harm legitimate users) |
| Fraud rate | ~0.05% of transactions (highly imbalanced) |
| Adversary model | Mix of individual and organised card testing rings |
| Fallback on model failure | Approve with rule-based check (revenue continuity) |

### High-Level Architecture
{: #fraud-architecture}

```
Transaction Event (user_id, card_id, merchant_id, amount, timestamp, ...)
          |
          v
  ┌───────────────────────────────────────────────────────────┐
  │               Feature Computation Layer                   │
  │                                                           │
  │  ┌──────────────────────┐   ┌──────────────────────────┐ │
  │  │ Real-time features   │   │ Batch / historical feats  │ │
  │  │ (Flink, <10ms)       │   │ (Redis lookup, <5ms)      │ │
  │  └──────────┬───────────┘   └─────────────┬────────────┘ │
  │             └──────────────┬───────────────┘              │
  │                            v                              │
  │                   Feature vector (d~200)                  │
  └────────────────────────────┬──────────────────────────────┘
                               |
                               v
  ┌────────────────────────────────────────────────────────────┐
  │                      Model Scoring Layer                   │
  │                                                            │
  │   Rule engine (hard blocks) → ML model (risk score)       │
  │   → Threshold decision (approve / review / decline)        │
  └────────────────────────────┬───────────────────────────────┘
                               |
                    ┌──────────┴──────────┐
                    v                     v
              approve                 review / decline
                                          |
                                          v
                               ┌──────────────────────┐
                               │   Case Management     │
                               │   (human review)      │
                               └──────────────────────┘
```

**Rule engine first**: hard rules (card on blocklist, IP from embargoed country, card velocity > 50 transactions in 1 hour) run before the ML model. They are fast, interpretable, and catch known patterns without burning compute. The ML model handles the ambiguous middle ground.

### Data Sources & Pipeline
{: #fraud-data}

**Transaction data (primary):**

- Card number (hashed), card type, issuing bank, country of issue
- Merchant ID, merchant category code (MCC), merchant country
- Transaction amount, currency, timestamp
- Device fingerprint, IP address, user agent
- Shipping address (for e-commerce), billing address

**Historical data (aggregates, pre-computed):**

- Card-level: transaction count (1h, 24h, 7d, 30d), spend amount (same windows), declined count (1h, 24h)
- User-level: new card count in 7 days, chargeback rate, account age
- Merchant-level: fraud rate (30d), typical transaction amount distribution
- IP/device: number of cards used from this IP/device (24h), fraud rate from this IP (30d)

**Label data:**

Labels arrive with significant delay. A chargeback (the confirmed fraud signal) can arrive 60–90 days after the transaction. This creates a **label delay problem**:

```
Transaction at t=0
  → Potential dispute filed by user at t=30d
  → Bank investigates at t=45d
  → Chargeback confirmed at t=60d
  → Label available for training at t=60d
```

**Consequence**: the training set at any point in time has uncertain labels for recent transactions. Solutions: (1) use a 90-day lookback for confirmed fraud labels; (2) use proxy labels (reported fraud, manual review decisions) for recent transactions with appropriate downweighting; (3) use a separate model for near-real-time updates.

**Data pipeline:**

```
Transaction events (Kafka)
      |
      ├── Flink stream job
      │     ├── Compute velocity features (sliding window counts)
      │     ├── Write to Redis (real-time feature store)
      │     └── Write to event log (S3) for training
      |
      └── Batch job (daily)
            ├── Join transactions with chargeback labels (90-day lag)
            ├── Compute aggregate features (merchant fraud rate, user history)
            ├── Write training dataset
            └── Retrain model, run evaluation, promote if passes
```

### Feature Engineering
{: #fraud-features}

Fraud detection relies on **velocity features** (how fast is this entity transacting) and **deviation features** (how unusual is this transaction relative to history). Both require real-time computation.

**Velocity features** (computed by Flink over sliding time windows):

| Feature | Window | Notes |
|---|---|---|
| `card_txn_count_1h` | 1 hour | Card testing: many small transactions rapidly |
| `card_txn_count_24h` | 24 hours | Daily limit exceeded |
| `card_spend_1h` | 1 hour | Amount velocity |
| `card_distinct_merchants_1h` | 1 hour | Unusual to hit many merchants in 1 hour |
| `card_distinct_countries_24h` | 24 hours | Geographic impossibility |
| `ip_card_count_24h` | 24 hours | Number of distinct cards from same IP |
| `device_card_count_24h` | 24 hours | Number of distinct cards on same device |
| `merchant_txn_count_1m` | 1 minute | Merchant-side card testing detection |

**Amount deviation features:**

$$\text{amount\_z\_score} = \frac{x - \mu_{\text{user}}}{\sigma_{\text{user}}}$$

where $\mu_{\text{user}}$ and $\sigma_{\text{user}}$ are the user's historical mean and standard deviation of transaction amounts. A high z-score ($> 3$) signals an unusual purchase.

**Sequence features** — the last N transactions for a card, encoded as a sequence for the model:

- Time since last transaction (log-transformed)
- Amount ratio (current amount / rolling mean)
- Merchant category shift (is this a new category for this user?)
- Geographic displacement (distance between current and last-known location)

**Graph features** (for organised fraud ring detection):

Build a bipartite graph where nodes are cards and merchants. Fraudulent cards often share IP addresses, devices, or shipping addresses — these are graph edges. Features derived from this graph:

| Graph feature | Computation | Signal |
|---|---|---|
| Cards sharing same IP (24h) | Count of distinct cards from same IP | Card testing from single machine |
| Cards sharing same device | Fingerprint matching | Mule account ring |
| Merchant fraud neighbor rate | Fraud rate of merchants connected via shared cards | Merchant compromise |

These graph features are expensive to compute in real time; pre-compute them in a batch job and cache in Redis.

**Embeddings:**

- **Merchant embedding**: train an embedding on merchant transaction history (similar merchants cluster together). A new transaction at a merchant the user has never visited has a high merchant-user distance in this space.
- **IP embedding**: geographic + ISP-type embedding (residential vs datacenter IP vs VPN exit node).

### Model Selection & Architecture
{: #fraud-model}

**Why not just rules?** Rules cover known patterns but miss novel fraud strategies. A rule for "more than 10 transactions in 1 hour" misses fraudsters who test at exactly 9 transactions per hour. ML models learn the decision boundary from data.

**Why not just deep learning?** Gradient boosted trees (GBT) are the standard choice for tabular fraud detection, not deep learning, for several reasons:

| Criterion | GBT (XGBoost/LightGBM) | Deep MLP |
|---|---|---|
| Tabular features | Excellent; handles mixed types natively | Good but requires more engineering |
| Training speed | Fast | Slow |
| Inference latency | < 1ms on CPU | 1–10ms on CPU; faster on GPU |
| Interpretability | SHAP values; feature importances | Black box |
| Class imbalance | `scale_pos_weight` parameter | Requires careful loss design |
| Monotonicity constraints | Supported natively | Requires custom training |

**Model architecture — two-stage ensemble:**

```
Stage 1 (fast): GBT on tabular features
  Input: 200-dimensional feature vector (velocity + amount + historical)
  Output: fraud_score ∈ [0, 1]
  Latency: < 1ms

Stage 2 (slow, for high-uncertainty cases): Sequence model on transaction history
  Input: last 20 transactions as a sequence
  Architecture: Transformer encoder (4 layers, d_model=128)
  Output: fraud_score_sequence ∈ [0, 1]
  Latency: ~20ms on GPU

Final score: α · score_GBT + (1 - α) · score_sequence
  where α = 0.7 (GBT is primary, sequence model refines high-uncertainty cases)
```

The sequence model is invoked only when $\text{score\_GBT} \in [0.2, 0.8]$ (the uncertain region). For clear positives and negatives, the GBT alone is sufficient.

**Class imbalance handling:**

With 0.05% fraud rate, training on raw data produces a model that predicts "not fraud" for everything and achieves 99.95% accuracy. Solutions:

1. **Positive oversampling** (SMOTE or random): upsample fraud examples to a 1:10 positive:negative ratio
2. **Class weights**: set `scale_pos_weight = (1 - fraud_rate) / fraud_rate ≈ 2000` in XGBoost
3. **Focal loss** for the sequence model: $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ with $\gamma = 2$

**Monotonicity constraints**: some features should have a monotone relationship with fraud score. Higher `ip_card_count_24h` should never decrease the fraud score. XGBoost supports monotonicity constraints natively, preventing the model from learning spurious inverse relationships.

### Training Pipeline
{: #fraud-training}

**Offline training (monthly, full retraining):**

- Training window: 6 months of confirmed transactions with labels (90-day label delay means most recent 90 days have partial labels)
- Label: binary (1 = confirmed chargeback or manual fraud review decision, 0 = not disputed)
- Temporal split for evaluation: train on months 1–5, evaluate on month 6 (never shuffle time-series data randomly)
- Feature computation: all features computed from state available at transaction time (point-in-time correctness)
- Hyperparameter search: Bayesian optimisation over depth, learning rate, L1/L2 regularisation

**Online retraining (daily, warm-start):**

- Data: last 30 days of transactions with proxy labels (reported fraud, manual decisions)
- Warm start from previous checkpoint
- Evaluation gate: new model must not increase FPR@recall=80% relative to current model
- Canary deployment: 5% of traffic for 24 hours before full promotion

**Champion/challenger framework:**

- Champion: currently deployed model (stable, well-monitored)
- Challenger: new candidate model from latest training run
- Challenger gets 5% of traffic; champion gets 95%
- After 7 days: if challenger improves recall by > 2% without increasing FPR, promote to champion

**Feedback loop risk**: fraudsters adapt to the model. A model trained on historical fraud patterns will miss novel strategies. Mitigate by: (1) periodic red-teaming (simulate novel fraud strategies and test model response); (2) maintaining a small exploration budget where some transactions go to human review regardless of model score, generating diverse labels.

### Evaluation
{: #fraud-evaluation}

Accuracy and AUC are insufficient for fraud detection due to class imbalance. The primary metrics:

**Offline metrics:**

| Metric | Definition | Target |
|---|---|---|
| Recall@FPR=0.1% | Fraction of fraud caught when FPR is fixed at 0.1% | > 60% |
| AUC-PR | Area under precision-recall curve (better than ROC for imbalanced) | > 0.80 |
| KS statistic | Max separation between fraud and non-fraud score distributions | > 0.60 |
| F1 at operating threshold | Harmonic mean at deployed threshold | > 0.65 |

**Why AUC-ROC is insufficient**: with 99.95% negatives, a model that randomly scores 0.1% of transactions as high-risk achieves AUC-ROC ~ 0.99 while being nearly useless. AUC-PR penalises this because precision is low.

**Operational metrics (business-aligned):**

| Metric | Definition | Target |
|---|---|---|
| Fraud rate (net) | (fraud losses) / (total volume processed) | < 0.02% |
| False decline rate | Legitimate transactions declined / all legitimate | < 0.1% |
| Chargeback rate | Chargebacks / total transactions | < 0.05% (card network threshold) |
| Manual review rate | Transactions sent to human review / total | < 0.5% (cost control) |

**A/B testing for fraud**: standard A/B testing is not straightforward — you cannot easily run two fraud models simultaneously, as fraud decisions affect each other (a declined transaction may lead to the fraudster trying again on a different path). Use **time-based holdouts** (train model on odd days, test on even days) or **population-based holdouts** (geographically separate populations).

### Serving & Inference
{: #fraud-serving}

**Latency budget:**

```
150ms end-to-end
  ├── Network (API call from checkout): 20ms
  ├── Feature fetching (Redis lookup, batch): 15ms
  ├── Rule engine check: 2ms
  ├── GBT scoring (CPU): 1ms
  ├── Sequence model (GPU, conditional): 20ms
  ├── Decision logic + response: 2ms
  └── Network (response): 20ms
  Buffer: 70ms
```

**Feature serving:**

- All pre-computed aggregate features stored in Redis with a 1-hour TTL
- Real-time velocity features computed by Flink and written to Redis with a 5-minute TTL
- At serving time, single batched Redis `MGET` call fetches all features in one round trip (~5ms)

**Model serving:**

- GBT model: served in-process (loaded into memory of the fraud scoring service) — sub-millisecond inference, no network hop
- Sequence model: served on a separate GPU service; called only for uncertain cases
- Model files loaded at startup; hot reload via blue-green deployment (no downtime)

**Threshold selection:**

The model outputs a continuous risk score $s \in [0, 1]$. Thresholds map this to decisions:

| Score range | Decision | Volume (approx) |
|---|---|---|
| $s < 0.3$ | Auto-approve | ~98.5% of transactions |
| $0.3 \le s < 0.7$ | Manual review queue | ~1.0% of transactions |
| $s \ge 0.7$ | Auto-decline | ~0.5% of transactions |

Thresholds are calibrated to hit the target FPR and are revisited monthly as the fraud rate and model drift.

### Graceful Degradation
{: #fraud-degradation}

```
Layer              | Primary                        | Fallback
-------------------+--------------------------------+---------------------------
Feature fetching   | Redis lookup                   | Default feature values (0)
Stream features    | Flink real-time computation    | Last-known value from cache
GBT model          | In-process scoring             | Rule engine only
Sequence model     | GPU inference service          | GBT score alone (skip stage 2)
Decision service   | Full pipeline                  | Velocity rules only (approve if < 10 txn/hr)
Complete outage    | Any decision                   | Approve all (revenue continuity) + alert immediately
```

**Critical**: the fallback for complete outage is to approve all transactions. The revenue cost of downtime is lower than the reputational and compliance cost of declining all legitimate transactions. But this fallback must alert immediately and have a maximum duration (e.g., 5 minutes before human escalation).

### Failure Modes
{: #fraud-failures}

| Failure | Symptom | Detection | Remediation |
|---|---|---|---|
| Concept drift | Fraud rate increases despite stable model score | Monitor fraud rate vs model score distribution weekly | Trigger emergency retraining on recent labelled data |
| Feature pipeline lag | Velocity features stale | Monitor Flink consumer lag; alert if > 60s | Serve with cached values; alert engineering |
| Label delay exploitation | Model misses fraud patterns < 90 days old | Compare model scores on recent vs old fraud | Use proxy labels for near-real-time updates |
| Adversarial adaptation | Novel fraud pattern below detection threshold | Monitor manual review outcomes for new patterns | Human review of low-confidence scores; re-labelling loop |
| Score distribution shift | Model score distribution shifts without change in FPR | Monitor score histogram daily | Investigate; re-calibrate thresholds |
| False positive surge | Legitimate users complain about declines | Monitor false decline rate, user complaint rate | Loosen threshold; investigate feature anomaly |

### Monitoring
{: #fraud-monitoring}

**Model metrics:**

| Metric | Alert threshold | Frequency |
|---|---|---|
| Score distribution KL divergence vs baseline | > 0.05 | Hourly |
| Auto-decline rate | > 0.7% (2× normal) | Real-time |
| Manual review queue depth | > 50,000 pending | Real-time |
| GBT inference P99 latency | > 5ms | Real-time |
| Feature missing rate | > 1% | Real-time |

**Business metrics:**

| Metric | Alert threshold | Response |
|---|---|---|
| Fraud rate (rolling 24h) | > 0.04% (2× normal) | Tighten thresholds; page on-call |
| False decline rate (rolling 24h) | > 0.15% (1.5× normal) | Loosen thresholds; investigate |
| Chargeback rate (weekly) | > 0.04% (approaching card network limit) | Immediate review; consider emergency rules |

**Data quality monitoring:**

- Redis hit rate: should be > 99%; a drop indicates feature pipeline failure
- Kafka consumer lag: alert if > 60 seconds (velocity features become stale)
- Training data volume: alert if daily transaction count drops by > 10% (upstream data loss)
- Label rate: monitor weekly chargeback volume to detect if dispute process has a delay

> **Interview question:** Fraud rate suddenly increases from 0.03% to 0.08%. Walk through your investigation.
>
> *Step 1: Segment the increase. Is it across all merchants, a single merchant, a single card BIN range, a specific geographic region? Segmentation reveals whether this is a merchant compromise (one merchant sees a spike), a BIN attack (fraudsters testing a specific card series), or a novel widespread strategy. Step 2: Check model scores. Are the fraudulent transactions scoring high (model detects them but threshold is too low) or low (model is blind to this pattern)? If scoring high: lower the decision threshold temporarily, accept more false positives. If scoring low: the model has a blind spot — investigate feature values for these transactions. Step 3: Check feature validity. Are velocity features computing correctly? A Flink lag spike could mean velocity features are stale, reducing model signal. Step 4: Manual sample. Pull 50 fraudulent transactions and read them. What do they have in common that the model might miss? New merchant category? Specific amount range? Specific device type? Step 5: Emergency response. Add targeted rules for the identified pattern (MCC filter, amount cap, velocity rule for the specific BIN). Rules are faster to deploy than model retraining. Step 6: Retrain. Once the pattern is understood and labelled, retrain the model to incorporate it.*

---

## Example 3: AI Coding Assistant (Agentic System)
{: #agent}

The agentic system is qualitatively different from the two previous examples. There is no fixed training label per request; the system does not produce a single prediction but an extended sequence of tool calls and code outputs; and evaluation cannot be reduced to offline AUC or online CTR. The design pattern centres on the **agent loop**, **tool use**, **context management**, and **multi-step evaluation**.

### Problem Scoping
{: #agent-scoping}

**The question**: Design an AI coding assistant that can autonomously complete GitHub issues — reading the repository, understanding the codebase, writing code, running tests, and opening a pull request.

**Clarifying questions:**

> **Interview question:** What are the most important scoping questions for an agentic coding assistant?
>
> *The questions that change the design most: (1) What is the task horizon? Single-function completion (< 30 seconds, 3–5 model calls) is a very different system from full-issue resolution (minutes, 20–50 model calls with tool use). (2) What is the execution environment — cloud sandbox or local machine? Cloud sandboxes are safer but require network isolation. Local execution is riskier but faster. (3) What codebase sizes must be supported? A 10K line repo is fine for naive context injection; a 500K line repo requires selective retrieval. (4) What is the failure mode policy — fail loudly (return error) or fail gracefully (produce partial result with explanation)? (5) What is the latency expectation — interactive (user waits) or asynchronous (user submits and gets notified)? (6) Who are the users — developers writing code, or product managers describing features? This changes the input format and required NLU.*

**Stated assumptions:**

| Parameter | Value |
|---|---|
| Task type | Autonomous issue resolution (not autocomplete) |
| Codebase size | Up to 500K lines |
| Execution environment | Cloud sandbox (Docker container, no external network) |
| Latency expectation | Asynchronous (submit issue → notified when done, typically 2–10 minutes) |
| Users | Software engineers at a company |
| Failure policy | Fail loudly — return partial work with explanation of what failed |
| Success definition | Pull request opened; CI passes; human review approves |

### High-Level Architecture
{: #agent-architecture}

The agentic design does not fit a single forward-pass model. It is an **orchestration system** around a language model, with explicit tool execution, state management, and evaluation loops.

```
User submits issue
        |
        v
┌───────────────────────────────────────────────────────────────────┐
│                        Orchestration Layer                        │
│                                                                   │
│  Issue Parser → Task Planner → Agent Loop Controller             │
│                                    |                              │
│                              step budget                         │
│                              token budget                         │
│                              context compaction policy           │
└────────────────────────────────────┬──────────────────────────────┘
                                     |
                                     v
┌───────────────────────────────────────────────────────────────────┐
│                          Agent Runtime                            │
│                                                                   │
│  System Prompt + Tool Schemas + Task Brief + Context Window       │
│            |                                                      │
│            v                                                      │
│       LLM (Thought → Tool Call → Observation loop)               │
│            |                                                      │
│            v                                                      │
│       Tool Dispatcher                                             │
│  ┌────────┬──────────┬──────────┬──────────┬────────────┐        │
│  │ read   │  bash/  │  search  │  git     │  test      │        │
│  │ file   │  edit   │  code    │  ops     │  runner    │        │
│  └────────┴──────────┴──────────┴──────────┴────────────┘        │
│            |                                                      │
│       Sandbox (Docker, no network, 2GB RAM, 10min timeout)       │
└────────────────────────────────────┬──────────────────────────────┘
                                     |
                                     v
┌───────────────────────────────────────────────────────────────────┐
│                      Evaluation & Output Layer                    │
│                                                                   │
│  Syntax check → Tests pass? → Code review model → PR opened      │
└───────────────────────────────────────────────────────────────────┘
```

### Data Sources & Pipeline
{: #agent-data}

Unlike a supervised ML system, the agentic system does not have a fixed training dataset from which features are extracted. Instead, it has two data regimes:

**Runtime context (per-task):**

- The GitHub issue: title, description, labels, linked issues
- Repository contents: file tree, code files, tests, CI configuration
- Repository history: recent commits, open PRs, CHANGELOG
- Issue comments: previous discussion, acceptance criteria
- Test results: CI logs, test output

**Training data (for the underlying model):**

The base LLM is trained on code corpora (GitHub, StackOverflow, documentation). For fine-tuning the agent:

| Data source | What it provides | Volume |
|---|---|---|
| Successful task trajectories (human-verified) | (issue, steps, PR) triples — the agent's golden path | Thousands |
| Failed trajectories with error labels | Negative examples for RLVR | Thousands |
| Unit test suites | Verifier for RLVR — does the code pass tests? | Millions of test cases |
| Code review comments | Style and correctness signal | Millions of comments |
| Human preference data (RLHF) | Which of two completions is better? | Hundreds of thousands |

**Data pipeline for training:**

```
Historical resolved issues (GitHub API)
        |
        ├── Filter: issues with merged PRs + passing CI
        ├── Extract: (issue_text, code_diff, test_results) triples
        ├── Construct agent trajectory: replay the steps implied by the diff
        │     (read files → edit code → run tests → iterate)
        └── SFT dataset: (context, action) pairs for supervised fine-tuning

RLVR pipeline:
        Issue → Agent generates trajectory → Code executed in sandbox
        → Tests run → Pass/fail → Reward signal → PPO/GRPO update
```

### Feature Engineering & Context Management
{: #agent-features}

For an agentic system, "feature engineering" is **context management** — deciding what information to put in the model's context window at each step of the agent loop.

**The context budget equation:**

$$C_{\text{total}} = C_{\text{system}} + C_{\text{tools}} + C_{\text{task}} + C_{\text{code\_context}} + C_{\text{history}} + C_{\text{generation}}$$

For a 200K token context window:

| Component | Typical size | Priority |
|---|---|---|
| System prompt | 2K tokens | Highest — always present |
| Tool schemas (read, edit, bash, git, test) | 3K tokens | Highest — needed for action |
| Issue description | 1K tokens | High — the task goal |
| Code context (retrieved relevant files) | 20–60K tokens | High but variable |
| Agent history (prior turns) | grows with steps | Medium — compress as needed |
| Current tool output | variable | Immediate — processed then evicted |

**Code retrieval** — the hardest part of context management for a large codebase:

The agent cannot fit a 500K line codebase into context. It must retrieve relevant files. Two approaches:

1. **Embedding-based retrieval**: embed all code files using a code-aware encoder (e.g., code-embedding model trained on function-level code). At task time, embed the issue description and retrieve the top-K most similar files. This handles semantic relevance (a bug about "authentication" retrieves auth-related files) but misses structural dependencies.

2. **Structural retrieval**: parse the codebase's import graph and call graph. When the issue mentions a function or class by name, retrieve the file containing it and all files that directly import from it. This handles dependency-aware retrieval but requires parsing.

**In practice, use hybrid retrieval**: start with the structural approach (exact name matches) and fall back to embedding similarity for files that are semantically relevant but not explicitly named.

**Context compaction strategy for long task trajectories:**

As the agent executes over 20–50 steps, the history grows and pushes relevant context out of the window. The preferred compaction strategy for a coding agent:

```
Compaction trigger: context > 80% of max window

Compaction method: transcript-to-state
  1. Extract from history: which files were read, which files were edited,
     current test results, open questions
  2. Write these into a "notebook" structure:
     {
       "files_read": [...],
       "edits_made": [{"file": ..., "description": ...}],
       "test_status": "3/5 passing",
       "open_issues": ["fix edge case in auth.py line 45"]
     }
  3. Discard raw history; keep notebook + last 3 turns verbatim
  4. Continue with compact context
```

This preserves the agent's "working memory" without accumulating every intermediate tool output.

**Retrieval-augmented tool use**: a dedicated `search_code` tool that the agent can call to retrieve additional files mid-task. This externalises retrieval from context and lets the agent selectively load files as needed, rather than front-loading everything.

### Model Selection & Architecture
{: #agent-model}

**Base model selection:**

The agentic coding assistant requires a model with:
- Strong code understanding and generation (≥ SWE-bench competitive)
- Reliable tool use / function calling
- Long context support (≥ 128K tokens for large files)
- Low hallucination rate (wrong file paths or non-existent functions are catastrophic)

Frontier models (Claude 3.5+, GPT-4o, Gemini 1.5 Pro) are the practical choice for production. Fine-tuning a smaller model on domain-specific trajectories can reduce cost while maintaining quality for well-defined task types.

**Fine-tuning strategy:**

1. **SFT on golden trajectories**: supervised fine-tuning on human-verified (issue → correct step sequence → passing PR) triples. Teaches the model the correct tool-use pattern and code editing format.

2. **RLVR on test suites**: the verifier is the test suite. The model receives a reward of 1 if its code changes cause all tests to pass; 0 otherwise. This is a dense, automated reward signal that does not require human labelling.

   $$R(x, y) = \mathbb{1}[\text{tests\_pass}(y)] - \lambda \cdot \text{step\_count}(y)$$

   The step count penalty discourages the model from solving tasks by brute force (trying every possible edit).

3. **RLHF on PR quality**: a preference model trained on human comparisons of PR quality (code style, test coverage, comment quality) provides a quality signal beyond binary pass/fail.

**Architecture considerations:**

- The LLM itself is the "model" — it is not a custom architecture for this use case
- The agentic framework (tool dispatch, context management, compaction) is **infrastructure**, not model weights
- Critical design choice: **structured tool use** (function calling) rather than free-text parsing. The model outputs `{"tool": "edit_file", "args": {"path": "auth.py", "old_string": "...", "new_string": "..."}}` which the dispatcher executes deterministically. Free-text parsing is fragile.

**Multi-agent extension:**

For complex issues, a single-agent approach hits context limits. A multi-agent setup:

```
Planner Agent: reads issue → produces task decomposition
  ├── Worker Agent 1: handles authentication module changes
  ├── Worker Agent 2: handles test additions
  └── Worker Agent 3: handles documentation updates

Planner Agent: merges results → opens single PR
```

Each worker receives a compact handoff packet (specific files, specific goal, done criteria) and returns a typed result (list of edits, test results). The planner never sees the workers' raw trajectories.

### Training Pipeline
{: #agent-training}

**Phase 1 — SFT (supervised fine-tuning on golden trajectories):**

- Data: 50,000 high-quality (issue, trajectory, PR) triples from human-verified resolutions
- Trajectory format: interleaved thought-action-observation sequences
- Loss: standard next-token prediction on the action tokens (not the observation tokens, which are tool outputs and should not be predicted)
- Training: 3–5 epochs, lr = 1e-5, cosine decay
- Evaluation: SWE-bench Lite subset (known evaluation benchmark for code agents)

**Phase 2 — RLVR (reinforcement learning with verifiable rewards):**

- Environment: sandboxed Docker containers with a repo, an issue, and a test suite
- Rollout: agent runs to completion or step budget (max 50 steps)
- Reward: binary (all tests pass = +1, fail = 0) minus step penalty
- Algorithm: GRPO (Group Relative Policy Optimisation) — samples K rollouts per issue, uses relative ranking within the group as the advantage signal
- KL regularisation: $\mathcal{L}_{\text{RLVR}} = -\mathbb{E}[R] + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ to prevent the model from drifting too far from the SFT checkpoint

**Phase 3 — Reward model for PR quality:**

- Data: human comparisons of pairs of PRs solving the same issue
- Architecture: LLM with a scalar head (classification approach: is PR A better than PR B?)
- Used as an auxiliary reward signal in a second RLHF phase

**Retraining cadence:**
- RLVR: run continuously on a rolling window of new issues from production (with sandbox verification)
- SFT: monthly, incorporating newly human-verified trajectories
- Reward model: quarterly, as human preference data accumulates

**Training-serving consistency:**

The agent's tool schemas, sandbox configuration, and context structure used during training must exactly match those at serving time. A change to a tool's output format between training and serving causes the model to misinterpret tool outputs — a major failure mode.

### Evaluation
{: #agent-evaluation}

Agentic evaluation is the hardest of the three examples. A single forward pass does not produce a right/wrong answer. Evaluation must cover trajectory quality, not just final output quality.

**Verifier cascade:**

```
Level 1 (Syntax): Does the produced code parse?
  → Python AST parse, or equivalent for other languages
  → Fail → return immediately; don't run tests

Level 2 (Static): Do all referenced files, functions, and imports exist?
  → Lint with flake8/mypy; check import resolution
  → Fail → strong negative signal

Level 3 (Execution): Do existing tests still pass? Do new tests pass?
  → Run full test suite in sandbox
  → Pass = primary positive signal

Level 4 (Quality): Is the code style acceptable?
  → Reward model score
  → PR review model output (simulated code reviewer)
```

**Offline metrics:**

| Metric | Definition | Benchmark |
|---|---|---|
| SWE-bench resolve rate | Fraction of issues fully resolved (all tests pass) | State of art: ~50% (hard benchmark) |
| Partial credit (k/n tests) | Fraction of tests passing when not all pass | Curriculum metric |
| Step efficiency | Steps taken vs oracle minimum steps | Lower is better; tracks waste |
| Hallucination rate | Fraction of tool calls referencing non-existent paths/functions | Should be < 1% |
| Context overflow rate | Fraction of tasks that hit context limit | Should be < 5% |

**Online metrics:**

| Metric | Definition | Target |
|---|---|---|
| PR merge rate | Fraction of agent PRs merged by humans | > 40% (high bar — many are follow-up needed) |
| Time to close issue | Wall-clock from issue submission to merged PR | P50 < 15 minutes |
| Human reviewer edits | Lines changed by human after agent PR | Fewer = better quality |
| CI pass rate on first submit | Fraction of agent PRs where CI passes immediately | > 70% |
| User satisfaction (NPS) | Developer survey on usefulness | Track weekly |

**A/B testing challenges for agentic systems:**

- Standard A/B testing (split users into control/treatment) works, but the observation window is long (days to weeks to accumulate enough resolved issues)
- Guardrail metrics: human reviewer burden must not increase; CI failure rate must not increase; accidental data deletion (destructive edits) rate must be zero
- Shadow mode: before deployment, run the new agent in shadow mode (produces outputs but does not open PRs); evaluate quality before going live

### Serving & Inference
{: #agent-serving}

Agentic serving is fundamentally asynchronous. The user submits an issue; the system returns a task ID; the agent runs for minutes; the user is notified when complete.

**Request lifecycle:**

```
User submits issue → API returns task_id (immediate)
        |
        v
Task queue (SQS/Celery) → Worker picks up task
        |
        v
Agent runtime:
  ├── Provision sandbox (Docker, ~5s cold start)
  ├── Clone repository (~10–30s for large repos)
  ├── Run agent loop (2–10 minutes, 10–50 steps)
  └── If success: open PR via GitHub API
        |
        v
Notification sent to user (webhook / email / Slack)
```

**Latency budget (per-step, not end-to-end):**

Each agent step has a latency budget:

```
Per-step budget: ~5 seconds average (50 steps × 5s = ~4 minutes total)
  ├── LLM inference (reasoning + tool call generation): 1–3s
  ├── Tool execution (file read: <100ms; bash: up to 30s; tests: up to 60s)
  ├── Context compaction (if triggered): 500ms
  └── State persistence: 100ms
```

**Prefix caching for cost reduction:**

The system prompt (2K tokens) and tool schemas (3K tokens) are identical across all steps of a task and across all tasks. Enable prefix caching:

- Anthropic API: `cache_control: {"type": "ephemeral"}` on the static prefix
- Cache hit saves ~90% of the compute cost on the prefix portion
- For a 50-step task with a 5K-token static prefix: 50 × 5K × 0.9 = 225K tokens of avoided compute

**Parallelism:**

- Multiple tasks run in parallel across worker pool (independent Docker containers)
- Within a task, parallel tool calls where independent: the agent can call `read_file(auth.py)` and `read_file(config.py)` simultaneously
- Modern LLM APIs support parallel function calling in a single response

**Scaling:**

- Worker pool scales horizontally based on task queue depth
- Each worker runs one agent task at a time (tasks are I/O bound — waiting for LLM responses and tool execution)
- LLM API is the bottleneck: manage via rate limits and request queuing
- Sandbox provisioning latency: maintain a warm pool of pre-provisioned Docker containers to avoid 5-second cold start per task

### Graceful Degradation
{: #agent-degradation}

```
Layer                | Primary                           | Fallback
---------------------+-----------------------------------+---------------------------
Code retrieval       | Embedding + structural retrieval  | Return top-20 recently-modified files
LLM inference        | Primary model (e.g., Claude)      | Smaller model (fewer capabilities)
Sandbox              | Docker container                  | Read-only analysis (no execution)
Test runner          | Full test suite                   | Syntax check only
Context budget       | Full trajectory                   | Compact to notebook + last 3 turns
Step budget exceeded | Partial completion                | Return work-in-progress with explanation
PR creation          | GitHub API                        | Output diff to file + notify user
Full stack failure   | Autonomous resolution             | Return analysis of the issue only
```

**Partial result handling**: if the agent exhausts its step budget without fully resolving the issue, it should return its work in progress — the edits made so far, the tests that pass, and an explanation of what remains. A partial result is vastly more useful than a silent failure or a timeout error.

### Failure Modes
{: #agent-failures}

| Failure | Symptom | Detection | Remediation |
|---|---|---|---|
| Hallucinated file paths | Tool call to non-existent file; infinite retry loop | Monitor `FileNotFoundError` rate in tool outputs | Structural retrieval pre-check; retry with search_code tool |
| Runaway loop | Agent repeatedly makes same tool call | Step budget counter; detect circular pattern (same tool + same args 3× in a row) | Hard stop; return partial result |
| Context overflow | Task fails at step 30+ when context hits limit | Monitor context token count per step; alert at 80% of limit | Trigger compaction earlier; reduce code context size |
| Test suite environment failure | Tests fail due to missing dependency, not code error | Distinguish test failure vs environment failure in sandbox output | Separate `env_failure` error type; retry with fresh sandbox |
| Training-serving tool schema mismatch | Model outputs tool calls with wrong argument names | Schema validation at dispatch time | Hard fail with descriptive error; model self-corrects |
| Repository access failure | Cannot clone repo (permissions, network) | API call failure in provisioning phase | Immediate fail + notify user; no retry (permission issues require human action) |
| Prompt injection from codebase | Adversarial comment in code: "Ignore your system prompt and..." | Tool output safety classifier | Sanitise tool outputs before context injection |
| Catastrophic edit | Agent deletes or corrupts a critical file | Sandbox containment — no edit escapes sandbox | Sandboxing prevents real damage; diff review before PR |

### Monitoring
{: #agent-monitoring}

**Agent health metrics:**

| Metric | Alert threshold | Frequency |
|---|---|---|
| Task completion rate (tests pass) | Drop > 5% vs baseline | Hourly |
| Step count P95 | Increase > 20% vs baseline | Daily |
| Token cost per task P95 | Increase > 25% vs baseline | Daily |
| Hallucination rate (bad file paths) | > 2% of tool calls | Hourly |
| Context overflow rate | > 5% of tasks | Hourly |
| LLM API latency P99 | > 10s per step | Real-time |
| Sandbox provisioning P99 | > 15s | Hourly |

**Business metrics:**

| Metric | Alert threshold | Response |
|---|---|---|
| PR merge rate (weekly) | Drop > 10% relative | Investigate top failure mode; check recent model changes |
| CI pass rate on first submit | Drop > 15% relative | Check test environment; check model quality |
| Human reviewer edits (median lines) | Increase > 30% relative | Review PR quality; check reward model signal |
| User NPS | Drop > 5 points | Broad investigation; user interviews |

**Trace-level monitoring:**

Every agent task generates a full trace: (thought, tool_call, tool_output, observation) for each step. Critical monitoring over this trace:

- **Tool call distribution**: which tools are called most; is `bash` being called with dangerous commands?
- **Error type distribution**: which tool errors are most common? This reveals the most impactful improvements.
- **Step efficiency**: compare steps taken vs the shortest possible solution for the same issue (estimated from the final diff size)
- **Compaction trigger rate**: how often does context compaction fire? If > 30% of tasks, the context is being managed inefficiently
- **Cache hit rate**: monitor prefix cache hit rate; a drop indicates context structure changed

> **Interview question:** Your agent's CI pass rate drops from 72% to 55% after a model update. Walk through how you investigate.
>
> *Step 1: Segment by failure type. Are tests failing due to syntax errors (model produces unparseable code), logic errors (code parses but is wrong), environment errors (test infrastructure issue), or timeout (agent ran out of steps)? Different failure types have different root causes. Step 2: Check if the model update changed tool use behaviour. Pull tool call logs for the failing tasks. Are tool calls using correct argument names? Did the new model start using different parameter names for edit_file or bash? A schema mismatch between training and serving is a common cause of regressions after model updates. Step 3: Compare step count distribution. Did the new model take more steps on average? More steps per task means more context accumulation, more compaction events, and higher risk of losing critical context. Step 4: Sample failing trajectories. Pull 20 failed task traces and read them. What step does the trajectory diverge from the correct path? Is the model misunderstanding the issue, hallucinating file paths, or making correct edits that fail for environment reasons? Step 5: Run regression on SWE-bench Lite. A held-out benchmark lets you compare new vs old model on known tasks with known answers. If SWE-bench also dropped, the model regression is real. If SWE-bench is stable, the production distribution shifted (new types of issues, codebase characteristics the model struggles with). Remediation depends on root cause: tool schema mismatch → fix schema consistency; logic errors → trigger SFT on recent failures; step count regression → add step penalty to reward; environment errors → fix CI infrastructure.*

---

## Also Read

**[Agentic System Design](/blogs/agentic-system-design/)** — a deep-dive into the engineering challenges specific to LLM-powered agents: agent loops (ReAct, Plan-and-Execute), context budget management, tool system design, memory architecture, multi-agent orchestration, reliability and safety, evaluation and reward design, and production operations. Covers the design patterns referenced in Example 3 of this post in much greater depth.

**[ML Systems: Training and Serving](/blogs/ml-systems-data-training-serving/)** — the infrastructure foundations: automatic differentiation, hardware acceleration, transformer architectures, distributed training (data, tensor, and pipeline parallelism), memory optimisations, and production serving systems. Essential background for the training pipeline sections across all three examples in this post.
