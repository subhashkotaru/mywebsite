---
title: "Evaluation & Benchmarking"
date: 2026-04-21
description: "A comprehensive reference for ML/AI evaluation — classification, regression, ranking, NLP, and generation metrics with equations, plus a full catalogue of LLM, code, math, agent, and VLM benchmarks with limitations and saturation analysis."
tags: [evaluation, benchmarks, llm, ml-systems]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#classification">Classification Metrics</a>
      <ul class="post-toc-sublist">
        <li><a href="#confusion-matrix">Confusion Matrix</a></li>
        <li><a href="#precision-recall">Precision, Recall & F-scores</a></li>
        <li><a href="#roc-auc">ROC, AUC & PR Curves</a></li>
        <li><a href="#mcc">MCC & Imbalanced Classes</a></li>
      </ul>
    </li>
    <li><a href="#regression">Regression Metrics</a>
      <ul class="post-toc-sublist">
        <li><a href="#mae-mse">MAE, MSE, RMSE</a></li>
        <li><a href="#r-squared">R² & Adjusted R²</a></li>
      </ul>
    </li>
    <li><a href="#ranking">Ranking & Retrieval Metrics</a>
      <ul class="post-toc-sublist">
        <li><a href="#ndcg">NDCG</a></li>
        <li><a href="#mrr-map">MRR & MAP</a></li>
        <li><a href="#hit-rate">Precision@K, Recall@K, Hit Rate</a></li>
      </ul>
    </li>
    <li><a href="#nlp-metrics">NLP & Generation Metrics</a>
      <ul class="post-toc-sublist">
        <li><a href="#perplexity">Perplexity</a></li>
        <li><a href="#bleu">BLEU</a></li>
        <li><a href="#rouge">ROUGE</a></li>
        <li><a href="#bertscore">BERTScore & MoverScore</a></li>
        <li><a href="#pass-k">PASS@k</a></li>
      </ul>
    </li>
    <li><a href="#detection">Detection & Segmentation Metrics</a></li>
    <li><a href="#calibration">Calibration Metrics</a></li>
    <li><a href="#llm-benchmarks">LLM Benchmarks</a>
      <ul class="post-toc-sublist">
        <li><a href="#knowledge-reasoning">Knowledge & Reasoning</a></li>
        <li><a href="#math-code">Math & Code</a></li>
        <li><a href="#instruction-following">Instruction Following</a></li>
        <li><a href="#factuality-safety">Factuality & Safety</a></li>
        <li><a href="#long-context">Long Context</a></li>
        <li><a href="#rag-benchmarks">RAG & Retrieval</a></li>
        <li><a href="#agent-benchmarks">Agent Benchmarks</a></li>
        <li><a href="#arena">Arena & Human Preference</a></li>
      </ul>
    </li>
    <li><a href="#vlm-benchmarks">VLM Benchmarks</a></li>
    <li><a href="#benchmark-pitfalls">Benchmark Pitfalls</a>
      <ul class="post-toc-sublist">
        <li><a href="#contamination">Data Contamination</a></li>
        <li><a href="#saturation">Saturation</a></li>
        <li><a href="#goodharts">Goodhart's Law</a></li>
      </ul>
    </li>
    <li><a href="#evaluation-design">Designing an Evaluation Suite</a></li>
  </ul>
</nav>

---

## Overview
{: #overview}

Evaluation is the feedback loop that distinguishes engineering from guessing. A metric answers a specific question about model behaviour — choosing the wrong metric means optimising the wrong objective.

**Core principle:** every metric is a compressed, lossy summary of model behaviour. No single metric captures everything. Report a *suite* of metrics, and always ask: what would a bad model look like on this metric, and would I want to deploy it?

| Domain | Primary metrics | Secondary metrics |
|---|---|---|
| Classification | Precision, Recall, F1, AUC-ROC | MCC, NPV, EER |
| Regression | MAE, RMSE | R², Huber loss |
| Ranking | NDCG@K, MRR | MAP, Hit Rate@K |
| NLP generation | BERTScore, ROUGE-L | BLEU, METEOR |
| LLM factuality | FActScore, RAGAS Faithfulness | TruthfulQA accuracy |
| Code generation | PASS@k | Syntax correctness, runtime |
| Calibration | ECE | MCE, reliability diagram |

---

## Classification Metrics
{: #classification}

### Confusion Matrix
{: #confusion-matrix}

All classification metrics derive from four counts:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP (True Positive) | FN (False Negative — Type II) |
| **Actual Negative** | FP (False Positive — Type I) | TN (True Negative) |

**Accuracy** — fraction of all predictions that are correct:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Misleading under class imbalance: a classifier that always predicts the majority class achieves high accuracy while being useless. With 99% negatives, a trivial "always predict negative" classifier gets 99% accuracy.

### Precision, Recall & F-scores
{: #precision-recall}

**Precision** — of everything the model labelled positive, what fraction was actually positive?

$$\text{Precision} = \frac{TP}{TP + FP}$$

High precision → few false alarms. Use when false positives are costly (spam filters, legal review, shoplifter detection).

**Recall (Sensitivity, TPR)** — of all actual positives, what fraction did the model find?

$$\text{Recall} = \frac{TP}{TP + FN}$$

High recall → few missed positives. Use when false negatives are costly (disease screening, fraud detection, safety systems).

**Specificity (TNR)** — of all actual negatives, what fraction did the model correctly label negative?

$$\text{Specificity} = \frac{TN}{TN + FP}$$

**Negative Predictive Value (NPV)** — of all predicted negatives, what fraction was actually negative?

$$\text{NPV} = \frac{TN}{TN + FN}$$

**Precision-Recall tradeoff.** Lowering the decision threshold increases recall (more positives caught) at the cost of precision (more false alarms). The right operating point depends on the relative cost of FP vs. FN in the application.

> **Interview question:** A medical test for a rare disease (1% prevalence) achieves 90% precision and 80% recall. Your manager wants to improve precision to 99% by raising the decision threshold. What are the consequences?
>
> *Raising the threshold to hit 99% precision will dramatically lower recall — the model will miss many true positives. With 1% prevalence, even high precision isn't as impressive as it sounds: 99% precision means 1% of positives are false alarms, but the missed cases (false negatives) in disease detection are far more costly. The better framing: compute the cost of a missed diagnosis vs. a false alarm, then choose the threshold that minimises expected cost. Report the full precision-recall curve, not a single operating point.*

**F₁ score** — harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

The harmonic mean penalises extreme imbalances: a model with precision 1.0 and recall 0.01 gets $F_1 \approx 0.02$, not 0.5.

**F$_\beta$ score** — weighted harmonic mean:

$$F_\beta = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

- $\beta < 1$: weights precision more (false positives are costlier)
- $\beta > 1$: weights recall more (false negatives are costlier)
- $\beta = 2$ (F₂) is common in medical screening

**Multi-class averaging:**

| Strategy | Formula | Use when |
|---|---|---|
| **Macro** | Unweighted mean across classes | All classes equally important |
| **Micro** | Aggregate TP/FP/FN globally then compute | Class-imbalanced; dominated by large classes |
| **Weighted** | Mean weighted by class support | Reflects real-world class distribution |

### ROC, AUC & PR Curves
{: #roc-auc}

**ROC curve** plots TPR (Recall) vs. FPR across all decision thresholds:

$$\text{TPR} = \frac{TP}{TP + FN}, \qquad \text{FPR} = \frac{FP}{FP + TN}$$

The diagonal ($\text{TPR} = \text{FPR}$) is a random classifier. A perfect classifier reaches the top-left corner (TPR=1, FPR=0).

**AUROC (AUC-ROC)** — area under the ROC curve:
- 0.5: random classifier
- 1.0: perfect classifier
- Probabilistic interpretation: the probability that the model ranks a random positive higher than a random negative

AUROC is threshold-independent — it summarises performance across all operating points. Best for balanced datasets or when class distribution at test time matches training.

**Precision-Recall curve** plots Precision vs. Recall across thresholds. Preferred over ROC for severely imbalanced datasets: ROC can look optimistic when negatives vastly outnumber positives (FPR stays small even with many false positives).

**AUPRC (AUC-PR)** — area under the PR curve. A no-skill baseline equals the fraction of positives in the dataset (the prior). AUPRC is the right summary metric for imbalanced classification (fraud, medical rare events, anomaly detection).

**Equal Error Rate (EER)** — the threshold where FPR equals FNR (false negative rate). Used in biometric authentication where both error types are equally undesirable.

**Detection Error Tradeoff (DET) curve** — plots False Rejection Rate (FRR) vs. False Acceptance Rate (FAR) on non-linear (normal deviate) scales. More visually informative than ROC in the low-error regime common in authentication systems.

### MCC & Imbalanced Classes
{: #mcc}

**Matthews Correlation Coefficient (MCC)** — the most reliable single metric for binary classification under class imbalance, because it uses all four confusion matrix cells:

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

Range: $[-1, +1]$. $+1$ is perfect prediction, $0$ is no better than random, $-1$ is perfect inverse.

**When F1 fails.** F1 is entirely insensitive to true negatives — a model predicting "positive" for every sample gets $F_1 = 2P/(P+1)$ which can be high when positives dominate. MCC cannot be gamed this way. Prefer MCC whenever class imbalance is severe and true negatives matter.

| Metric | TP | TN | FP | FN | Limitation |
|---|---|---|---|---|---|
| Accuracy | ✓ | ✓ | ✓ | ✓ | Misleading under imbalance |
| Precision | ✓ | — | ✓ | — | Ignores FN, TN |
| Recall | ✓ | — | — | ✓ | Ignores FP, TN |
| F1 | ✓ | — | ✓ | ✓ | Ignores TN |
| MCC | ✓ | ✓ | ✓ | ✓ | None |

---

## Regression Metrics
{: #regression}

### MAE, MSE, RMSE
{: #mae-mse}

**Mean Absolute Error (MAE / L1):**

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

Probabilistic basis: MLE under Laplace noise. Interpretable in the original units. Constant gradient ($\pm 1$) makes it robust to outliers but can cause oscillation near the optimum. Non-differentiable at zero — use subgradient or Huber for gradient-based optimisation.

**Mean Squared Error (MSE / L2):**

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

Probabilistic basis: MLE under Gaussian noise. Smooth, differentiable — preferred as a training loss. Quadratic penalty amplifies outliers: a single $10\times$ error contributes $100\times$ more than a $1\times$ error.

**Root MSE (RMSE):**

$$\text{RMSE} = \sqrt{\text{MSE}}$$

Returns to original units (same as MAE). Shares MSE's outlier sensitivity. RMSE $\geq$ MAE always; equality holds only when all errors are equal.

**RMSLE (Root Mean Squared Log Error):**

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(1+\hat{y}_i) - \log(1+y_i))^2}$$

Useful when targets span orders of magnitude (e.g., revenue prediction). Penalises under-prediction more than over-prediction.

**Huber loss** — hybrid:

$$\mathcal{L}_\delta(a) = \begin{cases} \frac{1}{2} a^2 & |a| \leq \delta \\ \delta(|a| - \frac{1}{2}\delta) & |a| > \delta \end{cases}$$

Quadratic near zero (smooth convergence), linear for large errors (outlier robustness). Best practical regression loss when outliers are present.

| Metric | Outlier robust | Units | Differentiable | Use for |
|---|---|---|---|---|
| MAE | Yes | Original | No (at 0) | Evaluation; robust training |
| MSE | No | Squared | Yes | Training loss |
| RMSE | No | Original | Yes | Evaluation; comparable to MAE |
| Huber | Yes | Original | Yes | Training with outliers |
| RMSLE | Partial | Log-scale | Yes | Large-range targets |

### R² & Adjusted R²
{: #r-squared}

**R² (Coefficient of Determination)** — fraction of variance explained by the model:

$$R^2 = 1 - \frac{SS_\text{res}}{SS_\text{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

Range: $(-\infty, 1]$. $R^2 = 1$ is perfect fit; $R^2 = 0$ means the model does no better than predicting the mean; $R^2 < 0$ means it's worse than the mean. Adding any predictor (even noise) can only increase $R^2$ — this motivates Adjusted $R^2$.

**Adjusted R²** — penalises adding irrelevant predictors:

$$R^2_\text{adj} = 1 - \left[\frac{n-1}{n-k-1}(1 - R^2)\right]$$

where $n$ is sample size and $k$ is number of predictors. Decreases when a new predictor adds less than one expected unit of explained variance.

---

## Ranking & Retrieval Metrics
{: #ranking}

### NDCG
{: #ndcg}

Normalised Discounted Cumulative Gain (NDCG) is the standard ranking metric when relevance is graded (not just binary).

**Cumulative Gain (CG):** sum of relevance scores for top $p$ positions:

$$\text{CG}_p = \sum_{i=1}^p \text{rel}_i$$

**Discounted Cumulative Gain (DCG):** positions lower in the ranking are discounted logarithmically:

$$\text{DCG}_p = \sum_{i=1}^p \frac{2^{\text{rel}_i} - 1}{\log_2(i+1)}$$

The $2^{\text{rel}_i} - 1$ numerator (alternative formulation) amplifies the value of highly relevant items. The $\log_2(i+1)$ denominator means position 1 has no discount, position 2 is divided by 1, position 3 by $\log_2 4 = 2$, etc.

**NDCG:** normalise by the Ideal DCG (IDCG — DCG of the perfect ranking):

$$\text{NDCG}_p = \frac{\text{DCG}_p}{\text{IDCG}_p}$$

Range $[0, 1]$. NDCG = 1 only when the ranking is perfect. The key insight: NDCG rewards getting the most relevant items to the top, not just anywhere in the list.

### MRR & MAP
{: #mrr-map}

**Mean Reciprocal Rank (MRR)** — for each query, reciprocal rank of the first relevant result:

$$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$

If the first relevant result is at position 1, 2, 3, the score is 1.0, 0.5, 0.33. Simple and interpretable; only considers the single highest-ranked relevant item. Best for tasks where there is one correct answer (e.g., FAQ retrieval).

**Mean Average Precision (MAP)** — averages precision at each relevant item's position:

$$\text{AP} = \frac{1}{R} \sum_{k=1}^{n} \text{Precision@}k \cdot \mathbb{1}[\text{doc}_k \text{ is relevant}]$$

$$\text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}_q$$

where $R$ is the number of relevant documents. MAP considers all relevant items, not just the first — penalises systems that retrieve relevant docs late even if they eventually find them all.

### Precision@K, Recall@K, Hit Rate
{: #hit-rate}

**Precision@K** — fraction of top-K results that are relevant:

$$\text{Precision@}K = \frac{|\text{relevant} \cap \text{top-}K|}{K}$$

**Recall@K** — fraction of all relevant results retrieved in top-K:

$$\text{Recall@}K = \frac{|\text{relevant} \cap \text{top-}K|}{|\text{relevant}|}$$

**Hit Rate@K (HR@K)** — binary: does any relevant item appear in the top-K? Averaged over queries:

$$\text{HR@}K = \frac{1}{|Q|} \sum_q \mathbb{1}[|\text{relevant}_q \cap \text{top-}K_q| > 0]$$

| Metric | Cares about order | Handles graded relevance | Best for |
|---|---|---|---|
| Precision@K | No | No | Quick quality check |
| Recall@K | No | No | Coverage check |
| MRR | Yes (first hit) | No | One-answer retrieval |
| MAP | Yes (all hits) | No | Multi-answer retrieval |
| NDCG@K | Yes | Yes | Recommendation, search |
| HR@K | No | No | Sparse relevant sets |

---

## NLP & Generation Metrics
{: #nlp-metrics}

### Perplexity
{: #perplexity}

Perplexity measures how well a language model predicts a held-out sequence. It is the exponentiated average negative log-likelihood per token:

$$\text{PPL}(w_{1:N}) = \exp\!\left(-\frac{1}{N} \sum_{i=1}^N \log p(w_i \mid w_{<i})\right)$$

Intuition: a perplexity of $k$ means the model is as confused as if it had to choose uniformly among $k$ equally likely tokens at each step. PPL = 1 is perfect; higher is worse.

**Limitations:** perplexity is model-specific — you cannot compare perplexity across models with different tokenisers, as token granularity changes the numbers. Also insensitive to many quality dimensions (factuality, coherence, instruction following). Use as a training-time signal, not a deployment quality metric.

**Burstiness** — detects AI-generated text. Human writing shows high variance in sentence length and structure (bursty); LLM output tends to be anti-bursty (uniform):

$$b = \frac{\sigma_\tau / m_\tau - 1}{\sigma_\tau / m_\tau + 1} \in [-1, 1]$$

Values near $-1$ indicate AI-like uniformity; values near $+1$ indicate human-like variation.

### BLEU
{: #bleu}

[BLEU](https://aclanthology.org/P02-1040/) (BiLingual Evaluation Understudy, Papineni et al. 2002) is a precision-oriented n-gram overlap metric for machine translation:

$$\text{BLEU}_N = BP \cdot \exp\!\left(\sum_{n=1}^N w_n \log p_n\right)$$

where $p_n$ is the modified n-gram precision at order $n$ and $w_n = 1/N$.

**Brevity penalty** — prevents gaming by outputting very short strings:

$$BP = \begin{cases} 1 & \ell_\text{hyp} > \ell_\text{ref} \\ \exp(1 - \ell_\text{ref}/\ell_\text{hyp}) & \ell_\text{hyp} \leq \ell_\text{ref} \end{cases}$$

**Modified n-gram precision** clips each n-gram count by its maximum occurrence in any reference:

$$p_n = \frac{\sum_{\text{n-gram} \in \hat{y}} \min(\text{count}(\text{n-gram}, \hat{y}),\; \max_r \text{count}(\text{n-gram}, r))}{\sum_{\text{n-gram} \in \hat{y}} \text{count}(\text{n-gram}, \hat{y})}$$

Unigram matches measure adequacy; longer n-gram matches measure fluency. BLEU-4 (up to 4-grams) is standard. Excellent scores are typically 0.6–0.7 — humans rarely score 1.0 against each other.

**Limitations:** no recall component; ignores synonyms and paraphrases; sensitive to tokenisation; poor correlation with human judgement on creative or diverse tasks.

### ROUGE
{: #rouge}

[ROUGE](https://aclanthology.org/W04-1013/) (Recall-Oriented Understudy for Gisting Evaluation, Lin 2004) is recall-oriented and primarily used for summarisation.

**ROUGE-N** — n-gram recall against reference:

$$\text{ROUGE-N} = \frac{\sum_{\text{n-gram} \in \text{ref}} \text{count}_\text{match}(\text{n-gram})}{\sum_{\text{n-gram} \in \text{ref}} \text{count}(\text{n-gram})}$$

**ROUGE-L** — based on Longest Common Subsequence (LCS):

$$R_\text{LCS} = \frac{\text{LCS}(\text{ref}, \hat{y})}{\ell_\text{ref}}, \qquad P_\text{LCS} = \frac{\text{LCS}(\text{ref}, \hat{y})}{\ell_{\hat{y}}}$$

$$\text{ROUGE-L} = \frac{(1+\beta^2) R_\text{LCS} P_\text{LCS}}{R_\text{LCS} + \beta^2 P_\text{LCS}}$$

LCS captures sentence-level structure without requiring contiguous matches — it rewards correct ordering even when words are non-adjacent. ROUGE-L is the standard for summarisation evaluation.

**BLEU vs. ROUGE:**

| | BLEU | ROUGE |
|---|---|---|
| Orientation | Precision | Recall |
| Primary task | Translation | Summarisation |
| N-gram | Precision over candidate | Recall over reference |
| LCS variant | None | ROUGE-L |
| Shared limitation | Syntactic only; no semantic understanding |

### BERTScore & MoverScore
{: #bertscore}

**[BERTScore](https://arxiv.org/abs/1904.09675)** (Zhang et al. 2020) uses contextual token embeddings from a pretrained transformer (RoBERTa, XLM-R) to compute semantic similarity:

1. Extract token embeddings for candidate $\hat{y}$ and reference $r$.
2. For each candidate token, find maximum cosine similarity to any reference token (precision).
3. For each reference token, find maximum cosine similarity to any candidate token (recall).
4. Aggregate:

$$P_\text{BERT} = \frac{1}{|\hat{y}|} \sum_{i \in \hat{y}} \max_{j \in r} \cos(\hat{y}_i, r_j)$$

$$R_\text{BERT} = \frac{1}{|r|} \sum_{j \in r} \max_{i \in \hat{y}} \cos(\hat{y}_i, r_j)$$

$$F_\text{BERT} = 2 \cdot \frac{P_\text{BERT} \cdot R_\text{BERT}}{P_\text{BERT} + R_\text{BERT}}$$

Captures synonyms and paraphrases that BLEU/ROUGE miss. Correlates better with human judgement across translation, summarisation, and generation tasks.

**MoverScore** computes semantic distance using Word Mover's Distance (optimal transport) over contextual embeddings — the "cost" of transforming the candidate distribution into the reference distribution in embedding space. Handles word-order variations and semantic reordering better than token-level matching.

**BLEURT** ([Sellam et al. 2020](https://arxiv.org/abs/2004.04696)) — fine-tunes BERT on human quality ratings. Currently the best-correlating reference-based metric with human judgement on translation and summarisation.

**Metric summary for NLP generation:**

| Metric | Semantic | Reference-free | Human correlation | Cost |
|---|---|---|---|---|
| BLEU | No | No | Low | Negligible |
| ROUGE-L | No | No | Medium | Negligible |
| BERTScore | Yes | No | High | Medium |
| BLEURT | Yes | No | Very high | Medium |
| LLM-as-judge | Yes | Yes | Very high | High |

### PASS@k
{: #pass-k}

For code generation, functional correctness matters more than token overlap. PASS@k is the probability that at least one of $k$ generated solutions passes all test cases:

$$\text{PASS@}k = \mathbb{E}_\text{problems}\!\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

where $n$ is the number of samples generated, $c$ is the number that pass. This unbiased estimator avoids the need to generate exactly $k$ samples. PASS@1 measures single-attempt success; PASS@10 and PASS@100 measure whether any solution in a larger sample works.

---

## Detection & Segmentation Metrics
{: #detection}

**Intersection over Union (IoU)** — measures bounding box overlap quality:

$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$$

Threshold: IoU $\geq 0.5$ is considered a true positive in most benchmarks (PASCAL VOC); COCO uses IoU from 0.5 to 0.95 in steps of 0.05.

**Average Precision (AP)** — area under the precision-recall curve at a specific IoU threshold. Summarises performance across confidence thresholds.

**Mean Average Precision (mAP)** — AP averaged over all object classes (and over IoU thresholds in COCO):

$$\text{mAP} = \frac{1}{|\mathcal{C}|} \sum_{c \in \mathcal{C}} \text{AP}_c$$

mAP is the primary metric for PASCAL VOC, COCO, and most object detection benchmarks. It simultaneously measures classification accuracy (is it the right class?), localisation accuracy (is the box tight?), and confidence calibration (are scores well-ordered?).

**FID (Fréchet Inception Distance)** — for generative image models, measures the distance between real and generated image feature distributions using Inception network embeddings:

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2})$$

Lower is better. FID = 0 means the distributions are identical. Sensitive to sample size — needs $\geq 10$K samples for stable estimates.

---

## Calibration Metrics
{: #calibration}

A calibrated model's confidence scores match empirical accuracy: when the model says 80% confidence, it should be correct 80% of the time.

**Expected Calibration Error (ECE)** — partitions predictions into $M$ confidence bins and measures the weighted average gap:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

**Maximum Calibration Error (MCE)** — worst-case bin gap. Relevant when any extreme overconfidence is unacceptable (medical, safety).

$$\text{MCE} = \max_m \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$

A reliability diagram plots $\text{conf}$ vs. $\text{acc}$ per bin — perfect calibration is the diagonal. RLHF-trained models often exhibit overconfidence (conf > acc) on factual questions because confident-sounding responses were preferred during training.

---

## LLM Benchmarks
{: #llm-benchmarks}

### Knowledge & Reasoning
{: #knowledge-reasoning}

**[MMLU](https://arxiv.org/abs/2009.03300)** (Massive Multitask Language Understanding, Hendrycks et al. 2021)
- 57 subjects spanning STEM, humanities, social sciences, medicine, law
- 4-choice multiple-choice; ~16K test questions
- Measures breadth of knowledge; mixes knowledge retrieval with reasoning
- **Limitation:** cannot distinguish memorisation from reasoning; saturating (frontier models >85%); contamination risk high due to age

**[MMLU-Pro](https://arxiv.org/abs/2406.01574)**
- 12K harder questions with 10 options each (vs. 4)
- CoT results ~20% higher than greedy — explicitly rewards reasoning
- Manually expert-reviewed; harder to game
- Better separation between frontier models

**[GPQA](https://arxiv.org/abs/2311.12022)** (Graduate-Level Google-Proof Q&A)
- 448 expert-written questions in biology, physics, chemistry
- PhD experts reach 65% accuracy; non-experts with web access reach 34%
- "Google-proof" — answers cannot be looked up easily
- Strong signal on genuine reasoning ability; low contamination risk

**[BIG-bench Hard (BBH)](https://arxiv.org/abs/2210.09261)**
- 23 hardest tasks from BIG-bench (200+ task meta-benchmark)
- Requires advanced reasoning; CoT strongly improves performance
- Tests: algorithmic reasoning, logical deduction, causal judgement

**[ARC-AGI-2](https://arcprize.org/)**
- Grid-based visual/pattern abstraction
- Tests generalisation beyond pattern completion
- Very hard for current models; minimal contamination

**[HellaSwag](https://arxiv.org/abs/1905.07830)** / **[WinoGrande](https://arxiv.org/abs/1907.10641)** / **[CommonsenseQA](https://arxiv.org/abs/1811.00937)**
- Common-sense reasoning benchmarks
- Largely saturated for frontier models; useful for smaller model comparisons

### Math & Code
{: #math-code}

**[GSM8K](https://arxiv.org/abs/2110.14168)** (Grade School Math)
- ~8,500 grade-school word problems
- Multi-step arithmetic reasoning; measures procedural fluency
- Largely saturated (frontier models >95%); useful for smaller models and CoT ablations

**[MATH](https://arxiv.org/abs/2103.03874)** (Hendrycks et al. 2021)
- 12,500 competition-level problems across 7 difficulty levels
- Covers algebra, calculus, number theory, combinatorics
- Still differentiates frontier models; CoT essential

**MATH-500** — 500-problem subset with step-level labels from OpenAI's PRM800K (800K step-level correctness annotations).

**AIME / AMC** — mathematical olympiad-level; AIME 2024 used as a frontier differentiator. Integer answers 0–999; near-zero contamination.

**[HumanEval](https://arxiv.org/abs/2107.03374)** (Chen et al. 2021)
- 164 hand-crafted Python function synthesis problems
- PASS@k evaluation; each problem has a test suite
- **Limitation:** small, largely saturated, Python-only

**[HumanEval+](https://arxiv.org/abs/2305.01210)** — 300 problems with wider difficulty range.

**[MBPP](https://arxiv.org/abs/2108.07732) / MBPP+** — 974/1500 basic Python problems; broader than HumanEval.

**[SWE-bench](https://arxiv.org/abs/2310.06770)** — real GitHub issues requiring bug fixes and feature implementation on real codebases. Verified subset uses triple-verified end-to-end tests. Far harder than HumanEval; tests genuine software engineering ability.

**[SWE-Lancer](https://arxiv.org/abs/2410.04752)** — 1,400+ real freelance tasks from Upwork with $1M in real payouts. Graded by experienced engineers. Evaluates economic value of generated code.

**[LiveCodeBench](https://arxiv.org/abs/2403.07974)** — real-time coding problems; contamination-free by design (pulls from recent contests).

**[BigCodeBench](https://arxiv.org/abs/2406.15877)** — 1,140 tasks covering 139 Python libraries; 5.6 test cases each. GPT-4 achieves 61.1%.

| Benchmark | Domain | Contamination risk | Saturation risk |
|---|---|---|---|
| GSM8K | Grade math | High | High |
| MATH | Competition math | Medium | Medium |
| AIME 2024 | Olympiad math | Low | Low |
| HumanEval | Python functions | High | High |
| SWE-bench | Real repos | Low | Low |
| LiveCodeBench | Recent contests | Very low | Low |

### Instruction Following
{: #instruction-following}

**[IFEval](https://arxiv.org/abs/2311.07911)** (Instruction Following Evaluation)
- 541 prompts with 1–3 programmatically verifiable constraints
- Constraint types: format (JSON, markdown), length (word count), content (include/exclude keywords), style
- Scores: prompt-level strict (all constraints satisfied) and instruction-level (per-constraint fraction)
- Clean signal; no human raters needed; hard to game

**[MT-Bench](https://arxiv.org/abs/2306.05685)** (Multi-Turn Benchmark)
- 80 multi-turn questions across 8 categories (writing, roleplay, extraction, reasoning, math, coding, knowledge, STEM)
- GPT-4 as judge; each turn scored 1–10
- Correlates well with LMSYS Arena human rankings

**[AlpacaEval 2.0](https://arxiv.org/abs/2404.04475)**
- Win rate against GPT-4 Turbo as reference
- Length-controlled version corrects for verbosity bias in LLM judges
- Good proxy for general instruction-following quality

**[Arena-Hard](https://arxiv.org/abs/2406.11939)**
- 500 high-difficulty technical questions
- GPT-4o as judge; measures ability on hard real-world tasks
- Strong correlation with LMSYS Arena

### Factuality & Safety
{: #factuality-safety}

**[TruthfulQA](https://arxiv.org/abs/2109.07958)** (Lin et al. 2022)
- 817 questions probing common human misconceptions
- Two-part: truthfulness (is the answer true?) and informativeness (is it useful?)
- Models trained on internet text often reproduce falsehoods with high confidence

**[SimpleQA](https://arxiv.org/abs/2411.07905)** (Wei et al. 2024)
- 4,326 short-form factoid questions with single verified answers
- Includes abstention scoring — penalises confident wrong answers, rewards admitting uncertainty
- Good signal on factual precision and calibration together

**[FreshQA](https://arxiv.org/abs/2310.03214)**
- Time-sensitive questions; answers change as facts change
- Tests temporal factuality and knowledge cutoff awareness

**HaluEval** — large-scale hallucination detection benchmark covering QA, dialogue, and summarisation; tests whether models can identify hallucinated content.

### Long Context
{: #long-context}

**[LongBench](https://arxiv.org/abs/2308.14508)**
- Tasks requiring 16K+ token contexts
- Long-form QA, summarisation, multi-document analysis, few-shot learning
- Tests whether models actually use long-range context vs. relying on position 0

**[RULER](https://arxiv.org/abs/2404.06654)**
- Synthetic and natural tasks testing effective context use
- Needle-in-a-haystack variants at various positions; multi-needle retrieval
- Good for characterising the usable context length (often much shorter than nominal)

**Needle-in-a-Haystack** — places a specific fact at varying depths in a long document and asks the model to retrieve it. Reveals "lost-in-the-middle" degradation at central positions.

### RAG & Retrieval
{: #rag-benchmarks}

**[BEIR](https://arxiv.org/abs/2104.08663)** (Benchmarking IR)
- 18 diverse retrieval tasks in zero-shot setting
- Covers news, medical, scientific, legal, argument retrieval
- Standard for evaluating retrieval generalisation

**[MS MARCO](https://arxiv.org/abs/1611.09268)**
- 8.8M web passages; 1M training queries from Bing
- Standard re-ranker training and evaluation dataset; MRR@10 is primary metric

**[HotpotQA](https://hotpotqa.github.io/)** — 2-hop reasoning across Wikipedia; tests multi-hop retrieval.

**[MultiHop-RAG](https://arxiv.org/abs/2401.15391)** — designed specifically for end-to-end RAG pipeline evaluation.

**[KILT](https://arxiv.org/abs/2009.02252)** — unified knowledge-intensive language tasks combining retrieval and generation (FEVER, TriviaQA, NaturalQuestions, Wizard of Wikipedia).

**[RAG-QA Arena](https://arxiv.org/abs/2407.13998)** — evaluates domain robustness for long-form retrieval-augmented QA.

### Agent Benchmarks
{: #agent-benchmarks}

**[AgentBench](https://arxiv.org/abs/2308.03688)**
- Multi-environment evaluation: OS tasks, DB queries, knowledge graph, web browsing, card games
- Tests planning, memory, tool use across diverse settings

**[τ-bench](https://arxiv.org/abs/2406.12045)**
- Policy-constrained tool-agent-user interaction
- Domain-specific rule-following with database reasoning
- Uses $\text{pass}^k$ metric — probability all $k$ attempts satisfy constraints

**[ToolBench](https://arxiv.org/abs/2307.16789)**
- Evaluates tool integration across 16,464 real-world APIs
- Tests tool selection, argument generation, error recovery

**[WebArena](https://arxiv.org/abs/2307.13854)** / **[OSWorld](https://arxiv.org/abs/2404.07972)**
- Web and OS task execution in realistic simulations
- Hard; requires multi-step planning and UI interaction

**[GAIA](https://arxiv.org/abs/2311.12983)**
- General AI assistants benchmark: real-world questions requiring multi-step reasoning, tool use, file handling
- Three levels of difficulty; frontier models still far from human performance

---

## VLM Benchmarks
{: #vlm-benchmarks}

**Visual QA:**

| Benchmark | Focus | Notes |
|---|---|---|
| **VQAv2** | Image-based Q&A | Balanced; reduces language bias vs. VQA v1 |
| **TextVQA** | Reading text in images | Signs, documents, packaging |
| **[MMMU](https://arxiv.org/abs/2311.16502)** | College-level multimodal | 30 subjects; text + image reasoning |
| **[MMBench](https://arxiv.org/abs/2307.06281)** | Comprehensive VLM | VQA, captioning, reasoning pipeline |
| **Humanity's Last Exam (HLE)** | Expert multimodal | 2,700 hard questions; unsolvable by current LLMs |

**Document & Chart Understanding:**

| Benchmark | Focus | Notes |
|---|---|---|
| **DocVQA** | Scanned document Q&A | OCR + comprehension |
| **ChartQA** | Chart interpretation | Bar/line/pie chart reasoning |
| **TabFact** | Table fact verification | 118K statements vs. 16K tables |
| **InfographicVQA** | Mixed visual/text/numeric | Complex infographics |

**Agent & World Interaction:**

| Benchmark | Focus | Notes |
|---|---|---|
| **OSWorld** | OS task execution | Multi-step GUI interaction |
| **WebVoyager** | Autonomous web navigation | Adapts to unfamiliar web structures |
| **IGLU** | 3D environment tasks | Spatial reasoning; real-time execution |

**Medical VLMs:**

| Benchmark | Focus | Notes |
|---|---|---|
| **CheXpert** | Chest X-ray classification | 200K+ images; 14 conditions |
| **MedQA** | USMLE-based Q&A | Detailed explanations; clinical reasoning |
| **BioASQ** | Biomedical QA | PubMed abstracts; yes/no + factoid |
| **MIMIC-III** | Clinical outcomes prediction | 40K+ patients; EHR data |

---

## Benchmark Pitfalls
{: #benchmark-pitfalls}

### Data Contamination
{: #contamination}

Contamination occurs when benchmark test examples appear in the model's pretraining data. Since most benchmarks are public, their questions and answers are crawled into web datasets. A model that has "seen" TruthfulQA or GSM8K during training will score higher through memorisation rather than reasoning.

**Detection approaches:**
- Canary strings: embed unique phrases in test sets; check if the model can complete them
- N-gram overlap: measure overlap between training data and benchmark questions
- Temporal holdout: use benchmarks released after the training cutoff (LiveCodeBench, AIME 2024)
- Perturbation testing: rephrase questions; true understanding should be robust to surface form changes

**Red flags in papers:** a model that scores dramatically higher on a benchmark than on comparable difficulty tasks, or that shows unusually low calibration on benchmark questions (overconfident correct answers).

### Saturation
{: #saturation}

Benchmark saturation occurs when frontier models approach the ceiling, making discrimination impossible. The signal-to-noise ratio collapses.

**Saturation timeline for key benchmarks:**

| Benchmark | Introduced | Approximate saturation year |
|---|---|---|
| GLUE | 2018 | 2020 |
| SuperGLUE | 2019 | 2021 |
| HellaSwag, WinoGrande | 2019 | 2022 |
| GSM8K | 2021 | 2024 |
| HumanEval | 2021 | 2024 |
| MMLU | 2021 | 2024–2025 |
| MATH | 2021 | 2025 |

Responses to saturation: harder variants (MMLU-Pro, MATH-500, HumanEval+), live benchmarks (LiveCodeBench), expert-curated hard sets (GPQA, AIME, HLE), and human preference arenas (LMSYS Arena) that are inherently hard to saturate.

### Goodhart's Law
{: #goodharts}

"When a measure becomes a target, it ceases to be a good measure." Models trained or selected against specific benchmarks overfit to their format and quirks — high scores no longer imply the underlying capability.

Examples:
- RLHF-trained models learn to produce outputs that score well on reward models trained from human preferences, which can diverge from actual quality (verbosity bias, sycophancy)
- Models can learn ARC-e shortcuts without genuine reasoning
- BLEU-optimised MT systems produce fluent text that nonetheless mistranslates meaning

**Mitigation:** rotate benchmarks regularly; use held-out test sets; prefer live/dynamic benchmarks; always pair automated metrics with human evaluation for high-stakes decisions.

### Arena & Human Preference
{: #arena}

**[LMSYS Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/)** — blind pairwise comparison by real users. Models are ranked via ELO rating computed from thousands of human preference votes.

Properties:
- Ground truth from real users on real queries (not a curated test set)
- Hard to overfit — the query distribution is unknown and diverse
- Slow to move — statistical significance requires many battles
- Reflects what people prefer, not necessarily what is correct or safe

The ELO score from Chatbot Arena is currently the most trusted overall quality signal for general-purpose LLMs, because it directly measures user preference at scale without benchmark-specific gaming.

**[MT-Bench](https://arxiv.org/abs/2306.05685)** — strong correlation with Arena ELO; faster to run but uses a fixed question set and is susceptible to prompt engineering.

---

## Designing an Evaluation Suite
{: #evaluation-design}

A production evaluation suite should answer three questions: (1) does the model do the right thing?, (2) does it avoid doing the wrong thing?, (3) how does it compare to alternatives?

**Step 1 — Define the task distribution.** Sample from real user queries (if available), not just benchmark data. The evaluation set should match the production query distribution.

**Step 2 — Choose metrics per quality dimension:**

| Dimension | Metric | Automated? |
|---|---|---|
| Task accuracy | Domain-specific (F1, NDCG, PASS@k) | Yes |
| Factuality | FActScore, RAGAS Faithfulness | Partial (LLM judge) |
| Instruction following | IFEval (programmatic constraints) | Yes |
| Calibration | ECE on confidence estimates | Yes |
| Safety | Refusal precision/recall | Partial |
| Overall preference | LLM-as-judge (MT-Bench style) or human | Partial/No |

**Step 3 — Build a regression suite.** Track all metrics on every checkpoint. Alert on any metric dropping >2% relative from the previous checkpoint.

**Step 4 — Stratify by difficulty and domain.** Aggregate metrics hide regressions in subpopulations. Report separately: easy/medium/hard, query category, language, context length.

**Step 5 — Human evaluation cadence.** Automated metrics are cheap and fast; human eval is expensive but authoritative. Run human eval: (a) before any production deployment, (b) monthly on a stratified sample, (c) whenever automated metrics show conflicting signals.

**Step 6 — Monitor live.** Instrument production for implicit signals: thumbs up/down, session length, follow-up questions, copy rate. These reflect real user satisfaction with no annotation cost.

> **Interview question:** You train a new model checkpoint and it improves on MMLU by 3% but drops on MT-Bench by 5%. How do you decide whether to deploy it?
>
> *MMLU and MT-Bench measure different things — MMLU tests knowledge breadth (mostly memorisation at this scale), while MT-Bench tests instruction following and multi-turn coherence. A 5% drop on MT-Bench is more concerning for a chat assistant deployment. First, decompose the MT-Bench drop: is it uniform across categories, or concentrated in specific task types (e.g., only coding, or only multi-turn)? Then check: (1) is the MMLU gain on tasks that actually appear in production, or on obscure academic subjects? (2) Run LMSYS Arena or internal human preference evaluation on a sample of production-like queries — does the new model win or lose? (3) Check safety and factuality metrics. Deploy only if human preference evaluation on the production distribution favours the new checkpoint despite the MT-Bench drop.*

---

## Also Read

**[Response Quality in LLMs](/blogs/response-quality/)** — factuality, reasoning, calibration, instruction following, and the full evaluation toolkit for LLM outputs.

**[RAG](/blogs/rag/)** — RAGAS metrics, BEIR, retrieval evaluation, and pipeline-level factuality measurement.

**[Fine-tuning LLMs](/blogs/finetuning/)** — reward modelling, RLHF, and how training objectives interact with evaluation targets.
