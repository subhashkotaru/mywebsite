---
title: "Classic Machine Learning"
date: 2026-04-21
display_order: 12
description: "The foundations of machine learning — supervised and unsupervised learning, core algorithms, model selection, and the statistical principles that underpin modern ML."
tags: [machine-learning, statistics, algorithms]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#supervised">Supervised Learning</a>
      <ul class="post-toc-sublist">
        <li><a href="#linear-models">Linear Models</a></li>
        <li><a href="#bayesian-linear-regression">Bayesian Linear Regression</a></li>
        <li><a href="#ridge-lasso">Ridge & Lasso Regression</a></li>
        <li><a href="#decision-trees">Decision Trees & Ensembles</a></li>
        <li><a href="#gradient-boosted-trees">Gradient Boosted Trees (XGBoost / LightGBM)</a></li>
        <li><a href="#neural-networks">Neural Networks & MLPs</a></li>
        <li><a href="#svms">Support Vector Machines</a></li>
        <li><a href="#lda">Linear Discriminant Analysis</a></li>
        <li><a href="#knn">k-Nearest Neighbours</a></li>
      </ul>
    </li>
    <li><a href="#unsupervised">Unsupervised Learning</a>
      <ul class="post-toc-sublist">
        <li><a href="#clustering">Clustering</a></li>
        <li><a href="#kmeans-deep">k-Means Deep Dive</a></li>
        <li><a href="#hierarchical">Hierarchical Clustering</a></li>
        <li><a href="#isolation-forest">Isolation Forest</a></li>
        <li><a href="#dimensionality-reduction">Dimensionality Reduction</a></li>
        <li><a href="#pca-deep">PCA Deep Dive</a></li>
        <li><a href="#umap">UMAP</a></li>
      </ul>
    </li>
    <li><a href="#bias-variance">Bias-Variance Tradeoff</a></li>
    <li><a href="#regularisation">Regularisation</a></li>
    <li><a href="#model-selection">Model Selection & Evaluation</a>
      <ul class="post-toc-sublist">
        <li><a href="#cross-validation">Cross-Validation</a></li>
        <li><a href="#metrics">Metrics</a></li>
        <li><a href="#hyperparameter-tuning">Hyperparameter Tuning</a></li>
      </ul>
    </li>
    <li><a href="#probabilistic">Probabilistic Models</a>
      <ul class="post-toc-sublist">
        <li><a href="#naive-bayes">Naive Bayes</a></li>
        <li><a href="#gmm">Gaussian Mixture Models</a></li>
        <li><a href="#hmm">Hidden Markov Models</a></li>
      </ul>
    </li>
    <li><a href="#optimisation">Optimisation</a></li>
  </ul>
</nav>

---

## Overview
{: #overview}

Classical machine learning predates deep learning by decades and remains practically important: it is faster to train, easier to interpret, more data-efficient, and often competitive with neural networks when data is tabular, structured, or small. More importantly, classical ML provides the conceptual foundation that deep learning is built on — loss functions, regularisation, optimisation, the bias-variance tradeoff, and cross-validation all originate here.

The central problem: given a dataset `{(xᵢ, yᵢ)}`, find a function `f: X → Y` that generalises — performs well on *unseen* examples, not just the training set. Every algorithm is a different answer to this problem, with different assumptions about the structure of `f`, different ways of measuring "performs well", and different tradeoffs between flexibility, interpretability, and sample efficiency.

<div class="post-flow post-flow--horizontal" role="group" aria-label="ML learning paradigms">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Supervised — labelled (x, y) pairs</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Unsupervised — unlabelled x only</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Semi-supervised — few labels, many unlabelled</span></li>
  </ol>
</div>

---

## Supervised Learning
{: #supervised}

### Linear Models
{: #linear-models}

**Linear regression** models the output as a linear function of the input features:

```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = wᵀx
```

The weights `w` are found by minimising mean squared error (MSE) over the training set:

```
L(w) = (1/n) Σ (yᵢ - wᵀxᵢ)²
```

The closed-form solution is `w = (XᵀX)⁻¹Xᵀy` — the **normal equations**. This is exact but requires inverting an `d×d` matrix (`d` = features), which is `O(d³)`. For large `d`, gradient descent is preferred.

**Logistic regression** replaces the linear output with a sigmoid to model binary classification:

```
P(y=1|x) = σ(wᵀx) = 1 / (1 + exp(−wᵀx))
```

Optimised by maximising log-likelihood (equivalently, minimising binary cross-entropy):

```
L(w) = -Σ [yᵢ log σ(wᵀxᵢ) + (1−yᵢ) log(1 − σ(wᵀxᵢ))]
```

No closed form — requires gradient descent. Despite the name, logistic regression is a classification algorithm that outputs calibrated probabilities.

**Why linear models still matter**: they are fast, interpretable (weights directly show feature importance), statistically well-understood, and work well when the relationship between features and target is approximately linear or when the dataset is small. Regularised logistic regression is still competitive on many tabular classification benchmarks.

**Softmax regression** generalises logistic regression to K classes: output a probability distribution over classes using the softmax function, train with cross-entropy loss. This is equivalent to a neural network with no hidden layers — the same loss and gradient used in deep learning classification.

**Pros and Cons — Linear Models**

| | Linear Regression | Logistic Regression |
|---|---|---|
| **Pros** | Closed-form solution, interpretable weights, fast training, well-calibrated uncertainty (with Bayesian treatment) | Outputs calibrated probabilities, interpretable, fast, works well with high-dimensional sparse features (text) |
| **Cons** | Assumes linear relationship; sensitive to outliers (MSE); requires feature engineering for non-linearity | Cannot model non-linear decision boundaries without manual feature expansion; requires independent features for coefficient interpretation |

> **Interview question:** Why does logistic regression output a probability, and is it actually well-calibrated?
>
> *Answer:* Logistic regression passes the linear score through a sigmoid function, which squashes the real line into (0, 1), giving an output that is interpretable as a probability. Logistic regression is generally well-calibrated because it is trained by maximising the log-likelihood — which is equivalent to minimising cross-entropy — a proper scoring rule that encourages predicted probabilities to match empirical frequencies. Empirically, logistic regression tends to produce calibrated probabilities without post-hoc adjustment, unlike SVMs or naive Bayes. However, calibration degrades when the model is very confident (probabilities near 0 or 1) or when the training distribution differs from the test distribution; Platt scaling or isotonic regression can recalibrate if needed.*

> **Interview question:** When would you choose linear regression with gradient descent over the normal equations?
>
> *Answer:* The normal equations require inverting the `d×d` matrix `XᵀX`, which costs `O(d³)` in time and `O(d²)` in memory. For problems where `d` is large — say, hundreds of thousands of features — this is prohibitive. Gradient descent iterates at `O(nd)` per epoch and requires only `O(d)` memory for the parameter vector, scaling far better with feature dimensionality. In practice, gradient descent is preferred whenever `d` exceeds a few thousand, or when the dataset is too large to materialise `XᵀX` in memory.*

### Decision Trees & Ensembles
{: #decision-trees}

A **decision tree** partitions the feature space into axis-aligned rectangular regions, assigning a prediction to each region. At each internal node, a single feature threshold splits the data; leaves hold predictions.

**Splitting criterion**: at each node, choose the feature and threshold that maximally reduce impurity. For classification, **Gini impurity**:

```
Gini(S) = 1 - Σₖ pₖ²
```

or **information gain** (entropy reduction):

```
IG = H(S) - Σ (|Sᵢ|/|S|) · H(Sᵢ)
```

For regression, variance reduction.

**Overfitting**: a tree grown to full depth memorises training data — every leaf contains exactly one sample. Pruning (removing leaves that don't improve validation loss) or setting a max depth/min samples per leaf controls this.

**Ensembles** are where decision trees truly excel:

<div class="post-flow post-flow--compare" role="group" aria-label="Bagging vs boosting">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Random Forest (Bagging)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Train B trees on bootstrap samples</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each tree uses random feature subset at each split</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Average predictions (regression) or majority vote (classification)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Reduces variance, robust to overfitting</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Gradient Boosting</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Train trees sequentially</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Each tree fits residuals of previous ensemble</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Shrinkage (learning rate) controls step size</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Reduces bias, often highest accuracy on tabular data</span></li>
    </ol>
  </div>
</div>

**Random forest** randomises both data (bootstrap) and features (random subsets at each split), making individual trees low-correlation. Averaging uncorrelated errors reduces variance without increasing bias.

**Gradient boosting** builds an additive model: `F(x) = Σₘ αₘhₘ(x)` where each `hₘ` is a shallow tree fitted to the negative gradient of the loss. **[XGBoost](https://arxiv.org/abs/1603.02754)**, **[LightGBM](https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)**, and **CatBoost** are engineering-optimised implementations that dominate structured/tabular ML competitions. LightGBM's histogram-based splitting and leaf-wise (vs level-wise) growth make it particularly fast on large datasets.

**Pros and Cons — Decision Trees & Ensembles**

| | Decision Trees | Random Forest | Gradient Boosting |
|---|---|---|---|
| **Pros** | Interpretable, handles mixed types, no normalisation needed, fast inference | Robust to overfitting, good out-of-box performance, parallelisable, feature importance | Highest accuracy on tabular data, handles missing values (XGBoost), flexible loss functions |
| **Cons** | High variance, prone to overfitting without pruning, unstable (small data changes → very different tree) | Slower inference than a single tree, harder to interpret than individual tree | Sequential training (slow to train), many hyperparameters, sensitive to noisy labels |

> **Interview question:** Explain the difference between bagging and boosting. Which reduces bias and which reduces variance?
>
> *Answer:* Bagging (bootstrap aggregating) trains many independent models on random subsamples of training data and averages their predictions. Because the models are trained independently on different data, they make uncorrelated errors, and averaging reduces variance without meaningfully changing bias. Boosting trains models sequentially, with each model correcting the errors of the previous ensemble; this progressively reduces bias. However, boosting can increase variance if too many rounds are used or the learning rate is too high — early stopping and regularisation (shrinkage, max depth, subsampling) are essential. In summary: bagging primarily reduces variance; boosting primarily reduces bias.*

> **Interview question:** How does a random forest differ from simply training many decision trees on the full dataset and averaging?
>
> *Answer:* If you train many trees on the identical full dataset with the same splitting criterion, they will all produce nearly identical trees — averaging identical predictions gains nothing. Random forests introduce two sources of randomness: bootstrap sampling (each tree trains on a different random subsample of rows) and feature randomisation (only a random subset of features is considered at each split). These two mechanisms decorrelate the trees, so their errors are partially independent. When uncorrelated errors are averaged, the ensemble variance decreases proportionally to `1/B` (for `B` trees with zero correlation), which is why random forests generalise far better than a single tree.*

### Support Vector Machines
{: #svms}

An SVM finds the **maximum-margin hyperplane** — the linear decision boundary that is as far as possible from the nearest training points (support vectors) of each class.

For linearly separable data, the hard-margin SVM solves:

```
min ½‖w‖²   subject to: yᵢ(wᵀxᵢ + b) ≥ 1  ∀i
```

The margin is `2/‖w‖` — maximising the margin is equivalent to minimising `‖w‖`. The **dual formulation** depends only on dot products `xᵢ·xⱼ`, enabling the **kernel trick**: replace dot products with a kernel function `K(xᵢ, xⱼ)` that implicitly maps inputs to a high-dimensional feature space without computing the mapping explicitly.

Common kernels:

| Kernel | Formula | Use case |
|---|---|---|
| Linear | `xᵢ·xⱼ` | High-dimensional, sparse data (text) |
| RBF (Gaussian) | `exp(−γ‖xᵢ−xⱼ‖²)` | General-purpose, smooth boundaries |
| Polynomial | `(γ xᵢ·xⱼ + r)ᵈ` | Interaction features |

The **soft-margin SVM** (for non-separable data) allows misclassifications with a penalty `C`: larger `C` = smaller margin but fewer violations, smaller `C` = larger margin but more violations. `C` is the primary hyperparameter.

SVMs with RBF kernels were state-of-the-art on many tasks before deep learning and remain competitive when data is small and features are hand-engineered. Their main limitation: training is `O(n²)` to `O(n³)` in samples, making them impractical for large datasets.

**Pros and Cons — SVMs**

| | Pros | Cons |
|---|---|---|
| **Generalisation** | Maximum-margin principle gives strong theoretical guarantees; effective in high-dimensional spaces where `d >> n` | Quadratic/cubic training complexity in `n` makes it impractical for large datasets (n > ~100k) |
| **Expressivity** | Kernel trick enables non-linear boundaries without explicit feature mapping | Kernel choice is problem-dependent and requires domain knowledge; RBF `γ` and `C` must both be tuned |
| **Interpretability** | Support vectors directly identify the most influential training points | Full model is not easily interpretable; no calibrated probability output without Platt scaling |

> **Interview question:** Explain the kernel trick. Why is it useful and what is its computational cost?
>
> *Answer:* The SVM dual formulation depends on the data only through dot products `xᵢ · xⱼ`. The kernel trick replaces these dot products with a kernel function `K(xᵢ, xⱼ)` that is equivalent to computing a dot product in a (possibly infinite-dimensional) feature space, without ever explicitly computing or storing the transformed features. For example, the RBF kernel is equivalent to an infinite-dimensional feature map. This is computationally beneficial because the cost of computing `K(xᵢ, xⱼ)` in the original space can be far cheaper than explicitly constructing the transformed representation. The cost is that the kernel matrix is `n×n`, so training scales as `O(n²)` in memory and `O(n²)` to `O(n³)` in time, limiting SVMs to smaller datasets.*

### k-Nearest Neighbours
{: #knn}

**kNN** is non-parametric: no explicit training step. To predict for a new point `x`, find the `k` closest training points and aggregate their labels (majority vote for classification, average for regression):

```
ŷ = (1/k) Σᵢ∈Nₖ(x) yᵢ
```

**Distance metric** matters: Euclidean distance works for continuous features of similar scale; Hamming for binary; cosine for high-dimensional sparse vectors (text, embeddings). Features must be normalised — kNN is highly sensitive to scale.

**Bias-variance**: `k=1` — zero training error, high variance (memorises training set). Large `k` — smooths predictions, increases bias. Optimal `k` is found by cross-validation.

**Computational cost**: naive kNN is `O(nd)` per query (`n` training points, `d` dimensions). Approximate nearest-neighbour structures (KD-trees for low `d`, HNSW/[FAISS](https://arxiv.org/abs/2401.08281) for high `d`) reduce this to sub-linear. kNN on top of learned embeddings (embedding + FAISS search) is the core of many modern retrieval systems.

**Pros and Cons — kNN**

| | Pros | Cons |
|---|---|---|
| **Training** | No training step; trivially online (just add new points) | Entire training set must be stored in memory |
| **Expressivity** | Makes no parametric assumptions about data distribution; naturally handles multi-class problems | Struggles in high dimensions (curse of dimensionality: all points become equidistant) |
| **Simplicity** | Simple to implement and understand | Inference is O(nd) per query without indexing; sensitive to irrelevant features and scale |

> **Interview question:** What is the curse of dimensionality and how does it affect kNN specifically?
>
> *Answer:* As the number of dimensions `d` increases, the volume of the space grows exponentially, meaning data points become increasingly sparse — even a large dataset covers only a tiny fraction of the input space. For kNN, this manifests as the distance between the nearest neighbour and the farthest neighbour becoming approximately equal as `d` grows; effectively, the concept of "nearest" breaks down because all points are roughly equidistant. This means kNN needs exponentially more data to maintain the same density as dimensionality increases. Practical mitigations include applying PCA or another dimensionality reduction before kNN, using domain-specific distance metrics, or learning embeddings (e.g., with a neural network) where relevant structure is preserved in a lower-dimensional space.*

---

## Unsupervised Learning
{: #unsupervised}

### Clustering
{: #clustering}

**k-Means** partitions `n` points into `k` clusters by alternating between two steps until convergence:

```
Assignment:   cᵢ = argminₖ ‖xᵢ − μₖ‖²         # assign each point to nearest centroid
Update:       μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ           # update centroid to cluster mean
```

k-Means minimises within-cluster sum of squares (WCSS). It converges to a local minimum — **k-Means++** initialisation (choose initial centroids with probability proportional to squared distance from existing centroids) reduces sensitivity to initialisation and typically converges faster to better solutions.

**Limitations**: assumes spherical clusters of similar size; sensitive to outliers; requires specifying `k`. The **elbow method** (plot WCSS vs k, find the "elbow") and **silhouette score** (measures how similar a point is to its own cluster vs others) help select `k`.

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) clusters by density rather than distance to a centroid:

- **Core point**: has ≥ `min_samples` neighbours within radius `ε`
- **Border point**: within `ε` of a core point but has fewer than `min_samples` neighbours
- **Noise**: neither core nor border

<div class="post-flow post-flow--compare" role="group" aria-label="k-Means vs DBSCAN">
  <div class="post-flow__col">
    <p class="post-flow__col-label">k-Means</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Must specify k</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Assumes spherical clusters</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">All points assigned to a cluster</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">DBSCAN</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Discovers k automatically</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Handles arbitrary shapes</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Explicitly identifies outliers as noise</span></li>
    </ol>
  </div>
</div>

**Pros and Cons — Clustering Algorithms**

| | k-Means | DBSCAN |
|---|---|---|
| **Pros** | Fast (`O(nkd)` per iteration), scales to large datasets, simple to understand and implement | No need to specify `k`; finds arbitrarily shaped clusters; explicitly labels outliers as noise |
| **Cons** | Requires specifying `k`; assumes spherical equal-size clusters; sensitive to outliers; converges to local optima | Sensitive to `ε` and `min_samples` hyperparameters; fails when clusters have varying density; high-dimensional data degrades distance-based density estimates |

> **Interview question:** What is the silhouette score and how do you use it to select k in k-Means?
>
> *Answer:* The silhouette score for a point `i` is `(b - a) / max(a, b)`, where `a` is the mean intra-cluster distance (average distance from `i` to all other points in its cluster) and `b` is the mean nearest-cluster distance (average distance from `i` to all points in the nearest different cluster). A score near +1 means the point is well inside its cluster and far from others; near 0 means it is on the boundary; negative means it may have been assigned to the wrong cluster. The average silhouette score across all points is computed for each candidate value of `k`, and the `k` that maximises the average score is selected. This approach is more principled than the elbow method because it directly measures cluster quality rather than the raw within-cluster sum of squares.*

> **Interview question:** When would you choose DBSCAN over k-Means?
>
> *Answer:* DBSCAN is preferable when the clusters have non-convex or irregular shapes that k-Means cannot capture (k-Means implicitly assumes spherical Voronoi regions). It is also the right choice when you expect noise or outliers in the data and want them explicitly labelled rather than forced into a cluster. Additionally, if the number of clusters is unknown and difficult to estimate (the elbow and silhouette methods can be ambiguous), DBSCAN discovers the number of clusters automatically from the density structure. The tradeoff is that DBSCAN is more sensitive to its two hyperparameters `ε` and `min_samples`, which often require domain knowledge or a k-distance plot to set well.*

### Dimensionality Reduction
{: #dimensionality-reduction}

**PCA (Principal Component Analysis)** projects data onto the directions of maximum variance. The principal components are the eigenvectors of the covariance matrix `Σ = XᵀX / n`, ordered by eigenvalue magnitude.

```
1. Centre data: X ← X − mean(X)
2. Compute covariance: Σ = XᵀX / n
3. Eigendecompose: Σ = VΛVᵀ
4. Project: Z = XV_k   (keep top k eigenvectors)
```

PCA is exact, fast (`O(nd²)` for `d < n`), and interpretable — each component is a linear combination of original features. It assumes the important variation is in directions of high variance, which is true for many but not all datasets.

**[t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html)** (t-Distributed Stochastic Neighbour Embedding) is a non-linear dimensionality reduction for **visualisation** (typically to 2D or 3D). It models pairwise similarities in high-dimensional space as probabilities and minimises KL divergence between those and the corresponding low-dimensional similarities:

```
High-dim similarity:  pᵢⱼ = (exp(−‖xᵢ−xⱼ‖²/2σ²)) / Σ exp(...)
Low-dim similarity:   qᵢⱼ = (1 + ‖yᵢ−yⱼ‖²)⁻¹ / Σ (1 + ...)   # t-distribution (heavier tail)
```

t-SNE preserves local structure (nearby points in high-dim stay nearby in 2D) but distorts global structure. It is not a general-purpose dimensionality reduction — it should not be used as a preprocessing step for other algorithms. **[UMAP](https://arxiv.org/abs/1802.03426)** is a faster alternative with better global structure preservation and is now preferred for most visualisation tasks.

**Pros and Cons — Dimensionality Reduction**

| | PCA | t-SNE |
|---|---|---|
| **Pros** | Exact, deterministic, fast; retains global structure; components are interpretable linear combinations; output can be used as features for downstream models | Excellent local structure preservation; reveals cluster structure not visible in linear projections |
| **Cons** | Only captures linear structure; sensitive to scale (must centre and normalise); number of components to retain requires choosing a threshold | Non-deterministic (random initialisation); does not preserve global distances; cannot embed new points without re-running; computationally expensive `O(n² log n)` |

> **Interview question:** How do you decide how many principal components to retain after PCA?
>
> *Answer:* There are three common approaches. First, the **explained variance threshold**: retain enough components to account for a target percentage of total variance (e.g., 95%), computed by summing sorted eigenvalues. Second, the **scree plot**: plot eigenvalues in descending order and look for an "elbow" — the point where the curve flattens, indicating that subsequent components explain diminishing variance. Third, **cross-validation**: treat the number of components as a hyperparameter and select the value that maximises downstream task performance on a validation set. The first two approaches are unsupervised and fast; the third is more expensive but directly optimises what matters for the final task. In practice, the explained variance threshold at 95% is the most common default.*

> **Interview question:** Why should you not use t-SNE embeddings as features for a downstream classifier?
>
> *Answer:* t-SNE is specifically designed to preserve local pairwise structure for visualisation, not to produce a faithful linear representation of the data for downstream tasks. Its objective — minimising KL divergence between high-dimensional and low-dimensional pairwise similarities — causes it to deliberately distort global distances: distances between clusters are not meaningful. Additionally, t-SNE is non-deterministic and the embedding changes with hyperparameters (particularly `perplexity`), and it cannot embed new out-of-sample points without re-running the entire algorithm on the augmented dataset. PCA or autoencoders are appropriate choices when dimensionality reduction is a preprocessing step for another model.*

---

## Bias-Variance Tradeoff
{: #bias-variance}

The expected test error of a model decomposes as:

```
E[(y − ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ²_noise
```

- **Bias** — error from wrong model assumptions. A linear model fit to quadratic data has high bias regardless of how much data you use.
- **Variance** — error from sensitivity to training data. A depth-20 decision tree trained on 100 samples has high variance — a different sample produces a completely different tree.
- **Irreducible noise** — measurement noise in the labels; cannot be reduced by any model.

<div class="post-flow post-flow--compare" role="group" aria-label="High bias vs high variance">
  <div class="post-flow__col">
    <p class="post-flow__col-label">High Bias (Underfitting)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Training error is high</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Test error ≈ training error</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Fix: more complex model, more features</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">High Variance (Overfitting)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Training error is low</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Test error >> training error</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Fix: regularise, more data, simpler model</span></li>
    </ol>
  </div>
</div>

The tradeoff: increasing model complexity reduces bias but increases variance. The optimal model sits at the minimum of total test error — complex enough to capture the true pattern, simple enough not to memorise noise. Regularisation and ensembling are the primary tools for managing this tradeoff.

---

## Regularisation
{: #regularisation}

Regularisation adds a penalty on model complexity to the loss function, discouraging the model from fitting noise:

```
L_reg(w) = L(w) + λ · Ω(w)
```

**L2 regularisation (Ridge)**: `Ω(w) = ‖w‖²`. Penalises large weights — equivalent to placing a Gaussian prior on `w`. Shrinks all weights toward zero proportionally. Has a closed-form solution: `w = (XᵀX + λI)⁻¹Xᵀy`. Stable even when `XᵀX` is singular.

**L1 regularisation (Lasso)**: `Ω(w) = ‖w‖₁`. Penalises the absolute value of weights — induces **sparsity**. Many weights become exactly zero, performing automatic feature selection. No closed form; requires coordinate descent or subgradient methods.

**Elastic net**: `Ω(w) = α‖w‖₁ + (1−α)‖w‖²`. Combines L1 sparsity with L2 stability. Useful when features are correlated (Lasso tends to pick one arbitrarily; Elastic Net groups correlated features).

| Method | Penalty | Effect | Use when |
|---|---|---|---|
| Ridge | `λ‖w‖²` | Shrinks weights uniformly | Features all relevant, correlated |
| Lasso | `λ‖w‖₁` | Sparsity — zeros out weights | Many irrelevant features |
| Elastic Net | `α‖w‖₁ + (1−α)‖w‖²` | Sparse + stable | Correlated features, many irrelevant |

`λ` (regularisation strength) is a hyperparameter: too large → high bias; too small → high variance. Selected by cross-validation.

---

## Model Selection & Evaluation
{: #model-selection}

### Cross-Validation
{: #cross-validation}

A model's training error is an optimistic estimate of test error — the model has seen the training data. **Cross-validation** estimates test error without a separate holdout set by rotating which data is used for training and evaluation.

**k-Fold CV**: partition data into k folds. Train on k−1 folds, evaluate on the held-out fold. Rotate k times. Average the k evaluation scores.

```
CV score = (1/k) Σᵢ metric(model_trained_on_all_but_fold_i, fold_i)
```

`k=5` or `k=10` is standard. Higher `k` → lower bias (more training data per fold), higher variance (more variance across folds), and higher compute cost. **Leave-one-out CV (LOOCV)** is the extreme case (`k=n`) — essentially unbiased but expensive and high-variance.

**Stratified k-Fold**: for classification, ensure each fold has the same class distribution as the full dataset. Important for imbalanced classes — without stratification, a fold might contain no examples of a minority class.

**Time series CV**: for temporal data, folds must respect time ordering — never train on future data to predict the past. Use an expanding window (train on all data up to time t, test on t+1 to t+h) or a sliding window.

### Metrics
{: #metrics}

Choosing the right metric is as important as choosing the right model — different metrics encode different tradeoffs.

**Classification:**

| Metric | Formula | Use when |
|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Balanced classes |
| Precision | TP/(TP+FP) | False positives costly (spam filter) |
| Recall | TP/(TP+FN) | False negatives costly (cancer detection) |
| F1 | 2·P·R/(P+R) | Imbalanced classes, balance P and R |
| ROC-AUC | Area under ROC curve | Ranking quality, threshold-independent |
| PR-AUC | Area under PR curve | Imbalanced classes, focus on positives |

**Regression:**

| Metric | Formula | Use when |
|---|---|---|
| MSE | `(1/n)Σ(y−ŷ)²` | Penalise large errors heavily |
| MAE | `(1/n)Σ|y−ŷ|` | Robust to outliers |
| RMSE | `√MSE` | Same units as target |
| R² | `1 − SS_res/SS_tot` | Fraction of variance explained |

**Imbalanced classes**: accuracy is misleading — a classifier that always predicts the majority class achieves 99% accuracy on a 99:1 dataset. Use F1, PR-AUC, or balanced accuracy instead.

### Hyperparameter Tuning
{: #hyperparameter-tuning}

Hyperparameters (regularisation strength, tree depth, number of neighbours, learning rate) are not learned during training — they must be set before training and tuned over a validation set.

**Grid search**: exhaustively evaluate all combinations from a predefined grid. Guaranteed to find the best combination in the grid but exponential in the number of hyperparameters.

**Random search**: sample hyperparameter combinations randomly. Empirically as effective as grid search in high-dimensional spaces — most hyperparameters have low sensitivity, so random search covers the important dimensions better with the same budget.

**Bayesian optimisation**: model the objective function (validation score as a function of hyperparameters) with a surrogate (Gaussian process or tree-based model). Use an acquisition function (expected improvement, upper confidence bound) to choose the next configuration to evaluate. Significantly more sample-efficient than grid or random search for expensive-to-evaluate models.

**Nested cross-validation**: when the dataset is small, use an outer cross-validation loop to estimate test error and an inner loop to select hyperparameters. This prevents overfitting to the validation set — a single train/val/test split can give misleading estimates when the val set is used for both model and hyperparameter selection.

---

## Probabilistic Models
{: #probabilistic}

### Naive Bayes
{: #naive-bayes}

Naive Bayes applies Bayes' theorem with a strong (naive) conditional independence assumption: given the class `y`, all features are independent.

```
P(y|x) ∝ P(y) · Π P(xᵢ|y)
```

The prior `P(y)` is estimated from class frequencies. The likelihood `P(xᵢ|y)` depends on the assumed feature distribution:
- **Gaussian NB**: continuous features, `P(xᵢ|y) = 𝒩(μᵢₖ, σ²ᵢₖ)`
- **Multinomial NB**: count features (word counts), `P(xᵢ|y) ∝ θᵢₖ^xᵢ`
- **Bernoulli NB**: binary features (word presence/absence)

Despite the independence assumption being almost always wrong, Naive Bayes works well in practice for text classification — especially with small datasets where more complex models overfit. It is also extremely fast to train and predict (`O(nd)` per class).

**Laplace smoothing**: add a pseudocount `α=1` to all feature counts before normalising, preventing zero probabilities for unseen feature-class combinations:

```
P(xᵢ|y) = (count(xᵢ, y) + α) / (count(y) + α · |V|)
```

**Pros and Cons — Naive Bayes**

| | Pros | Cons |
|---|---|---|
| **Speed** | Extremely fast to train and predict — `O(nd)` regardless of class count | Independence assumption is almost always violated; correlated features lead to over-confident posteriors |
| **Data efficiency** | Works well with very small training sets; naturally handles streaming/online updates | Probability estimates are poorly calibrated, especially near 0/1 (though predictions are often correct despite this) |
| **Applicability** | Effective baseline for text classification; naturally multi-class | Not suitable when feature interactions are important for the classification decision |

> **Interview question:** Why does Naive Bayes work well for text classification despite its independence assumption being clearly violated (words are not independent)?
>
> *Answer:* For classification, we only need the predicted class to be correct, not the probability estimates to be perfectly calibrated. Even though Naive Bayes produces over-confident posteriors when features are correlated (because it counts correlated evidence multiple times), it still usually ranks the correct class highest when the correlations are symmetric across classes. In text classification, while words are correlated (e.g., "machine" and "learning" often co-occur), these correlations tend to be similar across documents of the same class, so the relative ranking of class posteriors is preserved. Additionally, the simplicity of the model makes it resistant to overfitting with small training sets — a genuine advantage in many real-world settings.*

### Gaussian Mixture Models
{: #gmm}

A **GMM** models the data as a mixture of K Gaussian distributions:

```
P(x) = Σₖ πₖ · 𝒩(x; μₖ, Σₖ)
```

where `πₖ` are mixing weights (`Σπₖ = 1`), `μₖ` are component means, and `Σₖ` are covariance matrices.

GMMs are fit using **Expectation-Maximisation (EM)**:

<div class="post-flow" role="group" aria-label="EM algorithm for GMM">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">E-step: compute soft assignments rᵢₖ = P(component k | xᵢ) for each point</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">M-step: update μₖ, Σₖ, πₖ using weighted statistics from soft assignments</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Repeat until log-likelihood converges</span></li>
  </ol>
</div>

EM monotonically increases the log-likelihood and converges to a local maximum. Multiple restarts with different initialisations help find better solutions.

**GMM vs k-Means**: k-Means is hard assignment (each point belongs to exactly one cluster); GMM is soft assignment (each point has a probability distribution over clusters). GMM is strictly more expressive — k-Means is a special case of GMM with fixed spherical covariances and hard assignments in the limit.

GMMs also provide a generative model of the data: you can sample new points from `P(x)` by first sampling a component `k ~ Categorical(π)` then sampling `x ~ 𝒩(μₖ, Σₖ)`. This makes them useful for density estimation and anomaly detection (low `P(x)` → anomaly).

**Pros and Cons — GMMs**

| | Pros | Cons |
|---|---|---|
| **Expressivity** | Soft cluster assignments capture uncertainty; models elliptical clusters of varying size/orientation | Requires specifying `K`; EM converges to local optima — multiple restarts needed |
| **Generative** | Provides full density model `P(x)` — supports sampling, anomaly detection, and density estimation | Assumes Gaussian components — fails for multi-modal or heavy-tailed distributions within a cluster |
| **Interpretability** | Covariance matrices capture feature correlations within each cluster | Full covariance has `O(d²)` parameters per component — can overfit with high-dimensional data without covariance constraints |

> **Interview question:** How does the EM algorithm for GMMs relate to k-Means, and what are the key differences?
>
> *Answer:* k-Means can be viewed as a special case of EM for a GMM where all component covariances are fixed to `σ²I` (isotropic) and the E-step uses hard assignment (argmax over responsibilities) instead of soft assignment (full posterior probabilities). In the general EM-GMM, the E-step computes a soft responsibility `rᵢₖ = P(k | xᵢ)` for each point and component, and the M-step computes weighted statistics to update the means, covariances, and mixing proportions. The result is that GMM can model elliptical clusters of varying shape and size, and can express uncertainty about which cluster a point belongs to. The cost is more hyperparameters (covariance structure), slower M-steps, and greater sensitivity to local optima — especially in high dimensions.*

---

## Optimisation
{: #optimisation}

All ML training reduces to minimising a loss function over parameters. The choice of optimiser determines convergence speed, memory cost, and solution quality.

**Gradient descent** iteratively updates parameters in the direction of steepest descent:

```
θ ← θ − η · ∇L(θ)
```

**Variants** differ in how many samples are used to estimate the gradient:

| Variant | Gradient estimate | Update frequency | Use case |
|---|---|---|---|
| Batch GD | Full dataset | Once per epoch | Convex problems, small data |
| Stochastic GD (SGD) | One sample | Once per sample | Online learning, noisy but fast |
| Mini-batch GD | B samples | Once per batch | Standard practice; B=32–256 |

**Momentum** accumulates a velocity term to smooth noisy gradient updates and escape flat regions:

```
v ← β·v + (1−β)·∇L
θ ← θ − η·v
```

**[AdaGrad](https://jmlr.org/papers/v12/duchi11a.html)** adapts the learning rate per parameter — parameters with large gradients get smaller updates, sparse parameters get larger updates. Useful for sparse features (NLP), but the learning rate monotonically decreases and eventually stalls.

**[Adam](https://arxiv.org/abs/1412.6980)** combines momentum (first moment) with per-parameter adaptive rates (second moment):

```
m ← β₁·m + (1−β₁)·g
v ← β₂·v + (1−β₂)·g²
θ ← θ − η · m̂/(√v̂ + ε)
```

Adam is the default for neural networks. For linear models and SVMs, L-BFGS (a quasi-Newton method that approximates the Hessian) often converges in fewer iterations for small-to-medium datasets.

**Convexity**: for convex loss functions (linear/logistic regression, SVMs), gradient descent finds the global minimum regardless of initialisation. For non-convex losses (neural networks, GMMs), gradient descent finds a local minimum — which, in practice, is often good enough.

---

## Bayesian Linear Regression
{: #bayesian-linear-regression}

Standard (frequentist) linear regression produces a single point estimate of the weights `w`. **Bayesian linear regression** instead maintains a probability distribution over `w`, enabling uncertainty quantification in predictions.

**Frequentist vs Bayesian framing:**

| | Frequentist (OLS) | Bayesian |
|---|---|---|
| Parameters | Fixed unknowns estimated from data | Random variables with a prior distribution |
| Output | Point estimate `ŵ` | Posterior distribution `P(w | X, y)` |
| Uncertainty | Confidence intervals (frequentist coverage) | Posterior predictive distribution over `ŷ` |
| Regularisation | Ad hoc (Ridge, Lasso as penalties) | Natural — prior encodes regularisation |

**Setup:** Place a Gaussian prior on weights: `P(w) = 𝒩(0, τ²I)`. Given data `y = Xw + ε` with `ε ~ 𝒩(0, σ²)`, the posterior is also Gaussian (conjugate):

```
P(w | X, y) = 𝒩(w | μ_N, Σ_N)

Σ_N = (σ⁻² XᵀX + τ⁻²I)⁻¹
μ_N = σ⁻² Σ_N Xᵀy
```

Note that `μ_N` is exactly the Ridge regression solution with `λ = σ²/τ²`. This reveals that **Ridge regression is the MAP (maximum a posteriori) estimate of Bayesian linear regression with a Gaussian prior** — L2 regularisation has a clean Bayesian interpretation as placing a zero-mean Gaussian prior on the weights. Similarly, Lasso corresponds to a Laplace prior on `w`.

**Predictive distribution:** For a new point `x*`, the predictive distribution over `y*` integrates over all possible weights:

```
P(y* | x*, X, y) = ∫ P(y* | x*, w) P(w | X, y) dw = 𝒩(μ_N ᵀ x*, x*ᵀ Σ_N x* + σ²)
```

This gives not just a point prediction but a full predictive uncertainty — wider in regions far from training data. This is the key advantage over frequentist regression for decision-making under uncertainty.

**Pros and Cons — Bayesian Linear Regression**

| | Pros | Cons |
|---|---|---|
| **Uncertainty** | Full predictive distribution; naturally quantifies uncertainty — wider CI where data is sparse | Computationally expensive for large `d`: posterior covariance `Σ_N` is `d×d` |
| **Regularisation** | Prior provides principled regularisation; hyperparameter selection via marginal likelihood (Type II MLE) | Gaussian likelihood and prior assumptions may be inappropriate; inference scales as `O(d³)` |
| **Interpretability** | Bayesian model comparison via marginal likelihood; no separate validation set needed for regularisation selection | Requires specifying prior — choice of `τ` affects results |

> **Interview question:** What is the connection between Ridge regression and Bayesian linear regression, and why does this matter?
>
> *Answer:* Ridge regression adds an L2 penalty `λ‖w‖²` to the MSE loss, producing the solution `w = (XᵀX + λI)⁻¹Xᵀy`. In Bayesian linear regression with a Gaussian prior `P(w) = 𝒩(0, τ²I)` and Gaussian likelihood, the MAP estimate (mode of the posterior) is exactly this Ridge solution with `λ = σ²/τ²`. This connection matters for several reasons: it gives a principled interpretation of the regularisation strength in terms of the ratio of noise variance to prior variance; it suggests that the optimal `λ` can be estimated from the data by maximising the marginal likelihood `P(y | X, λ)` — called Type II MLE or empirical Bayes — avoiding the need for a separate cross-validation loop; and it shows that frequentist regularised regression and Bayesian MAP estimation are equivalent point-estimate procedures, with the Bayesian framework additionally providing the full posterior and predictive uncertainty.*

> **Interview question:** How does Bayesian linear regression handle uncertainty differently from a frequentist confidence interval?
>
> *Answer:* A frequentist 95% confidence interval means that if you repeated the experiment many times and computed the interval each time, 95% of the intervals would contain the true fixed parameter. It makes a statement about the procedure, not about the probability of the parameter lying in a specific interval. A Bayesian credible interval directly states that given the observed data, there is a 95% posterior probability that the parameter lies in the interval. More importantly, Bayesian linear regression provides a full predictive distribution `P(y* | x*, X, y)` rather than just an interval for the parameters — this predictive variance grows in regions far from training data (epistemic uncertainty) and includes irreducible measurement noise (aleatoric uncertainty), making it directly useful for downstream decision-making.*

---

## Ridge & Lasso Regression
{: #ridge-lasso}

While regularisation is covered briefly in the Regularisation section above, Ridge and Lasso deserve treatment as first-class supervised learning algorithms in their own right, with distinct mathematical properties, practical use cases, and selection criteria.

### Ridge Regression (L2)

Ridge regression minimises:

```
L_ridge(w) = ‖y − Xw‖² + λ‖w‖²
```

Closed-form solution: `w_ridge = (XᵀX + λI)⁻¹Xᵀy`

The addition of `λI` to `XᵀX` **regularises the matrix inversion** — even when `XᵀX` is singular (more features than samples, or perfectly collinear features), Ridge regression produces a unique, stable solution. As `λ → 0`, Ridge approaches OLS; as `λ → ∞`, all weights shrink toward zero.

**Geometric interpretation:** Ridge constrains the weight vector to lie within an L2 ball of radius `1/λ`. The optimal solution is where the MSE ellipses first touch the L2 ball — which happens at a smooth point, so Ridge **shrinks all weights proportionally but never zeros them out exactly**.

**Effect on correlated features:** when features are highly correlated, OLS has high variance (the normal equations are ill-conditioned). Ridge distributes weight among correlated features rather than assigning arbitrary large weights to one — this is the primary reason to use Ridge over OLS in practice.

### Lasso Regression (L1)

Lasso minimises:

```
L_lasso(w) = ‖y − Xw‖² + λ‖w‖₁
```

No closed form — the L1 penalty is non-differentiable at zero. Solved by **coordinate descent**: cycle through each weight `wⱼ`, hold all others fixed, and apply the **soft-thresholding** update:

```
wⱼ ← sign(ρⱼ) · max(|ρⱼ| − λ, 0)

where ρⱼ = Σᵢ xᵢⱼ (yᵢ − ŷᵢ^{(-j)})  # partial residual
```

**Sparsity:** the L1 ball has corners at the coordinate axes. The MSE ellipses tend to first touch the L1 ball at a corner, where one or more weights are exactly zero. This is why Lasso performs **automatic feature selection** — many weights become exactly zero, not just small.

**Limitations of Lasso with correlated features:** when features are highly correlated, Lasso tends to select one arbitrarily and zero out the others — even if all are genuinely informative. In this regime, Elastic Net (combining L1 and L2 penalties) is more appropriate.

### Choosing Ridge vs Lasso vs Elastic Net

| Scenario | Recommended |
|---|---|
| All features likely relevant, some correlated | Ridge |
| Many irrelevant features, want automatic selection | Lasso |
| Many irrelevant features, some correlated | Elastic Net |
| `d >> n` (many features, few samples) | Ridge or Lasso (Lasso if sparsity expected) |
| Want interpretable sparse model | Lasso |

**Selecting `λ`:** always use cross-validation — plot the cross-validation error vs `log(λ)` and select the minimum or, for more regularisation, the largest `λ` within one standard error of the minimum (the "one-SE rule").

**Pros and Cons — Ridge & Lasso**

| | Ridge | Lasso |
|---|---|---|
| **Pros** | Closed-form solution; stabilises collinear features; always unique solution | Automatic feature selection; sparse models easier to interpret and deploy; useful when `d >> n` |
| **Cons** | Never produces exactly zero weights — cannot do feature selection | No closed form; slower to solve; selects arbitrarily among correlated features; can miss some signal when many features are jointly relevant |

> **Interview question:** Why does L1 regularisation produce sparse solutions while L2 does not? Explain geometrically.
>
> *Answer:* The constrained form of regularised regression minimises the loss subject to the constraint `‖w‖_p ≤ t`. For L2, the constraint set is a smooth sphere — the contours of the MSE loss (ellipsoids) touch the sphere at a smooth point where none of the weights are exactly zero, they are merely shrunk. For L1, the constraint set is a diamond (in 2D) or hypercube (in higher dimensions) with sharp corners on the coordinate axes. The loss ellipsoids are far more likely to first intersect the L1 ball at one of these corners, where at least one weight is exactly zero. As dimensionality grows, the proportion of volume in the L1 ball concentrated near vertices and edges increases, making exact zeros even more common. This geometric argument shows sparsity is a structural consequence of the L1 penalty shape, not a coincidence.*

> **Interview question:** A colleague reports that Ridge regression performs better than Lasso on their dataset, even though they expected many features to be irrelevant. What might explain this?
>
> *Answer:* Several explanations are possible. First, the "irrelevant" features may actually have small but non-zero effects — Ridge's shrinkage accumulates many small contributions that Lasso would zero out, potentially improving prediction. Second, the features may be highly correlated with each other and with the target; Lasso arbitrarily selects one from each correlated group while Ridge distributes weight across all of them, which reduces variance and often improves prediction accuracy. Third, the sample size may be large enough that overfitting is not severe, so the aggressive sparsity of Lasso provides no advantage. Finally, Lasso's cross-validated `λ` may simply not have been tuned carefully — Elastic Net (which combines both penalties) often outperforms pure Lasso in these scenarios and is worth trying.*

---

## Gradient Boosted Trees Deep Dive: XGBoost & LightGBM
{: #gradient-boosted-trees}

Gradient boosting builds a prediction as an additive model of weak learners (typically shallow decision trees):

```
F_M(x) = F_0 + Σₘ₌₁ᴹ ηₘ · hₘ(x)
```

where `η` is the learning rate (shrinkage) and each `hₘ` fits the **negative gradient** of the loss evaluated at the current ensemble's predictions.

### Why Negative Gradient?

Fitting to the negative gradient is a gradient descent step in function space. For MSE loss, the negative gradient is simply the residual `yᵢ − Fₘ(xᵢ)` — so gradient boosting reduces to sequentially fitting residuals. For other losses (log-loss, Huber, quantile), the negative gradient gives the appropriate "pseudo-residual" that moves the predictions in the right direction:

```
rᵢₘ = − [∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)]_{F=Fₘ₋₁}
```

### XGBoost: Second-Order Approximation

Standard gradient boosting uses only the first-order gradient. **XGBoost** uses a second-order Taylor expansion of the loss to derive a closed-form optimal leaf weight for each tree:

```
L_approx ≈ Σᵢ [gᵢ f(xᵢ) + ½ hᵢ f(xᵢ)²] + Ω(f)

gᵢ = ∂L/∂Fₘ₋₁(xᵢ)   (first-order gradient)
hᵢ = ∂²L/∂Fₘ₋₁(xᵢ)²  (second-order Hessian)
Ω(f) = γ·T + ½λ‖w‖²   (regularisation: T = num leaves, w = leaf weights)
```

The optimal leaf weight for leaf `j` is:

```
w*_j = − (Σᵢ∈Iⱼ gᵢ) / (Σᵢ∈Iⱼ hᵢ + λ)
```

The tree structure is chosen to maximise the gain, which has a closed form from the second-order expansion. This makes XGBoost more stable and often requires fewer trees than standard gradient boosting.

**Key XGBoost features:**
- **Regularised objective:** L1 and L2 penalties on leaf weights, plus a penalty on tree complexity (number of leaves)
- **Column subsampling:** random feature subsets at each level (like Random Forest) — reduces overfitting
- **Row subsampling:** subsample training points per tree — reduces variance
- **Sparsity-aware split finding:** treats missing values natively by learning the best default direction at each split
- **Level-wise tree growth:** grows trees level by level (BFS), splitting all nodes at depth `d` before moving to depth `d+1`

### LightGBM: Histogram-Based Splitting and Leaf-Wise Growth

**LightGBM** achieves dramatically faster training than XGBoost through two key innovations:

<div class="post-flow post-flow--compare" role="group" aria-label="XGBoost vs LightGBM">
  <div class="post-flow__col">
    <p class="post-flow__col-label">XGBoost (Level-wise)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Split all nodes at depth d before depth d+1</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Balanced tree growth</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">May split nodes with low gain to maintain level structure</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">LightGBM (Leaf-wise)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Always split the leaf with the highest gain</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Asymmetric tree growth — deeper where it matters</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Fewer leaves for same loss reduction; faster convergence</span></li>
    </ol>
  </div>
</div>

**Histogram-based splitting (GOSS + EFB):**
- **GOSS (Gradient-based One-Side Sampling):** instead of using all training data to find the best split, keep all instances with large gradients (they contribute most to training loss) and randomly sample a fraction of those with small gradients. This approximates the gradient statistics used for split finding with far fewer data points.
- **EFB (Exclusive Feature Bundling):** many features in sparse datasets are mutually exclusive (rarely both non-zero). LightGBM bundles such features into a single feature, reducing effective feature dimensionality without loss of information.
- **Histogram splitting:** continuous features are discretised into `B` bins (typically 255). Split finding scans only `B` thresholds per feature (vs `n` unique values in exact algorithms), reducing complexity from `O(n)` to `O(B)` per feature per node.

**Practical differences (XGBoost vs LightGBM):**

| Aspect | XGBoost | LightGBM |
|---|---|---|
| Training speed | Moderate | Fast (often 5-10x faster on large data) |
| Memory | Moderate | Lower (histogram-based, no pre-sorted data) |
| Accuracy (large data) | Good | Often slightly better due to leaf-wise growth |
| Accuracy (small data) | Good | Risk of overfitting — use `num_leaves` carefully |
| GPU support | Yes | Yes |
| Categorical features | Manual encoding required | Native support (`categorical_feature` parameter) |

**Hyperparameters to tune (both):**

| Parameter | Effect |
|---|---|
| `n_estimators` / `num_boost_round` | More trees = lower bias, risk of overfitting; use early stopping |
| `learning_rate` / `eta` | Lower = better generalisation, more trees needed |
| `max_depth` / `num_leaves` | Controls tree complexity; leaf-wise models use `num_leaves` not `max_depth` |
| `subsample` / `bagging_fraction` | Row subsampling ratio; reduces variance |
| `colsample_bytree` / `feature_fraction` | Column subsampling; reduces correlation between trees |
| `min_child_weight` / `min_data_in_leaf` | Minimum sum of Hessian in leaf; prevents over-specific splits |
| `lambda` / `reg_lambda` | L2 regularisation on leaf weights |

**Pros and Cons — Gradient Boosted Trees**

| | Pros | Cons |
|---|---|---|
| **Accuracy** | State-of-the-art on tabular data; flexible loss functions; handles missing values natively | Sequential training makes parallelisation difficult (trees must be built in order) |
| **Features** | Native handling of mixed feature types; built-in feature importance; robust to outliers with appropriate loss | Many hyperparameters requiring tuning; training time scales with `n_estimators × n × d` |
| **Regularisation** | Built-in L1/L2 regularisation on leaf weights; subsampling reduces variance | Can overfit on small noisy datasets; early stopping and `num_leaves` control are critical |

> **Interview question:** Explain why gradient boosting is called "gradient" boosting. What is the role of the learning rate?
>
> *Answer:* Gradient boosting frames the problem of finding the best additive model as gradient descent in function space. Instead of minimising a loss by updating parameters, we minimise the loss by adding new functions (trees) that move the ensemble's predictions in the direction of the negative gradient of the loss evaluated at the current predictions. Each new tree is a "step" in function space — fitting the tree to the negative gradient ensures the ensemble improves on the loss, regardless of what loss function is used. The learning rate (shrinkage) `η` scales each tree's contribution: a smaller `η` means each step is smaller, requiring more trees but typically achieving better generalisation because the model is less likely to overfit to any individual tree's noise. Empirically, a small learning rate (0.01–0.1) combined with early stopping is the most robust configuration.*

> **Interview question:** How does XGBoost handle missing values during training and inference?
>
> *Answer:* XGBoost learns a default direction for each split node — it tries sending all missing values to the left child and to the right child separately, and keeps whichever direction gives the higher gain score. This default direction is learned during training and stored with the tree. At inference, any missing value at a split node is automatically routed via the learned default direction. This means no imputation is required — XGBoost can handle missing values natively without preprocessing, which is a significant practical advantage. The default direction effectively learns the relationship between missingness and the target, capturing MNAR (missing not at random) patterns that explicit imputation cannot.*

---

## Neural Networks & MLPs
{: #neural-networks}

A **multilayer perceptron (MLP)** is the classical feedforward neural network: a directed acyclic graph of neurons arranged in layers, where each neuron computes a weighted sum of its inputs followed by a non-linear activation function.

### Architecture

```
Input: x ∈ ℝᵈ
Hidden layer l: hˡ = σ(Wˡ hˡ⁻¹ + bˡ),   hˡ ∈ ℝⁿˡ
Output: ŷ = Wᴸ hᴸ⁻¹ + bᴸ   (then softmax for classification, identity for regression)
```

Where `σ` is an element-wise activation function applied after the affine transformation.

### Activation Functions

| Activation | Formula | Properties | Use when |
|---|---|---|---|
| Sigmoid | `1 / (1 + e⁻ˣ)` | Output in (0,1); vanishing gradients for large `|x|` | Output layer for binary classification |
| Tanh | `(eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)` | Output in (−1,1); zero-centred; still has vanishing gradients | Recurrent networks (historically) |
| ReLU | `max(0, x)` | No vanishing gradients for `x > 0`; sparse activations; dying ReLU problem | Default for hidden layers |
| Leaky ReLU | `max(αx, x)`, α ≈ 0.01 | Fixes dying ReLU by allowing small negative gradients | When dying ReLU is a concern |
| GELU | `x · Φ(x)` | Smooth approximation to ReLU; used in transformers | Modern architectures |
| Softmax | `eˣⁱ / Σeˣʲ` | Normalised to sum to 1; differentiable | Output layer for multi-class classification |

**Dying ReLU problem:** if a neuron's input is always negative (e.g., after a large gradient update pushes weights into a regime where the pre-activation is always < 0), the gradient through ReLU is identically zero. The neuron never recovers — it is "dead". Leaky ReLU, batch normalisation, and careful weight initialisation (He init for ReLU layers) mitigate this.

### Backpropagation

Backprop is an efficient application of the chain rule to compute `∂L/∂wᵢⱼ` for all weights simultaneously in `O(W)` operations (where `W` is the total number of parameters — the same order as a single forward pass):

<div class="post-flow" role="group" aria-label="Backpropagation algorithm">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Forward pass: compute activations hˡ for each layer l = 1, ..., L</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Compute loss L(ŷ, y)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Backward pass: propagate δˡ = (Wˡ⁺¹)ᵀ δˡ⁺¹ ⊙ σ'(zˡ) from output to input</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Compute weight gradients: ∂L/∂Wˡ = δˡ (hˡ⁻¹)ᵀ</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Update weights with gradient descent (or Adam)</span></li>
  </ol>
</div>

**Vanishing gradients:** when using sigmoid or tanh activations with many layers, gradients become exponentially small as they propagate backward — early layers learn very slowly or not at all. ReLU activations, residual connections, and batch normalisation are the primary solutions.

### Universal Approximation Theorem

A single hidden layer MLP with a sufficient number of neurons can approximate any continuous function on a compact domain to arbitrary precision (Cybenko 1989; Hornik 1991). This theorem is often misunderstood:

- It guarantees **existence** of such a network, not that gradient descent will find it
- The required width may be exponential in the input dimension — deep networks can approximate the same functions with far fewer parameters (depth efficiency)
- It says nothing about generalisation — a network that approximates the training data may not generalise to test data
- It is a qualitative result: in practice, architecture, regularisation, and optimisation choices are what determine whether a network learns a useful function

### Weight Initialisation

Random initialisation is critical. All-zeros initialisation causes all neurons in a layer to learn the same function (symmetry breaking fails). Standard initialisations:

- **Xavier / Glorot**: `W ~ 𝒩(0, 2/(nᵢₙ + nₒᵤₜ))` — designed for tanh/sigmoid to maintain variance through layers
- **He (Kaiming)**: `W ~ 𝒩(0, 2/nᵢₙ)` — designed for ReLU; accounts for the fact that ReLU zeros out half its inputs

### Regularisation for Neural Networks

- **Dropout**: randomly zero out each neuron's activation with probability `p` during training; at inference, scale weights by `(1 − p)`. Prevents co-adaptation of neurons; acts as an ensemble of `2^n` sub-networks.
- **Batch normalisation**: normalise pre-activations within a mini-batch, then apply learned scale and shift. Reduces internal covariate shift, enables higher learning rates, and provides mild regularisation.
- **Weight decay**: L2 penalty on weights (equivalent to Ridge regression for linear models).
- **Early stopping**: monitor validation loss and stop training when it starts increasing.

**Pros and Cons — Neural Networks / MLPs**

| | Pros | Cons |
|---|---|---|
| **Expressivity** | Universal approximators; can learn arbitrary feature representations; excel on high-dimensional data (images, text, audio) | Require large datasets to generalise; prone to overfitting on small data |
| **Features** | End-to-end learning — no manual feature engineering needed; transfer learning enables reuse of pretrained representations | Black-box — poor interpretability; difficult to debug; many hyperparameters |
| **Flexibility** | Flexible architecture for any input/output type; loss function can be customised | Long training times; require GPUs for large models; sensitive to learning rate and initialisation |

> **Interview question:** Explain backpropagation from first principles. Why is it efficient?
>
> *Answer:* Backpropagation computes the gradient of the loss with respect to every weight in the network by applying the chain rule in a specific order. During the forward pass, we compute and cache the activations at each layer. During the backward pass, we compute the error signal `δˡ` at each layer, starting from the output: `δᴸ = ∂L/∂zᴸ`. For each preceding layer, the delta is propagated backward as `δˡ = (Wˡ⁺¹)ᵀ δˡ⁺¹ ⊙ σ'(zˡ)`, where `⊙` is element-wise multiplication with the derivative of the activation. The weight gradients are then `∂L/∂Wˡ = δˡ (hˡ⁻¹)ᵀ`. The key efficiency insight is that each intermediate gradient is computed exactly once and reused — without backprop, computing gradients by finite differences would cost `O(W)` forward passes (one per parameter). Backprop computes all gradients in `O(1)` forward + `O(1)` backward passes, making it the same asymptotic cost as a single forward pass.*

> **Interview question:** What is the vanishing gradient problem and what architectural choices solve it?
>
> *Answer:* The vanishing gradient problem occurs when gradients become exponentially small as they are propagated backward through many layers. With sigmoid or tanh activations, the derivative `σ'(z)` is at most 0.25 for sigmoid and 1.0 for tanh, so each backward step multiplies by a value ≤ 1 (often much less). With 20 layers, the gradient at the first layer may be `0.25²⁰ ≈ 10⁻¹²`, making it effectively untrainable. ReLU activations solve this for positive inputs because their derivative is exactly 1, preserving gradient magnitude. Residual connections (ResNets) provide a direct gradient highway from loss to early layers, bypassing any activation. Batch normalisation normalises activations to prevent the pre-activations from entering the saturation region of sigmoid/tanh. In practice, deep networks trained for modern tasks combine all three: ReLU (or GELU) activations, residual connections, and normalisation layers.*

---

## Linear Discriminant Analysis
{: #lda}

**LDA** is a supervised dimensionality reduction and classification algorithm. Unlike PCA, which finds directions of maximum variance regardless of labels, LDA finds directions that **maximise class separability** — it maximises the ratio of between-class variance to within-class variance.

### Derivation

Given `C` classes with means `μ_c` and overall mean `μ`, define:

```
Between-class scatter:  S_B = Σ_c nᶜ (μᶜ − μ)(μᶜ − μ)ᵀ
Within-class scatter:   S_W = Σ_c Σᵢ∈c (xᵢ − μᶜ)(xᵢ − μᶜ)ᵀ
```

LDA finds projection directions `w` that maximise the **Fisher criterion**:

```
J(w) = (wᵀ S_B w) / (wᵀ S_W w)
```

The optimal directions are the generalised eigenvectors of `S_W⁻¹ S_B`. LDA can produce at most `min(C−1, d)` discriminant directions (since `S_B` has rank at most `C−1`).

### LDA as a Classifier

For Gaussian classes with equal covariance matrices, LDA is the optimal linear classifier (Bayes optimal). The decision rule for a new point `x` is:

```
class = argmaxᶜ [μᶜᵀ Σ⁻¹ x − ½ μᶜᵀ Σ⁻¹ μᶜ + log P(c)]
```

**Quadratic Discriminant Analysis (QDA)** relaxes the equal-covariance assumption, fitting a separate `Σᶜ` per class — this produces quadratic decision boundaries but requires estimating more parameters.

### LDA vs PCA

| | PCA | LDA |
|---|---|---|
| **Objective** | Maximise variance of projections | Maximise class separability |
| **Supervision** | Unsupervised | Supervised (uses class labels) |
| **Max components** | min(n, d) | C − 1 |
| **Use case** | Visualisation, compression, preprocessing | Classification, supervised dimensionality reduction |
| **Assumption** | No distributional assumptions | Gaussian classes with equal covariance |

**Pros and Cons — LDA**

| | Pros | Cons |
|---|---|---|
| **Classification** | Optimal Bayes classifier under Gaussian + equal-covariance assumptions; fast training and inference | Assumptions (Gaussianity, equal covariance) often violated in practice |
| **Dimensionality reduction** | Supervised — uses labels to find discriminative directions; at most C−1 components | Limited to C−1 dimensions — insufficient for problems with few classes and many needed dimensions |
| **Interpretability** | Discriminant directions are linear combinations of features; easy to visualise for C=3 | Sensitive to outliers (which distort scatter matrices); requires `S_W` to be invertible — fails when `d > n` |

> **Interview question:** What is the difference between LDA and logistic regression for binary classification, and when would you prefer each?
>
> *Answer:* Both are linear classifiers producing a decision boundary of the form `wᵀx + b = 0`. The key difference is how parameters are estimated. LDA assumes both classes are Gaussian with equal covariance, estimates the class means and shared covariance from the data, and derives the linear boundary analytically. Logistic regression makes no distributional assumptions — it directly models `P(y=1|x)` by maximising the conditional likelihood. If the Gaussian equal-covariance assumption holds, LDA is more statistically efficient (requires less data for the same accuracy) and more robust in small-sample settings. When the assumption is violated (e.g., skewed or multi-modal class distributions, different covariances), logistic regression is more reliable. In practice, logistic regression is more commonly used because it is robust to assumption violations and naturally extends to L1/L2 regularisation, while LDA is preferred when interpretable low-dimensional projections are needed and the Gaussian assumption is plausible.*

---

## k-Means Deep Dive
{: #kmeans-deep}

The basic k-Means algorithm is covered in the Clustering section above. Here we examine its mechanics in depth, focusing on initialisation, convergence, and the practical problem of choosing `k`.

### k-Means++ Initialisation

Standard k-Means initialises centroids uniformly at random, leading to poor local optima. **k-Means++** uses a smarter probabilistic initialisation:

<div class="post-flow" role="group" aria-label="k-Means++ initialisation">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Choose first centroid μ₁ uniformly at random from the data</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">For each point xᵢ, compute D(xᵢ) = min distance to any chosen centroid</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Choose next centroid with probability ∝ D(xᵢ)²</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Repeat steps 2–3 until k centroids are chosen</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Run standard k-Means assignment/update from these centroids</span></li>
  </ol>
</div>

By sampling proportional to squared distance, k-Means++ ensures initial centroids are spread across the data. This gives an `O(log k)` approximation guarantee on the WCSS objective and empirically reduces both the number of iterations to convergence and the final WCSS.

### Convergence Properties

k-Means is guaranteed to converge — WCSS strictly decreases (or stays the same) at each step, and there are finitely many partitions. However:

- **Local optima:** k-Means converges to a local minimum. The solution depends on initialisation. k-Means++ significantly improves quality but does not guarantee global optimality.
- **Running time:** each iteration costs `O(nkd)`. In practice, k-Means converges in tens to hundreds of iterations for well-separated clusters. Total complexity: `O(nkdT)` where `T` is the number of iterations.
- **Mini-batch k-Means:** for large datasets, update centroids on random mini-batches rather than the full dataset. Loses exact convergence guarantee but is much faster and gives near-identical results.

### Choosing k: Elbow Method and Silhouette Analysis

**Elbow method:** Plot WCSS as a function of `k`. As `k` increases, WCSS decreases monotonically (more clusters = smaller within-cluster distances). The "elbow" is the point where additional clusters provide diminishing WCSS reduction. This is subjective — the curve is often smooth without a clear elbow, especially on real data.

```
WCSS(k) = Σₖ Σᵢ∈Cₖ ‖xᵢ − μₖ‖²
```

**Silhouette analysis:** For each point `i`:

```
a(i) = mean distance from i to all other points in its cluster     (intra-cluster cohesion)
b(i) = min over other clusters c: mean distance from i to points in c    (nearest-cluster separation)
s(i) = (b(i) − a(i)) / max(a(i), b(i))    ∈ [−1, 1]
```

Average silhouette score over all points should be maximised. Unlike the elbow method, silhouette has a clear optimisation criterion — choose the `k` with the highest average silhouette.

**Gap statistic:** compares WCSS for your clustering to a reference null distribution (uniform random data). The optimal `k` is where the gap `log(WCSS_ref) − log(WCSS_data)` is maximised. More statistically grounded than the elbow method.

### When k-Means Fails

k-Means produces poor clusterings when:
- Clusters are non-convex or ring-shaped (use DBSCAN or spectral clustering)
- Clusters have very different sizes or densities (k-Means forces approximately equal-sized clusters)
- `d` is large (distances concentrate — all points become equally far from centroids)
- Data contains outliers (outliers drag centroids; consider k-Medoids or trimmed k-Means)

> **Interview question:** Why does k-Means++ provide better results than random initialisation, and what is the theoretical guarantee?
>
> *Answer:* k-Means++ spreads the initial centroids by sampling each successive centroid with probability proportional to the squared distance from the nearest already-chosen centroid. This prevents the common failure mode of random initialisation where multiple centroids land in the same dense region, leaving other clusters without a nearby initial centroid. Theoretically, Arthur and Vassilvitskii (2007) proved that the expected WCSS of k-Means++ initialisation (before running any Lloyd iterations) is `O(log k)` times the optimal WCSS — compared to no guarantee for random initialisation. In practice, k-Means++ usually reaches a solution within 5-10% of the global optimum and typically converges in fewer iterations, making it the default initialisation in all major ML libraries (scikit-learn's default is `init='k-means++'`).*

---

## Hierarchical Clustering
{: #hierarchical}

Hierarchical clustering builds a tree (dendrogram) of nested clusters rather than a flat partition. It does not require specifying `k` in advance — you can extract any number of clusters by "cutting" the dendrogram at a chosen level.

### Agglomerative (Bottom-Up) Clustering

Start with each point as its own cluster; repeatedly merge the two closest clusters until one cluster remains:

<div class="post-flow" role="group" aria-label="Agglomerative clustering steps">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Initialise: each of n points is its own cluster</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Compute pairwise distance matrix D (n × n)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Merge the two clusters with smallest linkage distance</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Update distance matrix (remove merged clusters, add merged cluster)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Repeat until one cluster remains</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Cut dendrogram at desired level to extract k clusters</span></li>
  </ol>
</div>

### Linkage Criteria

The linkage criterion defines the distance between two clusters:

| Linkage | Distance between clusters A and B | Properties |
|---|---|---|
| Single | `min_{a∈A, b∈B} d(a, b)` | Can find elongated/chain-like clusters; susceptible to chaining |
| Complete | `max_{a∈A, b∈B} d(a, b)` | Produces compact clusters; sensitive to outliers |
| Average (UPGMA) | `mean_{a∈A, b∈B} d(a, b)` | Compromise; robust to outliers |
| Ward | Minimise increase in total within-cluster variance | Tends to produce equally-sized compact clusters; most commonly used |

**Ward linkage** is analogous to k-Means in its objective (minimising WCSS) and tends to produce the most useful clusters in practice. However, it assumes Euclidean distance — other linkage methods can use any distance metric.

### Reading a Dendrogram

A dendrogram represents the merge history. The y-axis shows the linkage distance at which two clusters merged. To extract `k` clusters: draw a horizontal line at a height that cuts the dendrogram into `k` vertical subtrees. The height of the cut is related to cluster cohesion — a large gap between successive merge heights suggests a natural number of clusters.

### Divisive (Top-Down) Clustering

Start with all points in one cluster and recursively split the least cohesive cluster. Divisive methods are less common because finding the optimal split is computationally expensive (NP-hard in general). Bisecting k-Means (apply k-Means with k=2 recursively) is a practical approximation.

**Complexity:** Agglomerative clustering is `O(n² log n)` with efficient priority queue implementations (naive is `O(n³)`). Computing the initial distance matrix is `O(n²d)`. For large `n`, approximate methods (sampling, mini-batch hierarchical) are needed.

**Pros and Cons — Hierarchical Clustering**

| | Pros | Cons |
|---|---|---|
| **k selection** | No need to specify `k` in advance; dendrogram reveals natural cluster structure | Computationally expensive: `O(n² log n)` time and `O(n²)` memory for distance matrix |
| **Flexibility** | Can use any distance metric; different linkages reveal different cluster shapes | Merges are irreversible — a bad merge early propagates to all subsequent levels |
| **Interpretability** | Dendrogram provides a full hierarchical view of cluster relationships | Choice of linkage significantly affects results; no single "correct" choice |

> **Interview question:** Compare hierarchical clustering with k-Means. When would you choose each?
>
> *Answer:* k-Means is a flat partitioning algorithm — it produces exactly `k` non-overlapping clusters and requires you to specify `k` in advance. Hierarchical clustering builds a full tree of nested partitions, letting you examine cluster structure at all granularities and choose `k` post hoc by cutting the dendrogram. k-Means scales to large datasets (`O(nkdT)`), while agglomerative clustering requires `O(n²)` memory for the distance matrix, limiting it to datasets of perhaps tens of thousands of points. Hierarchical clustering is preferred when: the number of clusters is unknown and you want to explore the hierarchical structure; when clusters at multiple granularities are of interest; or when you are analysing small datasets in domains like bioinformatics (phylogenetic trees, gene expression) where the hierarchy itself is meaningful. k-Means is preferred for large-scale applications where speed and scalability matter more than hierarchy.*

---

## Isolation Forest
{: #isolation-forest}

**[Isolation Forest](https://arxiv.org/abs/2305.17691)** is an anomaly detection algorithm based on the insight that anomalies are rare and different — they are easier to isolate than normal points.

### Core Idea

An isolation tree randomly selects a feature and a split value (uniformly between the min and max of that feature). Anomalies, being extreme or unusual, require fewer random splits to isolate them into a leaf node. Normal points, being in dense regions, require many splits.

<div class="post-flow" role="group" aria-label="Isolation Forest mechanism">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Build T isolation trees on random subsamples (typically 256 points each)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">For each point, record path length h(x) in each tree (number of splits to isolate x)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Average path length E[h(x)] over all T trees</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Anomaly score s(x) = 2^(−E[h(x)]/c(n)) where c(n) is average path length for n points</span></li>
  </ol>
</div>

The anomaly score `s(x)` is in (0, 1):
- `s → 1`: short path length → anomaly
- `s → 0.5`: path length ≈ average → undecided
- `s → 0`: long path length → normal point

### Why Subsampling Works

Isolation Forest uses small subsamples (256 points by default) to build each tree. Anomalies are still isolated quickly in subsamples because they remain outliers relative to any subset of normal data. Subsampling also prevents normal points from having very long paths — it caps the path length effectively. This makes Isolation Forest fast and memory-efficient compared to distance-based methods.

**Complexity:** `O(T · ψ · log ψ)` training (where `ψ` is the subsample size, typically 256) and `O(T · log ψ)` per query — nearly constant in `n`, making it one of the most scalable anomaly detection methods.

### Comparison with Other Anomaly Detection Methods

| Method | Approach | Strengths | Weaknesses |
|---|---|---|---|
| Isolation Forest | Path length in random trees | Fast, scalable, no density estimation | Struggles with local anomalies in dense datasets |
| LOF (Local Outlier Factor) | Local density ratio | Detects local anomalies; no distributional assumptions | `O(n²)` training; sensitive to `k` |
| GMM / one-class SVM | Density estimation / boundary | Full density model | Assumes distributional form; slow for large `n` |
| z-score / IQR | Statistical | Simple, fast | Only detects global anomalies; assumes unimodal distributions |

**Pros and Cons — Isolation Forest**

| | Pros | Cons |
|---|---|---|
| **Scalability** | Near-linear training `O(T · ψ · log ψ)`; fast inference; scales to millions of points | Anomaly score threshold must be chosen; not inherently a probability |
| **Assumptions** | No distributional assumptions; works in high dimensions | Struggles when anomalies form small clusters (not individually isolated); biased for certain feature-space geometries |
| **Practical** | Handles high-dimensional data without distance computation; robust to irrelevant features (random feature selection acts as implicit feature selection) | Hyperparameter `contamination` (expected fraction of anomalies) must be estimated; can flag dense anomaly clusters as normal |

> **Interview question:** How does Isolation Forest detect anomalies without computing distances or densities?
>
> *Answer:* Isolation Forest exploits the structural property that anomalies are isolated — they are few and different from the majority. Rather than measuring density or distance, it measures how quickly a point can be separated from the rest using random axis-aligned splits. An isolation tree is built by recursively choosing a random feature and a random split threshold, partitioning the data, and stopping when each point is in its own leaf. An anomaly in an extreme region of the feature space is separated from all other points after very few random splits — it lives in a sparse region, so any split near it is likely to isolate it. A normal point in a dense region requires many splits because it is surrounded by similar points that must also be separated. By averaging path lengths across many trees (each built on a random subsample), Isolation Forest produces a robust anomaly score that does not depend on any distance metric, covariance estimate, or density model.*

---

## PCA Deep Dive
{: #pca-deep}

The basic PCA algorithm is covered in the Dimensionality Reduction section. Here we examine the statistical interpretation, explained variance, the scree plot, and the practical limitations of PCA.

### Explained Variance and Scree Plot

After computing the eigendecomposition `Σ = VΛVᵀ`, the eigenvalues `λ₁ ≥ λ₂ ≥ ... ≥ λ_d` represent the variance of the data along each principal component direction.

**Proportion of variance explained by component `k`:**

```
EVR_k = λₖ / Σᵢ λᵢ
```

**Cumulative explained variance:** `Σᵢ₌₁ᵏ EVR_i`. A common rule of thumb is to retain enough components to explain 95% of total variance.

**Scree plot:** a bar or line chart of `λₖ` vs component index `k`, sorted descending. The "elbow" — the point where the curve flattens — indicates where additional components explain little additional variance. Kaiser's rule (drop components with `λₖ < 1`, i.e., components that explain less variance than a single original feature after standardisation) is another common heuristic.

### PCA via SVD

In practice, PCA is computed using the **Singular Value Decomposition (SVD)** of the centred data matrix `X`:

```
X = U Σ Vᵀ   (X is n×d, U is n×n, Σ is n×d diagonal, V is d×d)
```

The principal components are the columns of `V`. The projected data is `Z = XV_k = U_k Σ_k`. SVD is numerically more stable than explicitly computing `XᵀX` and eigendecomposing — for large `d`, truncated SVD (computing only the top `k` singular values) is much more efficient.

### Assumptions and Limitations

| Assumption | Reality |
|---|---|
| Linearity — important variation is in linear combinations of features | Many real patterns are non-linear; PCA misses them (use kernel PCA or autoencoders) |
| Variance = information — high variance directions matter most | In supervised settings, low-variance directions may be discriminative (LDA captures this) |
| Gaussian data — PCA is optimal (MLE) under Gaussian assumption | Non-Gaussian data: ICA, sparse PCA, or non-linear methods may be better |
| Scale-invariance — PCA is not scale-invariant | Features must be standardised (zero mean, unit variance) before PCA; otherwise high-scale features dominate |

**When PCA fails:**
- **Rotational ambiguity:** principal components are defined up to sign; the direction is unique but the sign is arbitrary
- **Non-linear structure:** a Swiss roll in 3D cannot be "unrolled" by PCA — it sees it as a blob. Use UMAP, t-SNE, or manifold learning
- **Many correlated components:** if `d >> n`, most eigenvalues are zero — PCA only finds `min(n, d)` non-trivial components
- **Supervised tasks:** PCA may discard discriminative variance. Always check if PCA preprocessing hurts downstream classifier accuracy

> **Interview question:** When would PCA hurt the performance of a downstream classifier, and what should you use instead?
>
> *Answer:* PCA discards the directions of lowest variance, which are chosen purely based on unsupervised structure in `X`. However, the directions of lowest input variance may sometimes be the most discriminative — for example, if two classes differ primarily along a subtle direction that happens to have low overall variance. In this case, PCA preprocessing removes the most useful signal before the classifier even sees it. This is especially common in high-dimensional problems where class-discriminative structure is orthogonal to the principal variance. LDA is a supervised alternative that explicitly maximises class separability rather than total variance. Alternatively, the number of PCA components should be treated as a hyperparameter tuned by cross-validation on the downstream task rather than chosen purely by explained variance threshold.*

> **Interview question:** Explain the relationship between PCA and SVD. Why is SVD preferred for computation?
>
> *Answer:* PCA finds the eigenvectors of the covariance matrix `XᵀX/n`. The SVD of `X` (centred) is `X = UΣVᵀ`, where the columns of `V` are the right singular vectors, which are exactly the eigenvectors of `XᵀX`. The singular values `σᵢ` relate to the eigenvalues by `λᵢ = σᵢ²/n`. SVD is preferred for three reasons: numerical stability (explicitly computing `XᵀX` squares the condition number, amplifying floating-point errors); efficiency when `d < n` (SVD runs in `O(nd²)` vs `O(nd² + d³)` for the two-step approach); and most importantly, truncated SVD computes only the top `k` singular values/vectors in `O(nkd)` time using algorithms like Lanczos or randomised SVD, whereas eigendecomposing the full `XᵀX` always costs `O(d³)` regardless of how many components you need.*

---

## UMAP
{: #umap}

**UMAP (Uniform Manifold Approximation and Projection)** is a dimensionality reduction algorithm grounded in topology and Riemannian geometry, developed by McInnes et al. (2018). It has largely supplanted t-SNE for visualisation tasks and can also be used as a general-purpose dimensionality reduction preprocessing step — unlike t-SNE.

### Core Algorithm

UMAP constructs a fuzzy topological representation of the high-dimensional data and then optimises a low-dimensional representation to have as similar a topology as possible.

<div class="post-flow" role="group" aria-label="UMAP algorithm phases">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Phase 1 (Graph construction): for each point, find k nearest neighbours; compute fuzzy membership strengths using an adaptive local distance metric</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Phase 2 (Graph symmetrisation): combine directed graph into undirected: w(i,j) = w(i→j) + w(j→i) − w(i→j)·w(j→i)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Phase 3 (Low-dim embedding): randomly initialise 2D layout; optimise using stochastic gradient descent to minimise cross-entropy between high-dim and low-dim fuzzy graphs</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Output: low-dimensional embedding that preserves both local and global structure</span></li>
  </ol>
</div>

**Key difference from t-SNE:** UMAP normalises distances locally (each point uses its own scale based on its `k`th nearest neighbour distance), which causes it to better preserve the relative global distances between clusters. t-SNE normalises globally, which causes arbitrary distortion of inter-cluster distances.

### Hyperparameters

| Parameter | Effect |
|---|---|
| `n_neighbors` | Size of local neighbourhood. Small → local structure; large → global structure. Analogous to `perplexity` in t-SNE. Typical: 5–50 |
| `min_dist` | Minimum distance between points in embedding. Small → tightly packed clusters; large → more spread. Typical: 0.0–0.5 |
| `n_components` | Output dimensionality. 2 for visualisation; higher for preprocessing |
| `metric` | Distance metric for high-dimensional space. Can use any metric, including custom ones |

### UMAP vs t-SNE

| | UMAP | t-SNE |
|---|---|---|
| Speed | Much faster (`O(n log n)` vs `O(n² log n)`) | Slow for n > 10k |
| Global structure | Better preserved | Poor — inter-cluster distances arbitrary |
| Reproducibility | Stochastic but more stable | Highly sensitive to random seed and `perplexity` |
| Inference on new points | Supported (transform new points using fitted model) | Not supported without re-running |
| Theoretical grounding | Riemannian geometry, fuzzy topology | Information theory (KL divergence) |
| Use as preprocessing | Appropriate (with `n_components > 2`) | Not appropriate |

**Pros and Cons — UMAP**

| | Pros | Cons |
|---|---|---|
| **Speed** | `O(n log n)` — scales to millions of points with approximate NN | Stochastic — different runs give different embeddings (fix with `random_state`) |
| **Structure** | Preserves both local and global cluster structure better than t-SNE | `n_neighbors` and `min_dist` require tuning — different settings reveal different aspects of structure |
| **Flexibility** | Supports arbitrary metrics; can embed new points; works as preprocessing for downstream models | Mathematical complexity makes it harder to reason about what is distorted |

> **Interview question:** Why is UMAP preferred over t-SNE for most modern dimensionality reduction tasks?
>
> *Answer:* UMAP offers several practical advantages over t-SNE. First, it is significantly faster — `O(n log n)` compared to t-SNE's `O(n² log n)` — making it tractable for datasets with hundreds of thousands of points. Second, UMAP preserves global structure better: because distances are normalised locally per point rather than globally, the relative positions of clusters in the embedding are more meaningful. Third, UMAP supports out-of-sample extension — once the model is fitted, new points can be embedded using the learned mapping without re-running the full algorithm, which is essential for production pipelines. Finally, UMAP with `n_components > 2` can be used as a general dimensionality reduction preprocessing step for downstream classifiers, which t-SNE cannot because its objective distorts the geometry in ways that harm downstream tasks. The only real advantage of t-SNE is its longer track record and wider recognition — UMAP is generally the better default choice.*

---

## Hidden Markov Models
{: #hmm}

A **Hidden Markov Model (HMM)** is a probabilistic model for sequential data where the system is assumed to be a Markov process with unobserved (hidden) states. HMMs are the classical tool for sequence modelling in speech recognition, NLP (POS tagging), bioinformatics (gene finding), and time series.

### Model Definition

An HMM is defined by five components:

```
States:          S = {s₁, s₂, ..., sN}             (hidden states)
Observations:    O = {o₁, o₂, ..., oM}             (observed vocabulary)
Transition:      A[i][j] = P(sⱼ at t+1 | sᵢ at t)  (N×N matrix)
Emission:        B[i][k] = P(oₖ | sᵢ)              (N×M matrix)
Initial:         π[i] = P(s₁ = sᵢ)                 (N-vector)
```

The **Markov assumption**: the current state depends only on the immediately preceding state. The **output independence assumption**: the current observation depends only on the current state, not on previous states or observations.

### Three Fundamental Problems

**1. Evaluation (likelihood):** Given a model `λ = (A, B, π)` and observation sequence `O`, compute `P(O | λ)`.

Solved by the **forward algorithm** (dynamic programming):

```
α_t(i) = P(o₁, o₂, ..., oₜ, qₜ = sᵢ | λ)
α₁(i)  = π(i) · B[i][o₁]
α_{t+1}(j) = [Σᵢ α_t(i) · A[i][j]] · B[j][o_{t+1}]
P(O | λ) = Σᵢ α_T(i)
```

Complexity: `O(N²T)` vs naive `O(N^T)`.

**2. Decoding:** Given `O` and `λ`, find the most likely hidden state sequence `Q* = argmax_Q P(Q | O, λ)`.

Solved by the **Viterbi algorithm** — identical structure to the forward algorithm but replacing sum with max:

```
δ_t(i) = max_{q₁,...,q_{t-1}} P(q₁,...,q_{t-1}, qₜ=sᵢ, o₁,...,oₜ | λ)
```

Backtrack from the final state to recover the most probable path.

**3. Learning:** Given observation sequences, find `λ = (A, B, π)` that maximises `P(O | λ)`.

Solved by the **Baum-Welch algorithm** (an EM algorithm specialised for HMMs):

<div class="post-flow" role="group" aria-label="Baum-Welch EM for HMM">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">E-step: run forward-backward to compute γ_t(i) = P(qₜ=sᵢ | O, λ) and ξ_t(i,j) = P(qₜ=sᵢ, q_{t+1}=sⱼ | O, λ)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">M-step: update A[i][j] = Σₜ ξ_t(i,j) / Σₜ γ_t(i) and B[i][k] = Σ_{t: oₜ=k} γ_t(i) / Σₜ γ_t(i)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Repeat until log-likelihood converges</span></li>
  </ol>
</div>

### Practical Considerations

**Numerical underflow:** probabilities `α_t(i)` become exponentially small with sequence length. In practice, work in log-space or use scaled probabilities.

**Number of states:** must be specified as a hyperparameter. Use cross-validation on held-out sequences or the BIC/AIC criterion.

**Gaussian HMMs:** for continuous observations, replace the discrete emission matrix `B` with Gaussian (or GMM) emission distributions per state — `P(oₜ | qₜ = sᵢ) = 𝒩(oₜ; μᵢ, Σᵢ)`. This is the standard for speech recognition and time series.

**Pros and Cons — Hidden Markov Models**

| | Pros | Cons |
|---|---|---|
| **Sequence modelling** | Natural model for sequential data with latent structure; principled probabilistic framework | Strong Markov assumption (memoryless) and output independence assumption often violated in real sequences |
| **Algorithms** | Exact inference via forward-backward and Viterbi in `O(N²T)`; Baum-Welch provides principled parameter learning | Number of hidden states must be specified; EM converges to local optima |
| **Interpretability** | Hidden states can be interpreted as latent "modes" of a process | For long-range dependencies, HMMs are inadequate (LSTMs and transformers handle this better) |

> **Interview question:** Explain the Viterbi algorithm and why it is more efficient than brute-force decoding.
>
> *Answer:* The Viterbi algorithm finds the most probable hidden state sequence given an observation sequence using dynamic programming. Brute-force evaluation would consider all `N^T` possible state sequences (exponential in sequence length), computing the joint probability of each. Viterbi avoids this by exploiting the Markov property: the optimal state at time `t` depends only on the optimal state at time `t-1`. It computes `δ_t(i)` — the maximum probability of any state sequence ending in state `i` at time `t` — using the recurrence `δ_{t+1}(j) = max_i [δ_t(i) · A[i][j]] · B[j][o_{t+1}]`. By caching the argmax at each step (the "backpointer"), the full optimal path can be recovered by backtracking from the highest-probability final state. This reduces the complexity from `O(N^T)` to `O(N²T)` — polynomial rather than exponential in sequence length.*

> **Interview question:** How does an HMM differ from a Recurrent Neural Network for sequence modelling, and when would you prefer each?
>
> *Answer:* An HMM is a generative probabilistic model with explicit discrete latent states, transition probabilities, and emission distributions. It has limited expressive power — the number of states is fixed, the Markov assumption limits memory to one step, and the output independence assumption ignores context within a state. An RNN is a discriminative model (or can be generative) with a continuous hidden state of arbitrary dimension that is updated deterministically by the previous state and current input; it can in principle capture long-range dependencies. HMMs are preferred when: interpretable latent states are desired; the dataset is small (HMMs have few parameters); exact probabilistic inference is needed (posterior over states, likelihood computation); or the sequence length is short. RNNs (and especially LSTMs and transformers) are preferred when: the sequences are long with complex long-range dependencies; large labelled datasets are available; and raw predictive performance matters more than interpretability. For modern NLP, HMMs have been almost entirely replaced by transformer-based models.*
