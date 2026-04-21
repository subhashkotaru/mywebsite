---
title: "Classic Machine Learning"
date: 2026-04-21
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

**Gradient boosting** builds an additive model: `F(x) = Σₘ αₘhₘ(x)` where each `hₘ` is a shallow tree fitted to the negative gradient of the loss. **XGBoost**, **LightGBM**, and **CatBoost** are engineering-optimised implementations that dominate structured/tabular ML competitions. LightGBM's histogram-based splitting and leaf-wise (vs level-wise) growth make it particularly fast on large datasets.

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

### k-Nearest Neighbours
{: #knn}

**kNN** is non-parametric: no explicit training step. To predict for a new point `x`, find the `k` closest training points and aggregate their labels (majority vote for classification, average for regression):

```
ŷ = (1/k) Σᵢ∈Nₖ(x) yᵢ
```

**Distance metric** matters: Euclidean distance works for continuous features of similar scale; Hamming for binary; cosine for high-dimensional sparse vectors (text, embeddings). Features must be normalised — kNN is highly sensitive to scale.

**Bias-variance**: `k=1` — zero training error, high variance (memorises training set). Large `k` — smooths predictions, increases bias. Optimal `k` is found by cross-validation.

**Computational cost**: naive kNN is `O(nd)` per query (`n` training points, `d` dimensions). Approximate nearest-neighbour structures (KD-trees for low `d`, HNSW/FAISS for high `d`) reduce this to sub-linear. kNN on top of learned embeddings (embedding + FAISS search) is the core of many modern retrieval systems.

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
    <p class="post-flow__col-label">DBSCAN ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Discovers k automatically</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Handles arbitrary shapes</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Explicitly identifies outliers as noise</span></li>
    </ol>
  </div>
</div>

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

**t-SNE** (t-Distributed Stochastic Neighbour Embedding) is a non-linear dimensionality reduction for **visualisation** (typically to 2D or 3D). It models pairwise similarities in high-dimensional space as probabilities and minimises KL divergence between those and the corresponding low-dimensional similarities:

```
High-dim similarity:  pᵢⱼ = (exp(−‖xᵢ−xⱼ‖²/2σ²)) / Σ exp(...)
Low-dim similarity:   qᵢⱼ = (1 + ‖yᵢ−yⱼ‖²)⁻¹ / Σ (1 + ...)   # t-distribution (heavier tail)
```

t-SNE preserves local structure (nearby points in high-dim stay nearby in 2D) but distorts global structure. It is not a general-purpose dimensionality reduction — it should not be used as a preprocessing step for other algorithms. **UMAP** is a faster alternative with better global structure preservation and is now preferred for most visualisation tasks.

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

**AdaGrad** adapts the learning rate per parameter — parameters with large gradients get smaller updates, sparse parameters get larger updates. Useful for sparse features (NLP), but the learning rate monotonically decreases and eventually stalls.

**Adam** combines momentum (first moment) with per-parameter adaptive rates (second moment):

```
m ← β₁·m + (1−β₁)·g
v ← β₂·v + (1−β₂)·g²
θ ← θ − η · m̂/(√v̂ + ε)
```

Adam is the default for neural networks. For linear models and SVMs, L-BFGS (a quasi-Newton method that approximates the Hessian) often converges in fewer iterations for small-to-medium datasets.

**Convexity**: for convex loss functions (linear/logistic regression, SVMs), gradient descent finds the global minimum regardless of initialisation. For non-convex losses (neural networks, GMMs), gradient descent finds a local minimum — which, in practice, is often good enough.
