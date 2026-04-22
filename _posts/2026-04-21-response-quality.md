---
title: "Response Quality in LLMs"
date: 2026-04-21
description: "A deep technical guide to factuality, reasoning, calibration, instruction following, safety, and evaluation in large language models — covering hallucination taxonomy, CoT variants, process reward models, benchmarks, and production monitoring."
tags: [llm, evaluation, factuality, reasoning, alignment]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#factuality">Factuality</a>
      <ul class="post-toc-sublist">
        <li><a href="#factuality-taxonomy">Taxonomy</a></li>
        <li><a href="#hallucination-types">Hallucination Types</a></li>
        <li><a href="#hallucination-causes">Causes</a></li>
        <li><a href="#hallucination-mitigation">Detection & Mitigation Techniques</a></li>
        <li><a href="#factuality-evaluation">Evaluation Methods</a></li>
        <li><a href="#factuality-improvement">Improvement Strategies</a></li>
        <li><a href="#factuality-monitoring">Production Monitoring</a></li>
        <li><a href="#factuality-failure">Failure Modes</a></li>
      </ul>
    </li>
    <li><a href="#reasoning">Reasoning</a>
      <ul class="post-toc-sublist">
        <li><a href="#reasoning-types">Types of Reasoning</a></li>
        <li><a href="#prompting-reasoning">Prompting-Based Reasoning</a></li>
        <li><a href="#search-reasoning">Search-Based Reasoning</a></li>
        <li><a href="#tool-reasoning">Tool-Augmented Reasoning</a></li>
        <li><a href="#rl-reasoning">RL-Based Reasoning</a></li>
        <li><a href="#reasoning-failure">Failure Modes</a></li>
      </ul>
    </li>
    <li><a href="#calibration">Calibration & Uncertainty</a>
      <ul class="post-toc-sublist">
        <li><a href="#ece">Expected Calibration Error</a></li>
        <li><a href="#uncertainty-expression">Expressing Uncertainty</a></li>
        <li><a href="#calibration-improvement">Improving Calibration</a></li>
      </ul>
    </li>
    <li><a href="#instruction-following">Instruction Following</a>
      <ul class="post-toc-sublist">
        <li><a href="#constraint-types">Constraint Types</a></li>
        <li><a href="#ifeval">IFEval & Rubrics</a></li>
        <li><a href="#refusal-quality">Refusal Quality</a></li>
      </ul>
    </li>
    <li><a href="#safety">Safety & Alignment</a>
      <ul class="post-toc-sublist">
        <li><a href="#refusal-calibration">Refusal Calibration</a></li>
        <li><a href="#constitutional-ai">Constitutional AI & RLAIF</a></li>
        <li><a href="#safety-failure">Failure Modes</a></li>
      </ul>
    </li>
    <li><a href="#output-quality">Output Format & Coherence</a></li>
    <li><a href="#evaluation">Evaluation Methods</a>
      <ul class="post-toc-sublist">
        <li><a href="#human-eval">Human Evaluation</a></li>
        <li><a href="#llm-judge">LLM-as-Judge</a></li>
        <li><a href="#reference-metrics">Reference-Based Metrics</a></li>
        <li><a href="#benchmarks">Benchmarks</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## Overview
{: #overview}

Response quality is the aggregate of all properties that make an LLM output useful, correct, and safe. It sits downstream of pretraining and fine-tuning decisions but is ultimately what users experience. The key dimensions:

| Dimension | Core question | Primary failure |
|---|---|---|
| **Factuality** | Are claims grounded in verifiable truth? | Hallucination |
| **Reasoning** | Are intermediate steps logically sound? | Shortcut/spurious patterns |
| **Calibration** | Does expressed confidence match actual accuracy? | Overconfidence |
| **Instruction following** | Are all constraints in the prompt satisfied? | Constraint omission |
| **Safety** | Does the response avoid harm? | Over- or under-refusal |
| **Coherence** | Is the output well-structured and consistent? | Contradiction, verbosity |

The formal factuality score for a response $y$ to prompt $x$:

$$F(x,y) \in [0,1]$$

representing the fraction of claims in $y$ that are supported by ground-truth evidence. The fundamental reasoning decomposition externalises intermediate steps $z$:

$$p_\theta(y|x) = \sum_z p_\theta(y|x,z)\, p_\theta(z|x)$$

Externalising $z$ into visible chain-of-thought text makes reasoning debuggable and improvable.

---

## Factuality
{: #factuality}

Factuality measures whether LLM outputs align with verifiable external truths — not internal consistency, but correspondence with reality. Long-form responses compound errors because each sentence introduces new factual claims, expanding the verification space multiplicatively.

### Taxonomy
{: #factuality-taxonomy}

**Short-form factuality.** Binary correctness on single-claim responses (factoid Q&A):

$$F_\text{short}(x,y) = \mathbb{1}[\text{answer matches ground truth}]$$

Benchmarks: [SimpleQA](https://arxiv.org/abs/2411.07905) (Wei et al. 2024, 4,326 single-answer questions with abstention scoring), [TruthfulQA](https://arxiv.org/abs/2109.07958) (Lin et al. 2021), HaluEval, FreshQA.

**Truthfulness (intrinsic factuality).** Claims correspond to real-world facts independent of any provided context document:

$$F_\text{truth}(y) = \frac{1}{|\phi(y)|} \sum_i f_i, \quad f_i = \mathbb{1}[\text{claim } a_i \text{ is globally true}]$$

Challenges: knowledge staleness, plausibility bias (models generate plausible-sounding but false statements), gaps between internal representations and surface text.

**Faithfulness (extrinsic factuality).** Output consistency with a provided context document $D$:

$$F_\text{faith}(y|D) = \frac{1}{N} \sum_i \mathbb{1}[\text{Entail}(D, a_i)]$$

Faithful responses stay within what the evidence supports without adding unsupported claims. Key methods: [FEVER](https://arxiv.org/abs/1803.05355), FactCC, [QAGS](https://arxiv.org/abs/2004.04228), SummaC.

**Groundedness (retrieval-augmented factuality).** Extends faithfulness by requiring both claim support AND adequate evidence quality:

$$F_\text{ground}(y|D) = \frac{1}{N} \sum_i \mathbb{1}\!\left[\exists d_j \in D : \text{Entail}(d_j, a_i)\right] \cdot \mathbb{1}[\text{Suff}(D, y)]$$

The sufficiency term $\mathbb{1}[\text{Suff}(D,y)]$ penalises systems that retrieve irrelevant passages and still generate supported-sounding text. [RAGAS](https://arxiv.org/abs/2309.15217) (Es et al. 2023) provides four reference-free groundedness metrics:

$$F_\text{RAGAS} = \frac{1}{4}\left(F_\text{faith} + P_\text{ctx} + R_\text{ctx} + R_\text{ans}\right)$$

where $P_\text{ctx} = \text{relevant used}/\text{total used}$ and $R_\text{ctx} = \text{relevant used}/\text{relevant available}$.

**Long-form factuality.** Multi-sentence outputs require atomic claim decomposition:

$$F_\text{long}(y) = \frac{1}{M} \sum_{i=1}^M f_i$$

[FActScore](https://arxiv.org/abs/2305.14251) (Min et al. 2023) formalises this; [LongFact / SAFE](https://arxiv.org/abs/2403.18802) (Wei et al. 2024) adds web search verification per atomic claim, achieving ~0.8 correlation with human annotation.

### Hallucination Types
{: #hallucination-types}

Hallucination is the error-centric complement to factuality: $H(y) = 1 - F(y)$.

$$H(y) = \frac{1}{M} \sum_{i=1}^M \mathbb{1}[\text{Hallucination}(a_i)]$$

The full taxonomy (Ye et al. 2024, Ji et al. 2023, Zhou et al. 2023):

| Type | Description | Example |
|---|---|---|
| **Intrinsic** | Contradicts information provided in the same context | Summariser contradicts source document |
| **Extrinsic** | Contradicts verifiable external reality | Wrong date, wrong person |
| **Entity-level** | Wrong name, number, or date | "Einstein won the Nobel Prize in 1925" (correct: 1921) |
| **Relational** | Misrepresents a relationship between entities | Attributing a quote to the wrong person |
| **Compositional** | Combines real facts into a false composite claim | "The CEO of X founded Y" when only one is true |
| **Inferential** | Draws conclusions beyond what evidence supports | Concluding causation from correlation |
| **Unsupported synthesis** | Adds detail beyond what the source contains | RAG response inventing statistics |
| **Fabricated references** | Cites non-existent papers or URLs | Bogus arXiv IDs (Feng et al. 2024) |
| **Omission-induced** | Leaves out facts that reverse the meaning | "The drug had no serious side effects [in study A]" |

### Causes of Hallucination
{: #hallucination-causes}

| Cause | Mechanism | Mitigation lever |
|---|---|---|
| **Insufficient training data** | Model fails to establish accurate input-output correlations for rare facts | Data curation; retrieval augmentation |
| **Overfitting** | Outputs mirror training distribution rather than generalising to new inputs | Regularisation; diverse data |
| **Inadequate supervision** | Model relies on internal plausibility rather than verifiable grounding | RLHF with factuality rewards; process supervision |
| **Knowledge cutoff** | Temporal boundary means post-training facts are unknown | RAG; FreshQA-style temporal evaluation |
| **Over-reliance on fluency** | Beam search and sampling optimise for fluency, not truth | Factuality-weighted decoding |
| **Attention dilution** | Long contexts dilute relevant evidence, model fills gaps with plausible text | Retrieval chunking; selective compression |

### Hallucination Detection & Mitigation Techniques
{: #hallucination-mitigation}

**[Chain-of-Verification (CoVe)](https://arxiv.org/abs/2309.11495)** (Dhuliawala et al. 2023)

Four-step process that forces the model to explicitly verify its own claims before finalising a response:

```
Baseline response
    → Verification planning (generate verification questions)
    → Independent execution (answer each question without seeing baseline)
    → Final response (revise using verification results)
```

Variants ranked by effectiveness: **Factor+Revise** (factored verification, best) > Factored CoVe > Two-Step CoVe > Joint CoVe. Factored CoVe performs best by isolating verification questions completely from the baseline response, preventing the model from anchoring on its own errors.

**[DoLa — Decoding by Contrasting Layers](https://arxiv.org/abs/2309.03883)** (Chuang et al. 2023)

Observation: later (mature) transformer layers encode factual knowledge; earlier (premature) layers capture surface linguistic patterns. DoLa dynamically selects the premature layer per token using Jensen-Shannon Divergence, then subtracts premature layer log-probabilities from mature layer outputs:

$$\log p_\text{DoLa}(y_t \mid \cdot) = \log p_\text{mature}(y_t \mid \cdot) - \log p_\text{premature}(y_t \mid \cdot)$$

This amplifies factual signal relative to fluency signal. Results: **12–17% absolute improvement on TruthfulQA** with minimal computational overhead — no extra inference pass, no retriever needed.

**Active Detection via Low-Confidence Validation** (Varshney et al. 2023)

1. Generate initial response; identify low-confidence tokens via uncertainty metrics.
2. Generate validation questions specifically targeting the uncertain spans.
3. Answer validation questions; if answers contradict original claims, revise.

Results: 88% hallucination detection recall; mitigates 57.6% of detected hallucinations; does not introduce new false hallucinations.

**[SelfCheckGPT](https://arxiv.org/abs/2303.08896)** (Manakul et al. 2023)

Zero-resource detection without external databases. Premise: accurate knowledge yields consistent outputs across samples; hallucinations produce varied responses.

1. Sample $K$ stochastic completions from the model for the same prompt.
2. For each sentence $s_i$ in the primary response, measure semantic consistency with the $K$ samples.
3. Low consistency → high hallucination probability.

Effective at both sentence and passage levels. Works across model families; no access to logits required.

**MixAlign: Interactive Question-Knowledge Alignment** (Zhang et al.)

Addresses query-knowledge base misalignment — hallucinations caused when the model's interpretation of the query diverges from what the knowledge base actually contains. Two-pronged approach: model-based query refinement + human clarification when ambiguities persist.

Results: 22.2% performance improvement; 27.1% hallucination reduction vs. baselines.

**Factually Augmented RLHF (Fact-RLHF)** (Sun et al.)

Addresses multimodal hallucination in vision-language models. Standard RLHF reward models don't assess visual grounding. Fact-RLHF augments the reward model with image captions and ground-truth options during preference scoring:

```
Supervised fine-tuning → preference modelling (augmented with captions)
    → PPO-based RL → factual augmentation
```

Results: 94% of GPT-4 text performance; 60% improvement on MMHAL-BENCH (object attributes, adversarial objects, comparisons, counting, spatial relations).

**Technique selection guide:**

| Technique | Cost | When to use | Not suitable for |
|---|---|---|---|
| **DoLa** | ~0 (no extra pass) | Always-on factuality boost | Domains where linguistic patterns = factual signal |
| **CoVe** | 2–4× inference | High-stakes factual claims | Low-latency applications |
| **SelfCheckGPT** | $K$× inference | When logits unavailable; black-box APIs | Real-time responses |
| **Active detection** | ~1.5× inference | When uncertainty estimation available | Black-box models |
| **RAG** | +retrieval latency | Time-sensitive or domain-specific facts | Creative generation tasks |
| **Fact-RLHF** | Training-time only | Multimodal models | Text-only models |

### Evaluation Methods
{: #factuality-evaluation}

**Human annotation.** Gold standard: annotators label each claim as `Supported / Contradicted / Not Enough Info / Partially Supported`. Sentence-level segmentation via dependency parsing or OpenIE. Inter-annotator agreement via Cohen's κ or Krippendorff's α. High quality but expensive and slow.

**Entailment-based (NLI).** Fine-tuned models (DeBERTa-MNLI, domain-tuned) compute:

$$\Pr(\text{entail} \mid \text{premise} = D,\; \text{hypothesis} = a_i) \geq \tau$$

SummaC improves robustness by aggregating NLI scores across sentence pairs rather than the full document. Works well for short claims; degrades for complex compositional statements.

**QA-based (QAFactEval).** Tests that the model and source agree on question answers:

1. Generate questions $q_1 \ldots q_N$ from the output.
2. Answer each on both source $D$ and output $y$.
3. Score: $F_\text{QA} = \frac{1}{N} \sum_i \mathbb{1}[\text{sim}(\text{Ans}_\text{source}, \text{Ans}_\text{model}) > \tau]$

Semantic matching via sentence-BERT tolerates paraphrasing. Supports domain-specific QA variants (BioASQ, FinQA).

**Retrieval-augmented evaluation (SAFE/LongFact).** Per-claim web search then entailment:

1. GPT-4 extracts atomic claims from $y$.
2. Bing/Serper retrieve supporting snippets per claim.
3. DeBERTa-MNLI: $\Pr > 0.5$ → supported.
4. $F = \text{supported\_count}/\text{total\_claims}$

Parallelised async retrieval with caching. ~0.8 human correlation on open-domain QA.

**LLM-as-judge (FActScore).** Use GPT-4 to label each atomic fact:

1. Decompose $y$ into atomic facts.
2. Prompt GPT-4: "Is this fact supported by the evidence?" → Supported / Refuted / Not Enough Info.
3. $F_\text{fact} = \text{supported}/\text{total}$

Structured JSON output prompts; batch evaluation with retry. Correlates ~0.82 with expert annotation. Key risk: evaluator inherits shared hallucinations with the generator.

**Representation-based detection.** Intrinsic dimension of hidden states correlates with truthfulness (Yin et al. 2024). Activation probes trained on true/false pairs can predict hallucination likelihood before decoding completes (Orgad et al. 2025). Computationally cheap but requires calibration on domain-specific data.

**Confidence-aware decoding.** Trade fluency for factuality at inference:

$$\text{Score}(y) = \lambda \cdot p(y|x) + (1-\lambda) \cdot \text{FactualityVerifier}(y)$$

Tune $\lambda$ on a held-out factuality calibration set.

### Improvement Strategies
{: #factuality-improvement}

**Training-time:**

| Strategy | Mechanism | Notes |
|---|---|---|
| **Data curation** | Prefer Wikipedia, academic papers; periodic crawl refresh | Reduces plausibility bias at source |
| **Instruction tuning on truthful data** | SFT on Tulu-2 style truthful instruction datasets | +factuality on TruthfulQA |
| **RLHF with factuality reward** | Human raters penalise hallucinated claims | Requires careful reward model |
| **Constitutional AI** | Rule-based critiques flag and correct hallucinations | Bai et al. 2022; scalable but prompt-sensitive |
| **Knowledge editing** | [ROME](https://arxiv.org/abs/2202.05262) / [MEMIT](https://arxiv.org/abs/2210.07229) surgically update neurons | Precise but risks side effects |
| **Contrastive fine-tuning** | SFT on (true, false) pairs from FactCC dataset | Teaches discriminative factuality |

**Inference-time:**

| Strategy | Mechanism | Notes |
|---|---|---|
| **[RAG](https://arxiv.org/abs/2005.11401)** | Retrieve top-k passages, condition generation | Lewis et al. 2020; reduces knowledge cutoff risk |
| **Self-consistency (SelfCheckGPT)** | Sample $K$ generations; semantic similarity = confidence | Manakul et al. 2023; high confidence → lower hallucination rate |
| **Verifier-in-the-loop** | Block tokens contradicting evidence mid-decoding | Low latency; requires fast verifier |
| **Faithful RAG** | Rank retrieved passages by factual support; filter before generation | Reduces context pollution |
| **Post-hoc correction (ReFact)** | Detect unsupported claims → retrieve → regenerate corrections | Peng et al. 2023; adds one full inference pass |
| **Self-reflection (Reflexion)** | Critique output → regenerate with critique | Shinn et al. 2023; iterative until threshold |

### Production Monitoring
{: #factuality-monitoring}

Four-layer factuality-centric architecture:

```
Knowledge Layer  →  Reasoning Layer  →  Evaluation Layer  →  Governance Layer
(FAISS/BM25,        (generator +          (SAFE/RAGAS/          (dashboards,
 ETL pipelines)      verifier models)       FActScore)            feedback loops)
```

**Real-time fact logging schema:** `{prompt, response, claims[], scores[], sources[], timestamp}`.

**Factual drift detection:**

$$\text{drift} = 1 - \frac{F_t}{F_{t-1}}$$

Recompute historical scores as evidence updates. Trigger retraining if drift exceeds threshold.

**SLA target:** ≥95% supported claims. Shadow pipelines run factuality evaluation in parallel without adding production latency.

**Routing:** Low-confidence outputs (verifier score < 0.6) → human review queue → annotation feedback → continual fine-tuning.

### Failure Modes
{: #factuality-failure}

| Failure | Root cause | Mitigation |
|---|---|---|
| Reference incompleteness | Ground truth outside reference documents | Expand retrieval corpus; web search fallback |
| Semantic drift | Paraphrased equivalent claims incorrectly penalised | Use semantic similarity (sentence-BERT) not lexical match |
| Evaluator bias | One LLM judging another shares hallucinations | Cross-model or human-in-loop verification |
| Temporal decay | Static parameters can't track fast-changing facts | FreshQA evaluation; time-indexed retrieval |
| Verifier unreliability | Auto-verifiers themselves hallucinate | Ensemble multiple verifiers; human spot-checks |
| Creativity-factuality tension | Over-verification reduces stylistic diversity | Separate factual claims from creative framing |

---

## Reasoning
{: #reasoning}

Reasoning in LLMs is formalised as "learned, compositional computation over latent steps that yields verifiable conclusions." The fundamental decomposition:

$$p_\theta(y|x) = \sum_z p_\theta(y|x,z)\, p_\theta(z|x)$$

Externalising $z$ into chain-of-thought text makes it auditable and improvable.

### Types of Reasoning
{: #reasoning-types}

| Type | Definition | Example tasks |
|---|---|---|
| **Deductive** | Deriving logically necessary conclusions from premises | Symbolic algebra, formal proofs |
| **Inductive** | Generalising patterns from examples | Few-shot extrapolation, schema induction |
| **Abductive** | Inferring the most plausible explanation | Diagnosis, root-cause analysis |
| **Procedural** | Multi-step planning and task decomposition | Agentic task execution |

### Prompting-Based Reasoning
{: #prompting-reasoning}

**[Chain-of-Thought (CoT)](https://arxiv.org/abs/2201.11903)** (Wei et al. 2022; [Kojima et al. 2022](https://arxiv.org/abs/2205.11916))

Elicits step-by-step reasoning via few-shot exemplars or the zero-shot trigger "Let's think step by step."

$$\hat{y} = \arg\max_y \sum_z p_\theta(y,z \mid x)$$

Zero-shot CoT shows reasoning can emerge without demonstrations — the linguistic cue alone unlocks the behaviour. Emergent above ~100B parameters (Wei et al. 2022 scaling analysis).

**Self-Consistency** ([Wang et al. 2022](https://arxiv.org/abs/2203.11171))

Sample $K$ diverse reasoning paths and aggregate via majority vote:

$$\hat{y} = \arg\max_y \sum_{k=1}^K \mathbb{1}\!\left[y^{(k)} = y\right]$$

Reduces variance from individual chains. Typical $K = 10$–$40$; diminishing returns beyond ~20 samples. Primary cost is $K$ inference passes.

**Reflection and Self-Verification** ([Shinn et al. 2023](https://arxiv.org/abs/2303.11366); [Madaan et al. 2023](https://arxiv.org/abs/2303.17651))

Iterative refinement loop:

$$x \xrightarrow{\text{reason}} (z,y) \xrightarrow{\text{reflect}} c \xrightarrow{\text{revise}} (z', y')$$

Variants:
- **Reflexion**: verbal RL — critique stored as natural language memory fed into next attempt
- **Self-Refine**: separated roles (generator vs. critic) in the same model
- **RCOT**: structured correction with retrieval augmentation

**Implicit In-Context Reasoning** ([Brown et al. 2020](https://arxiv.org/abs/2005.14165))

Models perform structured reasoning internally through attention dynamics, formalised as Bayesian posterior inference ([Xie et al. 2022](https://arxiv.org/abs/2205.13109)):

$$p(h \mid x_{1:n}, y_{1:n}) \propto p(h)\prod_i p(y_i \mid x_i, h)$$

Von Oswald et al. (2023) showed transformers approximate gradient descent in-context — ICL may simulate learning rather than pure retrieval.

### Search-Based Reasoning
{: #search-reasoning}

**[Tree-of-Thoughts (ToT)](https://arxiv.org/abs/2305.10601)** (Yao et al. 2023)

Generalises CoT into a branching search tree of partial thoughts. Nodes represent partial reasoning sequences $z_{1:t}$; a value function guides expansion:

$$z_{t+1} \sim \pi_\theta(z_t \mid z_{1:t}), \qquad V_\phi(z_{1:t}) \approx \mathbb{E}[R \mid z_{1:t}]$$

Search strategies: BFS (breadth-first), DFS (depth-first), MCTS. Best for tasks with a clear correctness criterion (puzzles, proofs) where backtracking is valuable.

**Monte Carlo Tree Search (MCTS)**

Balances exploration and exploitation via upper-confidence bound:

$$a^* = \arg\max_a \left(Q(s,a) + c\sqrt{\frac{\log N(s)}{N(s,a)+1}}\right)$$

Four phases: Selection (UCB traversal) → Expansion (generate next steps) → Simulation (stochastic rollout) → Backpropagation (update $Q$-values). Enables non-linear, multi-path solution discovery; computationally expensive.

### Tool-Augmented Reasoning
{: #tool-reasoning}

**[ReAct](https://arxiv.org/abs/2210.03629)** (Yao et al. 2022)

Interleaves reasoning and action:

$$x \to \text{Thought}_1 \to \text{Action}_1 \to \text{Observation}_1 \to \ldots \to y$$

Hybrid policy that produces both reasoning steps and tool invocations:

$$\pi_\theta(a_t \mid s_t) = \begin{cases} z_t & \text{reasoning step} \\ \mathcal{T}_i(s_t) & \text{tool invocation} \end{cases}$$

**[Toolformer](https://arxiv.org/abs/2302.04761)** (Schick et al. 2023)

Self-supervised tool learning: model generates API call examples where calls improve prediction likelihood, then fine-tunes on these examples. Related: [PAL](https://arxiv.org/abs/2211.10435) (Gao et al. 2022) delegates arithmetic to Python; [Gorilla](https://arxiv.org/abs/2305.15334) (Patil et al. 2023) maps to thousands of APIs.

**Tool-Integrated Reinforcement Learning (TIRL)**

State: $s_t = \{r_1, c_1, o_1, \ldots, r_t, c_t, o_t\}$ where $r_t$ = reasoning, $c_t$ = tool command, $o_t = I(c_t)$ = interpreter output.

Objective:

$$J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^T \gamma^t\, r(s_t, a_t, o_t)\right]$$

Three complementary TIRL systems:

| System | Key idea | Result |
|---|---|---|
| **[ToRL](https://arxiv.org/abs/2503.23383)** (Li et al. 2025) | RL from base model without SFT; GRPO with Python execution | Strong math reasoning without supervised rationales |
| **[ReTool](https://arxiv.org/abs/2504.11536)** (Feng et al. 2025) | Two-phase RL: reason → execute → self-correct | Enables verification through code execution |
| **TIR-Judge** (Xu et al. 2025) | Train LLM judges as evaluation agents using TIRL | Better reward model calibration on tool-use tasks |

### RL-Based Reasoning
{: #rl-reasoning}

**Process vs. Outcome Reward Models** ([Lightman et al. 2023](https://arxiv.org/abs/2305.20050))

| Approach | What is rewarded | Advantage | Disadvantage |
|---|---|---|---|
| **Outcome RM (ORM)** | Final answer correctness only | No step labels needed | Sparse signal; credit assignment hard |
| **Process RM (PRM)** | Each intermediate reasoning step | Dense signal; catches early errors | Requires step-level labels (PRM800K) |

PRMs label intermediate steps; step-level supervision enables precise credit assignment for long reasoning chains.

**[DeepSeek-R1](https://arxiv.org/abs/2501.12948)** (Guo et al. 2025)

Learns reasoning without supervised rationales using outcome-based RL. Composite reward:

$$R(y,z) = \mathbb{1}[\text{correct}(y)] - \lambda \cdot \text{cost}(z)$$

Policy gradient update:

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}\!\left[(R - b)\, \nabla_\theta \log p_\theta(y,z \mid x)\right]$$

Design patterns: (1) reward verifiable outputs only; (2) stage training (cold-start SFT → RL); (3) keep decoding consistent with training; (4) prefer execution over narration; (5) budget thinking via self-consistency.

**WebGPT** ([Nakano et al. 2021](https://arxiv.org/abs/2112.09332))

Early RLHF for tool-augmented reasoning with web browsing. Combines imitation learning on human demonstrations with a learned reward model. Achieved 56% preference over human demonstrations on ELI5.

**Themis — Tool-Augmented Reward Modeling** ([Li et al. 2024](https://arxiv.org/abs/2310.01045))

Reward models call external tools (calculators, translators, search) during preference evaluation:

$$L_\text{total} = L_\text{RM} + \alpha\sum_t\!\left(L_\text{tool}(t) + \beta L_\text{obs}(t)\right) + \omega L_\text{rat}$$

Results: +17.7% accuracy on tool-based datasets, +7.3% on TruthfulQA.

**Budget allocation by system size:**

| Budget | Recommended stack |
|---|---|
| Small | Zero-shot CoT + self-consistency ($K = 5$–$10$) |
| Medium | Few-shot CoT + self-consistency + reflection loops |
| Large | RL-based policy (GRPO/PPO) + PRMs + tool execution + verifier-guided decoding |

### Failure Modes
{: #reasoning-failure}

| Failure | Description | Mitigation |
|---|---|---|
| **Reasoning shortcuts** | Surface-level pattern matching instead of genuine reasoning | Adversarial OOD evaluation; verifier-guided selection |
| **Hallucinated reasoning chains** | Plausible-sounding but false intermediate steps | Process reward models; code execution verification |
| **Sycophancy** | Reasoning degrades when user implies a preferred answer | Diverse prompt ensemble; robust aggregation |
| **Prompt sensitivity** | Quality varies with wording/order of few-shot examples | Temperature-controlled sampling; ensemble multiple prompts |
| **Reward hacking** | RL models exploit reward function weaknesses (format cues, guessable ranges) | Rotate perturbations; adversarial seeds; log chains with rewards |
| **Over-deliberation** | RL reasoners produce unnecessarily long chains | Chain-length penalty; early-stop verifiers; step-count caps |
| **Verification bottleneck** | Scaling process supervision requires expensive step labels | Outcome-only RL; template-based auto-labelling; execution verification |

> **Interview question:** Given a model that gets 72% on GSM8K with standard greedy decoding, what techniques would you apply to push accuracy higher without fine-tuning, and roughly how much gain do you expect from each?
>
> *In order of diminishing returns: (1) Self-consistency with $K=20$ samples — typically +5–8pp on GSM8K by marginalising over reasoning paths. (2) Zero-shot CoT with a better trigger ("Let's think step by step, checking each calculation") — +2–4pp if the base trigger is suboptimal. (3) PAL/code execution: delegate arithmetic to Python rather than asking the model to compute — eliminates arithmetic errors, +3–7pp on problems where the bottleneck is computation not reasoning. (4) ToT with BFS if the task has discrete checkable states — expensive but can recover from wrong early steps. Together these realistically get you from 72% to 85–88% on GSM8K without any fine-tuning.*

---

## Calibration & Uncertainty
{: #calibration}

A well-calibrated model's expressed confidence matches its empirical accuracy: when it says it's 80% confident, it should be correct 80% of the time. Poor calibration is dangerous — overconfident wrong answers are worse than admitted uncertainty.

### Expected Calibration Error
{: #ece}

Partition predictions into $M$ confidence bins. ECE is the weighted average gap between confidence and accuracy:

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left|\text{conf}(B_m) - \text{acc}(B_m)\right|$$

A reliability diagram plots $\text{conf}$ vs. $\text{acc}$ per bin; perfect calibration is the diagonal. Models trained with RLHF often become overconfident (conf > acc) on factual questions.

**Maximum Calibration Error (MCE)** is the worst-case bin deviation — useful for safety-critical applications where any extreme overconfidence is unacceptable.

### Expressing Uncertainty
{: #uncertainty-expression}

LLMs can signal uncertainty through several mechanisms:

| Mechanism | Example | Reliability |
|---|---|---|
| **Verbalised probability** | "I'm about 80% confident that…" | Low without calibration fine-tuning |
| **Linguistic hedges** | "This may be outdated / I'm not certain" | Reasonable proxy when consistently used |
| **Abstention** | "I don't know" / "I can't verify this" | Best option when uncertain; measure abstention rate |
| **Confidence scoring** | Per-token log-probs aggregated to sequence | Available from API; underestimates uncertainty for long outputs |
| **Ensemble variance** | Variance across $K$ self-consistency samples | Strong proxy — high variance → low confidence |

**SelfCheckGPT** (Manakul et al. 2023): sample $K$ completions; semantic similarity between completions via sentence-BERT estimates per-sentence confidence. Sentences where all $K$ completions agree are more likely to be correct.

### Improving Calibration
{: #calibration-improvement}

| Technique | Mechanism | Notes |
|---|---|---|
| **Temperature scaling** | Post-hoc division of logits by $T > 1$ | Cheap; effective for short-form; degrades at high $T$ for long outputs |
| **Platt scaling** | Fit logistic regression to confidence-accuracy pairs | Requires calibration held-out set |
| **Calibration-aware RLHF** | Add ECE penalty to reward function | Trades some performance for calibration |
| **Abstention training** | SFT on examples where model says "I don't know" on uncertain questions | Reduces overconfident wrong answers |
| **RAG for currency** | Retrieve evidence before asserting time-sensitive facts | Reduces knowledge cutoff miscalibration |

> **Interview question:** A deployed LLM assistant has ECE = 0.18 — much worse than the 0.04 you saw in evaluation. What do you investigate?
>
> *Distribution shift is the most likely cause. The evaluation set may be narrower than production queries. Investigate: (1) Compare confidence-accuracy calibration curves broken down by query category — a specific domain (medical, legal, recent events) may be badly miscalibrated. (2) Check if production queries are longer or more open-ended than eval — long-form generation compounds calibration errors. (3) Check if RLHF reward models were trained on queries similar to production — RLHF often induces overconfidence by rewarding confident-sounding answers. Fixes: domain-specific temperature scaling on the worst-performing query categories; add abstention examples for low-confidence domains to the fine-tuning data.*

---

## Instruction Following
{: #instruction-following}

Instruction following measures whether the model satisfies all explicit constraints in the prompt. Failures are often invisible to users who don't re-read the prompt against the output.

### Constraint Types
{: #constraint-types}

| Category | Examples | Typical failure |
|---|---|---|
| **Format** | "Respond as JSON", "use bullet points", "no markdown" | Adds unasked-for markdown; wrong JSON schema |
| **Length** | "Under 100 words", "exactly 5 bullets", "one paragraph" | Exceeds length; pads to hit minimum |
| **Style** | "Formal tone", "explain like I'm 5", "no jargon" | Reverts to default tone mid-response |
| **Content inclusion** | "Include the word 'however'", "mention X" | Omits required keyword |
| **Content exclusion** | "Do not mention competitors", "no code" | Includes forbidden content |
| **Persona** | "You are a pirate", "respond as a Socratic tutor" | Breaks character after a few turns |
| **Ordering** | "First do X, then Y" | Reverses or merges steps |

### IFEval & Rubrics
{: #ifeval}

[IFEval](https://arxiv.org/abs/2311.07911) (Zhou et al. 2023) is the standard benchmark: 541 prompts each with 1–3 verifiable constraints. Evaluation is programmatic — no human raters:

- **Prompt-level strict accuracy**: all constraints satisfied exactly
- **Instruction-level accuracy**: fraction of individual constraints satisfied

State-of-the-art models reach ~85% prompt-level strict accuracy on IFEval; common failures are length constraints and complex compositional instructions.

**Decomposition strategy for complex instructions:** parse the prompt into atomic constraint slots before generation. Chain-of-thought over constraints ("Constraint 1: format=JSON. Constraint 2: length<200 words…") before generating the response improves satisfaction rates.

### Refusal Quality
{: #refusal-quality}

Refusals are a special instruction-following case: the model must decide whether to comply and produce a useful declination if not.

**Good refusal:** specific, explains the constraint, offers alternatives where possible.
**Bad refusal:** vague ("I can't help with that"), refuses safe requests, ignores the user's actual need.

Metrics: refusal precision (of all refusals, what fraction were warranted?) and refusal recall (of all warranted refusals, what fraction were caught?). Over-refusal harms utility; under-refusal harms safety.

> **Interview question:** How would you evaluate instruction following for a model that needs to handle 10-part prompts with mixed format, length, and content constraints?
>
> *Use IFEval-style programmatic evaluation: parse each constraint type into a checker function (regex for keyword inclusion, word-count for length, JSON parser for format). Report per-constraint-type accuracy separately — this reveals where the model systematically fails. For a 10-part prompt, also measure "all-or-nothing" accuracy (all 10 satisfied simultaneously) vs. per-constraint accuracy — the former is usually 30–50% lower because constraint satisfaction events aren't independent. To improve: add constraint enumeration CoT in the system prompt ("Before responding, list all constraints you must satisfy"); evaluate with self-checking ("Does my response satisfy constraint N?") appended to the generation.*

---

## Safety & Alignment
{: #safety}

Safety is the constraint that the model avoids producing harmful outputs — but calibrating how strongly to apply this constraint is itself a quality problem.

### Refusal Calibration
{: #refusal-calibration}

The refusal calibration problem: a model that refuses everything is safe but useless; a model that complies with everything is useful but dangerous. The optimal model traces a Pareto frontier between helpfulness and harmlessness.

**Over-refusal** (false positives): refuses benign requests because surface patterns resemble harmful ones. Common examples: medical information, security research, historical violence, fictional violence.

**Under-refusal** (false negatives): complies with harmful requests because they are phrased indirectly or use jailbreak framing.

Measuring calibration: use a held-out set with ground-truth labels for `{should comply, should refuse}`. Compute precision and recall of refusals. Track separately by harm category (violence, CSAM, privacy, misinformation).

### Constitutional AI & RLAIF
{: #constitutional-ai}

**Constitutional AI** (Bai et al. 2022, Anthropic): The model critiques its own outputs against a written constitution of principles, then revises. No human labellers required for the critique step. Two phases:

1. **SL-CAI**: supervised fine-tuning on (prompt, critique, revision) triples generated by the model itself.
2. **RL-CAI (RLAIF)**: replace human preference labels with AI preference labels (from the same or a larger model judging against the constitution).

This enables scaling safety training without proportionally scaling human annotation.

**RLHF effects on safety:** RLHF shifts the model toward preferred outputs, which can inadvertently train the model to be more sycophantic (agreeing with stated opinions) or more verbose (if raters prefer longer responses). Separate safety reward models from quality reward models to avoid conflating these signals.

**Adversarial red-teaming:** systematically probe for jailbreaks before deployment. Techniques: manual red-teaming, automated adversarial prompting (GCG, AutoDAN), diverse persona attacks. Fix: fine-tune on red-team failures with correct refusals.

### Failure Modes
{: #safety-failure}

| Failure | Description | Mitigation |
|---|---|---|
| **Jailbreaking** | Role-play, indirect framing, or token manipulations bypass safety training | Adversarial fine-tuning; constitutional critique at inference |
| **Sycophancy** | Model agrees with false claims if user expresses strong preference | Calibration-aware RLHF; diverse human rater pool |
| **Prompt injection** | Adversarial content in retrieved documents hijacks instructions | Separate trusted system prompt from untrusted retrieved content |
| **Over-refusal** | Safety training generalises too broadly to benign queries | Precision-recall analysis of refusals; targeted fine-tuning on safe edge cases |
| **Dual-use harm** | Accurate, helpful information that enables harm | Harm-benefit analysis per domain; gating by use-case context |

> **Interview question:** Your model's refusal rate jumped from 3% to 11% after a safety fine-tuning update, but harm metrics didn't improve. What happened and how do you fix it?
>
> *This pattern is classic over-refusal: the fine-tuning updated the decision boundary too broadly, catching benign queries that share surface features with harmful ones (e.g., all chemistry questions refused because some chemistry questions can enable harm). Diagnose: sample the new refusals and manually label — what fraction are actually benign? Break down refusal rate by query category to find which domains over-triggered. Fix: add targeted "safe edge case" examples to the SFT and/or RLHF data — examples of benign chemistry, security, or medical queries where the correct behaviour is to comply. Re-tune the safety classifier threshold on a precision-recall curve; target refusal precision > 90% before deploying.*

---

## Output Format & Coherence
{: #output-quality}

A factually correct, well-reasoned response can still fail if its structure, length, or internal consistency undermines the user's ability to extract value from it.

**Length calibration.** The ideal response length matches task complexity. RLHF-trained models systematically over-generate because human raters often prefer longer responses, even when shorter ones are better. Symptoms: unnecessary preambles ("Great question!"), redundant summaries, padding to meet an implied length expectation.

Evaluation: compare output length distribution to task-optimal length (measured by human rating of trimmed responses). Add length penalty to the reward signal if over-generation is measured.

**Internal consistency.** In long responses, models can contradict earlier statements — claiming $X$ in paragraph 2 and $\neg X$ in paragraph 5. Causes: long context with attention dilution, or the model generating text locally without maintaining a global state of what has been asserted.

Detection: NLI-based contradiction check between sentence pairs within the same response. Mitigation: structured generation with an explicit outline step before prose generation.

**Format adherence.** Format requirements (JSON, Markdown, code blocks) have strict syntactic validity requirements. Failures: unclosed JSON braces, incorrect indentation, mixing prose into structured output.

Mitigation: constrained decoding (token masks that enforce syntactic validity), post-processing validators, or fine-tuning on format-correct examples.

| Coherence dimension | Metric | Fix |
|---|---|---|
| Length calibration | Token count vs task-optimal; rating of trimmed outputs | Length reward signal; explicit length instruction |
| Internal consistency | Contradiction rate (NLI between sentence pairs) | Outline-first generation; structured state tracking |
| Format validity | Parse error rate (JSON/code) | Constrained decoding; post-hoc parser |
| Verbosity | Compression ratio (key info / total tokens) | Conciseness preference in RLHF; explicit budget prompt |
| Coreference clarity | Ambiguous pronoun rate | Entity tracking in structured generation |

---

## Evaluation Methods
{: #evaluation}

### Human Evaluation
{: #human-eval}

The gold standard but expensive and slow. Key design decisions:

**Rating scales.** Likert 1–5 is common but annotation variance is high. Pairwise preference (A vs. B) is more reliable because relative judgements are easier than absolute ones — this is the design used by Chatbot Arena.

**Annotation guidelines.** Underspecified guidelines produce high inter-annotator disagreement. Require: per-dimension ratings (factuality separate from helpfulness separate from safety), concrete rubrics for each level, calibration examples.

**Inter-annotator agreement.** Cohen's κ < 0.4 indicates the task is underspecified or the raters need more calibration. For safety, target κ > 0.7 before trusting results.

**Limitations:** expensive ($10–$100 per query for expert annotation); slow; doesn't scale to continuous deployment; subjective dimensions (style, helpfulness) are hard to standardise.

### LLM-as-Judge
{: #llm-judge}

Using a stronger model (GPT-4, Claude) to evaluate outputs at scale. Formats:

| Format | Use case | Bias risk |
|---|---|---|
| **Pairwise (A vs. B)** | Ranking, preference learning | Position bias (prefers first or second); verbosity bias |
| **Single-score with rubric** | Absolute quality measurement | Anchor bias; inconsistent scale use |
| **Reference-free scoring** | When no ground truth exists | Style bias toward the judge's own outputs |
| **Chain-of-thought rationale** | When explanation is needed for trust | Can rationalise incorrect scores post-hoc |

**[G-Eval](https://arxiv.org/abs/2303.16634)** (Liu et al. 2023): structured rubric prompts with step-by-step evaluation; correlates better with human judgement than simple scoring.

**[MT-Bench](https://arxiv.org/abs/2306.05685)** (Zheng et al. 2023): 80 multi-turn questions across 8 categories; GPT-4 scores each turn; strong correlation with Chatbot Arena human rankings.

**Key biases to control for:** position bias (randomise A/B order; take symmetric average), verbosity bias (longer responses rated higher regardless of quality), self-enhancement bias (model prefers its own style).

### Reference-Based Metrics
{: #reference-metrics}

Metrics that compare outputs to human-written references:

| Metric | What it measures | Limitations |
|---|---|---|
| **BLEU** | N-gram precision against references | Misses semantically equivalent paraphrases; rewards short outputs |
| **ROUGE-L** | Longest common subsequence recall | Better for summarisation; still surface-level |
| **[BERTScore](https://arxiv.org/abs/1904.09675)** | Contextual embedding similarity (token-level F1) | Better than n-gram; expensive; sensitive to reference quality |
| **[BLEURT](https://arxiv.org/abs/2004.04696)** | Fine-tuned BERT on human rating data | Best correlation with humans among reference-based metrics |
| **chrF** | Character n-gram F-score | More robust for morphologically rich languages |

General rule: reference-based metrics are unreliable when the reference set is small or when valid paraphrases exist. Use them as supplementary signals, not primary evaluation.

### Benchmarks
{: #benchmarks}

**Knowledge and reasoning:**

| Benchmark | Focus | Notes |
|---|---|---|
| **[MMLU](https://arxiv.org/abs/2009.03300)** | 57-subject knowledge + reasoning | Mixes knowledge retrieval with reasoning; not a pure reasoning benchmark |
| **[GPQA](https://arxiv.org/abs/2311.12022)** | Graduate-level expert Q&A | Requires true domain expertise; hard to contaminate |
| **[AGIEval](https://arxiv.org/abs/2304.06364)** | Real-world cognitive exams (SAT, GRE, LSAT) | Strong gap between model and human performance |
| **[BIG-bench Hard (BBH)](https://arxiv.org/abs/2210.09261)** | 23 hardest BIG-bench tasks | CoT strongly improves performance; reveals non-smooth scaling |

**Mathematical reasoning:**

| Benchmark | Focus | Notes |
|---|---|---|
| **GSM8K** | Grade-school multi-step arithmetic | ~8,500 problems; measures procedural fluency |
| **[MATH](https://arxiv.org/abs/2103.03874)** | Competition-level math | 7 difficulty levels; tests reasoning depth |
| **AIME** | Mathematical olympiad | 450 problems; integer answers 0–999; minimal knowledge retrieval |
| **[ARC-AGI-2](https://arcprize.org/)** | Grid-based abstraction and pattern generalisation | Tests reasoning beyond pattern completion |

**Factuality:**

| Benchmark | Focus | Notes |
|---|---|---|
| **[TruthfulQA](https://arxiv.org/abs/2109.07958)** | Truthfulness | Probes common human misconceptions |
| **[SimpleQA](https://arxiv.org/abs/2411.07905)** | Short-form factoid accuracy | 4,326 questions; includes abstention behaviour |
| **[FreshQA](https://arxiv.org/abs/2310.03214)** | Time-sensitive Q&A | Tracks knowledge decay; tests temporal factuality |
| **HaluEval** | Hallucination detection | Large-scale; covers multiple domains |
| **[FEVER](https://arxiv.org/abs/1803.05355)** | Claim verification with evidence | Wikipedia-based; Supported / Refuted / Not Enough Info |

**Code:**

| Benchmark | Focus | Notes |
|---|---|---|
| **[HumanEval](https://arxiv.org/abs/2107.03374)** | Python function completion | 164 problems; pass@k evaluation |
| **[MBPP](https://arxiv.org/abs/2108.07732)** | Mostly Basic Programming Problems | 374 problems; broader coverage than HumanEval |
| **SWE-bench** | Real GitHub issues | End-to-end software engineering; harder than function completion |

**Instruction following:**

| Benchmark | Focus | Notes |
|---|---|---|
| **[IFEval](https://arxiv.org/abs/2311.07911)** | Verifiable constraint satisfaction | Programmatic; 541 prompts; strict and lenient metrics |
| **[MT-Bench](https://arxiv.org/abs/2306.05685)** | Multi-turn conversation quality | GPT-4 judge; 8 categories |
| **[Chatbot Arena / LMSYS](https://lmsys.org/blog/2023-05-03-arena/)** | Human pairwise preference at scale | ELO ranking; real user queries; gold standard for overall quality |

**Holistic evaluation:**

**[HELM](https://arxiv.org/abs/2211.09110)** (Holistic Evaluation of Language Models) — meta-evaluation across accuracy, robustness, calibration, toxicity, fairness, efficiency. Reasoning performance varies significantly by model scale and training regime.

> **Interview question:** You need to compare two LLM checkpoints for production deployment. What evaluation suite would you run and in what order?
>
> *Run in order of cost: (1) Automated benchmarks first (IFEval, MMLU, GSM8K, HumanEval, TruthfulQA) — cheap, fast, catches clear regressions. (2) LLM-as-judge on a representative sample of production queries (MT-Bench style with GPT-4 pairwise) — tests quality on the actual distribution. (3) Human evaluation on a stratified sample covering edge cases and safety-critical queries — expensive but authoritative for high-stakes decisions. (4) Shadow traffic: run both checkpoints on live traffic for 24–48 hours; measure real user engagement signals (thumbs up/down, session length, follow-up questions). Decision rule: new checkpoint must not regress on safety metrics at all; can accept ≤2% regression on benchmark tasks if production metrics improve.*

---

## Also Read

**[Context Engineering for LLMs](/blogs/context-engineering/)** — retrieval, RAG, memory systems, and the full lifecycle of context management for LLM systems.

**[Agentic System Design](/blogs/agentic-system-design/)** — production system design for LLM agents: tool use, memory, multi-agent orchestration, evaluation, and operations.

**[Fine-tuning LLMs](/blogs/finetuning/)** — RLHF, DPO, GRPO, and reward modelling — the training-time side of aligning models for quality.
