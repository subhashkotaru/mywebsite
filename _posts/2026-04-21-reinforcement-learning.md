---
title: "Reinforcement Learning"
date: 2026-04-21
description: "A comprehensive treatment of reinforcement learning from bandits and dynamic programming through deep RL, policy gradients, model-based methods, multi-agent RL, and RLHF for large language models."
tags: [rl, reinforcement-learning, deep-learning]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#foundations">Foundations</a>
      <ul class="post-toc-sublist">
        <li><a href="#mdp">MDP Formalism</a></li>
        <li><a href="#bellman">Bellman Equations</a></li>
        <li><a href="#value-functions">Value Functions</a></li>
        <li><a href="#policy-types">Policy Types</a></li>
      </ul>
    </li>
    <li><a href="#bandits">Multi-Armed Bandits</a>
      <ul class="post-toc-sublist">
        <li><a href="#explore-exploit">Explore-Exploit Tradeoff</a></li>
        <li><a href="#bandit-algorithms">Bandit Algorithms</a></li>
        <li><a href="#regret">Regret</a></li>
        <li><a href="#contextual-bandits">Contextual Bandits</a></li>
      </ul>
    </li>
    <li><a href="#dynamic-programming">Dynamic Programming</a>
      <ul class="post-toc-sublist">
        <li><a href="#policy-evaluation">Policy Evaluation</a></li>
        <li><a href="#policy-iteration">Policy Iteration</a></li>
        <li><a href="#value-iteration">Value Iteration</a></li>
        <li><a href="#dp-convergence">Convergence Guarantees</a></li>
      </ul>
    </li>
    <li><a href="#model-free-prediction">Model-Free Prediction</a>
      <ul class="post-toc-sublist">
        <li><a href="#monte-carlo">Monte Carlo Methods</a></li>
        <li><a href="#td">Temporal Difference Learning</a></li>
        <li><a href="#n-step">n-Step Returns</a></li>
      </ul>
    </li>
    <li><a href="#model-free-control">Model-Free Control</a>
      <ul class="post-toc-sublist">
        <li><a href="#sarsa">SARSA</a></li>
        <li><a href="#q-learning">Q-Learning</a></li>
        <li><a href="#double-q">Double Q-Learning</a></li>
        <li><a href="#expected-sarsa">Expected SARSA</a></li>
      </ul>
    </li>
    <li><a href="#function-approximation">Function Approximation</a>
      <ul class="post-toc-sublist">
        <li><a href="#linear-fa">Linear Approximation</a></li>
        <li><a href="#semi-gradient">Semi-Gradient Methods</a></li>
        <li><a href="#deadly-triad">Deadly Triad</a></li>
      </ul>
    </li>
    <li><a href="#dqn">Deep Q-Networks (DQN)</a>
      <ul class="post-toc-sublist">
        <li><a href="#experience-replay">Experience Replay</a></li>
        <li><a href="#target-network">Target Network</a></li>
        <li><a href="#dqn-extensions">DQN Extensions</a></li>
      </ul>
    </li>
    <li><a href="#policy-gradients">Policy Gradient Methods</a>
      <ul class="post-toc-sublist">
        <li><a href="#reinforce">REINFORCE</a></li>
        <li><a href="#pg-theorem">Policy Gradient Theorem</a></li>
        <li><a href="#actor-critic">Actor-Critic</a></li>
      </ul>
    </li>
    <li><a href="#advanced-policy-opt">Advanced Policy Optimization</a>
      <ul class="post-toc-sublist">
        <li><a href="#trpo">TRPO</a></li>
        <li><a href="#ppo">PPO</a></li>
        <li><a href="#sac">SAC</a></li>
      </ul>
    </li>
    <li><a href="#model-based">Model-Based RL</a>
      <ul class="post-toc-sublist">
        <li><a href="#dyna-q">Dyna-Q</a></li>
        <li><a href="#world-models">World Models</a></li>
        <li><a href="#alphazero">AlphaZero / MCTS</a></li>
      </ul>
    </li>
    <li><a href="#marl">Multi-Agent RL</a></li>
    <li><a href="#rl-llms">RL for LLMs</a>
      <ul class="post-toc-sublist">
        <li><a href="#rlhf">RLHF Pipeline</a></li>
        <li><a href="#reward-models">Reward Models</a></li>
        <li><a href="#ppo-llms">PPO for LLMs</a></li>
        <li><a href="#alternatives">GRPO and DPO</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## Overview
{: #overview}

Reinforcement learning (RL) is the study of how an agent learns to make sequences of decisions in order to maximise cumulative reward. Unlike supervised learning, there are no labelled input-output pairs — the agent must discover which actions lead to good outcomes through interaction with an environment. Unlike unsupervised learning, there is a clear objective: the reward signal.

RL formalises a wide class of sequential decision problems: game playing, robotics, recommendation systems, drug discovery, and now, the alignment of large language models. Its mathematical foundations draw from dynamic programming (Bellman, 1957), statistical learning theory, and stochastic approximation.

<div class="post-flow post-flow--horizontal" role="group" aria-label="RL paradigms">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Model-Free — learn directly from experience</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Model-Based — learn a model, plan within it</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Hybrid — model-free control with model-based planning</span></li>
  </ol>
</div>

The central tension in RL is the **explore-exploit tradeoff**: the agent must try new actions to discover better strategies (exploration) while also taking actions it already knows to be good (exploitation). Every major algorithm in RL can be understood as a different way of resolving this tension, handling the credit assignment problem, and scaling to high-dimensional state and action spaces.

---

## Foundations
{: #foundations}

### MDP Formalism
{: #mdp}

A **Markov Decision Process (MDP)** is the standard mathematical framework for RL. It is a tuple $$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$:

| Symbol | Name | Description |
|---|---|---|
| **S** | State space | Set of all possible states the environment can be in |
| **A** | Action space | Set of all actions the agent can take |
| P(s' \| s, a) | Transition function | Probability of transitioning to state s' after taking action a in state s |
| R(s, a, s') | Reward function | Expected reward received after transition (s, a, s') |
| γ ∈ [0, 1) | Discount factor | Degree to which future rewards are discounted relative to immediate rewards |

The **Markov property** is the key assumption: the next state depends only on the current state and action, not on the full history. Formally:

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} \mid s_t, a_t)$$

This assumption is satisfied exactly when the state representation is fully informative. When the agent can only observe partial information about the true state (e.g., a robot that cannot see behind itself), the problem becomes a **Partially Observable MDP (POMDP)**, which is significantly harder.

The agent interacts with the MDP to produce a **trajectory**:

$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$$

The **return** $$G_t$$ is the discounted sum of future rewards from time $$t$$:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \ldots$$

The discount factor $$\gamma$$ serves two purposes: mathematical convenience (ensures the infinite sum converges) and a preference for sooner rewards over later ones. When $$\gamma = 0$$ the agent is purely myopic; as $$\gamma \to 1$$ the agent becomes fully far-sighted.

> **Interview question:** What is the Markov property and why is it fundamental to RL?
>
> **Answer:** The Markov property states that the future is conditionally independent of the past given the present state: $$P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \ldots)$$. It is fundamental because it allows the entire history of an episode to be compressed into a single state vector without loss of predictive power. This makes it possible to define value functions over states (rather than histories) and to solve the resulting equations efficiently with dynamic programming or stochastic approximation. Without the Markov property, the optimal policy would in general depend on the full history, making the problem intractable. In practice, we often augment the state representation (e.g., stacking multiple frames in Atari) to approximate the Markov property even when raw observations are not Markovian.

### Bellman Equations
{: #bellman}

The **Bellman expectation equation** expresses the value of a state under a policy $$\pi$$ as the immediate reward plus the discounted value of the next state:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

Similarly, for the action-value function:

$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a') \right]$$

These equations express a **self-consistency condition**: the value of a state is defined in terms of the values of its successor states. They can be written in compact matrix form:

$$\mathbf{v}^\pi = \mathbf{r}^\pi + \gamma P^\pi \mathbf{v}^\pi \implies \mathbf{v}^\pi = (I - \gamma P^\pi)^{-1} \mathbf{r}^\pi$$

The matrix inversion is exact but costs $$O(\vert \mathcal{S}\vert ^3)$$ — practical only for small state spaces. Iterative methods are used in practice.

The **Bellman optimality equations** characterise the optimal value functions $$V^*$$ and $$Q^*$$:

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$

$$Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \right]$$

The optimal policy is greedy with respect to $$Q^*$$:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

The Bellman operators $$\mathcal{T}^\pi$$ (expectation) and $$\mathcal{T}^*$$ (optimality) are **contraction mappings** with modulus $$\gamma$$ under the $$\ell_\infty$$ norm — repeated application converges to the unique fixed point.

> **Interview question:** What is the Bellman optimality equation and why does it have a unique solution?
>
> **Answer:** The Bellman optimality equation expresses V<sup>*</sup>(s) as the maximum over actions of the expected discounted return. It is a fixed-point equation: V<sup>*</sup> = T<sup>*</sup> V<sup>*</sup>. The Bellman optimality operator T<sup>*</sup> is a γ-contraction in the ℓ<sub>∞</sub> norm — that is, ‖T<sup>*</sup>V − T<sup>*</sup>U‖<sub>∞</sub> ≤ γ ‖V − U‖<sub>∞</sub> for all V, U. By the Banach fixed-point theorem, any γ-contraction on a complete metric space has a unique fixed point, and iterative application of the operator converges geometrically to it. Since γ < 1, the contraction factor is strictly less than 1, guaranteeing both existence and uniqueness of V<sup>*</sup>.

### Value Functions
{: #value-functions}

The **state-value function** $$V^\pi(s)$$ is the expected return starting from state $$s$$ and following policy $$\pi$$ thereafter:

$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \mid s_t = s \right]$$

The **action-value function** (Q-function) $$Q^\pi(s, a)$$ is the expected return starting from state $$s$$, taking action $$a$$, then following $$\pi$$:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]$$

The relationship between $$V$$ and $$Q$$ is:

$$V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a)$$

$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

The **advantage function** $$A^\pi(s, a)$$ measures how much better action $$a$$ is compared to the average action under $$\pi$$:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

The advantage function is zero for actions that are exactly average, positive for better-than-average actions, and negative for worse-than-average actions. It plays a central role in policy gradient methods and actor-critic algorithms.

### Policy Types
{: #policy-types}

A **policy** $$\pi$$ maps states to a distribution over actions. There are several important distinctions:

**Deterministic vs stochastic:**
- Deterministic: $$a = \pi(s)$$ — a single action for each state
- Stochastic: $$a \sim \pi(\cdot \mid s)$$ — a distribution over actions given the state

Stochastic policies are necessary for exploration and for games with imperfect information where a mixed strategy is optimal (e.g., rock-paper-scissors). Deterministic policies suffice for fully observable MDPs after learning is complete.

**On-policy vs off-policy:**
- On-policy: the agent improves the policy it is currently using to generate data (SARSA, PPO)
- Off-policy: the agent learns about a target policy using data generated by a different behaviour policy (Q-Learning, DQN)

| | On-Policy | Off-Policy |
|---|---|---|
| **Data reuse** | Cannot reuse old data — must generate fresh experience with current policy | Can learn from any data, including human demonstrations or stored replay buffers |
| **Stability** | More stable; no distribution mismatch between behaviour and target | Can diverge if importance weights are too large (high variance) |
| **Examples** | SARSA, A2C, PPO | Q-learning, DQN, SAC |
| **Sample efficiency** | Lower — discards experience from old policies | Higher — replays stored transitions many times |

> **Interview question:** What is the difference between on-policy and off-policy learning, and when would you choose each?
>
> **Answer:** In on-policy learning, the agent evaluates and improves the same policy that it uses to collect data. In off-policy learning, the agent learns about a target policy from data collected by a potentially different behaviour policy. Off-policy methods are more sample-efficient because they can replay stored experience and learn from demonstrations. However, they introduce a distribution mismatch between where data was collected and where the target policy operates, which can cause instability or divergence — particularly with function approximation (the deadly triad). On-policy methods avoid this mismatch at the cost of discarding old experience. In practice, off-policy methods like DQN and SAC are preferred when sample efficiency is critical (e.g., real-world robotics where data collection is expensive), while on-policy methods like PPO are preferred when stability matters and environment interactions are cheap (e.g., simulated environments).

---

## Multi-Armed Bandits
{: #bandits}

### Explore-Exploit Tradeoff
{: #explore-exploit}

A **multi-armed bandit** is a simplified RL problem with a single state and $$K$$ actions (arms). Each arm $$a$$ has an unknown reward distribution with mean $$\mu_a$$. The agent must choose actions sequentially to maximise total reward, but it does not know $$\mu_a$$ a priori.

The problem captures the fundamental **explore-exploit tradeoff**: the agent can either exploit its current best estimate (take the action it thinks is best) or explore to gather more information about uncertain arms. Pure exploitation risks getting stuck on a suboptimal arm; pure exploration wastes reward on actions already known to be poor.

The optimal arm is $$a^* = \arg\max_a \mu_a$$. The goal is to maximise cumulative reward, or equivalently, to minimise **regret** — the difference between what the agent received and what the optimal agent would have received.

### Bandit Algorithms
{: #bandit-algorithms}

**epsilon-Greedy**

The simplest exploration strategy: with probability $$\varepsilon$$ take a uniformly random action (explore), with probability $$1 - \varepsilon$$ take the greedy action (exploit):

$$a_t = \begin{cases} \text{random arm} & \text{with probability } \varepsilon \\ \arg\max_a \hat{\mu}_a & \text{with probability } 1 - \varepsilon \end{cases}$$

The sample mean estimate $$\hat{\mu}_a$$ is updated incrementally:

$$\hat{\mu}_a \leftarrow \hat{\mu}_a + \frac{1}{n_a} (r - \hat{\mu}_a)$$

where $$n_a$$ is the number of times arm $$a$$ has been pulled. Decaying $$\varepsilon$$ (e.g., $$\varepsilon_t = 1/t$$) achieves logarithmic regret asymptotically but requires tuning.

**Upper Confidence Bound (UCB)**

UCB algorithms select the arm with the highest upper confidence bound on the mean reward, implementing the "optimism in the face of uncertainty" principle:

$$a_t = \arg\max_a \left[ \hat{\mu}_a + c \sqrt{\frac{\ln t}{n_a}} \right]$$

The confidence bonus $$c \sqrt{\ln t / n_a}$$ is large when an arm has been pulled few times (high uncertainty) and shrinks as the arm is sampled more. **UCB1** with $$c = \sqrt{2}$$ achieves the **Lai-Robbins lower bound** — $$O(\ln T)$$ regret — which is optimal up to constants.

**Thompson Sampling**

Thompson Sampling is a Bayesian approach that maintains a posterior distribution over each arm's mean reward and samples an action proportionally to the probability that it is optimal:

```
Initialise: Beta(1, 1) prior for each arm (for Bernoulli rewards)

For t = 1, 2, ...:
    For each arm a:
        Sample θ_a ~ Beta(S_a + 1, F_a + 1)   # S_a = successes, F_a = failures
    Take action: a_t = argmax_a θ_a
    Observe reward r_t
    Update: if r_t = 1: S_{a_t} += 1 else F_{a_t} += 1
```

Thompson Sampling achieves logarithmic regret and often outperforms UCB empirically. It extends naturally to more complex reward distributions by using conjugate priors or approximate posterior inference.

**Pros and Cons — Bandit Algorithms**

| | epsilon-Greedy | UCB1 | Thompson Sampling |
|---|---|---|---|
| **Pros** | Simple, no distributional assumptions, easy to implement | No hyperparameter tuning (with UCB1); deterministic action selection; near-optimal regret bounds | Bayesian; naturally balances exploration; often best empirical performance; handles batched feedback well |
| **Cons** | Requires tuning ε; explores uniformly regardless of arm quality | Assumes bounded rewards; can over-explore in early stages; not directly Bayesian | Requires specifying a prior; computationally expensive with non-conjugate likelihoods; stochastic (different runs differ) |

> **Interview question:** Why does UCB explore less-visited arms, and what is the intuition behind the confidence bonus?
>
> **Answer:** The UCB confidence bonus $$c\sqrt{\ln t / n_a}$$ implements the "optimism in the face of uncertainty" principle. An arm that has been pulled few times ($$n_a$$ small) has a high confidence bonus, meaning UCB treats it as potentially much better than its current estimated mean — uncertainty is interpreted as opportunity. An arm that has been pulled many times ($$n_a$$ large) has a small bonus, so it is only selected if its true mean is genuinely high. The $$\ln t$$ in the numerator ensures that the confidence bounds shrink slowly enough that even suboptimal arms are revisited occasionally — otherwise the algorithm might commit to a suboptimal arm if it happened to look good in the first few pulls. This design achieves logarithmic cumulative regret: the total exploration over T rounds is $$O(\ln T)$$, which matches the theoretical lower bound.

### Regret
{: #regret}

**Cumulative regret** is the total reward gap between the optimal policy and the agent over $$T$$ rounds:

$$R_T = \sum_{t=1}^{T} \left( \mu_{a^*} - \mu_{a_t} \right) = T \mu_{a^*} - \sum_{t=1}^{T} \mu_{a_t}$$

**Simple (instantaneous) regret** measures only the final recommendation quality, not cumulative performance:

$$r_T = \mu_{a^*} - \mu_{\hat{a}_T}$$

where $$\hat{a}_T$$ is the arm recommended after $$T$$ rounds. Simple regret is appropriate when the exploration phase is separate from the deployment phase (pure exploration).

The **Lai-Robbins lower bound** shows that any consistent algorithm must suffer at least:

$$\liminf_{T \to \infty} \frac{R_T}{\ln T} \geq \sum_{a: \mu_a < \mu_{a^*}} \frac{\mu_{a^*} - \mu_a}{\text{KL}(\mu_a \Vert \mu_{a^*})}$$

This is a fundamental lower bound — $$O(\ln T)$$ cumulative regret is the best achievable. UCB1 and Thompson Sampling both achieve this bound.

### Contextual Bandits
{: #contextual-bandits}

**Contextual bandits** extend the standard bandit by providing a context vector $$x_t$$ at each round. The agent must choose an arm, but the optimal arm depends on the context:

$$a_t = \arg\max_a \mu_a(x_t)$$

This is the setting of personalised recommendation: the context is user features, the arms are items, and the reward is a click or purchase signal. **LinUCB** assumes a linear reward model $$\mu_a(x) = x^\top \theta_a$$ and constructs confidence ellipsoids:

$$a_t = \arg\max_a \left[ x_t^\top \hat{\theta}_a + \alpha \sqrt{x_t^\top A_a^{-1} x_t} \right]$$

where $$A_a = \sum_\tau x_\tau x_\tau^\top + I$$ is the regularised design matrix. The second term is the uncertainty in the direction of $$x_t$$.

**Key papers:**
- [Auer et al. (2002) — Finite-time Analysis of the Multiarmed Bandit Problem](https://link.springer.com/article/10.1023/A:1013689704352) — UCB1 with finite-time regret bounds
- [Thompson (1933) — On the likelihood that one unknown probability exceeds another](https://academic.oup.com/biomet/article-abstract/25/3-4/285/253244) — original Thompson Sampling
- [Li et al. (2010) — LinUCB](https://arxiv.org/abs/1003.0146) — contextual bandit for news recommendation

---

## Dynamic Programming
{: #dynamic-programming}

Dynamic programming (DP) methods assume complete knowledge of the MDP: the transition function $$P$$ and reward function $$R$$ are known. They use the Bellman equations as update rules to compute value functions exactly.

### Policy Evaluation
{: #policy-evaluation}

**Policy evaluation** computes $$V^\pi$$ for a fixed policy $$\pi$$ by iterating the Bellman expectation operator:

$$V_{k+1}(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V_k(s') \right]$$

```
Input: MDP (S, A, P, R, γ), policy π, threshold θ
Initialise: V(s) = 0 for all s ∈ S

Repeat:
    Δ ← 0
    For each s ∈ S:
        v ← V(s)
        V(s) ← Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
        Δ ← max(Δ, |v - V(s)|)
Until Δ < θ

Return V
```

Since $$\mathcal{T}^\pi$$ is a $$\gamma$$-contraction, iterating converges geometrically: $$\Vert V_k - V^\pi\Vert_\infty \leq \gamma^k \Vert V_0 - V^\pi\Vert_\infty$$.

### Policy Iteration
{: #policy-iteration}

**Policy iteration** alternates between policy evaluation (compute $$V^\pi$$) and policy improvement (make policy greedy with respect to $$V^\pi$$):

```
Initialise: π(s) = random for all s ∈ S

Repeat:
    # Policy Evaluation
    V ← PolicyEvaluation(π)

    # Policy Improvement
    policy_stable ← True
    For each s ∈ S:
        old_action ← π(s)
        π(s) ← argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
        If old_action ≠ π(s): policy_stable ← False

Until policy_stable

Return π, V
```

**Policy improvement theorem**: if $$\pi'$$ is greedy with respect to $$V^\pi$$, then $$V^{\pi'}(s) \geq V^\pi(s)$$ for all $$s$$. Policy iteration converges in a finite number of iterations (at most $$\vert \mathcal{A}\vert ^{\vert \mathcal{S}\vert }$$ policies) to $$\pi^*$$.

### Value Iteration
{: #value-iteration}

**Value iteration** applies the Bellman optimality operator directly, bypassing explicit policy evaluation:

$$V_{k+1}(s) \leftarrow \max_a \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V_k(s') \right]$$

```
Initialise: V(s) = 0 for all s ∈ S

Repeat:
    Δ ← 0
    For each s ∈ S:
        v ← V(s)
        V(s) ← max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
        Δ ← max(Δ, |v - V(s)|)
Until Δ < θ

π(s) ← argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
Return π, V
```

Value iteration converges to $$V^*$$ in the limit. In practice, it is stopped when $$\Vert V_{k+1} - V_k\Vert_\infty < \theta (1-\gamma)/\gamma$$, which bounds the error in $$V^*$$ to $$\theta$$.

### Convergence Guarantees
{: #dp-convergence}

| Method | Convergence | Complexity per Sweep |
|---|---|---|
| Policy Evaluation | Geometric: γᵏ factor | O(\|S\|² \|A\|) |
| Policy Iteration | Finite: ≤ \|A\|^\|S\| iterations | O(\|S\|² \|A\|) per evaluation + improvement |
| Value Iteration | Geometric: γᵏ factor | O(\|S\|² \|A\|) |

In practice, **modified policy iteration** interpolates between the two: perform $$m$$ steps of policy evaluation (not until convergence) before improving. Setting $$m = 1$$ recovers value iteration; $$m = \infty$$ recovers policy iteration.

**Pros and Cons — Dynamic Programming**

| | Pros | Cons |
|---|---|---|
| **Exactness** | Computes exact V\* and π\* with guaranteed convergence | Requires complete knowledge of P and R — unavailable in most real problems |
| **Efficiency** | Polynomial in \|S\| and \|A\| — dramatically better than exhaustive search | State space must be enumerable; infeasible for continuous or high-dimensional states |
| **Foundation** | Every model-free algorithm can be understood as an approximate version of DP | Curse of dimensionality: \|S\| grows exponentially with state dimension |

> **Interview question:** What is the difference between policy iteration and value iteration, and which is faster in practice?
>
> **Answer:** Policy iteration alternates between full policy evaluation (iterate Bellman expectation until convergence) and a single greedy improvement step. Value iteration applies the Bellman optimality operator directly, which is equivalent to doing one step of policy evaluation followed by immediate improvement, repeated. Policy iteration converges in fewer outer iterations because each evaluation step moves to the exact Vπ, but each evaluation step itself takes many inner iterations. Value iteration trades fewer inner iterations per outer step for more outer steps. Empirically, value iteration is often faster in wall-clock time for small state spaces because the inner loop is cheap. For larger spaces, modified policy iteration (a few evaluation steps before improvement) typically performs best. The theoretical complexity of policy iteration is polynomial in $$\vert\mathcal{S}\vert$$ and $$\vert\mathcal{A}\vert$$; value iteration converges in $$O(1/(1-\gamma))$$ sweeps to $$\varepsilon$$-optimal.

---

## Model-Free Prediction
{: #model-free-prediction}

Model-free prediction estimates the value function $$V^\pi$$ from experience (trajectories sampled under $$\pi$$), without knowing $$P$$ or $$R$$.

### Monte Carlo Methods
{: #monte-carlo}

**Monte Carlo (MC)** methods estimate $$V^\pi(s)$$ by averaging the observed returns following visits to $$s$$:

$$V(s) \leftarrow V(s) + \alpha \left[ G_t - V(s) \right]$$

**First-visit MC**: update $$V(s)$$ only on the first time state $$s$$ is visited in an episode.

**Every-visit MC**: update $$V(s)$$ every time state $$s$$ is visited.

Both converge to $$V^\pi(s)$$ as the number of episodes grows. First-visit MC produces unbiased estimates; every-visit MC also converges and has lower variance in practice.

```
Initialise: V(s) = 0, Returns(s) = [] for all s

For each episode:
    Generate trajectory τ = (s_0, a_0, r_0, ..., s_T)
    G ← 0
    For t = T-1, T-2, ..., 0:
        G ← γG + r_{t+1}
        If s_t not in {s_0, ..., s_{t-1}}:  # First-visit check
            Append G to Returns(s_t)
            V(s_t) ← mean(Returns(s_t))
```

**Properties:**
- Unbiased: the expected value of $$G_t$$ is exactly $$V^\pi(s_t)$$
- High variance: $$G_t$$ is a sum of many random variables, so variance is high
- Only applicable to episodic tasks (must wait for episode to end)

### Temporal Difference Learning
{: #td}

**TD(0)** updates the value estimate at each step using the **TD target** $$r + \gamma V(s')$$, a single-step bootstrap estimate:

$$V(s_t) \leftarrow V(s_t) + \alpha \left[ \underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{TD target}} - V(s_t) \right]$$

The quantity $$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$ is the **TD error** — the difference between the estimated value and the bootstrapped estimate.

TD(0) is **biased** (because $$V(s')$$ is itself an estimate) but has **lower variance** than MC because it does not sum many random rewards. TD typically learns faster than MC in practice.

**TD($$\lambda$$)** interpolates between TD(0) and MC using **eligibility traces**. The $$\lambda$$-return is a weighted mixture of all $$n$$-step returns:

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

where $$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1} + \gamma^n V(s_{t+n})$$ is the $$n$$-step return.

The **eligibility trace** $$e_t(s)$$ tracks which states are responsible for current prediction errors:

$$e_t(s) = \gamma \lambda \, e_{t-1}(s) + \mathbf{1}[s_t = s]$$

The update uses the trace to assign credit to recently visited states:

$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s) \quad \forall s$$

When $$\lambda = 0$$, TD($$\lambda$$) reduces to TD(0). When $$\lambda = 1$$ and episodes are finite, it is equivalent to every-visit MC.

**Pros and Cons — MC vs TD**

| | Monte Carlo | TD(0) | TD(λ) |
|---|---|---|---|
| **Bias** | Zero bias — uses actual returns | Biased — bootstraps off current estimates | Interpolates; lower bias for larger λ |
| **Variance** | High — return is a sum of many random variables | Low — single-step TD error | Intermediate, controlled by λ |
| **Online learning** | Requires complete episodes | Online — updates at every step | Online with eligibility traces |
| **Convergence** | Converges to Vπ (tabular, decaying α) | Converges to Vπ (tabular, decaying α) | Converges for all λ ∈ [0,1] |

> **Interview question:** What is the TD error and what role does it play in the brain?
>
> **Answer:** The TD error $$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$ is the difference between the observed immediate reward plus the discounted next-state value and the current value estimate. It measures the surprise or prediction error at each step. Neuroscience research by Schultz, Dayan, and Montague showed that dopaminergic neurons in the basal ganglia fire in a pattern consistent with the TD error: they fire strongly for unexpected rewards (positive $$\delta$$), remain silent for fully predicted rewards ($$\delta \approx 0$$), and dip below baseline for expected rewards that do not arrive (negative $$\delta$$). This provided strong evidence that the brain implements something like TD learning for reward prediction, and it remains one of the most striking connections between computational RL and neuroscience.

### n-Step Returns
{: #n-step}

The $$n$$-step return uses $$n$$ actual rewards before bootstrapping:

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1} + \gamma^n V(s_{t+n})$$

The $$n$$-step TD update is:

$$V(s_t) \leftarrow V(s_t) + \alpha \left[ G_t^{(n)} - V(s_t) \right]$$

$$n = 1$$ gives TD(0); $$n = \infty$$ gives MC. Intermediate values of $$n$$ often perform best in practice. The TD($$\lambda$$) return is a geometric mixture of all $$n$$-step returns and typically outperforms any fixed $$n$$.

---

## Model-Free Control
{: #model-free-control}

Model-free control optimises the policy without knowing the MDP model. The key change from prediction is that we must ensure **exploration**: the agent must try all state-action pairs to find the optimal policy.

### SARSA
{: #sarsa}

**SARSA** (State-Action-Reward-State-Action) is an on-policy TD control algorithm that learns $$Q^\pi$$ for the current policy and improves it greedily:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$

The name comes from the five quantities used in each update: $$(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$$.

```
Initialise: Q(s, a) = 0 for all s, a; ε-greedy policy from Q

For each episode:
    s ← initial state
    a ← ε-greedy(Q, s)
    While s is not terminal:
        Take action a, observe r, s'
        a' ← ε-greedy(Q, s')
        Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]
        s ← s'; a ← a'
```

Because SARSA is on-policy, it learns $$Q^\pi$$ for the $$\varepsilon$$-greedy policy — including the cost of exploration. This makes SARSA safer in environments where exploratory actions are dangerous (e.g., the cliff walking problem where SARSA learns a safer path than Q-Learning).

### Q-Learning
{: #q-learning}

**Q-Learning** is an off-policy TD control algorithm that directly estimates $$Q^*$$ regardless of the behaviour policy:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

The key difference from SARSA: the TD target uses $$\max_{a'} Q(s_{t+1}, a')$$ — the value of the best action from $$s_{t+1}$$ — rather than the value of the action actually taken. This makes Q-Learning off-policy: it learns the optimal Q-function even while behaving $$\varepsilon$$-greedily.

```
Initialise: Q(s, a) = 0 for all s, a

For each episode:
    s ← initial state
    While s is not terminal:
        a ← ε-greedy(Q, s)              # behaviour policy
        Take action a, observe r, s'
        Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]
        s ← s'
```

Under standard conditions (all state-action pairs visited infinitely often, decaying step sizes satisfying the Robbins-Monro conditions), Q-Learning converges to $$Q^*$$.

### Double Q-Learning
{: #double-q}

Q-Learning suffers from **maximisation bias**: $$\max_{a'} Q(s', a')$$ overestimates the true maximum expected value because the same samples are used to both select and evaluate the best action.

**Double Q-Learning** maintains two independent estimators $$Q_1$$ and $$Q_2$$. One selects the best action; the other evaluates it:

$$Q_1(s, a) \leftarrow Q_1(s, a) + \alpha \left[ r + \gamma Q_2\!\left(s', \arg\max_{a'} Q_1(s', a')\right) - Q_1(s, a) \right]$$

With probability 0.5, swap the roles of $$Q_1$$ and $$Q_2$$. Because the action is selected by $$Q_1$$ and evaluated by $$Q_2$$ (trained on different experience), the bias is eliminated.

### Expected SARSA
{: #expected-sarsa}

**Expected SARSA** replaces the sampled next action with the expectation over the policy:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \sum_{a'} \pi(a' \mid s_{t+1}) Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

This eliminates variance from the random selection of $$a_{t+1}$$ (as in SARSA) while remaining on-policy. Expected SARSA typically has lower variance than SARSA and often performs better. With a greedy target policy it becomes Q-Learning.

**Pros and Cons — Model-Free Control Algorithms**

| | SARSA | Q-Learning | Double Q-Learning | Expected SARSA |
|---|---|---|---|---|
| **Policy** | On-policy | Off-policy | Off-policy | On-policy (or off) |
| **Bias** | Follows behaviour policy | Maximisation bias | Corrects max bias | Lower variance than SARSA |
| **Safety** | Safer (accounts for exploration) | Can find optimal regardless of behaviour | Finds optimal; unbiased | Good balance |
| **Convergence** | Converges to Qπ (on-policy) | Converges to Q\* | Converges to Q\* | Converges to Qπ or Q\* |

> **Interview question:** Why does Q-Learning overestimate action values and how does Double Q-Learning fix this?
>
> **Answer:** Q-Learning's target uses $$\max_{a'} Q(s', a')$$. If Q values are noisy — which they always are during learning — then taking the maximum over noisy estimates systematically overestimates the true maximum expected value. This is because $$\mathbb{E}[\max_a X_a] \geq \max_a \mathbb{E}[X_a]$$ for random variables $$X_a$$. The magnitude of the bias grows with the number of actions and the noise level, and it can substantially slow convergence or lead the agent to prefer suboptimal actions. Double Q-Learning decouples action selection from action evaluation by using two independent Q-networks: $$Q_1$$ selects the best action and $$Q_2$$ evaluates its value. Since the two networks are trained on different experience, their errors are independent, and the combined estimate is approximately unbiased.

---

## Function Approximation
{: #function-approximation}

Tabular methods store one value per state (or state-action pair), which is infeasible for large or continuous state spaces. **Function approximation** represents the value function with a parameterised function $$\hat{V}(s; \mathbf{w}) \approx V^\pi(s)$$.

### Linear Value Function Approximation
{: #linear-fa}

Linear approximation uses a feature vector $$\phi(s) \in \mathbb{R}^d$$ and a weight vector $$\mathbf{w} \in \mathbb{R}^d$$:

$$\hat{V}(s; \mathbf{w}) = \mathbf{w}^\top \phi(s)$$

The feature vector $$\phi(s)$$ encodes relevant information about state $$s$$ (e.g., polynomial basis functions, tile coding, radial basis functions). The mean squared value error is:

$$\overline{\text{VE}}(\mathbf{w}) = \sum_{s \in \mathcal{S}} \mu(s) \left[ V^\pi(s) - \hat{V}(s; \mathbf{w}) \right]^2$$

where $$\mu(s)$$ is the on-policy state distribution. The gradient descent update is:

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \left[ V^\pi(s) - \hat{V}(s; \mathbf{w}) \right] \phi(s)$$

Linear approximation converges to the **best linear approximation** of $$V^\pi$$ (minimum $$\overline{\text{VE}}$$) when the target is the true value. The challenge is choosing good features $$\phi(s)$$ — this requires domain knowledge.

### Semi-Gradient Methods
{: #semi-gradient}

When the target itself depends on $$\mathbf{w}$$ (as in bootstrapping), the update is not a true gradient descent step — it is a **semi-gradient** method:

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \delta_t \nabla_{\mathbf{w}} \hat{V}(s_t; \mathbf{w})$$

where $$\delta_t = r_{t+1} + \gamma \hat{V}(s_{t+1}; \mathbf{w}) - \hat{V}(s_t; \mathbf{w})$$. The gradient is taken only through $$\hat{V}(s_t; \mathbf{w})$$, not through the target $$\hat{V}(s_{t+1}; \mathbf{w})$$.

For **linear** function approximation with TD(0), the semi-gradient update converges to a fixed point near the minimum $$\overline{\text{VE}}$$ under the on-policy distribution. For non-linear approximators (neural networks), convergence is not guaranteed.

### Deadly Triad
{: #deadly-triad}

The **deadly triad** (Sutton & Barto) is the combination of three ingredients that can cause divergence:

1. **Function approximation** — representing values with a parameterised function (e.g., neural network) rather than a table
2. **Bootstrapping** — using the current value estimate as a target (as in TD learning)
3. **Off-policy learning** — training on data from a different distribution than the target policy

Any two of the three can be used together safely; all three together can diverge even for linear function approximation. Classic examples include Baird's counterexample (off-policy linear TD that diverges). This is why DQN uses techniques (experience replay, target networks) specifically designed to mitigate the deadly triad.

**Pros and Cons — Function Approximation**

| | Linear FA | Neural Network FA |
|---|---|---|
| **Pros** | Convergence guarantees; fast; interpretable weights | Can represent complex, non-linear value functions; no manual feature engineering |
| **Cons** | Requires manual feature engineering; limited expressivity | No convergence guarantees with bootstrapping; deadly triad; sensitive to hyperparameters |

> **Interview question:** What is the deadly triad and why does it cause instability?
>
> **Answer:** The deadly triad refers to the simultaneous use of function approximation, bootstrapping, and off-policy learning. The instability arises because bootstrapping creates a circular dependency: the target used to update $$\hat{V}(s; \mathbf{w})$$ is $$r + \gamma \hat{V}(s'; \mathbf{w})$$, which depends on the same parameters being updated. In tabular settings this is fine because each state's value is updated independently. But with function approximation, updating $$\hat{V}(s; \mathbf{w})$$ changes $$\hat{V}(s'; \mathbf{w})$$ too, potentially in directions that increase error rather than decrease it. Off-policy learning compounds this: the function approximator is evaluated on states sampled by the behaviour policy, but must generalise to states visited by the target policy, creating a distribution mismatch. Together, these can create positive feedback loops that drive weights to infinity — formal divergence examples exist even for linear approximation.

---

## Deep Q-Networks (DQN)
{: #dqn}

**DQN** ([Mnih et al., 2015](https://www.nature.com/articles/nature14236)) combines Q-Learning with deep neural networks to learn directly from high-dimensional inputs (raw pixels). It introduced two key stabilising mechanisms that mitigate the deadly triad.

### Experience Replay
{: #experience-replay}

The agent stores transitions $$(s_t, a_t, r_{t+1}, s_{t+1})$$ in a **replay buffer** $$\mathcal{D}$$ of fixed size $$N$$. At each training step, a random minibatch $$\mathcal{B}$$ is sampled from $$\mathcal{D}$$:

$$\mathcal{D} = \{(s_i, a_i, r_i, s_i')\}_{i=1}^{N}$$

Benefits:
- **Data efficiency**: each transition is used in many gradient updates, not just once
- **Decorrelation**: consecutive samples in the replay buffer are no longer temporally correlated, reducing gradient variance
- **Off-policy**: enables learning from past experience, making training more stable

### Target Network
{: #target-network}

DQN maintains two networks: the **online network** $$Q(s, a; \mathbf{w})$$ (updated every step) and a **target network** $$Q(s, a; \mathbf{w}^-)$$ (updated periodically, every $$C$$ steps):

$$\mathbf{w}^- \leftarrow \mathbf{w} \quad \text{every } C \text{ steps}$$

The **DQN loss** is:

$$\mathcal{L}(\mathbf{w}) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \mathbf{w}^-) - Q(s, a; \mathbf{w}) \right)^2 \right]$$

Using $$\mathbf{w}^-$$ in the target instead of $$\mathbf{w}$$ breaks the feedback loop that causes instability: the target is held fixed for $$C$$ steps, so updates to $$\mathbf{w}$$ do not immediately change the target. This is the key insight that made DQN stable on Atari games.

```
Initialise: online network Q(s,a;w), target network Q(s,a;w⁻) = Q(s,a;w)
Initialise: replay buffer D with capacity N

For each step t:
    Select a_t = argmax_a Q(s_t, a; w) with prob 1-ε, random with prob ε
    Execute a_t, observe r_t, s_{t+1}
    Store (s_t, a_t, r_t, s_{t+1}) in D

    Sample minibatch B = {(s, a, r, s')} from D
    For each sample:
        y = r + γ max_{a'} Q(s', a'; w⁻)   if s' is not terminal
        y = r                                 if s' is terminal
    Gradient step: minimize (y - Q(s, a; w))² over w

    Every C steps: w⁻ ← w
```

### DQN Extensions
{: #dqn-extensions}

**Double DQN** ([van Hasselt et al., 2016](https://arxiv.org/abs/1509.06461)): decouple action selection from evaluation using the online network for selection and target network for evaluation:

$$y = r + \gamma Q\!\left(s', \arg\max_{a'} Q(s', a'; \mathbf{w}); \mathbf{w}^-\right)$$

**Dueling DQN** ([Wang et al., 2016](https://arxiv.org/abs/1511.06581)): separate the network into two streams — one for $$V(s)$$ and one for $$A(s, a)$$ — combined as:

$$Q(s, a; \mathbf{w}) = V(s; \mathbf{w}_V) + A(s, a; \mathbf{w}_A) - \frac{1}{\vert \mathcal{A}\vert } \sum_{a'} A(s, a'; \mathbf{w}_A)$$

The subtraction of the mean advantage ensures identifiability. The dueling architecture helps in states where the action choice does not matter much, since $$V(s)$$ can be learned without visiting all actions.

**Prioritised Experience Replay (PER)** ([Schaul et al., 2016](https://arxiv.org/abs/1511.05952)): sample transitions with probability proportional to their TD error magnitude, so transitions with large errors (surprising transitions) are replayed more often:

$$P(i) = \frac{p_i^\alpha}{\sum_j p_j^\alpha}, \quad p_i = \vert \delta_i\vert  + \varepsilon$$

Importance-sampling weights $$w_i = (N \cdot P(i))^{-\beta}$$ correct for the non-uniform sampling.

**Rainbow** ([Hessel et al., 2018](https://arxiv.org/abs/1710.02298)): combines six DQN extensions — Double DQN, Prioritised Replay, Dueling Networks, multi-step returns, distributional RL (C51), and noisy networks — achieving state-of-the-art on Atari with all components contributing.

**Pros and Cons — DQN and Extensions**

| | DQN | Double DQN | Dueling DQN | PER |
|---|---|---|---|---|
| **Improvement** | Stable deep RL with pixels | Fixes overestimation | Better V/A decomposition | More efficient data use |
| **Pros** | Landmark result; learns from raw pixels; widely applicable | Reduces maximisation bias; often faster convergence | Robust in action-irrelevant states; better policy in many games | Faster convergence; focuses on informative transitions |
| **Cons** | Discrete actions only; slow convergence; many hyperparameters | Slight additional complexity | Requires careful architecture design | Hyperparameters α, β; stale priorities |

> **Interview question:** Why does DQN use a replay buffer and a target network? Would one without the other work?
>
> **Answer:** The replay buffer and target network address different sources of instability. The replay buffer breaks temporal correlation between consecutive training samples (which would bias gradient estimates) and enables each transition to be reused many times, improving sample efficiency. The target network stabilises the learning target by holding it fixed for C steps — without it, the target changes every step in a direction correlated with the gradient update, creating a feedback loop that can diverge. Using a replay buffer alone (but no target network) would reduce correlation but the chasing target would still cause instability. Using a target network alone (but no replay buffer) would stabilise targets but leave correlated consecutive samples biasing the gradient. In practice, both are needed: DQN without either diverges on most Atari games, with only a replay buffer is marginally better, with only a target network somewhat better, and with both achieves the stable superhuman performance reported in the original paper.

---

## Policy Gradient Methods
{: #policy-gradients}

Policy gradient methods directly optimise a parameterised policy $$\pi_\theta(a \mid s)$$ rather than learning a value function. They work in continuous action spaces where $$\arg\max$$ over actions is intractable.

### Policy Gradient Theorem
{: #pg-theorem}

The **policy gradient theorem** gives the gradient of the expected return $$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]$$ with respect to $$\theta$$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^{\pi_\theta}(s_t, a_t) \right]$$

This is remarkable: the gradient of an expectation over trajectories (which depends on the unknown environment dynamics $$P$$) equals an expectation that can be estimated from samples, without knowing $$P$$. The quantity $$\nabla_\theta \log \pi_\theta(a \mid s)$$ is the **score function** — it points in the direction that increases the probability of action $$a$$ in state $$s$$.

The intuition: actions that lead to high returns have their log-probabilities increased; actions that lead to low returns have their log-probabilities decreased.

### REINFORCE
{: #reinforce}

**REINFORCE** ([Williams, 1992](https://link.springer.com/article/10.1007/BF00992696)) is the simplest policy gradient algorithm. It estimates the gradient using Monte Carlo returns:

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^{(n)} \mid s_t^{(n)}) \cdot G_t^{(n)}$$

```
Initialise: policy parameters θ

For each episode n:
    Collect trajectory τ = (s_0, a_0, r_1, ..., s_T) under π_θ
    For t = 0, 1, ..., T-1:
        G_t ← Σ_{k=0}^{T-t-1} γ^k r_{t+k+1}   # compute return
    θ ← θ + α Σ_t ∇_θ log π_θ(a_t | s_t) G_t    # gradient ascent
```

**REINFORCE with baseline**: subtracting a baseline $$b(s_t)$$ from the return reduces variance without introducing bias (since $$\mathbb{E}[\nabla_\theta \log \pi_\theta(a \mid s) b(s)] = 0$$):

$$\nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot (G_t - b(s_t))$$

The optimal baseline is $$b(s_t) = V^{\pi_\theta}(s_t)$$, which is estimated by a critic network. This makes the effective signal the **advantage** $$A_t = G_t - V(s_t)$$.

**Pros and Cons — REINFORCE**

| | Pros | Cons |
|---|---|---|
| **Generality** | Works with any differentiable policy; handles continuous actions; naturally stochastic | High variance — returns are sums of many random variables |
| **Simplicity** | Conceptually clear; gradient is unbiased MC estimate | Requires complete episodes; slow convergence |
| **Baseline** | Baseline (b(s) = V(s)) reduces variance without bias | Still high variance even with baseline; many episodes needed |

> **Interview question:** Why is the REINFORCE gradient unbiased and why is variance a problem?
>
> **Answer:** The REINFORCE gradient is unbiased because it is a Monte Carlo estimate of the true policy gradient — the expectation of log π_θ(a&#124;s) · G_t over trajectories is exactly ∇J(θ) by the policy gradient theorem. However, variance is high because G_t is a sum of potentially hundreds or thousands of future rewards, each stochastic. Even if each reward has low variance, the sum can have enormous variance. High gradient variance means many samples are needed before the gradient estimate is reliable enough to make useful progress. This is the fundamental problem that TD-based actor-critic methods solve: instead of using the high-variance Monte Carlo return G_t, they use the lower-variance TD estimate r + γ V(s') - V(s) as the advantage signal, at the cost of introducing bias from the critic's imperfect value estimate.

### Actor-Critic
{: #actor-critic}

**Actor-Critic** methods maintain two function approximators:
- **Actor**: the policy $$\pi_\theta(a \mid s)$$, updated to maximise expected return
- **Critic**: the value function $$V_\phi(s)$$, updated to minimise TD error

The advantage is estimated as:

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t) \approx r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) = \delta_t$$

The actor update is:

$$\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot \delta_t$$

The critic update minimises the squared TD error:

$$\phi \leftarrow \phi - \alpha_\phi \delta_t \nabla_\phi V_\phi(s_t)$$

**A2C (Advantage Actor-Critic)**: synchronous version where multiple workers collect experience in parallel and gradients are averaged. The advantage is estimated using $$n$$-step returns or GAE.

**A3C (Asynchronous Advantage Actor-Critic)** ([Mnih et al., 2016](https://arxiv.org/abs/1602.01783)): asynchronous version with multiple workers running independently. Each worker maintains its own copy of the environment, collects experience, computes gradients, and applies them to a shared global network asynchronously. A3C was the first algorithm to achieve superhuman performance on a broad set of tasks without a replay buffer, using CPU-only parallelism.

**Generalised Advantage Estimation (GAE)** ([Schulman et al., 2016](https://arxiv.org/abs/1506.02438)) exponentially smooths TD errors to trade off bias and variance:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

When $$\lambda = 0$$: one-step TD advantage (low variance, high bias). When $$\lambda = 1$$: Monte Carlo advantage (high variance, low bias). GAE with $$\lambda \approx 0.95$$ is standard in PPO and A3C.

---

## Advanced Policy Optimization
{: #advanced-policy-opt}

Vanilla policy gradient methods suffer from instability: a single large gradient step can destroy the policy. The advanced methods below address this by constraining or clipping updates.

### TRPO
{: #trpo}

**Trust Region Policy Optimization (TRPO)** ([Schulman et al., 2015](https://arxiv.org/abs/1502.05477)) constrains the policy update to stay within a trust region — a neighbourhood where the surrogate objective is a reliable approximation of the true objective.

TRPO maximises a surrogate objective subject to a KL divergence constraint:

$$\max_\theta \; \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] \quad \text{subject to} \quad \hat{\mathbb{E}}_t \left[ \text{KL}\left[\pi_{\theta_{\text{old}}}(\cdot \mid s_t) \Vert \pi_\theta(\cdot \mid s_t)\right] \right] \leq \delta$$

The ratio $$r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$$ is the **importance ratio** — it allows reusing data from $$\pi_{\theta_{\text{old}}}$$ to estimate the performance of $$\pi_\theta$$.

The constraint $$\text{KL} \leq \delta$$ prevents the new policy from deviating too far from the old one — the policy is monotonically improved with each update. TRPO solves this constrained optimisation using conjugate gradient and a line search. This is theoretically principled but computationally expensive and complex to implement.

### PPO
{: #ppo}

**Proximal Policy Optimization (PPO)** ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) approximates TRPO's trust region constraint with a simple clipping mechanism, achieving similar performance with far less complexity:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \;\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t \right) \right]$$

where $$r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$$ and $$\varepsilon \approx 0.2$$.

The clipping removes the incentive to move $$r_t(\theta)$$ beyond $$[1-\varepsilon, 1+\varepsilon]$$:
- When $$\hat{A}_t > 0$$: the ratio is clipped at $$1+\varepsilon$$ — we can increase probability but not too much
- When $$\hat{A}_t < 0$$: the ratio is clipped at $$1-\varepsilon$$ — we decrease probability but not too much

The full PPO objective includes the clipped policy loss, a value function loss, and an entropy bonus:

$$\mathcal{L}(\theta) = \mathcal{L}^{\text{CLIP}}(\theta) - c_1 \mathcal{L}^{\text{VF}}(\theta) + c_2 \mathbb{H}[\pi_\theta]$$

```
Initialise: policy π_θ, value function V_φ

For each iteration:
    Collect T timesteps of experience under π_θ
    Compute advantages Â_t using GAE
    For K epochs:
        Sample minibatch of transitions
        Compute r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
        Compute L^CLIP(θ) = E[min(r_t Â_t, clip(r_t, 1-ε, 1+ε) Â_t)]
        Compute value loss L^VF = E[(V_φ(s_t) - V_t^target)²]
        Update θ by gradient ascent on L^CLIP - c₁ L^VF + c₂ H[π]
    θ_old ← θ
```

PPO is the de facto standard for on-policy deep RL: it is simple, stable, and achieves strong performance across continuous control and Atari tasks. It is the backbone of most RLHF pipelines for LLMs.

### SAC
{: #sac}

**Soft Actor-Critic (SAC)** ([Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)) is an off-policy actor-critic algorithm with **maximum entropy** regularisation. Instead of maximising only the expected return, SAC maximises a combination of return and policy entropy:

$$J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right]$$

where $$\mathcal{H}(\pi(\cdot \mid s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$$ is the policy entropy and $$\alpha > 0$$ is the temperature parameter.

The entropy bonus encourages exploration and prevents premature convergence to a deterministic policy. The soft Bellman equation becomes:

$$Q(s, a) = r + \gamma \mathbb{E}_{s' \sim P}[V(s')], \quad V(s) = \mathbb{E}_{a \sim \pi}[Q(s,a) - \alpha \log \pi(a \mid s)]$$

SAC uses:
- Two soft Q-networks (Double Q-Learning style) to mitigate overestimation
- A reparameterisation trick for the policy gradient (low variance)
- An automatic temperature adjustment rule for $$\alpha$$

SAC achieves state-of-the-art sample efficiency on continuous control benchmarks (MuJoCo) and is a standard baseline for robotic learning.

**Pros and Cons — Advanced Policy Optimization**

| | TRPO | PPO | SAC |
|---|---|---|---|
| **Stability** | Monotonic improvement guarantee | Strong stability via clipping; easy to implement | Very stable; off-policy data reuse |
| **Sample efficiency** | On-policy; moderate | On-policy; moderate | Off-policy; highest |
| **Complexity** | Complex: conjugate gradient, Fisher matrix | Simple: just add clipping | Moderate: two Q-networks, entropy term |
| **Action space** | Discrete or continuous | Discrete or continuous | Continuous only (by design) |
| **Best for** | Theoretical guarantees | LLM alignment, simulated control | Real-world robotics, sample-limited settings |

> **Interview question:** What does the PPO clipping objective actually do and why is it better than a plain KL penalty?
>
> **Answer:** The PPO clip objective min(r_t · Â_t, clip(r_t, 1−ε, 1+ε) · Â_t) limits the importance ratio r_t = π_θ / π_θ_old to the interval [1−ε, 1+ε]. By taking the minimum of the clipped and unclipped objective, it removes the incentive to push r_t beyond the clip boundary — the gradient becomes zero once the ratio exits the trust region. Compared to a KL penalty, the clipped objective avoids the need to tune a penalty coefficient β (which TRPO tries to adapt but often gets wrong) and is more numerically stable. Empirically, PPO-Clip and TRPO achieve similar performance, but PPO requires only first-order gradients — no Fisher matrix inversion or conjugate gradient — making it dramatically simpler to implement and roughly 10× faster in wall-clock time.

---

## Model-Based RL
{: #model-based}

Model-based RL methods learn a model of the environment dynamics $$\hat{P}(s' \mid s, a)$$ and use it to plan or generate synthetic experience, potentially achieving much greater sample efficiency than model-free methods.

### Dyna-Q
{: #dyna-q}

**Dyna-Q** ([Sutton, 1990](https://dl.acm.org/doi/10.1145/122344.122377)) integrates model learning with direct RL and model-based planning in a single unified architecture:

```
Initialise: Q(s, a) = 0; empty model M(s, a)

For each real step:
    # Direct RL: update Q from real experience
    Take action a in state s, observe r, s'
    Q(s, a) ← Q(s, a) + α [r + γ max_{a''} Q(s', a'') - Q(s, a)]
    
    # Model learning: update model
    M(s, a) ← (r, s')   # store transition (deterministic model)
    
    # Planning: n simulated updates from model
    For i = 1, 2, ..., n:
        s_sim ← random previously-seen state
        a_sim ← random action taken in s_sim
        r_sim, s'_sim ← M(s_sim, a_sim)    # model prediction
        Q(s_sim, a_sim) ← Q(s_sim, a_sim) + α [r_sim + γ max_{a''} Q(s'_sim, a'') - Q(s_sim, a_sim)]
```

With $$n$$ planning steps per real step, Dyna-Q achieves roughly $$n$$-fold improvement in sample efficiency. The model improves as more real experience is collected, and planning becomes more accurate. However, a wrong model produces wrong synthetic data, potentially biasing the Q-function — **model error** is the main challenge.

### World Models
{: #world-models}

**World models** ([Ha & Schmidhuber, 2018](https://arxiv.org/abs/1803.10122)) learn a compact latent representation of the environment and train an agent entirely inside the "dreamed" model:

<div class="post-flow post-flow--horizontal" role="group" aria-label="World model pipeline">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">V (Vision): VAE encodes observations to latent z</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">M (Memory): RNN predicts next latent z'</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">C (Controller): small linear policy trained in dream</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Deploy controller in real environment</span></li>
  </ol>
</div>

**DreamerV3** ([Hafner et al., 2023](https://arxiv.org/abs/2301.04104)) learns a world model in latent space using a recurrent state space model (RSSM) and trains an actor-critic entirely in imagination. It achieves superhuman performance on Atari, continuous control, and 3D environments with the same hyperparameters, demonstrating the generality of model-based approaches.

**MuZero** ([Schrittwieser et al., 2020](https://www.nature.com/articles/s41586-020-03051-4)) extends AlphaZero by learning the dynamics model jointly with the value and policy, without access to a ground-truth simulator. It learns to represent only the aspects of the environment dynamics that are relevant to planning.

### AlphaZero and MCTS
{: #alphazero}

**Monte Carlo Tree Search (MCTS)** is a planning algorithm that builds a search tree by simulating rollouts from the current state. At each node it maintains statistics: visit count $$N(s,a)$$ and cumulative value $$W(s,a)$$.

Selection uses a UCB-style rule (PUCT in AlphaZero):

$$a^* = \arg\max_a \left[ \frac{W(s,a)}{N(s,a)} + c \cdot P(s,a) \cdot \frac{\sqrt{\sum_{a'} N(s,a')}}{1 + N(s,a)} \right]$$

where $$P(s, a)$$ is the prior probability from the policy network.

**AlphaZero** ([Silver et al., 2018](https://science.sciencemag.org/content/362/6419/1140)) combines MCTS with deep neural networks (a joint policy and value head) through self-play:

```
For each self-play game:
    At each position s:
        Run MCTS using π_θ as prior and V_θ as rollout value
        π_MCTS(a|s) ← N(s,a)^{1/τ} / Σ N(s,a')^{1/τ}   # temperature-scaled visit counts
        Store (s, π_MCTS, result)

Train (π_θ, V_θ) to minimise:
    L = (z - V_θ(s))² - π_MCTS · log π_θ(s) + λ||θ||²
```

AlphaZero mastered chess, shogi, and Go from scratch using only the rules of the game, surpassing all prior systems. It is the strongest demonstration that model-based planning + learned value functions + self-play can solve previously intractable sequential decision problems.

**Pros and Cons — Model-Based RL**

| | Dyna-Q | World Models | AlphaZero/MCTS |
|---|---|---|---|
| **Sample efficiency** | High — planning multiplies real transitions | Very high — training in imagination | Very high — MCTS extracts maximum signal |
| **Model error** | Sensitive to wrong model | Sensitive; errors accumulate in long rollouts | Requires exact simulator; no dynamics learning in original AlphaZero |
| **Scalability** | Tabular or small state spaces | High-dimensional observations via latent space | Scales to complex games; requires fast simulation |
| **Best for** | Simple environments with reliable models | Pixel-based environments, partial observability | Games with perfect simulators |

> **Interview question:** What are the advantages and risks of training a model-based RL agent inside a learned world model?
>
> **Answer:** The main advantage is sample efficiency: an agent can perform many gradient updates from simulated experience generated by the world model at low computational cost, without needing real environment interactions. This is particularly valuable when real interactions are expensive or dangerous (e.g., robotics). The main risk is model error: the world model is only an approximation of reality, and errors compound over long imagined rollouts. An agent that trains too heavily on imagination can exploit model errors — finding policies that look optimal inside the dream but fail in the real environment (the "Dyna dilemma"). Mitigation strategies include: keeping imagined rollouts short, using ensembles of world models to quantify uncertainty, penalising actions where the ensemble disagrees, and periodically validating learned behaviours in the real environment.

---

## Multi-Agent RL
{: #marl}

**Multi-agent RL (MARL)** extends the single-agent MDP to settings with multiple agents interacting in a shared environment. The environment is now a **Markov Game** (or stochastic game): $$(\mathcal{N}, \mathcal{S}, \{\mathcal{A}^i\}_{i \in \mathcal{N}}, P, \{R^i\}_{i \in \mathcal{N}}, \gamma)$$ where $$\mathcal{N}$$ is the set of agents.

**Interaction types:**

| Type | Rewards | Examples |
|---|---|---|
| **Cooperative** | Shared reward: Rⁱ = Rʲ for all i, j | Multi-robot coordination, traffic control |
| **Competitive (zero-sum)** | Opposing rewards: Rⁱ = −Rʲ | Chess, Go, poker, StarCraft 1v1 |
| **Mixed (general-sum)** | Independent rewards | Real-world economic systems, most practical settings |

The key conceptual challenge: from each agent's perspective, other agents are part of the environment — but unlike the rest of the environment, other agents are also learning and adapting. This makes the environment **non-stationary** from each agent's viewpoint, violating the standard MDP assumption.

**CTDE (Centralised Training, Decentralised Execution)** is the dominant paradigm for cooperative MARL. During training, a centralised critic has access to global information (all agents' observations and actions). During execution, each agent acts using only its local observation.

**QMIX** ([Rashid et al., 2018](https://arxiv.org/abs/1803.11605)) is a prominent CTDE algorithm: each agent $$i$$ has a local Q-function $$Q_i(o^i, a^i)$$, and a mixing network combines them into a global $$Q_{\text{tot}}$$ with the constraint that $$\partial Q_{\text{tot}} / \partial Q_i \geq 0$$ (monotonicity). This ensures that the greedy joint action can be found by each agent maximising its local Q independently.

**MADDPG** ([Lowe et al., 2017](https://arxiv.org/abs/1706.02275)) extends DDPG to multi-agent settings: each agent has a centralised critic that takes all agents' observations and actions as input, but a decentralised actor that uses only local observations.

**Emergent Communication**: when cooperative agents cannot directly observe each other's state, they benefit from developing communication protocols. Works like [Mordatch & Abbeel (2018)](https://arxiv.org/abs/1703.04908) showed that language-like communication emerges naturally when agents have shared goals and need to coordinate.

> **Interview question:** Why does independent Q-Learning often fail in multi-agent settings?
>
> **Answer:** Independent Q-Learning (IQL) trains each agent as if it were a single-agent problem, ignoring the other agents. The fundamental problem is non-stationarity: as each agent's policy changes during training, the transition dynamics and reward distributions experienced by every other agent change too. This violates the stationarity assumption that Q-Learning (and all tabular RL methods) requires for convergence. From agent i's perspective, the environment keeps shifting unpredictably because its co-agents are learning simultaneously. This can cause oscillations, cycles, and failure to converge to a Nash equilibrium. Centralised training methods like CTDE address this by giving critics access to global information during training, making the effective environment stationary from the perspective of the training objective.

---

## RL for LLMs
{: #rl-llms}

Reinforcement learning has become central to aligning large language models with human preferences. The key insight is that human preference feedback can serve as a reward signal that is difficult to specify analytically but easy for humans to provide.

### RLHF Pipeline
{: #rlhf}

**Reinforcement Learning from Human Feedback (RLHF)** ([Ziegler et al., 2019](https://arxiv.org/abs/1909.08593); [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)) is a three-stage pipeline:

<div class="post-flow post-flow--horizontal" role="group" aria-label="RLHF pipeline">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Stage 1: Supervised Fine-Tuning (SFT) on curated demonstrations</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Stage 2: Reward model training from human pairwise preferences</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Stage 3: RL optimisation of SFT model against reward model</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Aligned model ready for deployment</span></li>
  </ol>
</div>

**Stage 1 — Supervised Fine-Tuning**: fine-tune a pretrained LLM on high-quality human-written demonstrations of desired behaviour. This provides a strong initialisation $$\pi_\text{SFT}$$ before RL.

**Stage 2 — Reward Model Training**: collect human preference labels: given a prompt $$x$$ and two responses $$(y_1, y_2)$$, a human labeller indicates which response is better. The reward model $$r_\phi(x, y)$$ is trained by maximum likelihood on these pairwise comparisons using the Bradley-Terry model:

$$\mathcal{L}_\text{RM} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \right]$$

where $$y_w$$ is the preferred ("winning") response and $$y_l$$ is the less preferred ("losing") response.

**Stage 3 — RL Fine-Tuning**: optimise the policy $$\pi_\theta$$ to maximise the reward model score, with a KL penalty to prevent the policy from deviating too far from $$\pi_\text{SFT}$$ (which prevents reward hacking and catastrophic forgetting):

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \vert  x)} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{SFT}(y \mid x)} \right]$$

The KL term $$\beta \log \pi_\theta / \pi_\text{SFT}$$ penalises the policy for producing outputs that are very different from the SFT model, preventing **reward hacking** — exploiting bugs in the reward model rather than genuinely improving.

### Reward Models
{: #reward-models}

A reward model $$r_\phi(x, y)$$ takes a prompt $$x$$ and a completion $$y$$ and outputs a scalar reward. In practice it is usually a fine-tuned version of the SFT model with the final token representation passed through a linear head to produce a scalar.

**Challenges:**
- **Distribution shift**: the reward model is trained on data from $$\pi_\text{SFT}$$ but must evaluate outputs from the fine-tuned policy $$\pi_\theta$$, which gradually moves out of distribution
- **Reward hacking**: the policy finds responses that the reward model rates highly but humans would not (e.g., verbose responses, sycophantic agreement, deceptive responses)
- **Annotation quality**: human preferences are noisy, inconsistent, and influenced by presentation order

**Constitutional AI (CAI)** ([Bai et al., 2022](https://arxiv.org/abs/2212.06950)) replaces human preference labels with AI-generated feedback: a model critiques its own outputs according to a set of principles and generates improved revisions, producing preference data at scale. Anthropic's Claude models use this approach.

### PPO for LLMs
{: #ppo-llms}

PPO is the standard RL algorithm for RLHF. The LLM is treated as the actor $$\pi_\theta$$: the state is the prompt and conversation history, the action is the next token, and an episode terminates when the end-of-sequence token is generated.

The reward at each step is sparse: $$r_t = 0$$ for all intermediate tokens, and $$r_T = r_\phi(x, y)$$ for the final token. The KL penalty is distributed across all tokens:

$$r_t^\text{total} = r_\phi(x, y) \cdot \mathbf{1}[t = T] - \beta \log \frac{\pi_\theta(a_t \mid s_t)}{\pi_\text{SFT}(a_t \mid s_t)}$$

A separate critic network $$V_\psi$$ estimates the expected future reward from each token position to compute advantages via GAE. Implementation challenges include:
- Memory: storing activations for both actor and critic of a large LLM
- Stability: LLM token distributions shift rapidly; $$\varepsilon$$ in PPO clip needs careful tuning
- Reference model: $$\pi_\text{SFT}$$ must be kept in memory to compute the per-token KL

### GRPO and DPO
{: #alternatives}

**Direct Preference Optimization (DPO)** ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) shows that the RLHF objective with KL regularisation has a closed-form optimal policy:

$$\pi^*(y \mid x) \propto \pi_\text{SFT}(y \mid x) \exp\left(\frac{r(x, y)}{\beta}\right)$$

This allows reparameterising the reward model in terms of the policy:

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_\text{SFT}(y \mid x)} + \beta \log Z(x)$$

Substituting into the Bradley-Terry reward model and dropping the partition function (which cancels), the DPO loss becomes:

$$\mathcal{L}_\text{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{SFT}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{SFT}(y_l \mid x)} \right) \right]$$

DPO completely eliminates the need for a separate reward model and RL training loop — preference optimisation becomes a simple classification loss. It is simpler, more stable, and requires less infrastructure than PPO-based RLHF.

**Group Relative Policy Optimization (GRPO)** ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)) is an RL algorithm used in DeepSeek-R1 that eliminates the critic network. For each prompt $$x$$, it samples $$G$$ completions $$\{y_1, \ldots, y_G\}$$ and uses the group mean reward as a normalisation baseline:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

The policy is updated with a clipped PPO-style objective on these relative advantages. Because baselines are computed per-group rather than by a critic network, GRPO requires significantly less memory and is simpler to implement. It has proven effective for reasoning-intensive tasks where rewards can be computed by a verifier (math problems, code execution).

**Comparison of LLM Alignment Approaches:**

| | PPO (RLHF) | DPO | GRPO |
|---|---|---|---|
| **Reward model** | Explicit, separately trained | Implicit — folded into policy loss | Explicit (verifiable or learned) |
| **Critic network** | Required | Not required | Not required |
| **Infrastructure** | Complex — 4 models in memory (actor, critic, reference, reward) | Simple — 2 models (policy, reference) | Moderate — 3 models (actor, reference, reward) |
| **Data** | Preference pairs or scalar rewards | Preference pairs required | Scalar rewards (verifiable preferred) |
| **Best for** | Open-ended generation, complex reward shaping | Simpler alignment tasks, research | Reasoning, math, code with verifiable outputs |
| **Stability** | Can be unstable; requires careful tuning | Very stable | Moderately stable |

> **Interview question:** What is reward hacking in RLHF and how does the KL penalty address it?
>
> **Answer:** Reward hacking occurs when the policy learns to exploit flaws or blind spots in the reward model rather than genuinely improving along the intended dimension. For example, a policy might learn to produce extremely long responses (if the reward model conflates length with quality), to be excessively agreeable (if the reward model was trained on human labellers who prefer agreement), or to use specific formatting patterns that score well regardless of content quality. The KL penalty β·log π_θ(y&#124;x) / π_SFT(y&#124;x) limits how far the policy can drift from the SFT model: responses very different from the SFT distribution are penalised proportionally to their KL divergence from it. Since the SFT model produces reasonable responses by construction, and the reward model is accurate near the SFT distribution (where it was trained), keeping the policy near π_SFT limits exploitation of out-of-distribution reward model errors. The coefficient β controls this tradeoff: larger β means more conservative updates but less reward hacking; smaller β means more aggressive optimisation but higher risk of hacking.

> **Interview question:** How does DPO differ from PPO-based RLHF and when would you prefer one over the other?
>
> **Answer:** DPO derives a closed-form relationship between the optimal RLHF policy and pairwise preference data, allowing the reward model to be implicitly represented by the policy ratio π_θ / π_SFT. This converts preference optimisation into a supervised loss on preference pairs, eliminating the need for a separately trained reward model and the RL training loop entirely. PPO-based RLHF requires training four models simultaneously — actor, critic, reference, and reward model — which is memory-intensive and operationally complex. DPO requires only the policy and a reference model. In practice, DPO is preferred when: you have a clean dataset of pairwise preferences, you want simpler infrastructure, or you are working at a scale where running PPO's RL loop is too expensive. PPO is preferred when: rewards come from a non-differentiable source (e.g., a human rater, a code execution environment), when you want to fine-tune the reward model jointly with the policy, or when you need to handle non-pairwise reward signals. On difficult reasoning tasks with verifiable rewards (math, code), RL methods like PPO and GRPO often outperform DPO because they can explore the action space more aggressively.

**Key papers:**
- [Mnih et al. (2015) — Human-level control through deep reinforcement learning (DQN)](https://www.nature.com/articles/nature14236)
- [Schulman et al. (2015) — Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [Schulman et al. (2017) — Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Haarnoja et al. (2018) — Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- [Silver et al. (2018) — AlphaZero](https://science.sciencemag.org/content/362/6419/1140)
- [Ouyang et al. (2022) — InstructGPT (RLHF)](https://arxiv.org/abs/2203.02155)
- [Rafailov et al. (2023) — Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [Shao et al. (2024) — DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [Sutton & Barto — Reinforcement Learning: An Introduction (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html)
