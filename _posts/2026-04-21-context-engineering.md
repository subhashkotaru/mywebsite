---
title: "Context Engineering for LLMs"
date: 2026-04-21
display_order: 2
description: "A systematic survey of context engineering — the discipline of designing, managing, and optimising the information payloads that govern LLM behaviour, covering retrieval, processing, compression, memory, RAG, tool use, and multi-agent coordination."
tags: [ml-systems, llm, agents, rag, context]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#definition">Formal Definition</a>
      <ul class="post-toc-sublist">
        <li><a href="#components">Context Components</a></li>
        <li><a href="#optimization">Optimization Objective</a></li>
        <li><a href="#vs-prompt-eng">Context vs Prompt Engineering</a></li>
      </ul>
    </li>
    <li><a href="#retrieval-generation">Context Retrieval & Generation</a>
      <ul class="post-toc-sublist">
        <li><a href="#prompt-engineering">Prompt Engineering Methods</a></li>
        <li><a href="#external-retrieval">External Knowledge Retrieval</a></li>
        <li><a href="#dynamic-assembly">Dynamic Context Assembly</a></li>
      </ul>
    </li>
    <li><a href="#processing">Context Processing</a>
      <ul class="post-toc-sublist">
        <li><a href="#long-context">Long Context Processing</a></li>
        <li><a href="#self-refinement">Self-Refinement</a></li>
        <li><a href="#multimodal">Multimodal Context</a></li>
        <li><a href="#structured">Structured & Relational Context</a></li>
      </ul>
    </li>
    <li><a href="#management">Context Management</a>
      <ul class="post-toc-sublist">
        <li><a href="#constraints">Fundamental Constraints</a></li>
        <li><a href="#memory-hierarchies">Memory Hierarchies</a></li>
        <li><a href="#compression">Context Compression</a></li>
      </ul>
    </li>
    <li><a href="#systems">System Implementations</a>
      <ul class="post-toc-sublist">
        <li><a href="#rag">RAG Architectures</a></li>
        <li><a href="#memory-systems">Memory Systems</a></li>
        <li><a href="#tool-reasoning">Tool-Integrated Reasoning</a></li>
        <li><a href="#multi-agent">Multi-Agent Coordination</a></li>
      </ul>
    </li>
    <li><a href="#pattern-catalogue">Nine Context Engineering Patterns</a>
      <ul class="post-toc-sublist">
        <li><a href="#pattern-rag">Pattern 1: RAG</a></li>
        <li><a href="#pattern-agentic-rag">Pattern 2: Agentic RAG</a></li>
        <li><a href="#pattern-tool-mcp">Pattern 3: Tool / MCP</a></li>
        <li><a href="#pattern-workflow">Pattern 4: Workflow / Typed Pipeline</a></li>
        <li><a href="#pattern-memory">Pattern 5: Memory-Augmented Context</a></li>
        <li><a href="#pattern-agent-loop">Pattern 6: Agentic Plan-Act-Observe</a></li>
        <li><a href="#pattern-subagents">Pattern 7: Subagents & Context Partitioning</a></li>
        <li><a href="#pattern-file-native">Pattern 8: File-Native Context</a></li>
        <li><a href="#pattern-long-context">Pattern 9: Long-Context Management</a></li>
      </ul>
    </li>
    <li><a href="#product-architectures">Product Architectures</a>
      <ul class="post-toc-sublist">
        <li><a href="#fast-answer-engines">Fast Answer Engines</a></li>
        <li><a href="#deep-research-engines">Deep Research Engines</a></li>
        <li><a href="#build-order">How to Build One</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## Overview
{: #overview}

The performance of an LLM is governed not just by its weights but by the information it receives at inference time. As models evolved from simple chatbots into reasoning engines orchestrating tools, databases, and other agents, "write a better prompt" ceased to be adequate guidance. **Context Engineering** is the formal discipline that fills this gap: the systematic design, retrieval, processing, and management of the full information payload delivered to an LLM.

Where prompt engineering treats context as a static string, context engineering treats it as a dynamically assembled, structured object with components drawn from instruction databases, knowledge stores, memory systems, tool definitions, and live world state. The goal is not a better prompt — it is a better information logistics system.

---

## Formal Definition
{: #definition}

### Context Components
{: #components}

The standard autoregressive LLM generates output `Y` by maximising:

```
Pθ(Y|C) = ∏ₜ Pθ(yₜ | y<t, C)
```

Traditional prompt engineering treated `C` as a monolithic static string. Context engineering re-conceptualises `C` as a dynamically assembled set of typed components:

```
C = A(c₁, c₂, ..., cₙ)
```

where `A` is an orchestration function that sources, filters, formats, and concatenates the components. Each component maps to a distinct engineering concern:

| Component | Content | Section |
|-----------|---------|---------|
| `c_instr` | System instructions, rules, persona | Prompt Engineering |
| `c_know` | External knowledge retrieved from databases, KGs | RAG / Context Retrieval |
| `c_tools` | Tool schemas and function signatures | Tool-Integrated Reasoning |
| `c_mem` | Persistent information from prior interactions | Memory Systems |
| `c_state` | Dynamic world / user / agent state | Multi-Agent Coordination |
| `c_query` | The user's immediate request | — |

### Optimization Objective
{: #optimization}

Context Engineering is formally the problem of finding the optimal set of context-generating functions `F = {A, Retrieve, Select, ...}` that maximises expected output quality over a distribution of tasks `T`:

```
F* = argmax_F  E_{τ~T} [ Reward( Pθ(Y | C_F(τ)), Y*_τ ) ]
```

subject to `|C| ≤ L_max` (the model's context length limit).

Knowledge retrieval can be framed as maximising mutual information with the target:

```
Retrieve* = argmax_Retrieve  I(Y*; c_know | c_query)
```

This ensures retrieved context is maximally informative, not merely semantically similar. The Bayesian view treats optimal context as a posterior:

```
P(C | c_query, ...) ∝ P(c_query | C) · P(C | History, World)
```

which provides a principled framework for adaptive retrieval and belief-state maintenance across multi-step reasoning.

### Context vs Prompt Engineering
{: #vs-prompt-eng}

| Dimension | Prompt Engineering | Context Engineering |
|-----------|-------------------|---------------------|
| Context model | `C = prompt` (static string) | `C = A(c₁,...,cₙ)` (dynamic structured assembly) |
| Optimisation target | Best string for the task | Best set of functions `F` over a task distribution |
| State | Stateless | Stateful — explicit `c_mem` and `c_state` components |
| Information | Fixed at write time | Dynamically sourced at inference time under `|C| ≤ L_max` |
| Scalability | Brittleness grows with complexity | Manages complexity through modular composition |
| Debugging | Manual prompt inspection | Systematic evaluation of individual context functions |

---

## Context Retrieval & Generation
{: #retrieval-generation}

This layer sources and constructs the contextual information that enters the model. Three primary mechanisms operate here.

### Prompt Engineering Methods
{: #prompt-engineering}

The **CLEAR Framework** governs effective prompt construction: Conciseness, Logic, Explicitness, Adaptability, Reflectiveness. Core prompt architecture integrates task instructions, contextual data, input, and output format indicators.

**Zero-shot and few-shot paradigms.** Zero-shot prompting relies entirely on instruction clarity and pre-trained knowledge. Few-shot extends this with carefully selected exemplars. In-context learning treats demonstration examples as a form of meta-learning — the model infers a task specification from examples without parameter updates.

**[Chain-of-Thought](https://arxiv.org/abs/2201.11903) (CoT).** CoT decomposes complex problems into intermediate reasoning steps, mirroring human cognition. Zero-shot CoT with "Let's think step by step" improved MultiArith accuracy from 17.7% to 78.7%. Automatic Prompt Engineer (APE) automates prompt search via LLM-generated candidates evaluated against a held-out set.

**Structured reasoning topologies.**

<div class="post-flow" role="group" aria-label="Reasoning topology evolution">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Chain-of-Thought — linear reasoning trace, decomposes problem into steps</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Tree-of-Thoughts — hierarchical exploration with lookahead and backtracking; Game of 24: 4% → 74%</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Graph-of-Thoughts — arbitrary dependency graphs; +62% quality, −31% cost vs ToT</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Cognitive Prompting — structured human-like ops: goal clarification, decomposition, abstraction, evaluation</span></li>
  </ol>
</div>

Long Chain-of-Thought (LongCoT) — as in OpenAI o1, [DeepSeek-R1](https://arxiv.org/abs/2501.12948), QwQ — uses substantially longer reasoning traces with self-reflection and error correction. Empirically, larger context windows correlate with stronger reasoning performance. Adaptive approaches like Auto Long-Short Reasoning dynamically adjust trace length to question complexity.

### External Knowledge Retrieval
{: #external-retrieval}

External retrieval addresses the fundamental limitation that parametric knowledge is static and bounded. The field has evolved through three generations:

**Dense retrieval.** Dense Passage Retrieval (DPR) encodes queries and passages into a shared embedding space; retrieval is approximate nearest-neighbour search. Self-RAG improves on this by training the model to emit special tokens that decide *when* to retrieve and assess retrieval quality — the model retrieves only when retrieval will help.

**Knowledge graph integration.** KAPING retrieves relevant KG facts based on semantic similarity and prepends them as natural language. KARPA performs training-free KG adaptation through pre-planning, semantic matching, and relation path reasoning. Think-on-Graph enables sequential reasoning over KG triples, conducting multi-hop exploration to find relevant facts.

**Agentic retrieval.** Agentic RAG systems treat retrieval as a dynamic operation: agents analyse content, cross-reference sources, decompose tasks, and refine queries iteratively. Systems like RAPTOR build hierarchical document summaries for multi-granularity retrieval. HippoRAG uses a memory-inspired architecture modelled on human hippocampal indexing.

### Dynamic Context Assembly
{: #dynamic-assembly}

Assembling retrieved components into a coherent context is itself an optimisation problem. Key techniques:

- **Verbalization**: converting structured data (KG triples, table rows, DB records) into natural language so it integrates cleanly with text context. Python/SQL representations outperform prose for complex reasoning tasks by leveraging structural properties.
- **Multi-level structurisation**: reorganising input into layered structures based on linguistic relationships.
- **Multi-agent assembly**: frameworks like LangChain, AutoGen, and CrewAI coordinate specialised agents (analyst, coder, tester) in context construction, achieving 29.9–47.1% relative improvement in Pass@1 over single-agent approaches.

---

## Context Processing
{: #processing}

Once acquired, context must be transformed and optimised for maximum utility. Four sub-problems dominate this layer.

### Long Context Processing
{: #long-context}

The transformer's self-attention has O(n²) complexity in sequence length. Scaling Mistral-7B from 4K to 128K tokens requires a 122× compute increase; Llama 3.1 8B needs up to 16GB for a single 128K-token request.

**Architectural solutions for linear complexity.**

| Approach | Key Idea | Complexity |
|----------|----------|-----------|
| SSMs / Mamba | Fixed-size hidden state recurrence | O(n) time, O(1) memory |
| LongNet (dilated attention) | Exponentially expanding attentive fields with token distance | O(n) |
| Linear attention | Self-attention as kernel feature map dot-products | O(n); up to 4000× speedup |
| Toeplitz Neural Networks (TNN) | Relative-position Toeplitz matrices | O(n log n); extrapolates 512 → 14K tokens |

**Position interpolation for context extension.** Models trained with a fixed context window struggle when asked to generalise beyond it. Instead of extrapolation:
- **YaRN**: combines NTK-based interpolation with linear interpolation and attention distribution correction.
- **LongRoPE**: two-stage — fine-tune to 256K tokens, then positional interpolation to 2048K.
- **PoSE**: position simulation extends effective length to 128K without full-length training.
- **Self-Extend**: bi-level grouped + neighbour attention, no fine-tuning required.

**Efficient attention implementations.**

| Technique | Mechanism | Benefit |
|-----------|-----------|---------|
| Grouped-Query Attention (GQA) | Query heads share key/value heads | Reduced KV cache memory during decoding |
| FlashAttention-2 | Tiled SRAM computation, no O(n²) HBM materialisation | ~2× throughput, linear memory |
| Ring Attention | Blockwise computation distributed across devices, overlapped comms | Arbitrarily long sequences |
| Sparse attention (S2-Attn) | Shifted sparse patterns | 92% of full-attention perplexity at fraction of compute |

**KV cache management for long inference.** As context grows, the KV cache becomes the memory bottleneck:
- **Rolling Buffer Cache**: maintains a fixed attention span, reducing cache memory ~8× on 32K sequences.
- **StreamingLLM**: retains "attention sink" tokens (initial tokens that absorb disproportionate attention mass) plus a recency window, enabling infinite-length inference with up to 22.2× speedup over sliding-window recomputation.
- **H2O (Heavy Hitter Oracle)**: evicts KV cache entries whose tokens receive little cumulative attention; improves throughput by up to 29× and reduces latency by 1.9×.

### Self-Refinement
{: #self-refinement}

Self-refinement enables LLMs to improve outputs through cyclical generate-evaluate-revise loops without additional training.

**Core frameworks.**

| Framework | Key Mechanism |
|-----------|---------------|
| Self-Refine | Same model acts as generator, feedback provider, and refiner — no supervised training required |
| Reflexion | Maintains reflective text in an episodic memory buffer; future decisions conditioned on linguistic feedback |
| N-CRITICS | Ensemble of critic models evaluates initial output; aggregated feedback drives refinement until stopping criterion |
| A2R | Explicit multi-dimensional evaluation (correctness, citation quality) generates targeted natural language feedback |
| Agent-R | Uses MCTS to construct training data from "on-the-fly" corrections of erroneous agent paths |

**Meta-learning and autonomous evolution.** SELF teaches LLMs meta-skills with limited examples, then lets the model generate and filter its own training data continuously. The Self-Developing framework goes further: the LLM generates improvement algorithms as executable code, evaluates them, and uses DPO to recursively bootstrap its own capabilities.

GPT-4 achieves approximately 20% absolute performance improvement through iterative self-refinement on text generation tasks. The key insight: identifying and fixing errors is often easier than producing perfect initial solutions.

### Multimodal Context
{: #multimodal}

Multimodal LLMs extend context engineering beyond text by integrating vision, audio, and 3D environments.

**Integration architecture.** The dominant paradigm connects specialised encoders (CLIP for vision, CLAP for audio) to the LLM backbone via alignment modules (Q-Former or simple MLPs). Visual inputs are projected into discrete tokens concatenated with text tokens. Visual Prompt Generators (VPGs) trained on image-caption pairs learn to map visual features into the LLM's embedding space.

**Core challenges.** Modality bias is the primary obstacle: models trained heavily on text favour textual patterns, generating linguistically plausible but visually ungrounded responses. VPGs trained on simple captioning tasks learn to extract only salient caption-worthy features, missing visual details needed for complex instruction following. Fine-grained spatial and temporal reasoning — precise object localisation, detailed event sequences in video — remains a major frontier.

**Long multimodal context.** Image tokens consume significant context budget, limiting many-shot in-context learning. Adaptive hierarchical token compression, variable visual position encoding (V2PE), and dynamic query-aware frame selection for video address this. In-context learning performance is sensitive to input ordering and modality weighting.

### Structured & Relational Context
{: #structured}

LLMs struggle with tables, databases, and knowledge graphs because linearisation fails to preserve relational structure.

**Representation strategies.** Programming language representations (Python for KGs, SQL for databases) outperform natural language prose by leveraging the inherent structural semantics of the representation. GraphToken achieves up to 73 percentage point improvement on graph reasoning tasks through parameter-efficient structural encoding. GraphFormers nest GNN components alongside transformer blocks for joint structural and linguistic encoding.

**Synergized architectures.** GreaseLM facilitates deep interaction across all model layers: language context representations are grounded by structured world knowledge while linguistic nuances simultaneously inform graph representations. QA-GNN implements bidirectional attention connecting QA contexts and KGs through joint graph formation and message passing.

Knowledge graphs reduce hallucinations by grounding responses in verifiable facts. Structured knowledge representations improve summarisation by 40% over unstructured memory on public benchmarks.

---

## Context Management
{: #management}

Context management addresses the efficient organisation, storage, and retrieval of contextual information given finite context windows and quadratic compute costs.

### Fundamental Constraints
{: #constraints}

Three interlocking problems define the constraint landscape:

**Context window overflow.** When conversation history exceeds `L_max`, early context is truncated. The model "forgets" earlier interactions.

**Lost-in-the-middle phenomenon.** LLMs retrieve information most reliably from the beginning and end of their context window. Information in the middle is systematically under-attended. Performance on long-context retrieval degrades by up to 73% when critical information is in the middle of the context.

**Context collapse.** In long conversations or multi-agent systems, enlarged context windows cause models to fail at distinguishing between different conversational threads — they can no longer tell which part of their context is relevant to the current query.

**Statelessness.** LLMs process each interaction independently. There are no native mechanisms to maintain state across exchanges. Persistent memory requires explicit external management.

### Memory Hierarchies
{: #memory-hierarchies}

Modern LLM memory architectures are explicitly inspired by OS memory management and human cognition.

**OS-inspired virtual memory.** MemGPT implements a paging system: a limited main context window (system instructions, FIFO message queue, scratchpad) pages information in and out from external storage via function calls. The model itself decides what to page in and when. PagedAttention applies the same virtual memory principle to KV cache management.

**Cognitively-inspired forgetting.** MemoryBank applies Ebbinghaus Forgetting Curve theory — memories decay exponentially with time unless reinforced — to determine which memories to retain. ReadAgent uses episode pagination (segment long content into episodes), memory gisting (create concise representations per episode), and interactive look-up (retrieve gists on demand).

**Memory taxonomy.** Agent memory systems typically implement four tiers:

| Tier | Storage | Access | Lifetime |
|------|---------|--------|----------|
| In-context (working memory) | LLM context window | Immediate, free | Current session |
| External semantic (episodic) | Vector database | kNN retrieval | Persistent |
| External procedural | KV store / code | Exact lookup | Persistent |
| External episodic log | Append-only log | Sequential / filtered | Persistent |

**Multi-agent distributed memory.** Centralised systems coordinate efficiently but overflow as shared context grows. Decentralised systems avoid overflow but incur inter-agent query latency. Hybrid approaches partition shared knowledge from specialised per-agent context.

### Context Compression
{: #compression}

When context must fit in a finite window, compression becomes necessary.

**Autoencoder compression.** In-context Autoencoder (ICAE) achieves 4× compression by condensing long contexts into compact memory slots the LLM can condition on directly. Recurrent Context Compression (RCC) expands the effective context window within constrained storage, using instruction reconstruction to prevent degradation when both instructions and context are compressed.

**Hierarchical caching.** Activation Refilling (ACRE) employs a bi-layer KV cache: L1 captures global information compactly; L2 provides detailed local information. L1 is dynamically refilled with query-relevant entries from L2, integrating broad understanding with specific detail.

**Reasoning chain compression.** Long CoT traces are verbose. Compression strategies:

| Method | Approach | Reduction |
|--------|----------|-----------|
| PREMISE | Gradient-inspired prompt optimisation with trace diagnostics | −87.5% tokens |
| O1-Pruner | RL fine-tuning to shorten chains while maintaining accuracy | — |
| InftyThink | Iterative reasoning with intermediate summarisation | +3–13% accuracy |
| Prune-on-Logic | Structure-aware removal of low-utility reasoning steps | Selective |

**Long-range memory with bounded compute.** InfLLM stores distant contexts in external memory units and retrieves token-relevant units for attention computation, enabling models pre-trained on 4K-token sequences to process up to 1,024K tokens. Infini-attention incorporates compressive memory directly into the attention mechanism — local masked attention handles recency while long-term linear attention accesses the compressed past.

---

## System Implementations
{: #systems}

Foundational components compose into four major system architectures.

### RAG Architectures
{: #rag}

RAG bridges parametric knowledge and dynamic information by integrating external knowledge sources into generation.

**Naive RAG**: query → retrieve → concatenate → generate. Simple but brittle — retrieval quality is the bottleneck.

**Modular RAG**: decomposes the pipeline into reconfigurable modules with routing, scheduling, and fusion mechanisms. Hierarchically: top-level RAG stages → middle-level sub-modules → bottom-level operational units. Enables dynamic reconfiguration at inference time.

**Agentic RAG**: treats retrieval as an agent action. The model decides when to retrieve, what to retrieve, and how to integrate results. Self-RAG emits special reflection tokens to control retrieval timing and quality assessment. CRAG adds a corrective step that evaluates retrieval quality and triggers web search when retrieved documents are insufficient.

**Graph-Enhanced RAG**: builds a knowledge graph from documents (entities as nodes, relations as edges), then retrieves subgraphs rather than passages. GraphRAG enables multi-hop reasoning over document collections. HippoRAG models retrieval on hippocampal memory indexing, achieving better associative recall across large corpora.

### Memory Systems
{: #memory-systems}

Memory systems enable persistent, stateful interaction across sessions.

**Memory operations:** write (encode and store new information), read (retrieve relevant past information), delete (remove stale or conflicting information), and update (merge new information with existing memories).

**Memory consolidation** mirrors human memory reconsolidation — deduplication, merging near-duplicate memories, resolving conflicts between contradictory memories. Reflective Memory Management combines prospective reflection (anticipating what will need to be remembered) with retrospective reflection (summarising what happened) for dynamic optimisation.

**Memory-enhanced agents.** MemOS implements a hierarchical memory operating system with tiered storage. MEM0 provides a unified API for reading and writing across memory tiers. A-MEM applies associative memory principles to enable flexible, context-aware retrieval. MemoryBank uses time-weighted retrieval to surface memories that are both relevant and recent.

### Tool-Integrated Reasoning
{: #tool-reasoning}

Tool use transforms LLMs from passive generators into active world-interactors. The evolution:

<div class="post-flow" role="group" aria-label="Tool use evolution">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">WebGPT — LLM browses web, cites sources for factual QA</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Toolformer — self-supervised tool use; model learns when to call APIs by inserting call tokens during training</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">ReAct — interleaved Reason + Act loop; tool outputs feed back as observations into reasoning</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Function calling — structured JSON schema; model emits structured tool calls, executor invokes, result returned</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">MCP (Model Context Protocol) — standardised client-server protocol for tool discovery and invocation</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">ReTool / ToolACE — RL-trained tool selection; model learns policies over large tool libraries</span></li>
  </ol>
</div>

**Tool-Integrated Reasoning.** Models like Chameleon compose tools into chains: natural language → tool selection → structured execution → result integration → next reasoning step. ChatCoT explicitly interleaves CoT reasoning with tool calls, forcing the model to articulate its reasoning before and after each tool invocation. ToRA generates code as an intermediate reasoning representation, executes it, and incorporates the output.

**Agent-environment interaction.** Computer use agents interact with GUI environments: they observe screen states, issue click/type/scroll actions, and receive updated screens as observations. WebArena provides a benchmark for multi-step web navigation tasks. This is structurally identical to the standard agent loop but with pixel-level environment observations.

### Multi-Agent Coordination
{: #multi-agent}

Multi-agent systems coordinate specialised agents toward shared goals, distributing context across a network rather than concentrating it in one model.

**Communication protocols.** Agent-to-agent communication requires shared message schemas. FIPA ACL and KQML are classical agent communication languages. Modern frameworks like A2A (Agent-to-Agent Protocol) and ANP (Agent Network Protocol) provide standardised APIs for agent discovery, capability advertisement, and message passing. [MCP](https://modelcontextprotocol.io/specification/2025-03-26) standardises tool/resource exposure between agents and orchestrators.

**Orchestration patterns.**

| Pattern | Structure | When to use |
|---------|-----------|-------------|
| Hierarchical | Orchestrator decomposes tasks, dispatches to specialised sub-agents, aggregates results | Complex multi-step tasks with clear decomposition |
| Peer-to-peer | Agents communicate directly, negotiate roles | Tasks requiring dynamic role assignment |
| Blackboard | Shared workspace all agents read/write | Incremental, collaborative problem building |

**Coordination strategies.** ChatDev assigns agents to software development roles (PM, developer, tester) with structured inter-agent dialogues. MetaGPT uses SOPs (standard operating procedures) to govern agent behaviour. AgentOrchestra coordinates specialist agents via a central orchestrator that manages task allocation, progress tracking, and result synthesis.

**Context synchronisation** is the core engineering challenge in multi-agent systems: ensuring all agents have consistent views of shared state while avoiding context overflow in any individual agent. Centralised shared memory is efficient but creates context bottlenecks. Decentralised per-agent context with explicit synchronisation messages reduces overflow but adds latency. The right balance depends on coordination frequency and knowledge-sharing requirements.

---

## Nine Context Engineering Patterns
{: #pattern-catalogue}

Every context engineering system is ultimately an implementation of one formula:

```
cₜ = f(instructions, task, retrieval, memory, tools, observations)
```

But "how does each slice of state enter the runtime context?" has many different answers. The nine patterns below are a taxonomy of those answers — each with its primary state type, key components, best fit, and dominant failure mode.

**Pattern taxonomy:**
- **Evidence patterns** (RAG, Agentic RAG) — how external knowledge enters context
- **Action patterns** (Tool/MCP, Workflows) — how live systems and typed pipelines feed context
- **Persistence patterns** (Memory) — how state persists across turns and sessions
- **Control patterns** (Agentic loops, Subagents) — how state is maintained across steps and agents
- **Substrate patterns** (File-native, Long-context) — the physical medium context lives in

### Pattern 1: RAG
{: #pattern-rag}

**What it is.** Retrieve external evidence for the current task and place it into context before generation. The flow: `user task → retrieve → pack evidence → grounded answer`.

**Best fit.** Questions whose answers depend on private, fresh, dynamic, or too-large-to-train knowledge, where the main job is answer synthesis (not multi-step reasoning over the retrieval).

**Six components:**

| Component | Role |
|---|---|
| Corpus + ingestion | Determine what content is searchable at all |
| Chunking + metadata | Set retrieval granularity and filter fields |
| Embeddings + lexical signals | Define the similarity space and exact-match support |
| Retriever + index | Generate candidate evidence under latency constraints |
| Reranker | Choose which evidence deserves the context budget |
| Prompt packer | Dedupe, order, and cite the final evidence payload |

**Dominant failure modes:**
- Wrong corpus or wrong chunk size (retrieved evidence doesn't contain the answer)
- Good recall but poor reranking (right documents, wrong passages in context)
- Redundant chunks wasting the token budget
- No citation or abstention policy (model hallucinates rather than saying "I don't know")
- Evidence is present in context but the model ignores it ("lost in the middle")

> **Interview question:** A RAG system has high recall@10 (the answer is in the top 10 retrieved chunks 90% of the time) but poor end-to-end answer accuracy. What are the likely failure points?
>
> *High recall but poor accuracy means the problem is downstream of retrieval. Three common culprits: (1) Reranking: the correct chunk is in the top 10 but ranked #8–10, so the prompt packer's budget limit excludes it — or the packer takes top-3 by embedding score rather than by reranker score. Fix: use a cross-encoder reranker that sees the query + chunk jointly. (2) Prompt packing: even if the correct chunk enters context, it's buried among 9 redundant or noisy chunks. Models exhibit "lost in the middle" — accuracy drops for evidence placed in the middle of a long context window. Fix: order evidence with the most relevant chunk first (or last), dedupe near-identical chunks, and keep the smallest support set that grounds the answer. (3) Citation/abstention: the model extracts the answer from its parametric knowledge instead of from the retrieved evidence. This happens when the question is close to training-data distribution — the model "knows" an answer and ignores the provided evidence. Fix: prompt the model to explicitly cite source IDs in its answer, making unsupported claims visible. Add an abstention policy: if no chunk has a relevance score above a threshold, say "I couldn't find this in the provided sources."*

### Pattern 2: Agentic RAG
{: #pattern-agentic-rag}

**What it is.** Retrieval becomes a trajectory rather than a single lookup. The system can decompose the task, search, inspect results, revise the query, switch stores, and search again: `task → plan → search → inspect → revise → search…`.

**Best fit.** Multi-hop questions, ambiguous store selection, incomplete first-pass evidence, and tasks where retrieval strategy must adapt online.

**Key components beyond plain RAG:**

- **Task decomposer/planner** — breaks the query into sub-questions
- **Query rewriter + store router** — rewrites based on what was found; selects among web, file, SQL, graph stores
- **Evidence judge** — "do I have enough to answer, or should I search more?"
- **Stopping rule + budget manager** — prevents infinite search loops
- **Final grounded synthesizer** — combines evidence from multiple search passes

**Dominant failure modes:**
- Search loops (no stopping rule; the system keeps searching without converging)
- One bad early query anchoring the entire trajectory (the rewriter never escapes a wrong initial framing)
- Evidence found in intermediate steps dropped before the final answer (poor state management across search turns)

> **Interview question:** In agentic RAG, the system searches, inspects results, then decides to search again. How do you prevent the system from looping indefinitely, and how do you ensure intermediate findings aren't lost?
>
> *Two separate problems — stopping and state management. For stopping: implement a multi-condition stopping rule. The system stops when: (a) the evidence judge scores sufficient confidence (e.g., "can I answer all sub-questions with current evidence?"), (b) a hard budget is reached (max N search steps, max M tokens of retrieved evidence), or (c) the last K queries returned no new information (diminishing returns detection via semantic deduplication of new vs. existing evidence). For state management: don't carry raw search results in the main transcript. After each search step, extract and store structured "findings" — compressed claims with source provenance — in a notebook object. The next search step reads the notebook, not the full transcript, to determine what's already known. This both reduces context bloat and prevents intermediate findings from being overwritten by later context. The key architectural choice: use a durable notebook as the accumulation medium, not the conversation transcript. Evidence accumulated across 10 search steps as notebook entries takes far fewer tokens than 10 raw result blocks in a chat transcript.*

### Pattern 3: Tool / MCP Context
{: #pattern-tool-mcp}

**What it is.** The runtime context includes a registry of tools and resources the model can call. Tools expose live systems as both action interfaces and context sources. A tool pattern is fundamentally a **context pattern**: tool descriptions shape the action prior, and tool outputs become the next evidence layer.

Flow: `task + tool registry → model selects tool → tool call → observation enters context → next step`.

**Registry-side components:**
- Names, descriptions, and JSON schemas (the model's action prior depends entirely on these)
- Permissions and auth scope
- Capability discovery and versioning (which tools are available changes dynamically in MCP)
- Prompts, resources, and tools (MCP's three primitive types)

**Execution-side components:**
- Executor or server boundary
- Retries, timeouts, and rate limits
- Observation reduction and normalisation (raw API responses are too large for context)
- Provenance and trust labels
- Clear error feedback to the model (so it can retry with corrections)

**Dominant failure modes:**
- Too many overlapping tools (model can't choose; description similarity creates ambiguity)
- Vague descriptions that blur tool choice ("search" vs "search_web" vs "search_docs")
- Raw outputs pasted with no reduction (a 200KB API response is not useful context)
- Unsafe permissions or weak trust zoning (a tool granted write access when only read is needed)

> **Interview question:** You give an agent 50 tools. Task completion drops compared to giving it 10 tools. Why, and how do you fix it?
>
> *This is the tool overload problem. With 50 tools, several things go wrong: (1) Attention dilution: the tool schema section of the context is large. The model must attend over all 50 schemas to select one — at long context lengths, attention to any particular schema weakens. (2) Ambiguity: with 50 tools there are likely several overlapping in capability. "search_web," "web_search," "search_online," "browse" are four tools that do similar things. The model makes worse choices because the schema descriptions are insufficiently differentiating. (3) Format sensitivity: a 50-tool schema takes many tokens. Models show reduced instruction-following accuracy when the instruction section grows large. Fixes: (1) Dynamic tool loading: only inject the tool schemas relevant to the current task. A coding agent in a Python file doesn't need the Slack or calendar tools. Use a tool router to select the relevant subset (typically 5–15 tools) given the task brief. (2) Tool consolidation: merge functionally similar tools. "search" with a `backend` parameter replaces four separate search tools. (3) Hierarchical tool discovery: expose a `list_tools(category)` meta-tool; the model first queries what's available, then gets the specific schemas. (4) Better descriptions: benchmark which tool descriptions most reliably trigger correct selection on a held-out task set; iterate on wording.*

### Pattern 4: Workflow / Typed Pipeline
{: #pattern-workflow}

**What it is.** A fixed pipeline or DAG where each step emits a typed intermediate object and only selected fields are passed forward: `input → route → extract → validate → answer`.

**Best fit.** Repetitive, auditable tasks where typed intermediates, validators, and fallbacks matter more than open-ended exploration. Customer support triage, structured document extraction, compliance checks.

**Core pieces:**
- Routers and classifiers (route the input to the right pipeline branch)
- Extractors into typed schemas (pull structured data from unstructured input)
- Validators and normalizers (reject malformed intermediates before they propagate)
- Graders, retries, and fallbacks (handle the expected failure modes at each stage)
- Final generator or formatter

**Dominant failure mode.** The pipeline becomes too rigid: it handles the median case well but breaks on anything slightly outside the template. A structured-output extractor trained on common formats fails silently on rare document layouts. Fix: always include a fallback branch and evaluate coverage (what % of real inputs flow through each branch).

> **Interview question:** You've built a 5-stage typed pipeline for contract review. Stage 3 (clause extraction) sometimes fails silently — it produces a valid schema but with empty fields. How do you detect and handle this?
>
> *Silent failures are the hardest failure mode in typed pipelines because the schema validates but the semantics are wrong. Detection strategy: (1) Schema validation catches structural failures (missing required fields, wrong types) but not semantic failures (empty optional fields that should have values). Add semantic validators: business rules like "a contract must have at least one party name" or "effective_date must be before expiry_date." These run after schema validation and catch the silent failure cases. (2) Coverage metrics: track what % of processed contracts have each field populated. A drop in population rate for `clause_type` signals that stage 3's extractor is silently failing on new document formats. Set alerts on these metrics in production. (3) Confidence scoring: have stage 3 output a confidence score alongside the extracted fields. Route low-confidence extractions to a fallback (human review, alternative extractor, or a more capable but slower model). (4) Sentinel values: use explicit null vs. missing — empty string "" is different from null. The validator can reject extractions where fields that should have values are null, triggering a retry or escalation. The broader point: typed pipelines need semantic validators at every stage boundary, not just structural ones. "Valid schema" is a necessary but not sufficient condition for a correct intermediate.*

### Pattern 5: Memory-Augmented Context
{: #pattern-memory}

**What it is.** The system stores selected information across turns or sessions and retrieves it later, so the runtime context includes persisted prior state.

**Critical distinction.** Long context is what you load *now*. Memory is what you choose to *keep, index, refresh, and retrieve later*. Memory engineering is about write gates, not just retrieval.

**Four memory schema types:**

| Type | Contents | Examples |
|---|---|---|
| Episodic | Past interactions with this user or task | "User prefers Python over JavaScript" |
| Semantic | Domain facts the agent has learned | "Product X was deprecated in v3.2" |
| Procedural | Workflow hints and conventions | "Always check the staging DB before prod" |
| Artifact | Pointers to produced outputs | "Summary of Q4 report is at /artifacts/q4.md" |

**What makes a good memory item.** Durable preferences, project conventions, reusable workflow hints, and links to past artifacts. What makes a bad memory item: transient chat, stale guesses, and giant transcript dumps.

**Five memory system components:**
1. **Write gate** — determines what is worth storing (not everything, not nothing)
2. **Schema** — profile / preference / episode / artifact type determines how to index
3. **Index + retriever** — enables efficient lookup at read time
4. **Freshness / TTL / invalidation** — stale memories are worse than no memory
5. **Consolidator** — deduplicates and resolves conflicts between overlapping memories

**Dominant failure modes:**
- Memory is too noisy to help (the write gate stores everything)
- Stale items are never invalidated (a user's preference from 6 months ago overrides their current request)
- Useful artifacts are not indexed for later reuse

> **Interview question:** Your memory-augmented agent is writing memories correctly but end-to-end task quality hasn't improved. What's likely wrong with the read policy?
>
> *Write quality and read quality are independent failure surfaces. Common read-side failures: (1) Over-retrieval: the system loads all memories touching any entity in the query. A request about "Python error handling" retrieves 40 memories including tangentially related items from months ago. The agent's context is dominated by stale, low-relevance memories. Fix: use a reranker on retrieved memories (not just embedding similarity), and cap the number of memories in context. (2) Unconditional loading: some implementations always load the full user profile regardless of whether it's relevant. A factual lookup query ("what's the capital of France?") doesn't benefit from the user's coding preferences. Fix: make memory retrieval conditional — only load if the task type matches the memory schema. (3) Stale memory winning over fresh context: if a memory from a past session says "user prefers X" but the current session context contradicts this ("actually I've switched to Y"), the memory overrides the explicit current signal. Fix: apply recency weighting in retrieval, and let the system prompt instruct the model to prefer explicit in-context signals over retrieved memories when they conflict. (4) Missing freshness check: memories retrieved without any freshness signal may represent outdated facts. Add a `last_verified` field and exclude memories older than a domain-appropriate TTL from automatic retrieval.*

### Pattern 6: Agentic Plan-Act-Observe
{: #pattern-agent-loop}

**What it is.** The system maintains explicit state over time and repeatedly builds the next context from that state:

```
sₜ → cₜ → aₜ → oₜ → sₜ₊₁
```

The model is embedded inside a controller, not answering a one-shot prompt. This is the base architecture for all autonomous agents.

**State-side components:**
- Task object and constraints (the goal that doesn't change)
- Current plan and subgoals (updated as evidence arrives)
- Retrieved facts and memory (accumulated from prior steps)
- Artifacts and tool state (files written, code run, results stored)
- Remaining token, time, and action budget

**Control-side components:**
- Planner/replanner (can the current plan still reach the goal?)
- Action selector (which tool/action to take next given current state)
- Observation summariser (compress raw tool output before appending to state)
- Stop policy and escalation policy
- Critics or graders for self-correction

**Dominant failure modes:**
- Transcript bloat: raw observations appended directly instead of being converted to typed state
- Plans that never update after new evidence arrives
- Poor observation reduction (a 50KB shell output fully in context)
- No clear stopping rule (agent runs until context budget exhausted)

> **Interview question:** An agent runs 20 tool calls but produces the wrong answer. The transcript shows it found the correct information in step 8, but the answer in step 20 contradicts it. What went wrong?
>
> *This is the "observation lost in the middle" failure — the agent found the right evidence but it was overwritten or diluted by subsequent context. Two mechanisms cause this: (1) Transcript bloat: raw tool outputs from steps 9–19 pushed the step-8 finding to the "lost in the middle" zone of the long context. The model's attention weight on the step-8 finding dropped below the threshold needed to influence the final answer. Fix: after each step, extract findings into a structured state object. The state is compact, always-near-top-of-context, and the agent reads state rather than re-scanning the full transcript. (2) Plan drift: after step 8 provided the answer, the agent's plan didn't update. It continued executing based on an outdated plan that assumed the answer wasn't yet found. The replanner should have noticed the evidence judge condition was satisfied and triggered synthesis. Fix: after every observation, run the evidence judge: "does current state satisfy the stopping condition?" If yes, exit the action loop immediately. The deeper fix: the agent should maintain an explicit "key findings" section in its state that is append-only and always included at the top of the next context. This ensures that critical intermediate findings survive long action trajectories.*

### Pattern 7: Subagents & Context Partitioning
{: #pattern-subagents}

**What it is.** Multi-agent systems are primarily **context partitioning**: many smaller `cₜ` instances plus explicit bridges, not a magical extra intelligence layer. A subagent is a control loop with its own transcript and a narrower tool allowlist. A supervisor delegates via a compact handoff; workers read/write shared artifacts instead of dumping full child transcripts into the parent's context.

**Why subagents help:**
- **Specialisation**: different tool surfaces per agent (read-only retrieval vs. shell vs. browser) → fewer, clearer action options per call
- **Isolation**: risky or noisy tools run in a child transcript; the parent keeps a summary or artifact diff, not every intermediate observation
- **Parallelism**: several workers with separate histories tackle subtasks simultaneously; supervisor merges via artifacts
- **Staging**: retrieval produces vetted snippets that a coding subagent consumes — a deliberate handoff instead of one bloated thread

**The handoff packet.** The critical object in a multi-agent system is the handoff from supervisor to worker: objective, constraints, artifact IDs the worker should read, and acceptance checks the worker should satisfy before returning. Vague handoffs are the #1 failure mode.

**Core components:**
- Supervisor/dispatcher (owns high-level plan and merge policy)
- Handoff packet (goal, constraints, artifact IDs, done criteria)
- Separate context history per worker (or explicit fork/snapshot)
- Local tool registry and permissions per worker
- Shared artifact store or message bus
- Return contract (structured result or diff — not raw logs)

**Dominant failure modes:**
- Pasting the child's full transcript into the parent (destroys isolation, reproduces monolithic bloat)
- Ambiguous handoffs (no file pointers, no done criteria)
- Two workers writing the same artifact without merge rules
- Supervisor never refreshes its plan after workers return new evidence

> **Interview question:** You're building a deep research system. Should you use one large agent or a supervisor + specialised subagents? What's the context engineering argument for each?
>
> *The context engineering argument determines this, not the "intelligence" argument. One large agent: all evidence, plans, and observations share one transcript. After 30 tool calls, the transcript is 50K+ tokens. The agent must attend over all prior steps to make decisions. This causes: (1) lost-in-the-middle failures for evidence found early; (2) context-window exhaustion before the task is complete; (3) tool confusion — a single agent with 40 tools (web, file, code, database, browser) has a large tool schema section and makes worse tool selections. Supervisor + subagents wins when: the task naturally decomposes into independent subtopics (parallel research threads), when tool surfaces are distinct enough to warrant specialisation (a browser agent vs. a code agent have non-overlapping tool sets), and when you want the final report writer to consume a compact notebook of findings rather than a 100K-token raw trajectory. The argument against subagents: orchestration overhead. Every handoff is a potential ambiguity point. If subtopics are highly interdependent (finding from subtopic A changes the scope of subtopic B), a supervisor that only sees artifact diffs may miss the dependency. Rule of thumb: use one agent for tasks that are fundamentally sequential and tightly coupled; use supervisor + subagents when the task has 3+ independent parallel dimensions and each dimension needs a different tool surface.*

### Pattern 8: File-Native Context
{: #pattern-file-native}

**What it is.** The file system becomes the primary external memory substrate. The model navigates files, diffs, logs, and artifacts with tools instead of pasting large text blobs into the prompt. This is the dominant pattern for coding agents (Claude Code, Codex-class CLIs).

**Main shift from RAG.** Retrieval becomes navigation plus execution, not just semantic nearest-neighbour search. The model uses `grep`, `read`, `edit`, `test` tools to actively explore the repository — it doesn't receive pre-retrieved evidence.

**What actually enters context:**
- Selected file excerpts, not full repositories
- Current path and artifact references
- Command outputs and failing test results (compressed)
- Repo-local instructions and conventions (e.g. CLAUDE.md)

**Two-level context engineering for coding harnesses.** The harness decides which paths load, how search results pack, how tool outputs append, and which project-level instructions always prepend (often via files the team checks in). The visible "prompt" is only part of `cₜ`.

**Dominant failure modes:**
- Too much raw file content in the prompt (reading a 2,000-line file when only 10 lines are relevant)
- Weak navigation discipline (the agent opens files by guessing paths rather than using grep/search tools)
- Oversized command outputs (a 5,000-line build log injected raw into context)
- Repo rules loaded without scoping (project-wide rules that aren't relevant to the current task)

> **Interview question:** A coding agent is failing to find the relevant code to fix a bug. It keeps opening the wrong files. What context engineering improvements would help?
>
> *This is a navigation discipline problem. The agent's file selection strategy is failing — likely because it's guessing paths based on naming conventions rather than searching. Fixes at the context engineering layer: (1) Structured navigation tools: replace "open file at path" with "search codebase for symbol/pattern" as the primary entry point. A grep-first approach (find all occurrences of the error message or function name) gives the agent a concrete starting point with high precision. (2) Search result reduction: grep results should be formatted as (file, line, context snippet) triples, not raw file contents. The agent selects which files to fully open based on these triples — avoiding large file reads for files that match but aren't the primary location. (3) Repo structure context: include a lightweight repo map at the start of each step — top-level directory names and the purpose of key modules (derivable from directory names + README fragments). This gives the agent a mental model for navigating a novel codebase. (4) Workspace memory: if the agent has already found the relevant file in a previous step, store its path in a "working set" state field that's always in context. This prevents the agent from re-searching for files it already found. The deeper problem: the agent is using parametric knowledge ("bug is probably in auth module") when it should be using tool-mediated search. Context engineering fix: make search tools the lowest-friction action — fast response, compact results — so the agent naturally searches before opening.*

### Pattern 9: Long-Context Management
{: #pattern-long-context}

**What it is.** A large context window changes the question from "can it fit?" to "what should occupy scarce attention and token budget inside the window?" Bigger windows do not remove context engineering — they make packing, ordering, caching, and compaction even more important.

**Context window anatomy:**

```
[static prefix | task + recent state | evidence slice | volatile observations | compaction checkpoints]
```

Static content (system instructions, tool schemas) should be at the start (or end) to take advantage of prompt caching. Volatile observations go in the suffix and are evicted first.

**DeepSeek BrowseComp ablations.** A direct experiment: same model, same harness, different context management policy, 128K window. When tool-call history exceeds ~80% of window:

| Policy | Strategy | Pass@1 |
|---|---|---|
| No management | Hit window limit | ~51.4 |
| Discard-75% | Remove oldest 75% of tool history | higher |
| Discard-all | Clear all prior tool history, continue | strong |
| Summary | Compress overflowed trajectory, restart | ~67.6 (best serial) |
| Parallel-fewest-step | Run N independent traces, keep shortest | comparable via parallel compute |

A 16 percentage point gain from context management alone, on the same model. The harness changed; the weights didn't.

**Packing principles:**
- Static prefix vs. volatile suffix — stable content stays in prompt cache
- Salience scoring and selection — not all evidence deserves equal space
- Section ordering and deduplication — most relevant evidence first or last, never middle
- Cache boundaries and reuse points — structure context so prefix-cache hits are maximised

**Compaction techniques:**
- Transcript-to-state compaction (replace raw observations with typed state summaries)
- Rolling summaries or checkpoints (every N steps, summarise and discard the detailed history)
- Differential updates since last turn (only append what changed, not the full state)
- Eviction policy for old observations (LRU eviction vs. salience-weighted eviction)

> **Interview question:** Your agent is hitting the 128K context limit on long research tasks. You have three options: summarise the full history and restart, discard the oldest 75%, or run parallel shorter traces. How do you choose?
>
> *The right strategy depends on the task's temporal dependency structure. Summarise + restart is best when: the task has a coherent goal that the summary can preserve (the model needs the "why" of what was found, not the raw "what"), and when the cost of extra steps (summary generation + restart steps) is outweighed by quality gains. It's the most expensive but recovers the most context. Discard-oldest-75% is best when: recent tool calls are highly informative (the trajectory is converging) and early exploration steps are largely irrelevant to the current state. The risk: if critical evidence was found in the discarded early steps, the agent re-discovers it expensively or misses it entirely. Mitigation: extract findings from steps before discarding them (notebook pattern). Parallel-fewest-step is best when: the task is embarrassingly parallel (each trace explores an independent sub-question), and total compute budget allows multiple simultaneous traces. It's the highest-throughput option because you're not compressing — you're restarting fresh traces that reach the answer faster by not accumulating history. The answer in the BrowseComp experiments: Summary wins on quality (67.6 vs ~51.4 baseline), but Parallel-fewest-step is competitive if compute is cheap. For a production system on a cost budget: discard-oldest + notebook is the practical middle ground — you preserve findings while evicting raw trajectory.*

---

**Pattern comparison table:**

| Pattern | Primary state | Best fit | Main failure |
|---|---|---|---|
| RAG | External evidence | Grounded Q&A over corpora | Wrong evidence in context |
| Agentic RAG | Retrieval trajectory | Multi-hop search | Search loops |
| Tool / MCP | Live systems | Tasks needing actions or live state | Tool confusion |
| Workflow | Typed intermediates | Repetitive, auditable flows | Brittle rigidity |
| Memory | Persisted state | Continuity across sessions | Stale noise |
| Agentic loop | Controller state | Long-horizon tasks | Transcript bloat |
| Subagents | Partitioned local state | Specialised parallel workers | Giant handoffs |
| File-native | Files and artifacts | Code and large repos | Navigation chaos |
| Long-context | Window budget | Huge sessions with reuse | Noisy windows |

---

## Product Architectures
{: #product-architectures}

Shipped products compose these nine patterns into two recognisable families. Identifying which family a product belongs to tells you what runtime context objects make it work.

### Fast Answer Engines
{: #fast-answer-engines}

**Products.** ChatGPT Search, Perplexity Standard/Pro, [Perplexica](https://github.com/ItzCrazyKns/Perplexica)/Vane.

**End artifact.** Short cited answer + conversational follow-up state.

**Reference pipeline:**

<div class="post-flow" role="list" aria-label="Fast answer engine pipeline">
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Step 1 — Query → Task Brief:</strong> A gating classifier decides: does this need live search? Which vertical (weather, maps, finance, sports, code)? How deep (quick, pro, research)? What citation policy? Output is a typed brief object — a workflow intermediate that controls all later retrieval and generation.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Step 2 — Query rewriting + fan-out:</strong> The raw user question is rewritten into a standalone query, entities and dates expanded, then fanned out into multiple targeted queries across web search, file search, and vertical widgets. Even a "search answer" system performs agentic RAG: retrieve, inspect, then search again more specifically.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Step 3 — Evidence cards (not raw results):</strong> SERP titles, opened page passages, widget outputs are processed into structured evidence cards: source ID + compressed claim or quote + provenance metadata. The packer deduplicates near-identical sources, prefers primary sources, trims raw HTML, and keeps the smallest support set that grounds the answer. The runtime context is not "top 10 results" — it is a carefully reduced evidence object.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--green"><strong>Step 4 — Synthesis + citations + conversational reuse:</strong> Final context bundle: `{instructions, task brief, recent chat, evidence cards, widget slots}`. Model synthesises, hedges conflicts, binds claim spans to source IDs. Product renders citation sidebar. Next-turn follow-up reuses the conversation tail unless scope changes, triggering fresh retrieval.</span></div>
</div>

**Perplexity: one skeleton, three budgets.** Standard, Pro, and Research all share the same `brief → retrieve → pack → generate` skeleton. What differs is the intermediate object size:
- **Standard**: short brief, modest fan-out, compact evidence, conversational answer
- **Pro**: deeper loop, more source types, optional code execution, richer bundle
- **Research**: multi-pass search, notebook-style intermediates, exportable report

The chat shell is stable; the intermediate context object (short bundle vs. notebook + report) is what changes the experience. Perplexity's product lineup demonstrates that "fast answer engine" and "deep research engine" are not different model species — they are different context-engineering policies inside one product family.

**ChatGPT Search architectural signals** (from public docs):
- A gating classifier decides whether live search is needed
- A query-rewrite layer expands the question into search-friendly forms
- A provider router covers general web + specialised verticals (weather, stocks, sports, maps)
- The system sends additional more-specific queries after inspecting initial results (agentic RAG, not one-shot)
- Citation-preserving answer generator reuses the conversation tail for follow-up turns

**Pattern composition.** Fast answer engines are primarily: **Agentic RAG + Tool/MCP + Workflow**. Memory is lighter-weight and mainly supports conversational continuity rather than long-term project state.

> **Interview question:** You're building a search answer engine. You've implemented retrieval and synthesis. What's the most impactful context engineering improvement to add next?
>
> *Evidence card construction — step 3 in the fast answer pipeline — has the highest marginal impact on answer quality once basic retrieval and synthesis work. Most early implementations pass raw search results (SERP snippets, opened page text) directly to the synthesis model. This fails because: (1) Raw SERP snippets are optimised for human scanning, not model consumption — they often lack the specific claim that supports the answer. (2) Near-duplicate sources (10 different news articles reporting the same Reuters wire) dilute the evidence budget. (3) No citation bindings — without source IDs attached to specific claims in the evidence, the synthesis model cannot reliably bind its generated claims to sources. The citation appears in the answer but isn't grounded. Evidence card construction fixes this: extract the specific claim or passage that answers the question (not the full page), attach a source ID, deduplicate near-identical content via semantic similarity, and format the evidence as a structured object the synthesis model can cite by ID. The result: same retrieval quality, but the synthesis model sees a compact, non-redundant, citation-ready payload — answer quality and citation accuracy both improve significantly. After evidence cards, the next highest-impact addition is a freshness classifier: identify which queries need live search (news, prices, recent events) vs. which can be answered from cache, and skip retrieval for the latter. This reduces latency without quality loss for stable-fact queries.*

### Deep Research Engines
{: #deep-research-engines}

**Products.** ChatGPT Deep Research, Perplexity Research, [STORM](https://arxiv.org/abs/2402.14207), [Open Deep Research](https://github.com/langchain-ai/open_deep_research).

**End artifact.** A structured report with citations, table of contents, and activity trace — not a chat blurb.

**Reference pipeline:**

<div class="post-flow" role="list" aria-label="Deep research engine pipeline">
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Step 1 — Goal → Research Brief:</strong> Open-ended research requests are underspecified. The system clarifies scope and converts the chat into a focused brief: objective, audience, deliverable, comparison axes, preferred sources, allowed tools, time horizon, and acceptance checks. This brief is the stable anchor for all subsequent planning, budget allocation, citation policy, and stop conditions. It replaces a long ambiguous transcript with a typed intermediate state.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Step 2 — Supervisor + Subagents:</strong> A single browsing thread accumulates too much raw context. The system splits the task into subtopics and delegates to specialised workers (search/open, find/files, code/MCP). Workers return compact findings or artifact diffs — not full child transcripts — to the supervisor. This is the catalogue's agentic loop + subagent partitioning in productised form.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Step 3 — Notebook (not transcript):</strong> Raw pages and tool logs are too large. Findings are compressed into a notebook: (claim/finding, source list, extracted notes, section tags, confidence/conflict, open questions). The notebook is the durable intermediate that survives through report writing — citations must survive several more model calls. This is memory + file-native artifacts + long-context compaction combined.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--green"><strong>Step 4 — Report Writing + Validation:</strong> The writer consumes: research brief + notebook of findings + selected quotations. It does not re-read every source page. Validation checks: missing citations, unsupported claims, coverage gaps that need re-search. Product layer: sources-used display, activity history, export to Markdown/Word/PDF, option to reopen research when coverage is thin.</span></div>
</div>

**STORM (Stanford OVAL) — explicit deep research pipeline.** STORM publishes its actual pipeline (unlike closed products), making it the clearest reference implementation:

1. **Knowledge curation** — discover perspectives, simulate writer-expert conversations grounded in web sources
2. **Outline generation** — convert collected evidence into a hierarchical outline (the compaction boundary)
3. **Article generation** — write sections against outline + references
4. **Polish** — optional final cleanup

The key design insight: the outline is not just presentation — it is the compaction boundary. The final writer's context is `{brief, outline, relevant notebook entries}`, not `{full browsing trajectory}`. Pre-writing context objects matter more than the final prompt.

**Open Deep Research (LangChain)** — three-stage pipeline: Scope → Research → Write. Separate models can be configured for summarisation, research, compression, and final report writing. The architecture emphasises: use multi-agent only when the task parallelises cleanly, compress user chat into a brief before the heavy run, and keep writing after research (not in many poorly coordinated section agents). Their engineering repeatedly emphasises that context engineering is what keeps the agent from drowning in raw tool output.

**ChatGPT Deep Research architectural signals:**
- User can choose sources (public web, uploaded files, connected apps, specified sites)
- Three pre-research stages before the main run: clarification, prompt rewriting, then deep research
- User-reviewable and modifiable research plan
- Report includes: structured sections, citations, table of contents, sources-used list, activity history

The three pre-research stages are the clearest public confirmation that Deep Research is a staged context pipeline, not "one big agent prompt."

**Pattern composition.** Deep research engines are: **Agentic RAG + Tool/MCP + Workflow + Subagents + File/artifact layer + Long-context management** — all nine patterns are present, with the notebook as the linchpin that makes the architecture tractable.

> **Interview question:** Why does a deep research engine need a notebook intermediary? Why not just have the supervisor read all the raw source pages at report-writing time?
>
> *The notebook solves three compounding problems that make "read everything at report time" infeasible: (1) Volume: a research run that reads 200 sources at average 5,000 tokens each = 1M tokens of raw content. No current context window handles this. Even 128K-window models would need 8 passes. More fundamentally, the synthesis model's quality degrades with noisy long context — 1M tokens of raw web pages include advertising text, navigation menus, and irrelevant paragraphs. (2) Citation survival: during browsing, the agent extracts claim X from Source Y. If this is stored as a raw transcript entry and then summarised, the claim-to-source binding is often lost. The notebook explicitly stores `{claim: "...", source_id: "url_Y", confidence: 0.9}` — the binding is a first-class object that survives into the report. (3) Redundancy: 200 sources may contain 50 unique claims, with high overlap. The notebook deduplicates at write time (when the agent records a finding, it checks if it's already known) rather than at report time (when the synthesis model must deduplicate under pressure to produce coherent text). The notebook is also the basis for the "open questions" tracker — gaps in coverage that trigger additional search rounds. Without it, the supervisor has no compact view of what's known and what's missing, so stopping decisions are unreliable.*

### How to Build One
{: #build-order}

Both product families share the same build order. The temptation is to start with the agent loop; the right answer is to start with the brief.

**Recommended build order:**

<div class="post-flow" role="list" aria-label="Build order for search/research products">
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>1. Build the brief.</strong> Build the router that decides: quick search, deeper search, or report workflow. Get typed task classification right before adding retrieval complexity.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>2. Build evidence cards.</strong> Make retrieval outputs compact, deduped, and citation-ready before adding agentic complexity. A non-agentic RAG system with good evidence cards outperforms an agentic system with raw evidence packing.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>3. Add the notebook.</strong> Once tasks become multi-step, store findings as artifacts rather than replaying long transcripts. This is the point at which the system can handle research-depth tasks.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>4. Add supervisors and subagents.</strong> Only when the task truly parallelises or when tool contexts must be isolated. Every subagent boundary adds orchestration surface; add it only when the workload demands it.</span></div>
  <div class="post-flow__step"><span class="post-flow__bar post-flow__bar--green"><strong>5. Add evaluation and safety.</strong> Trace logging, claim-source checks, staged access to private tools, prompt caching, and compaction. The product isn't shippable until you can measure claim-source support, plan coverage, citation accuracy, and latency/cost/tool-call budgets — not just whether the final paragraph "sounds good."</span></div>
</div>

**What to evaluate** (beyond "does it sound good"):
- Claim-source support: is every factual claim in the answer grounded in a cited source?
- Plan coverage: does the research brief's scope get covered by the final report?
- Citation accuracy: do cited source IDs actually contain the claim attributed to them?
- Budget compliance: latency, cost, and tool-call counts against SLOs

**The core insight.** The real deliverable is not "a better prompt." It is a stable context pipeline that repeatedly constructs the right object for the next model call. Quality depends on selection, ordering, deduplication, and refresh policy — not on wording alone.

---

*For the system-design perspective on building agents in production — tool design, memory architecture, guardrails, observability — see [Agentic System Design](/blogs/agentic-system-design/).*
