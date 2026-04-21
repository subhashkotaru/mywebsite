---
title: "Agentic System Design"
date: 2026-04-21
description: "Designing reliable LLM-powered agents — architectures, tool use, memory, multi-agent coordination, and the engineering challenges that make agents hard to productionise."
tags: [ml-systems, agents, llm, system-design]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#agent-loop">The Agent Loop</a>
      <ul class="post-toc-sublist">
        <li><a href="#react">ReAct</a></li>
        <li><a href="#plan-execute">Plan-and-Execute</a></li>
      </ul>
    </li>
    <li><a href="#tool-use">Tool Use</a>
      <ul class="post-toc-sublist">
        <li><a href="#function-calling">Function Calling</a></li>
        <li><a href="#tool-design">Tool Design Principles</a></li>
      </ul>
    </li>
    <li><a href="#memory">Memory</a>
      <ul class="post-toc-sublist">
        <li><a href="#in-context">In-Context Memory</a></li>
        <li><a href="#external-memory">External Memory & RAG</a></li>
        <li><a href="#episodic">Episodic & Procedural Memory</a></li>
      </ul>
    </li>
    <li><a href="#multi-agent">Multi-Agent Systems</a>
      <ul class="post-toc-sublist">
        <li><a href="#orchestration">Orchestration Patterns</a></li>
        <li><a href="#communication">Agent Communication</a></li>
      </ul>
    </li>
    <li><a href="#reliability">Reliability & Safety</a>
      <ul class="post-toc-sublist">
        <li><a href="#error-handling">Error Handling</a></li>
        <li><a href="#guardrails">Guardrails</a></li>
        <li><a href="#observability">Observability</a></li>
      </ul>
    </li>
    <li><a href="#structured-outputs">Structured Outputs</a></li>
    <li><a href="#infra">Infrastructure</a></li>
  </ul>
</nav>

---

## Overview
{: #overview}

A **language model** takes text in and produces text out — stateless, single-step. An **agent** wraps a language model in a loop that lets it take actions, observe results, and decide what to do next. The loop is what makes an agent an agent: it can browse the web, write and execute code, call APIs, read files, and coordinate with other agents to complete tasks that require many steps.

The promise is compelling. Tasks that would take a human hours — researching a topic across dozens of sources, writing and debugging a script, orchestrating a pipeline — can be delegated to an agent that runs autonomously. The reality is harder: agents fail in ways that are qualitatively different from single-step LLM calls. Errors compound across steps. The model's uncertainty is not always communicated. Tool calls have side effects. Long-running tasks hit context limits.

<div class="post-flow" role="group" aria-label="What makes an agent different from an LLM call">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Perceive — observe environment state (tool outputs, memory, user input)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Reason — decide what action to take next</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Act — call a tool, write output, or hand off to another agent</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Repeat — loop until task complete or termination condition met</span></li>
  </ol>
</div>

Building a reliable agent system requires decisions at every level: which reasoning loop, how to design tools, what memory architecture, how to handle failures, how to observe and debug the system. This post covers each.

---

## The Agent Loop
{: #agent-loop}

### ReAct
{: #react}

**ReAct** (Reasoning + Acting) is the foundational agent loop. At each step the model produces a structured trace interleaving thought and action:

```
Thought: I need to find the current population of Tokyo.
Action: web_search("Tokyo population 2025")
Observation: Tokyo metropolitan area population is approximately 37.4 million as of 2025.
Thought: I have the answer. I should return it.
Action: finish("Tokyo's population is approximately 37.4 million.")
```

The interleaving matters: the `Thought` step lets the model reason explicitly before committing to an action, and the `Observation` step grounds subsequent reasoning in real tool output rather than hallucinated facts. ReAct significantly outperforms chain-of-thought (which reasons but doesn't act) and direct tool use (which acts but doesn't reason) on multi-step tasks.

**Termination**: the loop ends when the model emits a `finish` action or a maximum step count is reached. Hard step limits are essential — without them, buggy agents can loop indefinitely burning API credits.

### Plan-and-Execute
{: #plan-execute}

ReAct is reactive — it decides one step at a time. For complex tasks, this leads to **myopic decisions** where early steps foreclose better approaches discovered later.

**Plan-and-Execute** separates planning from execution:

<div class="post-flow post-flow--compare" role="group" aria-label="ReAct vs plan-and-execute">
  <div class="post-flow__col">
    <p class="post-flow__col-label">ReAct</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Decide one action at a time</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Reactive to observations</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Lower latency per step</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Can get stuck in local loops</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Plan-and-Execute ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Planner produces full task decomposition upfront</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Executor carries out each step, re-plans on failure</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Better for long-horizon tasks with known structure</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Plan can become stale as observations arrive</span></li>
    </ol>
  </div>
</div>

A robust plan-and-execute agent includes a **re-planning trigger**: if an executor step fails or produces unexpected output, control returns to the planner with the new context. This handles the common case where a plan is partially wrong rather than completely wrong.

**Hierarchical planning**: for very complex tasks, the planner itself can delegate to sub-planners, each responsible for a sub-task. This creates a tree of agents — useful when sub-tasks are independent and can run in parallel.

---

## Tool Use
{: #tool-use}

Tools are the agent's interface to the world. Without tools, an agent can only reason over its training data. With tools, it can search the web, read documents, execute code, query databases, send messages, and call external APIs.

### Function Calling
{: #function-calling}

Modern LLM APIs (OpenAI, Anthropic, Gemini) expose **structured function calling**: the model is given a list of tool schemas (name, description, parameter types) and can respond with a structured tool call rather than plain text. The host application executes the call and returns the result.

```json
// Tool schema
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "City name"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
    },
    "required": ["location"]
  }
}

// Model response
{"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}
```

Structured function calling eliminates the need to parse free-text tool invocations — the model outputs valid JSON that the host can dispatch directly. Most production agents are built on this primitive.

**Parallel tool calls**: modern APIs allow the model to emit multiple tool calls in a single response when they are independent. The host executes them in parallel and returns all results at once, significantly reducing latency for tasks with independent sub-queries.

### Tool Design Principles
{: #tool-design}

Tool design is one of the highest-leverage decisions in agent system design. A well-designed tool is hard to misuse; a poorly designed one is a frequent source of agent failures.

**Atomic and composable**: each tool should do one thing. A tool that searches, summarises, and emails in a single call is hard for the model to reason about and impossible to retry at the right granularity.

**Idempotent where possible**: tools that have side effects (send email, delete file, post to API) should be designed so retries don't cause double-execution. Use a request ID or check-before-act pattern.

**Rich error messages**: when a tool fails, the error returned to the model should say *why* and *what to try instead* — not just "error 500". The model cannot recover from an opaque error but can often self-correct given a descriptive one.

**Explicit preconditions**: document what state the world must be in for the tool to succeed. A `read_file` tool that silently returns empty string for non-existent files will cause the model to hallucinate file contents rather than detecting the error.

| Anti-pattern | Better design |
|---|---|
| `search_and_summarise(query)` | Separate `search(query)` + `summarise(text)` |
| Silent failure (return `""`) | Return structured error with reason |
| No input validation | Validate and return descriptive error before execution |
| Unbounded output | Truncate with `truncated: true` flag so model knows |

---

## Memory
{: #memory}

A single LLM call has a context window — a fixed-size buffer of tokens the model can attend to. Agents need memory that persists across steps, across sessions, and across instances.

### In-Context Memory
{: #in-context}

The simplest memory: keep everything in the context window. The conversation history, tool outputs, and reasoning traces accumulate in the prompt. This is free — no external system needed — but hits context limits quickly and grows inference cost linearly with history length.

**Context management strategies**:
- **Truncation**: drop oldest messages when the context is full. Simple but loses potentially important history.
- **Summarisation**: compress older turns into a rolling summary. Loses detail but preserves semantics.
- **Selective retention**: score messages by relevance to the current task and keep only the most relevant. Requires a retrieval step.

For most agents, a hybrid works well: keep the last N turns verbatim, prepend a compressed summary of earlier history.

### External Memory & RAG
{: #external-memory}

**Retrieval-Augmented Generation (RAG)** externalises the knowledge base into a vector store. At query time, the agent embeds the query, retrieves the top-k most similar chunks, and injects them into the context:

<div class="post-flow" role="group" aria-label="RAG retrieval pipeline">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Embed query → dense vector</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Approximate nearest-neighbour search in vector store (FAISS, Pinecone, pgvector)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Retrieve top-k chunks + optional reranking</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Inject retrieved context into prompt → generation</span></li>
  </ol>
</div>

**Chunking strategy matters**: chunks that are too small lose surrounding context; chunks that are too large dilute the retrieved signal and bloat the prompt. Sentence-level and paragraph-level chunks with overlapping windows (stride < chunk size) are common. Metadata-aware chunking (keep table rows together, keep code blocks intact) outperforms naive character splitting.

**Hybrid retrieval**: combine dense (embedding) retrieval with sparse (BM25 keyword) retrieval, then fuse results with Reciprocal Rank Fusion (RRF). Dense retrieval handles semantic similarity; sparse handles exact keyword and entity matches. Neither alone is as robust as the combination.

### Episodic & Procedural Memory
{: #episodic}

Beyond knowledge retrieval, agents benefit from two additional memory types:

**Episodic memory** — a log of past task executions: what the task was, what steps were taken, what succeeded, what failed. Retrieved at the start of a new task to guide planning. Prevents the agent from repeatedly trying approaches that don't work.

**Procedural memory** — reusable skill programs: if the agent has solved a class of problem before (e.g. "query this database schema to answer questions"), the solution procedure can be stored and retrieved as a tool or prompt injection rather than re-derived from scratch each time.

Both are stored externally (vector store or key-value store) and retrieved by similarity to the current task context.

---

## Multi-Agent Systems
{: #multi-agent}

Single agents hit limits: context windows fill up, specialised tasks benefit from specialised models, and long-running workflows need parallelism. Multi-agent systems address this by distributing work across coordinated agents.

### Orchestration Patterns
{: #orchestration}

<div class="post-flow post-flow--compare" role="group" aria-label="Orchestration patterns">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Hierarchical (Manager-Worker)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Orchestrator breaks task into sub-tasks</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Delegates to specialised worker agents</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Aggregates results and decides next step</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Orchestrator is a bottleneck and single point of failure</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Peer (Handoff-Based)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Agents hand off directly to one another</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">No central coordinator — more resilient</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Each agent specialises in a domain</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Harder to reason about global task state</span></li>
    </ol>
  </div>
</div>

**Specialisation**: worker agents can be different model sizes (a cheap fast model for simple classification, a large expensive model for complex reasoning), different fine-tuned variants (a code agent fine-tuned on GitHub, a retrieval agent fine-tuned on documentation search), or different tool sets (a web agent vs a database agent).

**Parallelism**: independent sub-tasks can run concurrently across agents. A research agent, a data-gathering agent, and a fact-checking agent can all run simultaneously; the orchestrator waits for all to finish before synthesis.

### Agent Communication
{: #communication}

Agents communicate through structured messages. The message schema determines what context transfers between agents:

```python
class AgentMessage:
    task: str                    # what to do
    context: list[ContextItem]  # relevant background
    tools: list[ToolSchema]     # available tools
    constraints: dict            # time/cost/safety limits
    conversation_id: str         # for tracing
    parent_agent_id: str         # for hierarchy tracking
```

**Shared state vs message passing**: some systems use a shared blackboard (a dict or database all agents can read/write); others pass all state in messages. Shared state is convenient but creates race conditions under parallelism; message passing is safer but requires explicit handoffs.

**Context compression at handoff**: when an orchestrator delegates to a sub-agent, it should pass only the relevant context — not the entire conversation history. Passing too much wastes tokens and can confuse the sub-agent; passing too little loses necessary background.

---

## Reliability & Safety
{: #reliability}

Agents are harder to make reliable than single LLM calls. Errors compound: a wrong tool call in step 3 can cause every subsequent step to fail. Side effects accumulate: unlike a stateless inference call, an agent that sends emails or modifies files cannot be trivially retried.

### Error Handling
{: #error-handling}

**Retry with backoff**: transient tool failures (rate limits, network timeouts) should be retried automatically. Use exponential backoff with jitter. After N retries, return a structured error to the model so it can try a different approach.

**Self-correction**: when a tool call fails or returns unexpected output, include the error in the next prompt turn and let the model reason about how to recover. LLMs are surprisingly good at self-correcting given explicit error feedback — but only if the error is descriptive.

**Fallback tools**: for critical operations, register a fallback. If the primary web search tool fails, try a secondary. If the database query times out, try a cached result. The model should not be required to handle infrastructure failures.

**Step budget enforcement**: set a hard cap on the number of steps an agent can take per task. Track token usage per step and terminate early if the budget is exceeded. Return a partial result with a clear signal that the task was incomplete rather than silently timing out.

<div class="post-flow" role="group" aria-label="Error handling hierarchy">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Tool returns error → retry up to N times with backoff</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Retries exhausted → pass structured error to model for self-correction</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Self-correction fails → try fallback tool if available</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Fallback fails → escalate to human or return partial result with explanation</span></li>
  </ol>
</div>

### Guardrails
{: #guardrails}

Agents with access to external tools can cause real harm: deleting production data, sending emails to wrong recipients, making API calls that cost money. Guardrails prevent this:

**Confirmation gates**: for irreversible or high-stakes actions (delete, send, deploy), require explicit confirmation from the user or a separate approval agent before execution. This is a pause in the autonomy loop — the agent proposes, a human (or policy engine) approves.

**Action whitelisting**: define a strict set of allowed tool calls per task type. A customer support agent should never be able to call `delete_user`; a research agent should never be able to call `send_email`. Enforce at the tool dispatch layer, not just in the prompt.

**Scope limits**: constrain the agent to a defined environment. A file-system agent should operate only within a sandbox directory. A database agent should have read-only credentials unless write is explicitly required.

**Output filtering**: run agent outputs through a safety classifier before returning to the user. Catch prompt injection from tool outputs (malicious web pages that instruct the agent to ignore its system prompt), PII leakage, and harmful content.

### Observability
{: #observability}

An agent that fails silently is worse than no agent — you cannot debug what you cannot see.

**Trace every step**: log the full reasoning trace — thought, action, tool input, tool output, observation — for every step of every agent run. Attach a `trace_id` that links all steps of a single task and all sub-tasks in a multi-agent system.

**Latency and cost per step**: track time and token cost at each step. Steps with unusually high latency often indicate tool failures or infinite loops. Steps with unusually high token counts indicate context bloat.

**Success / failure classification**: tag each run as succeeded, failed (tool error), failed (model error), failed (timeout), or abandoned (user cancelled). Build dashboards over these tags to identify systematic failure modes — a particular tool that fails 30% of the time, a task type the agent consistently misunderstands.

**Replay**: store enough state to replay a failed run deterministically (same inputs, same tool outputs from cache) for debugging without re-executing expensive or side-effectful tools.

---

## Structured Outputs
{: #structured-outputs}

Agents that produce structured outputs — JSON, code, SQL, form fields — must produce *valid* structured outputs, not just text that looks like them. An agent that returns malformed JSON breaks the downstream system silently.

**Constrained decoding** enforces output validity at the token level: at each decoding step, mask the vocabulary to only tokens valid under the current grammar state. The model can only generate outputs that parse correctly. Libraries like Outlines, Guidance, and XGrammar implement this for JSON Schema, regex, and context-free grammars.

**Schema-first design**: define the output schema before writing the prompt. The schema is the contract between the agent and its caller. Use it to validate outputs at runtime and to generate test cases.

**Graceful degradation**: when the model produces invalid structured output despite constraints, have a fallback: re-prompt with the validation error, try a simpler schema, or return a plain-text response rather than crashing the caller.

---

## Infrastructure
{: #infra}

Production agent systems require infrastructure beyond a simple API wrapper.

**Async execution**: agent tasks are long-running (seconds to minutes). Expose them as async jobs: return a task ID immediately, poll for status, stream intermediate results via SSE or WebSocket. Never block an HTTP response thread for the full agent runtime.

**State persistence**: for multi-step tasks that may be interrupted, persist agent state (conversation history, tool outputs, current step) to durable storage after each step. This enables resume-on-failure without replaying expensive steps.

**Rate limiting and quotas**: agents can burn through API quotas rapidly, especially under bugs that cause loops. Implement per-task and per-user rate limits at the orchestration layer, separate from any limits enforced by the model provider.

**Sandboxing for code execution**: agents that write and execute code (a common and powerful capability) must run in isolated sandboxes — containers with no network access, restricted filesystem, CPU and memory limits, and timeout enforcement. A model that generates `rm -rf /` should not be able to execute it.

**Cost tracking**: track spend per agent run, per user, and per task type. Charge-back or quota enforcement requires accurate cost attribution. Cost anomalies (a run that spent 10× the expected budget) are often early signals of runaway loops or prompt injection.

<div class="post-flow" role="group" aria-label="Production agent infrastructure stack">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">API gateway — auth, rate limiting, cost tracking</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Task queue — async job dispatch, retry, priority</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Agent runtime — loop execution, tool dispatch, state management</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Tool layer — sandboxed execution, fallbacks, error normalisation</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Memory store — vector DB, KV store, episodic log</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Observability — trace store, dashboards, alerting</span></li>
  </ol>
</div>

---

## Also Read

**[Context Engineering for LLMs](/blogs/context-engineering/)** — a systematic survey of the formal discipline behind designing and managing the information payloads that govern LLM behaviour: context retrieval and generation (prompt engineering, RAG, dynamic assembly), context processing (long context architectures, self-refinement, multimodal), context management (memory hierarchies, KV cache compression, lost-in-the-middle constraints), and system implementations (modular/agentic/graph-enhanced RAG, memory systems, tool-integrated reasoning, multi-agent coordination).
