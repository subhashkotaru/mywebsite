---
title: "Agentic AI Frameworks: LangChain, LangGraph, MCP & Memory"
date: 2026-04-22
display_order: 4
description: "Practical guide to building agentic AI systems — LangChain primitives, LangGraph stateful multi-agent graphs, the Model Context Protocol, and every memory pattern from in-context to vector stores — all with working code."
tags: [agents, langchain, langgraph, mcp, memory, llm, ai-engineering]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#what-is-agentic">What Makes a System Agentic</a></li>
    <li><a href="#langchain">LangChain</a>
      <ul class="post-toc-sublist">
        <li><a href="#lc-primitives">Core Primitives: Models, Prompts, Parsers</a></li>
        <li><a href="#lc-chains">Chains & LCEL</a></li>
        <li><a href="#lc-tools">Tools & Tool Calling</a></li>
        <li><a href="#lc-agents">Agents: ReAct & OpenAI Tools</a></li>
        <li><a href="#lc-retrieval">Retrieval & RAG</a></li>
        <li><a href="#lc-pitfalls">LangChain Pitfalls</a></li>
      </ul>
    </li>
    <li><a href="#langgraph">LangGraph</a>
      <ul class="post-toc-sublist">
        <li><a href="#lg-why">Why LangGraph Exists</a></li>
        <li><a href="#lg-concepts">Graph Concepts: Nodes, Edges, State</a></li>
        <li><a href="#lg-basic">Building a Basic Agent Graph</a></li>
        <li><a href="#lg-conditional">Conditional Edges & Routing</a></li>
        <li><a href="#lg-multi-agent">Multi-Agent Graphs</a></li>
        <li><a href="#lg-human-in-loop">Human-in-the-Loop</a></li>
        <li><a href="#lg-persistence">Checkpointing & Persistence</a></li>
        <li><a href="#lg-streaming">Streaming</a></li>
        <li><a href="#lg-pitfalls">LangGraph Pitfalls</a></li>
      </ul>
    </li>
    <li><a href="#mcp">Model Context Protocol (MCP)</a>
      <ul class="post-toc-sublist">
        <li><a href="#mcp-what">What MCP Solves</a></li>
        <li><a href="#mcp-architecture">Architecture: Hosts, Clients, Servers</a></li>
        <li><a href="#mcp-primitives">MCP Primitives: Tools, Resources, Prompts</a></li>
        <li><a href="#mcp-server">Building an MCP Server</a></li>
        <li><a href="#mcp-client">Connecting a Client</a></li>
        <li><a href="#mcp-pitfalls">MCP Pitfalls</a></li>
      </ul>
    </li>
    <li><a href="#memory">Memory Handling</a>
      <ul class="post-toc-sublist">
        <li><a href="#memory-taxonomy">Memory Taxonomy</a></li>
        <li><a href="#in-context">In-Context Memory</a></li>
        <li><a href="#external-memory">External Memory: Vector Stores</a></li>
        <li><a href="#episodic">Episodic Memory</a></li>
        <li><a href="#semantic">Semantic / Entity Memory</a></li>
        <li><a href="#procedural">Procedural Memory</a></li>
        <li><a href="#memory-langgraph">Memory in LangGraph</a></li>
        <li><a href="#memory-pitfalls">Memory Pitfalls</a></li>
      </ul>
    </li>
    <li><a href="#putting-together">Putting It Together: Full Agent Stack</a></li>
  </ul>
</nav>

---

## What Makes a System Agentic {#what-is-agentic}

A standard LLM call is stateless and single-step: prompt in, completion out. An **agentic system** adds:

1. **Tools** — the model can call external functions (search, code execution, APIs, databases)
2. **Multi-step reasoning** — the model decides what to do next based on intermediate results
3. **State** — context persists across steps and across sessions
4. **Control flow** — loops, branches, retries, and escalation to humans

```
Standard LLM:                    Agent:
                                 
  Prompt → LLM → Response        Prompt → LLM → "I need to search"
                                                    │
                                              Tool call: search()
                                                    │
                                              Result → LLM → "I need more info"
                                                    │
                                              Tool call: lookup()
                                                    │
                                              Result → LLM → Final answer
```

The word "agentic" covers a wide spectrum:

| System | Agentic? | Why |
|---|---|---|
| Single LLM call | No | No tools, no loop |
| LLM + one tool call | Barely | One step, no iteration |
| ReAct loop until done | Yes | Iterative reasoning + action |
| Multi-agent pipeline | Yes | Agents coordinating |
| Autonomous agent with memory | Fully | Persistent state, long-horizon goals |

The frameworks in this post — LangChain, LangGraph, MCP — each address a different layer of this stack.

---

## LangChain {#langchain}

LangChain is a framework for composing LLM calls with tools, memory, and retrieval. It provides standard interfaces across model providers and a library of integrations. Think of it as the stdlib for LLM applications.

```bash
pip install langchain langchain-openai langchain-anthropic langchain-community
```

### Core Primitives: Models, Prompts, Parsers {#lc-primitives}

#### Chat Models

LangChain wraps every model provider with the same interface:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Both implement the same BaseLanguageModel interface
gpt = ChatOpenAI(model="gpt-4o", temperature=0)
claude = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# Invoke with messages
from langchain_core.messages import HumanMessage, SystemMessage

response = gpt.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is 2+2?"),
])
print(response.content)   # "4"
print(response.usage_metadata)  # token counts
```

#### Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Template with variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}. Be concise."),
    MessagesPlaceholder(variable_name="history"),   # inject chat history here
    ("human", "{question}"),
])

# Format the prompt
formatted = prompt.format_messages(
    domain="machine learning",
    history=[],
    question="What is backpropagation?"
)

# Or use as part of a chain (next section)
chain = prompt | gpt
result = chain.invoke({
    "domain": "machine learning",
    "history": [],
    "question": "What is backpropagation?"
})
```

#### Output Parsers

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# String parser — strips metadata, returns raw string
chain = prompt | gpt | StrOutputParser()
text = chain.invoke({"domain": "ml", "history": [], "question": "..."})

# Structured output with Pydantic
class ResearchSummary(BaseModel):
    topic: str = Field(description="Main topic of the research")
    key_findings: List[str] = Field(description="3-5 key findings")
    confidence: float = Field(description="Confidence 0-1", ge=0, le=1)

# Method 1: with_structured_output (preferred, uses tool calling under the hood)
structured_llm = gpt.with_structured_output(ResearchSummary)
result = structured_llm.invoke("Summarise recent findings on attention mechanisms")
print(result.key_findings)   # typed list, not a string

# Method 2: JsonOutputParser with format instructions
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser(pydantic_object=ResearchSummary)
prompt_with_format = ChatPromptTemplate.from_messages([
    ("system", "Answer in JSON. {format_instructions}"),
    ("human", "{question}"),
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt_with_format | gpt | parser
```

---

### Chains & LCEL {#lc-chains}

**LCEL (LangChain Expression Language)** uses the `|` pipe operator to compose components. Every component implements `invoke`, `stream`, and `batch` — so the same chain works synchronously, streaming, or in parallel.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# Simple chain
chain = (
    ChatPromptTemplate.from_template("Translate to French: {text}")
    | llm
    | StrOutputParser()
)

result = chain.invoke({"text": "Hello, world!"})  # "Bonjour, le monde!"

# Stream output token by token
for chunk in chain.stream({"text": "Hello, world!"}):
    print(chunk, end="", flush=True)

# Batch multiple inputs in parallel
results = chain.batch([
    {"text": "Good morning"},
    {"text": "Good night"},
    {"text": "Thank you"},
])
```

**Sequential chain with intermediate steps:**

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Step 1: extract keywords from a document
extract_keywords = (
    ChatPromptTemplate.from_template(
        "Extract 5 keywords from this text as a comma-separated list:\n{text}"
    )
    | llm
    | StrOutputParser()
)

# Step 2: generate a title from the keywords
generate_title = (
    ChatPromptTemplate.from_template(
        "Generate a blog post title for these keywords: {keywords}"
    )
    | llm
    | StrOutputParser()
)

# Compose: pass original text AND extract keywords, then generate title
full_chain = (
    RunnablePassthrough.assign(keywords=extract_keywords)  # adds 'keywords' key
    | RunnableLambda(lambda x: {"keywords": x["keywords"]})
    | generate_title
)

title = full_chain.invoke({"text": "Deep learning advances in protein folding..."})
```

**Branching with RunnableBranch:**

```python
from langchain_core.runnables import RunnableBranch

# Route to different prompts based on content
classifier_chain = (
    ChatPromptTemplate.from_template("Is this a question or statement? Reply 'question' or 'statement' only: {text}")
    | llm
    | StrOutputParser()
)

question_chain = ChatPromptTemplate.from_template("Answer this question: {text}") | llm | StrOutputParser()
statement_chain = ChatPromptTemplate.from_template("Expand on this statement: {text}") | llm | StrOutputParser()

router = RunnableBranch(
    (lambda x: "question" in x["type"].lower(), question_chain),
    statement_chain,   # default
)

full_chain = RunnablePassthrough.assign(type=classifier_chain) | router
```

---

### Tools & Tool Calling {#lc-tools}

Tools are functions the LLM can decide to call. LangChain wraps them with descriptions so the model knows when to use them.

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def search_web(query: str) -> str:
    """Search the web for current information. Use for recent events or facts you're unsure about."""
    # In practice: call Tavily, SerpAPI, etc.
    return f"Search results for '{query}': [simulated results]"

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_stock_price(ticker: str) -> dict:
    """Get the current stock price for a ticker symbol like AAPL, GOOGL."""
    # In practice: call a financial API
    return {"ticker": ticker, "price": 150.23, "change": "+1.2%"}

# Bind tools to the model
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([search_web, calculate, get_stock_price])

# The model decides whether to call a tool
response = llm_with_tools.invoke("What is 23 * 47?")
print(response.tool_calls)
# [{'name': 'calculate', 'args': {'expression': '23 * 47'}, 'id': 'call_abc'}]

# Execute the tool call manually
if response.tool_calls:
    tool_call = response.tool_calls[0]
    tool_map = {"calculate": calculate, "search_web": search_web}
    tool_result = tool_map[tool_call["name"]].invoke(tool_call["args"])
    print(tool_result)  # "1081"
```

**Tool with validation and error handling:**

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional

class DatabaseQueryInput(BaseModel):
    table: str = Field(description="Table name to query")
    filters: dict = Field(default={}, description="WHERE clause filters as key-value pairs")
    limit: int = Field(default=10, ge=1, le=1000, description="Max rows to return")

def query_database(table: str, filters: dict = {}, limit: int = 10) -> str:
    """Query the internal database. Only use for user data, orders, and products tables."""
    allowed_tables = {"users", "orders", "products"}
    if table not in allowed_tables:
        return f"Error: table '{table}' not allowed. Allowed: {allowed_tables}"
    # ... actual DB query ...
    return f"[{limit} rows from {table} with filters {filters}]"

db_tool = StructuredTool.from_function(
    func=query_database,
    name="query_database",
    description="Query internal database tables for user, order, or product data.",
    args_schema=DatabaseQueryInput,
    return_direct=False,
)
```

---

### Agents: ReAct & OpenAI Tools {#lc-agents}

#### ReAct Agent

ReAct (Reasoning + Acting) loops: think → act → observe → think → act → ... until done.

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for information online."""
    return f"[Result for: {query}]"

@tool  
def python_repl(code: str) -> str:
    """Execute Python code and return the output. Use for calculations and data processing."""
    import io, contextlib
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(code, {})
    return output.getvalue()

tools = [search, python_repl]
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Pull standard ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # print Thought/Action/Observation
    max_iterations=10,     # prevent infinite loops
    handle_parsing_errors=True,
)

result = executor.invoke({
    "input": "What is the square root of the number of days in a leap year?"
})
# Thought: I need to find the number of days in a leap year
# Action: python_repl
# Action Input: print(366)
# Observation: 366
# Thought: Now I need to compute sqrt(366)
# Action: python_repl
# Action Input: import math; print(math.sqrt(366))
# Observation: 19.131...
# Final Answer: The square root of 366 (days in a leap year) is approximately 19.13
```

#### OpenAI Tools Agent (preferred for modern models)

Modern models (GPT-4o, Claude 3.5+) support native tool calling — more reliable than ReAct's text parsing.

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. Use tools when needed."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # tool calls + results go here
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=15)

# With conversation history
from langchain_core.messages import HumanMessage, AIMessage

result = executor.invoke({
    "input": "Now what's the cube root of that?",
    "chat_history": [
        HumanMessage(content="What is the square root of 366?"),
        AIMessage(content="The square root of 366 is approximately 19.13."),
    ]
})
```

---

### Retrieval & RAG {#lc-retrieval}

```python
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. Load documents
loader = PyPDFLoader("research_paper.pdf")
docs = loader.load()

# Also works with web pages
web_loader = WebBaseLoader("https://arxiv.org/abs/2301.07041")
web_docs = web_loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,       # overlap prevents losing context at boundaries
    separators=["\n\n", "\n", ". ", " ", ""],  # tries these in order
)
chunks = splitter.split_documents(docs)

# 3. Embed and store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

# 4. Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",          # maximal marginal relevance: diverse results
    search_kwargs={"k": 5, "fetch_k": 20},
)

# 5. RAG chain
system_prompt = """You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know, say you don't know. Be concise.

Context:
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

result = rag_chain.invoke({"input": "What is the main contribution of the paper?"})
print(result["answer"])
print(result["context"])   # retrieved chunks used
```

**Contextual compression retriever** — filter retrieved chunks to only the relevant parts:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)
# Retrieved chunks are trimmed to only the relevant sentences
```

---

### LangChain Pitfalls {#lc-pitfalls}

| Pitfall | What happens | Fix |
|---|---|---|
| **Over-abstracting with chains** | 5 layers of wrappers for a 10-line task | Use LCEL only when composition genuinely helps; call the API directly for simple tasks |
| **No `max_iterations`** | Agent loops forever, burns tokens | Always set `max_iterations=10` and `max_execution_time=60` |
| **Tool descriptions are vague** | Model calls wrong tool or wrong args | Write descriptions like docstrings: what it does, when to use it, example input format |
| **Catching all errors silently** | `handle_parsing_errors=True` swallows real failures | Log errors; distinguish retryable from fatal |
| **Chunk size too large** | Retriever returns chunks that dilute the answer | Tune chunk size empirically; 512–1024 tokens works for most dense text |
| **Not filtering retrieved docs** | Irrelevant chunks confuse the model | Add metadata filters; use MMR or reranker |
| **Synchronous in async context** | `executor.invoke()` blocks the event loop | Use `ainvoke()`, `astream()` throughout |

---

## LangGraph {#langgraph}

### Why LangGraph Exists {#lg-why}

LangChain's `AgentExecutor` is a black box: you hand it a list of tools and it runs until done. This is fine for simple tasks. It breaks when you need:

- **Cycles**: loop back to a previous step based on a condition
- **Parallel branches**: run tool A and tool B simultaneously, merge results
- **Human approval**: pause mid-execution and wait for a human to review
- **Persistent state**: resume a conversation days later from exactly where it stopped
- **Multiple specialised agents**: a planner delegates subtasks to specialist workers

LangGraph models agent execution as a **directed graph** where:
- **Nodes** are Python functions (LLM calls, tool calls, any logic)
- **Edges** define flow (fixed or conditional)
- **State** is a typed dict shared across all nodes

```bash
pip install langgraph langchain-openai
```

---

### Graph Concepts: Nodes, Edges, State {#lg-concepts}

```
State (TypedDict):            Graph:
  messages: list               
  next_step: str              START
  tool_results: dict            │
  iteration: int                ▼
                             [agent]  ← LLM decides what to do
                                │
                    ┌───────────┴───────────┐
              (call tool?)           (done?)
                    │                   │
                    ▼                   ▼
               [tools]               END
                    │
                    └──────────────────┘
                         (loop back)
```

**State** is the single source of truth — every node reads from it and returns updates to it. LangGraph merges the updates (using reducers you define) rather than replacing the whole state.

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # Annotated with operator.add means: append new messages, don't replace
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # These are replaced entirely on update
    next: str
    iteration: int
```

---

### Building a Basic Agent Graph {#lg-basic}

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator
import json

# ── State ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# ── Tools ──────────────────────────────────────────────────────────────
@tool
def search(query: str) -> str:
    """Search for information. Use for current events or unknown facts."""
    return f"[Search result for '{query}': relevant information found]"

@tool
def calculator(expression: str) -> str:
    """Evaluate math expressions."""
    return str(eval(expression))

tools = [search, calculator]
tool_map = {t.name: t for t in tools}

# ── LLM with tools bound ───────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ── Nodes (plain Python functions) ────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """Call the LLM. Returns an AIMessage, possibly with tool_calls."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState) -> dict:
    """Execute all tool calls from the last AIMessage."""
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool = tool_map[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],
        ))
    return {"messages": results}

# ── Routing function ────────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """Decide: call tools, or we're done?"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# ── Build the graph ────────────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END},
)

graph.add_edge("tools", "agent")   # after tools, go back to agent

app = graph.compile()

# ── Run ────────────────────────────────────────────────────────────────
result = app.invoke({
    "messages": [HumanMessage(content="What is 15% of 847?")]
})

for msg in result["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content[:100]}")
```

---

### Conditional Edges & Routing {#lg-conditional}

Conditional edges let you route to different nodes based on state. This is the mechanism for loops, branching, and retries.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage
import operator

class ResearchState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    research_quality: float     # 0.0 to 1.0
    iterations: int
    final_report: str

def researcher_node(state: ResearchState) -> dict:
    """Do research, assess quality."""
    # ... LLM call to gather information ...
    return {
        "messages": [...],
        "research_quality": 0.7,
        "iterations": state["iterations"] + 1,
    }

def writer_node(state: ResearchState) -> dict:
    """Write the final report."""
    # ... LLM call to synthesise findings ...
    return {"final_report": "..."}

def critic_node(state: ResearchState) -> dict:
    """Critique the research and set quality score."""
    # ... LLM call to evaluate quality ...
    return {"research_quality": 0.95}

# ── Routing logic ──────────────────────────────────────────────────────
def route_after_research(state: ResearchState) -> Literal["critic", "writer", END]:
    if state["iterations"] >= 5:
        return "writer"                    # max iterations reached
    if state["research_quality"] < 0.8:
        return "critic"                    # needs review
    return "writer"                        # quality good enough

def route_after_critic(state: ResearchState) -> Literal["researcher", "writer"]:
    if state["research_quality"] >= 0.8:
        return "writer"
    return "researcher"                    # go research more

graph = StateGraph(ResearchState)
graph.add_node("researcher", researcher_node)
graph.add_node("critic", critic_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("researcher")

graph.add_conditional_edges("researcher", route_after_research)
graph.add_conditional_edges("critic", route_after_critic)
graph.add_edge("writer", END)

app = graph.compile()
```

---

### Multi-Agent Graphs {#lg-multi-agent}

**Supervisor pattern**: one agent routes tasks to specialist workers.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage
import operator

llm = ChatOpenAI(model="gpt-4o", temperature=0)

class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str

WORKERS = ["researcher", "coder", "writer"]

# ── Supervisor: decides who works next ────────────────────────────────
def supervisor_node(state: MultiAgentState) -> dict:
    system = f"""You are a supervisor managing these workers: {WORKERS}.
Given the conversation, decide who should act next, or 'FINISH' if done.
Reply with ONLY the worker name or FINISH."""
    
    response = llm.invoke([
        SystemMessage(content=system),
        *state["messages"],
    ])
    
    next_agent = response.content.strip()
    return {"next_agent": next_agent}

# ── Worker nodes ───────────────────────────────────────────────────────
def make_worker(name: str, system_prompt: str):
    def worker_node(state: MultiAgentState) -> dict:
        response = llm.invoke([
            SystemMessage(content=f"You are {name}. {system_prompt}"),
            *state["messages"],
        ])
        # Wrap response with agent name so supervisor knows who spoke
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content=f"[{name}] {response.content}")]}
    return worker_node

researcher = make_worker("researcher", "Research and gather facts. Be thorough.")
coder = make_worker("coder", "Write clean Python code. Include docstrings.")
writer = make_worker("writer", "Write clear, concise prose. Summarise findings.")

# ── Routing from supervisor ────────────────────────────────────────────
def route_from_supervisor(state: MultiAgentState) -> str:
    next_agent = state["next_agent"]
    if next_agent == "FINISH":
        return END
    return next_agent

# ── Build graph ────────────────────────────────────────────────────────
graph = StateGraph(MultiAgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)
graph.add_node("writer", writer)

graph.set_entry_point("supervisor")

graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {"researcher": "researcher", "coder": "coder", "writer": "writer", END: END},
)

# All workers return to supervisor
for worker in WORKERS:
    graph.add_edge(worker, "supervisor")

app = graph.compile()

result = app.invoke({
    "messages": [HumanMessage(content="Write a Python script to compute Fibonacci numbers and explain how it works.")],
    "next_agent": "",
})
```

**Parallel subgraph pattern** — run multiple agents simultaneously:

```python
from langgraph.graph import StateGraph, END
import asyncio

# Run researcher and fact-checker in parallel, merge results
async def parallel_research(state):
    results = await asyncio.gather(
        researcher_chain.ainvoke(state),
        fact_checker_chain.ainvoke(state),
    )
    return {"messages": results[0]["messages"] + results[1]["messages"]}
```

---

### Human-in-the-Loop {#lg-human-in-loop}

LangGraph can pause execution at any node and wait for human input — useful for approval gates, ambiguous decisions, and sensitive actions.

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

class ApprovalState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    pending_action: str
    approved: bool

def plan_node(state: ApprovalState) -> dict:
    """Agent plans what action to take."""
    response = llm.invoke(state["messages"])
    # Extract the action the agent wants to take
    return {
        "messages": [response],
        "pending_action": response.content,
        "approved": False,
    }

def human_approval_node(state: ApprovalState) -> dict:
    """This node pauses and waits for human input."""
    # LangGraph interrupts HERE and returns control to the caller
    # The caller provides input via .update_state() then resumes with .invoke()
    pass   # body doesn't matter — the interrupt happens before this node runs

def execute_node(state: ApprovalState) -> dict:
    """Only reached if approved."""
    if not state["approved"]:
        return {"messages": [AIMessage(content="Action cancelled by user.")]}
    # ... execute the action ...
    return {"messages": [AIMessage(content=f"Executed: {state['pending_action']}")]}

def route_after_plan(state: ApprovalState) -> str:
    return "human_approval"

def route_after_approval(state: ApprovalState) -> str:
    if state["approved"]:
        return "execute"
    return END

# ── Compile with checkpointer (required for interrupts) ───────────────
checkpointer = MemorySaver()

graph = StateGraph(ApprovalState)
graph.add_node("plan", plan_node)
graph.add_node("human_approval", human_approval_node)
graph.add_node("execute", execute_node)
graph.set_entry_point("plan")
graph.add_edge("plan", "human_approval")
graph.add_conditional_edges("human_approval", route_after_approval)
graph.add_edge("execute", END)

# interrupt_before: pause BEFORE entering these nodes
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_approval"],
)

# ── Usage ──────────────────────────────────────────────────────────────
thread_config = {"configurable": {"thread_id": "thread-001"}}

# Run until the interrupt
state = app.invoke(
    {"messages": [HumanMessage(content="Delete all logs older than 30 days")]},
    config=thread_config,
)
print("Agent wants to:", state["pending_action"])
print("Waiting for approval...")

# Human reviews and approves
human_approved = True   # in practice: CLI prompt, Slack message, web form

# Inject human decision into state, then resume
app.update_state(
    config=thread_config,
    values={"approved": human_approved},
)

# Resume from the interrupt point
final_state = app.invoke(None, config=thread_config)
```

---

### Checkpointing & Persistence {#lg-persistence}

Every graph invocation can be checkpointed so you can resume later — from a crash, after human input, or across sessions.

```python
from langgraph.checkpoint.memory import MemorySaver          # in-process (dev)
from langgraph.checkpoint.sqlite import SqliteSaver          # SQLite (single server)
from langgraph.checkpoint.postgres import PostgresSaver      # Postgres (production)

# Dev: in-memory (lost on restart)
memory_checkpointer = MemorySaver()

# Production: Postgres (pip install langgraph-checkpoint-postgres)
import psycopg
conn = psycopg.connect("postgresql://user:pass@localhost/langgraph")
pg_checkpointer = PostgresSaver(conn)
pg_checkpointer.setup()   # creates tables on first use

# Compile with checkpointer
app = graph.compile(checkpointer=pg_checkpointer)

# Each thread_id is an independent conversation
config_alice = {"configurable": {"thread_id": "user-alice-session-42"}}
config_bob   = {"configurable": {"thread_id": "user-bob-session-7"}}

# Alice's conversation
app.invoke({"messages": [HumanMessage("My name is Alice")]}, config=config_alice)
app.invoke({"messages": [HumanMessage("What is my name?")]}, config=config_alice)
# → "Your name is Alice" (remembers from same thread_id)

# Bob's conversation (completely separate state)
app.invoke({"messages": [HumanMessage("What is my name?")]}, config=config_bob)
# → "I don't know your name yet"

# Inspect checkpoint history
history = list(app.get_state_history(config=config_alice))
print(f"Number of checkpoints: {len(history)}")
for checkpoint in history[:3]:
    print(checkpoint.step, checkpoint.values["messages"][-1].content[:50])

# Get current state
current = app.get_state(config=config_alice)
print(current.values["messages"])

# Time-travel: resume from an earlier checkpoint
old_checkpoint = history[2]
app.invoke(
    {"messages": [HumanMessage("Let's go back to this point")]},
    config={**config_alice, "configurable": {"checkpoint_id": old_checkpoint.config["configurable"]["checkpoint_id"]}}
)
```

---

### Streaming {#lg-streaming}

```python
# Stream graph execution events
for event in app.stream(
    {"messages": [HumanMessage("Research quantum computing")]},
    config={"configurable": {"thread_id": "stream-test"}},
    stream_mode="updates",   # emit state updates from each node
):
    for node_name, node_output in event.items():
        print(f"\n── {node_name} ──")
        if "messages" in node_output:
            for msg in node_output["messages"]:
                print(msg.content[:200])

# Stream tokens as they're generated (for UI display)
async def stream_tokens():
    async for event in app.astream_events(
        {"messages": [HumanMessage("Explain diffusion models")]},
        config={"configurable": {"thread_id": "token-stream"}},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(chunk.content, end="", flush=True)

import asyncio
asyncio.run(stream_tokens())
```

---

### LangGraph Pitfalls {#lg-pitfalls}

| Pitfall | What happens | Fix |
|---|---|---|
| **State grows unboundedly** | `messages` list accumulates forever, hits context limit | Trim messages in a node: keep last N or summarise older ones |
| **No max_iterations guard** | Graph loops forever between two nodes | Add `iteration` counter to state; route to END if `>= limit` |
| **Using `MemorySaver` in production** | State lost on server restart | Use `SqliteSaver` or `PostgresSaver` |
| **Shared thread_id across users** | Users see each other's conversations | One `thread_id` per user session, stored in your auth layer |
| **Interrupt without checkpointer** | `interrupt_before` raises error without a checkpointer | Always compile with a checkpointer when using interrupts |
| **Nodes that mutate state directly** | Unpredictable behaviour; breaks checkpointing | Nodes must return dicts; never mutate `state` in place |
| **Parallel nodes writing same key** | Last write wins, earlier writes lost | Use reducers (`Annotated[list, operator.add]`) for keys written by multiple nodes |

---

## Model Context Protocol (MCP) {#mcp}

### What MCP Solves {#mcp-what}

Before MCP: every AI application that needed external tools had to write a custom integration. Want Claude to query your database? Write a custom tool. Want it to read files? Another custom integration. 100 AI apps × 100 data sources = 10,000 custom connectors.

MCP defines a **standard protocol** between AI applications (hosts) and external data/tool providers (servers). It's analogous to what USB is to hardware peripherals, or what LSP is to language servers in IDEs.

```
Before MCP:                          With MCP:
                                     
App A ──custom──→ Database           App A ──MCP──→ MCP Server ──→ Database
App A ──custom──→ Files              App B ──MCP──┘
App A ──custom──→ Slack              App C ──MCP──┘
App B ──custom──→ Database           
App B ──custom──→ Files              One MCP server, many clients.
...                                  Standardised protocol.
N×M integrations                     N+M integrations
```

MCP uses **JSON-RPC 2.0** over stdio (local) or HTTP+SSE (remote). An MCP server exposes capabilities; an MCP client (embedded in the AI host) discovers and calls them.

---

### Architecture: Hosts, Clients, Servers {#mcp-architecture}

```
┌─────────────────────────────────────────────────────┐
│  Host (Claude Desktop, Cursor, your app)            │
│                                                     │
│  ┌─────────────┐     ┌─────────────┐               │
│  │ MCP Client 1│     │ MCP Client 2│               │
│  └──────┬──────┘     └──────┬──────┘               │
│         │ JSON-RPC           │ JSON-RPC              │
└─────────┼────────────────────┼─────────────────────┘
          │                    │
          ▼                    ▼
   ┌─────────────┐      ┌─────────────┐
   │ MCP Server A│      │ MCP Server B│
   │ (Filesystem)│      │ (Database)  │
   └─────────────┘      └─────────────┘
   
Transports:
  Local: stdio (subprocess, stdin/stdout)
  Remote: HTTP + Server-Sent Events (SSE)
```

**Connection lifecycle:**
1. Host spawns or connects to MCP server
2. Client sends `initialize` request (protocol version, capabilities)
3. Server responds with its capabilities (tools, resources, prompts it provides)
4. Client can now call `tools/call`, `resources/read`, `prompts/get`
5. Session ends when host closes connection

---

### MCP Primitives: Tools, Resources, Prompts {#mcp-primitives}

**Tools** — functions the model can call (like LangChain tools):
```
tools/list     → list available tools with their JSON schemas
tools/call     → invoke a tool with arguments, get result
```

**Resources** — data sources the model can read (files, DB tables, API responses):
```
resources/list     → list available resources (URIs)
resources/read     → read a resource by URI
resources/subscribe → watch for resource changes (optional)
```

**Prompts** — reusable prompt templates the server provides:
```
prompts/list   → list available prompt templates
prompts/get    → get a rendered prompt with filled arguments
```

The key distinction: **tools** do things (side effects OK), **resources** expose data (read-only), **prompts** provide reusable instructions.

---

### Building an MCP Server {#mcp-server}

```bash
pip install mcp
```

```python
# server.py — a simple MCP server
import asyncio
import json
import sqlite3
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ── Create server ──────────────────────────────────────────────────────
app = Server("my-data-server")

# ── Tools ──────────────────────────────────────────────────────────────
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_database",
            description="Run a read-only SQL query on the application database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT query to execute"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return",
                        "default": 20
                    }
                },
                "required": ["sql"]
            }
        ),
        types.Tool(
            name="send_notification",
            description="Send a notification to a user by email.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "message": {"type": "string"},
                    "channel": {"type": "string", "enum": ["email", "slack"]},
                },
                "required": ["user_id", "message"]
            }
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "query_database":
        sql = arguments["sql"]
        limit = arguments.get("limit", 20)
        
        # Safety: only allow SELECT
        if not sql.strip().upper().startswith("SELECT"):
            return [types.TextContent(
                type="text",
                text="Error: only SELECT queries are allowed"
            )]
        
        conn = sqlite3.connect("app.db")
        try:
            cursor = conn.execute(sql + f" LIMIT {limit}")
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            result = {"columns": cols, "rows": rows, "count": len(rows)}
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error: {e}")]
        finally:
            conn.close()
    
    elif name == "send_notification":
        user_id = arguments["user_id"]
        message = arguments["message"]
        channel = arguments.get("channel", "email")
        # ... actual notification logic ...
        return [types.TextContent(
            type="text",
            text=f"Notification sent to {user_id} via {channel}"
        )]
    
    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

# ── Resources ──────────────────────────────────────────────────────────
@app.list_resources()
async def list_resources() -> list[types.Resource]:
    # Expose database tables as resources
    conn = sqlite3.connect("app.db")
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    
    return [
        types.Resource(
            uri=f"db://tables/{table[0]}",
            name=f"Table: {table[0]}",
            description=f"Schema and sample data for the {table[0]} table",
            mimeType="application/json",
        )
        for table in tables
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri.startswith("db://tables/"):
        table_name = uri.split("/")[-1]
        conn = sqlite3.connect("app.db")
        try:
            # Return schema
            schema = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchall()
            return json.dumps({
                "schema": schema,
                "sample_rows": sample
            }, indent=2)
        finally:
            conn.close()
    
    # Also expose local files
    if uri.startswith("file://"):
        path = Path(uri[7:])
        if path.exists() and path.is_file():
            return path.read_text()
    
    raise ValueError(f"Unknown resource URI: {uri}")

# ── Prompts ────────────────────────────────────────────────────────────
@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="analyse_table",
            description="Prompt to analyse a database table and suggest optimisations",
            arguments=[
                types.PromptArgument(
                    name="table_name",
                    description="Name of the table to analyse",
                    required=True,
                )
            ]
        )
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict | None) -> types.GetPromptResult:
    if name == "analyse_table":
        table_name = (arguments or {}).get("table_name", "unknown")
        return types.GetPromptResult(
            description=f"Analyse the {table_name} table",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"""Please analyse the '{table_name}' database table:
1. Read the table schema and sample data using the db://tables/{table_name} resource
2. Identify potential performance issues (missing indexes, wide rows, etc.)
3. Suggest SQL optimisations
4. Note any data quality concerns"""
                    )
                )
            ]
        )
    raise ValueError(f"Unknown prompt: {name}")

# ── Run the server ──────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

**Configure in Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "my-data-server": {
      "command": "python",
      "args": ["/path/to/server.py"],
      "env": {
        "DATABASE_URL": "sqlite:///app.db"
      }
    }
  }
}
```

---

### Connecting a Client {#mcp-client}

```python
# client.py — use an MCP server from your own application
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_mcp_server():
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env=None,
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialise the connection
            await session.initialize()
            
            # Discover available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Call a tool
            result = await session.call_tool(
                "query_database",
                arguments={"sql": "SELECT count(*) FROM users", "limit": 1}
            )
            print("\nQuery result:", result.content[0].text)
            
            # List and read a resource
            resources = await session.list_resources()
            print("\nAvailable resources:")
            for r in resources.resources:
                print(f"  - {r.uri}: {r.name}")
            
            resource_content = await session.read_resource("db://tables/users")
            print("\nUsers table info:", resource_content.contents[0].text[:200])

asyncio.run(use_mcp_server())
```

**Using MCP tools with LangChain:**

```python
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI

async def run_agent_with_mcp():
    server_params = StdioServerParameters(command="python", args=["server.py"])
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Load MCP tools as LangChain tools
            tools = await load_mcp_tools(session)
            
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data analyst. Use the available tools to answer questions."),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ])
            
            agent = create_tool_calling_agent(llm, tools, prompt)
            executor = AgentExecutor(agent=agent, tools=tools)
            
            result = await executor.ainvoke({
                "input": "How many users signed up last month?"
            })
            print(result["output"])

asyncio.run(run_agent_with_mcp())
```

---

### MCP Pitfalls {#mcp-pitfalls}

| Pitfall | What happens | Fix |
|---|---|---|
| **No auth on remote server** | Anyone can call your MCP server's tools | Add OAuth 2.0 / API key auth; MCP spec includes auth extension |
| **Tool descriptions too vague** | Model calls wrong tool or with wrong args | Write precise descriptions with example inputs in the docstring |
| **No input validation** | Model passes SQL injection in `sql` param | Validate args against JSON schema; whitelist operations |
| **Blocking calls in async server** | sqlite3 blocks the event loop | Use `asyncio.to_thread()` for synchronous DB calls |
| **Resource URIs not stable** | Cached URIs break when resources change | Make URIs deterministic; version them if schema changes |
| **Returning too much data** | 10MB JSON response in a tool call | Always paginate and limit; summarise large results |
| **stdio transport in production** | One process per client, no multiplexing | Use HTTP+SSE transport for multi-client production deployments |

---

## Memory Handling {#memory}

### Memory Taxonomy {#memory-taxonomy}

Agents need different kinds of memory for different purposes, mirroring human cognition:

```
┌────────────────────────────────────────────────────────────────┐
│                    Agent Memory Types                          │
│                                                                │
│  In-Context (short-term)                                       │
│  └─ The current conversation window: messages, tool results    │
│     Fast. Limited by context window. Lost after session.       │
│                                                                │
│  Episodic (what happened)                                      │
│  └─ Log of past conversations, actions, outcomes              │
│     Persisted externally. Retrieved by similarity or time.     │
│                                                                │
│  Semantic (what is known)                                      │
│  └─ Factual knowledge about entities, users, world state       │
│     Persisted as structured records or vector embeddings.      │
│                                                                │
│  Procedural (how to do things)                                 │
│  └─ Learned skills, preferred patterns, few-shot examples      │
│     Encoded in prompts, fine-tuning, or retrieved examples.    │
└────────────────────────────────────────────────────────────────┘
```

---

### In-Context Memory {#in-context}

The simplest form: include everything in the prompt. The model "remembers" by seeing it.

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing import List

llm = ChatOpenAI(model="gpt-4o")

class ConversationMemory:
    def __init__(self, max_messages: int = 20):
        self.messages = []
        self.max_messages = max_messages
    
    def add_turn(self, human: str, assistant: str):
        self.messages.append(HumanMessage(content=human))
        self.messages.append(AIMessage(content=assistant))
        # Sliding window: keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self, system_prompt: str = "") -> List:
        msgs = []
        if system_prompt:
            msgs.append(SystemMessage(content=system_prompt))
        msgs.extend(self.messages)
        return msgs

memory = ConversationMemory(max_messages=20)

def chat(user_input: str) -> str:
    msgs = memory.get_messages("You are a helpful assistant.")
    msgs.append(HumanMessage(content=user_input))
    response = llm.invoke(msgs)
    memory.add_turn(user_input, response.content)
    return response.content

print(chat("My name is Alice and I work at Anthropic."))
print(chat("What company do I work at?"))   # "You work at Anthropic."
```

**Summary memory** — compress old turns instead of dropping them:

```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

# Keeps last N tokens verbatim, summarises older turns
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=1000,   # summarise anything older than 1000 tokens
    return_messages=True,
)

# When buffer exceeds limit, calls LLM to summarise older messages:
# "The human introduced themselves as Alice, a researcher at..."
# This summary replaces the raw old messages
```

**Token budget management:**

```python
import tiktoken

def count_tokens(messages, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += len(enc.encode(msg.content)) + 4  # 4 tokens per message overhead
    return total

def trim_to_budget(messages, budget=100_000):
    """Remove oldest messages until total tokens fit budget."""
    while count_tokens(messages) > budget and len(messages) > 2:
        # Always keep system message (index 0) and last message
        messages.pop(1)
    return messages
```

---

### External Memory: Vector Stores {#external-memory}

For long-term memory across sessions, store memories as vector embeddings. Retrieve relevant memories by semantic similarity at query time.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from datetime import datetime
import uuid

class VectorMemoryStore:
    def __init__(self, persist_dir: str = "./memory_store"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.store = Chroma(
            collection_name="agent_memory",
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )
    
    def save(self, content: str, metadata: dict = {}):
        """Save a memory with metadata for filtering."""
        doc = Document(
            page_content=content,
            metadata={
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                **metadata
            }
        )
        self.store.add_documents([doc])
    
    def recall(self, query: str, k: int = 5, filter: dict = None) -> list[str]:
        """Retrieve k most relevant memories."""
        results = self.store.similarity_search(
            query,
            k=k,
            filter=filter,   # e.g., {"user_id": "alice"}
        )
        return [doc.page_content for doc in results]
    
    def recall_with_scores(self, query: str, k: int = 5, threshold: float = 0.7):
        """Only return memories above similarity threshold."""
        results = self.store.similarity_search_with_relevance_scores(query, k=k)
        return [
            (doc.page_content, score)
            for doc, score in results
            if score >= threshold
        ]

# Usage
memory = VectorMemoryStore()

# Store memories during conversation
memory.save(
    "User Alice prefers Python over JavaScript. She's a senior ML engineer at Anthropic.",
    metadata={"user_id": "alice", "type": "user_preference"}
)
memory.save(
    "Alice asked about RLHF in the previous session. Explained PPO and DPO.",
    metadata={"user_id": "alice", "type": "conversation_history"}
)
memory.save(
    "Alice is working on a project to reduce hallucinations in medical AI.",
    metadata={"user_id": "alice", "type": "user_goal"}
)

# Retrieve relevant memories for a new query
query = "What programming language does Alice prefer?"
relevant = memory.recall(query, k=3, filter={"user_id": "alice"})
for mem in relevant:
    print("Memory:", mem)

# Inject into prompt
def build_prompt_with_memory(user_query: str, user_id: str) -> str:
    memories = memory.recall(user_query, k=5, filter={"user_id": user_id})
    memory_context = "\n".join(f"- {m}" for m in memories)
    return f"""You are a helpful assistant. Here is what you remember about this user:

{memory_context}

User: {user_query}"""
```

**Choosing a vector store:**

| Store | Deployment | Best for |
|---|---|---|
| **Chroma** | In-process or server | Development, small scale |
| **FAISS** | In-process | Fast local similarity search |
| **Pinecone** | Managed cloud | Production, large scale |
| **Weaviate** | Self-hosted or cloud | Hybrid search (vector + keyword) |
| **pgvector** | Postgres extension | Already using Postgres |
| **Qdrant** | Self-hosted or cloud | High performance, filtering |

---

### Episodic Memory {#episodic}

Store full conversation episodes; retrieve and replay relevant past conversations as context.

```python
import json
import sqlite3
from datetime import datetime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

class EpisodicMemory:
    """Stores complete conversation episodes and retrieves relevant past sessions."""
    
    def __init__(self, db_path: str = "episodes.db", vector_dir: str = "./episode_vectors"):
        # SQLite for full episode storage
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                messages TEXT,        -- JSON serialised
                summary TEXT,
                outcome TEXT
            )
        """)
        self.conn.commit()
        
        # Vector store for semantic search over episode summaries
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Chroma(
            collection_name="episode_summaries",
            embedding_function=self.embeddings,
            persist_directory=vector_dir,
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini")
    
    def save_episode(self, user_id: str, session_id: str, messages: list, outcome: str = ""):
        """Summarise and store a completed conversation episode."""
        # Generate summary using LLM
        conversation_text = "\n".join([
            f"{m['role']}: {m['content']}" for m in messages
        ])
        summary_prompt = f"""Summarise this conversation in 2-3 sentences.
Focus on: what the user wanted, what was accomplished, key facts mentioned.

Conversation:
{conversation_text}

Summary:"""
        summary = self.llm.invoke(summary_prompt).content
        
        episode_id = f"{user_id}-{session_id}"
        now = datetime.utcnow().isoformat()
        
        # Store full episode in SQLite
        self.conn.execute(
            "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?)",
            (episode_id, user_id, session_id, now, now,
             json.dumps(messages), summary, outcome)
        )
        self.conn.commit()
        
        # Store summary embedding for semantic retrieval
        from langchain_core.documents import Document
        self.vector_store.add_documents([
            Document(
                page_content=summary,
                metadata={"episode_id": episode_id, "user_id": user_id, "timestamp": now}
            )
        ])
    
    def recall_similar_episodes(self, query: str, user_id: str, k: int = 3) -> list[dict]:
        """Find past episodes relevant to the current query."""
        results = self.vector_store.similarity_search(
            query, k=k, filter={"user_id": user_id}
        )
        episodes = []
        for doc in results:
            episode_id = doc.metadata["episode_id"]
            row = self.conn.execute(
                "SELECT summary, messages, ended_at FROM episodes WHERE id=?",
                (episode_id,)
            ).fetchone()
            if row:
                episodes.append({
                    "summary": row[0],
                    "messages": json.loads(row[1]),
                    "when": row[2],
                })
        return episodes
    
    def format_for_context(self, episodes: list[dict]) -> str:
        """Format retrieved episodes for injection into current prompt."""
        if not episodes:
            return ""
        parts = ["Relevant past conversations:"]
        for ep in episodes:
            parts.append(f"\n[{ep['when'][:10]}] {ep['summary']}")
        return "\n".join(parts)

# Usage
episodic = EpisodicMemory()

# After a session ends, save it
episodic.save_episode(
    user_id="alice",
    session_id="session-2024-01-15",
    messages=[
        {"role": "user", "content": "How do I set up DDP training in PyTorch?"},
        {"role": "assistant", "content": "Here's how to set up DistributedDataParallel..."},
    ],
    outcome="user successfully set up multi-GPU training"
)

# At the start of a new session, retrieve relevant past episodes
past = episodic.recall_similar_episodes(
    query="PyTorch distributed training",
    user_id="alice",
    k=2
)
context = episodic.format_for_context(past)
# Inject context into the system prompt for the new session
```

---

### Semantic / Entity Memory {#semantic}

Maintain a structured knowledge base about entities (users, organisations, concepts) that the agent updates over time.

```python
import json
import sqlite3
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class EntityMemory:
    """Maintains structured facts about entities, updated incrementally."""
    
    def __init__(self, db_path: str = "entities.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT,
                entity_type TEXT,
                facts TEXT,   -- JSON dict of fact_name → value
                updated_at TEXT,
                PRIMARY KEY (entity_id, entity_type)
            )
        """)
        self.conn.commit()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def extract_and_update(self, conversation: str, entity_id: str):
        """Extract new facts from a conversation and merge into entity memory."""
        existing = self.get_entity(entity_id) or {}
        
        extraction_prompt = f"""Extract factual information about user '{entity_id}' from this conversation.
Return a JSON object with keys like: name, job_title, company, preferences, goals, technical_skills, location.
Only include facts explicitly mentioned. Return {{}} if nothing new.

Existing known facts: {json.dumps(existing)}

Conversation:
{conversation}

New facts (JSON):"""
        
        response = self.llm.invoke([HumanMessage(content=extraction_prompt)])
        try:
            new_facts = json.loads(response.content.strip())
        except json.JSONDecodeError:
            new_facts = {}
        
        # Merge: new facts override old (except lists, which extend)
        merged = {**existing, **new_facts}
        self._save_entity(entity_id, "user", merged)
        return merged
    
    def get_entity(self, entity_id: str) -> dict:
        row = self.conn.execute(
            "SELECT facts FROM entities WHERE entity_id=?", (entity_id,)
        ).fetchone()
        return json.loads(row[0]) if row else {}
    
    def _save_entity(self, entity_id: str, entity_type: str, facts: dict):
        from datetime import datetime
        self.conn.execute(
            "INSERT OR REPLACE INTO entities VALUES (?,?,?,?)",
            (entity_id, entity_type, json.dumps(facts), datetime.utcnow().isoformat())
        )
        self.conn.commit()
    
    def format_for_prompt(self, entity_id: str) -> str:
        facts = self.get_entity(entity_id)
        if not facts:
            return ""
        lines = [f"What you know about this user:"]
        for key, val in facts.items():
            lines.append(f"  - {key}: {val}")
        return "\n".join(lines)

# Usage across sessions
entity_mem = EntityMemory()

# Session 1: user mentions their background
conv1 = "User: I'm a senior ML engineer at Anthropic working on RLHF. I prefer Python."
entity_mem.extract_and_update(conv1, "alice")
# Stored: {"job_title": "senior ML engineer", "company": "Anthropic", "focus": "RLHF", "preferences": {"language": "Python"}}

# Session 2: new info added automatically
conv2 = "User: I've started using PyTorch 2.0's compile feature for my training runs."
entity_mem.extract_and_update(conv2, "alice")
# Merged: previous facts + {"tools": ["PyTorch 2.0 compile"]}

# Use in prompt
print(entity_mem.format_for_prompt("alice"))
# What you know about this user:
#   - job_title: senior ML engineer
#   - company: Anthropic
#   - preferences: {"language": "Python"}
#   - tools: ["PyTorch 2.0 compile"]
```

---

### Procedural Memory {#procedural}

Store and retrieve successful approaches, workflows, and few-shot examples.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class ProceduralMemory:
    """Stores successful task patterns and retrieves them as few-shot examples."""
    
    def __init__(self, vector_dir: str = "./procedural_memory"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.store = Chroma(
            collection_name="procedures",
            embedding_function=self.embeddings,
            persist_directory=vector_dir,
        )
    
    def record_success(self, task_description: str, approach: str, outcome: str, rating: float = 1.0):
        """Record a successful task completion for future retrieval."""
        content = f"""Task: {task_description}
Approach: {approach}
Outcome: {outcome}"""
        self.store.add_documents([
            Document(
                page_content=content,
                metadata={"task": task_description, "rating": rating}
            )
        ])
    
    def retrieve_examples(self, current_task: str, k: int = 3) -> str:
        """Get relevant past successful approaches as few-shot examples."""
        results = self.store.similarity_search(current_task, k=k)
        if not results:
            return ""
        
        examples = ["Here are similar tasks you've handled successfully before:\n"]
        for i, doc in enumerate(results, 1):
            examples.append(f"Example {i}:\n{doc.page_content}\n")
        return "\n".join(examples)

# Pre-populate with known good patterns
proc_mem = ProceduralMemory()

proc_mem.record_success(
    task_description="Debug a Python script that hangs indefinitely",
    approach="1. Add timeout with signal.alarm. 2. Use py-spy to get stack trace. 3. Check for deadlock with threading.enumerate()",
    outcome="Found deadlock in database connection pool. Fixed by adding connection timeout."
)

proc_mem.record_success(
    task_description="Optimise a slow SQL query on a large table",
    approach="1. Run EXPLAIN ANALYZE. 2. Check for sequential scans. 3. Add index on filter columns. 4. Consider partitioning.",
    outcome="Query went from 45s to 0.3s by adding composite index on (user_id, created_at)."
)

# At agent start, inject relevant examples
examples = proc_mem.retrieve_examples("My Python service is stuck and not responding")
# Returns the debugging example above as context
```

---

### Memory in LangGraph {#memory-langgraph}

LangGraph's `thread_id` checkpointing handles short-term (within-session) memory automatically. For long-term cross-session memory, integrate external stores into your nodes.

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

# ── State ──────────────────────────────────────────────────────────────
class MemoryAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_id: str
    recalled_memories: list[str]

# ── Initialise memory stores ───────────────────────────────────────────
vector_memory = VectorMemoryStore("./long_term_memory")
entity_memory = EntityMemory("entities.db")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ── Memory recall node: runs at start of each turn ───────────────────
def recall_node(state: MemoryAgentState) -> dict:
    """Retrieve relevant long-term memories before the agent responds."""
    user_query = state["messages"][-1].content
    user_id = state["user_id"]
    
    # Semantic recall
    relevant = vector_memory.recall(user_query, k=5, filter={"user_id": user_id})
    
    # Entity facts
    entity_facts = entity_memory.format_for_prompt(user_id)
    
    all_memories = relevant + ([entity_facts] if entity_facts else [])
    return {"recalled_memories": all_memories}

# ── Agent node: uses recalled memories as context ─────────────────────
def agent_node(state: MemoryAgentState) -> dict:
    memory_context = "\n".join(f"- {m}" for m in state.get("recalled_memories", []))
    
    system = f"""You are a helpful assistant with long-term memory.

{f'What you remember about this user:{chr(10)}{memory_context}' if memory_context else ''}

Use your memories to give personalised, contextually aware responses."""
    
    msgs = [SystemMessage(content=system)] + list(state["messages"])
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}

# ── Memory save node: runs after agent responds ───────────────────────
def save_memory_node(state: MemoryAgentState) -> dict:
    """Extract and save important information from this turn."""
    user_id = state["user_id"]
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    last_ai = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        ""
    )
    
    if last_human and last_ai:
        # Save to vector store
        memory_text = f"User said: {last_human}\nAssistant responded: {last_ai[:200]}"
        vector_memory.save(memory_text, metadata={"user_id": user_id, "type": "conversation"})
        
        # Update entity memory
        conversation = f"User: {last_human}\nAssistant: {last_ai}"
        entity_memory.extract_and_update(conversation, user_id)
    
    return {}

# ── Tool execution node ────────────────────────────────────────────────
def tool_node(state: MemoryAgentState) -> dict:
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool = tool_map[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        from langchain_core.messages import ToolMessage
        results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": results}

def should_continue(state: MemoryAgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "save_memory"

# ── Build graph ────────────────────────────────────────────────────────
graph = StateGraph(MemoryAgentState)
graph.add_node("recall", recall_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("save_memory", save_memory_node)

graph.set_entry_point("recall")
graph.add_edge("recall", "agent")
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "save_memory": "save_memory",
})
graph.add_edge("tools", "agent")
graph.add_edge("save_memory", END)

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app = graph.compile(checkpointer=checkpointer)

# ── Run with persistent memory ─────────────────────────────────────────
def chat(user_id: str, message: str, session_id: str):
    config = {"configurable": {"thread_id": f"{user_id}-{session_id}"}}
    result = app.invoke({
        "messages": [HumanMessage(content=message)],
        "user_id": user_id,
        "recalled_memories": [],
    }, config=config)
    return result["messages"][-1].content

# Session 1
print(chat("alice", "I'm building a RAG system using Chroma.", "session-1"))

# Session 2 (new thread_id but long-term memory persists via VectorMemoryStore)
print(chat("alice", "What vector store should I use?", "session-2"))
# → Remembers Alice is building a RAG system, recommends Chroma or alternatives
```

---

### Memory Pitfalls {#memory-pitfalls}

| Pitfall | What happens | Fix |
|---|---|---|
| **Storing everything verbatim** | Vector store bloats; retrieval degrades (too many irrelevant memories) | Summarise before storing; only save non-trivial turns |
| **No TTL on memories** | Outdated facts (old job, old preferences) persist and mislead | Add `expires_at` metadata; let agent invalidate stale memories |
| **Retrieving too many memories** | Context window fills with irrelevant history | Retrieve top-3 to 5; score and filter below threshold |
| **No memory for anonymous users** | Every session starts cold | Use session-level episodic memory even without a user ID |
| **Injecting memories naively** | "Alice's dog died" injected in an unrelated coding session | Categorise memories; only inject relevant categories per task |
| **Writing to vector store on every token** | Massive embedding costs, slow saves | Save only at end of session; batch embedding calls |
| **No deduplication** | Same fact stored 50 times after 50 sessions | Check similarity before inserting; update existing entry if score > 0.95 |

---

## Putting It Together: Full Agent Stack {#putting-together}

Here's how all the pieces compose in a production agentic system:

```
User Request
     │
     ▼
┌────────────────────────────────────────────────────────────────────┐
│  LangGraph Graph                                                   │
│                                                                    │
│  recall_node                                                       │
│  ├── VectorMemoryStore.recall()   → recent relevant memories      │
│  └── EntityMemory.get_entity()    → structured user facts         │
│         │                                                          │
│  agent_node  (system prompt = memories + instructions)            │
│  ├── ChatOpenAI with bound tools (LangChain tools)                │
│  ├── MCP tools loaded via load_mcp_tools()                        │
│  └── Responds with tool calls or final answer                     │
│         │                                                          │
│  tools_node (if tool calls)                                       │
│  ├── LangChain tools: search, calculator, code_repl               │
│  └── MCP tools: database, filesystem, external APIs               │
│         │                                                          │
│  [human_approval_node]  ← optional interrupt for sensitive actions│
│         │                                                          │
│  save_memory_node                                                  │
│  ├── VectorMemoryStore.save()      → save turn for future recall  │
│  └── EntityMemory.extract_and_update() → update user facts        │
│                                                                    │
│  Checkpointer: PostgresSaver → resume any thread at any point     │
└────────────────────────────────────────────────────────────────────┘
     │
     ▼
Response + updated state persisted to Postgres
```

**Deployment architecture:**

```bash
# Start MCP server (tool provider)
python mcp_server.py &

# Start LangGraph server (agent executor)
# langgraph-cli handles the graph as an HTTP service
pip install langgraph-cli
langgraph up --config langgraph.json

# langgraph.json:
# {
#   "dependencies": ["./requirements.txt"],
#   "graphs": {"agent": "./agent.py:app"},
#   "env": ".env"
# }
```

**The right tool for the right job:**

| Need | Use |
|---|---|
| Simple LLM call with tools | LangChain LCEL + `bind_tools` |
| Looping agent with state | LangGraph |
| Multi-agent coordination | LangGraph supervisor pattern |
| Human-in-the-loop | LangGraph `interrupt_before` |
| Cross-session persistence | LangGraph checkpointer (Postgres) |
| Standardised tool server | MCP server |
| Short-term context | In-context messages (LangGraph state) |
| Long-term semantic memory | VectorMemoryStore (Chroma/Pinecone) |
| User profile/facts | EntityMemory (SQLite + LLM extraction) |
| Past conversation replay | EpisodicMemory (SQLite + vector index) |
| Successful patterns | ProceduralMemory (vector retrieval) |
