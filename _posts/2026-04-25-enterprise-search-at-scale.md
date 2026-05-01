---
layout: post
title: "Enterprise Search at Scale: M365 Copilot"
date: 2026-04-25
display_order: 13
description: "How Microsoft 365 Copilot Search works end-to-end — learning-to-rank, enterprise knowledge graphs, permission-aware retrieval, multi-stage RAG grounding, and the eval infrastructure that keeps it honest. Interview prep for Search Relevance / Applied Scientist roles."
tags: [search, rag, enterprise, ml-systems, information-retrieval, knowledge-graphs]
math: true
---

Enterprise search is not web search with a corporate VPN. The data is heterogeneous — files, emails, Teams threads, meeting transcripts, SharePoint pages, external connectors — owned by different people, governed by ACLs, freshness-critical, and sparse in user feedback. The query surface is natural-language work queries that share almost no vocabulary with the target document. And the retrieval result is not a link the user clicks — it is grounding context that an LLM uses to answer, where a wrong retrieval produces a wrong answer with a confident citation.

This is what Microsoft 365 Copilot Search is solving. This blog covers the full system: what makes enterprise retrieval different, learning-to-rank, enterprise knowledge graphs, permission-aware indexing, multi-stage RAG grounding, and the evaluation infrastructure that is honestly the hardest part of the job.

Prerequisite reading: [Search Fundamentals](/mywebsite/search) (BM25, dense retrieval, hybrid, HNSW, DiskANN, ColBERT) and [RAG](/mywebsite/rag) (chunking, GraphRAG, multi-hop, evaluation).

---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#enterprise-vs-web">Enterprise vs Web Search</a></li>
    <li><a href="#query-understanding">Enterprise Query Understanding</a>
      <ul class="post-toc-sublist">
        <li><a href="#query-taxonomy">Query Taxonomy</a></li>
        <li><a href="#entity-linking">Entity Linking in the Org Graph</a></li>
        <li><a href="#temporal-understanding">Temporal & Conversational Context</a></li>
        <li><a href="#query-rewriting">Query Rewriting Strategies</a></li>
      </ul>
    </li>
    <li><a href="#ltr">Learning to Rank</a>
      <ul class="post-toc-sublist">
        <li><a href="#ltr-fundamentals">Pointwise, Pairwise, Listwise</a></li>
        <li><a href="#lambdarank">LambdaRank & LambdaMART</a></li>
        <li><a href="#neural-ltr">Neural Learning-to-Rank</a></li>
        <li><a href="#click-models">Click Models & Bias Correction</a></li>
        <li><a href="#enterprise-ltr-signals">Enterprise Signals</a></li>
      </ul>
    </li>
    <li><a href="#knowledge-graph">Enterprise Knowledge Graph</a>
      <ul class="post-toc-sublist">
        <li><a href="#microsoft-graph">Microsoft Graph as Search Signal</a></li>
        <li><a href="#entity-extraction">Entity Extraction at Scale</a></li>
        <li><a href="#graphrag-deep">GraphRAG Internals</a></li>
        <li><a href="#graph-retrieval">Graph-Based Retrieval</a></li>
      </ul>
    </li>
    <li><a href="#permission-aware">Permission-Aware Retrieval</a>
      <ul class="post-toc-sublist">
        <li><a href="#acl-indexing">ACL Indexing Mechanics</a></li>
        <li><a href="#freshness-acl">Freshness & Permission Propagation</a></li>
        <li><a href="#secure-grounding">Secure Grounding for RAG</a></li>
      </ul>
    </li>
    <li><a href="#copilot-rag">Copilot as a RAG System</a>
      <ul class="post-toc-sublist">
        <li><a href="#retrieval-for-generation">Retrieval for Generation</a></li>
        <li><a href="#context-packing">Context Packing & Chunk Selection</a></li>
        <li><a href="#citation-quality">Citation Quality</a></li>
        <li><a href="#answerability">Answerability Detection</a></li>
      </ul>
    </li>
    <li><a href="#evaluation">Evaluation Infrastructure</a>
      <ul class="post-toc-sublist">
        <li><a href="#offline-eval">Offline Metrics & Annotation</a></li>
        <li><a href="#online-eval">Online Experiments & Click Signals</a></li>
        <li><a href="#rag-eval">RAG-Specific Evaluation</a></li>
        <li><a href="#failure-slicing">Failure Slicing</a></li>
      </ul>
    </li>
    <li><a href="#system-design">End-to-End System Design</a></li>
    <li><a href="#interview-questions">Interview Questions</a></li>
  </ul>
</nav>

---

## Enterprise vs Web Search {#enterprise-vs-web}

The hardest part of enterprise search is not scale — it is the fundamental mismatch between query language and document language, compounded by security constraints that prevent the usual shortcut of "use more data."

| Dimension | Web Search | Enterprise Search |
|---|---|---|
| Query language | Keywords, short phrases | Natural-language work queries, pronouns ("the deck Alex sent") |
| Document vocabulary | Rich, keyword-dense | Jargon, acronyms, project names unknown to any pretrained model |
| Feedback signal | Billions of clicks/day | Thousands of opens/saves/shares/day per tenant |
| Personalisation | Implicit from history | Explicit org graph: who you work with, what you own, what your team touched |
| Security | Public or login-gated | Per-document ACL, tenant isolation, need-to-know |
| Content types | HTML pages | Files, emails, chat, meetings, people, calendar, external connectors |
| Freshness | Hours | Minutes (Teams messages, emails) |
| Duplication | Near-dup web pages | Authoritative doc + 40 copies in different mailboxes |
| Ground truth | Click-through rates | Human judgments (clicks are too sparse) |

The vocabulary mismatch is severe. A user asks *"find the deck from the QBR about EMEA margin"*. The PowerPoint file is titled *"Q3 Business Review — EMEA Region"* and contains the word "margin" twice in 60 slides. BM25 score is near-zero. A pretrained dense model that has never seen this company's internal terminology scores it randomly. Neither works without adaptation.

---

## Enterprise Query Understanding {#query-understanding}

Before retrieving anything, you need to understand what the user actually wants. Enterprise queries are multi-dimensional — they express a content need, a person reference, a time reference, an intent, and often a conversational context from a prior Copilot turn.

### Query Taxonomy {#query-taxonomy}

Enterprise queries cluster into distinct types with different optimal retrieval strategies:

| Query type | Example | Best retrieval strategy |
|---|---|---|
| File search | "Find the budget spreadsheet from finance" | Metadata (file type, author, recency) + sparse |
| Person search | "Who is the PM for Azure Functions?" | People index, org graph |
| Decision search | "What did we decide about the onboarding latency?" | Dense semantic + Teams/email index |
| Status search | "What's the current status of Project Phoenix?" | Knowledge graph + recent docs |
| Expert search | "Who knows about COGS reduction?" | Entity graph, expertise model |
| Time-anchored | "What was shared after the QBR?" | Temporal filter + graph (who attended QBR) |
| Acronym/jargon | "Find the LTV deck from Rev Ops" | Lexical + domain adaptation |
| Conversational | "Actually, make that from last month" | Session context + reformulation |

Intent classification is the first gate. A people-search intent routed to the file index produces zero useful results. A classification model (fine-tuned BERT or T5) identifies intent, and the retrieval strategy is adjusted accordingly.

### Entity Linking in the Org Graph {#entity-linking}

Entities in enterprise queries are not Wikipedia entities. They are:
- **People**: "Alex", "the PM", "my manager", "the team"
- **Projects**: "Phoenix", "QBR", "the rebrand"
- **Orgs**: "finance", "Rev Ops", "the EMEA team"
- **Time references**: "last quarter", "after the reorg", "yesterday's standup"
- **File types**: "deck", "model", "tracker", "brief"

Entity linking maps these surface forms to nodes in Microsoft Graph — the org-wide knowledge graph of people, groups, documents, meetings, and their relationships.

```
Query: "Find the deck Alex shared after the QBR"
Entity linking:
  "Alex"  → user_id:abc123 (resolved via org graph, working-with signal)
  "QBR"   → event:Q3BusinessReview (resolved via calendar entity, Oct 15)
  "deck"  → file_type:PowerPoint
  "after" → temporal_filter: created_after:2024-10-15

Rewritten query:
  author:abc123 AND file_type:pptx AND created_after:2024-10-15
  + semantic query: "Q3 business review presentation"
```

**The working-with graph** is the key enterprise signal. If I search for "Alex," Microsoft Graph knows which Alex I work with most frequently (meetings, email threads, shared files) and resolves the ambiguity without me specifying a last name.

**Ambiguity resolution** is hard. "The PM" could be five people depending on what project you're talking about. "Finance" could be a team, a project, or a file tag. The system uses conversation history, the user's org position, and recent activity to disambiguate.

### Temporal & Conversational Context {#temporal-understanding}

Enterprise queries are heavily time-sensitive in two ways:

**Explicit temporal references:** "latest," "last quarter," "after the meeting," "before the reorg." These need to be normalized to absolute timestamps relative to the current time, resolved against calendar events and known org milestones.

**Conversational context:** In Copilot Chat, a follow-up query "show me more from that project" has no meaning without the prior turn. The system must maintain a session context window and reformulate each query to be self-contained before retrieval.

```python
class ConversationalQueryRewriter:
    """
    Reformulates follow-up queries using session context.
    """
    def __init__(self, llm):
        self.llm = llm
        self.session_history = []
    
    def rewrite(self, current_query: str) -> str:
        if not self.session_history:
            return current_query
        
        context = "\n".join([
            f"Turn {i+1}: Q: {h['query']} A: {h['summary']}"
            for i, h in enumerate(self.session_history[-3:])
        ])
        
        prompt = f"""Given this conversation:
{context}

Rewrite the following follow-up query to be fully self-contained
(resolve all pronouns, "that", "it", "same", temporal references):
Follow-up: {current_query}
Rewritten:"""
        
        rewritten = self.llm.generate(prompt, max_tokens=100)
        self.session_history.append({"query": current_query, "rewritten": rewritten})
        return rewritten

# Example:
# Turn 1: "Find the EMEA budget deck" → retrieves deck
# Turn 2: "Who created it?" 
# Rewritten: "Who created the EMEA budget deck?"
# Turn 3: "Show me their other files from last month"
# Rewritten: "Show files created by [author of EMEA budget deck] in [last month timestamp range]"
```

### Query Rewriting Strategies {#query-rewriting}

Multiple rewriting techniques can be applied in sequence:

**1. Term expansion with domain vocabulary.** Pretrained models do not know that "LTV" means "Lifetime Value" in your company's context, or that "Phoenix" is a project name not the city. A domain vocabulary extracted from the tenant's own documents (TF-IDF over titles, common acronym patterns) enables query expansion.

**2. HyDE (Hypothetical Document Embeddings).** Generate a synthetic document that would answer the query, embed it, use the embedding for dense retrieval. Works well for decision/status queries where the answer is a paragraph, not a keyword.

```python
def hyde_embedding(query: str, llm, embedder) -> np.ndarray:
    # Generate a hypothetical passage that would answer the query
    hypothetical = llm.generate(
        f"Write a short work document passage that answers: {query}",
        max_tokens=200
    )
    # Embed the hypothetical passage, not the query
    return embedder.encode(hypothetical)
```

**3. Sub-question decomposition.** Complex queries like "What did we decide about onboarding latency and who owns the action items?" decompose into: (a) retrieve decision about onboarding latency, (b) retrieve action items from same context, (c) find owners of those items.

**4. Step-back prompting.** For specific queries ("Why is sprint 42 delayed?"), step back to a more general query ("sprint 42 status updates") which retrieves more context, then use that context to answer the specific question.

---

## Learning to Rank {#ltr}

First-stage retrieval (BM25 + dense + metadata filters) returns hundreds of candidates. Learning-to-rank (LTR) is the second stage that decides the final ordering. In enterprise search, LTR is especially important because first-stage signals are noisy and the click feedback needed to train ranking models is sparse.

### Pointwise, Pairwise, Listwise {#ltr-fundamentals}

**Pointwise** — predict a relevance score for each (query, document) pair independently. Treat as regression or classification. Optimize: $\min \sum_{(q,d,r)} \text{Loss}(f(q,d), r)$ where $r$ is the relevance label.

Problem: ignores the ranking nature of the problem. A document scored 0.8 is not necessarily better than one scored 0.7 — what matters is their relative order.

**Pairwise** — for each query, optimize over pairs $(d_i, d_j)$ where $d_i$ is known to be more relevant. Optimize: $\min \sum_{(q, d_i \succ d_j)} \text{Loss}(f(q,d_i) - f(q,d_j))$. RankNet, RankSVM.

Problem: all pairs are treated equally, but swapping the top-2 results matters more than swapping results 50 and 51.

**Listwise** — optimize a list-level metric directly. ListNet optimizes KL-divergence between predicted and true permutation distributions. SoftRank differentiates through a smooth approximation of NDCG. LambdaRank (below) directly optimizes NDCG gradients.

### LambdaRank & LambdaMART {#lambdarank}

**LambdaRank** is the core algorithm behind Microsoft's RankNet/LambdaMART family — likely still in the enterprise ranking stack.

The key insight: you do not need to know *what* you are optimizing, only *which direction* to push each result. Define $\lambda_{ij}$ as the gradient signal for pair $(i, j)$:

$$\lambda_{ij} = \frac{\partial C}{\partial s_i} = -\frac{1}{1 + e^{s_i - s_j}} \cdot |\Delta\text{NDCG}_{ij}|$$

where $s_i, s_j$ are the model scores and $\lvert \Delta\text{NDCG}_{ij} \rvert$ is the absolute change in NDCG if $i$ and $j$ were swapped. The $\lvert \Delta\text{NDCG} \rvert$ factor weights pair gradients by how much the swap would matter — swapping rank 1 and 2 gets a large weight; swapping 40 and 41 gets a tiny weight.

The net gradient for document $i$:

$$\lambda_i = \sum_{j: i \succ j} \lambda_{ij} - \sum_{j: j \succ i} \lambda_{ij}$$

**LambdaMART** applies the LambdaRank gradients to train gradient-boosted trees (MART = Multiple Additive Regression Trees). Each tree learns a correction to the current ranking.

```python
# Pseudocode for LambdaMART training iteration
for query in training_queries:
    docs = first_stage_candidates(query)
    scores = current_model.predict(feature_vectors(query, docs))
    ranked = sort_by_scores(docs, scores)
    
    for i, doc_i in enumerate(ranked):
        lambda_i = 0.0
        for j, doc_j in enumerate(ranked):
            if relevance[doc_i] > relevance[doc_j]:  # i should rank above j
                delta_ndcg = compute_delta_ndcg(ranked, i, j)
                pair_grad = sigmoid(scores[j] - scores[i]) * abs(delta_ndcg)
                lambda_i += pair_grad
            elif relevance[doc_i] < relevance[doc_j]:  # j should rank above i
                delta_ndcg = compute_delta_ndcg(ranked, j, i)
                pair_grad = sigmoid(scores[j] - scores[i]) * abs(delta_ndcg)
                lambda_i -= pair_grad
    
    # Fit a regression tree to predict lambda_i for each doc
    tree.fit(feature_vectors(query, docs), lambdas)
    current_model.add_tree(tree, learning_rate=0.1)
```

**Features for enterprise LTR** (the feature engineering is where applied-science work lives):

| Feature category | Examples |
|---|---|
| **Textual match** | BM25 score, TF-IDF overlap, exact title match, acronym match |
| **Semantic** | Dense retrieval score, cross-encoder score |
| **Metadata** | File type match, author = query person entity, recency |
| **Graph/social** | Working-with score (how often user interacts with author), file shared in user's meetings |
| **Engagement** | Historical opens/saves for similar queries, dwell time |
| **Freshness** | Days since last modified, days since last viewed by anyone |
| **Authority** | Number of unique viewers, shares, replies |
| **Personalization** | User's past interaction with this file, this author, this topic |
| **Position** | Prior rank in BM25 results, prior rank in dense results |

### Neural Learning-to-Rank {#neural-ltr}

**Cross-encoder rerankers** are the most powerful point-level LTR models. They jointly encode (query, document) through a transformer and output a relevance score. Unlike bi-encoders (which encode query and document independently), cross-encoders can model query-document interaction at every token.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def rerank(self, query: str, candidates: list[str], top_k: int = 10) -> list[tuple]:
        pairs = [(query, doc) for doc in candidates]
        
        # Batch encode all pairs
        encoded = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True, truncation=True, max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            logits = self.model(**encoded).logits.squeeze(-1)
            scores = torch.sigmoid(logits).numpy()
        
        ranked = sorted(zip(scores, candidates), reverse=True)
        return ranked[:top_k]
```

**ColBERT late interaction** (already in your search blog, but the enterprise-specific point): the **token importance** problem. In enterprise docs, a few key tokens (the project name, the author's name, the decision keyword) are highly informative and the rest is boilerplate. Microsoft Research has published work on token importance weighting in multi-vector retrieval — learning which tokens' MaxSim scores to weight more heavily for enterprise queries.

$$\text{WeightedColBERT}(q, d) = \sum_{i \in q} w_i \cdot \max_{j \in d} \vec{q}_i \cdot \vec{d}_j$$

where $w_i$ is a learned importance weight for query token $i$ (e.g., "EMEA" gets high weight, "the" gets near-zero).

**Listwise neural rankers** like RankT5 (Google) and RankGPT use an encoder-decoder or LLM to directly generate the ranked permutation of candidates. Given a query and 20 candidates, the model outputs a ranked list. These are expensive but powerful for the top reranking stage.

### Click Models & Bias Correction {#click-models}

Enterprise search has sparse click data but it exists — file opens, email opens, document saves, reply-to-sender, meeting acceptances. Raw click-through rate is a **biased signal** because:

- **Position bias**: users are more likely to click higher-ranked results regardless of relevance
- **Presentation bias**: files with rich previews get more clicks
- **Trust bias**: documents from known colleagues get preferential opens

**Inverse Propensity Scoring (IPS)** corrects for position bias. Let $p_k$ be the probability a user examines position $k$ (the "propensity"). The unbiased relevance estimate for a clicked document at position $k$:

$$\hat{r} = \frac{\text{click}}{p_k}$$

Documents clicked at rank 10 (low examination probability) get upweighted; documents clicked at rank 1 (high examination probability) get downweighted.

**Estimating propensities** without randomization: use swap experiments (randomly swap pairs of results for a fraction of queries and observe click rate changes), or use the EM-based position-based click model.

**Counterfactual learning-to-rank** directly trains on logged data with IPS correction:

$$\mathcal{L}_\text{IPS} = -\sum_{(q, d, k, c)} \frac{c}{p_k} \cdot \log f(q, d)$$

where $c$ is the click indicator and $p_k$ is the position propensity.

### Enterprise LTR Signals {#enterprise-ltr-signals}

The most distinctive enterprise signals — things consumer search does not have:

**Working-with score.** For each (user, document_author) pair, compute a collaboration intensity score from: shared meetings, email thread co-participants, @mentions, file co-editors, shared group membership. A document from someone I work with closely ranks higher, all else equal.

```python
def working_with_score(user_id: str, author_id: str, graph_client) -> float:
    """
    Compute collaboration intensity from Microsoft Graph signals.
    """
    signals = graph_client.get_collaboration_signals(user_id, author_id)
    
    score = 0.0
    score += 0.3 * min(signals["shared_meetings_30d"] / 20, 1.0)
    score += 0.3 * min(signals["email_threads_30d"] / 50, 1.0)
    score += 0.2 * min(signals["shared_files_30d"] / 10, 1.0)
    score += 0.2 * min(signals["mentions_30d"] / 5, 1.0)
    return score
```

**File interaction history.** Has this user viewed, edited, commented on, or shared this exact file before? If yes, it is likely relevant for repeat-access queries.

**Meeting context.** Was this file shared in a meeting the user attended? Was this email thread part of a meeting invite chain?

**Recency decay.** Enterprise documents have a strong recency bias. A budget doc from last quarter is almost never the right answer when "latest" is implicit. Recency decay:

$$r_\text{fresh}(t) = r_0 \cdot e^{-\lambda (t_\text{now} - t_\text{modified})}$$

Different content types have different decay rates: Teams messages decay in days, budget docs decay in quarters, company policies decay in years.

---

## Enterprise Knowledge Graph {#knowledge-graph}

### Microsoft Graph as Search Signal {#microsoft-graph}

Microsoft Graph is the API surface over the organizational knowledge graph — people, files, meetings, emails, groups, and their relationships. For search, it provides signals that no amount of text analysis can produce:

- **Who created this file** and their organizational position
- **Who has accessed this file** and how recently
- **What meetings referenced this file** and who attended
- **What Teams channels discussed this topic** and which users are active there
- **What external data sources** are connected and what permissions apply

The Graph is queryable at search time for personalization and at index time for feature extraction.

### Entity Extraction at Scale {#entity-extraction}

To build search-useful knowledge from unstructured enterprise content, you need to extract structured entities from documents, emails, meetings, and chats at ingestion time.

**Entities to extract:**

| Entity type | Examples | Source |
|---|---|---|
| Person mentions | "@Alex", "the CEO", "our PM" | All content |
| Project/initiative | "Project Phoenix", "Q3 planning" | Docs, emails, chats |
| Decision | "We decided to...", "Agreed that..." | Meeting transcripts, emails |
| Action item | "John will...", "TODO:", "Follow up on..." | All content |
| Metric/KPI | "conversion rate", "p99 latency", "COGS" | Docs, slides |
| Date/time event | "the reorg", "last QBR", "sprint 42" | All content |
| External entity | Customer names, product names, competitor names | Docs, emails |

**Extraction pipeline:**

```python
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class ExtractedEntity:
    entity_type: str
    surface_form: str
    canonical_id: Optional[str]
    confidence: float
    context: str   # surrounding text

class EnterpriseEntityExtractor:
    """
    Extracts entities from enterprise content using an LLM.
    In production: fine-tuned NER model for latency, LLM for quality.
    """
    def __init__(self, llm, org_resolver):
        self.llm = llm
        self.org_resolver = org_resolver  # resolves names → user IDs
    
    def extract(self, text: str, doc_metadata: dict) -> list[ExtractedEntity]:
        prompt = f"""Extract named entities from this enterprise document.
Return JSON array with fields: entity_type, surface_form, context (surrounding sentence).
Entity types: PERSON, PROJECT, DECISION, ACTION_ITEM, METRIC, DATE_EVENT, EXTERNAL.

Document title: {doc_metadata.get('title', '')}
Author: {doc_metadata.get('author', '')}
Text: {text[:2000]}

JSON:"""
        
        raw = self.llm.generate(prompt, max_tokens=500)
        entities = json.loads(raw)
        
        result = []
        for e in entities:
            # Resolve PERSON entities to canonical user IDs
            canonical_id = None
            if e["entity_type"] == "PERSON":
                canonical_id = self.org_resolver.resolve(
                    e["surface_form"],
                    context=doc_metadata.get("author_dept")
                )
            
            result.append(ExtractedEntity(
                entity_type=e["entity_type"],
                surface_form=e["surface_form"],
                canonical_id=canonical_id,
                confidence=e.get("confidence", 0.8),
                context=e["context"],
            ))
        
        return result
```

**Building the document-entity index:** For each document chunk, store extracted entities as structured fields alongside the embedding. At query time, entity-based filtering augments semantic retrieval: "decision about onboarding latency" → retrieve chunks with entity_type=DECISION + semantic similarity to "onboarding latency."

### GraphRAG Internals {#graphrag-deep}

Microsoft Research's **GraphRAG** (Edge et al. 2024) goes beyond entity extraction to build a full knowledge graph and use it for retrieval. The pipeline:

**Phase 1 — Graph extraction.** For each document chunk, prompt an LLM to extract (entity, relationship, entity) triples:

```
Chunk: "Alex presented the Q3 results showing EMEA missed targets by 12%.
        Sarah suggested a new pricing strategy. The team agreed to review in 30 days."

Extracted triples:
(Alex) --[PRESENTED]--> (Q3 Results)
(EMEA) --[MISSED_TARGET_BY]--> (12%)
(Sarah) --[SUGGESTED]--> (New Pricing Strategy)
(Team) --[AGREED_TO_REVIEW]--> (New Pricing Strategy) [in:30 days]
```

**Phase 2 — Community detection.** Run Leiden algorithm (a graph clustering algorithm) over the entity graph to find communities — groups of highly-interconnected entities. Each community corresponds to a topic/project/initiative.

```python
import networkx as nx
from graspologic.partition import leiden

def build_entity_graph(triples: list[tuple]) -> nx.Graph:
    G = nx.Graph()
    for (e1, rel, e2, weight) in triples:
        if G.has_edge(e1, e2):
            G[e1][e2]["weight"] += weight
        else:
            G.add_edge(e1, e2, weight=weight, relations=[rel])
    return G

def detect_communities(G: nx.Graph) -> dict:
    # Leiden: better than Louvain for large graphs
    partition = leiden(G, trials=3)
    return partition  # {node: community_id}
```

**Phase 3 — Community summarization.** For each community, collect all triples and all source chunks, prompt an LLM to generate a summary:

```
Community: {Alex, Q3 Results, EMEA, Sarah, Pricing Strategy, ...}
Source chunks: [5 chunks about Q3 review meeting]
Summary: "Q3 business review. EMEA region missed targets by 12%. 
          Sarah proposed revised pricing strategy. Team reviewing options 
          with 30-day checkpoint. Key stakeholders: Alex (presenter), Sarah (strategy lead)."
```

**Phase 4 — Hierarchical indexing.** Index entity-level summaries, community-level summaries, and document-level summaries at different granularities. At query time:

- **Local queries** ("What did Sarah say about pricing?") → retrieve entity-level chunks
- **Global queries** ("What are the main themes in our Q3 review?") → retrieve community summaries
- **Mixed queries** → retrieve both, let LLM synthesize

**Phase 5 — Query-time retrieval.** Map query entities to graph nodes, retrieve their communities and summaries, augment the standard vector retrieval result:

```python
def graphrag_retrieve(query: str, entity_extractor, entity_graph, 
                       chunk_index, community_summaries) -> list[str]:
    # Standard dense retrieval
    dense_results = chunk_index.search(query, k=20)
    
    # Graph-based retrieval
    query_entities = entity_extractor.extract(query)
    graph_chunks = []
    
    for entity in query_entities:
        if entity.canonical_id in entity_graph:
            # Get neighbors in entity graph
            neighbors = list(entity_graph.neighbors(entity.canonical_id))
            community_id = entity_graph.nodes[entity.canonical_id].get("community")
            
            # Get community summary
            if community_id in community_summaries:
                graph_chunks.append(community_summaries[community_id])
            
            # Get chunks mentioning this entity and its neighbors
            entity_chunks = chunk_index.filter_by_entities(
                [entity.canonical_id] + neighbors[:5], k=5
            )
            graph_chunks.extend(entity_chunks)
    
    # Merge and deduplicate
    all_chunks = dense_results + graph_chunks
    return deduplicate_and_rank(all_chunks, query)
```

### Graph-Based Retrieval {#graph-retrieval}

Beyond GraphRAG, the organizational graph enables several retrieval patterns not possible with pure vector search:

**Expertise search.** "Who knows about COGS reduction?" — find people who have authored, co-authored, or heavily engaged with content about COGS. The entity graph has (Person) --[AUTHORED/ENGAGED]--> (Document) --[ABOUT]--> (Topic) edges. Traversal retrieves experts.

**Provenance search.** "Where did this number come from?" — trace a metric backward through document-entity-document links to find the source of record.

**Cross-document reasoning.** "What changed between the March and June forecasts?" — retrieve both documents via temporal + topic matching, compare their entities.

**Community of practice discovery.** "Find everyone working on ML infrastructure" — community detection clusters people by shared document activity.

---

## Permission-Aware Retrieval {#permission-aware}

This is the biggest difference between enterprise search and everything else. Every retrieved result must respect the querying user's permissions. Getting this wrong is a security incident.

### ACL Indexing Mechanics {#acl-indexing}

**The naive approach:** at query time, retrieve top-K results, then filter to those the user has permission for. Problem: if only 10% of results are accessible, you need to over-retrieve significantly (top-100 to get top-10 after trimming), which is expensive and may not even work if accessible results are rare.

**The better approach: ACL-aware indexing.**

Store ACL information with each document in the index. At query time, push the permission filter into the retrieval layer — only retrieve documents the user can access.

```
Document index record:
{
  "doc_id": "sharepoint_doc_abc",
  "embedding": [...],            // 768-dim vector
  "bm25_tokens": {...},          // sparse representation
  "acl": {
    "allowed_users": ["user:alice", "user:bob"],
    "allowed_groups": ["group:finance", "group:emea_team"],
    "denied_users": [],
    "inheritance": "parent_site",
    "sensitivity_label": "confidential"
  },
  "tenant_id": "contoso.com"
}
```

**Group membership expansion.** Users are members of groups, which are members of other groups (nested). For fast filtering, pre-expand each user's group memberships and cache the set. At query time, the filter becomes: `user_id ∈ acl.allowed_users OR any(group ∈ user_groups AND group ∈ acl.allowed_groups)`.

**The scale problem.** A large enterprise has 100K users, each potentially in hundreds of groups, across millions of documents with dynamic ACLs. Storing expanded ACLs per document doubles index size. Querying unexpanded ACLs requires group expansion at query time (expensive).

**Practical solution — BitSet ACL index:** For each group, maintain a bitset over document IDs indicating which documents that group can access. A user's accessible document set is the OR of their groups' bitsets. Bitset AND with retrieval results gives the permission-filtered set. This can be done efficiently with SIMD operations.

**DiskANN with filtering.** The FreshDiskANN variant supports predicate-filtered ANN search — the graph traversal skips nodes that don't satisfy the permission predicate, rather than post-hoc filtering.

### Freshness & Permission Propagation {#freshness-acl}

ACLs change constantly: documents are shared, unshared, moved to different sites, deleted, sensitivity labels are upgraded. The search index must reflect these changes promptly.

**Permission change events** (from Microsoft Graph change notifications) must trigger re-indexing of affected documents. The challenge: a group permission change can affect millions of documents (if the group is large). Batch re-indexing is too slow; you need **lazy invalidation** — mark affected documents as stale and re-evaluate permissions on demand.

```
ACL change event: group "all_employees" removed from document set X
→ Invalidate permission cache for all documents in set X
→ On next access attempt for any document in X: re-evaluate from source of truth
→ Async background job: re-index documents in X with updated ACL
```

**Document deletion.** A deleted document must be removed from the index immediately — a user searching after deletion should not find it, and the LLM should not ground answers on it. This requires real-time deletion events propagated to all index shards.

**Sensitivity label upgrades.** A document reclassified from "Internal" to "Confidential" must immediately have its accessible user set shrunk. This is a security-critical event — late propagation is a compliance violation.

### Secure Grounding for RAG {#secure-grounding}

In Copilot Chat, the retrieved chunks become grounding context for the LLM's answer. Two security constraints:

**1. No cross-user leakage.** The LLM must not include information from documents User A cannot access in a response to User A. This requires that the retrieval layer applies permission filtering *before* any content reaches the LLM context.

**2. No summary leakage.** Even if the LLM does not quote the document directly, it could implicitly reveal the document's content through the answer. This is harder to prevent and is an active research problem ("membership inference via LLM outputs").

**Practical guardrail:**
```
Rule: Only documents that pass the ACL filter for the querying user
      may appear in the LLM context window.
Rule: Citations in the LLM response must be linked to source documents,
      and those documents must be accessible to the response recipient.
Rule: If Copilot summarizes a document, the summary may only be shown
      to users who could access the original document.
```

---

## Copilot as a RAG System {#copilot-rag}

### Retrieval for Generation {#retrieval-for-generation}

When Copilot generates an answer, it needs grounding context — retrieved chunks that contain the information needed to answer correctly. This creates a different retrieval objective than search-result ranking:

| Search ranking | RAG grounding |
|---|---|
| Return the most relevant documents | Return the chunks most useful for answering this specific question |
| User evaluates results themselves | LLM uses chunks; wrong chunks → wrong answer |
| Top-10 results, user picks | Need exactly the right evidence |
| Diversity across sources is good | Duplicated chunks waste context window |
| Partial match is acceptable | Incomplete evidence causes hallucination |

**Answerability routing.** Before full retrieval, a lightweight classifier determines whether the query is answerable from enterprise data at all. "What is 2+2?" does not need retrieval. "Who are the top performers in Q3?" needs retrieval. "What should I have for lunch?" is not enterprise-relevant and should not trigger retrieval. This saves cost and latency.

### Context Packing & Chunk Selection {#context-packing}

The LLM has a fixed context window (128K tokens for GPT-4). Retrieved chunks must be selected and ordered to maximize the useful information density within that window.

**The chunk selection problem:** Given 50 retrieved chunks and a 4K-token budget, which 15 chunks do you include?

```python
def select_chunks_for_context(
    query: str,
    candidates: list[Chunk],
    token_budget: int,
    reranker,
    dedup_threshold: float = 0.85,
) -> list[Chunk]:
    
    # Step 1: Rerank by relevance to query
    scored = reranker.score(query, candidates)
    scored.sort(key=lambda x: x.score, reverse=True)
    
    # Step 2: Deduplicate — remove near-duplicate chunks
    selected = []
    selected_embeddings = []
    
    for chunk in scored:
        if selected_embeddings:
            # Check similarity to already-selected chunks
            sim = max(cosine_sim(chunk.embedding, e) for e in selected_embeddings)
            if sim > dedup_threshold:
                continue  # skip near-duplicate
        
        selected.append(chunk)
        selected_embeddings.append(chunk.embedding)
    
    # Step 3: Fill within token budget, respecting source diversity
    packed = []
    tokens_used = 0
    sources_seen = set()
    
    for chunk in selected:
        chunk_tokens = count_tokens(chunk.text)
        if tokens_used + chunk_tokens > token_budget:
            break
        packed.append(chunk)
        tokens_used += chunk_tokens
        sources_seen.add(chunk.source_id)
    
    # Step 4: Order by document position within source (preserves coherence)
    packed.sort(key=lambda c: (c.source_id, c.position))
    
    return packed
```

**Multi-hop retrieval.** Complex queries require chaining retrievals:

```
Query: "What are the open action items from the Azure Functions planning meeting?"
Hop 1: Retrieve: Azure Functions planning meeting (gets meeting metadata + transcript chunk)
       → Found: Meeting on 2024-03-15, attendees: [Alice, Bob, Carol]
Hop 2: Retrieve: action items from 2024-03-15 Azure Functions meeting
       → Found: 3 action item chunks from the transcript
Hop 3: Retrieve: current status of those action items
       → Found: email thread and task tracker entries
Synthesize: List open items with owners and current status
```

### Citation Quality {#citation-quality}

Copilot answers include citations — links to the source documents the answer is grounded in. Citation quality is a first-class metric: a citation should point to a document that actually contains the stated information, and the user should be able to click through and verify.

**Attribution granularity.** Ideally, a specific claim in the answer is linked to a specific chunk (paragraph, email, meeting segment) that contains that claim — not just the parent document. This requires chunk-level provenance tracking.

**Hallucination detection via citation checking.** After generation, a checking model verifies that each claim in the answer is supported by a cited chunk. Claims without citation support are flagged or removed.

```python
def verify_citations(answer: str, cited_chunks: list[Chunk], llm) -> list[dict]:
    """
    Check each sentence in the answer against cited chunks.
    Returns list of {claim, supported_by_chunk, confidence}.
    """
    sentences = split_into_claims(answer)
    results = []
    
    for claim in sentences:
        prompt = f"""Does the following document excerpt support this claim?
Claim: {claim}
Excerpts:
{chr(10).join(c.text for c in cited_chunks)}

Answer: supported / not_supported / partially_supported
Confidence (0-1):
Supporting excerpt (or none):"""
        
        response = llm.generate(prompt, max_tokens=100)
        results.append(parse_verification_response(response, claim))
    
    return results
```

### Answerability Detection {#answerability}

Before generating, the system should know whether it has enough evidence to answer correctly. If the retrieved chunks are tangentially related or insufficient, the system should say so rather than hallucinate.

**Answerability score** — a classifier that takes (query, retrieved_chunks) and predicts whether a grounded answer is possible:

```python
def answerability_score(query: str, chunks: list[Chunk], model) -> float:
    """
    Returns probability that the query is answerable from the retrieved chunks.
    0 = definitely not answerable, 1 = definitely answerable.
    """
    context = "\n".join(c.text for c in chunks[:5])
    prompt = f"""Question: {query}
Context: {context}

Is this question answerable from the context above?
Options: yes / partially / no
Answer:"""
    
    response = model.generate(prompt, max_tokens=10)
    return {"yes": 1.0, "partially": 0.5, "no": 0.0}.get(response.strip(), 0.3)
```

If answerability is below threshold (e.g., 0.4), Copilot responds with "I couldn't find enough information about that in your files" rather than generating an unsupported answer.

---

## Evaluation Infrastructure {#evaluation}

Evaluation is arguably the hardest and most important part of enterprise search. You cannot improve what you cannot measure, and measuring enterprise search quality is hard for every reason: sparse clicks, private data, subjective relevance, and the interaction between retrieval and generation quality.

### Offline Metrics & Annotation {#offline-eval}

**The annotation workflow.** Human judges evaluate (query, document) pairs for relevance using a graded scale:

| Label | Meaning |
|---|---|
| 4 — Perfect | Directly answers the query; user would immediately open this |
| 3 — Excellent | Highly relevant; user would very likely find this useful |
| 2 — Good | Relevant but not the best answer; useful secondary result |
| 1 — Fair | Weakly related; some relevant content but mostly noise |
| 0 — Bad | Not relevant; user would not find this useful |

For enterprise queries, you need judges who understand the organizational context. Options:
- **Crowdsourced judges** (MTurk, Scale AI): cheap but cannot understand private enterprise terminology
- **Internal judges** (employees who understand the org): expensive but accurate
- **Synthetic judges** (LLM-as-judge): scalable, surprisingly good for clear relevance distinctions

**Inter-annotator agreement.** Cohen's kappa is the standard measure. For enterprise search, kappa > 0.6 is acceptable; > 0.7 is good. Low agreement signals ambiguous query/document pairs that should be excluded or adjudicated.

**Annotation design for enterprise queries.** Each annotation task must include:
- The raw query
- Simulated user context (role, department, recent activity summary)
- The candidate document (or its excerpt — no PII)
- The actual Microsoft Graph context (sharing history, author info)

Without the context, annotators cannot judge relevance — "Is this budget spreadsheet relevant?" depends entirely on whether the judge plays the role of a finance manager or an engineer.

**Offline metrics:**

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}, \quad \text{DCG@k} = \sum_{i=1}^k \frac{2^{r_i} - 1}{\log_2(i+1)}$$

where $r_i$ is the relevance label at position $i$.

$$\text{MRR} = \frac{1}{\lvert Q \rvert} \sum_{q \in Q} \frac{1}{\text{rank of first relevant result}}$$

$$\text{Recall@k} = \frac{\lvert \text{relevant docs in top-}k \rvert}{\lvert \text{total relevant docs} \rvert}$$

For enterprise RAG specifically, you need additional retrieval metrics:
- **Grounding precision**: fraction of retrieved chunks that appear in the LLM's answer
- **Grounding recall**: fraction of answer claims supported by retrieved chunks
- **Citation accuracy**: fraction of citations that actually support their attributed claims

### Online Experiments & Click Signals {#online-eval}

**A/B testing** is the gold standard but has serious complications in enterprise search:

**Novelty effect.** A new ranking algorithm produces different results. Users click on them out of curiosity, not because they are better. Novelty bias inflates early metrics. Mitigation: wait 2+ weeks for the novelty effect to decay before drawing conclusions.

**Interleaving.** Instead of A/B testing at user level, interleave results from both algorithms in a single ranked list and observe which algorithm's results get more clicks (Team-Draft Interleaving). More statistically efficient than A/B; requires only 1/10 the traffic for the same power.

**Surrogate metrics.** Direct relevance is rarely observable online. Use surrogates:
- **Document open rate** (much stronger signal than impression)
- **Dwell time** after open (> 30 seconds suggests the result was useful)
- **Subsequent action rate**: did the user share, reply, or take another action involving the document?
- **Query reformulation rate**: if the user immediately reformulates, the results were bad
- **Session abandonment**: no click on any result in 30 seconds = failure

**Sparse click problem.** Enterprise search has 100-1000× fewer user events than web search. A Bing ranking change can be evaluated in hours; an M365 Copilot ranking change may need weeks for sufficient statistical power.

Mitigation strategies:
- Pre-registered hypothesis to reduce multiple-comparison corrections
- Variance reduction via CUPED (using pre-experiment metrics as covariates)
- Stratified analysis by query type (file queries vs people queries have different baselines)

### RAG-Specific Evaluation {#rag-eval}

For Copilot answers (not just search results), you need to evaluate the full retrieval → generation pipeline:

**RAGAS framework** (Recall, Answer Relevance, Groundedness, Answer Similarity):

| Metric | What it measures | How computed |
|---|---|---|
| **Context Recall** | Does retrieved context contain info needed to answer? | LLM judges if each answer sentence is in context |
| **Context Precision** | Is retrieved context relevant? | Fraction of context chunks that are useful |
| **Answer Relevance** | Does answer address the question? | Cosine sim of answer embedding to query embedding |
| **Faithfulness** | Is answer grounded in retrieved context? | LLM checks each claim against context |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

# Build evaluation dataset
eval_data = {
    "question": ["What are the action items from the Q3 review?"],
    "answer": ["The main action items are: 1) Review pricing strategy by Oct 30..."],
    "contexts": [["[chunk1 text]", "[chunk2 text]"]],
    "ground_truth": ["Action items include pricing review, EMEA target revision..."]
}

dataset = Dataset.from_dict(eval_data)
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision]
)
print(results)
```

**Enterprise-specific RAG failure modes:**

| Failure | Description | Detection |
|---|---|---|
| **Stale grounding** | Answer based on outdated document version | Check doc modification time vs retrieval time |
| **Permission bleed** | Answer reveals content user cannot access | ACL audit trail |
| **Hallucinated citation** | Citation doesn't support the claim | Citation verification model |
| **Missing evidence** | Answer confident despite no supporting chunks | Answerability score vs answer confidence |
| **Entity confusion** | "Alex" in answer refers to wrong Alex | Named entity disambiguation audit |
| **Acronym drift** | LTV means different things in different documents | Acronym disambiguation in extraction |

### Failure Slicing {#failure-slicing}

Aggregate metrics hide the distribution of failures. A model with NDCG@10 = 0.75 might be excellent for file queries but completely broken for people queries. **Failure slicing** systematically decomposes metrics by:

- **Query type**: file / people / decision / status / expert / temporal / conversational
- **Content type**: SharePoint / OneDrive / Outlook / Teams / external connector
- **Query length**: short (< 5 words) / medium / long conversational
- **Temporal reference**: no time reference / explicit date / relative ("last quarter") / event-anchored ("after the reorg")
- **Entity type**: person-referenced / project-referenced / metric-referenced
- **User cohort**: by org level, by department, by location
- **Freshness**: query for recent content vs archival content
- **Acronym density**: queries with enterprise jargon vs general queries

```python
def slice_metrics(eval_results: list[dict], slice_config: dict) -> dict:
    """
    Compute NDCG@10 for each slice of the eval set.
    """
    slices = {}
    for slice_name, filter_fn in slice_config.items():
        slice_results = [r for r in eval_results if filter_fn(r)]
        if len(slice_results) < 30:  # too small to be meaningful
            continue
        slices[slice_name] = {
            "ndcg_10": compute_ndcg(slice_results, k=10),
            "mrr": compute_mrr(slice_results),
            "recall_10": compute_recall(slice_results, k=10),
            "n": len(slice_results),
        }
    return slices

slice_config = {
    "file_queries": lambda r: r["intent"] == "file",
    "people_queries": lambda r: r["intent"] == "people",
    "temporal_queries": lambda r: r["has_temporal_ref"],
    "acronym_heavy": lambda r: r["acronym_density"] > 0.2,
    "teams_content": lambda r: r["content_type"] == "teams",
    "long_queries": lambda r: len(r["query"].split()) > 10,
}
```

Failure slicing tells you where to focus improvement effort. In practice: people queries often underperform because person disambiguation is hard; temporal queries fail because time references require calendar context; acronym-heavy queries fail because pretrained models don't know enterprise terminology.

---

## End-to-End System Design {#system-design}

Putting it all together — a request through M365 Copilot Search:

```
User query: "Find the latest budget model from Sarah's team"
                │
                ▼
┌───────────────────────────────────────────────────────────┐
│  Query Understanding                                       │
│  • Intent: file_search                                    │
│  • Entities: Sarah → user_id:xyz (working-with graph)    │
│              Sarah's team → group:finance_emea            │
│              "latest" → recency sort, no explicit date    │
│              "budget model" → file_type hint: xlsx/pptx   │
│  • Rewritten: author_group:finance_emea file_type:xlsx    │
│               semantic:"budget financial model"           │
│               sort:recency                                │
└──────────────────────┬────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌──────────────┐ ┌──────────┐ ┌───────────────┐
│ Sparse/BM25  │ │  Dense   │ │   Metadata    │
│ "budget"     │ │ DiskANN  │ │  author_group │
│ "model"      │ │  vector  │ │  file_type    │
│ "financial"  │ │  search  │ │  recency      │
└──────┬───────┘ └────┬─────┘ └───────┬───────┘
       │              │               │
       └──────────────┴───────────────┘
                       │ ~200 candidates
                       ▼
┌───────────────────────────────────────────────────────────┐
│  ACL Filtering                                            │
│  • Bitset filter: user can access these doc IDs           │
│  • Tenant isolation: contoso.com only                     │
│  • Sensitivity: hide confidential from non-entitled users │
│  → ~80 accessible candidates                              │
└──────────────────────┬────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────┐
│  Reranking (Cross-encoder + LTR features)                 │
│  • Cross-encoder score (query × doc content)              │
│  • Working-with score (Sarah's team proximity)            │
│  • Recency score (modified_date decay)                    │
│  • File interaction history (user opened this before?)    │
│  • LambdaMART model combines all features                 │
│  → Top 10 results                                         │
└──────────────────────┬────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐    ┌───────────────────────┐
│  Search UI       │    │  Copilot Chat / RAG   │
│  Show top-10     │    │  • Chunk selection    │
│  with previews   │    │  • Context packing    │
│  and citations   │    │  • Generate answer    │
└──────────────────┘    │  • Citation linking   │
                        │  • Answerability check│
                        └───────────────────────┘
```

**Latency budget** (p99 target: 200ms for search UI, 2s for Copilot answer):

| Stage | Budget |
|---|---|
| Query understanding + entity linking | 20ms |
| Parallel first-stage retrieval (sparse + dense + metadata) | 40ms |
| ACL filtering | 10ms |
| Reranking (top-200 → top-10) | 50ms |
| Response assembly + previews | 30ms |
| **Total search UI** | **~150ms** |
| Chunk selection + context packing | 100ms |
| LLM generation (streaming) | 1–5s (streamed) |
| Citation verification | 200ms (async) |

---

## Interview Questions {#interview-questions}

These are the questions you should be able to answer for a Microsoft Search Relevance / Applied Scientist role:

> **"How would you improve retrieval quality for enterprise Copilot queries?"**

Strong answer: Start with the failure taxonomy — what types of queries fail and why. Vocabulary mismatch → domain adaptation of embedding model on enterprise corpora. Person/project queries → entity linking to org graph. Temporal queries → time-aware retrieval with calendar integration. Then: hybrid retrieval (BM25 + dense + metadata filters), cross-encoder reranking with enterprise signals (working-with, recency), graph-augmented retrieval for status/decision queries. Measure with offline NDCG on query slices + online A/B with surrogate metrics (open rate, dwell time). Never claim one thing fixes everything.

> **"How do you evaluate a new retriever when online labels are sparse?"**

Offline first: build a labeled set with human annotators using simulated user context, measure NDCG per query slice. For online: use interleaving (more statistically efficient than A/B at low traffic). Use surrogate signals: open rate, dwell time, reformulation rate. Use CUPED to reduce variance. Accept slow rollout — 2–4 weeks for significance is normal for enterprise.

> **"How do you know query rewriting improved search and didn't just overfit to the test set?"**

Hold out a rewriting test set from the annotation set. Measure on unseen query types separately. Monitor online for query reformulation rate (users immediately reformulating suggests they didn't like the results). Check that improvements hold across query slices — if improvement is only on the queries used to develop rewriting rules, it's overfit.

> **"How do you evaluate Copilot answer quality when retrieval and generation interact?"**

Decompose: measure retrieval quality (context recall, context precision) separately from generation quality (faithfulness, answer relevance). This tells you whether a bad answer came from bad retrieval or bad generation. Use RAGAS for both jointly. For enterprise-specific failure modes: citation accuracy, freshness correctness, permission correctness. Use LLM-as-judge with rubric for fluency + factuality + citation quality separately.

> **"How would you handle permission changes at scale — a shared drive suddenly made private?"**

Immediate: invalidate the permission cache for all docs in that drive, remove from any live results caches. Short-term: re-evaluate ACLs on demand for any query returning those docs. Background: async re-indexing job to update ACL fields in the index. The user experience goal: no more than 60 seconds between a permission revocation and the document disappearing from all search results. Test this explicitly — it is a compliance boundary.

> **"Design a system to answer 'What are the key themes in our last QBR?'"**

This is a global query — no single chunk answers it. Route to GraphRAG: extract entity graph from all QBR-tagged documents and meeting transcripts, run community detection, generate community summaries. Retrieve top-3 communities by similarity to query. Feed community summaries + top document chunks to LLM. The LLM synthesizes a thematic answer with citations to specific documents. Evaluate: does the answer cover themes mentioned by a human who read all the QBR materials?

---

The core insight for this role: enterprise search is a product where every component failure is visible to the user and potentially a security incident. The job is not to build the most sophisticated model — it is to build a reliable system that users trust, with evaluation infrastructure rigorous enough to know when you have improved it.
