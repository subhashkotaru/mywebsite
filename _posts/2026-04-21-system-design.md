---
title: "Software System Design for Interviews"
date: 2026-04-21
display_order: 17
description: "A comprehensive guide to distributed systems design for interviews — covering architecture patterns, databases, caching, message queues, reliability, failure modes, and worked examples with diagrams, equations, and decision tables."
tags: [system-design, distributed-systems, interview]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#interview-framework">Interview Framework</a>
      <ul class="post-toc-sublist">
        <li><a href="#approach">How to Approach a System Design Interview</a></li>
        <li><a href="#estimation">Scale Estimation Heuristics</a></li>
        <li><a href="#envelope-math">Back-of-Envelope Examples</a></li>
      </ul>
    </li>
    <li><a href="#architecture-patterns">High-Level Architecture Patterns</a>
      <ul class="post-toc-sublist">
        <li><a href="#monolith-microservices">Monolith vs Microservices vs Serverless</a></li>
        <li><a href="#sync-async">Synchronous vs Asynchronous Communication</a></li>
        <li><a href="#event-driven">Event-Driven Architecture, CQRS, Event Sourcing</a></li>
        <li><a href="#api-gateway">API Gateway Pattern</a></li>
      </ul>
    </li>
    <li><a href="#databases">Databases</a>
      <ul class="post-toc-sublist">
        <li><a href="#sql-nosql">SQL vs NoSQL Decision Framework</a></li>
        <li><a href="#relational">Relational Databases</a></li>
        <li><a href="#key-value">Key-Value Stores</a></li>
        <li><a href="#document">Document Stores</a></li>
        <li><a href="#wide-column">Wide-Column Stores</a></li>
        <li><a href="#timeseries">Time-Series Databases</a></li>
        <li><a href="#search">Search Engines</a></li>
        <li><a href="#oltp-olap">OLTP vs OLAP</a></li>
        <li><a href="#cap">CAP Theorem and PACELC</a></li>
        <li><a href="#replication">Replication Strategies</a></li>
        <li><a href="#sharding">Sharding Strategies</a></li>
      </ul>
    </li>
    <li><a href="#caching">Caching</a>
      <ul class="post-toc-sublist">
        <li><a href="#cache-patterns">Cache Patterns</a></li>
        <li><a href="#eviction">Eviction Policies</a></li>
        <li><a href="#stampede">Cache Stampede and Thundering Herd</a></li>
        <li><a href="#cdn">CDN as Distributed Cache</a></li>
        <li><a href="#redis-memcached">Redis vs Memcached</a></li>
        <li><a href="#invalidation">Cache Invalidation</a></li>
      </ul>
    </li>
    <li><a href="#messaging">Message Queues and Streaming</a>
      <ul class="post-toc-sublist">
        <li><a href="#kafka">Kafka Architecture</a></li>
        <li><a href="#delivery-semantics">Delivery Semantics</a></li>
        <li><a href="#queue-pubsub">Queue vs Pub-Sub vs Stream</a></li>
        <li><a href="#backpressure">Backpressure Handling</a></li>
      </ul>
    </li>
    <li><a href="#latency">Latency Optimisations</a>
      <ul class="post-toc-sublist">
        <li><a href="#percentiles">Percentile Latency Targets</a></li>
        <li><a href="#connection-pooling">Connection Pooling</a></li>
        <li><a href="#async-io">Async I/O and Non-Blocking</a></li>
        <li><a href="#geo-distribution">Geographic Distribution</a></li>
        <li><a href="#protocols">Protocol Optimisations</a></li>
      </ul>
    </li>
    <li><a href="#reliability">Reliability and Graceful Degradation</a>
      <ul class="post-toc-sublist">
        <li><a href="#circuit-breaker">Circuit Breakers</a></li>
        <li><a href="#retry">Retry with Exponential Backoff</a></li>
        <li><a href="#bulkhead">Bulkhead Pattern</a></li>
        <li><a href="#rate-limiting">Rate Limiting</a></li>
        <li><a href="#timeouts">Timeout Strategies</a></li>
        <li><a href="#fallbacks">Fallback Patterns</a></li>
        <li><a href="#health-checks">Health Checks</a></li>
      </ul>
    </li>
    <li><a href="#failure-modes">Failure Modes and Detection</a>
      <ul class="post-toc-sublist">
        <li><a href="#network-partitions">Network Partitions and Split-Brain</a></li>
        <li><a href="#cascading">Cascading Failures</a></li>
        <li><a href="#deadlocks">Distributed Deadlocks</a></li>
        <li><a href="#data-corruption">Data Corruption and Checksums</a></li>
        <li><a href="#observability">Observability: Metrics, Logs, Traces</a></li>
        <li><a href="#slo">SLOs, SLAs, and SLIs</a></li>
      </ul>
    </li>
    <li><a href="#ab-testing">A/B Testing and Feature Flags</a>
      <ul class="post-toc-sublist">
        <li><a href="#experiment-design">Experiment Design</a></li>
        <li><a href="#statistics">Statistical Significance</a></li>
        <li><a href="#feature-flags">Feature Flag Architecture</a></li>
        <li><a href="#holdouts">Holdout Groups</a></li>
      </ul>
    </li>
    <li><a href="#worked-examples">Worked Examples</a>
      <ul class="post-toc-sublist">
        <li><a href="#url-shortener">URL Shortener</a></li>
        <li><a href="#twitter-feed">Twitter/X Feed</a></li>
        <li><a href="#rate-limiter">Distributed Rate Limiter</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## Interview Framework
{: #interview-framework}

### How to Approach a System Design Interview
{: #approach}

System design interviews are open-ended by design. There is no single correct answer. What the interviewer evaluates is your **process** — how you gather requirements, how you reason through tradeoffs, and how you identify the parts of the design that matter most for the stated constraints. A strong candidate drives the conversation with a clear, structured framework.

<div class="post-flow" role="group" aria-label="System design interview framework">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Clarify — ask questions to nail down functional and non-functional requirements</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Estimate — compute scale: QPS, storage, bandwidth, memory</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Design — sketch the high-level architecture; identify the core components</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Deep-dive — explore the 2-3 hardest sub-problems in detail</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Tradeoffs — explain what you chose and what you gave up</span></li>
  </ol>
</div>

**Step 1 — Clarify.** Do not start designing immediately. Spend 5 minutes asking questions:
- Who are the users and what are the core use cases?
- What is the expected scale (users, requests per day, data volume)?
- What are the consistency requirements? Is it okay to show stale data?
- What is the acceptable latency? P99 under 100ms or under 1 second?
- Are there geographic constraints (single region or global)?
- What is the read/write ratio?

Document your assumptions explicitly. "I'm assuming 100M daily active users, a 10:1 read/write ratio, and 99.99% availability."

**Step 2 — Estimate.** Rough numbers guide every subsequent decision. A system serving 100 QPS has fundamentally different architectural requirements than one serving 1M QPS. Compute the numbers before choosing databases or caches.

**Step 3 — Design.** Draw the architecture diagram on the whiteboard. Start with the clients, then the load balancer, then the application tier, then the data stores. Identify how data flows between components. Name the external services you would use (S3, Kafka, Redis).

**Step 4 — Deep-dive.** Pick the hardest 2-3 sub-problems. Typical candidates: the database schema, the caching strategy, the fanout mechanism, the consistency model, the failure recovery path. Go deep on these — this is where you demonstrate senior-level thinking.

**Step 5 — Tradeoffs.** Explicitly state what your design optimises for and what it sacrifices. "We chose eventual consistency here because strong consistency would require distributed transactions that triple write latency. In this domain, stale reads are acceptable." Interviewers want to hear you reason about tradeoffs, not just recite architectures.

| Phase | Time budget (45-min interview) | Output |
|---|---|---|
| Clarify | 5 min | Written assumptions |
| Estimate | 5 min | QPS, storage, bandwidth numbers |
| Design | 10 min | Component diagram |
| Deep-dive | 20 min | Detailed treatment of hardest parts |
| Tradeoffs | 5 min | Explicit list of decisions and alternatives |

### Scale Estimation Heuristics
{: #estimation}

Knowing a handful of numbers by heart lets you compute estimates in seconds during an interview.

**Latency reference numbers:**

| Operation | Approximate latency |
|---|---|
| L1 cache reference | 1 ns |
| L2 cache reference | 4 ns |
| Main memory reference | 100 ns |
| SSD random read | 16 us |
| HDD seek | 10 ms |
| Same-datacenter round trip | 500 us |
| Cross-region round trip (US–EU) | 150 ms |
| DNS lookup | 10–100 ms |

**Storage units:**

| Quantity | Bytes |
|---|---|
| 1 KB | 10^3 |
| 1 MB | 10^6 |
| 1 GB | 10^9 |
| 1 TB | 10^12 |
| 1 PB | 10^15 |

**Throughput rules of thumb:**

| System | Approximate throughput |
|---|---|
| Commodity HTTP server (1 core) | 10K–50K req/s |
| PostgreSQL (optimised, indexed) | 10K–100K reads/s |
| Cassandra (single node) | 20K–50K writes/s |
| Kafka (single partition) | 100K–1M msg/s |
| Redis (single instance) | 100K–1M ops/s |
| Network card (1 GbE) | 125 MB/s |
| Network card (10 GbE) | 1.25 GB/s |

**Time unit conversions:**

$$\text{1 day} = 86{,}400 \text{ s} \approx 10^5 \text{ s}$$

$$\text{QPS} = \frac{\text{requests per day}}{86{,}400}$$

For a rough upper bound: if you have 100M daily active users each making 10 requests per day, that is 1B requests/day or roughly 11,600 QPS average. Peak is typically 2-3x average, so plan for 25,000-35,000 QPS.

### Back-of-Envelope Examples
{: #envelope-math}

**Example 1: Twitter-like feed storage**

Assumptions: 300M daily active users, each reads 50 tweets/day, each tweet is 280 characters (560 bytes), 500M tweets posted per day.

Storage per day:

$$500\text{M tweets} \times 560\text{ B} = 280\text{ GB/day}$$

With metadata, user IDs, timestamps (assume 1KB per full tweet row):

$$500\text{M} \times 1\text{ KB} = 500\text{ GB/day} \approx 180\text{ TB/year}$$

Write QPS:

$$\frac{500\text{M}}{86{,}400} \approx 5{,}800 \text{ writes/s (average)}$$

Read QPS (300M users × 50 reads / 86,400):

$$\approx 174{,}000 \text{ reads/s}$$

Read/write ratio: ~30:1. This immediately tells you caching is critical.

**Example 2: Image hosting (Instagram-like)**

Assumptions: 100M photos uploaded/day, average photo size 3 MB.

Storage per day:

$$100\text{M} \times 3\text{ MB} = 300\text{ TB/day}$$

Storage per year: ~110 PB. This immediately rules out on-premises storage — you need object storage (S3, GCS). Bandwidth for uploads at peak (assuming 10x spike):

$$\frac{300\text{ TB}}{86{,}400} \times 10 \approx 34.7\text{ GB/s}$$

This requires a CDN for uploads and aggressive compression.

**Example 3: URL shortener**

Assumptions: 100M URLs shortened/day, 10B redirects/day (100:1 read:write ratio), URL stored as 100 bytes, 5-year retention.

Storage:

$$100\text{M URLs/day} \times 365 \times 5 \times 100\text{ B} \approx 18\text{ TB}$$

Redirect QPS:

$$\frac{10\text{B}}{86{,}400} \approx 116{,}000 \text{ reads/s}$$

At 116K reads/s with a 100:1 read:write ratio, aggressive caching is mandatory. A single Redis instance handles this comfortably.

> **Interview question:** Estimate the QPS, storage, and bandwidth requirements for a WhatsApp-like messaging system with 2B users.
>
> *Assumptions: 2B users, 50% DAU = 1B DAU. Each user sends 40 messages/day = 40B messages/day. Each message is 200 bytes average (text; media separate). Write QPS: 40B / 86,400 = ~463,000 msg/s. Add 10x peak = 4.6M msg/s peak. Storage: 40B messages/day x 200 B = 8 TB/day, 2.9 PB/year. For media: assume 5% of messages contain 1 MB media = 2B x 0.05 x 1 MB x 40 = 4 PB/day — media must live in object storage (S3) with CDN. Bandwidth for delivery (each message delivered to 1-1 or group, assume average 2 recipients): 40B messages x 400 B x 2 = 32 TB/day delivered = 370 GB/s. This requires a CDN and regional edge nodes. The key insight: media is the dominant storage/bandwidth term, not text messages.*

---

## High-Level Architecture Patterns
{: #architecture-patterns}

### Monolith vs Microservices vs Serverless
{: #monolith-microservices}

The first architectural decision is the deployment model. Each has genuine strengths and the right choice depends on team size, scale, and operational maturity.

| Dimension | Monolith | Microservices | Serverless |
|---|---|---|---|
| Deployment complexity | Low | High | Low (managed) |
| Operational overhead | Low | High | Very low |
| Independent scaling | No | Yes | Yes (per function) |
| Network latency | None (in-process) | Adds per-hop latency | Cold start latency |
| Technology diversity | Low | High | Medium |
| Testing complexity | Low | High | Medium |
| Best team size | Small (1-10 engineers) | Large (10+ engineers, many teams) | Any |
| Best when | Early stage, fast iteration | Many independent domains, independent scale needs | Event-driven, spiky workloads |

**Monolith.** All components run in the same process. Simple to develop, test, and deploy. Internal calls are in-process function calls — no serialisation, no network, no retries. The failure modes are well-understood. Scaling requires vertical scaling or running multiple identical instances behind a load balancer. The classic mistake is treating "monolith" as a dirty word — most successful companies start with a monolith and migrate selectively.

**Microservices.** Each domain is a separately deployed service communicating over a network. Independent deployment and scaling is the core value proposition. The cost: distributed systems problems appear everywhere. A call that was a function call is now an HTTP request with latency, failures, and serialisation. You need a service registry, load balancers, distributed tracing, circuit breakers, and a sophisticated deployment pipeline.

**Serverless.** Functions-as-a-Service (AWS Lambda, Google Cloud Functions). No servers to manage — you deploy code, the platform handles scaling. Excellent for event-driven workloads and spiky traffic. Cold start latency (100ms-1s) makes it unsuitable for latency-sensitive hot paths. State must be externalised — functions are stateless.

```
Monolith:
    +---------+
    |  App    |
    | [Auth]  |
    | [Cart]  |
    | [Order] |
    +---------+
        |
    [Single DB]

Microservices:
    [API GW] --> [Auth Svc] --> [Auth DB]
             --> [Cart Svc] --> [Cart DB]
             --> [Order Svc] --> [Order DB]
                            --> [Payment Svc]
```

### Synchronous vs Asynchronous Communication
{: #sync-async}

| Property | Synchronous (HTTP/gRPC) | Asynchronous (Queue/Events) |
|---|---|---|
| Caller blocks? | Yes | No |
| Latency | Low (direct call) | Higher (queue hop) |
| Coupling | Tight — caller must know callee's address | Loose — caller sends to broker |
| Retry behaviour | Caller must retry | Broker handles retry |
| Fan-out | Difficult | Natural |
| Ordering guarantee | N/A | Depends on queue |
| Backpressure | Caller sees slow responses | Queue absorbs bursts |

**Synchronous** is appropriate when the caller needs the result immediately to continue. User-facing reads — "fetch my shopping cart" — are typically synchronous. The tradeoff is that the caller fails if the callee is slow or unavailable.

**Asynchronous** is appropriate when the result is not needed immediately or when the workload is bursty. "Process this order" does not need to complete before returning a 202 Accepted response to the user. The queue absorbs traffic spikes and decouples the producer from the consumer's processing rate.

### Event-Driven Architecture, CQRS, Event Sourcing
{: #event-driven}

**Event-Driven Architecture (EDA).** Services communicate by publishing and subscribing to events. An event represents something that happened: `OrderPlaced`, `PaymentProcessed`, `InventoryUpdated`. Producers and consumers are decoupled — the producer does not know which services will consume its events.

```
[Order Svc] ---(OrderPlaced)---> [Event Bus]
                                      |
                           +----------+----------+
                           |          |          |
                     [Inventory] [Notification] [Analytics]
```

**CQRS (Command Query Responsibility Segregation).** Separate the write model (commands that mutate state) from the read model (queries that read state). This allows them to be optimised independently. Writes go through a command handler that validates and applies changes; reads go through a query handler that reads from a denormalised read replica optimised for the query shape.

```
                  [Command Handler] --> [Write DB]
    [Client] -->                                 --> [Sync/Event] --> [Read DB]
                  [Query Handler]  <---------------------------------/
```

The read model can be a different database entirely — a document store for flexible querying, or a materialised view that's pre-joined for the most common query pattern.

**Event Sourcing.** Instead of storing current state, store the sequence of events that led to that state. The current state is derived by replaying events. This gives you a full audit log by default, temporal queries ("what was the state at time T?"), and the ability to project new read models by replaying the event log.

```
Event store: [AccountOpened] [Deposited $100] [Withdrew $30] [Deposited $50]
Current state: balance = $120 (derived by replaying all events)
```

**When to use each:**

| Pattern | Use when | Avoid when |
|---|---|---|
| CQRS | Read/write workloads have very different shapes; read scale >> write scale | Simple CRUD applications |
| Event Sourcing | Need full audit log; complex domain logic; temporal queries | Simple domains; high event volume without clear benefit |
| EDA | Multiple consumers need to react to state changes; loose coupling needed | Simple request-response flows; low event volume |

### API Gateway Pattern
{: #api-gateway}

The API gateway is the single entry point for all client requests. It handles cross-cutting concerns so individual services do not have to.

```
[Mobile App] ----+
[Web App]    ----|---> [API Gateway] --> [Auth Svc]
[3rd Party]  ----+           |       --> [User Svc]
                             |       --> [Order Svc]
                             |       --> [Product Svc]
                         [Rate Limit]
                         [Auth/JWT]
                         [SSL Termination]
                         [Logging]
                         [Request Routing]
```

**Responsibilities of an API gateway:**
- Authentication and authorisation (validate JWT, check permissions)
- Rate limiting and throttling per client or per API key
- SSL/TLS termination (offload TLS from application servers)
- Request routing and load balancing
- Request/response transformation (version negotiation, protocol translation)
- Caching for read-heavy endpoints
- Observability (logging, tracing request IDs, metrics)

**Failure mode:** the API gateway becomes a single point of failure. Mitigation: deploy multiple gateway instances behind a load balancer. The gateway must itself be stateless so it can be horizontally scaled. Do not put business logic in the gateway — keep it thin.

> **Interview question:** When would you use a Backend-for-Frontend (BFF) pattern instead of a single API gateway?
>
> *A single API gateway is fine when all clients have similar data needs. The BFF pattern is warranted when clients have significantly different data shapes. A mobile client may need a condensed response (small payload, fewer fields) because of bandwidth constraints. A web client may need rich nested data for a complex dashboard. With a single gateway, you either over-fetch on mobile or under-fetch on web. A BFF creates one backend per client type (mobile BFF, web BFF) that aggregates data from the same downstream services but shapes the response optimally for each client. This reduces client-side orchestration and avoids over-fetching. The cost: more code to maintain. The BFF is appropriate when client needs diverge significantly, not for minor differences.*

---

## Databases
{: #databases}

### SQL vs NoSQL Decision Framework
{: #sql-nosql}

The SQL vs NoSQL question is not about which is better — it is about which fits the access pattern and consistency requirements of your specific workload.

| Factor | Choose SQL (Relational) | Choose NoSQL |
|---|---|---|
| Data shape | Structured, relations between entities | Flexible schema, nested/hierarchical data |
| Query pattern | Ad-hoc joins, aggregations, complex filtering | Known access patterns, key-based lookups |
| Consistency | Strong ACID needed | Eventual consistency acceptable |
| Scale | Vertical or moderate horizontal | Massive horizontal scale |
| Transaction needs | Multi-entity transactions required | Single-document or idempotent operations |
| Team familiarity | SQL expertise available | NoSQL expertise available |

**The dangerous middle ground.** Many candidates choose NoSQL for "scale" without considering the loss of joins and transactions. If your data is inherently relational (users have orders, orders have line items, line items reference products), a relational database with proper indexing scales further than most teams realise. PostgreSQL with read replicas and connection pooling handles tens of thousands of QPS. Start relational unless you have a specific reason not to.

### Relational Databases
{: #relational}

**ACID properties** are the foundation of relational database guarantees:

| Property | Meaning | Implementation |
|---|---|---|
| Atomicity | All operations in a transaction succeed or all are rolled back | Write-ahead log (WAL), undo log |
| Consistency | Database transitions from one valid state to another | Constraints, triggers, referential integrity |
| Isolation | Concurrent transactions do not interfere | Locking, MVCC (multi-version concurrency control) |
| Durability | Committed transactions survive crashes | WAL flushed to disk before acknowledgment |

**Isolation levels** (from weakest to strongest):

| Level | Dirty Read | Non-repeatable Read | Phantom Read |
|---|---|---|---|
| Read Uncommitted | Possible | Possible | Possible |
| Read Committed | Prevented | Possible | Possible |
| Repeatable Read | Prevented | Prevented | Possible |
| Serialisable | Prevented | Prevented | Prevented |

Most applications use Read Committed (PostgreSQL default) or Repeatable Read. Serialisable is expensive — it serialises concurrent transactions, limiting throughput.

**Indexing.** Indexes are the primary performance lever in relational databases.

*B-tree index* — the default index type. Balanced tree structure where each node contains a sorted range of keys with pointers to child nodes or heap rows. Supports equality lookups, range queries, and ordering. Write cost: O(log N). Read cost: O(log N). Good for most access patterns.

*LSM-tree (Log-Structured Merge tree)* — used in write-optimised storage engines (RocksDB, LevelDB, Cassandra). Writes go to an in-memory memtable first, then are flushed to immutable SSTables on disk. Reads must check multiple levels. Excellent write throughput; read amplification is managed by compaction. Used when write throughput dominates.

```
B-tree:                    LSM-tree:
      [50]                  Writes -> [MemTable] (in memory)
     /    \                             |
  [25]    [75]              flush -> [L0 SSTable]
  /  \    /  \                         |
[10][30][60][90]           compact -> [L1 SSTable]
                                       |
                                    [L2 SSTable] ...
```

**Query optimisation.** The query planner selects an execution plan from a set of candidates. Key concepts:
- **Index selectivity**: an index on `status` with 3 distinct values over 1M rows is low-selectivity — the planner may choose a full table scan. An index on `user_id` with 1M distinct values is high-selectivity — the planner uses the index.
- **EXPLAIN ANALYZE**: always check the query plan before deploying. Look for sequential scans on large tables, hash joins on non-indexed columns, and excessive row estimates.
- **Composite indexes**: `CREATE INDEX ON orders (user_id, created_at)` supports queries filtering by `user_id` and optionally `created_at`. The leftmost prefix rule: a query on `created_at` alone cannot use this index.
- **Covering indexes**: an index that contains all columns needed by a query (including SELECT columns) allows an index-only scan — no heap access needed.

### Key-Value Stores
{: #key-value}

Key-value stores provide the simplest data model: get(key), set(key, value), delete(key). The simplicity enables extreme performance.

**Redis.** In-memory key-value store with rich data structures (strings, hashes, lists, sets, sorted sets, streams, HyperLogLog). Data is held in RAM — reads and writes are O(1) for most operations, ~100K-1M ops/s on a single instance.

| Data structure | Use case | Example |
|---|---|---|
| String | Caching serialised objects, counters | Session data, rate limit counters |
| Hash | Storing object fields | User profile, config |
| List | Message queue, activity feed | Job queue (LPUSH/BRPOP) |
| Sorted Set | Leaderboards, range queries by score | Top-N users by score |
| Set | Unique membership, intersection | Followers, tags |
| HyperLogLog | Approximate cardinality (count unique) | Unique visitors (1% error) |

**DynamoDB.** AWS managed key-value and document store. Data stored on SSDs; strongly consistent reads optional but double the cost. Pricing on read/write capacity units or on-demand. Partition key determines the partition a row lives on — poor partition key choice leads to hot partitions and throttling.

**Failure modes for key-value stores:**

| Failure | Cause | Mitigation |
|---|---|---|
| Hot key | Single key (e.g., celebrity user) receives disproportionate traffic | Local in-process cache; key sharding (user:123:shard0..9) |
| Memory eviction | Redis OOMs, evicts entries unexpectedly | Set maxmemory policy; monitor eviction rate |
| Cold start | Cache is empty (restart, deployment) | Warm cache from database on startup; gradual rollout |
| Stale data | Cache not invalidated after write | Use TTL; implement write-through or event-based invalidation |

### Document Stores
{: #document}

Document stores (MongoDB, Firestore) store JSON/BSON documents. Each document is a self-contained unit that can have nested objects and arrays.

**Schema flexibility tradeoffs:**

| Flexible Schema Benefit | Flexible Schema Cost |
|---|---|
| No migration needed for new fields | No enforcement of required fields |
| Arbitrary nesting maps naturally to application objects | Query performance depends on index design |
| Polymorphic documents possible | Inconsistent documents hard to query uniformly |
| Fast iteration in early development | Data quality degrades without application-level validation |

**Embedding vs referencing.** The core schema design decision in document stores:

*Embed* when: the nested data is always read together with the parent, the nested data is bounded in size, and the nested data has no independent lifecycle.

*Reference* (use foreign-key equivalent) when: the nested data is large, frequently read independently, or shared across many parent documents.

```json
// Embedded: good for blog post with comments (always read together)
{
  "_id": "post123",
  "title": "System Design",
  "comments": [
    {"author": "alice", "text": "great post"},
    {"author": "bob", "text": "very helpful"}
  ]
}

// Referenced: good for orders referencing products (product used by many orders)
{
  "_id": "order456",
  "product_id": "prod789",   // reference, not embedded
  "quantity": 2
}
```

### Wide-Column Stores
{: #wide-column}

Cassandra, HBase, and Google Bigtable are wide-column stores. Data is organised by row key with variable columns per row, stored sorted by column key within each row.

**Partition key design** is the most critical Cassandra design decision. The partition key determines which node a row lives on. A good partition key:
- Has high cardinality (many distinct values)
- Distributes writes evenly across nodes (no hot partitions)
- Enables the most common query pattern to hit a single partition

**Anti-patterns:**
- Partition key with low cardinality (e.g., `country_code`) — most data ends up on 3-4 nodes
- Partition key with unbounded data (e.g., `user_id` for a "messages per user" table) — one partition grows without bound, causing hot partition + compaction problems
- Allowing tombstone accumulation (deletes in Cassandra create tombstones; queries read through them until compaction) — use TTL instead of explicit deletes

**Consistency levels.** Cassandra uses quorum-based consistency. With a replication factor of N, you choose a consistency level for each read/write:

| Level | Requires | Use when |
|---|---|---|
| ONE | 1 replica responds | Maximum availability; accept stale reads |
| QUORUM | Majority (N/2 + 1) respond | Strong consistency with multiple data centers |
| ALL | All N replicas respond | Strongest consistency; unavailable if any replica down |
| LOCAL_QUORUM | Quorum in local datacenter | Multi-DC with low cross-DC latency requirement |

For strong consistency: use QUORUM for both reads and writes.

### Time-Series Databases
{: #timeseries}

Time-series databases (InfluxDB, TimescaleDB, Prometheus) are optimised for append-heavy workloads where the primary query dimension is time.

**Core features:**
- Optimised for append-only writes at high throughput
- Automatic time-based partitioning (chunks/shards by time range)
- Retention policies: automatically delete data older than N days
- Downsampling: aggregate high-resolution data into lower-resolution summaries to save storage

**Retention and downsampling example:**

```
Raw data (1s resolution) --> retain 7 days
1-minute aggregates      --> retain 30 days
1-hour aggregates        --> retain 1 year
1-day aggregates         --> retain 5 years
```

Downsampling trades query precision for storage efficiency. A 5-year trend query does not need second-level resolution — day-level aggregates suffice and are 86,400x smaller.

**TimescaleDB** is PostgreSQL with a time-series extension — you get SQL, joins, and ACID with the performance of a dedicated time-series engine. **InfluxDB** is purpose-built with its own query language (Flux) and is better for extremely high-cardinality series.

### Search Engines
{: #search}

Elasticsearch (and OpenSearch) provide full-text search via an **inverted index**: for each term, the index stores a sorted list of document IDs containing that term.

```
Inverted index example:
"system"   -> [doc1, doc3, doc7, doc12]
"design"   -> [doc1, doc2, doc7, doc15]
"interview" -> [doc1, doc4, doc15]
```

A search for "system design" performs an intersection of the posting lists: `[doc1, doc7]`.

**Relevance scoring.** Elasticsearch uses BM25 (Best Match 25) by default:

$$\text{score}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

Where $f(t,d)$ is term frequency in document $d$, $\text{IDF}(t)$ is inverse document frequency, $|d|$ is document length, $\text{avgdl}$ is average document length, and $k_1$, $b$ are tuning parameters.

**Failure modes for search engines:**

| Failure | Cause | Mitigation |
|---|---|---|
| Index lag | Writes replicated async to Elasticsearch from primary DB | Accept eventual consistency; use refresh interval |
| Split brain | Two master-eligible nodes both claim master status | Set minimum_master_nodes = (N/2 + 1); use dedicated master nodes |
| Hot shards | One shard receives all traffic | Increase shard count; use routing to spread load |

### OLTP vs OLAP
{: #oltp-olap}

| Dimension | OLTP (Online Transaction Processing) | OLAP (Online Analytical Processing) |
|---|---|---|
| Query type | Short point reads and writes | Long scans, aggregations, joins |
| Data freshness | Real-time | Batch or near-real-time |
| Schema | Normalised (3NF) | Denormalised (star/snowflake schema) |
| Index type | B-tree for point lookups | Column-oriented for scan performance |
| Examples | PostgreSQL, MySQL, Cassandra | Snowflake, BigQuery, Redshift, ClickHouse |
| Row vs column | Row-oriented storage | Column-oriented storage |
| Users | Application code | Analysts, data scientists |

**Why columnar storage for OLAP.** A query like "average revenue by country" touches one or two columns across millions of rows. Column-oriented storage keeps all values of a column contiguous on disk — the query reads only the two relevant columns, not every field of every row. Compression ratios are also much higher within a column (homogeneous data types, high repetition).

**Data warehouse architecture:**

```
[OLTP DBs] --> [ETL / CDC] --> [Data Lake (S3)] --> [Data Warehouse (Snowflake)]
                                                             |
                                              [BI Tool / Analyst Queries]
```

CDC (Change Data Capture) extracts database changes in real time (via WAL in Postgres) and streams them to the data lake, enabling near-real-time analytics without burdening the OLTP database.

### CAP Theorem and PACELC
{: #cap}

**CAP Theorem** states that a distributed data store can guarantee at most two of three properties simultaneously:

- **Consistency (C)**: every read sees the most recent write
- **Availability (A)**: every request receives a non-error response
- **Partition Tolerance (P)**: the system continues operating despite network partitions

Since network partitions are unavoidable in distributed systems, the practical choice is between **CP** (consistency + partition tolerance, sacrificing availability when partitioned) and **AP** (availability + partition tolerance, serving possibly stale data during partitions).

| System | Classification | Behaviour during partition |
|---|---|---|
| Zookeeper, HBase | CP | May refuse writes to prevent inconsistency |
| Cassandra, DynamoDB | AP | Serves reads/writes with potential stale data |
| RDBMS (single node) | CA | Not partition tolerant; single point of failure |

**PACELC model** extends CAP to the normal (non-partition) case: during normal operation, there is a latency (L) vs consistency (C) tradeoff even without partitions.

$$\text{PACELC: if P then } \{A \text{ or } C\} \text{ else } \{L \text{ or } C\}$$

| System | P: A or C | E: L or C |
|---|---|---|
| DynamoDB (eventual consistency) | PA | EL |
| DynamoDB (strong consistency) | PC | EC |
| Cassandra (QUORUM) | PA | EC |
| MongoDB (primary reads) | PC | EC |

### Replication Strategies
{: #replication}

**Leader-follower (primary-replica).** All writes go to one leader node. The leader replicates changes to followers asynchronously or synchronously. Reads can be served from followers for read scaling.

- Async replication: lower write latency; followers may be seconds behind; data loss possible if leader fails before replicating
- Sync replication: no data loss; write latency increases as you wait for follower acknowledgment; follower slowness blocks writes

**Multi-leader.** Multiple nodes accept writes. Useful for multi-datacenter deployments where each datacenter has a local leader. Conflict resolution is the hard problem: last-write-wins (LWW), CRDTs (conflict-free replicated data types), or application-level conflict resolution.

**Leaderless (quorum).** No designated leader. Writes are sent to N replicas; a write succeeds when W replicas acknowledge. Reads are sent to R replicas; the most recent value is returned. Consistency guarantee when:

$$R + W > N$$

Example: N=3, W=2, R=2 (quorum): any read sees at least one replica that received the latest write.

```
Leader-follower:               Leaderless:
[Client] --> [Leader] -sync-> [Follower A]     [Client] --> [Node A] ✓
             [Leader] -async-> [Follower B]              --> [Node B] ✓
                                                         --> [Node C] (any 2 of 3)
```

### Sharding Strategies
{: #sharding}

Sharding (horizontal partitioning) distributes data across multiple nodes. The sharding key determines which node stores each row.

**Range sharding.** Shard by a contiguous range of key values. Example: users A-M on shard 1, N-Z on shard 2.
- Advantage: range queries within a single shard are efficient
- Disadvantage: hot spots if data is skewed (most users have last names starting with 'S')

**Hash sharding.** Apply a hash function to the key; the hash mod N determines the shard. Example: `shard = hash(user_id) % num_shards`.
- Advantage: even distribution for uniformly distributed keys
- Disadvantage: range queries require scanning all shards; resharding (adding nodes) requires remapping many keys

**Consistent hashing.** A variant of hash sharding that minimises key remapping when nodes are added or removed. Each node is assigned a position on a hash ring. A key's shard is the nearest node clockwise on the ring. Adding a node only remaps keys from one adjacent node, not all nodes.

**Directory-based sharding.** A lookup service maps each key to a shard. Most flexible — you can implement arbitrary sharding logic. Cost: the directory is a bottleneck and single point of failure; must be cached.

| Strategy | Even distribution | Range query support | Resharding cost |
|---|---|---|---|
| Range | No (depends on data) | Yes | Low |
| Hash | Yes | No (requires scatter-gather) | High |
| Consistent hash | Yes | No | Low |
| Directory-based | Yes | Possible | Medium |

> **Interview question:** Your PostgreSQL database is hitting write limits. You decide to shard. How do you choose a sharding key and strategy?
>
> *Step 1: identify the dominant access pattern. If 90% of queries filter by user_id, user_id is the natural sharding key — it keeps all data for a user on one shard and avoids cross-shard queries. Step 2: evaluate distribution. If user activity is uniform, hash(user_id) % N gives even distribution. If there are power users with 1000x more data/traffic (e.g., organisations in a B2B product), hash sharding on user_id creates hot shards. In that case, add a sub-key or use consistent hashing with virtual nodes. Step 3: handle cross-shard queries. Analytics queries that aggregate across all users must scatter-gather across all shards — these are expensive. Move analytics to an OLAP system (Snowflake/BigQuery) via CDC. Step 4: resharding plan. Choose consistent hashing to minimise key remapping when you add shards. Use a double-write period when migrating: write to old and new location; read from new; verify; cut over. Step 5: foreign keys. Sharding breaks cross-shard foreign key constraints. Model relationships within a shard (user_id on all child tables); use application-level consistency for cross-user references.*

---

## Caching
{: #caching}

### Cache Patterns
{: #cache-patterns}

**Cache-aside (lazy loading).** The application checks the cache before querying the database. On a miss, it reads from the database and populates the cache.

```
Read:  App -> Cache -> (miss) -> DB -> populate cache -> return
Write: App -> DB (cache NOT updated; TTL or explicit invalidation needed)
```

Advantage: only requested data is cached; resilient to cache failures (falls back to DB). Disadvantage: cache miss penalty (two round trips); possible stale reads until TTL expires.

**Read-through.** The cache is in front of the database. On a miss, the cache layer reads from the database and populates itself.

```
Read:  App -> Cache -> (miss) -> Cache reads DB -> Cache returns data
Write: App -> DB (same as cache-aside for writes)
```

Similar to cache-aside but the cache population logic is in the cache layer, not the application. Useful with a managed caching layer (e.g., AWS ElastiCache with read-through support).

**Write-through.** Every write goes through the cache to the database synchronously.

```
Write: App -> Cache -> DB (both updated synchronously before acknowledgment)
Read:  App -> Cache -> (always fresh)
```

Advantage: cache is always fresh; no stale reads. Disadvantage: write latency is higher (two writes); cache stores data that may never be read (write-heavy with cold reads wastes cache memory).

**Write-behind (write-back).** Writes go to the cache immediately; the cache asynchronously flushes to the database.

```
Write: App -> Cache (acknowledged immediately) --> (async) --> DB
```

Advantage: lowest write latency. Disadvantage: data loss risk if the cache node fails before flushing; complex to implement correctly.

| Pattern | Read freshness | Write latency | Data loss risk | Complexity |
|---|---|---|---|---|
| Cache-aside | Stale until TTL | Low | None | Low |
| Read-through | Stale until TTL | Low | None | Medium |
| Write-through | Always fresh | Higher | None | Medium |
| Write-behind | Always fresh | Lowest | Yes | High |

### Eviction Policies
{: #eviction}

When the cache is full, an eviction policy determines which item to remove.

| Policy | Description | Best for |
|---|---|---|
| LRU (Least Recently Used) | Evict the item not accessed for the longest time | General-purpose workloads; temporal locality |
| LFU (Least Frequently Used) | Evict the item accessed fewest times | Stable hot-set workloads; power-law access distributions |
| FIFO (First In, First Out) | Evict the oldest inserted item | Simple; not access-aware |
| ARC (Adaptive Replacement Cache) | Dynamically balances between LRU and LFU | Mixed workloads; recency and frequency matter |
| Random | Evict a random item | Simple; surprisingly competitive in practice |

**LRU implementation.** LRU is most commonly implemented as a doubly-linked list + hashmap. The hashmap provides O(1) lookup; the linked list maintains recency order. On access, move the item to the head. On eviction, remove the tail. Total O(1) per operation.

**ARC.** Maintains four lists: recently used once (T1), frequently used (T2), ghost of T1 (B1), ghost of T2 (B2). When a ghost entry is hit, it indicates the cache is too biased and adjusts dynamically. ARC outperforms LRU on workloads that mix recency and frequency — it is the eviction policy in ZFS.

### Cache Stampede and Thundering Herd
{: #stampede}

**Cache stampede** occurs when a popular cache entry expires and simultaneously N threads all attempt to recompute it, flooding the backend. The classic failure mode for high-traffic systems.

```
Time T:   Key "homepage_feed" expires
Time T+1: 10,000 concurrent requests arrive
Time T+1: All 10,000 miss cache, all hit the database
          DB receives 10,000 simultaneous queries => DB overload
```

**Solutions:**

*Mutex / lock.* The first thread acquires a lock to recompute the value; other threads wait and read from the cache once the lock is released. Simple but adds latency on misses and introduces a lock contention point.

*Probabilistic early expiry (XFetch).* Items are refreshed before they expire, probabilistically:

$$\text{refresh if: } \text{now} - \Delta \cdot \beta \cdot \ln(\text{rand}()) > \text{expiry\_time}$$

Where $\Delta$ is the recomputation time, $\beta$ is a tuning parameter, and rand() is a uniform random number in (0, 1). As expiry approaches, the probability of early refresh increases. Multiple workers independently compute this; one will refresh early, preventing stampede.

*Stale-while-revalidate.* Serve the stale cached value immediately while asynchronously refreshing the cache in the background. Cache-Control HTTP header supports this pattern natively. Clients always get fast responses; the cache is eventually refreshed.

*Redundant cache entries.* Use two keys: one with a short TTL (serves fresh data) and one "fallback" with a longer TTL. On short-TTL miss, serve from the fallback while triggering a background refresh.

### CDN as Distributed Cache
{: #cdn}

A CDN (Content Delivery Network) is a globally distributed cache of static and dynamic content. Edge nodes are colocated with ISPs near end users.

```
[User in Tokyo]   --> [CDN Edge Tokyo]   --> [Origin Server (US)]
[User in London]  --> [CDN Edge London]  --> (cache hit, no origin request)
[User in Sydney]  --> [CDN Edge Sydney]  --> (cache hit, no origin request)
```

**Cache-Control headers** control CDN behaviour:
- `max-age=3600` — cache for 1 hour
- `s-maxage=86400` — cache at CDN for 1 day (overrides max-age for shared caches)
- `stale-while-revalidate=300` — serve stale for 5 min while revalidating
- `no-store` — never cache (for sensitive data)

**Cache invalidation at CDN scale.** Most CDNs support explicit invalidation (purge by URL, by tag, by wildcard). Purge propagates to all edge nodes but takes seconds to minutes. For time-sensitive invalidation, use surrogate keys (cache tags) to invalidate by logical group.

### Redis vs Memcached
{: #redis-memcached}

| Dimension | Redis | Memcached |
|---|---|---|
| Data structures | Rich (string, hash, list, set, sorted set, stream) | String only |
| Persistence | Optional (RDB snapshot, AOF log) | None |
| Replication | Built-in leader-follower and Redis Cluster | Third-party only |
| Lua scripting | Yes | No |
| Pub/Sub | Yes | No |
| Memory efficiency | Higher overhead per key | Lower overhead; slightly more memory-efficient for pure string caching |
| Multi-threading | Single-threaded command processing (I/O multi-threaded) | Multi-threaded |
| Use case | General-purpose: sessions, leaderboards, queues, pub-sub, rate limiting | Pure caching of serialised blobs at very high throughput |

**Choose Redis** for almost every new use case. The richer data structures, persistence options, and built-in clustering make it more versatile. **Choose Memcached** only if you need multi-threaded performance for pure string caching at extreme scale and have no need for persistence or complex data structures.

### Cache Invalidation
{: #invalidation}

Phil Karlton's famous observation: "There are only two hard things in Computer Science: cache invalidation and naming things."

**TTL-based invalidation.** Every cache entry has a time-to-live. After TTL expires, the entry is evicted. Simple but you always have a window of stale data equal to the TTL.

**Event-based invalidation.** When the source data changes, publish an invalidation event. Cache subscribers invalidate the affected entry immediately. Zero stale window but requires reliable event delivery and careful event design.

**Write-through invalidation.** On every write to the database, also write to the cache (or delete the cache entry). The cache is always fresh but writes are slower.

**Versioned keys.** Include a version number in the cache key: `user:123:v7`. When the user updates, increment the version. Old entries become unreachable (eventually evicted by LRU/TTL). No explicit invalidation needed but storage grows until eviction.

| Strategy | Stale window | Implementation complexity | Failure mode |
|---|---|---|---|
| TTL | Up to TTL duration | Trivial | Always some staleness |
| Event-based | Near-zero | High (event pipeline) | Event loss = stale indefinitely |
| Write-through | Zero | Medium | Write amplification |
| Versioned keys | Zero | Low | Storage growth until eviction |

---

## Message Queues and Streaming
{: #messaging}

### Kafka Architecture
{: #kafka}

Apache Kafka is a distributed, fault-tolerant event streaming platform. Its core abstractions:

```
Producer --> [Topic: orders] --> Consumer Group A
                |
           [Partition 0]  [Partition 1]  [Partition 2]
                |               |               |
           [Broker 1]      [Broker 2]      [Broker 3]
                |
         [Offset 0,1,2...] (log-structured, immutable, retained)
```

**Topics and partitions.** A topic is a logical log of events. Topics are split into partitions for parallelism. Each partition is an ordered, immutable sequence of records. Within a partition, ordering is guaranteed. Across partitions within a topic, ordering is not guaranteed.

**Consumer groups.** A consumer group is a set of consumers that cooperate to consume a topic. Kafka assigns partitions to consumers within a group — each partition is consumed by exactly one consumer in the group. This enables parallel consumption: a topic with 12 partitions can have up to 12 consumers in a group consuming in parallel.

**Offsets.** Each record within a partition has a unique offset (monotonically increasing integer). Consumers track their position via offsets. This allows replaying events from any past offset — unlike traditional queues where consumed messages are deleted.

**Retention.** Kafka retains messages for a configurable period (e.g., 7 days) or up to a configurable size. This decouples consumer speed from producer speed and enables multiple consumer groups to consume the same data independently.

**Compacted topics.** Kafka can compact a topic by retaining only the latest record for each key. This enables event-sourcing patterns where the compacted topic represents the latest state of each entity.

| Component | Failure mode | Mitigation |
|---|---|---|
| Broker | Broker goes down | Replication factor >= 3; min.insync.replicas = 2 |
| Partition leader | Leader fails | Kafka elects new leader from ISR (in-sync replicas) |
| Consumer lag | Consumers fall behind | Monitor consumer lag; add consumers; increase partitions |
| Message loss | Producer acks=0 or acks=1 | Use acks=all for durability; enable idempotent producer |

### Delivery Semantics
{: #delivery-semantics}

| Semantic | Guarantee | How | Risk |
|---|---|---|---|
| At-most-once | Message delivered 0 or 1 time | Fire and forget; no retry on failure | Data loss on failure |
| At-least-once | Message delivered 1 or more times | Retry until acknowledged | Duplicate processing |
| Exactly-once | Message delivered exactly once | Kafka transactions + idempotent consumer | Complexity; higher latency |

**At-least-once** is the practical default for most systems. Pair it with idempotent consumers: design message processing so that processing the same message twice produces the same result as processing it once. Use a deduplication key (message ID) checked against a processed-IDs store.

**Exactly-once in Kafka** requires: idempotent producers (producer assigns sequence numbers; broker deduplicates), transactions (atomic write across multiple partitions), and transactional consumers (commit offset atomically with application state change). This is complex but achievable within Kafka. Cross-system exactly-once (Kafka + external database) requires two-phase commit or outbox pattern.

**Outbox pattern.** For exactly-once delivery from database write to Kafka:

```
Transaction: Write to [orders table] AND [outbox table] atomically
CDC process: Read from outbox table -> publish to Kafka -> mark outbox entry as sent
```

The outbox table is in the same database. The CDC process reads it transactionally — if the process crashes after publishing to Kafka but before marking the outbox entry, it publishes again (at-least-once) but the consumer is idempotent.

### Queue vs Pub-Sub vs Stream Processing
{: #queue-pubsub}

| Model | Description | Delivery | Examples |
|---|---|---|---|
| Queue | Messages consumed by exactly one consumer; deleted after consumption | Point-to-point | SQS, RabbitMQ |
| Pub-Sub | Messages broadcast to all subscribers of a topic | One-to-many | SNS, Google Pub/Sub, Redis Pub/Sub |
| Stream | Ordered, replayable log; multiple independent consumer groups | One-to-many + replay | Kafka, Kinesis |

**Queue** is appropriate for task distribution: N workers share a queue; each job is processed by exactly one worker. Good for background jobs, email sending, image processing.

**Pub-Sub** is appropriate for notifications: when an event happens, all interested services should know. A payment completed event should notify the order service, the notification service, and the analytics service. Each gets its own copy.

**Stream** is appropriate when you need both pub-sub semantics and replay: audit logs, event sourcing, reprocessing historical data with a new consumer.

### Backpressure Handling
{: #backpressure}

Backpressure occurs when producers generate events faster than consumers can process them. Without backpressure control, the queue grows unboundedly, consuming memory, until the system falls over.

**Strategies:**

*Rate limiting producers.* Slow down the producer if the queue depth exceeds a threshold. Simple but upstream latency increases.

*Bounded queues.* Use a queue with a maximum capacity. When the queue is full, producers receive an error (fail fast) or block. Bounded queues make backpressure visible rather than hiding it behind unbounded growth.

*Load shedding.* Drop low-priority messages when the system is overloaded. Not acceptable for every domain (financial transactions) but valid for analytics events.

*Consumer scaling.* Automatically add consumer instances when lag exceeds a threshold (KEDA for Kubernetes scales consumers based on Kafka consumer lag).

*Flow control.* Reactive Streams standard defines a pull-based protocol where the consumer signals how many items it can process (`request(N)`). The producer sends at most N items. Implemented in RxJava, Project Reactor, Akka Streams.

> **Interview question:** You have a Kafka topic with 6 partitions. Your consumer group has 10 consumer instances. How many consumers are actively processing messages?
>
> *Exactly 6. In Kafka, each partition is consumed by exactly one consumer within a consumer group. With 10 consumers and 6 partitions, 6 consumers are assigned one partition each and 4 consumers are idle (they are ready to take over if one of the active consumers fails, but they receive no partitions during normal operation). To utilise all 10 consumers, you must increase the number of partitions to at least 10. Note: you can have more partitions than consumers without waste (e.g., 12 partitions, 6 consumers = 2 partitions per consumer). Partitions can be reassigned as you scale consumers up. The tradeoff: more partitions mean higher metadata overhead at the broker and longer leader election during failover.*

---

## Latency Optimisations
{: #latency}

### Percentile Latency Targets
{: #percentiles}

Average latency is a misleading metric. A system with average latency 50ms may have P99 latency of 5 seconds — 1 in 100 requests takes 100x the average. Users who experience high-latency requests are disproportionately more likely to churn.

| Percentile | Meaning | Typical target |
|---|---|---|
| P50 | 50% of requests complete in this time | Best-case user experience |
| P95 | 95% of requests complete in this time | Typical SLO target |
| P99 | 99% of requests complete in this time | Captures tail latency |
| P99.9 | 99.9% of requests complete in this time | High-stakes services |

**Tail latency amplification.** In a system where a request calls 10 services sequentially, the overall P99 latency is dominated by the worst service:

$$P99_{\text{system}} \approx 1 - (1 - P99_{\text{service}})^{10}$$

If each of 10 services has P99 = 1%, the probability that at least one exceeds P99 in a sequential chain is $1 - 0.99^{10} \approx 10\%$. This is why tail latency matters: a small tail in any component becomes a significant tail for the overall request.

**Hedged requests.** For read operations, send the same request to two replicas simultaneously and use whichever responds first. Reduces tail latency at the cost of increased load. Google's Bigtable uses hedged requests to cut P99 latency by 10x at a 5% increase in total requests.

### Connection Pooling
{: #connection-pooling}

Establishing a new TCP connection and database connection is expensive: TCP handshake (~1 RTT), TLS handshake (~2 RTTs), database authentication and session setup (~1 RTT). For a database with 10ms per query, connection setup might add 50ms on top.

Connection pooling reuses established connections. The pool maintains N open connections; application threads borrow a connection, execute queries, and return it to the pool.

**Sizing the pool.** The optimal pool size is not "as large as possible." More connections than the database can service simultaneously causes connection queue contention at the database. A useful heuristic: pool size = (number of CPU cores on the DB server) × 2 + number of disk spindles. PgBouncer is the standard connection pooler for PostgreSQL.

**Connection pool failure modes:**
- Pool exhaustion: all connections in use; new requests queue or fail
- Connection leak: a thread acquires a connection and never returns it; pool shrinks until exhausted
- Stale connections: connections held in the pool for hours become stale (server-side timeout); use `testOnBorrow` or keepalive pings

### Async I/O and Non-Blocking
{: #async-io}

**Blocking I/O.** Each request occupies a thread from request start to response. While waiting for a database query or downstream HTTP call, the thread is blocked but consuming memory and scheduler overhead. Under high concurrency, you run out of threads before running out of CPU.

**Async I/O (event loop).** One or a few threads handle many concurrent I/O operations. When an I/O operation is initiated, the thread registers a callback and immediately handles another request. When the I/O completes, the event loop invokes the callback. Node.js, Nginx, and Netty use this model.

**Comparison:**

| Model | Threads for 10K concurrent requests | CPU utilisation | Programming model |
|---|---|---|---|
| Thread-per-request (blocking) | 10K threads | Low (most threads blocked) | Simple, synchronous |
| Async / event loop | 1-few threads | High | Complex (callbacks/async-await) |
| Reactive (Project Reactor, RxJava) | 1-few threads | High | Functional reactive; steep learning curve |

For I/O-bound services (most web backends), async I/O dramatically improves throughput per instance. For CPU-bound services, threads are appropriate.

### Geographic Distribution
{: #geo-distribution}

Latency is bounded by physics: light travels ~300,000 km/s through fibre at ~67% efficiency, so a New York–London round trip is a minimum of ~70ms. No software optimisation can reduce this.

**Read replicas.** Place read replicas in each geographic region. Route read traffic to the nearest replica. Write traffic still goes to the primary (typically in one region), accepting replication lag of seconds to minutes.

**Multi-region active-active.** Each region has a full copy of the database and accepts both reads and writes. Conflicts (two regions writing the same record simultaneously) must be resolved — last-write-wins, CRDTs, or application-level resolution.

**Latency-based routing.** DNS resolver returns the IP of the nearest endpoint (AWS Route 53 Latency Routing, CloudFlare Load Balancing). Combined with read replicas, this routes each user to their nearest region.

### Protocol Optimisations
{: #protocols}

**HTTP/1.1** opens a new TCP connection per request (or reuses with keep-alive but only one request at a time per connection). Head-of-line blocking: a slow response blocks subsequent requests on the same connection.

**HTTP/2** multiplexes multiple requests over a single TCP connection. Eliminates head-of-line blocking at the HTTP layer. Supports server push. Reduces handshake overhead for many small requests. Use HTTP/2 between clients and load balancers and between services.

**gRPC** is built on HTTP/2 and uses Protocol Buffers (binary serialisation). Advantages over REST+JSON: smaller message size (~5-10x), faster serialisation/deserialisation, strict schema, bidirectional streaming. Ideal for service-to-service communication where payload size and throughput matter.

**WebSocket.** Full-duplex communication over a single TCP connection. Use for real-time features: chat, live notifications, collaborative editing, gaming. The connection upgrade happens over HTTP and then the protocol switches to WebSocket.

| Protocol | Use case | Overhead | Streaming |
|---|---|---|---|
| HTTP/1.1 + JSON | Public REST APIs, simple services | High | No |
| HTTP/2 + JSON | External APIs needing multiplexing | Medium | Partial |
| gRPC (HTTP/2 + Protobuf) | Internal service-to-service | Low | Full (bidirectional) |
| WebSocket | Real-time, bidirectional, long-lived | Low (after handshake) | Full |

---

## Reliability and Graceful Degradation
{: #reliability}

### Circuit Breakers
{: #circuit-breaker}

A circuit breaker wraps a remote call and monitors its failure rate. When failures exceed a threshold, the circuit "opens" and subsequent calls fail fast without attempting the remote call — protecting the downstream service from additional load during an outage.

```
[Client] --> [Circuit Breaker] --> [Remote Service]

States:
CLOSED:    Calls pass through; failures tracked
           If failure_rate > threshold -> transition to OPEN
OPEN:      Calls fail immediately (no remote call); timer starts
           If timer expires -> transition to HALF-OPEN
HALF-OPEN: Allow one probe call through
           If probe succeeds -> transition to CLOSED
           If probe fails -> transition to OPEN
```

**Configuration parameters:**
- `failure_threshold`: percentage of failures that trips the breaker (e.g., 50%)
- `slow_call_threshold`: duration above which a call counts as slow (e.g., 2000ms)
- `slow_call_rate_threshold`: percentage of slow calls that trips the breaker
- `wait_duration_in_open_state`: time to wait before trying HALF-OPEN (e.g., 60s)
- `permitted_calls_in_half_open_state`: number of probe calls in HALF-OPEN (e.g., 3)

Libraries: Resilience4j (Java), Hystrix (deprecated), Polly (.NET), go-circuit-breaker (Go).

### Retry with Exponential Backoff
{: #retry}

Naive retry (retry immediately on failure) can amplify load on a struggling service. Exponential backoff spreads retries over time, giving the service room to recover.

$$\text{wait}(n) = \min(\text{base} \times 2^n + \text{jitter}, \text{max\_wait})$$

Where $n$ is the retry attempt number (0-indexed), $\text{base}$ is the initial wait (e.g., 100ms), and $\text{jitter}$ is a random value in $[0, \text{base}]$ to prevent synchronized retries from multiple clients.

**Example:** base=100ms, max_wait=30s, jitter=random(0, 100ms)

| Attempt | Wait (without jitter) | Wait (with jitter, example) |
|---|---|---|
| 0 (first failure) | 100ms | 142ms |
| 1 | 200ms | 267ms |
| 2 | 400ms | 519ms |
| 3 | 800ms | 873ms |
| 4 | 1600ms | 1723ms |
| 5+ | 30000ms (capped) | ~30000ms |

**Full jitter vs equal jitter.** Full jitter: `wait = random(0, base × 2^n)`. Equal jitter: `wait = base × 2^n / 2 + random(0, base × 2^n / 2)`. AWS recommends full jitter for reducing load on recovering services.

**Retry budget.** To prevent retry storms, track the ratio of retries to total calls across all clients. If the retry ratio exceeds the budget (e.g., 10% of all requests are retries), stop retrying. This prevents a short outage from being amplified 10x by cascading retries.

**Idempotency is prerequisite.** Only retry idempotent operations. Retrying a non-idempotent operation (e.g., charging a credit card) on an ambiguous failure (request succeeded but response was lost) causes double-execution. Use idempotency keys: include a UUID in the request; the server stores the key and returns the cached response on duplicate requests.

### Bulkhead Pattern
{: #bulkhead}

Bulkheads (named after ship hull compartments) isolate resources so that failures in one area do not cascade to others. In software: dedicate separate thread pools, connection pools, or processes to different subsystems.

```
Without bulkhead:
[Shared thread pool, 100 threads]
  |-- Request type A (slow) fills all 100 threads
  |-- Request type B (fast) has no threads; queues and fails

With bulkhead:
[Thread pool A, 60 threads] --> Service A
[Thread pool B, 40 threads] --> Service B
  |-- Service A is slow; its 60 threads fill up
  |-- Service B is unaffected; its 40 threads are free
```

Bulkheads prevent the noisy-neighbour problem where one service consuming all shared resources starves other services. Implement via:
- Separate thread pools per service or request type (Hystrix/Resilience4j bulkhead)
- Separate connection pools per database or downstream service
- Process isolation (separate containers per service tier)
- Resource quotas via cgroups (CPU, memory limits per container)

### Rate Limiting
{: #rate-limiting}

Rate limiting protects services from overload and abuse by capping the number of requests per time window.

**Token bucket.** A bucket holds up to `capacity` tokens. Tokens are added at a constant rate `r` tokens/second. Each request consumes 1 token. If the bucket is empty, the request is rejected.

$$\text{tokens}(t) = \min\left(\text{capacity},\ \text{tokens}(t_0) + r \times (t - t_0)\right)$$

Allows short bursts up to `capacity`. The most natural model for user-facing rate limits.

**Leaky bucket.** Requests enter a bucket at any rate and are processed at a constant rate. Excess requests are dropped or queued. Smooths bursty traffic into a constant output rate. Used for traffic shaping rather than simple rate limiting.

**Fixed window counter.** Count requests in a fixed time window (e.g., per minute). On the boundary, the counter resets. Problem: a burst at the end of one window and the beginning of the next can double the allowed rate.

**Sliding window log.** Maintain a log of request timestamps. On each request, evict timestamps older than the window and count the remaining. Accurate but memory-intensive ($O(n)$ per user per window).

**Sliding window counter.** Approximate the sliding window log by combining the current window count and the previous window count weighted by overlap:

$$\text{rate} = \text{prev\_count} \times \frac{\text{remaining in window}}{\text{window size}} + \text{current\_count}$$

Accurate to ~0.003% in practice, $O(1)$ memory per user. Used in Cloudflare's rate limiter.

**Distributed rate limiting with Redis:**

```lua
-- Sliding window counter in Redis (Lua script for atomicity)
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])

-- Remove timestamps older than window
redis.call('zremrangebyscore', key, '-inf', now - window)
local count = redis.call('zcard', key)

if count < limit then
  redis.call('zadd', key, now, now)
  redis.call('expire', key, window / 1000)
  return 1  -- allowed
else
  return 0  -- rate limited
end
```

### Timeout Strategies
{: #timeouts}

Every remote call must have a timeout. Without timeouts, a slow downstream can hold threads indefinitely, eventually exhausting the thread pool and causing the entire service to fail.

**Setting timeouts.** Set timeouts at the P99 latency of the downstream service under normal conditions, plus a safety margin. If the downstream P99 is 200ms, set a timeout of 300-500ms. Do not use "generous" timeouts (5s, 30s) — they provide little protection.

**Timeout budget / deadline propagation.** In a chain of services (A calls B, which calls C), propagate the remaining timeout. If A's request timeout is 500ms and A spent 100ms before calling B, B's timeout should be at most 400ms. gRPC supports deadline propagation natively. Implement in other frameworks by including the deadline timestamp in the request header.

**Timeout hierarchy.** Set client timeout < server timeout. If the client times out and cancels the request, the server should also stop processing. If the server timeout is shorter than the client timeout, the server stops but the client is still waiting — wasted client resources.

| Timeout type | What it covers | Typical value |
|---|---|---|
| Connection timeout | TCP handshake, TLS handshake | 100ms–1s |
| Read timeout | Time for server to start responding | Per-service P99 + margin |
| Write timeout | Time to send the request body | Per-payload size |
| Idle timeout | Keep-alive connection maximum idle time | 60s–300s |

### Fallback Patterns
{: #fallbacks}

When a service is unavailable or slow, fallbacks define the degraded response.

**Cached response.** Return the last known good response from cache. Appropriate when stale data is acceptable (product catalog, user preferences, news feed from 5 minutes ago). The cache entry must be maintained beyond its normal TTL as a "stale fallback" — use `stale-if-error` HTTP cache directive.

**Default value.** Return a safe default when the service is unavailable. A recommendation service that is down returns a curated default list of popular items. A personalisation service that is down returns unPersonalised content.

**Graceful degradation.** Remove the failing feature from the response rather than failing the entire request. A product page that cannot load review scores returns the product details without the review section rather than returning an error page.

**Fail fast with clear error.** For some features, it is better to fail immediately with a clear user message than to return misleading degraded data. A payment processing service that is down should return "payment service temporarily unavailable, try again in a few minutes" rather than silently failing.

<div class="post-flow" role="group" aria-label="Fallback decision chain">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Service call fails or times out</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Is there a fresh cached result? Use it.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Is there a stale cached result? Use it with a staleness indicator.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Is a default value appropriate? Return the default.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">No fallback available. Fail fast with a clear, user-friendly error message.</span></li>
  </ol>
</div>

### Health Checks
{: #health-checks}

Load balancers use health checks to determine which instances can receive traffic. An unhealthy instance is removed from the rotation.

**Liveness check.** Is the process alive? A simple "am I responding?" check — if the process is deadlocked or OOM-killed, it does not respond. Used by Kubernetes to decide if a container should be restarted.

**Readiness check.** Is the instance ready to receive traffic? A deeper check: has the database connection pool initialised, have configuration files been loaded, are dependent services reachable? An instance that is alive but not ready should not receive traffic.

**Deep health check.** Does the service's critical path work end-to-end? Execute a minimal representative query — write a test record and read it back, for example. Only appropriate for low-frequency checks (every minute) as it creates load.

**Health check design principles:**
- Health checks must respond quickly (<100ms) and must not create load on the database or downstream services
- Include a version endpoint (`/version`) that returns the currently deployed commit hash — invaluable for debugging which version is on which instance
- Report degraded state separately from failing state: a service that is alive but slow should report degraded so the load balancer can deprioritise it without removing it entirely

---

## Failure Modes and Detection
{: #failure-modes}

### Network Partitions and Split-Brain
{: #network-partitions}

A **network partition** is a failure in the network that prevents some nodes from communicating with others. In a distributed database, this forces the CAP choice: sacrifice consistency (serve possibly stale data) or availability (refuse requests that cannot be validated against all replicas).

**Split-brain** in a leader-election system occurs when two nodes simultaneously believe they are the leader. Classic scenario: the leader is isolated from followers by a network partition. The followers elect a new leader (the partition looks like a failure). The old leader, still able to serve some clients, continues accepting writes. Now two nodes accept writes — when the partition heals, their states must be reconciled.

**Prevention:**
- Fencing tokens: each leader has an epoch number; followers reject requests from leaders with a lower epoch than the latest observed
- Quorum-based leadership: a leader must maintain a lease confirmed by a quorum of followers; if it cannot reach quorum, it voluntarily steps down
- Single-region deployment: eliminates inter-datacenter partitions but sacrifices geographic redundancy

### Cascading Failures
{: #cascading}

Cascading failures occur when a failure in one component propagates to dependent components, causing progressively more components to fail.

**Classic cascade:**

```
DB is slow (high latency) 
  -> App server requests pile up (threads blocked waiting for DB)
  -> Thread pool exhausted 
  -> Load balancer health check times out 
  -> Load balancer marks instance unhealthy 
  -> Remaining instances receive more traffic 
  -> They also slow down 
  -> All instances unhealthy -> service down
```

**Prevention:**
- Circuit breakers: fail fast instead of letting slow calls block threads
- Bulkheads: isolate thread pools so one service's slowness does not exhaust shared resources
- Load shedding: drop low-priority requests when the system is overloaded rather than queuing them
- Timeouts: every remote call must have a timeout; blocked threads are the primary cascade mechanism

**Retry storms.** When a service recovers from an outage, all clients that have been waiting begin retrying simultaneously, creating a traffic spike that can overwhelm the recovering service — causing another outage. Mitigation: exponential backoff with full jitter, retry budgets, and gradual traffic restoration.

### Distributed Deadlocks
{: #deadlocks}

Distributed deadlocks occur when two or more services hold resources that the others need and each waits for the other to release.

```
Service A holds lock on resource X, waits for lock on resource Y
Service B holds lock on resource Y, waits for lock on resource X
Result: neither can proceed
```

**Detection:** cycle detection in the wait-for graph. Maintain a graph where an edge from A to B means A is waiting for a resource held by B. A deadlock exists if the graph contains a cycle. This requires a centralised lock manager or distributed cycle detection algorithms (Chandy-Misra-Haas).

**Prevention:**
- Lock ordering: all services acquire locks in the same global order; deadlocks cannot form (no circular wait)
- Timeout and abort: acquire locks with a timeout; if timeout expires, abort and retry
- Optimistic concurrency: avoid locks entirely; detect conflicts at commit time (e.g., database CAS operations)
- Distributed transactions with two-phase commit: the coordinator detects and resolves deadlocks by aborting one participant

### Data Corruption and Checksums
{: #data-corruption}

Data corruption occurs when data is modified without the application's knowledge — hardware failure, cosmic rays flipping bits, buggy software, truncated writes. Silent corruption is the most dangerous failure mode: the system continues operating but produces wrong results.

**Detection layers:**

*Checksum at rest.* Compute a checksum (CRC32, SHA-256) of data when writing; verify it when reading. Modern filesystems (ZFS, Btrfs) do this automatically. Databases can optionally compute page checksums.

*Checksum in transit.* TLS provides integrity guarantees (HMAC) for data in transit. TCP checksum is a weak 16-bit CRC — insufficient for detecting all corruption; rely on application-layer checksums for critical data.

*End-to-end checksums.* Compute a checksum of the original data at the source and verify it at the destination. Catches corruption introduced anywhere in the pipeline — network, storage, processing. Include the checksum in the data envelope (API response, file format).

*Periodic data audits.* Schedule offline jobs that read data from storage and verify checksums. Detect corruption that occurred between writes and reads.

**Merkle trees.** Cassandra and DynamoDB use Merkle trees (hash trees) for anti-entropy repair: compute a hash of each row, then a hash of hashes, building up to a tree root. Compare Merkle tree roots between replicas to detect divergence; walk the tree to identify the specific records that differ.

### Observability: Metrics, Logs, Traces
{: #observability}

Observability is the ability to understand the internal state of a system from its external outputs. The three pillars:

**Metrics.** Numerical measurements over time. Aggregated by dimension (service, endpoint, region). Used for dashboards and alerting.

| Metric type | Description | Example |
|---|---|---|
| Counter | Monotonically increasing; measures total count | `http_requests_total` |
| Gauge | Point-in-time value; can increase or decrease | `active_connections`, `memory_bytes` |
| Histogram | Distributes values into buckets; enables percentiles | `http_request_duration_seconds` |
| Summary | Precomputed percentiles; less flexible than histograms | `rpc_duration_seconds{quantile="0.99"}` |

Prometheus is the standard metric collection system for cloud-native services. Grafana provides dashboards over Prometheus data.

**Logs.** Structured records of discrete events. Use structured logging (JSON) rather than plain text — structured logs are queryable by field. Include a trace ID in every log line to correlate logs across services.

Key log events to emit: request received, request completed (with duration and status), external service call (with duration and outcome), errors with stack traces, and business events (order placed, payment processed).

**Traces.** Distributed traces follow a single request as it travels across multiple services. Each service emits spans (start time, duration, service name, operation name) that are joined by a shared trace ID.

```
Trace ID: abc123
  [API Gateway]  0ms - 450ms
    [Auth Svc]   10ms - 50ms
    [Order Svc]  55ms - 420ms
      [DB query] 60ms - 380ms  <-- bottleneck
      [Cache]    385ms - 395ms
```

OpenTelemetry is the standard instrumentation library. Jaeger and Zipkin are open-source trace backends. Datadog, Honeycomb, and AWS X-Ray are managed options.

**The three pillars together.** Metrics tell you *something is wrong*. Traces tell you *where it is wrong*. Logs tell you *why it is wrong*. A complete observability strategy requires all three.

### SLOs, SLAs, and SLIs
{: #slo}

**SLI (Service Level Indicator).** A metric that measures a specific aspect of service quality. Examples: request success rate, P99 latency, error rate, availability percentage.

**SLO (Service Level Objective).** An internal target for an SLI. "P99 latency < 200ms, measured over a rolling 28-day window." SLOs are the operational contract the team holds itself to.

**SLA (Service Level Agreement).** An external commitment to customers, with penalties for breach. SLAs are typically looser than SLOs — the SLO is the internal target, the SLA is the external guarantee.

**Error budget.** If the SLO is 99.9% availability, the error budget is 0.1% = 43.8 minutes/month of allowed downtime. The error budget is consumed by outages, deployments, and experiments. If the budget is exhausted, new deployments are frozen until the budget resets.

**SLO-based alerting.** Alert on error budget consumption rate rather than on raw metrics:
- Burn rate: alert if the error budget is being consumed N times faster than the allowed rate
- Multi-window alert: require both a short window (high burn, immediate) and a long window (lower burn, sustained) to reduce false positives

| Burn rate | Time to budget exhaustion | Alert window |
|---|---|---|
| 14.4x | 1 hour | 1h short + 5m long |
| 6x | 2.5 hours | 6h short + 30m long |
| 3x | 5 days | 1d short + 6h long |
| 1x | 30 days | No alert (healthy) |

> **Interview question:** Your service SLO is P99 latency < 300ms. During a peak traffic period, your P99 is 450ms. Walk through how you would diagnose and address this.
>
> *Step 1: is this load-induced or structural? Pull traces from the high-latency period. If the slow component is the database, check database metrics: CPU, I/O, connection pool utilisation, query plan changes. If the slow component is a downstream service, check that service's latency metrics. Step 2: database bottleneck. High CPU + slow queries = missing index or query plan regression (check EXPLAIN ANALYZE; has a new query been deployed?). High I/O = data volume crossed a threshold where index fits no longer fit in memory. High connection pool utilisation = pool sized too small or too many queries per request. Step 3: downstream service bottleneck. Check if the downstream service's own P99 degraded — this is a dependency failure, not a capacity problem. Enable circuit breaker if not already active; return cached or default response while the dependency recovers. Step 4: load-induced. If queries are fine but you are simply receiving more traffic than the current fleet can handle, scale horizontally (add instances) or vertically (larger instances). Check autoscaling triggers — did they fire? Was there a delay? Step 5: fix and verify. Deploy fix; verify P99 drops below 300ms on production traffic. Add an alert for P99 > 250ms to catch regressions before they breach the SLO.*

---

## A/B Testing and Feature Flags
{: #ab-testing}

### Experiment Design
{: #experiment-design}

A/B testing (controlled experiment) measures the causal effect of a change by comparing a control group (A, existing experience) with a treatment group (B, new experience). Randomised assignment eliminates confounding variables.

**Randomisation unit.** The entity randomised into control or treatment. Choosing the wrong unit causes SUTVA (Stable Unit Treatment Value Assumption) violations — treatment of one unit affects other units.

| Unit | Use when | Risk |
|---|---|---|
| User ID | Most experiments; persistent assignment across sessions | Network effects (social features can violate SUTVA) |
| Session ID | Short-lived experiments; users should see both variants | Carryover bias between sessions |
| Device ID | Mobile experiments where user accounts are not logged in | Multi-device users see inconsistent experiences |
| Cookie | Anonymous visitors | Cookie churn; returning users may be re-randomised |
| Request | Performance experiments; each request independent | No per-user consistency (poor UX for feature changes) |

**Minimum detectable effect (MDE).** The smallest effect size worth detecting. Smaller MDE requires larger sample size. Define MDE before running the experiment — choosing it based on what you observed is p-hacking.

$$n = \frac{2 \cdot (z_{\alpha/2} + z_{\beta})^2 \cdot \sigma^2}{\delta^2}$$

Where $n$ is the required sample size per group, $z_{\alpha/2}$ is the critical value for Type I error rate (1.96 for 95% confidence), $z_{\beta}$ is the critical value for desired power (0.84 for 80% power), $\sigma^2$ is the variance of the metric, and $\delta$ is the MDE.

**Key experiment design decisions:**

| Decision | Recommendation |
|---|---|
| Traffic allocation | Start 5-10% treatment; scale after sanity checks pass |
| Duration | At least 1-2 business cycles (1-2 weeks) to capture weekly seasonality |
| Primary metric | One pre-registered primary metric; multiple secondary |
| Guardrail metrics | Metrics that must not regress (latency, error rate, revenue) |

### Statistical Significance
{: #statistics}

**p-value.** The probability of observing an effect as large as (or larger than) the observed one, assuming the null hypothesis is true (no real effect). A p-value < 0.05 means that if there were truly no effect, you would only see this large a difference 5% of the time due to chance.

**p-value misinterpretations:**
- A p-value of 0.05 does NOT mean there is a 95% probability the effect is real
- A p-value of 0.05 does NOT mean the effect size is meaningful (statistical significance != practical significance)
- A p-value below threshold does NOT mean the null hypothesis is false

**Confidence interval.** A 95% CI for the treatment effect is an interval that, if you ran the experiment 100 times, would contain the true effect 95 of those times. Report effect size with confidence intervals, not just p-values.

**Multiple hypothesis testing.** If you test 20 metrics, you expect 1 to show p < 0.05 by chance. Apply a correction:
- Bonferroni correction: divide the significance threshold by the number of tests ($\alpha' = \alpha / m$). Conservative.
- Benjamini-Hochberg procedure: controls false discovery rate (FDR) rather than family-wise error rate. Less conservative.

**Sequential testing / peeking.** Stopping an experiment early because it looks significant is a common mistake. The p-value crosses 0.05 randomly during the experiment even with no true effect — if you check repeatedly and stop when it does, you inflate the Type I error rate. Solutions: pre-register the end date; use sequential testing methods (SPRT, mSPRT) that are designed for continuous monitoring.

### Feature Flag Architecture
{: #feature-flags}

Feature flags (feature toggles) decouple deployment from release. Code is deployed to production with a flag disabled; the flag is enabled for a subset of users or all users when ready.

```
[Feature Flag Service]
        |
    [Flag Config]
     - flag_name: "new_checkout_flow"
     - enabled: true
     - rollout: { type: "user_id_percentage", percentage: 10 }
     - targeting: { user_segments: ["beta_users"] }
        |
[SDK integrated in application code]
        |
if (flagService.isEnabled("new_checkout_flow", userId)):
    renderNewCheckout()
else:
    renderOldCheckout()
```

**LaunchDarkly pattern.** A centralised flag service stores flag configurations. SDKs embedded in application code evaluate flags locally using a cached copy of the configuration (fetched from the service and updated via streaming). Flag evaluation is synchronous and in-memory — no network call on each flag check.

**Flag storage.** Flag configurations stored in a distributed key-value store (Redis, DynamoDB). SDKs subscribe to changes via SSE or polling. Updates propagate to all instances within seconds.

**Targeting rules.** Flags can be evaluated per user based on:
- User attributes (plan, country, registration date, beta_opt_in)
- Random percentage (10% of users by user ID hash — deterministic; the same user always gets the same assignment)
- Explicitly listed user IDs

**Flag lifecycle.** Flags accumulate technical debt if not cleaned up. Every flag should have:
- An issue or ticket tracking its intended removal date
- A rollout plan: 0% -> 5% -> 25% -> 50% -> 100% -> remove flag
- A kill switch path: the flag can be turned off instantly if a regression is detected

| Flag type | Use case | Expected lifetime |
|---|---|---|
| Release flag | Roll out new feature gradually | Weeks; remove after 100% rollout |
| Experiment flag | A/B test | Duration of experiment; remove after decision |
| Ops flag | Kill switch for a feature | Indefinite but should be rare |
| Permission flag | Enable feature for specific user segments | Long-lived; clean up if segment dissolves |

### Holdout Groups
{: #holdouts}

A **holdout group** is a persistent control group excluded from all experiments in a product area. Unlike regular A/B tests (which end after a decision), holdouts continue for months to measure cumulative effects.

**Why holdouts.** Individual experiments may show positive effects that reverse when combined with other changes. A holdout measures the cumulative impact of all changes over a period — useful for answering "have our last 6 months of changes actually improved the product?"

**Holdout design:**
- Reserve 1-5% of users as a holdout who see no changes in a product area
- Measure the treatment effect between holdout and the full-experience group monthly or quarterly
- Holdouts must be excluded from all experiments in the relevant area — cannot be in treatment or control of another experiment

**Long-term experiments.** Some effects only manifest over months: habit formation, churn prediction, subscription renewal. Short experiments (2 weeks) miss these. Long-term holdouts and cohort analysis are used to measure sustained effects.

---

## Worked Examples
{: #worked-examples}

### URL Shortener
{: #url-shortener}

**Requirements:** shorten URLs, redirect with low latency, track click analytics. Scale: 100M new URLs/day, 10B redirects/day.

**Scale estimation:**
- Write QPS: 100M / 86,400 ≈ 1,160 writes/s
- Read QPS: 10B / 86,400 ≈ 116,000 reads/s (100:1 read:write ratio)
- Storage: 100M × 365 × 5 years × 100 bytes ≈ 18 TB over 5 years

**Architecture:**

```
[Client] --> [CDN (cache redirects)] --> [Load Balancer]
                                              |
                                     [Redirect Service]
                                              |
                                     [Cache (Redis)]
                                              |
                                     [URL DB (Cassandra or MySQL)]

[Write Path]
[Client] --> [API Gateway] --> [Shorten Service] --> [URL DB]
                                      |
                              [Short Code Generator]
```

**Hashing strategy.** Generate a unique 7-character short code (base62: a-z, A-Z, 0-9 = 62^7 = 3.5 trillion codes). Two approaches:

*Random generation + collision check.* Generate a random 7-character string; check if it exists in the DB; retry if it does. Simple but requires a round-trip to the DB for each generation. Collision probability at 100M URLs is 100M / 3.5T ≈ 0.003% — acceptable.

*Counter + base62 encoding.* Maintain a distributed counter (auto-increment in DB, or a dedicated counter service like ZooKeeper). Base62-encode the counter value. Guaranteed uniqueness but the counter is a single point of failure and the codes are sequential (guessable). Use a 4-byte random salt to prevent enumeration.

*MD5 / SHA-256 + take first 7 characters.* Hash the URL and take the first 7 characters. Deterministic (same URL always produces the same code) — but collisions occur more frequently than random generation when the URL space is not uniformly distributed.

**Redirect flow:**

```
GET /abc1234
  1. Check CDN (cache hit -> 301 redirect immediately)
  2. Miss: check Redis (cache hit -> 301 redirect; populate CDN)
  3. Miss: query DB (Cassandra lookup by short code)
  4. Store in Redis (TTL = 24h); return 301 redirect
```

Use **301 (Permanent Redirect)** for caching at the client and CDN — future requests are served from cache without hitting the server. Use **302 (Temporary Redirect)** if you want every redirect to hit your servers (for analytics or if the destination may change).

**Analytics.** Click events are write-heavy, high-volume. Do not write directly to the URL database:

```
Redirect Service --> [Kafka Topic: clicks] --> [Stream Processor (Flink)]
                                                      |
                                              [Analytics Store (Cassandra or ClickHouse)]
```

Each click event: `{short_code, timestamp, referrer, user_agent, ip_geo_country}`. Stream processor aggregates into per-URL click counts by time window.

**Failure modes:**

| Component | Failure | Mitigation |
|---|---|---|
| Short code DB | Node failure | Cassandra replication factor = 3 |
| Redis | Cache down | Fall through to DB; degraded performance |
| Analytics Kafka | Lag | Analytics is eventual; acceptable delay |
| Counter service | Counter node failure | Use DB auto-increment with a single shard; or range-based counter allocation |

### Twitter/X Feed
{: #twitter-feed}

**Requirements:** users post tweets; followers see tweets in their feed; support 300M DAU, 500M tweets/day, celebrities with 50M followers.

**Scale estimation:**
- Write QPS: 500M / 86,400 ≈ 5,800 tweets/s
- Read QPS (300M DAU × 50 reads / 86,400): ≈ 174,000 reads/s

**Core design decision: fanout on write vs fanout on read.**

*Fanout on write (push model).* When a user tweets, immediately push the tweet into each follower's feed (pre-computed timeline). Feed reads are fast — just read from the user's pre-built timeline cache.

- Read: O(1) — fetch user's timeline from cache
- Write: O(followers) — fan out to all followers' timelines
- Problem: celebrity with 50M followers creates 50M write operations per tweet → fanout storm

*Fanout on read (pull model).* When a user requests their feed, query the tweets of everyone they follow, merge, and sort. Feed reads are expensive — requires N queries for N followees.

- Read: O(following_count × recent_tweets_per_user) — expensive at read time
- Write: O(1) — just write the tweet once
- Problem: users who follow 2,000 accounts with high tweet volume create expensive read-time merges

**Hybrid approach (Twitter's actual design):**

```
[Tweet by regular user (< 1M followers)]
  -> Fanout on write to followers' timeline caches

[Tweet by celebrity (> 1M followers)]  
  -> Store tweet only (no fanout)
  -> At read time, inject celebrity tweets into the timeline
```

```
Timeline Read:
  1. Fetch pre-computed timeline from Redis (most entries)
  2. For each celebrity the user follows:
     - Fetch last 20 tweets from celebrity's tweet cache
  3. Merge + sort by timestamp
  4. Return unified timeline
```

**Data model:**

```
Tweets table (Cassandra):
  tweet_id (time-based UUID), user_id, content, created_at, media_urls, likes, retweets

User timeline cache (Redis sorted set):
  Key: timeline:{user_id}
  Score: tweet timestamp (Unix ms)
  Member: tweet_id
  Max entries: 800 (older tweets fetched from cold storage)

Followers index (Cassandra):
  user_id -> [follower_id_1, follower_id_2, ...] (paginated)
```

**Fanout service:**

```
[Tweet] --> [Kafka: new_tweets] --> [Fanout Service (workers)]
                                          |
                                   Reads follower list
                                          |
                                   Batch writes to Redis timelines
                                   (pipeline write: ZADD timeline:{follower_id} ts tweet_id)
```

The fanout service consumes from Kafka asynchronously. Regular users' timelines are populated within seconds. Celebrity tweets appear with a short delay (acceptable for a social feed).

**Storage tiers:**

```
Hot tier: Redis sorted sets (last 800 tweets per user) -> 1ms reads
Warm tier: Cassandra (tweets from last 30 days) -> 10ms reads
Cold tier: S3 (older tweets, compressed) -> seconds
```

### Distributed Rate Limiter
{: #rate-limiter}

**Requirements:** rate limit API requests per user (1,000 requests/minute), globally consistent across all server instances, low latency (<5ms overhead per request).

**Why distributed.** A local in-memory rate limiter counts requests only on the current server. With 10 servers and 1,000 requests/minute limit, each server allows 1,000 requests/minute locally = 10,000 requests/minute globally. A distributed rate limiter uses shared state (Redis) to enforce limits globally.

**Algorithm: sliding window counter with Redis:**

```
On each request for user_id:
  current_window = floor(now_ms / 60000)  # minute-level window
  prev_window = current_window - 1
  
  key_current = "rl:{user_id}:{current_window}"
  key_prev    = "rl:{user_id}:{prev_window}"
  
  # Atomic: get both counts, increment current
  PIPELINE:
    GET key_prev
    INCR key_current
    EXPIRE key_current 120  # keep for 2 windows
  
  elapsed_fraction = (now_ms % 60000) / 60000
  estimated_count = prev_count * (1 - elapsed_fraction) + current_count
  
  if estimated_count > limit:
    return 429 Too Many Requests
  else:
    return allow
```

This Lua script executes atomically on Redis:

```lua
local key_prev = KEYS[1]
local key_curr = KEYS[2]
local limit    = tonumber(ARGV[1])
local fraction = tonumber(ARGV[2])

local prev  = tonumber(redis.call('GET', key_prev) or 0)
local curr  = tonumber(redis.call('INCR', key_curr))
redis.call('EXPIRE', key_curr, 120)

local estimated = prev * (1 - fraction) + curr
if estimated > limit then
  -- Undo the increment
  redis.call('DECR', key_curr)
  return 0
end
return 1
```

**Architecture:**

```
[Client] --> [API Gateway / App Server]
                      |
              [Rate Limit Middleware]
                      |
               [Redis Cluster]
               (rate limit state)
```

**Handling Redis failure.** If Redis is unavailable, the rate limiter cannot enforce limits. Options:
- Fail open: allow all requests when Redis is down (risk: abuse during outage)
- Fail closed: reject all requests when Redis is down (risk: service unavailable for all users)
- Local fallback: fall back to a per-instance local rate limit at a fraction of the global limit (10 servers × 100/min local limit ≈ 1000/min global, approximately correct)

**Rate limit response headers:**

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 1714291200   # Unix timestamp when window resets
Retry-After: 12                 # seconds until next request allowed (on 429)
```

**Tiered rate limits.** Different users have different limits:

```
Free tier:    100 req/min
Pro tier:   1,000 req/min
Enterprise: 10,000 req/min
```

Store the tier limit in the Redis hash for the user (or look up from a user service with a local cache). The rate limit middleware reads the limit from the local cache on each request — only falls back to the user service on cache miss.

**Failure modes:**

| Failure | Effect | Mitigation |
|---|---|---|
| Redis hot key | Single user key overwhelms one Redis shard | Spread user keys across shards using consistent hashing on user_id |
| Redis latency spike | Rate limit check adds >5ms to every request | Use Redis pipeline; set aggressive timeout; fall back to local limit on timeout |
| Counter overflow | Integer overflow for very active users | Redis INCR max is 2^63 - 1; not a practical concern with per-window keys |
| Clock skew | Servers have different clocks; window boundaries misalign | Use Redis server time (TIME command) rather than application server time |

> **Interview question:** Your rate limiter allows 1,000 requests per minute per user. A user sends 999 requests in the last second of minute 1 and 999 requests in the first second of minute 2. How does a fixed-window counter handle this, and how does a sliding window fix it?
>
> *With a fixed window counter: the counter for minute 1 has 999 requests (under limit). The counter resets at the start of minute 2, so the 999 requests in minute 2 also pass (under limit). In the 2-second window spanning the boundary, the user sent 1,998 requests — nearly 2x the intended limit. This is the boundary exploit. A sliding window evaluates the count over the last 60 seconds at any point in time, not aligned to fixed minute boundaries. When the user sends request 999 in the first second of minute 2, the sliding window looks back 60 seconds and counts 999 (from the last second of minute 1) + the current minute 2 count. At 999 + 999 = 1,998, the requests are rejected. The sliding window log (exact) is O(n) memory per user. The sliding window counter approximation (using two adjacent fixed windows) is O(1) per user and accurate to within 0.003% under realistic traffic patterns — this is the recommended implementation.*

---

## Also Read

**[Agentic System Design](/blogs/agentic-system-design/)** — designing reliable LLM-powered agents: architectures, tool use, memory, multi-agent coordination, and the engineering challenges that make agents hard to productionise.
