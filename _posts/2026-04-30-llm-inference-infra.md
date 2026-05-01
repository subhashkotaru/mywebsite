---
title: "LLM Inference Infrastructure"
date: 2026-04-30
display_order: 9
description: "From basic web serving to production LLM deployment — containers, Kubernetes, Envoy, Gateway API, and the infrastructure decisions that determine whether your model can handle real traffic."
tags: [ml-systems, inference, llm, kubernetes, infrastructure]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#web-serving-basics">Web Serving Basics</a>
      <ul class="post-toc-sublist">
        <li><a href="#request-response">The Request-Response Model</a></li>
        <li><a href="#threads-processes">Threads, Processes, and Event Loops</a></li>
        <li><a href="#reverse-proxies">Reverse Proxies and Load Balancers</a></li>
        <li><a href="#why-stateless">Why Stateless Servers Are Simpler</a></li>
      </ul>
    </li>
    <li><a href="#containers">Containers: Portable Environments</a>
      <ul class="post-toc-sublist">
        <li><a href="#what-is-a-container">What Is a Container?</a></li>
        <li><a href="#images">Container Images</a></li>
        <li><a href="#container-vs-vm">Container vs VM</a></li>
        <li><a href="#gpu-containers">GPU Containers</a></li>
      </ul>
    </li>
    <li><a href="#kubernetes">Kubernetes: Orchestrating Many Containers</a>
      <ul class="post-toc-sublist">
        <li><a href="#why-k8s">Why Kubernetes?</a></li>
        <li><a href="#k8s-primitives">Core Primitives</a></li>
        <li><a href="#k8s-control-plane">Control Plane vs Data Plane</a></li>
        <li><a href="#scheduling">Scheduling and GPU Resources</a></li>
        <li><a href="#k8s-networking">Networking: Services and DNS</a></li>
        <li><a href="#k8s-storage">Storage: ConfigMaps, Secrets, PVCs</a></li>
        <li><a href="#health-checks">Health Checks and Readiness Gates</a></li>
        <li><a href="#autoscaling">Autoscaling</a></li>
      </ul>
    </li>
    <li><a href="#envoy">Envoy: The Universal Proxy</a>
      <ul class="post-toc-sublist">
        <li><a href="#what-is-envoy">What Is Envoy?</a></li>
        <li><a href="#envoy-architecture">Envoy Architecture</a></li>
        <li><a href="#xds">xDS: Dynamic Configuration</a></li>
        <li><a href="#envoy-filters">Filter Chains and Extension Points</a></li>
        <li><a href="#envoy-observability">Observability Built In</a></li>
      </ul>
    </li>
    <li><a href="#gateway-api">Kubernetes Gateway API</a>
      <ul class="post-toc-sublist">
        <li><a href="#ingress-vs-gateway">Ingress vs Gateway API</a></li>
        <li><a href="#gateway-resources">Gateway API Resources</a></li>
        <li><a href="#traffic-splitting">Traffic Splitting and Canaries</a></li>
        <li><a href="#llm-gateway">LLM-Specific Gateway Concerns</a></li>
      </ul>
    </li>
    <li><a href="#service-mesh">Service Mesh: East-West Traffic</a>
      <ul class="post-toc-sublist">
        <li><a href="#sidecar-model">The Sidecar Model</a></li>
        <li><a href="#mtls">mTLS and Zero-Trust</a></li>
        <li><a href="#mesh-observability">Mesh Observability</a></li>
      </ul>
    </li>
    <li><a href="#llm-serving-stack">LLM Serving Stack</a>
      <ul class="post-toc-sublist">
        <li><a href="#llm-vs-web">Why LLMs Are Different From Web Servers</a></li>
        <li><a href="#serving-layers">The Four Layers</a></li>
        <li><a href="#engine-layer">Engine Layer: vLLM, TGI, TensorRT-LLM</a></li>
        <li><a href="#request-routing">Request Routing for LLMs</a></li>
        <li><a href="#pd-disaggregation">Prefill-Decode Disaggregation</a></li>
        <li><a href="#kv-cache-routing">KV Cache-Aware Routing</a></li>
        <li><a href="#llm-autoscaling">LLM Autoscaling</a></li>
      </ul>
    </li>
    <li><a href="#llm-d">llm-d: Distributed LLM Serving</a>
      <ul class="post-toc-sublist">
        <li><a href="#llm-d-arch">Architecture Overview</a></li>
        <li><a href="#llm-d-scheduler">Disaggregated Inference Scheduler</a></li>
        <li><a href="#llm-d-gateway">Gateway and Routing</a></li>
      </ul>
    </li>
    <li><a href="#observability">Observability for LLM Serving</a>
      <ul class="post-toc-sublist">
        <li><a href="#metrics">Key Metrics</a></li>
        <li><a href="#tracing">Distributed Tracing</a></li>
        <li><a href="#slos">SLOs for LLM APIs</a></li>
      </ul>
    </li>
    <li><a href="#production-checklist">Production Readiness Checklist</a></li>
  </ul>
</nav>

---

## Overview
{: #overview}

Running a language model locally is straightforward — load weights, call `model.generate()`, done. Serving that same model to thousands of concurrent users in production is a fundamentally different problem. You need containers to make the environment reproducible, an orchestrator to run many containers across many machines, a proxy to route and load-balance requests, a gateway to expose the service to the outside world, and on top of all that, an inference engine designed for the unique cost structure of autoregressive generation.

This post builds the mental model from the ground up: how web serving works, how containers package it, how Kubernetes orchestrates it at scale, how Envoy and the Gateway API manage traffic, and finally how all these layers compose into a production LLM serving system. If you are coming from a modeling background, this is the infra layer you need to understand to ship models at scale.

> **Intuition question:** Why can't I just `ssh` into a big server, load my model, and call it done? What breaks as traffic grows?
>
> *A single server is a single point of failure. When it crashes, your service goes down. When traffic spikes, it gets overwhelmed. When you update your model, you have to take the service down. And when you need a GPU, you have to physically provision one — no elastic scaling. Distributed serving solves all of this, but requires orchestration infrastructure to manage the moving parts.*

---

## Web Serving Basics
{: #web-serving-basics}

Before containers and Kubernetes, there was a web server. Understanding its fundamentals gives you the mental model for everything that follows.

### The Request-Response Model
{: #request-response}

At its core, HTTP is a request-response protocol: a client sends a request (verb + path + headers + body), the server processes it, returns a response (status code + headers + body), and the connection closes (or is reused with HTTP/1.1 keep-alive).

```
Client                                 Server
  │                                       │
  │  GET /v1/completions HTTP/1.1         │
  │  Host: api.example.com                │
  │  Content-Type: application/json  ───► │
  │  {"model": "llama3", "prompt": "..."}│
  │                                       │  ← forward to GPU worker
  │                                       │  ← run inference
  │                                       │  ← encode response
  │  HTTP/1.1 200 OK              ◄────── │
  │  Content-Type: application/json       │
  │  {"text": "...generated text..."}     │
  │                                       │
```

For regular web servers this works fine — each request takes milliseconds. For LLMs generating 500 tokens, a single request can take seconds, and the client needs to wait. This is why **streaming** (Server-Sent Events or HTTP chunked transfer) is the dominant pattern for LLM APIs: the server pushes each generated token as it is produced rather than buffering the whole response.

### Threads, Processes, and Event Loops
{: #threads-processes}

How does a server handle multiple requests simultaneously?

<div class="post-flow post-flow--compare" role="group" aria-label="Three concurrency models">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Thread-per-request</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">One OS thread per active request</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Easy to reason about — blocking I/O is fine</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Threads are expensive: ~8MB stack each</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Breaks at 10K+ concurrent connections</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Event loop (async)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Single thread, non-blocking I/O (epoll/kqueue)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Handles 100K+ connections on one core</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Nginx, Node.js, Envoy use this model</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">CPU-heavy work blocks the loop — must offload</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Process pool</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Fork N worker processes, each handles a subset</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Crash isolation: one bad request can't kill all</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Used by Gunicorn (Python WSGI), uWSGI</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">More memory overhead than threads</span></li>
    </ol>
  </div>
</div>

Most LLM serving frontends (FastAPI with uvicorn) use an **async event loop** for the HTTP layer, but offload GPU work to a separate inference process/thread — the event loop handles hundreds of concurrent streaming connections while GPU work happens in the background.

### Reverse Proxies and Load Balancers
{: #reverse-proxies}

A **reverse proxy** sits between clients and your server fleet. Clients see one address; the proxy fans traffic out to whichever backend is available.

```
                    ┌────────────────┐
                    │  Reverse Proxy │
Internet ──────────►│  (Nginx/Envoy) │
                    └───────┬────────┘
                            │  picks a backend
               ┌────────────┼────────────┐
               ▼            ▼            ▼
          Backend-1     Backend-2     Backend-3
          (vLLM)        (vLLM)        (vLLM)
```

Load balancing algorithms:
- **Round-robin**: requests cycle through backends 1→2→3→1→... Simple, works when requests are uniform.
- **Least connections**: send to the backend with fewest active requests. Better for variable-length LLM requests.
- **Random two choices (P2C)**: sample 2 backends randomly, pick the less-loaded one. Near-optimal at scale with no coordination overhead.
- **Consistent hashing**: hash a request attribute (e.g., user ID or prefix hash) to always route to the same backend. Critical for KV cache reuse in LLM serving.

> **Intuition question:** Round-robin is simple and fair. Why does it work poorly for LLM serving?
>
> *LLM requests have wildly variable latency — a request asking for 10 tokens takes 100ms; one asking for 2000 tokens takes 20 seconds. Round-robin sends the same number of requests to each backend regardless of their current load. Result: one backend gets 50 long requests and is saturated; another gets 50 short ones and is idle. Least-connections or P2C measures actual load and routes accordingly.*

### Why Stateless Servers Are Simpler
{: #why-stateless}

A server is **stateless** if any replica can handle any request without needing information stored on another replica. Stateless servers are trivial to scale: add more replicas, load-balance across them, restart any of them without data loss.

LLM serving is **partially stateful** because of the KV cache. A running conversation's KV cache lives on one GPU. If you route the next request to a different GPU, the KV cache must be recomputed from scratch. This forces a choice: either treat the KV cache as ephemeral (stateless routing, pay recompute cost), or route subsequent requests to the same replica (stateful routing, enable prefix caching). Most production systems do prefix-hash-based routing to strike a balance.

---

## Containers: Portable Environments
{: #containers}

### What Is a Container?
{: #what-is-a-container}

A container is a process running in an isolated environment provided by three Linux kernel features:

| Kernel feature | What it isolates | Why it matters |
|---|---|---|
| **Namespaces** | PID, network, filesystem, user, hostname | Container sees its own process tree, IP, filesystem |
| **cgroups** | CPU, memory, I/O limits | Container can't starve its neighbors |
| **Union filesystem** | Layered filesystem (OverlayFS) | Efficient image storage and sharing |

The critical insight: a container is **not a VM**. It shares the host OS kernel. There is no hardware virtualization. Starting a container takes milliseconds; starting a VM takes seconds. The tradeoff is weaker isolation — a kernel exploit in one container can affect others. For trusted workloads (your own ML inference servers), this is an acceptable tradeoff.

### Container Images
{: #images}

A container **image** is a read-only snapshot of a filesystem, built in layers. Each line in a Dockerfile creates a new layer:

```dockerfile
# Layer 1: CUDA + cuDNN runtime from NVIDIA
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Layer 2: Python and system deps (cached aggressively)
RUN apt-get update && apt-get install -y python3-pip && \
    pip install torch==2.3.0 vllm==0.4.2

# Layer 3: model server code (changes often, separate layer)
COPY serve.py /app/serve.py

# Default command when container starts
CMD ["python3", "/app/serve.py"]
```

Layers are **content-addressed** (SHA256 hash). If layer 2 hasn't changed, Docker reuses it from cache. This means the 8GB PyTorch layer is only downloaded once even if you rebuild your server code 50 times. For LLM images that can be 15–20GB, this layering discipline is essential.

> **Intuition question:** Why should model weights NOT be baked into the container image?
>
> *Model weights for a 70B model are 140GB in BF16. Baking them into the image means every rebuild pushes 140GB to your registry, every node that runs the container must download 140GB, and you can't share weights between containers running different server code versions. The right pattern is: weights live in an external volume (PVC backed by object storage or NFS) and are mounted into the container at runtime. The container image stays small and fast to deploy.*

### Container vs VM
{: #container-vs-vm}

```
┌─────────────────────────────────────────────────────────┐
│                        VM                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ Guest OS   │  │ Guest OS   │  │ Guest OS   │        │
│  │ + App      │  │ + App      │  │ + App      │        │
│  └────────────┘  └────────────┘  └────────────┘        │
│  ──────────── Hypervisor ──────────────────────         │
│  ──────────── Host OS + Hardware ──────────────        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   Containers                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │ App        │  │ App        │  │ App        │        │
│  │ (isolated) │  │ (isolated) │  │ (isolated) │        │
│  └────────────┘  └────────────┘  └────────────┘        │
│  ──────────── Container Runtime (containerd) ──────────│
│  ──────────── Host OS + Hardware ──────────────        │
└─────────────────────────────────────────────────────────┘
```

### GPU Containers
{: #gpu-containers}

The standard Linux namespace model does not include GPU. NVIDIA's **Container Toolkit** (`nvidia-container-toolkit`) solves this: it hooks into the container runtime to inject NVIDIA drivers and device files into the container's namespace, giving the container direct access to the physical GPU without copying data through a virtualization layer.

```bash
# Run a container with GPU access
docker run --gpus all \
  -v /models:/models \
  nvcr.io/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
  nvidia-smi
```

In Kubernetes, GPU resources are exposed as **extended resources** (`nvidia.com/gpu`) that pods request explicitly:

```yaml
resources:
  requests:
    nvidia.com/gpu: "2"   # request 2 GPUs
  limits:
    nvidia.com/gpu: "2"
```

The NVIDIA Device Plugin (a DaemonSet) advertises GPU slots to the Kubernetes scheduler so it can place pods correctly.

---

## Kubernetes: Orchestrating Many Containers
{: #kubernetes}

### Why Kubernetes?
{: #why-k8s}

Managing containers manually does not scale. You need to answer: which machine runs which container? What happens if a machine crashes? How do you roll out a new model version without downtime? How do containers find each other? Kubernetes answers all of these.

> **Intuition question:** I can script all of this with bash — start containers on machines, check health with curl, restart on failure. Why do I need Kubernetes?
>
> *The bash script works for 3 servers. At 300, you need: distributed state about what is running where, a scheduler that accounts for resource constraints (GPUs!), health checking at scale, service discovery that updates as pods come and go, declarative rollouts with rollback, secret management, storage provisioning, and autoscaling. Kubernetes is a battle-tested distributed system for exactly these problems. Building it yourself would take years.*

### Core Primitives
{: #k8s-primitives}

Kubernetes has a small set of objects that compose into any deployment:

<div class="post-flow" role="group" aria-label="Kubernetes object hierarchy">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Pod</strong> — one or more containers sharing a network namespace and storage. The smallest deployable unit. Usually managed by a higher-level object.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Deployment</strong> — declares desired state: "run 5 replicas of this pod spec." Handles rollouts, rollbacks, and self-healing.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Service</strong> — stable DNS name + virtual IP that load-balances across matching pods. Pods come and go; the Service endpoint is stable.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>StatefulSet</strong> — like Deployment but gives each pod a stable identity (pod-0, pod-1) and stable storage. Used for stateful workloads.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>DaemonSet</strong> — runs exactly one pod per node. Used for infrastructure: GPU plugins, log collectors, monitoring agents.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Job / CronJob</strong> — runs a pod to completion (batch). CronJob schedules them on a cron expression.</span></li>
  </ol>
</div>

A typical LLM serving deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 4                       # 4 GPU workers
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
      - name: vllm
        image: vllm-ai/vllm:0.4.2
        args:
        - "--model"
        - "/models/llama-3-8b"
        - "--tensor-parallel-size"
        - "1"
        resources:
          requests:
            nvidia.com/gpu: "1"
            memory: "40Gi"
          limits:
            nvidia.com/gpu: "1"
            memory: "40Gi"
        volumeMounts:
        - name: model-weights
          mountPath: /models
          readOnly: true
        ports:
        - containerPort: 8000
      volumes:
      - name: model-weights
        persistentVolumeClaim:
          claimName: model-weights-pvc
```

### Control Plane vs Data Plane
{: #k8s-control-plane}

Kubernetes has a strict separation:

```
┌─────────────────────────────────────────────────────────────┐
│                    Control Plane (masters)                   │
│                                                             │
│  API Server ──► etcd (distributed key-value store)          │
│      │                                                      │
│  Scheduler ── decides which node runs each pod              │
│  Controller Manager ── reconciles desired vs actual state   │
└─────────────────────────────────────────────────────────────┘
         │ watches for new pods │ updates node status
         ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                     Data Plane (workers)                     │
│                                                             │
│  Node-1                Node-2               Node-3          │
│  ┌───────────┐         ┌───────────┐        ┌───────────┐  │
│  │ kubelet   │         │ kubelet   │        │ kubelet   │  │
│  │ kube-proxy│         │ kube-proxy│        │ kube-proxy│  │
│  │ Pod-A     │         │ Pod-B     │        │ Pod-C     │  │
│  │ Pod-D     │         │ Pod-E     │        │ Pod-F     │  │
│  └───────────┘         └───────────┘        └───────────┘  │
└─────────────────────────────────────────────────────────────┘
```

The **kubelet** on each node watches the API server for pods assigned to that node and starts/stops containers accordingly. The **kube-proxy** programs iptables or eBPF rules to implement Service virtual IPs. The **control plane never touches user traffic** — it only manages configuration.

> **Intuition question:** Why is the control plane a separate fleet of machines rather than running on the same nodes as workloads?
>
> *If the control plane runs on the same nodes as workloads, a noisy workload can starve the scheduler, and a node crash takes down both the API server and the workloads. Separating them means you can OOM-kill workload pods without affecting the control plane's ability to reschedule them. Cloud providers (GKE, EKS, AKS) fully manage the control plane — you only provision the worker nodes.*

### Scheduling and GPU Resources
{: #scheduling}

The Kubernetes scheduler assigns pods to nodes. It runs two phases:

1. **Filter**: eliminate nodes that cannot satisfy the pod's constraints (not enough CPU/memory/GPU, wrong labels, taints that the pod doesn't tolerate).
2. **Score**: rank the remaining nodes (balanced resource usage, pod affinity, topology spread).

For GPU workloads, key scheduling features:

| Feature | What it does | LLM use case |
|---|---|---|
| **Resource requests/limits** | Reserve `nvidia.com/gpu: N` slots | Prevent over-subscription of GPU nodes |
| **Node selectors / affinity** | Pin pods to nodes with specific GPU models | Route large models to A100 nodes |
| **Taints and tolerations** | Nodes repel pods unless pods tolerate the taint | Dedicate GPU nodes to inference only |
| **Topology spread constraints** | Spread pods across failure domains | Distribute replicas across AZs |
| **Pod affinity** | Co-locate pods that communicate heavily | Co-locate prefill/decode pods on same host |

```yaml
# Ensure replicas spread across availability zones
topologySpreadConstraints:
- maxSkew: 1
  topologyKey: topology.kubernetes.io/zone
  whenUnsatisfiable: DoNotSchedule
  labelSelector:
    matchLabels:
      app: vllm-server
```

### Networking: Services and DNS
{: #k8s-networking}

Every pod gets an IP address from the cluster's pod CIDR. Pods can communicate directly by IP — but pod IPs are ephemeral (pod restarts = new IP). **Services** provide stable endpoints:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-server
spec:
  selector:
    app: vllm-server           # matches pods with this label
  ports:
  - port: 80
    targetPort: 8000           # pod's container port
  type: ClusterIP              # internal only
```

The Service gets a stable ClusterIP and a DNS name: `vllm-server.default.svc.cluster.local`. Any pod in the cluster can reach it at `http://vllm-server` — Kubernetes DNS resolves it to the ClusterIP, and kube-proxy load-balances across the backing pods.

Service types:
- **ClusterIP**: internal only. Standard for inter-service communication.
- **NodePort**: exposes a port on every node's external IP. Rarely used directly in production.
- **LoadBalancer**: provisions a cloud load balancer (GCP LB, AWS NLB). Used to expose services externally.
- **Headless** (`clusterIP: None`): returns pod IPs directly from DNS — used when clients want to connect to specific pods (e.g., StatefulSet members).

### Storage: ConfigMaps, Secrets, PVCs
{: #k8s-storage}

| Object | What it stores | How pods use it |
|---|---|---|
| **ConfigMap** | Non-sensitive config (model name, batch sizes) | Mounted as file or injected as env vars |
| **Secret** | Sensitive data (API keys, HuggingFace token) | Same as ConfigMap, but encrypted at rest |
| **PersistentVolumeClaim (PVC)** | Durable storage (model weights, datasets) | Mounted as a filesystem volume |

Model weights pattern:

```yaml
# PVC backed by cloud NFS / object storage CSI driver
kind: PersistentVolumeClaim
metadata:
  name: model-weights-pvc
spec:
  storageClassName: gcs-csi           # GCS bucket via CSI driver
  accessModes: [ReadOnlyMany]         # multiple pods read same weights
  resources:
    requests:
      storage: 300Gi
```

`ReadOnlyMany` lets multiple vLLM pods on different nodes mount the same weight volume simultaneously — essential for horizontal scaling without copying weights to each node.

### Health Checks and Readiness Gates
{: #health-checks}

Kubernetes probes control when traffic is sent to a pod and when a pod is restarted:

```yaml
livenessProbe:                    # kill and restart if this fails
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30         # wait for model to load
  periodSeconds: 10

readinessProbe:                   # remove from Service endpoints if this fails
  httpGet:
    path: /ready                  # /ready returns 200 only after model is loaded
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 5

startupProbe:                     # gives more time for initial startup
  httpGet:
    path: /health
    port: 8000
  failureThreshold: 60            # 60 × 10s = 10 minutes to load a large model
  periodSeconds: 10
```

The liveness/readiness distinction is critical for LLM serving: a vLLM pod loading a 70B model takes several minutes. During that time, the pod is alive (kubelet shouldn't restart it) but not ready (the Service shouldn't send requests to it). The `startupProbe` covers the long startup window.

> **Interview question:** Your LLM server pod passes its liveness probe but starts returning 503s under load. What Kubernetes mechanism can take it out of rotation, and how do you implement it?
>
> *The readiness probe. If the `/ready` endpoint returns a non-2xx response, Kubernetes removes the pod from the Service's endpoints list — no new requests are routed to it. For LLM servers, you can implement a readiness check that measures queue depth: if the number of waiting requests exceeds a threshold (the server is saturated), return 503. This creates natural backpressure: saturated pods are temporarily removed from rotation, forcing the load balancer to route to less-loaded replicas. Combine with exponential backoff on retries at the proxy layer to prevent thundering herd.*

### Autoscaling
{: #autoscaling}

Kubernetes has three autoscaling dimensions:

<div class="post-flow" role="group" aria-label="Three autoscaling axes">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>HPA (Horizontal Pod Autoscaler)</strong> — adds/removes pods based on metrics (CPU, custom metrics like queue depth). Works by adjusting Deployment replicas.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>VPA (Vertical Pod Autoscaler)</strong> — adjusts CPU/memory requests of existing pods. Requires a restart to apply. Rarely used for GPU workloads.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Cluster Autoscaler / Karpenter</strong> — adds/removes nodes when pods can't be scheduled. Karpenter (AWS) is faster and more flexible: provisions the right instance type for each workload.</span></li>
  </ol>
</div>

LLM-specific HPA example using custom metrics:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: vllm_pending_requests      # custom metric from Prometheus
      target:
        type: AverageValue
        averageValue: "5"               # scale up when >5 pending requests per pod
```

GPU nodes are expensive and slow to provision (5–10 minutes). For LLM serving with variable traffic, you typically keep a minimum of warm replicas and scale up proactively based on queue depth rather than reactively based on CPU.

---

## Envoy: The Universal Proxy
{: #envoy}

### What Is Envoy?
{: #what-is-envoy}

Envoy is a high-performance L4/L7 proxy written in C++, originally built at Lyft. It is the data plane for almost every service mesh and API gateway in the cloud-native ecosystem (Istio, Contour, AWS App Mesh, Google Cloud Traffic Director). Unlike Nginx (configured with static files), Envoy is designed for **dynamic configuration** — its entire config can be updated at runtime via gRPC APIs without restarting.

Key design principles:
1. **Out-of-process architecture**: the proxy runs as a sidecar next to your app. The app doesn't need network code.
2. **L7-aware**: understands HTTP/1.1, HTTP/2, gRPC, WebSocket, and protocols above. Can route based on headers, paths, and request bodies.
3. **Observability first**: every request generates metrics, access logs, and distributed traces automatically.
4. **Dynamic config (xDS)**: the control plane pushes config changes over gRPC without a restart.

### Envoy Architecture
{: #envoy-architecture}

```
┌─────────────────────────────────────────────────────────┐
│                        Envoy                            │
│                                                         │
│  Downstream                                 Upstream    │
│  (clients)                                 (backends)   │
│     │                                           │       │
│     ▼                                           ▼       │
│  Listeners ──► Filter Chain ──► Router ──► Clusters    │
│     │                 │                        │        │
│     │         (HTTP filters:                   │        │
│     │          auth, rate-limit,               │        │
│     │          header manipulation,            │        │
│     │          WAF, RBAC...)                   │        │
│                                                         │
│  Stats sink (Prometheus / StatsD)                       │
│  Access log (stdout / gRPC sink)                        │
│  Trace exporter (Zipkin / OTLP)                         │
└─────────────────────────────────────────────────────────┘
```

**Listeners** bind to a port and accept connections. **Filter chains** process each connection/request through a series of filters. The **router** matches requests to **clusters** (groups of upstream endpoints). Each cluster has a **load balancer** that picks an endpoint.

### xDS: Dynamic Configuration
{: #xds}

xDS ("x Discovery Service") is the protocol by which a control plane pushes config to Envoy. It is a set of gRPC APIs, each responsible for a resource type:

| xDS API | Resources managed | Example |
|---|---|---|
| **LDS** (Listener DS) | Listeners + filter chains | "Add a new listener on port 8080" |
| **RDS** (Route DS) | Route tables | "Route /v1/models/llama3 to cluster-llama3" |
| **CDS** (Cluster DS) | Upstream cluster definitions | "Cluster llama3 uses round-robin LB" |
| **EDS** (Endpoint DS) | Endpoints within clusters | "Add 10.0.0.5:8000 to cluster llama3" |
| **SDS** (Secret DS) | TLS certificates | "Rotate the certificate for api.example.com" |

When a new vLLM pod starts and passes its readiness check, the control plane pushes an EDS update: "add this pod's IP to the cluster." Envoy starts routing traffic to it within seconds, with zero restarts. This is the mechanism that makes Kubernetes Services work with Envoy-based proxies.

> **Intuition question:** Kubernetes already has Services that update endpoints automatically. Why do I need Envoy and xDS on top?
>
> *Kubernetes Services use kube-proxy + iptables/eBPF for load balancing — simple, fast, but dumb. They support only round-robin, have no circuit breaking, no retry logic, no header-based routing, no rate limiting, and no observability. Envoy gives you L7 routing (route /v1/completions to cluster-A and /v1/embeddings to cluster-B), circuit breakers (stop sending to a backend that is returning 500s), retries with backoff, per-route timeouts, request mirroring for canaries, JWT validation, and full telemetry. For LLM serving, you need all of this.*

### Filter Chains and Extension Points
{: #envoy-filters}

Envoy filters are composable middleware for the request pipeline:

```yaml
http_filters:
- name: envoy.filters.http.jwt_authn          # validate JWT tokens
  typed_config:
    "@type": type.googleapis.com/envoy.extensions.filters.http.jwt_authn.v3.JwtAuthentication

- name: envoy.filters.http.local_ratelimit    # rate limit per client
  typed_config:
    "@type": type.googleapis.com/envoy.extensions.filters.http.local_rate_limit.v3.LocalRateLimit

- name: envoy.filters.http.ext_authz          # call external auth service
  typed_config:
    "@type": type.googleapis.com/envoy.extensions.filters.http.ext_authz.v3.ExtAuthz

- name: envoy.filters.http.router             # must be last: routes to upstream
  typed_config:
    "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
```

For LLM gateways, relevant filters:
- **JWT auth**: validate API keys / user tokens before forwarding to the GPU cluster
- **Rate limiting**: per-user token quotas (charge by tokens consumed, not requests)
- **Ext proc (External Processing)**: call an external gRPC service to transform the request — used to inject system prompts, log inputs, or run guardrails

### Observability Built In
{: #envoy-observability}

Envoy emits three observability signals automatically:

1. **Metrics**: per-cluster and per-route stats in StatsD/Prometheus format. `envoy_cluster_upstream_rq_total`, `envoy_cluster_upstream_rq_time` (latency histogram), `envoy_cluster_upstream_cx_active` (active connections).

2. **Access logs**: structured JSON log per request with timing breakdown, response code, upstream host, and custom fields.

3. **Distributed traces**: injects/propagates B3 or W3C TraceContext headers, exports spans to Zipkin/Jaeger/OTLP. Every hop in the request path contributes a span.

---

## Kubernetes Gateway API
{: #gateway-api}

### Ingress vs Gateway API
{: #ingress-vs-gateway}

The original Kubernetes object for north-south traffic (external clients → cluster services) was **Ingress**. It works but has significant limitations:

<div class="post-flow post-flow--compare" role="group" aria-label="Ingress vs Gateway API">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Ingress (old)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Single resource type — everything in one spec</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Vendor features via annotations (not portable)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">No role separation: infra + app config mixed</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">No traffic splitting, no header matching in spec</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Gateway API (new) ✓</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Role-oriented: GatewayClass → Gateway → Route</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Portable across implementations (Envoy Gateway, Cilium, Istio)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Traffic splitting, header matching, URL rewriting in spec</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Supports TCP/UDP/gRPC routes, not just HTTP</span></li>
    </ol>
  </div>
</div>

### Gateway API Resources
{: #gateway-resources}

The Gateway API separates concerns across three personas:

```
┌────────────────────────────────────────────────────────────┐
│  Infra team                                                │
│  GatewayClass: "use envoy-gateway implementation"          │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│  Platform team                                             │
│  Gateway: "provision a gateway on port 443 with TLS"       │
│  (creates cloud load balancer + Envoy fleet)               │
└──────────────────────────┬─────────────────────────────────┘
                           │
┌──────────────────────────▼─────────────────────────────────┐
│  App team (per namespace)                                  │
│  HTTPRoute: "route /v1/completions → vllm-service:80"      │
│  HTTPRoute: "route /v1/embeddings → embedding-service:80"  │
└────────────────────────────────────────────────────────────┘
```

Example HTTPRoute for an LLM API:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-api-routes
spec:
  parentRefs:
  - name: prod-gateway                    # attach to this Gateway
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /v1/completions
    - headers:
      - name: X-Model
        value: llama3-8b
    backendRefs:
    - name: vllm-llama3-8b
      port: 80
      weight: 90                          # 90% to stable
    - name: vllm-llama3-8b-canary
      port: 80
      weight: 10                          # 10% to canary
```

### Traffic Splitting and Canaries
{: #traffic-splitting}

The `weight` field on `backendRefs` enables weighted traffic splitting at the Gateway layer. This is the mechanism for:

- **Canary deployments**: 5% of traffic to new model version, 95% to old. Monitor error rates and latency. Shift weight when healthy.
- **A/B testing**: split by header value (`X-Experiment-Group`) to test model variants.
- **Shadow mirroring**: send a copy of real traffic to a new deployment for testing without affecting users.

```yaml
# HTTPRoute with request mirroring for shadow testing
filters:
- type: RequestMirror
  requestMirror:
    backendRef:
      name: vllm-llama3-shadow
      port: 80
```

### LLM-Specific Gateway Concerns
{: #llm-gateway}

Standard web API gateways assume short requests. LLM streaming responses break several assumptions:

| Challenge | Root cause | Gateway solution |
|---|---|---|
| **Timeout misconfiguration** | Default HTTP timeout is 30–60s; LLM responses can take minutes | Set per-route timeouts to 300s+; use streaming |
| **Token-based rate limiting** | HTTP rate limiters count requests; LLM cost is in output tokens | Custom ext_proc filter counts tokens from streaming chunks |
| **Sticky routing for KV cache** | Repeated requests for same prefix should hit the same backend | Consistent hash by prefix hash in HTTPRoute |
| **SSE / chunked encoding** | Some proxies buffer streaming responses | Disable response buffering on the route |
| **Model-header-based routing** | Route by `X-Model` header to different backends | HTTPRoute header match |

---

## Service Mesh: East-West Traffic
{: #service-mesh}

The Gateway API handles **north-south** traffic (external → cluster). A **service mesh** handles **east-west** traffic (service → service inside the cluster).

### The Sidecar Model
{: #sidecar-model}

A service mesh injects an Envoy sidecar into every pod. The sidecar intercepts all inbound and outbound traffic transparently (via iptables rules that redirect traffic to the sidecar's port). The application needs no code changes.

```
┌──────────────────────────────────┐     ┌──────────────────────────────────┐
│  Pod A                           │     │  Pod B                           │
│                                  │     │                                  │
│  ┌─────────┐  ┌──────────────┐   │     │  ┌──────────────┐  ┌─────────┐  │
│  │ App     │◄─│ Envoy sidecar│◄──┼─────┼─►│ Envoy sidecar│─►│ App     │  │
│  │         │  │ (inbound)    │   │mTLS │  │ (inbound)    │  │         │  │
│  └─────────┘  └──────────────┘   │     │  └──────────────┘  └─────────┘  │
│               ┌──────────────┐   │     │                                  │
│               │ Envoy sidecar│───┼─────┘                                  │
│               │ (outbound)   │   │                                        │
│               └──────────────┘   │                                        │
└──────────────────────────────────┘
```

The control plane (Istiod for Istio) pushes xDS config to all sidecars: routes, certificates, load balancing policy, and retry config. A pod gets the right route to any service without any DNS or hardcoded addresses.

**Ambient mesh** (Istio's newer model, also used by Cilium): eliminates sidecars by moving traffic interception to a per-node proxy ("ztunnel") and optional per-service L7 "waypoint" proxy. Lower memory overhead — relevant for GPU pods where every sidecar wastes container slots.

### mTLS and Zero-Trust
{: #mtls}

The mesh enforces **mutual TLS (mTLS)** on all east-west traffic automatically:

1. Each pod gets a short-lived X.509 certificate from the control plane, tied to its Kubernetes Service Account.
2. When Pod A connects to Pod B, both present certificates. Both verify each other.
3. If authentication fails, the connection is rejected.
4. **No code changes** required in the application — the sidecar handles TLS termination.

For LLM infrastructure, mTLS means:
- The vLLM engine pods can only be reached by authorized services (the router/gateway) — not by arbitrary pods.
- Worker-to-worker communication (tensor parallel shards, prefill → decode transfers) is encrypted in transit.
- Identity is based on SPIFFE/X.509, not network topology — works across clusters and clouds.

### Mesh Observability
{: #mesh-observability}

With sidecars on every pod, the mesh knows the full topology of every request. The Prometheus metrics and traces that Envoy emits automatically give you:

- **Golden signals per service pair**: latency, error rate, request volume for every (source, destination) pair.
- **Full distributed trace**: request → gateway → router → prefill worker → decode worker, with timing at each hop.
- **Topology visualization**: real-time service dependency graph with health overlays (Kiali for Istio).

---

## LLM Serving Stack
{: #llm-serving-stack}

### Why LLMs Are Different From Web Servers
{: #llm-vs-web}

Everything so far applies to any web service. LLMs have properties that demand specialized infrastructure:

| Property | Web server assumption | LLM reality | Infrastructure implication |
|---|---|---|---|
| Request duration | Milliseconds | Seconds to minutes | Long timeouts, streaming required |
| Resource usage | CPU + memory | GPU (scarce, expensive) | GPU scheduling, packing, disaggregation |
| State per request | Stateless | KV cache (GB per request) | Memory management, cache-aware routing |
| Cost model | Uniform per request | Proportional to output tokens | Token-aware rate limiting and billing |
| Batching | Not needed at L7 | Critical for GPU utilization | Continuous batching in the engine |
| Failure mode | Crash → restart | OOM → stall | KV cache eviction, preemption |

### The Four Layers
{: #serving-layers}

A production LLM serving system has four distinct layers:

<div class="post-flow" role="group" aria-label="LLM serving layers">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Gateway layer</strong> — auth, rate limiting, SSL termination, model routing by header, token counting for billing. Runs on CPU. Implemented with Envoy + Gateway API.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Router / scheduler layer</strong> — picks which engine instance handles each request. KV cache-aware routing, load balancing, queue management. Runs on CPU. New: llm-d inference scheduler, vLLM router.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Engine layer</strong> — the actual model inference. Manages KV cache, continuous batching, tensor parallelism. Runs on GPU. vLLM, TGI, TensorRT-LLM.</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue"><strong>Storage layer</strong> — model weights (PVC / object store), KV cache spill (NVMe / disaggregated KV store), prefix cache (in-memory or Redis).</span></li>
  </ol>
</div>

```
User request
      │
      ▼
┌─────────────────────┐
│   Gateway (Envoy)   │  Auth, rate limit, TLS
│   HTTPRoute         │  Route by X-Model header
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Inference Router   │  Prefix-hash routing
│  (llm-d / vLLM)    │  Least-loaded selection
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌────────┐
│ vLLM-0 │   │ vLLM-1 │   GPU workers
│  GPU   │   │  GPU   │   KV cache, batching
└────────┘   └────────┘
    │             │
    ▼             ▼
Model weights  KV cache
(NFS PVC)     (HBM + NVMe)
```

### Engine Layer: vLLM, TGI, TensorRT-LLM
{: #engine-layer}

| Engine | Key strength | Best for |
|---|---|---|
| **vLLM** | PagedAttention KV cache, continuous batching, wide model support | General-purpose, research, multi-model |
| **TGI** (Text Generation Inference) | HuggingFace ecosystem integration, simple deployment | HuggingFace models, quick start |
| **TensorRT-LLM** | NVIDIA-optimized kernels, maximum throughput on A100/H100 | Throughput-maximized production on NVIDIA |
| **SGLang** | RadixAttention (tree-structured prefix cache), fast for shared prefixes | Long-system-prompt workloads, structured output |
| **MLC-LLM** | Mobile/edge deployment, diverse hardware | Edge inference |

Each engine exposes an OpenAI-compatible REST API (`/v1/completions`, `/v1/chat/completions`). This means the gateway layer is engine-agnostic — you can route some requests to vLLM and others to TensorRT-LLM based on model or user tier.

### Request Routing for LLMs
{: #request-routing}

Naive round-robin is wrong for LLMs. The right routing strategy depends on what you're optimizing:

<div class="post-flow post-flow--compare" role="group" aria-label="LLM routing strategies">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Minimize TTFT (time-to-first-token)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Route to backend with shortest queue</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Metric: pending_requests per backend</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Ignores KV cache — pays recompute cost</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Maximize KV cache hits</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Hash the prompt prefix, route to same backend</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Reuses existing KV cache — saves prefill cost</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Can overload one backend if many requests share a prefix</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Balance: P2C with cache score</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Sample 2 backends, score each by: load − cache_bonus</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Route to better score</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Used by llm-d and production vLLM deployments</span></li>
    </ol>
  </div>
</div>

### Prefill-Decode Disaggregation
{: #pd-disaggregation}

The **prefill** phase (processing the input prompt) and **decode** phase (generating output tokens) have radically different hardware profiles:

| Phase | Compute profile | GPU utilization | Optimal hardware |
|---|---|---|---|
| **Prefill** | Compute-bound (large batch of input tokens) | High MFU | Compute-dense (H100) |
| **Decode** | Memory-bandwidth-bound (stream weights per token) | Low MFU | High-bandwidth (HBM3) |

Running both phases on the same GPU means the compute-hungry prefill steals GPU time from the latency-sensitive decode, causing **TTFT jitter** (spiky time-to-first-token). Disaggregation splits them:

```
Request
   │
   ▼
┌──────────────────────────────────────────┐
│            Inference Scheduler           │
└──────────────────────────────────────────┘
          │                      │
          ▼                      ▼
  ┌───────────────┐      ┌───────────────┐
  │ Prefill Pool  │      │ Decode Pool   │
  │ (2× H100)     │      │ (8× A100)     │
  │               │      │               │
  │ Process input │      │ Generate      │
  │ → KV cache    │─────►│ tokens using  │
  │               │      │ KV cache      │
  └───────────────┘      └───────────────┘
```

The scheduler routes each new request to a prefill worker. After prefill completes, the KV cache is transferred to a decode worker over RDMA (RoCE or InfiniBand) or NVLink. The decode worker generates tokens and streams them back to the client. This transfer adds a few milliseconds of latency but enables better GPU utilization and tail latency isolation.

> **Interview question:** You're running vLLM on 8 A100s. Users complain about spiky TTFT — sometimes fast, sometimes 5–10 seconds. Decoding speed is fine. What is the likely cause and how do you fix it?
>
> *The likely cause is prefill contention: a long prompt (from another user) fills the GPU for several seconds, blocking decode for all other requests on that GPU. This is the "prefill stalls decode" problem. Fix with disaggregation: dedicate some GPUs to prefill and others to decode. Requests go to a prefill GPU, the KV cache is transferred (via shared memory if co-located, or RDMA if cross-node), and decoding proceeds on a dedicated decode GPU. Alternatively, **chunked prefill** (split the long prompt into chunks, interleave prefill chunks with decode steps) can reduce the worst-case stall without requiring separate hardware.*

### KV Cache-Aware Routing
{: #kv-cache-routing}

**Prefix caching** saves enormous compute: if two requests share the same system prompt (e.g., a 4096-token system prompt for a customer support bot), the KV cache for that prefix only needs to be computed once per GPU. Subsequent requests that share the prefix reuse the cached KV.

For prefix caching to work, the router must send requests with the same prefix to the same backend. The standard approach:

1. Hash the prefix (first N tokens of the prompt, or the system prompt).
2. Consistent-hash to a backend.
3. If that backend is overloaded (queue depth > threshold), fall back to least-loaded.

SGLang's **RadixAttention** extends this: it builds a radix tree of all cached KV prefixes and routes requests to the backend that has the longest cached prefix match. This maximizes cache hits even for partially-shared prefixes.

### LLM Autoscaling
{: #llm-autoscaling}

Standard CPU-based HPA metrics (CPU%, memory%) are poor signals for LLM scaling:
- GPU utilization can be low (batch=1 decode) while the queue is growing.
- Memory is pinned to the KV cache, not a proxy for load.

Better LLM autoscaling signals:

| Metric | Why it works | Threshold guidance |
|---|---|---|
| `vllm_num_requests_waiting` | Direct measure of unserved demand | Scale up if >N per pod for >60s |
| `vllm_gpu_cache_usage_perc` | KV cache fills up → new requests can't start | Scale up if >80% sustained |
| Request queue latency (P95) | SLO-direct: scale if P95 queue time exceeds budget | Depends on SLO |
| Tokens per second per pod | Throughput headroom | Scale if utilization >70% of max tested throughput |

The challenge: GPU nodes take 5–15 minutes to provision. Reactive scaling always lags. Solutions:
- **Predictive scaling**: model traffic patterns (daily cycles), pre-scale before peak.
- **Minimum warm pool**: keep N replicas always warm even at zero traffic.
- **Scale-up-fast, scale-down-slow**: aggressive scale-up triggers, long cooldown before scale-down (avoid churn).
- **Spot/preemptible nodes**: use cheaper interruptible GPUs for burst capacity with graceful drain on preemption.

---

## llm-d: Distributed LLM Serving
{: #llm-d}

**llm-d** is a Kubernetes-native, open-source framework for distributed LLM inference, built around the concept of disaggregated prefill-decode inference with smart routing.

### Architecture Overview
{: #llm-d-arch}

```
                     External Traffic
                           │
                           ▼
              ┌────────────────────────┐
              │   Envoy Gateway        │
              │   (Gateway API)        │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Inference Gateway     │  ← llm-d component
              │  (Envoy + ext_proc)    │
              │  • Model routing       │
              │  • Auth / rate limit   │
              │  • Token counting      │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Disaggregated         │  ← llm-d component
              │  Inference Scheduler   │
              │  • KV cache-aware LB   │
              │  • Prefill/decode split│
              │  • Queue management    │
              └───────────┬────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │ Prefill Pod │ │ Decode Pod  │ │ Decode Pod  │
   │ (vLLM)     │ │ (vLLM)     │ │ (vLLM)     │
   └─────────────┘ └─────────────┘ └─────────────┘
```

llm-d builds on Kubernetes and the Gateway API — it is not a replacement for them but a set of custom resources and controllers that extend them for LLM-specific concerns.

### Disaggregated Inference Scheduler
{: #llm-d-scheduler}

The llm-d scheduler is a Kubernetes controller that:
1. Watches inference request queues.
2. Maintains a real-time view of each worker's KV cache state (what prefixes are cached, how full the KV cache is, queue depth).
3. Routes new requests using a **KV-cache-aware scoring function**: score = α × cache_hit_bonus − β × queue_depth.
4. Splits requests across prefill and decode pools, managing the KV transfer.

It exposes metrics via Prometheus and integrates with HPA for autoscaling based on queue depth and cache pressure.

### Gateway and Routing
{: #llm-d-gateway}

llm-d uses the Kubernetes Gateway API (via Envoy Gateway or a compatible implementation) for north-south traffic. It defines a custom **InferencePool** CRD — a group of LLM worker pods — and an **InferenceModel** CRD that maps a model name to an InferencePool. The Gateway's HTTPRoute sends requests to the InferencePool, and the scheduler handles the rest.

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferencePool
metadata:
  name: llama3-pool
spec:
  targetPortNumber: 8000
  selector:
    matchLabels:
      app: vllm-llama3
  extensionRef:
    name: llm-d-scheduler          # pluggable scheduler
---
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModel
metadata:
  name: llama3-8b
spec:
  modelName: llama-3-8b-instruct
  criticality: Critical
  poolRef:
    name: llama3-pool
```

This is the **Gateway API inference extension** (GIE), an emerging standard for expressing LLM serving topology in Kubernetes-native terms. llm-d is one of the first implementations.

---

## Observability for LLM Serving
{: #observability}

### Key Metrics
{: #metrics}

LLM serving has a different metric taxonomy than standard web services:

| Metric | What it measures | SLO typical value |
|---|---|---|
| **TTFT** (Time-To-First-Token) | Prefill latency — how long until first token arrives | P50 < 500ms, P99 < 2s |
| **ITL** (Inter-Token Latency) | Decode speed — time between consecutive tokens | P50 < 50ms/token |
| **E2E latency** | Total time from request to final token | Depends on output length |
| **Throughput** | Tokens/sec per GPU | Depends on model and hardware |
| **KV cache utilization** | % of GPU memory used for KV cache | Alert at >80% |
| **Queue depth** | Requests waiting for a free KV cache slot | Alert at >N |
| **Request success rate** | 1 − (5xx / total requests) | >99.9% |
| **Preemption rate** | Requests evicted from KV cache and retried | Should be near zero |

```
# Prometheus scrape from vLLM /metrics endpoint
vllm:e2e_request_latency_seconds_bucket{model_name="llama-3-8b"}
vllm:time_to_first_token_seconds_bucket{model_name="llama-3-8b"}
vllm:time_per_output_token_seconds_bucket{model_name="llama-3-8b"}
vllm:num_requests_waiting{model_name="llama-3-8b"}
vllm:gpu_cache_usage_perc{model_name="llama-3-8b"}
```

### Distributed Tracing
{: #tracing}

A distributed trace for an LLM request:

```
Trace: req-abc123 (total: 2.3s)
├── Gateway (Envoy): 5ms
│   ├── JWT auth filter: 2ms
│   └── Rate limit check: 3ms
├── Inference Router: 8ms
│   ├── KV cache score computation: 4ms
│   └── Backend selection: 4ms
├── Prefill (vLLM worker-2): 840ms
│   ├── Tokenize: 2ms
│   ├── KV cache lookup: 10ms
│   └── Prefill forward pass: 828ms
├── KV transfer (worker-2 → worker-5): 45ms
└── Decode (vLLM worker-5): 1.4s
    ├── token-1: 48ms
    ├── token-2: 46ms
    ...
    └── token-30: 47ms
```

OpenTelemetry is the standard instrumentation layer. vLLM exports OTLP traces; Envoy propagates trace context headers; Jaeger or Tempo stores and visualizes them.

### SLOs for LLM APIs
{: #slos}

Service Level Objectives for LLM APIs must account for the variable-length nature of generation:

| User experience goal | SLO metric | Typical target |
|---|---|---|
| "Feels responsive" | P95 TTFT | < 1 second |
| "Smooth streaming" | P99 ITL | < 100ms/token |
| "Doesn't hang" | P99.9 E2E latency (for fixed output length) | < 60 seconds |
| "Reliable" | Error rate | < 0.1% |
| "Available" | Uptime | > 99.9% |

> **Interview question:** Your P95 TTFT is 3 seconds but P50 is 300ms. What does this distribution tell you, and what are the likely causes?
>
> *A fat tail (P95 >> P50) for TTFT means most requests are fast but a few are very slow. For TTFT, which is dominated by prefill latency, the likely causes are: (1) Long input prompts — a 10K token prompt takes 10× longer to prefill than a 1K prompt. Segment TTFT by prompt length to confirm. (2) Prefill-decode contention — long prefills block other requests' decoding, and when the GPU becomes available, there is a burst of queued prefills that create a spike. Fix with chunked prefill or disaggregation. (3) Cold start — pods are scaling up under load; new pods take time to load weights. Pre-warm replicas. (4) KV cache OOM — when KV cache fills, new requests queue. Fix with better cache eviction or scale out.*

---

## Production Readiness Checklist
{: #production-checklist}

Before serving real users:

**Containers and Images**
- [ ] CUDA base image version pinned — not `latest`
- [ ] Model weights on a PVC, not in the image
- [ ] Image built without secrets in layers (use multi-stage builds)
- [ ] Container runs as non-root user

**Kubernetes**
- [ ] Resource requests and limits set for all containers
- [ ] GPU resources declared as extended resources (`nvidia.com/gpu`)
- [ ] `startupProbe` configured with enough time for model loading
- [ ] `readinessProbe` reflects actual serving readiness (not just process alive)
- [ ] `topologySpreadConstraints` distributes replicas across AZs
- [ ] `PodDisruptionBudget` ensures minimum replicas during node maintenance
- [ ] Secrets stored in Kubernetes Secrets (not ConfigMaps or env vars)

**Networking and Gateway**
- [ ] TLS terminated at the Gateway with auto-rotating certificates
- [ ] JWT / API key validation in the filter chain
- [ ] Per-user rate limits configured (request and token-based)
- [ ] Route timeouts set to 300s+ for completion endpoints
- [ ] SSE / streaming response buffering disabled on routes
- [ ] mTLS enforced for all east-west traffic in the mesh

**LLM Serving**
- [ ] Continuous batching enabled in the engine
- [ ] KV cache size tuned for target batch size and context length
- [ ] Prefix caching enabled with consistent-hash routing at the load balancer
- [ ] Preemption policy configured (recompute vs. swap vs. abort)
- [ ] Model weights checksummed at startup to detect corruption

**Observability**
- [ ] TTFT, ITL, and queue depth metrics exported to Prometheus
- [ ] Alerting on P95 TTFT > SLO threshold
- [ ] Alerting on KV cache utilization > 80%
- [ ] Distributed tracing enabled end-to-end
- [ ] GPU utilization dashboards in Grafana
- [ ] HPA configured on queue depth metric

**Reliability**
- [ ] Graceful shutdown: drain in-flight requests before pod terminates
- [ ] Rolling update strategy with `maxSurge` and `maxUnavailable` configured
- [ ] Canary deploy process tested with HTTPRoute weight splitting
- [ ] Runbook for: pod OOM, GPU driver failure, model loading failure, cache pressure
- [ ] Load test at 2× expected peak to find failure modes before they affect users

> **Intuition question:** I've covered all the reliability checklist items. Why do I still need load testing?
>
> *Checklists verify configuration correctness, not system behavior under real load. Load testing reveals: how throughput and latency degrade under concurrency (often non-linear — the system falls off a cliff at a specific concurrency level due to KV cache OOM or CPU scheduler contention); whether autoscaling keeps up fast enough; which component saturates first (the router CPU? the KV cache? the network to the storage backend?); and how failures cascade — does one overloaded pod cause a thundering herd on others? You cannot reason about these from the checklist alone.*
