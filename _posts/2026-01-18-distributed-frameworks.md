---
layout: post
title: "Distributed Computing Frameworks"
date: 2026-01-18
categories: [engineering, distributed-systems]
tags: [ray, dask, horovod, pytorch-distributed, celery, apache-beam, distributed-computing]
description: "Ray, Dask, Horovod, PyTorch Distributed, Celery, Apache Beam — practical guide to distributed computing frameworks with real code"
math: true
---

Distributed computing means different things in different contexts: ML training across GPUs, parallel pandas on a laptop cluster, task queues for web backends, or unified batch+stream pipelines. Picking the wrong tool is expensive. This guide covers the industry-relevant frameworks with enough depth to use them.

---

## Table of Contents
- [Why Distributed?](#why-distributed)
- [Ray — Distributed Python for ML](#ray)
  - [Ray Core: Tasks and Actors](#ray-core)
  - [Ray Data](#ray-data)
  - [Ray Train](#ray-train)
  - [Ray Serve](#ray-serve)
  - [Ray Tune](#ray-tune)
- [Dask — Parallel Pandas/NumPy](#dask)
  - [Dask DataFrame](#dask-dataframe)
  - [Dask Array](#dask-array)
  - [Dask Delayed](#dask-delayed)
  - [Dask ML](#dask-ml)
- [PyTorch Distributed](#pytorch-distributed)
  - [DDP — Data Parallel](#ddp)
  - [FSDP — Fully Sharded](#fsdp)
  - [torch.distributed Primitives](#torch-distributed-primitives)
- [Horovod](#horovod)
- [Celery — Distributed Task Queues](#celery)
- [Apache Beam — Unified Batch + Stream](#apache-beam)
- [Framework Comparison](#framework-comparison)
- [Choosing the Right Tool](#choosing)

---

## Why Distributed? {#why-distributed}

Three problems force you into distributed computing:

**Data doesn't fit in RAM.** A 500 GB dataset can't be loaded into a single machine. You need to partition it across nodes and process chunks in parallel.

**Compute takes too long.** Training a large model on one GPU takes weeks. Distributing across 64 GPUs cuts it to hours.

**Throughput requirements.** A web service needs to handle 10k requests/second. One Python process maxes out at ~1k.

The frameworks in this guide solve different subsets of these problems. The mental model before going deep:

```
Data-parallel batch processing:  Spark (see big-data post), Dask, Beam
ML distributed training:         PyTorch Distributed, Horovod, Ray Train
Flexible task/actor model:       Ray Core
Task queues (web backend):       Celery
ML serving + HPO:                Ray Serve, Ray Tune
Unified batch+stream:            Apache Beam
```

---

## Ray — Distributed Python for ML {#ray}

Ray is the most ML-native distributed framework. It started as a research project at UC Berkeley (RISELab) and is now used at OpenAI, Anthropic, Uber, and most large ML shops. The core insight: **Python functions and classes should scale to a cluster with minimal code changes.**

Ray's architecture:

```
┌────────────────────────────────────────────────────────┐
│                     Head Node                          │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │  Global  │  │   GCS    │  │    Raylet (local   │   │
│  │ Control  │  │ (metadata│  │    scheduler)      │   │
│  │  Store   │  │  store)  │  │                    │   │
│  └──────────┘  └──────────┘  └───────────────────┘   │
└────────────────────────────────────────────────────────┘
        │                              │
┌───────────────┐              ┌───────────────┐
│  Worker Node  │              │  Worker Node  │
│  ┌─────────┐  │              │  ┌─────────┐  │
│  │ Raylet  │  │              │  │ Raylet  │  │
│  │(sched)  │  │              │  │(sched)  │  │
│  └─────────┘  │              │  └─────────┘  │
│  ┌─────────┐  │              │  ┌─────────┐  │
│  │ Object  │  │              │  │ Object  │  │
│  │  Store  │  │              │  │  Store  │  │
│  │(Plasma) │  │              │  │(Plasma) │  │
│  └─────────┘  │              │  └─────────┘  │
│  Workers...   │              │  Workers...   │
└───────────────┘              └───────────────┘
```

**Global Control Store (GCS)** is the brain — tracks all actors, tasks, and resources. **Raylet** is the local scheduler on each node — handles task dispatch and object transfers. **Plasma Object Store** is a shared-memory store for zero-copy reads between co-located workers.

### Ray Core: Tasks and Actors {#ray-core}

```python
import ray
import time

ray.init()  # connects to cluster, or starts local cluster

# --- Tasks: stateless remote functions ---

@ray.remote
def fetch_data(url: str) -> dict:
    import requests
    return requests.get(url).json()

@ray.remote(num_cpus=2, num_gpus=0.5)  # resource annotation
def preprocess(data: dict) -> list:
    # ... CPU-intensive work
    return processed

# Tasks return ObjectRefs (futures), not values
ref1 = fetch_data.remote("https://api.example.com/data/1")
ref2 = fetch_data.remote("https://api.example.com/data/2")

# ray.get blocks until result is ready
results = ray.get([ref1, ref2])

# ray.wait for partial completion (don't wait for all)
refs = [fetch_data.remote(url) for url in urls]
ready, not_ready = ray.wait(refs, num_returns=1, timeout=5.0)
# process ready results while others are still running
first_done = ray.get(ready[0])


# --- Actors: stateful distributed objects ---

@ray.remote
class ModelServer:
    def __init__(self, model_path: str):
        import torch
        self.model = torch.load(model_path)
        self.request_count = 0
    
    def predict(self, batch: list) -> list:
        self.request_count += 1
        with torch.no_grad():
            return self.model(batch).tolist()
    
    def get_stats(self) -> dict:
        return {"requests": self.request_count}

# Create actor (starts a remote process)
server = ModelServer.remote("model.pt")

# Calls return ObjectRefs
pred_ref = server.predict.remote(my_batch)
predictions = ray.get(pred_ref)

# Actor pool for parallel serving
from ray.util import ActorPool

servers = [ModelServer.remote("model.pt") for _ in range(4)]
pool = ActorPool(servers)

# Map over inputs using the pool
results = list(pool.map(lambda s, batch: s.predict.remote(batch), batches))
```

**Passing large objects efficiently:**

```python
# Put large object in object store once, share reference
large_data = ray.put(my_numpy_array)  # returns ObjectRef

@ray.remote
def process_shard(data_ref, shard_idx: int):
    data = ray.get(data_ref)  # zero-copy if on same node
    return data[shard_idx * 1000:(shard_idx + 1) * 1000].sum()

# All tasks share the same object — no redundant serialization
futures = [process_shard.remote(large_data, i) for i in range(100)]
```

**Named actors for service discovery:**

```python
@ray.remote
class ParameterServer:
    def __init__(self):
        self.params = {}
    
    def push(self, key, value):
        self.params[key] = value
    
    def pull(self, key):
        return self.params.get(key)

# Create with a name so other processes can find it
ps = ParameterServer.options(name="global_ps", lifetime="detached").remote()

# In another process/node:
ps = ray.get_actor("global_ps")
params = ray.get(ps.pull.remote("model_weights"))
```

### Ray Data {#ray-data}

Ray Data is for distributed data loading and preprocessing — the stage before ML training.

```python
import ray.data as rd

# Read from various sources
ds = rd.read_parquet("s3://my-bucket/train/")
ds = rd.read_csv("data/*.csv")
ds = rd.read_images("images/", size=(224, 224))

# Transformations are lazy — build execution plan
ds = ds.filter(lambda row: row["label"] != -1)
ds = ds.map(lambda row: {**row, "feature": row["x"] ** 2})

# Batched transforms (more efficient for GPU preprocessing)
def preprocess_batch(batch: dict) -> dict:
    import numpy as np
    batch["image"] = batch["image"].astype(np.float32) / 255.0
    return batch

ds = ds.map_batches(preprocess_batch, batch_size=256)

# Parallel transforms with GPU
@ray.remote(num_gpus=1)
class ImageEmbedder:
    def __init__(self):
        import clip
        self.model, self.preprocess = clip.load("ViT-B/32")
    
    def embed(self, batch):
        # GPU preprocessing
        ...

ds = ds.map_batches(
    ImageEmbedder,
    batch_size=64,
    num_gpus=1,          # each batch actor gets 1 GPU
    concurrency=4,       # 4 parallel actors
)

# Materialize and iterate
for batch in ds.iter_batches(batch_size=512):
    train_step(batch)

# Split for distributed training
train_ds, val_ds = ds.train_test_split(test_size=0.1)
shards = train_ds.split(n=4)  # one shard per GPU worker
```

### Ray Train {#ray-train}

Ray Train is the distributed training library. It handles GPU communication, fault tolerance, and checkpoint management.

```python
import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def train_loop_per_worker(config: dict):
    # This function runs on each worker
    # Ray Train automatically wraps model with DDP
    
    model = MyModel(config["hidden_size"])
    model = train.torch.prepare_model(model)  # wraps in DDP
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    # Get the dataset shard for this worker
    dataset_shard = train.get_dataset_shard("train")
    
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        
        for batch in dataset_shard.iter_torch_batches(
            batch_size=config["batch_size"],
            dtypes={"x": torch.float32, "y": torch.long},
        ):
            optimizer.zero_grad()
            outputs = model(batch["x"])
            loss = criterion(outputs, batch["y"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Report metrics — Ray Train aggregates across workers
        train.report(
            {"loss": total_loss / len(dataset_shard), "epoch": epoch},
            checkpoint=train.Checkpoint.from_dict(
                {"model_state": model.state_dict()}
            ),
        )

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={
        "hidden_size": 256,
        "lr": 1e-3,
        "epochs": 10,
        "batch_size": 128,
    },
    scaling_config=ScalingConfig(
        num_workers=4,         # 4 GPU workers
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 4},
    ),
    run_config=RunConfig(
        name="my_experiment",
        storage_path="s3://my-bucket/ray-results/",
        checkpoint_config=CheckpointConfig(
            num_to_keep=3,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
        ),
    ),
    datasets={"train": ray.data.read_parquet("s3://data/train/")},
)

result = trainer.fit()
best_checkpoint = result.best_checkpoints[0][0]
print(result.metrics)
```

**Fault tolerance** is where Ray Train shines — if a worker dies, Ray restores from the last checkpoint and restarts only the failed worker. This is critical for long training runs on preemptible VMs.

### Ray Serve {#ray-serve}

Ray Serve is for model serving — it handles HTTP, batching, model composition, and autoscaling.

```python
from ray import serve
import ray
from transformers import pipeline

ray.init()
serve.start()

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 8,
        "target_num_ongoing_requests_per_replica": 10,
    },
)
class SentimentModel:
    def __init__(self):
        self.model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0,  # GPU 0
        )
    
    async def __call__(self, request):
        data = await request.json()
        result = self.model(data["text"])
        return {"sentiment": result[0]["label"], "score": result[0]["score"]}


# Model composition — chain multiple models
@serve.deployment
class EnsembleModel:
    def __init__(self, model_a, model_b):
        self.model_a = model_a
        self.model_b = model_b
    
    async def __call__(self, request):
        data = await request.json()
        # Call both models in parallel
        ref_a = self.model_a.predict.remote(data)
        ref_b = self.model_b.predict.remote(data)
        results = await asyncio.gather(
            asyncio.wrap_future(ref_a.future()),
            asyncio.wrap_future(ref_b.future()),
        )
        return {"ensemble": (results[0] + results[1]) / 2}


# Bind and deploy
sentiment_app = SentimentModel.bind()
serve.run(sentiment_app, route_prefix="/sentiment")

# Test
import requests
resp = requests.post(
    "http://localhost:8000/sentiment",
    json={"text": "This is amazing!"}
)
print(resp.json())
```

**Batching for throughput:**

```python
@serve.deployment
class BatchedInference:
    def __init__(self):
        import torch
        self.model = load_model()
    
    @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)
    async def handle_batch(self, requests: list) -> list:
        # requests are batched automatically
        inputs = [r["input"] for r in requests]
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs.tolist()
    
    async def __call__(self, request):
        data = await request.json()
        result = await self.handle_batch(data)
        return {"output": result}
```

### Ray Tune {#ray-tune}

Ray Tune is hyperparameter optimization at scale — runs many trials in parallel, prunes bad ones early.

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

def trainable(config):
    model = build_model(
        hidden_size=config["hidden_size"],
        dropout=config["dropout"],
        lr=config["lr"],
    )
    
    for epoch in range(50):
        train_loss = train_one_epoch(model, config["lr"])
        val_loss = evaluate(model)
        
        # Report metric to Tune
        tune.report(
            loss=val_loss,
            train_loss=train_loss,
            epoch=epoch,
        )

# ASHA: Asynchronous Successive Halving — kills bad trials early
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=50,           # max epochs
    grace_period=5,     # min epochs before pruning
    reduction_factor=2, # halve resources each round
)

# Optuna for Bayesian search
search_alg = OptunaSearch(metric="loss", mode="min")

tuner = tune.Tuner(
    trainable,
    param_space={
        "hidden_size": tune.choice([64, 128, 256, 512]),
        "dropout": tune.uniform(0.1, 0.5),
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
    },
    tune_config=tune.TuneConfig(
        num_samples=100,      # total trials
        scheduler=scheduler,
        search_alg=search_alg,
    ),
    run_config=tune.RunConfig(
        name="hpo_experiment",
        storage_path="~/ray_results",
    ),
)

results = tuner.fit()
best_config = results.get_best_result(metric="loss", mode="min").config
print(f"Best config: {best_config}")
```

**Population-Based Training (PBT)** — adapts hyperparameters during training (used by DeepMind for RL):

```python
from ray.tune.schedulers import PopulationBasedTraining

pbt = PopulationBasedTraining(
    time_attr="epoch",
    metric="val_loss",
    mode="min",
    perturbation_interval=5,   # perturb every 5 epochs
    hyperparam_mutations={
        "lr": tune.loguniform(1e-5, 1e-1),
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    },
)
```

---

## Dask — Parallel Pandas/NumPy {#dask}

Dask is the "scale pandas/numpy to a cluster" tool. It uses the same API as pandas and numpy, but builds a lazy task graph that executes in parallel. Much easier to adopt than Spark if your team already knows pandas.

```
Dask vs Spark mental model:

  Spark: JVM-based, starts from scratch, great for PB-scale
  Dask:  Python-native, extends pandas/numpy, great for 10GB–10TB
```

Dask architecture:

```
┌──────────────────────┐
│       Client         │  ← your code
│  (builds task graph) │
└──────────┬───────────┘
           │ submits graph
┌──────────▼───────────┐
│      Scheduler       │  ← distributed.LocalCluster or remote
│  (assigns tasks)     │
└──────┬───────┬───────┘
       │       │
┌──────▼──┐ ┌──▼──────┐
│ Worker  │ │ Worker  │  ← run tasks, communicate results
└─────────┘ └─────────┘
```

### Dask DataFrame {#dask-dataframe}

```python
import dask.dataframe as dd
import pandas as pd

# Read partitioned data — each file becomes a partition
df = dd.read_parquet("s3://bucket/data/*.parquet")
df = dd.read_csv("data/year=2024/month=*/*.csv")

# Check structure without loading
print(df.dtypes)         # known from metadata
print(df.npartitions)    # number of partitions
print(df.divisions)      # partition boundaries (if sorted)

# Operations look identical to pandas
df_filtered = df[df["revenue"] > 1000]
df_enriched = df_filtered.assign(margin=df_filtered["profit"] / df_filtered["revenue"])
result = df_enriched.groupby("region")["margin"].mean()

# Nothing has run yet — result is a "lazy" object
# .compute() triggers execution
final = result.compute()   # returns a pandas Series

# Merge two large datasets
orders = dd.read_parquet("s3://bucket/orders/")
customers = dd.read_parquet("s3://bucket/customers/")

# Dask handles the distributed join
merged = orders.merge(customers, on="customer_id", how="left")
merged.compute()

# Repartition for better parallelism
df = df.repartition(npartitions=100)         # by count
df = df.set_index("date", sorted=True)       # enables fast slicing
df_sorted = df.sort_values("timestamp")      # expensive shuffle

# Rolling/time-series
df = df.set_index("timestamp")
rolling_avg = df["value"].rolling("7d").mean()

# Apply custom function per partition
def clean_partition(df):
    df["text"] = df["text"].str.lower().str.strip()
    return df

df = df.map_partitions(clean_partition)

# Sample without loading all data
sample = df.sample(frac=0.01).compute()
```

**Distributed setup:**

```python
from dask.distributed import Client, LocalCluster

# Local cluster (multiprocess)
cluster = LocalCluster(
    n_workers=4,
    threads_per_worker=2,
    memory_limit="4GB",
)
client = Client(cluster)

# Remote cluster (Kubernetes, YARN, AWS)
from dask_kubernetes import KubeCluster
cluster = KubeCluster(pod_spec="worker-pod.yaml")
cluster.scale(20)  # 20 workers
client = Client(cluster)

# Check cluster status
print(client.scheduler_info())
client.dashboard_link  # http://localhost:8787 — visual task graph
```

### Dask Array {#dask-array}

```python
import dask.array as da
import numpy as np

# Create from numpy chunks
x = da.from_array(large_numpy_array, chunks=(1000, 1000))

# Create directly
x = da.zeros((100_000, 100_000), chunks=(5000, 5000))

# Operations mirror numpy
y = da.sin(x) + da.cos(x)
z = da.dot(x, x.T)         # distributed matrix multiply
svd_u, svd_s, svd_v = da.linalg.svd(x)  # distributed SVD

# Slicing works naturally
subset = x[1000:5000, 2000:6000]

result = z.compute()  # triggers computation

# Rechunk for different access patterns
x = x.rechunk({0: "auto", 1: -1})  # auto rows, full columns
```

### Dask Delayed {#dask-delayed}

For arbitrary Python code that doesn't fit the DataFrame/Array APIs:

```python
from dask import delayed
import time

@delayed
def load_file(path: str) -> dict:
    # any Python — runs in parallel
    with open(path) as f:
        return json.load(f)

@delayed
def process(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data["records"])

@delayed
def aggregate(dfs: list) -> pd.DataFrame:
    return pd.concat(dfs)

# Build the graph
paths = glob.glob("data/*.json")
loaded = [load_file(p) for p in paths]
processed = [process(d) for d in loaded]
result = aggregate(processed)

# Visualize the task graph (requires graphviz)
result.visualize("task_graph.png")

# Execute
final_df = result.compute()

# Or submit to cluster and get a future
future = client.compute(result)
final_df = future.result()
```

### Dask ML {#dask-ml}

```python
import dask_ml.model_selection as dcv
from dask_ml.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import dask.array as da

# Load large dataset
X = da.from_zarr("features.zarr")
y = da.from_zarr("labels.zarr")

# Dask-ML wraps sklearn to work on dask arrays
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=100)),
])

# Distributed cross-validation
cv = dcv.GridSearchCV(
    model,
    param_grid={"clf__C": [0.1, 1.0, 10.0]},
    cv=5,
    scoring="accuracy",
)
cv.fit(X, y)
print(cv.best_params_)

# Incremental learning (fits model in chunks)
from dask_ml.linear_model import PartialSGDClassifier
clf = PartialSGDClassifier()
for X_chunk, y_chunk in zip(X.blocks, y.blocks):
    clf.partial_fit(X_chunk.compute(), y_chunk.compute())
```

---

## PyTorch Distributed {#pytorch-distributed}

PyTorch has first-class distributed training support. It's what most industry ML training uses — Ray Train and Horovod both build on top of it.

### DDP — Data Parallel {#ddp}

Distributed Data Parallel (DDP): each GPU holds a full model copy. On backward pass, gradients are all-reduced (averaged) across GPUs using NCCL.

```
Forward pass:   each GPU processes its data shard independently
Backward pass:  ring all-reduce syncs gradients across all GPUs
Weight update:  identical on all GPUs (same gradients, same update)
```

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank: int, world_size: int):
    dist.init_process_group(
        backend="nccl",       # GPU-to-GPU: nccl; CPU-only: gloo
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank: int, world_size: int, config: dict):
    setup(rank, world_size)
    
    # Build model and move to this rank's GPU
    model = MyModel(config).to(rank)
    
    # Wrap in DDP — handles gradient sync automatically
    model = DDP(model, device_ids=[rank])
    
    # DistributedSampler ensures each GPU gets different data
    dataset = MyDataset(config["data_path"])
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config["epochs"]):
        sampler.set_epoch(epoch)  # ensures different shuffle each epoch
        
        model.train()
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(rank), labels.to(rank)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # all-reduce happens here automatically
            optimizer.step()
            
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Only save checkpoint on rank 0 to avoid conflicts
        if rank == 0:
            torch.save(model.module.state_dict(), f"checkpoint_{epoch}.pt")
    
    cleanup()

# Launch with torchrun (replaces torch.multiprocessing.spawn)
# torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
#   --master_addr="192.168.1.1" --master_port=1234 train.py

# Or programmatically:
import torch.multiprocessing as mp
world_size = 4
mp.spawn(train, args=(world_size, config), nprocs=world_size)
```

**Key DDP gotchas:**
- Batch size scales with world size. If you use batch_size=32 per GPU and 8 GPUs, effective batch size is 256 — scale your learning rate accordingly (linear scaling rule: `lr = base_lr * world_size`)
- `model.module` accesses the underlying model inside DDP wrapper — needed when saving
- `sampler.set_epoch(epoch)` before each epoch — otherwise all epochs use the same shuffle

### FSDP — Fully Sharded {#fsdp}

DDP keeps a full model copy on each GPU — impossible for large models. FSDP shards parameters, gradients, and optimizer states across GPUs.

```
DDP:   each GPU holds full model (100B params = 200GB per GPU at bf16)
FSDP:  each GPU holds 1/N of model (100B params across 64 GPUs = 3.1GB per GPU)
```

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from transformers import LlamaForCausalLM
import functools

def train_fsdp(rank: int, world_size: int):
    setup(rank, world_size)
    
    # Auto wrap policy — wraps transformer blocks individually
    # This is key: each block is a separate FSDP unit
    llama_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},  # your layer class
    )
    
    # Mixed precision: compute in bf16, master weights in fp32
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
    
    model = FSDP(
        model,
        auto_wrap_policy=llama_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # shard params+grads+optim
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # overlap comm+compute
        cpu_offload=CPUOffload(offload_params=False),     # True if GPU OOM
        device_id=rank,
    )
    
    # Optimizer sees only the local shard's parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop same as DDP
    for epoch in range(epochs):
        for batch in loader:
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # Save checkpoint (different from DDP — need to gather shards)
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()
        if rank == 0:
            torch.save(state_dict, "model.pt")
```

**FSDP sharding strategies:**

| Strategy | Params | Gradients | Optimizer State |
|---|---|---|---|
| `NO_SHARD` | Full | Full | Full (= DDP) |
| `SHARD_GRAD_OP` | Full (sharded during forward) | Sharded | Sharded |
| `FULL_SHARD` | Sharded | Sharded | Sharded |
| `HYBRID_SHARD` | Full within node, sharded across | Mixed | Mixed |

### torch.distributed Primitives {#torch-distributed-primitives}

```python
import torch.distributed as dist

# Collective operations — all processes participate
tensor = torch.tensor([rank], dtype=torch.float32).cuda()

# All-reduce: sum/mean across all ranks
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
# tensor is now [sum across all ranks] on every rank

# Broadcast: rank 0 sends to all
if rank == 0:
    tensor = torch.randn(100).cuda()
dist.broadcast(tensor, src=0)
# all ranks now have rank 0's tensor

# Gather: collect from all ranks to rank 0
gathered = [torch.zeros(1).cuda() for _ in range(world_size)]
dist.gather(tensor, gather_list=gathered if rank == 0 else None, dst=0)

# All-gather: every rank gets all tensors
gathered_all = [torch.zeros(1).cuda() for _ in range(world_size)]
dist.all_gather(gathered_all, tensor)

# Scatter: rank 0 distributes different tensors to each rank
if rank == 0:
    scatter_list = [torch.tensor([i]).cuda() for i in range(world_size)]
dist.scatter(tensor, scatter_list if rank == 0 else None, src=0)

# Reduce-scatter: reduce then scatter (used in FSDP)
output = torch.zeros(100 // world_size).cuda()
dist.reduce_scatter(output, input_list)

# Barrier: synchronization point (all ranks wait)
dist.barrier()

# Point-to-point
if rank == 0:
    dist.send(tensor, dst=1)
elif rank == 1:
    dist.recv(tensor, src=0)
```

**Ring all-reduce** (what NCCL does for gradient averaging):

```
Step 1 (Scatter-reduce): each GPU sends 1/N of its gradient to the next
  GPU0: sends chunk0 to GPU1
  GPU1: sends chunk1 to GPU2
  ...  (N-1 steps, each GPU accumulates one chunk)

Step 2 (All-gather): each GPU broadcasts its accumulated chunk
  GPU0: sends its chunk to GPU1
  GPU1: sends its chunk to GPU2
  ...  (N-1 steps, all GPUs get all chunks)

Communication cost: 2 * (N-1)/N * data_size ≈ 2 * data_size
Independent of N — bandwidth efficient!
```

---

## Horovod {#horovod}

Horovod (Uber, 2018) is an MPI-based distributed training library that works on top of TensorFlow, PyTorch, and MXNet. It predates PyTorch's built-in DDP and was the go-to until ~2021. Still used in HPC environments where MPI is the native communication substrate.

```python
import horovod.torch as hvd
import torch
import torch.nn as nn

# Initialize Horovod (wraps MPI_Init)
hvd.init()

# Each process gets a rank
rank = hvd.rank()           # process rank
local_rank = hvd.local_rank()  # GPU index on this machine
world_size = hvd.size()

torch.cuda.set_device(local_rank)

# Build model
model = MyModel().cuda()

# Scale learning rate linearly with world size
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01 * world_size,
    momentum=0.9,
)

# Broadcast initial parameters from rank 0 to all workers
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# Wrap optimizer — adds all-reduce of gradients
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters(),
    compression=hvd.Compression.fp16,  # compress gradients for speed
)

# Distributed sampler
train_sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, num_replicas=world_size, rank=rank
)
loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)

# Training — identical to normal PyTorch
for epoch in range(100):
    for inputs, labels in loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()  # all-reduce + step

# Only rank 0 saves
if rank == 0:
    torch.save(model.state_dict(), "model.pt")
```

**Launch with MPI:**
```bash
# 4 GPUs on one machine
horovodrun -np 4 python train.py

# 16 GPUs across 2 machines (8 GPUs each)
horovodrun -np 16 -H server1:8,server2:8 python train.py

# On SLURM cluster
srun --ntasks=16 --gpus-per-task=1 python train.py
```

**Horovod Elastic** — handles node failures and autoscaling:
```python
@hvd.elastic.run
def train(state):
    # state.epoch, state.model, state.optimizer
    for epoch in range(state.epoch, config.epochs):
        train_one_epoch(state)
        state.epoch = epoch + 1
        state.commit()  # checkpoint this epoch

state = hvd.elastic.TorchState(model, optimizer, epoch=0)
train(state)
```

**When Horovod over DDP?**
- Running on SLURM/MPI clusters (HPC centers)
- Multi-framework teams (some TF, some PyTorch)
- Need elastic training with heterogeneous hardware
- Legacy codebases already using Horovod

---

## Celery — Distributed Task Queues {#celery}

Celery is the standard distributed task queue for Python web applications. It's not for ML training — it's for offloading work from web servers: sending emails, processing uploads, running reports.

```
Web Request → Django/FastAPI → Celery Task → Redis/RabbitMQ → Celery Worker
```

```python
# celery_app.py
from celery import Celery
from celery.utils.log import get_task_logger
import redis

# Redis as broker (job queue) and backend (result store)
app = Celery(
    "myapp",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=["tasks"],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    task_track_started=True,
    task_acks_late=True,          # ack after completion (at-least-once)
    worker_prefetch_multiplier=1, # one task per worker at a time
    task_soft_time_limit=3600,    # soft limit: raises SoftTimeLimitExceeded
    task_time_limit=3700,         # hard limit: kills worker
)
```

```python
# tasks.py
from celery_app import app
from celery.utils.log import get_task_logger
import time

logger = get_task_logger(__name__)

@app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,  # seconds between retries
)
def process_video(self, video_id: str, user_id: str):
    try:
        logger.info(f"Processing video {video_id}")
        
        # Update task state for progress tracking
        self.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Downloading"}
        )
        
        download_video(video_id)
        
        self.update_state(
            state="PROGRESS",
            meta={"current": 50, "total": 100, "status": "Transcoding"}
        )
        
        transcode_video(video_id)
        notify_user(user_id, "Your video is ready!")
        
        return {"status": "success", "video_id": video_id}
    
    except VideoDownloadError as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
    
    except Exception as exc:
        logger.error(f"Failed to process {video_id}: {exc}")
        raise


@app.task
def send_email(to: str, subject: str, body: str):
    import smtplib
    # ... send email
    return {"sent": True}


# Chains — sequential tasks
from celery import chain, group, chord

# Run tasks sequentially
workflow = chain(
    download_video.s(video_id),
    transcode_video.s(),         # receives result from previous
    notify_user.s(user_id),
)
result = workflow.delay()

# Groups — parallel tasks, same data
workflow = group(
    transcode_to_720p.s(video_id),
    transcode_to_1080p.s(video_id),
    generate_thumbnail.s(video_id),
)
result = workflow.apply_async()

# Chord — parallel tasks, then callback when all done
workflow = chord(
    group(
        transcode_to_720p.s(video_id),
        transcode_to_1080p.s(video_id),
    ),
    notify_completion.s(user_id),  # called with list of results
)
result = workflow.delay()
```

```python
# In Django/FastAPI view
from tasks import process_video
from celery.result import AsyncResult

# Submit task (non-blocking)
result = process_video.delay(video_id="abc123", user_id="user42")
task_id = result.id

# Check status endpoint
def get_task_status(task_id: str):
    result = AsyncResult(task_id)
    if result.state == "PENDING":
        return {"status": "queued"}
    elif result.state == "PROGRESS":
        return {"status": "processing", "progress": result.info}
    elif result.state == "SUCCESS":
        return {"status": "done", "result": result.result}
    elif result.state == "FAILURE":
        return {"status": "failed", "error": str(result.info)}
```

**Start workers:**
```bash
# Start worker
celery -A celery_app worker --loglevel=info --concurrency=8

# Named queues — route heavy tasks to GPU workers
celery -A celery_app worker -Q gpu_tasks --concurrency=1
celery -A celery_app worker -Q default --concurrency=16

# Monitor with Flower
pip install flower
celery -A celery_app flower --port=5555

# Periodic tasks (cron-like)
celery -A celery_app beat --scheduler django_celery_beat.schedulers:DatabaseScheduler
```

**Routing tasks to specific queues:**
```python
@app.task(queue="gpu_tasks")
def run_inference(input_data):
    ...

# Or at call time
process_video.apply_async(args=[video_id], queue="gpu_tasks", priority=9)
```

**Celery pitfalls:**
- Never pass SQLAlchemy model instances as task arguments — pass IDs and re-fetch inside the task (database connections aren't picklable)
- `task_acks_late=True` prevents task loss on worker crash, but can cause duplicates — make tasks idempotent
- Workers consume memory over time — use `--max-tasks-per-child=100` to restart workers periodically
- Redis as broker doesn't persist — use RabbitMQ for guaranteed delivery at the cost of complexity

---

## Apache Beam — Unified Batch + Stream {#apache-beam}

Apache Beam is a unified programming model for batch and streaming data pipelines. Write the pipeline once, run it on Spark, Flink, Google Dataflow, or locally. Developed by Google, used heavily in GCP.

The key insight: **a bounded dataset is just an unbounded stream with a known end**. Beam abstracts over this distinction.

```
PCollection: distributed dataset (bounded or unbounded)
PTransform:  operation on PCollections (Map, Filter, GroupByKey, etc.)
Pipeline:    DAG of PTransforms
Runner:      execution engine (DirectRunner, DataflowRunner, FlinkRunner)
```

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Batch pipeline
options = PipelineOptions([
    "--runner=DataflowRunner",     # or DirectRunner, FlinkRunner
    "--project=my-gcp-project",
    "--region=us-central1",
    "--staging_location=gs://bucket/staging",
    "--temp_location=gs://bucket/temp",
    "--job_name=my-pipeline",
    "--max_num_workers=50",
])

with beam.Pipeline(options=options) as p:
    # Read input
    lines = p | "ReadCSV" >> beam.io.ReadFromText("gs://bucket/input/*.csv")
    
    # Parse
    def parse_line(line: str) -> dict:
        parts = line.split(",")
        return {"user_id": parts[0], "event": parts[1], "ts": int(parts[2])}
    
    records = (
        lines
        | "Parse" >> beam.Map(parse_line)
        | "FilterValid" >> beam.Filter(lambda r: r["event"] != "")
    )
    
    # Group and aggregate
    by_user = (
        records
        | "KeyByUser" >> beam.Map(lambda r: (r["user_id"], r))
        | "GroupByUser" >> beam.GroupByKey()
        | "CountEvents" >> beam.Map(
            lambda kv: {"user_id": kv[0], "event_count": len(list(kv[1]))}
        )
    )
    
    # Write output
    by_user | "WriteOutput" >> beam.io.WriteToText("gs://bucket/output/counts")


# Streaming pipeline with Pub/Sub
streaming_options = PipelineOptions([
    "--runner=DataflowRunner",
    "--streaming",  # enable streaming mode
    ...
])

with beam.Pipeline(options=streaming_options) as p:
    messages = (
        p
        | "ReadPubSub" >> beam.io.ReadFromPubSub(
            subscription="projects/my-project/subscriptions/my-sub"
        )
        | "DecodeJSON" >> beam.Map(lambda msg: json.loads(msg.decode()))
    )
    
    # Windowing — group events into time windows
    windowed = (
        messages
        | "AddTimestamps" >> beam.Map(
            lambda r: beam.window.TimestampedValue(r, r["timestamp"])
        )
        | "WindowInto5min" >> beam.WindowInto(
            beam.window.FixedWindows(5 * 60)  # 5-minute windows
        )
        | "KeyByEvent" >> beam.Map(lambda r: (r["event_type"], 1))
        | "CountPerWindow" >> beam.CombinePerKey(sum)
    )
    
    windowed | "WriteBigQuery" >> beam.io.WriteToBigQuery(
        "my-project:dataset.event_counts",
        schema="event_type:STRING, count:INTEGER, window_start:TIMESTAMP",
        write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
    )
```

**DoFn — stateful custom transforms:**

```python
class ParseAndEnrichDoFn(beam.DoFn):
    def setup(self):
        # Called once per worker — initialize connections
        self.db = connect_to_db()
    
    def process(self, element, *args, **kwargs):
        # Called for each element — must be a generator
        try:
            record = parse(element)
            enriched = self.db.lookup(record["id"])
            yield {**record, **enriched}
        except ParseError as e:
            yield beam.pvalue.TaggedOutput("parse_errors", {"raw": element, "error": str(e)})
    
    def teardown(self):
        self.db.close()

# Multiple output tags
results = (
    records
    | "ParseAndEnrich" >> beam.ParDo(ParseAndEnrichDoFn()).with_outputs(
        "parse_errors",
        main="valid",
    )
)
valid = results.valid
errors = results.parse_errors
```

**Side inputs — broadcast small dataset to all workers:**

```python
# Load lookup table as side input
lookup_table = p | "ReadLookup" >> beam.io.ReadFromText("gs://bucket/lookup.csv")

# Use as a side input in map
enriched = records | "Enrich" >> beam.Map(
    lambda record, lookup: {**record, "category": lookup.get(record["id"])},
    lookup=beam.pvalue.AsDict(lookup_table),  # materialised as dict on each worker
)
```

**Beam vs alternatives:**

| | Beam | Spark | Flink | Dataflow |
|---|---|---|---|---|
| Portability | Any runner | Spark only | Flink only | GCP only |
| Batch | ✓ | ✓ | ✓ | ✓ |
| Streaming | ✓ | Micro-batch | True streaming | ✓ |
| Language | Python/Java/Go | Python/Scala/Java | Python/Java | Python/Java |
| Local dev | DirectRunner | Local mode | Local mode | DirectRunner |
| State | Per-key | Per-partition | Per-key | Per-key |

---

## Framework Comparison {#framework-comparison}

### Head-to-head: Ray vs Dask vs Spark

| Dimension | Ray | Dask | Spark |
|---|---|---|---|
| Primary use case | ML workflows end-to-end | Parallel pandas/numpy | Large-scale SQL/ETL |
| API style | Task graph + actors | Familiar pandas/numpy | SQL + DataFrame |
| Ecosystem | Ray Train/Serve/Tune | Dask-ML, Dask-Kubernetes | MLlib, Structured Streaming |
| Fault tolerance | Yes (actor state too) | Yes (task retry) | Yes (RDD lineage) |
| Data scale | 10GB–10TB | 10GB–10TB | 100GB–PB |
| JVM required | No | No | Yes |
| GPU support | First-class | Via CuDF | Limited |
| Setup complexity | Medium | Low | High |
| Overhead | Low per-task | Low | High (JVM startup) |
| Best for | ML training, serving, HPO | Data science, sklearn scale-out | ETL, SQL, structured data |

### When each tool wins

```
┌─────────────────────────────────────────────────────────────────┐
│  Task                              → Best Tool                  │
├─────────────────────────────────────────────────────────────────┤
│  Distributed LLM training          → PyTorch FSDP + DeepSpeed  │
│  Parallel sklearn on 100GB         → Dask-ML                   │
│  ML experiment orchestration       → Ray (Core + Tune + Train) │
│  Real-time model serving           → Ray Serve or Triton       │
│  Hyperparameter search             → Ray Tune                  │
│  PB-scale SQL/ETL                  → Spark                     │
│  Streaming ETL (event time)        → Flink                     │
│  Web task queue (email, reports)   → Celery                    │
│  Multi-cloud/portable pipeline     → Apache Beam               │
│  GCP managed streaming             → Beam on Dataflow          │
│  HPC cluster (SLURM/MPI)          → Horovod or PyTorch+MPI    │
│  Parallel pandas on a laptop       → Dask (no cluster needed)  │
└─────────────────────────────────────────────────────────────────┘
```

### Gradient communication comparison

All distributed training frameworks ultimately reduce to a few communication patterns:

```
Parameter Server (old):
  Workers push gradients to PS, PS averages, sends back weights
  Bottleneck: PS bandwidth is O(N * model_size)

Ring All-Reduce (DDP, Horovod):
  Workers form a ring, pass gradients around
  No bottleneck: bandwidth cost is O(2 * model_size) regardless of N

Sharding (FSDP, DeepSpeed ZeRO):
  Split model, gradients, optimizer state across GPUs
  Memory cost is O(model_size / N) per GPU
  Communication: all-gather before forward, reduce-scatter after backward
```

---

## Choosing the Right Tool {#choosing}

**Are you doing ML?**

```
Yes → Is your model > 40GB?
        Yes → PyTorch FSDP or DeepSpeed ZeRO-3
        No  → DDP (torchrun, Ray Train, or Horovod on SLURM)
      Do you need HPO?
        Yes → Ray Tune
      Do you need serving?
        Yes → Ray Serve (flexible) or Triton (pure inference throughput)
      Do you need end-to-end ML orchestration?
        Yes → Ray (Core + Data + Train + Tune + Serve)

No → Is this ETL/data processing?
       Scale > 1TB or SQL-heavy?
         Yes → Spark
         No  → Dask (if pandas-like) or Beam (if multi-runner portability)
       Is this a web task queue?
         Yes → Celery
       Is this a streaming pipeline?
         GCP → Beam on Dataflow
         Otherwise → Flink (true streaming) or Spark Structured Streaming
```

**Common mistakes:**

1. **Using Spark for 10GB of data** — JVM overhead makes it slower than pandas or Dask. Spark is for when data genuinely doesn't fit on one machine.

2. **Using DDP for a 70B parameter model** — each GPU needs 140GB at fp16. FSDP or DeepSpeed ZeRO-3 required.

3. **Using Celery for ML training** — Celery is a task queue, not a compute framework. Workers don't share GPU memory or coordinate gradients.

4. **Not pinning memory in DataLoaders** — `pin_memory=True` in PyTorch DataLoader enables async CPU→GPU transfers. Free performance.

5. **Ignoring communication overhead** — gradient all-reduce is blocking. For small models on many GPUs, communication cost > compute cost. Use gradient compression (Horovod fp16 compression, PowerSGD) or larger batch sizes.

6. **Ray actor state not checkpointed** — if an actor dies, its state is lost unless you checkpoint to durable storage (S3, Redis).

7. **Dask partition count mismatch** — too few partitions → cores idle; too many → scheduling overhead. Target 2–4 partitions per CPU core, 100–200MB per partition.

---

## Practical Setup Reference

**Ray cluster on Kubernetes:**

```yaml
# ray-cluster.yaml
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: ray-cluster
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
          - name: ray-head
            image: rayproject/ray:2.9.0-gpu
            resources:
              limits:
                cpu: "4"
                memory: "16Gi"
  workerGroupSpecs:
    - groupName: gpu-workers
      replicas: 4
      rayStartParams: {}
      template:
        spec:
          containers:
            - name: ray-worker
              image: rayproject/ray:2.9.0-gpu
              resources:
                limits:
                  cpu: "8"
                  memory: "32Gi"
                  nvidia.com/gpu: "1"
```

**torchrun multi-node:**

```bash
# On each node — torchrun handles MASTER_ADDR/PORT env vars
torchrun \
  --nproc_per_node=8 \
  --nnodes=4 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=29500 \
  train.py --config config.yaml
```

**Dask on SLURM:**

```python
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

cluster = SLURMCluster(
    queue="gpu",
    account="my_project",
    cores=8,
    memory="32GB",
    walltime="02:00:00",
    job_extra=["--gres=gpu:1"],
)
cluster.scale(jobs=10)  # 10 SLURM jobs = 10 workers
client = Client(cluster)
```

**DeepSpeed ZeRO-3** (worth knowing — often faster than FSDP for LLMs):

```json
// ds_config.json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"},
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8
  },
  "bf16": {"enabled": true},
  "gradient_clipping": 1.0,
  "train_batch_size": 2048,
  "train_micro_batch_size_per_gpu": 4
}
```

```bash
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json
```

---

All of these frameworks have their place. Ray has become the de facto choice for ML infrastructure teams because it covers the whole lifecycle — data preprocessing, training, HPO, serving — with a consistent Python API. Dask wins for data science teams who know pandas and want to scale with minimal friction. PyTorch Distributed (DDP/FSDP) is what you actually use inside Ray Train or standalone for training. Celery handles the web backend use case that none of the ML frameworks address. Beam is the choice when you need multi-runner portability or are already on GCP.
