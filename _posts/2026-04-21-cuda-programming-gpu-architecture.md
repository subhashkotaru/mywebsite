---
title: "CUDA Programming & GPU Architecture"
date: 2026-04-21
display_order: 9
description: "How CUDA threads, warps, and shared memory map onto GPU hardware — with worked examples on matmul tiling and parallel reduction."
tags: [ml-systems, cuda, gpu, hardware]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#cuda-abstractions">CUDA Programming Abstractions</a>
      <ul class="post-toc-sublist">
        <li><a href="#thread-hierarchy">Thread Hierarchy</a></li>
        <li><a href="#memory-model">Memory Model</a></li>
        <li><a href="#divergence">Warp Divergence</a></li>
      </ul>
    </li>
    <li><a href="#gpu-server">GPU Server Architecture</a>
      <ul class="post-toc-sublist">
        <li><a href="#server-components">Components & Interconnects</a></li>
        <li><a href="#gpu-lineup">GPU Generations</a></li>
      </ul>
    </li>
    <li><a href="#gpu-architecture">GPU Architecture</a>
      <ul class="post-toc-sublist">
        <li><a href="#smm">Streaming Multiprocessors</a></li>
        <li><a href="#tensor-cores">Tensor Cores</a></li>
      </ul>
    </li>
    <li><a href="#matmul-cuda">Case Study: Matrix Multiplication</a>
      <ul class="post-toc-sublist">
        <li><a href="#register-tiling">Register Tiling</a></li>
        <li><a href="#shared-mem-tiling">Shared Memory Tiling</a></li>
      </ul>
    </li>
    <li><a href="#parallel-reduction">Case Study: Parallel Reduction</a>
      <ul class="post-toc-sublist">
        <li><a href="#reduction-v1">V1 — Interleaved Addressing</a></li>
        <li><a href="#reduction-v2">V2 — Sequential Addressing</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## CUDA Programming Abstractions
{: #cuda-abstractions}

CUDA (Compute Unified Device Architecture) was introduced in 2007 alongside the NVIDIA Tesla architecture. It exposes GPU parallelism through a **C-like language** designed to closely match GPU hardware capabilities — keeping the abstraction distance low so you can reason directly about performance.

### Thread Hierarchy
{: #thread-hierarchy}

<div class="post-flow" role="group" aria-label="CUDA thread hierarchy">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Grid — all thread blocks for a kernel launch</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Block — group of threads sharing shared memory &amp; syncing</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Warp — 32 threads executing in lockstep (SIMD)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Thread — one scalar execution lane</span></li>
  </ol>
</div>

Each thread knows its position via four built-in variables:

| Variable | Meaning |
|---|---|
| `gridDim` | Dimensions of the grid (in blocks) |
| `blockIdx` | This block's index within the grid |
| `blockDim` | Dimensions of a block (in threads) |
| `threadIdx` | This thread's index within the block |

A typical kernel launch for an `Nx × Ny` matrix add:

```cuda
dim3 threadsPerBlock(4, 3, 1);   // 12 threads per block
dim3 numBlocks(Nx/4, Ny/3, 1);   // as many blocks as needed
matrixAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);

__global__ void matrixAdd(float A[Ny][Nx], float B[Ny][Nx], float C[Ny][Nx]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    C[j][i] = A[j][i] + B[j][i];
}
```

`__global__` marks a **kernel** — executed on the GPU, callable from the CPU. `__device__` marks a helper callable only from GPU code. The number of thread blocks launched is explicit and independent of data size — you must guard against out-of-bounds access when dimensions aren't multiples of block size.

### Memory Model
{: #memory-model}

<div class="post-flow post-flow--compare" role="group" aria-label="Host vs device memory">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Host (CPU)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Host DRAM — main memory</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Separate address space</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">cudaMemcpy to move data →</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Device (GPU)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Global memory (HBM) — large, slow</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Shared memory — per-block, fast (~L1)</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Registers — per-thread, fastest</span></li>
    </ol>
  </div>
</div>

Host and device have **distinct address spaces** — you cannot dereference a device pointer from the CPU or vice versa. Data must be explicitly moved with `cudaMemcpy`. Shared memory (`__shared__`) is allocated per thread block and is the main tool for reducing expensive global memory traffic.

### Warp Divergence
{: #divergence}

All 32 threads in a warp execute the **same instruction** simultaneously. When a branch condition differs across threads, the GPU must **serialize** both paths — active threads on each side are masked off while the other side runs.

<div class="post-flow post-flow--compare" role="group" aria-label="Coherent vs divergent execution">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Coherent (fast)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">All threads take same branch</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Full warp utilisation</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">1× throughput</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Divergent (slow)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Threads split across branches</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Each path runs sequentially</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Up to 32× throughput loss</span></li>
    </ol>
  </div>
</div>

**Design rule**: ensure threads within a warp take the same branch. Rewrite divergent `if (tid % (2*s) == 0)` loops as strided-index loops where only the *first half* of threads are active — all active threads agree on the branch.

---

## GPU Server Architecture
{: #gpu-server}

Before diving into the SM, it helps to understand the full hardware stack a GPU sits in — because memory bandwidth and interconnect topology directly constrain what optimisations matter.

### Components & Interconnects
{: #server-components}

A modern GPU server pairs multiple CPUs with multiple GPUs. A typical research cluster node looks like:

- **2× AMD EPYC CPUs** (128 cores, 256 threads each), connected to 1 TB of DDR5 host DRAM
- **8× NVIDIA A6000 GPUs** (48 GB each), connected to each other and the CPUs via NVLink and PCIe

The memory bandwidth at each level of the hierarchy:

<div class="post-flow" role="group" aria-label="Memory bandwidth hierarchy in a GPU server">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">GPU HBM → SM: ~768 GB/s (A6000) to 3.2 TB/s (H100)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">GPU ↔ GPU via NVLink: 112.5 GB/s bidirectional</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">GPU ↔ CPU via PCIe Gen4: 32 GB/s (16 lanes × 2 GB/s)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">CPU ↔ Host DRAM: 64–512 GB/s</span></li>
  </ol>
</div>

The 20–24× gap between NVLink and PCIe is why data-parallel training uses NVLink for AllReduce within a node and pays the PCIe penalty only when crossing nodes. It is also why `cudaMemcpy` between CPU and GPU is a bottleneck that kernel overlap (async copies) is designed to hide.

Compared to CPU, a GPU trades per-thread performance for massive parallelism:

| | AMD EPYC 9754 | NVIDIA A6000 |
|---|---|---|
| Cores / threads | 128 / 256 | 10,752 |
| Clock | 2.25 GHz | 1.8 GHz |
| Peak compute | 576 GFlops | 38.7 TFlops |
| Power | 360 W | 300 W |

The A6000 delivers **67× more compute** at slightly lower power by replacing a handful of complex out-of-order cores with thousands of simple in-order ones — a good trade when the workload is data-parallel and latency-insensitive.

### GPU Generations
{: #gpu-lineup}

NVIDIA's datacenter GPU lineup has scaled dramatically across three recent architectures:

| | Ampere (A100) | Hopper (H100) | Blackwell (B200) |
|---|---|---|---|
| FP32 | 19.5 TFLOPS | 67 TFLOPS | 75 TFLOPS |
| FP32 Tensor Core | 312 TFLOPS | 989 TFLOPS | 2,200 TFLOPS |
| FP16/BF16 Tensor Core | 624 TFLOPS | 1,979 TFLOPS | 4,500 TFLOPS |
| FP8 Tensor Core | — | 3,958 TFLOPS | 9,000 TFLOPS |
| FP4 Tensor Core | — | — | 18,000 TFLOPS |
| HBM capacity | 80 GB HBM2e | 80 GB HBM3 | 192 GB HBM3e |
| Memory bandwidth | 2 TB/s | 3.2 TB/s | 7.7 TB/s |
| SMs | 108 | 132 | — |

Three trends stand out. First, tensor core throughput scales much faster than scalar FP32 — the gap between "TFLOPS" and "Tensor Core TFLOPS" grows each generation, making layout alignment for tensor core access increasingly important. Second, lower-precision formats (FP8, FP4) multiply throughput further at the cost of quantisation noise — the system-level challenge is keeping that noise within acceptable bounds. Third, memory bandwidth has grown ~3.8× from A100 to B200 while capacity has grown 2.4×, reflecting that LLM inference is bandwidth-bound rather than compute-bound for most serving scenarios.

---

## GPU Architecture
{: #gpu-architecture}

### Streaming Multiprocessors (SMMs)
{: #smm}

A GPU is a collection of **Streaming Multiprocessors (SMs)**. Each SM is an independent processing unit that runs thread blocks assigned to it by the GPU scheduler. The A6000 has 84 SMs; the H100 has 132.

**SM internal structure (Ampere/A6000):**

Each SM is divided into **4 partitions**, each with 32 CUDA cores — giving 128 cores per SM total. Within a partition:

- **32 CUDA cores** — one FP32 (or INT32) operation per core per cycle
- **64 KB register file** — the fastest storage, private to each thread, lives here for the warp's entire lifetime
- **Warp scheduler** — selects a ready warp each cycle and issues its next instruction

All 4 partitions share a **128 KB L1/shared memory** (256 KB on H100). Shared memory is carved out of this pool at kernel launch — more shared memory per block means fewer blocks can co-reside.

<div class="post-flow" role="group" aria-label="SM resource hierarchy">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">SM — 4 partitions × 32 cores = 128 CUDA cores</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">64 KB register file per partition — fastest, private per warp</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">128 KB shared/L1 cache per SM (256 KB on H100)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Up to 64 resident warps (2048 threads) per SM</span></li>
  </ol>
</div>

**Warp scheduling**: the warp scheduler selects a warp with a ready instruction each cycle and issues it. Because execution context (PC, registers, shared memory) stays on the SM for the warp's lifetime, switching between warps is instant — zero overhead. This is how GPUs hide memory latency: while one warp waits on a global memory load (~200 cycles), the scheduler runs other warps. A block's resource footprint (registers per thread × thread count + shared memory) determines how many blocks co-reside on a single SM. If a block's demand exceeds available resources, it waits until a running block completes.

**GTX 980 → H100 evolution:**

| | GTX 980 (2014) | H100 (2022) |
|---|---|---|
| SMMs | 16 | 132 |
| Shared mem / SMM | 96 KB | 256 KB |
| Warps / SMM | 64 | 64 |
| Clock speed | 1064 MHz | 1110 MHz |
| Peak TFLOPS | 4.6 | ~1000 (with tensor cores) |

The 200× throughput gain from GTX 980 to H100 came almost entirely from **more SMMs** and **tensor cores**, not faster clocks.

### Tensor Cores
{: #tensor-cores}

<div class="post-flow post-flow--horizontal" role="group" aria-label="Compute units in an SMM">
  <ol class="post-flow__list post-flow__list--row">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">CUDA Cores (scalar FP32/INT32)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue post-flow__bar--accent">Tensor Cores (matrix MMA units)</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Special Function Units (sin, cos, rcp…)</span></li>
  </ol>
</div>

Tensor Cores perform a **4×4 matrix multiply-accumulate** (`D = A×B + C`) in a single instruction, operating on FP16/BF16 inputs and FP32 accumulators. They are the reason H100 peaks at ~1000 TFLOPS in mixed precision versus ~4.6 TFLOPS on scalar CUDA cores for GTX 980. Reaching tensor core throughput requires carefully shaped, aligned tiles — the exact shape that register + shared memory tiling produces.

---

## Case Study: Matrix Multiplication in CUDA
{: #matmul-cuda}

### Strawman → Register Tiling
{: #register-tiling}

**Strawman** — each thread computes one scalar `C[x][y]`:

```cuda
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float result = 0;
    for (int k = 0; k < N; ++k)
        result += A[x][k] * B[k][y];
    C[x][y] = result;
}
// Global memory accesses: 2N per thread × N² threads = 2N³ total
```

**Register tiling** — each thread computes a `V×V` submatrix, reusing loaded values `V` times:

```cuda
__global__ void mm_tiled(...) {
    float c[V][V] = {0};
    float a[V], b[V];
    for (int k = 0; k < N; ++k) {
        a[:] = A[xbase*V : xbase*V+V, k];   // loaded once, reused V times
        b[:] = B[k, ybase*V : ybase*V+V];
        for (int y = 0; y < V; ++y)
            for (int x = 0; x < V; ++x)
                c[x][y] += a[x] * b[y];
    }
    C[...] = c[:];
}
// Global memory accesses: 2NV per thread × N²/V² threads = 2N³/V total
```

<div class="post-flow post-flow--compare" role="group" aria-label="Strawman vs register tiled memory access">
  <div class="post-flow__col">
    <p class="post-flow__col-label">Strawman</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">1 element per thread</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">2N global loads per thread</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Total: 2N³ global loads</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">Register Tiled (V×V)</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">V×V elements per thread</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">2NV global loads per thread</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Total: 2N³/V global loads</span></li>
    </ol>
  </div>
</div>

### Shared Memory Tiling
{: #shared-mem-tiling}

Register tiling still reads from slow global memory every iteration. **Shared memory tiling** adds a second level: cooperatively load an `S×L` tile of A and B into fast shared memory, then run the register-tiled inner loop against it.

```cuda
__global__ void mm_smem(...) {
    __shared__ float sA[S][L], sB[S][L];
    float c[V][V] = {0};
    for (int ko = 0; ko < N; ko += S) {
        __syncthreads();
        // cooperative fetch: all threads in block load sA, sB together
        sA[:,:] = A[ko:ko+S, yblock*L : yblock*L+L];
        sB[:,:] = B[ko:ko+S, xblock*L : xblock*L+L];
        __syncthreads();
        for (int ki = 0; ki < S; ++ki) {
            a[:] = sA[ki, threadIdx.y*V : threadIdx.y*V+V];
            b[:] = sB[ki, threadIdx.x*V : threadIdx.x*V+V];
            for (int y = 0; y < V; ++y)
                for (int x = 0; x < V; ++x)
                    c[y][x] += a[y] * b[x];
        }
    }
}
```

Two `__syncthreads()` calls are required: one **before** loading (ensure previous iteration's reads are done) and one **after** (ensure all threads see the new tile before compute begins).

**Combined cost**: `2N³/L` global loads + `2N³/V` shared memory loads — an `L×` reduction in the expensive DRAM traffic.

---

## Case Study: Parallel Reduction
{: #parallel-reduction}

Parallel reduction — computing `Σ A[i]` over millions of elements — underlies softmax, layer norm, and many other ML ops. The naive tree reduction works within a block; scaling to multiple blocks requires careful design because **CUDA has no global synchronization between thread blocks**.

**Solution**: decompose into multiple kernel launches. Each launch is an implicit barrier; partial sums from the first kernel feed the second.

<div class="post-flow" role="group" aria-label="Parallel reduction pipeline">
  <ol class="post-flow__list">
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Kernel 1 — each block reduces its chunk into one partial sum</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--blue">Kernel launch = global sync barrier</span></li>
    <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Kernel 2 — reduce the array of partial sums to a single value</span></li>
  </ol>
</div>

### V1 — Interleaved Addressing (Divergent)
{: #reduction-v1}

```cuda
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0)          // ← divergent: half the warp is idle
        sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

Pattern of active threads across iterations:

```
s=1:  T F T F T F T F ...   (every other thread idle)
s=2:  T F F F T F F F ...
s=4:  T F F F F F F F ...
```

Highly divergent — later iterations leave most of each warp masked.

### V2 — Sequential Addressing (Coalesced + Non-divergent)
{: #reduction-v2}

<div class="post-flow post-flow--compare" role="group" aria-label="V1 interleaved vs V2 sequential reduction">
  <div class="post-flow__col">
    <p class="post-flow__col-label">V1 — Interleaved</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Strided memory access → non-coalesced</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">tid % (2*s) → divergent warps</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--muted">Suboptimal throughput</span></li>
    </ol>
  </div>
  <div class="post-flow__col">
    <p class="post-flow__col-label">V2 — Sequential</p>
    <ol class="post-flow__list">
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Threads access consecutive addresses → coalesced</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">threadIdx.x &lt; s → only first half active, no divergence within warp</span></li>
      <li class="post-flow__step"><span class="post-flow__bar post-flow__bar--green">Full memory bandwidth utilisation</span></li>
    </ol>
  </div>
</div>

```cuda
// V2: reversed loop, threadId-based index
for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
    if (threadIdx.x < s)
        sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
}
```

Active thread pattern per iteration:

```
s=blockDim/2:  T T T T ... T T F F ... F F   (first half active, coalesced)
s=blockDim/4:  T T T T ... F F F F ... F F
```

All active threads in each warp access **consecutive** addresses → fully coalesced reads → maximum memory bus utilisation.

**Coalesced access** means multiple threads in a warp read consecutive memory addresses so the GPU can satisfy all reads in a single memory transaction. Non-coalesced access triggers multiple transactions, wasting bandwidth.
