---
title: "Python & PyTorch: Basics to Expert"
date: 2026-04-21
display_order: 16
description: "A complete reference from Python fundamentals through advanced patterns, PyTorch from tensors to distributed training, GPU cluster workflows (SLURM, CUDA), environment management with Conda, and profiling tools — with code snippets and bash command explanations throughout."
tags: [python, pytorch, gpu, cuda, slurm, conda, deep-learning]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#python-basics">Python Basics</a>
      <ul class="post-toc-sublist">
        <li><a href="#data-types">Data Types & Collections</a></li>
        <li><a href="#control-flow">Control Flow</a></li>
        <li><a href="#functions">Functions</a></li>
        <li><a href="#oop">Object-Oriented Programming</a></li>
        <li><a href="#exceptions">Exceptions & Context Managers</a></li>
      </ul>
    </li>
    <li><a href="#python-intermediate">Python Intermediate</a>
      <ul class="post-toc-sublist">
        <li><a href="#comprehensions">Comprehensions & Generators</a></li>
        <li><a href="#itertools">itertools & functools</a></li>
        <li><a href="#decorators">Decorators</a></li>
        <li><a href="#dataclasses">Dataclasses & NamedTuples</a></li>
        <li><a href="#typing">Type Hints & Annotations</a></li>
        <li><a href="#concurrency">Concurrency: Threading, Multiprocessing, Async</a></li>
      </ul>
    </li>
    <li><a href="#python-expert">Python Expert</a>
      <ul class="post-toc-sublist">
        <li><a href="#metaclasses">Metaclasses & Descriptors</a></li>
        <li><a href="#memory">Memory Management & GC</a></li>
        <li><a href="#profiling-python">Profiling Python Code</a></li>
        <li><a href="#packaging">Packaging & Project Layout</a></li>
      </ul>
    </li>
    <li><a href="#numpy">NumPy Essentials</a></li>
    <li><a href="#pytorch-basics">PyTorch Basics</a>
      <ul class="post-toc-sublist">
        <li><a href="#tensors">Tensors</a></li>
        <li><a href="#autograd">Autograd</a></li>
        <li><a href="#nn-module">nn.Module</a></li>
        <li><a href="#datasets">Datasets & DataLoaders</a></li>
        <li><a href="#training-loop">Training Loop</a></li>
      </ul>
    </li>
    <li><a href="#pytorch-intermediate">PyTorch Intermediate</a>
      <ul class="post-toc-sublist">
        <li><a href="#custom-layers">Custom Layers & Loss Functions</a></li>
        <li><a href="#hooks">Hooks & Gradient Manipulation</a></li>
        <li><a href="#checkpointing">Checkpointing & Serialisation</a></li>
        <li><a href="#mixed-precision">Mixed Precision Training</a></li>
        <li><a href="#torch-compile">torch.compile</a></li>
      </ul>
    </li>
    <li><a href="#pytorch-expert">PyTorch Expert</a>
      <ul class="post-toc-sublist">
        <li><a href="#distributed">Distributed Training (DDP & FSDP)</a></li>
        <li><a href="#cuda-extensions">Custom CUDA Extensions</a></li>
        <li><a href="#torch-profiler">torch.profiler</a></li>
        <li><a href="#memory-opt">Memory Optimisation</a></li>
        <li><a href="#jit">TorchScript & JIT</a></li>
      </ul>
    </li>
    <li><a href="#gpu-cluster">GPU Cluster & SLURM</a>
      <ul class="post-toc-sublist">
        <li><a href="#slurm-basics">SLURM Basics</a></li>
        <li><a href="#sbatch">sbatch & Job Scripts</a></li>
        <li><a href="#squeue-sinfo">squeue, sinfo & Monitoring</a></li>
        <li><a href="#gpu-monitoring">GPU Monitoring</a></li>
      </ul>
    </li>
    <li><a href="#conda">Conda Environment Management</a></li>
    <li><a href="#bash-tools">Essential Bash for ML</a></li>
    <li><a href="#bash-general">General Bash & Shell Productivity</a>
      <ul class="post-toc-sublist">
        <li><a href="#bash-nav">Navigation & Files</a></li>
        <li><a href="#bash-text">Text Processing</a></li>
        <li><a href="#bash-scripting">Scripting Essentials</a></li>
        <li><a href="#bash-network">Network & Remote</a></li>
        <li><a href="#bash-git">Git Quick Reference</a></li>
      </ul>
    </li>
  </ul>
</nav>

---

## Python Basics
{: #python-basics}

### Data Types & Collections
{: #data-types}

Python's built-in types cover most needs. Understanding their performance characteristics matters at scale.

```python
# Numeric types
x: int = 42
y: float = 3.14
z: complex = 1 + 2j

# Strings — immutable sequences of Unicode code points
s = "hello"
s_raw = r"C:\Users\name"       # raw string — backslashes not escaped
s_f = f"value is {x:.2f}"     # f-string — evaluated at runtime
s_b = b"bytes"                 # bytes literal

# Lists — mutable, ordered, O(1) append, O(n) insert at front
lst = [1, 2, 3]
lst.append(4)          # O(1)
lst.insert(0, 0)       # O(n) — shifts all elements right
lst.pop()              # O(1) — removes last element
lst.pop(0)             # O(n) — removes first element (use deque instead)

# Tuples — immutable, faster than lists for fixed data
t = (1, 2, 3)
a, b, c = t            # unpacking
first, *rest = t       # star unpacking

# Dicts — hash map, O(1) average lookup/insert/delete
# Ordered by insertion since Python 3.7
d = {"a": 1, "b": 2}
d.get("c", 0)          # default value on missing key
d.setdefault("c", []).append(1)   # initialise if missing

# Sets — hash set, O(1) membership, unordered
s1 = {1, 2, 3}
s2 = {2, 3, 4}
s1 & s2   # intersection: {2, 3}
s1 | s2   # union: {1, 2, 3, 4}
s1 - s2   # difference: {1}

# collections — specialised containers
from collections import defaultdict, Counter, deque, OrderedDict

counter = Counter("abracadabra")
# Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})
counter.most_common(2)   # [('a', 5), ('b', 2)]

dd = defaultdict(list)
dd["key"].append(1)      # no KeyError on missing key

dq = deque([1, 2, 3], maxlen=5)
dq.appendleft(0)   # O(1) — deque supports O(1) at both ends
dq.popleft()       # O(1)
```

**Complexity cheat sheet:**

| Operation | list | dict | set | deque |
|---|---|---|---|---|
| Append / insert end | O(1) amortised | — | — | O(1) |
| Insert front | O(n) | — | — | O(1) |
| Lookup by index | O(1) | — | — | O(n) |
| Lookup by key | — | O(1) avg | O(1) avg | — |
| Delete by index | O(n) | — | — | O(n) |
| Delete by key | — | O(1) avg | O(1) avg | — |

### Control Flow
{: #control-flow}

```python
# Walrus operator := — assign and test in one expression (Python 3.8+)
import re
if m := re.match(r"(\d+)", "42abc"):
    print(m.group(1))   # prints "42"

# Match statement (structural pattern matching, Python 3.10+)
def classify(point):
    match point:
        case (0, 0):
            return "origin"
        case (x, 0):
            return f"x-axis at {x}"
        case (0, y):
            return f"y-axis at {y}"
        case (x, y):
            return f"point ({x}, {y})"
        case _:
            return "not a point"

# for-else: else runs if loop completes without break
for i in range(10):
    if i == 5:
        break
else:
    print("no break hit")   # not printed here
```

### Functions
{: #functions}

```python
# Positional, keyword, *args, **kwargs, keyword-only
def func(a, b, /, c, *, d, **kwargs):
    # a, b: positional-only (before /)
    # c: positional or keyword
    # d: keyword-only (after *)
    # kwargs: extra keyword args
    pass

# Closures — inner function captures enclosing scope by reference
def make_counter(start=0):
    count = start
    def counter():
        nonlocal count
        count += 1
        return count
    return counter

c = make_counter(10)
c()   # 11
c()   # 12

# Lambda — anonymous single-expression function
square = lambda x: x ** 2
sorted([(1, 'b'), (2, 'a')], key=lambda t: t[1])

# functools.partial — fix some arguments
from functools import partial
def power(base, exp): return base ** exp
square = partial(power, exp=2)
square(3)   # 9

# Annotations — no runtime enforcement without a library
def add(x: int, y: int) -> int:
    return x + y
```

### Object-Oriented Programming
{: #oop}

```python
class Animal:
    # Class variable — shared across all instances
    kingdom = "Animalia"

    def __init__(self, name: str, sound: str):
        self.name = name       # instance variable
        self._sound = sound    # convention: protected
        self.__id = id(self)   # name-mangled to _Animal__id

    def speak(self) -> str:
        return f"{self.name} says {self._sound}"

    @classmethod
    def from_dict(cls, d: dict) -> "Animal":
        return cls(d["name"], d["sound"])

    @staticmethod
    def is_valid_name(name: str) -> bool:
        return bool(name.strip())

    def __repr__(self) -> str:
        return f"Animal(name={self.name!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Animal) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Dog(Animal):
    def __init__(self, name: str):
        super().__init__(name, "woof")

    def speak(self) -> str:          # override
        return super().speak() + "!"

    def fetch(self) -> str:
        return f"{self.name} fetches!"


# Abstract base classes
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float: ...

    @abstractmethod
    def perimeter(self) -> float: ...

class Circle(Shape):
    def __init__(self, r: float):
        self.r = r
    def area(self) -> float:
        import math; return math.pi * self.r ** 2
    def perimeter(self) -> float:
        import math; return 2 * math.pi * self.r

# Dunder (magic) methods
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    def __rmul__(self, scalar):      # scalar * vector
        return self.__mul__(scalar)
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        return (self.x, self.y)[idx]
    def __iter__(self):
        yield self.x; yield self.y
```

### Exceptions & Context Managers
{: #exceptions}

```python
# Exception hierarchy — inherit from Exception, not BaseException
class ModelError(Exception):
    """Base for model-related errors."""

class ShapeError(ModelError):
    def __init__(self, expected, got):
        super().__init__(f"expected {expected}, got {got}")
        self.expected = expected
        self.got = got

# try / except / else / finally
try:
    result = risky_op()
except (ValueError, TypeError) as e:
    logger.error(f"Input error: {e}")
    raise   # re-raise
except Exception as e:
    logger.critical(f"Unexpected: {e}", exc_info=True)
    raise RuntimeError("wrapped") from e   # exception chaining
else:
    # runs only if no exception
    process(result)
finally:
    # always runs — cleanup
    cleanup()

# Context managers with contextlib
from contextlib import contextmanager

@contextmanager
def timer(label: str):
    import time
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.4f}s")

with timer("forward pass"):
    output = model(x)

# __enter__ / __exit__ protocol
class ManagedResource:
    def __enter__(self):
        self.acquire()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False   # don't suppress exceptions
```

---

## Python Intermediate
{: #python-intermediate}

### Comprehensions & Generators
{: #comprehensions}

```python
# List comprehension — creates a full list in memory
squares = [x**2 for x in range(10) if x % 2 == 0]

# Dict / set comprehensions
inv = {v: k for k, v in d.items()}
evens = {x for x in range(20) if x % 2 == 0}

# Generator expression — lazy; yields one item at a time; O(1) memory
gen = (x**2 for x in range(10**9))   # no memory allocation yet
next(gen)   # 0 — pull one item

# Generator function — uses yield
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def read_chunks(path, chunk_size=8192):
    """Memory-efficient file reader."""
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            yield chunk

# yield from — delegates to sub-generator
def chain(*iterables):
    for it in iterables:
        yield from it

# send() — bidirectional generator (coroutine pattern)
def accumulator():
    total = 0
    while True:
        value = yield total
        if value is None:
            break
        total += value

acc = accumulator()
next(acc)       # prime the generator
acc.send(10)    # 10
acc.send(5)     # 15
```

### itertools & functools
{: #itertools}

```python
import itertools, functools

# itertools — building blocks for efficient looping
list(itertools.chain([1,2], [3,4]))        # [1, 2, 3, 4]
list(itertools.islice(range(100), 5))      # [0, 1, 2, 3, 4]
list(itertools.combinations([1,2,3], 2))   # [(1,2),(1,3),(2,3)]
list(itertools.permutations("AB"))         # [('A','B'),('B','A')]
list(itertools.product([0,1], repeat=3))   # all 3-bit strings
list(itertools.groupby("AAABBBCC", key=lambda x: x))
# [('A', iter), ('B', iter), ('C', iter)]

# accumulate — running total / scan
list(itertools.accumulate([1,2,3,4], lambda a,b: a*b))  # [1,2,6,24]

# functools
functools.reduce(lambda a,b: a+b, [1,2,3,4])   # 10

@functools.lru_cache(maxsize=None)
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)

@functools.cached_property   # lazy, computed once per instance
class Circle:
    def __init__(self, r): self.r = r
    @functools.cached_property
    def area(self): return 3.14159 * self.r ** 2

functools.total_ordering   # fill in comparison methods from __eq__ + one of lt/le/gt/ge
```

### Decorators
{: #decorators}

```python
import functools, time

# Basic decorator
def timer(func):
    @functools.wraps(func)   # preserves __name__, __doc__
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.perf_counter()-start:.4f}s")
        return result
    return wrapper

# Decorator with arguments — three levels of nesting
def retry(max_tries=3, exceptions=(Exception,)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_tries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_tries - 1:
                        raise
                    time.sleep(2 ** attempt)   # exponential backoff
        return wrapper
    return decorator

@retry(max_tries=5, exceptions=(ConnectionError,))
def fetch_data(url): ...

# Class-based decorator
class Cached:
    def __init__(self, func):
        self.func = func
        self.cache = {}
        functools.update_wrapper(self, func)

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]

# Descriptor-based property (how @property works internally)
class Property:
    def __init__(self, fget): self.fget = fget
    def __get__(self, obj, objtype=None):
        if obj is None: return self
        return self.fget(obj)
```

### Dataclasses & NamedTuples
{: #dataclasses}

```python
from dataclasses import dataclass, field
from typing import NamedTuple

@dataclass
class TrainingConfig:
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    tags: list = field(default_factory=list)   # mutable default
    _checksum: str = field(default="", repr=False, compare=False)

    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

# frozen=True makes it immutable and hashable
@dataclass(frozen=True, order=True)
class Point:
    x: float
    y: float

# NamedTuple — immutable, lighter than dataclass, tuple-compatible
class Token(NamedTuple):
    id: int
    text: str
    logprob: float = 0.0

t = Token(42, "hello")
t.id, t.text   # attribute access
t[0], t[1]     # index access (it's a tuple)
```

### Type Hints & Annotations
{: #typing}

```python
from typing import (
    Any, Union, Optional, Literal,
    List, Dict, Tuple, Set,
    Callable, Iterator, Generator,
    TypeVar, Generic, Protocol,
    overload
)

# Modern syntax (Python 3.10+): use | instead of Union
def process(x: int | str) -> str: ...

# Optional[X] == X | None
def find(key: str) -> Optional[int]: ...

# TypeVar for generic functions
T = TypeVar("T")
def first(lst: list[T]) -> T:
    return lst[0]

# Protocol — structural subtyping (duck typing)
class Drawable(Protocol):
    def draw(self) -> None: ...

def render(obj: Drawable) -> None:
    obj.draw()

# Generic classes
class Stack(Generic[T]):
    def __init__(self): self._items: list[T] = []
    def push(self, item: T) -> None: self._items.append(item)
    def pop(self) -> T: return self._items.pop()

# Callable[[arg_types], return_type]
Transform = Callable[[torch.Tensor], torch.Tensor]

# Literal — restricts to specific values
Mode = Literal["train", "eval", "test"]
def set_mode(mode: Mode) -> None: ...
```

### Concurrency: Threading, Multiprocessing, Async
{: #concurrency}

```python
import threading, multiprocessing, asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Threading — I/O-bound; GIL limits CPU parallelism
def download(url):
    import urllib.request
    with urllib.request.urlopen(url) as r:
        return r.read()

with ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(download, url) for url in urls]
    results = [f.result() for f in futures]

# Multiprocessing — CPU-bound; bypasses GIL; separate memory
def cpu_task(n): return sum(i**2 for i in range(n))

with ProcessPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(cpu_task, [10**6]*4))

# multiprocessing.Pool for more control
with multiprocessing.Pool(processes=4) as pool:
    results = pool.starmap(func, [(a, b) for a, b in pairs])

# asyncio — async I/O; single thread; event loop
async def fetch(session, url):
    async with session.get(url) as resp:
        return await resp.text()

async def main(urls):
    import aiohttp
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)

asyncio.run(main(urls))

# asyncio.Queue for producer-consumer
async def producer(queue: asyncio.Queue):
    for item in data:
        await queue.put(item)
    await queue.put(None)   # sentinel

async def consumer(queue: asyncio.Queue):
    while (item := await queue.get()) is not None:
        await process(item)
        queue.task_done()
```

**Choosing the right concurrency model:**

| Scenario | Tool | Why |
|---|---|---|
| I/O-bound, many requests | asyncio | Single thread; no overhead |
| I/O-bound, existing sync code | ThreadPoolExecutor | Threads handle blocking I/O |
| CPU-bound, pure Python | ProcessPoolExecutor | Bypasses GIL |
| CPU-bound, NumPy/PyTorch | Release GIL or use GPU | C extensions often release GIL |

---

## Python Expert
{: #python-expert}

### Metaclasses & Descriptors
{: #metaclasses}

```python
# Descriptor protocol — __get__, __set__, __delete__
class Validated:
    """Descriptor that validates a numeric range."""
    def __set_name__(self, owner, name):
        self.name = name
        self.private = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None: return self
        return getattr(obj, self.private, None)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be numeric")
        setattr(obj, self.private, value)

class Model:
    lr = Validated()
    dropout = Validated()

m = Model()
m.lr = 1e-3      # calls Validated.__set__
m.lr             # calls Validated.__get__

# Metaclass — controls class creation
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Config(metaclass=SingletonMeta):
    def __init__(self):
        self.lr = 1e-3

Config() is Config()   # True — same instance

# __init_subclass__ — hook called when a subclass is created
class Registry:
    _registry = {}
    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name:
            Registry._registry[name] = cls

class ModelA(Registry, name="model_a"): ...
class ModelB(Registry, name="model_b"): ...

Registry._registry["model_a"]   # ModelA
```

### Memory Management & GC
{: #memory}

```python
import sys, gc, weakref

# Reference counting — primary memory management
x = [1, 2, 3]
sys.getrefcount(x)   # 2 (x + argument to getrefcount)

# Circular references — not freed by refcount; GC handles them
class Node:
    def __init__(self): self.ref = None

a, b = Node(), Node()
a.ref = b; b.ref = a   # cycle; refcount never reaches 0

gc.collect()   # manually trigger cyclic GC

# weakref — reference that doesn't prevent garbage collection
cache = {}
def get_obj(key):
    ref = cache.get(key)
    if ref is not None:
        obj = ref()    # dereference weak reference
        if obj is not None:
            return obj
    obj = ExpensiveObj(key)
    cache[key] = weakref.ref(obj)
    return obj

# __slots__ — skip per-instance __dict__; saves ~30-50% memory for small objects
class Point:
    __slots__ = ("x", "y")
    def __init__(self, x, y):
        self.x, self.y = x, y

# sys.getsizeof — shallow size in bytes
sys.getsizeof([])       # 56
sys.getsizeof({})       # 232
sys.getsizeof(Point(1,2))   # 48 with __slots__
```

### Profiling Python Code
{: #profiling-python}

```bash
# cProfile — deterministic profiler; measures function call times
python -m cProfile -s cumulative script.py
# -s cumulative: sort by cumulative time (includes subcalls)
# -s tottime:    sort by total time in function only

# Save profile output for later analysis
python -m cProfile -o profile.out script.py
python -c "import pstats; p = pstats.Stats('profile.out'); p.sort_stats('cumulative'); p.print_stats(20)"

# line_profiler — per-line timing (install: pip install line_profiler)
# Decorate functions you want to profile with @profile, then:
kernprof -l -v script.py
# -l: line-by-line mode; -v: print results immediately

# memory_profiler — per-line memory usage (pip install memory_profiler)
python -m memory_profiler script.py

# py-spy — sampling profiler; attaches to running process; zero overhead
py-spy top --pid 12345          # live top-like view
py-spy record -o profile.svg --pid 12345   # flame graph

# timeit — micro-benchmarking in IPython/Jupyter
%timeit sorted(lst)
%timeit -n 1000 -r 5 sorted(lst)   # 1000 loops, 5 repetitions
```

```python
# timeit in scripts
import timeit
timeit.timeit("sorted(lst)", setup="lst=list(range(1000))", number=10000)

# contextlib.contextmanager timer
from contextlib import contextmanager
import time

@contextmanager
def profile_block(name):
    start = time.perf_counter()
    yield
    print(f"{name}: {(time.perf_counter()-start)*1000:.2f}ms")

with profile_block("embedding lookup"):
    emb = embedding_layer(input_ids)
```

### Packaging & Project Layout
{: #packaging}

```
myproject/
├── pyproject.toml       # PEP 517/518: build system + metadata
├── src/
│   └── myproject/
│       ├── __init__.py
│       ├── model.py
│       └── utils.py
├── tests/
│   ├── conftest.py      # pytest fixtures
│   └── test_model.py
├── README.md
└── .github/workflows/ci.yml
```

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "myproject"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["torch>=2.0", "numpy>=1.24"]

[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy"]
```

```bash
pip install -e ".[dev]"    # editable install with dev extras
ruff check src/            # fast linter (replaces flake8/pylint)
mypy src/                  # static type checking
pytest tests/ -v --tb=short
```

---

## NumPy Essentials
{: #numpy}

```python
import numpy as np

# Array creation
a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.zeros((3, 4), dtype=np.float64)
c = np.random.randn(100, 50)    # standard normal
d = np.arange(0, 10, 0.5)      # like range() but float
e = np.linspace(0, 1, 100)     # 100 evenly spaced points

# Indexing and slicing
a[0]          # element
a[1:3]        # slice
a[::2]        # every 2nd element
c[:, 0]       # first column
c[c > 0]      # boolean indexing — returns 1D array of matches

# Broadcasting — operations between arrays of compatible shapes
# Shape rules: align trailing dims, size must be equal or 1
x = np.ones((3, 1))
y = np.ones((1, 4))
(x + y).shape    # (3, 4) — broadcast both

# Universal functions (ufuncs) — vectorised, C speed
np.exp(a); np.log(a); np.sqrt(a)
np.maximum(a, b)          # element-wise max
np.dot(c, c.T)            # matrix multiply: (100,50)@(50,100) -> (100,100)
c @ c.T                   # same; @ operator since Python 3.5

# Axis operations
c.sum(axis=0)    # sum over rows -> shape (50,)
c.mean(axis=1)   # mean over columns -> shape (100,)
c.argmax(axis=1) # index of max in each row -> shape (100,)

# Reshape and view vs copy
c.reshape(50, 100)     # view if possible (contiguous), else copy
c.ravel()              # flatten to 1D view
c.T                    # transpose — view, not copy
np.ascontiguousarray(c.T)  # make contiguous copy (needed for some ops)

# Fancy indexing
idx = np.array([0, 2, 4])
c[idx]               # rows 0, 2, 4 — returns copy, not view
c[np.ix_([0,1],[2,3])]  # outer indexing: rows 0,1 x cols 2,3

# Structured arrays
dt = np.dtype([("name", "U10"), ("score", "f4")])
records = np.array([("alice", 0.95), ("bob", 0.87)], dtype=dt)
records["score"]     # array([0.95, 0.87])
```

---

## PyTorch Basics
{: #pytorch-basics}

### Tensors
{: #tensors}

```python
import torch

# Creation
x = torch.tensor([1.0, 2.0, 3.0])
x = torch.zeros(3, 4)
x = torch.randn(2, 3, 4)           # standard normal
x = torch.arange(0, 10, dtype=torch.float32)
x = torch.from_numpy(np_array)     # shares memory if on CPU

# Device management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
x = x.cuda()           # shorthand; deprecated in favour of .to()
x = x.cpu()            # move back to CPU

# Dtype
x = x.float()          # float32
x = x.half()           # float16
x = x.bfloat16()       # bfloat16 (brain float — wider exponent range)
x.dtype                # torch.float32

# Shape operations
x.shape                 # torch.Size([2, 3, 4])
x.reshape(6, 4)         # may copy
x.view(6, 4)            # never copies; errors if not contiguous
x.contiguous().view(6, 4)  # safe
x.permute(2, 0, 1)      # reorder dims; (4, 2, 3)
x.unsqueeze(0)          # add dim at position 0: (1, 2, 3, 4)
x.squeeze(0)            # remove dim of size 1 at position 0
x.expand(5, 2, 3, 4)    # broadcast without copying data
x.repeat(2, 1, 1, 1)    # actually copies data

# Indexing — same syntax as numpy
x[0]                    # first element along dim 0
x[:, :, 1]              # slice
x[x > 0]               # boolean mask
torch.where(x > 0, x, torch.zeros_like(x))  # conditional

# Math
torch.matmul(a, b)      # @ operator; handles batched matmul
torch.bmm(a, b)         # batch matmul: (B, n, m) @ (B, m, k) -> (B, n, k)
torch.einsum("bik,bkj->bij", a, b)   # Einstein summation
torch.cat([a, b], dim=0)   # concatenate along dim
torch.stack([a, b], dim=0) # new dim
torch.chunk(x, 4, dim=0)   # split into 4 equal pieces

# In-place operations — trailing underscore
x.add_(1)    # adds 1 in-place; cannot be used with autograd graphs
x.zero_()    # zero in-place
```

### Autograd
{: #autograd}

```python
# Leaf tensors with requires_grad=True track operations
x = torch.randn(3, requires_grad=True)
y = x ** 2 + 2 * x + 1    # builds computation graph
loss = y.sum()
loss.backward()             # computes gradients via backprop
x.grad                      # d(loss)/dx = 2x + 2

# gradient accumulation — grads accumulate by default
optimizer.zero_grad()      # must call before each backward pass

# Detach from graph — stops gradient tracking
x_detached = x.detach()    # shares storage, no grad
with torch.no_grad():      # context manager; no graph built
    val = model(x)

# .grad_fn shows the last operation in the graph
y = x * 2
y.grad_fn                  # <MulBackward0>

# retain_graph=True — keep graph after backward (multiple losses)
loss1.backward(retain_graph=True)
loss2.backward()

# Gradient hooks — inspect or modify gradients mid-backward
def hook(grad):
    print(f"grad norm: {grad.norm():.4f}")
    return grad * 0.1   # scale gradients down

x.register_hook(hook)

# Custom autograd function
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sig = torch.sigmoid(x)
        return grad_output * (sig + x * sig * (1 - sig))

swish = Swish.apply
```

### nn.Module
{: #nn-module}

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Module inspection
model = MLP(128, 512, 10)
print(model)                          # pretty-print architecture
sum(p.numel() for p in model.parameters())          # total params
sum(p.numel() for p in model.parameters() if p.requires_grad)  # trainable

model.state_dict()                    # OrderedDict of parameter tensors
list(model.named_parameters())        # [(name, param), ...]
list(model.named_modules())           # all sub-modules recursively

model.train()   # sets training mode (enables dropout, batch norm stats)
model.eval()    # sets eval mode (disables dropout, freezes BN)

# Freeze / unfreeze parameters
for param in model.parameters():
    param.requires_grad = False

# Common layers
nn.Linear(in, out, bias=True)
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True)
nn.LayerNorm(normalized_shape)
nn.BatchNorm1d(num_features)
nn.Embedding(num_embeddings, embedding_dim)
nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
nn.Transformer(d_model=512, nhead=8)
```

### Datasets & DataLoaders
{: #datasets}

```python
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

dataset = TextDataset(texts, labels, tokenizer)
train_ds, val_ds = random_split(dataset, [0.9, 0.1])

loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,        # parallel data loading processes
    pin_memory=True,      # faster CPU->GPU transfer (pinned memory)
    prefetch_factor=2,    # prefetch batches per worker
    persistent_workers=True,  # keep workers alive between epochs
    drop_last=True,       # drop incomplete last batch
)

# Custom collate function for variable-length sequences
def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    return {"input_ids": padded}
```

### Training Loop
{: #training-loop}

```python
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        inputs = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()

        # Mixed precision forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs)
            loss = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0

    for batch in loader:
        inputs = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inputs)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


# Full training loop
model = MLP(128, 512, 10).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = torch.cuda.amp.GradScaler()  # for FP16

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")
```

---

## PyTorch Intermediate
{: #pytorch-intermediate}

### Custom Layers & Loss Functions
{: #custom-layers}

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, H, T, d_k)
        q, k, v = qkv.unbind(0)

        scale = self.d_k ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale   # (B, H, T, T)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


# Custom loss
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean": return focal.mean()
        if self.reduction == "sum": return focal.sum()
        return focal
```

### Hooks & Gradient Manipulation
{: #hooks}

```python
# Forward hook — inspect activations
activation_cache = {}

def save_activation(name):
    def hook(module, input, output):
        activation_cache[name] = output.detach()
    return hook

for name, module in model.named_modules():
    if isinstance(module, nn.ReLU):
        module.register_forward_hook(save_activation(name))

# Backward hook — inspect/modify gradients
grad_cache = {}

def save_gradient(name):
    def hook(module, grad_input, grad_output):
        grad_cache[name] = grad_output[0].detach()
    return hook

model.layer1.register_full_backward_hook(save_gradient("layer1"))

# Gradient clipping by value or norm
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# Gradient accumulation — simulate larger batch sizes
accumulation_steps = 4
optimizer.zero_grad()
for i, batch in enumerate(loader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

### Checkpointing & Serialisation
{: #checkpointing}

```python
# Save checkpoint
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, path)

# Load checkpoint
def load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"]

# Inference only — just the model weights
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

# Gradient checkpointing — trades compute for memory
from torch.utils.checkpoint import checkpoint

class TransformerLayer(nn.Module):
    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def _forward(self, x):
        return self.mlp(self.attn(x))
```

### Mixed Precision Training
{: #mixed-precision}

```python
# BF16 autocast (preferred on Ampere+ GPUs — wider exponent, no overflow risk)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

# FP16 requires GradScaler to handle underflow
scaler = torch.cuda.amp.GradScaler(
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)

optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(x)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)               # unscale before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()

# Check if scaler is healthy
scaler.get_scale()    # current loss scale value
```

### torch.compile
{: #torch-compile}

```python
# torch.compile (PyTorch 2.0+) — JIT compiles the model via Triton kernels
model = torch.compile(
    model,
    mode="reduce-overhead",  # "default" | "reduce-overhead" | "max-autotune"
    fullgraph=False,          # True: error if graph breaks; False: fall back to eager
    dynamic=False,            # True: handle dynamic shapes (slower but more flexible)
    backend="inductor",       # default; alternatives: "aot_eager", "cudagraphs"
)

# mode guide:
# "default"        — balanced; good for most cases
# "reduce-overhead"— minimise kernel launch overhead; best for small batches
# "max-autotune"   — exhaustive kernel search; slowest compile, fastest runtime

# Inspect compilation
import torch._dynamo
torch._dynamo.explain(model)(x)   # show what gets compiled vs. falls back

# Reset compiled state
torch._dynamo.reset()
```

---

## PyTorch Expert
{: #pytorch-expert}

### Distributed Training (DDP & FSDP)
{: #distributed}

```python
# DistributedDataParallel (DDP) — replicate model across GPUs; sync gradients
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",         # NCCL for GPU; Gloo for CPU
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # important: ensures different shuffling per epoch
        for batch in loader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    cleanup()

# Launch with torchrun (replaces torch.multiprocessing.spawn)
# torchrun --nproc_per_node=4 train.py

# FSDP — Fully Sharded Data Parallel; shards parameters across GPUs
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},
)

mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)

model = FSDP(
    model,
    auto_wrap_policy=policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    mixed_precision=mixed_precision_policy,
    device_id=torch.cuda.current_device(),
)

# Save FSDP checkpoint
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
):
    state_dict = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state_dict, "model.pt")
```

```bash
# Launch distributed training
torchrun \
  --nproc_per_node=8 \       # 8 GPUs per node
  --nnodes=2 \               # 2 nodes total
  --node_rank=0 \            # rank of this node (0 or 1)
  --master_addr=10.0.0.1 \   # IP of rank-0 node
  --master_port=29500 \
  train.py

# Explanation:
# nproc_per_node: one process per GPU
# nnodes: total number of machines
# node_rank: 0 for the master node, 1,2,... for workers
# master_addr/port: rendezvous point for all processes to find each other
```

### Custom CUDA Extensions
{: #cuda-extensions}

```python
# Option 1: inline CUDA with torch.utils.cpp_extension.load_inline
from torch.utils.cpp_extension import load_inline

cuda_source = """
__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int n = a.numel();
    add_kernel<<<(n+255)/256, 256>>>(a.data_ptr<float>(),
                                      b.data_ptr<float>(),
                                      c.data_ptr<float>(), n);
    return c;
}
"""

ext = load_inline(name="add_ext", cuda_sources=[cuda_source],
                  functions=["add_tensors"], verbose=True)
c = ext.add_tensors(a, b)

# Option 2: Triton kernels (Python-based GPU programming)
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add_triton(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out
```

### torch.profiler
{: #torch-profiler}

```python
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# Basic profiling
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,        # record tensor shapes
    profile_memory=True,       # track memory allocations
    with_stack=True,           # include Python call stack
) as prof:
    with record_function("forward"):
        output = model(input)
    with record_function("backward"):
        output.sum().backward()

# Print top 10 ops by CUDA time
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=10
))

# Export to Chrome trace (open at chrome://tracing)
prof.export_chrome_trace("trace.json")

# Export to TensorBoard
from torch.profiler import tensorboard_trace_handler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch)
        prof.step()   # advance scheduler

# Key metrics in profiler output:
# cpu_time_total:  time spent in CPU ops (including kernel launches)
# cuda_time_total: time spent in CUDA kernels (actual GPU time)
# self_cpu_time:   time in this op, excluding subcalls
# cpu_memory_usage / cuda_memory_usage: allocation deltas
```

```bash
# Run TensorBoard to visualise profiler output
tensorboard --logdir=./log/profiler
# Then open http://localhost:6006 and go to "PyTorch Profiler" tab

# nvprof — NVIDIA profiler (older; prefer nsys for new workflows)
nvprof --print-gpu-trace python train.py

# nsys (Nsight Systems) — timeline profiler
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=profile \
  python train.py
nsys-ui profile.nsys-rep   # open GUI

# ncu (Nsight Compute) — kernel-level profiler
ncu --set full \
    --target-processes all \
    -o kernel_profile \
    python train.py
ncu-ui kernel_profile.ncu-rep
```

### Memory Optimisation
{: #memory-opt}

```python
# Check memory usage
torch.cuda.memory_allocated()     # bytes currently allocated
torch.cuda.max_memory_allocated()  # peak allocation since last reset
torch.cuda.memory_reserved()      # bytes in PyTorch's cache (not all allocated)
torch.cuda.reset_peak_memory_stats()

# Memory snapshot (PyTorch 2.1+)
torch.cuda.memory._record_memory_history()
# ... run your code ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
# Upload to https://pytorch.org/memory_viz to visualise

# Empty cache — release cached but unallocated memory back to OS
torch.cuda.empty_cache()

# Activation checkpointing — recompute activations during backward
from torch.utils.checkpoint import checkpoint_sequential

model_layers = [layer1, layer2, ..., layerN]
output = checkpoint_sequential(model_layers, segments=4, input=x)
# Divides layers into 4 segments; only keeps segment boundaries in memory

# Offloading — move inactive tensors to CPU
class CPUOffloadOptimiser:
    """Offload optimiser states to CPU (ZeRO-3 style)."""
    # Use fairscale or DeepSpeed for production; concept only here
    pass

# Flash Attention — O(N) memory instead of O(N²) for attention
# PyTorch 2.0+ uses Flash Attention automatically via F.scaled_dot_product_attention
output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0 if not training else dropout_p,
    is_causal=True,       # causal mask for autoregressive models
)
```

### TorchScript & JIT
{: #jit}

```python
# torch.jit.script — compile Python to TorchScript IR
@torch.jit.script
def fused_op(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.relu(x * scale + 1.0)

# Script a module
scripted = torch.jit.script(model)
scripted.save("model_scripted.pt")
loaded = torch.jit.load("model_scripted.pt")

# torch.jit.trace — record ops from a single example input (no control flow)
traced = torch.jit.trace(model, example_input)

# TorchScript restrictions:
# - No Python dynamic features (arbitrary dicts, *args in loops, etc.)
# - Type annotations required
# - Limited Python stdlib

# Export to ONNX for cross-framework deployment
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
```

---

## GPU Cluster & SLURM
{: #gpu-cluster}

### SLURM Basics
{: #slurm-basics}

SLURM (Simple Linux Utility for Resource Management) is the job scheduler on most HPC/GPU clusters. You submit jobs to a queue; SLURM allocates nodes and GPUs, runs your script, and returns output.

**Core concepts:**

| Term | Meaning |
|---|---|
| **Job** | A unit of work submitted to SLURM |
| **Node** | A physical machine with CPUs, memory, and optionally GPUs |
| **Partition** | A logical group of nodes (e.g., `gpu`, `cpu`, `debug`) |
| **QOS** | Quality of Service — limits on priority, runtime, resources |
| **Account** | Billing/quota group; your PI's allocation |
| **GRES** | Generic Resource Scheduling — how GPUs are requested |

### sbatch & Job Scripts
{: #sbatch}

```bash
# Submit a job script
sbatch job.sh
# Returns: Submitted batch job 123456
# 123456 is your job ID — use it for monitoring and cancellation

# Submit with overrides (override script directives)
sbatch --partition=gpu --gres=gpu:4 job.sh

# Submit interactively (allocate resources and drop into shell)
salloc --partition=gpu --gres=gpu:2 --time=2:00:00 --mem=32G
# This allocates 2 GPUs for 2 hours — you get a shell on the compute node

# Interactive job that runs a command directly
srun --partition=gpu --gres=gpu:1 --pty bash
```

```bash
#!/bin/bash
#SBATCH --job-name=train_bert          # name shown in squeue
#SBATCH --partition=gpu                # which partition to run on
#SBATCH --nodes=2                      # number of nodes
#SBATCH --ntasks-per-node=1            # one process per node (torchrun handles the rest)
#SBATCH --gres=gpu:a100:8              # 8 A100 GPUs per node
#SBATCH --cpus-per-task=32             # CPU cores per task (for DataLoader workers)
#SBATCH --mem=256G                     # total memory per node
#SBATCH --time=24:00:00               # wall clock limit (HH:MM:SS)
#SBATCH --output=logs/%j_stdout.log    # %j = job ID
#SBATCH --error=logs/%j_stderr.log
#SBATCH --account=my_project           # billing account / PI group
#SBATCH --mail-type=END,FAIL           # email on completion or failure
#SBATCH --mail-user=you@nyu.edu

# --nodes: physical machines; --ntasks-per-node: processes per machine
# With torchrun we use 1 task per node and torchrun spawns nproc_per_node processes
# --gres=gpu:a100:8 requests 8 A100 GPUs specifically; use gpu:8 for any GPU type
# --cpus-per-task=32: gives 32 CPU cores to your process (for DataLoader num_workers)
# --mem is per node; --mem-per-cpu is per CPU core

# Load modules (cluster-specific software management)
module purge                    # clear any pre-loaded modules
module load cuda/12.1           # load CUDA toolkit
module load python/3.11

# Activate conda environment
source activate myenv
# or: conda activate myenv (if conda is initialised in .bashrc)

# Export distributed training env vars
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# scontrol show hostnames: expands SLURM node list (e.g., "node[01-02]") to individual hostnames
# head -n 1: take the first node as the master
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 8))   # total GPUs = nodes * GPUs per node

# Run with srun (executes one task per node as specified by ntasks-per-node)
srun torchrun \
  --nproc_per_node=8 \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py \
    --batch_size=64 \
    --lr=1e-4 \
    --epochs=10

# SLURM_NNODES: number of allocated nodes (set by SLURM)
# SLURM_NODEID: rank of this node (0, 1, 2, ...)
# SLURM_JOB_ID: job ID
# SLURM_JOB_NODELIST: list of allocated node names
```

### squeue, sinfo & Monitoring
{: #squeue-sinfo}

```bash
# squeue — show jobs in the queue
squeue                           # all jobs on the cluster
squeue -u $USER                  # only your jobs
squeue -u $USER --format="%.18i %.9P %.30j %.8T %.12M %.6D %R"
# Format fields: %i=jobID, %P=partition, %j=name, %T=state, %M=runtime, %D=nodes, %R=reason
# States: PD=pending, R=running, CG=completing, F=failed

squeue -j 123456                 # info about specific job
squeue -u $USER -t R             # only running jobs
squeue --start -j 123456         # estimated start time for pending job

# scancel — cancel jobs
scancel 123456                   # cancel job by ID
scancel -u $USER                 # cancel ALL your jobs (be careful)
scancel -u $USER -t PD           # cancel only pending jobs
scancel -u $USER --name=train_bert  # cancel by job name

# sinfo — show cluster partition and node status
sinfo                            # all partitions
sinfo -p gpu                     # only the gpu partition
sinfo -N -l                      # per-node detailed view
# Node states: idle, alloc, mix, down, drain
# idle: no jobs; alloc: fully allocated; mix: partially allocated

# sacct — accounting info for completed jobs
sacct -j 123456 --format=JobID,Elapsed,CPUTime,MaxRSS,State
sacct -u $USER --starttime=2026-04-01 --format=JobID,JobName,Elapsed,State
# MaxRSS: max resident set size (memory peak) — useful for tuning --mem

# scontrol — show detailed job/node info
scontrol show job 123456         # full job details (all SLURM variables)
scontrol show node node01        # node details (CPUs, GPUs, load, state)
scontrol hold 123456             # hold a pending job (prevent it from starting)
scontrol release 123456          # release a held job

# Check GPU allocation on your node (run inside a job)
scontrol show job $SLURM_JOB_ID | grep GRES
```

### GPU Monitoring
{: #gpu-monitoring}

```bash
# nvidia-smi — NVIDIA System Management Interface
nvidia-smi                       # snapshot: all GPUs, memory, utilisation, processes
nvidia-smi -l 1                  # refresh every 1 second (like watch)
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
           --format=csv          # custom query; great for logging

# Explanation of nvidia-smi output:
# GPU %: fraction of time GPU was executing a CUDA kernel (not GPU memory util)
# Mem-Usage: allocated / total VRAM
# SM %: streaming multiprocessor utilisation (same as GPU %)
# Power: current draw vs. limit

# dmon — continuous device monitoring
nvidia-smi dmon -s u             # utilisation metrics only (-s: select columns)
nvidia-smi dmon -s pucmt         # power, utilisation, clocks, memory, temp

# pmon — process monitoring (which processes use which GPUs)
nvidia-smi pmon -i 0             # monitor GPU 0
nvidia-smi pmon -s um            # show memory and sm utilisation

# nvitop — rich TUI GPU monitor (pip install nvitop)
nvitop                           # interactive; shows per-process GPU/CPU/memory

# watch — run any command periodically
watch -n 2 nvidia-smi            # run nvidia-smi every 2 seconds

# Check peer-to-peer GPU connectivity (NVLink / NVSwitch)
nvidia-smi topo -m               # topology matrix (NV2 = 2 NVLink paths, PHB = PCIe)

# DCGM — datacenter GPU manager (cluster-level)
dcgmi dmon -e 203,204,1001,1002  # field IDs for SM util, mem util, etc.

# Inside Python — memory monitoring
import subprocess, re

def gpu_memory_mb():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    return [int(x) for x in result.stdout.strip().split("\n")]
```

```bash
# Check CUDA version
nvcc --version              # CUDA compiler version
nvidia-smi | grep "CUDA Version"  # driver-supported CUDA version (may differ)

# The driver CUDA version is the maximum supported; nvcc is what's installed.
# You can run code compiled for older CUDA if driver version >= code's CUDA version.

# Check GPU compute capability (important for knowing what ops are supported)
python -c "import torch; print(torch.cuda.get_device_capability())"
# (8, 0) = A100 (Ampere); (7, 0) = V100 (Volta); (9, 0) = H100 (Hopper)

# Set which GPUs a process can see
export CUDA_VISIBLE_DEVICES=0,1      # only GPUs 0 and 1 are visible
export CUDA_VISIBLE_DEVICES=0        # single GPU training even if 8 are present

# In Python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"   # must set before importing torch
```

---

## Conda Environment Management
{: #conda}

Conda manages both Python packages and non-Python dependencies (CUDA, MKL, HDF5). It creates isolated environments so projects don't conflict.

```bash
# Installation — use Miniconda (minimal) or Mambaforge (faster solver)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts; say yes to conda init to add conda to PATH

# mamba — drop-in conda replacement; dramatically faster dependency resolution
conda install -n base -c conda-forge mamba
# After: use mamba instead of conda for create/install

# --- Environment management ---

# Create environment
conda create -n myenv python=3.11
# Creates a new isolated env named myenv with Python 3.11

conda create -n myenv python=3.11 numpy pandas  # create with packages
conda create --file environment.yml             # from spec file

# Activate / deactivate
conda activate myenv     # switch to myenv; all conda/pip installs go here
conda deactivate         # return to base

# List environments
conda env list           # or: conda info --envs
# * marks the currently active environment

# Delete an environment
conda env remove -n myenv    # removes all packages and the env directory
conda remove -n myenv --all  # equivalent

# --- Package management ---

# Install packages
conda install numpy scipy matplotlib    # install from defaults channel
conda install -c conda-forge pytorch    # install from conda-forge channel
pip install wandb                       # pip works inside conda envs

# Key rule: do all conda installs first, then pip. Mixing order can break environments.

# Update
conda update numpy                  # update specific package
conda update --all                  # update everything (can be slow)

# Remove
conda remove numpy                  # uninstall package

# List installed packages
conda list                          # all packages in active env
conda list | grep torch             # filter for torch

# --- Export and reproduce ---

# Export environment (exact versions; platform-specific)
conda env export > environment.yml

# Export without build strings (cross-platform; recommended for sharing)
conda env export --no-builds > environment.yml

# Export only manually installed packages (not all transitive deps)
conda env export --from-history > environment_minimal.yml

# Recreate from file
conda env create -f environment.yml
conda env update -f environment.yml --prune   # update existing; remove unlisted

# --- PyTorch + CUDA install (canonical command) ---

# Install PyTorch with CUDA 12.1 (always check pytorch.org for latest)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# --- Useful config ---

# Set default channels (add to ~/.condarc)
conda config --add channels conda-forge
conda config --set channel_priority strict
# strict: conda-forge packages take priority over defaults

# Speed up solver
conda config --set solver libmamba    # use libmamba solver (much faster)

# Show conda info
conda info              # current env, conda version, paths
conda config --show     # all configuration values

# Clean up cached packages (free disk space)
conda clean --all       # remove unused packages, tarballs, cache
# -p: packages; -t: tarballs; -l: logfiles; -a: all
```

### pip flags explained
{: #pip-flags}

```bash
# --- pip install flags you will actually use ---

pip install numpy                     # install latest version from PyPI
pip install "numpy==1.26.4"           # pin exact version
pip install "numpy>=1.24,<2.0"        # version range (use quotes!)
pip install --upgrade numpy           # upgrade to latest
pip install --upgrade pip             # upgrade pip itself

# -e / --editable: install a local package in "development mode"
pip install -e .
# What this does:
#   Instead of copying the package to site-packages, pip creates a
#   .egg-link (or .pth file) pointing back to your source directory.
#   Any edits you make to the source are immediately reflected —
#   no reinstall needed. Essential when developing your own library.
#
# Example use-case:
#   You are building a shared utils library used by multiple projects.
#   cd my_utils && pip install -e .
#   Now `import my_utils` works from anywhere in the env, and editing
#   my_utils/core.py takes effect instantly.

pip install -e ".[dev]"               # install package + optional "dev" deps
pip install -e ".[dev,test]"          # multiple extras groups

# Install from a specific git repo (no PyPI needed)
pip install git+https://github.com/user/repo.git
pip install git+https://github.com/user/repo.git@main        # specific branch
pip install git+https://github.com/user/repo.git@v1.2.0      # specific tag
pip install git+https://github.com/user/repo.git@abc1234     # specific commit

# Install from a local directory (non-editable copy)
pip install ./my_package/

# Install from a requirements file
pip install -r requirements.txt
pip install -r requirements.txt --no-cache-dir   # skip cache (fresh download)

# --no-deps: install package WITHOUT its dependencies
# Useful when you've already pinned deps and don't want pip to override them
pip install some_package --no-deps

# --index-url / --extra-index-url: custom package index
# PyTorch uses this to serve CUDA-specific wheels:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
# --index-url   : replace PyPI entirely with this index
# --extra-index-url: search this index IN ADDITION to PyPI

# Show what's installed
pip list                              # all installed packages
pip list --outdated                   # packages with newer versions available
pip show numpy                        # details: version, location, deps, homepage
pip freeze                            # pinned requirements format (for requirements.txt)
pip freeze > requirements.txt         # capture current env

# Uninstall
pip uninstall numpy                   # with confirmation prompt
pip uninstall numpy -y                # skip prompt

# --- pip cache ---
pip cache list                        # show cached wheels
pip cache purge                       # clear all cached wheels (free disk space)
pip download numpy -d ./wheels/       # download wheel without installing

# --- Debugging installs ---
pip install numpy -v                  # verbose — shows what pip is doing
pip install numpy --dry-run           # show what would be installed without doing it
pip check                             # verify all installed deps are consistent
```

---

## Essential Bash for ML
{: #bash-tools}

```bash
# --- File and directory navigation ---
ls -lh                    # long listing with human-readable sizes
ls -lhS                   # sort by size, largest first
du -sh */                 # disk usage of each subdirectory
du -sh checkpoints/       # total size of a directory

# --- Watching training logs ---
tail -f train.log                     # follow log as it grows
tail -f train.log | grep "val_loss"   # filter for specific pattern
less +F train.log                     # like tail -f but scrollable; Ctrl+C to scroll

# --- grep — search text ---
grep -r "learning_rate" configs/      # recursive search in directory
grep -n "Error" train.log             # show line numbers
grep -c "step" train.log              # count matching lines
grep -v "DEBUG" train.log             # show lines NOT matching
grep -E "loss|accuracy" train.log     # extended regex (multiple patterns)

# --- Screen / tmux — keep jobs running after logout ---
tmux new -s training                  # new session named "training"
tmux attach -t training               # re-attach to session
# Inside tmux: Ctrl+b d to detach; Ctrl+b [ to scroll

screen -S training                    # new screen session
screen -r training                    # re-attach
# Inside screen: Ctrl+a d to detach

# --- Process management ---
ps aux | grep python                  # find Python processes
top -u $USER                          # processes for your user
htop                                  # interactive process viewer
kill -9 <PID>                         # force kill a process
pkill -f "train.py"                   # kill by command name pattern

# --- Resource usage ---
free -h                               # RAM usage (human-readable)
df -h                                 # disk usage by filesystem
lscpu                                 # CPU info (cores, sockets)
lspci | grep -i nvidia               # check GPU hardware

# --- File transfer ---
scp file.py user@cluster:/path/       # copy to cluster
scp -r checkpoints/ user@cluster:~/   # copy directory recursively
rsync -avz --progress checkpoints/ user@cluster:~/checkpoints/
# rsync: -a=archive (preserve permissions), -v=verbose, -z=compress
# --progress: show per-file progress; much better than scp for large dirs

# --- wget / curl — download files ---
wget https://example.com/dataset.tar.gz
wget -c url                           # -c: resume interrupted download
curl -L url -o output.file            # -L: follow redirects; -o: output file

# --- Archive / compress ---
tar -czf results.tar.gz results/      # create compressed tarball
# -c: create; -z: gzip; -f: filename
tar -xzf results.tar.gz               # extract
tar -tzf results.tar.gz               # list contents without extracting
zip -r results.zip results/           # zip alternative

# --- Environment variables ---
export WANDB_PROJECT=my_project       # set for current shell session
printenv | grep CUDA                  # print matching env vars
env -u CUDA_VISIBLE_DEVICES python train.py  # unset for one command

# --- Job arrays with SLURM (for hyperparameter sweeps) ---
#SBATCH --array=0-9                   # 10 jobs, SLURM_ARRAY_TASK_ID = 0..9
#SBATCH --array=0-9%3                 # max 3 running simultaneously

# In the script, use SLURM_ARRAY_TASK_ID to select hyperparameters
LRS=(1e-5 3e-5 1e-4 3e-4 1e-3)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
python train.py --lr $LR

# --- Useful one-liners for ML workflows ---
# Count lines in training data
wc -l data/train.jsonl

# Check number of GPUs visible to Python
python -c "import torch; print(torch.cuda.device_count())"

# Monitor GPU memory every 5 seconds
watch -n 5 "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"

# Find largest files in directory
find . -name "*.pt" -exec du -sh {} \; | sort -rh | head -20

# Check if a port is in use (for distributed training master port)
lsof -i :29500
ss -tlnp | grep 29500

# Redirect stdout and stderr to the same file
python train.py > train.log 2>&1

# Run in background, output to log
nohup python train.py > train.log 2>&1 &
echo $!                       # print PID of background process

# String formatting in bash (useful for checkpoint naming)
CKPT_DIR="checkpoints/run_lr${LR}_bs${BS}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $CKPT_DIR

# Loop over hyperparameters (for quick local sweeps)
for lr in 1e-4 3e-4 1e-3; do
  for bs in 32 64; do
    echo "Running lr=$lr bs=$bs"
    python train.py --lr $lr --batch_size $bs --output_dir runs/lr${lr}_bs${bs}
  done
done
```

```bash
# --- Weights & Biases (wandb) integration ---
wandb login                           # authenticate once
wandb offline                         # disable sync (useful on cluster without internet)
wandb sync ./wandb/offline-run-*      # sync offline runs later

# In Python
import wandb
wandb.init(project="my_project", config={"lr": 1e-3, "epochs": 10})
wandb.log({"loss": loss.item(), "step": step})
wandb.finish()

# --- Module system (cluster-specific) ---
module list                           # currently loaded modules
module avail                          # all available modules
module avail cuda                     # search for cuda modules
module load cuda/12.1 cudnn/8.9       # load specific versions
module unload cuda/12.1               # unload
module purge                          # unload all modules
module show cuda/12.1                 # show what a module does (paths it sets)

# Modules set environment variables like:
# CUDA_HOME, PATH, LD_LIBRARY_PATH
# This is why you must load cuda before running GPU code
```

**Quick reference: common SLURM job states**

| State | Meaning |
|---|---|
| `PD` | Pending — waiting for resources |
| `R` | Running |
| `CG` | Completing — cleaning up after job ended |
| `F` | Failed — non-zero exit code |
| `TO` | Timed Out — hit wall time limit |
| `OOM` | Out of Memory — killed by OOM |
| `CA` | Cancelled — by user or admin |
| `DL` | Deadline — QOS deadline exceeded |
| `PR` | Preempted — higher-priority job took resources |

**Common `#SBATCH` directives quick reference:**

| Directive | Effect |
|---|---|
| `--job-name` | Name shown in squeue |
| `--partition` | Which queue/partition to use |
| `--nodes` | Number of nodes to allocate |
| `--ntasks-per-node` | Processes per node |
| `--gres=gpu:N` | Request N GPUs per node |
| `--cpus-per-task` | CPU cores per task |
| `--mem` | Memory per node (e.g., `128G`) |
| `--time` | Wall clock limit (`HH:MM:SS`) |
| `--output` | stdout log file (`%j`=job ID, `%N`=node) |
| `--error` | stderr log file |
| `--array` | Submit job array |
| `--dependency=afterok:JOBID` | Only start after JOBID succeeds |
| `--requeue` | Requeue if preempted |
| `--account` | Billing account |
| `--qos` | Quality of service override |

---

## General Bash & Shell Productivity
{: #bash-general}

Everything below applies to any Linux/macOS terminal — not ML-specific.

---

### Navigation & Files
{: #bash-nav}

```bash
# --- Directory navigation ---
cd -                        # go back to previous directory (toggle)
cd ~                        # home directory
pushd /some/path            # push current dir onto stack, cd to path
popd                        # pop stack and return to previous dir
dirs                        # show directory stack

# --- ls variants ---
ls -lhSr                    # sort by size, smallest first
ls -lht                     # sort by modification time, newest first
ls -lhtu                    # sort by access time
ls -A                       # include hidden files, exclude . and ..
ls **/*.py                  # glob recursively (bash 4+ with globstar)

# --- Find files ---
find . -name "*.log"                       # find by name
find . -name "*.py" -newer config.py       # files modified after config.py
find . -size +100M                         # files larger than 100 MB
find . -type d -name "__pycache__"         # find directories named __pycache__
find . -type f -name "*.pyc" -delete       # find and delete .pyc files
find . -maxdepth 2 -name "*.yaml"          # limit recursion depth
find . -name "*.log" -mtime +7             # files not modified in 7+ days
find . -name "*.log" -mtime +7 -delete     # delete logs older than 7 days

# --- File inspection ---
file some_file              # show file type (binary? text? which encoding?)
stat file.txt               # inode info: size, permissions, timestamps
wc -l file.txt              # count lines
wc -w file.txt              # count words
wc -c file.txt              # count bytes

# --- Viewing files ---
less file.txt               # paginated view; / to search, q to quit
less +G file.txt            # open at the end
head -n 20 file.txt         # first 20 lines
tail -n 20 file.txt         # last 20 lines
tail -f file.txt            # follow (live append)
cat -n file.txt             # show with line numbers
xxd file.bin | head         # hex dump (inspect binary files)

# --- Permissions ---
chmod +x script.sh          # make executable
chmod 644 file.txt          # rw-r--r--  (owner rw, group r, others r)
chmod 755 dir/              # rwxr-xr-x  (standard for directories/executables)
chown user:group file       # change owner and group
ls -la                      # show permissions, owner, size

# --- Disk & space ---
du -sh *                    # size of each item in current directory
du -sh * | sort -rh         # sort by size, largest first
df -h                       # disk usage by filesystem/mount
ncdu                        # interactive disk usage explorer (install separately)

# --- Symlinks ---
ln -s /path/to/real link_name   # create symbolic link
ls -la link_name                # shows -> target
readlink -f link_name           # resolve to absolute real path
unlink link_name                # remove symlink (or rm link_name)
```

---

### Text Processing
{: #bash-text}

```bash
# --- grep ---
grep "pattern" file.txt               # basic search
grep -i "pattern" file.txt            # case insensitive
grep -r "pattern" ./src/              # recursive
grep -l "pattern" ./src/              # only filenames, not lines
grep -n "pattern" file.txt            # show line numbers
grep -c "pattern" file.txt            # count matches
grep -v "pattern" file.txt            # invert: lines NOT matching
grep -A 3 "pattern" file.txt          # 3 lines After match
grep -B 3 "pattern" file.txt          # 3 lines Before match
grep -C 3 "pattern" file.txt          # 3 lines Context (before + after)
grep -E "err|warn|fail" file.txt      # extended regex (multiple patterns)
grep -P "\d{3}-\d{4}" file.txt        # Perl regex (lookaheads, \d etc.)
grep -o "lr=[0-9.e-]*" log.txt        # print only the matching part

# --- cut, awk — extract columns ---
cut -d',' -f1,3 data.csv              # fields 1 and 3 from CSV
awk '{print $2}' file.txt             # print 2nd whitespace-delimited field
awk -F',' '{print $1, $3}' data.csv   # CSV with custom delimiter
awk 'NR>1' file.txt                   # skip first line (header)
awk '{sum += $1} END {print sum}'     # sum first column
awk '$3 > 0.9 {print $0}' scores.txt  # filter rows where col 3 > 0.9

# --- sed — stream editor ---
sed 's/foo/bar/' file.txt             # replace first occurrence per line
sed 's/foo/bar/g' file.txt            # replace ALL occurrences per line
sed 's/foo/bar/gi' file.txt           # case-insensitive replace all
sed -i 's/foo/bar/g' file.txt         # edit file IN PLACE (dangerous — backup first!)
sed -i.bak 's/foo/bar/g' file.txt     # in-place with .bak backup
sed '/^#/d' config.txt                # delete lines starting with #
sed '1d' file.txt                     # delete first line
sed -n '10,20p' file.txt              # print lines 10-20 only

# --- sort, uniq ---
sort file.txt                         # lexicographic sort
sort -n numbers.txt                   # numeric sort
sort -rn numbers.txt                  # reverse numeric sort
sort -k2 -t',' data.csv               # sort by 2nd field, CSV
sort file.txt | uniq                  # remove duplicate lines (must sort first)
sort file.txt | uniq -c               # count occurrences of each line
sort file.txt | uniq -d               # only print duplicates

# --- tr — character translation ---
echo "Hello World" | tr '[:upper:]' '[:lower:]'    # lowercase
echo "a:b:c" | tr ':' '\n'                         # replace : with newline
echo "hello" | tr -d 'aeiou'                       # delete vowels

# --- xargs — build commands from stdin ---
find . -name "*.pyc" | xargs rm       # delete all .pyc files
find . -name "*.txt" | xargs wc -l    # count lines in each .txt file
cat urls.txt | xargs -n 1 wget        # download each URL (-n 1 = one arg per call)
cat files.txt | xargs -P 4 -I {} cp {} /backup/
# -P 4: run 4 in parallel; -I {}: replace {} with input

# --- Pipes and redirection ---
command > out.txt           # stdout to file (overwrite)
command >> out.txt          # stdout to file (append)
command 2> err.txt          # stderr to file
command > out.txt 2>&1      # both stdout and stderr to same file
command 2>/dev/null         # discard stderr
command | tee out.txt       # stdout to screen AND file simultaneously
command1 | command2         # pipe stdout of command1 to stdin of command2

# --- Combine files ---
cat a.txt b.txt > combined.txt        # concatenate
paste -d',' a.txt b.txt               # side-by-side with comma separator
diff a.txt b.txt                      # show line differences
diff -u a.txt b.txt                   # unified diff (git-style)
```

---

### Scripting Essentials
{: #bash-scripting}

```bash
#!/usr/bin/env bash
# Best practice: use /usr/bin/env bash instead of /bin/bash for portability

# --- Safety flags (put at top of every script) ---
set -e          # exit immediately on any error
set -u          # treat undefined variables as errors
set -o pipefail # pipe fails if ANY command in the pipe fails
# Combined idiom:
set -euo pipefail

# --- Variables ---
NAME="Alice"
echo "Hello, $NAME"
echo "Hello, ${NAME}!"        # braces needed before non-space chars like !

# Default values
LOG_DIR=${LOG_DIR:-/tmp/logs}  # use /tmp/logs if LOG_DIR is unset or empty
: ${PORT:=8080}                # set PORT to 8080 if unset (: is a no-op)

# Command substitution
TODAY=$(date +%Y-%m-%d)
FILES=$(ls *.py | wc -l)

# Arithmetic
N=10
echo $(( N * 2 + 1 ))         # 21
echo $(( N % 3 ))             # 1

# --- Conditionals ---
if [ -f file.txt ]; then       # -f: file exists and is regular file
    echo "found"
elif [ -d dir/ ]; then         # -d: directory exists
    echo "dir exists"
else
    echo "not found"
fi

# Common test flags:
# -f file    file exists and is regular
# -d dir     directory exists
# -z "$var"  string is empty
# -n "$var"  string is non-empty
# -e path    path exists (file, dir, symlink, etc.)

# String comparison (use [[ ]] for safer comparisons)
[[ "$STATUS" == "done" ]] && echo "complete"
[[ "$NAME" != "root" ]] && echo "not root"
[[ "$FILE" == *.py ]] && echo "python file"   # glob in [[ ]]

# Numeric comparison
[[ $N -gt 5 ]] && echo "greater"
[[ $N -le 10 ]] && echo "at most 10"

# --- Loops ---
# C-style for loop
for (( i=0; i<5; i++ )); do
    echo "step $i"
done

# Iterate over array
LRS=(1e-4 3e-4 1e-3)
for lr in "${LRS[@]}"; do
    echo "lr=$lr"
done

# Iterate over files
for f in *.py; do
    echo "Processing $f"
    python lint.py "$f"
done

# While loop
while IFS= read -r line; do   # read file line by line (IFS= preserves spaces)
    echo "Line: $line"
done < input.txt

# --- Functions ---
greet() {
    local name=$1              # local: scope variable to function
    echo "Hello, $name!"
}
greet "World"

# Return a value via stdout (capture with $())
get_timestamp() {
    date +%Y%m%d_%H%M%S
}
TS=$(get_timestamp)

# --- Error handling ---
run_or_die() {
    "$@" || { echo "Command failed: $*" >&2; exit 1; }
}
run_or_die python train.py --config config.yaml

# Trap: run cleanup even if script exits early
cleanup() { echo "Cleaning up..."; rm -f /tmp/lock; }
trap cleanup EXIT             # always run on exit
trap cleanup INT TERM         # also run on Ctrl+C or kill

# --- Arrays ---
arr=("a" "b" "c")
echo "${arr[0]}"              # first element
echo "${arr[@]}"              # all elements
echo "${#arr[@]}"             # length
arr+=("d")                    # append

# Associative array (bash 4+)
declare -A scores
scores["alice"]=95
scores["bob"]=87
echo "${scores["alice"]}"

# --- Useful builtins ---
time python train.py           # measure wall clock time
sleep 2                        # pause for 2 seconds
read -p "Continue? [y/N] " yn  # prompt user for input
[[ $yn == "y" ]] || exit 0

echo $?                        # exit code of last command (0=success)
echo $$                        # PID of current shell
echo $!                        # PID of last background process

# --- Here-doc (multi-line string) ---
cat <<EOF > config.yaml
model: gpt2
lr: 1e-4
epochs: 10
EOF

# --- Aliases & functions in ~/.bashrc or ~/.zshrc ---
alias ll='ls -lh'
alias gs='git status'
alias gd='git diff'
alias gc='git commit -m'
alias ..='cd ..'
alias ...='cd ../..'

mkcd() { mkdir -p "$1" && cd "$1"; }   # create dir and cd into it
```

---

### Network & Remote
{: #bash-network}

```bash
# --- ssh ---
ssh user@host                          # connect
ssh -p 2222 user@host                  # custom port
ssh -i ~/.ssh/my_key user@host         # specific key
ssh -L 8888:localhost:8888 user@host   # local port forward
# Forwards local port 8888 to host's port 8888 — lets you open a Jupyter
# notebook running on the cluster in your local browser at localhost:8888

ssh -N -f -L 8888:localhost:8888 user@host
# -N: don't execute command, just tunnel; -f: background the ssh process

# ssh config (~/.ssh/config) — avoid typing long ssh commands:
# Host mycluster
#   HostName gpu-cluster.university.edu
#   User myusername
#   IdentityFile ~/.ssh/cluster_key
#   ServerAliveInterval 60
# Then just: ssh mycluster

# --- scp / rsync ---
scp file.py user@host:/path/           # copy file to remote
scp user@host:/path/file.py .          # copy file from remote
scp -r dir/ user@host:/path/           # copy directory

rsync -avz src/ user@host:/dst/        # sync directory (-a archive, -v verbose, -z compress)
rsync -avz --progress src/ user@host:/dst/   # show per-file progress
rsync -avz --exclude='*.pyc' --exclude='__pycache__' src/ user@host:/dst/
rsync -avz --dry-run src/ user@host:/dst/    # preview without transferring
rsync -avz --delete src/ user@host:/dst/     # delete files on remote not in src

# --- curl ---
curl https://api.example.com/data              # GET request
curl -X POST -d '{"key":"val"}' -H "Content-Type: application/json" https://api.example.com/
curl -O https://example.com/file.tar.gz        # download, keep original filename
curl -L -o output.tar.gz https://example.com/  # -L follow redirects, -o custom name
curl -C - -O https://example.com/big.tar.gz    # -C -: resume interrupted download
curl -u user:pass https://protected.com/       # basic auth
curl -s https://api.example.com/ | python -m json.tool  # pretty-print JSON response

# --- wget ---
wget https://example.com/file.tar.gz
wget -c url                            # resume interrupted download
wget -q url                            # quiet mode
wget -P /tmp/ url                      # save to directory
wget --limit-rate=1m url               # throttle download speed
wget -r -l 2 https://example.com/      # recursive download, depth 2

# --- Check connectivity ---
ping -c 4 google.com                   # send 4 ICMP packets
traceroute google.com                  # trace network path
nslookup hostname                      # DNS lookup
curl -Is https://google.com | head -1  # quick HTTP status check
nc -zv host 8080                       # check if port 8080 is open on host

# --- Ports and processes ---
lsof -i :8080                          # what's using port 8080
ss -tlnp                               # all listening TCP ports and owning process
netstat -tlnp                          # same (older systems)
kill $(lsof -t -i:8080)                # kill whatever is on port 8080
```

---

### Git Quick Reference
{: #bash-git}

```bash
# --- Setup ---
git config --global user.name "Name"
git config --global user.email "you@example.com"
git config --global core.editor "vim"        # or nano, code, etc.
git config --list                            # show all config

# --- Everyday commands ---
git status                                   # working tree status
git diff                                     # unstaged changes
git diff --staged                            # staged (cached) changes
git add .                                    # stage all changes
git add -p                                   # interactively stage hunks
git commit -m "message"                      # commit staged changes
git commit --amend                           # amend last commit (don't push after!)
git push origin main                         # push to remote
git pull --rebase                            # pull and rebase (cleaner history than merge)

# --- Branches ---
git branch                                   # list local branches (* = current)
git branch -a                                # list all including remote
git checkout -b feature/xyz                  # create and switch to new branch
git switch -c feature/xyz                    # modern equivalent
git switch main                              # switch to existing branch
git branch -d feature/xyz                    # delete merged branch
git branch -D feature/xyz                    # force delete unmerged branch

# --- Stash ---
git stash                                    # save dirty working tree temporarily
git stash push -m "WIP: experiment"          # stash with message
git stash list                               # list all stashes
git stash pop                                # apply most recent stash + remove it
git stash apply stash@{2}                    # apply specific stash, keep it
git stash drop stash@{0}                     # delete specific stash
git stash branch new-branch                  # create branch from stash

# --- History ---
git log --oneline --graph --all              # visual branch graph
git log --oneline -20                        # last 20 commits
git log -p -- path/to/file                   # history with diffs for a file
git log --author="Name"                      # commits by author
git blame file.py                            # who last changed each line
git show abc1234                             # show a specific commit

# --- Undo ---
git restore file.py                          # discard unstaged changes in file
git restore --staged file.py                 # unstage (keep changes in working tree)
git reset HEAD~1                             # undo last commit, keep changes staged
git reset --soft HEAD~1                      # undo last commit, keep changes staged
git reset --hard HEAD~1                      # undo last commit, DISCARD changes (dangerous!)
git revert abc1234                           # create new commit that undoes abc1234 (safe for pushed commits)

# --- Remotes ---
git remote -v                                # list remotes
git remote add upstream https://github.com/org/repo.git
git fetch upstream                           # fetch without merging
git rebase upstream/main                     # rebase your branch onto upstream/main
git push origin --delete old-branch         # delete remote branch

# --- Tags ---
git tag v1.0.0                               # lightweight tag
git tag -a v1.0.0 -m "Release 1.0"          # annotated tag
git push origin --tags                       # push all tags
git tag -d v1.0.0                            # delete local tag
git push origin :refs/tags/v1.0.0           # delete remote tag

# --- Useful one-liners ---
git diff --stat HEAD~5                       # what changed in last 5 commits
git shortlog -sn                             # commit count by author
git grep "TODO" -- "*.py"                    # search across all tracked files
git clean -fd                                # delete untracked files and dirs (careful!)
git clean -n                                 # dry run: show what would be deleted
git bisect start                             # binary search for bad commit
git bisect bad                               # mark current commit as bad
git bisect good v1.0.0                       # mark known good commit → git checks out midpoint
```

| Command | What it does |
|---|---|
| `git add -p` | Stage changes hunk-by-hunk — review every diff before committing |
| `git commit --amend` | Edit last commit message or add forgotten files (before push only) |
| `git pull --rebase` | Avoid merge commits when pulling; keeps history linear |
| `git stash pop` | Restore stashed changes and remove from stash list |
| `git reset --soft HEAD~1` | Undo commit but keep all changes staged |
| `git reset --hard HEAD~1` | Undo commit AND discard changes (irreversible) |
| `git revert` | Safe undo for pushed commits — adds a reverting commit |
| `git bisect` | Binary search to find which commit introduced a bug |
| `git log --oneline --graph --all` | Visual branch topology |
