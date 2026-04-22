---
title: "Low-Level Design: OOP, Python Internals & Databases"
date: 2026-01-18
display_order: 19
description: "Practical low-level design — Python OOP from first principles (self, dunders, inheritance, SOLID), design patterns, SQL from basics to indexing and query planning, concurrency in databases and Python, and real interview problems worked end-to-end."
tags: [lld, oop, python, sql, databases, interviews]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#python-oop-fundamentals">Python OOP Fundamentals</a>
      <ul class="post-toc-sublist">
        <li><a href="#what-is-self">What is `self`?</a></li>
        <li><a href="#dunders">Dunder (Magic) Methods</a></li>
        <li><a href="#classes-internals">Class Internals: `__dict__`, MRO, `type`</a></li>
        <li><a href="#inheritance">Inheritance & MRO</a></li>
        <li><a href="#properties">Properties and Descriptors</a></li>
        <li><a href="#class-static">Class Methods vs Static Methods</a></li>
        <li><a href="#abstract">Abstract Base Classes</a></li>
      </ul>
    </li>
    <li><a href="#solid">SOLID Principles (with Python)</a>
      <ul class="post-toc-sublist">
        <li><a href="#srp">Single Responsibility</a></li>
        <li><a href="#ocp">Open/Closed</a></li>
        <li><a href="#lsp">Liskov Substitution</a></li>
        <li><a href="#isp">Interface Segregation</a></li>
        <li><a href="#dip">Dependency Inversion</a></li>
      </ul>
    </li>
    <li><a href="#design-patterns">Design Patterns</a>
      <ul class="post-toc-sublist">
        <li><a href="#creational">Creational: Singleton, Factory, Builder</a></li>
        <li><a href="#structural">Structural: Adapter, Decorator, Composite</a></li>
        <li><a href="#behavioral">Behavioral: Observer, Strategy, Command, Iterator</a></li>
      </ul>
    </li>
    <li><a href="#lld-problems">LLD Interview Problems</a>
      <ul class="post-toc-sublist">
        <li><a href="#parking-lot">Parking Lot</a></li>
        <li><a href="#lru-cache">LRU Cache</a></li>
        <li><a href="#rate-limiter">Rate Limiter</a></li>
        <li><a href="#pub-sub">Pub-Sub System</a></li>
        <li><a href="#snake-game">Snake Game</a></li>
      </ul>
    </li>
    <li><a href="#sql-fundamentals">SQL Fundamentals</a>
      <ul class="post-toc-sublist">
        <li><a href="#schema-design">Schema Design & Normalisation</a></li>
        <li><a href="#joins">JOINs</a></li>
        <li><a href="#aggregations">Aggregations & Window Functions</a></li>
        <li><a href="#subqueries">Subqueries & CTEs</a></li>
        <li><a href="#transactions">Transactions & ACID</a></li>
      </ul>
    </li>
    <li><a href="#indexing">Indexing Deep Dive</a>
      <ul class="post-toc-sublist">
        <li><a href="#btree">B-Tree Index</a></li>
        <li><a href="#hash-index">Hash Index</a></li>
        <li><a href="#composite">Composite Indexes & Column Order</a></li>
        <li><a href="#covering">Covering Index</a></li>
        <li><a href="#query-plan">Reading EXPLAIN ANALYZE</a></li>
        <li><a href="#index-pitfalls">Index Pitfalls</a></li>
      </ul>
    </li>
    <li><a href="#concurrency">Database Concurrency</a>
      <ul class="post-toc-sublist">
        <li><a href="#isolation-levels">Isolation Levels & Anomalies</a></li>
        <li><a href="#locking">Locking: Row, Table, Advisory</a></li>
        <li><a href="#mvcc">MVCC</a></li>
        <li><a href="#deadlocks">Deadlocks</a></li>
        <li><a href="#optimistic">Optimistic vs Pessimistic Concurrency</a></li>
      </ul>
    </li>
    <li><a href="#python-concurrency">Python Concurrency</a>
      <ul class="post-toc-sublist">
        <li><a href="#gil">The GIL</a></li>
        <li><a href="#threading">Threading</a></li>
        <li><a href="#asyncio">asyncio</a></li>
        <li><a href="#multiprocessing">Multiprocessing</a></li>
      </ul>
    </li>
    <li><a href="#practical-sql">Practical SQL Problems</a></li>
  </ul>
</nav>

---

## Python OOP Fundamentals {#python-oop-fundamentals}

### What is `self`? {#what-is-self}

`self` is not a keyword — it's a naming convention for the first parameter of instance methods. Python passes the instance automatically when you call a method on it.

```python
class Dog:
    def bark(self):
        print(f"Woof from {self}")

d = Dog()
d.bark()          # Python calls Dog.bark(d) under the hood
Dog.bark(d)       # identical — explicit call
```

When you write `d.bark()`, Python looks up `bark` on the class (not the instance), sees it's a function, wraps it as a *bound method* that prepends `d` as the first argument. `self` is literally just `d`.

```python
import inspect
print(type(Dog.bark))   # <class 'function'>
print(type(d.bark))     # <class 'method'>   — bound to d
```

**`__init__` is not the constructor** — `__new__` creates the object; `__init__` initialises it. 99% of the time you only override `__init__`.

```python
class Point:
    def __init__(self, x, y):   # self is the freshly-created Point instance
        self.x = x              # sets instance attribute
        self.y = y
```

Instance attributes live in `self.__dict__`; class attributes live in `MyClass.__dict__`. Attribute lookup walks instance → class → base classes (MRO).

```python
class Counter:
    count = 0           # class attribute — shared across all instances

    def __init__(self):
        Counter.count += 1   # modify class attribute explicitly
        self.id = Counter.count  # instance attribute
```

---

### Dunder (Magic) Methods {#dunders}

Dunder = **d**ouble **under**score on both sides. They are how Python's operators and built-in functions hook into your classes. You never call them directly — Python calls them for you.

#### Object Representation

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        # unambiguous, aimed at developers; used in debugger and REPL
        return f"Vector({self.x}, {self.y})"

    def __str__(self):
        # readable, aimed at end users; used by print()
        return f"<{self.x}, {self.y}>"

v = Vector(3, 4)
repr(v)   # "Vector(3, 4)"
str(v)    # "<3, 4>"
print(v)  # <3, 4>
```

Rule: always implement `__repr__`. Only implement `__str__` if you want a different user-facing string.

#### Arithmetic Operators

```python
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):        # v * 3
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):       # 3 * v  — reflected (right-side)
        return self.__mul__(scalar)

    def __neg__(self):                # -v
        return Vector(-self.x, -self.y)

    def __abs__(self):                # abs(v)
        return (self.x**2 + self.y**2) ** 0.5
```

#### Comparison

```python
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return abs(self) < abs(other)

    def __le__(self, other):
        return abs(self) <= abs(other)

    # If you implement __eq__, Python sets __hash__ = None (unhashable).
    # Restore it explicitly if you want the object in sets/dicts:
    def __hash__(self):
        return hash((self.x, self.y))
```

`@functools.total_ordering` generates the missing comparison methods from `__eq__` + one of `__lt__/__le__/__gt__/__ge__`.

#### Container Protocol

```python
class Playlist:
    def __init__(self):
        self._songs = []

    def add(self, song):
        self._songs.append(song)

    def __len__(self):          # len(playlist)
        return len(self._songs)

    def __getitem__(self, idx): # playlist[0], playlist[1:3], for s in playlist
        return self._songs[idx]

    def __contains__(self, song): # "Bohemian Rhapsody" in playlist
        return song in self._songs

    def __iter__(self):         # explicit iterator (fallback: __getitem__ + IndexError)
        return iter(self._songs)
```

Implementing `__getitem__` alone is enough to make the class iterable and support `in` — Python falls back to sequential search. Implement `__contains__` when you can do better than O(n).

#### Context Manager Protocol

```python
class Timer:
    import time

    def __enter__(self):
        self.start = time.perf_counter()
        return self          # value bound to `as` variable

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start
        print(f"Elapsed: {self.elapsed:.4f}s")
        return False         # False = don't suppress exceptions

with Timer() as t:
    do_work()
print(t.elapsed)
```

#### Callable Objects

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):     # lets you use the instance like a function
        return x * self.factor

double = Multiplier(2)
double(5)   # 10
```

#### Complete Dunder Reference

| Dunder | Triggered by |
|---|---|
| `__init__` | `MyClass(...)` |
| `__new__` | object creation (before `__init__`) |
| `__del__` | garbage collection |
| `__repr__` / `__str__` | `repr()` / `str()` / `print()` |
| `__len__` | `len()` |
| `__getitem__` / `__setitem__` / `__delitem__` | `obj[key]` / `obj[key]=v` / `del obj[key]` |
| `__contains__` | `x in obj` |
| `__iter__` / `__next__` | `for`, `iter()`, `next()` |
| `__enter__` / `__exit__` | `with` statement |
| `__call__` | `obj(...)` |
| `__add__` / `__radd__` / `__iadd__` | `+` / reflected `+` / `+=` |
| `__eq__` / `__lt__` / `__le__` | `==` / `<` / `<=` |
| `__hash__` | `hash()`, dict key, set membership |
| `__bool__` | `bool()`, `if obj:` |
| `__getattr__` | attribute not found normally |
| `__getattribute__` | *every* attribute access |
| `__setattr__` / `__delattr__` | `obj.x = v` / `del obj.x` |
| `__class_getitem__` | `MyClass[T]` (generics) |

---

### Class Internals: `__dict__`, MRO, `type` {#classes-internals}

Every class and instance has a `__dict__` mapping names to values.

```python
class Foo:
    class_var = 42

    def __init__(self, x):
        self.x = x

f = Foo(10)
print(f.__dict__)      # {'x': 10}         — instance namespace
print(Foo.__dict__)    # {'class_var': 42, '__init__': <function>, ...}
```

Attribute lookup (`f.x`) follows this order:
1. Data descriptors on the class (e.g., properties with `__set__`)
2. Instance `__dict__`
3. Non-data descriptors and other class attributes

**`type` is the metaclass of all classes.** A class is itself an instance of `type`.

```python
print(type(42))      # <class 'int'>
print(type(int))     # <class 'type'>
print(type(Foo))     # <class 'type'>

# You can create a class dynamically:
MyClass = type('MyClass', (object,), {'greet': lambda self: 'hello'})
obj = MyClass()
obj.greet()  # 'hello'
```

---

### Inheritance & MRO {#inheritance}

Python uses **C3 linearisation** to determine Method Resolution Order (MRO) — the order in which base classes are searched.

```python
class A:
    def greet(self): return "A"

class B(A):
    def greet(self): return "B → " + super().greet()

class C(A):
    def greet(self): return "C → " + super().greet()

class D(B, C):
    pass

print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

D().greet()   # "B → C → A"
```

`super()` does not mean "parent class" — it means "next in MRO". This is why cooperative multiple inheritance works: each class calls `super()` and the chain proceeds down the MRO.

**Common interview trap**: never call `A.method(self)` explicitly in multiple inheritance — it skips the MRO chain. Always use `super()`.

```python
# Mixin pattern — compose behaviours without deep inheritance
class JSONMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class TimestampMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from datetime import datetime
        self.created_at = datetime.utcnow()

class User(TimestampMixin, JSONMixin):
    def __init__(self, name, email):
        super().__init__()   # TimestampMixin.__init__ → object.__init__
        self.name = name
        self.email = email

u = User("Alice", "alice@example.com")
print(u.to_json())   # includes created_at
```

---

### Properties and Descriptors {#properties}

A **property** is syntactic sugar for a descriptor that controls attribute access.

```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius   # private backing attribute

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Below absolute zero")
        self._celsius = value

    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32    # computed, read-only

t = Temperature(25)
print(t.fahrenheit)   # 77.0
t.celsius = -300      # raises ValueError
```

A **descriptor** is any class implementing `__get__`, `__set__`, or `__delete__`. Properties are descriptors. You write your own when you want reusable attribute logic across many classes.

```python
class Validated:
    """Reusable descriptor for range-checked numeric attributes."""

    def __set_name__(self, owner, name):
        self.name = name               # called when class is created
        self.private = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self                # accessed on class, not instance
        return getattr(obj, self.private, None)

    def __set__(self, obj, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be numeric")
        setattr(obj, self.private, value)


class Circle:
    radius = Validated()
    x = Validated()
    y = Validated()

    def __init__(self, x, y, radius):
        self.x, self.y, self.radius = x, y, radius

c = Circle(0, 0, 5)
c.radius = "big"   # TypeError
```

---

### Class Methods vs Static Methods {#class-static}

```python
class Pizza:
    _price_per_inch = 1.5

    def __init__(self, size, toppings):
        self.size = size
        self.toppings = toppings

    @classmethod
    def from_string(cls, s):
        """Alternative constructor — receives the class, not an instance."""
        size, *toppings = s.split(",")
        return cls(int(size), toppings)    # cls allows subclassing to work

    @staticmethod
    def is_valid_size(size):
        """Utility that doesn't need class or instance state."""
        return size in (8, 12, 16, 20)

    @property
    def price(self):
        return self.size * self._price_per_inch


p = Pizza.from_string("12,cheese,pepperoni")
Pizza.is_valid_size(14)   # False
```

| | `self` | `cls` | Neither |
|---|---|---|---|
| Instance method | ✓ | ✓ via `type(self)` | ✓ via globals |
| `@classmethod` | ✗ | ✓ | ✓ |
| `@staticmethod` | ✗ | ✗ | ✓ |

Use `@classmethod` for alternative constructors and factory methods. Use `@staticmethod` for pure utility functions that happen to live in the class namespace.

---

### Abstract Base Classes {#abstract}

`abc.ABC` + `@abstractmethod` enforces that subclasses implement required methods — the Python equivalent of interfaces.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        ...

    @abstractmethod
    def perimeter(self) -> float:
        ...

    def describe(self):
        # concrete method using abstract interface
        return f"Area={self.area():.2f}, Perimeter={self.perimeter():.2f}"


class Rectangle(Shape):
    def __init__(self, w, h):
        self.w, self.h = w, h

    def area(self):
        return self.w * self.h

    def perimeter(self):
        return 2 * (self.w + self.h)


Shape()          # TypeError: Can't instantiate abstract class
Rectangle(3, 4)  # fine
```

`ABC` also supports `__subclasshook__` for structural subtyping — a class can be considered a subclass without explicitly inheriting from the ABC.

---

## SOLID Principles (with Python) {#solid}

### Single Responsibility Principle {#srp}

A class should have one reason to change.

```python
# BAD — Order does too much
class Order:
    def __init__(self, items): self.items = items
    def calculate_total(self): ...
    def save_to_db(self): ...       # storage concern
    def send_confirmation_email(self): ...  # notification concern

# GOOD — each class has one job
class Order:
    def __init__(self, items): self.items = items
    def total(self): return sum(i.price for i in self.items)

class OrderRepository:
    def save(self, order): ...

class OrderNotifier:
    def confirm(self, order): ...
```

### Open/Closed Principle {#ocp}

Open for extension, closed for modification. Add new behaviour by adding new code, not changing existing code.

```python
from abc import ABC, abstractmethod

class Discount(ABC):
    @abstractmethod
    def apply(self, price: float) -> float: ...

class NoDiscount(Discount):
    def apply(self, price): return price

class PercentDiscount(Discount):
    def __init__(self, pct): self.pct = pct
    def apply(self, price): return price * (1 - self.pct / 100)

class BuyOneGetOne(Discount):
    def apply(self, price): return price / 2

# Adding a new discount type = new class, zero changes to existing code
class LoyaltyDiscount(Discount):
    def apply(self, price): return price * 0.85
```

### Liskov Substitution Principle {#lsp}

A subclass must be substitutable for its base class without breaking correctness.

```python
# VIOLATES LSP — Square breaks Rectangle's invariant
class Rectangle:
    def __init__(self, w, h): self.w, self.h = w, h
    def set_width(self, w): self.w = w
    def set_height(self, h): self.h = h
    def area(self): return self.w * self.h

class Square(Rectangle):
    def set_width(self, w): self.w = self.h = w   # silently changes height!
    def set_height(self, h): self.w = self.h = h

# Code that works for Rectangle breaks for Square:
def double_width(r: Rectangle):
    r.set_width(r.w * 2)
    assert r.area() == r.w * r.h   # fails for Square!

# Fix: don't inherit. Use composition or a common abstract base.
```

### Interface Segregation Principle {#isp}

Don't force clients to depend on methods they don't use. Split fat interfaces.

```python
# BAD — not all workers can eat or sleep
class Worker(ABC):
    @abstractmethod
    def work(self): ...
    @abstractmethod
    def eat(self): ...

# GOOD — segregated
class Workable(ABC):
    @abstractmethod
    def work(self): ...

class Feedable(ABC):
    @abstractmethod
    def eat(self): ...

class HumanWorker(Workable, Feedable):
    def work(self): ...
    def eat(self): ...

class RobotWorker(Workable):
    def work(self): ...   # robots don't eat
```

### Dependency Inversion Principle {#dip}

High-level modules should not depend on low-level modules — both should depend on abstractions.

```python
# BAD — UserService hardcodes MySQL
class UserService:
    def __init__(self):
        self.db = MySQLDatabase()   # tightly coupled

# GOOD — depends on abstraction
class UserRepository(ABC):
    @abstractmethod
    def find_by_id(self, uid: int): ...
    @abstractmethod
    def save(self, user): ...

class MySQLUserRepository(UserRepository):
    def find_by_id(self, uid): ...
    def save(self, user): ...

class InMemoryUserRepository(UserRepository):   # for testing
    def __init__(self): self._store = {}
    def find_by_id(self, uid): return self._store.get(uid)
    def save(self, user): self._store[user.id] = user

class UserService:
    def __init__(self, repo: UserRepository):   # injected
        self.repo = repo

# Wire up:
service = UserService(MySQLUserRepository())
# Or in tests:
service = UserService(InMemoryUserRepository())
```

---

## Design Patterns {#design-patterns}

### Creational {#creational}

#### Singleton

```python
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config=None):
        if not hasattr(self, '_initialised'):
            self.config = config
            self._initialised = True

s1 = Singleton({"debug": True})
s2 = Singleton()
assert s1 is s2   # same object

# Thread-safe version:
import threading

class ThreadSafeSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:   # double-checked locking
                    cls._instance = super().__new__(cls)
        return cls._instance
```

#### Factory Method

```python
from abc import ABC, abstractmethod

class Notification(ABC):
    @abstractmethod
    def send(self, message: str): ...

class EmailNotification(Notification):
    def send(self, message): print(f"Email: {message}")

class SMSNotification(Notification):
    def send(self, message): print(f"SMS: {message}")

class PushNotification(Notification):
    def send(self, message): print(f"Push: {message}")

class NotificationFactory:
    _registry = {
        "email": EmailNotification,
        "sms": SMSNotification,
        "push": PushNotification,
    }

    @classmethod
    def create(cls, kind: str) -> Notification:
        if kind not in cls._registry:
            raise ValueError(f"Unknown notification type: {kind}")
        return cls._registry[kind]()

    @classmethod
    def register(cls, kind: str, notification_cls):
        cls._registry[kind] = notification_cls

notif = NotificationFactory.create("email")
notif.send("Hello!")
```

#### Builder

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class QueryBuilder:
    _table: str = ""
    _conditions: List[str] = field(default_factory=list)
    _columns: List[str] = field(default_factory=list)
    _limit: int = None
    _order_by: str = None

    def select(self, *cols):
        self._columns = list(cols)
        return self   # return self for chaining

    def from_table(self, table):
        self._table = table
        return self

    def where(self, condition):
        self._conditions.append(condition)
        return self

    def order_by(self, col):
        self._order_by = col
        return self

    def limit(self, n):
        self._limit = n
        return self

    def build(self):
        cols = ", ".join(self._columns) if self._columns else "*"
        sql = f"SELECT {cols} FROM {self._table}"
        if self._conditions:
            sql += " WHERE " + " AND ".join(self._conditions)
        if self._order_by:
            sql += f" ORDER BY {self._order_by}"
        if self._limit:
            sql += f" LIMIT {self._limit}"
        return sql

query = (QueryBuilder()
    .select("id", "name", "email")
    .from_table("users")
    .where("age > 18")
    .where("active = TRUE")
    .order_by("name")
    .limit(10)
    .build())
# "SELECT id, name, email FROM users WHERE age > 18 AND active = TRUE ORDER BY name LIMIT 10"
```

---

### Structural {#structural}

#### Adapter

Makes an incompatible interface compatible without changing original code.

```python
class LegacyLogger:
    def write_log(self, severity, msg):
        print(f"[{severity}] {msg}")

class ModernLoggerInterface(ABC):
    @abstractmethod
    def info(self, msg): ...
    @abstractmethod
    def error(self, msg): ...

class LegacyLoggerAdapter(ModernLoggerInterface):
    def __init__(self, legacy: LegacyLogger):
        self._legacy = legacy

    def info(self, msg):
        self._legacy.write_log("INFO", msg)

    def error(self, msg):
        self._legacy.write_log("ERROR", msg)

logger = LegacyLoggerAdapter(LegacyLogger())
logger.info("System started")
```

#### Decorator (structural — not Python `@decorator`)

Adds behaviour to objects dynamically without subclassing.

```python
class Coffee(ABC):
    @abstractmethod
    def cost(self) -> float: ...
    @abstractmethod
    def description(self) -> str: ...

class SimpleCoffee(Coffee):
    def cost(self): return 1.0
    def description(self): return "Coffee"

class CoffeeDecorator(Coffee):
    def __init__(self, coffee: Coffee):
        self._coffee = coffee

class Milk(CoffeeDecorator):
    def cost(self): return self._coffee.cost() + 0.25
    def description(self): return self._coffee.description() + ", Milk"

class Sugar(CoffeeDecorator):
    def cost(self): return self._coffee.cost() + 0.10
    def description(self): return self._coffee.description() + ", Sugar"

drink = Sugar(Milk(Milk(SimpleCoffee())))
print(drink.description())   # Coffee, Milk, Milk, Sugar
print(drink.cost())          # 1.60
```

---

### Behavioral {#behavioral}

#### Observer

```python
from typing import List, Callable

class EventEmitter:
    def __init__(self):
        self._listeners: dict[str, List[Callable]] = {}

    def on(self, event: str, callback: Callable):
        self._listeners.setdefault(event, []).append(callback)
        return self

    def emit(self, event: str, *args, **kwargs):
        for cb in self._listeners.get(event, []):
            cb(*args, **kwargs)

    def off(self, event: str, callback: Callable):
        if event in self._listeners:
            self._listeners[event].remove(callback)

emitter = EventEmitter()
emitter.on("login", lambda user: print(f"User {user} logged in"))
emitter.on("login", lambda user: log_audit(user))
emitter.emit("login", "Alice")
```

#### Strategy

```python
from typing import Protocol

class SortStrategy(Protocol):
    def sort(self, data: list) -> list: ...

class QuickSort:
    def sort(self, data):
        if len(data) <= 1: return data
        pivot = data[len(data)//2]
        left = [x for x in data if x < pivot]
        mid  = [x for x in data if x == pivot]
        right= [x for x in data if x > pivot]
        return self.sort(left) + mid + self.sort(right)

class MergeSort:
    def sort(self, data):
        if len(data) <= 1: return data
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right= self.sort(data[mid:])
        return self._merge(left, right)

    def _merge(self, l, r):
        result = []
        i = j = 0
        while i < len(l) and j < len(r):
            if l[i] <= r[j]: result.append(l[i]); i += 1
            else: result.append(r[j]); j += 1
        return result + l[i:] + r[j:]

class Sorter:
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: SortStrategy):
        self._strategy = strategy

    def sort(self, data):
        return self._strategy.sort(data)

sorter = Sorter(QuickSort())
sorter.sort([3, 1, 4, 1, 5, 9])
sorter.set_strategy(MergeSort())
sorter.sort([3, 1, 4, 1, 5, 9])
```

#### Command

Encapsulates actions as objects — enables undo/redo, queuing, logging.

```python
from abc import ABC, abstractmethod
from collections import deque

class Command(ABC):
    @abstractmethod
    def execute(self): ...
    @abstractmethod
    def undo(self): ...

class TextEditor:
    def __init__(self): self.text = ""
    def insert(self, s): self.text += s
    def delete(self, n): self.text = self.text[:-n]

class InsertCommand(Command):
    def __init__(self, editor: TextEditor, text: str):
        self.editor = editor
        self.text = text

    def execute(self): self.editor.insert(self.text)
    def undo(self): self.editor.delete(len(self.text))

class CommandHistory:
    def __init__(self):
        self._history: deque[Command] = deque()

    def execute(self, cmd: Command):
        cmd.execute()
        self._history.append(cmd)

    def undo(self):
        if self._history:
            self._history.pop().undo()

editor = TextEditor()
history = CommandHistory()
history.execute(InsertCommand(editor, "Hello"))
history.execute(InsertCommand(editor, " World"))
print(editor.text)   # "Hello World"
history.undo()
print(editor.text)   # "Hello"
```

---

## LLD Interview Problems {#lld-problems}

### Parking Lot {#parking-lot}

Classic OOP interview — shows polymorphism, enums, and state management.

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid

class VehicleType(Enum):
    MOTORCYCLE = auto()
    CAR = auto()
    BUS = auto()

class SpotSize(Enum):
    SMALL = auto()
    MEDIUM = auto()
    LARGE = auto()

# Which spots fit which vehicle
VEHICLE_TO_SPOT = {
    VehicleType.MOTORCYCLE: SpotSize.SMALL,
    VehicleType.CAR: SpotSize.MEDIUM,
    VehicleType.BUS: SpotSize.LARGE,
}

HOURLY_RATE = {
    SpotSize.SMALL: 2.0,
    SpotSize.MEDIUM: 4.0,
    SpotSize.LARGE: 8.0,
}

@dataclass
class Vehicle:
    plate: str
    type: VehicleType

@dataclass
class ParkingSpot:
    spot_id: str
    size: SpotSize
    floor: int
    vehicle: Optional[Vehicle] = None

    @property
    def is_free(self): return self.vehicle is None

    def park(self, v: Vehicle):
        if not self.is_free: raise RuntimeError("Spot occupied")
        self.vehicle = v

    def leave(self):
        self.vehicle = None

@dataclass
class Ticket:
    ticket_id: str
    vehicle: Vehicle
    spot: ParkingSpot
    entry_time: datetime = field(default_factory=datetime.utcnow)

    def fee(self) -> float:
        hours = (datetime.utcnow() - self.entry_time).total_seconds() / 3600
        return max(1, hours) * HOURLY_RATE[self.spot.size]

class ParkingLot:
    def __init__(self):
        self._spots: dict[SpotSize, list[ParkingSpot]] = {s: [] for s in SpotSize}
        self._tickets: dict[str, Ticket] = {}

    def add_spot(self, spot: ParkingSpot):
        self._spots[spot.size].append(spot)

    def _find_spot(self, size: SpotSize) -> Optional[ParkingSpot]:
        for spot in self._spots[size]:
            if spot.is_free:
                return spot
        return None

    def park(self, vehicle: Vehicle) -> Ticket:
        size = VEHICLE_TO_SPOT[vehicle.type]
        spot = self._find_spot(size)
        if spot is None:
            raise RuntimeError(f"No {size.name} spots available")
        spot.park(vehicle)
        ticket = Ticket(str(uuid.uuid4()), vehicle, spot)
        self._tickets[ticket.ticket_id] = ticket
        return ticket

    def checkout(self, ticket_id: str) -> float:
        ticket = self._tickets.pop(ticket_id)
        fee = ticket.fee()
        ticket.spot.leave()
        return fee

# Usage
lot = ParkingLot()
for i in range(50):
    lot.add_spot(ParkingSpot(f"M{i}", SpotSize.SMALL, floor=i//10))
for i in range(100):
    lot.add_spot(ParkingSpot(f"C{i}", SpotSize.MEDIUM, floor=i//20))

car = Vehicle("ABC-123", VehicleType.CAR)
ticket = lot.park(car)
# ... time passes ...
fee = lot.checkout(ticket.ticket_id)
print(f"Fee: ${fee:.2f}")
```

---

### LRU Cache {#lru-cache}

Implement `get` and `put` in O(1). Uses a doubly-linked list + hash map.

```python
class Node:
    __slots__ = ('key', 'val', 'prev', 'next')

    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.map: dict[int, Node] = {}
        # Sentinel nodes — no edge case handling needed
        self.head = Node()   # least recent
        self.tail = Node()   # most recent
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _insert_at_tail(self, node: Node):
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def get(self, key: int) -> int:
        if key not in self.map:
            return -1
        node = self.map[key]
        self._remove(node)
        self._insert_at_tail(node)
        return node.val

    def put(self, key: int, value: int):
        if key in self.map:
            self._remove(self.map[key])
        node = Node(key, value)
        self.map[key] = node
        self._insert_at_tail(node)
        if len(self.map) > self.cap:
            lru = self.head.next
            self._remove(lru)
            del self.map[lru.key]

cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
cache.get(1)     # 1 (marks 1 as recently used)
cache.put(3, 3)  # evicts key 2
cache.get(2)     # -1
```

Note: Python's `collections.OrderedDict` provides this in 3 lines, but interviewers want the manual implementation.

---

### Rate Limiter {#rate-limiter}

Three algorithms — know all three.

```python
import time
import threading
from collections import deque

# 1. Token Bucket — smooth bursting allowed
class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # tokens added per second
        self.capacity = capacity  # max tokens
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def allow(self, tokens=1) -> bool:
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

# 2. Sliding Window Log — exact but memory-heavy
class SlidingWindowLog:
    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window = window_seconds
        self.log: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.monotonic()
            cutoff = now - self.window
            while self.log and self.log[0] <= cutoff:
                self.log.popleft()
            if len(self.log) < self.limit:
                self.log.append(now)
                return True
            return False

# 3. Fixed Window Counter — simplest, but boundary burst problem
class FixedWindowCounter:
    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window = window_seconds
        self.count = 0
        self.window_start = time.monotonic()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.monotonic()
            if now - self.window_start >= self.window:
                self.count = 0
                self.window_start = now
            if self.count < self.limit:
                self.count += 1
                return True
            return False

# Per-user rate limiter wrapping any algorithm
class RateLimiterService:
    def __init__(self, rate=10, capacity=10):
        self._limiters: dict[str, TokenBucket] = {}
        self._rate = rate
        self._capacity = capacity
        self._lock = threading.Lock()

    def is_allowed(self, user_id: str) -> bool:
        with self._lock:
            if user_id not in self._limiters:
                self._limiters[user_id] = TokenBucket(self._rate, self._capacity)
        return self._limiters[user_id].allow()
```

---

### Pub-Sub System {#pub-sub}

```python
from typing import Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading
import uuid
from collections import defaultdict

@dataclass
class Message:
    topic: str
    payload: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

class Broker:
    def __init__(self):
        self._subscribers: dict[str, dict[str, Callable]] = defaultdict(dict)
        self._lock = threading.RLock()

    def subscribe(self, topic: str, callback: Callable, subscriber_id: str = None) -> str:
        sid = subscriber_id or str(uuid.uuid4())
        with self._lock:
            self._subscribers[topic][sid] = callback
        return sid

    def unsubscribe(self, topic: str, subscriber_id: str):
        with self._lock:
            self._subscribers[topic].pop(subscriber_id, None)

    def publish(self, topic: str, payload: Any):
        msg = Message(topic, payload)
        with self._lock:
            callbacks = list(self._subscribers[topic].values())
        for cb in callbacks:
            threading.Thread(target=cb, args=(msg,), daemon=True).start()

    def publish_sync(self, topic: str, payload: Any):
        msg = Message(topic, payload)
        with self._lock:
            callbacks = list(self._subscribers[topic].values())
        for cb in callbacks:
            cb(msg)

broker = Broker()
sid = broker.subscribe("orders", lambda m: print(f"Order received: {m.payload}"))
broker.subscribe("orders", lambda m: send_to_warehouse(m.payload))
broker.publish("orders", {"item": "book", "qty": 2})
broker.unsubscribe("orders", sid)
```

---

### Snake Game {#snake-game}

Tests data structures (deque for snake body), state machine, collision detection.

```python
from collections import deque
from enum import Enum

class Direction(Enum):
    UP = (-1, 0); DOWN = (1, 0); LEFT = (0, -1); RIGHT = (0, 1)

OPPOSITES = {
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT,
}

class SnakeGame:
    def __init__(self, rows: int, cols: int, food: list[tuple[int, int]]):
        self.rows, self.cols = rows, cols
        self.food = deque(food)
        self.snake: deque[tuple[int, int]] = deque([(0, 0)])
        self.body_set: set[tuple[int, int]] = {(0, 0)}  # O(1) collision check
        self.direction = Direction.RIGHT
        self.score = 0

    def move(self, direction: Direction) -> int:
        if direction == OPPOSITES[self.direction]:
            direction = self.direction   # can't reverse
        self.direction = direction

        head = self.snake[0]
        dr, dc = direction.value
        new_head = (head[0] + dr, head[1] + dc)

        # Wall collision
        if not (0 <= new_head[0] < self.rows and 0 <= new_head[1] < self.cols):
            return -1

        # Check if eating food
        eating = self.food and self.food[0] == new_head

        # Remove tail before collision check (tail moves away unless eating)
        if not eating:
            tail = self.snake.pop()
            self.body_set.remove(tail)

        # Self collision
        if new_head in self.body_set:
            return -1

        if eating:
            self.food.popleft()
            self.score += 1

        self.snake.appendleft(new_head)
        self.body_set.add(new_head)
        return self.score
```

---

## SQL Fundamentals {#sql-fundamentals}

### Schema Design & Normalisation {#schema-design}

**Normal forms** prevent data anomalies:

| NF | Rule | Violation Example |
|---|---|---|
| 1NF | Atomic values, no repeating groups | `phone = "555-1234, 555-5678"` |
| 2NF | No partial dependency on composite PK | `order_item` storing `product_name` (depends only on `product_id`) |
| 3NF | No transitive dependency | `orders` storing `customer_city` (city depends on customer, not order) |
| BCNF | Every determinant is a candidate key | Rare; violates when a non-key column determines part of the key |

**Practical schema for an e-commerce system:**

```sql
CREATE TABLE users (
    id          BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    email       TEXT UNIQUE NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE products (
    id          BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    name        TEXT NOT NULL,
    price_cents INT NOT NULL CHECK (price_cents >= 0),
    stock       INT NOT NULL DEFAULT 0,
    category_id INT REFERENCES categories(id)
);

CREATE TABLE orders (
    id          BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    user_id     BIGINT NOT NULL REFERENCES users(id),
    status      TEXT NOT NULL CHECK (status IN ('pending','paid','shipped','cancelled')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE order_items (
    order_id    BIGINT NOT NULL REFERENCES orders(id),
    product_id  BIGINT NOT NULL REFERENCES products(id),
    quantity    INT NOT NULL CHECK (quantity > 0),
    unit_price  INT NOT NULL,     -- snapshot price at time of order
    PRIMARY KEY (order_id, product_id)
);
```

Key decisions:
- Store `unit_price` in `order_items` (denormalize intentionally) because product price changes over time
- Use `BIGINT` for IDs (int exhaustion is real at scale)
- `TIMESTAMPTZ` not `TIMESTAMP` (timezone-aware)
- Enum-like status as `CHECK` constraint for portability; use a proper enum type in PostgreSQL for performance

---

### JOINs {#joins}

```sql
-- INNER JOIN — only rows with matches in both tables
SELECT u.email, COUNT(o.id) AS order_count
FROM users u
INNER JOIN orders o ON o.user_id = u.id
GROUP BY u.id, u.email;

-- LEFT JOIN — all users, even those with no orders
SELECT u.email, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
GROUP BY u.id, u.email;

-- Find users with NO orders
SELECT u.email
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE o.id IS NULL;

-- Self-join: find employees and their managers (same table)
SELECT e.name AS employee, m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;

-- Multi-table join
SELECT u.email, p.name, oi.quantity, oi.unit_price
FROM order_items oi
JOIN orders o     ON o.id = oi.order_id
JOIN users u      ON u.id = o.user_id
JOIN products p   ON p.id = oi.product_id
WHERE o.status = 'paid';
```

---

### Aggregations & Window Functions {#aggregations}

```sql
-- Basic aggregation
SELECT
    status,
    COUNT(*) AS cnt,
    AVG(total_cents) AS avg_total,
    SUM(total_cents) AS revenue
FROM orders
GROUP BY status
HAVING COUNT(*) > 100;   -- filter on aggregated value (not WHERE)

-- Window functions — aggregate without collapsing rows
SELECT
    user_id,
    order_id,
    created_at,
    SUM(total_cents) OVER (PARTITION BY user_id) AS user_lifetime_value,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at) AS order_num,
    LAG(total_cents) OVER (PARTITION BY user_id ORDER BY created_at) AS prev_order_total,
    RANK() OVER (ORDER BY total_cents DESC) AS rank_by_value
FROM orders;

-- Top N per group: latest order per user
SELECT * FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn
    FROM orders
) ranked
WHERE rn = 1;

-- Running total
SELECT
    created_at::date AS day,
    SUM(total_cents) AS daily_revenue,
    SUM(SUM(total_cents)) OVER (ORDER BY created_at::date) AS running_total
FROM orders
GROUP BY created_at::date;
```

---

### Subqueries & CTEs {#subqueries}

```sql
-- Correlated subquery: users who spent more than average
SELECT email
FROM users u
WHERE (
    SELECT COALESCE(SUM(total_cents), 0)
    FROM orders
    WHERE user_id = u.id
) > (SELECT AVG(total_cents) FROM orders);

-- CTE (Common Table Expression) — cleaner, often same performance
WITH user_spend AS (
    SELECT user_id, COALESCE(SUM(total_cents), 0) AS total
    FROM orders
    GROUP BY user_id
),
avg_spend AS (
    SELECT AVG(total) AS avg_total FROM user_spend
)
SELECT u.email, us.total
FROM users u
JOIN user_spend us ON us.user_id = u.id
CROSS JOIN avg_spend
WHERE us.total > avg_spend.avg_total;

-- Recursive CTE: org chart traversal
WITH RECURSIVE org AS (
    -- Base: top-level (no manager)
    SELECT id, name, manager_id, 0 AS depth
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive: employees whose manager is already in the CTE
    SELECT e.id, e.name, e.manager_id, org.depth + 1
    FROM employees e
    JOIN org ON e.manager_id = org.id
)
SELECT * FROM org ORDER BY depth;
```

---

### Transactions & ACID {#transactions}

```sql
-- Basic transaction
BEGIN;
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;
    UPDATE accounts SET balance = balance + 100 WHERE id = 2;
    -- If anything fails between BEGIN and COMMIT, ROLLBACK is automatic on error
COMMIT;

-- Savepoints — partial rollback
BEGIN;
    INSERT INTO orders (user_id, status) VALUES (42, 'pending') RETURNING id;
    SAVEPOINT after_order;
    INSERT INTO order_items ... ;
    -- If item insert fails:
    ROLLBACK TO SAVEPOINT after_order;
    -- Order is preserved; items rolled back
COMMIT;
```

**ACID** in plain language:

| Property | What it means | How Postgres achieves it |
|---|---|---|
| **Atomicity** | All-or-nothing | Write-ahead log (WAL); rollback on failure |
| **Consistency** | DB moves from one valid state to another | Constraints, triggers, foreign keys checked at commit |
| **Isolation** | Transactions don't see each other's in-progress work | MVCC (each transaction sees a snapshot) |
| **Durability** | Committed data survives crashes | WAL flushed to disk before COMMIT returns |

---

## Indexing Deep Dive {#indexing}

### B-Tree Index {#btree}

The default index type. A balanced tree where each leaf node holds `(key, row_pointer)` pairs, sorted.

```
                   [30 | 70]
                  /    |    \
           [10|20]  [40|60]  [80|90]
```

- **O(log n)** for equality and range queries
- Supports: `=`, `<`, `>`, `<=`, `>=`, `BETWEEN`, `LIKE 'prefix%'`
- Does NOT support: `LIKE '%suffix'`, regex, unordered equality at scale

```sql
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_created ON orders(created_at DESC);

-- Range query — uses B-tree efficiently
SELECT * FROM orders WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31';
```

### Hash Index {#hash-index}

```sql
CREATE INDEX idx_users_email_hash ON users USING HASH (email);
```

- O(1) average for `=` lookups only
- Cannot support range queries or ordering
- In Postgres, B-tree is usually preferred (its equality performance is close enough)

### Composite Indexes & Column Order {#composite}

Column order matters enormously.

```sql
CREATE INDEX idx_orders_user_status ON orders(user_id, status, created_at);
```

This index efficiently answers:
- `WHERE user_id = 5` ✓
- `WHERE user_id = 5 AND status = 'paid'` ✓
- `WHERE user_id = 5 AND status = 'paid' AND created_at > '2024-01-01'` ✓
- `WHERE status = 'paid'` ✗ (skips leading column)
- `WHERE created_at > '2024-01-01'` ✗ (skips leading columns)

**Left-prefix rule**: you can use the index for any prefix of the column list. A range condition on a column stops index usage for subsequent columns.

```sql
-- This uses idx_orders_user_status only for user_id and status (range on created_at stops after)
SELECT * FROM orders
WHERE user_id = 5 AND status = 'paid' AND created_at > '2024-01-01';
```

**Cardinality rule**: put high-cardinality columns first for selective filtering, unless your query pattern dictates otherwise.

### Covering Index {#covering}

An index that contains all columns the query needs — the DB never touches the heap (the actual table).

```sql
-- Query:
SELECT user_id, status, total_cents FROM orders WHERE user_id = 5;

-- Covering index (includes all queried columns):
CREATE INDEX idx_covering ON orders(user_id) INCLUDE (status, total_cents);
-- Postgres 11+ INCLUDE syntax keeps extra cols in leaf only (not searchable)
```

If the index covers the query, you'll see `Index Only Scan` in `EXPLAIN` — the fastest possible path.

### Reading EXPLAIN ANALYZE {#query-plan}

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT u.email, COUNT(o.id)
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at > '2024-01-01'
GROUP BY u.id, u.email;
```

Key terms:

| Term | Meaning |
|---|---|
| `Seq Scan` | Full table scan — missing or unused index |
| `Index Scan` | Uses index, then fetches heap rows |
| `Index Only Scan` | Covering index — never touches heap |
| `Bitmap Index Scan` | Builds a bitmap of matching pages, then fetches (good for low selectivity) |
| `Hash Join` | Builds hash table of smaller relation, probes with larger |
| `Nested Loop` | For each row in outer, probe inner — fast when inner is small |
| `Merge Join` | Both inputs sorted; walk in parallel — great for large sorted sets |
| `cost=X..Y` | Estimated startup cost .. total cost (in arbitrary units) |
| `actual time=X..Y` | Actual wall time in ms |
| `rows=N` | Estimated rows; large discrepancy = stale statistics → run `ANALYZE` |
| `Buffers: hit=N read=M` | N pages from cache, M from disk |

### Index Pitfalls {#index-pitfalls}

```sql
-- 1. Function on indexed column defeats index
WHERE UPPER(email) = 'ALICE@EXAMPLE.COM'   -- bad: can't use idx on email
-- Fix: expression index
CREATE INDEX idx_email_upper ON users(UPPER(email));
-- Or: store lowercase, enforce at insert

-- 2. Implicit type cast
WHERE user_id = '42'   -- user_id is INT; cast defeats index on some DBs

-- 3. Leading wildcard
WHERE name LIKE '%smith'   -- can't use B-tree; use full-text search instead

-- 4. OR can prevent index use
WHERE user_id = 5 OR status = 'paid'
-- Fix: rewrite as UNION
SELECT * FROM orders WHERE user_id = 5
UNION
SELECT * FROM orders WHERE status = 'paid';

-- 5. NULL comparisons
WHERE col IS NULL      -- B-tree indexes NULLs; this is fine in Postgres
WHERE col != 5         -- often causes seq scan (high % of rows match)

-- 6. Too many indexes slow down writes
-- Every INSERT/UPDATE/DELETE must update all indexes on the table
-- Profile write-heavy tables; drop unused indexes
```

---

## Database Concurrency {#concurrency}

### Isolation Levels & Anomalies {#isolation-levels}

| Anomaly | Description | Prevented by |
|---|---|---|
| Dirty read | Read uncommitted data from another tx | Read Committed+ |
| Non-repeatable read | Same row returns different values within tx | Repeatable Read+ |
| Phantom read | Re-running query returns different set of rows | Serializable |
| Write skew | Two txs read overlapping data and each writes based on the read | Serializable |
| Lost update | Two txs read then write same row; one overwrites the other | Depends on locking |

```sql
-- Set isolation level for a session
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN;
...
COMMIT;

-- Or per-transaction
BEGIN ISOLATION LEVEL SERIALIZABLE;
```

**Postgres defaults to Read Committed** — each statement sees a new snapshot of committed data.

**Serializable in Postgres** uses Serializable Snapshot Isolation (SSI) — optimistic, detects serialization conflicts at commit time and aborts one transaction rather than locking.

---

### Locking {#locking}

```sql
-- Explicit row lock (SELECT FOR UPDATE)
BEGIN;
SELECT balance FROM accounts WHERE id = 1 FOR UPDATE;
-- Now you hold an exclusive row lock — no other tx can modify this row
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;

-- Advisory locks — application-level named locks
SELECT pg_advisory_lock(12345);   -- blocks until acquired
-- ... critical section ...
SELECT pg_advisory_unlock(12345);

-- Try-lock (non-blocking)
SELECT pg_try_advisory_lock(12345);   -- returns true/false immediately

-- FOR SHARE — allows other readers but blocks writers
SELECT * FROM orders WHERE id = 42 FOR SHARE;

-- SKIP LOCKED — skip rows locked by other transactions (queue pattern)
SELECT * FROM jobs
WHERE status = 'pending'
ORDER BY created_at
LIMIT 1
FOR UPDATE SKIP LOCKED;
```

**`SKIP LOCKED` pattern** is how you implement a job queue in SQL without a separate queue system — workers grab the next unlocked job.

---

### MVCC {#mvcc}

Multi-Version Concurrency Control — PostgreSQL's key concurrency mechanism.

Instead of locking rows on reads, Postgres keeps **multiple versions** of each row:
- Each row version has `xmin` (transaction that created it) and `xmax` (transaction that deleted/updated it)
- A transaction's snapshot says "I can see rows where `xmin <= my_txid` and `xmax` is either 0 or committed after my snapshot"
- Writers never block readers; readers never block writers

```sql
-- See MVCC internals
SELECT xmin, xmax, cmin, ctid, * FROM orders WHERE id = 42;
-- ctid = (page, offset) — physical location of this row version
```

MVCC creates **dead tuples** — old versions that no longer any snapshot needs to see. `VACUUM` reclaims them.

```sql
-- Check table bloat from dead tuples
SELECT schemaname, tablename,
       n_dead_tup, n_live_tup,
       round(n_dead_tup::numeric / NULLIF(n_live_tup, 0) * 100, 2) AS dead_pct
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;

-- Manual vacuum analyze
VACUUM ANALYZE orders;
```

---

### Deadlocks {#deadlocks}

```sql
-- Transaction 1                  Transaction 2
BEGIN;                             BEGIN;
UPDATE accounts SET ...            UPDATE accounts SET ...
  WHERE id = 1;    -- locks row 1    WHERE id = 2;    -- locks row 2
                                   
UPDATE accounts SET ...            UPDATE accounts SET ...
  WHERE id = 2;    -- WAITS          WHERE id = 1;    -- WAITS → DEADLOCK
```

Postgres detects deadlocks and aborts one transaction (the one that caused the cycle). Your application must handle this:

```python
import psycopg2
from psycopg2 import errors
import time

def transfer(conn, from_id, to_id, amount, retries=3):
    for attempt in range(retries):
        try:
            with conn.cursor() as cur:
                # Canonical deadlock prevention: always lock in ID order
                low, high = min(from_id, to_id), max(from_id, to_id)
                cur.execute("SELECT balance FROM accounts WHERE id = %s FOR UPDATE", (low,))
                cur.execute("SELECT balance FROM accounts WHERE id = %s FOR UPDATE", (high,))
                cur.execute("UPDATE accounts SET balance = balance - %s WHERE id = %s", (amount, from_id))
                cur.execute("UPDATE accounts SET balance = balance + %s WHERE id = %s", (amount, to_id))
            conn.commit()
            return
        except errors.DeadlockDetected:
            conn.rollback()
            time.sleep(0.01 * (2 ** attempt))  # exponential backoff
    raise RuntimeError("Transfer failed after retries")
```

**Deadlock prevention rule**: always acquire locks in a consistent global order (e.g., by primary key ascending).

---

### Optimistic vs Pessimistic Concurrency {#optimistic}

```sql
-- Pessimistic: lock the row immediately
BEGIN;
SELECT * FROM inventory WHERE product_id = 5 FOR UPDATE;
UPDATE inventory SET stock = stock - 1 WHERE product_id = 5;
COMMIT;

-- Optimistic: no lock; detect conflict at update time using version column
-- Schema: inventory has a `version` INT column

-- Read phase (no lock):
SELECT stock, version FROM inventory WHERE product_id = 5;
-- stock=10, version=7

-- Application logic: check stock > 0

-- Write phase: include version in WHERE
UPDATE inventory
SET stock = stock - 1, version = version + 1
WHERE product_id = 5 AND version = 7;
-- If 0 rows updated → someone else updated between read and write → retry
```

```python
def decrement_stock(conn, product_id):
    with conn.cursor() as cur:
        for _ in range(5):
            cur.execute("SELECT stock, version FROM inventory WHERE product_id = %s", (product_id,))
            stock, version = cur.fetchone()
            if stock == 0:
                raise ValueError("Out of stock")
            cur.execute(
                "UPDATE inventory SET stock = stock - 1, version = version + 1 "
                "WHERE product_id = %s AND version = %s",
                (product_id, version)
            )
            if cur.rowcount == 1:
                conn.commit()
                return
            conn.rollback()
        raise RuntimeError("Optimistic lock conflict, max retries exceeded")
```

**When to use which:**
- **Pessimistic**: high contention on same rows (e.g., limited-inventory flash sale), short transactions
- **Optimistic**: low contention, reads >> writes, you can tolerate retries

---

## Python Concurrency {#python-concurrency}

### The GIL {#gil}

The Global Interpreter Lock (GIL) is a mutex that allows only one Python thread to execute bytecode at a time. It exists to make CPython's memory management thread-safe without per-object locking.

**Implications:**
- Threading does NOT parallelize CPU-bound Python code
- Threading DOES work for I/O-bound code (GIL is released during I/O waits)
- For CPU parallelism: use `multiprocessing` or C extensions (numpy releases the GIL)

```
CPU-bound: threads actually slower than single-threaded (GIL contention overhead)
I/O-bound: threads work fine — thread waiting on I/O releases GIL for others
```

Python 3.13+ introduces a free-threaded mode (`python3.13t`) that removes the GIL experimentally.

### Threading {#threading}

```python
import threading
import queue
import time

# Basic thread
def worker(name, delay):
    print(f"{name} started")
    time.sleep(delay)
    print(f"{name} done")

threads = [threading.Thread(target=worker, args=(f"T{i}", i*0.1)) for i in range(5)]
for t in threads: t.start()
for t in threads: t.join()

# Thread-safe producer-consumer
work_queue = queue.Queue(maxsize=10)

def producer():
    for i in range(20):
        work_queue.put(i)          # blocks if queue full
    work_queue.put(None)           # sentinel

def consumer():
    while True:
        item = work_queue.get()    # blocks if queue empty
        if item is None:
            work_queue.put(None)   # pass sentinel to other consumers
            break
        process(item)
        work_queue.task_done()

# Synchronization primitives
lock = threading.Lock()
rlock = threading.RLock()          # reentrant (same thread can acquire multiple times)
event = threading.Event()          # one-shot signal
semaphore = threading.Semaphore(3) # at most 3 concurrent

# Condition variable: wait for a condition while holding a lock
condition = threading.Condition()

def waiter():
    with condition:
        condition.wait_for(lambda: data_ready)  # atomically releases lock and waits
        process_data()

def notifier():
    with condition:
        prepare_data()
        data_ready = True
        condition.notify_all()
```

### asyncio {#asyncio}

Single-threaded cooperative concurrency. No GIL issues; one event loop runs coroutines.

```python
import asyncio
import aiohttp

# Coroutine — defined with async def, called with await
async def fetch(session, url):
    async with session.get(url) as resp:
        return await resp.json()

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)   # run concurrently
    return results

# Run
asyncio.run(fetch_all(["http://api.example.com/1", "http://api.example.com/2"]))

# asyncio.gather vs asyncio.wait
# gather: returns results in order, raises if any task raises (use return_exceptions=True to suppress)
# wait: returns (done, pending) sets — more control over partial completion

# Semaphore to limit concurrency
async def limited_fetch(semaphore, session, url):
    async with semaphore:
        return await fetch(session, url)

async def main():
    sem = asyncio.Semaphore(10)   # max 10 concurrent requests
    async with aiohttp.ClientSession() as s:
        tasks = [limited_fetch(sem, s, url) for url in urls]
        return await asyncio.gather(*tasks)

# async context managers and iterators
class AsyncDB:
    async def __aenter__(self):
        self.conn = await connect()
        return self

    async def __aexit__(self, *args):
        await self.conn.close()

    async def __aiter__(self):
        async for row in self.conn.cursor():
            yield row
```

### Multiprocessing {#multiprocessing}

True parallelism by spawning separate OS processes (each has its own GIL).

```python
from multiprocessing import Pool, Process, Queue, Manager
import os

def cpu_bound_task(n):
    # Each worker runs in its own process with its own interpreter
    return sum(i * i for i in range(n))

# Process pool — like ThreadPoolExecutor but for processes
with Pool(processes=os.cpu_count()) as pool:
    results = pool.map(cpu_bound_task, [10**6, 10**6, 10**6, 10**6])

# Non-blocking: pool.map_async
async_result = pool.map_async(cpu_bound_task, data)
# ... do other work ...
results = async_result.get(timeout=30)

# ProcessPoolExecutor (concurrent.futures — uniform API)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(cpu_bound_task, n): n for n in [10**5, 10**6, 10**7]}
    for future in as_completed(futures):
        n = futures[future]
        print(f"n={n}: {future.result()}")
```

**Choosing the right tool:**

| Scenario | Best choice |
|---|---|
| I/O-bound, many connections | `asyncio` |
| I/O-bound, simple/legacy code | `threading` |
| CPU-bound Python | `multiprocessing` |
| CPU-bound with C extension (numpy) | `threading` (numpy releases GIL) |
| Mix of CPU + I/O | `ProcessPoolExecutor` for CPU tasks, `asyncio` for I/O |

---

## Practical SQL Problems {#practical-sql}

**1. Second-highest salary**
```sql
-- Using OFFSET
SELECT DISTINCT salary FROM employees ORDER BY salary DESC LIMIT 1 OFFSET 1;

-- Using window function (handles ties correctly)
SELECT salary FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) t WHERE rnk = 2;
```

**2. Running total of daily revenue**
```sql
SELECT
    created_at::date AS day,
    SUM(total_cents) AS daily_rev,
    SUM(SUM(total_cents)) OVER (ORDER BY created_at::date ROWS UNBOUNDED PRECEDING) AS running
FROM orders
GROUP BY created_at::date;
```

**3. Users who ordered every product in a set**
```sql
-- Users who bought ALL products in category 5
SELECT user_id
FROM order_items oi
JOIN orders o ON o.id = oi.order_id
JOIN products p ON p.id = oi.product_id
WHERE p.category_id = 5
GROUP BY o.user_id
HAVING COUNT(DISTINCT oi.product_id) = (
    SELECT COUNT(*) FROM products WHERE category_id = 5
);
```

**4. Department-wise top earner**
```sql
SELECT department, name, salary FROM (
    SELECT department, name, salary,
           RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rnk
    FROM employees
) t WHERE rnk = 1;
```

**5. Detect duplicate emails**
```sql
SELECT email, COUNT(*) AS cnt
FROM users
GROUP BY email
HAVING COUNT(*) > 1;
```

**6. Delete duplicates, keep one**
```sql
DELETE FROM users
WHERE id NOT IN (
    SELECT MIN(id)
    FROM users
    GROUP BY email
);

-- Faster with ctid in Postgres
DELETE FROM users a
USING users b
WHERE a.email = b.email AND a.id > b.id;
```

**7. Pivot: orders per status per month**
```sql
SELECT
    DATE_TRUNC('month', created_at) AS month,
    COUNT(*) FILTER (WHERE status = 'paid')      AS paid,
    COUNT(*) FILTER (WHERE status = 'cancelled') AS cancelled,
    COUNT(*) FILTER (WHERE status = 'pending')   AS pending
FROM orders
GROUP BY 1
ORDER BY 1;
```

**8. Find accounts with insufficient funds before transfer**
```sql
-- Safe transfer with constraint check in one query
WITH transfer AS (
    UPDATE accounts
    SET balance = balance - 100
    WHERE id = 1 AND balance >= 100
    RETURNING id, balance
)
UPDATE accounts
SET balance = balance + 100
WHERE id = 2 AND EXISTS (SELECT 1 FROM transfer);
```

**9. Recursive category tree**
```sql
WITH RECURSIVE tree AS (
    SELECT id, name, parent_id, name::text AS path
    FROM categories
    WHERE parent_id IS NULL

    UNION ALL

    SELECT c.id, c.name, c.parent_id, tree.path || ' > ' || c.name
    FROM categories c
    JOIN tree ON c.parent_id = tree.id
)
SELECT * FROM tree ORDER BY path;
```

**10. Moving 7-day average**
```sql
SELECT
    day,
    revenue,
    AVG(revenue) OVER (
        ORDER BY day
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS ma7
FROM daily_revenue;
```
