---
title: "DSA Interview Preparation"
date: 2026-04-21
display_order: 13
description: "A comprehensive reference for data structures and algorithms interviews — every major topic with complexity analysis, key patterns, LeetCode problems, and Python implementations."
tags: [dsa, algorithms, interview]
---

<nav class="post-toc" aria-label="On this page">
  <p class="post-toc-title">On this page</p>
  <ul class="post-toc-list">
    <li><a href="#overview">Overview</a></li>
    <li><a href="#arrays-strings">Arrays & Strings</a>
      <ul class="post-toc-sublist">
        <li><a href="#two-pointers">Two Pointers</a></li>
        <li><a href="#sliding-window">Sliding Window</a></li>
        <li><a href="#prefix-sums">Prefix Sums</a></li>
        <li><a href="#kadanes">Kadane's Algorithm</a></li>
        <li><a href="#string-manipulation">String Manipulation</a></li>
      </ul>
    </li>
    <li><a href="#hashmaps-sets">Hash Maps & Sets</a></li>
    <li><a href="#linked-lists">Linked Lists</a></li>
    <li><a href="#stacks-queues">Stacks & Queues</a></li>
    <li><a href="#trees">Trees</a>
      <ul class="post-toc-sublist">
        <li><a href="#tree-traversal">DFS & BFS Traversal</a></li>
        <li><a href="#bst">Binary Search Trees</a></li>
        <li><a href="#tree-problems">Classic Tree Problems</a></li>
      </ul>
    </li>
    <li><a href="#heaps">Heaps & Priority Queues</a></li>
    <li><a href="#graphs">Graphs</a>
      <ul class="post-toc-sublist">
        <li><a href="#graph-bfs-dfs">BFS & DFS</a></li>
        <li><a href="#topological-sort">Topological Sort</a></li>
        <li><a href="#union-find">Union-Find (DSU)</a></li>
        <li><a href="#shortest-paths">Shortest Paths</a></li>
      </ul>
    </li>
    <li><a href="#dynamic-programming">Dynamic Programming</a>
      <ul class="post-toc-sublist">
        <li><a href="#dp-1d">1D DP Patterns</a></li>
        <li><a href="#dp-2d">2D DP Patterns</a></li>
        <li><a href="#dp-classic">Classic DP Problems</a></li>
      </ul>
    </li>
    <li><a href="#backtracking">Backtracking</a></li>
    <li><a href="#binary-search">Binary Search</a></li>
    <li><a href="#tries">Tries</a></li>
    <li><a href="#intervals">Intervals</a></li>
    <li><a href="#bit-manipulation">Bit Manipulation</a></li>
    <li><a href="#greedy">Greedy Algorithms</a></li>
    <li><a href="#math-number-theory">Math & Number Theory</a></li>
  </ul>
</nav>

---

## Overview
{: #overview}

Data structures and algorithms interviews test your ability to decompose a problem, choose the right abstraction, and implement a correct and efficient solution under time pressure. Doing well is not about memorising solutions — it is about recognising *patterns*. Nearly every LeetCode problem is a composition of a small number of canonical patterns: two pointers, sliding window, BFS/DFS, dynamic programming, binary search on the answer space, union-find, and so on.

This reference covers all 15 major topic areas that appear in top-company interviews. For each section you will find:

- The core concept and complexity table
- The key pattern or template in Python
- Representative LeetCode problems with brief solution approaches
- Common mistakes and an interview tip

**How to use this guide.** Read through each section once to build pattern recognition. Then, for each topic, solve the listed problems on your own — without looking at solutions — and return here only to check your approach. Recognising which template to reach for is the skill; coding it cleanly under pressure is what practice builds.

| Complexity | Name | Typical Context |
|---|---|---|
| $O(1)$ | Constant | Hash table lookup, array index |
| $O(\log n)$ | Logarithmic | Binary search, balanced BST |
| $O(n)$ | Linear | Single-pass scan, BFS/DFS on tree |
| $O(n \log n)$ | Linearithmic | Sorting, divide-and-conquer |
| $O(n^2)$ | Quadratic | Nested loops, naive string matching |
| $O(2^n)$ | Exponential | Subset enumeration, backtracking |
| $O(n!)$ | Factorial | Permutation enumeration |

---

## Arrays & Strings
{: #arrays-strings}

Arrays are the most common interview topic. Most array problems reduce to one of five sub-patterns: two pointers, sliding window, prefix sums, Kadane's, or direct string manipulation. Learn to identify which sub-pattern applies within the first minute of reading a problem.

### Two Pointers
{: #two-pointers}

Two pointers work on sorted arrays (or arrays with a monotonic property) by maintaining a left and right index that move toward each other or in the same direction. The key invariant is that the search space shrinks by at least one element per step, giving $O(n)$ time with $O(1)$ extra space.

| Variant | Movement | Use Case |
|---|---|---|
| Opposite ends | Left right, right left | Target sum in sorted array, palindrome check |
| Same direction | Both move right (fast/slow) | Remove duplicates, partition |
| Multiple arrays | One pointer per array | Merge sorted arrays |

**Template — opposite-end two pointers:**

```python
def two_pointer(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        current = arr[left] + arr[right]
        if current == target:
            return [left, right]
        elif current < target:
            left += 1
        else:
            right -= 1
    return []
```

**LC-1 — Two Sum**
Hash map approach: store `{value: index}` as you iterate; for each element check if `target - element` is already in the map. $O(n)$ time, $O(n)$ space. Note: for the sorted-array variant (LC-167), use opposite-end two pointers instead.

**LC-15 — 3Sum**
Sort the array first. Fix one element with an outer loop, then run an opposite-end two-pointer scan on the remainder. Skip duplicate values at each pointer position to avoid duplicate triplets. $O(n^2)$ time.

**LC-42 — Trapping Rain Water**
Two pointers from both ends. Maintain `left_max` and `right_max`. At each step, the side with the smaller max determines the trapped water for that cell: `min(left_max, right_max) - height[i]`. Move the pointer on the side with the smaller max. $O(n)$ time, $O(1)$ space.

**LC-125 — Valid Palindrome**
Two pointers from both ends. Skip non-alphanumeric characters. Compare lowercased characters. $O(n)$ time.

**LC-11 — Container With Most Water**
Opposite-end two pointers. Area at each step is `min(height[l], height[r]) * (r - l)`. Always move the pointer pointing to the shorter line — moving the taller line can only make things worse. $O(n)$ time.

**Common mistakes.** Forgetting to handle duplicate elements in 3Sum leads to duplicate triplets. Off-by-one errors when the while condition is `left < right` vs `left <= right` — use `left < right` for opposite-end, `left <= right` for binary search.

> **Interview tip.** When a problem involves a sorted array and a target value, your first instinct should be two pointers. When the array is unsorted and you need a pair, your first instinct should be a hash map. Sort first if the problem allows mutation of the input or explicitly gives you a sorted input.

### Sliding Window
{: #sliding-window}

A sliding window maintains a contiguous subarray (or substring) satisfying some constraint. The right pointer expands the window; the left pointer contracts it when the constraint is violated. The window size can be fixed or variable. This gives $O(n)$ time because each element enters and exits the window at most once.

| Window Type | When to Use |
|---|---|
| Fixed size $k$ | "Subarray of length $k$" problems |
| Variable size | "Longest/shortest subarray with property X" |

**Template — variable sliding window:**

```python
def sliding_window(s):
    left = 0
    window = {}           # tracks what's in the window
    result = 0
    for right in range(len(s)):
        # expand: add s[right] to window
        window[s[right]] = window.get(s[right], 0) + 1
        # shrink: move left until constraint is satisfied
        while not valid(window):
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1
        # update result with current valid window
        result = max(result, right - left + 1)
    return result
```

**LC-3 — Longest Substring Without Repeating Characters**
Variable window. Expand right; when a duplicate enters the window, advance left until the duplicate is removed. Track characters in a set or frequency map. $O(n)$ time.

**LC-76 — Minimum Window Substring**
Variable window. Maintain a frequency map of characters needed. Track how many distinct characters are fully satisfied. Shrink the window from the left once all characters are covered, recording the minimum. $O(n + m)$ time where $m$ is the pattern length.

**LC-239 — Sliding Window Maximum**
Fixed window of size $k$. Use a monotonic deque (see Stacks & Queues) to track the maximum. The front of the deque is always the index of the maximum in the current window. $O(n)$ time.

**LC-567 — Permutation in String**
Fixed window of size `len(s1)`. Compare frequency maps of the window and `s1`. Use an integer counter of "satisfied" character positions to avoid full map comparison each step. $O(n)$ time.

**LC-424 — Longest Repeating Character Replacement**
Variable window. Track the count of the most frequent character in the window. If `window_size - max_count > k`, the window is invalid — advance left. $O(n)$ time.

**Common mistakes.** Not resetting or correctly updating the frequency map when contracting the window. Confusing "number of distinct characters" with "number of satisfied characters" in LC-76.

> **Interview tip.** The sliding window pattern applies whenever the problem asks for an optimal contiguous subarray/substring and there is a monotonic relationship: making the window larger makes it "more valid" (or less valid). If validity is not monotonic with size, sliding window will not work and you may need DP instead.

### Prefix Sums
{: #prefix-sums}

A prefix sum array `prefix[i]` stores the sum of `arr[0..i-1]`. The sum of any subarray `arr[l..r]` is then `prefix[r+1] - prefix[l]` — a constant-time query after $O(n)$ preprocessing.

```python
def build_prefix(arr):
    prefix = [0] * (len(arr) + 1)
    for i, x in enumerate(arr):
        prefix[i + 1] = prefix[i] + x
    return prefix

# Sum of arr[l..r] inclusive
def range_sum(prefix, l, r):
    return prefix[r + 1] - prefix[l]
```

**LC-303 — Range Sum Query — Immutable**
Build prefix array once; answer each query in $O(1)$.

**LC-560 — Subarray Sum Equals K**
Prefix sum + hash map. Store counts of prefix sums seen so far. For each index, check if `prefix[i] - k` exists in the map — if so, there are that many subarrays ending at `i` with sum `k`. $O(n)$ time.

**LC-238 — Product of Array Except Self**
Two-pass prefix product. First pass fills a `left_product` array; second pass multiplies by a running `right_product`. $O(n)$ time, $O(1)$ extra space (output array doesn't count).

**LC-1248 — Count Number of Nice Subarrays**
Reduce to subarray sum equals $k$ by replacing even numbers with 0 and odd numbers with 1. Then apply the prefix sum hash map technique.

**LC-304 — Range Sum Query 2D — Immutable**
2D prefix sums. `prefix[i][j]` = sum of rectangle from `(0,0)` to `(i-1, j-1)`. Query uses inclusion-exclusion: `prefix[r2+1][c2+1] - prefix[r1][c2+1] - prefix[r2+1][c1] + prefix[r1][c1]`.

**Common mistakes.** Off-by-one indexing — using a 1-indexed prefix array (size $n+1$ with `prefix[0] = 0`) avoids most edge cases. Forgetting to initialise the hash map with `{0: 1}` for the "empty prefix has sum 0" case in LC-560.

> **Interview tip.** Whenever you see "subarray sum equals target" or "number of subarrays with property X", think prefix sums combined with a hash map. The key insight is that a subarray sum can be expressed as a difference of two prefix sums.

### Kadane's Algorithm
{: #kadanes}

Kadane's algorithm finds the maximum sum contiguous subarray in $O(n)$ time and $O(1)$ space. The key insight: the maximum subarray ending at index $i$ is either the element itself, or the element plus the maximum subarray ending at $i-1$.

```python
def max_subarray(nums):
    max_sum = current = nums[0]
    for num in nums[1:]:
        current = max(num, current + num)
        max_sum = max(max_sum, current)
    return max_sum
```

**LC-53 — Maximum Subarray**
Direct application of Kadane's. $O(n)$ time.

**LC-918 — Maximum Sum Circular Subarray**
Either the maximum subarray does not wrap around (standard Kadane's), or it wraps — which is equivalent to the total sum minus the minimum subarray. Take the max of both cases. Edge case: if all elements are negative, return the standard Kadane's result.

**LC-152 — Maximum Product Subarray**
Track both maximum and minimum running products (negative times negative = positive). At each step: `max_prod = max(num, max_prod * num, min_prod * num)` and similarly for `min_prod`. $O(n)$ time.

**Common mistakes.** Initialising `max_sum = 0` instead of `nums[0]` — this fails when all elements are negative. Not tracking the minimum product in LC-152.

> **Interview tip.** Kadane's is the prototypical "optimal substructure without overlapping subproblems" problem — it is DP without a table. If you see a variation asking about products, XOR, or circular arrays, the same two-variable rolling approach applies.

### String Manipulation
{: #string-manipulation}

String problems often combine hashing, sorting, two pointers, or dynamic programming. Key operations: character frequency counting, sliding windows over characters, building strings efficiently with lists.

```python
# Character frequency (faster than Counter for single chars)
from collections import Counter
freq = Counter(s)

# Build string efficiently (avoid repeated concatenation O(n^2))
parts = []
parts.append(char)
result = "".join(parts)
```

**LC-242 — Valid Anagram**
Sort both strings and compare, or compare character frequency maps. $O(n \log n)$ or $O(n)$.

**LC-49 — Group Anagrams**
Use sorted string (or tuple of character counts) as the hash map key. Group strings by key. $O(n k \log k)$ where $k$ is the average string length.

**LC-647 — Palindromic Substrings**
Expand around each centre (there are $2n-1$ centres for odd and even length palindromes). Count each palindrome as the expansion succeeds. $O(n^2)$ time. Manacher's algorithm achieves $O(n)$ but is rarely required in interviews.

**LC-5 — Longest Palindromic Substring**
Same expand-around-centre approach; track the longest palindrome found. $O(n^2)$ time.

**Common mistakes.** Using string concatenation inside a loop — each concatenation is $O(n)$, making the loop $O(n^2)$ total. Always accumulate into a list and join at the end.

> **Interview tip.** Python strings are immutable. Never build a string by repeated `+=` in a loop. Use a list and `"".join()`. This distinction will come up in follow-up questions about your solution's time complexity.

---

## Hash Maps & Sets
{: #hashmaps-sets}

Hash maps (dictionaries in Python) and hash sets provide $O(1)$ average-case insert, lookup, and delete. They are the single most useful data structure in interview problems — if you are doing a nested loop to check membership or count frequencies, a hash map almost certainly gives you a linear-time solution.

| Operation | Average | Worst Case |
|---|---|---|
| Insert | $O(1)$ | $O(n)$ (rehash) |
| Lookup | $O(1)$ | $O(n)$ (collision) |
| Delete | $O(1)$ | $O(n)$ |

**Frequency counting pattern:**

```python
from collections import Counter, defaultdict

# Frequency map
freq = Counter(arr)

# Default dict avoids KeyError
graph = defaultdict(list)
graph[node].append(neighbour)

# Two-sum pattern: complement lookup
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

**LC-1 — Two Sum**
Store `{value: index}` as you iterate. For each number, check if the complement is already in the map. Single pass, $O(n)$ time.

**LC-49 — Group Anagrams**
Use `tuple(sorted(word))` as the key — anagrams share the same sorted form. Group into lists by key. $O(nk \log k)$.

**LC-128 — Longest Consecutive Sequence**
Build a set of all numbers. For each number that has no left-neighbour (`num - 1` not in set), walk right counting the streak. Each number is visited at most twice total. $O(n)$ time.

**LC-349 — Intersection of Two Arrays**
Convert both to sets, return `set(nums1) & set(nums2)`. $O(n + m)$ time.

**LC-387 — First Unique Character in a String**
Count frequencies with a hash map, then scan again to find the first character with frequency 1. $O(n)$ time.

**Common mistakes.** Using a list for membership checks when a set suffices — `x in list` is $O(n)$, `x in set` is $O(1)$. Modifying a dictionary while iterating over it.

> **Interview tip.** The two-sum hash map pattern generalises: whenever you need "does X exist such that f(X) = target", store previously seen values and query for the required complement. This pattern appears in 3Sum (fix one element, two-sum the rest), subarray sum equals K (prefix sum complement), and many graph problems.

---

## Linked Lists
{: #linked-lists}

Linked list problems are about pointer manipulation. The key techniques are: reversal in-place, Floyd's cycle detection (slow/fast pointers), dummy head nodes to simplify edge cases, and merging sorted lists.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

| Operation | Time | Space |
|---|---|---|
| Access by index | $O(n)$ | $O(1)$ |
| Insert/delete at known node | $O(1)$ | $O(1)$ |
| Search | $O(n)$ | $O(1)$ |

**Reversal template:**

```python
def reverse_list(head):
    prev, curr = None, head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

**Floyd's cycle detection:**

```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None
    slow = head                 # reset one pointer to head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow                 # meeting point is cycle start
```

**LC-206 — Reverse Linked List**
Use the reversal template above. Iterative is $O(n)$ time, $O(1)$ space. Recursive is $O(n)$ time, $O(n)$ stack space.

**LC-141 — Linked List Cycle**
Floyd's cycle detection. If `fast` and `slow` ever meet, there is a cycle. $O(n)$ time, $O(1)$ space.

**LC-142 — Linked List Cycle II**
Find the cycle start using Floyd's: after detecting the meeting point, reset one pointer to `head` and advance both one step at a time — they meet at the cycle start. Mathematical proof: if the distance to cycle start is $F$ and cycle length is $C$, the meeting point is $F$ steps from the cycle start.

**LC-21 — Merge Two Sorted Lists**
Use a dummy head node. Advance through both lists, always appending the smaller node. $O(m + n)$ time.

**LC-19 — Remove Nth Node From End of List**
Use two pointers $n$ apart. Move both until the fast pointer reaches the end; slow is then at the node before the one to delete. A dummy head simplifies edge cases. $O(n)$ time.

**LC-143 — Reorder List**
Three steps: (1) find the middle with slow/fast pointers, (2) reverse the second half, (3) interleave the two halves. $O(n)$ time, $O(1)$ space.

**Common mistakes.** Not using a dummy head — this forces special-casing the empty list or operations on the head node. Losing a pointer by overwriting `curr.next` before saving it.

> **Interview tip.** Whenever a linked list problem says "in-place" or has an $O(1)$ space requirement, think about pointer manipulation and the slow/fast pointer pattern. Always draw out the pointer state on paper — visualising which nodes the pointers reference prevents subtle bugs.

---

## Stacks & Queues
{: #stacks-queues}

Stacks (LIFO) and queues (FIFO) are the workhorses of graph traversal and expression evaluation. The most powerful stack technique in interviews is the **monotonic stack** — a stack that maintains elements in sorted order (increasing or decreasing).

| Structure | Python Implementation | Key Operation |
|---|---|---|
| Stack | `list` with `append`/`pop` | LIFO, $O(1)$ |
| Queue | `collections.deque` with `append`/`popleft` | FIFO, $O(1)$ |
| Deque | `collections.deque` | $O(1)$ both ends |
| Priority Queue | `heapq` (min-heap) | $O(\log n)$ push/pop |

**Monotonic stack template (next greater element):**

```python
def next_greater(nums):
    n = len(nums)
    result = [-1] * n
    stack = []          # stores indices, values are decreasing
    for i in range(n):
        # while stack top is less than current element, pop and record
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    return result
```

**LC-20 — Valid Parentheses**
Use a stack. Push open brackets; on a close bracket, check if the top of the stack is the matching opener. If the stack is empty at the end, the string is valid. $O(n)$ time.

**LC-739 — Daily Temperatures**
Monotonic decreasing stack of indices. When a warmer day is found, pop all colder indices and record the difference. $O(n)$ time.

**LC-84 — Largest Rectangle in Histogram**
Monotonic increasing stack of indices. When a bar shorter than the stack top is encountered, pop and compute the area using the popped bar's height and the width between the current index and the new stack top. $O(n)$ time.

**LC-155 — Min Stack**
Maintain a second stack tracking the running minimum alongside the main stack. Each push records the current min; pop restores the previous min. $O(1)$ per operation.

**LC-225 — Implement Stack using Queues**
Two queues: push to `q2`, then move all of `q1` into `q2`, then swap. Pop from `q1`. Each push is $O(n)$; pop is $O(1)$.

**LC-102 — Binary Tree Level Order Traversal**
BFS with a queue. Add the root; at each step, process all nodes at the current level (drain the queue), add their children. Track levels by processing one "generation" at a time. $O(n)$ time.

**Common mistakes.** Using `list.pop(0)` for a queue — this is $O(n)$, not $O(1)$. Always use `collections.deque` with `popleft()`. Forgetting to check if the stack is empty before peeking.

> **Interview tip.** The monotonic stack pattern solves a family of "previous/next smaller/larger element" problems. Recognise this pattern when you see: "for each element, find the nearest element to the left/right that is greater/smaller". The stack maintains candidate elements whose fate (whether they are the answer for some future element) is not yet determined.

---

## Trees
{: #trees}

Trees are hierarchical data structures where each node has zero or more children. Binary trees (at most two children) are the focus of most interview questions. The recursive structure of trees makes recursive solutions natural, but iterative solutions (with an explicit stack or queue) are equally important.

### DFS & BFS Traversal
{: #tree-traversal}

```python
# Recursive DFS — all three traversal orders
def inorder(root):    # left, root, right — gives sorted order in BST
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def preorder(root):   # root, left, right — useful for serialisation
    if not root: return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def postorder(root):  # left, right, root — useful for deletion/evaluation
    if not root: return []
    return postorder(root.left) + postorder(root.right) + [root.val]

# Iterative BFS — level-order traversal
from collections import deque
def level_order(root):
    if not root: return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result
```

### Binary Search Trees
{: #bst}

A BST has the invariant: all values in the left subtree are strictly less than the root, and all values in the right subtree are strictly greater. This enables $O(\log n)$ average-case search, insert, and delete — but degrades to $O(n)$ on a skewed tree.

| Operation | Average (balanced) | Worst (skewed) |
|---|---|---|
| Search | $O(\log n)$ | $O(n)$ |
| Insert | $O(\log n)$ | $O(n)$ |
| Delete | $O(\log n)$ | $O(n)$ |
| Inorder traversal | $O(n)$ | $O(n)$ |

**BST search template:**

```python
def search(root, target):
    if not root or root.val == target:
        return root
    if target < root.val:
        return search(root.left, target)
    return search(root.right, target)
```

### Classic Tree Problems
{: #tree-problems}

**LC-104 — Maximum Depth of Binary Tree**
DFS: `1 + max(max_depth(root.left), max_depth(root.right))`. Base case: `None` returns 0. $O(n)$ time.

**LC-226 — Invert Binary Tree**
Recursively swap left and right children at every node. $O(n)$ time.

**LC-236 — Lowest Common Ancestor of a Binary Tree**
Recursive DFS. If the current node is `p` or `q`, return it. Recurse on both subtrees. If both return non-null, the current node is the LCA. If only one returns non-null, propagate that result upward. $O(n)$ time.

**LC-543 — Diameter of Binary Tree**
At each node, the diameter passing through it is `left_depth + right_depth`. Use a global variable to track the maximum, while the recursive function returns height. $O(n)$ time.

**LC-297 — Serialize and Deserialize Binary Tree**
Preorder DFS. Use a sentinel (e.g., `"#"`) for null nodes. Serialise to a comma-separated string; deserialise by consuming tokens from a queue in preorder sequence. $O(n)$ time and space.

**LC-98 — Validate Binary Search Tree**
DFS with min/max bounds. Pass the valid range down: left subtree must be in `(min, root.val)`, right subtree in `(root.val, max)`. $O(n)$ time. Common mistake: only checking immediate children rather than the full subtree.

**LC-105 — Construct Binary Tree from Preorder and Inorder Traversal**
The first element of preorder is the root. Find it in inorder to split left and right subtrees. Recurse. Use a hash map for $O(1)$ inorder index lookup. $O(n)$ total time.

**Common mistakes.** Confusing preorder/inorder/postorder. In BST validation, checking only direct children rather than propagating valid ranges through the recursion.

> **Interview tip.** Most tree problems are solved by a single DFS that computes multiple quantities in one pass. Before writing code, ask: "what does my recursive function return, and what side effect does it track?" Often you need to return one quantity (e.g., height) while tracking another (e.g., diameter) via a nonlocal variable.

---

## Heaps & Priority Queues
{: #heaps}

A heap is a complete binary tree that maintains the heap property: in a min-heap, every node is smaller than its children. Python's `heapq` module provides a min-heap. For a max-heap, negate values.

| Operation | Time |
|---|---|
| Push | $O(\log n)$ |
| Pop (min) | $O(\log n)$ |
| Peek (min) | $O(1)$ |
| Build heap from list | $O(n)$ |

```python
import heapq

# Min-heap
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
min_val = heapq.heappop(heap)   # returns 1

# Max-heap: negate values
heapq.heappush(heap, -val)
max_val = -heapq.heappop(heap)

# Build heap in-place O(n)
heapq.heapify(arr)

# Top-K smallest elements
top_k = heapq.nsmallest(k, arr)     # O(n log k)
top_k = heapq.nlargest(k, arr)      # O(n log k)
```

**LC-215 — Kth Largest Element in an Array**
Min-heap of size $k$. Iterate through the array; push each element. If heap size exceeds $k$, pop the minimum. At the end, the heap top is the $k$th largest. $O(n \log k)$ time. Alternative: QuickSelect for $O(n)$ average.

**LC-23 — Merge K Sorted Lists**
Push the first node of each list onto a min-heap keyed by node value. Pop the minimum, add it to the result list, and push its successor. $O(N \log k)$ where $N$ is total nodes and $k$ is number of lists.

**LC-295 — Find Median from Data Stream**
Maintain two heaps: a max-heap for the lower half and a min-heap for the upper half. Balance them so their sizes differ by at most one. The median is the top of the larger heap, or the average of both tops if equal. $O(\log n)$ per insertion, $O(1)$ per query.

**LC-347 — Top K Frequent Elements**
Count frequencies with a hash map. Then use a min-heap of size $k$ over `(frequency, element)` pairs — or use `heapq.nlargest(k, freq.items(), key=lambda x: x[1])`. $O(n \log k)$ time.

**LC-378 — Kth Smallest Element in a Sorted Matrix**
Min-heap seeded with the first element of each row. Pop the minimum and push the next element from the same row. After $k$ pops, return the last popped value. $O(k \log n)$ time where $n$ is the number of rows.

**Common mistakes.** Using `heapq` with tuples for ordering — if the first element of two tuples is equal, Python compares the second. For custom objects, provide a comparator key or use `(priority, counter, item)` tuples where `counter` breaks ties.

> **Interview tip.** Heaps shine for "streaming" or "online" problems where data arrives one element at a time and you need the running top-K or median. For static arrays, sorting is often simpler and has the same $O(n \log n)$ complexity — use a heap when the data is dynamic or you need partial sorting.

---

## Graphs
{: #graphs}

Graphs generalise trees: nodes (vertices) connected by edges, which may be directed or undirected, weighted or unweighted, and may contain cycles. Most graph algorithms are built on BFS or DFS.

**Graph representation:**

```python
# Adjacency list (most common)
from collections import defaultdict
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)   # undirected

# Adjacency matrix (dense graphs)
n = 5
adj = [[0] * n for _ in range(n)]
adj[u][v] = weight
```

### BFS & DFS on Graphs
{: #graph-bfs-dfs}

Unlike trees, graphs can have cycles. Always maintain a `visited` set.

```python
# BFS — shortest path in unweighted graph
from collections import deque
def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    dist = {start: 0}
    while queue:
        node = queue.popleft()
        for neighbour in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                dist[neighbour] = dist[node] + 1
                queue.append(neighbour)
    return dist

# DFS — iterative
def dfs(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited: continue
        visited.add(node)
        for neighbour in graph[node]:
            if neighbour not in visited:
                stack.append(neighbour)
```

**LC-200 — Number of Islands**
DFS/BFS on a 2D grid. Treat cells as nodes; edges connect adjacent land cells. Count connected components. Mark visited cells as '0' to avoid revisiting. $O(mn)$ time.

**LC-133 — Clone Graph**
BFS or DFS with a hash map from original node to cloned node. When visiting a node, create its clone if not already in the map, then recurse on its neighbours. $O(V + E)$ time.

**LC-207 — Course Schedule**
Detect a cycle in a directed graph. Use DFS with three states: unvisited (0), in-stack (1), done (2). If DFS encounters a node in state 1, there is a cycle. $O(V + E)$ time.

**LC-994 — Rotting Oranges**
Multi-source BFS. Start with all rotten oranges in the queue simultaneously. BFS spreads rot layer by layer. Track the number of fresh oranges; return the number of BFS rounds needed or -1 if unreachable. $O(mn)$ time.

### Topological Sort
{: #topological-sort}

Topological sort orders vertices of a DAG such that for every directed edge $u \to v$, $u$ comes before $v$. Two approaches: Kahn's algorithm (BFS with in-degrees) and DFS-based (reverse postorder).

```python
from collections import deque

def topological_sort_kahn(n, edges):
    graph = defaultdict(list)
    indegree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1
    queue = deque(i for i in range(n) if indegree[i] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph[node]:
            indegree[neighbour] -= 1
            if indegree[neighbour] == 0:
                queue.append(neighbour)
    return order if len(order) == n else []   # empty if cycle
```

**LC-207 — Course Schedule**
If topological sort produces all $n$ nodes, no cycle exists. Otherwise, return `False`. $O(V + E)$.

**LC-210 — Course Schedule II**
Return the topological order itself, or an empty list if a cycle is detected. Use Kahn's. $O(V + E)$.

**LC-269 — Alien Dictionary**
Build a DAG from the relative ordering of characters in adjacent words. Run topological sort. Watch for contradictions (a word is a prefix of the next but appears after it — invalid).

### Union-Find (DSU)
{: #union-find}

Union-Find (Disjoint Set Union) efficiently answers: "are these two nodes in the same connected component?" It supports union and find in near-$O(1)$ amortised time with path compression and union by rank.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False    # already connected
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

**LC-684 — Redundant Connection**
Process edges one by one; use union-find. The first edge that connects two already-connected nodes is the redundant edge. $O(n \alpha(n)) \approx O(n)$.

**LC-547 — Number of Provinces**
Count the number of distinct roots in the union-find structure after processing all edges. $O(n^2 \alpha(n))$ for a dense adjacency matrix input.

**LC-1202 — Smallest String With Swaps**
All positions in the same connected component can be freely rearranged. Group positions by component, sort characters within each group, and place them back. $O(n \log n)$.

### Shortest Paths
{: #shortest-paths}

| Algorithm | Graph Type | Time |
|---|---|---|
| BFS | Unweighted | $O(V + E)$ |
| Dijkstra | Non-negative weights | $O((V + E) \log V)$ |
| Bellman-Ford | Any weights (detects negative cycles) | $O(VE)$ |
| Floyd-Warshall | All-pairs, dense | $O(V^3)$ |

**Dijkstra's algorithm:**

```python
import heapq
def dijkstra(graph, start, n):
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]          # (distance, node)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]: continue    # stale entry — skip
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    return dist
```

**Bellman-Ford** (handles negative weights):

```python
def bellman_ford(n, edges, start):
    dist = [float('inf')] * n
    dist[start] = 0
    for _ in range(n - 1):           # relax all edges n-1 times
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    # check for negative cycles
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return None             # negative cycle detected
    return dist
```

**LC-743 — Network Delay Time**
Dijkstra from source node; return the maximum distance to any node. If any node is unreachable, return -1. $O((V + E) \log V)$.

**LC-787 — Cheapest Flights Within K Stops**
Modified Bellman-Ford with at most $k+1$ relaxation rounds (not $n-1$). Use a copy of distances to avoid using edges added in the same round. $O(kE)$.

**LC-1091 — Shortest Path in Binary Matrix**
BFS on 8-directional grid. Start from `(0, 0)`, reach `(n-1, n-1)`. BFS guarantees shortest path in unweighted graphs. $O(n^2)$.

**Common mistakes.** In Dijkstra's, not skipping stale entries causes incorrect results and performance degradation. Dijkstra fails on negative weights — use Bellman-Ford instead.

> **Interview tip.** BFS guarantees shortest path only in unweighted (or unit-weight) graphs. For weighted graphs, Dijkstra is the standard choice. Always clarify whether the graph is directed/undirected and whether weights can be negative before choosing an algorithm.

---

## Dynamic Programming
{: #dynamic-programming}

Dynamic programming (DP) solves problems by breaking them into overlapping subproblems and storing results to avoid recomputation. Two equivalent approaches: top-down (memoisation — recursive with a cache) and bottom-up (tabulation — iterative, fill a table in order).

**When to use DP:** the problem asks for an optimal value (min/max/count) and has overlapping subproblems plus optimal substructure (optimal solution to the whole problem depends on optimal solutions to subproblems).

```python
# Top-down memoisation template
from functools import lru_cache

def solve(n):
    @lru_cache(maxsize=None)
    def dp(state):
        if base_case(state):
            return base_value
        return optimal(dp(subproblem1), dp(subproblem2), ...)
    return dp(initial_state)

# Bottom-up tabulation template
def solve(n):
    dp = [0] * (n + 1)
    dp[base] = base_value
    for i in range(1, n + 1):
        dp[i] = optimal(dp[i-1], dp[i-2], ...)
    return dp[n]
```

### 1D DP Patterns
{: #dp-1d}

**LC-70 — Climbing Stairs**
`dp[i] = dp[i-1] + dp[i-2]` — the Fibonacci recurrence. $O(n)$ time, $O(1)$ space with two variables.

**LC-198 — House Robber**
`dp[i] = max(dp[i-1], dp[i-2] + nums[i])`. Either skip house $i$ or rob it (can't rob $i-1$). $O(n)$ time, $O(1)$ space.

**LC-300 — Longest Increasing Subsequence (LIS)**
For each index $i$, `dp[i]` = length of LIS ending at $i$ = `1 + max(dp[j] for j < i if nums[j] < nums[i])`. $O(n^2)$. The patience sorting / binary search approach achieves $O(n \log n)$: maintain a list `tails` where `tails[i]` is the smallest tail of all LIS of length $i+1$; binary search for the insertion position of each element.

```python
import bisect
def length_of_lis(nums):
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

**LC-322 — Coin Change**
`dp[amount] = min(dp[amount - coin] + 1 for coin in coins if amount >= coin)`. Initialise `dp[0] = 0`, all others `inf`. $O(\text{amount} \times |\text{coins}|)$ time.

**LC-139 — Word Break**
`dp[i] = True` if `s[:i]` can be segmented. For each `i`, check all `j < i`: if `dp[j]` is True and `s[j:i]` is in the word set, set `dp[i] = True`. $O(n^2)$ time.

### 2D DP Patterns
{: #dp-2d}

**LC-62 — Unique Paths**
`dp[i][j] = dp[i-1][j] + dp[i][j-1]`. A robot can only move right or down. $O(mn)$ time, $O(n)$ space with 1D rolling array.

**LC-1143 — Longest Common Subsequence (LCS)**
`dp[i][j]` = LCS of `s1[:i]` and `s2[:j]`. If `s1[i-1] == s2[j-1]`, `dp[i][j] = dp[i-1][j-1] + 1`. Else `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`. $O(mn)$ time and space.

**LC-72 — Edit Distance**
`dp[i][j]` = min edits to transform `s1[:i]` to `s2[:j]`. If characters match: `dp[i-1][j-1]`. Else: `1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])` for delete, insert, replace. $O(mn)$.

### Classic DP Problems
{: #dp-classic}

**0/1 Knapsack** — each item can be used at most once:

```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]   # skip item i
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])
    return dp[n][capacity]
```

Space optimised to $O(W)$ by iterating `w` in reverse (right to left) so each item is considered at most once.

**LC-416 — Partition Equal Subset Sum**
Reduce to 0/1 knapsack: can we select a subset that sums to `total // 2`? Use a boolean DP set tracking achievable sums. $O(n \cdot \text{sum})$ time.

**LC-518 — Coin Change II (counting combinations)**
Unbounded knapsack variant. `dp[amount] += dp[amount - coin]`. Process coins in outer loop to avoid counting permutations as distinct. $O(\text{amount} \times |\text{coins}|)$.

**LC-10 — Regular Expression Matching**
`dp[i][j]` = whether `s[:i]` matches `p[:j]`. Careful case analysis for `*` — it can match zero or more of the preceding character. $O(mn)$.

**Common mistakes.** Confusing 0/1 knapsack (iterate weight in reverse) with unbounded knapsack (iterate weight forward). Incorrectly defining the subproblem — the state definition is everything.

> **Interview tip.** Before coding DP, write out: (1) the state definition — what does `dp[i]` or `dp[i][j]` represent? (2) the recurrence — how does the state depend on smaller states? (3) the base cases. Say these out loud to the interviewer. A clear state definition prevents the most common DP bugs.

---

## Backtracking
{: #backtracking}

Backtracking is systematic enumeration: build a solution incrementally, abandon a branch as soon as it violates constraints ("prune"), and backtrack to try the next option. It is $O(b^d)$ in the worst case where $b$ is the branching factor and $d$ is the depth.

**General template:**

```python
def backtrack(state, choices, result):
    if is_complete(state):
        result.append(state[:])   # append a copy
        return
    for choice in choices:
        if is_valid(state, choice):
            state.append(choice)           # make choice
            backtrack(state, next_choices(choice), result)
            state.pop()                    # undo choice
```

**LC-46 — Permutations**
At each step, choose any unused number. Use a `used` boolean array or swap elements in place. $O(n \cdot n!)$ time (there are $n!$ permutations, each taking $O(n)$ to record).

```python
def permute(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i, num in enumerate(remaining):
            path.append(num)
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()
    backtrack([], nums)
    return result
```

**LC-78 — Subsets**
At each step, decide to include or exclude the current element. Total $2^n$ subsets. Start index `i` prevents reuse of earlier elements. $O(2^n \cdot n)$ time.

**LC-39 — Combination Sum**
Candidates can be reused. Pass the current index to allow reuse (same element) but not earlier elements. Prune when running sum exceeds target. $O(2^{t/m})$ where $t$ = target, $m$ = minimum candidate.

**LC-51 — N-Queens**
Track occupied columns, diagonals (`row - col`), and anti-diagonals (`row + col`) in sets. At each row, try each column not blocked by any queen. $O(n!)$ time.

**LC-37 — Sudoku Solver**
At each empty cell, try digits 1–9 that do not violate row/column/box constraints. Track valid digits with three sets per row, column, and 3×3 box. $O(9^m)$ where $m$ is the number of empty cells; constraint propagation makes practical runtime much lower.

**Common mistakes.** Forgetting to make a deep copy when appending to results — `result.append(path)` appends a reference, not a copy. Not pruning early enough (e.g., not sorting candidates first to enable early termination in combination sum).

> **Interview tip.** Backtracking problems often have a "with duplicates" variant (LC-47 Permutations II, LC-40 Combination Sum II). The standard technique: sort the input and skip an element if it equals the previous element at the same recursion depth (i.e., `if i > start and candidates[i] == candidates[i-1]: continue`).

---

## Binary Search
{: #binary-search}

Binary search reduces the search space by half each iteration, achieving $O(\log n)$ time. It applies to: sorted arrays, search spaces with a monotonic property, and the "answer space" (binary search on the answer itself, not an index).

**Canonical template — finding the leftmost target:**

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2    # avoids overflow
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Find leftmost position (lower bound)
def lower_bound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left
```

**Binary search on answer space** — use when you can check "is X feasible?" in $O(n)$:

```python
def solve(nums):
    def feasible(mid):
        # return True if mid is achievable
        ...
    left, right = min_possible, max_possible
    while left < right:
        mid = (left + right) // 2
        if feasible(mid):
            right = mid        # mid works, try smaller
        else:
            left = mid + 1     # mid too small
    return left
```

**LC-704 — Binary Search**
Straightforward application of the canonical template. $O(\log n)$.

**LC-33 — Search in Rotated Sorted Array**
One half is always sorted. Determine which half by comparing `arr[mid]` to `arr[left]`. Binary search within the sorted half; if target is not there, search the other. $O(\log n)$.

**LC-153 — Find Minimum in Rotated Sorted Array**
The minimum is at the inflection point. If `arr[mid] > arr[right]`, the minimum is in the right half. Else it is in the left half (inclusive of mid). $O(\log n)$.

**LC-875 — Koko Eating Bananas**
Binary search on the eating speed. `feasible(speed)` returns True if Koko can eat all bananas in $h$ hours at that speed. $O(n \log W)$ where $W$ is the maximum pile size.

**LC-410 — Split Array Largest Sum**
Binary search on the answer (the largest subarray sum). `feasible(mid)` greedily counts whether the array can be split into at most $m$ subarrays each with sum at most `mid`. $O(n \log(\text{sum}))$.

**LC-4 — Median of Two Sorted Arrays**
Binary search on the partition point of the first array. Ensure the left halves of both arrays contain exactly $(m+n+1)/2$ elements. $O(\log(\min(m, n)))$.

**Common mistakes.** Integer overflow: use `mid = left + (right - left) // 2`, not `(left + right) // 2` (in Python this is not an issue due to arbitrary-precision integers, but good habit). Using `left < right` vs `left <= right` — use `<=` for exact-match search, `<` for bounds-finding.

> **Interview tip.** When a problem asks for the "minimum X such that condition Y holds" or "maximum X such that condition Z holds", binary search on the answer. The condition must be monotonic: if $X$ works, then $X+1$ works (or $X-1$ works). Identifying this monotonic property is the key insight.

---

## Tries
{: #tries}

A trie (prefix tree) stores strings as paths from the root, where each node represents one character. It enables $O(L)$ insert and lookup where $L$ is the string length — independent of the number of stored strings. Tries are ideal for prefix queries, autocomplete, and word search.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

**LC-208 — Implement Trie (Prefix Tree)**
Direct implementation of the class above. $O(L)$ per operation.

**LC-212 — Word Search II**
Build a trie from the word list. DFS on the grid; at each cell, follow the trie path. When a word-end node is reached, record the word. Prune branches not present in the trie. $O(M \cdot N \cdot 4^L)$ where $M \times N$ is the grid and $L$ is the max word length.

**LC-211 — Design Add and Search Words Data Structure**
Trie with wildcard `.` character. When searching, a `.` branches into all children via DFS/recursion. $O(26^L)$ worst case for a pattern of all dots.

**LC-745 — Prefix and Suffix Search**
Augment the trie or use a combined hash map. One approach: for each word, insert all `suffix#word` combinations into a trie keyed by `suffix + '#' + prefix`. $O(L^2)$ per insert.

**LC-472 — Concatenated Words**
For each word, check if it can be formed by other words in the list. Use DP with a trie: `dp[i]` = True if `word[:i]` is constructable. $O(n L^2)$ total.

**Common mistakes.** Using an array of 26 characters instead of a dict for the children — arrays are faster but only work for lowercase ASCII. Forgetting to mark `is_end` on insertion leads to `search` always returning False.

> **Interview tip.** A trie beats a hash set for prefix queries — a hash set requires storing all prefixes explicitly, while a trie stores them implicitly. If the problem involves "find all words matching a prefix" or "check if any stored word starts with X", reach for a trie.

---

## Intervals
{: #intervals}

Interval problems involve ranges $[start, end]$ and operations like merging, inserting, or scheduling. The canonical technique is to sort by start time, then scan linearly.

**Merge intervals template:**

```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)   # overlap — extend
        else:
            merged.append([start, end])
    return merged
```

**LC-56 — Merge Intervals**
Sort by start, then greedily merge overlapping intervals. $O(n \log n)$ for sort, $O(n)$ to merge.

**LC-57 — Insert Interval**
Scan the sorted list of existing intervals. Add all intervals ending before the new interval starts. Merge all intervals overlapping with the new interval. Add the remainder. $O(n)$ time (assuming already sorted).

**LC-252 — Meeting Rooms**
Sort by start time. If any meeting starts before the previous one ends, return False. $O(n \log n)$.

**LC-253 — Meeting Rooms II**
Minimum number of rooms needed. Use a min-heap of end times. Sort meetings by start time; for each meeting, if the earliest-ending current meeting has ended, reuse that room (pop heap). Otherwise, add a new room. $O(n \log n)$. Alternatively, use a sweep line: mark +1 at each start and -1 at each end, scan for the maximum overlap.

**LC-435 — Non-overlapping Intervals**
Greedy: sort by end time. Greedily select the interval with the earliest end that does not overlap the previously selected interval. Count removed = total - selected. $O(n \log n)$.

**Common mistakes.** Sorting by end time vs start time — which to sort by depends on the problem. Merge intervals requires sorting by start; activity selection (remove minimum) requires sorting by end. Handling inclusive vs exclusive endpoints — clarify with the interviewer.

> **Interview tip.** When the problem involves scheduling or resource allocation, think sweep line: create events for start and end of each interval, sort by time, and sweep through counting active intervals. This handles many variants that are awkward with direct interval comparison.

---

## Bit Manipulation
{: #bit-manipulation}

Bit manipulation operates directly on binary representations. Key operations and their complexities are all $O(1)$. Python integers have arbitrary precision, but for interview problems assume 32-bit integers.

| Operation | Syntax | Result |
|---|---|---|
| AND | `a & b` | 1 only where both are 1 |
| OR | `a \| b` | 1 where either is 1 |
| XOR | `a ^ b` | 1 where they differ |
| NOT | `~a` | Flip all bits |
| Left shift | `a << k` | Multiply by $2^k$ |
| Right shift | `a >> k` | Divide by $2^k$ (floor) |
| Clear lowest set bit | `a & (a - 1)` | Strips rightmost 1 |
| Isolate lowest set bit | `a & (-a)` | Only rightmost 1 |
| Check bit $k$ | `(a >> k) & 1` | 0 or 1 |
| Set bit $k$ | `a \| (1 << k)` | Force bit $k$ to 1 |
| Clear bit $k$ | `a & ~(1 << k)` | Force bit $k$ to 0 |

**XOR properties** — the most useful in interviews:
- `x ^ x = 0` (self-cancellation)
- `x ^ 0 = x` (identity)
- XOR is commutative and associative

```python
# Find the single non-duplicate in an array where all others appear twice
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result      # all paired elements cancel out
```

**LC-136 — Single Number**
XOR all elements. Pairs cancel; the unique element remains. $O(n)$ time, $O(1)$ space.

**LC-191 — Number of 1 Bits (Hamming Weight)**
Repeatedly apply `n & (n-1)` to clear the lowest set bit, counting iterations. $O(\text{number of set bits})$. Or use `bin(n).count('1')` in Python.

**LC-231 — Power of Two**
A power of two has exactly one bit set. Check `n > 0 and n & (n-1) == 0`. $O(1)$.

**LC-268 — Missing Number**
XOR all indices 0 to $n$ with all array values. Missing number remains. Alternatively, use the arithmetic formula: `n*(n+1)//2 - sum(nums)`. $O(n)$ time, $O(1)$ space.

**LC-338 — Counting Bits**
`dp[i] = dp[i >> 1] + (i & 1)`. The number of set bits in $i$ equals the number in $i/2$ (right shift drops the last bit) plus the last bit of $i$. $O(n)$ time.

**LC-371 — Sum of Two Integers Without +/-**
Use bit manipulation: `a ^ b` gives the sum without carry; `(a & b) << 1` gives the carry. Repeat until carry is zero. $O(\log(\max(a, b)))$.

**Common mistakes.** In Python, `~n` gives `-(n+1)` due to two's complement semantics with arbitrary precision. For 32-bit operations, mask with `0xFFFFFFFF`. Using `>>` instead of `>>>` (Python has no unsigned right shift).

> **Interview tip.** XOR is the most powerful bit trick in interviews. Memorise: XOR finds the odd-one-out (LC-136), distinguishes two different elements (LC-260), and detects differences between two datasets. The `n & (n-1)` trick for clearing the lowest set bit is the second most useful pattern.

---

## Greedy Algorithms
{: #greedy}

A greedy algorithm makes the locally optimal choice at each step, hoping to reach a global optimum. Unlike DP, it does not reconsider past choices. Greedy works when the problem has the **greedy choice property** (local optimal choices lead to global optimum) and **optimal substructure**.

Proving a greedy algorithm is correct typically uses an exchange argument: assume an optimal solution differs from the greedy solution at some point, then show swapping to the greedy choice cannot make things worse.

**Activity selection / interval scheduling template:**

```python
def activity_selection(intervals):
    # Sort by end time; greedily pick intervals that don't overlap
    intervals.sort(key=lambda x: x[1])
    count = 0
    last_end = float('-inf')
    for start, end in intervals:
        if start >= last_end:
            count += 1
            last_end = end
    return count
```

**LC-55 — Jump Game**
Greedily track the maximum reachable index. If the current index ever exceeds the maximum reachable, return False. $O(n)$ time.

**LC-45 — Jump Game II**
Greedy with two pointers tracking the current reachable boundary and next reachable boundary. Increment jumps each time you exhaust the current boundary. $O(n)$ time.

**LC-435 — Non-overlapping Intervals**
Sort by end time. Greedily keep the interval with the earliest end time that does not conflict with the previous kept interval. Count removals = total - kept. $O(n \log n)$.

**LC-406 — Queue Reconstruction by Height**
Sort by height descending; among equal heights, sort by index ascending. Insert each person at the index equal to their `k` value. Taller people don't affect shorter people's visibility count. $O(n^2)$ insertion (use a list or skip list for $O(n \log n)$).

**LC-134 — Gas Station**
If total gas >= total cost, a solution exists. Find the starting station: greedily track the running tank; when it goes negative, reset and try the next station as start. $O(n)$ time.

**Common mistakes.** Applying greedy to a problem that requires DP — greedy fails when past choices constrain future options in complex ways. Always verify the greedy choice property before trusting a greedy solution.

> **Interview tip.** Greedy problems often involve sorting by one criterion (end time, ratio, height, etc.) and then making a single linear pass. If you find yourself reaching for DP, step back and check whether a greedy sort-and-scan would work — it often does for scheduling and optimisation problems.

---

## Math & Number Theory
{: #math-number-theory}

Mathematical insight often reduces an $O(n^2)$ or exponential problem to $O(n \log n)$ or better. The key tools are GCD, prime sieves, and modular arithmetic.

| Concept | Formula / Algorithm | Time |
|---|---|---|
| GCD (Euclidean) | `gcd(a, b) = gcd(b, a % b)` | $O(\log \min(a, b))$ |
| LCM | `lcm(a, b) = a * b // gcd(a, b)` | $O(\log \min(a, b))$ |
| Prime sieve | Sieve of Eratosthenes | $O(n \log \log n)$ |
| Modular exponentiation | Fast power | $O(\log n)$ |
| Modular inverse | Fermat's little theorem (prime modulus) | $O(\log p)$ |

**GCD — Euclidean algorithm:**

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Python built-in
from math import gcd
```

**Sieve of Eratosthenes:**

```python
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]
```

**Modular arithmetic** — essential when answers involve large numbers and the problem says "return modulo $10^9 + 7$":

```python
MOD = 10**9 + 7

# Modular exponentiation
def power(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result

# Python built-in
pow(base, exp, mod)   # O(log exp)
```

**LC-204 — Count Primes**
Sieve of Eratosthenes up to $n$. Count primes. $O(n \log \log n)$ time.

**LC-172 — Factorial Trailing Zeroes**
Each trailing zero comes from a factor of 10 = 2 × 5. Fives are the bottleneck. Count factors of 5 in $n!$: `n//5 + n//25 + n//125 + ...`. $O(\log n)$.

**LC-50 — Pow(x, n)**
Modular exponentiation / fast power. Handle negative $n$ by computing `1 / power(x, -n)`. $O(\log n)$.

**LC-149 — Max Points on a Line**
For each pair of points, compute the slope as a reduced fraction `(dy/gcd, dx/gcd)` to avoid floating-point issues. Use a hash map per anchor point. $O(n^2)$ time.

**LC-263 — Ugly Number**
Repeatedly divide by 2, 3, and 5; check if the result is 1. $O(\log n)$ time.

**LC-279 — Perfect Squares**
DP: `dp[n] = min(dp[n - i^2] + 1 for i in range(1, sqrt(n)+1))`. Alternatively, Lagrange's four-square theorem (every number is the sum of four squares) gives an $O(\sqrt{n})$ mathematical solution.

**Common mistakes.** Integer overflow in languages without arbitrary precision — in Python this is not an issue, but multiplying before taking modulo can cause issues in other languages. Using floating-point for slope comparisons — always represent slopes as reduced-fraction tuples.

> **Interview tip.** Modular arithmetic problems follow the pattern: `(a + b) % m`, `(a * b) % m`, and `(a - b + m) % m` (the `+m` prevents negative results). For division under modulo, multiply by the modular inverse — which exists only when the modulus is prime. Python's three-argument `pow(base, exp, mod)` is the cleanest way to compute modular exponentiation.

---

## Closing Notes

Every problem in a DSA interview maps to a small number of templates. When you are stuck:

1. **Classify the problem.** What are you optimising (min/max/count/existence)? What is the input structure (sorted array, graph, tree, string)?

2. **Identify the pattern.** Sorted array with a target — two pointers or binary search. Contiguous subarray — sliding window or prefix sums. Tree — DFS or BFS. Optimal substructure — DP or greedy. Connectivity — union-find or BFS/DFS.

3. **State the brute force first.** Always mention the naive solution and its complexity. Then optimise.

4. **Verify with small examples.** Run through a 3–4 element example by hand before coding. Check edge cases: empty input, single element, all elements equal, already sorted.

5. **Analyse complexity out loud.** After coding, walk through time and space complexity. Interviewers listen for you to correctly identify bottlenecks.

The patterns compound: the hardest problems (LC-Hard) are usually two or three of these patterns composed together — a BFS over a graph whose nodes are DP states, or a binary search over an answer verified by a greedy scan. Build fluency with each pattern in isolation; composing them under pressure will follow naturally.
