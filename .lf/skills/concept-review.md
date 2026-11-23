---
extends: builtin
---

For engine, RL, and training infrastructure (`managym` and `manabot` outside
Academy and CLI surfaces), the user is the algorithm or caller, not a human
operator. Judge concepts, APIs, and data layouts by what is optimal for the
training, search, and evaluation algorithms that consume them—throughput,
memory and cache ownership, thread safety, statistical validity, and
composability—rather than human UX.
