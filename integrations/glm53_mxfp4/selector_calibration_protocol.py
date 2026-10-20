"""Define independent, authored prompt splits before collecting any training data."""

from __future__ import annotations

import hashlib
from typing import Any

_CODE = {
    "train": [
        "merge overlapping closed integer intervals, including touching endpoints",
        "perform iterative binary search and return the first matching index",
        "serialize and deserialize a binary tree using a breadth-first traversal",
        "find a cycle in a directed graph using three-color depth-first search",
        "implement an LRU cache using OrderedDict with a strictly positive capacity",
        "parse a simple CSV row with quoted fields and doubled quote escapes",
        "compute Levenshtein distance using two rolling rows",
        "topologically sort a directed acyclic graph using indegrees",
        "validate balanced brackets while ignoring all other characters",
        "combine several sorted iterators lazily with a heap",
        "perform stable deduplication of hashable items without reordering",
        "flatten a nested list iteratively without treating strings as lists",
        "count connected components in an undirected graph",
        "compute integer powers by repeated squaring with a nonnegative exponent",
        "implement a bounded producer-consumer queue with threading.Condition",
        "write a context manager that closes an acquired resource on exceptions",
        "implement an async semaphore-limited map preserving input order",
        "parse a duration containing integer hours, minutes and seconds",
        "implement a trie supporting insert, contains and prefix queries",
        "compute a median from a stream with two heaps",
        "calculate a longest increasing subsequence length in n log n time",
        "apply run-length encoding and decoding to a sequence of integers",
        "perform breadth-first shortest-path search on a rectangular grid",
        "implement union-find with path compression and union by size",
    ],
    "validation": [
        "evaluate an integer reverse-Polish expression using a stack",
        "find the earliest non-repeating character in a string",
        "enumerate unique permutations of a list with duplicate values",
        "implement a binary min-heap without using heapq",
        "convert a nested JSON object into dotted key paths",
        "implement a recursive-descent parser for addition and multiplication",
    ],
    "test": [
        "rotate a square matrix clockwise in place",
        "implement a persistent immutable singly linked list",
        "compute strongly connected components using Kosaraju's algorithm",
        "add two arbitrary-length signed decimal integers stored as strings",
        "calculate a minimum spanning tree using Kruskal's algorithm",
        "implement a file iterator that yields complete UTF-8 lines across byte chunks",
    ],
}

_EXPLANATION = {
    "train": [
        "snapshot isolation and the write-skew anomaly",
        "why TCP flow control and congestion control are separate",
        "how reference counting differs from tracing garbage collection",
        "the consistency guarantees of atomic file replacement",
        "the tradeoffs between a B-tree and an LSM tree",
        "how DNS negative caching affects a newly created record",
        "why floating-point addition depends on evaluation order",
        "the difference between authentication and authorization",
    ],
    "validation": [
        "how database deadlocks can be detected and retried",
        "the limitations of Bloom filters for membership tests",
    ],
    "test": [
        "why a monotonic clock is useful for elapsed-time measurement",
        "how copy-on-write snapshots preserve an older filesystem view",
    ],
}


def cases() -> list[dict[str, Any]]:
    """Return 32 training, eight validation and eight sealed test prompts."""
    result = []
    for split in ("train", "validation", "test"):
        for domain, topics in (("code", _CODE[split]), ("explanation", _EXPLANATION[split])):
            for index, topic in enumerate(topics):
                if domain == "code":
                    prompt = (
                        f"Write Python 3 code to {topic}. State the contract, handle empty "
                        "inputs and edge cases, and include a few assert-based examples. "
                        "Keep the explanation concise."
                    )
                else:
                    prompt = (
                        f"Explain {topic} to an experienced application developer. "
                        "Use a concrete example, describe one common mistake, and keep it concise."
                    )
                result.append({
                    "id": f"{split}-{domain}-{index:02d}", "split": split,
                    "domain": domain, "text": prompt,
                    "text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                })
    assert len({row["text_sha256"] for row in result}) == 48
    return result
