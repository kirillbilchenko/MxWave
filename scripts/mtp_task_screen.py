"""Authored, automatically scored diagnostic tasks; not a benchmark accuracy suite."""

from __future__ import annotations

import argparse
import ast
import json
import re
import resource
import subprocess
import sys
from typing import Any


def task_cases() -> list[dict[str, Any]]:
    """Return eight arithmetic and eight Python tasks with fixed answer contracts."""
    math = [
        (
            (
                "An account starts with 1200 dollars, grows by 5% each year for two years, "
                "and then pays out 300 dollars. What balance remains?"
            ),
            "1023",
        ),
        (
            (
                "A cyclist travels 45 km at 15 km/h and 60 km at 20 km/h. "
                "What is the average speed over the whole journey in km/h?"
            ),
            "17.5",
        ),
        (
            (
                "A store buys 60 items at 8 dollars each. It sells 45 at 12 dollars each "
                "and the rest at 6 dollars each. What is its profit in dollars?"
            ),
            "150",
        ),
        (
            (
                "A bag has 5 red, 3 blue and 2 green balls. Two balls are drawn without "
                "replacement. What is the probability both are red? Give a reduced fraction."
            ),
            "2/9",
        ),
        (
            (
                "Machine A makes 18 parts an hour and machine B makes 12. They work together "
                "for 4 hours, then B alone for 3 hours. How many parts are made?"
            ),
            "156",
        ),
        ("Five consecutive odd positive integers sum to 175. What is the largest?", "39"),
        (
            (
                "A rectangle has perimeter 54 cm. Its length is twice its width plus 3 cm. "
                "What is its area in square centimetres?"
            ),
            "152",
        ),
        (
            (
                "A bus initially has 38 passengers. At each of three stops, 7 get off and "
                "4 get on. How many passengers remain after the third stop?"
            ),
            "29",
        ),
    ]
    codes = [
        (
            "stable_unique",
            "values",
            "Return distinct integers in first-occurrence order.",
            [([[3, 1, 3, 2, 1]], [3, 1, 2]), ([[]], []), ([[0, 0, -1]], [0, -1])],
        ),
        (
            "rotate_right",
            "values, steps",
            (
                "Return a new list rotated right by steps. "
                "Support negative steps and empty input without modifying values."
            ),
            [
                ([[1, 2, 3, 4], 1], [4, 1, 2, 3]),
                ([[1, 2, 3], -1], [2, 3, 1]),
                ([[], 5], []),
                ([[1, 2, 3], 8], [2, 3, 1]),
            ],
        ),
        (
            "run_lengths",
            "text",
            (
                "Return a list of [character, count] pairs for consecutive "
                "runs. Empty input returns an empty list."
            ),
            [
                (["aaabbcaa"], [["a", 3], ["b", 2], ["c", 1], ["a", 2]]),
                ([""], []),
                (["xx"], [["x", 2]]),
            ],
        ),
        (
            "merge_counts",
            "left, right",
            (
                "Return a new dictionary summing integer counts "
                "for each key. Do not modify either input dictionary."
            ),
            [
                ([{"a": 2, "b": 1}, {"b": 4, "c": 3}], {"a": 2, "b": 5, "c": 3}),
                ([{}, {"z": 0}], {"z": 0}),
                ([{"q": -2}, {"q": 2}], {"q": 0}),
            ],
        ),
        (
            "first_missing",
            "values",
            (
                "Return the smallest positive integer missing from "
                "the integer list; support duplicates and negative integers."
            ),
            [([[3, 4, -1, 1]], 2), ([[1, 2, 0]], 3), ([[]], 1), ([[1, 1, 2]], 3)],
        ),
        (
            "is_balanced",
            "text",
            (
                "Check balanced parentheses, square brackets and braces, "
                "ignoring all other characters. Return a boolean."
            ),
            [
                (["a{b[c](d)}"], True),
                (["([)]"], False),
                ([""], True),
                (["text]"], False),
                (["(()"], False),
            ],
        ),
        (
            "transpose",
            "rows",
            (
                "Return the transpose of a rectangular list of lists. "
                "Empty input returns []; rows may contain arbitrary values."
            ),
            [
                ([[[1, 2, 3], [4, 5, 6]]], [[1, 4], [2, 5], [3, 6]]),
                ([[]], []),
                ([[[]]], []),
                ([[["a"], ["b"]]], [["a", "b"]]),
            ],
        ),
        (
            "window_sums",
            "values, width",
            (
                "Return sums of every contiguous window of "
                "the positive integer width. If width exceeds the input length, return []."
            ),
            [
                ([[1, 2, 3, 4], 2], [3, 5, 7]),
                ([[2, -1, 4], 1], [2, -1, 4]),
                ([[], 2], []),
                ([[1, 2], 3], []),
            ],
        ),
    ]
    return [
        {
            "id": f"math-{i}",
            "category": "math",
            "content": prompt
            + " Return only the number or reduced fraction, without units or explanation.",
            "answer": answer,
        }
        for i, (prompt, answer) in enumerate(math)
    ] + [
        {
            "id": f"code-{i}",
            "category": "code",
            "function": name,
            "content": f"Implement Python def {name}({arguments}): {description} "
            "Do not modify inputs. Return only function definitions, no imports, markdown "
            "or example calls.",
            "tests": [{"args": args, "expected": expected} for args, expected in tests],
        }
        for i, (name, arguments, description, tests) in enumerate(codes)
    ]


def _check_source(source: str) -> str:
    source = source.strip()
    match = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", source, re.DOTALL)
    if match:
        source = match.group(1)
    if len(source) > 20000:
        raise ValueError("Code exceeds diagnostic limit")
    tree = ast.parse(source)
    if not tree.body or any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        raise ValueError("Only function definitions are allowed")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal)):
            raise TypeError("Unsupported code construct")
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise ValueError("Decorators are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private attributes are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Private names are not allowed")
    return source


def _worker(payload: dict[str, Any]) -> dict[str, Any]:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    # macOS does not support lowering RLIMIT_AS; Spark's Linux worker does.
    if sys.platform == "linux":
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    allowed = (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "max",
        "min",
        "next",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "Exception",
        "ValueError",
        "TypeError",
    )
    import builtins

    namespace: dict[str, Any] = {"__builtins__": {k: getattr(builtins, k) for k in allowed}}
    # The worker is isolated, resource-limited and restricted to public builtins.
    exec(compile(_check_source(payload["text"]), "<generated-task>", "exec"), namespace)  # noqa: S102
    function = namespace[payload["case"]["function"]]
    outcomes = []
    for test in payload["case"]["tests"]:
        arguments = json.loads(json.dumps(test["args"]))
        before = json.loads(json.dumps(arguments))
        actual = function(*arguments)
        outcomes.append(
            {"passed": actual == test["expected"] and arguments == before, "actual": actual}
        )
    return {"passed": all(item["passed"] for item in outcomes), "tests": outcomes}


def score(case: dict[str, Any], text: str) -> dict[str, Any]:
    """Score an exact numeric answer or isolated, bounded Python unit cases."""
    if case["category"] == "math":
        return {"passed": text.strip() == case["answer"], "answer": text.strip()}
    try:
        _check_source(text)
        result = subprocess.run(
            [sys.executable, "-I", str(__file__), "--worker"],
            input=json.dumps({"case": case, "text": text}),
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
        if result.returncode:
            return {"passed": False, "error": f"worker exit {result.returncode}"}
        return json.loads(result.stdout)
    except (TypeError, ValueError, SyntaxError, subprocess.TimeoutExpired) as error:
        return {"passed": False, "error": str(error)[:300]}


def main() -> None:
    """Run only the bounded scoring worker; generated code cannot import modules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.parse_args()
    try:
        value = _worker(json.load(sys.stdin))
    except Exception as error:  # noqa: BLE001 - generated task errors are scored failures
        value = {"passed": False, "error": f"{type(error).__name__}: {error}"[:300]}
    print(json.dumps(value))


if __name__ == "__main__":
    main()
