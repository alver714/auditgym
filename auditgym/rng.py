"""Deterministic, label-scoped random streams. Never use hash() or wall-clock for randomness."""
from __future__ import annotations

import hashlib
import random


def derive_int(seed: int, *labels: object) -> int:
    h = hashlib.sha256(repr((int(seed),) + tuple(str(x) for x in labels)).encode()).digest()
    return int.from_bytes(h[:8], "big")


def sub_rng(seed: int, *labels: object) -> random.Random:
    """Independent stream for (seed, labels...). Adding a new label elsewhere never shifts this one."""
    return random.Random(derive_int(seed, *labels))


def hex_id(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))
