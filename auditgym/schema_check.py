"""Tiny stdlib JSON-Schema checker for the files in schemas/ (no third-party dependency).

Supports the subset the schemas use: type (incl. unions), required, properties,
additionalProperties (bool), items, enum, const, pattern, minimum, maximum, minLength, maxLength,
minItems, maxItems, oneOf and $ref to a sibling schema file.

    python -m auditgym.schema_check episodes/train/<episode_id>     # public files
    python -m auditgym.schema_check episodes/train/<id> grader_data/train/<id>   # + truth/provenance
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

SCHEMA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas")
_cache: dict[str, dict] = {}


def load_schema(name: str) -> dict:
    if name not in _cache:
        with open(os.path.join(SCHEMA_DIR, name), encoding="utf-8") as f:
            _cache[name] = json.load(f)
    return _cache[name]


def _type_ok(v: Any, t: str) -> bool:
    if t == "object":
        return isinstance(v, dict)
    if t == "array":
        return isinstance(v, list)
    if t == "string":
        return isinstance(v, str)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "boolean":
        return isinstance(v, bool)
    if t == "null":
        return v is None
    return False


def check(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Return a list of violation strings (empty list = valid)."""
    if "$ref" in schema:
        return check(value, load_schema(schema["$ref"]), path)
    errs: list[str] = []
    t = schema.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_type_ok(value, x) for x in types):
            return [f"{path}: expected {types}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        errs.append(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errs.append(f"{path}: {value!r} not in {schema['enum']}")
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{path}: {value[:40]!r} does not match {schema['pattern']}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            errs.append(f"{path}: too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errs.append(f"{path}: too long")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errs.append(f"{path}: {value} < {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errs.append(f"{path}: {value} > {schema['maximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errs.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errs.append(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                errs += check(v, schema["items"], f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                errs.append(f"{path}: missing key {k!r}")
        for k, v in value.items():
            if k in props:
                errs += check(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected key {k!r}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errs += check(v, schema["additionalProperties"], f"{path}.{k}")
    if "oneOf" in schema:
        ok = [i for i, sub in enumerate(schema["oneOf"]) if not check(value, sub, path)]
        if len(ok) != 1:
            errs.append(f"{path}: matches {len(ok)} of the oneOf alternatives (need exactly 1)")
    return errs


def check_episode(ep, private: bool = True, limit: int = 20) -> list[str]:
    """Validate every public file of an Episode (and truth / provenance when present)."""
    errs: list[str] = []

    def run(label: str, value: Any, schema_name: str) -> None:
        for e in check(value, load_schema(schema_name))[:3]:
            errs.append(f"{label}: {e}")

    run("manifest", ep.manifest, "manifest.schema.json")
    run("policy", ep.policy, "policy.schema.json")
    for label, rows, sch in (("ledger", ep.ledger, "ledger_row.schema.json"),
                             ("invoices", ep.invoices, "invoice.schema.json"),
                             ("payments", ep.payments, "payment.schema.json"),
                             ("fx", ep.fx, "fx_row.schema.json")):
        n_bad = 0
        for i, r in enumerate(rows):
            es = check(r, load_schema(sch))
            if es and n_bad < limit:
                errs.append(f"{label}[{i}]: {es[0]}")
            n_bad += bool(es)
    for r in ep.ledger:
        if (r["debit_minor"] > 0) == (r["credit_minor"] > 0):
            errs.append(f"ledger {r['row_id']}: not exactly one of debit/credit > 0")
            break
    if private:
        if ep.truth is not None:
            run("truth", ep.truth, "truth.schema.json")
        if ep.provenance is not None:
            run("provenance", ep.provenance, "provenance.schema.json")
    return errs


def main(argv: list[str] | None = None) -> int:
    from .episode import load_episode, load_public
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    ep = load_episode(args[0], args[1]) if len(args) > 1 else load_public(args[0])
    errs = check_episode(ep, private=len(args) > 1)
    for e in errs:
        print("SCHEMA VIOLATION", e)
    print("schema check:", "FAIL" if errs else "ok", f"({ep.episode_id})")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
