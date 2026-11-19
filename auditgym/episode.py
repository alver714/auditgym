"""In-memory episode container plus on-disk (de)serialisation.

Public part  -> episodes/<split>/<episode_id>/   (what an agent could ever see)
Private part -> grader_data/<split>/<episode_id>/ (truth.json, provenance.json; grader only)

The two roots are deliberately separate so no public path ever contains a private file.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any

from .constants import (FX_FIELDS, INVOICE_FIELDS, LEDGER_FIELDS, PAYMENT_FIELDS,
                        TABLE_FX, TABLE_INVOICES, TABLE_LEDGER, TABLE_PAYMENTS, fx_id)

PUBLIC_FILES = ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl",
                "fx.csv", "policy.json")
PRIVATE_FILES = ("truth.json", "provenance.json")

_INT_LEDGER = ("debit_minor", "credit_minor")
_NULLABLE_LEDGER = ("entity_id", "invoice_id", "payment_id")


@dataclass
class Episode:
    manifest: dict
    policy: dict
    ledger: list[dict]
    invoices: list[dict]
    payments: list[dict]
    fx: list[dict]
    truth: dict = field(default_factory=dict)        # PRIVATE
    provenance: dict = field(default_factory=dict)   # PRIVATE

    @property
    def episode_id(self) -> str:
        return self.manifest["episode_id"]

    # ------------------------------------------------------------ public views
    def tables(self) -> dict[str, list[dict]]:
        return {TABLE_LEDGER: self.ledger, TABLE_INVOICES: self.invoices,
                TABLE_PAYMENTS: self.payments, TABLE_FX: self.fx}

    def public_ids(self) -> set[str]:
        """Every id an agent may legitimately cite as evidence."""
        ids: set[str] = set()
        ids.update(r["row_id"] for r in self.ledger)
        ids.update(d["doc_id"] for d in self.invoices)
        ids.update(d["doc_id"] for d in self.payments)
        ids.update(fx_id(r["date"], r["source_currency"], r["target_currency"]) for r in self.fx)
        return ids

    def public_copy(self) -> "Episode":
        return Episode(copy.deepcopy(self.manifest), copy.deepcopy(self.policy),
                       copy.deepcopy(self.ledger), copy.deepcopy(self.invoices),
                       copy.deepcopy(self.payments), copy.deepcopy(self.fx))

    def public_hash(self) -> str:
        h = hashlib.sha256()
        for name, blob in sorted(render_public(self).items()):
            h.update(name.encode()); h.update(b"\0"); h.update(blob.encode()); h.update(b"\0")
        return h.hexdigest()


# ---------------------------------------------------------------- canonical JSON
def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _csv_text(rows: list[dict], fields: tuple[str, ...]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(fields)
    for r in rows:
        w.writerow(["" if r.get(f) is None else r[f] for f in fields])
    return buf.getvalue()


def _jsonl_text(rows: list[dict], fields: tuple[str, ...]) -> str:
    return "".join(dumps({f: r.get(f) for f in fields}) + "\n" for r in rows)


def render_public(ep: Episode) -> dict[str, str]:
    return {
        "manifest.json": json.dumps(ep.manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "policy.json": json.dumps(ep.policy, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        "ledger.csv": _csv_text(ep.ledger, LEDGER_FIELDS),
        "invoices.jsonl": _jsonl_text(ep.invoices, INVOICE_FIELDS),
        "payments.jsonl": _jsonl_text(ep.payments, PAYMENT_FIELDS),
        "fx.csv": _csv_text(ep.fx, FX_FIELDS),
    }


def save_episode(ep: Episode, public_root: str, private_root: str) -> tuple[str, str]:
    pub = os.path.join(public_root, ep.episode_id)
    priv = os.path.join(private_root, ep.episode_id)
    os.makedirs(pub, exist_ok=True)
    os.makedirs(priv, exist_ok=True)
    for name, text in render_public(ep).items():
        with open(os.path.join(pub, name), "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(priv, "truth.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(ep.truth, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    with open(os.path.join(priv, "provenance.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(ep.provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return pub, priv


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_public(pub_dir: str) -> Episode:
    with open(os.path.join(pub_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(pub_dir, "policy.json"), encoding="utf-8") as f:
        policy = json.load(f)
    ledger = []
    for r in _read_csv(os.path.join(pub_dir, "ledger.csv")):
        for k in _INT_LEDGER:
            r[k] = int(r[k])
        for k in _NULLABLE_LEDGER:
            r[k] = r[k] or None
        ledger.append(r)
    fx = _read_csv(os.path.join(pub_dir, "fx.csv"))
    return Episode(manifest, policy, ledger,
                   _read_jsonl(os.path.join(pub_dir, "invoices.jsonl")),
                   _read_jsonl(os.path.join(pub_dir, "payments.jsonl")), fx)


def load_episode(pub_dir: str, priv_dir: str | None) -> Episode:
    ep = load_public(pub_dir)
    if priv_dir:
        with open(os.path.join(priv_dir, "truth.json"), encoding="utf-8") as f:
            ep.truth = json.load(f)
        with open(os.path.join(priv_dir, "provenance.json"), encoding="utf-8") as f:
            ep.provenance = json.load(f)
    return ep
