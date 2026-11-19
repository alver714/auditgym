"""Benign, truth-preserving episode transforms used by paired checks.

rename_ids(ep, salt)  -> deep copy with every id consistently remapped (public records, truth, provenance).
                         A pure relabelling: record ORDER is untouched, so it commutes with ``permute``.
permute(ep, salt)     -> deep copy with record order shuffled per table. The truth object (issues, amounts,
                         evidence, ids) is not rewritten, but the ORDER in which issues surface in the
                         tables - and the order a scripted policy meets them in - is permuted.
Both are deterministic in (episode, salt) and never mutate their input.

Two rename modes
----------------
* ``"shape"`` (default; any int / str salt): every id keeps its prefix, length and character classes
  (``INV-2026-0042`` -> ``INV-2026-7731``). Cheap sanity check that nothing hard-codes concrete ids.
* ``"aggressive"`` (salt ``"aggressive:<n>"`` or ``mode="aggressive"``): every id is replaced by an id of a
  completely different format - different prefix, length, character set AND lexicographic sort order
  (numeric-only ids and ``z`` + hex ids, mixed within one table). A policy that parses ids, assumes a
  prefix, a width, or that sorted ids follow chronology, fails this transform. The salt is a plain JSON
  string, so it travels through the unchanged ``reset(options={"transform": {"rename_salt": ...}})`` wire
  format (the subprocess grader accepts any JSON salt).

What is intentionally NOT renamed: ``fx:<date>:<src>:<dst>`` keys. They are natural keys (a pure function of
the public ``date`` / ``source_currency`` / ``target_currency`` columns, see ``constants.fx_id``), not
opaque identifiers: renaming them would break the contract that an fx evidence id is derivable from the row.
Id-format independence is therefore tested for every opaque id (row, entity, invoice, payment, document),
and fx keys are kept identical in truth and in observations.
"""
from __future__ import annotations

import copy
import re

from .content_id import retag
from .episode import Episode
from .rng import sub_rng

# literal, non-random id prefixes produced by the generator's id styles (kept so ids stay well-formed)
_PREFIX = re.compile(r"^(INV-\d{4}-|INV-|SI/|F-|CUST-|C(?=\d)|PAY-|RCPT-|BNK|VND-|D-|R-)(.*)$")
_ID_FIELDS = ("row_id", "entity_id", "invoice_id", "payment_id", "doc_id")


RENAME_MODES = ("shape", "aggressive")
AGGRESSIVE_PREFIX = "aggressive:"


def aggressive_salt(n) -> str:
    """The wire-format salt that selects the aggressive rename mode (a plain JSON string)."""
    return f"{AGGRESSIVE_PREFIX}{n}"


def split_rename_salt(salt, mode: str | None = None) -> tuple[str, object]:
    """``(mode, salt)``: ``"aggressive:<n>"`` strings (or ``mode="aggressive"``) select the aggressive mode."""
    if mode is None:
        mode = "aggressive" if isinstance(salt, str) and salt.startswith(AGGRESSIVE_PREFIX) else "shape"
    if mode not in RENAME_MODES:
        raise ValueError(f"unknown rename mode {mode!r}; choose from {RENAME_MODES}")
    return mode, salt


def _remap_aggressive(old: str, salt, taken: set[str]) -> str:
    """Format-destroying remap: numeric-only (9-14 digits) or ``z`` + 8-20 hex chars, chosen per id.

    The result shares no prefix, length class, charset or sort position with the original, and the two
    styles interleave inside one table (so neither "all ids look alike" nor "ids sort chronologically"
    can be relied on)."""
    for attempt in range(1000):
        r = sub_rng(0, "rename_ids_aggressive", salt, old, attempt)
        if r.random() < 0.5:
            new = r.choice("123456789") + "".join(r.choice("0123456789") for _ in range(r.randint(8, 13)))
        else:
            new = "z" + "".join(r.choice("0123456789abcdef") for _ in range(r.randint(8, 20)))
        if new != old and new not in taken:
            return new
    raise RuntimeError("could not remap id " + old)


def _remap_one(old: str, salt, taken: set[str], mode: str = "shape") -> str:
    if mode == "aggressive":
        return _remap_aggressive(old, salt, taken)
    m = _PREFIX.match(old)
    prefix, tail = (m.group(1), m.group(2)) if m else ("", old)
    for attempt in range(1000):
        r = sub_rng(0, "rename_ids", salt, old, attempt)
        out = []
        for ch in tail:
            if ch.isdigit():
                out.append(r.choice("0123456789"))
            elif "A" <= ch <= "Z":
                out.append(r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            elif ch in "abcdef":
                out.append(r.choice("abcdef0123456789"))
            elif "a" <= ch <= "z":
                out.append(r.choice("abcdefghijklmnopqrstuvwxyz"))
            else:
                out.append(ch)
        new = prefix + "".join(out)
        if new != old and new not in taken:
            return new
    raise RuntimeError("could not remap id " + old)


def _collect_ids(ep: Episode) -> list[str]:
    seen: dict[str, None] = {}

    def add(v):
        if isinstance(v, str) and v:
            seen.setdefault(v, None)

    def tables(ledger, invoices, payments):
        for r in ledger:
            for f in ("row_id", "entity_id", "invoice_id", "payment_id"):
                add(r.get(f))
        for d in invoices:
            for f in ("doc_id", "invoice_id", "entity_id"):
                add(d.get(f))
        for d in payments:
            add(d.get("doc_id"))
            add(d.get("payment_id"))
            for ref in d.get("invoice_refs") or []:
                add(ref.get("invoice_id"))

    tables(ep.ledger, ep.invoices, ep.payments)
    cr = (ep.provenance or {}).get("clean_records")
    if cr:
        tables(cr.get("ledger", []), cr.get("invoices", []), cr.get("payments", []))
    for iss in (ep.truth or {}).get("issues", []):
        for a in iss.get("affected_ids", []):
            add(a)
        for s in iss.get("evidence_sets", []):
            for e in s:
                if not e.startswith("fx:"):
                    add(e)
    for c in (ep.provenance or {}).get("changes", []):
        for k in ("records_added", "records_removed"):
            for x in c.get(k, []):
                add(x)
        for m in c.get("records_modified", []):
            add(m.get("id"))
    for d in (ep.provenance or {}).get("distractors", []):
        for x in d.get("record_ids", []):
            add(x)
    return list(seen)


def rename_ids(ep: Episode, salt, mode: str | None = None) -> Episode:
    """Consistently relabel every opaque id (public records, truth, provenance). Record order is left
    exactly as it is (no re-sorting), so ``rename_ids`` and ``permute`` commute:
    ``permute(rename_ids(e, a), b) == rename_ids(permute(e, b), a)`` up to the content-addressed id of an
    identical public render. ``mode``: see the module docstring (default: derived from the salt)."""
    mode, salt = split_rename_salt(salt, mode)
    out = copy.deepcopy(ep)
    old_ids = _collect_ids(out)
    taken = set(old_ids)
    mapping: dict[str, str] = {}
    for old in old_ids:
        new = _remap_one(old, salt, taken, mode)
        taken.add(new)
        mapping[old] = new

    def m(x):
        return mapping.get(x, x) if isinstance(x, str) else x

    text_re = None
    if mapping:
        alt = "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
        text_re = re.compile(r"(?<![A-Za-z0-9/_-])(?:" + alt + r")(?![A-Za-z0-9/_-])")

    def text(s):
        return text_re.sub(lambda mo: mapping[mo.group(0)], s) if (text_re and isinstance(s, str)) else s

    def do_tables(ledger, invoices, payments):
        for r in ledger:
            for f in ("row_id", "entity_id", "invoice_id", "payment_id"):
                r[f] = m(r.get(f))
        for d in invoices:
            for f in ("doc_id", "invoice_id", "entity_id"):
                d[f] = m(d.get(f))
        for d in payments:
            d["doc_id"] = m(d["doc_id"])
            d["payment_id"] = m(d["payment_id"])
            for ref in d.get("invoice_refs") or []:
                ref["invoice_id"] = m(ref["invoice_id"])

    do_tables(out.ledger, out.invoices, out.payments)
    if out.truth:
        for iss in out.truth.get("issues", []):
            iss["affected_ids"] = [m(a) for a in iss["affected_ids"]]
            iss["evidence_sets"] = [[e if e.startswith("fx:") else m(e) for e in s] for s in iss["evidence_sets"]]
    if out.provenance:
        pv = out.provenance
        cr = pv.get("clean_records")
        if cr:
            do_tables(cr["ledger"], cr["invoices"], cr["payments"])
        for c in pv.get("changes", []):
            c["records_added"] = [m(x) for x in c.get("records_added", [])]
            c["records_removed"] = [m(x) for x in c.get("records_removed", [])]
            for mod in c.get("records_modified", []):
                mod["id"] = m(mod["id"])
                if mod.get("field") in _ID_FIELDS:
                    mod["before"] = m(mod["before"])
                    mod["after"] = m(mod["after"])
            c["description"] = text(c.get("description"))
        for d in pv.get("distractors", []):
            d["record_ids"] = [m(x) for x in d.get("record_ids", [])]
    retag(out)       # content-addressed id recomputed from the renamed public records
    return out


def permute(ep: Episode, salt) -> Episode:
    """Shuffle the record order of every table (ledger, invoices, payments, fx).

    The ``truth`` dict is NOT rewritten (same issues, ids, amounts, evidence sets), but because the records
    move, the ORDER in which the issues appear in the data is permuted; the matcher is order independent, so
    a correct policy must score identically. Pure reorder with no id change, so it commutes with
    ``rename_ids``. The episode id is recomputed because record order is public content."""
    out = copy.deepcopy(ep)
    for name in ("ledger", "invoices", "payments", "fx"):
        sub_rng(0, "permute", salt, name).shuffle(getattr(out, name))
    retag(out)       # record order is part of the public content, so the id is recomputed
    return out
