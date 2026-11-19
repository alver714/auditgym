"""Independent rules engine (the "perfect checker") for AuditGym.

This module re-implements the accounting policy of SPEC section 1 from a *public* episode only
(manifest, policy, ledger, invoices, payments, fx).  It deliberately does NOT import anything
from ``auditgym.generator``: it is a second, separate derivation used to cross-check the
generator's truth (``validate.py``) and the oracle policy.

Public API
----------
find_violations(ep)                         -> list[dict]
check_evidence(issue, evidence_ids, ep)     -> (ok: bool, amount_minor: int)
"""
from __future__ import annotations

import bisect
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Iterable

from .constants import (ACC_AR, ACC_CASH, DUPLICATE_INVOICE, FX_ERROR, ISSUE_TYPES,
                        MAX_EVIDENCE_IDS, PAYMENT_MISALLOCATION, PERIOD_CUTOFF, fx_id)
from .money import ROUNDING_MODES, convert_minor

_TYPE_ORDER = {t: i for i, t in enumerate(ISSUE_TYPES)}


# ---------------------------------------------------------------- time helpers
def _parse_offset(text: str) -> timezone:
    """'+05:30' / '-03:00' / 'Z' / 'UTC' -> fixed-offset tzinfo."""
    t = (text or "+00:00").strip()
    if t in ("Z", "UTC", "z"):
        return timezone.utc
    sign = -1 if t.startswith("-") else 1
    body = t.lstrip("+-")
    if ":" in body:
        hh, mm = body.split(":")[:2]
    elif len(body) == 4:
        hh, mm = body[:2], body[2:]
    else:
        hh, mm = body, "0"
    return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))


def _line_key(items: Iterable[dict]) -> tuple:
    """Canonical identity of an invoice's line items (the 'line' number is not part of it)."""
    out = []
    for li in items or []:
        try:
            vat = str(Decimal(str(li.get("vat_rate"))).normalize())
        except (InvalidOperation, ValueError):
            vat = str(li.get("vat_rate"))
        out.append((li.get("description"), li.get("quantity"), li.get("unit_price_minor"),
                    vat, li.get("service_period"), li.get("po_ref")))
    return tuple(out)


# ---------------------------------------------------------------- context
class _Ctx:
    """Pre-computed lookups over a public episode."""

    def __init__(self, ep):
        m = ep.manifest
        self.ep = ep
        self.base: str = m["base_currency"]
        self.tz = _parse_offset(m.get("utc_offset") or "+00:00")
        self.p_start = self.ts(m["period_start"])
        self.p_end = self.ts(m["period_end"])
        mode = (m.get("rounding") or {}).get("mode", "half_even")
        self.rounding = mode if mode in ROUNDING_MODES else "half_even"

        # fx: (source, target) -> {date: row}, sorted date list
        self.fx_rows: dict[tuple[str, str], dict[str, dict]] = {}
        for r in ep.fx:
            self.fx_rows.setdefault((r["source_currency"], r["target_currency"]), {})[r["date"]] = r
        self.fx_dates = {k: sorted(v) for k, v in self.fx_rows.items()}
        self.fx_by_id = {fx_id(r["date"], r["source_currency"], r["target_currency"]): r for r in ep.fx}

        self.invoices = {i["invoice_id"]: i for i in ep.invoices}
        self.payments = {p["payment_id"]: p for p in ep.payments}
        self.inv_by_doc = {i["doc_id"]: i for i in ep.invoices}
        self.pay_by_doc = {p["doc_id"]: p for p in ep.payments}
        self.row_by_id = {r["row_id"]: r for r in ep.ledger}
        # earliest AR-debit posting instant per invoice (id-independent tie-break of the duplicate order)
        self.first_post: dict[str, datetime] = {}
        for r in ep.ledger:
            if r.get("account") == ACC_AR and r.get("invoice_id") and int(r.get("debit_minor") or 0) > 0:
                t = self.ts(r["timestamp"])
                cur = self.first_post.get(r["invoice_id"])
                if cur is None or t < cur:
                    self.first_post[r["invoice_id"]] = t

    # -- time
    def ts(self, s: str) -> datetime:
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=self.tz)
        return d

    def company_date(self, s: str) -> str:
        return self.ts(s).astimezone(self.tz).date().isoformat()

    def in_period(self, s: str) -> bool:
        return self.p_start <= self.ts(s) <= self.p_end

    # -- fx
    def rate(self, currency: str, date: str, fx_rows=None):
        """(fx_id | None, rate_str) for currency->base on `date` with weekend fallback.

        Returns None if no usable rate.  Base currency => (None, '1').
        """
        if currency == self.base:
            return (None, "1")
        key = (currency, self.base)
        dates = self.fx_dates.get(key)
        if not dates:
            return None
        i = bisect.bisect_right(dates, date) - 1
        if i < 0:
            return None
        d = dates[i]
        return (fx_id(d, currency, self.base), self.fx_rows[key][d]["rate"])

    def conv(self, amount: int, currency: str, rate: str) -> int:
        return convert_minor(int(amount), currency, self.base, rate, self.rounding)


def _ar_debits(rows: Iterable[dict], invoice_id: str) -> list[dict]:
    return [r for r in rows if r.get("account") == ACC_AR and r.get("invoice_id") == invoice_id
            and int(r.get("debit_minor") or 0) > 0]


def _ar_credits(rows: Iterable[dict], payment_id: str) -> list[dict]:
    return [r for r in rows if r.get("account") == ACC_AR and r.get("payment_id") == payment_id
            and int(r.get("credit_minor") or 0) > 0]


def _cash_debits(rows: Iterable[dict], payment_id: str) -> list[dict]:
    return [r for r in rows if r.get("account") == ACC_CASH and r.get("payment_id") == payment_id
            and int(r.get("debit_minor") or 0) > 0]


_LATE = datetime.max.replace(tzinfo=timezone.utc)


def _dup_order_key(ctx: _Ctx, inv: dict):
    """Which of two duplicates is the later one: issued_at (an instant), then due_at, then the earliest AR posting
    time -- all id-independent, so renaming ids can never swap original and duplicate -- and only as the very
    last resort the invoice_id.  The generator guarantees distinct issued_at for every injected pair."""
    try:
        due = ctx.ts(inv["due_at"])
    except (KeyError, ValueError, TypeError):
        due = _LATE
    return (ctx.ts(inv["issued_at"]), due, ctx.first_post.get(inv["invoice_id"], _LATE), inv["invoice_id"])


def _dedupe(seq: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ================================================================= find_violations
def find_violations(ep) -> list[dict]:
    """Recompute policy postings from public records and report every discrepancy.

    Each item: {type, affected_ids, amount_minor, currency, evidence_ids, detail}.
    """
    ctx = _Ctx(ep)
    out: list[dict] = []
    ledger = ep.ledger

    ar_deb: dict[str, list[dict]] = {}
    ar_cred: dict[str, list[dict]] = {}
    cash_deb: dict[str, list[dict]] = {}
    pay_rows: dict[str, list[dict]] = {}
    for r in ledger:
        if r.get("invoice_id") and r.get("account") == ACC_AR and int(r.get("debit_minor") or 0) > 0:
            ar_deb.setdefault(r["invoice_id"], []).append(r)
        pid = r.get("payment_id")
        if pid:
            pay_rows.setdefault(pid, []).append(r)
            if r.get("account") == ACC_AR and int(r.get("credit_minor") or 0) > 0:
                ar_cred.setdefault(pid, []).append(r)
            if r.get("account") == ACC_CASH and int(r.get("debit_minor") or 0) > 0:
                cash_deb.setdefault(pid, []).append(r)

    # ---- duplicate_invoice -------------------------------------------------------
    groups: dict[tuple, list[dict]] = {}
    for inv in ep.invoices:
        groups.setdefault((inv["entity_id"], inv["currency"], _line_key(inv.get("line_items"))), []).append(inv)
    for grp in groups.values():
        if len(grp) < 2:
            continue
        grp = sorted(grp, key=lambda i: _dup_order_key(ctx, i))
        first = grp[0]
        for dup in grp[1:]:
            rows = ar_deb.get(dup["invoice_id"], [])
            if not rows:
                continue  # never posted -> no ledger impact
            amount = sum(int(r["debit_minor"]) for r in rows)
            out.append({
                "type": DUPLICATE_INVOICE,
                "affected_ids": [dup["invoice_id"]],
                "amount_minor": amount,
                "currency": ctx.base,
                "evidence_ids": [first["doc_id"], dup["doc_id"]] + [r["row_id"] for r in rows],
                "detail": f"invoice {dup['invoice_id']} duplicates {first['invoice_id']} "
                          f"(same entity/currency/line items); AR debit {amount} posted",
            })

    # ---- period_cutoff -----------------------------------------------------------
    for inv in ep.invoices:
        rows = ar_deb.get(inv["invoice_id"], [])
        if not rows or ctx.in_period(inv["issued_at"]):
            continue
        amount = sum(int(r["debit_minor"]) for r in rows)
        out.append({
            "type": PERIOD_CUTOFF,
            "affected_ids": [inv["invoice_id"]],
            "amount_minor": amount,
            "currency": ctx.base,
            "evidence_ids": [inv["doc_id"]] + [r["row_id"] for r in rows],
            "detail": f"invoice {inv['invoice_id']} issued {inv['issued_at']} is outside the period "
                      f"but AR debit {amount} is posted",
        })

    # ---- fx_error: invoices ------------------------------------------------------
    for inv in ep.invoices:
        cur = inv["currency"]
        rows = ar_deb.get(inv["invoice_id"], [])
        if cur == ctx.base or not rows:
            continue
        ri = ctx.rate(cur, ctx.company_date(inv["issued_at"]))
        if ri is None:
            continue
        correct = ctx.conv(inv["total_minor"], cur, ri[1])
        posted = sum(int(r["debit_minor"]) for r in rows)
        if posted != correct:
            out.append({
                "type": FX_ERROR,
                "affected_ids": [inv["invoice_id"]],
                "amount_minor": abs(posted - correct),
                "currency": ctx.base,
                "evidence_ids": [inv["doc_id"]] + [r["row_id"] for r in rows] + [ri[0]],
                "detail": f"invoice {inv['invoice_id']} AR debit {posted} != correct {correct} "
                          f"(rate {ri[1]} from {ri[0]})",
            })

    # ---- fx_error: payments (cash row) -------------------------------------------
    for pay in ep.payments:
        cur = pay["currency"]
        rows = cash_deb.get(pay["payment_id"], [])
        if cur == ctx.base or not rows or int(pay["amount_minor"]) <= 0:
            continue
        ri = ctx.rate(cur, ctx.company_date(pay["timestamp"]))
        if ri is None:
            continue
        correct = ctx.conv(pay["amount_minor"], cur, ri[1])
        posted = sum(int(r["debit_minor"]) for r in rows)
        if posted != correct:
            out.append({
                "type": FX_ERROR,
                "affected_ids": [pay["payment_id"]],
                "amount_minor": abs(posted - correct),
                "currency": ctx.base,
                "evidence_ids": [pay["doc_id"]] + [r["row_id"] for r in rows] + [ri[0]],
                "detail": f"payment {pay['payment_id']} cash debit {posted} != correct {correct} "
                          f"(rate {ri[1]} from {ri[0]})",
            })

    # ---- payment_misallocation ---------------------------------------------------
    for pay in ep.payments:
        refs = pay.get("invoice_refs")
        if not refs:
            continue
        cur = pay["currency"]
        expected = 0
        ok = True
        inv_docs: list[str] = []
        fx_ids: list[str] = []
        for ref in refs:
            inv = ctx.invoices.get(ref["invoice_id"])
            if inv is None:
                ok = False
                break
            if cur == ctx.base:
                expected += int(ref["amount_minor"])
                continue
            ri = ctx.rate(cur, ctx.company_date(inv["issued_at"]))
            if ri is None:
                ok = False
                break
            expected += ctx.conv(ref["amount_minor"], cur, ri[1])
            inv_docs.append(inv["doc_id"])
            fx_ids.append(ri[0])
        if not ok:
            continue
        credits = ar_cred.get(pay["payment_id"], [])
        posted = sum(int(r["credit_minor"]) for r in credits)
        if expected - posted > 0:
            ar_ids = [r["row_id"] for r in credits]
            other = [r["row_id"] for r in pay_rows.get(pay["payment_id"], []) if r["row_id"] not in set(ar_ids)]
            essential = _dedupe([pay["doc_id"]] + ar_ids + inv_docs + fx_ids)
            room = max(0, MAX_EVIDENCE_IDS - len(essential))
            if room == 0 and not ar_ids and other:
                room = 1
            ev = essential + other[:room]
            out.append({
                "type": PAYMENT_MISALLOCATION,
                "affected_ids": [pay["payment_id"]],
                "amount_minor": expected - posted,
                "currency": ctx.base,
                "evidence_ids": ev,
                "detail": f"payment {pay['payment_id']}: invoice refs convert to {expected} but AR credits "
                          f"posted are {posted}",
            })

    out.sort(key=lambda v: (_TYPE_ORDER[v["type"]], tuple(v["affected_ids"]), v["amount_minor"]))
    return out


# ================================================================= check_evidence
def _fail() -> tuple[bool, int]:
    return False, 0


def check_evidence(issue: dict, evidence_ids: Iterable[str], ep) -> tuple[bool, int]:
    """Recompute `issue` using ONLY the cited records.

    Returns (ok, amount_minor).  ok is False when any id is unknown, the issue is malformed, or
    the cited subset does not suffice to establish the finding (e.g. missing the document, the
    posted ledger row(s), or the correct fx row).  The full fx table is consulted only to decide
    which fx row the fallback rule selects; the rate itself is read from the cited fx record.
    """
    try:
        return _check_evidence(issue, evidence_ids, ep)
    except (KeyError, ValueError, TypeError, InvalidOperation, AttributeError):
        return _fail()


def _check_evidence(issue, evidence_ids, ep):
    ids = list(evidence_ids or [])
    if not ids or not all(isinstance(i, str) for i in ids):
        return _fail()
    ctx = _Ctx(ep)
    itype = issue.get("type")
    aff = issue.get("affected_ids")
    if itype not in ISSUE_TYPES or not isinstance(aff, (list, tuple)) or len(aff) != 1:
        return _fail()
    target = aff[0]

    ev_rows: list[dict] = []
    ev_inv: dict[str, dict] = {}      # invoice_id -> record, only those cited by doc_id
    ev_pay: dict[str, dict] = {}
    ev_fx: dict[str, dict] = {}       # fx_id -> row
    for i in dict.fromkeys(ids):
        if i in ctx.row_by_id:
            ev_rows.append(ctx.row_by_id[i])
        elif i in ctx.inv_by_doc:
            rec = ctx.inv_by_doc[i]
            ev_inv[rec["invoice_id"]] = rec
        elif i in ctx.pay_by_doc:
            rec = ctx.pay_by_doc[i]
            ev_pay[rec["payment_id"]] = rec
        elif i in ctx.fx_by_id:
            ev_fx[i] = ctx.fx_by_id[i]
        else:
            return _fail()

    def evidence_rate(currency: str, ts: str):
        """Rate for (currency, date of ts) that is backed by a cited fx record."""
        if currency == ctx.base:
            return "1"
        ri = ctx.rate(currency, ctx.company_date(ts))
        if ri is None or ri[0] not in ev_fx:
            return None
        return ev_fx[ri[0]]["rate"]

    if itype == DUPLICATE_INVOICE:
        dup = ev_inv.get(target)
        if dup is None:
            return _fail()
        rows = _ar_debits(ev_rows, target)
        if not rows:
            return _fail()
        key = (dup["entity_id"], dup["currency"], _line_key(dup.get("line_items")))
        dk = _dup_order_key(ctx, dup)
        has_orig = any(o["invoice_id"] != target
                       and (o["entity_id"], o["currency"], _line_key(o.get("line_items"))) == key
                       and _dup_order_key(ctx, o) < dk
                       for o in ev_inv.values())
        if not has_orig:
            return _fail()
        return True, sum(int(r["debit_minor"]) for r in rows)

    if itype == PERIOD_CUTOFF:
        inv = ev_inv.get(target)
        if inv is None or ctx.in_period(inv["issued_at"]):
            return _fail()
        rows = _ar_debits(ev_rows, target)
        if not rows:
            return _fail()
        return True, sum(int(r["debit_minor"]) for r in rows)

    if itype == FX_ERROR:
        if target in ctx.invoices:
            inv = ev_inv.get(target)
            if inv is None or inv["currency"] == ctx.base:
                return _fail()
            rows = _ar_debits(ev_rows, target)
            rate = evidence_rate(inv["currency"], inv["issued_at"])
            if not rows or rate is None:
                return _fail()
            correct = ctx.conv(inv["total_minor"], inv["currency"], rate)
            posted = sum(int(r["debit_minor"]) for r in rows)
            return (posted != correct), abs(posted - correct)
        if target in ctx.payments:
            pay = ev_pay.get(target)
            if pay is None or pay["currency"] == ctx.base or int(pay["amount_minor"]) <= 0:
                return _fail()
            rows = _cash_debits(ev_rows, target)
            rate = evidence_rate(pay["currency"], pay["timestamp"])
            if not rows or rate is None:
                return _fail()
            correct = ctx.conv(pay["amount_minor"], pay["currency"], rate)
            posted = sum(int(r["debit_minor"]) for r in rows)
            return (posted != correct), abs(posted - correct)
        return _fail()

    # PAYMENT_MISALLOCATION
    pay = ev_pay.get(target)
    if pay is None or not pay.get("invoice_refs"):
        return _fail()
    if not any(r.get("payment_id") == target for r in ev_rows):
        return _fail()
    cur = pay["currency"]
    expected = 0
    for ref in pay["invoice_refs"]:
        if cur == ctx.base:
            expected += int(ref["amount_minor"])
            continue
        inv = ev_inv.get(ref["invoice_id"])
        if inv is None:
            return _fail()
        rate = evidence_rate(cur, inv["issued_at"])
        if rate is None:
            return _fail()
        expected += ctx.conv(ref["amount_minor"], cur, rate)
    posted = sum(int(r["credit_minor"]) for r in _ar_credits(ev_rows, target))
    amount = expected - posted
    return (amount > 0), amount
