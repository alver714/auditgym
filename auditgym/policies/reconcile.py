"""reconcile: a naive but reasonable sweep baseline.

It pages through every table (limit 50, within the 100-step budget) and then applies simple
heuristics. They are deliberately naive so the benchmark has headroom:

* duplicate_invoice  - same entity + total + currency within 10 days (falls for recurring billing);
* payment_misallocation - AR credits posted for a payment < its amount (falls for prepayments,
  overpayments and realised FX differences);
* fx_error - recompute the base amount using the *date string* of issued_at / payment timestamp
  (ignores the offset) and compare with the posted row (falls for timezone-boundary cases);
* period_cutoff - compare the *date strings* of issued_at with the period dates (ignores
  offsets: misses tricky-offset cutoff errors, flags legitimate timezone-boundary invoices).

Evidence = the ids it actually used. It only uses observations; it never sees truth.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..constants import (ACC_AR, ACC_CASH, MAX_FINDINGS, TABLE_FX, TABLE_INVOICES, TABLE_LEDGER,
                         TABLE_PAYMENTS, fx_id)
from ..money import convert_minor
from .base import ScriptedPolicy, make_finding, submit_action


def _tol(expected: int) -> int:
    return max(1, abs(expected) // 1000)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


class Reconcile(ScriptedPolicy):
    name = "reconcile"

    def on_reset(self) -> None:
        m = self.first_obs.get("manifest") or {}
        self.base = m.get("base_currency", "USD")
        self.period_start = m.get("period_start", "")
        self.period_end = m.get("period_end", "")
        self._findings: list[dict] = []

    def fallback_findings(self) -> list[dict]:
        return self._findings

    # ------------------------------------------------------------ script
    def script(self):
        # small tables first so a tight budget always loses ledger pages, not the documents
        invoices = yield from self.paginate(TABLE_INVOICES)
        payments = yield from self.paginate(TABLE_PAYMENTS)
        fx_rows = yield from self.paginate(TABLE_FX)
        ledger = yield from self.paginate(TABLE_LEDGER)
        self._findings = self.analyse(invoices, payments, fx_rows, ledger)
        yield submit_action(self.episode_id, self._findings)

    # ------------------------------------------------------------ analysis (pure)
    def analyse(self, invoices, payments, fx_rows, ledger) -> list[dict]:
        base = self.base
        rates: dict[str, dict[str, str]] = {}
        for r in fx_rows:
            if r.get("target_currency") == base:
                rates.setdefault(r["source_currency"], {})[r["date"]] = r["rate"]
        by_inv: dict[str, list[dict]] = {}
        by_pay: dict[str, list[dict]] = {}
        for r in ledger:
            if r.get("invoice_id"):
                by_inv.setdefault(r["invoice_id"], []).append(r)
            if r.get("payment_id"):
                by_pay.setdefault(r["payment_id"], []).append(r)

        def ar_debit(inv_id):
            rows = [r for r in by_inv.get(inv_id, []) if r["account"] == ACC_AR and r["debit_minor"] > 0]
            return rows[0] if rows else None

        def naive_rate(cur: str, date_str: str):
            """(rate, fx_id) for the date string, falling back to the latest earlier date."""
            table = rates.get(cur)
            if not table:
                return None
            if date_str in table:
                d = date_str
            else:
                earlier = [x for x in table if x < date_str]
                if not earlier:
                    return None
                d = max(earlier)
            return table[d], fx_id(d, cur, base)

        findings: list[dict] = []
        seen_keys: set[tuple] = set()

        sort_keys: list[tuple] = []

        def add(ftype, affected, amount, evidence, why, when):
            """Collect a candidate; `when` is a content-based instant so the final ordering (and thus
            which 30 candidates survive the cap) does not depend on row order or on id spelling."""
            key = (ftype, tuple(affected))
            if key in seen_keys or amount <= 0:
                return
            seen_keys.add(key)
            findings.append(make_finding(ftype, affected, amount, base, sorted(evidence), why))
            sort_keys.append((ftype, -amount, when))

        # --- duplicates: same entity + total + currency within 10 days
        groups: dict[tuple, list[dict]] = {}
        for inv in invoices:
            groups.setdefault((inv["entity_id"], inv["total_minor"], inv["currency"]), []).append(inv)
        for grp in groups.values():
            grp.sort(key=lambda i: (_parse(i["issued_at"]), i["invoice_id"]))
            for a, b in zip(grp, grp[1:]):
                if _parse(b["issued_at"]) - _parse(a["issued_at"]) <= timedelta(days=10):
                    row = ar_debit(b["invoice_id"])
                    if row:
                        add("duplicate_invoice", [b["invoice_id"]], row["debit_minor"],
                            [a["doc_id"], b["doc_id"], row["row_id"]],
                            "same customer, currency and total issued within 10 days",
                            _parse(b["issued_at"]).timestamp())

        # --- cutoff: date strings only
        p0, p1 = self.period_start[:10], self.period_end[:10]
        for inv in invoices:
            d = inv["issued_at"][:10]
            if d > p1 or d < p0:
                row = ar_debit(inv["invoice_id"])
                if row:
                    add("period_cutoff", [inv["invoice_id"]], row["debit_minor"],
                        [inv["doc_id"], row["row_id"]],
                        "issued after period end by date but posted inside the period",
                        _parse(inv["issued_at"]).timestamp())

        # --- fx on invoices: naive date string of issued_at
        for inv in invoices:
            if inv["currency"] == base:
                continue
            row = ar_debit(inv["invoice_id"])
            nr = naive_rate(inv["currency"], inv["issued_at"][:10])
            if not row or not nr:
                continue
            expected = convert_minor(inv["total_minor"], inv["currency"], base, nr[0])
            diff = abs(row["debit_minor"] - expected)
            if diff > _tol(expected):
                add("fx_error", [inv["invoice_id"]], diff, [inv["doc_id"], row["row_id"], nr[1]],
                    "AR debit differs from total at the issue-date rate",
                    _parse(inv["issued_at"]).timestamp())

        # --- payments: fx on cash row, and AR credits vs amount
        for pay in payments:
            if pay["amount_minor"] <= 0:
                continue
            rows = by_pay.get(pay["payment_id"], [])
            cash = [r for r in rows if r["account"] == ACC_CASH and r["debit_minor"] > 0]
            ar_cr = [r for r in rows if r["account"] == ACC_AR and r["credit_minor"] > 0]
            expected = None
            if pay["currency"] == base:
                expected = pay["amount_minor"]
                fx_ev: list[str] = []
            else:
                nr = naive_rate(pay["currency"], pay["timestamp"][:10])
                if nr:
                    expected = convert_minor(pay["amount_minor"], pay["currency"], base, nr[0])
                    fx_ev = [nr[1]]
            if pay["currency"] != base and cash and expected is not None:
                diff = abs(cash[0]["debit_minor"] - expected)
                if diff > _tol(expected):
                    add("fx_error", [pay["payment_id"]], diff,
                        [pay["doc_id"], cash[0]["row_id"]] + fx_ev,
                        "cash debit differs from amount at the payment-date rate",
                        _parse(pay["timestamp"]).timestamp())
            if rows and expected is not None:
                credited = sum(r["credit_minor"] for r in ar_cr)
                gap = expected - credited
                if gap > max(1, expected // 200):
                    add("payment_misallocation", [pay["payment_id"]], gap,
                        [pay["doc_id"]] + [r["row_id"] for r in rows][:8],
                        "AR credits posted are less than the payment amount",
                        _parse(pay["timestamp"]).timestamp())
        order = sorted(range(len(findings)), key=lambda i: sort_keys[i])
        return [findings[i] for i in order][:MAX_FINDINGS]
