"""careful_auditor: a strong but HONEST scripted agent (EXTRA diagnostic policy, not a required baseline).

Purpose: show that the environment is solvable from public evidence within the 100-step budget and
that the reward separates careful work from naive heuristics (``reconcile``).

Honesty contract
----------------
* It only uses observations: the first observation (task / manifest / policy.json) and the records
  it fetches through ``list_records`` / ``get_record``.
* It never imports truth, the generator, ``rules.py`` or ``money.py`` - the policy text in the first
  observation is re-implemented here with its own ``Decimal`` code (half-even rounding, the
  company-offset rate date, and the "most recent earlier date" fallback).
* Constants are only the public ones (table names, account names, finding schema) from
  ``constants.py`` plus the action builders in ``base.py``.

Plan (<= 100 steps)
-------------------
1. Page invoices, payments, fx and the whole ledger with limit 50 (at most ~50 steps for the largest
   2000-row episodes).
2. Pure analysis of the paged rows:
   * period_cutoff   - compare ``issued_at`` as an *instant* with the period bounds; an invoice
     outside the period that still has an AR debit in the (in-period only) ledger is a finding;
   * fx_error        - recompute the base amount of every foreign invoice (AR debit) and foreign
     payment (cash debit) with the rate of the company-offset calendar date (+ fallback);
   * payment_misallocation - sum of conv(ref.amount, rate of the *invoice* date) minus the AR credits
     actually posted under the payment_id. Prepayments (no refs), overpayments (excess goes to
     unapplied cash) and realised FX gain/loss therefore do not trigger it;
   * duplicate_invoice - candidates are invoices with the same entity + currency + total (a
     necessary condition for identical line items). Only candidates are fetched with ``get_record``
     (to read ``line_items``) and compared on all six line fields, so recurring billing with a
     different ``service_period`` is correctly rejected. Candidates are fetched closest-in-time
     first, so a tight budget loses the least likely ones first.
3. Submit with the evidence sets described in the task text (<= 10 ids per finding).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

from ..constants import (ACC_AR, ACC_CASH, MAX_EVIDENCE_IDS, MAX_FINDINGS, TABLE_FX, TABLE_INVOICES,
                         TABLE_LEDGER, TABLE_PAYMENTS)
from .base import ScriptedPolicy, get_record_action, make_finding, submit_action

_LINE_KEYS = ("description", "quantity", "unit_price_minor", "vat_rate", "service_period", "po_ref")


def _material(diff: int) -> bool:
    """A discrepancy is reportable when it is distinguishable from +-1 minor-unit noise.

    The published amount tolerance is ``max(1, claimed // 1000)`` of the *finding amount* (the
    discrepancy), not of the invoice total, so even a small discrepancy has to be claimed exactly.
    All the amounts here are recomputed exactly (same rounding as the posting rules), so a clean
    ledger gives a difference of exactly 0; anything above 2 minor units is real.
    """
    return abs(diff) > 2


def _parse_offset(text: str) -> timezone:
    sign = -1 if text.startswith("-") else 1
    hh, mm = text.lstrip("+-").split(":")
    return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))


def _parse_ts(text: str) -> datetime:
    return datetime.fromisoformat(text)


class CarefulAuditor(ScriptedPolicy):
    name = "careful_auditor"

    # ------------------------------------------------------------ lifecycle
    def on_reset(self) -> None:
        m = self.first_obs.get("manifest") or {}
        self.base = m.get("base_currency", "USD")
        self.exps: dict[str, int] = dict((m.get("money") or {}).get("exponents") or {})
        self.company_tz = _parse_offset(m.get("utc_offset") or "+00:00")
        self.period_start = _parse_ts(m["period_start"]) if m.get("period_start") else None
        self.period_end = _parse_ts(m["period_end"]) if m.get("period_end") else None
        self._findings: list[dict] = []
        self._base_findings: list[dict] = []
        self.docs: dict[str, dict] = {}        # invoice_id -> full record fetched (line_items)

    def fallback_findings(self) -> list[dict]:
        return self._findings or self._base_findings

    # ------------------------------------------------------------ script
    def script(self):
        invoices = yield from self.paginate(TABLE_INVOICES)
        payments = yield from self.paginate(TABLE_PAYMENTS)
        fx_rows = yield from self.paginate(TABLE_FX)
        ledger = yield from self.paginate(TABLE_LEDGER)

        index = self._index(invoices, payments, fx_rows, ledger)
        self._base_findings = self.rule_findings(index)
        self._findings = list(self._base_findings)

        # duplicate candidates: need line_items, fetched only for same entity+currency+total
        for inv_id in self.candidate_fetch_order(index):
            if not self.can_inspect():
                break
            obs = yield get_record_action(TABLE_INVOICES, inv_id)
            if "steps_remaining" in obs:
                self.remaining = int(obs["steps_remaining"])
            rec = ((obs.get("result") or {}).get("record")) if not obs.get("error") else None
            if isinstance(rec, dict) and isinstance(rec.get("line_items"), list):
                self.docs[inv_id] = rec
            self._findings = self._merge(self._base_findings, self.duplicate_findings(index))

        self._findings = self._merge(self._base_findings, self.duplicate_findings(index))
        yield submit_action(self.episode_id, self._findings)

    @staticmethod
    def _merge(a: list[dict], b: list[dict]) -> list[dict]:
        return (list(a) + list(b))[:MAX_FINDINGS]

    # ------------------------------------------------------------ exact FX arithmetic (own code)
    def conv(self, amount_minor: int, src: str, rate: str) -> int:
        """round_half_even(amount_minor / 10**exp_src * rate * 10**exp_base)."""
        if src == self.base:
            return int(amount_minor)
        with localcontext() as ctx:
            ctx.prec = 50
            value = Decimal(int(amount_minor)) * Decimal(rate)
            value = value.scaleb(self.exps[self.base] - self.exps[src])
            return int(value.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))

    def company_date(self, ts: str) -> str:
        """Calendar date (YYYY-MM-DD) of an ISO timestamp expressed in the company offset."""
        return _parse_ts(ts).astimezone(self.company_tz).date().isoformat()

    def in_period(self, issued_at: str) -> bool:
        """True iff the invoice was issued inside the period, compared as an *instant* (offset-aware).
        (Hook: the degraded diagnostic variant overrides this to ignore the offsets.)"""
        if not (self.period_start and self.period_end):
            return True
        return self.period_start <= _parse_ts(issued_at) <= self.period_end

    # ------------------------------------------------------------ indexing
    def _index(self, invoices, payments, fx_rows, ledger) -> dict:
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
        return {"invoices": invoices, "payments": payments, "rates": rates, "by_inv": by_inv,
                "by_pay": by_pay, "inv_by_id": {i["invoice_id"]: i for i in invoices}}

    def rate_for(self, index: dict, currency: str, ts: str):
        """(rate, fx_id) for the company-offset date of ``ts`` with the earlier-date fallback."""
        table = index["rates"].get(currency)
        if not table:
            return None
        d = self.company_date(ts)
        if d not in table:
            earlier = [x for x in table if x < d]
            if not earlier:
                return None
            d = max(earlier)
        return table[d], f"fx:{d}:{currency}:{self.base}"

    @staticmethod
    def ar_debits(index: dict, inv_id: str) -> list[dict]:
        return [r for r in index["by_inv"].get(inv_id, []) if r["account"] == ACC_AR and r["debit_minor"] > 0]

    # ------------------------------------------------------------ rule-based findings (no fetches)
    def rule_findings(self, index: dict) -> list[dict]:
        base = self.base
        out: list[dict] = []
        seen: set[tuple] = set()

        def add(ftype, affected, amount, evidence, why):
            key = (ftype, tuple(affected))
            if key in seen or amount <= 0:
                return
            seen.add(key)
            out.append(make_finding(ftype, affected, amount, base, evidence[:MAX_EVIDENCE_IDS], why))

        # --- period cutoff + fx on invoices
        for inv in index["invoices"]:
            rows = self.ar_debits(index, inv["invoice_id"])
            if not rows:
                continue
            row = rows[0]
            posted = sum(r["debit_minor"] for r in rows)
            if not self.in_period(inv["issued_at"]):
                add("period_cutoff", [inv["invoice_id"]], posted, [inv["doc_id"], row["row_id"]],
                    "invoice issued outside the period (as an instant) but its AR debit is posted inside it")
            if inv["currency"] != base:
                rr = self.rate_for(index, inv["currency"], inv["issued_at"])
                if rr:
                    correct = self.conv(inv["total_minor"], inv["currency"], rr[0])
                    diff = abs(posted - correct)
                    if _material(diff):
                        add("fx_error", [inv["invoice_id"]], diff, [inv["doc_id"], row["row_id"], rr[1]],
                            "AR debit differs from the total converted at the company-date rate")

        # --- payments: cash-row fx and AR-credit allocation
        for pay in index["payments"]:
            if pay["amount_minor"] <= 0:
                continue
            rows = index["by_pay"].get(pay["payment_id"], [])
            foreign = pay["currency"] != base
            if foreign:
                cash = [r for r in rows if r["account"] == ACC_CASH and r["debit_minor"] > 0]
                rr = self.rate_for(index, pay["currency"], pay["timestamp"])
                if cash and rr:
                    correct = self.conv(pay["amount_minor"], pay["currency"], rr[0])
                    diff = abs(cash[0]["debit_minor"] - correct)
                    if _material(diff):
                        add("fx_error", [pay["payment_id"]], diff,
                            [pay["doc_id"], cash[0]["row_id"], rr[1]],
                            "cash debit differs from the amount converted at the company-date rate")
            if not rows:
                continue
            refs = pay.get("invoice_refs") or []
            expected = 0
            inv_docs: list[str] = []
            fx_ids: list[str] = []
            resolvable = True
            for ref in refs:
                inv = index["inv_by_id"].get(ref["invoice_id"])
                if inv is None:
                    resolvable = False
                    break
                if inv["currency"] == base:
                    expected += int(ref["amount_minor"])
                else:
                    rr = self.rate_for(index, inv["currency"], inv["issued_at"])
                    if rr is None:
                        resolvable = False
                        break
                    expected += self.conv(ref["amount_minor"], inv["currency"], rr[0])
                    if inv["doc_id"] not in inv_docs:
                        inv_docs.append(inv["doc_id"])
                    if rr[1] not in fx_ids:
                        fx_ids.append(rr[1])
            if not resolvable or not refs:
                continue
            credited = sum(r["credit_minor"] for r in rows if r["account"] == ACC_AR and r["credit_minor"] > 0)
            gap = expected - credited
            if _material(gap):
                evidence = [pay["doc_id"]] + [r["row_id"] for r in rows] + inv_docs + fx_ids
                add("payment_misallocation", [pay["payment_id"]], gap, evidence,
                    "AR credits posted for this payment are less than its invoice_refs imply")
        return out

    # ------------------------------------------------------------ duplicates (need line_items)
    def _groups(self, index: dict) -> list[list[dict]]:
        groups: dict[tuple, list[dict]] = {}
        for inv in index["invoices"]:
            groups.setdefault((inv["entity_id"], inv["currency"], inv["total_minor"]), []).append(inv)
        out = []
        for grp in groups.values():
            if len(grp) > 1:
                grp.sort(key=lambda i: (_parse_ts(i["issued_at"]), i["invoice_id"]))
                out.append(grp)
        return out

    def candidate_fetch_order(self, index: dict) -> list[str]:
        """Invoice ids whose line_items are needed, most suspicious (closest in time) first."""
        scored: list[tuple[timedelta, int, str]] = []
        for grp in self._groups(index):
            for k in range(1, len(grp)):
                later = grp[k]
                if not self.ar_debits(index, later["invoice_id"]):
                    continue                       # not posted -> no AR debit amount to claim
                gap = min(_parse_ts(later["issued_at"]) - _parse_ts(grp[j]["issued_at"]) for j in range(k))
                scored.append((gap, -later["total_minor"], later["invoice_id"]))
                for j in range(k):
                    scored.append((gap, -later["total_minor"], grp[j]["invoice_id"]))
        order: list[str] = []
        for _gap, _amt, inv_id in sorted(scored):
            if inv_id not in order:
                order.append(inv_id)
        return order

    @staticmethod
    def _lines(rec: dict) -> list[tuple]:
        return [tuple(li.get(k) for k in _LINE_KEYS) for li in rec["line_items"]]

    def duplicate_findings(self, index: dict) -> list[dict]:
        out: list[dict] = []
        for grp in self._groups(index):
            for k in range(1, len(grp)):
                later = grp[k]
                rec_l = self.docs.get(later["invoice_id"])
                rows = self.ar_debits(index, later["invoice_id"])
                if rec_l is None or not rows:
                    continue
                earlier = [grp[j] for j in range(k)
                           if grp[j]["invoice_id"] in self.docs
                           and self._lines(self.docs[grp[j]["invoice_id"]]) == self._lines(rec_l)]
                if not earlier:
                    continue
                evidence = [e["doc_id"] for e in earlier][:MAX_EVIDENCE_IDS - 2] + [later["doc_id"], rows[0]["row_id"]]
                out.append(make_finding(
                    "duplicate_invoice", [later["invoice_id"]], sum(r["debit_minor"] for r in rows),
                    self.base, evidence,
                    "same entity, currency and identical line items (incl. service_period, po_ref) as an "
                    "earlier invoice; the later one is the duplicate"))
        return out
