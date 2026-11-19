"""flag_all (attack baseline): list the tables, then accuse everything of everything.

Submits up to 30 findings spread over all four issue types and many ids, with amounts copied
from records and all plausible evidence it has seen. A sound reward must score this badly.
"""
from __future__ import annotations

from ..constants import (ACC_AR, ISSUE_TYPES, MAX_FINDINGS, TABLE_FX, TABLE_INVOICES,
                         TABLE_LEDGER, TABLE_PAYMENTS, fx_id)
from .base import ScriptedPolicy, make_finding, submit_action


class FlagAll(ScriptedPolicy):
    name = "flag_all"

    def on_reset(self) -> None:
        self.base = (self.first_obs.get("manifest") or {}).get("base_currency", "USD")
        self._findings: list[dict] = []

    def fallback_findings(self) -> list[dict]:
        return self._findings

    def script(self):
        invoices = yield from self.paginate(TABLE_INVOICES, max_pages=1)
        payments = yield from self.paginate(TABLE_PAYMENTS, max_pages=1)
        ledger = yield from self.paginate(TABLE_LEDGER, max_pages=2)
        fx = yield from self.paginate(TABLE_FX, max_pages=1)

        by_inv: dict[str, list[dict]] = {}
        by_pay: dict[str, list[dict]] = {}
        for r in ledger:
            if r.get("invoice_id"):
                by_inv.setdefault(r["invoice_id"], []).append(r)
            if r.get("payment_id"):
                by_pay.setdefault(r["payment_id"], []).append(r)
        fx_ids = [fx_id(r["date"], r["source_currency"], r["target_currency"]) for r in fx]

        targets: list[tuple[str, dict]] = []
        # interleave invoices/payments so every type gets many ids
        for i in range(max(len(invoices), len(payments))):
            if i < len(invoices):
                targets.append(("invoice", invoices[i]))
            if i < len(payments):
                targets.append(("payment", payments[i]))

        findings: list[dict] = []
        k = 0
        for kind, rec in targets:
            if len(findings) >= MAX_FINDINGS:
                break
            if kind == "invoice":
                ftype = ("duplicate_invoice", "fx_error", "period_cutoff")[k % 3]
                rid, amount = rec["invoice_id"], int(rec.get("total_minor") or 0)
                rows = by_inv.get(rid, [])
            else:
                ftype = ("payment_misallocation", "fx_error")[k % 2]
                rid, amount = rec["payment_id"], abs(int(rec.get("amount_minor") or 0))
                rows = by_pay.get(rid, [])
            k += 1
            ev = [rec["doc_id"]] + [r["row_id"] for r in rows][:6] + fx_ids[:1]
            findings.append(make_finding(ftype, [rid], amount, self.base, ev,
                                         "flagged: looks suspicious"))
        # make sure every type appears even on tiny episodes
        for ftype in ISSUE_TYPES:
            if len(findings) >= MAX_FINDINGS:
                break
            if not any(f["type"] == ftype for f in findings) and (invoices or payments):
                rec = (invoices or payments)[0]
                rid = rec.get("invoice_id") or rec.get("payment_id")
                findings.append(make_finding(ftype, [rid], 1000, self.base, [rec["doc_id"]], "flagged"))
        self._findings = findings[:MAX_FINDINGS]
        yield submit_action(self.episode_id, self._findings)
