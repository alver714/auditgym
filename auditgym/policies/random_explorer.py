"""random_explorer: 15-30 random inspection actions on ids it has seen, then 0-3 random findings.

Diagnostic floor policy, not a discriminating one: it guesses ids, so it ends at the -1.0 reward floor on
nearly every episode. Because of that it is deliberately NOT used in the paired invariance checks (a pair of
two floor scores is trivially equal); ``reconcile`` and ``careful_auditor`` carry those."""
from __future__ import annotations

from ..constants import (CURRENCY_EXPONENT, FX_FIELDS, ISSUE_TYPES, PRIMARY_KEY, TABLE_FX,
                         TABLE_INVOICES, TABLE_LEDGER, TABLE_PAYMENTS, TABLES, fx_id)
from ..rng import sub_rng
from .base import (ScriptedPolicy, calculator_action, get_document_action, get_record_action,
                   list_action, make_finding)


class RandomExplorer(ScriptedPolicy):
    name = "random_explorer"

    def on_reset(self) -> None:
        self.rng = sub_rng(self.seed, "random_explorer")
        self.seen: dict[str, list[str]] = {t: [] for t in TABLES}
        self.docs: list[str] = []
        self.totals: list[int] = []
        self.cursors: dict[str, str | None] = {}
        self.base = (self.first_obs.get("manifest") or {}).get("base_currency", "USD")

    # ------------------------------------------------------------ observation intake
    def _ingest(self, obs: dict) -> None:
        res = obs.get("result")
        if not isinstance(res, dict):
            return
        table = res.get("table")
        for row in res.get("rows") or []:
            if not isinstance(row, dict):
                continue
            if table == TABLE_FX:
                rid = row.get("fx_id") or fx_id(row.get("date", ""), row.get("source_currency", ""),
                                                row.get("target_currency", ""))
            else:
                rid = row.get(PRIMARY_KEY.get(table, ""))
            if rid and rid not in self.seen[table]:
                self.seen[table].append(rid)
            if row.get("doc_id") and row["doc_id"] not in self.docs:
                self.docs.append(row["doc_id"])
            if isinstance(row.get("total_minor"), int):
                self.totals.append(row["total_minor"])
            if isinstance(row.get("amount_minor"), int):
                self.totals.append(abs(row["amount_minor"]))
        if table in self.cursors or table in TABLES:
            self.cursors[table] = res.get("next_cursor")

    # ------------------------------------------------------------ script
    def _random_action(self) -> dict:
        rng = self.rng
        kind = rng.choice(["list", "list", "get", "doc", "calc"])
        tables_with_ids = [t for t in TABLES if self.seen[t]]
        if kind == "get" and tables_with_ids:
            t = rng.choice(tables_with_ids)
            return get_record_action(t, rng.choice(self.seen[t]))
        if kind == "doc" and self.docs:
            return get_document_action(rng.choice(self.docs))
        if kind == "calc":
            a = rng.choice(self.totals) if self.totals else rng.randint(1, 10**6)
            b = rng.choice(["1.12", "0.2", "2", "0.001", "100"])
            return calculator_action(f"{a} * {b}")
        t = rng.choice(TABLES)
        cursor = self.cursors.get(t) if rng.random() < 0.4 else None
        return list_action(t, cursor, rng.choice([5, 10, 20, 50]))

    def _random_findings(self) -> list[dict]:
        rng = self.rng
        ids_inv = self.seen[TABLE_INVOICES]
        ids_pay = self.seen[TABLE_PAYMENTS]
        if not (ids_inv or ids_pay):
            return []
        findings = []
        evidence_pool = (self.docs + self.seen[TABLE_LEDGER] + self.seen[TABLE_FX])
        for _ in range(rng.randint(0, 3)):
            ftype = rng.choice(ISSUE_TYPES)
            pool = ids_pay if (ftype == "payment_misallocation" and ids_pay) else (ids_inv or ids_pay)
            if ftype == "payment_misallocation" and not ids_pay:
                pool = ids_inv
            affected = [rng.choice(pool)]
            amount = rng.randint(1_000, 5_000_000)
            ev = rng.sample(evidence_pool, min(len(evidence_pool), rng.randint(1, 3))) if evidence_pool else []
            findings.append(make_finding(ftype, affected, amount, self.base, ev,
                                         "random guess based on browsing"))
        return findings

    def script(self):
        n_actions = self.rng.randint(15, 30)
        # always look at something first so ids exist
        first_table = self.rng.choice(TABLES)
        obs = yield list_action(first_table, None, self.rng.choice([10, 20, 50]))
        self._ingest(obs)
        for _ in range(n_actions - 1):
            if not self.can_inspect():
                break
            obs = yield self._random_action()
            self._ingest(obs)
        yield self._submit(self._random_findings())

    def _submit(self, findings: list[dict]) -> dict:
        from .base import submit_action
        return submit_action(self.episode_id, findings)
