"""generate_episode(seed, config) -> Episode : the single entry point of the generator."""
from __future__ import annotations

import copy

from ..constants import CURRENCY_EXPONENT, ENV_VERSION, SCHEMA_VERSION
from ..content_id import retag
from ..episode import Episode
from . import distractors, injectors
from .config import config_hash, resolve_config, sample_spec
from .injectors import InjectionError
from .world import World, WorldError, check_balanced, iso

MAX_ATTEMPTS = 60


class GenerationError(RuntimeError):
    pass


def build_policy(spec: dict, w: World) -> dict:
    return {
        "revenue_recognition": {"rule": "AR and revenue are recognised at invoice issued_at. An invoice belongs to the period iff period_start <= issued_at (as an instant) <= period_end. Compare instants, not date strings."},
        "period_boundaries": {"period_start": iso(w.period_start), "period_end": iso(w.period_end),
                              "inclusive": True, "timezone": spec["timezone"], "utc_offset": spec["utc_offset"]},
        "fx": {"direction": "rate converts 1 unit of source_currency into target_currency (=base)",
               "invoice_rate_date": "calendar date of issued_at expressed in company utc_offset",
               "payment_rate_date": "calendar date of payment timestamp expressed in company utc_offset",
               "fallback": "if no rate for that date, use the most recent earlier date present in fx.csv",
               "conversion": "base_minor = round_half_even(amount_minor / 10**exp_src * rate * 10**exp_base)"},
        "ledger": {"currency": "all ledger rows are in base_currency",
                   "double_entry": "each posting group (same timestamp + same invoice_id/payment_id) balances"},
        "postings": {
            "invoice": "Dr 1200_accounts_receivable = conv(total); Cr 2200_vat_payable = conv(vat); Cr 4000_revenue = conv(total)-conv(vat)",
            "payment": "Dr 1000_cash = conv(amount, payment date rate); for each invoice_refs entry Cr 1200_accounts_receivable = conv(ref.amount_minor, that invoice's rate); amount not covered by invoice_refs -> Cr 2300_unapplied_cash (converted at payment rate); remaining difference -> 7900_fx_gain_loss (Cr if gain, Dr if loss)",
            "refund": "negative amount_minor; Dr 2300_unapplied_cash, Cr 1000_cash, both at conv(|amount|, refund-date rate). For a foreign-currency refund the unapplied balance (booked at the payment-date rate) can therefore keep a small base-currency residual; no 7900_fx_gain_loss row is posted for it and this is NOT an error"},
        "duplicate_rule": "two invoices of the same entity with identical currency and identical line_items (description, quantity, unit_price_minor, vat_rate, service_period, po_ref) are duplicates; the later one (by issued_at instant; ties broken by due_at, then first posting time, then invoice_id) is the duplicate. Same amount with a different service_period is NOT a duplicate.",
        "finding_amounts": {
            "duplicate_invoice": "AR debit posted for the duplicate invoice (base minor)",
            "payment_misallocation": "sum over invoice_refs of conv(ref.amount_minor, invoice rate) minus AR credits actually posted with this payment_id (base minor)",
            "fx_error": "abs(posted - correct) on the AR debit row (invoice) or the cash debit row (payment), base minor",
            "period_cutoff": "AR debit posted inside the period for an invoice issued outside it (base minor)"},
        "finding_affected_ids": {"duplicate_invoice": "[duplicate invoice_id]", "payment_misallocation": "[payment_id]",
                                 "fx_error": "[invoice_id] or [payment_id]", "period_cutoff": "[invoice_id]"},
        "finding_currency": "always base_currency",
        "amount_tolerance": "|claimed - expected| <= max(1, expected // 1000) minor units",
        "evidence": "evidence_ids are ledger row_id, document doc_id or fx ids 'fx:<date>:<source>:<target>'; every cited id must exist; a finding needs a full sufficient evidence set (see task text)",
    }


def _clean_snapshot(w: World) -> dict:
    return {"ledger": copy.deepcopy(w.sorted_ledger()),
            "invoices": copy.deepcopy(w.sorted_invoices()),
            "payments": copy.deepcopy(w.sorted_payments())}


def _generate(seed: int, cfg: dict, spec: dict, attempt: int) -> Episode:
    w = World(seed, attempt, spec, cfg)
    w.build_customers()
    w.build_fx()
    w.gen_ordinary_invoices()
    w.gen_post_period_invoices()
    w.gen_pre_period_invoices()
    distractors.apply_distractors(w)
    w.gen_boundary_payments()
    w.gen_ordinary_payments()
    w.verify_no_duplicates()
    w.post_clean()
    check_balanced(list(w.ledger.values()))
    clean = _clean_snapshot(w)

    ctx = injectors.apply_injections(w, seed)
    check_balanced(list(w.ledger.values()))
    issues = injectors.finalize_truth(w, ctx.drafts, seed)

    ledger, invoices, payments, fx = w.sorted_ledger(), w.sorted_invoices(), w.sorted_payments(), w.fx_rows()
    docs = len(invoices) + len(payments)
    if not (200 <= len(ledger) <= 2000) or not (20 <= docs <= 200):
        raise WorldError(f"size out of range: ledger={len(ledger)} docs={docs}")
    if len(issues) > 12:
        raise WorldError("too many issues")

    # content-addressed; filled by retag() once the public records are final.  The PUBLIC manifest deliberately
    # carries no config_id / config_version / config_hash: they would tell the policy which split (train, dev,
    # holdout) it is being evaluated on.  They live in provenance.json (grader side), the grader report and the
    # trajectory headers, and `configs/CONFIG_VERSIONS.json` binds id@version to the config content hash.
    eid = ""
    exps = {k: v for k, v in CURRENCY_EXPONENT.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION, "env_version": ENV_VERSION, "episode_id": eid,
        "company_name": spec["company_name"], "base_currency": w.base,
        "timezone": spec["timezone"], "utc_offset": spec["utc_offset"],
        "period_start": iso(w.period_start), "period_end": iso(w.period_end),
        "counts": {"ledger_rows": len(ledger), "invoices": len(invoices), "payments": len(payments),
                   "documents": docs, "fx_rates": len(fx)},
        "rounding": {"mode": "half_even", "unit": "base_minor", "fx_rate_places": 6},
        "money": {"amounts": "integer minor units", "exponents": exps},
    }
    key_to_id = {i["_key"]: i["issue_id"] for i in issues}
    for i in issues:
        del i["_key"]
    truth = {"schema_version": SCHEMA_VERSION, "episode_id": eid, "base_currency": w.base,
             "issues": sorted(issues, key=lambda x: x["issue_id"])}
    changes = []
    for c in ctx.changes:
        c = dict(c)
        keys = c.pop("issue_keys")
        c["issue_ids"] = sorted(key_to_id[k] for k in keys)
        changes.append(c)
    spec_out = copy.deepcopy(spec)
    spec_out["attempt"] = attempt
    provenance = {
        "episode_id": eid, "env_version": ENV_VERSION, "config_id": cfg["config_id"],
        "config_version": cfg["config_version"], "seed": seed, "spec": spec_out,
        "config_hash": config_hash(cfg),
        "clean_records": clean, "changes": changes,
        "distractors": copy.deepcopy(w.distractors),
    }
    ep = Episode(manifest=manifest, policy=build_policy(spec, w), ledger=ledger, invoices=invoices,
                 payments=payments, fx=fx, truth=truth, provenance=provenance)
    retag(ep)
    return ep


def generate_episode(seed: int, config) -> Episode:
    """Deterministic: same (seed, config, ENV_VERSION) => byte-identical public files and truth."""
    cfg = resolve_config(config)
    seed = int(seed)
    spec = sample_spec(seed, cfg)
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return _generate(seed, cfg, spec, attempt)
        except (WorldError, InjectionError) as e:   # infeasible draw: deterministic retry
            last = e
    raise GenerationError(f"could not generate episode for seed={seed} config={cfg['config_id']}: {last}")
