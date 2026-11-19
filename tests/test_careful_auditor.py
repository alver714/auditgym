"""careful_auditor: an honest, observation-only scripted agent (EXTRA diagnostic policy).

What these tests establish
--------------------------
* the policy is honest (static import scan: no truth / generator / rules / money / env access);
* its reasoning is correct on hand-built records (instant-based cutoff with offsets, company-offset
  FX date + weekend fallback, half-even conversion, refs vs AR credits, distractors ignored,
  duplicates decided on line items, evidence sets as the task text prescribes);
* the environment is solvable from public evidence inside the 100-step budget: on the 20 holdout
  episodes (eval mode) it scores near-perfect F1 while the naive ``reconcile`` baseline does not,
  and it never accuses a clean company;
* it keeps working at maximum episode size and under a tiny budget (still submits a valid answer).

Everything is deterministic and fast (whole file well under 60 s).
"""
from __future__ import annotations

import ast
import functools
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from auditgym.constants import (ACC_AR, ACC_CASH, ACC_FX, ACC_REVENUE, ACC_UNAPPLIED, ACC_VAT,  # noqa: E402
                                MAX_STEPS, MODE_EVAL)
from auditgym.env import AuditEnv  # noqa: E402
from auditgym.policies import EXTRA_POLICY_NAMES, POLICY_NAMES, get_policy  # noqa: E402
from auditgym.policies.careful_auditor import CarefulAuditor  # noqa: E402
from auditgym.runner import (load_splits, replay_actions, report_scores, resolve_config,  # noqa: E402
                             run_episode, submitted_findings, trajectory_hash)

BASE = "KZT"
EXPS = {"KZT": 2, "EUR": 2, "USD": 2, "JPY": 0}


# =============================================================================== helpers
def _auditor(utc_offset="+05:00", base=BASE) -> CarefulAuditor:
    pol = CarefulAuditor(0)
    pol.reset(0, {"episode_id": "ep_test", "steps_remaining": 100,
                  "manifest": {"base_currency": base, "utc_offset": utc_offset,
                               "period_start": "2026-03-01T00:00:00+05:00",
                               "period_end": "2026-03-31T23:59:59+05:00",
                               "money": {"exponents": EXPS}}})
    return pol


def _inv(n, issued_at, total, currency=BASE, entity="C1"):
    return {"doc_id": f"D-inv{n}", "invoice_id": f"INV-{n}", "entity_id": entity, "issued_at": issued_at,
            "due_at": issued_at, "currency": currency, "total_minor": total}


def _row(rid, account, debit=0, credit=0, inv=None, pay=None, ts="2026-03-10T10:00:00+05:00"):
    return {"row_id": rid, "timestamp": ts, "account": account, "debit_minor": debit,
            "credit_minor": credit, "currency": BASE, "entity_id": "C1", "invoice_id": inv, "payment_id": pay}


def _ar(n, amount):
    return _row(f"R-ar{n}", ACC_AR, debit=amount, inv=f"INV-{n}")


def _pay(n, ts, currency, amount, refs):
    return {"doc_id": f"D-pay{n}", "payment_id": f"PAY-{n}", "timestamp": ts, "currency": currency,
            "amount_minor": amount, "invoice_refs": refs}


def _fx(date, cur, rate):
    return {"date": date, "source_currency": cur, "target_currency": BASE, "rate": rate,
            "fx_id": f"fx:{date}:{cur}:{BASE}"}


def _findings(pol, invoices, payments, fx_rows, ledger):
    return pol.rule_findings(pol._index(invoices, payments, fx_rows, ledger))


def _by_type(findings, ftype):
    return [f for f in findings if f["type"] == ftype]


# =============================================================================== honesty / registry
def test_policy_is_observation_only_by_static_import_scan():
    """No truth, generator, rules engine, matcher, env or money module is imported."""
    path = os.path.join(ROOT, "auditgym", "policies", "careful_auditor.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            imported.add(mod)
            imported.update(f"{mod}.{a.name}" for a in node.names)
    banned = ("generator", "rules", "money", "matcher", "env", "validate", "runner", "server", "client",
              "oracle", "episode", "transforms", "provenance", "truth")
    offenders = [m for m in imported if any(part in banned for part in m.replace("..", ".").split("."))]
    assert not offenders, offenders
    src = open(path, encoding="utf-8").read()
    assert "truth.json" not in src and "provenance.json" not in src


def test_registered_as_extra_not_as_a_required_baseline():
    assert "careful_auditor" not in POLICY_NAMES
    assert len(POLICY_NAMES) == 4
    assert "careful_auditor" in EXTRA_POLICY_NAMES
    pol = get_policy("careful_auditor", 3)
    assert isinstance(pol, CarefulAuditor) and pol.name == "careful_auditor"
    with pytest.raises(ValueError):
        get_policy("oracle")                       # still grader-only
    with pytest.raises(ValueError):
        get_policy("no_such_policy")


# =============================================================================== arithmetic
def test_conversion_is_half_even_with_own_decimal_code():
    pol = _auditor()
    assert pol.conv(1, "EUR", "2.500000") == 2          # 2.5 -> 2 (half-even; half-up would give 3)
    assert pol.conv(1, "EUR", "3.500000") == 4          # 3.5 -> 4
    assert pol.conv(1, "EUR", "0.500000") == 0          # 0.5 -> 0
    assert pol.conv(1, "EUR", "1.500000") == 2
    assert pol.conv(12345, "EUR", "512.340000") == 6324837   # exact: 123.45 * 512.34 * 100 = 6324837.30
    assert pol.conv(7, "JPY", "3.141500") == 2199        # exponent 0 -> 2: 7 * 3.1415 * 100 = 2199.05
    assert pol.conv(999, BASE, "9.9") == 999            # same currency -> unchanged, rate ignored


def test_rate_date_uses_company_offset_not_own_offset_or_utc():
    pol = _auditor("+05:00")
    assert pol.company_date("2026-03-31T22:00:00-03:00") == "2026-04-01"   # looks in-period, is not
    assert pol.company_date("2026-04-01T01:00:00+08:00") == "2026-03-31"   # looks outside, is inside
    assert pol.company_date("2026-03-10T03:00:00+05:00") == "2026-03-10"
    assert _auditor("-05:00").company_date("2026-03-10T03:00:00+05:00") == "2026-03-09"
    assert _auditor("+05:30").company_date("2026-03-10T18:45:00+00:00") == "2026-03-11"


def test_fx_fallback_uses_most_recent_earlier_date_per_currency():
    pol = _auditor()
    fx = [_fx("2026-03-12", "EUR", "500.000000"), _fx("2026-03-13", "EUR", "510.000000"),
          _fx("2026-03-16", "EUR", "520.000000"), _fx("2026-03-13", "USD", "450.000000")]
    ix = pol._index([], [], fx, [])
    # Sunday 2026-03-15 (company date) -> falls back to Friday 03-13
    assert pol.rate_for(ix, "EUR", "2026-03-15T10:00:00+05:00") == ("510.000000", "fx:2026-03-13:EUR:KZT")
    assert pol.rate_for(ix, "EUR", "2026-03-16T00:00:01+05:00") == ("520.000000", "fx:2026-03-16:EUR:KZT")
    assert pol.rate_for(ix, "USD", "2026-03-14T10:00:00+05:00") == ("450.000000", "fx:2026-03-13:USD:KZT")
    assert pol.rate_for(ix, "EUR", "2026-03-01T10:00:00+05:00") is None      # nothing earlier
    assert pol.rate_for(ix, "GBP", "2026-03-13T10:00:00+05:00") is None      # unknown pair


# =============================================================================== period cutoff
def test_cutoff_is_decided_on_instants_not_date_strings():
    pol = _auditor()
    late = _inv(1, "2026-03-31T22:00:00-03:00", 100000)    # date string 03-31 but instant is 04-01 08:00 +05
    tz_ok = _inv(2, "2026-04-01T01:00:00+08:00", 120000)   # date string 04-01 but instant is 03-31 22:00 +05
    edge = _inv(3, "2026-03-31T23:59:59+05:00", 130000)    # exactly period_end: inclusive
    clean = _inv(4, "2026-03-05T10:00:00+05:00", 140000)
    ledger = [_ar(1, 100000), _ar(2, 120000), _ar(3, 130000), _ar(4, 140000)]
    out = _by_type(_findings(pol, [late, tz_ok, edge, clean], [], [], ledger), "period_cutoff")
    assert [f["affected_ids"] for f in out] == [["INV-1"]]
    f = out[0]
    assert f["claimed_amount_minor"] == 100000 and f["currency"] == BASE
    assert set(f["evidence_ids"]) == {"D-inv1", "R-ar1"}


def test_unposted_post_period_invoice_is_not_a_cutoff_finding():
    pol = _auditor()
    after = _inv(1, "2026-04-02T10:00:00+05:00", 100000)    # correctly absent from the ledger
    assert _findings(pol, [after], [], [], []) == []


# =============================================================================== fx
def test_fx_error_on_invoice_uses_correct_rate_date_and_evidence():
    pol = _auditor()
    fx = [_fx("2026-03-13", "EUR", "510.000000"), _fx("2026-03-16", "EUR", "520.000000")]
    # issued Sat 03-14 23:30 -05:00 = Sun 03-15 09:30 company time -> fallback to 03-13 (510)
    inv_ok = _inv(1, "2026-03-14T23:30:00-05:00", 10000, "EUR")
    inv_bad = _inv(2, "2026-03-14T23:30:00-05:00", 20000, "EUR", entity="C2")
    ledger = [_ar(1, pol.conv(10000, "EUR", "510.000000")),
              _ar(2, pol.conv(20000, "EUR", "520.000000"))]     # posted with the Monday rate
    out = _by_type(_findings(pol, [inv_ok, inv_bad], [], fx, ledger), "fx_error")
    assert [f["affected_ids"] for f in out] == [["INV-2"]]
    f = out[0]
    assert f["claimed_amount_minor"] == pol.conv(20000, "EUR", "520.000000") - pol.conv(20000, "EUR", "510.000000")
    assert f["evidence_ids"] == ["D-inv2", "R-ar2", "fx:2026-03-13:EUR:KZT"]


def test_small_but_exact_fx_discrepancy_is_still_reported():
    """Tolerance applies to the finding amount, not to the invoice total: a 0.05% error counts."""
    pol = _auditor()
    fx = [_fx("2026-03-10", "EUR", "500.000000")]
    inv = _inv(1, "2026-03-10T10:00:00+05:00", 1_000_000, "EUR")        # 5,000,000 KZT minor
    correct = pol.conv(1_000_000, "EUR", "500.000000")
    out = _findings(pol, [inv], [], fx, [_ar(1, correct + 2500)])
    assert [(f["type"], f["claimed_amount_minor"]) for f in out] == [("fx_error", 2500)]
    assert _findings(pol, [inv], [], fx, [_ar(1, correct)]) == []       # exact posting -> silence
    assert _findings(pol, [inv], [], fx, [_ar(1, correct + 1)]) == []   # +-1 rounding noise ignored


def test_fx_error_on_payment_cash_row_and_correct_date_evidence():
    pol = _auditor()
    fx = [_fx("2026-03-09", "EUR", "500.000000"), _fx("2026-03-10", "EUR", "505.000000")]
    inv = _inv(1, "2026-03-09T09:00:00+05:00", 10000, "EUR")
    # payment 2026-03-09T23:30:00-03:00 is 03-10 company time (07:30+05:00 on 03-10): rate 505
    pay = _pay(1, "2026-03-09T23:30:00-03:00", "EUR", 10000, [{"invoice_id": "INV-1", "amount_minor": 10000}])
    ar_credit = pol.conv(10000, "EUR", "500.000000")
    wrong_cash = pol.conv(10000, "EUR", "500.000000")                   # booked with the own-offset date
    right_cash = pol.conv(10000, "EUR", "505.000000")
    ledger = [_ar(1, ar_credit),
              _row("R-cash", ACC_CASH, debit=wrong_cash, pay="PAY-1"),
              _row("R-arc", ACC_AR, credit=ar_credit, pay="PAY-1")]
    out = _findings(pol, [inv], [pay], fx, ledger)
    fxs = _by_type(out, "fx_error")
    assert len(fxs) == 1 and fxs[0]["affected_ids"] == ["PAY-1"]
    assert fxs[0]["claimed_amount_minor"] == right_cash - wrong_cash
    assert fxs[0]["evidence_ids"] == ["D-pay1", "R-cash", "fx:2026-03-10:EUR:KZT"]
    assert _by_type(out, "payment_misallocation") == []                  # credits equal conv(refs)


# =============================================================================== payments
def _pay_ledger(pay_id, cash, ar_credit=0, unapplied=0, fx_gain=0, fx_loss=0, n=1):
    rows = [_row(f"R-c{n}", ACC_CASH, debit=cash, pay=pay_id)]
    if ar_credit:
        rows.append(_row(f"R-a{n}", ACC_AR, credit=ar_credit, pay=pay_id))
    if unapplied:
        rows.append(_row(f"R-u{n}", ACC_UNAPPLIED, credit=unapplied, pay=pay_id))
    if fx_gain:
        rows.append(_row(f"R-g{n}", ACC_FX, credit=fx_gain, pay=pay_id))
    if fx_loss:
        rows.append(_row(f"R-l{n}", ACC_FX, debit=fx_loss, pay=pay_id))
    return rows


def test_payment_distractors_are_not_misallocations():
    pol = _auditor()
    invoices = [_inv(1, "2026-03-02T10:00:00+05:00", 100000), _inv(2, "2026-03-03T10:00:00+05:00", 200000)]
    payments = [
        # unreferenced prepayment: refs null, everything to unapplied cash
        _pay(1, "2026-03-10T10:00:00+05:00", BASE, 50000, None),
        # overpayment: refs 100000, paid 130000, excess unapplied
        _pay(2, "2026-03-11T10:00:00+05:00", BASE, 130000, [{"invoice_id": "INV-1", "amount_minor": 100000}]),
        # split payment: one invoice paid in two parts, each fully credited
        _pay(3, "2026-03-12T10:00:00+05:00", BASE, 120000, [{"invoice_id": "INV-2", "amount_minor": 120000}]),
        _pay(4, "2026-03-13T10:00:00+05:00", BASE, 80000, [{"invoice_id": "INV-2", "amount_minor": 80000}]),
        # refund clearing the overpayment: negative amount, no refs
        _pay(5, "2026-03-20T10:00:00+05:00", BASE, -30000, None),
    ]
    ledger = [_ar(1, 100000), _ar(2, 200000)]
    ledger += _pay_ledger("PAY-1", 50000, unapplied=50000, n=1)
    ledger += _pay_ledger("PAY-2", 130000, ar_credit=100000, unapplied=30000, n=2)
    ledger += _pay_ledger("PAY-3", 120000, ar_credit=120000, n=3)
    ledger += _pay_ledger("PAY-4", 80000, ar_credit=80000, n=4)
    ledger += [_row("R-r1", ACC_UNAPPLIED, debit=30000, pay="PAY-5"), _row("R-r2", ACC_CASH, credit=30000, pay="PAY-5")]
    assert _findings(pol, invoices, payments, [], ledger) == []


def test_realised_fx_gain_loss_is_not_an_error_but_a_missing_ar_credit_is():
    pol = _auditor()
    fx = [_fx("2026-03-02", "EUR", "500.000000"), _fx("2026-03-10", "EUR", "520.000000")]
    inv = _inv(1, "2026-03-02T10:00:00+05:00", 10000, "EUR")
    ar_deb = pol.conv(10000, "EUR", "500.000000")
    cash = pol.conv(10000, "EUR", "520.000000")
    pay = _pay(1, "2026-03-10T10:00:00+05:00", "EUR", 10000, [{"invoice_id": "INV-1", "amount_minor": 10000}])
    good = [_ar(1, ar_deb)] + _pay_ledger("PAY-1", cash, ar_credit=ar_deb, fx_gain=cash - ar_deb)
    assert _findings(pol, [inv], [pay], fx, good) == []              # gain absorbed by 7900, all correct
    # misallocation: only 60% credited, the rest parked in unapplied cash
    part = ar_deb * 6 // 10
    bad = [_ar(1, ar_deb)] + _pay_ledger("PAY-1", cash, ar_credit=part, unapplied=ar_deb - part,
                                         fx_gain=cash - ar_deb)
    out = _findings(pol, [inv], [pay], fx, bad)
    assert [(f["type"], f["affected_ids"], f["claimed_amount_minor"]) for f in out] == [
        ("payment_misallocation", ["PAY-1"], ar_deb - part)]
    ev = out[0]["evidence_ids"]
    # payment doc + every ledger row of the payment + foreign invoice doc + invoice-date fx id
    assert ev[0] == "D-pay1" and "D-inv1" in ev and "fx:2026-03-02:EUR:KZT" in ev
    assert {"R-c1", "R-a1", "R-u1", "R-g1"} <= set(ev) and len(ev) <= 10


def test_misallocation_amount_uses_invoice_rate_and_base_currency_needs_no_fx_evidence():
    pol = _auditor()
    inv = _inv(1, "2026-03-02T10:00:00+05:00", 100000)
    pay = _pay(1, "2026-03-10T10:00:00+05:00", BASE, 100000, [{"invoice_id": "INV-1", "amount_minor": 100000}])
    ledger = [_ar(1, 100000)] + _pay_ledger("PAY-1", 100000, unapplied=100000)       # AR credit missing
    out = _findings(pol, [inv], [pay], [], ledger)
    assert len(out) == 1 and out[0]["type"] == "payment_misallocation"
    assert out[0]["claimed_amount_minor"] == 100000
    assert out[0]["evidence_ids"] == ["D-pay1", "R-c1", "R-u1"]          # no invoice doc / fx for base ccy


# =============================================================================== duplicates
def _lines(period="2026-02", po="PO-1", qty=2):
    return [{"line": 1, "description": "Ceramic tiles", "quantity": qty, "unit_price_minor": 50000,
             "vat_rate": "0.12", "service_period": period, "po_ref": po}]


def _setup_dups():
    pol = _auditor()
    a = _inv(1, "2026-03-02T10:00:00+05:00", 112000)
    b = _inv(2, "2026-03-04T10:00:00+05:00", 112000)          # same everything -> duplicate of INV-1
    c = _inv(3, "2026-03-05T10:00:00+05:00", 112000, entity="C9")   # other entity: never a candidate
    d = _inv(4, "2026-03-30T10:00:00+05:00", 112000)          # same entity & total, other service period
    ledger = [_ar(1, 112000), _ar(2, 112000), _ar(3, 112000), _ar(4, 112000)]
    ix = pol._index([a, b, c, d], [], [], ledger)
    return pol, ix


def test_duplicate_candidates_are_closest_first_and_exclude_other_entities():
    pol, ix = _setup_dups()
    order = pol.candidate_fetch_order(ix)
    assert set(order) == {"INV-1", "INV-2", "INV-4"}
    assert order[:2] in (["INV-1", "INV-2"], ["INV-2", "INV-1"])      # the 2-day pair before the 28-day one


def test_duplicate_decided_on_line_items_and_recurring_service_period_is_ignored():
    pol, ix = _setup_dups()
    assert pol.duplicate_findings(ix) == []                            # nothing fetched yet -> no claims
    pol.docs["INV-1"] = {"line_items": _lines("2026-02")}
    pol.docs["INV-2"] = {"line_items": _lines("2026-02")}
    pol.docs["INV-4"] = {"line_items": _lines("2026-03")}               # legit recurring: new service_period
    out = pol.duplicate_findings(ix)
    assert [(f["type"], f["affected_ids"], f["claimed_amount_minor"]) for f in out] == [
        ("duplicate_invoice", ["INV-2"], 112000)]
    assert set(out[0]["evidence_ids"]) == {"D-inv1", "D-inv2", "R-ar2"}


def test_different_po_ref_or_quantity_is_not_a_duplicate():
    pol, ix = _setup_dups()
    pol.docs["INV-1"] = {"line_items": _lines("2026-02", po="PO-1")}
    pol.docs["INV-2"] = {"line_items": _lines("2026-02", po="PO-2")}
    pol.docs["INV-4"] = {"line_items": _lines("2026-02", po="PO-3")}
    assert pol.duplicate_findings(ix) == []


def test_duplicate_order_uses_instants_not_date_strings():
    pol = _auditor()
    # INV-A has the earlier date string but the LATER instant (+14:00 vs -10:00): INV-B is the original
    a = _inv(1, "2026-03-10T01:00:00-10:00", 112000)    # 11:00Z on 03-10
    b = _inv(2, "2026-03-10T20:00:00+14:00", 112000)    # 06:00Z on 03-10  (earlier instant)
    ix = pol._index([a, b], [], [], [_ar(1, 112000), _ar(2, 112000)])
    pol.docs["INV-1"] = {"line_items": _lines()}
    pol.docs["INV-2"] = {"line_items": _lines()}
    out = pol.duplicate_findings(ix)
    assert [f["affected_ids"] for f in out] == [["INV-1"]]


def test_unposted_duplicate_makes_no_claim_and_costs_no_fetch():
    pol = _auditor()
    a = _inv(1, "2026-03-02T10:00:00+05:00", 112000)
    b = _inv(2, "2026-04-03T10:00:00+05:00", 112000)    # post-period copy, not in the ledger
    ix = pol._index([a, b], [], [], [_ar(1, 112000)])
    assert pol.candidate_fetch_order(ix) == []


# =============================================================================== real episodes
@functools.lru_cache(maxsize=None)
def _holdout_runs(policy: str):
    out = []
    for e in load_splits()["holdout"]:
        r = run_episode(policy, e["seed"], resolve_config(e), mode=MODE_EVAL, remote=False)
        s = report_scores(r["grader_report"])
        s.update(steps=r["steps"], category=e.get("category"), seed=e["seed"],
                 ended_by=r["grader_report"].get("ended_by"), run=r)
        out.append(s)
    return out


def _mean(rows, key):
    return sum(float(r[key]) for r in rows) / len(rows)


def test_holdout_eval_careful_auditor_solves_the_environment_from_public_evidence():
    rows = _holdout_runs("careful_auditor")
    assert len(rows) == 20
    assert all(r["ended_by"] == "submit" and not r["invalid"] for r in rows)
    assert all(r["steps"] <= MAX_STEPS for r in rows)
    assert _mean(rows, "f1") >= 0.95
    # precision / recall are 0 (0/0) on the clean books by convention; judge them on books with findings
    with_issues = [r for r in rows if r["category"] != "clean"]
    assert _mean(with_issues, "precision") >= 0.95
    assert _mean(with_issues, "recall") >= 0.95
    assert _mean(rows, "shared") >= 0.85          # near the ceiling of 1 - 0.002*steps


def test_holdout_clean_companies_get_no_false_accusations():
    clean = [r for r in _holdout_runs("careful_auditor") if r["category"] == "clean"]
    assert len(clean) == 5
    for r in clean:
        assert r["fp"] == 0 and r["tp"] == 0 and r["f1"] == 1.0
        assert submitted_findings(r["run"]["trajectory"])["findings"] == []


def test_holdout_interactions_are_resolved_into_both_linked_findings():
    inter = [r for r in _holdout_runs("careful_auditor") if r["category"] == "interaction"]
    assert len(inter) == 5
    assert sum(r["fn"] for r in inter) == 0 and sum(r["fp"] for r in inter) == 0
    assert all(r["tp"] >= 2 for r in inter)


def test_reward_separates_careful_work_from_the_naive_reconcile_heuristic():
    careful, naive = _holdout_runs("careful_auditor"), _holdout_runs("reconcile")
    assert _mean(careful, "f1") - _mean(naive, "f1") >= 0.3
    assert _mean(careful, "shared") - _mean(naive, "shared") >= 0.5
    assert _mean(careful, "precision") > _mean(naive, "precision")
    # same inspection effort order of magnitude: the gain comes from correct reasoning, not from budget
    assert _mean(careful, "steps") <= MAX_STEPS


def test_careful_auditor_uses_only_documented_tools_and_valid_args():
    for r in _holdout_runs("careful_auditor")[:4]:
        acts = [t["action"] for t in r["run"]["trajectory"][1:]]
        assert acts[-1]["action"] == "submit"
        assert {a["action"] for a in acts} <= {"list_records", "get_record", "submit"}
        for t in r["run"]["trajectory"][1:]:
            assert t["observation"]["error"] is None, t["observation"]["error"]    # no wasted error steps
            a = t["action"]
            if a["action"] == "list_records":
                assert a["args"]["limit"] == 50
        sub = submitted_findings(r["run"]["trajectory"])
        for f in sub["findings"]:
            assert set(f) == {"type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids",
                              "explanation"}
            assert len(f["evidence_ids"]) <= 10 and f["currency"] == r["run"]["trajectory"][0][
                "observation"]["manifest"]["base_currency"]


def test_runs_are_deterministic_and_replayable():
    e = load_splits()["holdout"][9]                        # an interaction episode
    cfg = resolve_config(e)
    r1 = run_episode("careful_auditor", e["seed"], cfg, mode=MODE_EVAL, remote=False)
    r2 = run_episode("careful_auditor", e["seed"], cfg, mode=MODE_EVAL, remote=False)
    assert trajectory_hash(r1["trajectory"]) == trajectory_hash(r2["trajectory"])
    actions = [t["action"] for t in r1["trajectory"][1:]]
    rp = replay_actions(actions, e["seed"], cfg, mode=MODE_EVAL, remote=False)
    assert trajectory_hash(rp["trajectory"]) == trajectory_hash(r1["trajectory"])


def test_remote_grader_run_matches_the_in_process_run():
    e = load_splits()["holdout"][5]
    cfg = resolve_config(e)
    local = run_episode("careful_auditor", e["seed"], cfg, mode=MODE_EVAL, remote=False)
    remote = run_episode("careful_auditor", e["seed"], cfg, mode=MODE_EVAL, remote=True)
    assert trajectory_hash(local["trajectory"]) == trajectory_hash(remote["trajectory"])
    assert report_scores(local["grader_report"]) == report_scores(remote["grader_report"])


def test_eval_observations_do_not_reveal_the_score_to_the_policy():
    r = _holdout_runs("careful_auditor")[6]["run"]
    last = r["trajectory"][-1]
    assert last["terminated"] and last["reward"] == 0.0           # eval mode: reward stays with the grader
    assert "metrics" not in json.dumps(last["observation"])


# =============================================================================== size / budget stress
def _big_config() -> dict:
    from auditgym.generator.config import config_with_force
    cfg = config_with_force("holdout", None)
    cfg = json.loads(json.dumps(cfg))
    cfg["company"]["n_noise_postings"] = [900, 1000]
    cfg["company"]["n_invoices"] = [85, 90]
    cfg["company"]["n_customers"] = [15, 18]
    cfg["issues"] = {"duplicate_invoice": [2, 2], "payment_misallocation": [2, 2], "fx_error": [2, 2],
                     "period_cutoff": [2, 2]}
    cfg["interactions"] = {"fx_partial_payment_at_boundary": [1, 1], "cutoff_with_wrong_rate_date": [1, 1]}
    cfg["clean_probability"] = 0.0
    cfg.pop("force", None)
    return cfg


def test_maximum_size_episode_fits_in_the_step_budget_and_finds_every_issue():
    r = run_episode("careful_auditor", 3, _big_config(), mode=MODE_EVAL, remote=False)
    counts = r["trajectory"][0]["observation"]["manifest"]["counts"]
    assert counts["ledger_rows"] >= 1800                       # ~38 ledger pages
    s = report_scores(r["grader_report"])
    assert r["steps"] <= MAX_STEPS and r["grader_report"]["ended_by"] == "submit"
    assert s["fn"] == 0 and s["fp"] == 0 and s["tp"] == 12
    assert r["steps"] >= counts["ledger_rows"] // 50           # it really paged the whole ledger


@pytest.mark.parametrize("budget", [3, 12, 30])
def test_tiny_budget_degrades_gracefully_with_a_valid_submission(budget):
    e = load_splits()["holdout"][12]
    env = AuditEnv(mode=MODE_EVAL, max_steps=budget)
    pol = get_policy("careful_auditor", 0)
    obs, _ = env.reset(seed=e["seed"], config=resolve_config(e))
    pol.reset(0, obs)
    terminated = truncated = False
    n = 0
    while not (terminated or truncated) and n < budget + 5:
        obs, _r, terminated, truncated, _i = env.step(pol.act(obs))
        n += 1
    rep = env.grader_report()
    assert terminated and not truncated and n <= budget
    assert rep["ended_by"] == "submit" and not rep.get("invalid_submission")


def test_whole_holdout_sweep_is_fast():
    import time
    t0 = time.perf_counter()
    for e in load_splits()["holdout"][:5]:
        run_episode("careful_auditor", e["seed"], resolve_config(e), mode=MODE_EVAL, remote=False)
    assert time.perf_counter() - t0 < 20.0
