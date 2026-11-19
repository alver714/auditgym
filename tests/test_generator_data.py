"""Generator data & truth tests (SPEC sections 1-3, 8).

Covers: validator ok across splits/profiles, double-entry balance, rules engine == truth,
evidence sufficiency, distractors, interactions, per-type configurability, sizes, id variety,
manifest/format conventions, explicit nulls, episode_id opacity.

Episodes are cached per (seed, config) for speed; tests must treat them as read-only.
"""
from __future__ import annotations

import copy
import os
import csv
import hashlib
import io
import json
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from functools import lru_cache

import pytest

from auditgym import rules, validate
from auditgym.constants import (CURRENCY_EXPONENT, DISTRACTORS, ENV_VERSION, INTERACTION_CUTOFF_FX_DATE,
                                INTERACTION_FX_PARTIAL_BOUNDARY, INTERACTIONS, ISSUE_TYPES, LEDGER_FIELDS,
                                INVOICE_FIELDS, PAYMENT_FIELDS, FX_FIELDS, SCHEMA_VERSION, ACC_AR,
                                ACC_FX, ACC_UNAPPLIED, ACC_CASH, ACC_REVENUE, ACC_VAT, ACCOUNTS)
from auditgym.episode import Episode, render_public
from auditgym.content_id import content_episode_id
from auditgym.generator.build import config_hash, generate_episode
from auditgym.generator.config import (ConfigError, config_with_force, resolve_config, sample_spec)

# ------------------------------------------------------------------ helpers
SPLITS = ("train", "dev", "holdout")
BASE_SEED = {"train": 1000, "dev": 2000, "holdout": 3000}
N_NATURAL = 24                      # natural (unforced) seeds per split


def _freeze(force):
    return json.dumps(force, sort_keys=True)


@lru_cache(maxsize=None)
def _cfg(split: str, force_json: str):
    return config_with_force(split, json.loads(force_json))


@lru_cache(maxsize=None)
def gen(seed: int, split: str = "train", force_json: str = "null"):
    force = json.loads(force_json)
    cfg = _cfg(split, force_json) if force is not None else split
    return generate_episode(seed, cfg)


def gen_force(seed, force, split="train"):
    return gen(seed, split, _freeze(force))


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def period(ep):
    return dt(ep.manifest["period_start"]), dt(ep.manifest["period_end"])


def tol_of(expected: int) -> int:
    return max(1, expected // 1000)


def group_balance(ledger):
    groups = {}
    for r in ledger:
        k = (r["timestamp"], r.get("payment_id") or r.get("invoice_id"))
        groups[k] = groups.get(k, 0) + r["debit_minor"] - r["credit_minor"]
    return groups


def assert_balanced(ledger, what=""):
    assert sum(r["debit_minor"] for r in ledger) == sum(r["credit_minor"] for r in ledger), what
    unb = {k: v for k, v in group_balance(ledger).items() if v != 0}
    assert not unb, f"{what}: unbalanced posting groups {list(unb.items())[:3]}"


def match_truth(violations, issues):
    """Multiset match of rules violations to truth issues on (type, affected set, amount within tol)."""
    pool = list(violations)
    for iss in issues:
        exp = iss["expected_amount_minor"]
        hit = None
        for v in pool:
            if (v["type"] == iss["type"] and set(v["affected_ids"]) == set(iss["affected_ids"])
                    and abs(v["amount_minor"] - exp) <= iss["tolerance_minor"]):
                hit = v
                break
        assert hit is not None, f"truth issue {iss['issue_id']} {iss['type']} not found by rules engine"
        pool.remove(hit)
    assert not pool, f"rules engine reports extra violations: {[(v['type'], v['affected_ids']) for v in pool]}"


# profiles: (name, force) -- every issue type alone, each interaction, combos, max 12, clean
def _single(t, n):
    return {"issues": {t: n}}


PROFILES = (
    [("clean", {"clean": True})]
    + [(f"only_{t}_1", _single(t, 1)) for t in ISSUE_TYPES]
    + [(f"only_{t}_3", _single(t, 3)) for t in ISSUE_TYPES]
    + [(f"inter_{i}_1", {"interactions": {i: 1}}) for i in INTERACTIONS]
    + [(f"inter_{i}_2", {"interactions": {i: 2}}) for i in INTERACTIONS]
    + [("both_interactions", {"interactions": {INTERACTIONS[0]: 1, INTERACTIONS[1]: 1}}),
       ("all_types_1", {"issues": {t: 1 for t in ISSUE_TYPES}}),
       ("all_types_plus_inter", {"issues": {t: 1 for t in ISSUE_TYPES},
                                 "interactions": {i: 1 for i in INTERACTIONS}}),
       ("max12_issues", {"issues": {t: 3 for t in ISSUE_TYPES}}),
       ("max12_interactions", {"interactions": {INTERACTIONS[0]: 3, INTERACTIONS[1]: 3}}),
       ("max12_mixed", {"issues": {t: 1 for t in ISSUE_TYPES}, "interactions": {INTERACTIONS[0]: 2, INTERACTIONS[1]: 2}}),
       ("max12_dups", _single("duplicate_invoice", 12)),
       ("max12_fx", _single("fx_error", 12))]
)
PROFILE_IDS = [p[0] for p in PROFILES]
PROFILE_SEEDS = (11, 12, 13, 14, 15, 16)


def all_profile_episodes(split="train", seeds=PROFILE_SEEDS):
    for name, force in PROFILES:
        for s in seeds:
            yield name, s, gen_force(s + BASE_SEED[split], force, split)


# ------------------------------------------------------------------ validator ok
@pytest.mark.parametrize("split", SPLITS)
def test_validate_episode_ok_natural_seeds(split):
    bad = []
    for i in range(N_NATURAL):
        ep = gen(BASE_SEED[split] + i, split)
        rep = validate.validate_episode(ep)
        if not rep["ok"]:
            bad.append((split, BASE_SEED[split] + i, {k: v["detail"] for k, v in rep["checks"].items() if not v["ok"]}))
    assert not bad, bad[:3]


@pytest.mark.parametrize("name,force", PROFILES, ids=PROFILE_IDS)
def test_validate_episode_ok_forced_profiles(name, force):
    for split in SPLITS:
        for s in PROFILE_SEEDS[:3]:
            ep = gen_force(BASE_SEED[split] + s, force, split)
            rep = validate.validate_episode(ep)
            assert rep["ok"], (name, split, s, {k: v["detail"] for k, v in rep["checks"].items() if not v["ok"]})


def test_validate_reports_all_check_groups():
    rep = validate.validate_episode(gen(1001))
    names = set(rep["checks"])
    for must in ("a_clean_balanced", "a_clean_groups_balanced", "a_clean_no_violations", "b_final_balanced",
                 "b_final_groups_balanced", "b_rules_match_truth", "c_evidence_sets", "d_no_leak", "e_counts",
                 "e_sizes"):
        assert must in names, must


# ------------------------------------------------------------------ balance
def test_ledger_posting_groups_balanced_clean_snapshot_and_final():
    n = 0
    for name, s, ep in all_profile_episodes():
        clean = ep.provenance["clean_records"]["ledger"]
        assert_balanced(clean, f"clean snapshot {name}/{s}")
        assert_balanced(ep.ledger, f"final {name}/{s}")
        n += 1
    assert n == len(PROFILES) * len(PROFILE_SEEDS)


def test_every_ledger_row_is_one_sided_in_base_currency_with_known_account():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:2]):
        base = ep.manifest["base_currency"]
        for r in ep.ledger:
            assert (r["debit_minor"] > 0) != (r["credit_minor"] > 0), (name, s, r)
            assert r["debit_minor"] >= 0 and r["credit_minor"] >= 0
            assert r["currency"] == base
            assert r["account"] in ACCOUNTS


def test_ledger_rows_only_inside_period_and_canonical_order():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:2]):
        ps, pe = period(ep)
        keys = []
        for r in ep.ledger:
            assert ps <= dt(r["timestamp"]) <= pe, (name, s, r["timestamp"])
            keys.append((dt(r["timestamp"]), r["row_id"]))
        assert keys == sorted(keys), "ledger not in canonical (timestamp, row_id) order"
        assert [dt(i["issued_at"]) for i in ep.invoices] == sorted(dt(i["issued_at"]) for i in ep.invoices)
        assert [dt(p["timestamp"]) for p in ep.payments] == sorted(dt(p["timestamp"]) for p in ep.payments)


def test_posting_structure_of_invoices_follows_policy():
    """Dr AR = conv(total); VAT+revenue credits = AR debit (ignoring injected issues: use the clean snapshot)."""
    for s in range(1001, 1013):
        ep = gen(s)
        cl = ep.provenance["clean_records"]
        by_inv = {}
        for r in cl["ledger"]:
            if r["invoice_id"] and r["payment_id"] is None:
                by_inv.setdefault(r["invoice_id"], []).append(r)
        for inv in cl["invoices"]:
            rows = by_inv.get(inv["invoice_id"])
            ps, pe = period(ep)
            inside = ps <= dt(inv["issued_at"]) <= pe
            if not inside:
                assert rows is None, f"post-period invoice {inv['invoice_id']} posted in clean books"
                continue
            assert rows, f"in-period invoice {inv['invoice_id']} not posted"
            dr = [r for r in rows if r["account"] == ACC_AR]
            assert len(dr) == 1 and dr[0]["debit_minor"] > 0
            cr = sum(r["credit_minor"] for r in rows if r["account"] in (ACC_VAT, ACC_REVENUE))
            assert cr == dr[0]["debit_minor"]
            assert {r["timestamp"] for r in rows} == {dr[0]["timestamp"]}


# ------------------------------------------------------------------ rules engine == truth
def test_rules_engine_violations_equal_truth_across_profiles():
    for name, s, ep in all_profile_episodes():
        match_truth(rules.find_violations(ep), ep.truth["issues"])


@pytest.mark.parametrize("split", SPLITS)
def test_rules_engine_violations_equal_truth_natural(split):
    for i in range(N_NATURAL):
        ep = gen(BASE_SEED[split] + i, split)
        match_truth(rules.find_violations(ep), ep.truth["issues"])


def test_clean_snapshot_has_no_violations_and_clean_episodes_have_no_truth():
    for name, force in PROFILES:
        for s in PROFILE_SEEDS:
            ep = gen_force(s + 1000, force)
            snap = type(ep)(ep.manifest, ep.policy, ep.provenance["clean_records"]["ledger"],
                            ep.provenance["clean_records"]["invoices"], ep.provenance["clean_records"]["payments"], ep.fx)
            assert rules.find_violations(snap) == [], (name, s)
    for s in range(1, 15):
        ep = gen_force(s, {"clean": True})
        assert ep.truth["issues"] == []
        assert rules.find_violations(ep) == []
        assert ep.provenance["changes"] == []
        assert ep.ledger == ep.provenance["clean_records"]["ledger"]


def test_rules_violation_currency_is_base_and_amounts_positive():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:2]):
        for v in rules.find_violations(ep):
            assert v["currency"] == ep.manifest["base_currency"]
            assert v["amount_minor"] > 0


def test_truth_issue_shape_and_ids_do_not_encode_type():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:3]):
        issues = ep.truth["issues"]
        assert ep.truth["episode_id"] == ep.episode_id
        assert ep.truth["base_currency"] == ep.manifest["base_currency"]
        assert [i["issue_id"] for i in issues] == [f"iss_{n:02d}" for n in range(1, len(issues) + 1)]
        for i in issues:
            assert set(i) == {"issue_id", "type", "affected_ids", "expected_amount_minor", "currency",
                              "tolerance_minor", "evidence_sets", "scenario"}
            assert i["type"] in ISSUE_TYPES
            assert isinstance(i["expected_amount_minor"], int) and not isinstance(i["expected_amount_minor"], bool)
            assert i["expected_amount_minor"] > 0
            assert i["tolerance_minor"] == tol_of(i["expected_amount_minor"])
            assert i["currency"] == ep.manifest["base_currency"]
            assert i["affected_ids"] and all(isinstance(a, str) for a in i["affected_ids"])


def test_issue_ids_are_shuffled_not_sorted_by_type():
    """Across many multi-type episodes the type order along iss_01.. must not be constant."""
    orders = set()
    for s in range(1, 25):
        ep = gen_force(s, {"issues": {t: 1 for t in ISSUE_TYPES}})
        orders.add(tuple(i["type"] for i in ep.truth["issues"]))
    assert len(orders) >= 4, orders


def test_affected_ids_refer_to_correct_tables_and_match_policy_shape():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:3]):
        inv = {i["invoice_id"] for i in ep.invoices}
        pay = {p["payment_id"] for p in ep.payments}
        for i in ep.truth["issues"]:
            assert len(i["affected_ids"]) == 1
            a = i["affected_ids"][0]
            if i["type"] in ("duplicate_invoice", "period_cutoff"):
                assert a in inv
            elif i["type"] == "payment_misallocation":
                assert a in pay
            else:
                assert a in inv or a in pay


# ------------------------------------------------------------------ evidence
def test_every_evidence_set_sufficient_public_and_bounded():
    n_sets = 0
    for name, s, ep in all_profile_episodes():
        pub = ep.public_ids()
        for iss in ep.truth["issues"]:
            assert iss["evidence_sets"]
            for ev in iss["evidence_sets"]:
                n_sets += 1
                assert 0 < len(ev) <= 10 and len(set(ev)) == len(ev)
                assert all(e in pub for e in ev), (name, s, [e for e in ev if e not in pub])
                ok, amt = rules.check_evidence(iss, ev, ep)
                assert ok, (name, s, iss["issue_id"], iss["type"])
                assert abs(amt - iss["expected_amount_minor"]) <= iss["tolerance_minor"]
    assert n_sets > 100


def test_evidence_sets_have_spec_composition():
    seen = Counter()
    for name, s, ep in all_profile_episodes():
        ledger_ids = {r["row_id"] for r in ep.ledger}
        inv_docs = {i["doc_id"] for i in ep.invoices}
        pay_docs = {p["doc_id"] for p in ep.payments}
        for iss in ep.truth["issues"]:
            for ev in iss["evidence_sets"]:
                rows = [e for e in ev if e in ledger_ids]
                docs = [e for e in ev if e in inv_docs or e in pay_docs]
                fxs = [e for e in ev if e.startswith("fx:")]
                t = iss["type"]
                if t == "duplicate_invoice":
                    assert len(docs) == 2 and len(rows) == 1 and not fxs
                elif t == "period_cutoff":
                    assert len(docs) == 1 and len(rows) == 1 and not fxs
                elif t == "fx_error":
                    assert len(docs) == 1 and len(rows) == 1 and len(fxs) == 1
                else:
                    assert any(d in pay_docs for d in docs) and len(rows) >= 1
                seen[t] += 1
    assert set(seen) == set(ISSUE_TYPES)


def test_evidence_fx_id_is_the_correct_rate_for_fx_errors():
    n = 0
    for s in range(1, 21):
        ep = gen_force(s, _single("fx_error", 2))
        fxids = {fx_id for fx_id in (f"fx:{r['date']}:{r['source_currency']}:{r['target_currency']}" for r in ep.fx)}
        for iss in ep.truth["issues"]:
            fx_ev = [e for e in iss["evidence_sets"][0] if e.startswith("fx:")]
            assert len(fx_ev) == 1 and fx_ev[0] in fxids
            assert fx_ev[0].endswith(":" + ep.manifest["base_currency"])
            n += 1
    assert n >= 30


def test_evidence_insufficient_when_key_ids_removed_or_bogus_ids_added():
    checked = Counter()
    for s in range(1, 13):
        ep = gen_force(s, {"issues": {t: 1 for t in ISSUE_TYPES}})
        for iss in ep.truth["issues"]:
            ev = iss["evidence_sets"][0]
            assert rules.check_evidence(iss, [], ep) == (False, 0)
            assert not rules.check_evidence(iss, ev + ["R-deadbeef"], ep)[0]
            docs = [e for e in ev if e.startswith("D-")]
            assert not rules.check_evidence(iss, [e for e in ev if e not in docs], ep)[0]
            if iss["type"] != "payment_misallocation":
                for e in ev:
                    assert not rules.check_evidence(iss, [x for x in ev if x != e], ep)[0], (iss["type"], e)
            else:
                for e in ev:
                    if e.startswith("fx:") or e.startswith("D-"):
                        assert not rules.check_evidence(iss, [x for x in ev if x != e], ep)[0], (iss["type"], e)
            # evidence for one issue must not prove a different issue of another type
            checked[iss["type"]] += 1
    assert set(checked) == set(ISSUE_TYPES)


def test_evidence_set_of_one_issue_does_not_certify_a_different_affected_id():
    for s in range(1, 9):
        ep = gen_force(s, _single("fx_error", 2))
        a, b = ep.truth["issues"][:2]
        if a["affected_ids"] != b["affected_ids"]:
            assert not rules.check_evidence(b, a["evidence_sets"][0], ep)[0]


# ------------------------------------------------------------------ distractors
def test_distractors_present_in_clean_and_dirty_episodes_and_match_spec_counts():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:3]):
        spec_d = ep.provenance["spec"]["distractors"]
        got = Counter(d["kind"] for d in ep.provenance["distractors"])
        for k in DISTRACTORS:
            assert got.get(k, 0) == spec_d[k], (name, s, k, got, spec_d)
        for k, lo in (("split_payment", 1), ("legit_recurring_amount", 1), ("timezone_boundary_ok", 1),
                      ("realized_fx_gain_loss", 1)):
            assert got[k] >= lo, (name, s, k)


def test_clean_books_with_distractors_produce_zero_truth_and_zero_violations():
    kinds = Counter()
    for s in range(1, 41):
        ep = gen_force(s, {"clean": True})
        for d in ep.provenance["distractors"]:
            kinds[d["kind"]] += 1
        assert ep.truth["issues"] == []
        assert rules.find_violations(ep) == []
        assert validate.validate_episode(ep)["ok"]
    assert set(kinds) == set(DISTRACTORS), kinds


def test_distractor_patterns_exist_in_public_data():
    for s in range(1, 31):
        ep = gen_force(s, {"clean": True})
        inv = {i["invoice_id"]: i for i in ep.invoices}
        pay = {p["payment_id"]: p for p in ep.payments}
        ps, pe = period(ep)
        off = ep.manifest["utc_offset"]
        base = ep.manifest["base_currency"]
        for d in ep.provenance["distractors"]:
            k, r = d["kind"], d["record_ids"]
            assert all(x in inv or x in pay for x in r), (k, r)
            if k == "split_payment":
                paying = [p for p in ep.payments if any(x["invoice_id"] == r[0] for x in (p["invoice_refs"] or []))]
                assert len(paying) >= 2
                assert sum(x["amount_minor"] for p in paying for x in p["invoice_refs"] if x["invoice_id"] == r[0]) \
                    == inv[r[0]]["total_minor"]
            elif k == "overpayment_refund":
                over, refund = pay[r[1]], pay[r[2]]
                assert over["amount_minor"] > sum(x["amount_minor"] for x in over["invoice_refs"])
                assert refund["amount_minor"] < 0 and refund["invoice_refs"] is None
            elif k == "legit_recurring_amount":
                a, b = inv[r[0]], inv[r[1]]
                assert a["entity_id"] == b["entity_id"] and a["total_minor"] == b["total_minor"]
                assert a["currency"] == b["currency"]
                assert a["line_items"][0]["service_period"] != b["line_items"][0]["service_period"]
            elif k == "timezone_boundary_ok":
                a = inv[r[0]]
                assert ps <= dt(a["issued_at"]) <= pe
                assert not a["issued_at"].endswith(off)
            elif k == "realized_fx_gain_loss":
                assert inv[r[0]]["currency"] != base
                assert any(x["payment_id"] == r[1] and x["account"] == ACC_FX for x in ep.ledger)
            elif k == "unreferenced_prepayment":
                assert pay[r[0]]["invoice_refs"] is None and pay[r[0]]["amount_minor"] > 0
                rows = [x for x in ep.ledger if x["payment_id"] == r[0]]
                assert any(x["account"] == ACC_UNAPPLIED for x in rows)


def test_distractors_do_not_leak_into_truth_for_dirty_books():
    """Every violation reported by the rules engine is in truth, so no distractor can be a hidden extra issue."""
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:3]):
        assert len(rules.find_violations(ep)) == len(ep.truth["issues"])


# ------------------------------------------------------------------ interactions
def test_interactions_yield_two_linked_issues_same_scenario():
    for inter, types in ((INTERACTION_FX_PARTIAL_BOUNDARY, {"payment_misallocation", "fx_error"}),
                         (INTERACTION_CUTOFF_FX_DATE, {"period_cutoff", "fx_error"})):
        for n in (1, 2, 3):
            for s in range(1, 7):
                ep = gen_force(s * 7 + n, {"interactions": {inter: n}})
                issues = ep.truth["issues"]
                assert len(issues) == 2 * n
                assert all(i["scenario"] == inter for i in issues)
                by_aff = {}
                for i in issues:
                    by_aff.setdefault(tuple(i["affected_ids"]), []).append(i)
                assert len(by_aff) == n
                for aff, pair in by_aff.items():
                    assert len(pair) == 2 and {p["type"] for p in pair} == types
                chg = [c for c in ep.provenance["changes"] if c["injector"] == inter]
                assert len(chg) == n
                for c in chg:
                    assert len(c["issue_ids"]) == 2


def test_interaction_payment_scenario_has_partial_allocation_and_boundary_date_mismatch():
    for s in range(1, 13):
        ep = gen_force(s, {"interactions": {INTERACTION_FX_PARTIAL_BOUNDARY: 1}})
        pid = ep.truth["issues"][0]["affected_ids"][0]
        p = next(x for x in ep.payments if x["payment_id"] == pid)
        assert p["currency"] != ep.manifest["base_currency"]
        own = dt(p["timestamp"])
        co_off = rules._parse_offset(ep.manifest["utc_offset"])
        assert own.date() != own.astimezone(co_off).date(), "payment not near a day boundary in its own offset"
        posted_ar = sum(r["credit_minor"] for r in ep.ledger if r["payment_id"] == pid and r["account"] == ACC_AR)
        mis = next(i for i in ep.truth["issues"] if i["type"] == "payment_misallocation")
        assert mis["expected_amount_minor"] > 0
        assert posted_ar >= 0


def test_interaction_cutoff_scenario_is_foreign_invoice_just_after_period_end():
    for s in range(1, 13):
        ep = gen_force(s, {"interactions": {INTERACTION_CUTOFF_FX_DATE: 1}})
        iid = ep.truth["issues"][0]["affected_ids"][0]
        inv = next(x for x in ep.invoices if x["invoice_id"] == iid)
        ps, pe = period(ep)
        assert inv["currency"] != ep.manifest["base_currency"]
        issued = dt(inv["issued_at"])
        assert issued > pe
        assert issued - pe <= timedelta(days=11)
        ar = [r for r in ep.ledger if r["invoice_id"] == iid and r["account"] == ACC_AR]
        assert len(ar) == 1 and ps <= dt(ar[0]["timestamp"]) <= pe


def test_non_interaction_issues_have_null_scenario_and_interactions_only_when_forced():
    for t in ISSUE_TYPES:
        for s in range(1, 6):
            ep = gen_force(s, _single(t, 2))
            assert all(i["scenario"] is None for i in ep.truth["issues"])
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:2]):
        if "inter" not in name and "max12_mixed" != name and "both" not in name and name != "all_types_plus_inter" \
                and name != "max12_interactions":
            assert all(i["scenario"] is None for i in ep.truth["issues"]), name


# ------------------------------------------------------------------ independent configurability
@pytest.mark.parametrize("t", ISSUE_TYPES)
def test_forcing_one_type_yields_only_that_type_with_exact_count(t):
    for n in (1, 2, 4):
        for s in range(1, 9):
            ep = gen_force(s * 13 + n, _single(t, n))
            assert Counter(i["type"] for i in ep.truth["issues"]) == Counter({t: n}), (t, n, s)
            assert [c["injector"] for c in ep.provenance["changes"]] == [t] * n


def test_forcing_zero_for_all_types_and_unlisted_default_to_zero():
    ep = gen_force(5, {"issues": {"fx_error": 0}})
    assert ep.truth["issues"] == []
    ep = gen_force(5, {"interactions": {INTERACTION_CUTOFF_FX_DATE: 1}})
    assert Counter(i["type"] for i in ep.truth["issues"]) == Counter({"period_cutoff": 1, "fx_error": 1})


def test_mixed_counts_are_exact_for_every_type():
    force = {"issues": {"duplicate_invoice": 2, "payment_misallocation": 3, "fx_error": 1, "period_cutoff": 2}}
    for s in range(1, 9):
        ep = gen_force(s, force)
        assert Counter(i["type"] for i in ep.truth["issues"]) == Counter(force["issues"])


def test_max_twelve_issues_reached_and_cap_enforced():
    ep = gen_force(3, {"issues": {t: 3 for t in ISSUE_TYPES}})
    assert len(ep.truth["issues"]) == 12
    ep = gen_force(3, {"interactions": {INTERACTIONS[0]: 3, INTERACTIONS[1]: 3}})
    assert len(ep.truth["issues"]) == 12
    with pytest.raises(ConfigError):
        sample_spec(1, _cfg("train", _freeze({"issues": {"fx_error": 13}})))
    with pytest.raises(ConfigError):
        sample_spec(1, _cfg("train", _freeze({"issues": {t: 3 for t in ISSUE_TYPES}, "interactions": {INTERACTIONS[0]: 1}})))


def test_natural_sampling_never_exceeds_twelve_even_with_wide_ranges():
    cfg = {"config_id": "wide", "config_version": "1.0.0",
           "issues": {t: [3, 6] for t in ISSUE_TYPES},
           "interactions": {i: [1, 3] for i in INTERACTIONS}, "clean_probability": 0.0}
    sizes = Counter()
    for s in range(1, 41):
        spec = sample_spec(s, resolve_config(cfg))
        total = sum(spec["issues"].values()) + 2 * sum(spec["interactions"].values())
        assert total <= 12
        sizes[total] += 1
    assert 12 in sizes
    # and the generator honours a capped spec
    ep = generate_episode(1, cfg)
    assert 0 < len(ep.truth["issues"]) <= 12
    assert validate.validate_episode(ep)["ok"]


def test_clean_probability_is_respected_by_natural_sampling():
    always = {"config_id": "c1", "config_version": "1.0.0", "clean_probability": 1.0}
    never = {"config_id": "c0", "config_version": "1.0.0", "clean_probability": 0.0,
             "issues": {t: [1, 1] for t in ISSUE_TYPES}, "interactions": {i: [0, 0] for i in INTERACTIONS}}
    for s in range(1, 8):
        assert generate_episode(s, always).truth["issues"] == []
        assert len(generate_episode(s, never).truth["issues"]) == 4


def test_train_dev_holdout_natural_mix_contains_clean_and_dirty_episodes():
    for split in SPLITS:
        counts = [len(gen(BASE_SEED[split] + i, split).truth["issues"]) for i in range(N_NATURAL)]
        assert any(c == 0 for c in counts), split
        assert any(c > 0 for c in counts), split


# ------------------------------------------------------------------ issue-specific semantic checks
def test_duplicate_invoice_is_same_entity_currency_lineitems_and_later_in_period():
    for s in range(1, 21):
        ep = gen_force(s, _single("duplicate_invoice", 2))
        inv = {i["invoice_id"]: i for i in ep.invoices}
        for iss in ep.truth["issues"]:
            dup = inv[iss["affected_ids"][0]]
            twins = [i for i in ep.invoices if i is not dup and i["entity_id"] == dup["entity_id"]
                     and i["currency"] == dup["currency"] and i["line_items"] == dup["line_items"]]
            assert twins, "duplicate has no original"
            orig = twins[0]
            assert (dt(orig["issued_at"]), orig["invoice_id"]) < (dt(dup["issued_at"]), dup["invoice_id"])
            assert timedelta(0) <= dt(dup["issued_at"]) - dt(orig["issued_at"]) <= timedelta(days=5, hours=1)
            ps, pe = period(ep)
            assert ps <= dt(dup["issued_at"]) <= pe
            ar = [r for r in ep.ledger if r["invoice_id"] == dup["invoice_id"] and r["account"] == ACC_AR]
            assert len(ar) == 1 and iss["expected_amount_minor"] == ar[0]["debit_minor"]
            assert orig["doc_id"] in iss["evidence_sets"][0] and dup["doc_id"] in iss["evidence_sets"][0]


def test_legit_recurring_is_not_flagged_as_duplicate_by_line_items():
    """Recurring pairs share entity+total but differ in line_items (service_period)."""
    for s in range(1, 21):
        ep = gen_force(s, {"clean": True})
        for d in ep.provenance["distractors"]:
            if d["kind"] == "legit_recurring_amount":
                inv = {i["invoice_id"]: i for i in ep.invoices}
                a, b = inv[d["record_ids"][0]], inv[d["record_ids"][1]]
                assert a["line_items"] != b["line_items"]


def test_period_cutoff_invoice_is_outside_period_but_posted_inside_and_tricky_half_the_time():
    """The cutoff invoice is outside the period as an instant -- after period_end (the common case) or, in the mirror
    case, before period_start -- yet its AR debit is posted inside the period."""
    tricky = plain = after = before = 0
    for s in range(1, 41):
        ep = gen_force(s, _single("period_cutoff", 2))
        ps, pe = period(ep)
        for iss in ep.truth["issues"]:
            inv = next(i for i in ep.invoices if i["invoice_id"] == iss["affected_ids"][0])
            issued = dt(inv["issued_at"])
            assert issued > pe or issued < ps, "cutoff invoice not outside period as an instant"
            ar = [r for r in ep.ledger if r["invoice_id"] == inv["invoice_id"] and r["account"] == ACC_AR]
            assert len(ar) == 1 and ps <= dt(ar[0]["timestamp"]) <= pe
            assert iss["expected_amount_minor"] == ar[0]["debit_minor"]
            if issued > pe:
                after += 1
                assert pe - dt(ar[0]["timestamp"]) <= timedelta(days=3, hours=1)
                # "looks in-period" if the date written in the invoice's own offset is <= period_end's date
                if issued.date() <= pe.date() or inv["issued_at"][:10] <= ep.manifest["period_end"][:10]:
                    tricky += 1
                else:
                    plain += 1
            else:
                before += 1
                assert dt(ar[0]["timestamp"]) - ps <= timedelta(days=3, hours=1)
    assert tricky >= 5 and plain >= 5, (tricky, plain)
    assert before >= 5 and after >= 20, (before, after)


def test_period_cutoff_hits_the_inclusive_boundaries_in_both_directions():
    """period_end + 1 s (just out, after) and period_start - 1 s (just out, before) are generated as cutoff
    candidates and are really picked; period_end / period_start themselves are correct, in-period invoices."""
    end_plus = start_minus = 0
    for s in range(1, 61):
        ep = gen_force(s, _single("period_cutoff", 3))
        ps, pe = period(ep)
        for iss in ep.truth["issues"]:
            inv = next(i for i in ep.invoices if i["invoice_id"] == iss["affected_ids"][0])
            issued = dt(inv["issued_at"])
            end_plus += issued == pe + timedelta(seconds=1)
            start_minus += issued == ps - timedelta(seconds=1)
        assert any(dt(i["issued_at"]) == pe + timedelta(seconds=1) for i in ep.invoices)
        assert any(dt(i["issued_at"]) == ps - timedelta(seconds=1) for i in ep.invoices)
    assert end_plus >= 5 and start_minus >= 5, (end_plus, start_minus)


def test_timezone_boundary_distractor_has_exact_period_start_and_end_invoices():
    seen = Counter()
    for s in range(1, 81):
        ep = gen_force(s, {"clean": True, "distractors": {"timezone_boundary_ok": 2}})
        ps, pe = period(ep)
        inv = {i["invoice_id"]: i for i in ep.invoices}
        assert rules.find_violations(ep) == [] and ep.truth["issues"] == []
        for d in ep.provenance["distractors"]:
            if d["kind"] != "timezone_boundary_ok":
                continue
            a = inv[d["record_ids"][0]]
            seen[d["variant"]] += 1
            assert ps <= dt(a["issued_at"]) <= pe
            if d["variant"] == "exact_end":
                assert dt(a["issued_at"]) == pe
            if d["variant"] == "exact_start":
                assert dt(a["issued_at"]) == ps
            rows = [r for r in ep.ledger if r["invoice_id"] == a["invoice_id"]]
            assert rows and all(ps <= dt(r["timestamp"]) <= pe for r in rows)
    assert set(seen) == {"offset_edge", "exact_end", "exact_start"}, seen


def test_post_period_invoices_present_and_not_posted_in_ledger_unless_cutoff_issue():
    for s in range(1, 21):
        ep = gen_force(s, {"issues": {"duplicate_invoice": 1, "fx_error": 1}})
        ps, pe = period(ep)
        post = [i for i in ep.invoices if dt(i["issued_at"]) > pe]
        assert 3 <= len(post) <= 9, len(post)
        pre = [i for i in ep.invoices if dt(i["issued_at"]) < ps]      # mirror set: prior-period documents
        assert 2 <= len(pre) <= 4, len(pre)
        paid = {x["invoice_id"] for p in ep.payments for x in (p["invoice_refs"] or [])}
        posted = {r["invoice_id"] for r in ep.ledger if r["invoice_id"]}
        for i in pre:
            assert i["invoice_id"] not in posted and i["invoice_id"] not in paid
            assert ps - dt(i["issued_at"]) <= timedelta(days=8)
        for i in post:
            assert i["invoice_id"] not in posted
            assert dt(i["issued_at"]) - pe <= timedelta(days=11)
    for s in range(1, 11):
        ep = gen_force(s, _single("period_cutoff", 2))
        _, pe = period(ep)
        posted = {r["invoice_id"] for r in ep.ledger if r["invoice_id"]}
        ps, _ = period(ep)
        outside = [i for i in ep.invoices if dt(i["issued_at"]) > pe or dt(i["issued_at"]) < ps]
        flagged = {i["affected_ids"][0] for i in ep.truth["issues"]}
        assert {i["invoice_id"] for i in outside if i["invoice_id"] in posted} == flagged


def test_fx_error_changes_base_amount_by_more_than_twice_tolerance():
    for s in range(1, 21):
        ep = gen_force(s, _single("fx_error", 2))
        for iss in ep.truth["issues"]:
            assert iss["expected_amount_minor"] >= 2 * iss["tolerance_minor"] + 1
            # the foreign document is in a non-base currency
            a = iss["affected_ids"][0]
            doc = next((d for d in ep.invoices if d["invoice_id"] == a), None) or \
                next(p for p in ep.payments if p["payment_id"] == a)
            assert doc["currency"] != ep.manifest["base_currency"]


def test_fx_error_variants_cover_invoice_and_payment_targets():
    kinds = Counter()
    for s in range(1, 41):
        ep = gen_force(s, _single("fx_error", 2))
        inv = {i["invoice_id"] for i in ep.invoices}
        for iss in ep.truth["issues"]:
            kinds["invoice" if iss["affected_ids"][0] in inv else "payment"] += 1
    assert kinds["invoice"] > 0 and kinds["payment"] > 0, kinds


def test_payment_misallocation_has_missing_or_partial_ar_credit():
    full = partial = 0
    for s in range(1, 31):
        ep = gen_force(s, _single("payment_misallocation", 2))
        for iss in ep.truth["issues"]:
            pid = iss["affected_ids"][0]
            p = next(x for x in ep.payments if x["payment_id"] == pid)
            assert p["invoice_refs"], "misallocation on unreferenced payment"
            rows = [r for r in ep.ledger if r["payment_id"] == pid]
            assert any(r["account"] == ACC_UNAPPLIED for r in rows)
            if any(r["account"] == ACC_AR for r in rows):
                partial += 1
            else:
                full += 1
    assert partial > 0 and full > 0, (partial, full)


# ------------------------------------------------------------------ sizes
@pytest.mark.parametrize("split", SPLITS)
def test_sizes_in_ranges(split):
    for i in range(N_NATURAL):
        ep = gen(BASE_SEED[split] + i, split)
        assert 200 <= len(ep.ledger) <= 2000, len(ep.ledger)
        assert 20 <= len(ep.invoices) + len(ep.payments) <= 200
        assert 0 <= len(ep.truth["issues"]) <= 12
        assert ep.invoices and ep.payments and ep.fx


def test_sizes_in_ranges_forced_profiles():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:3]):
        assert 200 <= len(ep.ledger) <= 2000
        assert 20 <= len(ep.invoices) + len(ep.payments) <= 200
        assert len(ep.truth["issues"]) <= 12


def test_extreme_company_sizes_still_valid():
    for company in ({"n_invoices": 14, "n_customers": 4, "n_noise_postings": 60},
                    {"n_invoices": 105, "n_customers": 18, "n_noise_postings": 250}):
        for s in range(1, 4):
            ep = generate_episode(s, config_with_force("train", {"issues": {"fx_error": 1, "duplicate_invoice": 1},
                                                                  "company": company}))
            rep = validate.validate_episode(ep)
            assert rep["ok"], {k: v["detail"] for k, v in rep["checks"].items() if not v["ok"]}


def test_company_force_pins_base_currency_and_timezone():
    for base in ("KZT", "USD", "EUR"):
        ep = generate_episode(4, config_with_force("train", {"clean": True,
                              "company": {"base_currency": base, "timezone": ["Asia/Almaty", "+05:00"]}}))
        assert ep.manifest["base_currency"] == base
        assert ep.manifest["timezone"] == "Asia/Almaty" and ep.manifest["utc_offset"] == "+05:00"
        assert validate.validate_episode(ep)["ok"]


# ------------------------------------------------------------------ ids
INV_RE = {"year_seq": re.compile(r"^INV-\d{4}-\d{4}$"), "slash": re.compile(r"^SI/\d{4}-[A-Z]$"),
          "alnum": re.compile(r"^F-[A-Z0-9]{6}$")}
PAY_RE = {"pay": re.compile(r"^PAY-\d{6}$"), "rcpt": re.compile(r"^RCPT-\d{5}$"), "bnk": re.compile(r"^BNK[A-Z0-9]{8}$")}
ROW_RE = re.compile(r"^R-[0-9a-f]{8}$")
DOC_RE = re.compile(r"^D-[0-9a-f]{10}$")


def test_id_formats_and_single_style_per_company():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:3]):
        ist = ep.provenance["spec"]["id_styles"]
        for i in ep.invoices:
            assert INV_RE[ist["invoice"]].match(i["invoice_id"]), (ist, i["invoice_id"])
            assert DOC_RE.match(i["doc_id"])
        for p in ep.payments:
            assert PAY_RE[ist["payment"]].match(p["payment_id"]), (ist, p["payment_id"])
            assert DOC_RE.match(p["doc_id"])
        for r in ep.ledger:
            assert ROW_RE.match(r["row_id"])


def test_ids_unique_within_episode_and_doc_ids_disjoint_from_row_ids():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:3]):
        for vals in ([r["row_id"] for r in ep.ledger], [d["doc_id"] for d in ep.invoices + ep.payments],
                     [i["invoice_id"] for i in ep.invoices], [p["payment_id"] for p in ep.payments]):
            assert len(vals) == len(set(vals))
        assert len(ep.public_ids()) == len(ep.ledger) + len(ep.invoices) + len(ep.payments) + len(ep.fx)


def test_ids_vary_across_seeds_and_styles_vary():
    styles = Counter()
    inv_sets, row_sets = [], []
    for s in range(1, 41):
        ep = gen_force(s, {"clean": True})
        ist = ep.provenance["spec"]["id_styles"]
        styles["inv_" + ist["invoice"]] += 1
        styles["pay_" + ist["payment"]] += 1
        styles["ent_" + ist["entity"]] += 1
        inv_sets.append({i["invoice_id"] for i in ep.invoices})
        row_sets.append({r["row_id"] for r in ep.ledger})
    for k in INV_RE:
        assert styles["inv_" + k] > 0, styles
    for k in PAY_RE:
        assert styles["pay_" + k] > 0, styles
    assert sum(1 for k in styles if k.startswith("ent_")) >= 2
    for a in range(0, 40, 2):
        assert len(row_sets[a] & row_sets[a + 1]) == 0
        assert len(inv_sets[a] & inv_sets[a + 1]) <= 0.2 * len(inv_sets[a])


def test_row_ids_are_not_sequential_and_injected_rows_indistinguishable():
    for s in range(1, 21):
        ep = gen_force(s, {"issues": {t: 2 for t in ISSUE_TYPES}})
        ids = [int(r["row_id"][2:], 16) for r in ep.ledger]
        diffs = [abs(b - a) for a, b in zip(ids, ids[1:])]
        assert sum(1 for d in diffs if d == 1) < 3, "row ids look sequential"
        added = [x for c in ep.provenance["changes"] for x in c["records_added"]]
        assert added
        public = {r["row_id"] for r in ep.ledger} | {i["invoice_id"] for i in ep.invoices} | \
                 {p["payment_id"] for p in ep.payments} | {i["doc_id"] for i in ep.invoices + ep.payments}
        for a in added:
            assert a in public
            if a.startswith("R-"):
                assert ROW_RE.match(a)
            if a.startswith("D-"):
                assert DOC_RE.match(a)
        # noise: injected invoice follows the company's style exactly
        ist = ep.provenance["spec"]["id_styles"]["invoice"]
        for a in added:
            if a in {i["invoice_id"] for i in ep.invoices}:
                assert INV_RE[ist].match(a)


def test_duplicate_invoice_ids_drawn_from_same_sequence_style_as_normal_ones():
    for s in range(1, 21):
        ep = gen_force(s, _single("duplicate_invoice", 2))
        ist = ep.provenance["spec"]["id_styles"]["invoice"]
        for iss in ep.truth["issues"]:
            assert INV_RE[ist].match(iss["affected_ids"][0])
        if ist == "year_seq":
            nums = sorted(int(i["invoice_id"].split("-")[2]) for i in ep.invoices)
            dup_nums = {int(i["affected_ids"][0].split("-")[2]) for i in ep.truth["issues"]}
            lo, hi = nums[0], nums[-1]
            assert all(lo <= d <= hi for d in dup_nums)


# ------------------------------------------------------------------ manifest & formats
def test_manifest_fields_and_formats():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:3]):
        m = ep.manifest
        # no config_id / config_version / config_hash: the policy must not be able to tell which split it is on
        assert set(m) == {"schema_version", "env_version", "episode_id",
                          "company_name", "base_currency", "timezone", "utc_offset", "period_start",
                          "period_end", "counts", "rounding", "money"}
        assert m["schema_version"] == SCHEMA_VERSION and m["env_version"] == ENV_VERSION
        assert ep.provenance["config_id"] == "train" and ep.provenance["config_version"] == "1.3.0"
        assert re.fullmatch(r"ep_[0-9a-f]{12}", m["episode_id"])
        assert m["company_name"].startswith("Fictional ")
        assert m["base_currency"] in CURRENCY_EXPONENT
        assert m["timezone"] in ("Asia/Almaty", "Asia/Tokyo", "UTC", "Asia/Dubai", "America/Bogota", "Asia/Kolkata")
        assert re.fullmatch(r"[+-]\d{2}:\d{2}", m["utc_offset"])
        ps, pe = dt(m["period_start"]), dt(m["period_end"])
        assert ps.utcoffset() is not None and ps.utcoffset() == pe.utcoffset()
        assert m["period_start"].endswith(m["utc_offset"]) and m["period_end"].endswith(m["utc_offset"])
        assert (ps.hour, ps.minute, ps.second) == (0, 0, 0) and ps.day == 1
        assert (pe.hour, pe.minute, pe.second) == (23, 59, 59)
        assert (pe + timedelta(seconds=1)).day == 1 and (pe + timedelta(seconds=1)).month != pe.month
        assert m["counts"] == {"ledger_rows": len(ep.ledger), "invoices": len(ep.invoices),
                               "payments": len(ep.payments), "documents": len(ep.invoices) + len(ep.payments),
                               "fx_rates": len(ep.fx)}
        assert m["rounding"] == {"mode": "half_even", "unit": "base_minor", "fx_rate_places": 6}
        assert m["money"]["amounts"] == "integer minor units"
        assert m["money"]["exponents"] == CURRENCY_EXPONENT


def test_episode_id_is_content_addressed_and_does_not_reveal_seed():
    for split in SPLITS:
        for i in range(10):
            seed = BASE_SEED[split] + i
            ep = gen(seed, split)
            cfg = resolve_config(split)
            assert re.fullmatch(r"ep_[0-9a-f]{12}", ep.episode_id)
            # pure function of the public records (manifest minus episode_id included)
            assert ep.episode_id == content_episode_id(ep)
            probe = Episode(dict(ep.manifest, episode_id="ep_000000000000"), ep.policy, ep.ledger,
                            ep.invoices, ep.payments, ep.fx)
            assert content_episode_id(probe) == ep.episode_id
            # every private/public place that embeds the id carries the same value
            assert ep.truth["episode_id"] == ep.provenance["episode_id"] == ep.manifest["episode_id"]
            # neither the legacy unsalted formula nor a seed-keyed formula reproduces it
            legacy = "ep_" + hashlib.sha256(
                f"{seed}|{split}|{cfg['config_version']}|{ENV_VERSION}".encode()).hexdigest()[:12]
            assert ep.episode_id != legacy
            assert ep.provenance["config_hash"] == config_hash(cfg)
            assert "id_salt" not in ep.provenance and "id_salt" not in cfg
            # changing any record changes the id
            mod = copy.deepcopy(ep)
            mod.ledger[0]["debit_minor"] = int(mod.ledger[0]["debit_minor"]) + 1
            assert content_episode_id(mod) != ep.episode_id
    ids = {gen(s, "train").episode_id for s in range(1000, 1024)}
    assert len(ids) == 24
    # same seed under different configs gives different ids
    assert gen(1001, "train").episode_id != gen(1001, "dev").episode_id
    # no seed in any public text of the manifest/policy, and ids are not seed-derived numbers
    for s in (123457, 987651, 7654321):
        ep = generate_episode(s, "train")
        assert str(s) not in ep.episode_id and format(s, "x") not in ep.episode_id
        for name in ("manifest.json", "policy.json"):
            assert str(s) not in render_public(ep)[name]
        assert '"seed"' not in render_public(ep)["manifest.json"]


def test_episode_id_depends_on_config_content_including_force():
    base = copy.deepcopy(resolve_config("train"))
    a = generate_episode(77, base).episode_id
    assert generate_episode(77, base).episode_id == a
    # same seed + config_id + version, different force blocks / contents => different ids (no aliasing)
    f1 = generate_episode(77, config_with_force(base, {"issues": {"fx_error": 1}})).episode_id
    f2 = generate_episode(77, config_with_force(base, {"issues": {"fx_error": 2}})).episode_id
    f3 = generate_episode(77, config_with_force(base, {"clean": True})).episode_id
    assert len({a, f1, f2, f3}) == 4
    changed = copy.deepcopy(base)
    changed["company"]["foreign_invoice_share"] = 0.9
    assert generate_episode(77, changed).episode_id != a


def test_id_salt_mechanism_is_gone(monkeypatch):
    base = resolve_config("train")
    a = generate_episode(5, base).episode_id
    monkeypatch.setenv("AUDITGYM_ID_SALT", "hidden-grader-secret")
    assert generate_episode(5, base).episode_id == a


def _all_timestamps(ep):
    for r in ep.ledger:
        yield "ledger", r["timestamp"]
    for i in ep.invoices:
        yield "inv.issued", i["issued_at"]
        yield "inv.due", i["due_at"]
    for p in ep.payments:
        yield "pay", p["timestamp"]


ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")


def test_all_timestamps_iso8601_with_explicit_offsets():
    offsets = Counter()
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:2]):
        for kind, ts in _all_timestamps(ep):
            assert ISO_RE.match(ts), (kind, ts)
            assert dt(ts).utcoffset() is not None
            offsets[ts[-6:]] += 1
    assert len(offsets) >= 3, offsets                      # customers use varied offsets
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:1]):
        for r in ep.ledger:                                  # ledger is booked in company offset
            assert r["timestamp"].endswith(ep.manifest["utc_offset"])


def test_due_dates_are_15_30_or_45_days_after_issue_and_payments_inside_period():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:2]):
        ps, pe = period(ep)
        for i in ep.invoices:
            assert dt(i["due_at"]) - dt(i["issued_at"]) in (timedelta(days=15), timedelta(days=30), timedelta(days=45))
            assert i["due_at"][-6:] == i["issued_at"][-6:]
        for p in ep.payments:
            assert ps <= dt(p["timestamp"]) <= pe


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def test_money_fields_are_integer_minor_units_and_invoice_totals_follow_line_rule():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:2]):
        for r in ep.ledger:
            assert _is_int(r["debit_minor"]) and _is_int(r["credit_minor"])
        for i in ep.invoices:
            assert _is_int(i["total_minor"]) and i["total_minor"] > 0
            total = 0
            for li in i["line_items"]:
                assert _is_int(li["quantity"]) and _is_int(li["unit_price_minor"])
                assert set(li) == {"line", "description", "quantity", "unit_price_minor", "vat_rate",
                                   "service_period", "po_ref"}
                assert re.fullmatch(r"\d{4}-(\d{2}|W\d{2})", li["service_period"])
                net = li["quantity"] * li["unit_price_minor"]
                vat = (Decimal(net) * Decimal(li["vat_rate"])).quantize(Decimal(1), rounding="ROUND_HALF_EVEN")
                total += net + int(vat)
            assert total == i["total_minor"], (i["invoice_id"], total, i["total_minor"])
            assert [li["line"] for li in i["line_items"]] == list(range(1, len(i["line_items"]) + 1))
        for p in ep.payments:
            assert _is_int(p["amount_minor"])
            for ref in p["invoice_refs"] or []:
                assert _is_int(ref["amount_minor"]) and ref["amount_minor"] > 0


def test_fx_table_format_business_days_range_and_random_walk():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:2]):
        base = ep.manifest["base_currency"]
        ps, pe = period(ep)
        lo = ps.date() - timedelta(days=10)
        hi = pe.date() + timedelta(days=12)
        by_cur = {}
        for r in ep.fx:
            assert list(r) == list(FX_FIELDS)
            assert r["target_currency"] == base and r["source_currency"] != base
            assert re.fullmatch(r"\d+\.\d{6}", r["rate"]), r["rate"]
            assert Decimal(r["rate"]) > 0
            d = date.fromisoformat(r["date"])
            assert d.weekday() < 5, "weekend fx rate present"
            assert lo <= d <= hi
            by_cur.setdefault(r["source_currency"], []).append((d, Decimal(r["rate"])))
        used = {i["currency"] for i in ep.invoices} | {p["currency"] for p in ep.payments}
        assert used - {base} <= set(by_cur)
        for cur, rows in by_cur.items():
            rows.sort()
            expect = [lo + timedelta(days=k) for k in range((hi - lo).days + 1) if (lo + timedelta(days=k)).weekday() < 5]
            assert [d for d, _ in rows] == expect, cur
            for (d1, r1), (d2, r2) in zip(rows, rows[1:]):
                assert r1 != r2, "consecutive rates identical"


def test_foreign_currency_documents_and_payment_currency_matches_invoice_currency():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:3]):
        inv = {i["invoice_id"]: i for i in ep.invoices}
        foreign = 0
        for i in ep.invoices:
            foreign += i["currency"] != ep.manifest["base_currency"]
        assert foreign > 0
        for p in ep.payments:
            for ref in p["invoice_refs"] or []:
                assert inv[ref["invoice_id"]]["currency"] == p["currency"]


def test_record_field_sets_are_exact():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:1]):
        for r in ep.ledger:
            assert set(r) == set(LEDGER_FIELDS)
        for r in ep.invoices:
            assert set(r) == set(INVOICE_FIELDS)
        for r in ep.payments:
            assert set(r) == set(PAYMENT_FIELDS)


def test_policy_json_structure_and_boundaries_match_manifest():
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:2]):
        pol = ep.policy
        for k in ("revenue_recognition", "period_boundaries", "fx", "ledger", "postings", "duplicate_rule",
                  "finding_amounts", "finding_affected_ids", "finding_currency", "amount_tolerance", "evidence"):
            assert k in pol, k
        pb = pol["period_boundaries"]
        assert pb["period_start"] == ep.manifest["period_start"] and pb["period_end"] == ep.manifest["period_end"]
        assert pb["inclusive"] is True
        assert pb["timezone"] == ep.manifest["timezone"] and pb["utc_offset"] == ep.manifest["utc_offset"]
        assert set(pol["finding_amounts"]) == set(ISSUE_TYPES)
        assert set(pol["postings"]) == {"invoice", "payment", "refund"}


# ------------------------------------------------------------------ explicit nulls
def test_null_refs_explicit_in_rendered_files_and_loaded_as_none():
    from auditgym.episode import load_public, save_episode
    import tempfile
    n_null_ref = n_null_entity = 0
    for s in range(1, 13):
        ep = gen_force(s, {"issues": {"fx_error": 1}})
        text = render_public(ep)
        for line in text["payments.jsonl"].splitlines():
            rec = json.loads(line)
            assert set(rec) == set(PAYMENT_FIELDS), "payment record missing a key (nulls must be explicit)"
            if rec["invoice_refs"] is None:
                assert '"invoice_refs":null' in line
                n_null_ref += 1
        for line in text["invoices.jsonl"].splitlines():
            assert set(json.loads(line)) == set(INVOICE_FIELDS)
        rows = list(csv.DictReader(io.StringIO(text["ledger.csv"])))
        assert list(rows[0].keys()) == list(LEDGER_FIELDS)
        for r in rows:
            for f in ("entity_id", "invoice_id", "payment_id"):
                assert f in r
        n_null_entity += sum(1 for r in rows if r["invoice_id"] == "" and r["payment_id"] == "")
        with tempfile.TemporaryDirectory() as td:
            pub, _ = save_episode(ep, td + "/pub", td + "/priv")
            back = load_public(pub)
        for r in back.ledger:
            for f in ("entity_id", "invoice_id", "payment_id"):
                assert r[f] is None or (isinstance(r[f], str) and r[f] != "")
        assert back.ledger == ep.ledger
        assert back.payments == ep.payments
    assert n_null_ref > 0 and n_null_entity > 0


def test_null_ref_payments_are_prepayments_or_refunds_only():
    for s in range(1, 25):
        ep = gen_force(s, {"clean": True})
        for p in ep.payments:
            if p["invoice_refs"] is None:
                assert p["amount_minor"] < 0 or any(
                    d["kind"] == "unreferenced_prepayment" and p["payment_id"] in d["record_ids"]
                    for d in ep.provenance["distractors"]), p["payment_id"]
            else:
                assert isinstance(p["invoice_refs"], list) and p["invoice_refs"]


def test_ledger_noise_rows_have_null_links_and_invoice_rows_have_ids():
    for s in range(1, 9):
        ep = gen_force(s, {"clean": True})
        noise = [r for r in ep.ledger if r["invoice_id"] is None and r["payment_id"] is None]
        assert len(noise) >= 100
        for r in noise:
            assert r["account"] not in (ACC_REVENUE, ACC_VAT) or r["entity_id"] is None
        inv_ids = {i["invoice_id"] for i in ep.invoices}
        pay_ids = {p["payment_id"] for p in ep.payments}
        for r in ep.ledger:
            assert r["invoice_id"] is None or r["invoice_id"] in inv_ids
            assert r["payment_id"] is None or r["payment_id"] in pay_ids


# ------------------------------------------------------------------ leakage in public data
def test_public_files_do_not_contain_truth_or_provenance_tokens():
    tokens = ("iss_", "expected_amount", "evidence_sets", "provenance", "truth", "scenario",
              INTERACTION_FX_PARTIAL_BOUNDARY, INTERACTION_CUTOFF_FX_DATE, "clean_records", "tolerance_minor",
              "root_cause", "records_added")
    for name, s, ep in all_profile_episodes(seeds=PROFILE_SEEDS[:2]):
        text = "\n".join(render_public(ep).values()).lower()
        for t in tokens:
            assert t.lower() not in text, (name, t)


def test_fictional_names_only():
    for s in range(1, 20):
        ep = gen(s)
        assert ep.manifest["company_name"].startswith("Fictional ")


# ------------------------------------------------------------------ config validation
def test_config_errors_raise_configerror():
    bad = [{"issues": {"nope": [0, 1]}}, {"issues": {"fx_error": [3, 1]}}, {"clean_probability": 2},
           {"force": {"bogus": 1}}, {"company": {"n_invoices": [3, 9]}}, {"company": {"base_currencies": ["XXX"]}}]
    for b in bad:
        with pytest.raises(ConfigError):
            resolve_config(dict(b, config_id="x", config_version="1"))
    with pytest.raises(ConfigError):
        resolve_config("definitely_not_a_config")


def test_named_configs_load_and_have_distinct_ids():
    ids = {resolve_config(n)["config_id"] for n in SPLITS}
    assert ids == set(SPLITS)


# ================================================================== verified jury issues: generator & truth
def _agree_matcher_and_rules(ep, iss, ev):
    """(matcher accepts, independent verifier accepts) for claiming exactly what `ev` proves."""
    from auditgym import matcher
    ok, amt = rules.check_evidence(iss, ev, ep)
    verifier = bool(ok) and abs(amt - iss["expected_amount_minor"]) <= iss["tolerance_minor"]
    f = matcher.normalize_finding({"type": iss["type"], "affected_ids": list(iss["affected_ids"]),
                                   "claimed_amount_minor": int(amt), "currency": iss["currency"],
                                   "evidence_ids": list(ev), "explanation": ""})
    return matcher.eligible(f, iss, ep.public_ids())[0], verifier


def test_misallocation_matcher_and_rules_agree_on_every_evidence_subset():
    """Every subset of the payment's evidence pool: the matcher (any-of-sets) accepts it exactly when
    rules.check_evidence proves the amount from it -- the canonical maximal set is not the only acceptable one."""
    import itertools
    cfg = config_with_force("holdout", {"issues": {"payment_misallocation": 2}})
    n_sets, subsets, accepted = Counter(), 0, 0
    for seed in list(range(94000, 94012)) + list(range(107100, 107106)):
        ep = generate_episode(seed, cfg)
        pay = {p["payment_id"]: p for p in ep.payments}
        for iss in ep.truth["issues"]:
            assert 2 <= len(iss["evidence_sets"]) <= 4
            n_sets[len(iss["evidence_sets"])] += 1
            for s in iss["evidence_sets"]:
                assert len(s) <= 10 and len(set(s)) == len(s)
            p = pay[iss["affected_ids"][0]]
            pool = {x for s in iss["evidence_sets"] for x in s}
            pool |= {r["row_id"] for r in ep.ledger if r["payment_id"] == p["payment_id"]}
            pool = sorted(pool)
            assert len(pool) <= 14
            for k in range(len(pool) + 1):
                for sub in itertools.combinations(pool, k):
                    if not sub:
                        continue
                    m, v = _agree_matcher_and_rules(ep, iss, sub)
                    assert m == v, (seed, iss["issue_id"], sub)
                    subsets += 1
                    accepted += m
    assert subsets > 2000 and accepted > 50
    assert n_sets


def test_misallocation_minimal_evidence_set_is_accepted_by_matcher_on_the_reported_seeds():
    """Seeds 94000-94039 (holdout) used to demand rows the proof does not need (40/40 mismatches)."""
    cfg = config_with_force("holdout", {"issues": {"payment_misallocation": 1}})
    foreign = base = 0
    for seed in range(94000, 94040):
        ep = generate_episode(seed, cfg)
        iss = ep.truth["issues"][0]
        pay = next(p for p in ep.payments if p["payment_id"] == iss["affected_ids"][0])
        rows = [r for r in ep.ledger if r["payment_id"] == pay["payment_id"]]
        ar = [r["row_id"] for r in rows if r["account"] == ACC_AR and r["credit_minor"] > 0]
        other = [r["row_id"] for r in rows if r["row_id"] not in ar]
        refs = [next(i["doc_id"] for i in ep.invoices if i["invoice_id"] == x["invoice_id"])
                for x in pay["invoice_refs"]] if pay["currency"] != ep.manifest["base_currency"] else []
        fx = [e for e in iss["evidence_sets"][0] if e.startswith("fx:")]
        minimal = [pay["doc_id"]] + ar + refs + fx + ([] if ar else other[:1])
        m, v = _agree_matcher_and_rules(ep, iss, minimal)
        assert m and v, seed
        # the canonical maximal set is still there (oracle uses evidence_sets[0])
        assert set(iss["evidence_sets"][0]) >= set(minimal)
        foreign += bool(fx)
        base += not fx
    assert foreign and base


def test_misallocation_evidence_sets_survive_id_transforms():
    from auditgym import transforms
    ep = generate_episode(94003, config_with_force("holdout", {"issues": {"payment_misallocation": 1}}))
    out = transforms.rename_ids(ep, 7)
    for iss in out.truth["issues"]:
        assert len(iss["evidence_sets"]) >= 2
        for s in iss["evidence_sets"]:
            assert all(e in out.public_ids() for e in s)
            assert rules.check_evidence(iss, s, out)[0]


# ------------------------------------------------------------------ strict config validation
def _good_cfg(**over):
    cfg = {"config_id": "t", "config_version": "1.0.0"}
    cfg.update(over)
    return cfg


def test_config_validator_rejects_typo_keys_bad_versions_and_out_of_range_values():
    bad = [
        _good_cfg(isues={"fx_error": [0, 1]}),                           # top-level typo
        _good_cfg(company={"foreign_invoice_shar": 0.3}),                # company typo
        _good_cfg(company={"foreign_invoice_share": 7}),                 # share outside [0, 1]
        _good_cfg(company={"foreign_invoice_share": -0.1}),
        _good_cfg(company={"foreign_invoice_share": True}),
        _good_cfg(company={"foreign_invoice_share": float("nan")}),
        _good_cfg(clean_probability=1.5),
        _good_cfg(clean_probability="0.2"),
        _good_cfg(company={"vat_rates": ["1.5"]}),
        _good_cfg(company={"vat_rates": []}),
        _good_cfg(company={"years": [1066]}),
        _good_cfg(company={"timezones": [["UTC"]]}),
        _good_cfg(issues={"fx_error": [0, 10 ** 9]}),
        _good_cfg(force={"company": {"month": 13}}),
        _good_cfg(force={"company": {"foreign_invoice_shar": 0.2}}),
        _good_cfg(force={"company": {"foreign_invoice_share": 7}}),
        _good_cfg(force={"clean": "yes"}),
        _good_cfg(force={"issues": [1]}),
        _good_cfg(company=[1]),
        _good_cfg(distractors=None),
    ]
    for b in bad:
        with pytest.raises(ConfigError):
            resolve_config(b)
    for ver in ("1", "1.0", "v1.0.0", "1.0.0-rc1", "1.0.0\n", " 1.0.0", "", "1.0.x", 1, None):
        with pytest.raises(ConfigError):
            resolve_config(_good_cfg(config_version=ver))
    for cid in ("", "has space", "../x", None, 3, "x" * 65):
        with pytest.raises(ConfigError):
            resolve_config(_good_cfg(config_id=cid))
    with pytest.raises(ConfigError):
        resolve_config("not even a dict")                               # name that does not exist
    with pytest.raises(ConfigError):
        resolve_config([1, 2])


def test_config_id_and_version_are_required_and_never_defaulted():
    for missing in ("config_id", "config_version"):
        cfg = _good_cfg()
        del cfg[missing]
        with pytest.raises(ConfigError, match=missing):
            resolve_config(cfg)
    with pytest.raises(ConfigError):
        resolve_config({})
    ok = resolve_config(_good_cfg(config_version="12.34.56"))
    assert ok["config_version"] == "12.34.56" and ok["config_id"] == "t"
    # validation is idempotent: a resolved config validates again unchanged
    assert resolve_config(ok) == ok


def test_config_boundary_values_are_accepted():
    cfg = resolve_config(_good_cfg(company={"foreign_invoice_share": 1, "vat_rates": ["0", 0.2]},
                                   clean_probability=0))
    assert cfg["company"]["foreign_invoice_share"] == 1 and cfg["clean_probability"] == 0
    resolve_config(_good_cfg(company={"foreign_invoice_share": 0}, clean_probability=1))
    ep = generate_episode(3, _good_cfg(company={"foreign_invoice_share": 1.0}))
    assert validate.validate_episode(ep)["ok"]


def test_shipped_configs_validate_against_the_json_schema_and_the_lock():
    import os
    from auditgym import schema_check
    from auditgym.generator.config import CONFIG_DIR, shipped_config_hashes, verify_config_lock
    sch = schema_check.load_schema("config.schema.json")
    names = [f for f in sorted(os.listdir(CONFIG_DIR)) if f.endswith(".json")
             and f not in ("splits.json", "CONFIG_VERSIONS.json")]
    assert names == ["dev.json", "holdout.json", "train.json"]
    for f in names:
        with open(os.path.join(CONFIG_DIR, f), encoding="utf-8") as fh:
            assert schema_check.check(json.load(fh), sch) == [], f
    assert schema_check.check(_good_cfg(config_version="1", bogus=1), sch)
    assert verify_config_lock() == []
    assert set(shipped_config_hashes()) == {"train@1.3.0", "dev@1.3.0", "holdout@1.3.0"}


def test_config_lock_fails_on_content_change_without_version_bump(tmp_path):
    import shutil
    from auditgym.generator.config import CONFIG_DIR, build_config_lock, verify_config_lock
    for fn in ("train.json", "dev.json", "holdout.json", "CONFIG_VERSIONS.json"):
        shutil.copy(os.path.join(CONFIG_DIR, fn), tmp_path / fn)
    assert verify_config_lock(str(tmp_path)) == []
    cfg = json.loads((tmp_path / "dev.json").read_text())
    cfg["company"]["foreign_invoice_share"] = 0.41                       # content change, same version
    (tmp_path / "dev.json").write_text(json.dumps(cfg))
    problems = verify_config_lock(str(tmp_path))
    assert len(problems) == 1 and "dev@1.3.0" in problems[0] and "without a config_version bump" in problems[0]
    cfg["config_version"] = "1.4.0"                                      # bump but forget to re-lock
    (tmp_path / "dev.json").write_text(json.dumps(cfg))
    problems = verify_config_lock(str(tmp_path))
    assert any("dev@1.4.0" in p and "not in" in p for p in problems)
    assert any("dev@1.3.0" in p and "locked but" in p for p in problems)
    (tmp_path / "CONFIG_VERSIONS.json").write_text(json.dumps(build_config_lock(str(tmp_path))))
    assert verify_config_lock(str(tmp_path)) == []                       # re-locked: consistent again
    os.remove(tmp_path / "CONFIG_VERSIONS.json")
    assert verify_config_lock(str(tmp_path))


def test_config_hash_binds_content_and_lives_only_in_grader_side_records():
    from auditgym.generator.config import config_hash as ch
    a = resolve_config("train")
    b = copy.deepcopy(a)
    assert ch(a) == ch(b) and re.fullmatch(r"[0-9a-f]{64}", ch(a))
    b["company"]["foreign_invoice_share"] = 0.9
    assert ch(a) != ch(b)
    # the same (id, version) with different content gives a different episode AND a different recorded hash
    e1, e2 = generate_episode(5, a), generate_episode(5, b)
    assert e1.episode_id != e2.episode_id
    assert e1.provenance["config_hash"] == ch(a) != e2.provenance["config_hash"]
    # nothing about the config identity is visible to the policy (public files / manifest / policy)
    for ep in (e1, e2):
        blob = "\n".join(render_public(ep).values())
        for tok in ("config_id", "config_version", "config_hash", e1.provenance["config_hash"][:12]):
            assert tok not in blob, tok
    assert (e1.provenance["config_id"], e1.provenance["config_version"]) == ("train", "1.3.0")
    # all three splits produce manifests of the same shape: the split cannot be read off the manifest
    shapes = {split: tuple(sorted(generate_episode(BASE_SEED[split], split).manifest)) for split in SPLITS}
    assert len(set(shapes.values())) == 1


# ------------------------------------------------------------------ duplicate invoices
def test_duplicate_is_strictly_later_than_original_and_rules_order_is_id_independent():
    for s in range(1, 41):
        ep = gen_force(s, _single("duplicate_invoice", 3))
        inv = {i["invoice_id"]: i for i in ep.invoices}
        chg = [c for c in ep.provenance["changes"] if c["injector"] == "duplicate_invoice"]
        assert len(chg) == 3
        for iss in ep.truth["issues"]:
            dup = inv[iss["affected_ids"][0]]
            orig = next(i for i in ep.invoices if i["invoice_id"] != dup["invoice_id"]
                        and i["entity_id"] == dup["entity_id"] and i["currency"] == dup["currency"]
                        and [{k: v for k, v in l.items() if k != "line"} for l in i["line_items"]]
                        == [{k: v for k, v in l.items() if k != "line"} for l in dup["line_items"]])
            assert dt(dup["issued_at"]) > dt(orig["issued_at"])          # distinct instants: no id tie-break needed
        ctx = rules._Ctx(ep)
        for iss in ep.truth["issues"]:
            dup = inv[iss["affected_ids"][0]]
            key = rules._dup_order_key(ctx, dup)
            assert key[0] == ctx.ts(dup["issued_at"]) and key[1] == ctx.ts(dup["due_at"])
            assert key[2] == ctx.first_post[dup["invoice_id"]]


def test_duplicate_tie_break_does_not_depend_on_ids():
    """Two identical-looking invoices with the SAME issued_at: due_at, then first posting time decide before any id."""
    ep = gen_force(11, _single("duplicate_invoice", 1))
    dup_id = ep.truth["issues"][0]["affected_ids"][0]
    dup = next(i for i in ep.invoices if i["invoice_id"] == dup_id)
    ctx = rules._Ctx(ep)
    twin = copy.deepcopy(dup)
    twin["invoice_id"], twin["doc_id"] = "AAA-first-by-id", "D-0000000000"
    ctx.first_post["AAA-first-by-id"] = ctx.first_post[dup_id] + timedelta(seconds=5)
    # same issued_at / due_at: the earlier posting wins even though the twin's id sorts first
    assert rules._dup_order_key(ctx, dup) < rules._dup_order_key(ctx, twin)
    twin["due_at"] = (dt(dup["due_at"]) + timedelta(days=1)).isoformat()
    assert rules._dup_order_key(ctx, dup) < rules._dup_order_key(ctx, twin)


def test_a_share_of_duplicates_is_paid_with_unchanged_evidence_and_still_found_by_rules():
    paid = unpaid = partial = 0
    for s in range(1, 61):
        ep = gen_force(s, _single("duplicate_invoice", 2))
        assert validate.validate_episode(ep)["ok"], s
        by_inv = {}
        for p in ep.payments:
            for x in p["invoice_refs"] or []:
                by_inv.setdefault(x["invoice_id"], []).append((p, x))
        for iss in ep.truth["issues"]:
            dup_id = iss["affected_ids"][0]
            dup = next(i for i in ep.invoices if i["invoice_id"] == dup_id)
            ev = iss["evidence_sets"][0]
            assert len(iss["evidence_sets"]) == 1 and len(ev) == 3
            ar = [r for r in ep.ledger if r["invoice_id"] == dup_id and r["account"] == ACC_AR]
            assert len(ar) == 1 and iss["expected_amount_minor"] == ar[0]["debit_minor"]
            if dup_id in by_inv:
                paid += 1
                partial += sum(x["amount_minor"] for _, x in by_inv[dup_id]) < dup["total_minor"]
                for p, _ in by_inv[dup_id]:
                    assert dt(p["timestamp"]) > dt(dup["issued_at"])
            else:
                unpaid += 1
        assert sorted(tuple(v["affected_ids"]) for v in rules.find_violations(ep)) == \
            sorted(tuple(i["affected_ids"]) for i in ep.truth["issues"])
    assert paid >= 15 and unpaid >= 15 and partial >= 3, (paid, unpaid, partial)


# ------------------------------------------------------------------ sampling / realism nits
def test_natural_sampling_rarely_saturates_the_cap_and_interactions_are_a_minority():
    from auditgym.generator.config import _sample_error_counts
    from auditgym.rng import sub_rng
    for split in SPLITS:
        cfg = resolve_config(split)
        n = 1200
        totals, with_inter = [], 0
        for seed in range(n):
            issues, inter = _sample_error_counts(sub_rng(seed, "spec"), cfg)
            totals.append(sum(issues.values()) + 2 * sum(inter.values()))
            with_inter += sum(inter.values()) > 0
            for t in ISSUE_TYPES:
                assert cfg["issues"][t][0] <= issues[t] <= cfg["issues"][t][1]
            for t in INTERACTIONS:
                assert cfg["interactions"][t][0] <= inter[t] <= cfg["interactions"][t][1]
        assert max(totals) <= 12 and min(totals) >= 1
        assert sum(t == 12 for t in totals) / n < 0.03, split           # was ~12 % saturated at the cap
        assert 0.15 < with_inter / n < 0.5, (split, with_inter / n)       # was ~72 %
        assert len(set(totals)) >= 9                                      # still a wide spread of difficulty


def test_forced_split_categories_are_untouched_by_the_new_sampling():
    from auditgym.runner import load_splits, resolve_config as rc
    for e in load_splits()["holdout"]:
        ep = generate_episode(e["seed"], rc(e))
        force = e["force"]
        if force.get("clean"):
            assert ep.truth["issues"] == []
        elif force.get("interactions"):
            assert any(i["scenario"] for i in ep.truth["issues"])


def test_legit_recurring_distractor_uses_weekly_labels_with_seven_day_gaps():
    n = 0
    for s in range(1, 41):
        ep = gen_force(s, {"clean": True, "distractors": {"legit_recurring_amount": 3}})
        inv = {i["invoice_id"]: i for i in ep.invoices}
        for d in ep.provenance["distractors"]:
            if d["kind"] != "legit_recurring_amount":
                continue
            seq = [inv[x] for x in d["record_ids"]]
            labels = [i["line_items"][0]["service_period"] for i in seq]
            assert all(re.fullmatch(r"\d{4}-W\d{2}", l) for l in labels), labels
            assert len(set(labels)) == len(labels)
            for a, b in zip(seq, seq[1:]):
                gap = dt(b["issued_at"]) - dt(a["issued_at"])
                assert timedelta(days=6, hours=19) <= gap <= timedelta(days=7, hours=5), gap
            n += 1
    assert n >= 60


def test_company_name_pool_is_wide():
    names = {generate_episode(s, "train").manifest["company_name"] for s in range(1, 61)}
    assert len(names) >= 50
    assert all(n.startswith("Fictional ") for n in names)
