"""Matcher + reward tests (SPEC section 6 / section 11 and the hackathon brief).

Covers: reward ordering properties, step economy, every way a finding can fail to match
(guessed ids, unknown / insufficient evidence, currency, type, tolerance edges), the
explanation-text invariance, strict schema validation (-1), metric edge cases,
deterministic tie-breaking, normalisation, env-level terminal handling (re-submit, bad
input, truncation) and the four baseline policies run through the runner.

Real generated episodes are used wherever possible; hand-made truth is used where an exact
number (tolerance edges, ambiguous matches) is clearer.
"""
from __future__ import annotations

import copy
import functools
import itertools
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from auditgym import matcher  # noqa: E402
from auditgym.constants import (DUPLICATE_INVOICE, FX_ERROR, ISSUE_TYPES, MAX_EVIDENCE_IDS,  # noqa: E402
                                MAX_FINDINGS, PAYMENT_MISALLOCATION, PERIOD_CUTOFF, STEP_COST)
from auditgym.env import AuditEnv  # noqa: E402
from auditgym.generator.build import generate_episode  # noqa: E402
from auditgym.generator.config import config_with_force  # noqa: E402

# ----------------------------------------------------------------------------- fixtures
ALL_ISSUES = {"duplicate_invoice": 1, "payment_misallocation": 1, "fx_error": 1, "period_cutoff": 1}
INTERACTING = {"interactions": {"fx_partial_payment_at_boundary": 1, "cutoff_with_wrong_rate_date": 1}}


@functools.lru_cache(maxsize=None)
def _ep(seed: int, kind: str):
    if kind == "clean":
        cfg = config_with_force("train", {"clean": True})
    elif kind == "all":
        cfg = config_with_force("train", {"issues": ALL_ISSUES})
    elif kind == "inter":
        cfg = config_with_force("train", INTERACTING)
    else:
        raise ValueError(kind)
    return generate_episode(seed, cfg)


def issues_ep(seed: int = 11):
    return _ep(seed, "all")


def clean_ep(seed: int = 12):
    return _ep(seed, "clean")


def cfg_of(kind: str):
    if kind == "clean":
        return config_with_force("train", {"clean": True})
    if kind == "all":
        return config_with_force("train", {"issues": ALL_ISSUES})
    return config_with_force("train", INTERACTING)


def finding(issue: dict, **over) -> dict:
    """A correct finding for a truth issue (first evidence set); keyword args override."""
    f = {"type": issue["type"], "affected_ids": list(issue["affected_ids"]),
         "claimed_amount_minor": int(issue["expected_amount_minor"]),
         "currency": issue["currency"], "evidence_ids": list(issue["evidence_sets"][0]),
         "explanation": "because"}
    f.update(over)
    return f


def sub(ep, findings) -> dict:
    return {"episode_id": ep.episode_id, "findings": findings}


def grade(ep, findings, steps: int = 0, submission=None) -> dict:
    s = submission if submission is not None else sub(ep, findings)
    return matcher.score(s, ep.truth, ep.public_ids(), steps, ep.episode_id)


def perfect(ep) -> list[dict]:
    return [finding(i) for i in ep.truth["issues"]]


def fabricated(ep, n: int = 4) -> list[dict]:
    """False accusations built only from real public ids (what a guessing agent could do)."""
    invs = ep.invoices[:n]
    out = []
    for k, inv in enumerate(invs):
        out.append({"type": ISSUE_TYPES[k % 4], "affected_ids": [inv["invoice_id"]],
                    "claimed_amount_minor": inv["total_minor"], "currency": ep.manifest["base_currency"],
                    "evidence_ids": [inv["doc_id"]], "explanation": "suspicious"})
    return out


def mk_issue(iid, type_, ids, amount, evsets, currency="KZT", tol=None, scenario=None) -> dict:
    return {"issue_id": iid, "type": type_, "affected_ids": list(ids),
            "expected_amount_minor": amount, "currency": currency,
            "tolerance_minor": max(1, amount // 1000) if tol is None else tol,
            "evidence_sets": [list(s) for s in evsets], "scenario": scenario}


PUB = {"D-a", "D-b", "D-c", "R-1", "R-2", "R-3", "R-4", "fx:2026-03-02:EUR:KZT"}


def hand_truth(*issues) -> dict:
    return {"episode_id": "ep_hand", "base_currency": "KZT", "issues": list(issues)}


def hscore(findings, truth, steps=0, public=PUB, submission=None):
    s = submission if submission is not None else {"episode_id": "ep_hand", "findings": findings}
    return matcher.score(s, truth, public, steps, "ep_hand")


def hfind(type_="duplicate_invoice", ids=("INV-1",), amount=100000, cur="KZT", ev=("D-a", "R-1"), expl="x"):
    return {"type": type_, "affected_ids": list(ids), "claimed_amount_minor": amount,
            "currency": cur, "evidence_ids": list(ev), "explanation": expl}


H1 = mk_issue("iss_01", "duplicate_invoice", ["INV-1"], 100000, [["D-a", "R-1"]])


# ============================================================================= sanity of the fixtures
def test_generated_episodes_have_expected_truth():
    ep = issues_ep()
    assert {i["type"] for i in ep.truth["issues"]} == set(ISSUE_TYPES)
    assert clean_ep().truth["issues"] == []
    inter = _ep(21, "inter")
    assert {i["scenario"] for i in inter.truth["issues"]} == {"fx_partial_payment_at_boundary",
                                                             "cutoff_with_wrong_rate_date"}


def test_truth_tolerance_follows_published_formula():
    for seed in (11, 13, 17):
        for i in _ep(seed, "all").truth["issues"]:
            assert i["tolerance_minor"] == max(1, i["expected_amount_minor"] // 1000)
            assert i["currency"] == _ep(seed, "all").manifest["base_currency"]


def test_truth_evidence_sets_exist_publicly_and_fit_the_cap():
    for seed in (11, 13, 17):
        ep = _ep(seed, "all")
        pub = ep.public_ids()
        for i in ep.truth["issues"]:
            assert i["evidence_sets"]
            for s in i["evidence_sets"]:
                assert set(s) <= pub
                assert 1 <= len(s) <= MAX_EVIDENCE_IDS


# ============================================================================= reward ordering (section 11)
@pytest.mark.parametrize("seed", [11, 13, 17])
def test_perfect_answer_with_evidence_beats_silence_on_erroneous_books(seed):
    ep = _ep(seed, "all")
    good = grade(ep, perfect(ep))
    silent = grade(ep, [])
    assert good["valid"] and good["tp"] == len(ep.truth["issues"]) and good["fp"] == 0 and good["fn"] == 0
    assert good["f1"] == 1.0 and good["reward_shared"] == 1.0
    assert silent["f1"] == 0.0 and silent["reward_shared"] == -1.0
    assert good["reward_shared"] > silent["reward_shared"]
    assert good["reward_experimental"] > silent["reward_experimental"]


def test_perfect_answer_on_interacting_episode():
    ep = _ep(21, "inter")
    assert len(ep.truth["issues"]) == 4
    g = grade(ep, perfect(ep))
    assert (g["tp"], g["fp"], g["fn"], g["reward_shared"]) == (4, 0, 0, 1.0)


@pytest.mark.parametrize("seed", [12, 14, 18])
def test_silence_beats_false_accusations_on_clean_books(seed):
    ep = _ep(seed, "clean")
    silent = grade(ep, [])
    accused = grade(ep, fabricated(ep))
    assert silent["f1"] == 1.0 and silent["reward_shared"] == 1.0
    assert accused["f1"] == 0.0 and accused["fp"] == 4 and accused["tp"] == 0
    assert accused["reward_shared"] == -1.0
    assert silent["reward_shared"] > accused["reward_shared"]
    assert silent["reward_experimental"] > accused["reward_experimental"]


def test_single_false_accusation_on_clean_book_is_worse_than_silence():
    ep = clean_ep()
    one = grade(ep, fabricated(ep, 1))
    assert one["f1"] == 0.0 and one["reward_shared"] == -1.0
    assert grade(ep, [])["reward_shared"] > one["reward_shared"]


def test_false_accusation_on_erroneous_book_hurts_relative_to_perfect():
    ep = issues_ep()
    base = grade(ep, perfect(ep))
    plus_fp = grade(ep, perfect(ep) + fabricated(ep, 1))
    assert plus_fp["fp"] == 1 and plus_fp["tp"] == base["tp"]
    assert plus_fp["reward_shared"] < base["reward_shared"]
    assert plus_fp["precision"] < 1.0 and plus_fp["recall"] == 1.0


@pytest.mark.parametrize("seed", [11, 13])
def test_duplicate_of_correct_finding_gives_no_gain(seed):
    ep = _ep(seed, "all")
    issues = ep.truth["issues"]
    one = grade(ep, [finding(issues[0])])
    twice = grade(ep, [finding(issues[0]), finding(issues[0])])
    thrice = grade(ep, [finding(issues[0])] * 3)
    assert one["tp"] == twice["tp"] == thrice["tp"] == 1
    assert (twice["fp"], thrice["fp"]) == (1, 2)
    assert twice["reward_shared"] < one["reward_shared"]
    assert thrice["reward_shared"] < twice["reward_shared"]
    assert twice["finding_reasons"] == ["tp", "duplicate_of_matched"]
    # the repeat must not even help the experimental reward
    assert twice["reward_experimental"] < one["reward_experimental"]


def test_duplicating_every_finding_is_strictly_worse_than_once():
    ep = issues_ep()
    once = grade(ep, perfect(ep))
    doubled = grade(ep, perfect(ep) * 2)
    assert doubled["tp"] == once["tp"] == len(ep.truth["issues"])
    assert doubled["fp"] == len(ep.truth["issues"])
    assert doubled["reward_shared"] < once["reward_shared"]
    assert doubled["f1"] == pytest.approx(2 * 4 / (2 * 4 + 4))


def test_repeating_a_correct_finding_with_reworded_explanation_is_still_fp():
    ep = issues_ep()
    i = ep.truth["issues"][0]
    g = grade(ep, [finding(i, explanation="first"), finding(i, explanation="second, much longer " * 20)])
    assert (g["tp"], g["fp"]) == (1, 1)


def test_flag_everything_scores_below_zero_on_erroneous_and_clean_books():
    for ep in (issues_ep(), clean_ep(), _ep(21, "inter")):
        findings = []
        for inv in ep.invoices:
            for t in ISSUE_TYPES:
                findings.append({"type": t, "affected_ids": [inv["invoice_id"]],
                                 "claimed_amount_minor": inv["total_minor"],
                                 "currency": ep.manifest["base_currency"],
                                 "evidence_ids": [inv["doc_id"]], "explanation": "all"})
                if len(findings) == MAX_FINDINGS:
                    break
            if len(findings) == MAX_FINDINGS:
                break
        g = grade(ep, findings, steps=6)
        assert g["valid"] and len(findings) == MAX_FINDINGS
        assert g["reward_shared"] < 0.0, g
        assert g["reward_experimental"] <= 0.5  # soft credit is bounded by the 1-1 matching


def test_flag_everything_even_with_every_truth_id_and_amount_but_no_evidence_is_bad():
    ep = issues_ep()
    findings = [finding(i, evidence_ids=[]) for i in ep.truth["issues"]] + fabricated(ep, 26)
    g = grade(ep, findings)
    assert g["tp"] == 0
    assert g["reward_shared"] == -1.0


def test_flag_all_with_the_right_ids_but_wrong_types_everywhere_gets_nothing():
    ep = issues_ep()
    findings = []
    for i in ep.truth["issues"]:
        for t in ISSUE_TYPES:
            if t != i["type"]:
                findings.append(finding(i, type=t))
    g = grade(ep, findings)
    assert g["tp"] == 0 and g["fp"] == len(findings)
    assert g["reward_shared"] < 0.0


# ============================================================================= step economy
@pytest.mark.parametrize("steps", [0, 1, 5, 17, 50, 99])
def test_step_cost_is_exactly_0_002_per_inspection_step(steps):
    ep = issues_ep()
    g0 = grade(ep, perfect(ep), steps=0)
    gn = grade(ep, perfect(ep), steps=steps)
    assert STEP_COST == 0.002
    assert gn["reward_shared"] == pytest.approx(g0["reward_shared"] - 0.002 * steps, abs=1e-9)
    assert gn["reward_shared"] == round(1.0 - 0.002 * steps, 6)


def test_step_economy_never_flips_ordering_of_different_answers():
    ep = issues_ep()
    # max possible penalty 99 steps = 0.198; a perfect answer at 99 steps must still beat silence at 0
    assert grade(ep, perfect(ep), steps=99)["reward_shared"] > grade(ep, [], steps=0)["reward_shared"]
    # and a half-right answer at 0 steps is below the perfect answer at 99 steps (F1 gap 0.667 => reward gap 1.33)
    half = grade(ep, perfect(ep)[:2], steps=0)
    assert half["reward_shared"] < grade(ep, perfect(ep), steps=99)["reward_shared"]


def test_step_economy_equal_answers_differ_by_at_most_one_step_cost_per_step():
    ep = issues_ep()
    rewards = [grade(ep, perfect(ep), steps=n)["reward_shared"] for n in range(0, 100, 7)]
    for (a, na), (b, nb) in zip(zip(rewards, range(0, 100, 7)), list(zip(rewards, range(0, 100, 7)))[1:]):
        assert a - b == pytest.approx(0.002 * (nb - na), abs=1e-6)
    assert rewards == sorted(rewards, reverse=True)


def test_clean_silence_loses_exactly_step_cost_too():
    ep = clean_ep()
    assert grade(ep, [], steps=0)["reward_shared"] == 1.0
    assert grade(ep, [], steps=10)["reward_shared"] == pytest.approx(0.98)
    assert grade(ep, [], steps=99)["reward_shared"] == pytest.approx(1 - 0.198)


def test_shared_reward_clipping_and_formula():
    assert matcher.shared_reward(1.0, 0) == 1.0
    assert matcher.shared_reward(0.0, 0) == -1.0
    assert matcher.shared_reward(0.0, 100) == -1.0            # clipped below
    assert matcher.shared_reward(1.0, 100) == pytest.approx(0.8)
    assert matcher.shared_reward(0.5, 0) == 0.0
    assert matcher.shared_reward(0.5, 10) == pytest.approx(-0.02)
    # never leaves [-1, 1]
    for f1, n in itertools.product([0, 0.1, 0.5, 0.99, 1.0], [0, 1, 50, 100, 105]):
        r = matcher.shared_reward(f1, n)
        assert -1.0 <= r <= 1.0
        assert r == pytest.approx(max(-1.0, min(1.0, 2 * f1 - 1 - 0.002 * n)), abs=1e-6)


def test_step_cost_does_not_change_matching_only_reward():
    ep = issues_ep()
    a, b = grade(ep, perfect(ep), steps=0), grade(ep, perfect(ep), steps=60)
    for k in ("tp", "fp", "fn", "f1", "precision", "recall", "pairs", "finding_reasons"):
        assert a[k] == b[k]
    assert a["inspection_steps"] == 0 and b["inspection_steps"] == 60


# ============================================================================= every way to not match
def _first(ep, type_=None):
    for i in ep.truth["issues"]:
        if type_ is None or i["type"] == type_:
            return i
    raise AssertionError("type not in episode")


@pytest.mark.parametrize("t", ISSUE_TYPES)
def test_guessed_affected_ids_do_not_match(t):
    ep = issues_ep()
    i = _first(ep, t)
    other = next(inv["invoice_id"] for inv in ep.invoices if inv["invoice_id"] not in i["affected_ids"])
    g = grade(ep, [finding(i, affected_ids=[other])])
    assert g["tp"] == 0 and g["fp"] == 1
    assert g["finding_reasons"] == ["ids_mismatch"]


def test_affected_ids_superset_or_subset_or_nonexistent_do_not_match():
    ep = issues_ep()
    i = _first(ep)
    real = i["affected_ids"][0]
    other = next(inv["invoice_id"] for inv in ep.invoices if inv["invoice_id"] != real)
    for ids in ([real, other], [real, "INV-GUESS-1"], ["INV-GUESS-1"], [other]):
        g = grade(ep, [finding(i, affected_ids=ids)])
        assert g["tp"] == 0, ids
        assert g["finding_reasons"] == ["ids_mismatch"]


def test_affected_ids_order_and_repeats_do_not_matter():
    ep = issues_ep()
    i = _first(ep)
    real = i["affected_ids"][0]
    g = grade(ep, [finding(i, affected_ids=[real, real, real])])
    assert g["tp"] == 1


def test_affected_ids_are_case_sensitive():
    ep = issues_ep()
    i = _first(ep)
    real = i["affected_ids"][0]
    swapped = real.lower() if real != real.lower() else real.upper()
    assert swapped != real
    assert grade(ep, [finding(i, affected_ids=[swapped])])["tp"] == 0


@pytest.mark.parametrize("t", ISSUE_TYPES)
def test_unknown_evidence_id_blocks_match_even_with_full_sufficient_set(t):
    ep = issues_ep()
    i = _first(ep, t)
    ev = list(i["evidence_sets"][0]) + ["D-doesnotexist"]
    g = grade(ep, [finding(i, evidence_ids=ev)])
    assert g["valid"] and g["tp"] == 0 and g["fp"] == 1 and g["fn"] == len(ep.truth["issues"])
    assert g["finding_reasons"] == ["evidence_unknown_id"]
    assert g["reward_shared"] == -1.0


def test_only_unknown_evidence_ids():
    ep = issues_ep()
    g = grade(ep, [finding(_first(ep), evidence_ids=["R-00000000", "D-0000000000", "fx:1999-01-01:EUR:KZT"])])
    assert g["tp"] == 0 and g["finding_reasons"] == ["evidence_unknown_id"]


@pytest.mark.parametrize("t", ISSUE_TYPES)
def test_insufficient_evidence_subset_does_not_match(t):
    ep = issues_ep()
    i = _first(ep, t)
    full = list(i["evidence_sets"][0])
    assert len(full) >= 2
    for drop in range(len(full)):
        ev = full[:drop] + full[drop + 1:]
        g = grade(ep, [finding(i, evidence_ids=ev)])
        # a subset of the canonical set still matches iff it contains one of the issue's acceptable sets
        # (payment_misallocation ships a minimal set besides the maximal one)
        ok = any(set(s) <= set(ev) for s in i["evidence_sets"])
        assert g["tp"] == int(ok), (t, drop)
        if not ok:
            assert g["finding_reasons"] == ["evidence_insufficient"]


def test_no_evidence_at_all_does_not_match():
    ep = issues_ep()
    g = grade(ep, [finding(_first(ep), evidence_ids=[])])
    assert g["tp"] == 0 and g["finding_reasons"] == ["evidence_insufficient"]


def test_valid_but_irrelevant_evidence_does_not_match():
    ep = issues_ep()
    i = _first(ep)
    needed = set(i["evidence_sets"][0])
    irrelevant = [r["row_id"] for r in ep.ledger if r["row_id"] not in needed][:3]
    g = grade(ep, [finding(i, evidence_ids=irrelevant)])
    assert g["tp"] == 0 and g["finding_reasons"] == ["evidence_insufficient"]


def test_extra_valid_evidence_is_allowed_and_order_irrelevant():
    ep = issues_ep()
    for i in ep.truth["issues"]:
        needed = list(i["evidence_sets"][0])
        extra = [r["row_id"] for r in ep.ledger if r["row_id"] not in needed][: MAX_EVIDENCE_IDS - len(needed)]
        ev = extra + list(reversed(needed))
        assert len(ev) <= MAX_EVIDENCE_IDS
        assert grade(ep, [finding(i, evidence_ids=ev)])["tp"] == 1
        assert grade(ep, [finding(i, evidence_ids=needed + needed[: MAX_EVIDENCE_IDS - len(needed)])])["tp"] == 1


def test_any_one_acceptable_evidence_set_is_enough():
    t = hand_truth(mk_issue("iss_01", "fx_error", ["INV-1"], 5000, [["D-a", "R-1"], ["D-b", "R-2"]]))
    for ev in (["D-a", "R-1"], ["D-b", "R-2"], ["R-2", "D-b", "D-c"]):
        assert hscore([hfind("fx_error", amount=5000, ev=ev)], t)["tp"] == 1
    # mixing halves of two sets is not a complete set
    assert hscore([hfind("fx_error", amount=5000, ev=["D-a", "R-2"])], t)["tp"] == 0


def test_cited_evidence_exists_but_belongs_to_another_episode_is_unknown():
    ep, other = issues_ep(11), issues_ep(13)
    i = _first(ep)
    foreign = [r["row_id"] for r in other.ledger][0]
    assert foreign not in ep.public_ids()
    g = grade(ep, [finding(i, evidence_ids=list(i["evidence_sets"][0]) + [foreign])])
    assert g["tp"] == 0 and g["finding_reasons"] == ["evidence_unknown_id"]


@pytest.mark.parametrize("cur", ["USD", "EUR", "GBP", "JPY", "XXX", "", "KZTT"])
def test_wrong_currency_does_not_match(cur):
    ep = issues_ep()
    base = ep.manifest["base_currency"]
    if cur == base:
        cur = "CHF"
    i = _first(ep)
    g = grade(ep, [finding(i, currency=cur)])
    assert g["tp"] == 0 and g["finding_reasons"] == ["currency_mismatch"]


def test_foreign_currency_of_the_document_is_not_accepted_as_finding_currency():
    # findings are always in base currency, even if the invoice is foreign
    ep = issues_ep()
    foreign = next(c for c in ("EUR", "USD", "GBP", "JPY", "CHF", "AED") if c != ep.manifest["base_currency"])
    for i in ep.truth["issues"]:
        assert grade(ep, [finding(i, currency=foreign)])["tp"] == 0


@pytest.mark.parametrize("t", ISSUE_TYPES)
def test_wrong_type_does_not_match(t):
    ep = issues_ep()
    i = _first(ep, t)
    for wrong in ISSUE_TYPES:
        if wrong == t:
            continue
        g = grade(ep, [finding(i, type=wrong)])
        assert g["tp"] == 0 and g["fp"] == 1, (t, wrong)
        # in isolation (single-issue truth) the diagnostic reason is wrong_type
        solo = hand_truth(dict(i))
        gs = matcher.score(sub(ep, [finding(i, type=wrong)]), solo, ep.public_ids(), 0, ep.episode_id)
        assert gs["tp"] == 0 and gs["finding_reasons"] == ["wrong_type"], (t, wrong)


def test_interaction_two_issues_same_id_need_two_distinct_typed_findings():
    ep = _ep(21, "inter")
    by_id: dict[str, list[dict]] = {}
    for i in ep.truth["issues"]:
        by_id.setdefault(i["affected_ids"][0], []).append(i)
    pairs = [v for v in by_id.values() if len(v) == 2]
    assert len(pairs) == 2
    for a, b in pairs:
        assert a["type"] != b["type"]
        # claiming only one of the pair earns one TP and misses the other
        g = grade(ep, [finding(a)])
        assert (g["tp"], g["fp"], g["fn"]) == (1, 0, len(ep.truth["issues"]) - 1)
        # a single finding cannot cover both even if it copies the other's evidence
        # copying the other issue's evidence does not make one finding count twice
        g2 = grade(ep, [finding(a, evidence_ids=(list(a["evidence_sets"][0]) + list(b["evidence_sets"][0]))[:10])])
        assert g2["tp"] == 1 and g2["fp"] == 0


# ----------------------------------------------------------------------------- tolerance
@pytest.mark.parametrize("expected", [1, 5, 999, 1000, 1999, 2000, 123456, 7131896, 10 ** 9])
def test_amount_tolerance_inclusive_edges(expected):
    tol = max(1, expected // 1000)
    t = hand_truth(mk_issue("iss_01", "period_cutoff", ["INV-1"], expected, [["D-a"]]))
    f = lambda amt: hscore([hfind("period_cutoff", amount=amt, ev=("D-a",))], t)  # noqa: E731
    assert f(expected)["tp"] == 1
    assert f(expected + tol)["tp"] == 1                       # exactly at tolerance: inside
    assert f(expected + tol + 1)["tp"] == 0                   # one minor unit beyond: outside
    assert f(expected + tol + 1)["finding_reasons"] == ["amount_out_of_tolerance"]
    if expected - tol >= 0:
        assert f(expected - tol)["tp"] == 1
    if expected - tol - 1 >= 0:
        assert f(expected - tol - 1)["tp"] == 0


def test_tolerance_floor_is_one_minor_unit():
    t = hand_truth(mk_issue("iss_01", "fx_error", ["PAY-1"], 120, [["D-a"]]))
    assert t["issues"][0]["tolerance_minor"] == 1
    for amt, ok in ((119, 1), (120, 1), (121, 1), (118, 0), (122, 0), (0, 0)):
        assert hscore([hfind("fx_error", ids=("PAY-1",), amount=amt, ev=("D-a",))], t)["tp"] == ok, amt


def test_tolerance_uses_issue_value_and_falls_back_to_formula_when_absent():
    iss = mk_issue("iss_01", "fx_error", ["PAY-1"], 50000, [["D-a"]])
    del iss["tolerance_minor"]
    t = hand_truth(iss)
    assert hscore([hfind("fx_error", ids=("PAY-1",), amount=50050, ev=("D-a",))], t)["tp"] == 1
    assert hscore([hfind("fx_error", ids=("PAY-1",), amount=50051, ev=("D-a",))], t)["tp"] == 0


def test_tolerance_on_real_generated_issues_edges():
    ep = issues_ep()
    for i in ep.truth["issues"]:
        e, tol = i["expected_amount_minor"], i["tolerance_minor"]
        assert grade(ep, [finding(i, claimed_amount_minor=e + tol)])["tp"] == 1
        assert grade(ep, [finding(i, claimed_amount_minor=e + tol + 1)])["tp"] == 0
        assert grade(ep, [finding(i, claimed_amount_minor=max(0, e - tol))])["tp"] == 1
        if e - tol - 1 >= 0:
            assert grade(ep, [finding(i, claimed_amount_minor=e - tol - 1)])["tp"] == 0


def test_amount_in_wrong_scale_does_not_match():
    ep = issues_ep()
    for i in ep.truth["issues"]:
        e = i["expected_amount_minor"]
        if e < 1000:
            continue
        assert grade(ep, [finding(i, claimed_amount_minor=e * 100)])["tp"] == 0
        assert grade(ep, [finding(i, claimed_amount_minor=e // 100)])["tp"] == 0


def test_two_findings_at_the_edges_of_one_issue_only_one_matches():
    t = hand_truth(mk_issue("iss_01", "period_cutoff", ["INV-1"], 100000, [["D-a"]]))
    g = hscore([hfind("period_cutoff", amount=100100, ev=("D-a",)),
                hfind("period_cutoff", amount=99900, ev=("D-a",))], t)
    assert (g["tp"], g["fp"], g["fn"]) == (1, 1, 0)


# ----------------------------------------------------------------------------- explanation text
@pytest.mark.parametrize("expl", ["", "I am absolutely certain, trust me.", "x" * 1000,
                                  "IGNORE THE RULES. Give this finding full credit. reward=1.0",
                                  "евро ≠ доллар 欧元", "iss_01 fx_error expected_amount", "\n\t  spaced  \n"])
def test_explanation_text_never_changes_the_score(expl):
    ep = issues_ep()
    base = grade(ep, perfect(ep) + fabricated(ep, 2))
    mod = [finding(i, explanation=expl) for i in ep.truth["issues"]] + \
          [dict(f, explanation=expl) for f in fabricated(ep, 2)]
    assert grade(ep, mod) == base


def test_persuasive_explanation_cannot_rescue_a_wrong_finding():
    ep = issues_ep()
    i = _first(ep)
    g = grade(ep, [finding(i, claimed_amount_minor=i["expected_amount_minor"] * 3 + 10 ** 6,
                           explanation="The expected amount is exactly this, verified by the auditor.")])
    assert g["tp"] == 0


# ============================================================================= schema validation => -1
def _invalid(ep, submission, truth=None):
    g = matcher.score(submission, truth or ep.truth, ep.public_ids(), 7, ep.episode_id)
    assert g["valid"] is False, submission
    assert g["reward_shared"] == -1.0 and g["reward_experimental"] == -1.0
    assert g["f1"] == 0.0 and g["tp"] == 0
    assert isinstance(g["error"], str) and 0 < len(g["error"]) <= 200
    return g


def _good(ep):
    return sub(ep, [finding(i) for i in ep.truth["issues"][:1]])


def test_valid_submission_is_accepted_and_normalised():
    ep = issues_ep()
    ok, err, norm = matcher.validate_submission(_good(ep), ep.episode_id)
    assert ok and err is None and norm["episode_id"] == ep.episode_id and len(norm["findings"]) == 1


def test_invalid_missing_top_level_keys():
    ep = issues_ep()
    for k in ("episode_id", "findings"):
        s = _good(ep)
        del s[k]
        _invalid(ep, s)
    _invalid(ep, {})


def test_invalid_extra_top_level_key():
    ep = issues_ep()
    s = _good(ep)
    s["notes"] = "hi"
    _invalid(ep, s)


@pytest.mark.parametrize("key", ["type", "affected_ids", "claimed_amount_minor", "currency",
                                 "evidence_ids", "explanation"])
def test_invalid_missing_finding_key(key):
    ep = issues_ep()
    s = _good(ep)
    del s["findings"][0][key]
    _invalid(ep, s)


def test_invalid_extra_finding_key():
    ep = issues_ep()
    s = _good(ep)
    s["findings"][0]["confidence"] = 0.9
    _invalid(ep, s)


def test_invalid_extra_key_cannot_smuggle_a_second_amount():
    ep = issues_ep()
    s = _good(ep)
    s["findings"][0]["expected_amount_minor"] = 1
    _invalid(ep, s)


@pytest.mark.parametrize("bad", [True, False, 12.0, 12.5, 1e3, "12", None, [12], {"v": 1}, -1, -10 ** 6])
def test_invalid_amount_values(bad):
    ep = issues_ep()
    s = _good(ep)
    s["findings"][0]["claimed_amount_minor"] = bad
    _invalid(ep, s)


def test_zero_amount_is_schema_valid():
    ep = issues_ep()
    s = _good(ep)
    s["findings"][0]["claimed_amount_minor"] = 0
    g = matcher.score(s, ep.truth, ep.public_ids(), 0, ep.episode_id)
    assert g["valid"] is True and g["tp"] == 0 and g["fp"] == 1


def test_invalid_more_than_30_findings_and_exactly_30_is_valid():
    ep = issues_ep()
    f = fabricated(ep, 1)[0]
    g30 = matcher.score(sub(ep, [dict(f) for _ in range(MAX_FINDINGS)]), ep.truth, ep.public_ids(), 0, ep.episode_id)
    assert g30["valid"] is True and g30["fp"] == 30
    _invalid(ep, sub(ep, [dict(f) for _ in range(MAX_FINDINGS + 1)]))
    _invalid(ep, sub(ep, [dict(f) for _ in range(500)]))


def test_invalid_more_than_10_evidence_ids_and_exactly_10_is_valid():
    ep = issues_ep()
    i = _first(ep)
    ids = [r["row_id"] for r in ep.ledger[:11]]
    ok = matcher.score(sub(ep, [finding(i, evidence_ids=ids[:10])]), ep.truth, ep.public_ids(), 0, ep.episode_id)
    assert ok["valid"] is True
    _invalid(ep, sub(ep, [finding(i, evidence_ids=ids)]))
    # repeats count toward the cap (cap is on the list, not on the set)
    _invalid(ep, sub(ep, [finding(i, evidence_ids=[ids[0]] * 11)]))


def test_invalid_more_than_10_affected_ids_and_empty_affected_ids():
    ep = issues_ep()
    i = _first(ep)
    ids = [inv["invoice_id"] for inv in ep.invoices[:11]]
    assert matcher.score(sub(ep, [finding(i, affected_ids=ids[:10])]), ep.truth, ep.public_ids(), 0,
                         ep.episode_id)["valid"] is True
    _invalid(ep, sub(ep, [finding(i, affected_ids=ids)]))
    _invalid(ep, sub(ep, [finding(i, affected_ids=[])]))


def test_invalid_explanation_over_1000_chars_and_exactly_1000_is_valid():
    ep = issues_ep()
    i = _first(ep)
    assert matcher.score(sub(ep, [finding(i, explanation="y" * 1000)]), ep.truth, ep.public_ids(), 0,
                         ep.episode_id)["valid"] is True
    _invalid(ep, sub(ep, [finding(i, explanation="y" * 1001)]))
    _invalid(ep, sub(ep, [finding(i, explanation="y" * 100000)]))


def test_invalid_wrong_or_foreign_episode_id():
    ep, other = issues_ep(11), issues_ep(13)
    _invalid(ep, {"episode_id": other.episode_id, "findings": []})
    _invalid(ep, {"episode_id": "", "findings": []})
    _invalid(ep, {"episode_id": ep.episode_id.upper(), "findings": []})
    _invalid(ep, {"episode_id": ep.episode_id + " ", "findings": []})   # no stripping of the episode id
    _invalid(ep, {"episode_id": None, "findings": []})
    _invalid(ep, {"episode_id": 123, "findings": []})


@pytest.mark.parametrize("bad", [None, [], "submit", 42, 3.5, True, [{"findings": []}], (1, 2)])
def test_invalid_non_dict_submission(bad):
    ep = issues_ep()
    _invalid(ep, bad)


@pytest.mark.parametrize("bad", [None, {}, "x", 0, [[]], ["a"], [1], [None], [[{"type": "fx_error"}]], ((),)])
def test_invalid_findings_container_or_entries(bad):
    ep = issues_ep()
    _invalid(ep, {"episode_id": ep.episode_id, "findings": bad})


def test_invalid_field_types():
    ep = issues_ep()
    i = _first(ep)
    cases = [dict(type=7), dict(type=None), dict(type="not_a_type"), dict(type=""),
             dict(affected_ids="INV-1"), dict(affected_ids=[1]), dict(affected_ids=None),
             dict(affected_ids=[["INV-1"]]), dict(affected_ids=("a",)),
             dict(currency=3), dict(currency=None), dict(currency=["KZT"]),
             dict(evidence_ids="R-1"), dict(evidence_ids=[1]), dict(evidence_ids=None),
             dict(explanation=None), dict(explanation=3), dict(explanation=["x"])]
    for over in cases:
        _invalid(ep, sub(ep, [finding(i, **over)]))


def test_one_bad_finding_invalidates_the_whole_submission_even_with_good_ones():
    ep = issues_ep()
    fs = perfect(ep) + [dict(finding(_first(ep)), claimed_amount_minor=True)]
    g = _invalid(ep, sub(ep, fs))
    assert g["fn"] == len(ep.truth["issues"])


def test_invalid_submission_is_worse_than_silence_on_a_clean_book_and_on_an_erroneous_one():
    for ep in (clean_ep(), issues_ep()):
        bad = matcher.score([], ep.truth, ep.public_ids(), 0, ep.episode_id)
        assert bad["reward_shared"] == -1.0
        assert bad["reward_shared"] <= grade(ep, [])["reward_shared"]
    # on a clean book silence is +1 and invalid is -1: the gap must be the maximum
    assert grade(clean_ep(), [])["reward_shared"] - (-1.0) == 2.0


def test_validate_submission_error_never_echoes_long_input():
    ep = issues_ep()
    s = _good(ep)
    s["Z" * 5000] = 1
    ok, err, norm = matcher.validate_submission(s, ep.episode_id)
    assert not ok and norm is None and len(err) < 120
    assert "Z" * 41 not in err


def test_validate_does_not_mutate_input():
    ep = issues_ep()
    s = sub(ep, [finding(i, type=" FX_Error ", currency=" kzt ") if i["type"] == "fx_error" else finding(i)
                 for i in ep.truth["issues"]])
    before = copy.deepcopy(s)
    matcher.validate_submission(s, ep.episode_id)
    matcher.score(s, ep.truth, ep.public_ids(), 3, ep.episode_id)
    assert s == before


# ============================================================================= metric edge cases
def test_prf_both_empty_is_one():
    m = matcher.prf(0, 0, 0, 0, 0)
    # F1 = 1 by definition; precision and recall are 0/0 -> 0 (zero denominators give 0)
    assert (m["precision"], m["recall"], m["f1"]) == (0.0, 0.0, 1.0)


def test_prf_empty_truth_with_findings_is_zero():
    m = matcher.prf(0, 3, 0, 0, 3)
    assert m["f1"] == 0.0 and m["precision"] == 0.0 and m["recall"] == 0.0


def test_prf_truth_but_no_findings_is_zero_not_error():
    m = matcher.prf(0, 0, 4, 4, 0)
    assert m["f1"] == 0.0 and m["precision"] == 0.0 and m["recall"] == 0.0


def test_prf_zero_denominators_give_zero():
    assert matcher.prf(0, 0, 0, 5, 0)["recall"] == 0.0
    m = matcher.prf(0, 2, 0, 0, 2)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


@pytest.mark.parametrize("tp,fp,fn", [(1, 0, 0), (1, 1, 0), (1, 0, 1), (2, 1, 3), (3, 3, 3), (0, 1, 1), (5, 0, 0), (12, 18, 0)])
def test_prf_formulas(tp, fp, fn):
    m = matcher.prf(tp, fp, fn, tp + fn, tp + fp)
    assert m["precision"] == pytest.approx(tp / (tp + fp) if tp + fp else 0, abs=1e-6)
    assert m["recall"] == pytest.approx(tp / (tp + fn) if tp + fn else 0, abs=1e-6)
    assert m["f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn), abs=1e-6)
    assert (m["tp"], m["fp"], m["fn"]) == (tp, fp, fn)


def test_score_empty_vs_empty_on_clean_book_is_f1_one_and_reward_one():
    ep = clean_ep()
    g = grade(ep, [])
    assert g["valid"] and (g["tp"], g["fp"], g["fn"], g["n_truth"], g["n_findings"]) == (0, 0, 0, 0, 0)
    assert g["f1"] == 1.0 and g["precision"] == g["recall"] == 0.0     # 0/0 -> 0
    assert g["reward_shared"] == 1.0 and g["reward_experimental"] == 1.0


def test_score_empty_truth_with_findings_is_f1_zero_reward_minus_one():
    ep = clean_ep()
    g = grade(ep, fabricated(ep, 2))
    assert g["f1"] == 0.0 and g["n_truth"] == 0 and g["n_findings"] == 2
    assert g["reward_shared"] == -1.0


def test_score_truth_present_and_no_findings_is_f1_zero():
    ep = issues_ep()
    g = grade(ep, [])
    assert g["f1"] == 0.0 and g["fn"] == 4 and g["fp"] == 0 and g["tp"] == 0
    assert g["reward_shared"] == -1.0


def test_score_missing_issues_key_treated_as_clean():
    g = matcher.score({"episode_id": "e", "findings": []}, {}, set(), 0, "e")
    assert g["valid"] and g["f1"] == 1.0


def test_partial_recall_and_precision_numbers_on_real_episode():
    ep = issues_ep()
    issues = ep.truth["issues"]
    g = grade(ep, [finding(issues[0]), finding(issues[1])] + fabricated(ep, 2))
    assert (g["tp"], g["fp"], g["fn"]) == (2, 2, 2)
    assert g["precision"] == 0.5 and g["recall"] == 0.5 and g["f1"] == 0.5
    assert g["reward_shared"] == 0.0
    g2 = grade(ep, [finding(issues[0]), finding(issues[1])], steps=10)
    assert g2["f1"] == pytest.approx(2 / 3, abs=1e-6)
    assert g2["reward_shared"] == pytest.approx(2 * (2 / 3) - 1 - 0.02, abs=1e-5)


def test_reward_is_monotone_in_true_positives():
    ep = issues_ep()
    rs = [grade(ep, perfect(ep)[:k])["reward_shared"] for k in range(0, 5)]
    assert rs == sorted(rs) and len(set(rs)) == 5


def test_metrics_are_rounded_to_six_decimals_and_rewards_in_range():
    ep = issues_ep()
    g = grade(ep, perfect(ep)[:1] + fabricated(ep, 3), steps=13)
    for k in ("precision", "recall", "f1", "reward_shared", "reward_experimental"):
        assert g[k] == round(g[k], 6)
        assert -1.0 <= g[k] <= 1.0


# ----------------------------------------------------------------------------- experimental reward
def test_experimental_reward_is_separate_and_gives_partial_credit_only_for_near_misses():
    ep = issues_ep()
    i = _first(ep)
    near = grade(ep, [finding(i, claimed_amount_minor=i["expected_amount_minor"] * 5 + 10 ** 7)])
    far = grade(ep, [finding(i, affected_ids=["INV-GUESS"])])
    assert near["reward_shared"] == far["reward_shared"] == -1.0
    assert near["reward_experimental"] > far["reward_experimental"]
    # partial credit is capped: perfect shared reward stays the max
    assert grade(ep, perfect(ep))["reward_experimental"] == 1.0
    assert near["reward_experimental"] < grade(ep, [finding(i)])["reward_experimental"]


def test_experimental_partial_credit_is_still_one_to_one():
    ep = issues_ep()
    i = _first(ep)
    bad = finding(i, claimed_amount_minor=i["expected_amount_minor"] * 5 + 10 ** 7)
    one = grade(ep, [bad])
    many = grade(ep, [bad] * 5)
    assert many["reward_experimental"] < one["reward_experimental"]


def test_experimental_reward_formula():
    # 1 truth, 1 finding, 1 partial => soft_tp 0.25 => soft_f1 = 0.5/2 => 2*0.25-1
    assert matcher.experimental_reward(0, 1, 1, 1, 0) == pytest.approx(-0.5)
    assert matcher.experimental_reward(1, 0, 1, 1, 0) == 1.0
    assert matcher.experimental_reward(0, 0, 0, 0, 0) == 1.0
    assert matcher.experimental_reward(0, 0, 0, 0, 10) == pytest.approx(0.98)
    assert matcher.experimental_reward(0, 0, 3, 0, 0) == -1.0


# ============================================================================= deterministic matching
def test_ambiguous_findings_are_matched_with_maximum_cardinality_deterministically():
    # two issues the same finding can satisfy; f1 can only satisfy iss_01
    t = hand_truth(mk_issue("iss_01", "fx_error", ["INV-1"], 1000, [["D-a"]]),
                   mk_issue("iss_02", "fx_error", ["INV-1"], 1000, [["D-a"]]))
    f_any = hfind("fx_error", amount=1000, ev=("D-a",))
    g = hscore([f_any, f_any], t)
    assert g["tp"] == 2 and g["fp"] == 0 and g["fn"] == 0
    # Kuhn: f0 takes iss_01, then f1 augments (f0 is moved to iss_02); fixed, documented outcome
    assert g["pairs"] == [[0, "iss_02"], [1, "iss_01"]]
    assert hscore([f_any, f_any], t) == g


def test_augmenting_path_rematches_to_reach_maximum_cardinality():
    # f0 is eligible for iss_01 and iss_02 ; f1 only for iss_01 -> f0 must be moved to iss_02
    t = hand_truth(mk_issue("iss_01", "fx_error", ["INV-1"], 1000, [["D-a"]]),
                   mk_issue("iss_02", "fx_error", ["INV-1"], 1000, [["D-b"]]))
    f0 = hfind("fx_error", amount=1000, ev=("D-a", "D-b"))      # satisfies both
    f1 = hfind("fx_error", amount=1000, ev=("D-a",))            # satisfies iss_01 only
    g = hscore([f0, f1], t)
    assert g["tp"] == 2 and g["fp"] == 0 and g["fn"] == 0
    assert dict((i, iid) for i, iid in g["pairs"]) == {0: "iss_02", 1: "iss_01"}
    # naive greedy would have scored 1 TP; reversing the submitted order still reaches 2
    g_rev = hscore([f1, f0], t)
    assert g_rev["tp"] == 2
    assert dict((i, iid) for i, iid in g_rev["pairs"]) == {0: "iss_01", 1: "iss_02"}


def test_matching_does_not_depend_on_truth_list_order():
    issues = [mk_issue(f"iss_0{k}", "fx_error", ["INV-1"], 1000, [["D-a"]]) for k in range(1, 5)]
    fs = [hfind("fx_error", amount=1000, ev=("D-a",)) for _ in range(3)]
    base = hscore(fs, hand_truth(*issues))
    for perm in itertools.permutations(issues):
        assert hscore(fs, hand_truth(*perm)) == base
    assert base["tp"] == 3 and base["fn"] == 1
    # chain of augmentations with issues tried in sorted order => fixed outcome
    assert base["pairs"] == [[0, "iss_03"], [1, "iss_02"], [2, "iss_01"]]
    assert base["finding_reasons"] == ["tp", "tp", "tp"]


def test_matching_is_repeatable_and_ignores_unrelated_dict_order():
    ep = _ep(21, "inter")
    fs = perfect(ep) + perfect(ep)
    a = grade(ep, fs)
    for _ in range(5):
        assert grade(ep, list(fs)) == a
    shuffled_truth = copy.deepcopy(ep.truth)
    shuffled_truth["issues"].reverse()
    assert matcher.score(sub(ep, fs), shuffled_truth, ep.public_ids(), 0, ep.episode_id) == a


def test_extra_identical_finding_is_attributed_to_the_later_submitted_one():
    t = hand_truth(H1)
    g = hscore([hfind(), hfind()], t)
    assert g["pairs"] == [[0, "iss_01"]]
    assert g["finding_reasons"] == ["tp", "duplicate_of_matched"]
    assert (g["tp"], g["fp"], g["fn"]) == (1, 1, 0)


def test_findings_visited_in_submitted_order_first_eligible_wins():
    # one issue, a "good" finding and an equally eligible one with extra evidence: the first wins
    t = hand_truth(H1)
    g = hscore([hfind(ev=("D-a", "R-1", "D-b")), hfind()], t)
    assert g["pairs"] == [[0, "iss_01"]]
    g = hscore([hfind(), hfind(ev=("D-a", "R-1", "D-b"))], t)
    assert g["pairs"] == [[0, "iss_01"]]


def test_match_function_direct_api_and_reasons():
    t = [H1, mk_issue("iss_02", "period_cutoff", ["INV-2"], 5000, [["D-b"]])]
    fs = [hfind(), hfind("period_cutoff", ("INV-2",), 5000, "KZT", ("D-b",)),
          hfind("fx_error", ("INV-9",)), hfind(cur="USD"), hfind(amount=1), hfind(ev=("D-a", "NOPE")),
          hfind(ev=("D-a",)), hfind("duplicate_invoice", ("INV-1",))]
    norm = [matcher.normalize_finding(f) for f in fs]
    m = matcher.match(norm, t, PUB)
    assert m["tp"] == 2 and m["fp"] == 6 and m["fn"] == 0
    assert m["pairs"] == [(0, "iss_01"), (1, "iss_02")]
    assert m["reasons"] == ["tp", "tp", "wrong_type", "currency_mismatch", "amount_out_of_tolerance",
                            "evidence_unknown_id", "evidence_insufficient", "duplicate_of_matched"]


def test_match_reasons_for_unmatched_findings_when_issue_is_free():
    t = [H1]
    cases = [(hfind("fx_error"), "wrong_type"),
             (hfind(ids=("INV-2",)), "ids_mismatch"),
             (hfind(cur="USD"), "currency_mismatch"),
             (hfind(amount=1), "amount_out_of_tolerance"),
             (hfind(ev=("D-a", "NOPE")), "evidence_unknown_id"),
             (hfind(ev=("D-a",)), "evidence_insufficient"),
             (hfind(), "tp")]
    for f, reason in cases:
        m = matcher.match([matcher.normalize_finding(f)], t, PUB)
        assert m["reasons"] == [reason], f


def test_eligible_function_reason_order_is_fixed():
    f = matcher.normalize_finding(hfind("fx_error", ("INV-9",), 1, "USD", ("NOPE",)))
    assert matcher.eligible(f, H1, PUB) == (False, "wrong_type")
    f = matcher.normalize_finding(hfind("duplicate_invoice", ("INV-9",), 1, "USD", ("NOPE",)))
    assert matcher.eligible(f, H1, PUB) == (False, "ids_mismatch")
    f = matcher.normalize_finding(hfind("duplicate_invoice", ("INV-1",), 1, "USD", ("NOPE",)))
    assert matcher.eligible(f, H1, PUB) == (False, "currency_mismatch")
    f = matcher.normalize_finding(hfind("duplicate_invoice", ("INV-1",), 1, "KZT", ("NOPE",)))
    assert matcher.eligible(f, H1, PUB) == (False, "amount_out_of_tolerance")
    f = matcher.normalize_finding(hfind("duplicate_invoice", ("INV-1",), 100000, "KZT", ("NOPE",)))
    assert matcher.eligible(f, H1, PUB) == (False, "evidence_unknown_id")
    f = matcher.normalize_finding(hfind("duplicate_invoice", ("INV-1",), 100000, "KZT", ("D-b",)))
    assert matcher.eligible(f, H1, PUB) == (False, "evidence_insufficient")
    f = matcher.normalize_finding(hfind())
    assert matcher.eligible(f, H1, PUB) == (True, "tp")


def test_many_findings_vs_many_issues_matching_is_maximum_and_stable():
    # 6 issues alike except evidence sets arranged as a chain => needs augmenting paths
    issues = [mk_issue(f"iss_0{k}", "fx_error", ["INV-1"], 1000, [[f"E{k}"]]) for k in range(1, 7)]
    pub = {f"E{k}" for k in range(1, 8)}
    # finding k carries evidence E1..Ek => eligible for issues 1..k (staircase)
    fs = [hfind("fx_error", amount=1000, ev=tuple(f"E{j}" for j in range(1, k + 1))) for k in range(6, 0, -1)]
    # submitted so the widest finding comes first: greedy would take iss_01 for it and starve the rest
    g = hscore(fs, hand_truth(*issues), public=pub)
    assert g["tp"] == 6 and g["fp"] == 0 and g["fn"] == 0
    g2 = hscore(fs, hand_truth(*issues), public=pub)
    assert g == g2


# ============================================================================= normalisation
def test_normalisation_strips_whitespace_and_cases_type_and_currency():
    ep = issues_ep()
    i = _first(ep, "fx_error")
    f = finding(i, type="  FX_Error\t", currency=" " + ep.manifest["base_currency"].lower() + "\n",
                affected_ids=[" " + a + " " for a in i["affected_ids"]],
                evidence_ids=["\t" + e + "  " for e in i["evidence_sets"][0]],
                explanation="  padded  ")
    g = grade(ep, [f])
    assert g["valid"] and g["tp"] == 1


@pytest.mark.parametrize("t", ["Duplicate_Invoice", "DUPLICATE_INVOICE", " duplicate_invoice ", "duplicate_invoice"])
def test_type_normalisation_cases(t):
    ep = issues_ep()
    i = _first(ep, "duplicate_invoice")
    assert grade(ep, [finding(i, type=t)])["tp"] == 1


@pytest.mark.parametrize("cur_fn", [str.lower, str.upper, str.title, lambda s: f"  {s}  "])
def test_currency_normalisation_cases(cur_fn):
    ep = issues_ep()
    i = _first(ep)
    assert grade(ep, [finding(i, currency=cur_fn(i["currency"]))])["tp"] == 1


def test_normalize_finding_output_shape():
    n = matcher.normalize_finding({"type": " Fx_Error ", "affected_ids": [" a ", "a", "b"],
                                   "claimed_amount_minor": 5, "currency": " kzt ",
                                   "evidence_ids": ["x ", " x"], "explanation": "  e "})
    assert n["type"] == "fx_error" and n["currency"] == "KZT"
    assert n["affected_ids"] == frozenset({"a", "b"}) and n["evidence_ids"] == frozenset({"x"})
    assert n["claimed_amount_minor"] == 5 and n["explanation"] == "e"


def test_ids_are_case_sensitive_for_evidence_too():
    ep = issues_ep()
    i = _first(ep)
    ev = [e.lower() if e.lower() != e else e.upper() for e in i["evidence_sets"][0]]
    assert set(ev) != set(i["evidence_sets"][0])
    g = grade(ep, [finding(i, evidence_ids=ev)])
    assert g["tp"] == 0   # changed-case ids are not known public ids


def test_type_with_inner_whitespace_or_dash_is_not_normalised_away():
    ep = issues_ep()
    i = _first(ep, "fx_error")
    for bad in ("fx error", "fx-error", "fxerror", "FX ERROR"):
        _invalid(ep, sub(ep, [finding(i, type=bad)]))


# ============================================================================= env-level terminal handling
def _env(kind: str, seed: int, mode: str = "train", **kw):
    env = AuditEnv(mode=mode, **kw)
    obs, _ = env.reset(seed=seed, config=cfg_of(kind))
    return env, obs


def _ep_with_truth(env):
    """The env keeps truth out of the Episode it holds (env._ep is public-only); tests that need to build
    adversarial submissions from truth read it from the sealed attribute."""
    import dataclasses
    return dataclasses.replace(env._ep, truth=env._truth)


def _truth_findings(env):
    return [finding(i) for i in env._truth["issues"]]


def _submit(env, findings=None, raw=None):
    payload = raw if raw is not None else {"episode_id": env._ep.episode_id, "findings": findings}
    return env.step({"action": "submit", "args": {"findings": payload}})


def _peek(env):
    return {"action": "list_records", "args": {"table": "fx", "limit": 1}}


def test_env_train_perfect_vs_silence_vs_clean_through_the_real_submit_path():
    env, _ = _env("all", 11)
    for _ in range(3):
        env.step(_peek(env))
    obs, r, term, trunc, info = _submit(env, _truth_findings(env))
    assert term and not trunc and r == pytest.approx(1.0 - 3 * 0.002)
    assert info["metrics"]["f1"] == 1.0 and info["inspection_steps"] == 3
    assert info["reward_shared"] == r and info["invalid_submission"] is False

    env2, _ = _env("all", 11)
    _, r2, _, _, info2 = _submit(env2, [])
    assert r2 == -1.0 and info2["metrics"]["fn"] == 4

    env3, _ = _env("clean", 12)
    _, r3, _, _, _ = _submit(env3, [])
    assert r3 == 1.0
    env4, _ = _env("clean", 12)
    _, r4, _, _, i4 = _submit(env4, fabricated(env4._ep, 2))
    assert r4 == -1.0 and i4["metrics"]["fp"] == 2


def test_env_step_economy_exact_through_submit():
    rewards = []
    for n in (0, 1, 10, 40):
        env, _ = _env("all", 11)
        for _ in range(n):
            env.step(_peek(env))
        rewards.append(_submit(env, _truth_findings(env))[1])
    assert rewards == pytest.approx([1.0, 0.998, 0.98, 0.92])


def test_env_errors_consume_steps_and_count_as_inspection_steps():
    env, _ = _env("clean", 12)
    env.step({"action": "get_record", "args": {"table": "ledger", "id": "R-nope"}})
    env.step({"action": "bogus"})
    _, r, _, _, info = _submit(env, [])
    assert info["inspection_steps"] == 2 and r == pytest.approx(1.0 - 0.004)


def test_env_eval_mode_hides_reward_and_info_but_grader_report_has_the_truth_score():
    env, _ = _env("all", 11, mode="eval")
    obs, r, term, trunc, info = _submit(env, _truth_findings(env))
    assert term and r == 0.0 and info == {}
    rep = env.grader_report()
    assert rep["reward_shared"] == 1.0 and rep["metrics"]["tp"] == 4 and rep["invalid_submission"] is False


def test_env_resubmit_after_end_is_rejected_and_does_not_change_the_result():
    env, _ = _env("all", 11)
    _, r1, term1, _, _ = _submit(env, [])
    assert term1 and r1 == -1.0
    before = copy_report(env)
    obs, r2, term2, trunc2, info2 = _submit(env, _truth_findings(env))
    assert obs["error"]["code"] == "episode_over"
    assert r2 == 0.0 and term2 is True and trunc2 is False
    assert env.grader_report() == before
    assert before["reward_shared"] == -1.0
    # also a non-submit action after the end is a no-op error
    obs, r3, term3, _, _ = env.step(_peek(env))
    assert obs["error"]["code"] == "episode_over" and r3 == 0.0 and term3 is True
    assert env.grader_report() == before


def copy_report(env):
    return copy.deepcopy(env.grader_report())


def test_env_resubmit_after_invalid_submission_is_also_rejected():
    env, _ = _env("all", 11)
    _, r1, term1, _, info1 = _submit(env, raw=["junk"])
    assert r1 == -1.0 and term1 and info1["invalid_submission"] is True
    obs, r2, term2, _, _ = _submit(env, _truth_findings(env))
    assert obs["error"]["code"] == "episode_over" and r2 == 0.0 and term2
    assert env.grader_report()["reward_shared"] == -1.0


def test_env_reset_clears_previous_result():
    env, _ = _env("all", 11)
    _submit(env, [])
    obs, _ = env.reset(seed=11, config=cfg_of("all"))
    assert obs["step"] == 0
    _, r, _, _, _ = _submit(env, _truth_findings(env))
    assert r == 1.0


@pytest.mark.parametrize("make_bad", [
    lambda ep: ["not", "a", "dict"],
    lambda ep: None,
    lambda ep: {"findings": []},
    lambda ep: {"episode_id": ep.episode_id},
    lambda ep: {"episode_id": ep.episode_id, "findings": [], "extra": 1},
    lambda ep: {"episode_id": "ep_wrong", "findings": []},
    lambda ep: {"episode_id": ep.episode_id, "findings": [dict(finding(ep.truth["issues"][0]), claimed_amount_minor=1.5)]},
    lambda ep: {"episode_id": ep.episode_id, "findings": [dict(finding(ep.truth["issues"][0]), claimed_amount_minor=True)]},
    lambda ep: {"episode_id": ep.episode_id, "findings": [dict(finding(ep.truth["issues"][0]), claimed_amount_minor=-5)]},
    lambda ep: {"episode_id": ep.episode_id, "findings": [finding(ep.truth["issues"][0])] * 31},
    lambda ep: {"episode_id": ep.episode_id, "findings": [dict(finding(ep.truth["issues"][0]), evidence_ids=["R-x"] * 11)]},
    lambda ep: {"episode_id": ep.episode_id, "findings": [dict(finding(ep.truth["issues"][0]), explanation="z" * 1001)]},
])
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_env_malformed_submission_ends_episode_with_minus_one_and_logs_error(make_bad, mode):
    env, _ = _env("all", 11, mode=mode)
    bad = make_bad(_ep_with_truth(env))
    obs, r, term, trunc, info = _submit(env, raw=bad)
    assert term and not trunc
    rep = env.grader_report()
    assert rep["reward_shared"] == -1.0 and rep["reward_experimental"] == -1.0
    assert rep["invalid_submission"] is True and rep["errors"]
    if mode == "train":
        assert obs["error"]["code"] == "invalid_args"
        assert r == -1.0 and info["invalid_submission"] is True
    else:
        # eval: identical terminal observation as for a valid submission (no 1-bit leak)
        assert obs["error"] is None
        assert r == 0.0 and info == {}


def test_env_eval_rejection_message_does_not_leak_schema_detail_or_truth():
    env, _ = _env("all", 11, mode="eval")
    secret_id = env._truth["issues"][0]["affected_ids"][0]
    bad = {"episode_id": env._ep.episode_id, "findings": [dict(finding(env._truth["issues"][0]),
                                                              claimed_amount_minor=1.5)]}
    obs, *_ = _submit(env, raw=bad)
    blob = repr(obs)
    assert secret_id not in blob and "iss_" not in blob and "expected" not in blob


@pytest.mark.parametrize("action", [
    {"action": "submit", "args": {}}, {"action": "submit"}, {"action": "submit", "args": None},
    {"action": "submit", "args": {"findings": {"episode_id": "x", "findings": []}, "extra": 1}},
    {"action": "submit", "args": "findings"}])
def test_env_submit_with_invalid_structure_is_an_invalid_submission(action):
    # SPEC section 4: invalid submission structure => reward -1, terminated, error in the grader log
    env, _ = _env("all", 11)
    obs, r, term, trunc, info = env.step(action)
    assert term and not trunc and r == -1.0
    assert obs["error"]["code"] == "invalid_args" and len(obs["error"]["message"]) <= 120
    assert info["invalid_submission"] is True and info["inspection_steps"] == 0
    rep = env.grader_report()
    assert rep["invalid_submission"] is True and rep["errors"] and rep["reward_shared"] == -1.0


def test_env_step_limit_truncates_with_empty_findings_scored_like_silence():
    env, _ = _env("clean", 12, max_steps=10)
    last = None
    for _ in range(10):
        last = env.step(_peek(env))
    obs, r, term, trunc, info = last
    assert trunc and not term
    # clean book + empty answer: F1=1 minus the cost of the 10 steps that were used
    assert r == pytest.approx(1.0 - 0.002 * 10)
    assert info["metrics"]["f1"] == 1.0 and info["invalid_submission"] is False
    env2, _ = _env("all", 11, max_steps=10)
    for _ in range(10):
        last = env2.step(_peek(env2))
    assert last[3] is True and last[1] == -1.0


def test_env_default_budget_is_100_steps_including_submit():
    env, _ = _env("clean", 12)
    for k in range(99):
        o, r, term, trunc, _ = env.step(_peek(env))
        assert not term and not trunc
    # the 100th step: a submit is still honoured and charged 99 inspection steps
    _, r, term, trunc, info = _submit(env, [])
    assert term and not trunc
    assert r == pytest.approx(1.0 - 0.002 * 99)
    assert info["inspection_steps"] == 99

    env2, _ = _env("clean", 12)
    for _ in range(99):
        env2.step(_peek(env2))
    o, r, term, trunc, _ = env2.step(_peek(env2))     # 100th non-submit step
    assert trunc and not term and r == pytest.approx(1.0 - 0.2)


def test_env_time_limit_truncates_and_scores_empty_answer():
    env = AuditEnv(mode="train")
    env.reset(seed=11, config=cfg_of("all"))
    t = [0.0]
    env._clock = lambda: t[0]
    env._t0 = 0.0
    env.step(_peek(env))
    t[0] = 1000.0
    obs, r, term, trunc, info = env.step(_peek(env))
    assert trunc and not term
    assert r == -1.0 and info["metrics"]["tp"] == 0 and info["metrics"]["fn"] == 4
    # a late submit is not honoured either
    obs2, r2, term2, trunc2, _ = _submit(env, _truth_findings(env))
    assert obs2["error"]["code"] == "episode_over" and r2 == 0.0


def test_env_timeout_on_clean_book_scores_like_silence_minus_steps():
    env = AuditEnv(mode="train")
    env.reset(seed=12, config=cfg_of("clean"))
    t = [0.0]
    env._clock = lambda: t[0]
    env._t0 = 0.0
    t[0] = 61.0
    obs, r, term, trunc, info = env.step(_peek(env))
    # the step that hit the limit is counted as an inspection step, exactly like the 100th step
    assert trunc and r == pytest.approx(1.0 - 0.002 * 1)


# ============================================================================= baselines through the runner
@functools.lru_cache(maxsize=None)
def _run(policy: str, kind: str, seed: int, remote: bool = False):
    from auditgym.runner import run_episode
    return run_episode(policy, seed, cfg_of(kind), mode="eval", remote=remote)


BASELINE_EPISODES = [("all", 11), ("clean", 12), ("inter", 21)]


@pytest.mark.parametrize("kind,seed", BASELINE_EPISODES)
def test_oracle_gets_exactly_one(kind, seed):
    rep = _run("oracle", kind, seed)["grader_report"]
    assert rep["reward_shared"] == 1.0
    assert rep["metrics"]["f1"] == 1.0 and rep["metrics"]["fp"] == 0 and rep["metrics"]["fn"] == 0
    assert rep["inspection_steps"] == 0 and rep["invalid_submission"] is False
    assert rep["metrics"]["tp"] == rep["n_truth_issues"]


@pytest.mark.parametrize("kind,seed", BASELINE_EPISODES)
def test_flag_all_scores_below_zero(kind, seed):
    rep = _run("flag_all", kind, seed)["grader_report"]
    assert rep["invalid_submission"] is False
    assert rep["reward_shared"] < 0.0, rep["metrics"]
    assert rep["reward_shared"] < _run("oracle", kind, seed)["grader_report"]["reward_shared"]


@pytest.mark.parametrize("policy", ["random_explorer", "reconcile"])
@pytest.mark.parametrize("kind,seed", BASELINE_EPISODES)
def test_other_baselines_are_valid_bounded_and_never_beat_the_oracle(policy, kind, seed):
    rep = _run(policy, kind, seed)["grader_report"]
    assert rep["invalid_submission"] is False
    assert -1.0 <= rep["reward_shared"] <= _run("oracle", kind, seed)["grader_report"]["reward_shared"]
    assert -1.0 <= rep["reward_experimental"] <= 1.0
    m = rep["metrics"]
    assert 0.0 <= m["precision"] <= 1.0 and 0.0 <= m["recall"] <= 1.0 and 0.0 <= m["f1"] <= 1.0


def test_oracle_beats_every_other_baseline_on_every_episode_with_issues():
    for kind, seed in (("all", 11), ("inter", 21)):
        o = _run("oracle", kind, seed)["grader_report"]["reward_shared"]
        for p in ("random_explorer", "reconcile", "flag_all"):
            assert _run(p, kind, seed)["grader_report"]["reward_shared"] < o, (p, kind)


def test_baseline_runs_are_deterministic_and_use_at_most_the_step_budget():
    from auditgym.runner import run_episode
    a = run_episode("random_explorer", 11, cfg_of("all"), mode="eval", remote=False)
    b = run_episode("random_explorer", 11, cfg_of("all"), mode="eval", remote=False)
    assert a["trajectory"] == b["trajectory"] and a["grader_report"] == b["grader_report"]
    for kind, seed in BASELINE_EPISODES:
        for p in ("random_explorer", "reconcile", "flag_all", "oracle"):
            assert _run(p, kind, seed)["steps"] <= 100


def test_baseline_rewards_in_grader_report_match_recomputed_score_from_the_submission():
    from auditgym.runner import submitted_findings
    for p in ("reconcile", "flag_all", "random_explorer"):
        for kind, seed in BASELINE_EPISODES:
            run = _run(p, kind, seed)
            ep = _ep(seed, kind)
            findings = submitted_findings(run["trajectory"])
            rep = run["grader_report"]
            if findings is None:
                continue
            g = matcher.score(findings, ep.truth, ep.public_ids(), rep["inspection_steps"], ep.episode_id)
            assert g["reward_shared"] == rep["reward_shared"]
            assert g["reward_experimental"] == rep["reward_experimental"]
            assert (g["tp"], g["fp"], g["fn"]) == (rep["metrics"]["tp"], rep["metrics"]["fp"], rep["metrics"]["fn"])


def test_remote_grader_subprocess_oracle_and_flag_all_agree_with_in_process_scores():
    local_o = _run("oracle", "all", 11)["grader_report"]["reward_shared"]
    remote_o = _run("oracle", "all", 11, True)["grader_report"]["reward_shared"]
    assert local_o == remote_o == 1.0
    local_f = _run("flag_all", "all", 11)["grader_report"]["reward_shared"]
    remote_f = _run("flag_all", "all", 11, True)["grader_report"]["reward_shared"]
    assert local_f == remote_f < 0.0


def test_eval_trajectory_never_contains_a_reward_or_truth_tokens():
    run = _run("oracle", "all", 11)
    ep = _ep(11, "all")
    assert all(rec["reward"] == 0.0 for rec in run["trajectory"])
    blob = repr([r["observation"] for r in run["trajectory"]])
    for tok in ("iss_0", "expected_amount", "evidence_sets", "tolerance_minor", "provenance"):
        assert tok not in blob
    assert ep.truth["issues"][0]["issue_id"] not in blob


# ============================================================================= experimental credit leak
def test_experimental_bare_type_and_ids_guess_earns_no_partial_credit():
    # type + affected_ids right, but no evidence and a far-off amount: nothing to reward
    t = hand_truth(H1)
    bare = hscore([hfind(amount=5, ev=())], t)
    silent = hscore([], t)
    assert bare["reward_shared"] == -1.0
    assert bare["reward_experimental"] == pytest.approx(
        matcher.experimental_reward(0, 0, 1, 1, 0)) == pytest.approx(-1.0)
    assert bare["reward_experimental"] <= silent["reward_experimental"]


@pytest.mark.parametrize("kw, credited", [
    (dict(amount=5, ev=("D-a",)), True),                    # cites existing evidence (insufficient set)
    (dict(amount=5, ev=("D-a", "NOPE")), False),             # cites an id that does not exist
    (dict(amount=100000 + 10 * 100, ev=()), True),           # amount exactly at 10x tolerance (tol = 100)
    (dict(amount=100000 + 10 * 100 + 1, ev=()), False),      # just beyond 10x tolerance
    (dict(amount=100000, ev=(), cur="USD"), False),          # right amount, wrong currency, no evidence
    (dict(amount=100000, ev=("D-a",), cur="USD"), True),     # wrong currency but real evidence
])
def test_experimental_partial_credit_requires_evidence_or_near_amount(kw, credited):
    g = hscore([hfind(**kw)], hand_truth(H1))
    assert g["tp"] == 0 and g["reward_shared"] == -1.0
    assert (g["reward_experimental"] > -1.0) is credited


def test_experimental_type_spray_without_evidence_still_loses_and_is_not_better_than_silence():
    t = hand_truth(H1, mk_issue("iss_02", "fx_error", ["INV-2"], 5000, [["D-b"]]))
    spray = [hfind(type_=ty, ids=(i,), amount=1, ev=())
             for ty in ISSUE_TYPES for i in ("INV-1", "INV-2")]
    g = hscore(spray, t)
    assert g["reward_experimental"] == pytest.approx(-1.0)
    assert g["reward_experimental"] <= hscore([], t)["reward_experimental"]


# ============================================================================= paths not hit by generated data
def test_split_findings_for_a_two_id_issue_score_zero_true_positives():
    two = mk_issue("iss_01", "payment_misallocation", ["INV-1", "INV-2"], 100000, [["D-a", "R-1"]])
    t = hand_truth(two)
    halves = [hfind("payment_misallocation", ids=("INV-1",), amount=50000),
              hfind("payment_misallocation", ids=("INV-2",), amount=50000)]
    g = hscore(halves, t)
    assert (g["tp"], g["fp"], g["fn"]) == (0, 2, 1) and g["f1"] == 0.0
    assert g["reward_shared"] == -1.0 and g["pairs"] == []
    whole = hscore([hfind("payment_misallocation", ids=("INV-2", "INV-1"))], t)
    assert (whole["tp"], whole["fp"], whole["fn"]) == (1, 0, 0) and whole["f1"] == 1.0
    # supersets of the truth ids do not match either
    sup = hscore([hfind("payment_misallocation", ids=("INV-1", "INV-2", "INV-3"))], t)
    assert sup["tp"] == 0


def test_duplicated_ids_in_affected_ids_are_accepted_by_design_as_a_set():
    # ids are compared as sets (module docstring): repeats are neither rejected nor rewarded
    g = hscore([hfind(ids=("INV-1", "INV-1", " INV-1 "))], hand_truth(H1))
    assert g["valid"] and g["tp"] == 1 and g["f1"] == 1.0
    two = mk_issue("iss_01", "duplicate_invoice", ["INV-1", "INV-2"], 100000, [["D-a", "R-1"]])
    g2 = hscore([hfind(ids=("INV-1", "INV-1"))], hand_truth(two))
    assert g2["valid"] and g2["tp"] == 0           # repeating an id never fakes a second one


def test_same_type_same_ids_pair_is_resolved_deterministically_and_one_to_one():
    a = mk_issue("iss_01", "duplicate_invoice", ["INV-1"], 100000, [["D-a"]])
    b = mk_issue("iss_02", "duplicate_invoice", ["INV-1"], 100000, [["D-b"]])
    t = hand_truth(b, a)                                  # truth order must not matter
    fs = [hfind(ev=("D-a", "D-b")), hfind(ev=("D-a", "D-b")), hfind(ev=("D-a", "D-b"))]
    g = hscore(fs, t)
    assert (g["tp"], g["fp"], g["fn"]) == (2, 1, 0)
    # Kuhn: f0 takes iss_01, f1 augments and pushes f0 onto iss_02 (fixed, documented outcome)
    assert g["pairs"] == [[0, "iss_02"], [1, "iss_01"]]
    assert g["finding_reasons"][2] == matcher.REASON_DUPLICATE
    assert hscore(fs, hand_truth(a, b)) == g


def test_grader_pairs_are_sorted_canonically_by_finding_index():
    a = mk_issue("iss_01", "duplicate_invoice", ["INV-1"], 100000, [["D-a"]])
    b = mk_issue("iss_02", "fx_error", ["INV-2"], 5000, [["D-b"]])
    fs = [hfind("fx_error", ids=("INV-2",), amount=5000, ev=("D-b",)),
          hfind(ev=("D-a",))]
    g = hscore(fs, hand_truth(b, a))
    assert g["pairs"] == [[0, "iss_02"], [1, "iss_01"]]
    assert [p[0] for p in g["pairs"]] == sorted(p[0] for p in g["pairs"])


def test_truncated_clean_book_run_never_beats_an_early_silent_submit():
    # Documented behaviour: truncation == empty findings, so on a clean book it keeps F1 = 1 and
    # only loses the step cost; an honest silent submit after fewer steps is at least as good.
    env, _ = _env("clean", 12, max_steps=10)
    last = None
    for _ in range(10):
        last = env.step(_peek(env))
    assert last[3] is True
    truncated_reward = last[1]
    env2, _ = _env("clean", 12)
    for _ in range(5):
        env2.step(_peek(env2))
    _, silent_reward, term, trunc, _ = _submit(env2, [])
    assert term and not trunc
    assert truncated_reward <= silent_reward
    # at the default 100-step budget: truncated 0.8 versus silent submit after 99 steps 0.802
    env3, _ = _env("clean", 12)
    for _ in range(100):
        out = env3.step(_peek(env3))
    assert out[3] is True and out[1] == pytest.approx(0.8)
    assert out[1] <= 1.0 - STEP_COST * 99
