"""Episode / truth validation (SPEC section 8 a-e).

``validate_episode(ep)`` needs a full episode (public records + truth + provenance).  It checks that
the *hidden* truth, the *provenance snapshot* and the *public records* tell one consistent story,
using the independent rules engine in ``rules.py``.

CLI:  python -m auditgym.validate PUBLIC_ROOT PRIVATE_ROOT [--json]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Any

from .constants import (ACCOUNTS, DISTRACTORS, ENV_VERSION, INTERACTIONS, ISSUE_TYPES,
                        LEDGER_FIELDS, MAX_EVIDENCE_IDS)
from .content_id import content_episode_id
from .episode import PUBLIC_FILES, Episode, load_episode, render_public
from . import rules

# default sizes from SPEC section 1 ("Sizes")
DEFAULT_SIZES = {"ledger_rows": (200, 2000), "documents": (20, 200), "issues": (0, 12)}


# ---------------------------------------------------------------- tiny helpers
def _res(ok: bool, detail: str = "") -> dict:
    return {"ok": bool(ok), "detail": detail}


def _short(items, n: int = 4) -> str:
    items = list(items)
    s = ", ".join(str(x) for x in items[:n])
    return s + (f" (+{len(items) - n} more)" if len(items) > n else "")


def _tol(expected: int) -> int:
    return max(1, int(expected) // 1000)


def _group_key(r: dict):
    # a payment posting group is keyed by payment_id (its AR credit rows may also carry invoice_id)
    return (r["timestamp"], r.get("payment_id") or r.get("invoice_id") or None)


def _balance_checks(ledger: list[dict]) -> tuple[dict, dict]:
    """(totals check, posting-group check)."""
    dr = sum(int(r["debit_minor"]) for r in ledger)
    cr = sum(int(r["credit_minor"]) for r in ledger)
    totals = _res(dr == cr, f"sum debit {dr} vs sum credit {cr}")
    groups: dict[Any, int] = {}
    for r in ledger:
        k = _group_key(r)
        groups[k] = groups.get(k, 0) + int(r["debit_minor"]) - int(r["credit_minor"])
    bad = [(k, v) for k, v in groups.items() if v != 0]
    grp = _res(not bad, f"{len(groups)} posting groups, {len(bad)} unbalanced"
               + (f": {_short(bad, 3)}" if bad else ""))
    return totals, grp


def _ledger_shape(ep: Episode) -> dict:
    base = ep.manifest["base_currency"]
    problems = []
    for r in ep.ledger:
        d, c = int(r["debit_minor"]), int(r["credit_minor"])
        if d < 0 or c < 0 or (d > 0) == (c > 0):
            problems.append(f"{r['row_id']}: debit/credit not exclusive positive")
        elif r["currency"] != base:
            problems.append(f"{r['row_id']}: currency {r['currency']}")
        elif r["account"] not in ACCOUNTS:
            problems.append(f"{r['row_id']}: account {r['account']}")
        if set(LEDGER_FIELDS) - set(r):
            problems.append(f"{r['row_id']}: missing fields")
    ids = [r["row_id"] for r in ep.ledger]
    if len(set(ids)) != len(ids):
        problems.append("duplicate row_id")
    return _res(not problems, "ok" if not problems else _short(problems))


def _invoice_total(inv: dict) -> int:
    from decimal import Decimal, ROUND_HALF_EVEN
    total = 0
    for li in inv.get("line_items") or []:
        net = int(li["quantity"]) * int(li["unit_price_minor"])
        vat = int((Decimal(net) * Decimal(str(li["vat_rate"]))).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
        total += net + vat
    return total


def _structure(ep: Episode) -> dict:
    ctx = rules._Ctx(ep)
    problems = []
    ids = [i["invoice_id"] for i in ep.invoices] + [p["payment_id"] for p in ep.payments]
    docs = [i["doc_id"] for i in ep.invoices] + [p["doc_id"] for p in ep.payments]
    if len(set(ids)) != len(ids):
        problems.append("duplicate invoice/payment ids")
    if len(set(docs)) != len(docs):
        problems.append("duplicate doc_ids")
    paid = {ref["invoice_id"] for p in ep.payments for ref in (p.get("invoice_refs") or [])}
    for inv in ep.invoices:
        # prior-period invoices exist as (unposted) documents -- the mirror of the post-period ones; a payment
        # inside the period may not settle one
        if ctx.ts(inv["issued_at"]) < ctx.p_start and inv["invoice_id"] in paid:
            problems.append(f"{inv['invoice_id']} issued before period_start but paid inside the period")
        if _invoice_total(inv) != int(inv["total_minor"]):
            problems.append(f"{inv['invoice_id']} total_minor != sum of lines")
    for pay in ep.payments:
        if not ctx.in_period(pay["timestamp"]):
            problems.append(f"{pay['payment_id']} outside period")
        for ref in pay.get("invoice_refs") or []:
            inv = ctx.invoices.get(ref["invoice_id"])
            if inv is None:
                problems.append(f"{pay['payment_id']} refs unknown invoice")
            elif inv["currency"] != pay["currency"]:
                problems.append(f"{pay['payment_id']} currency != {inv['invoice_id']} currency")
    for r in ep.ledger:
        if not ctx.in_period(r["timestamp"]):
            problems.append(f"ledger {r['row_id']} outside period")
        if r.get("invoice_id") and r["invoice_id"] not in ctx.invoices:
            problems.append(f"ledger {r['row_id']} unknown invoice")
        if r.get("payment_id") and r["payment_id"] not in ctx.payments:
            problems.append(f"ledger {r['row_id']} unknown payment")
    return _res(not problems, "ok" if not problems else _short(problems))


# ---------------------------------------------------------------- (a) clean snapshot
def _clean_episode(ep: Episode) -> Episode | None:
    clean = (ep.provenance or {}).get("clean_records")
    if not clean:
        return None
    return Episode(ep.manifest, ep.policy, clean["ledger"], clean["invoices"], clean["payments"], ep.fx)


# ---------------------------------------------------------------- (b) rules vs truth
def _compare_violations(violations: list[dict], issues: list[dict], base: str) -> tuple[bool, str]:
    key = lambda t, a: (t, frozenset(a))
    pool: dict[Any, list[dict]] = {}
    for v in violations:
        pool.setdefault(key(v["type"], v["affected_ids"]), []).append(v)
    missing, mismatched = [], []
    for iss in issues:
        k = key(iss["type"], iss["affected_ids"])
        exp = int(iss["expected_amount_minor"])
        tol = int(iss.get("tolerance_minor", _tol(exp)))
        cands = pool.get(k, [])
        best = None
        for v in cands:
            if abs(int(v["amount_minor"]) - exp) <= tol and (
                    best is None or abs(int(v["amount_minor"]) - exp) < abs(int(best["amount_minor"]) - exp)):
                best = v
        if best is not None:
            cands.remove(best)
        elif cands:
            cands_snapshot = [c["amount_minor"] for c in cands]
            cands.clear()
            mismatched.append(f"{iss['issue_id']}({iss['type']} {_short(iss['affected_ids'])}: "
                              f"truth {exp} vs rules {cands_snapshot})")
        else:
            missing.append(f"{iss['issue_id']}({iss['type']} {_short(iss['affected_ids'])} amt {exp})")
    extra = [f"{v['type']} {_short(v['affected_ids'])} amt {v['amount_minor']}"
             for vs in pool.values() for v in vs]
    wrong_cur = [v for v in violations if v["currency"] != base]
    ok = not (missing or mismatched or extra or wrong_cur)
    if ok:
        return True, f"{len(issues)} truth issues == {len(violations)} rule violations"
    parts = []
    if missing:
        parts.append(f"truth-not-found-by-rules: {_short(missing)}")
    if mismatched:
        parts.append(f"amount-mismatch: {_short(mismatched)}")
    if extra:
        parts.append(f"rules-found-not-in-truth: {_short(extra)}")
    if wrong_cur:
        parts.append("rules currency != base")
    return False, "; ".join(parts)


# ---------------------------------------------------------------- truth well-formedness
def _truth_wellformed(ep: Episode) -> dict:
    t = ep.truth or {}
    problems = []
    if t.get("episode_id") != ep.episode_id:
        problems.append("truth.episode_id != manifest.episode_id")
    if t.get("base_currency") != ep.manifest["base_currency"]:
        problems.append("truth.base_currency mismatch")
    issues = t.get("issues", [])
    ids = [i.get("issue_id") for i in issues]
    if len(set(ids)) != len(ids):
        problems.append("duplicate issue_id")
    expect = [f"iss_{n:02d}" for n in range(1, len(issues) + 1)]
    if sorted(ids) != expect:
        problems.append("issue_ids are not iss_01..iss_NN")
    scen = Counter()
    for i in issues:
        if i.get("type") not in ISSUE_TYPES:
            problems.append(f"{i.get('issue_id')} bad type")
        if not i.get("affected_ids"):
            problems.append(f"{i.get('issue_id')} empty affected_ids")
        if i.get("currency") != ep.manifest["base_currency"]:
            problems.append(f"{i.get('issue_id')} currency != base")
        exp = i.get("expected_amount_minor")
        if not isinstance(exp, int) or isinstance(exp, bool) or exp < 0:
            problems.append(f"{i.get('issue_id')} bad expected amount")
        elif i.get("tolerance_minor") != _tol(exp):
            problems.append(f"{i.get('issue_id')} tolerance != max(1, expected//1000)")
        if not i.get("evidence_sets"):
            problems.append(f"{i.get('issue_id')} no evidence_sets")
        sc = i.get("scenario")
        if sc is not None:
            if sc not in INTERACTIONS:
                problems.append(f"{i.get('issue_id')} unknown scenario")
            scen[sc] += 1
    for sc, n in scen.items():
        if n % 2:
            problems.append(f"scenario {sc} has odd issue count {n}")
    return _res(not problems, f"{len(issues)} issues well-formed" if not problems else _short(problems))


# ---------------------------------------------------------------- (c) evidence
def _evidence_check(ep: Episode) -> dict:
    pub = ep.public_ids()
    problems = []
    n_sets = 0
    for iss in ep.truth.get("issues", []):
        exp = int(iss["expected_amount_minor"])
        tol = int(iss.get("tolerance_minor", _tol(exp)))
        for ev in iss.get("evidence_sets", []):
            n_sets += 1
            tag = f"{iss['issue_id']}[{_short(ev, 3)}]"
            if len(ev) > MAX_EVIDENCE_IDS:
                problems.append(f"{tag}: {len(ev)} ids > {MAX_EVIDENCE_IDS}")
                continue
            if len(set(ev)) != len(ev):
                problems.append(f"{tag}: repeated ids")
            unknown = [e for e in ev if e not in pub]
            if unknown:
                problems.append(f"{tag}: unknown ids {_short(unknown, 2)}")
                continue
            ok, amt = rules.check_evidence(iss, ev, ep)
            if not ok:
                problems.append(f"{tag}: evidence insufficient for {iss['type']}")
            elif abs(amt - exp) > tol:
                problems.append(f"{tag}: evidence amount {amt} vs expected {exp}")
    return _res(not problems, f"{n_sets} evidence sets verified" if not problems else _short(problems))


# ---------------------------------------------------------------- (d) leakage
def _leak_check(ep: Episode, public_dir: str | None = None) -> dict:
    blobs = render_public(ep)
    tokens = {"iss_", "expected_amount", "evidence_sets", "provenance", "truth.json", "scenario",
              "tolerance_minor", '"seed"', "clean_records"}
    tokens.update(INTERACTIONS)
    tokens.update(DISTRACTORS)
    tokens.update(i.get("issue_id", "") for i in ep.truth.get("issues", []))
    tokens.discard("")
    hits = []
    for name, text in blobs.items():
        low = text.lower()
        for tok in sorted(tokens):
            if tok.lower() in low:
                hits.append(f"{name}:{tok}")
    if content_episode_id(ep) != ep.manifest.get("episode_id"):
        hits.append("manifest:episode_id != content hash of the public records")
    if public_dir is not None and os.path.isdir(public_dir):
        extra = sorted(set(os.listdir(public_dir)) - set(PUBLIC_FILES))
        if extra:
            hits.append(f"public dir has extra files: {_short(extra)}")
        if ep.episode_id not in os.path.basename(os.path.normpath(public_dir)):
            hits.append("public dir name != episode_id")
    return _res(not hits, "no truth/provenance tokens in public files" if not hits else _short(hits, 6))


# ---------------------------------------------------------------- (e) counts & sizes
def _counts_check(ep: Episode) -> dict:
    c = ep.manifest.get("counts", {})
    actual = {"ledger_rows": len(ep.ledger), "invoices": len(ep.invoices), "payments": len(ep.payments),
              "documents": len(ep.invoices) + len(ep.payments), "fx_rates": len(ep.fx)}
    bad = [f"{k}: manifest {c.get(k)} vs actual {v}" for k, v in actual.items() if c.get(k) != v]
    return _res(not bad, f"counts {actual}" if not bad else _short(bad))


def _sizes_check(ep: Episode, sizes: dict) -> dict:
    vals = {"ledger_rows": len(ep.ledger), "documents": len(ep.invoices) + len(ep.payments),
            "issues": len((ep.truth or {}).get("issues", []))}
    bad = []
    for k, (lo, hi) in sizes.items():
        if k in vals and not (lo <= vals[k] <= hi):
            bad.append(f"{k} {vals[k]} not in [{lo},{hi}]")
    return _res(not bad, f"sizes {vals}" if not bad else _short(bad))


# ================================================================= public entry points
def validate_episode(ep: Episode, sizes: dict | None = None, public_dir: str | None = None) -> dict:
    """Run all SPEC section 8 (a)-(e) checks.  Returns {"ok": bool, "checks": {name: {"ok","detail"}}}."""
    checks: dict[str, dict] = {}

    def run(name, fn, *a):
        try:
            r = fn(*a)
            if r is not None:
                checks[name] = r
        except Exception as e:  # a crashing check is a failing check, never a crash of the validator
            checks[name] = _res(False, f"check crashed: {type(e).__name__}: {e}")

    # (a) clean snapshot
    clean = _clean_episode(ep)
    if clean is None:
        for n in ("a_clean_balanced", "a_clean_groups_balanced", "a_clean_no_violations"):
            checks[n] = _res(False, "provenance.clean_records missing")
    else:
        def a_all():
            t, g = _balance_checks(clean.ledger)
            v = rules.find_violations(clean)
            checks["a_clean_balanced"] = t
            checks["a_clean_groups_balanced"] = g
            checks["a_clean_no_violations"] = _res(
                not v, "rules engine finds 0 violations on clean snapshot" if not v else
                f"{len(v)} violations on clean snapshot: "
                + _short([f"{x['type']} {_short(x['affected_ids'])} {x['amount_minor']}" for x in v]))
        run("a_clean_balanced", a_all)

    # (b) final records
    def b_all():
        t, g = _balance_checks(ep.ledger)
        checks["b_final_balanced"] = t
        checks["b_final_groups_balanced"] = g
        v = rules.find_violations(ep)
        ok, detail = _compare_violations(v, (ep.truth or {}).get("issues", []), ep.manifest["base_currency"])
        checks["b_rules_match_truth"] = _res(ok, detail)
    run("b_final_balanced", b_all)
    run("b_ledger_shape", _ledger_shape, ep)
    run("b_structure", _structure, ep)
    run("b_truth_wellformed", _truth_wellformed, ep)

    # (c) evidence sets
    run("c_evidence_sets", _evidence_check, ep)
    # (d) leakage
    run("d_no_leak", _leak_check, ep, public_dir)
    # (e) counts and sizes
    run("e_counts", _counts_check, ep)
    run("e_sizes", _sizes_check, ep, sizes or DEFAULT_SIZES)

    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


def validate_dir(public_root: str, private_root: str | None, sizes: dict | None = None) -> dict:
    """Validate every episode directory under `public_root` (or a single episode dir).

    Private data is read from `private_root/<episode_id>` (or `private_root` itself if it directly
    contains truth.json).  Returns {"ok", "n", "n_ok", "episodes": {episode_id: result}}.
    """
    if os.path.exists(os.path.join(public_root, "manifest.json")):
        pubs = [public_root]
    else:
        pubs = sorted(os.path.join(public_root, d) for d in os.listdir(public_root)
                      if os.path.exists(os.path.join(public_root, d, "manifest.json")))
    results: dict[str, dict] = {}
    for pub in pubs:
        name = os.path.basename(os.path.normpath(pub))
        priv = None
        if private_root:
            if os.path.exists(os.path.join(private_root, "truth.json")):
                priv = private_root
            else:
                priv = os.path.join(private_root, name)
        try:
            ep = load_episode(pub, priv)
            res = validate_episode(ep, sizes=sizes, public_dir=pub)
            res["episode_id"] = ep.episode_id
        except Exception as e:
            res = {"ok": False, "checks": {"load": _res(False, f"{type(e).__name__}: {e}")}}
        results[name] = res
    n_ok = sum(1 for r in results.values() if r["ok"])
    return {"ok": bool(results) and n_ok == len(results), "n": len(results), "n_ok": n_ok,
            "episodes": results}


def format_summary(report: dict) -> str:
    lines = [f"validated {report['n']} episode(s): {report['n_ok']} ok, {report['n'] - report['n_ok']} failed"]
    for name, res in report["episodes"].items():
        lines.append(f"  [{'OK ' if res['ok'] else 'FAIL'}] {name}")
        for cn, c in res["checks"].items():
            if not c["ok"]:
                lines.append(f"       - {cn}: {c['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    argv = [a for a in argv if a != "--json"]
    if not argv:
        print("usage: python -m auditgym.validate PUBLIC_ROOT [PRIVATE_ROOT] [--json]", file=sys.stderr)
        return 2
    report = validate_dir(argv[0], argv[1] if len(argv) > 1 else None)
    print(json.dumps(report, indent=2, sort_keys=True) if as_json else format_summary(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
