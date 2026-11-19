"""AuditGym matcher: submission schema, normalisation, 1-1 matching, metrics and rewards.

This module is published, deterministic, stdlib-only code. It is the whole of the grading
logic: nothing about an explanation text, the order of ids or the iteration order of any
dict/set ever changes the result.

Submission schema (strict)
--------------------------
A submission is a JSON object with exactly the keys ``episode_id`` (must equal the
episode's id) and ``findings`` (a list of at most 30 findings). Each finding is an object
with exactly these keys:

    type                  str, one of ISSUE_TYPES after normalisation
    affected_ids          non-empty list of str, at most 10
    claimed_amount_minor  int (``bool`` is rejected), 0 <= x <= 10**18, base-currency minor units
                          (larger values are a malformed submission)
    currency              str
    evidence_ids          list of str, at most 10
    explanation           str, at most 1000 characters (never affects the score)

Every string must be UTF-8 encodable (a lone surrogate is a malformed submission).
A missing, extra or wrongly typed key anywhere makes the whole submission invalid:
reward -1 for both the shared and the experimental reward, F1 reported as 0.

Normalisation
-------------
Leading/trailing whitespace is stripped from every string. ``type`` is lowercased,
``currency`` is uppercased. Ids are otherwise kept as-is (case-sensitive). ``affected_ids``
and ``evidence_ids`` are compared as *sets* (order and repeats do not matter).

Eligibility of a finding for a truth issue (all must hold)
----------------------------------------------------------
  1. type equal;
  2. set(affected_ids) == set(issue.affected_ids);
  3. currency == issue.currency;
  4. |claimed_amount_minor - expected_amount_minor| <= issue.tolerance_minor
     (inclusive; the policy states ``max(1, expected // 1000)``);
  5. every evidence id exists in the episode's public ids;
  6. some evidence set S of the issue satisfies S <= set(evidence_ids)
     (extra, valid evidence ids are allowed).

Matching and tie-breaking
-------------------------
Eligibility defines a bipartite graph between findings and issues. The matcher computes a
maximum-cardinality matching with Kuhn's augmenting-path algorithm, visiting findings in
submitted order and trying issues in sorted ``issue_id`` order. This makes the result
deterministic even when several findings are eligible for several issues. Every finding
is matched to at most one issue and vice versa, so a repeated finding is a false
positive. Unmatched findings are FP, unmatched issues are FN.

Metrics
-------
precision = TP / (TP + FP), recall = TP / (TP + FN) (0 when the denominator is 0),
F1 = 2TP / (2TP + FP + FN). If truth and findings are both empty, F1 = 1 while precision and
recall are 0 (0/0 -> 0, as everywhere else). If truth is empty and findings are not, F1 = 0.
Metrics are rounded to 6 decimals.

Rewards
-------
shared (the official one):  clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1).
experimental (reported separately, never replaces the shared one; it is shaped, so do not use
it for model selection): soft-F1 = 2*(TP + 0.25*P) / (n_truth + n_findings), where P counts
findings that match an issue on type and affected_ids but fail the amount, currency or
evidence checks AND show some work (cite evidence ids that all exist publicly, or claim an
amount within 10x the tolerance in the right currency). A bare type + ids guess earns
nothing. Still strictly one-to-one, assigned by the same deterministic augmenting-path
procedure on the leftovers after the exact matching, with the same step cost. Rewards are rounded to 6 decimals. An invalid submission gets -1.0
for both.

Floats are used only for metrics and rewards; amounts are integers.
"""
from __future__ import annotations

from typing import Any

from .constants import (ISSUE_TYPES, MAX_AFFECTED_IDS, MAX_EVIDENCE_IDS,
                        MAX_EXPLANATION_CHARS, MAX_FINDINGS, STEP_COST)

SUBMISSION_KEYS = frozenset({"episode_id", "findings"})
FINDING_KEYS = frozenset({"type", "affected_ids", "claimed_amount_minor", "currency",
                          "evidence_ids", "explanation"})

REASON_TP = "tp"
REASON_WRONG_TYPE = "wrong_type"
REASON_IDS_MISMATCH = "ids_mismatch"
REASON_CURRENCY_MISMATCH = "currency_mismatch"
REASON_AMOUNT = "amount_out_of_tolerance"
REASON_EVIDENCE_UNKNOWN = "evidence_unknown_id"
REASON_EVIDENCE_INSUFFICIENT = "evidence_insufficient"
REASON_DUPLICATE = "duplicate_of_matched"
REASON_UNMATCHED = "unmatched"

# Order in which eligibility is checked; a finding's diagnostic reason for an unmatched
# finding is the furthest stage it reached against any issue.
_STAGES = (REASON_WRONG_TYPE, REASON_IDS_MISMATCH, REASON_CURRENCY_MISMATCH,
           REASON_AMOUNT, REASON_EVIDENCE_UNKNOWN, REASON_EVIDENCE_INSUFFICIENT, REASON_TP)
_STAGE_RANK = {r: i for i, r in enumerate(_STAGES)}

PARTIAL_CREDIT = 0.25
MAX_CLAIMED_AMOUNT_MINOR = 10 ** 18


# ------------------------------------------------------------------ validation
def _short(s: Any) -> str:
    return str(s)[:40]


def _is_str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _encodable(s: str) -> bool:
    """False for strings that cannot be encoded as UTF-8 (lone surrogates)."""
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def validate_submission(sub: Any, episode_id: str) -> tuple[bool, str | None, dict | None]:
    """Strict schema check. Returns (ok, error_message, normalised_submission).

    The normalised submission is ``{"episode_id": str, "findings": [normalised finding]}``.
    The error message is short and never echoes more than 40 chars of any offending key.
    """
    if not isinstance(sub, dict):
        return False, "submission must be an object", None
    keys = set(sub.keys())
    if not all(isinstance(k, str) for k in keys):
        return False, "submission keys must be strings", None
    missing = SUBMISSION_KEYS - keys
    if missing:
        return False, "missing key: " + _short(sorted(missing)[0]), None
    extra = keys - SUBMISSION_KEYS
    if extra:
        return False, "unexpected key: " + _short(sorted(extra)[0]), None
    if not isinstance(sub["episode_id"], str) or sub["episode_id"] != episode_id:
        return False, "episode_id mismatch", None
    findings = sub["findings"]
    if not isinstance(findings, list):
        return False, "findings must be a list", None
    if len(findings) > MAX_FINDINGS:
        return False, f"too many findings (max {MAX_FINDINGS})", None

    out: list[dict] = []
    for i, f in enumerate(findings):
        if not isinstance(f, dict):
            return False, f"finding {i}: must be an object", None
        fkeys = set(f.keys())
        if not all(isinstance(k, str) for k in fkeys):
            return False, f"finding {i}: keys must be strings", None
        missing = FINDING_KEYS - fkeys
        if missing:
            return False, f"finding {i}: missing key {_short(sorted(missing)[0])}", None
        extra = fkeys - FINDING_KEYS
        if extra:
            return False, f"finding {i}: unexpected key {_short(sorted(extra)[0])}", None
        if not isinstance(f["type"], str):
            return False, f"finding {i}: type must be a string", None
        if f["type"].strip().lower() not in ISSUE_TYPES:
            return False, f"finding {i}: unknown type", None
        aff = f["affected_ids"]
        if not _is_str_list(aff):
            return False, f"finding {i}: affected_ids must be a list of strings", None
        if not aff or len(aff) > MAX_AFFECTED_IDS:
            return False, f"finding {i}: affected_ids needs 1..{MAX_AFFECTED_IDS} ids", None
        amt = f["claimed_amount_minor"]
        if isinstance(amt, bool) or not isinstance(amt, int):
            return False, f"finding {i}: claimed_amount_minor must be an integer", None
        if amt < 0:
            return False, f"finding {i}: claimed_amount_minor must be >= 0", None
        if amt > MAX_CLAIMED_AMOUNT_MINOR:
            return False, f"finding {i}: claimed_amount_minor must be <= 10**18", None
        if not isinstance(f["currency"], str):
            return False, f"finding {i}: currency must be a string", None
        ev = f["evidence_ids"]
        if not _is_str_list(ev):
            return False, f"finding {i}: evidence_ids must be a list of strings", None
        if len(ev) > MAX_EVIDENCE_IDS:
            return False, f"finding {i}: too many evidence_ids (max {MAX_EVIDENCE_IDS})", None
        if not isinstance(f["explanation"], str):
            return False, f"finding {i}: explanation must be a string", None
        if len(f["explanation"]) > MAX_EXPLANATION_CHARS:
            return False, f"finding {i}: explanation too long", None
        strings = [f["type"], f["currency"], f["explanation"], *aff, *ev]
        if not all(_encodable(x) for x in strings):
            return False, f"finding {i}: strings must be valid UTF-8 text", None
        out.append(normalize_finding(f))
    return True, None, {"episode_id": episode_id, "findings": out}


def normalize_finding(f: dict) -> dict:
    """Normalise one (schema-valid) finding.

    Returns ``{"type", "affected_ids" (frozenset), "claimed_amount_minor", "currency",
    "evidence_ids" (frozenset), "explanation"}``.
    """
    return {
        "type": f["type"].strip().lower(),
        "affected_ids": frozenset(x.strip() for x in f["affected_ids"]),
        "claimed_amount_minor": f["claimed_amount_minor"],
        "currency": f["currency"].strip().upper(),
        "evidence_ids": frozenset(x.strip() for x in f["evidence_ids"]),
        "explanation": f["explanation"].strip(),
    }


# ------------------------------------------------------------------ eligibility
def _tolerance(issue: dict) -> int:
    tol = issue.get("tolerance_minor")
    if tol is None:
        tol = max(1, int(issue["expected_amount_minor"]) // 1000)
    return int(tol)


def eligible(f_norm: dict, issue: dict, public_ids) -> tuple[bool, str]:
    """Return (is_eligible, reason_code). Checks run in a fixed order; the first failure
    is the reason. On success the reason is ``"tp"``."""
    if f_norm["type"] != issue["type"]:
        return False, REASON_WRONG_TYPE
    if set(f_norm["affected_ids"]) != set(issue["affected_ids"]):
        return False, REASON_IDS_MISMATCH
    if f_norm["currency"] != issue["currency"]:
        return False, REASON_CURRENCY_MISMATCH
    if abs(f_norm["claimed_amount_minor"] - int(issue["expected_amount_minor"])) > _tolerance(issue):
        return False, REASON_AMOUNT
    evidence = set(f_norm["evidence_ids"])
    if any(e not in public_ids for e in evidence):
        return False, REASON_EVIDENCE_UNKNOWN
    if not any(set(s) <= evidence for s in issue["evidence_sets"]):
        return False, REASON_EVIDENCE_INSUFFICIENT
    return True, REASON_TP


PARTIAL_AMOUNT_TOLERANCE_FACTOR = 10


def _partial_edge(f_norm: dict, issue: dict, public_ids) -> bool:
    """Soft-credit edge (experimental reward only): type and affected_ids match AND the
    finding shows some work, i.e. it cites at least one evidence id and all cited ids exist
    publicly, or its amount is within 10x the issue tolerance in the issue's currency. A bare
    type + ids guess (no evidence, no near amount) earns nothing."""
    if (f_norm["type"] != issue["type"]
            or set(f_norm["affected_ids"]) != set(issue["affected_ids"])):
        return False
    evidence = set(f_norm["evidence_ids"])
    if evidence and all(e in public_ids for e in evidence):
        return True
    return (f_norm["currency"] == issue["currency"]
            and abs(f_norm["claimed_amount_minor"] - int(issue["expected_amount_minor"]))
            <= PARTIAL_AMOUNT_TOLERANCE_FACTOR * _tolerance(issue))


# ------------------------------------------------------------------ matching
def _kuhn(adj: list[list[int]], n_left: int, n_right: int,
          skip_left: set[int] | None = None,
          taken_right: set[int] | None = None) -> dict[int, int]:
    """Maximum bipartite matching by augmenting paths.

    Left vertices are visited in index order and ``adj[l]`` must already be sorted in the
    preferred right-vertex order. Left vertices in ``skip_left`` and right vertices in
    ``taken_right`` are excluded. Returns {left: right}.
    """
    skip_left = skip_left or set()
    taken_right = taken_right or set()
    match_r: dict[int, int] = {}

    def try_aug(l: int, seen: set[int]) -> bool:
        for r in adj[l]:
            if r in taken_right or r in seen:
                continue
            seen.add(r)
            if r not in match_r or try_aug(match_r[r], seen):
                match_r[r] = l
                return True
        return False

    for l in range(n_left):
        if l in skip_left:
            continue
        try_aug(l, set())
    return {l: r for r, l in match_r.items()}


def _sorted_issues(issues: list[dict]) -> list[dict]:
    return sorted(issues, key=lambda t: t["issue_id"])


def _match_full(findings_norm: list[dict], issues: list[dict], public_ids) -> dict:
    """Core routine; also returns the partial (soft) matching for the experimental reward."""
    iss = _sorted_issues(issues)
    nf, ni = len(findings_norm), len(iss)

    adj: list[list[int]] = []
    best_reason: list[str] = []
    for f in findings_norm:
        edges: list[int] = []
        best = None
        for j, t in enumerate(iss):
            ok, reason = eligible(f, t, public_ids)
            if ok:
                edges.append(j)
            if best is None or _STAGE_RANK[reason] > _STAGE_RANK[best]:
                best = reason
        adj.append(edges)
        best_reason.append(best if best is not None else REASON_UNMATCHED)

    m = _kuhn(adj, nf, ni)                       # {finding index: issue index}
    matched_issue = set(m.values())

    # soft matching on leftovers (type + affected_ids), same deterministic procedure
    padj = [[j for j, t in enumerate(iss) if _partial_edge(f, t, public_ids)]
            for f in findings_norm]
    soft = _kuhn(padj, nf, ni, skip_left=set(m.keys()), taken_right=matched_issue)

    reasons: list[str] = []
    for i in range(nf):
        if i in m:
            reasons.append(REASON_TP)
        elif any(j in matched_issue for j in adj[i]):
            reasons.append(REASON_DUPLICATE)     # eligible, but its issue is already taken
        else:
            reasons.append(best_reason[i])

    pairs = [(i, iss[m[i]]["issue_id"]) for i in sorted(m)]
    tp = len(m)
    return {"pairs": pairs, "tp": tp, "fp": nf - tp, "fn": ni - tp,
            "reasons": reasons, "partial": len(soft),
            "partial_pairs": [(i, iss[soft[i]]["issue_id"]) for i in sorted(soft)]}


def match(findings_norm: list[dict], issues: list[dict], public_ids) -> dict:
    """Deterministic maximum-cardinality 1-1 matching.

    Returns ``{"pairs": [(finding_index, issue_id)], "tp", "fp", "fn", "reasons": [...]}``
    where ``reasons[i]`` is the reason code of finding i (see module docstring).
    """
    r = _match_full(findings_norm, issues, public_ids)
    return {"pairs": r["pairs"], "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
            "reasons": r["reasons"]}


# ------------------------------------------------------------------ metrics/rewards
def prf(tp: int, fp: int, fn: int, n_truth: int, n_findings: int) -> dict:
    """Precision / recall / F1 (rounded to 6 decimals).

    Zero denominators give 0 (P = TP/(TP+FP), R = TP/(TP+FN)); the single special case is the
    both-empty one (clean books, silent agent): F1 = 1 by the task definition while P = R = 0
    because 0/0 is reported as 0."""
    if n_truth == 0 and n_findings == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 1.0, "tp": tp, "fp": fp, "fn": fn}
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(f1, 6), "tp": tp, "fp": fp, "fn": fn}


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def shared_reward(f1: float, inspection_steps: int) -> float:
    """clip(2*F1 - 1 - 0.002*steps, -1, 1), rounded to 6 decimals."""
    return round(_clip(2.0 * f1 - 1.0 - STEP_COST * inspection_steps), 6)


def experimental_reward(tp: int, partial: int, n_truth: int, n_findings: int,
                        inspection_steps: int) -> float:
    """Soft-F1 reward: each partial match (type + affected_ids right, amount/currency/
    evidence wrong) counts as 0.25 TP. Same step cost and clipping as the shared reward."""
    if n_truth == 0 and n_findings == 0:
        soft_f1 = 1.0
    else:
        soft_tp = tp + PARTIAL_CREDIT * partial
        denom = n_truth + n_findings          # == 2*soft_tp + soft_fp + soft_fn
        soft_f1 = (2.0 * soft_tp / denom) if denom else 0.0
    return round(_clip(2.0 * soft_f1 - 1.0 - STEP_COST * inspection_steps), 6)


# ------------------------------------------------------------------ top level
def score(submission: Any, truth: dict, public_ids, inspection_steps: int,
          episode_id: str) -> dict:
    """Grade a submission against ``truth`` (the parsed truth.json).

    Keys: valid, error, tp, fp, fn, n_truth, n_findings, precision, recall, f1,
    reward_shared, reward_experimental, inspection_steps, pairs ([finding_index, issue_id]),
    finding_reasons (reason code per finding).
    """
    issues = list(truth.get("issues", []))
    n_truth = len(issues)
    ok, err, norm = validate_submission(submission, episode_id)
    if not ok or norm is None:
        return {"valid": False, "error": err, "tp": 0, "fp": 0, "fn": n_truth,
                "n_truth": n_truth, "n_findings": 0, "precision": 0.0, "recall": 0.0,
                "f1": 0.0, "reward_shared": -1.0, "reward_experimental": -1.0,
                "inspection_steps": inspection_steps, "pairs": [], "finding_reasons": []}
    findings = norm["findings"]
    r = _match_full(findings, issues, public_ids)
    m = prf(r["tp"], r["fp"], r["fn"], n_truth, len(findings))
    return {
        "valid": True, "error": None,
        "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
        "n_truth": n_truth, "n_findings": len(findings),
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "reward_shared": shared_reward(m["f1"], inspection_steps),
        "reward_experimental": experimental_reward(
            r["tp"], r["partial"], n_truth, len(findings), inspection_steps),
        "inspection_steps": inspection_steps,
        "pairs": [[i, iid] for i, iid in r["pairs"]],
        "finding_reasons": list(r["reasons"]),
    }
