"""`python -m auditgym check [--quick] [--out reports/]` - the ONE official verification command.

Offline, deterministic. Steps (SPEC section 10):
 1. pytest (if importable)             5. 20 paired checks (shape + aggressive rename, permute, replay hash), oracle vs committed truth.json
 2. reproducibility of train/dev       6. leak scan over every recorded observation
 3. validate 20 holdout episodes       7. reward property checks (section 11)
 4. 4 baselines, eval mode, remote     8. JSON + MD report, exit code (0 iff every hard check passes)

Statuses: PASS / FAIL (hard, makes the exit code non-zero) / WARN (diagnostic) / SKIP.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

from . import matcher
from .constants import (ENV_VERSION, INTERACTIONS, ISSUE_TYPES, MAX_STEPS, MODE_EVAL, MODE_TRAIN,
                        STEP_COST)
from .episode import dumps, load_public
from .policies import EXTRA_POLICY_NAMES, POLICY_NAMES
from .runner import (GOLDEN_PATH, ROOT, committed_truth, config_hash, golden_digests, load_splits,
                     replay_actions, report_scores, resolve_config, run_episode, submitted_findings,
                     trajectory_hash)
from .transforms import aggressive_salt

BOOT_SEED = 20260101
BOOT_N = 2000
LEAK_TOKENS = ("iss_", "expected_amount", "evidence_sets", "tolerance_minor", "truth", "provenance",
               "scenario", "clean_records", "injector", "root_cause", "injected", "distractor") + INTERACTIONS


# ================================================================== reporting
class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.sections: dict[str, Any] = {}
        self.out_dir = ""

    def add(self, step: str, name: str, ok: bool | None, detail: str = "", hard: bool = True) -> bool:
        if ok is None:
            status = "SKIP"
        elif ok:
            status = "PASS"
        else:
            status = "FAIL" if hard else "WARN"
        self.checks.append({"step": step, "name": name, "status": status, "hard": hard, "detail": detail})
        print(f"{status:<4}  [{step}] {name}" + (f" - {detail}" if detail else ""), flush=True)
        return bool(ok)

    @property
    def failed(self) -> list[dict]:
        return [c for c in self.checks if c["status"] == "FAIL"]


def guarded(rep: Report, step: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception as e:  # a crashing step is a failing step, never a crashing check
        tb = traceback.format_exc().strip().splitlines()[-1]
        rep.add(step, "step crashed", False, f"{type(e).__name__}: {e} ({tb})")


# ================================================================== helpers
def bootstrap_ci(values: list[float], n: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap CI (stdlib, fixed seed)."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    m = len(values)
    means = sorted(sum(values[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(values) / m, means[int(0.025 * n)], means[min(n - 1, int(0.975 * n))]


def _gen(entry: dict, cache: dict):
    key = (entry["seed"], dumps(resolve_config(entry)))
    if key not in cache:
        from .generator.build import generate_episode
        cache[key] = generate_episode(entry["seed"], resolve_config(entry))
    return cache[key]


def classify(ep) -> str:
    issues = (ep.truth or {}).get("issues", [])
    if not issues:
        return "clean"
    if any(i.get("scenario") for i in issues):
        return "interaction"
    return "mixed"


def perfect_submission(ep) -> dict:
    findings = []
    for t in ep.truth["issues"]:
        findings.append({"type": t["type"], "affected_ids": list(t["affected_ids"]),
                         "claimed_amount_minor": int(t["expected_amount_minor"]),
                         "currency": t["currency"], "evidence_ids": list(t["evidence_sets"][0]),
                         "explanation": "x"})
    return {"episode_id": ep.episode_id, "findings": findings}


def silence(ep) -> dict:
    return {"episode_id": ep.episode_id, "findings": []}


def sc(ep, sub, steps: int = 0) -> dict:
    return matcher.score(sub, ep.truth, ep.public_ids(), steps, ep.episode_id)


def independent_score(ep, run: dict) -> dict:
    """Grader-side recomputation from the recorded submit action (cross-check of the env report)."""
    sub = submitted_findings(run["trajectory"])
    rep = run["grader_report"]
    steps = rep.get("inspection_steps")
    if steps is None:
        steps = max(0, run["steps"] - 1)
    if sub is None:
        sub = silence(ep)
    return sc(ep, sub, steps)


def canon_answer(sub: dict | None) -> list:
    out = []
    for f in (sub or {}).get("findings", []):
        out.append((f["type"], f["claimed_amount_minor"], f["currency"],
                    len(f["affected_ids"]), len(f["evidence_ids"])))
    return sorted(out)


def subprocess_env() -> dict:
    """Environment of every subprocess `check` starts: it must leave the source tree untouched, so no
    bytecode (``__pycache__``) is written; pytest additionally runs with ``-p no:cacheprovider`` (no
    ``.pytest_cache``)."""
    env = dict(os.environ)
    env["AUDITGYM_CHECK_RUNNING"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


PYTEST_ARGS = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]


def run_cmd(args: list[str], timeout: int) -> tuple[int, str]:
    env = subprocess_env()
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    p = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)


# ================================================================== step 1: pytest
def step_pytest(rep: Report) -> None:
    s = "1"
    if os.environ.get("AUDITGYM_CHECK_RUNNING"):
        rep.add(s, "pytest", None, "skipped (invoked from inside check)")
        return
    if importlib.util.find_spec("pytest") is None:
        rep.add(s, "pytest", None, "pytest not installed - skipped (notice: install requirements-dev.txt)")
        return
    tdir = os.path.join(ROOT, "tests")
    has_tests = os.path.isdir(tdir) and any(f.startswith("test_") and f.endswith(".py") for f in os.listdir(tdir))
    if not has_tests:
        rep.add(s, "pytest", None, "no tests found in tests/ - skipped")
        return
    t0 = time.perf_counter()
    rc, out = run_cmd([sys.executable] + PYTEST_ARGS, 1800)
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    rep.add(s, "pytest", rc == 0, f"{tail} ({time.perf_counter() - t0:.1f}s)")
    rep.sections["pytest"] = {"returncode": rc, "tail": out.strip().splitlines()[-15:]}


# ================================================================== step 2: reproducibility
def step_repro(rep: Report, splits: dict, quick: bool, cache: dict) -> None:
    s = "2"
    n_checked = n_committed = 0
    for split in ("train", "dev"):
        entries = splits.get(split) or []
        if not entries:
            rep.add(s, f"{split} entries in splits.json", False, "no entries")
            continue
        if quick:
            entries = entries[:3]
        bad_det, bad_commit, missing = [], [], 0
        for e in entries:
            from .generator.build import generate_episode
            cfg = resolve_config(e)
            a = generate_episode(e["seed"], cfg)
            b = generate_episode(e["seed"], cfg)
            n_checked += 1
            if a.public_hash() != b.public_hash() or dumps(a.truth) != dumps(b.truth) \
                    or dumps(a.provenance) != dumps(b.provenance):
                bad_det.append(e["seed"])
            pub = os.path.join(ROOT, "episodes", split, a.episode_id)
            if os.path.isdir(pub):
                n_committed += 1
                if load_public(pub).public_hash() != a.public_hash():
                    bad_commit.append(a.episode_id)
                tpath = os.path.join(ROOT, "grader_data", split, a.episode_id, "truth.json")
                if os.path.exists(tpath):
                    with open(tpath, encoding="utf-8") as f:
                        if json.load(f) != json.loads(dumps(a.truth)):
                            bad_commit.append(a.episode_id + ":truth")
            else:
                missing += 1
        rep.add(s, f"{split}: regeneration is byte-identical (x2)", not bad_det,
                f"{len(entries)} episodes" + (f"; mismatched seeds {bad_det}" if bad_det else ""))
        if missing == len(entries):
            rep.add(s, f"{split}: matches committed episodes/{split}/", None,
                    "not committed yet (run `python -m auditgym generate`)")
        else:
            rep.add(s, f"{split}: matches committed episodes/{split}/", not bad_commit,
                    f"{len(entries) - missing} compared, {missing} not committed"
                    + (f"; differ: {bad_commit[:5]}" if bad_commit else ""))
    rep.sections["reproducibility"] = {"checked": n_checked, "committed_compared": n_committed}


# ------------------------------------------------------------------ step 2b: config inventory + lock
CONFIG_DIR = os.path.join(ROOT, "configs")
LOCK_FILE = "CONFIG_VERSIONS.json"


def config_inventory() -> dict:
    """``{name: {config_id, config_version, file_sha256, config_hash}}`` for every shipped config
    (configs/*.json except splits.json and the lock file). ``config_hash`` is the runner's resolved-config hash
    that every grader_report / trajectory header carries."""
    import hashlib
    out: dict[str, dict] = {}
    for fn in sorted(os.listdir(CONFIG_DIR)):
        if not fn.endswith(".json") or fn in ("splits.json", LOCK_FILE):
            continue
        path = os.path.join(CONFIG_DIR, fn)
        with open(path, "rb") as f:
            raw = f.read()
        cfg = json.loads(raw.decode("utf-8"))
        name = fn[:-5]
        out[name] = {"config_id": cfg.get("config_id"), "config_version": cfg.get("config_version"),
                     "file_sha256": hashlib.sha256(raw).hexdigest(), "config_hash": config_hash(name)}
    return out


def step_config_lock(rep: Report) -> None:
    """Every shipped config is version-stamped and bound to its content: ``configs/CONFIG_VERSIONS.json`` maps
    ``config_id@config_version`` to the config's content hash, so changing a config without bumping its
    version (or bumping it without re-locking) fails here."""
    from .generator.config import verify_config_lock
    s = "2"
    inv = config_inventory()
    rep.sections["configs"] = inv
    bad = [n for n, v in inv.items() if not v["config_id"] or not v["config_version"] or not v["config_hash"]]
    rep.add(s, "every shipped config has config_id, config_version and a resolvable config_hash", not bad and bool(inv),
            ", ".join(f"{n}@{v['config_version']}" for n, v in inv.items()) if not bad else f"bad: {bad}")
    if not os.path.exists(os.path.join(CONFIG_DIR, LOCK_FILE)):
        rep.add(s, f"configs lock ({LOCK_FILE}) matches the shipped configs", False, "lock file missing")
        return
    problems = verify_config_lock(CONFIG_DIR)
    rep.add(s, f"configs lock ({LOCK_FILE}) binds every config_id@config_version to its content hash "
               "(no content change without a version bump)", not problems,
            "; ".join(problems) if problems else f"{len(inv)} configs locked")


# ------------------------------------------------------------------ step 2c: golden digests
def step_golden(rep: Report) -> None:
    """Cross-platform determinism evidence: digests of generator output and of two deterministic
    trajectories per episode, committed in tests/golden_hashes.json (the same file the test suite reads),
    must equal what THIS machine produces (OS, CPU and Python build independent)."""
    s = "2"
    if not os.path.exists(GOLDEN_PATH):
        rep.add(s, "golden digests (tests/golden_hashes.json) match this machine", False, "file missing")
        return
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        want = json.load(f)
    got = golden_digests()
    diff = []
    for w, g in zip(want.get("episodes", []), got["episodes"]):
        if w != g:
            keys = [k for k in g if w.get(k) != g[k]]
            diff.append(f"{g['split']}:{g['seed']} ({','.join(keys)})")
    if len(want.get("episodes", [])) != len(got["episodes"]):
        diff.append("episode set differs")
    rep.add(s, "golden digests (tests/golden_hashes.json) match this machine (episodes, truth, trajectories)",
            not diff, f"{len(got['episodes'])} episodes, python {platform.python_version()} on {platform.system()}"
            + (f"; differ: {diff[:3]}" if diff else ""))
    rep.sections["golden"] = {"platform": platform.platform(), "python": platform.python_version(),
                              "episodes": len(got["episodes"]), "match": not diff}


# ------------------------------------------------------------------ step 2d: shipped trajectory files
TRAJ_DIR = os.path.join(ROOT, "trajectories")


def _read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def step_trajectory_files(rep: Report) -> None:
    """The shipped trajectories are documented (schemas/trajectory_row.schema.json), oracle rows live in
    their own labelled file, and samples.jsonl holds an eval-mode run and truncated runs (step budget +
    wall clock)."""
    from . import schema_check
    s = "2"
    paths = {n: os.path.join(TRAJ_DIR, n) for n in ("trajectories.jsonl", "oracle_reference.jsonl", "samples.jsonl")}
    missing = [n for n, p in paths.items() if not os.path.exists(p)]
    if missing:
        rep.add(s, "shipped trajectory files present", None, f"missing {missing} (run `python -m auditgym generate`)")
        return
    sch = schema_check.load_schema("trajectory_row.schema.json")
    rows = {n: _read_rows(p) for n, p in paths.items()}
    bad = []
    for n, rs in rows.items():
        for i, r in enumerate(rs):
            es = schema_check.check(r, sch)
            if es:
                bad.append(f"{n}[{i}]: {es[0]}")
                break
    rep.add(s, "every shipped trajectory row conforms to schemas/trajectory_row.schema.json",
            not bad, f"{sum(len(r) for r in rows.values())} rows in 3 files" + (f"; {bad[0]}" if bad else ""))
    main_pol = {r["policy"] for r in rows["trajectories.jsonl"]}
    orc_pol = {r["policy"] for r in rows["oracle_reference.jsonl"]}
    labelled = all(r["header"].get("label") for r in rows["oracle_reference.jsonl"] if r["step"] == 0)
    rep.add(s, "oracle rows are in oracle_reference.jsonl (labelled), not mixed into trajectories.jsonl",
            "oracle" not in main_pol and orc_pol == {"oracle"} and labelled,
            f"trajectories.jsonl policies {sorted(main_pol)}")
    heads = [r["header"] for r in rows["samples.jsonl"] if r["step"] == 0]
    have_trunc = {("time" if h["time_truncated"] else "steps") for h in heads if h["truncated"]}
    have_eval = [h for h in heads if h["mode"] == MODE_EVAL and not h["reward_visible_to_policy"]]
    rep.add(s, "samples.jsonl has an eval-mode run (reward hidden) and truncated=true runs (steps and wall clock)",
            have_trunc == {"steps", "time"} and bool(have_eval),
            f"truncated by {sorted(have_trunc)}, eval-mode samples {len(have_eval)}")
    versions = {(h["env_version"], h["config_id"], h["config_version"]) for r in rows.values()
                for h in (x["header"] for x in r if x["step"] == 0 and "header" in x)}
    rep.add(s, "every trajectory header carries env_version, config_id, config_version and config_hash",
            all(h.get("config_hash") and h.get("env_version") for r in rows.values()
                for h in (x["header"] for x in r if x["step"] == 0 and "header" in x))
            and all(x.get("header") for r in rows.values() for x in r if x["step"] == 0),
            f"{len(versions)} distinct (env, config, version) combinations")


# ================================================================== step 3: holdout validation
def select_holdout(rep: Report, splits: dict, quick: bool, cache: dict):
    s = "3"
    entries = splits.get("holdout") or []
    rep.add(s, "holdout has 20 episodes in splits.json", len(entries) == 20, f"found {len(entries)}")
    from .validate import validate_episode
    eps, classes, failures = [], {"clean": 0, "interaction": 0, "mixed": 0}, []
    for e in entries:
        ep = _gen(e, cache)
        res = validate_episode(ep)
        kind = classify(ep)
        classes[kind] += 1
        eps.append((e, ep, kind))
        if not res["ok"]:
            bad = [f"{k}: {v['detail']}" for k, v in res["checks"].items() if not v["ok"]]
            failures.append(f"{ep.episode_id}: {'; '.join(bad)[:200]}")
    rep.add(s, "all holdout episodes pass validate.py (identities, rules==truth, evidence, leak, counts)",
            not failures and bool(entries), f"{len(entries) - len(failures)}/{len(entries)} valid"
            + (f"; first failure {failures[0]}" if failures else ""))
    from .schema_check import check_episode
    schema_bad = [f"{ep.episode_id}: {errs[0]}" for _e, ep, _k in eps for errs in [check_episode(ep)] if errs]
    rep.add(s, "every holdout episode (public files, truth, provenance) conforms to schemas/*.json",
            not schema_bad and bool(eps), f"{len(eps) - len(schema_bad)}/{len(eps)}"
            + (f"; first {schema_bad[0]}" if schema_bad else ""))
    rep.add(s, "holdout composition 5 clean / 5 interaction / 10 mixed", 
            classes == {"clean": 5, "interaction": 5, "mixed": 10}, f"found {classes}")
    declared_bad = [e["seed"] for e, _ep, kind in eps if e.get("category") and e["category"] != kind]
    rep.add(s, "declared splits.json category matches the category measured from truth", not declared_bad,
            "all consistent" if not declared_bad else f"mismatched seeds {declared_bad}")
    rep.sections["holdout"] = {"classes": classes, "n": len(entries), "validation_failures": failures}
    if quick:
        picked, per = [], {"clean": 0, "interaction": 0, "mixed": 0}
        for t in eps:
            if per[t[2]] < 2:
                per[t[2]] += 1
                picked.append(t)
        eps = picked
    return eps


def _pooled(sc_list: list[dict], which: str) -> float:
    """Micro-averaged precision / recall over all episodes of a run (sum of TP, FP, FN). Per-episode
    P and R are 0 on an empty/empty (clean, silent) episode, so a mean over episodes would be misleading."""
    tp = sum(x["tp"] for x in sc_list)
    other = sum(x["fp"] if which == "precision" else x["fn"] for x in sc_list)
    return tp / (tp + other) if (tp + other) else 0.0


# ================================================================== step 4: baselines
def step_baselines(rep: Report, eps: list, remote: bool, log_root: str | None = None) -> dict:
    s = "4"
    all_policies = list(POLICY_NAMES) + list(EXTRA_POLICY_NAMES)   # extras = diagnostic rows, not baselines
    runs: dict[str, list[dict]] = {p: [] for p in all_policies}
    crashes, extra_crashes = [], []
    for p in all_policies:
        for e, ep, kind in eps:
            try:
                r = run_episode(p, e["seed"], resolve_config(e), mode=MODE_EVAL, remote=remote, policy_seed=0,
                                log_dir=os.path.join(log_root, p) if (log_root and remote) else None)
                r["kind"] = kind
                runs[p].append(r)
            except Exception as ex:
                (extra_crashes if p in EXTRA_POLICY_NAMES else crashes).append(
                    f"{p}/{e['seed']}: {type(ex).__name__}: {ex}")
    rep.add(s, "every (policy, episode) run completed", not crashes,
            f"{sum(len(runs[p]) for p in POLICY_NAMES)} runs" + (f"; {crashes[:3]}" if crashes else ""))
    if EXTRA_POLICY_NAMES:
        rep.add(s, "extra diagnostic policies completed (not one of the 4 required baselines)",
                not extra_crashes,
                f"{sum(len(runs[p]) for p in EXTRA_POLICY_NAMES)} runs"
                + (f"; {extra_crashes[:3]}" if extra_crashes else ""), hard=False)

    rows, table = {}, []
    for p in all_policies:
        sc_list = []
        for r, (e, ep, kind) in zip(runs[p], eps):
            s_ = report_scores(r["grader_report"])
            s_["steps"], s_["wall_s"], s_["kind"] = r["steps"], r["wall_s"], kind
            s_["episode_id"], s_["seed"] = r["episode_id"], e["seed"]
            sc_list.append(s_)
        rows[p] = sc_list
        if not sc_list:
            continue
        def col(k):
            return [float(x[k] if x[k] is not None else 0.0) for x in sc_list]
        shared = bootstrap_ci(col("shared"))
        f1 = bootstrap_ci(col("f1"))
        table.append({"policy": f"{p} (extra)" if p in EXTRA_POLICY_NAMES else p, "n": len(sc_list),
                      "precision": _pooled(sc_list, "precision"),
                      "recall": _pooled(sc_list, "recall"),
                      "f1": f1, "shared": shared,
                      "experimental": sum(col("experimental")) / len(sc_list),
                      "steps": sum(col("steps")) / len(sc_list),
                      "wall_s": sum(col("wall_s")) / len(sc_list)})
    rep.sections["baselines"] = {"table": table, "rows": rows}
    rep.sections["failure_analysis"] = failure_analysis(eps, runs)
    print("\n  policy                    n   P(pool) R(pool) F1 [95% CI]            shared [95% CI]          exp.    steps  wall_s")
    for t in table:
        print(f"  {t['policy']:<24} {t['n']:>2}  {t['precision']:.3f}  {t['recall']:.3f}  "
              f"{t['f1'][0]:.3f} [{t['f1'][1]:.3f},{t['f1'][2]:.3f}]  "
              f"{t['shared'][0]:+.3f} [{t['shared'][1]:+.3f},{t['shared'][2]:+.3f}]  "
              f"{t['experimental']:+.3f}  {t['steps']:5.1f}  {t['wall_s']:.2f}")
    print()

    allruns = [r for v in runs.values() for r in v]
    ended = all(r["grader_report"].get("ended") for r in allruns)
    rep.add(s, "all episodes ended with a grader report", ended and bool(allruns))
    rep.add(s, f"all runs within {MAX_STEPS} steps", all(r["steps"] <= MAX_STEPS for r in allruns))
    if log_root and remote:
        n_logs = sum(len([f for f in os.listdir(os.path.join(log_root, p)) if f.startswith("report_")])
                     for p in all_policies if os.path.isdir(os.path.join(log_root, p)))
        rep.add(s, "the grader wrote one per-episode log file for every run (--log-dir, never sent to the agent)",
                n_logs == len(allruns), f"{n_logs} log files for {len(allruns)} runs")
    inv = [r["episode_id"] for r in allruns if r["grader_report"].get("invalid_submission")]
    rep.add(s, "no baseline submitted a malformed submission", not inv, f"{len(inv)} invalid" if inv else "")
    rec = runs["reconcile"]
    rep.add(s, "reconcile obeys the budget (never truncated, submits)",
            bool(rec) and all(r["grader_report"].get("ended_by") == "submit" for r in rec))

    # independent recomputation with matcher.score from the recorded submit action
    disagree = []
    for p in all_policies:
        for r, (e, ep, kind) in zip(runs[p], eps):
            ind = independent_score(ep, r)
            got = r["grader_report"].get("reward_shared")
            if got is None or abs(got - ind["reward_shared"]) > 1e-9:
                disagree.append(f"{p}/{ep.episode_id}")
    rep.add(s, "env grader_report == independent matcher.score(recorded submission)", not disagree,
            f"{len(disagree)} disagreements: {disagree[:3]}" if disagree else "all runs agree")

    orc = rows["oracle"]
    rep.add(s, "oracle: F1 = 1.0 and shared reward = +1.0 on every episode (0 inspection steps)",
            bool(orc) and all(abs(x["f1"] - 1.0) < 1e-9 and abs((x["shared"] or 0) - 1.0) < 1e-9 for x in orc),
            f"mean shared {sum(x['shared'] or 0 for x in orc) / max(1, len(orc)):+.3f}")
    fa = rows["flag_all"]
    fa_mean = sum(x["shared"] or 0 for x in fa) / max(1, len(fa))
    rep.add(s, "flag_all mean shared reward < 0", bool(fa) and fa_mean < 0, f"{fa_mean:+.3f}")
    means = {t["policy"]: t["shared"][0] for t in table}
    rep.add(s, "ordering oracle > reconcile > random_explorer, flag_all (mean shared)",
            means.get("oracle", -9) > means.get("reconcile", 9) > max(means.get("random_explorer", 9),
                                                                      means.get("flag_all", 9)),
            ", ".join(f"{k} {v:+.3f}" for k, v in means.items()), hard=False)
    deg = [t for t in table if t["policy"].startswith("careful_auditor_degraded")]
    car = [t for t in table if t["policy"].startswith("careful_auditor") and "degraded" not in t["policy"]]
    if deg and car:
        rep.add(s, "careful_auditor_degraded is a measurable near miss (0 < mean F1 < careful_auditor's; reward below it)",
                0 < deg[0]["f1"][0] < car[0]["f1"][0] and deg[0]["shared"][0] < car[0]["shared"][0],
                f"degraded F1 {deg[0]['f1'][0]:.3f} / shared {deg[0]['shared'][0]:+.3f} vs careful_auditor "
                f"{car[0]['f1'][0]:.3f} / {car[0]['shared'][0]:+.3f}", hard=False)
    rmean = [t for t in table if t["policy"] == "reconcile"]
    rep.add(s, "reconcile earns partial credit (0 < mean F1 < 1)",
            bool(rmean) and 0 < rmean[0]["f1"][0] < 1, f"mean F1 {rmean[0]['f1'][0]:.3f}" if rmean else "", hard=False)
    return runs


# ================================================================== failure analysis (measured)
def failure_analysis(eps: list, runs: dict) -> dict:
    """Per policy, from the grader reports of the eval runs: recall by issue type / interaction,
    findings by type and the matcher reason histogram of every false positive, and which distractor
    kinds the false positives landed on. Everything is measured, nothing is hand-written."""
    out: dict[str, Any] = {}
    for pol, rs in runs.items():
        types = {t: {"truth": 0, "tp": 0} for t in ISSUE_TYPES}
        scen = {sc_: {"truth": 0, "tp": 0} for sc_ in INTERACTIONS}
        found = {t: {"findings": 0, "tp": 0} for t in ISSUE_TYPES}
        reasons: dict[str, int] = {}
        distr: dict[str, int] = {}
        fp_total = 0
        n_steps = 0
        shared_sum = 0.0
        for r, (e, ep, kind) in zip(rs, eps):
            rep_ = r["grader_report"]
            n_steps += rep_.get("inspection_steps") if rep_.get("inspection_steps") is not None else r["steps"]
            shared_sum += float(rep_.get("reward_shared") or 0.0)
            score_ = rep_.get("score") or {}
            by_id = {i["issue_id"]: i for i in ep.truth["issues"]}
            matched = {iid for _fi, iid in score_.get("pairs", [])}
            for iid, iss in by_id.items():
                types[iss["type"]]["truth"] += 1
                types[iss["type"]]["tp"] += iid in matched
                if iss.get("scenario"):
                    scen[iss["scenario"]]["truth"] += 1
                    scen[iss["scenario"]]["tp"] += iid in matched
            sub = submitted_findings(r["trajectory"])
            fnds = (sub or {}).get("findings", []) if isinstance(sub, dict) else []
            reas = score_.get("finding_reasons", [])
            tp_idx = {fi for fi, _ in score_.get("pairs", [])}
            dmap: dict[str, str] = {}
            for d in (ep.provenance or {}).get("distractors", []):
                for rid in d.get("record_ids", []):
                    dmap.setdefault(rid, d["kind"])
            for fi, f in enumerate(fnds):
                if not isinstance(f, dict) or f.get("type") not in found:
                    continue
                found[f["type"]]["findings"] += 1
                if fi in tp_idx:
                    found[f["type"]]["tp"] += 1
                    continue
                fp_total += 1
                reason = reas[fi] if fi < len(reas) else "unmatched"
                reasons[reason] = reasons.get(reason, 0) + 1
                hit = {dmap[a] for a in f.get("affected_ids", []) if a in dmap}
                for k in (sorted(hit) or ["(not a distractor)"]):
                    distr[k] = distr.get(k, 0) + 1
        n_runs = max(1, len(rs))
        out[pol] = {"recall_by_type": types, "recall_by_interaction": scen, "precision_by_type": found,
                    "fp_total": fp_total, "fp_reasons": dict(sorted(reasons.items())),
                    "fp_by_distractor_kind": dict(sorted(distr.items())),
                    "misses_by_type": {t: v["truth"] - v["tp"] for t, v in types.items()},
                    "fp_by_type": {t: v["findings"] - v["tp"] for t, v in found.items()},
                    "mean_inspection_steps": n_steps / n_runs,
                    "mean_step_cost": n_steps / n_runs * STEP_COST,
                    "mean_shared_reward": shared_sum / n_runs, "n_runs": len(rs)}
    return out

# ================================================================== step 5: paired checks
AGGR_ID = re.compile(r"\d{9,14}|z[0-9a-f]{8,20}")           # the only id shapes the aggressive rename emits
PAIRED_POLICIES = ("reconcile", "careful_auditor")           # hard: honest, non-saturated-at-the-floor policies
PAIRED_SOFT: tuple[str, ...] = ()                            # (none: see the note below)
# careful_auditor_degraded is NOT paired: its step cap makes the partial ledger it sees depend on row order, so
# it is order-dependent by construction (that is part of what "ran out of budget" means).
# random_explorer / flag_all are NOT paired: they read the first pages / follow row order by design and sit at
# the -1.0 floor on almost every episode, so "identical score" would be trivially true and prove nothing.


def _variant_options(variant: str, i: int) -> dict:
    if variant == "aggressive":
        return {"transform": {"rename_salt": aggressive_salt(3000 + i), "permute_salt": 4000 + i}}
    return {"transform": {"rename_salt": 3000 + i, "permute_salt": 4000 + i}}


def step_paired(rep: Report, eps: list, runs: dict, remote: bool, n_pairs: int) -> list[dict]:
    """Pair every holdout episode with its transformed twin. ``shape`` keeps id formats; ``aggressive``
    replaces every opaque id by a different prefix / length / charset / sort order (fx:* keys are natural
    keys and stay). Rewards are read from ``grader_report.reward_shared`` (eval mode hides the trajectory
    reward by design)."""
    s = "5"
    extra_runs: list[dict] = []
    pairs = list(zip(range(n_pairs), eps))[:n_pairs]
    detail: list[dict] = []
    oracle_fail: dict[str, list] = {"shape": [], "aggressive": []}
    inv_fail: dict[str, dict[str, list]] = {p: {"shape": [], "aggressive": []} for p in PAIRED_POLICIES + PAIRED_SOFT}
    rn_cases: dict[str, list] = {"shape": [], "aggressive": []}
    aggr_bad: list[str] = []
    rp_fail: list = []
    for i, (e, ep, kind) in pairs:
        cfg = resolve_config(e)
        for variant in ("shape", "aggressive"):
            opts = _variant_options(variant, i)
            # a real (non-oracle) policy must also be invariant to ids and order: same metrics, same reward
            for pol in PAIRED_POLICIES + PAIRED_SOFT:
                if not runs.get(pol) or len(runs[pol]) <= i:
                    continue
                try:
                    tr_ = run_episode(pol, e["seed"], cfg, mode=MODE_EVAL, options=opts, remote=remote)
                    extra_runs.append(tr_)
                    b_, t_ = runs[pol][i]["grader_report"], tr_["grader_report"]
                    ok_ = (b_["metrics"], b_["reward_shared"], b_["inspection_steps"]) == \
                          (t_["metrics"], t_["reward_shared"], t_["inspection_steps"])
                    detail.append({"episode_id": ep.episode_id, "kind": kind, "policy": pol, "variant": variant,
                                   "f1": b_["metrics"]["f1"], "f1_transformed": t_["metrics"]["f1"],
                                   "shared": b_["reward_shared"], "shared_transformed": t_["reward_shared"],
                                   "steps": b_["inspection_steps"], "steps_transformed": t_["inspection_steps"],
                                   "identical": ok_})
                    if not ok_:
                        inv_fail[pol][variant].append(ep.episode_id)
                except Exception as ex:
                    inv_fail[pol][variant].append(f"{ep.episode_id}: {type(ex).__name__}: {ex}")
            base_o = runs["oracle"][i]
            try:
                tr = run_episode("oracle", e["seed"], cfg, mode=MODE_EVAL, options=opts, remote=remote)
                extra_runs.append(tr)
                b, t = base_o["grader_report"], tr["grader_report"]
                sb, st = submitted_findings(base_o["trajectory"]), submitted_findings(tr["trajectory"])
                ids_b = {x for f in sb["findings"] for x in f["affected_ids"] + f["evidence_ids"]}
                ids_t = {x for f in st["findings"] for x in f["affected_ids"] + f["evidence_ids"]}
                ok = (b["reward_shared"] == t["reward_shared"] and abs(t["metrics"]["f1"] - 1.0) < 1e-9
                      and canon_answer(sb) == canon_answer(st) and b["metrics"] == t["metrics"])
                renamed = (not ids_b) or bool(ids_b - ids_t)
                rn_cases[variant].append({"episode_id": ep.episode_id, "ok": ok, "ids_renamed": renamed})
                if variant == "aggressive":
                    cited = {x for x in ids_t if not x.startswith("fx:")}
                    if any(not AGGR_ID.fullmatch(x) or x in ids_b for x in cited):
                        aggr_bad.append(ep.episode_id)
                if not ok:
                    oracle_fail[variant].append(ep.episode_id)
            except Exception as ex:
                oracle_fail[variant].append(f"{ep.episode_id}: {type(ex).__name__}: {ex}")
        try:
            r0 = runs["reconcile"][i]
            actions = [x["action"] for x in r0["trajectory"][1:]]
            rr = replay_actions(actions, e["seed"], cfg, mode=MODE_EVAL, remote=remote)
            if trajectory_hash(r0["trajectory"]) != trajectory_hash(rr["trajectory"]):
                rp_fail.append(ep.episode_id)
        except Exception as ex:
            rp_fail.append(f"{ep.episode_id}: {type(ex).__name__}: {ex}")
    n = len(pairs)
    for variant, label in (("shape", "rename (shape-preserving ids) + permute"),
                           ("aggressive", "AGGRESSIVE rename (other prefix/length/charset/sort order) + permute")):
        bad = oracle_fail[variant]
        rep.add(s, f"{label}: oracle reward identical and answer maps ({n} pairs)", not bad and n > 0,
                f"{n - len(bad)}/{n}" + (f"; failures {bad[:3]}" if bad else ""))
    for pol in PAIRED_POLICIES + PAIRED_SOFT:
        soft = pol in PAIRED_SOFT
        for variant, label in (("shape", "rename+permute"), ("aggressive", "aggressive rename+permute")):
            bad = inv_fail[pol][variant]
            rep.add(s, f"{label}: {pol} (a non-oracle policy) gets identical metrics, reward and steps ({n} pairs)",
                    not bad and n > 0, f"{n - len(bad)}/{n}" + (f"; failures {bad[:3]}" if bad else ""), hard=not soft)
    not_renamed = [c["episode_id"] for v in rn_cases.values() for c in v if not c["ids_renamed"]]
    rep.add(s, "rename actually changed the ids the oracle cites", not not_renamed,
            f"{len(not_renamed)} episodes unchanged" if not_renamed else "ids changed on every episode", hard=False)
    rep.add(s, "aggressive rename: every cited opaque id is numeric-only or z+hex and differs from the original "
               "(fx:* natural keys are intentionally kept)", not aggr_bad and n > 0,
            f"{n - len(aggr_bad)}/{n}" + (f"; failures {aggr_bad[:3]}" if aggr_bad else ""))
    # the pairs must discriminate: a policy sitting at the -1.0 floor is trivially "invariant"
    disc = {}
    for pol in PAIRED_POLICIES:
        rows = [d for d in detail if d["policy"] == pol and d["variant"] == "shape"]
        disc[pol] = sum(1 for d in rows if d["shared"] > -1.0 + 1e-9)
    need = max(1, n // 5)
    rep.add(s, f"paired checks are discriminating: reconcile and careful_auditor score above the -1.0 floor on "
               f">= {need} of {n} pairs", n > 0 and all(v >= need for v in disc.values()),
            ", ".join(f"{k}: {v}/{n} above floor" for k, v in disc.items()))
    rep.add(s, f"replay of the same seed + actions gives an identical trajectory sha256 ({n} pairs)",
            not rp_fail and n > 0, f"{n - len(rp_fail)}/{n}" + (f"; failures {rp_fail[:3]}" if rp_fail else ""))
    per_pair = {}
    for pol in PAIRED_POLICIES + PAIRED_SOFT:
        rows = [d for d in detail if d["policy"] == pol and d["variant"] == "shape"]
        per_pair[pol] = [{"episode_id": d["episode_id"], "kind": d["kind"], "f1": d["f1"],
                          "shared": d["shared"], "steps": d["steps"]} for d in rows]
    rep.sections["paired"] = {
        "pairs": n, "rename_permute_failures": oracle_fail["shape"], "aggressive_failures": oracle_fail["aggressive"],
        "replay_failures": rp_fail, "discriminating_pairs": disc, "per_pair_f1": per_pair, "detail": detail,
        "note": "random_explorer / flag_all are excluded: trivially invariant at the -1.0 floor"}
    return extra_runs


def step_oracle_truth(rep: Report, splits: dict, remote: bool, quick: bool) -> None:
    """The oracle's truth comes from ``runner.grader_truth`` = the same generator the env runs, so on its own
    "oracle F1 = 1.0" is partly self-consistent. For the OPEN splits the shipped ``grader_data/<split>/<id>/
    truth.json`` is an independent artifact: (a) the oracle's submission must equal the one derived from the
    committed file, (b) scoring it against the committed truth and committed public ids gives F1 = 1, and
    (c) a non-oracle policy's env report must equal an independent score against the committed truth."""
    s = "5"
    n = a_bad = b_bad = c_bad = missing = 0
    for split in ("train", "dev"):
        entries = (splits.get(split) or [])[: 3 if quick else None]
        for e in entries:
            cfg = resolve_config(e)
            from .generator.build import generate_episode
            eid = generate_episode(e["seed"], cfg).episode_id
            truth = committed_truth(split, eid)
            pub_dir = os.path.join(ROOT, "episodes", split, eid)
            if truth is None or not os.path.isdir(pub_dir):
                missing += 1
                continue
            n += 1
            pub = load_public(pub_dir)
            ids = pub.public_ids()
            orc = run_episode("oracle", e["seed"], cfg, mode=MODE_EVAL, remote=remote)
            sub = submitted_findings(orc["trajectory"]) or {"findings": []}
            want = sorted((t["type"], tuple(t["affected_ids"]), int(t["expected_amount_minor"]), t["currency"])
                          for t in truth["issues"])
            got = sorted((f["type"], tuple(f["affected_ids"]), f["claimed_amount_minor"], f["currency"])
                         for f in sub["findings"])
            a_bad += want != got
            res = matcher.score(sub, truth, ids, 0, eid)
            b_bad += abs(res["f1"] - 1.0) > 1e-9
            car = run_episode("careful_auditor", e["seed"], cfg, mode=MODE_EVAL, remote=remote)
            csub = submitted_findings(car["trajectory"]) or {"episode_id": eid, "findings": []}
            steps = car["grader_report"].get("inspection_steps", max(0, car["steps"] - 1))
            ind = matcher.score(csub, truth, ids, steps, eid)
            c_bad += abs(ind["reward_shared"] - car["grader_report"]["reward_shared"]) > 1e-9
    rep.add(s, "oracle submission == the one derived from the committed grader_data truth.json (open splits)",
            n > 0 and not a_bad, f"{n - a_bad}/{n} episodes" + (f"; {missing} without committed truth" if missing else ""))
    rep.add(s, "oracle scored against the COMMITTED truth.json + committed public ids: F1 = 1.0",
            n > 0 and not b_bad, f"{n - b_bad}/{n}")
    rep.add(s, "careful_auditor env reward == independent score against the committed truth.json",
            n > 0 and not c_bad, f"{n - c_bad}/{n}")


# ================================================================== step 6: leak scan
def _walk_keys(x: Any):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk_keys(v)


def step_leaks(rep: Report, all_runs: list[dict]) -> None:
    s = "6"
    hits: list[str] = []
    n_obs = 0
    schema_bad = 0
    for r in all_runs:
        seed = r.get("seed")
        for rec in r["trajectory"]:
            obs = rec["observation"]
            n_obs += 1
            text = json.dumps(obs, ensure_ascii=False).lower()
            for tok in LEAK_TOKENS:
                if tok.lower() in text:
                    hits.append(f"{r.get('policy')}/{r['episode_id']}/step{rec['step']}: token {tok!r}")
            if "seed" in {str(k).lower() for k in _walk_keys(obs)}:
                hits.append(f"{r['episode_id']}/step{rec['step']}: key 'seed'")
            expect = {"episode_id", "step", "steps_remaining", "result", "error"}
            if rec["step"] == 0:
                expect |= {"task", "manifest", "policy"}
            if set(obs) != expect:
                schema_bad += 1
            if rec["step"] == 0 and isinstance(seed, int):
                head = json.dumps({k: obs.get(k) for k in ("task", "manifest", "policy", "episode_id")})
                if re.search(rf"(?<!\d){seed}(?!\d)", head):
                    hits.append(f"{r['episode_id']}: seed value in first observation")
            if rec["step"] > 0 and obs.get("error") and isinstance(seed, int) \
                    and re.search(rf"(?<!\d){seed}(?!\d)", json.dumps(obs["error"])):
                hits.append(f"{r['episode_id']}/step{rec['step']}: seed in error")
    rep.add(s, f"no hidden-data tokens in {n_obs} recorded observations", not hits,
            f"{len(hits)} hits, e.g. {hits[:3]}" if hits else f"scanned {len(all_runs)} runs for {len(LEAK_TOKENS)} tokens")
    rep.add(s, "observation schema is exactly {episode_id, step, steps_remaining, result, error} (+task/manifest/policy at reset)",
            schema_bad == 0, f"{schema_bad} deviating observations" if schema_bad else "")


def step_isolation(rep: Report, eps: list, out_dir: str) -> None:
    """Trust-boundary checks: seed not recoverable from the manifest, harness-only commands, hostile
    payloads, per-episode grader log for an invalid submission."""
    import tempfile
    from .client import RemoteAuditEnv, RemoteEnvError
    from .content_id import content_episode_id
    s = "6"

    # (a) episode_id is content-addressed: a pure function of the public records, so it adds no
    #     information about the seed beyond the records themselves. (The generator is open source; seed
    #     secrecy rests on the seed space and on the agent having no code execution, see README threat model.)
    bad_hash, seed_in_id = [], []
    for e, ep, _k in eps:
        if content_episode_id(ep) != ep.manifest["episode_id"]:
            bad_hash.append(ep.episode_id)
        sd, hx = str(e["seed"]), ep.episode_id[3:]
        # a short decimal string occurs in a random hex id by chance, so: the whole id is the seed, or a
        # seed of >= 6 digits appears inside it
        if (hx.isdigit() and int(hx) == e["seed"]) or (len(sd) >= 6 and sd in hx):
            seed_in_id.append(ep.episode_id)
    rep.add(s, "episode_id equals the content hash of the public records (carries no extra information "
               "about the seed)", not bad_hash, f"{len(bad_hash)} mismatched" if bad_hash else f"{len(eps)} episodes")
    rep.add(s, "episode_id is not / does not embed the decimal seed (seeds of >= 6 digits searched as substring)", not seed_in_id,
            f"{len(seed_in_id)} found" if seed_in_id else f"{len(eps)} episodes")

    # (b) harness-only commands + hostile payloads + invalid-submission log
    logdir = tempfile.mkdtemp(prefix="auditgym_logs_")
    try:
        env = RemoteAuditEnv(mode=MODE_EVAL, log_dir=logdir, allow_grader_report=True)
        try:
            e0 = eps[0][0]
            env.reset(seed=e0["seed"], config=resolve_config(e0))
            refused = []
            for cmd in ({"cmd": "reset", "seed": 1, "config": "train"}, {"cmd": "grader_report"}):
                try:
                    env._call(cmd)
                    refused.append(False)
                except RemoteEnvError:
                    refused.append(True)
            rep.add(s, "reset and grader_report are harness-only: refused without the control token", all(refused),
                    f"refused={refused}")
            deep: Any = {}
            cur = deep
            for _ in range(800):
                cur["x"] = {}
                cur = cur["x"]
            o, rw, term, trunc, info = env.step({"action": "calculator", "args": {"expression": "1"}, "x": deep})
            o2, *_ = env.step({"action": "calculator", "args": {"expression": "1+1"}})
            ok = (o["error"] and o["error"]["code"] == "invalid_args" and rw == 0.0 and not term
                  and o2["result"] == {"value": "2"} and o2["step"] == 2)
            rep.add(s, "a deeply nested action gives a bounded invalid_args error, consumes one step, keeps the run alive",
                    bool(ok), f"code={(o['error'] or {}).get('code')}, next step={o2.get('step')}")
            o, rw, term, trunc, info = env.step({"action": "submit", "args": {}})
            rpt = env.grader_report()
            rep.add(s, "submit without args.findings: ends the episode; eval shows no error (validity is not leaked), "
                       "grader log has reward -1 and the reason",
                    bool(term and o["error"] is None and rw == 0.0 and rpt["reward_shared"] == -1.0
                         and rpt["errors"]), f"terminated={term}, grader reward {rpt['reward_shared']}")
            env.reset(seed=e0["seed"], config=resolve_config(e0))     # mid-episode re-reset abandons the first
            env.step({"action": "calculator", "args": {"expression": "1"}})
            env.reset(seed=e0["seed"], config=resolve_config(e0))
        finally:
            env.close()
        logs = sorted(f for f in os.listdir(logdir) if f.startswith("report_"))
        texts = []
        for f in logs:
            with open(os.path.join(logdir, f), encoding="utf-8") as fh:
                texts.append(json.load(fh))
        has_err = any(t.get("errors") and t.get("invalid_submission") for t in texts)
        abandoned = [t for t in texts if t.get("ended_by") == "abandoned"]
        rep.add(s, "grader log file written for the invalid submission and for an episode abandoned by re-reset",
                has_err and len(abandoned) == 1, f"{len(logs)} logs, invalid-with-error={has_err}, abandoned={len(abandoned)}")
    finally:
        shutil.rmtree(logdir, ignore_errors=True)

# ================================================================== step 7: reward properties
def _env(mode: str = MODE_TRAIN, **kw):
    from .env import AuditEnv
    return AuditEnv(mode=mode, **kw)


def step_rewards(rep: Report, eps: list, cache: dict) -> None:
    s = "7"
    issue_t = next(((e, ep) for e, ep, k in eps if ep.truth["issues"]), None)
    multi_t = next(((e, ep) for e, ep, k in eps if len(ep.truth["issues"]) >= 2), issue_t)
    clean_t = next(((e, ep) for e, ep, k in eps if not ep.truth["issues"]), None)
    if not issue_t or not clean_t:
        rep.add(s, "need one clean and one issue episode", False, "missing in holdout selection")
        return
    (ie, iep), (ce, cep) = issue_t, clean_t
    perfect, sil = perfect_submission(iep), silence(iep)
    P = lambda sub, ep=iep, steps=0: sc(ep, sub, steps)
    R = lambda sub, ep=iep, steps=0: sc(ep, sub, steps)["reward_shared"]

    def prop(name: str, fn: Callable[[], tuple[bool, str]]) -> None:
        try:
            ok, detail = fn()
        except Exception as ex:
            ok, detail = False, f"crashed: {type(ex).__name__}: {ex}"
        rep.add(s, name, ok, detail)

    n_iss = len(iep.truth["issues"])
    prop("complete correct answer with evidence > silence on books with issues",
         lambda: (R(perfect) > R(sil), f"{R(perfect):+.3f} vs {R(sil):+.3f} ({n_iss} issues)"))

    false_acc = {"episode_id": cep.episode_id, "findings": [{
        "type": "fx_error", "affected_ids": [cep.invoices[0]["invoice_id"]],
        "claimed_amount_minor": 12345, "currency": cep.manifest["base_currency"],
        "evidence_ids": [cep.invoices[0]["doc_id"]], "explanation": "guess"}]} if cep.invoices else silence(cep)
    prop("silence > false accusations on clean books",
         lambda: (R(silence(cep), cep) > R(false_acc, cep), f"{R(silence(cep), cep):+.3f} vs {R(false_acc, cep):+.3f}"))

    def dup():
        one = perfect["findings"][0]
        a = R({"episode_id": iep.episode_id, "findings": [one]})
        b = R({"episode_id": iep.episode_id, "findings": [one, dict(one)]})
        pa = P({"episode_id": iep.episode_id, "findings": [one, dict(one)]})
        return b <= a and pa["tp"] == 1 and pa["fp"] == 1, f"single {a:+.3f}, duplicated {b:+.3f}, tp={pa['tp']} fp={pa['fp']}"
    prop("duplicating a correct finding gives no gain (second copy is FP)", dup)

    def reorder():
        f = perfect["findings"]
        g = []
        for x in reversed(f):
            y = dict(x, affected_ids=list(reversed(x["affected_ids"])), evidence_ids=list(reversed(x["evidence_ids"])),
                     explanation="completely different words")
            g.append(y)
        a, b = P(perfect), P({"episode_id": iep.episode_id, "findings": g})
        return (a["tp"], a["fp"], a["fn"], a["reward_shared"]) == (b["tp"], b["fp"], b["fn"], b["reward_shared"]), \
            "order, id order and explanation text do not change the score"
    prop("score ignores explanation text and the order of findings/ids", reorder)

    def step_cost():
        r0, r10 = R(perfect, steps=0), R(perfect, steps=10)
        per = (r0 - r10) / 10
        return abs(per - STEP_COST) < 1e-9 and per <= 0.002 + 1e-12, f"{per:.6f} per step"
    prop("step cost is tiny: <= 0.002 per inspection step", step_cost)

    def step_cost_env():
        env = _env()
        env.reset(seed=ie["seed"], config=resolve_config(ie))
        for _ in range(5):
            env.step({"action": "list_records", "args": {"table": "fx", "limit": 1}})
        _, rew, term, _, info = env.step({"action": "submit", "args": {"findings": perfect}})
        exp = matcher.shared_reward(1.0, 5)
        return term and abs(rew - exp) < 1e-9 and info["inspection_steps"] == 5, f"reward {rew:+.4f} (expected {exp:+.4f})"
    prop("env: perfect answer after 5 inspection steps earns 1 - 5*0.002", step_cost_env)

    def unknown_ev():
        f = dict(perfect["findings"][0]); f["evidence_ids"] = list(f["evidence_ids"])[:9] + ["D-ffffffffff"]
        r = P({"episode_id": iep.episode_id, "findings": [f]})
        return r["finding_reasons"][0] == matcher.REASON_EVIDENCE_UNKNOWN and r["tp"] == 0, r["finding_reasons"][0]
    prop("unknown/non-existent evidence id invalidates the finding", unknown_ev)

    def guessed():
        f = dict(perfect["findings"][0]); f["affected_ids"] = ["INV-GUESS-0001"]
        r = P({"episode_id": iep.episode_id, "findings": [f]})
        return r["tp"] == 0 and r["fp"] == 1, f"reason {r['finding_reasons'][0]}"
    prop("guessed affected ids are false positives", guessed)

    def wrong_cur():
        f = dict(perfect["findings"][0]); f["currency"] = "JPY" if f["currency"] != "JPY" else "USD"
        r = P({"episode_id": iep.episode_id, "findings": [f]})
        return r["tp"] == 0 and r["finding_reasons"][0] == matcher.REASON_CURRENCY_MISMATCH, r["finding_reasons"][0]
    prop("wrong currency invalidates the finding", wrong_cur)

    def insufficient():
        f = dict(perfect["findings"][0]); f["evidence_ids"] = []
        r1 = P({"episode_id": iep.episode_id, "findings": [f]})
        g = dict(perfect["findings"][0])
        sets = iep.truth["issues"][0]["evidence_sets"]
        g["evidence_ids"] = list(sets[0])[:-1]
        r2 = P({"episode_id": iep.episode_id, "findings": [g]})
        ok2 = r2["tp"] == 0 if len(sets) == 1 else True
        return r1["tp"] == 0 and ok2, f"empty evidence: {r1['finding_reasons'][0]}; one id missing: {r2['finding_reasons'][0]}"
    prop("missing or incomplete evidence set invalidates the finding", insufficient)

    def tolerance():
        t = iep.truth["issues"][0]
        tol, exp = t["tolerance_minor"], t["expected_amount_minor"]
        mk = lambda a: P({"episode_id": iep.episode_id, "findings": [dict(perfect["findings"][0], claimed_amount_minor=a)]})["tp"]
        return mk(exp + tol) == 1 and mk(max(0, exp - tol)) == 1 and mk(exp + tol + 1) == 0, \
            f"tolerance {tol}: +tol ok, +tol+1 rejected"
    prop("amount tolerance is inclusive max(1, expected//1000)", tolerance)

    def malformed():
        base = perfect["findings"][0] if perfect["findings"] else None
        eid = iep.episode_id
        cases = {
            "not an object": "oops", "missing findings": {"episode_id": eid},
            "extra key": {"episode_id": eid, "findings": [], "x": 1},
            "wrong episode_id": {"episode_id": "ep_000000000000", "findings": []},
            "findings not a list": {"episode_id": eid, "findings": "none"},
            "31 findings": {"episode_id": eid, "findings": [base] * 31},
            "bool amount": {"episode_id": eid, "findings": [dict(base, claimed_amount_minor=True)]},
            "negative amount": {"episode_id": eid, "findings": [dict(base, claimed_amount_minor=-1)]},
            "string amount": {"episode_id": eid, "findings": [dict(base, claimed_amount_minor="5")]},
            "float amount": {"episode_id": eid, "findings": [dict(base, claimed_amount_minor=5.0)]},
            "unknown type": {"episode_id": eid, "findings": [dict(base, type="theft")]},
            "empty affected_ids": {"episode_id": eid, "findings": [dict(base, affected_ids=[])]},
            "11 evidence ids": {"episode_id": eid, "findings": [dict(base, evidence_ids=["a"] * 11)]},
            "extra finding key": {"episode_id": eid, "findings": [dict(base, note="x")]},
            "missing finding key": {"episode_id": eid, "findings": [{k: v for k, v in base.items() if k != "currency"}]},
        }
        bad = [k for k, v in cases.items() if not (P(v)["reward_shared"] == -1.0 and not P(v)["valid"])]
        return not bad, f"{len(cases)} malformed variants all score -1" if not bad else f"not rejected: {bad}"
    prop("malformed submissions score -1 (15 variants)", malformed)

    # ---- env-level properties
    def resubmit():
        env = _env()
        env.reset(seed=ie["seed"], config=resolve_config(ie))
        _, r1, t1, _, _ = env.step({"action": "submit", "args": {"findings": perfect}})
        n = len(env.trajectory())
        obs, r2, t2, tr2, info = env.step({"action": "submit", "args": {"findings": sil}})
        obs2, r3, _, _, _ = env.step({"action": "list_records", "args": {"table": "fx"}})
        ok = (t1 and t2 and r2 == 0.0 and r3 == 0.0 and obs["error"]["code"] == "episode_over"
              and obs2["error"]["code"] == "episode_over" and len(env.trajectory()) == n)
        return ok, f"first {r1:+.3f}, re-submit reward {r2}, code {obs['error']['code']}"
    prop("re-submit / any step after the end -> episode_over, no state change, reward 0", resubmit)

    def malformed_env():
        out = []
        for mode in (MODE_TRAIN, MODE_EVAL):
            env = _env(mode)
            env.reset(seed=ie["seed"], config=resolve_config(ie))
            obs, rew, term, trunc, info = env.step({"action": "submit", "args": {"findings": {"episode_id": "x"}}})
            rpt = env.grader_report()
            out.append((mode, rew, term, obs["error"], info, rpt.get("errors"), rpt.get("reward_shared")))
        t, e = out
        ok = (t[1] == -1.0 and t[2] and t[4].get("invalid_submission") is True and t[3]["code"] == "invalid_args"
              and t[3]["message"].startswith("submission rejected")
              and e[1] == 0.0 and e[2] and e[4] == {} and e[3] is None
              and bool(e[5]) and e[6] == -1.0)
        return ok, f"train reward {t[1]}, eval error {e[3]!r} (no validity oracle), grader reward {e[6]}"
    prop("malformed submit: train -> reward -1 + reason; eval -> no error shown (no validity oracle), grader keeps -1 and error log", malformed_env)

    def truncation_steps():
        res = []
        for e, ep in ((ce, cep), (ie, iep)):
            env = _env()
            env.reset(seed=e["seed"], config=resolve_config(e))
            last = None
            for _ in range(MAX_STEPS):
                last = env.step({"action": "list_records", "args": {"table": "fx", "limit": 1}})
            obs, rew, term, trunc, info = last
            exp = matcher.shared_reward(0.0 if ep.truth["issues"] else 1.0, MAX_STEPS)
            res.append((trunc, term, rew, exp, info.get("invalid_submission")))
        ok = all(t and not m and abs(r - x) <= 0.0021 and inv is False for t, m, r, x, inv in res)
        return ok, f"clean {res[0][2]:+.3f}, issues {res[1][2]:+.3f} (truncated at step {MAX_STEPS}, scored as empty)"
    prop("truncation by steps: 100th non-submit step -> truncated, findings treated as empty and scored", truncation_steps)

    def truncation_time():
        env = _env(time_limit_s=5.0)
        ticks = iter(range(0, 1000000, 10))
        env._clock = lambda: next(ticks)
        env.reset(seed=ce["seed"], config=resolve_config(ce))
        obs, rew, term, trunc, info = env.step({"action": "list_records", "args": {"table": "fx"}})
        obs2, *_ = env.step({"action": "calculator", "args": {"expression": "1+1"}})
        exp = matcher.shared_reward(1.0, 0)
        ok = trunc and not term and abs(rew - exp) <= 0.0021 and obs2["error"]["code"] == "episode_over"
        return ok, f"truncated={trunc}, reward {rew:+.3f}"
    prop("truncation by wall-clock limit -> truncated and scored as empty", truncation_time)

    def not_reset():
        obs, rew, term, trunc, info = _env().step({"action": "list_records", "args": {"table": "fx"}})
        return obs["error"]["code"] == "not_reset" and rew == 0.0 and not term and not trunc, obs["error"]["code"]
    prop("step before reset -> not_reset", not_reset)

    def eval_hidden():
        env = _env(MODE_EVAL)
        env.reset(seed=ie["seed"], config=resolve_config(ie))
        obs, rew, term, trunc, info = env.step({"action": "submit", "args": {"findings": perfect}})
        rpt = env.grader_report()
        ok = term and rew == 0.0 and info == {} and rpt["reward_shared"] == 1.0 and rpt["metrics"]["f1"] == 1.0
        return ok, f"policy sees reward {rew}, info {info}; grader keeps {rpt['reward_shared']:+.3f}"
    prop("eval mode: terminal reward 0.0 and empty info to the policy; grader keeps the real result", eval_hidden)

    def train_info():
        env = _env()
        env.reset(seed=ie["seed"], config=resolve_config(ie))
        *_, info = env.step({"action": "submit", "args": {"findings": perfect}})
        keys = {"metrics", "reward_shared", "reward_experimental", "inspection_steps", "invalid_submission"}
        blob = json.dumps(info).lower()
        ok = set(info) == keys and not any(t.lower() in blob for t in LEAK_TOKENS) \
            and not any(i["issue_id"].lower() in blob for i in iep.truth["issues"])
        return ok, f"info keys {sorted(info)}"
    prop("train-mode terminal info has only counts/metrics (no ids)", train_info)

    def errors_cost_steps():
        env = _env()
        env.reset(seed=ie["seed"], config=resolve_config(ie))
        longname = "x" * 500
        o1, r1, *_ = env.step({"action": longname, "args": {}})
        o2, r2, *_ = env.step({"action": "get_record", "args": {"table": "ledger", "id": longname}})
        o3, r3, *_ = env.step({"action": "list_records", "args": {"table": "ledger", "limit": 51}})
        o4, r4, *_ = env.step({"action": "list_records", "args": {"table": "ledger", "limit": 0}})
        o5, r5, *_ = env.step({"action": "list_records", "args": {"table": "ledger", "cursor": "garbage"}})
        o6, r6, *_ = env.step({"action": "list_records", "args": {"table": "nope"}})
        o7, r7, *_ = env.step({"action": "calculator", "args": {"expression": "__import__('os')"}})
        errs = [o["error"] for o in (o1, o2, o3, o4, o5, o6, o7)]
        codes = [e["code"] for e in errs]
        ok = (codes == ["unknown_action", "unknown_id", "invalid_args", "invalid_args", "invalid_cursor",
                        "unknown_table", "calculator_error"]
              and all(len(e["message"]) <= 120 for e in errs) and all(r == 0.0 for r in (r1, r2, r3, r4, r5, r6, r7))
              and o7["steps_remaining"] == MAX_STEPS - 7)
        return ok, f"codes {codes}"
    prop("bad actions: bounded error code, message <= 120 chars, consume a step, reward 0", errors_cost_steps)

    def pagination():
        env = _env()
        env.reset(seed=ie["seed"], config=resolve_config(ie))
        seen, cur, pages = [], None, 0
        while pages < 80:
            o, *_ = env.step({"action": "list_records", "args": {"table": "ledger", "cursor": cur, "limit": 50}})
            res = o["result"]
            seen += [r["row_id"] for r in res["rows"]]
            cur = res["next_cursor"]
            pages += 1
            if not cur:
                break
        # a cursor from another table must be rejected
        o, *_ = env.step({"action": "list_records", "args": {"table": "ledger", "limit": 1}})
        foreign = o["result"]["next_cursor"]
        o2, *_ = env.step({"action": "list_records", "args": {"table": "payments", "cursor": foreign}})
        ok = len(seen) == len(set(seen)) == res["total"] == len(iep.ledger) and o2["error"]["code"] == "invalid_cursor"
        return ok, f"{len(seen)} ledger rows over {pages} pages; foreign cursor rejected"
    prop("pagination: stable cursors, every row exactly once, foreign cursor rejected", pagination)

    def flag_all_train():
        res = []
        for e, ep, kind in eps[:6]:
            r = run_episode("flag_all", e["seed"], resolve_config(e), mode=MODE_TRAIN, remote=False)
            res.append(r["trajectory"][-1]["reward"])
        return all(x < 0 for x in res), "rewards " + ", ".join(f"{x:+.2f}" for x in res)
    prop("flag-all scores badly: reward < 0 on the first 6 episodes (train mode, env)", flag_all_train)

    def policies_never_import_truth():
        out = []
        pol_dir = os.path.join(ROOT, "auditgym", "policies")
        for fn in sorted(os.listdir(pol_dir)):
            if fn.endswith(".py") and fn != "oracle.py":
                with open(os.path.join(pol_dir, fn), encoding="utf-8") as f:
                    src = f.read()
                for bad in ("generator", "provenance", "truth", "grader_data", "episode import", "load_episode"):
                    if re.search(rf"^\s*(from|import)\s.*{bad}", src, re.M):
                        out.append(f"{fn}:{bad}")
        return not out, "policies (except oracle) have no truth/generator imports" if not out else str(out)
    prop("agent-side policies import neither generator nor truth", policies_never_import_truth)


# ================================================================== step 8: output
def fmt_ci(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def results_lines(rep: Report, level: str = "##") -> list[str]:
    """Baseline table + failure analysis as markdown lines (shared by the report and the docs sync)."""
    L: list[str] = []
    base = rep.sections.get("baselines")
    if base:
        L += ["", f"{level} Baseline policies (eval mode, holdout, remote env)", "",
              "Shared reward, F1, experimental reward, steps and wall time are means over episodes (95% percentile "
              "bootstrap CI, 2000 resamples, fixed seed). Precision and recall are POOLED over the episodes "
              "(sum TP / (sum TP + sum FP or FN)), because a silent agent on a clean book has P = R = 0 (0/0) "
              "while F1 = 1.", "",
              "| policy | n | precision (pooled) | recall (pooled) | F1 [95% CI] | shared reward [95% CI] | experimental | steps | wall s |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t in base["table"]:
            L.append(f"| {t['policy']} | {t['n']} | {t['precision']:.3f} | {t['recall']:.3f} | {fmt_ci(t['f1'])} | "
                     f"{fmt_ci(t['shared'])} | {t['experimental']:+.3f} | {t['steps']:.1f} | {t['wall_s']:.2f} |")
        L += ["", "`oracle` reads the truth grader-side and only demonstrates the reward ceiling; it is not agent ability."]
        if any(t["policy"].endswith("(extra)") for t in base["table"]):
            L += ["", "`careful_auditor (extra)` is an additional diagnostic row, not one of the 4 required baselines: an honest "
                  "observation-only scripted agent (no truth, no generator, no rules engine) showing the environment is "
                  "solvable from public evidence within the 100-step budget."]
        if any(t["policy"].startswith("careful_auditor_degraded") for t in base["table"]):
            L += ["", "`careful_auditor_degraded (extra)` is the same agent with a 20-inspection-step cap and period membership "
                  "decided on the wall-clock fields of `issued_at` (the time-zone rule ignored). It is the near-miss row: "
                  "the reward and F1 gap to `careful_auditor` is measured, not asserted, and the failure analysis below "
                  "lists which issue types it misses."]
    fa = rep.sections.get("failure_analysis")
    if fa:
        L += ["", f"{level} Failure analysis (measured on the same eval runs; from the grader reports)", "",
              "Recall by issue type = matched truth issues / truth issues of that type (all episodes of the run).", "",
              "| policy | " + " | ".join(ISSUE_TYPES) + " | issues born of the 2 interaction scenarios |",
              "|---|" + "---|" * (len(ISSUE_TYPES) + 1)]
        for pol, d in fa.items():
            cells = []
            for t in ISSUE_TYPES:
                v = d["recall_by_type"][t]
                cells.append(f"{v['tp']}/{v['truth']}")
            sv = d["recall_by_interaction"]
            cells.append(f"{sum(v['tp'] for v in sv.values())}/{sum(v['truth'] for v in sv.values())}")
            L.append(f"| {pol} | " + " | ".join(cells) + " |")
        L += ["", "Findings by type: true positives / findings submitted (the rest are false positives).", "",
              "| policy | " + " | ".join(ISSUE_TYPES) + " | FP total |", "|---|" + "---|" * (len(ISSUE_TYPES) + 1)]
        for pol, d in fa.items():
            cells = [f"{d['precision_by_type'][t]['tp']}/{d['precision_by_type'][t]['findings']}" for t in ISSUE_TYPES]
            L.append(f"| {pol} | " + " | ".join(cells) + f" | {d['fp_total']} |")
        L += ["", "Why the false positives failed to match (matcher reason = furthest stage reached against any issue; "
              "`ids_mismatch` = no truth issue has this type + affected_ids, i.e. a plain false accusation), and "
              "which distractor kind they landed on:", "",
              "| policy | FP reasons | FP by distractor kind |", "|---|---|---|"]
        for pol, d in fa.items():
            L.append(f"| {pol} | {', '.join(f'{k}: {v}' for k, v in d['fp_reasons'].items()) or '-'} | "
                     f"{', '.join(f'{k}: {v}' for k, v in d['fp_by_distractor_kind'].items()) or '-'} |")
        L += ["", "Near-miss analysis: what each policy misses (false negatives) and wrongly claims (false positives) "
              "per issue type, and what its inspection steps cost. `careful_auditor_degraded` is `careful_auditor` "
              "with a 20-step cap and period membership decided on wall-clock fields instead of instants; the gap "
              "to `careful_auditor` is the price of those two mistakes. Step cost = mean inspection steps x "
              f"{STEP_COST} (it is the only reward term a perfect answer can still lose).", "",
              "| policy | missed (FN) per type: " + " / ".join(ISSUE_TYPES) + " | FP per type | mean steps | "
              "step cost | mean shared reward |", "|---|---|---|---|---|---|"]
        for pol, d in fa.items():
            if "misses_by_type" not in d:
                continue
            L.append(f"| {pol} | " + " / ".join(str(d["misses_by_type"][t]) for t in ISSUE_TYPES) + " | "
                     + " / ".join(str(d["fp_by_type"][t]) for t in ISSUE_TYPES)
                     + f" | {d['mean_inspection_steps']:.1f} | {d['mean_step_cost']:.3f} | "
                     f"{d['mean_shared_reward']:+.3f} |")
    pr = rep.sections.get("paired")
    if pr and pr.get("per_pair_f1"):
        L += ["", f"{level} Paired invariance pairs (per-pair F1 of the untransformed run)", "",
              "A pair only proves something when the policy is above the -1.0 floor, so the discriminating policies "
              "are `reconcile` and `careful_auditor`; `random_explorer` and `flag_all` are excluded (trivially "
              "invariant at the floor). Each cell is the F1 of one holdout episode (order: clean, interaction, mixed, "
              "as in `splits.json`); every listed pair gave identical metrics, reward and steps under the shape and "
              "the aggressive rename + permute.", "",
              "| policy | pairs above floor | per-pair F1 |", "|---|---|---|"]
        for pol, rows in pr["per_pair_f1"].items():
            above = sum(1 for r in rows if r["shared"] > -1.0 + 1e-9)
            L.append(f"| {pol} | {above}/{len(rows)} | " + " ".join(f"{r['f1']:.2f}" for r in rows) + " |")
    return L


def write_reports(rep: Report, out_dir: str, quick: bool, elapsed: float) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    passed = not rep.failed
    counts = {k: sum(1 for c in rep.checks if c["status"] == k) for k in ("PASS", "FAIL", "WARN", "SKIP")}
    cfgs = rep.sections.get("configs") or {}
    data = {"env_version": ENV_VERSION, "python": platform.python_version(), "platform": platform.platform(),
            "config_ids": {n: v.get("config_id") for n, v in cfgs.items()},
            "config_versions": {n: v.get("config_version") for n, v in cfgs.items()},
            "config_hashes": {n: v.get("config_hash") for n, v in cfgs.items()},
            "config_file_sha256": {n: v.get("file_sha256") for n, v in cfgs.items()},
            "quick": quick,
            "all_hard_checks_passed": passed, "counts": counts, "checks": rep.checks,
            "sections": rep.sections, "elapsed_s": round(elapsed, 1)}
    jp = os.path.join(out_dir, "check_report.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    L = [f"# AuditGym check report", "",
         f"- env_version: `{ENV_VERSION}`  python: `{platform.python_version()}`  platform: `{platform.system()}`  "
         f"mode: `{'quick' if quick else 'full'}`",
         "- configs: " + (", ".join(f"`{n}` {v.get('config_id')}@{v.get('config_version')} "
                                   f"(hash `{(v.get('config_hash') or '')[:12]}`)" for n, v in cfgs.items()) or "n/a"),
         f"- result: **{'PASS' if passed else 'FAIL'}**  ({counts['PASS']} pass, {counts['FAIL']} fail, "
         f"{counts['WARN']} warn, {counts['SKIP']} skip)", ""]
    cur = None
    for c in rep.checks:
        if c["step"] != cur:
            cur = c["step"]
            L += ["", f"## Step {cur}", "", "| status | check | detail |", "|---|---|---|"]
        L.append(f"| {c['status']} | {c['name']} | {c['detail'].replace('|', '/')} |")
    L += results_lines(rep)
    with open(os.path.join(out_dir, "check_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return jp, os.path.join(out_dir, "check_report.md")


BEGIN_MARK, END_MARK = "<!-- RESULTS:BEGIN (generated by `python -m auditgym check --update-docs`; do not edit) -->", "<!-- RESULTS:END -->"


def update_docs(rep: Report, quick: bool) -> None:
    """Rewrite the generated results blocks of README.md and docs/DESIGN.md from this run, and keep a
    copy of the full report in docs/ (reports/ is scratch output). Full runs only."""
    if quick:
        print("--update-docs ignored: only a full run may update the documentation")
        return
    counts = {k: sum(1 for c in rep.checks if c["status"] == k) for k in ("PASS", "FAIL", "WARN", "SKIP")}
    pyt = rep.sections.get("pytest", {}).get("tail", [""])
    pyt_line = next((x for x in reversed(pyt) if "passed" in x or "failed" in x), "")
    block = [BEGIN_MARK, "",
             f"Generated by the latest **full** `python -m auditgym check` run: {counts['PASS']} PASS, "
             f"{counts['FAIL']} FAIL, {counts['WARN']} WARN, {counts['SKIP']} SKIP"
             + (f"; pytest: {pyt_line.strip().rsplit(' in ', 1)[0]}" if pyt_line else "")
             + ". Full per-check table: [docs/check_report.md](docs/check_report.md).", ""]
    block += results_lines(rep, "####")
    block += ["", END_MARK]
    text = "\n".join(block)
    for rel in ("README.md", os.path.join("docs", "DESIGN.md")):
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as f:
            doc = f.read()
        if BEGIN_MARK not in doc or END_MARK not in doc:
            print(f"warning: no results markers in {rel}")
            continue
        head = doc[: doc.index(BEGIN_MARK)]
        tail = doc[doc.index(END_MARK) + len(END_MARK):]
        t = text if rel == "README.md" else text.replace("(docs/check_report.md)", "(check_report.md)")
        with open(path, "w", encoding="utf-8") as f:
            f.write(head + t + tail)
    for ext in ("md", "json"):
        shutil.copyfile(os.path.join(rep.out_dir, f"check_report.{ext}"), os.path.join(ROOT, "docs", f"check_report.{ext}"))
    print("docs updated: README.md, docs/DESIGN.md, docs/check_report.{md,json}")


# ================================================================== entry point
def run_check(quick: bool = False, out_dir: str | None = None, remote: bool = True,
              update_docs_flag: bool = False) -> int:
    t0 = time.perf_counter()
    sys.dont_write_bytecode = True           # `check` must not litter the source tree with __pycache__
    out_dir = out_dir or os.path.join(ROOT, "reports")
    rep = Report()
    rep.out_dir = out_dir
    cache: dict = {}
    print(f"AuditGym check ({'quick' if quick else 'full'}) - {ENV_VERSION}\n")

    guarded(rep, "1", lambda: step_pytest(rep))
    try:
        splits = load_splits(ROOT)
    except Exception as e:
        rep.add("2", "configs/splits.json readable", False, f"{type(e).__name__}: {e}")
        splits = {}
    state: dict[str, Any] = {}
    guarded(rep, "2", lambda: step_repro(rep, splits, quick, cache))
    guarded(rep, "2", lambda: step_config_lock(rep))
    guarded(rep, "2", lambda: step_golden(rep))
    guarded(rep, "2", lambda: step_trajectory_files(rep))

    def s3():
        state["eps"] = select_holdout(rep, splits, quick, cache)
    guarded(rep, "3", s3)
    eps = state.get("eps") or []
    runs: dict = {}
    extra: list[dict] = []
    if eps:
        def s4():
            log_root = os.path.join(out_dir, "grader_logs")
            shutil.rmtree(log_root, ignore_errors=True)
            runs.update(step_baselines(rep, eps, remote, log_root))
        guarded(rep, "4", s4)
        if runs.get("oracle") and runs.get("reconcile") and len(runs["oracle"]) == len(eps) == len(runs["reconcile"]):
            n_pairs = 5 if quick else 20
            guarded(rep, "5", lambda: extra.extend(step_paired(rep, eps, runs, remote, min(n_pairs, len(eps)))))
            guarded(rep, "5", lambda: step_oracle_truth(rep, splits, remote, quick))
        else:
            rep.add("5", "paired checks", False, "baseline runs missing")
        guarded(rep, "6", lambda: step_leaks(rep, [r for v in runs.values() for r in v] + extra))
        guarded(rep, "6", lambda: step_isolation(rep, eps, out_dir))
        guarded(rep, "7", lambda: step_rewards(rep, eps, cache))
    else:
        for st in ("4", "5", "6", "7"):
            rep.add(st, "skipped: no holdout episodes available", False)

    elapsed = time.perf_counter() - t0
    jp, mp = write_reports(rep, out_dir, quick, elapsed)
    if update_docs_flag and not rep.failed:
        update_docs(rep, quick)
    n_fail = len(rep.failed)
    print(f"\n{'ALL HARD CHECKS PASSED' if not n_fail else f'{n_fail} HARD CHECK(S) FAILED'}  "
          f"({elapsed:.1f}s)\nreports: {jp}\n         {mp}")
    return 0 if not n_fail else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="auditgym check")
    ap.add_argument("--quick", action="store_true", help="fewer episodes (smoke)")
    ap.add_argument("--out", default=None, help="report directory (default reports/)")
    ap.add_argument("--local", action="store_true", help="in-process env instead of the subprocess grader (debug)")
    ap.add_argument("--update-docs", action="store_true",
                    help="after a passing FULL run, regenerate the results blocks in README/DESIGN and docs/check_report.*")
    a = ap.parse_args(argv)
    return run_check(quick=a.quick, out_dir=a.out, remote=not a.local, update_docs_flag=a.update_docs)


if __name__ == "__main__":
    raise SystemExit(main())
