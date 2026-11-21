"""CLI: python -m auditgym {generate | check | run | demo | golden | lock | isolate}."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from .constants import MODE_EVAL, MODE_TRAIN
from .policies import EXTRA_POLICY_NAMES, POLICY_NAMES

ALL_POLICY_NAMES = POLICY_NAMES + EXTRA_POLICY_NAMES


def _root(a) -> str:
    from .runner import ROOT
    return os.path.abspath(getattr(a, "root", None) or ROOT)


# ------------------------------------------------------------------ generate
def cmd_generate(a) -> int:
    from .episode import save_episode
    from .generator.build import generate_episode
    from .runner import load_splits, resolve_config, write_trajectory_files

    root = _root(a)
    splits = load_splits(root)
    n_ep = 0
    for split in ("train", "dev"):
        entries = splits.get(split) or []
        if not entries:
            print(f"warning: no entries for split {split!r} in configs/splits.json", file=sys.stderr)
            continue
        pub_root = os.path.join(root, "episodes", split)
        priv_root = os.path.join(root, "grader_data", split)
        for d in (pub_root, priv_root):          # exact regeneration: drop stale episodes first
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d)
        for e in entries:
            cfg = resolve_config(e, root)
            ep = generate_episode(e["seed"], cfg)
            save_episode(ep, pub_root, priv_root)
            n_ep += 1
        print(f"{split}: wrote {len(entries)} episodes -> episodes/{split}/, grader_data/{split}/")
    if not a.no_trajectories:
        counts = write_trajectory_files(splits, root, remote=a.remote)
        for name, n in counts.items():
            print(f"trajectories/{name}: {n} lines")
    print(f"done: {n_ep} episodes")
    return 0


# ------------------------------------------------------------------ check
def cmd_check(a) -> int:
    from .check import run_check
    return run_check(quick=a.quick, out_dir=a.out, remote=not a.local, update_docs_flag=a.update_docs,
                     allow_skip_tests=a.allow_skip_tests)


# ------------------------------------------------------------------ run
def _options(a) -> dict | None:
    if a.rename_salt is None and a.permute_salt is None:
        return None
    salt = a.rename_salt
    if salt is not None and getattr(a, "rename_mode", "shape") == "aggressive":
        from .transforms import aggressive_salt
        salt = aggressive_salt(salt)
    return {"transform": {"rename_salt": salt, "permute_salt": a.permute_salt}}


def _load_config_arg(config):
    """A config name stays a name; a path to a JSON file is loaded into a dict."""
    if isinstance(config, str) and os.path.isfile(config):
        with open(config, encoding="utf-8") as f:
            return json.load(f)
    return config


def cmd_run(a) -> int:
    from .runner import run_episode
    config = _load_config_arg(a.config)
    options = _options(a)
    local = a.local or a.deterministic
    r = run_episode(a.policy, a.seed, config, mode=a.mode, options=options, remote=not local,
                    policy_seed=a.policy_seed, log_dir=a.log_dir, deterministic=a.deterministic)
    if a.pretty:
        print(format_run(r, a.seed, config, options))
        return 0
    last = r["trajectory"][-1]
    out = {"episode_id": r["episode_id"], "policy": a.policy, "steps": r["steps"], "wall_s": r["wall_s"],
           "terminal_reward_seen_by_policy": last["reward"], "terminated": last["terminated"],
           "truncated": last["truncated"], "grader_report": r["grader_report"]}
    if a.mode == MODE_EVAL:
        out["note"] = ("eval mode hides the terminal reward from the policy (0.0); the real reward is "
                       "grader_report.reward_shared")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


# ------------------------------------------------------------------ pretty formatter (demo + run --pretty)
def _short(x, n=96) -> str:
    s = json.dumps(x, ensure_ascii=False, sort_keys=True) if not isinstance(x, str) else x
    return s if len(s) <= n else s[: n - 3] + "..."


def _describe_action(act) -> str:
    if act is None:
        return "(reset)"
    name, args = act.get("action"), act.get("args") or {}
    if name == "list_records":
        return f"list_records table={args.get('table')} limit={args.get('limit')} cursor={'yes' if args.get('cursor') else 'null'}"
    if name == "submit":
        n = len((args.get("findings") or {}).get("findings", []))
        return f"submit {n} finding(s)"
    return f"{name} {_short(args, 70)}"


def _describe_result(obs) -> str:
    if obs.get("error"):
        return f"ERROR {obs['error']['code']}: {obs['error']['message']}"
    res = obs.get("result")
    if isinstance(res, dict) and "rows" in res:
        return f"{len(res['rows'])} rows (total {res['total']}, more={'yes' if res['next_cursor'] else 'no'})"
    return _short(res) if res is not None else "-"


def _score_run(r: dict, seed: int, config, options: dict | None):
    """Grader-side view of a finished run: (submission, truth, matcher result)."""
    from . import matcher
    from .runner import grader_episode, submitted_findings
    ep = grader_episode(seed, config, options)
    traj = r["trajectory"]
    sub = submitted_findings(traj) or {"episode_id": r["episode_id"], "findings": []}
    steps = r["grader_report"].get("inspection_steps", r["steps"] - 1)
    res = matcher.score(sub, ep.truth, ep.public_ids(), steps, r["episode_id"])
    return sub, ep.truth, res


def _score_line(res: dict) -> str:
    return (f"TP={res['tp']} FP={res['fp']} FN={res['fn']}  precision={res['precision']:.3f} "
            f"recall={res['recall']:.3f} F1={res['f1']:.3f}  shared reward={res['reward_shared']:+.3f}  "
            f"experimental={res['reward_experimental']:+.3f}")


def format_run(r: dict, seed: int, config, options: dict | None = None) -> str:
    """Readable step log + grader-side findings/truth/score of one run (shared by `demo` and `run --pretty`)."""
    traj = r["trajectory"]
    m = traj[0]["observation"]["manifest"]
    rep = r["grader_report"]
    L = [f"Episode {r['episode_id']}  company: {m['company_name']}  base: {m['base_currency']}  "
         f"period: {m['period_start'][:10]}..{m['period_end'][:10]}  counts: {m['counts']}",
         f"Policy: {r['policy']}   mode: {r['mode']}   env: {rep.get('env_version')}   "
         f"config: {rep.get('config_id')}@{rep.get('config_version')} (hash {str(rep.get('config_hash'))[:12]})", "",
         f"{'step':>4}  {'left':>4}  {'action':<56}  result"]
    for rec in traj[1:]:
        o = rec["observation"]
        L.append(f"{rec['step']:>4}  {o['steps_remaining']:>4}  {_describe_action(rec['action']):<56}  {_describe_result(o)}")
    last = traj[-1]
    shown = f"{last['reward']:+.4f}"
    if r["mode"] == MODE_EVAL:
        shown += f"  (eval mode hides it; grader reward_shared = {rep.get('reward_shared'):+.4f})" \
            if isinstance(rep.get("reward_shared"), (int, float)) else "  (eval mode hides it)"
    L.append(f"\nended: terminated={last['terminated']} truncated={last['truncated']} reward={shown}")
    sub, truth, res = _score_run(r, seed, config, options)
    L += ["", "Submitted findings (grader-side view with match reasons):"]
    for i, f in enumerate(sub["findings"]):
        reason = res["finding_reasons"][i] if res.get("finding_reasons") else "?"
        L.append(f"  {i + 1:>2}. {f['type']:<22} {','.join(f['affected_ids']):<22} {f['claimed_amount_minor']:>12} "
                 f"{f['currency']}  -> {reason}")
    L += ["", "Truth issues (grader-side):"]
    matched = {iid for _, iid in res.get("pairs", [])}
    for t in truth["issues"]:
        L.append(f"  {'FOUND ' if t['issue_id'] in matched else 'MISSED'} {t['type']:<22} {','.join(t['affected_ids']):<22} "
                 f"{t['expected_amount_minor']:>12} {t['currency']}  scenario={t.get('scenario')}")
    L.append(f"\n{_score_line(res)}")
    return "\n".join(L)


def format_contrast(main: dict, other: dict, seed: int, config) -> str:
    """Compact side-by-side of two policies on the same episode (what the naive one misses)."""
    _s1, _t1, r1 = _score_run(main, seed, config, None)
    sub2, truth, r2 = _score_run(other, seed, config, None)
    matched = {iid for _, iid in r2.get("pairs", [])}
    missed = [t for t in truth["issues"] if t["issue_id"] not in matched]
    L = [f"Contrast: {other['policy']} on the same episode ({other['steps'] - 1} inspection steps)",
         f"  {main['policy']:<18} {_score_line(r1)}",
         f"  {other['policy']:<18} {_score_line(r2)}"]
    if missed:
        L.append(f"  {other['policy']} missed: " + "; ".join(f"{t['type']} {','.join(t['affected_ids'])}" for t in missed[:6])
                 + (f" (+{len(missed) - 6} more)" if len(missed) > 6 else ""))
    wrong = [i for i, why in enumerate(r2.get("finding_reasons") or []) if i not in {fi for fi, _ in r2.get("pairs", [])}]
    if wrong:
        L.append(f"  {other['policy']} false positives: {len(wrong)}")
    L.append(f"  reward gap: {r1['reward_shared'] - r2['reward_shared']:+.3f} in favour of {main['policy']}"
             if r1["reward_shared"] >= r2["reward_shared"] else
             f"  reward gap: {r2['reward_shared'] - r1['reward_shared']:+.3f} in favour of {other['policy']}")
    return "\n".join(L)


def cmd_demo(a) -> int:
    from .runner import load_splits, resolve_config, run_episode

    root = _root(a)
    if a.seed is not None:
        config = _load_config_arg(a.config or "dev")
        seed = a.seed
    else:
        splits = load_splits(root)
        entries = splits.get("dev") or splits.get("train") or []
        if not entries:
            print("no dev/train entries in configs/splits.json", file=sys.stderr)
            return 1
        e = entries[min(a.index, len(entries) - 1)]
        config = _load_config_arg(a.config) if a.config else resolve_config(e, root)
        seed = e["seed"]
    r = run_episode(a.policy, seed, config, mode=MODE_TRAIN, remote=False)
    print(format_run(r, seed, config))
    if a.contrast and a.contrast != "none" and a.contrast != a.policy:
        other = run_episode(a.contrast, seed, config, mode=MODE_TRAIN, remote=False)
        print("\n" + "=" * 100 + "\n" + format_contrast(r, other, seed, config))
    return 0


# ------------------------------------------------------------------ golden
def cmd_golden(a) -> int:
    from .runner import GOLDEN_PATH, golden_digests
    got = golden_digests()
    text = json.dumps(got, indent=2, sort_keys=True) + "\n"
    if a.write:
        with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {GOLDEN_PATH} ({len(got['episodes'])} episodes)")
        return 0
    if not os.path.exists(GOLDEN_PATH):
        print("tests/golden_hashes.json is missing; run `python -m auditgym golden --write`", file=sys.stderr)
        return 1
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        same = json.load(f) == json.loads(text)
    print("golden digests " + ("match" if same else "DIFFER from tests/golden_hashes.json"))
    return 0 if same else 1


# ------------------------------------------------------------------ config lock
def cmd_lock(a) -> int:
    from .generator.config import CONFIG_DIR, LOCK_FILE, build_config_lock, verify_config_lock
    path = os.path.join(CONFIG_DIR, LOCK_FILE)
    if a.write:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(build_config_lock(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")
        return 0
    problems = verify_config_lock()
    for p in problems:
        print("config lock: " + p, file=sys.stderr)
    print("config lock " + ("OK" if not problems else "BROKEN (bump config_version, then `lock --write`)"))
    return 1 if problems else 0


# ------------------------------------------------------------------ isolate
def cmd_isolate(a) -> int:
    from .isolate import main_args
    return main_args(a)


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m auditgym", description="AuditGym RL environment")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write episodes/, grader_data/, trajectories/")
    g.add_argument("--root", default=None)
    g.add_argument("--no-trajectories", action="store_true")
    g.add_argument("--remote", action="store_true", help="use the subprocess grader for trajectories")
    g.set_defaults(fn=cmd_generate)

    c = sub.add_parser("check", help="the one official verification command")
    c.add_argument("--quick", action="store_true")
    c.add_argument("--out", default=None)
    c.add_argument("--local", action="store_true")
    c.add_argument("--update-docs", action="store_true",
                   help="after a passing full run, regenerate results in README/DESIGN and docs/check_report.*")
    c.add_argument("--allow-skip-tests", action="store_true",
                   help="do not fail when pytest is missing or tests/ is empty (the suite is then SKIP, "
                        "and the run is NOT a full verification)")
    c.set_defaults(fn=cmd_check)

    r = sub.add_parser("run", help="run one policy on one episode")
    r.add_argument("--policy", required=True, choices=ALL_POLICY_NAMES)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--config", required=True, help="config name (train/dev/holdout) or path")
    r.add_argument("--mode", choices=(MODE_EVAL, MODE_TRAIN), default=MODE_EVAL)
    r.add_argument("--policy-seed", type=int, default=0)
    r.add_argument("--rename-salt", type=int, default=None)
    r.add_argument("--rename-mode", choices=("shape", "aggressive"), default="shape",
                   help="shape: ids keep their format; aggressive: other prefix/length/charset/sort order")
    r.add_argument("--permute-salt", type=int, default=None)
    r.add_argument("--local", action="store_true", help="in-process env instead of subprocess grader")
    r.add_argument("--deterministic", action="store_true",
                   help="in-process, wall clock frozen: the run can never be time-truncated (implies --local)")
    r.add_argument("--pretty", action="store_true", help="readable step log + findings vs truth (the demo format)")
    r.add_argument("--log-dir", default=None, help="write the per-episode grader report here (subprocess grader)")
    r.set_defaults(fn=cmd_run)

    d = sub.add_parser("demo", help="pretty step log of one policy (default careful_auditor) on a dev episode, "
                                    "then reconcile for contrast")
    d.add_argument("--root", default=None)
    d.add_argument("--policy", default="careful_auditor", choices=list(ALL_POLICY_NAMES))
    d.add_argument("--contrast", default="reconcile", choices=list(ALL_POLICY_NAMES) + ["none"],
                   help="second policy run on the same episode for comparison (default reconcile; none = skip)")
    d.add_argument("--index", type=int, default=0, help="which dev episode (ignored with --seed)")
    d.add_argument("--seed", type=int, default=None, help="run this seed instead of a dev episode")
    d.add_argument("--config", default=None,
                   help="config name or path (default: the dev entry's config, or `dev` with --seed)")
    d.set_defaults(fn=cmd_demo)

    gd = sub.add_parser("golden", help="compare (or --write) the committed golden digests of tests/golden_hashes.json")
    gd.add_argument("--write", action="store_true")
    gd.set_defaults(fn=cmd_golden)

    from .isolate import build_parser as isolate_parser
    iso = sub.add_parser("isolate", help="run an agent in its own process and uid against the grader server "
                                          "(--self-test: escape-probe agent)")
    isolate_parser(iso)
    iso.set_defaults(fn=cmd_isolate)

    lk = sub.add_parser("lock", help="verify (or --write) configs/CONFIG_VERSIONS.json: config_id@version -> content hash")
    lk.add_argument("--write", action="store_true")
    lk.set_defaults(fn=cmd_lock)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
