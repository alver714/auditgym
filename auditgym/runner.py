"""Runner: drive a policy through an episode, record the trajectory, collect the grader report.

Trust boundary
--------------
* The policy lives on the *agent side*: it only sees observations and emits actions.
* ``run_episode`` is the *grader context*. For the ``oracle`` policy ONLY it generates the
  episode itself (``generator.generate_episode`` + the same benign transforms as the env) to read
  the truth and hands it to the oracle constructor. No other policy ever receives truth, and
  the truth never enters any observation.
* With ``remote=True`` the environment (and its truth) lives in a separate subprocess
  (``client.RemoteAuditEnv``); the grader report is fetched only after the episode ended.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from typing import Any

from .constants import ENV_VERSION, MAX_STEPS, MODE_EVAL, MODE_TRAIN
from .episode import dumps
from .policies import POLICY_NAMES, Policy, get_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFETY_STEPS = MAX_STEPS + 5          # the env ends the episode by step MAX_STEPS; hard loop guard


# ------------------------------------------------------------------ paths / splits
def default_root() -> str:
    return ROOT


def load_config_arg(config: Any) -> Any:
    """Config may be a dict, a config name, or a path; returned unchanged (env/generator resolve it)."""
    return config


def load_splits(root: str | None = None) -> dict:
    """Parse configs/splits.json into ``{split: [ {"seed", "config", "tag"} ... ]}``.

    Tolerant of a few layouts: ``{"train": {"config": "train", "seeds": [..]}}``,
    ``{"train": [seed, ...]}``, ``{"holdout": {"config":..., "clean": [...], "interaction": [...],
    "mixed": [...]}}`` (group keys become ``tag``), or lists of ``{"seed", "config", "tag"}`` entries.
    A top-level ``"splits"`` key is unwrapped.
    """
    path = os.path.join(root or ROOT, "configs", "splits.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw.get("splits"), dict):
        raw = raw["splits"]
    out: dict[str, list[dict]] = {}
    for split, spec in raw.items():
        if split in ("version", "schema_version", "description", "notes", "splits_version"):
            continue
        entries = _parse_split(split, spec)
        if entries is not None:
            out[split] = entries
    return out


def _entry(seed: Any, config: Any, tag: str | None, force: Any = None, category: str | None = None) -> dict:
    e = {"seed": int(seed), "config": config, "tag": tag}
    if force:
        e["force"] = force
    if category:
        e["category"] = category
    return e


def _parse_split(split: str, spec: Any, default_cfg: Any = None, tag: str | None = None):
    cfg_default = default_cfg if default_cfg is not None else split
    if isinstance(spec, list):
        out = []
        for item in spec:
            if isinstance(item, dict):
                out.append(_entry(item["seed"], item.get("config", item.get("config_id", cfg_default)),
                                  item.get("tag", item.get("kind", item.get("category", tag))),
                                  item.get("force"), item.get("category")))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append(_entry(item[0], item[1], tag))
            else:
                out.append(_entry(item, cfg_default, tag))
        return out
    if isinstance(spec, dict):
        cfg = spec.get("config", spec.get("config_id", cfg_default))
        if "seeds" in spec:
            sub = _parse_split(split, spec["seeds"], cfg, tag)
            force = spec.get("force")
            if force:
                for e in sub:
                    e["force"] = force
            return sub
        out = []
        found = False
        for key, val in spec.items():
            if key in ("config", "config_id", "force", "description", "version", "config_version"):
                continue
            if isinstance(val, (list, dict)):
                sub = _parse_split(split, val, cfg, key)
                if sub is not None:
                    found = True
                    force = val.get("force") if isinstance(val, dict) else None
                    if force:
                        for e in sub:
                            e["force"] = force
                    out.extend(sub)
        return out if found else None
    return None


def resolve_config(entry: dict, root: str | None = None) -> Any:
    """Concrete config argument for an entry: a name, or a dict if the entry pins ``force``."""
    cfg = entry["config"]
    force = entry.get("force")
    if not force:
        return cfg
    from .generator.config import config_with_force
    base = config_with_force(cfg, None)
    merged = dict(base.get("force") or {})
    merged.update(force)
    return config_with_force(base, merged)


# ------------------------------------------------------------------ env construction
def _frozen_clock() -> float:
    return 0.0


def _make_env(mode: str, remote: bool, log_dir: str | None = None, deterministic: bool = False):
    if deterministic and remote:
        raise ValueError("deterministic mode (wall clock disabled) needs the in-process env: pass remote=False")
    if not remote:
        from .env import AuditEnv
        env = AuditEnv(mode=mode)
        if deterministic:
            # the wall clock never advances: no time truncation, observations (incl. the task text with
            # its nominal time limit) stay byte-identical to a normal run that did not time out
            env._clock = _frozen_clock
        return env
    from .client import RemoteAuditEnv
    params = inspect.signature(RemoteAuditEnv.__init__).parameters
    kwargs: dict[str, Any] = {}
    for name in params:
        if name == "mode":
            kwargs["mode"] = mode
        elif "grader" in name:
            kwargs[name] = True                      # runner fetches the report AFTER the episode
        elif name == "log_dir" and log_dir:
            kwargs[name] = log_dir
    return RemoteAuditEnv(**kwargs)


def _close(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


# ------------------------------------------------------------------ grader context (oracle)
def grader_episode(seed: int, config: Any, options: dict | None = None):
    """GRADER CONTEXT ONLY: the full episode (public records + truth) the env will serve, including the
    env's benign transforms. Regenerated from (seed, config), or loaded from ``episode_dir``."""
    from .episode import load_episode
    if isinstance(config, dict) and "episode_dir" in config:
        ep = load_episode(config["episode_dir"], config.get("grader_dir"))
    else:
        from .generator.build import generate_episode
        ep = generate_episode(seed, config)
    return _apply_transforms(ep, options)


def grader_truth(seed: int, config: Any, options: dict | None = None) -> dict:
    """GRADER CONTEXT ONLY: regenerate the episode (with the env's transforms) to read the truth.

    Used solely to construct the ``oracle`` policy. Never call from agent-side code.

    Note: this is the *same generator* the env runs, so the oracle's reward of +1.0 alone proves little about
    the truth itself. ``check`` therefore cross-checks it against the committed ``grader_data/<split>/<id>/
    truth.json`` (see ``committed_truth`` / ``check.step_oracle_truth``) for the open splits.
    """
    return grader_episode(seed, config, options).truth


def committed_truth(split: str, episode_id: str, root: str | None = None) -> dict | None:
    """The shipped ``grader_data/<split>/<episode_id>/truth.json`` (open splits only), or None."""
    path = os.path.join(root or ROOT, "grader_data", split, episode_id, "truth.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _apply_transforms(ep, options):
    t = (options or {}).get("transform") or {}
    if t.get("rename_salt") is not None:
        from .transforms import rename_ids
        out = rename_ids(ep, t["rename_salt"])
        ep = out if out is not None else ep
    if t.get("permute_salt") is not None:
        from .transforms import permute
        out = permute(ep, t["permute_salt"])
        ep = out if out is not None else ep
    return ep


# ------------------------------------------------------------------ core loop
def trajectory_hash(trajectory: list[dict]) -> str:
    # backslashreplace: a policy that emits lone surrogates must not be able to abort its own recording
    return hashlib.sha256(dumps(trajectory).encode("utf-8", "backslashreplace")).hexdigest()


def _drive(env: Any, policy: Any, seed: int, config: Any, options: dict | None,
           policy_seed: int) -> tuple[list[dict], bool]:
    obs, _info = env.reset(seed=seed, config=config, options=options)
    traj: list[dict] = [{"episode_id": obs["episode_id"], "step": 0, "action": None,
                         "observation": obs, "reward": 0.0, "terminated": False, "truncated": False}]
    policy.reset(policy_seed, obs)
    terminated = truncated = False
    n = 0
    while not (terminated or truncated) and n < SAFETY_STEPS:
        action = policy.act(obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        n += 1
        traj.append({"episode_id": obs["episode_id"], "step": obs["step"], "action": action,
                     "observation": obs, "reward": reward,
                     "terminated": terminated, "truncated": truncated})
    return traj, bool(terminated or truncated)


def config_hash(config: Any) -> str | None:
    """sha256 of the RESOLVED config (defaults filled, ``force`` block included) as canonical JSON.
    None when the config is not a generator config (e.g. an ``episode_dir`` load)."""
    if isinstance(config, dict) and "episode_dir" in config:
        return None
    try:
        from .generator.config import config_hash as _hash, resolve_config as _resolve
        return _hash(_resolve(config))
    except Exception:
        return None


def config_identity(config: Any) -> tuple[str | None, str | None]:
    """(config_id, config_version) of the RESOLVED config. The public manifest no longer carries them (it
    would reveal the split to the policy), so the grader side reads them from the config itself."""
    if isinstance(config, dict) and "episode_dir" in config:
        return None, None
    try:
        from .generator.config import resolve_config as _resolve
        cfg = _resolve(config)
        return cfg["config_id"], cfg["config_version"]
    except Exception:
        return None, None


def reproducibility_meta(first_obs: dict, config: Any, report: dict, deterministic: bool = False) -> dict:
    """Version stamp of a run: which environment, which config (id, version, content hash), whether the
    wall clock cut the run short and whether it ran in deterministic (clock-free) mode."""
    man = (first_obs or {}).get("manifest") or {}
    cid, cver = config_identity(config)
    return {"env_version": man.get("env_version") or ENV_VERSION,
            "config_id": cid, "config_version": cver,
            "config_hash": config_hash(config),
            "time_truncated": bool(report.get("ended_by") == "time"),
            "deterministic": bool(deterministic)}


def run_episode(policy_name: str, seed: int, config: Any, mode: str = MODE_EVAL,
                options: dict | None = None, remote: bool = True, policy_seed: int = 0,
                log_dir: str | None = None, deterministic: bool = False) -> dict:
    """Run one policy on one episode.

    Returns ``{"episode_id", "trajectory", "grader_report", "wall_s", "steps", "policy",
    "seed", "mode", "meta"}``. ``trajectory`` holds the policy-visible records (reset = step 0).

    ``grader_report`` is the env's report plus the version stamp (``env_version``, ``config_id``,
    ``config_version``, ``config_hash``, ``time_truncated``, ``deterministic``), added here on the runner
    side. NOTE (eval mode): the trajectory's terminal reward is hidden (0.0) and ``info`` is empty by
    design; paired / regression checks must read ``grader_report["reward_shared"]``, never the trajectory.
    ``deterministic=True`` (in-process only) freezes the wall clock so a run can never be time-truncated.
    """
    truth = grader_truth(seed, config, options) if policy_name == "oracle" else None
    policy = get_policy(policy_name, policy_seed, truth)
    env = _make_env(mode, remote, log_dir, deterministic)
    t0 = time.perf_counter()
    try:
        traj, ended = _drive(env, policy, seed, config, options, policy_seed)
        wall = time.perf_counter() - t0
        report = env.grader_report() if ended else {"ended": False}
    finally:
        _close(env)
    meta = reproducibility_meta(traj[0]["observation"], config, report, deterministic)
    report = dict(report, **meta)
    return {"episode_id": traj[0]["episode_id"], "trajectory": traj, "grader_report": report,
            "wall_s": round(wall, 4), "steps": len(traj) - 1, "policy": policy_name,
            "seed": seed, "mode": mode, "meta": meta}


def replay_actions(actions: list[dict], seed: int, config: Any, mode: str = MODE_EVAL,
                   options: dict | None = None, remote: bool = True) -> dict:
    """Feed a recorded action list to a fresh env; returns the same record shape as run_episode."""
    env = _make_env(mode, remote)
    t0 = time.perf_counter()
    try:
        obs, _ = env.reset(seed=seed, config=config, options=options)
        traj = [{"episode_id": obs["episode_id"], "step": 0, "action": None, "observation": obs,
                 "reward": 0.0, "terminated": False, "truncated": False}]
        ended = False
        for action in actions:
            obs, reward, term, trunc, _ = env.step(action)
            traj.append({"episode_id": obs["episode_id"], "step": obs["step"], "action": action,
                         "observation": obs, "reward": reward, "terminated": term, "truncated": trunc})
            if term or trunc:
                ended = True
                break
        report = env.grader_report() if ended else {"ended": False}
    finally:
        _close(env)
    meta = reproducibility_meta(traj[0]["observation"], config, report)
    return {"episode_id": traj[0]["episode_id"], "trajectory": traj, "grader_report": dict(report, **meta),
            "wall_s": round(time.perf_counter() - t0, 4), "steps": len(traj) - 1, "seed": seed, "mode": mode,
            "meta": meta}


def submitted_findings(trajectory: list[dict]) -> Any:
    """The submission object of the last submit action (or None if the run never submitted)."""
    for rec in reversed(trajectory):
        a = rec.get("action")
        if isinstance(a, dict) and a.get("action") == "submit":
            return (a.get("args") or {}).get("findings")
    return None


# ------------------------------------------------------------------ trajectories on disk
ROW_KEYS = ("episode_id", "step", "action", "observation", "reward", "terminated", "truncated", "policy")
ORACLE_LABEL = ("oracle reference: reads the truth grader-side (zero inspection steps); shows the reward "
                "ceiling, it is NOT agent behaviour")


def trajectory_header(rec: dict) -> dict:
    """Run-level header stored on the step-0 row of every recorded trajectory (see
    ``schemas/trajectory_row.schema.json``). Deterministic: no wall time, no paths."""
    rep = rec.get("grader_report") or {}
    meta = rec.get("meta") or {}
    traj = rec["trajectory"]
    last = traj[-1]
    mode = rec.get("mode")
    h: dict[str, Any] = {
        "header_version": 1,
        "env_version": meta.get("env_version", ENV_VERSION), "config_id": meta.get("config_id"),
        "config_version": meta.get("config_version"), "config_hash": meta.get("config_hash"),
        "mode": mode, "policy": rec.get("policy"), "seed": rec.get("seed"), "split": rec.get("split"),
        "steps": rec.get("steps", len(traj) - 1),
        "terminated": bool(last["terminated"]), "truncated": bool(last["truncated"]),
        "ended_by": rep.get("ended_by"),
        "time_truncated": bool(meta.get("time_truncated", rep.get("ended_by") == "time")),
        "deterministic": bool(meta.get("deterministic", False)),
        # eval mode hides the terminal reward from the policy (rows carry 0.0); the grader-side value is here
        "reward_visible_to_policy": mode == MODE_TRAIN,
        "grader_reward_shared": rep.get("reward_shared"),
        "grader_f1": (rep.get("metrics") or {}).get("f1"),
        "trajectory_sha256": trajectory_hash(traj),
    }
    if rec.get("label"):
        h["label"] = rec["label"]
    return h


def write_trajectories(records: list[dict], path: str) -> int:
    """Write jsonl with keys episode_id, step, action, observation, reward, terminated, truncated,
    policy; the step-0 row of a run additionally carries ``header`` (see ``trajectory_header``).
    ``records`` may be run_episode results (each with "trajectory" and "policy") or flat
    step records that already carry "policy". Returns the number of lines written."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", errors="backslashreplace") as f:
        for rec in records:
            if "trajectory" in rec:
                steps = [dict(s, policy=rec.get("policy")) for s in rec["trajectory"]]
                head = trajectory_header(rec) if "grader_report" in rec else None
            else:
                steps = [rec]
                head = None
            for i, s in enumerate(steps):
                row = {k: s.get(k) for k in ROW_KEYS}
                if i == 0 and head is not None:
                    row["header"] = head
                f.write(dumps(row) + "\n")
                n += 1
    return n


class StallPolicy(Policy):
    """Scripted sample policy that NEVER submits: it pages the fx table forever, so the episode ends by
    truncation (step budget, or wall clock with a fake clock). Used only to ship truncated example
    trajectories; not a registered baseline."""
    name = "stall"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._cursor = None

    def reset(self, seed: int, first_obs: dict) -> None:
        self._cursor = None

    def act(self, obs: dict) -> dict:
        res = obs.get("result")
        if isinstance(res, dict) and res.get("next_cursor"):
            self._cursor = res["next_cursor"]
        elif isinstance(res, dict):
            self._cursor = None
        from .policies.base import list_action
        return list_action("fx", self._cursor, 1)


def run_truncated_sample(seed: int, config: Any, by: str = "steps", mode: str = MODE_TRAIN) -> dict:
    """A scripted run that ends ``truncated=true``. ``by="steps"``: the 100-step budget is exhausted on a
    non-submit step. ``by="time"``: a fake clock jumps past the wall-clock limit on the second step (so the
    sample is reproducible, no real waiting). Both are scored as an empty submission."""
    from .env import AuditEnv
    if by not in ("steps", "time"):
        raise ValueError("by must be 'steps' or 'time'")
    env = AuditEnv(mode=mode, time_limit_s=5.0) if by == "time" else AuditEnv(mode=mode)
    if by == "time":
        ticks = iter([0.0, 0.0, 10.0] + [10.0] * 1000)      # reset at 0, step 1 at 0, step 2 at 10 s > 5 s
        env._clock = lambda: next(ticks)
    pol = StallPolicy(0)
    t0 = time.perf_counter()
    traj, ended = _drive(env, pol, seed, config, None, 0)
    report = env.grader_report() if ended else {"ended": False}
    meta = reproducibility_meta(traj[0]["observation"], config, report)
    return {"episode_id": traj[0]["episode_id"], "trajectory": traj, "grader_report": dict(report, **meta),
            "wall_s": round(time.perf_counter() - t0, 4), "steps": len(traj) - 1,
            "policy": f"stall_truncated_by_{by}", "seed": seed, "mode": mode, "meta": meta,
            "label": f"scripted sample: never submits, ends truncated=true by the {by} limit, scored as empty"}


def trajectory_files(entries_by_split: dict[str, list[dict]], root: str, remote: bool = False) -> dict[str, list[dict]]:
    """The three shipped trajectory sets, as ``{filename: [run records]}``.

    * ``trajectories.jsonl``     train-mode runs of the non-oracle policies (random_explorer, reconcile,
                                 flag_all, careful_auditor, careful_auditor_degraded) on every open episode;
    * ``oracle_reference.jsonl`` the grader-side oracle, kept apart and labelled (it reads the truth);
    * ``samples.jsonl``          one eval-mode run (reward hidden, header carries the grader reward) and two
                                 scripted truncated runs (step budget; wall clock via a fake clock).
    """
    from .policies import EXTRA_POLICY_NAMES
    main_runs: list[dict] = []
    oracle_runs: list[dict] = []
    first: tuple[str, dict] | None = None
    for split in ("train", "dev"):
        for e in entries_by_split.get(split) or []:
            first = first or (split, e)
            cfg = resolve_config(e, root)
            for p in tuple(POLICY_NAMES) + tuple(EXTRA_POLICY_NAMES):
                r = run_episode(p, e["seed"], cfg, mode=MODE_TRAIN, remote=remote)
                r["split"] = split
                if p == "oracle":
                    r["label"] = ORACLE_LABEL
                    oracle_runs.append(r)
                else:
                    main_runs.append(r)
    samples: list[dict] = []
    if first is not None:
        split, e = first
        cfg = resolve_config(e, root)
        ev = run_episode("careful_auditor", e["seed"], cfg, mode=MODE_EVAL, remote=remote)
        ev["label"] = ("eval-mode sample: the policy-visible terminal reward is 0.0 by design; the real "
                       "reward is header.grader_reward_shared (== grader_report.reward_shared)")
        samples.append(ev)
        for by in ("steps", "time"):
            samples.append(run_truncated_sample(e["seed"], cfg, by=by))
        for r in samples:
            r["split"] = split
    return {"trajectories.jsonl": main_runs, "oracle_reference.jsonl": oracle_runs, "samples.jsonl": samples}


def write_trajectory_files(entries_by_split: dict[str, list[dict]], root: str, remote: bool = False) -> dict[str, int]:
    """Write the shipped trajectory sets into ``<root>/trajectories/``; returns ``{file: n_lines}``."""
    out = {}
    for name, runs in trajectory_files(entries_by_split, root, remote).items():
        out[name] = write_trajectories(runs, os.path.join(root, "trajectories", name))
    return out


# ------------------------------------------------------------------ golden digests (cross-platform)
GOLDEN_PATH = os.path.join(ROOT, "tests", "golden_hashes.json")
GOLDEN_POLICIES = ("reconcile", "careful_auditor")


def golden_selection(root: str | None = None) -> list[tuple[str, dict]]:
    """Which episodes carry a committed digest: 2 train, 2 dev, 2 holdout (clean + interaction forced)."""
    splits = load_splits(root)
    out: list[tuple[str, dict]] = []
    for split, n in (("train", 2), ("dev", 2)):
        out += [(split, e) for e in (splits.get(split) or [])[:n]]
    hold = splits.get("holdout") or []
    picked, seen = [], set()
    for e in hold:
        k = e.get("category") or e.get("tag")
        if k not in seen:
            seen.add(k)
            picked.append(e)
    out += [("holdout", e) for e in picked[:2]]
    return out


def golden_digests(root: str | None = None) -> dict:
    """sha256 digests of generator output and of deterministic train-mode trajectories (in-process).
    Pure functions of (seed, config, code): equal on every platform / Python build if the project is
    deterministic. Committed as ``tests/golden_hashes.json``; compared by ``test_golden_hashes``."""
    from .generator.build import generate_episode
    eps = []
    for split, e in golden_selection(root):
        cfg = resolve_config(e, root)
        ep = generate_episode(e["seed"], cfg)
        row: dict[str, Any] = {
            "split": split, "seed": e["seed"], "config_id": ep.provenance.get("config_id"),
            "config_version": ep.provenance.get("config_version"), "config_hash": config_hash(cfg),
            "episode_id": ep.episode_id, "public_hash": ep.public_hash(),
            "truth_sha256": hashlib.sha256(dumps(ep.truth).encode("utf-8")).hexdigest(),
            "provenance_sha256": hashlib.sha256(dumps(ep.provenance).encode("utf-8")).hexdigest(),
            "trajectory_sha256": {}}
        for pol in GOLDEN_POLICIES:
            r = run_episode(pol, e["seed"], cfg, mode=MODE_TRAIN, remote=False)
            row["trajectory_sha256"][pol] = trajectory_hash(r["trajectory"])
        eps.append(row)
    return {"env_version": ENV_VERSION,
            "note": "regenerate with `python -m auditgym golden --write`; any diff means output changed "
                    "(code change, or a platform/Python difference in determinism)",
            "episodes": eps}


# ------------------------------------------------------------------ scoring helpers
def report_scores(report: dict) -> dict:
    """Normalise a grader report to {f1, precision, recall, tp, fp, fn, shared, experimental, steps}."""
    m = report.get("metrics") or {}
    return {"f1": m.get("f1", 0.0), "precision": m.get("precision", 0.0),
            "recall": m.get("recall", 0.0), "tp": m.get("tp", 0), "fp": m.get("fp", 0),
            "fn": m.get("fn", 0), "shared": report.get("reward_shared"),
            "experimental": report.get("reward_experimental"),
            "inspection_steps": report.get("inspection_steps"),
            "invalid": bool(report.get("invalid_submission"))}


__all__ = ["run_episode", "replay_actions", "write_trajectories", "trajectory_hash", "load_splits",
           "resolve_config", "grader_truth", "grader_episode", "committed_truth", "config_hash",
           "reproducibility_meta", "trajectory_header", "write_trajectory_files", "trajectory_files",
           "run_truncated_sample", "golden_digests", "submitted_findings", "report_scores", "POLICY_NAMES",
           "MODE_EVAL", "MODE_TRAIN"]
