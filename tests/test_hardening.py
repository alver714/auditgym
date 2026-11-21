"""Hardening tests from the jury review: trust boundary, hostile payloads, config feasibility,
log files, baseline invariance, recording robustness, shipped-artefact consistency."""
from __future__ import annotations

import json
import os

import pytest

from auditgym import runner
from auditgym.client import RemoteAuditEnv, RemoteEnvError
from auditgym.constants import ISSUE_TYPES, MODE_EVAL, MODE_TRAIN
from auditgym.env import AuditEnv
from auditgym.generator.build import GenerationError, generate_episode
from auditgym.generator.config import ConfigError, config_with_force, resolve_config
from auditgym.validate import validate_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nested(depth: int):
    root: dict = {}
    cur = root
    for _ in range(depth):
        cur["x"] = {}
        cur = cur["x"]
    return root


def nested_list(depth: int):
    root: list = []
    cur = root
    for _ in range(depth):
        nxt: list = []
        cur.append(nxt)
        cur = nxt
    return root


# ------------------------------------------------------------------ hostile payloads (in-process)
@pytest.mark.parametrize("depth", [30, 500, 2000, 100_000])
def test_deeply_nested_action_is_a_bounded_error_that_consumes_one_step(clean_config, depth):
    env = AuditEnv(mode=MODE_TRAIN)
    env.reset(seed=3, config=clean_config)
    obs, r, term, trunc, info = env.step({"action": "calculator", "args": {"expression": "1"},
                                          "extra": nested_list(depth)})
    assert obs["error"]["code"] == "invalid_args" and r == 0.0 and not term and not trunc
    assert obs["step"] == 1 and obs["steps_remaining"] == 99
    obs2, *_ = env.step({"action": "calculator", "args": {"expression": "1+1"}})
    assert obs2["result"] == {"value": "2"} and obs2["step"] == 2
    traj = env.trajectory()
    assert [t["step"] for t in traj] == [0, 1, 2]                 # no gap in the trajectory
    assert depth < 100 or "extra" not in json.dumps(traj[1]["action"])


def test_deeply_nested_submit_is_an_invalid_submission(clean_config):
    env = AuditEnv(mode=MODE_TRAIN)
    obs0, _ = env.reset(seed=3, config=clean_config)
    sub = {"episode_id": obs0["episode_id"], "findings": [], "x": nested(5000)}
    obs, r, term, trunc, info = env.step({"action": "submit", "args": {"findings": sub}})
    assert term and r == -1.0 and info["invalid_submission"] is True
    assert env.grader_report()["errors"]


def test_oversized_action_node_count_is_bounded(clean_config):
    env = AuditEnv(mode=MODE_EVAL)
    env.reset(seed=3, config=clean_config)
    obs, r, term, _, _ = env.step({"action": "calculator", "args": {"expression": "1"}, "pad": [0] * 200_000})
    assert obs["error"]["code"] == "invalid_args" and r == 0.0 and not term


def test_remote_deep_nesting_does_not_abort_the_run(clean_config, make_remote):
    env = make_remote(mode=MODE_EVAL, allow_grader_report=True)
    env.reset(seed=3, config=clean_config)
    obs, r, term, trunc, info = env.step({"action": "calculator", "args": {"expression": "1"},
                                          "extra": nested(800)})
    assert obs["error"]["code"] == "invalid_args" and not term
    obs2, *_ = env.step({"action": "calculator", "args": {"expression": "2*3"}})
    assert obs2["result"] == {"value": "6"} and obs2["step"] == 2
    # absurdly deep python object: the client does not crash either
    obs3, *_ = env.step({"action": "calculator", "args": {"expression": "1"}, "extra": nested_list(5000)})
    assert obs3["error"] is not None


# ------------------------------------------------------------------ submit structure
@pytest.mark.parametrize("args", [None, {}, {"findings": None}, {"findings": []}, "x",
                                  {"findings": {"episode_id": "?", "findings": []}, "unknown": 1}])
def test_every_structurally_invalid_submit_ends_the_episode_with_minus_one(clean_config, args):
    for mode in (MODE_TRAIN, MODE_EVAL):
        env = AuditEnv(mode=mode)
        env.reset(seed=3, config=clean_config)
        action = {"action": "submit"} if args is None else {"action": "submit", "args": args}
        obs, r, term, trunc, info = env.step(action)
        assert term and not trunc
        assert r == (-1.0 if mode == MODE_TRAIN else 0.0)
        if mode == MODE_EVAL:
            assert obs["error"] is None and info == {}      # identical to a valid submit's terminal obs
        else:
            assert obs["error"]["code"] == "invalid_args"
        rep = env.grader_report()
        assert rep["reward_shared"] == -1.0 and rep["errors"] and rep["inspection_steps"] == 0


def test_invalid_submit_as_hundredth_step_is_invalid_not_truncated(clean_config):
    env = AuditEnv(mode=MODE_TRAIN)
    env.reset(seed=3, config=clean_config)
    for _ in range(99):
        env.step({"action": "list_records", "args": {"table": "fx", "limit": 1}})
    obs, r, term, trunc, info = env.step({"action": "submit", "args": {}})
    assert term and not trunc and r == -1.0


def test_time_and_step_truncation_count_the_final_step_alike(clean_config):
    env = AuditEnv(mode=MODE_TRAIN)
    env.reset(seed=3, config=clean_config)
    for _ in range(100):
        last = env.step({"action": "list_records", "args": {"table": "fx", "limit": 1}})
    steps_inspection = last[4]["inspection_steps"]

    env2 = AuditEnv(mode=MODE_TRAIN, time_limit_s=5.0)
    ticks = iter(range(0, 10**6, 100))
    env2._clock = lambda: next(ticks)
    env2.reset(seed=3, config=clean_config)
    for _ in range(3):
        pass
    obs, r, term, trunc, info = env2.step({"action": "calculator", "args": {"expression": "1"}})
    assert trunc and info["inspection_steps"] == 1      # the step that hit the limit counts, like step 100
    assert steps_inspection == 100


def test_abandon_scores_an_unfinished_episode_as_empty(clean_config):
    env = AuditEnv(mode=MODE_EVAL)
    env.reset(seed=3, config=clean_config)
    env.step({"action": "calculator", "args": {"expression": "1"}})
    assert env.in_progress()
    env.abandon()
    rep = env.grader_report()
    assert rep["ended"] and rep["truncated"] and rep["ended_by"] == "abandoned"
    assert rep["metrics"]["f1"] == 1.0                    # clean book, empty answer
    assert not env.in_progress()


# ------------------------------------------------------------------ trust boundary (server/client)
def test_reset_and_grader_report_need_the_harness_token(clean_config, make_remote):
    env = make_remote(mode=MODE_EVAL, allow_grader_report=True)
    env.reset(seed=3, config=clean_config)
    for cmd in ({"cmd": "reset", "seed": 1, "config": "train"}, {"cmd": "grader_report"},
                {"cmd": "reset", "seed": 1, "config": "train", "token": "guess"},
                {"cmd": "reset", "seed": 1, "config": "train", "token": 12345}):
        with pytest.raises(RemoteEnvError, match="harness command"):
            env._call(cmd)
    # the episode was not touched by the refused resets
    obs, *_ = env.step({"action": "calculator", "args": {"expression": "1"}})
    assert obs["step"] == 1
    # the harness itself still works
    env.step({"action": "submit", "args": {"findings": {"episode_id": obs["episode_id"], "findings": []}}})
    assert env.grader_report()["ended"]


def test_agent_handle_exposes_step_only(clean_config, make_remote):
    env = make_remote(mode=MODE_EVAL, allow_grader_report=True)
    env.reset(seed=3, config=clean_config)
    h = env.agent_handle()
    assert not hasattr(h, "reset") and not hasattr(h, "grader_report") and not hasattr(h, "_call")
    obs, *_ = h.step({"action": "calculator", "args": {"expression": "7*6"}})
    assert obs["result"] == {"value": "42"}


def test_each_server_gets_its_own_token(make_remote):
    a, b = make_remote(mode=MODE_EVAL), make_remote(mode=MODE_EVAL)
    assert a._token != b._token and len(a._token) >= 32


def test_agent_cannot_point_the_grader_at_its_own_truth_files(clean_config, make_remote, tmp_path):
    env = make_remote(mode=MODE_EVAL, allow_grader_report=True)
    env.reset(seed=3, config=clean_config)
    forged = {"cmd": "reset", "config": {"episode_dir": str(tmp_path), "grader_dir": str(tmp_path)}}
    with pytest.raises(RemoteEnvError, match="harness command"):
        env._call(forged)


def test_failed_reset_replies_do_not_disclose_paths_or_exception_text(make_remote):
    env = make_remote(mode=MODE_EVAL)
    for cfg in ({"episode_dir": "/etc"}, "/tmp/../etc/passwd", {"episode_dir": "/nonexistent/dir"}):
        with pytest.raises(RemoteEnvError) as ei:
            env.reset(seed=1, config=cfg)
        msg = str(ei.value)
        assert "/etc" not in msg and "nonexistent" not in msg and "Errno" not in msg and "JSONDecode" not in msg
        assert msg == "reset failed (invalid seed, config or options)"
    # our own validation errors are passed on (their text is controlled) so a bad config is diagnosable
    bad = resolve_config("holdout")
    bad["company"]["customer_offsets"] = ["+00:00"]
    bad["company"]["timezones"] = [["UTC", "+00:00"]]
    with pytest.raises(RemoteEnvError, match="ConfigError"):
        env.reset(seed=1, config=bad)
    # the server survived all of that
    obs, _ = env.reset(seed=1, config="train")
    assert obs["step"] == 0


def test_reset_of_an_unfinished_episode_is_scored_and_logged(clean_config, tmp_path):
    env = RemoteAuditEnv(mode=MODE_EVAL, log_dir=str(tmp_path), allow_grader_report=True)
    try:
        env.reset(seed=3, config=clean_config)
        env.step({"action": "calculator", "args": {"expression": "1"}})
        env.reset(seed=4, config=clean_config)           # abandons the first episode
        env.step({"action": "submit", "args": {}})        # invalid structure ends the second one
    finally:
        env.close()
    reports = [json.load(open(tmp_path / f)) for f in sorted(os.listdir(tmp_path)) if f.startswith("report_")]
    assert [r["ended_by"] for r in reports] == ["abandoned", "submit"]
    assert reports[1]["invalid_submission"] and reports[1]["errors"]


def test_runner_passes_log_dir_to_the_grader(clean_config, tmp_path):
    r = runner.run_episode("reconcile", 3, clean_config, mode=MODE_EVAL, log_dir=str(tmp_path))
    files = [f for f in os.listdir(tmp_path) if f.startswith("report_")]
    assert len(files) == 1 and r["episode_id"] in files[0]
    rep = json.load(open(tmp_path / files[0]))
    assert rep["reward_shared"] == r["grader_report"]["reward_shared"]


# ------------------------------------------------------------------ config feasibility
def test_customer_offsets_equal_to_the_company_offset_is_rejected_at_validation():
    cfg = resolve_config("holdout")
    cfg["company"]["customer_offsets"] = ["+00:00"]
    cfg["company"]["timezones"] = [["UTC", "+00:00"]]
    with pytest.raises(ConfigError, match="differing"):
        generate_episode(1, cfg)
    with pytest.raises(ConfigError):
        resolve_config(cfg)


def test_pinned_timezone_without_a_differing_customer_offset_is_rejected():
    cfg = config_with_force("train", {"company": {"timezone": ["Asia/Tokyo", "+09:00"],
                                                  "customer_offsets": ["+09:00"]}})
    with pytest.raises(ConfigError, match="differing"):
        generate_episode(1, cfg)


@pytest.mark.parametrize("key,val", [("n_invoices", [300, 500]), ("n_invoices", [2, 10]),
                                     ("n_noise_postings", [5000, 6000]), ("n_customers", [1, 3])])
def test_sizes_outside_the_supported_range_raise_instead_of_clamping_silently(key, val):
    cfg = resolve_config("train")
    cfg["company"][key] = val
    with pytest.raises(ConfigError):
        generate_episode(1, cfg)


def test_forced_sizes_outside_the_supported_range_raise():
    for pin in ({"n_invoices": 500}, {"n_invoices": 3}, {"n_noise_postings": 5000}):
        with pytest.raises(ConfigError):
            generate_episode(1, config_with_force("train", {"issues": {"fx_error": 1}, "company": pin}))


def test_requested_vs_effective_noise_is_recorded_in_provenance():
    cfg = resolve_config("train")
    cfg["company"]["n_noise_postings"] = [900, 1000]
    cfg["company"]["n_invoices"] = [85, 90]
    ep = generate_episode(5, cfg)
    sp = ep.provenance["spec"]
    assert sp["n_noise_postings"] >= 900 and 0 < sp["n_noise_effective"] < sp["n_noise_postings"]
    assert len(ep.ledger) <= 2000


CAPACITY_CASES = [
    {"issues": {"period_cutoff": 12}},
    {"issues": {"period_cutoff": 9}},
    {"interactions": {"cutoff_with_wrong_rate_date": 6}},
    {"interactions": {"cutoff_with_wrong_rate_date": 5}},
    {"issues": {"period_cutoff": 4}, "interactions": {"cutoff_with_wrong_rate_date": 4}},
    {"issues": {"period_cutoff": 7}, "interactions": {"cutoff_with_wrong_rate_date": 2}},
]


@pytest.mark.parametrize("force", CAPACITY_CASES)
@pytest.mark.parametrize("split", ["dev", "holdout"])
def test_forced_cutoff_counts_up_to_the_issue_cap_are_generatable_and_valid(split, force):
    cfg = config_with_force(split, force)
    for seed in (90000, 90001, 90002, 90003, 90004, 90005):
        ep = generate_episode(seed, cfg)
        res = validate_episode(ep)
        assert res["ok"], (seed, force, {k: v["detail"] for k, v in res["checks"].items() if not v["ok"]})
        want = force.get("issues", {}).get("period_cutoff", 0) + \
            force.get("interactions", {}).get("cutoff_with_wrong_rate_date", 0)
        got = sum(1 for i in ep.truth["issues"] if i["type"] == "period_cutoff")
        assert got == want
        assert len(ep.invoices) + len(ep.payments) <= 200


# ------------------------------------------------------------------ distractor documentation
def test_policy_documents_the_foreign_refund_fx_residual():
    pol = generate_episode(1, "train").policy
    txt = pol["postings"]["refund"]
    assert "residual" in txt and "NOT an error" in txt


# ------------------------------------------------------------------ baselines are order/id invariant
@pytest.mark.parametrize("seed", [9556, 9001, 9186, 9300])
def test_reconcile_and_careful_auditor_are_invariant_to_ids_and_order(seed):
    from auditgym.runner import load_splits
    splits = load_splits(ROOT)
    entry = next(e for e in splits["holdout"] if e["seed"] == seed) if any(
        e["seed"] == seed for e in splits["holdout"]) else splits["holdout"][0]
    cfg = runner.resolve_config(entry)
    for pol in ("reconcile", "careful_auditor"):
        base = runner.run_episode(pol, entry["seed"], cfg, mode=MODE_EVAL, remote=False)["grader_report"]
        for opts in ({"transform": {"permute_salt": 12}}, {"transform": {"rename_salt": 5}},
                     {"transform": {"rename_salt": 6, "permute_salt": 7}}):
            alt = runner.run_episode(pol, entry["seed"], cfg, mode=MODE_EVAL, options=opts,
                                     remote=False)["grader_report"]
            assert (alt["metrics"], alt["reward_shared"]) == (base["metrics"], base["reward_shared"]), (pol, opts)


# ------------------------------------------------------------------ recording robustness
def test_lone_surrogates_in_actions_do_not_break_trajectory_recording(tmp_path, clean_config):
    env = AuditEnv(mode=MODE_EVAL)
    obs, _ = env.reset(seed=3, config=clean_config)
    traj = [{"episode_id": obs["episode_id"], "step": 0, "action": None, "observation": obs,
             "reward": 0.0, "terminated": False, "truncated": False}]
    action = {"action": "get_record", "args": {"table": "ledger", "id": "\ud800x"}}
    o, r, te, tr, _ = env.step(action)
    traj.append({"episode_id": o["episode_id"], "step": o["step"], "action": action, "observation": o,
                 "reward": r, "terminated": te, "truncated": tr})
    h1 = runner.trajectory_hash(traj)
    assert h1 == runner.trajectory_hash(traj) and len(h1) == 64
    n = runner.write_trajectories([{"trajectory": traj, "policy": "t"}], str(tmp_path / "t.jsonl"))
    assert n == 2


# ------------------------------------------------------------------ CLI
def test_run_and_demo_accept_the_careful_auditor_extra():
    from auditgym.__main__ import build_parser
    p = build_parser()
    a = p.parse_args(["run", "--policy", "careful_auditor", "--seed", "1", "--config", "train"])
    assert a.policy == "careful_auditor"
    a = p.parse_args(["demo", "--policy", "careful_auditor"])
    assert a.policy == "careful_auditor"


# ------------------------------------------------------------------ shipped-artefact consistency
def _ignore_entries(name: str) -> set[str]:
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}


def test_dockerignore_keeps_everything_the_official_check_needs():
    # `docker run ... check` runs pytest, which reads the committed open-split truth (grader_data/)
    ignored = {e.rstrip("/") for e in _ignore_entries(".dockerignore")}
    for needed in ("grader_data", "episodes", "configs", "trajectories", "tests", "auditgym", "docs",
                   "schemas"):
        assert needed not in ignored, needed
    assert os.path.isdir(os.path.join(ROOT, "grader_data", "train"))
    assert os.path.isdir(os.path.join(ROOT, "grader_data", "dev"))
    assert not os.path.exists(os.path.join(ROOT, "grader_data", "holdout"))


def test_shipped_configs_carry_a_version_and_no_salt():
    for name in ("train", "dev", "holdout"):
        cfg = resolve_config(name)
        assert "id_salt" not in cfg and cfg["config_version"] == "1.3.0"


def test_requirements_are_pinned_and_the_docker_base_image_has_an_exact_patch_tag():
    with open(os.path.join(ROOT, "requirements-dev.txt"), encoding="utf-8") as f:
        reqs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    assert reqs and all("==" in r for r in reqs)
    assert any(r.startswith("pytest==") for r in reqs) and any(r.startswith("pluggy==") for r in reqs)
    with open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8") as f:
        first = next(ln for ln in f if ln.startswith("FROM"))
    import re
    assert re.fullmatch(r"FROM python:3\.11\.\d+-slim\s*", first), first


# ================================================================== core hardening round 2 (jury cluster)
import base64
import hashlib
import io
import subprocess
import sys
import time

from auditgym import calculator
from auditgym.client import AgentHandle
from auditgym.env import GraderReportNotReady
from auditgym.server import Server, TOKEN_ENV


def _lines(*msgs) -> io.StringIO:
    return io.StringIO("".join((m if isinstance(m, str) else json.dumps(m)) + "\n" for m in msgs))


def _serve(server: Server, *msgs) -> list[dict]:
    out = io.StringIO()
    server.serve(_lines(*msgs), out)
    return [json.loads(line) for line in out.getvalue().splitlines()]


# ---- 1. server fails closed without a token
def test_server_without_token_refuses_reset_and_grader_report_over_the_wire(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    srv = Server(MODE_EVAL, None, True)
    r = _serve(srv, {"cmd": "reset", "seed": 1, "config": "train"},
               {"cmd": "reset", "seed": 2, "config": "train"}, {"cmd": "grader_report"},
               {"cmd": "step", "action": {"action": "calculator", "args": {"expression": "1"}}})
    assert [x["ok"] for x in r] == [False, False, False, True]
    assert "harness command" in r[0]["error"] and r[3]["obs"]["error"]["code"] == "not_reset"


def test_server_without_token_subprocess_fails_closed_and_insecure_open_is_explicit():
    def run(*extra):
        p = subprocess.run([sys.executable, "-m", "auditgym.server", "--mode", "eval", "--allow-grader-report",
                            *extra], cwd=ROOT, capture_output=True, text=True, timeout=60,
                           env={k: v for k, v in os.environ.items() if k != TOKEN_ENV}
                           | {"PYTHONPATH": ROOT, "PYTHONDONTWRITEBYTECODE": "1"},
                           input='{"cmd":"reset","seed":1,"config":"train"}\n'
                                 '{"cmd":"reset","seed":2,"config":"train"}\n{"cmd":"close"}\n')
        return [json.loads(x)["ok"] for x in p.stdout.splitlines()]
    assert run() == [False, False, False]          # close is a harness command too; the server stops at EOF
    assert run("--insecure-open") == [True, True, True]


@pytest.mark.parametrize("msg", [
    {"seed": 1, "config": {"episode_dir": "/tmp", "grader_dir": "/tmp"}},
    {"seed": 1, "config": {"grader_dir": "/tmp"}},
    {"seed": 1, "config": {"surprise": 1}},
    {"seed": 1, "config": "/etc/passwd"},
    {"seed": 1, "config": "../configs/train"},
    {"seed": 1, "config": 5},
    {"seed": -1, "config": "train"},
    {"seed": 2 ** 63, "config": "train"},
    {"seed": True, "config": "train"},
    {"seed": 1.5, "config": "train"},
    {"seed": "1", "config": "train"},
    {"seed": 1, "config": "train", "options": {"surprise": 1}},
    {"seed": 1, "config": "train", "options": {"transform": {"surprise": 1}}},
    {"seed": 1, "config": "train", "options": "x"},
])
def test_wire_reset_rejects_paths_unknown_keys_and_out_of_range_seeds(monkeypatch, msg):
    monkeypatch.setenv(TOKEN_ENV, "tok")
    srv = Server(MODE_EVAL, None, False)
    r = _serve(srv, {"cmd": "reset", "token": "tok", **msg})
    assert r[0] == {"ok": False, "error": "reset failed (invalid seed, config or options)"}
    assert srv.env.in_progress() is False                  # nothing was loaded


def test_wire_reset_accepts_the_documented_forms(monkeypatch, clean_config):
    monkeypatch.setenv(TOKEN_ENV, "tok")
    srv = Server(MODE_EVAL, None, False)
    r = _serve(srv, {"cmd": "reset", "token": "tok", "seed": 0, "config": "train"},
               {"cmd": "reset", "token": "tok", "seed": 2 ** 63 - 1, "config": clean_config},
               {"cmd": "reset", "token": "tok", "options": {"transform": {"rename_salt": "s"}}})
    assert [x["ok"] for x in r] == [True, True, True]


def test_server_rejects_duplicate_json_keys(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "tok")
    srv = Server(MODE_EVAL, None, False)
    r = _serve(srv, '{"cmd":"reset","token":"tok","seed":1,"seed":2,"config":"train"}',
               '{"cmd":"step","action":{"action":"calculator","action":"submit"}}')
    assert [x["ok"] for x in r] == [False, False]
    assert all("duplicate" in x["error"] for x in r)


# ---- 2. AgentHandle holds no env reference
def test_agent_handle_has_no_path_to_the_env_through_attributes(clean_config, make_remote):
    env = make_remote(mode=MODE_EVAL, allow_grader_report=True)
    env.reset(seed=3, config=clean_config)
    h = env.agent_handle()
    assert type(h) is AgentHandle and not hasattr(h, "__dict__")
    assert not hasattr(h, "_step") and not hasattr(h, "_env") and not hasattr(h, "_call")
    assert not hasattr(h.step, "__self__")                  # a plain function, not a bound method of env
    for name in ("step", "reset", "_token", "x"):
        with pytest.raises(AttributeError):
            setattr(h, name, 1)
    with pytest.raises(AttributeError):
        del h.step
    assert h.step({"action": "calculator", "args": {"expression": "2+2"}})[0]["result"] == {"value": "4"}


# ---- 3. truth / grader state sealed in-process
def test_in_process_env_keeps_truth_out_of_its_episode_and_refuses_early_grader_report(issue_config):
    env = AuditEnv(mode=MODE_EVAL)
    env.reset(seed=5, config=issue_config)
    assert env._ep.truth == {} and env._ep.provenance == {}
    with pytest.raises(GraderReportNotReady):
        env.grader_report()
    env.step({"action": "calculator", "args": {"expression": "1"}})
    with pytest.raises(GraderReportNotReady):
        env.grader_report()
    obs, *_ = env.step({"action": "submit", "args": {"findings": {"episode_id": env._ep.episode_id,
                                                                   "findings": []}}})
    rep = env.grader_report()
    assert rep["ended"] and rep["n_truth_issues"] >= 1 and rep["metrics"]["fn"] >= 1


# ---- 4. hostile values never break recording
def _submit_with(env, **finding_overrides):
    base = {"type": "fx_error", "affected_ids": ["x"], "claimed_amount_minor": 1, "currency": "KZT",
            "evidence_ids": [], "explanation": "e"}
    base.update(finding_overrides)
    sub = {"episode_id": env._ep.episode_id, "findings": [base]}
    return env.step({"action": "submit", "args": {"findings": sub}})


def _strict_json_utf8(traj) -> bytes:
    return json.dumps(traj, allow_nan=False, ensure_ascii=False).encode("utf-8")      # must not raise


@pytest.mark.parametrize("override", [
    {"claimed_amount_minor": 10 ** 5000},
    {"claimed_amount_minor": 10 ** 18 + 1},
    {"affected_ids": ["\ud800"]},
    {"evidence_ids": ["ok", "\udfff"]},
    {"explanation": "bad \ud83d surrogate"},
    {"currency": "\ud800"},
])
def test_hostile_submission_values_are_invalid_and_trajectory_stays_json_safe(clean_config, override):
    env = AuditEnv(mode=MODE_TRAIN)
    env.reset(seed=3, config=clean_config)
    obs, r, term, trunc, info = _submit_with(env, **override)
    assert term and r == -1.0 and info["invalid_submission"] is True
    _strict_json_utf8(env.trajectory())
    _strict_json_utf8(env.grader_report())
    json.dumps(env.trajectory())                       # the default encoder (4300-digit limit) works too


def test_claimed_amount_upper_bound_is_inclusive_at_1e18(clean_config):
    from auditgym import matcher
    f = {"type": "fx_error", "affected_ids": ["x"], "claimed_amount_minor": 10 ** 18, "currency": "KZT",
         "evidence_ids": [], "explanation": ""}
    ok, err, _ = matcher.validate_submission({"episode_id": "e", "findings": [f]}, "e")
    assert ok, err
    ok, err, _ = matcher.validate_submission({"episode_id": "e", "findings": [dict(f, claimed_amount_minor=10 ** 18 + 1)]}, "e")
    assert not ok and "<=" in err


def test_huge_and_non_finite_values_in_actions_are_recorded_as_placeholders(clean_config):
    env = AuditEnv(mode=MODE_TRAIN)
    env.reset(seed=3, config=clean_config)
    env.step({"action": "list_records", "args": {"table": "fx", "limit": 10 ** 5000}})
    env.step({"action": "list_records", "args": {"table": "fx", "limit": float("nan"), "cursor": float("inf")}})
    env.step({"action": "get_record", "args": {"table": "fx", "id": "\ud800" * 5}})
    env.step({"action": "get_record", "args": {"table": "fx", "id": "z" * 150_000}})
    env.step({"action": "bogus" * 10})
    traj = env.trajectory()
    _strict_json_utf8(traj)
    assert all(t["observation"]["error"] is None or len(t["observation"]["error"]["message"]) <= 120
               for t in traj)
    assert len(json.dumps(traj)) < 20_000              # the 150k-char id is truncated in the record


def test_ten_megabyte_action_name_is_a_bounded_error_and_not_stored(clean_config):
    env = AuditEnv(mode=MODE_TRAIN)
    env.reset(seed=3, config=clean_config)
    obs, r, term, trunc, info = env.step({"action": "x" * 10_000_000})
    assert obs["error"]["code"] == "invalid_args" and not term and not trunc and obs["step"] == 1
    assert len(json.dumps(env.trajectory())) < 20_000


def test_total_string_budget_applies_across_many_small_strings(clean_config):
    env = AuditEnv(mode=MODE_TRAIN)
    obs0, _ = env.reset(seed=3, config=clean_config)
    findings = [{"type": "fx_error", "affected_ids": ["a" * 1000] * 10, "claimed_amount_minor": 1,
                 "currency": "KZT", "evidence_ids": ["b" * 1000] * 10, "explanation": "c" * 1000}] * 30
    obs, r, term, trunc, info = env.step({"action": "submit", "args": {"findings": {
        "episode_id": obs0["episode_id"], "findings": findings}}})
    assert term and r == -1.0                          # > 200k chars in total: rejected as too large
    _strict_json_utf8(env.trajectory())


def test_hostile_action_on_time_limit_step_does_not_break_recording(clean_config):
    env = AuditEnv(mode=MODE_TRAIN, time_limit_s=1.0)
    env._clock = lambda: 10_000.0 if env._step_n >= 1 else 0.0
    env.reset(seed=3, config=clean_config)
    obs, r, term, trunc, info = env.step({"action": "calculator", "args": nested_list(100_000)})
    assert trunc and not term
    _strict_json_utf8(env.trajectory())


def test_write_trajectories_survives_hostile_run(clean_config, tmp_path):
    env = AuditEnv(mode=MODE_TRAIN)
    env.reset(seed=3, config=clean_config)
    _submit_with(env, claimed_amount_minor=10 ** 5000, affected_ids=["\ud800"])
    rec = {"episode_id": env._ep.episode_id, "policy": "x", "trajectory": env.trajectory(),
           "grader_report": env.grader_report()}
    n = runner.write_trajectories([rec], str(tmp_path / "t.jsonl"))
    assert n >= 1


# ---- 5/6. watchdog and distinct time-limit signalling
def test_server_watchdog_truncates_a_stalled_episode_and_logs_it(monkeypatch, tmp_path, clean_config):
    monkeypatch.setenv(TOKEN_ENV, "tok")
    srv = Server(MODE_EVAL, str(tmp_path), True, time_limit_s=0.3)
    try:
        r = srv.handle({"cmd": "reset", "token": "tok", "seed": 3, "config": clean_config})[0]
        assert r["ok"]
        deadline = time.monotonic() + 5
        while srv.env.in_progress() and time.monotonic() < deadline:
            time.sleep(0.05)                                   # no step ever arrives
        assert srv.env.ended()
        rep = srv.handle({"cmd": "grader_report", "token": "tok"})[0]["report"]
        assert rep["ended_by"] == "time" and rep["truncated"] and not rep["terminated"]
        assert [f for f in os.listdir(tmp_path) if f.startswith("report_")]      # logged
        late = srv.handle({"cmd": "step", "action": {"action": "calculator", "args": {"expression": "1"}}})[0]
        assert late["ok"] and late["obs"]["error"]["code"] == "episode_over" and late["truncated"] is True
        assert late["obs"]["error"]["note"] == "already_over"
    finally:
        srv._disarm()


def test_remote_policy_that_stalls_is_truncated_by_the_server(clean_config, tmp_path):
    env = RemoteAuditEnv(mode=MODE_EVAL, log_dir=str(tmp_path), allow_grader_report=True, time_limit_s=0.4)
    try:
        env.reset(seed=3, config=clean_config)
        time.sleep(1.2)                                         # the policy stalls; nothing is sent
        files = [f for f in os.listdir(tmp_path) if f.startswith("report_")]
        assert len(files) == 1                                  # logged without any further message
        assert json.load(open(tmp_path / files[0]))["ended_by"] == "time"
        obs, r, term, trunc, info = env.step({"action": "calculator", "args": {"expression": "1"}})
        assert trunc and obs["error"]["code"] == "episode_over"
    finally:
        env.close()


def test_time_limit_truncating_step_is_distinguishable_from_already_over(clean_config):
    env = AuditEnv(mode=MODE_TRAIN, time_limit_s=5.0)
    t = [0.0]
    env._clock = lambda: t[0]
    env.reset(seed=3, config=clean_config)
    t[0] = 100.0
    ending = env.step({"action": "calculator", "args": {"expression": "1"}})
    later = env.step({"action": "calculator", "args": {"expression": "1"}})
    assert ending[3] and later[3]
    assert ending[0]["error"]["code"] == later[0]["error"]["code"] == "episode_over"
    assert ending[0]["error"]["note"] == "ended_by_this_step" and later[0]["error"]["note"] == "already_over"
    assert ending[0]["step"] == 1 and later[0]["step"] == 1


def test_client_read_timeout_closes_a_server_that_does_not_answer():
    env = RemoteAuditEnv(mode=MODE_EVAL, read_timeout_s=0.0)
    with pytest.raises(RemoteEnvError, match="did not answer"):
        env.reset(seed=1, config="train")
    with pytest.raises(RemoteEnvError, match="closed"):
        env.step({"action": "calculator", "args": {"expression": "1"}})


# ---- 7. small contract items
def test_top_level_step_keys_other_than_action_and_args_are_rejected(clean_config):
    env = AuditEnv(mode=MODE_TRAIN)
    env.reset(seed=3, config=clean_config)
    obs, r, term, trunc, info = env.step({"action": "calculator", "args": {"expression": "1"}, "args2": {}})
    assert obs["error"]["code"] == "invalid_args" and "unexpected key" in obs["error"]["message"]
    assert not term and obs["step"] == 1
    obs, r, term, trunc, info = env.step({"action": "submit", "args": {"findings": {
        "episode_id": obs["episode_id"], "findings": []}}, "cmd": "reset"})
    assert term and r == -1.0 and info["invalid_submission"] is True        # smuggled key = malformed submit


def test_in_process_episode_dir_load_errors_are_path_free_config_errors():
    env = AuditEnv(mode=MODE_EVAL)
    for cfg in ({"episode_dir": "/nonexistent/secret/path"}, {"episode_dir": "/etc", "grader_dir": "/etc"}):
        with pytest.raises(ConfigError) as ei:
            env.reset(seed=1, config=cfg)
        assert "/nonexistent" not in str(ei.value) and "/etc" not in str(ei.value)
        assert ei.value.__cause__ is None and ei.value.__suppress_context__


@pytest.mark.parametrize("expr", ["1e-100*1e-100", "1e-100/1e100", "(1e-100)**2", "1e-60*1e-60*1e-60"])
def test_calculator_nonzero_operands_underflowing_to_zero_are_out_of_range(expr):
    with pytest.raises(calculator.CalculatorError, match="out of range"):
        calculator.evaluate(expr)


def test_calculator_genuine_zero_results_are_still_fine():
    assert calculator.evaluate("0*1e-100") == "0"
    assert calculator.evaluate("0/5") == "0"
    assert calculator.evaluate("1e-50*1e-50") != "0"       # representable: not an underflow


# ---- 8. keyed cursors
def test_pagination_cursor_cannot_be_forged_from_public_values(issue_config):
    from auditgym.constants import ENV_VERSION
    env = AuditEnv(mode=MODE_EVAL)
    obs, _ = env.reset(seed=3, config=issue_config)
    real = env.step({"action": "list_records", "args": {"table": "ledger", "limit": 5}})[0]["result"]["next_cursor"]
    _, off, _chk = base64.urlsafe_b64decode(real + "=" * (-len(real) % 4)).decode().split(":")
    old_sum = hashlib.sha256(f"{obs['episode_id']}|{ENV_VERSION}|ledger|{off}".encode()).hexdigest()[:10]
    forged = base64.urlsafe_b64encode(f"ledger:{off}:{old_sum}".encode()).decode().rstrip("=")
    o = env.step({"action": "list_records", "args": {"table": "ledger", "cursor": forged}})[0]
    assert o["error"]["code"] == "invalid_cursor"
    # real cursors still work, are deterministic per episode and are not valid in another episode
    assert env.step({"action": "list_records", "args": {"table": "ledger", "cursor": real}})[0]["error"] is None
    env2 = AuditEnv(mode=MODE_EVAL)
    env2.reset(seed=3, config=issue_config)
    assert env2.step({"action": "list_records", "args": {"table": "ledger", "limit": 5}})[0]["result"]["next_cursor"] == real
    env3 = AuditEnv(mode=MODE_EVAL)
    env3.reset(seed=4, config=issue_config)
    assert env3.step({"action": "list_records", "args": {"table": "ledger", "cursor": real}})[0]["error"]["code"] == "invalid_cursor"
