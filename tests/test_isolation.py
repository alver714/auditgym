"""Regression tests for the grader-robustness and isolation fixes:

* a malformed control token (lone surrogate) or invalid UTF-8 never stops the server;
* `close` is a harness command, and an episode still running when the server stops is scored and logged;
* `check` fails when the pytest suite cannot run (unless --allow-skip-tests);
* rules.check_evidence rejects an incomplete citation of the posted ledger rows;
* `python -m auditgym isolate` runs the agent in its own process/uid and survives a hostile agent.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest

from auditgym import rules
from auditgym.constants import ACC_AR, MODE_EVAL
from auditgym.server import TOKEN_ENV, Server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_ROOT = os.name == "posix" and os.geteuid() == 0
needs_root = pytest.mark.skipif(not IS_ROOT, reason="sandbox 'uid' needs root (run inside the Docker image)")


def _serve(server: Server, *lines: str) -> list[dict]:
    out = io.StringIO()
    server.serve(io.StringIO("".join(x + "\n" for x in lines)), out)
    return [json.loads(x) for x in out.getvalue().splitlines()]


def _server_proc(*extra: str, token: str | None = "tok", stdin: bytes) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != TOKEN_ENV} | {"PYTHONPATH": ROOT, "PYTHONDONTWRITEBYTECODE": "1"}
    if token is not None:
        env[TOKEN_ENV] = token
    return subprocess.run([sys.executable, "-m", "auditgym.server", "--mode", "eval", *extra], cwd=ROOT, env=env,
                          input=stdin, capture_output=True, timeout=120)


# ------------------------------------------------------------------ 1. malformed token / bytes
@pytest.mark.parametrize("tok", ['"\\ud800"', '"\\udfff\\ud800"', '"' + "a" * 5000 + '"', "null", "[]", "{}", "1.5"])
def test_malformed_token_is_refused_and_server_keeps_running(monkeypatch, tok):
    monkeypatch.setenv(TOKEN_ENV, "tok")
    srv = Server(MODE_EVAL, None, True)
    r = _serve(srv, '{"cmd":"reset","seed":1,"config":"train","token":' + tok + '}',
               '{"cmd":"grader_report","token":' + tok + '}', '{"cmd":"close","token":' + tok + '}',
               '{"cmd":"reset","seed":1,"config":"train","token":"tok"}')
    assert [x["ok"] for x in r] == [False, False, False, True]
    assert all("harness command" in x["error"] for x in r[:3])


def test_surrogate_token_does_not_kill_the_server_process():
    p = _server_proc(stdin=b'{"cmd":"reset","token":"\\ud800"}\n'
                           b'\xff\xfe not utf-8 at all\n'
                           b'{"cmd":"reset","seed":1,"config":"train","token":"tok"}\n'
                           b'{"cmd":"step","action":{"action":"calculator","args":{"expression":"6*7"}}}\n'
                           b'{"cmd":"close","token":"tok"}\n')
    assert p.returncode == 0, p.stderr.decode()
    replies = [json.loads(x) for x in p.stdout.decode().splitlines()]
    assert [x["ok"] for x in replies] == [False, False, True, True, True]
    assert replies[3]["obs"]["result"] == {"value": "42"}
    assert b"Traceback" not in p.stderr


# ------------------------------------------------------------------ 2. close / abandon / EOF
def test_close_without_token_is_refused_and_the_episode_continues(monkeypatch, tmp_path):
    monkeypatch.setenv(TOKEN_ENV, "tok")
    srv = Server(MODE_EVAL, str(tmp_path), True)
    r = _serve(srv, '{"cmd":"reset","seed":1,"config":"train","token":"tok"}', '{"cmd":"close"}',
               '{"cmd":"close","token":"wrong"}',
               '{"cmd":"step","action":{"action":"calculator","args":{"expression":"1"}}}')
    assert [x["ok"] for x in r] == [True, False, False, True]
    assert r[3]["obs"]["step"] == 1
    # the loop only ended at EOF; the running episode was scored empty and logged
    reps = [json.load(open(tmp_path / f)) for f in os.listdir(tmp_path) if f.startswith("report_")]
    assert [x["ended_by"] for x in reps] == ["disconnected"]


def test_authorised_close_scores_and_logs_a_running_episode(monkeypatch, tmp_path):
    monkeypatch.setenv(TOKEN_ENV, "tok")
    srv = Server(MODE_EVAL, str(tmp_path), True)
    r = _serve(srv, '{"cmd":"reset","seed":1,"config":"train","token":"tok"}', '{"cmd":"close","token":"tok"}',
               '{"cmd":"step","action":null}')
    assert [x["ok"] for x in r] == [True, True]                 # nothing is read after close
    assert srv.env.ended()
    reps = [json.load(open(tmp_path / f)) for f in os.listdir(tmp_path) if f.startswith("report_")]
    assert len(reps) == 1 and reps[0]["ended_by"] == "closed" and reps[0]["truncated"]


def test_close_after_a_finished_episode_logs_it_once(monkeypatch, tmp_path):
    monkeypatch.setenv(TOKEN_ENV, "tok")
    first = _serve(Server(MODE_EVAL, None, False), '{"cmd":"reset","seed":1,"config":"train","token":"tok"}')
    ep_id = first[0]["obs"]["episode_id"]
    monkeypatch.setenv(TOKEN_ENV, "tok")                       # each Server takes the token out of the env
    srv = Server(MODE_EVAL, str(tmp_path), True)
    sub = {"cmd": "step", "action": {"action": "submit", "args": {"findings": {"episode_id": ep_id, "findings": []}}}}
    r = _serve(srv, '{"cmd":"reset","seed":1,"config":"train","token":"tok"}', json.dumps(sub),
               '{"cmd":"close","token":"tok"}')
    assert [x["ok"] for x in r] == [True, True, True]
    reps = [json.load(open(tmp_path / f)) for f in os.listdir(tmp_path) if f.startswith("report_")]
    assert [x["ended_by"] for x in reps] == ["submit"]


def test_abandon_is_a_harness_command(monkeypatch, tmp_path):
    monkeypatch.setenv(TOKEN_ENV, "tok")
    srv = Server(MODE_EVAL, str(tmp_path), True)
    r = _serve(srv, '{"cmd":"abandon","token":"tok"}', '{"cmd":"reset","seed":1,"config":"train","token":"tok"}',
               '{"cmd":"abandon"}', '{"cmd":"abandon","token":"tok"}', '{"cmd":"grader_report","token":"tok"}')
    assert [x["ok"] for x in r] == [False, True, False, True, True]
    assert r[4]["report"]["ended_by"] == "agent_failed"


def test_remote_client_close_logs_an_unfinished_episode(make_remote, tmp_path):
    env = make_remote(mode=MODE_EVAL, log_dir=str(tmp_path))
    env.reset(seed=1, config="train")
    env.close()
    reps = [json.load(open(tmp_path / f)) for f in os.listdir(tmp_path) if f.startswith("report_")]
    assert [x["ended_by"] for x in reps] == ["closed"]


# ------------------------------------------------------------------ 3. check: the suite is a hard check
def test_check_fails_when_pytest_is_missing_unless_explicitly_allowed(monkeypatch):
    import importlib.util

    from auditgym import check
    monkeypatch.delenv("AUDITGYM_CHECK_RUNNING", raising=False)
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "pytest" else real(name, *a))
    rep = check.Report()
    check.step_pytest(rep)
    assert rep.checks[-1]["status"] == "FAIL" and rep.failed
    rep2 = check.Report()
    check.step_pytest(rep2, allow_skip_tests=True)
    assert rep2.checks[-1]["status"] == "SKIP" and not rep2.failed


def test_check_fails_when_tests_dir_is_empty(monkeypatch, tmp_path):
    from auditgym import check
    monkeypatch.delenv("AUDITGYM_CHECK_RUNNING", raising=False)
    monkeypatch.setattr(check, "ROOT", str(tmp_path))
    (tmp_path / "tests").mkdir()
    rep = check.Report()
    check.step_pytest(rep)
    assert rep.checks[-1]["status"] == "FAIL"


# ------------------------------------------------------------------ 4. evidence completeness
def test_check_evidence_rejects_a_misallocation_citation_missing_an_ar_credit_row():
    from auditgym.generator.build import generate_episode
    from auditgym.runner import resolve_config
    n = 0
    for seed in range(1, 31):
        ep = generate_episode(seed, resolve_config({"seed": seed, "config": "train"}))
        rows = {r["row_id"]: r for r in ep.ledger}
        for iss in ep.truth["issues"]:
            for ev in iss["evidence_sets"]:
                assert rules.check_evidence(iss, ev, ep)[0]
                ledger_cited = [e for e in ev if e in rows and rows[e]["account"] == ACC_AR]
                for e in ledger_cited:
                    r = rows[e]
                    if iss["type"] == "payment_misallocation" and int(r.get("credit_minor") or 0) <= 0:
                        continue
                    if iss["type"] != "payment_misallocation" and int(r.get("debit_minor") or 0) <= 0:
                        continue
                    ok, amt = rules.check_evidence(iss, [x for x in ev if x != e], ep)
                    assert not ok, (seed, iss["type"], e, amt)
                    n += 1
    assert n > 20


def test_review_case_seed1_train_misallocation_without_its_credit_row():
    from auditgym.generator.build import generate_episode
    from auditgym.runner import resolve_config
    ep = generate_episode(1, resolve_config({"seed": 1, "config": "train"}))
    rows = {r["row_id"]: r for r in ep.ledger}
    hit = False
    for iss in ep.truth["issues"]:
        if iss["type"] != "payment_misallocation":
            continue
        for ev in iss["evidence_sets"]:
            credits = [e for e in ev if e in rows and rows[e]["account"] == ACC_AR and int(rows[e]["credit_minor"]) > 0]
            for c in credits:
                hit = True
                assert rules.check_evidence(iss, [x for x in ev if x != c], ep) == (False, 0)
    assert hit


# ------------------------------------------------------------------ 5. isolated run
def _iso(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONDONTWRITEBYTECODE="1")
    env.pop(TOKEN_ENV, None)
    return subprocess.run([sys.executable, "-m", "auditgym", "isolate", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)


@needs_root
def test_isolation_self_test_passes_and_restores_permissions(tmp_path):
    before = os.stat(os.path.join(ROOT, "grader_data")).st_mode
    p = _iso("--self-test", "--log-dir", str(tmp_path / "logs"))
    assert p.returncode == 0, p.stdout + p.stderr
    assert "isolation self-test PASSED" in p.stdout and "LEAK" not in p.stdout
    assert os.stat(os.path.join(ROOT, "grader_data")).st_mode == before
    s = json.load(open(tmp_path / "logs" / "isolated_summary.json"))
    assert s["isolation_boundary"] and s["episodes"][0]["ended_by"] == "submit"


@needs_root
def test_escape_probe_detects_leaks_when_there_is_no_boundary(tmp_path):
    """The probe is not a rubber stamp: without the uid split it finds the token and the grader files."""
    p = _iso("--self-test", "--sandbox", "none", "--log-dir", str(tmp_path / "logs"))
    assert p.returncode == 1
    assert "LEAK" in p.stdout and "isolation self-test FAILED" in p.stdout


@needs_root
def test_isolated_policy_run_scores_like_the_normal_runner(tmp_path):
    p = _iso("--policy", "careful_auditor", "--split", "dev", "--log-dir", str(tmp_path / "logs"))
    assert p.returncode == 0, p.stdout + p.stderr
    s = json.load(open(tmp_path / "logs" / "isolated_summary.json"))
    assert len(s["episodes"]) == 10
    assert all(e["agent_status"] == "ok" and e["ended_by"] == "submit" and e["f1"] == 1.0 for e in s["episodes"])
    assert len([f for f in os.listdir(tmp_path / "logs") if f.startswith("report_")]) == 10


@needs_root
@pytest.mark.parametrize("code,status", [
    ("import sys; sys.exit(3)", "exited"),
    ("import time; time.sleep(30)", "timeout"),
    ("import sys; sys.stdout.write('x' * 3000000); sys.stdout.flush(); import time; time.sleep(30)", "too_long"),
    ("import os, time\nif os.fork() == 0:\n    os.setsid(); time.sleep(60)\ntime.sleep(60)", "timeout"),
])
def test_failing_agent_is_killed_and_its_episode_scored_empty(tmp_path, code, status):
    import tempfile
    adir = tempfile.mkdtemp(prefix="auditgym-test-agent-")       # readable by the agent uid (tmp_path is 0700)
    os.chmod(adir, 0o755)
    agent = os.path.join(adir, "agent.py")
    with open(agent, "w") as f:
        f.write(code)
    os.chmod(agent, 0o644)
    p = _iso("--agent", f"{sys.executable} -I {agent}", "--seed", "1", "--config", "train",
             "--step-timeout", "2", "--log-dir", str(tmp_path / "logs"))
    assert p.returncode == 0, p.stdout + p.stderr
    s = json.load(open(tmp_path / "logs" / "isolated_summary.json"))
    e = s["episodes"][0]
    assert e["agent_status"] == status and e["ended_by"] == "agent_failed" and e["reward_shared"] is not None
    # no process of the agent uid survived (the forked child included)
    left = [pid for pid in os.listdir("/proc") if pid.isdigit() and _uid_of(pid) == s["agent_uid"]]
    assert not left


def _uid_of(pid: str):
    """uid of a LIVE process (zombies waiting to be reaped by init do not count)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                return None
        return os.stat(f"/proc/{pid}").st_uid
    except (OSError, IndexError):
        return None


def test_auto_sandbox_refuses_without_root_instead_of_degrading(monkeypatch):
    from auditgym import isolate
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    with pytest.raises(isolate.IsolationError, match="--sandbox none"):
        isolate.Sandbox("auto")
    with pytest.raises(isolate.IsolationError):
        isolate.Sandbox("uid")


def test_agent_environment_is_rebuilt_from_scratch(monkeypatch, tmp_path):
    from auditgym import isolate
    monkeypatch.setenv(TOKEN_ENV, "secret")
    monkeypatch.setenv("SOME_API_KEY", "secret2")
    env = isolate._agent_env(str(tmp_path))
    assert set(env) == {"PATH", "LANG", "LC_ALL", "HOME", "TMPDIR", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED"}
    assert "secret" not in json.dumps(env)


def test_agent_lines_can_only_ever_become_step_actions():
    from auditgym.isolate import _parse_action
    assert _parse_action(b'{"cmd":"close"}') is None
    assert _parse_action(b'{"cmd":"reset","token":"\\ud800"}') is None
    assert _parse_action(b"\xff\xfe") is None
    assert _parse_action(b'{"action":1,"action":2}') is None
    assert _parse_action(b'{"action":{"action":"calculator"},"cmd":"close"}') == {"action": "calculator"}
