"""Agent-side handle on the out-of-process grader (SPEC section 7).

RemoteAuditEnv mirrors AuditEnv.reset/step/close.  The environment (and its truth) lives in a
subprocess running `python -m auditgym.server`; this process never imports truth.
"""
from __future__ import annotations

import json
import os
import secrets
import select
import subprocess
import sys
from typing import Any, Callable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RemoteEnvError(RuntimeError):
    """The server process failed, died, or rejected a command."""


class AgentHandle:
    """Agent-facing view of a RemoteAuditEnv: the single `step` action channel.

    The handle is slotted, immutable and keeps no attribute that points at the env (no bound method,
    so no `__self__`, no `_step`/`_call`/`_token`). `step` is a plain function closing over the env.
    This is API hygiene, NOT a sandbox: Python closures still expose `step.__closure__`, so an
    in-process policy that deliberately digs through private attributes is trusted code. Real
    isolation is a separate process: the server (truth, seed, grader report) never shares memory with
    the policy, and the control token never reaches it.
    """
    __slots__ = ("step",)

    def __init__(self, step_fn: Callable[[Any], tuple[dict, float, bool, bool, dict]]):
        object.__setattr__(self, "step", step_fn)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("AgentHandle is read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("AgentHandle is read-only")


class RemoteAuditEnv:
    def __init__(self, mode: str = "eval", log_dir: str | None = None,
                 allow_grader_report: bool = False, max_steps: int | None = None,
                 time_limit_s: float | None = None, read_timeout_s: float | None = 120.0):
        self.mode = mode
        self._read_timeout_s = read_timeout_s      # transport watchdog: a silent server is killed, not waited on
        cmd = [sys.executable, "-m", "auditgym.server", "--mode", mode]
        if log_dir:
            cmd += ["--log-dir", str(log_dir)]
        if allow_grader_report:
            cmd.append("--allow-grader-report")
        if max_steps is not None:
            cmd += ["--max-steps", str(int(max_steps))]
        if time_limit_s is not None:
            cmd += ["--time-limit", str(float(time_limit_s))]
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self._token = secrets.token_hex(16)      # harness-only capability (reset / grader_report)
        env["AUDITGYM_CONTROL_TOKEN"] = self._token
        self._proc: subprocess.Popen | None = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)

    # -------------------------------------------------------------- transport
    def _call(self, msg: dict) -> dict:
        p = self._proc
        if p is None or p.stdin is None or p.stdout is None:
            raise RemoteEnvError("remote env is closed")
        try:
            p.stdin.write(json.dumps(msg, ensure_ascii=True) + "\n")
            p.stdin.flush()
            if self._read_timeout_s is not None and os.name != "nt":
                # exactly one reply line per request, so nothing is ever buffered ahead of select()
                ready, _, _ = select.select([p.stdout], [], [], self._read_timeout_s)
                if not ready:
                    self.close()
                    raise RemoteEnvError(f"server did not answer within {self._read_timeout_s:g} s; closed")
            line = p.stdout.readline()
        except (BrokenPipeError, OSError, ValueError) as e:
            raise RemoteEnvError(f"server connection lost: {e}") from None
        if not line:
            raise RemoteEnvError("server exited unexpectedly")
        try:
            reply = json.loads(line)
        except ValueError:
            raise RemoteEnvError("server sent invalid JSON") from None
        if not isinstance(reply, dict):
            raise RemoteEnvError("server reply is not an object")
        if not reply.get("ok"):
            raise RemoteEnvError(str(reply.get("error", "server error")))
        return reply

    # -------------------------------------------------------------- env API
    def reset(self, seed: int | None = None, config: Any = None,
              options: dict | None = None) -> tuple[dict, dict]:
        r = self._call({"cmd": "reset", "seed": seed, "config": config, "options": options,
                        "token": self._token})
        return r["obs"], r["info"]

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        try:
            json.dumps(action)
        except (TypeError, ValueError, RecursionError):
            action = None            # server answers with a bounded unknown_action error
        r = self._call({"cmd": "step", "action": action})
        return r["obs"], r["reward"], r["terminated"], r["truncated"], r["info"]

    def grader_report(self) -> dict:
        return self._call({"cmd": "grader_report", "token": self._token})["report"]

    def abandon(self) -> None:
        """Harness: score the running episode as truncated/empty (ended_by=agent_failed) and log it, e.g.
        because the agent process died or timed out. The grader report is then available as usual."""
        self._call({"cmd": "abandon", "token": self._token})

    def agent_handle(self) -> "AgentHandle":
        """What a policy may hold: `step` only (no reset, no grader report, no token, no env attribute).
        See AgentHandle: closures are introspectable, so this is hygiene; process isolation is the boundary."""
        return AgentHandle(lambda a: self.step(a))

    def close(self) -> None:
        p, self._proc = self._proc, None
        if p is None:
            return
        try:
            if p.poll() is None and p.stdin is not None:
                try:
                    p.stdin.write(json.dumps({"cmd": "close", "token": self._token}) + "\n")
                    p.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            for s in (p.stdin, p.stdout):
                try:
                    if s:
                        s.close()
                except OSError:
                    pass
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
        finally:
            pass

    def __enter__(self) -> "RemoteAuditEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
