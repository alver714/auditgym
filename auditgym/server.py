"""Out-of-process grader server (SPEC section 7): JSON lines on stdin/stdout.

    python -m auditgym.server --mode eval|train [--log-dir DIR] [--allow-grader-report]
                              [--max-steps N] [--time-limit S] [--insecure-open]

Commands (one JSON object per line):
  {"cmd":"reset","seed":..,"config":..,"options":..,"token":..}  -> {"ok":true,"obs":..,"info":..}
  {"cmd":"step","action":{..}}                          -> {"ok":true,"obs":..,"reward":..,"terminated":..,"truncated":..,"info":..}
  {"cmd":"grader_report","token":..}   (only after the episode ended and only with --allow-grader-report)
  {"cmd":"close"}
Failures reply {"ok":false,"error":"<short message>"}; the loop keeps running. Truth lives only in
this process; per-episode grader reports are written to --log-dir, never sent to the agent.

Trust boundary: `step` is the only agent-facing command. `reset` (which chooses seed / config) and
`grader_report` are HARNESS commands and FAIL CLOSED: they are refused unless the message carries the
token from the environment variable AUDITGYM_CONTROL_TOKEN. With no token configured they are refused
outright, unless the operator explicitly passes --insecure-open (local experiments only: any stdin
peer may then reset with any seed/config and read grader reports). client.RemoteAuditEnv creates a
fresh random token per server process and keeps it; the `AgentHandle` it hands to policies carries only
a `step` function, so a policy that does not introspect private attributes can neither re-reset an
episode (adaptive truth oracle) nor read the grader report. (In-process Python cannot be sandboxed:
a policy that digs through closures is trusted code. The isolation boundary is this separate process.)
Even for the harness the wire is narrow: config may be null, a bare config name or a dict with only known
keys; `episode_dir` / `grader_dir` (file paths) are never accepted over the wire; seed must be an int in
0 <= seed < 2**63; JSON objects with duplicate keys are rejected. An episode abandoned by a re-reset is
scored as truncated/empty and logged.

Time limit: besides the cooperative check on every step, a watchdog timer truncates an unfinished episode
(ended_by=time, empty answer, report logged) when the wall-clock limit passes while the server is waiting
for the next action; the client then sees episode_over on its next step.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import sys
import threading
import traceback

from .constants import EPISODE_TIME_LIMIT_S, MAX_STEPS, MODE_EVAL, MODE_TRAIN
from .env import AuditEnv

MAX_LINE = 4_000_000
TOKEN_ENV = "AUDITGYM_CONTROL_TOKEN"
HARNESS_CMDS = ("reset", "grader_report")
MAX_SEED = 2 ** 63
_CONFIG_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_CONFIG_KEYS = frozenset({"config_id", "config_version", "company", "issues", "interactions",
                          "distractors", "clean_probability", "force"})
_FORBIDDEN_CONFIG_KEYS = ("episode_dir", "grader_dir")
_TRANSFORM_KEYS = frozenset({"rename_salt", "permute_salt"})
BAD_RESET_ARGS = "reset failed (invalid seed, config or options)"


class _DuplicateKey(ValueError):
    pass


def _no_duplicate_keys(pairs: list) -> dict:
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise _DuplicateKey(k)
        out[k] = v
    return out


def _reset_args_ok(msg: dict) -> bool:
    """Wire-level validation of reset arguments (before anything touches the filesystem)."""
    seed = msg.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < MAX_SEED):
        return False
    cfg = msg.get("config")
    if isinstance(cfg, dict):
        if any(k in cfg for k in _FORBIDDEN_CONFIG_KEYS) or not set(cfg) <= _CONFIG_KEYS:
            return False
    elif isinstance(cfg, str):
        if not _CONFIG_NAME.fullmatch(cfg):          # a bare name in configs/, never a path
            return False
    elif cfg is not None:
        return False
    opts = msg.get("options")
    if opts is not None:
        if not isinstance(opts, dict) or not set(opts) <= {"transform"}:
            return False
        t = opts.get("transform")
        if t is not None and (not isinstance(t, dict) or not set(t) <= _TRANSFORM_KEYS):
            return False
    return True


def _short(s: object, n: int = 120) -> str:
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


class Server:
    def __init__(self, mode: str, log_dir: str | None, allow_grader_report: bool,
                 max_steps: int = MAX_STEPS, time_limit_s: float = EPISODE_TIME_LIMIT_S,
                 insecure_open: bool = False):
        self.env = AuditEnv(mode=mode, max_steps=max_steps, time_limit_s=time_limit_s)
        self.insecure_open = insecure_open
        self._lock = threading.RLock()            # handle() vs. the watchdog timer thread
        self._timer: threading.Timer | None = None
        self.log_dir = log_dir
        self.allow_grader_report = allow_grader_report
        self._token = os.environ.pop(TOKEN_ENV, None) or None     # never inherited by anything else
        self._episodes_logged = 0
        self._logged_current = True
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # -------------------------------------------------------------- logging
    def _log_exc(self) -> None:
        text = traceback.format_exc()
        if self.log_dir:
            with open(os.path.join(self.log_dir, "server_errors.log"), "a", encoding="utf-8") as f:
                f.write(text + "\n")
        else:
            sys.stderr.write(text)
            sys.stderr.flush()

    def _log_report_if_done(self) -> None:
        if self._logged_current or not self.log_dir or not self.env.ended():
            return
        self._logged_current = True
        rep = self.env.grader_report()
        self._episodes_logged += 1
        name = f"report_{self._episodes_logged:04d}_{rep.get('episode_id')}.json"
        with open(os.path.join(self.log_dir, name), "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, sort_keys=True)
            f.write("\n")

    # -------------------------------------------------------------- commands
    @staticmethod
    def _reset_error(e: Exception) -> str:
        """Fixed, path-free reply for a failed reset; details go to the server log only. Our own
        validation errors (whose text we control) are passed on so a bad config is diagnosable."""
        from .generator.build import GenerationError
        from .generator.config import ConfigError
        if isinstance(e, (ConfigError, GenerationError)):
            return _short(f"reset failed: {type(e).__name__}: {e}")
        return BAD_RESET_ARGS

    # -------------------------------------------------------------- watchdog
    def _disarm(self) -> None:
        t, self._timer = self._timer, None
        if t is not None:
            t.cancel()

    def _arm(self) -> None:
        """(Re)start the wall-clock watchdog for the running episode."""
        self._disarm()
        left = self.env.seconds_left()
        if left is None:
            return
        t = threading.Timer(max(0.0, left) + 0.05, self._on_deadline)
        t.daemon = True
        self._timer = t
        t.start()

    def _on_deadline(self) -> None:
        with self._lock:
            try:
                if self.env.expire_if_overdue():
                    self._log_report_if_done()
            except Exception:
                self._log_exc()

    def handle(self, msg: object) -> tuple[dict, bool]:
        with self._lock:
            reply, stop = self._handle(msg)
            if stop:
                self._disarm()
            elif isinstance(msg, dict) and msg.get("cmd") in ("reset", "step"):
                self._arm()
            return reply, stop

    def _handle(self, msg: object) -> tuple[dict, bool]:
        """Return (reply, stop)."""
        if not isinstance(msg, dict):
            return {"ok": False, "error": "message must be a JSON object"}, False
        cmd = msg.get("cmd")
        if cmd in HARNESS_CMDS:
            if self._token is not None:
                tok = msg.get("token")
                if not isinstance(tok, str) or not hmac.compare_digest(tok.encode(), self._token.encode()):
                    return {"ok": False, "error": f"{_short(cmd, 20)} is a harness command"}, False
            elif not self.insecure_open:             # fail closed: no token configured, no --insecure-open
                return {"ok": False, "error": f"{_short(cmd, 20)} is a harness command"}, False
        try:
            if cmd == "reset":
                if not _reset_args_ok(msg):
                    return {"ok": False, "error": BAD_RESET_ARGS}, False
                if self.env.in_progress():            # abandoned episode: score (empty) + log it
                    self.env.abandon()
                    self._log_report_if_done()
                try:
                    obs, info = self.env.reset(seed=msg.get("seed"), config=msg.get("config"),
                                               options=msg.get("options"))
                except Exception as e:
                    self._log_exc()
                    return {"ok": False, "error": self._reset_error(e)}, False
                self._logged_current = False
                return {"ok": True, "obs": obs, "info": info}, False
            if cmd == "step":
                obs, reward, term, trunc, info = self.env.step(msg.get("action"))
                self._log_report_if_done()
                return {"ok": True, "obs": obs, "reward": reward, "terminated": term,
                        "truncated": trunc, "info": info}, False
            if cmd == "grader_report":
                if not self.allow_grader_report:
                    return {"ok": False, "error": "grader_report not enabled"}, False
                if not self.env.ended():
                    return {"ok": False, "error": "episode not finished"}, False
                return {"ok": True, "report": self.env.grader_report()}, False
            if cmd == "close":
                return {"ok": True}, True
            return {"ok": False, "error": "unknown cmd"}, False
        except Exception as e:  # never die on a bad request; never leak a traceback to the agent
            self._log_exc()
            return {"ok": False, "error": _short(f"{cmd} failed: {type(e).__name__}: {e}")}, False

    # -------------------------------------------------------------- loop
    def serve(self, fin, fout) -> None:
        while True:
            line = fin.readline(MAX_LINE + 1)
            if not line:
                break
            if len(line) > MAX_LINE and not line.endswith("\n"):
                # drain the rest of an oversized line
                while True:
                    rest = fin.readline(MAX_LINE)
                    if not rest or rest.endswith("\n"):
                        break
                reply, stop = {"ok": False, "error": "line too long"}, False
            elif not line.strip():
                continue
            else:
                try:
                    msg = json.loads(line, object_pairs_hook=_no_duplicate_keys)
                except _DuplicateKey:
                    reply, stop = {"ok": False, "error": "duplicate keys in json object"}, False
                except (ValueError, RecursionError):
                    reply, stop = {"ok": False, "error": "bad json"}, False
                else:
                    reply, stop = self.handle(msg)
            try:
                out = json.dumps(reply, ensure_ascii=True, allow_nan=False)
            except (TypeError, ValueError):
                self._log_exc()
                out = json.dumps({"ok": False, "error": "reply not serialisable"})
            try:
                fout.write(out + "\n")
                fout.flush()
            except BrokenPipeError:
                break
            if stop:
                break
        self._disarm()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="auditgym.server")
    ap.add_argument("--mode", choices=(MODE_EVAL, MODE_TRAIN), default=MODE_EVAL)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--allow-grader-report", action="store_true")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--time-limit", type=float, default=EPISODE_TIME_LIMIT_S)
    ap.add_argument("--insecure-open", action="store_true",
                    help="accept reset/grader_report from any stdin peer when no control token is set "
                         "(local experiments only)")
    a = ap.parse_args(argv)
    for stream in (sys.stdin, sys.stdout):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    srv = Server(a.mode, a.log_dir, a.allow_grader_report, a.max_steps, a.time_limit,
                 insecure_open=a.insecure_open)
    if srv._token is None and not a.insecure_open:
        sys.stderr.write("auditgym.server: no AUDITGYM_CONTROL_TOKEN and no --insecure-open: "
                         "reset/grader_report will be refused\n")
        sys.stderr.flush()
    srv.serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
