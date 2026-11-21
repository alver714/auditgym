"""Isolated run: grader, harness and agent in separate processes, the agent under its own uid.

    python -m auditgym isolate --agent "python3 my_agent.py" --split dev --log-dir /grader/logs
    python -m auditgym isolate --policy careful_auditor --split dev
    python -m auditgym isolate --self-test          # an escape-probe agent tries to break out; exit 1 on a leak

Process layout (one harness process, i.e. this one):

    grader server  <-- JSON lines + control token -->  harness  <-- JSON lines, observations/actions -->  agent
    (truth, token,      (RemoteAuditEnv: reset/abandon/close,       (a fresh process per episode: own uid,
     grader reports)     grader_report; relays `step` only)          empty env, empty cwd, no extra fds)

Agent protocol (stdin/stdout, one JSON object per line, UTF-8):
    harness -> agent  {"type": "observation", "obs": {..}, "reward": 0.0, "terminated": false,
                       "truncated": false, "info": {}}
    agent -> harness  {"action": {"action": "list_records", "args": {..}}}
After every action the agent receives the next observation; after the terminal one its stdin is closed. A line that
is not a JSON object with an "action" key is forwarded as an invalid action (the env answers unknown_action);
anything else in the line is ignored, so the agent can never send a server command. An agent that exits, stays
silent longer than --step-timeout or sends a line longer than --max-line is killed and its episode is scored empty
(ended_by=agent_failed) and logged.

What the agent can and cannot reach (sandbox "uid", the default when the harness runs as root on Linux/POSIX):
  * it runs as a separate unprivileged uid/gid (default 65534) with no supplementary groups, no_new_privs set, in
    its own process group and session, with an empty temporary working directory it owns;
  * its environment is rebuilt from scratch (PATH, LANG, HOME/TMPDIR = its workdir): the control token and every
    other variable of the harness are absent; it inherits no file descriptor besides stdin/stdout/stderr;
  * the grader log directory, the episodes file and grader_data/ are made unreadable to it for the run (mode 0700,
    owned by the harness user; the original mode is restored afterwards), and a preflight probe running as the
    agent uid must fail to read every one of them, otherwise the run is refused (fail closed);
  * the grader server and the harness run as the harness user, so /proc/<pid>/environ, /proc/<pid>/mem and ptrace
    of those processes are denied to the agent by the kernel;
  * after each episode every process of the agent uid is killed (use a DEDICATED uid), and the workdir is deleted.
  * --no-network additionally puts the agent into an empty network namespace (needs CAP_SYS_ADMIN; refused,
    not silently skipped, when unavailable).
Sandbox "none" keeps only the process split, the scrubbed environment and the separate cwd; the agent runs as the
harness user and could read the harness's files. It exists for local debugging and is labelled in every summary.
This is process-level isolation, not a VM: a kernel exploit, a world-readable copy of private data elsewhere on the
host, or a too-wide --protect list are out of scope. Holdout truth is never written to disk (the server generates it
in memory), so the files worth protecting are the episodes/seeds file and the grader logs.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import select
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any

from .constants import MODE_EVAL, MODE_TRAIN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS_DIR = os.path.join(ROOT, "auditgym", "agents")
PROBE_SCRIPT = os.path.join(AGENTS_DIR, "escape_probe.py")
POLICY_AGENT_SCRIPT = os.path.join(AGENTS_DIR, "policy_agent.py")
SANDBOXES = ("auto", "uid", "none")
DEFAULT_UID = 65534
DEFAULT_LOG_DIR = os.path.join(ROOT, "reports", "isolated_logs")
MAX_LINE = 1_000_000
SAFETY_STEPS = 1000                 # the env ends an episode by its step budget; this is only a loop guard
_PR_SET_NO_NEW_PRIVS = 38
_CLONE_NEWNET = 0x40000000


class IsolationError(RuntimeError):
    """The requested isolation cannot be provided; the run is refused instead of degraded."""


# ------------------------------------------------------------------ agent process
def _libc():
    try:
        return ctypes.CDLL(None, use_errno=True)
    except OSError:
        return None


def _make_preexec(no_network: bool, uid: int | None, gid: int | None):
    libc = _libc()

    def preexec() -> None:            # runs in the child between fork and exec
        if no_network:                # before dropping root: unshare needs CAP_SYS_ADMIN
            if libc is None or libc.unshare(_CLONE_NEWNET) != 0:
                os.write(2, b"isolate: unshare(CLONE_NEWNET) failed\n")
                os._exit(97)
        if uid is not None:
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)
            if os.getuid() != uid or os.geteuid() != uid:
                os._exit(98)
        if libc is not None:
            libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except Exception:
            pass
    return preexec


def _agent_env(workdir: str) -> dict:
    return {"PATH": os.environ.get("PATH") or "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8", "HOME": workdir, "TMPDIR": workdir,
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}


class Sandbox:
    def __init__(self, kind: str, uid: int = DEFAULT_UID, gid: int | None = None, no_network: bool = False):
        if kind == "auto":
            kind = "uid" if os.name == "posix" and os.geteuid() == 0 else ""
            if not kind:
                raise IsolationError(
                    "sandbox 'uid' needs the harness to run as root (e.g. inside the Docker image) so that it can "
                    "start the agent under a separate uid; pass --sandbox none to run WITHOUT a boundary")
        if kind not in ("uid", "none"):
            raise IsolationError(f"unknown sandbox {kind!r}")
        if kind == "uid":
            if os.name != "posix" or os.geteuid() != 0:
                raise IsolationError("sandbox 'uid' requires root on a POSIX system")
            if uid == 0 or uid == os.geteuid():
                raise IsolationError("the agent uid must differ from the harness uid and from root")
        self.kind = kind
        self.uid = uid
        self.gid = uid if gid is None else gid
        self.no_network = no_network
        self._restored: list[tuple[str, int]] = []

    @property
    def is_boundary(self) -> bool:
        return self.kind == "uid"

    # ---- workdir and spawning
    def make_workdir(self) -> str:
        d = tempfile.mkdtemp(prefix="auditgym-agent-")
        if self.kind == "uid":
            os.chown(d, self.uid, self.gid)
        os.chmod(d, 0o700)
        return d

    def popen(self, argv: list[str], workdir: str, stderr) -> subprocess.Popen:
        kw: dict[str, Any] = {}
        if os.name == "posix":
            uid, gid = (self.uid, self.gid) if self.kind == "uid" else (None, None)
            kw.update(start_new_session=True, preexec_fn=_make_preexec(self.no_network, uid, gid))
        return subprocess.Popen(argv, cwd=workdir, env=_agent_env(workdir), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=stderr, close_fds=True, **kw)

    def kill(self, proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, AttributeError):
                proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        if self.kind == "uid":
            self.sweep()

    def sweep(self) -> int:
        """Kill every process of the agent uid (a daemon that escaped the process group included)."""
        n = 0
        for _ in range(20):
            found = 0
            for pid in _pids():
                try:
                    if os.stat(f"/proc/{pid}").st_uid == self.uid and not _is_zombie(pid):
                        os.kill(pid, signal.SIGKILL)
                        found += 1
                except OSError:
                    pass
            n += found
            if not found:
                break
            time.sleep(0.05)
        return n

    # ---- protected paths
    def protect(self, paths: list[str]) -> None:
        """Make every path unreachable for the agent uid and prove it with a probe running AS the agent."""
        paths = [os.path.abspath(p) for p in paths if p and os.path.exists(p)]
        if self.kind != "uid":
            return
        for p in paths:
            st = os.stat(p)
            if st.st_uid == self.uid:
                raise IsolationError(f"protected path {p} is owned by the agent uid {self.uid}")
            if st.st_mode & 0o077:
                self._restored.append((p, stat.S_IMODE(st.st_mode)))
                os.chmod(p, stat.S_IMODE(st.st_mode) & 0o700)
        leaks = self.readable_by_agent(paths)
        if leaks:
            self.restore()
            raise IsolationError("the agent uid can still read protected paths: " + ", ".join(leaks))

    def restore(self) -> None:
        while self._restored:
            p, mode = self._restored.pop()
            try:
                os.chmod(p, mode)
            except OSError:
                pass

    def readable_by_agent(self, paths: list[str]) -> list[str]:
        if not paths:
            return []
        code = ("import os,sys,json\nbad=[]\n"
                "for p in json.loads(sys.argv[1]):\n"
                "    try:\n"
                "        (os.listdir(p) if os.path.isdir(p) else open(p,'rb').read(1)); bad.append(p)\n"
                "    except OSError:\n"
                "        pass\n"
                "print(json.dumps(bad))\n")
        wd = self.make_workdir()
        try:
            p = self.popen([sys.executable, "-I", "-S", "-c", code, json.dumps(paths)], wd, subprocess.DEVNULL)
            out, _ = p.communicate(timeout=30)
            if p.returncode != 0:
                raise IsolationError("the preflight probe could not run as the agent uid")
            return json.loads(out.decode() or "[]")
        finally:
            shutil.rmtree(wd, ignore_errors=True)


def _is_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except (OSError, IndexError):
        return True


def _pids() -> list[int]:
    try:
        return [int(x) for x in os.listdir("/proc") if x.isdigit()]
    except OSError:
        return []


# ------------------------------------------------------------------ line transport to the agent
class _LineReader:
    def __init__(self, fd: int, max_line: int):
        self.fd, self.max_line, self.buf = fd, max_line, b""

    def readline(self, timeout_s: float) -> tuple[str, bytes | None]:
        """('ok', line) | ('eof', None) | ('timeout', None) | ('too_long', None)."""
        deadline = time.monotonic() + timeout_s
        while b"\n" not in self.buf:
            if len(self.buf) > self.max_line:
                return "too_long", None
            left = deadline - time.monotonic()
            if left <= 0:
                return "timeout", None
            r, _, _ = select.select([self.fd], [], [], left)
            if not r:
                return "timeout", None
            chunk = os.read(self.fd, 65536)
            if not chunk:
                return "eof", None
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        if len(line) > self.max_line:
            return "too_long", None
        return "ok", line


def _parse_action(line: bytes) -> Any:
    from .server import _DuplicateKey, _no_duplicate_keys
    try:
        msg = json.loads(line.decode("utf-8", errors="replace"), object_pairs_hook=_no_duplicate_keys)
    except (_DuplicateKey, ValueError, RecursionError):
        return None
    if not isinstance(msg, dict) or "action" not in msg:
        return None
    return msg["action"]


def _send(proc: subprocess.Popen, payload: dict) -> bool:
    try:
        proc.stdin.write((json.dumps(payload, ensure_ascii=True) + "\n").encode())
        proc.stdin.flush()
        return True
    except (BrokenPipeError, OSError, ValueError):
        return False


# ------------------------------------------------------------------ episodes
def load_episodes(path: str | None, split: str | None, seeds: list[int] | None, config: str | None) -> list[dict]:
    from .runner import load_splits, resolve_config
    if path:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        items = raw.get("episodes", raw) if isinstance(raw, dict) else raw
        out = []
        for it in items:
            if isinstance(it, int):
                out.append({"seed": it, "config": config or "holdout"})
            else:
                out.append({"seed": int(it["seed"]), "config": it.get("config", config or "holdout")})
        return out
    if seeds:
        return [{"seed": s, "config": config or "holdout"} for s in seeds]
    entries = load_splits(ROOT).get(split or "dev") or []
    return [{"seed": e["seed"], "config": resolve_config(e)} for e in entries]


def _run_one(env, sandbox: Sandbox, agent_argv: list[str], ep: dict, stderr_path: str,
             step_timeout_s: float, max_line: int) -> dict:
    obs, info = env.reset(seed=ep["seed"], config=ep["config"])
    workdir = sandbox.make_workdir()
    status, steps = "ok", 0
    with open(stderr_path, "ab") as errf:
        proc = sandbox.popen(agent_argv, workdir, errf)
        try:
            reader = _LineReader(proc.stdout.fileno(), max_line)
            payload = {"type": "observation", "obs": obs, "reward": 0.0, "terminated": False,
                       "truncated": False, "info": info}
            done = False
            while not done:
                if not _send(proc, payload):
                    status = "exited"
                    break
                st, line = reader.readline(step_timeout_s)
                if st != "ok":
                    status = "exited" if st == "eof" else st
                    break
                obs, reward, term, trunc, info = env.step(_parse_action(line))
                steps += 1
                done = bool(term or trunc) or steps >= SAFETY_STEPS
                payload = {"type": "observation", "obs": obs, "reward": reward, "terminated": term,
                           "truncated": trunc, "info": info}
            if done:
                _send(proc, payload)              # the terminal observation, then EOF
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            sandbox.kill(proc)
            shutil.rmtree(workdir, ignore_errors=True)
    if status != "ok":
        try:
            env.abandon()
        except Exception:
            pass                                   # already ended (e.g. the step budget ran out)
    rep = env.grader_report()
    return {"episode_id": rep.get("episode_id"), "seed": ep["seed"],
            "config": ep["config"] if isinstance(ep["config"], str) else ep["config"].get("config_id"),
            "agent_status": status, "agent_steps": steps, "ended_by": rep.get("ended_by"),
            "reward_shared": rep.get("reward_shared"), "f1": (rep.get("metrics") or {}).get("f1")}


def run_isolated(agent_argv: list[str], episodes: list[dict], *, log_dir: str = DEFAULT_LOG_DIR,
                 mode: str = MODE_EVAL, sandbox: str = "auto", agent_uid: int = DEFAULT_UID,
                 protect: list[str] | None = None, step_timeout_s: float = 60.0, max_line: int = MAX_LINE,
                 no_network: bool = False) -> dict:
    from .client import RemoteAuditEnv
    sb = Sandbox(sandbox, agent_uid, no_network=no_network)
    os.makedirs(log_dir, exist_ok=True)
    os.chmod(log_dir, 0o700)
    stderr_dir = os.path.join(log_dir, "agent_stderr")
    os.makedirs(stderr_dir, exist_ok=True)
    protected = [log_dir, os.path.join(ROOT, "grader_data")] + list(protect or [])
    results: list[dict] = []
    sb.protect(protected)
    try:
        env = RemoteAuditEnv(mode=mode, log_dir=log_dir, allow_grader_report=True)
        try:
            for i, ep in enumerate(episodes):
                results.append(_run_one(env, sb, agent_argv, ep,
                                        os.path.join(stderr_dir, f"episode_{i + 1:04d}.log"),
                                        step_timeout_s, max_line))
        finally:
            env.close()
    finally:
        if sb.kind == "uid":
            sb.sweep()
        sb.restore()
    summary = {"sandbox": sb.kind, "isolation_boundary": sb.is_boundary, "agent_uid": sb.uid if sb.is_boundary else None,
               "no_network": no_network, "mode": mode, "agent": agent_argv, "log_dir": log_dir,
               "protected": [os.path.abspath(p) for p in protected], "episodes": results}
    with open(os.path.join(log_dir, "isolated_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


# ------------------------------------------------------------------ self-test (escape probe)
def probe_findings(log_dir: str) -> list[dict]:
    """PROBE_RESULT lines the escape probe wrote to its stderr, one per episode."""
    out = []
    d = os.path.join(log_dir, "agent_stderr")
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        with open(os.path.join(d, name), encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("PROBE_RESULT "):
                    out.append(json.loads(line[len("PROBE_RESULT "):]))
    return out


def self_test(log_dir: str, sandbox: str = "auto", agent_uid: int = DEFAULT_UID, episodes_file: str | None = None,
              no_network: bool = False) -> tuple[bool, dict]:
    """Run the escape probe on one dev episode. Passes iff the probe found no leak, the harness survived every
    hostile line and the episode ended through the probe's own submit (not through a forged close/reset)."""
    if os.path.isdir(os.path.join(log_dir, "agent_stderr")):
        shutil.rmtree(os.path.join(log_dir, "agent_stderr"))
    eps = load_episodes(episodes_file, "dev", None, None)[:1]
    protect = [episodes_file] if episodes_file else []
    summary = run_isolated([sys.executable, "-I", "-S", PROBE_SCRIPT], eps, log_dir=log_dir, sandbox=sandbox,
                           agent_uid=agent_uid, protect=protect, step_timeout_s=30, no_network=no_network)
    found = probe_findings(log_dir)
    leaks = [c for r in found for c in r.get("checks", []) if c.get("leak")]
    ep = summary["episodes"][0] if summary["episodes"] else {}
    ok = (bool(found) and not leaks and ep.get("agent_status") == "ok" and ep.get("ended_by") == "submit"
          and summary["isolation_boundary"])
    summary["probe"] = found
    summary["self_test_passed"] = ok
    return ok, summary


# ------------------------------------------------------------------ CLI
def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(prog="python -m auditgym isolate")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--agent", help="agent command line (split like a shell); speaks the JSON-lines agent protocol")
    who.add_argument("--policy", help="run a built-in policy as the isolated agent (auditgym/agents/policy_agent.py)")
    who.add_argument("--self-test", action="store_true",
                     help="run the escape-probe agent; exit 1 if it reads anything private or breaks the protocol")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--episodes", help="JSON file: [{seed, config}, ...] or [seed, ...] (protected from the agent)")
    src.add_argument("--split", choices=("train", "dev", "holdout"), help="episodes from configs/splits.json (default dev)")
    src.add_argument("--seed", type=int, action="append", help="one episode seed (repeatable), with --config")
    ap.add_argument("--config", default=None, help="config name for --seed / bare seeds in --episodes (default holdout)")
    ap.add_argument("--mode", choices=(MODE_EVAL, MODE_TRAIN), default=MODE_EVAL)
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR, help="grader-only directory: reports, agent stderr, summary")
    ap.add_argument("--sandbox", choices=SANDBOXES, default="auto",
                    help="auto/uid: agent under its own uid (needs root); none: NO boundary (debug only)")
    ap.add_argument("--agent-uid", type=int, default=DEFAULT_UID, help="dedicated uid (and gid) for the agent")
    ap.add_argument("--protect", action="append", default=[], help="extra path the agent must not read (repeatable)")
    ap.add_argument("--step-timeout", type=float, default=60.0, help="seconds the agent may take per action")
    ap.add_argument("--no-network", action="store_true", help="agent in an empty network namespace (needs CAP_SYS_ADMIN)")
    return ap


def _print_summary(summary: dict) -> None:
    print(f"sandbox: {summary['sandbox']}  isolation boundary: {'yes' if summary['isolation_boundary'] else 'NO'}"
          f"  agent uid: {summary['agent_uid']}  mode: {summary['mode']}  no_network: {summary['no_network']}")
    print(f"{'#':>3}  {'episode':<16} {'agent':<9} {'steps':>5}  {'ended_by':<13} {'reward':>8} {'f1':>6}")
    for i, r in enumerate(summary["episodes"]):
        rew = r["reward_shared"]
        f1 = r["f1"]
        print(f"{i + 1:>3}  {str(r['episode_id']):<16} {r['agent_status']:<9} {r['agent_steps']:>5}  "
              f"{str(r['ended_by']):<13} {rew if rew is None else f'{rew:+.3f}':>8} "
              f"{f1 if f1 is None else f'{f1:.3f}':>6}")
    print(f"grader reports and agent stderr: {summary['log_dir']}")


def main_args(a) -> int:
    try:
        if a.self_test:
            ok, summary = self_test(a.log_dir, a.sandbox, a.agent_uid, a.episodes, a.no_network)
            _print_summary(summary)
            for r in summary["probe"]:
                for c in r.get("checks", []):
                    print(f"  {'LEAK' if c['leak'] else 'ok  '}  {c['check']}: {c['detail']}")
            print("isolation self-test " + ("PASSED" if ok else "FAILED"))
            return 0 if ok else 1
        if a.policy:
            argv = [sys.executable, "-I", POLICY_AGENT_SCRIPT, a.policy]
        else:
            argv = shlex.split(a.agent)
        eps = load_episodes(a.episodes, a.split, a.seed, a.config)
        protect = list(a.protect) + ([a.episodes] if a.episodes else [])
        summary = run_isolated(argv, eps, log_dir=a.log_dir, mode=a.mode, sandbox=a.sandbox,
                               agent_uid=a.agent_uid, protect=protect, step_timeout_s=a.step_timeout,
                               no_network=a.no_network)
    except IsolationError as e:
        print(f"isolate: refused: {e}", file=sys.stderr)
        return 2
    _print_summary(summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    return main_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
