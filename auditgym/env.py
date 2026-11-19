"""AuditEnv: the reset/step contract (SPEC section 4).

The env owns the (private) truth but never places it in an observation, info dict, error message
or trajectory record.  Observations are deterministic and JSON-serialisable (no wall-clock values).
"""
from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import math
import time
from typing import Any

from . import calculator
from .constants import (ACTIONS, ENV_VERSION, ERR_CALC, ERR_EPISODE_OVER, ERR_INVALID_ARGS,
                        ERR_INVALID_CURSOR, ERR_NOT_RESET, ERR_UNKNOWN_ACTION, ERR_UNKNOWN_ID,
                        ERR_UNKNOWN_TABLE, EPISODE_TIME_LIMIT_S, MAX_AFFECTED_IDS,
                        MAX_EVIDENCE_IDS, MAX_EXPLANATION_CHARS, MAX_FINDINGS, MAX_STEPS,
                        MODE_EVAL, MODE_TRAIN, PAGE_LIMIT_MAX, PRIMARY_KEY, STEP_COST,
                        TABLE_FX, TABLE_INVOICES, TABLE_LEDGER, TABLE_PAYMENTS, TABLES, fx_id)
from .episode import Episode, dumps as _canonical_json, load_episode

MAX_MSG = 120
MAX_ECHO = 64
MAX_ACTION_DEPTH = 24        # nesting limit for any agent-supplied action (a valid submit is ~7 deep)
MAX_ACTION_NODES = 50_000    # total containers + scalars in one action
MAX_ACTION_STR_CHARS = 200_000   # total characters of all strings and dict keys in one action
MAX_RECORD_STR = 2_000       # longest string kept verbatim in a trajectory / grader record
MAX_RECORD_INT_BITS = 64     # larger ints are replaced by a placeholder in records
NOTE_ENDED_BY_THIS_STEP = "ended_by_this_step"   # episode_over error: this very step hit the time limit
NOTE_ALREADY_OVER = "already_over"               # episode_over error: the episode had ended before this step


class GraderReportNotReady(RuntimeError):
    """grader_report() was requested before the episode ended (truth/grader state stays sealed)."""


# ------------------------------------------------------------------ helpers
def _trunc(x: Any, n: int) -> str:
    if isinstance(x, str):
        s = x
    else:
        try:
            s = repr(x)
        except (ValueError, RecursionError):      # e.g. int with > 4300 digits
            s = "<unprintable>"
    s = "".join(ch if ch.isprintable() else "?" for ch in s)
    return s if len(s) <= n else s[: max(0, n - 3)] + "..."


def _err(code: str, msg: str, note: str | None = None) -> dict:
    out = {"code": code, "message": _trunc(msg, MAX_MSG)}
    if note is not None:
        out["note"] = note
    return out


def _safe_str(s: str) -> str:
    """UTF-8 encodable and bounded: lone surrogates become '?', oversized strings are cut."""
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        s = s.encode("utf-8", "replace").decode("utf-8")
    if len(s) > MAX_RECORD_STR:
        s = s[:MAX_RECORD_STR] + f"...<{len(s) - MAX_RECORD_STR} more chars>"
    return s


def _jsonable(x: Any, _depth: int = 0) -> Any:
    """Always strict-JSON-safe (json.dumps(..., allow_nan=False, ensure_ascii=False).encode('utf-8') works):
    non-finite floats and huge ints become placeholders, strings are sanitised and truncated."""
    if _depth > 64:
        return "<too deep>"
    d = _depth + 1
    if isinstance(x, dict):
        return {_safe_str(str(k)): _jsonable(v, d) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v, d) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((_jsonable(v, d) for v in x), key=repr)
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x if x.bit_length() <= MAX_RECORD_INT_BITS else "<int out of range>"
    if isinstance(x, float):
        return x if math.isfinite(x) else "<non-finite float>"
    if isinstance(x, str):
        return _safe_str(x)
    try:
        return _safe_str(str(x))
    except (ValueError, RecursionError):
        return "<unprintable>"


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _within_bounds(x: Any) -> bool:
    """Iterative (non-recursive) depth, size and total-string-length check of an agent-supplied action."""
    stack = [(x, 1)]
    nodes = 0
    chars = 0
    while stack:
        cur, depth = stack.pop()
        nodes += 1
        if depth > MAX_ACTION_DEPTH or nodes > MAX_ACTION_NODES:
            return False
        if isinstance(cur, str):
            chars += len(cur)
            if chars > MAX_ACTION_STR_CHARS:
                return False
        elif isinstance(cur, dict):
            if len(cur) > MAX_ACTION_NODES:
                return False
            for k, v in cur.items():
                if isinstance(k, str):
                    chars += len(k)
                    if chars > MAX_ACTION_STR_CHARS:
                        return False
                stack.append((v, depth + 1))
        elif isinstance(cur, (list, tuple)):
            stack.extend((v, depth + 1) for v in cur)
    return True


class _ArgError(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code
        self.msg = msg


# ------------------------------------------------------------------ task text
def build_task_text(manifest: dict, policy: dict, max_steps: int, time_limit_s: float) -> str:
    base = manifest.get("base_currency", "?")
    ep_id = manifest.get("episode_id", "?")
    fa = policy.get("finding_amounts", {}) if isinstance(policy, dict) else {}
    fi = policy.get("finding_affected_ids", {}) if isinstance(policy, dict) else {}
    tol = policy.get("amount_tolerance", "|claimed - expected| <= max(1, expected // 1000) minor units")
    types = {
        "duplicate_invoice": (
            "an invoice that duplicates an earlier invoice of the same entity (see duplicate_rule in policy).",
            "both invoice documents (the original's and the duplicate's doc_id) + the ledger row of the "
            "duplicate's AR debit."),
        "payment_misallocation": (
            "a payment whose AR credits posted in the ledger differ from what its invoice_refs say "
            "(partly or fully unapplied).",
            "the payment document + every ledger row carrying that payment_id; if the referenced invoices "
            "are in a foreign currency also add those invoice documents and the fx ids of their invoice rate dates."),
        "fx_error": (
            "an invoice or payment converted to base currency with the wrong rate (wrong date or wrong rate).",
            "the invoice or payment document + the AR debit ledger row (invoice) or the cash debit ledger row "
            "(payment) + the fx id of the CORRECT rate (after the fallback rule)."),
        "period_cutoff": (
            "an invoice issued outside the reporting period (as an instant) but posted inside it.",
            "the invoice document + its AR debit ledger row."),
    }
    lines: list[str] = []
    a = lines.append
    a("TASK: You are a financial auditor. Inspect one fictional company's records for ONE reporting period "
      "and submit the findings (accounting errors) you can prove with evidence. Records are read only "
      "through the actions below. The policy and manifest in this first observation define the rules. "
      "Some suspicious-looking patterns are legitimate; false findings cost you.")
    a("")
    a("FINDING TYPES (type -> affected_ids, claimed amount):")
    for t, (desc, _ev) in types.items():
        a(f"- {t}: {desc}")
        a(f"    affected_ids: {fi.get(t, 'see policy')}")
        a(f"    amount: {fa.get(t, 'see policy')}")
    a("")
    a(f"CURRENCY: every claimed amount is an integer in minor units of the base currency ({base}); "
      f"set currency to \"{base}\". Amount tolerance: {tol}.")
    a("")
    a("EVIDENCE: evidence_ids may contain ledger row_id values, document doc_id values (invoices.doc_id, "
      "payments.doc_id) and fx ids of the form 'fx:<date>:<source>:<target>' (the fx_id field of fx rows). "
      "Every cited id must exist. Maximum 10 evidence ids per finding. A finding needs a sufficient "
      "evidence set (extra existing ids are allowed while the total stays within 10). Sufficient sets:")
    for t, (_d, ev) in types.items():
        a(f"- {t}: {ev}")
    a("")
    a("NOT ERRORS: a foreign-currency overpayment refund can leave a small base-currency residual in "
      "2300_unapplied_cash (the excess is booked at the payment-date rate, the refund at the refund-date rate; "
      "no 7900_fx_gain_loss row is posted for it). Prepayments, overpayments, split payments, recurring "
      "billing with a different service_period and realised FX gain/loss are legitimate (see policy).")
    a("")
    a("ACTIONS (send {\"action\": <name>, \"args\": {...}}):")
    a(f"- list_records {{\"table\": \"ledger|invoices|payments|fx\", \"cursor\": null, \"limit\": 50}}: "
      f"page of rows (limit 1..{PAGE_LIMIT_MAX}); pass result.next_cursor back as cursor for the next page. "
      "Invoice rows omit line_items (use get_record/get_document). Result: {table, rows, next_cursor, total}.")
    a("- get_record {\"table\": <table>, \"id\": <primary key>}: primary keys are ledger row_id, invoices "
      "invoice_id, payments payment_id, fx fx_id. Invoices include line_items.")
    a("- get_document {\"doc_id\": <doc_id>}: full invoice or payment document.")
    a("- calculator {\"expression\": <str>}: exact decimal arithmetic (+ - * / // % **, parentheses, "
      "round(x, n), abs, min, max); returns result.value as a string.")
    a("- submit {\"findings\": {\"episode_id\": \"%s\", \"findings\": [<finding>, ...]}}: ends the episode." % ep_id)
    a("")
    a("FINDING OBJECT (exactly these keys, no others): {\"type\": <one of the 4 types>, "
      "\"affected_ids\": [<id>, ...], \"claimed_amount_minor\": <int >= 0>, \"currency\": \"%s\", "
      "\"evidence_ids\": [<id>, ...], \"explanation\": <string>}. Limits: at most %d findings, %d affected_ids, "
      "%d evidence_ids, %d characters of explanation. A malformed submission scores -1."
      % (base, MAX_FINDINGS, MAX_AFFECTED_IDS, MAX_EVIDENCE_IDS, MAX_EXPLANATION_CHARS))
    a("")
    a(f"LIMITS: {max_steps} steps in total including submit (errors also use a step), wall-clock limit "
      f"{int(time_limit_s)} s. If the step budget runs out on a non-submit step or time runs out, the "
      "episode is truncated and scored as an empty submission. Steps after the end change nothing.")
    a("")
    a("REWARD (terminal): reward = clip(2*F1 - 1 - %s*inspection_steps, -1, 1), inspection_steps = steps "
      "other than submit. A finding is a true positive if it matches a real issue (same type, same affected_ids "
      "set, same currency, amount within tolerance, sufficient evidence); matching is one-to-one, so repeats "
      "and unmatched findings are false positives. F1 = 2TP/(2TP+FP+FN); no findings on clean books gives "
      "F1 = 1; findings on clean books give F1 = 0." % STEP_COST)
    return "\n".join(lines)


# ------------------------------------------------------------------ env
class AuditEnv:
    def __init__(self, mode: str = MODE_TRAIN, max_steps: int = MAX_STEPS,
                 time_limit_s: float = EPISODE_TIME_LIMIT_S):
        if mode not in (MODE_TRAIN, MODE_EVAL):
            raise ValueError(f"mode must be {MODE_TRAIN!r} or {MODE_EVAL!r}")
        if not _is_int(max_steps) or max_steps < 1:
            raise ValueError("max_steps must be a positive int")
        self.mode = mode
        self.max_steps = max_steps
        self.time_limit_s = time_limit_s
        self._clock = time.monotonic          # replaceable in tests
        self._reset_state()
        self._ep: Episode | None = None          # PUBLIC part only (truth/provenance are stripped)
        self._truth: dict = {}                   # sealed until scoring; see README: only the server is a boundary
        self._cursor_key = b""
        self._started = False

    # -------------------------------------------------------------- state
    def _reset_state(self) -> None:
        self._step_n = 0
        self._terminated = False
        self._truncated = False
        self._t0 = 0.0
        self._traj: list[dict] = []
        self._errors: list[dict] = []
        self._result: dict | None = None
        self._public_ids: set[str] = set()
        self._rows: dict[str, list[dict]] = {}
        self._index: dict[str, dict[str, dict]] = {}
        self._docs: dict[str, tuple[str, dict]] = {}
        self._ended_by: str | None = None

    # -------------------------------------------------------------- reset
    def reset(self, seed: int | None = None, config: Any = None,
              options: dict | None = None) -> tuple[dict, dict]:
        self._reset_state()
        self._ep = None
        self._truth = {}
        self._cursor_key = b""
        self._started = False
        ep = self._load(seed, config)
        ep = self._transform(ep, options)
        # Keep truth out of the Episode object this env holds; it is only touched by _score().
        self._truth = ep.truth or {}
        self._cursor_key = hashlib.sha256(
            b"auditgym-cursor-key|" + _canonical_json(ep.provenance or self._truth).encode("utf-8")).digest()
        ep = ep.public_copy()
        self._ep = ep
        self._started = True
        self._build_views(ep)
        self._t0 = self._clock()
        obs = {
            "episode_id": ep.episode_id,
            "step": 0,
            "steps_remaining": self.max_steps,
            "result": None,
            "error": None,
            "task": build_task_text(ep.manifest, ep.policy, self.max_steps, self.time_limit_s),
            "manifest": copy.deepcopy(ep.manifest),
            "policy": copy.deepcopy(ep.policy),
        }
        self._traj.append({"episode_id": ep.episode_id, "step": 0, "action": None,
                           "observation": copy.deepcopy(obs), "reward": 0.0,
                           "terminated": False, "truncated": False})
        return obs, {}

    def _load(self, seed: Any, config: Any) -> Episode:
        if isinstance(config, dict) and "episode_dir" in config:
            try:
                return load_episode(config["episode_dir"], config.get("grader_dir"))
            except Exception:           # never echo paths or OS error text
                from .generator.config import ConfigError
                raise ConfigError("episode could not be loaded from episode_dir/grader_dir") from None
        try:
            from .generator import generate_episode  # lazy: generator is built by another module
        except ImportError:
            from .generator.build import generate_episode
        if seed is None:
            seed = 0
        if not _is_int(seed):
            raise ValueError("seed must be an int")
        return generate_episode(seed, "train" if config is None else config)

    @staticmethod
    def _transform(ep: Episode, options: dict | None) -> Episode:
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

    def _build_views(self, ep: Episode) -> None:
        fx_rows = []
        for r in ep.fx:
            row = {k: r[k] for k in ("date", "source_currency", "target_currency", "rate")}
            row["fx_id"] = fx_id(r["date"], r["source_currency"], r["target_currency"])
            fx_rows.append(row)
        self._rows = {TABLE_LEDGER: ep.ledger, TABLE_INVOICES: ep.invoices,
                      TABLE_PAYMENTS: ep.payments, TABLE_FX: fx_rows}
        self._index = {t: {} for t in TABLES}
        for t in TABLES:
            pk = PRIMARY_KEY[t]
            for r in self._rows[t]:
                self._index[t].setdefault(r[pk], r)
        for kind, rows in (("invoice", ep.invoices), ("payment", ep.payments)):
            for r in rows:
                self._docs.setdefault(r["doc_id"], (kind, r))
        self._public_ids = ep.public_ids()

    # -------------------------------------------------------------- observation
    def _obs(self, result: Any = None, error: dict | None = None) -> dict:
        return {"episode_id": self._ep.episode_id if self._ep else None,
                "step": self._step_n,
                "steps_remaining": max(0, self.max_steps - self._step_n),
                "result": result, "error": error}

    def _record(self, action: Any, obs: dict, reward: float) -> None:
        self._traj.append({"episode_id": obs["episode_id"], "step": obs["step"],
                           "action": _jsonable(action),
                           "observation": _jsonable(obs), "reward": reward,
                           "terminated": self._terminated, "truncated": self._truncated})

    # -------------------------------------------------------------- step
    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        if not self._started or self._ep is None:
            return (self._obs(error=_err(ERR_NOT_RESET, "call reset before step")), 0.0, False, False, {})
        if self._terminated or self._truncated:
            return (self._obs(error=_err(ERR_EPISODE_OVER, "episode is over", NOTE_ALREADY_OVER)), 0.0,
                    self._terminated, self._truncated, {})

        self._step_n += 1
        in_bounds = _within_bounds(action)
        is_submit = isinstance(action, dict) and action.get("action") == "submit"
        shown = {"action": "submit" if is_submit else "?", "args": "<rejected: too large or nested>"}
        # wall-clock limit (monotonic, measured from reset; includes the policy's own latency). This is a
        # cooperative check on the next step; the out-of-process server adds a stdin watchdog (expire_if_overdue).
        if self._clock() - self._t0 > self.time_limit_s:
            return self._finish_truncated(action if in_bounds else shown,
                                          "time limit exceeded; episode truncated", "time")

        if not in_bounds:
            # hostile / absurd payload: bounded error (or rejected submission), the step is consumed and
            # the stored record never holds the raw payload
            if is_submit:
                return self._do_submit(shown, None, reject="submission too large or too deeply nested")
            return self._finish_error(shown, _err(ERR_INVALID_ARGS, "action too large or too deeply nested"))

        try:
            return self._step_inner(action)
        except RecursionError:
            return self._finish_error({"action": "?"}, _err(ERR_INVALID_ARGS, "action could not be processed"))

    def _step_inner(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        if isinstance(action, dict) and action.get("action") == "submit":
            if any(k not in ("action", "args") for k in action):
                return self._do_submit(action, None, reject="submit action has unexpected top-level keys")
            return self._do_submit(action, action.get("args", {}))

        name, args, bad = self._parse_action(action)
        if bad is not None:
            return self._finish_error(action, bad)

        try:
            result = self._dispatch(name, args)
            error = None
        except _ArgError as e:
            result, error = None, _err(e.code, e.msg)
        except calculator.CalculatorError as e:
            result, error = None, _err(ERR_CALC, str(e))
        if error is not None:
            return self._finish_error(action, error)

        obs = self._obs(result=result)
        if self._step_n >= self.max_steps:
            return self._finish_truncated(action, None, "steps", obs=obs)
        self._record(action, obs, 0.0)
        return obs, 0.0, False, False, {}

    def _parse_action(self, action: Any) -> tuple[str | None, dict, dict | None]:
        if not isinstance(action, dict):
            return None, {}, _err(ERR_UNKNOWN_ACTION, "action must be an object {action, args}")
        name = action.get("action")
        if not isinstance(name, str) or name not in ACTIONS:
            shown = _trunc(name, MAX_ECHO) if isinstance(name, str) else "?"
            return None, {}, _err(ERR_UNKNOWN_ACTION, f"unknown action: {shown}")
        extra = [k for k in action if k not in ("action", "args")]
        if extra:
            return None, {}, _err(ERR_INVALID_ARGS, f"unexpected key: {_trunc(extra[0], MAX_ECHO)}")
        args = action.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return None, {}, _err(ERR_INVALID_ARGS, "args must be an object")
        return name, args, None

    def _finish_error(self, action: Any, error: dict) -> tuple[dict, float, bool, bool, dict]:
        self._errors.append({"step": self._step_n, "code": error["code"], "message": error["message"]})
        obs = self._obs(error=error)
        if self._step_n >= self.max_steps:
            return self._finish_truncated(action, None, "steps", obs=obs)
        self._record(action, obs, 0.0)
        return obs, 0.0, False, False, {}

    # -------------------------------------------------------------- actions
    def _dispatch(self, name: str, args: dict) -> dict:
        if name == "list_records":
            return self._list_records(args)
        if name == "get_record":
            return self._get_record(args)
        if name == "get_document":
            return self._get_document(args)
        if name == "calculator":
            return self._calculator(args)
        raise _ArgError(ERR_UNKNOWN_ACTION, "unknown action")

    @staticmethod
    def _check_keys(args: dict, allowed: tuple[str, ...]) -> None:
        extra = [k for k in args if k not in allowed]
        if extra:
            raise _ArgError(ERR_INVALID_ARGS, f"unexpected argument: {_trunc(extra[0], MAX_ECHO)}")

    def _table(self, args: dict) -> str:
        t = args.get("table")
        if not isinstance(t, str):
            raise _ArgError(ERR_INVALID_ARGS, "table must be a string")
        if t not in TABLES:
            raise _ArgError(ERR_UNKNOWN_TABLE, f"unknown table: {_trunc(t, MAX_ECHO)}")
        return t

    def _cursor_sum(self, table: str, offset: int) -> str:
        # keyed (HMAC) with a per-episode secret derived from grader-side material the agent never sees,
        # so a cursor cannot be forged from public values
        eid = self._ep.episode_id
        msg = f"{eid}|{ENV_VERSION}|{table}|{offset}".encode()
        return hmac.new(self._cursor_key, msg, hashlib.sha256).hexdigest()[:16]

    def _encode_cursor(self, table: str, offset: int) -> str:
        raw = f"{table}:{offset}:{self._cursor_sum(table, offset)}".encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    def _decode_cursor(self, cursor: Any, table: str, total: int) -> int:
        bad = _ArgError(ERR_INVALID_CURSOR, "invalid cursor")
        if not isinstance(cursor, str) or not cursor or len(cursor) > 200:
            raise bad
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode("ascii")
            tbl, off, chk = raw.split(":")
        except Exception:
            raise bad from None
        if tbl != table or not (off.isascii() and off.isdigit()) or len(off) > 9:
            raise bad
        offset = int(off)
        if offset > total or not hmac.compare_digest(chk, self._cursor_sum(table, offset)):
            raise bad
        return offset

    def _list_records(self, args: dict) -> dict:
        self._check_keys(args, ("table", "cursor", "limit"))
        table = self._table(args)
        limit = args.get("limit", PAGE_LIMIT_MAX)
        if not _is_int(limit) or not 1 <= limit <= PAGE_LIMIT_MAX:
            raise _ArgError(ERR_INVALID_ARGS, f"limit must be an int in 1..{PAGE_LIMIT_MAX}")
        rows = self._rows[table]
        total = len(rows)
        cursor = args.get("cursor")
        offset = 0 if cursor is None else self._decode_cursor(cursor, table, total)
        page = rows[offset: offset + limit]
        if table == TABLE_INVOICES:
            page = [{k: v for k, v in r.items() if k != "line_items"} for r in page]
        nxt = offset + len(page)
        return {"table": table, "rows": copy.deepcopy(page),
                "next_cursor": self._encode_cursor(table, nxt) if nxt < total else None,
                "total": total}

    def _get_record(self, args: dict) -> dict:
        self._check_keys(args, ("table", "id"))
        table = self._table(args)
        rid = args.get("id")
        if not isinstance(rid, str):
            raise _ArgError(ERR_INVALID_ARGS, "id must be a string")
        rec = self._index[table].get(rid)
        if rec is None:
            raise _ArgError(ERR_UNKNOWN_ID, f"unknown id in {table}: {_trunc(rid, MAX_ECHO)}")
        return {"table": table, "record": copy.deepcopy(rec)}

    def _get_document(self, args: dict) -> dict:
        self._check_keys(args, ("doc_id",))
        did = args.get("doc_id")
        if not isinstance(did, str):
            raise _ArgError(ERR_INVALID_ARGS, "doc_id must be a string")
        hit = self._docs.get(did)
        if hit is None:
            raise _ArgError(ERR_UNKNOWN_ID, f"unknown doc_id: {_trunc(did, MAX_ECHO)}")
        kind, rec = hit
        return {"doc_type": kind, "document": copy.deepcopy(rec)}

    def _calculator(self, args: dict) -> dict:
        self._check_keys(args, ("expression",))
        expr = args.get("expression")
        if not isinstance(expr, str):
            raise _ArgError(ERR_INVALID_ARGS, "expression must be a string")
        return {"value": calculator.evaluate(expr)}

    # -------------------------------------------------------------- submit / scoring
    def _do_submit(self, action: Any, args: Any, reject: str | None = None
                   ) -> tuple[dict, float, bool, bool, dict]:
        """submit ends the episode. Any structurally invalid submission (args missing / not an object /
        unknown keys / findings not an object / bad findings) scores -1, terminates and is logged."""
        inspection = self._step_n - 1
        if reject is None:
            if args is None:
                args = {}
            if not isinstance(args, dict) or set(args) != {"findings"}:
                reject = "submit args must be exactly {findings: <object>}"
        if reject is not None:
            res = {"valid": False, "error": reject}
        else:
            res = self._score(args["findings"], inspection)
        invalid = not res.get("valid", False)
        reason = _trunc(res.get("error") or "invalid submission", 90)
        if invalid:
            self._errors.append({"step": self._step_n, "code": ERR_INVALID_ARGS,
                                 "message": str(res.get("error"))[:500]})
        self._terminated = True
        self._ended_by = "submit"
        self._store_result(res, inspection, invalid)
        reward = -1.0 if invalid else float(res["reward_shared"])
        # Eval mode must not leak whether the submission was valid: identical terminal observation either
        # way (reason and -1 live only in grader_report). Train mode keeps the explicit error.
        if invalid and self.mode != MODE_EVAL:
            err = _err(ERR_INVALID_ARGS, f"submission rejected: {reason}")
        else:
            err = None
        obs = self._obs(error=err)
        return self._finish_terminal(action, obs, reward)

    def _score(self, submission: Any, inspection: int) -> dict:
        from . import matcher
        truth = self._truth or {"issues": [], "episode_id": self._ep.episode_id,
                                "base_currency": self._ep.manifest.get("base_currency")}
        try:
            res = matcher.score(submission, truth, self._public_ids, inspection, self._ep.episode_id)
        except Exception as e:  # defensive: a malformed submission must never crash the env
            res = {"valid": False, "error": f"submission could not be scored ({type(e).__name__})"}
        return res

    def _store_result(self, res: dict, inspection: int, invalid: bool) -> None:
        res = _jsonable(res)
        if invalid:
            res["reward_shared"] = -1.0
            res["reward_experimental"] = -1.0
        self._result = {"score": res, "inspection_steps": inspection, "invalid": invalid}

    def _finish_truncated(self, action: Any, msg: str | None, why: str,
                          obs: dict | None = None) -> tuple[dict, float, bool, bool, dict]:
        inspection = self._step_n          # the step that hit the limit is an (unfinished) inspection step
        empty = {"episode_id": self._ep.episode_id, "findings": []}
        res = self._score(empty, inspection)
        self._truncated = True
        self._ended_by = why
        self._store_result(res, inspection, not res.get("valid", False))
        if obs is None:
            obs = self._obs(error=_err(ERR_EPISODE_OVER, msg or "episode truncated", NOTE_ENDED_BY_THIS_STEP))
        reward = -1.0 if self._result["invalid"] else float(res["reward_shared"])
        return self._finish_terminal(action, obs, reward)

    def _finish_terminal(self, action: Any, obs: dict, reward: float) -> tuple[dict, float, bool, bool, dict]:
        if self.mode == MODE_TRAIN:
            s = self._result["score"]
            info = {
                "metrics": {k: s.get(k, 0) for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
                "reward_shared": reward,
                "reward_experimental": s.get("reward_experimental", reward),
                "inspection_steps": self._result["inspection_steps"],
                "invalid_submission": self._result["invalid"],
            }
            out_reward = reward
        else:
            info, out_reward = {}, 0.0
        self._record(action, obs, out_reward)
        return obs, out_reward, self._terminated, self._truncated, info

    # -------------------------------------------------------------- grader / trajectory
    def abandon(self) -> None:
        """Grader side: the harness re-reset an unfinished episode. It is scored as truncated with an
        empty answer (never silently dropped) so a grader log exists for it."""
        if not self._started or self._ep is None or self.ended():
            return
        empty = {"episode_id": self._ep.episode_id, "findings": []}
        res = self._score(empty, self._step_n)
        self._truncated = True
        self._ended_by = "abandoned"
        self._store_result(res, self._step_n, not res.get("valid", False))

    def expire_if_overdue(self) -> bool:
        """Harness-side watchdog hook: truncate (ended_by=time, empty answer) an unfinished episode whose
        wall-clock limit has passed although no step arrived. Returns True if it truncated."""
        if not self.in_progress() or self._clock() - self._t0 <= self.time_limit_s:
            return False
        empty = {"episode_id": self._ep.episode_id, "findings": []}
        res = self._score(empty, self._step_n)
        self._truncated = True
        self._ended_by = "time"
        self._store_result(res, self._step_n, not res.get("valid", False))
        return True

    def seconds_left(self) -> float | None:
        """Wall-clock seconds until the time limit for an unfinished episode (None if not running)."""
        if not self.in_progress():
            return None
        return self.time_limit_s - (self._clock() - self._t0)

    def in_progress(self) -> bool:
        return bool(self._started and self._ep is not None and not self.ended())

    def ended(self) -> bool:
        return self._terminated or self._truncated

    def grader_report(self) -> dict:
        """Grader side only. Contains match tables and errors; never given to a policy in eval.
        Raises GraderReportNotReady until the episode has ended (or before reset)."""
        if not self._started or self._ep is None or not self.ended():
            raise GraderReportNotReady("grader_report is available only after the episode has ended")
        rep: dict[str, Any] = {
            "episode_id": self._ep.episode_id, "mode": self.mode, "ended": self.ended(),
            "terminated": self._terminated, "truncated": self._truncated,
            "ended_by": self._ended_by, "steps": self._step_n,
            "n_truth_issues": len((self._truth or {}).get("issues", [])),
            "errors": _jsonable(self._errors),
        }
        if self._result is not None:
            s = self._result["score"]
            rep.update({
                "inspection_steps": self._result["inspection_steps"],
                "invalid_submission": self._result["invalid"],
                "metrics": {k: s.get(k, 0) for k in ("precision", "recall", "f1", "tp", "fp", "fn")},
                "reward_shared": s.get("reward_shared"),
                "reward_experimental": s.get("reward_experimental"),
                "pairs": s.get("pairs", []),
                "score": s,
            })
        return rep

    def trajectory(self) -> list[dict]:
        return copy.deepcopy(self._traj)      # always strict-JSON-safe (see _jsonable)
