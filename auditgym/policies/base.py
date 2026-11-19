"""Policy interface and small helpers shared by the baseline policies.

A policy only ever sees observations (JSON dicts) and emits action dicts
``{"action": name, "args": {...}}``. Policies never import truth; the single grader-side
exception (``oracle``) receives the truth in its constructor from the runner.

``ScriptedPolicy`` lets a policy be written as a generator::

    def script(self):
        obs = yield list_action("invoices")      # obs of the step that action produced
        ...
        yield submit_action(self.episode_id, findings)

``act(obs)`` is called with the observation that resulted from the previous action (the
first call receives the reset observation). The generator gets each later observation via
``send``. A safety net in ``act`` forces a submit when only one step remains.
"""
from __future__ import annotations

from typing import Any, Generator, Iterator

from ..constants import PAGE_LIMIT_MAX

Action = dict
Obs = dict


class Policy:
    name = "policy"

    def reset(self, seed: int, first_obs: Obs) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def act(self, obs: Obs) -> Action:  # pragma: no cover - interface
        raise NotImplementedError


# ------------------------------------------------------------------ action builders
def list_action(table: str, cursor: str | None = None, limit: int = PAGE_LIMIT_MAX) -> Action:
    return {"action": "list_records", "args": {"table": table, "cursor": cursor, "limit": limit}}


def get_record_action(table: str, rec_id: str) -> Action:
    return {"action": "get_record", "args": {"table": table, "id": rec_id}}


def get_document_action(doc_id: str) -> Action:
    return {"action": "get_document", "args": {"doc_id": doc_id}}


def calculator_action(expression: str) -> Action:
    return {"action": "calculator", "args": {"expression": expression}}


def make_finding(type_: str, affected_ids: list[str], amount_minor: int, currency: str,
                 evidence_ids: list[str], explanation: str = "") -> dict:
    return {"type": type_, "affected_ids": list(affected_ids),
            "claimed_amount_minor": int(amount_minor), "currency": currency,
            "evidence_ids": list(evidence_ids)[:10], "explanation": explanation[:1000]}


def submit_action(episode_id: str, findings: list[dict]) -> Action:
    return {"action": "submit", "args": {"findings": {"episode_id": episode_id,
                                                       "findings": list(findings)}}}


def is_submit(action: Action) -> bool:
    return isinstance(action, dict) and action.get("action") == "submit"


# ------------------------------------------------------------------ scripted base
class ScriptedPolicy(Policy):
    """Generator-driven policy. Subclasses implement ``script``."""

    name = "scripted"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.episode_id = ""
        self.first_obs: Obs = {}
        self.remaining = 0
        self._gen: Iterator[Action] | None = None

    # -- lifecycle
    def reset(self, seed: int, first_obs: Obs) -> None:
        self.seed = seed
        self.first_obs = first_obs
        self.episode_id = first_obs.get("episode_id", "")
        self.remaining = int(first_obs.get("steps_remaining", 0))
        self._gen = None
        self.on_reset()

    def on_reset(self) -> None:
        """Hook for subclasses to clear per-episode state."""

    def script(self) -> Generator[Action, Obs, None]:  # pragma: no cover - abstract
        raise NotImplementedError
        yield  # noqa

    def fallback_findings(self) -> list[dict]:
        """Findings submitted if the budget runs out or the script ends without submitting."""
        return []

    def act(self, obs: Obs) -> Action:
        if "steps_remaining" in obs:
            self.remaining = int(obs["steps_remaining"])
        action: Action | None = None
        try:
            if self._gen is None:
                self._gen = self.script()
                action = next(self._gen)
            else:
                action = self._gen.send(obs)
        except StopIteration:
            action = None
        if action is None or (self.remaining <= 1 and not is_submit(action)):
            # script ended, or one step left and the script wants to keep inspecting
            return submit_action(self.episode_id, self.fallback_findings())
        return action

    # -- budget helpers for scripts
    def can_inspect(self, reserve: int = 1) -> bool:
        """True iff one more non-submit action still leaves ``reserve`` steps for submit."""
        return self.remaining - 1 >= reserve

    def paginate(self, table: str, limit: int = PAGE_LIMIT_MAX,
                 max_pages: int | None = None) -> Generator[Action, Obs, list[dict]]:
        """Sub-generator: ``rows = yield from self.paginate("ledger")``."""
        rows: list[dict] = []
        cursor: str | None = None
        pages = 0
        while True:
            if not self.can_inspect() or (max_pages is not None and pages >= max_pages):
                break
            obs = yield list_action(table, cursor, limit)
            if "steps_remaining" in obs:
                self.remaining = int(obs["steps_remaining"])
            res = obs.get("result")
            if obs.get("error") or not isinstance(res, dict):
                break
            rows.extend(res.get("rows") or [])
            pages += 1
            cursor = res.get("next_cursor")
            if not cursor:
                break
        return rows
