"""oracle (GRADER ONLY): submits exactly the truth issues with their first evidence set.

Constructed by the runner inside the grader context with the parsed truth.json. It makes no
inspection steps, so its reward is the maximum achievable (about +1.0). It demonstrates that
the reward ceiling is reachable; it is not evidence of agent ability.
"""
from __future__ import annotations

from .base import Policy, make_finding, submit_action


class Oracle(Policy):
    name = "oracle"

    def __init__(self, truth: dict, seed: int = 0):
        if not isinstance(truth, dict) or "issues" not in truth:
            raise ValueError("oracle needs the truth dict (grader-side only)")
        self.truth = truth
        self.seed = seed
        self.episode_id = ""

    def reset(self, seed: int, first_obs: dict) -> None:
        self.seed = seed
        self.episode_id = first_obs.get("episode_id", self.truth.get("episode_id", ""))

    def act(self, obs: dict) -> dict:
        findings = []
        for issue in self.truth["issues"]:
            sets = issue.get("evidence_sets") or [[]]
            findings.append(make_finding(
                issue["type"], list(issue["affected_ids"]), int(issue["expected_amount_minor"]),
                issue["currency"], list(sets[0]), "oracle: ground-truth answer"))
        return submit_action(self.episode_id, findings)
