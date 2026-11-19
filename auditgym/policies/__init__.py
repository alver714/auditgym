"""Baseline policies and registry. ``get_policy(name, seed, truth=None)``."""
from __future__ import annotations

from .base import Policy, ScriptedPolicy

POLICY_NAMES = ("random_explorer", "reconcile", "flag_all", "oracle")
# extra diagnostic policies (NOT part of the 4 required baselines)
EXTRA_POLICY_NAMES = ("careful_auditor", "careful_auditor_degraded")


def get_policy(name: str, seed: int = 0, truth: dict | None = None) -> Policy:
    """Build a policy. ``truth`` is accepted only for ``oracle`` (grader-side, from the runner)."""
    if name == "random_explorer":
        from .random_explorer import RandomExplorer
        return RandomExplorer(seed)
    if name == "reconcile":
        from .reconcile import Reconcile
        return Reconcile(seed)
    if name == "flag_all":
        from .flag_all import FlagAll
        return FlagAll(seed)
    if name == "careful_auditor":
        from .careful_auditor import CarefulAuditor
        return CarefulAuditor(seed)
    if name == "careful_auditor_degraded":
        from .careful_auditor_degraded import CarefulAuditorDegraded
        return CarefulAuditorDegraded(seed)
    if name == "oracle":
        if truth is None:
            raise ValueError("oracle requires truth (grader context only)")
        from .oracle import Oracle
        return Oracle(truth, seed)
    raise ValueError(f"unknown policy {name!r}; choose from {POLICY_NAMES + EXTRA_POLICY_NAMES}")


__all__ = ["Policy", "ScriptedPolicy", "POLICY_NAMES", "EXTRA_POLICY_NAMES", "get_policy"]
