"""Shared pytest fixtures for AuditGym tests (stdlib + pytest only)."""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def repo_root() -> str:
    return ROOT


@pytest.fixture(scope="session")
def clean_config() -> dict:
    """`train` config pinned to a clean company (no issues, no interactions)."""
    from auditgym.generator import config_with_force
    return config_with_force("train", {"clean": True})


@pytest.fixture(scope="session")
def issue_config() -> dict:
    """`train` config pinned to a company with 4 plain issues + both interactions."""
    from auditgym.generator import config_with_force
    return config_with_force("train", {
        "issues": {"duplicate_invoice": 1, "payment_misallocation": 1, "fx_error": 1, "period_cutoff": 1},
        "interactions": {"fx_partial_payment_at_boundary": 1, "cutoff_with_wrong_rate_date": 1},
    })


@pytest.fixture
def make_env():
    """Factory for in-process AuditEnv instances."""
    from auditgym.env import AuditEnv

    def _make(mode: str = "train", **kw) -> AuditEnv:
        return AuditEnv(mode=mode, **kw)
    return _make


@pytest.fixture
def make_remote():
    """Factory for RemoteAuditEnv instances; all are closed at teardown."""
    from auditgym.client import RemoteAuditEnv
    made = []

    def _make(*a, **kw):
        r = RemoteAuditEnv(*a, **kw)
        made.append(r)
        return r
    yield _make
    for r in made:
        try:
            r.close()
        except Exception:
            pass
