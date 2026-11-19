"""Documentation / repo-hygiene regression tests: the docs must not drift from the shipped code and configs."""
from __future__ import annotations

import json
import os
import re

import pytest

from auditgym.constants import EPISODE_TIME_LIMIT_S, MAX_STEPS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts: str) -> str:
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def _config_versions() -> dict[str, str]:
    return {n: json.load(open(os.path.join(ROOT, "configs", f"{n}.json"), encoding="utf-8"))["config_version"]
            for n in ("train", "dev", "holdout")}


def test_spec_config_example_matches_the_shipped_config_version_and_is_marked_illustrative():
    spec = _read("SPEC.md")
    m = re.search(r'\{"config_id":"train","config_version":"([0-9.]+)"', spec)
    assert m, "SPEC.md has no config example"
    assert m.group(1) == _config_versions()["train"]
    assert "Illustrative example" in spec


def test_changelog_covers_the_shipped_config_versions_back_to_1_0_0():
    log = _read("CHANGELOG.md")
    for v in set(_config_versions().values()):
        assert f"Config {v}" in log
    assert "Config 1.0.0" in log


def test_ci_workflow_runs_the_quick_official_check():
    ci = _read(".github", "workflows", "ci.yml")
    assert "python -m auditgym check --quick" in ci


def test_pyproject_is_valid_toml_with_ruff_config():
    tomllib = pytest.importorskip("tomllib")
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        cfg = tomllib.load(f)
    assert cfg["project"]["dependencies"] == []
    assert "ruff" in cfg["tool"]


def test_readme_and_design_state_the_isolation_boundary_honestly():
    for name in ("README.md", os.path.join("docs", "DESIGN.md")):
        t = _read(name)
        assert "Only the server path is an isolation boundary" in t, name
        assert "trusted-code only" in t, name
        assert "never stored" in t or "never written to disk" in t, name
        assert "HMAC" in t, name
        assert "watchdog" in t, name
    readme = _read("README.md")
    assert "docker run" in readme and "grader_data/" in readme
    assert "hygiene, not a sandbox" in " ".join(readme.split())          # the AgentHandle is not presented as isolation


def test_readme_documents_runtime_network_caveat_and_public_holdout():
    readme = _read("README.md")
    assert "Network is needed once" in readme and "offline" in readme
    assert "NOT hidden-seed results" in readme and "own hidden seeds" in readme
    assert "pip install -e ." in readme
    assert "wall s" in readme and "snapshot of the last run" in readme
    assert "reward-ceiling" in readme and "train-mode" in readme.lower().replace("*", "")
    assert "careful_auditor_degraded" in readme and "failure analysis" in readme.lower()


def test_design_limitations_cover_the_generator_and_policy_notes():
    d = _read("docs", "DESIGN.md")
    for needle in ("canonical maximal and minimal evidence sets", "Single-id issues by design", "natural keys",
                   "no DST", "Cross-platform determinism status", "Repeated ids", "step cost",
                   "world-readable by design", "two-directional"):
        assert needle.lower() in d.lower(), needle


def test_documented_limits_match_the_constants():
    readme = _read("README.md")
    assert f"{int(EPISODE_TIME_LIMIT_S)} s" in readme
    assert f"{MAX_STEPS} steps" in readme
