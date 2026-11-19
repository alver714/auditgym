"""schemas/*.json describe every public and private file, the actions and the observations."""
from __future__ import annotations

import json
import os

import pytest

from auditgym import schema_check
from auditgym.client import RemoteAuditEnv  # noqa: F401  (import check only)
from auditgym.constants import ISSUE_TYPES
from auditgym.env import AuditEnv
from auditgym.episode import load_episode
from auditgym.generator.build import generate_episode
from auditgym.generator.config import config_with_force

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_FILES = sorted(f for f in os.listdir(os.path.join(ROOT, "schemas")) if f.endswith(".schema.json"))


def test_schema_files_are_valid_json_with_titles():
    assert len(SCHEMA_FILES) >= 11
    for f in SCHEMA_FILES:
        d = schema_check.load_schema(f)
        assert d.get("title") and d.get("$schema")


@pytest.mark.parametrize("seed,force", [
    (11, {"clean": True}),
    (12, {"issues": {t: 1 for t in ISSUE_TYPES}}),
    (13, {"interactions": {"fx_partial_payment_at_boundary": 1, "cutoff_with_wrong_rate_date": 1}}),
    (14, {"issues": {"fx_error": 12}}),
])
def test_generated_episodes_conform_to_the_schemas(seed, force):
    ep = generate_episode(seed, config_with_force("holdout", force))
    assert schema_check.check_episode(ep) == []


@pytest.mark.parametrize("split", ["train", "dev"])
def test_committed_episodes_conform_to_the_schemas(split):
    pub, priv = os.path.join(ROOT, "episodes", split), os.path.join(ROOT, "grader_data", split)
    for eid in sorted(os.listdir(pub)):
        ep = load_episode(os.path.join(pub, eid), os.path.join(priv, eid))
        assert schema_check.check_episode(ep) == [], eid


def test_schema_checker_actually_rejects_bad_data():
    ep = generate_episode(12, config_with_force("train", {"issues": {"fx_error": 1}}))
    bad = json.loads(json.dumps(ep.invoices[0]))
    bad["issued_at"] = "2026-03-01"                       # no time / offset
    bad["total_minor"] = "5"
    bad["extra"] = 1
    assert len(schema_check.check(bad, schema_check.load_schema("invoice.schema.json"))) >= 3
    pay = json.loads(json.dumps(ep.payments[0]))
    del pay["invoice_refs"]
    assert schema_check.check(pay, schema_check.load_schema("payment.schema.json"))
    assert schema_check.check({"episode_id": "x", "findings": [{"type": "theft"}]},
                              schema_check.load_schema("findings.schema.json"))


def test_observations_and_actions_conform():
    env = AuditEnv(mode="train")
    obs, _ = env.reset(seed=12, config=config_with_force("train", {"issues": {"fx_error": 1}}))
    assert schema_check.check(obs, schema_check.load_schema("observation.schema.json")) == []
    actions = [
        {"action": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 5}},
        {"action": "get_record", "args": {"table": "fx", "id": "nope"}},
        {"action": "get_document", "args": {"doc_id": "D-x"}},
        {"action": "calculator", "args": {"expression": "1+1"}},
        {"action": "submit", "args": {"findings": {"episode_id": obs["episode_id"], "findings": []}}},
    ]
    sch = schema_check.load_schema("action.schema.json")
    for a in actions:
        assert schema_check.check(a, sch) == [], a
        o, *_ = env.step(a)
        assert schema_check.check(o, schema_check.load_schema("observation.schema.json")) == [], o
    assert schema_check.check({"action": "list_records", "args": {"table": "ledger", "limit": 51}}, sch)
    assert schema_check.check({"action": "shell", "args": {}}, sch)
