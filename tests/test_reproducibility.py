"""Reproducibility tests: determinism, replay, truth-preserving transforms, committed artefacts,
save/load round trip (SPEC sections 3, 4, 10).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from functools import lru_cache

import pytest

from auditgym import rules, runner, validate
from auditgym.constants import INTERACTIONS, ISSUE_TYPES, MODE_EVAL, MODE_TRAIN
from auditgym.content_id import content_episode_id
from auditgym.env import AuditEnv
from auditgym.episode import (PRIVATE_FILES, PUBLIC_FILES, Episode, dumps, load_episode, load_public,
                              render_public, save_episode)
from auditgym.generator.build import generate_episode
from auditgym.generator.config import config_with_force
from auditgym.transforms import permute, rename_ids

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROFILES = {
    "clean": {"clean": True},
    "plain": {"issues": {t: 1 for t in ISSUE_TYPES}},
    "interactions": {"interactions": {i: 1 for i in INTERACTIONS}},
    "max12": {"issues": {t: 1 for t in ISSUE_TYPES}, "interactions": {i: 2 for i in INTERACTIONS}},
    "fx_only": {"issues": {"fx_error": 3}},
}


@lru_cache(maxsize=None)
def cfg(profile: str, split: str = "train"):
    return config_with_force(split, PROFILES[profile])


def ep_of(seed, profile="plain", split="train") -> Episode:
    return generate_episode(seed, cfg(profile, split))


def jd(x):
    return json.dumps(x, sort_keys=True)


# =================================================================== determinism of generation
@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_same_seed_and_config_identical_public_hash_truth_and_provenance(profile):
    for seed in (3, 17, 4242):
        a, b = ep_of(seed, profile), ep_of(seed, profile)
        assert a.public_hash() == b.public_hash()
        assert render_public(a) == render_public(b)
        assert jd(a.truth) == jd(b.truth)
        assert jd(a.provenance) == jd(b.provenance)
        assert a is not b


def test_config_given_as_name_dict_or_path_gives_identical_episode():
    from auditgym.generator.config import CONFIG_DIR, resolve_config
    for seed in (5, 6):
        by_name = generate_episode(seed, "train")
        by_dict = generate_episode(seed, resolve_config("train"))
        by_path = generate_episode(seed, os.path.join(CONFIG_DIR, "train.json"))
        with open(os.path.join(CONFIG_DIR, "train.json")) as f:
            by_raw = generate_episode(seed, json.load(f))
        for other in (by_dict, by_path, by_raw):
            assert other.public_hash() == by_name.public_hash()
            assert jd(other.truth) == jd(by_name.truth)


def test_generate_does_not_mutate_or_depend_on_input_config():
    c = copy.deepcopy(cfg("plain"))
    before = jd(c)
    generate_episode(9, c)
    assert jd(c) == before


def test_different_seeds_and_configs_give_different_episodes():
    hashes = {ep_of(s, "plain").public_hash() for s in range(1, 21)}
    assert len(hashes) == 20
    assert ep_of(7, "plain", "train").public_hash() != ep_of(7, "plain", "dev").public_hash()
    assert ep_of(7, "plain").public_hash() != ep_of(7, "clean").public_hash()


def test_episode_id_is_bound_to_the_resolved_config_including_force():
    # the id hashes the resolved config (force included): different force blocks never alias on one id
    ids = {ep_of(7, "plain").episode_id, ep_of(7, "clean").episode_id, generate_episode(7, "train").episode_id}
    assert len(ids) == 3
    assert ep_of(7, "plain").episode_id == ep_of(7, "plain").episode_id


def test_generation_independent_of_global_random_state_and_call_order():
    ref = ep_of(31, "max12")
    random.seed(12345)
    for _ in range(100):
        random.random()
    ep_of(32, "plain")                      # unrelated work in between
    ep_of(33, "clean")
    again = ep_of(31, "max12")
    assert again.public_hash() == ref.public_hash() and jd(again.truth) == jd(ref.truth)
    random.seed(99999)
    assert ep_of(31, "max12").public_hash() == ref.public_hash()


def test_generation_identical_across_processes_and_hash_randomisation():
    code = (
        "import json,hashlib,sys\n"
        "from auditgym.generator.build import generate_episode\n"
        "from auditgym.generator.config import config_with_force\n"
        "from auditgym.episode import dumps\n"
        "out=[]\n"
        "for s,f in ((3,{'clean':True}),(4,{'issues':{'fx_error':2,'duplicate_invoice':1},'interactions':{'fx_partial_payment_at_boundary':1}}),(5,None)):\n"
        "    c=config_with_force('train',f) if f is not None else 'train'\n"
        "    e=generate_episode(s,c)\n"
        "    out.append([e.public_hash(),hashlib.sha256(dumps(e.truth).encode()).hexdigest(),hashlib.sha256(dumps(e.provenance).encode()).hexdigest()])\n"
        "print(json.dumps(out))\n")
    results = []
    for hs in ("0", "1", "424242"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=ROOT)
        env.pop("PYTHONDONTWRITEBYTECODE", None)
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr[-2000:]
        results.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert results[0] == results[1] == results[2]
    here = [[generate_episode(3, cfg("clean")).public_hash()]]
    assert results[0][0][0] == here[0][0]


def test_public_render_is_stable_text_with_trailing_newlines_and_sorted_json():
    ep = ep_of(8, "plain")
    files = render_public(ep)
    assert set(files) == set(PUBLIC_FILES)
    for name, text in files.items():
        assert text.endswith("\n") and "\r" not in text
    for line in files["invoices.jsonl"].splitlines():
        assert line == dumps(json.loads(line))
    assert json.loads(files["manifest.json"]) == ep.manifest


# =================================================================== replay of action lists
def _recorded_actions(traj):
    return [t["action"] for t in traj[1:]]


def _rewards(traj):
    return [t["reward"] for t in traj]


@pytest.mark.parametrize("policy", ["reconcile", "random_explorer", "flag_all", "oracle"])
@pytest.mark.parametrize("mode", [MODE_TRAIN, MODE_EVAL])
def test_replaying_action_list_gives_identical_trajectory_and_rewards(policy, mode):
    for seed, profile in ((21, "plain"), (22, "clean"), (23, "interactions")):
        c = cfg(profile)
        first = runner.run_episode(policy, seed, c, mode=mode, remote=False)
        acts = _recorded_actions(first["trajectory"])
        again = runner.replay_actions(acts, seed, c, mode=mode, remote=False)
        assert runner.trajectory_hash(again["trajectory"]) == runner.trajectory_hash(first["trajectory"])
        assert _rewards(again["trajectory"]) == _rewards(first["trajectory"])
        assert again["grader_report"] == first["grader_report"]
        # and a fresh run of the same (deterministic) policy repeats too
        third = runner.run_episode(policy, seed, c, mode=mode, remote=False)
        assert runner.trajectory_hash(third["trajectory"]) == runner.trajectory_hash(first["trajectory"])
        assert third["grader_report"] == first["grader_report"]


def test_replay_across_the_remote_process_boundary_matches_in_process():
    c = cfg("plain")
    local = runner.run_episode("reconcile", 41, c, mode=MODE_EVAL, remote=False)
    remote = runner.run_episode("reconcile", 41, c, mode=MODE_EVAL, remote=True)
    assert runner.trajectory_hash(local["trajectory"]) == runner.trajectory_hash(remote["trajectory"])
    assert remote["grader_report"]["reward_shared"] == local["grader_report"]["reward_shared"]
    acts = _recorded_actions(remote["trajectory"])
    rep = runner.replay_actions(acts, 41, c, mode=MODE_EVAL, remote=True)
    assert runner.trajectory_hash(rep["trajectory"]) == runner.trajectory_hash(remote["trajectory"])


def test_replay_with_errors_and_budget_exhaustion_is_deterministic():
    c = cfg("plain")
    acts = [{"action": "list_records", "args": {"table": "ledger", "limit": 7}},
            {"action": "list_records", "args": {"table": "nope"}},
            {"action": "get_record", "args": {"table": "invoices", "id": "x" * 100}},
            {"action": "calculator", "args": {"expression": "round(1/3, 4) + 2**10"}},
            {"action": "calculator", "args": {"expression": "__import__('os')"}},
            {"action": "bogus", "args": {}},
            {"action": "list_records", "args": {"table": "ledger", "cursor": "garbage"}}] + \
           [{"action": "list_records", "args": {"table": "fx", "limit": 3}}] * 120
    a = runner.replay_actions(acts, 51, c, mode=MODE_TRAIN, remote=False)
    b = runner.replay_actions(acts, 51, c, mode=MODE_TRAIN, remote=False)
    assert runner.trajectory_hash(a["trajectory"]) == runner.trajectory_hash(b["trajectory"])
    assert a["trajectory"][-1]["truncated"] is True and len(a["trajectory"]) == 101
    assert a["grader_report"] == b["grader_report"]


def test_reset_fully_isolates_state_between_episodes_on_same_env():
    env = AuditEnv(mode=MODE_TRAIN)
    c = cfg("plain")
    acts = [{"action": "list_records", "args": {"table": "ledger", "limit": 5}},
            {"action": "list_records", "args": {"table": "invoices", "limit": 5}}]

    def run(seed):
        obs, _ = env.reset(seed=seed, config=c)
        out = [obs]
        for a in acts:
            out.append(env.step(a)[0])
        return dumps(out), dumps(env.trajectory())

    r1 = run(61)
    run(62)                                   # different episode in between
    r2 = run(61)
    assert r1 == r2
    fresh = AuditEnv(mode=MODE_TRAIN)
    obs, _ = fresh.reset(seed=61, config=c)
    assert dumps(obs) == dumps(json.loads(r1[0])[0])


def test_list_records_pages_and_cursors_are_deterministic_across_instances():
    c = cfg("plain")
    runs = []
    for _ in range(2):
        env = AuditEnv(mode=MODE_TRAIN)
        env.reset(seed=71, config=c)
        cur, seen = None, []
        while True:
            obs, *_ = env.step({"action": "list_records", "args": {"table": "ledger", "cursor": cur, "limit": 50}})
            seen.append(obs)
            cur = obs["result"]["next_cursor"]
            if cur is None:
                break
        runs.append(dumps(seen))
    assert runs[0] == runs[1]


# =================================================================== transforms: rename + permute
TRANSFORM_CASES = [(81, "plain"), (82, "interactions"), (83, "max12"), (84, "clean"), (85, "fx_only")]
OPTS = {"transform": {"rename_salt": 1234, "permute_salt": 777}}


def _content_multisets(ep):
    return {
        "ledger": Counter((r["timestamp"], r["account"], r["debit_minor"], r["credit_minor"], r["currency"])
                          for r in ep.ledger),
        "invoices": Counter((i["issued_at"], i["due_at"], i["currency"], i["total_minor"], jd(i["line_items"]))
                            for i in ep.invoices),
        "payments": Counter((p["timestamp"], p["currency"], p["amount_minor"],
                             tuple(r["amount_minor"] for r in (p["invoice_refs"] or [])) if p["invoice_refs"] is not None else None)
                            for p in ep.payments),
        "fx": Counter(jd(r) for r in ep.fx),
    }


@pytest.mark.parametrize("seed,profile", TRANSFORM_CASES)
def test_oracle_reward_is_one_under_rename_and_permute(seed, profile):
    c = cfg(profile)
    base = runner.run_episode("oracle", seed, c, mode=MODE_TRAIN, remote=False)
    for opts in (OPTS, {"transform": {"rename_salt": 5, "permute_salt": None}},
                 {"transform": {"rename_salt": None, "permute_salt": 6}},
                 {"transform": {"rename_salt": "abc", "permute_salt": 0}}):
        r = runner.run_episode("oracle", seed, c, mode=MODE_TRAIN, options=opts, remote=False)
        rep = r["grader_report"]
        assert r["trajectory"][-1]["reward"] == 1.0
        assert rep["reward_shared"] == 1.0 and rep["metrics"]["f1"] == 1.0
        assert rep["metrics"]["fp"] == 0 and rep["metrics"]["fn"] == 0
        assert rep["metrics"]["tp"] == base["grader_report"]["metrics"]["tp"] == rep["n_truth_issues"]
        assert rep["inspection_steps"] == 0
        # the transformed episode is a different episode id but identical score
        if opts["transform"]["rename_salt"] is not None:
            assert r["episode_id"] != base["episode_id"]


def test_oracle_reward_under_transform_in_eval_mode_and_remote():
    c = cfg("max12")
    r = runner.run_episode("oracle", 91, c, mode=MODE_EVAL, options=OPTS, remote=True)
    assert r["trajectory"][-1]["reward"] == 0.0 and r["grader_report"]["reward_shared"] == 1.0
    assert r["grader_report"]["metrics"]["tp"] == 12


def test_transformed_episode_is_validated_by_independent_rules_engine():
    for seed, profile in TRANSFORM_CASES:
        ep = ep_of(seed, profile)
        for t in (rename_ids(ep, 11), permute(ep, 12), permute(rename_ids(ep, 11), 12)):
            rep = validate.validate_episode(t)
            failed = {k: v["detail"] for k, v in rep["checks"].items() if not v["ok"]}
            # transformed episodes carry a recomputed content-addressed id, so every check (including
            # the id-from-content check) must pass
            assert not failed, (seed, profile, failed)


def test_rename_ids_maps_truth_consistently_and_preserves_content():
    for seed, profile in TRANSFORM_CASES:
        ep = ep_of(seed, profile)
        before = (render_public(ep), jd(ep.truth), jd(ep.provenance))
        rn = rename_ids(ep, 99)
        assert (render_public(ep), jd(ep.truth), jd(ep.provenance)) == before, "rename_ids mutated its input"
        assert rn.episode_id != ep.episode_id
        assert rn.truth["episode_id"] == rn.episode_id == rn.provenance["episode_id"]
        assert rn.episode_id == content_episode_id(rn)
        assert _content_multisets(rn) == _content_multisets(ep)
        for k in ("policy",):
            assert getattr(rn, k) == getattr(ep, k)
        assert {k: v for k, v in rn.manifest.items() if k != "episode_id"} == \
               {k: v for k, v in ep.manifest.items() if k != "episode_id"}
        # truth: same issues in the same order, ids remapped, nothing else changed
        assert len(rn.truth["issues"]) == len(ep.truth["issues"])
        old_inv = {i["invoice_id"]: i for i in ep.invoices}
        new_inv = {i["invoice_id"]: i for i in rn.invoices}
        old_pay = {p["payment_id"]: p for p in ep.payments}
        new_pay = {p["payment_id"]: p for p in rn.payments}
        old_rows = {r["row_id"]: r for r in ep.ledger}
        new_rows = {r["row_id"]: r for r in rn.ledger}
        old_docs = {d["doc_id"]: ("i", d) for d in ep.invoices} | {d["doc_id"]: ("p", d) for d in ep.payments}
        new_docs = {d["doc_id"]: ("i", d) for d in rn.invoices} | {d["doc_id"]: ("p", d) for d in rn.payments}
        mapping = {}
        for o, n in zip(ep.truth["issues"], rn.truth["issues"]):
            assert o["issue_id"] == n["issue_id"] and o["type"] == n["type"]
            assert o["expected_amount_minor"] == n["expected_amount_minor"]
            assert o["tolerance_minor"] == n["tolerance_minor"] and o["scenario"] == n["scenario"]
            assert o["currency"] == n["currency"]
            assert len(o["affected_ids"]) == len(n["affected_ids"])
            for oa, na in zip(o["affected_ids"], n["affected_ids"]):
                assert oa != na
                if oa in old_inv:
                    a, b = old_inv[oa], new_inv[na]
                    assert (a["issued_at"], a["total_minor"], a["currency"], jd(a["line_items"])) == \
                           (b["issued_at"], b["total_minor"], b["currency"], jd(b["line_items"]))
                else:
                    a, b = old_pay[oa], new_pay[na]
                    assert (a["timestamp"], a["amount_minor"], a["currency"]) == (b["timestamp"], b["amount_minor"], b["currency"])
                assert mapping.setdefault(oa, na) == na, "inconsistent mapping of an affected id"
            assert len(o["evidence_sets"]) == len(n["evidence_sets"])
            for so, sn in zip(o["evidence_sets"], n["evidence_sets"]):
                assert len(so) == len(sn)
                for eo, en in zip(so, sn):
                    if eo.startswith("fx:"):
                        assert eo == en
                        continue
                    assert eo != en
                    assert mapping.setdefault(eo, en) == en
                    if eo in old_rows:
                        ro, rn_ = old_rows[eo], new_rows[en]
                        assert (ro["timestamp"], ro["account"], ro["debit_minor"], ro["credit_minor"]) == \
                               (rn_["timestamp"], rn_["account"], rn_["debit_minor"], rn_["credit_minor"])
                        for f in ("invoice_id", "payment_id", "entity_id"):
                            if ro[f] is None:
                                assert rn_[f] is None
                            else:
                                assert mapping.setdefault(ro[f], rn_[f]) == rn_[f], f"row link {f} inconsistent"
                    else:
                        ko, do = old_docs[eo]
                        kn, dn = new_docs[en]
                        assert ko == kn
                        if ko == "i":
                            assert mapping.setdefault(do["invoice_id"], dn["invoice_id"]) == dn["invoice_id"]
                            assert mapping.setdefault(do["entity_id"], dn["entity_id"]) == dn["entity_id"]
                        else:
                            assert mapping.setdefault(do["payment_id"], dn["payment_id"]) == dn["payment_id"]
        # the mapping is injective and never maps an id to itself or onto an unrelated old id
        assert len(set(mapping.values())) == len(mapping)
        # every public id changed (full relabel) except fx ids
        old_pub, new_pub = ep.public_ids(), rn.public_ids()
        non_fx = lambda s: {x for x in s if not x.startswith("fx:")}
        assert not (non_fx(old_pub) & non_fx(new_pub))
        assert {x for x in old_pub if x.startswith("fx:")} == {x for x in new_pub if x.startswith("fx:")}


def test_rename_is_deterministic_and_salt_dependent_and_keeps_id_formats():
    import re
    ep = ep_of(81, "plain")
    a, b, c = rename_ids(ep, 5), rename_ids(ep, 5), rename_ids(ep, 6)
    assert a.public_hash() == b.public_hash() and jd(a.truth) == jd(b.truth)
    assert a.public_hash() != c.public_hash() and a.episode_id != c.episode_id
    assert all(re.fullmatch(r"R-[0-9a-f]{8}", r["row_id"]) for r in a.ledger)
    assert all(re.fullmatch(r"D-[0-9a-f]{10}", i["doc_id"]) for i in a.invoices + a.payments)
    # referential integrity
    inv = {i["invoice_id"] for i in a.invoices}
    pay = {p["payment_id"] for p in a.payments}
    for r in a.ledger:
        assert r["invoice_id"] is None or r["invoice_id"] in inv
        assert r["payment_id"] is None or r["payment_id"] in pay
    for p in a.payments:
        for ref in p["invoice_refs"] or []:
            assert ref["invoice_id"] in inv
    # entity partitioning preserved
    assert sorted(Counter(i["entity_id"] for i in a.invoices).values()) == \
           sorted(Counter(i["entity_id"] for i in ep.invoices).values())
    # unique ids
    assert len({r["row_id"] for r in a.ledger}) == len(a.ledger)


def test_permute_changes_order_only_and_leaves_truth_untouched():
    for seed, profile in TRANSFORM_CASES:
        ep = ep_of(seed, profile)
        before = render_public(ep)
        pm = permute(ep, 3)
        assert render_public(ep) == before, "permute mutated its input"
        # record order is public content, so the content-addressed id is recomputed
        assert pm.episode_id != ep.episode_id and pm.episode_id == content_episode_id(pm)
        assert pm.truth["episode_id"] == pm.provenance["episode_id"] == pm.episode_id
        assert {k: v for k, v in pm.truth.items() if k != "episode_id"} == \
               {k: v for k, v in ep.truth.items() if k != "episode_id"}
        assert {k: v for k, v in pm.manifest.items() if k != "episode_id"} == \
               {k: v for k, v in ep.manifest.items() if k != "episode_id"} and pm.policy == ep.policy
        for name in ("ledger", "invoices", "payments", "fx"):
            assert sorted(jd(r) for r in getattr(pm, name)) == sorted(jd(r) for r in getattr(ep, name))
        assert [r["row_id"] for r in pm.ledger] != [r["row_id"] for r in ep.ledger]
        assert pm.public_hash() != ep.public_hash()
        again = permute(ep, 3)
        assert again.public_hash() == pm.public_hash()
        assert permute(ep, 4).public_hash() != pm.public_hash()


def test_grader_truth_with_transform_equals_transform_of_generated_truth():
    for seed, profile in TRANSFORM_CASES:
        c = cfg(profile)
        ep = generate_episode(seed, c)
        expect = permute(rename_ids(ep, OPTS["transform"]["rename_salt"]), OPTS["transform"]["permute_salt"]).truth
        assert jd(runner.grader_truth(seed, c, OPTS)) == jd(expect)


def test_env_observation_ids_come_from_transformed_episode_and_no_old_ids_leak():
    c = cfg("plain")
    ep = generate_episode(81, c)
    env = AuditEnv(mode=MODE_TRAIN)
    obs, _ = env.reset(seed=81, config=c, options=OPTS)
    assert obs["episode_id"] != ep.episode_id
    seen = {}
    obs1, *_ = env.step({"action": "list_records", "args": {"table": "invoices", "limit": 50}})
    old_inv_ids = {i["invoice_id"] for i in ep.invoices}
    for row in obs1["result"]["rows"]:
        assert row["invoice_id"] not in old_inv_ids
    text = dumps(obs1)
    assert not any(o in text for o in list(old_inv_ids)[:30])


def test_paired_original_and_transformed_runs_give_equal_scores_for_baselines():
    """A transform must not change what a deterministic content-based policy achieves."""
    c = cfg("plain")
    for pol in ("reconcile",):
        a = runner.run_episode(pol, 86, c, mode=MODE_TRAIN, remote=False)
        b = runner.run_episode(pol, 86, c, mode=MODE_TRAIN, options={"transform": {"rename_salt": 3, "permute_salt": None}},
                               remote=False)
        ma, mb = a["grader_report"]["metrics"], b["grader_report"]["metrics"]
        assert (ma["tp"], ma["fp"], ma["fn"]) == (mb["tp"], mb["fp"], mb["fn"])


# =================================================================== committed episodes vs regeneration
def _split_entries():
    return runner.load_splits(ROOT)


def _committed_dir(kind, split):
    return os.path.join(ROOT, kind, split)


def test_splits_json_structure_and_disjoint_seeds():
    sp = _split_entries()
    assert set(sp) == {"train", "dev", "holdout"}
    assert len(sp["train"]) >= 10 and len(sp["dev"]) >= 10
    assert len(sp["holdout"]) == 20
    seeds = [e["seed"] for s in sp.values() for e in s]
    assert len(seeds) == len(set(seeds)), "seeds shared between/within splits"
    for split, entries in sp.items():
        assert all(e["config"] == split for e in entries)
    cats = Counter(e.get("category") for e in sp["holdout"])
    assert cats == Counter({"clean": 5, "interaction": 5, "mixed": 10})


@pytest.mark.parametrize("split", ["train", "dev"])
def test_committed_episodes_match_regeneration_from_splits(split):
    pub_root, priv_root = _committed_dir("episodes", split), _committed_dir("grader_data", split)
    assert os.path.isdir(pub_root), "episodes/ not committed (run: python -m auditgym generate)"
    assert os.path.isdir(priv_root)
    entries = _split_entries()[split]
    expected_ids = set()
    for e in entries:
        ep = generate_episode(e["seed"], runner.resolve_config(e))
        expected_ids.add(ep.episode_id)
        d = os.path.join(pub_root, ep.episode_id)
        assert os.path.isdir(d), f"committed episode missing for seed {e['seed']}"
        assert sorted(os.listdir(d)) == sorted(PUBLIC_FILES)
        for name, text in render_public(ep).items():
            with open(os.path.join(d, name), encoding="utf-8", newline="") as f:
                assert f.read() == text, f"{split}/{ep.episode_id}/{name} differs from regeneration"
        pd = os.path.join(priv_root, ep.episode_id)
        assert sorted(os.listdir(pd)) == sorted(PRIVATE_FILES)
        with open(os.path.join(pd, "truth.json"), encoding="utf-8") as f:
            assert json.load(f) == json.loads(json.dumps(ep.truth))
        with open(os.path.join(pd, "provenance.json"), encoding="utf-8") as f:
            assert json.load(f) == json.loads(json.dumps(ep.provenance))
        loaded = load_episode(d, pd)
        assert loaded.public_hash() == ep.public_hash()
    assert set(os.listdir(pub_root)) == expected_ids, "stray or missing committed episodes"
    assert set(os.listdir(priv_root)) == expected_ids


def test_committed_public_dirs_contain_no_private_files_and_holdout_not_committed():
    for split in ("train", "dev"):
        for eid in os.listdir(_committed_dir("episodes", split)):
            assert not (set(os.listdir(os.path.join(ROOT, "episodes", split, eid))) & set(PRIVATE_FILES))
    sp = _split_entries()
    holdout_ids = {generate_episode(e["seed"], runner.resolve_config(e)).episode_id for e in sp["holdout"]}
    for kind in ("episodes", "grader_data"):
        for split in ("train", "dev", "holdout"):
            d = _committed_dir(kind, split)
            if os.path.isdir(d):
                assert not (set(os.listdir(d)) & holdout_ids), f"holdout episode committed under {kind}/{split}"
    assert not os.path.exists(_committed_dir("episodes", "holdout"))


def test_committed_episodes_validate():
    for split in ("train", "dev"):
        rep = validate.validate_dir(_committed_dir("episodes", split), _committed_dir("grader_data", split))
        assert rep["ok"], rep if not isinstance(rep, dict) else {k: v for k, v in rep.items() if k != "ok"}


def test_holdout_entries_regenerate_deterministically_and_match_their_force():
    sp = _split_entries()["holdout"]
    for e in sp:
        c = runner.resolve_config(e)
        a, b = generate_episode(e["seed"], c), generate_episode(e["seed"], c)
        assert a.public_hash() == b.public_hash() and jd(a.truth) == jd(b.truth)
        assert validate.validate_episode(a)["ok"], e
        force, cat = e["force"], e["category"]
        n = len(a.truth["issues"])
        if cat == "clean":
            assert n == 0
        else:
            want = sum((force.get("issues") or {}).values()) + 2 * sum((force.get("interactions") or {}).values())
            assert n == want, (e, n, want)
            scen = [i for i in a.truth["issues"] if i["scenario"]]
            if cat == "interaction":
                assert len(scen) == 2 * sum(force["interactions"].values())
            else:
                assert not scen
    assert sum(1 for e in sp if e["category"] == "clean") == 5


def test_committed_trajectories_file_replays_for_a_sample_of_records():
    path = os.path.join(ROOT, "trajectories", "trajectories.jsonl")
    if not os.path.exists(path):
        pytest.skip("no committed trajectories")
    by_ep = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_ep.setdefault((r["episode_id"], r["policy"]), []).append(r)
    assert by_ep
    sp = _split_entries()
    id_to_entry = {}
    for split in ("train", "dev"):
        for e in sp[split]:
            id_to_entry[generate_episode(e["seed"], runner.resolve_config(e)).episode_id] = e
    checked = 0
    for (eid, pol), recs in sorted(by_ep.items())[:6]:
        if eid not in id_to_entry:
            continue
        e = id_to_entry[eid]
        acts = [r["action"] for r in sorted(recs, key=lambda r: r["step"]) if r["step"] > 0]
        rep = runner.replay_actions(acts, e["seed"], runner.resolve_config(e), mode=MODE_TRAIN, remote=False)
        got = [(t["step"], t["reward"], dumps(t["observation"])) for t in rep["trajectory"]]
        want = [(r["step"], r["reward"], dumps(r["observation"])) for r in sorted(recs, key=lambda r: r["step"])]
        assert got == want, (eid, pol)
        checked += 1
    assert checked > 0


# =================================================================== save / load round trip
@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_save_load_round_trip(tmp_path, profile):
    for seed in (101, 102):
        ep = ep_of(seed, profile)
        pub_root, priv_root = str(tmp_path / profile / "pub"), str(tmp_path / profile / "priv")
        pub, priv = save_episode(ep, pub_root, priv_root)
        assert os.path.basename(pub) == os.path.basename(priv) == ep.episode_id
        assert sorted(os.listdir(pub)) == sorted(PUBLIC_FILES)
        assert sorted(os.listdir(priv)) == sorted(PRIVATE_FILES)
        loaded = load_episode(pub, priv)
        assert loaded.public_hash() == ep.public_hash()
        assert loaded.manifest == ep.manifest and loaded.policy == ep.policy
        assert loaded.ledger == ep.ledger and loaded.invoices == ep.invoices
        assert loaded.payments == ep.payments and loaded.fx == ep.fx
        assert jd(loaded.truth) == jd(ep.truth) and jd(loaded.provenance) == jd(ep.provenance)
        assert validate.validate_episode(loaded)["ok"]
        # types survive: ints stay ints, absent links are None, fx rates stay strings with 6 places
        for r in loaded.ledger:
            assert isinstance(r["debit_minor"], int) and isinstance(r["credit_minor"], int)
            assert all(r[f] is None or r[f] != "" for f in ("entity_id", "invoice_id", "payment_id"))
        assert all(isinstance(r["rate"], str) and len(r["rate"].split(".")[1]) == 6 for r in loaded.fx)
        # idempotent re-save is byte-identical
        pub2, priv2 = save_episode(loaded, str(tmp_path / profile / "pub2"), str(tmp_path / profile / "priv2"))
        for name in PUBLIC_FILES:
            assert open(os.path.join(pub, name), encoding="utf-8", newline="").read() == \
                   open(os.path.join(pub2, name), encoding="utf-8", newline="").read()
        for name in PRIVATE_FILES:
            assert open(os.path.join(priv, name), encoding="utf-8").read() == \
                   open(os.path.join(priv2, name), encoding="utf-8").read()


def test_public_load_needs_no_private_files_and_public_dir_is_leak_free(tmp_path):
    ep = ep_of(103, "max12")
    pub, priv = save_episode(ep, str(tmp_path / "pub"), str(tmp_path / "priv"))
    pub_ep = load_public(pub)
    assert pub_ep.truth == {} and pub_ep.provenance == {}
    assert pub_ep.public_hash() == ep.public_hash()
    blob = "".join(open(os.path.join(pub, n), encoding="utf-8").read() for n in os.listdir(pub)).lower()
    for tok in ("iss_", "expected_amount", "evidence_sets", "provenance", "truth", "scenario", "root_cause"):
        assert tok not in blob
    assert not os.path.commonpath([pub, priv]) == pub
    # private tokens live only in the private dir
    assert "iss_01" in open(os.path.join(priv, "truth.json"), encoding="utf-8").read()


def test_env_and_policies_behave_identically_on_loaded_episode_dir(tmp_path):
    ep = ep_of(104, "plain")
    pub, priv = save_episode(ep, str(tmp_path / "pub"), str(tmp_path / "priv"))
    disk_cfg = {"episode_dir": pub, "grader_dir": priv}
    live = runner.run_episode("reconcile", 104, cfg("plain"), mode=MODE_TRAIN, remote=False)
    disk = runner.run_episode("reconcile", 104, disk_cfg, mode=MODE_TRAIN, remote=False)
    assert runner.trajectory_hash(live["trajectory"]) == runner.trajectory_hash(disk["trajectory"])
    assert disk["grader_report"]["reward_shared"] == live["grader_report"]["reward_shared"]
    orc = runner.run_episode("oracle", 104, disk_cfg, mode=MODE_TRAIN, remote=False)
    assert orc["grader_report"]["reward_shared"] == 1.0
    assert orc["grader_report"]["metrics"]["tp"] == len(ep.truth["issues"])
    assert orc["grader_report"]["metrics"]["f1"] == 1.0
    orc_t = runner.run_episode("oracle", 104, disk_cfg, mode=MODE_TRAIN, options=OPTS, remote=False)
    assert orc_t["grader_report"]["reward_shared"] == 1.0


def test_json_round_trip_of_truth_and_provenance_is_lossless():
    for seed, profile in TRANSFORM_CASES:
        ep = ep_of(seed, profile)
        assert json.loads(json.dumps(ep.truth)) == ep.truth
        assert json.loads(json.dumps(ep.provenance)) == ep.provenance
        assert json.loads(json.dumps(ep.manifest)) == ep.manifest
        assert json.loads(json.dumps(ep.policy)) == ep.policy


def test_public_copy_has_no_private_data_and_same_public_hash():
    ep = ep_of(105, "max12")
    pc = ep.public_copy()
    assert pc.truth == {} and pc.provenance == {}
    assert pc.public_hash() == ep.public_hash()
    assert ep.truth["issues"], "public_copy must not strip the source episode"


# =================================================================== cluster 5: runner / CLI / check polish
import re  # noqa: E402

from auditgym import __main__ as cli  # noqa: E402
from auditgym import check as check_mod  # noqa: E402
from auditgym import schema_check  # noqa: E402
from auditgym.episode import render_public  # noqa: E402
from auditgym.policies import EXTRA_POLICY_NAMES, POLICY_NAMES, get_policy  # noqa: E402
from auditgym.transforms import aggressive_salt  # noqa: E402

TRAJ_DIR = os.path.join(ROOT, "trajectories")


def _rows(name):
    with open(os.path.join(TRAJ_DIR, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- issue 1: CLI
def test_demo_defaults_to_careful_auditor_then_contrasts_with_reconcile(capsys):
    assert cli.build_parser().parse_args(["demo"]).policy == "careful_auditor"
    assert cli.main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "Policy: careful_auditor" in out
    first = out.index("Policy: careful_auditor")
    assert "Contrast: reconcile" in out[first:]
    # the headline run ends well (positive reward), the contrast is the failing naive baseline
    m = re.search(r"ended: terminated=True truncated=False reward=([+-]\d\.\d+)", out)
    assert m and float(m.group(1)) > 0.5
    contrast = out[out.index("Contrast: reconcile"):]
    assert re.search(r"reconcile\s+TP=\d+ FP=\d+ FN=\d+.*shared reward=-", contrast)


def test_demo_accepts_seed_and_config_and_can_skip_contrast(capsys):
    assert cli.main(["demo", "--seed", "5303", "--config", "dev", "--policy", "reconcile", "--contrast", "none"]) == 0
    out = capsys.readouterr().out
    assert "Policy: reconcile" in out and "Contrast:" not in out
    ep = generate_episode(5303, "dev")
    assert ep.episode_id in out
    # a config given as a path to a JSON file works the same
    assert cli.main(["demo", "--seed", "5303", "--config", os.path.join(ROOT, "configs", "dev.json"),
                     "--contrast", "none"]) == 0
    assert ep.episode_id in capsys.readouterr().out


def test_run_pretty_reuses_the_demo_formatter_and_plain_run_stays_json(capsys):
    assert cli.main(["run", "--policy", "reconcile", "--seed", "5303", "--config", "dev", "--local", "--pretty"]) == 0
    out = capsys.readouterr().out
    assert "Submitted findings (grader-side view with match reasons):" in out and "Truth issues (grader-side):" in out
    assert "mode: eval" in out and "eval mode hides it; grader reward_shared" in out
    assert cli.main(["run", "--policy", "reconcile", "--seed", "5303", "--config", "dev", "--local"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["terminal_reward_seen_by_policy"] == 0.0 and data["grader_report"]["reward_shared"] < 1.0
    assert "reward_shared" in data["note"]


def test_run_deterministic_flag_and_aggressive_rename_via_cli(capsys):
    args = ["run", "--policy", "careful_auditor", "--seed", "5303", "--config", "dev", "--local"]
    assert cli.main(args) == 0
    base = json.loads(capsys.readouterr().out)["grader_report"]
    assert cli.main(args + ["--deterministic", "--rename-salt", "3", "--rename-mode", "aggressive",
                            "--permute-salt", "4"]) == 0
    other = json.loads(capsys.readouterr().out)["grader_report"]
    assert other["deterministic"] is True and base["deterministic"] is False
    assert other["metrics"] == base["metrics"] and other["reward_shared"] == base["reward_shared"]


# ---------------------------------------------------------------- issue 2: check must not write into the tree
def test_check_pytest_step_runs_with_no_cache_and_no_bytecode(monkeypatch):
    env = check_mod.subprocess_env()
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and env["AUDITGYM_CHECK_RUNNING"] == "1"
    assert "-p" in check_mod.PYTEST_ARGS and "no:cacheprovider" in check_mod.PYTEST_ARGS
    seen = {}

    def fake_run(args, **kw):
        seen["args"], seen["env"], seen["cwd"] = args, kw["env"], kw["cwd"]
        return subprocess.CompletedProcess(args, 0, "1 passed in 0.1s\n", "")
    monkeypatch.delenv("AUDITGYM_CHECK_RUNNING", raising=False)
    monkeypatch.setattr(check_mod.subprocess, "run", fake_run)
    rep = check_mod.Report()
    check_mod.step_pytest(rep)
    assert rep.checks[-1]["status"] == "PASS"
    assert "no:cacheprovider" in seen["args"] and seen["env"]["PYTHONDONTWRITEBYTECODE"] == "1"


def test_check_subprocess_really_writes_no_bytecode(tmp_path):
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "mod.py").write_text("X = 1\n")
    code = f"import sys; sys.path.insert(0, {str(probe)!r}); import mod; print(sys.dont_write_bytecode)"
    p = subprocess.run([sys.executable, "-c", code], env=check_mod.subprocess_env(), capture_output=True, text=True)
    assert p.stdout.strip() == "True" and not (probe / "__pycache__").exists()


# ---------------------------------------------------------------- issue 3: version stamp in reports
def test_grader_report_carries_env_and_config_version_and_hash_local_and_remote():
    for remote in (False, True):
        r = runner.run_episode("careful_auditor", 5303, "dev", mode=MODE_EVAL, remote=remote)
        rep = r["grader_report"]
        assert rep["env_version"] == "auditgym-1.0.0" and rep["config_id"] == "dev"
        assert re.fullmatch(r"\d+\.\d+\.\d+", rep["config_version"])
        assert re.fullmatch(r"[0-9a-f]{64}", rep["config_hash"])
        assert rep["time_truncated"] is False and rep["deterministic"] is False
        man = r["trajectory"][0]["observation"]["manifest"]
        assert "config_id" not in man and "config_version" not in man and "config_hash" not in man
    # the hash is that of the RESOLVED config incl. force, so it separates pinned holdout episodes
    h_plain = runner.config_hash("holdout")
    h_forced = runner.config_hash(runner.resolve_config(runner.load_splits()["holdout"][0]))
    assert h_plain != h_forced and runner.config_hash("dev") != runner.config_hash("train")
    assert runner.config_hash("dev") == runner.config_hash(os.path.join(ROOT, "configs", "dev.json"))
    assert runner.config_hash({"episode_dir": "x"}) is None


def test_eval_mode_hides_trajectory_reward_so_paired_checks_must_use_the_grader_report():
    r = runner.run_episode("careful_auditor", 5303, "dev", mode=MODE_EVAL, remote=False)
    assert r["trajectory"][-1]["reward"] == 0.0
    assert r["grader_report"]["reward_shared"] > 0.9
    h = runner.trajectory_header(r)
    assert h["reward_visible_to_policy"] is False and h["grader_reward_shared"] == r["grader_report"]["reward_shared"]
    t = runner.run_episode("careful_auditor", 5303, "dev", mode=MODE_TRAIN, remote=False)
    assert t["trajectory"][-1]["reward"] == t["grader_report"]["reward_shared"]
    assert runner.trajectory_header(t)["reward_visible_to_policy"] is True
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    assert "reward_shared" in readme and "hidden" in readme.lower() and "0.0" in readme


def test_check_report_json_and_md_carry_env_and_config_versions(tmp_path):
    rep = check_mod.Report()
    rep.out_dir = str(tmp_path)
    check_mod.step_config_lock(rep)
    jp, mp = check_mod.write_reports(rep, str(tmp_path), True, 1.0)
    data = json.load(open(jp, encoding="utf-8"))
    assert data["env_version"] == "auditgym-1.0.0"
    assert set(data["config_versions"]) >= {"train", "dev", "holdout"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in data["config_hashes"].values())
    assert data["config_ids"]["dev"] == "dev" and data["platform"]
    md = open(mp, encoding="utf-8").read()
    assert "dev@" in md and "hash `" in md


# ---------------------------------------------------------------- issue 4: aggressive rename, commutation
AGGR_RE = re.compile(r"\d{9,14}|z[0-9a-f]{8,20}")


def _opaque_ids(ep):
    ids = [r["row_id"] for r in ep.ledger] + [i["invoice_id"] for i in ep.invoices] + \
          [i["doc_id"] for i in ep.invoices + ep.payments] + [p["payment_id"] for p in ep.payments] + \
          [i["entity_id"] for i in ep.invoices]
    return ids


@pytest.mark.parametrize("seed,profile", TRANSFORM_CASES)
def test_aggressive_rename_changes_prefix_length_charset_and_sort_order(seed, profile):
    ep = ep_of(seed, profile)
    a = rename_ids(ep, aggressive_salt(7))
    assert a.public_hash() == rename_ids(ep, aggressive_salt(7)).public_hash()
    assert a.public_hash() != rename_ids(ep, aggressive_salt(8)).public_hash()
    new_ids, old_ids = _opaque_ids(a), _opaque_ids(ep)
    assert all(AGGR_RE.fullmatch(x) for x in new_ids)
    assert not any(re.match(r"^(R-|D-|INV|SI/|F-|CUST|PAY|RCPT|BNK|VND)", x) for x in new_ids)
    assert any(x.isdigit() for x in new_ids) and any(x.startswith("z") for x in new_ids)
    assert {len(x) for x in new_ids} != {len(x) for x in old_ids}
    assert not (set(new_ids) & set(old_ids))
    # sort order: ids no longer sort like the originals (old ledger ids sort == record order is not preserved)
    old_order = [r["row_id"] for r in ep.ledger]
    new_order = [r["row_id"] for r in a.ledger]
    assert sorted(range(len(old_order)), key=old_order.__getitem__) != \
           sorted(range(len(new_order)), key=new_order.__getitem__)
    # fx keys are natural keys and intentionally kept; truth cites the same fx ids
    assert a.fx == ep.fx
    fx_old = {e for i in ep.truth["issues"] for s in i["evidence_sets"] for e in s if e.startswith("fx:")}
    fx_new = {e for i in a.truth["issues"] for s in i["evidence_sets"] for e in s if e.startswith("fx:")}
    assert fx_old == fx_new
    # content preserved and the independent rules engine still validates it
    assert _content_multisets(a) == _content_multisets(ep)
    assert validate.validate_episode(a)["ok"]
    assert len(a.truth["issues"]) == len(ep.truth["issues"])


def test_rename_is_a_pure_relabel_and_commutes_with_permute():
    for seed, profile in TRANSFORM_CASES:
        ep = ep_of(seed, profile)
        for salt in (5, "abc", aggressive_salt(5)):
            r = rename_ids(ep, salt)
            # record order untouched: same position <-> same record content
            assert [x["timestamp"] for x in r.ledger] == [x["timestamp"] for x in ep.ledger]
            assert [x["issued_at"] for x in r.invoices] == [x["issued_at"] for x in ep.invoices]
            x = permute(rename_ids(ep, salt), 9)
            y = rename_ids(permute(ep, 9), salt)
            assert render_public(x) == render_public(y) and x.episode_id == y.episode_id
            assert jd(x.truth) == jd(y.truth)


def test_unknown_rename_mode_rejected_and_shape_default_unchanged():
    ep = ep_of(81, "plain")
    with pytest.raises(ValueError):
        rename_ids(ep, 1, mode="nope")
    assert rename_ids(ep, 1).public_hash() == rename_ids(ep, 1, mode="shape").public_hash()
    assert "fx:" in (permute.__doc__ + rename_ids.__doc__ + __import__("auditgym.transforms").transforms.__doc__)


@pytest.mark.parametrize("policy", ["oracle", "reconcile", "careful_auditor"])
def test_policies_are_invariant_under_aggressive_rename_and_permute(policy):
    for seed, split in ((5303, "dev"), (5333, "dev"), (4117, "train")):
        base = runner.run_episode(policy, seed, split, mode=MODE_EVAL, remote=False)
        t = runner.run_episode(policy, seed, split, mode=MODE_EVAL, remote=False,
                               options={"transform": {"rename_salt": aggressive_salt(11), "permute_salt": 12}})
        b, c = base["grader_report"], t["grader_report"]
        assert (b["metrics"], b["reward_shared"], b["inspection_steps"]) == (c["metrics"], c["reward_shared"], c["inspection_steps"])
        assert t["episode_id"] != base["episode_id"]


def test_aggressive_rename_works_through_the_subprocess_grader():
    r = runner.run_episode("oracle", 5303, "dev", mode=MODE_EVAL, remote=True,
                           options={"transform": {"rename_salt": aggressive_salt(2), "permute_salt": 3}})
    assert r["grader_report"]["reward_shared"] == 1.0
    cited = {x for f in runner.submitted_findings(r["trajectory"])["findings"] for x in f["affected_ids"]}
    assert cited and all(AGGR_RE.fullmatch(x) for x in cited)


# ---------------------------------------------------------------- issue 5: oracle vs committed truth, per-pair F1
def test_oracle_truth_is_cross_checked_against_committed_truth_json():
    rep = check_mod.Report()
    check_mod.step_oracle_truth(rep, runner.load_splits(), remote=False, quick=True)
    assert [c["status"] for c in rep.checks] == ["PASS"] * 3, rep.checks


def test_oracle_cross_check_fails_when_the_committed_truth_disagrees(monkeypatch):
    real = runner.committed_truth

    def tampered(split, eid, root=None):
        t = real(split, eid, root)
        if t and t["issues"]:
            t = json.loads(json.dumps(t))
            t["issues"][0]["expected_amount_minor"] += 1000
        return t
    monkeypatch.setattr(check_mod, "committed_truth", tampered)
    rep = check_mod.Report()
    check_mod.step_oracle_truth(rep, runner.load_splits(), remote=False, quick=True)
    assert any(c["status"] == "FAIL" for c in rep.checks)


def test_paired_step_reports_per_pair_f1_and_is_discriminating():
    splits = runner.load_splits()
    hold = splits["holdout"]
    entries = [hold[0], hold[6], hold[12]]               # a clean, an interaction and a mixed episode
    cache: dict = {}
    eps = [(e, check_mod._gen(e, cache), check_mod.classify(check_mod._gen(e, cache))) for e in entries]
    runs = {p: [runner.run_episode(p, e["seed"], runner.resolve_config(e), mode=MODE_EVAL, remote=False)
                for e, _ep, _k in eps] for p in ("oracle", "reconcile", "careful_auditor")}
    rep = check_mod.Report()
    check_mod.step_paired(rep, eps, runs, False, 3)
    bad = [c for c in rep.checks if c["status"] == "FAIL"]
    assert not bad, bad
    names = " | ".join(c["name"] for c in rep.checks)
    assert "AGGRESSIVE rename" in names and "aggressive rename+permute: careful_auditor" in names
    sec = rep.sections["paired"]
    assert set(sec["per_pair_f1"]) >= {"reconcile", "careful_auditor"}
    assert all("f1" in d for d in sec["per_pair_f1"]["careful_auditor"])
    assert {d["variant"] for d in sec["detail"]} == {"shape", "aggressive"}
    assert "random_explorer" not in sec["per_pair_f1"] and "floor" in sec["note"]
    lines = "\n".join(check_mod.results_lines(rep))
    assert "Paired invariance pairs" in lines and "| careful_auditor | 3/3 |" in lines


# ---------------------------------------------------------------- issue 6: degraded careful auditor
def test_degraded_careful_auditor_is_registered_and_a_measurable_near_miss():
    assert "careful_auditor_degraded" in EXTRA_POLICY_NAMES and "careful_auditor_degraded" not in POLICY_NAMES
    f1 = {"careful_auditor": [], "careful_auditor_degraded": []}
    shared = {k: [] for k in f1}
    for e in runner.load_splits()["dev"][:6]:
        for p in f1:
            rep = runner.run_episode(p, e["seed"], e["config"], mode=MODE_EVAL, remote=False)["grader_report"]
            f1[p].append(rep["metrics"]["f1"])
            shared[p].append(rep["reward_shared"])
    mean = lambda x: sum(x) / len(x)
    assert mean(f1["careful_auditor"]) == 1.0
    assert 0.3 < mean(f1["careful_auditor_degraded"]) < 0.95
    assert mean(shared["careful_auditor_degraded"]) < mean(shared["careful_auditor"])


def test_degraded_variant_with_both_handicaps_off_equals_careful_auditor():
    from auditgym.policies.careful_auditor_degraded import CarefulAuditorDegraded
    e = runner.load_splits()["dev"][1]
    a = runner.run_episode("careful_auditor", e["seed"], e["config"], mode=MODE_TRAIN, remote=False)
    orig = runner.get_policy
    try:
        runner.get_policy = lambda n, s=0, t=None: CarefulAuditorDegraded(s, step_cap=None, ignore_tz=False)
        b = runner.run_episode("x", e["seed"], e["config"], mode=MODE_TRAIN, remote=False)
    finally:
        runner.get_policy = orig
    assert runner.submitted_findings(a["trajectory"]) == runner.submitted_findings(b["trajectory"])
    with pytest.raises(ValueError):
        CarefulAuditorDegraded(0, ignore_tz="sometimes")


def test_degraded_step_cap_is_respected_and_still_submits():
    r = runner.run_episode("careful_auditor_degraded", 5311, "dev", mode=MODE_TRAIN, remote=False)
    assert r["steps"] <= 21 and runner.submitted_findings(r["trajectory"]) is not None


def test_failure_analysis_has_per_type_misses_and_step_cost():
    cache: dict = {}
    entries = runner.load_splits()["holdout"][:3]
    eps = [(e, check_mod._gen(e, cache), "x") for e in entries]
    runs = {p: [runner.run_episode(p, e["seed"], runner.resolve_config(e), mode=MODE_EVAL, remote=False)
                for e, _ep, _k in eps] for p in ("careful_auditor", "careful_auditor_degraded")}
    fa = check_mod.failure_analysis(eps, runs)
    for pol, d in fa.items():
        assert set(d["misses_by_type"]) == set(ISSUE_TYPES) == set(d["fp_by_type"])
        assert d["mean_step_cost"] == pytest.approx(d["mean_inspection_steps"] * 0.002)
        assert d["n_runs"] == 3
        for t in ISSUE_TYPES:
            assert d["misses_by_type"][t] == d["recall_by_type"][t]["truth"] - d["recall_by_type"][t]["tp"]
    rep = check_mod.Report()
    rep.sections["failure_analysis"] = fa
    assert any("Near-miss analysis" in line for line in check_mod.results_lines(rep))


# ---------------------------------------------------------------- issue 7: golden digests, time-truncation flag
def test_golden_hashes_match_this_machine():
    path = os.path.join(ROOT, "tests", "golden_hashes.json")
    want = json.load(open(path, encoding="utf-8"))
    got = runner.golden_digests()
    assert want["env_version"] == got["env_version"]
    assert len(want["episodes"]) == len(got["episodes"]) >= 6
    for w, g in zip(want["episodes"], got["episodes"]):
        assert w == g, f"golden digest changed for {g['split']}:{g['seed']}; if intended run `python -m auditgym golden --write`"
    kinds = {e["split"] for e in got["episodes"]}
    assert kinds == {"train", "dev", "holdout"}
    assert all(set(e["trajectory_sha256"]) == {"reconcile", "careful_auditor"} for e in got["episodes"])


def test_golden_digests_change_if_the_output_changes(monkeypatch):
    from auditgym.generator import build
    real = build.generate_episode
    first = runner.golden_digests()["episodes"][0]["public_hash"]
    monkeypatch.setattr(build, "generate_episode", lambda seed, cfg: real(seed + 1, cfg))
    assert runner.golden_digests()["episodes"][0]["public_hash"] != first


def test_deterministic_mode_freezes_the_clock_and_matches_a_normal_run():
    a = runner.run_episode("careful_auditor", 5303, "dev", mode=MODE_TRAIN, remote=False)
    b = runner.run_episode("careful_auditor", 5303, "dev", mode=MODE_TRAIN, remote=False, deterministic=True)
    assert runner.trajectory_hash(a["trajectory"]) == runner.trajectory_hash(b["trajectory"])
    assert b["grader_report"]["deterministic"] is True and b["meta"]["time_truncated"] is False
    with pytest.raises(ValueError):
        runner.run_episode("careful_auditor", 5303, "dev", remote=True, deterministic=True)


def test_time_truncation_is_flagged_in_the_report_and_header_but_step_truncation_is_not():
    t = runner.run_truncated_sample(5303, "dev", by="time")
    s = runner.run_truncated_sample(5303, "dev", by="steps")
    for r in (t, s):
        assert r["trajectory"][-1]["truncated"] is True and not r["trajectory"][-1]["terminated"]
        assert all(rec["action"]["action"] != "submit" for rec in r["trajectory"][1:])
    assert t["grader_report"]["time_truncated"] is True and t["grader_report"]["ended_by"] == "time"
    assert s["grader_report"]["time_truncated"] is False and s["grader_report"]["ended_by"] == "steps"
    assert runner.trajectory_header(t)["time_truncated"] is True
    assert runner.trajectory_header(s)["truncated"] is True and runner.trajectory_header(s)["steps"] == 100
    assert t["steps"] == 2
    with pytest.raises(ValueError):
        runner.run_truncated_sample(5303, "dev", by="vibes")


# ---------------------------------------------------------------- issue 8: shipped trajectory files
def test_shipped_samples_contain_eval_mode_and_both_truncation_kinds():
    heads = [r["header"] for r in _rows("samples.jsonl") if r["step"] == 0]
    by = {h["policy"]: h for h in heads}
    assert set(by) == {"careful_auditor", "stall_truncated_by_steps", "stall_truncated_by_time"}
    ev = by["careful_auditor"]
    assert ev["mode"] == "eval" and ev["reward_visible_to_policy"] is False and ev["grader_reward_shared"] > 0.9
    assert by["stall_truncated_by_steps"]["truncated"] and not by["stall_truncated_by_steps"]["time_truncated"]
    assert by["stall_truncated_by_time"]["truncated"] and by["stall_truncated_by_time"]["time_truncated"]
    rows = _rows("samples.jsonl")
    last_eval = [r for r in rows if r["policy"] == "careful_auditor" and r["terminated"]][0]
    assert last_eval["reward"] == 0.0
    for h in heads:
        assert h["label"] and re.fullmatch(r"[0-9a-f]{64}", h["config_hash"]) and h["env_version"] == "auditgym-1.0.0"


def test_oracle_rows_are_in_their_own_labelled_file():
    main = _rows("trajectories.jsonl")
    assert "oracle" not in {r["policy"] for r in main}
    assert {"random_explorer", "reconcile", "flag_all", "careful_auditor", "careful_auditor_degraded"} == \
           {r["policy"] for r in main}
    orc = _rows("oracle_reference.jsonl")
    assert {r["policy"] for r in orc} == {"oracle"}
    assert all("NOT agent behaviour" in r["header"]["label"] for r in orc if r["step"] == 0)


def test_every_shipped_trajectory_row_matches_the_row_schema():
    sch = schema_check.load_schema("trajectory_row.schema.json")
    for name in ("trajectories.jsonl", "oracle_reference.jsonl", "samples.jsonl"):
        rows = _rows(name)
        assert rows
        for i, r in enumerate(rows):
            errs = schema_check.check(r, sch)
            assert not errs, (name, i, errs[:2])
            assert ("header" in r) == (r["step"] == 0)
    # the schema actually rejects a bad row
    bad = dict(_rows("samples.jsonl")[0])
    bad["reward"] = "high"
    bad["extra"] = 1
    assert len(schema_check.check(bad, sch)) >= 2


def test_shipped_samples_replay_to_the_recorded_trajectories():
    rows = _rows("samples.jsonl")
    by = {}
    for r in rows:
        by.setdefault(r["policy"], []).append(r)
    e = runner.load_splits()["train"][0]
    for pol, recs in by.items():
        recs.sort(key=lambda r: r["step"])
        head = recs[0]["header"]
        acts = [r["action"] for r in recs[1:]]
        if pol == "stall_truncated_by_time":
            continue          # needs the fake clock of run_truncated_sample (checked above)
        rep = runner.replay_actions(acts, e["seed"], runner.resolve_config(e), mode=head["mode"], remote=False)
        assert runner.trajectory_hash(rep["trajectory"]) == head["trajectory_sha256"], pol
        assert rep["grader_report"]["reward_shared"] == head["grader_reward_shared"]


def test_write_trajectories_puts_the_header_on_step_zero_only(tmp_path):
    r = runner.run_episode("reconcile", 5303, "dev", mode=MODE_TRAIN, remote=False)
    n = runner.write_trajectories([r], str(tmp_path / "t.jsonl"))
    rows = [json.loads(x) for x in open(tmp_path / "t.jsonl", encoding="utf-8")]
    assert n == len(rows) == r["steps"] + 1
    assert [("header" in x) for x in rows] == [True] + [False] * r["steps"]
    assert rows[0]["header"]["trajectory_sha256"] == runner.trajectory_hash(r["trajectory"])


def test_check_trajectory_file_step_passes_on_the_shipped_files_and_fails_without_oracle_separation(monkeypatch, tmp_path):
    rep = check_mod.Report()
    check_mod.step_trajectory_files(rep)
    assert all(c["status"] == "PASS" for c in rep.checks), rep.checks
    # oracle mixed into trajectories.jsonl must be caught
    d = tmp_path / "trajectories"
    d.mkdir()
    for n in ("trajectories.jsonl", "oracle_reference.jsonl", "samples.jsonl"):
        (d / n).write_text(open(os.path.join(TRAJ_DIR, n), encoding="utf-8").read())
    with open(d / "trajectories.jsonl", "a", encoding="utf-8") as f:
        for r in _rows("oracle_reference.jsonl")[:2]:
            f.write(json.dumps(r) + "\n")
    monkeypatch.setattr(check_mod, "TRAJ_DIR", str(d))
    rep2 = check_mod.Report()
    check_mod.step_trajectory_files(rep2)
    assert any(c["status"] == "FAIL" and "oracle" in c["name"] for c in rep2.checks)
