"""AuditEnv / RemoteAuditEnv contract tests (SPEC sections 4 and 7, hackathon brief acceptance)."""
from __future__ import annotations

import base64
import copy
import json
import os
import random
import subprocess
import sys
import time

import pytest

from auditgym.client import RemoteAuditEnv, RemoteEnvError
from auditgym.constants import (ACTIONS, ERR_CALC, ERR_EPISODE_OVER, ERR_INVALID_ARGS,
                                ERR_INVALID_CURSOR, ERR_NOT_RESET, ERR_UNKNOWN_ACTION,
                                ERR_UNKNOWN_ID, ERR_UNKNOWN_TABLE, ERROR_CODES, INTERACTIONS,
                                MAX_STEPS, PAGE_LIMIT_MAX, PRIMARY_KEY, STEP_COST, TABLES, fx_id)
from auditgym.env import AuditEnv, GraderReportNotReady
from auditgym.generator import generate_episode

SEED = 11
OBS_KEYS = {"episode_id", "step", "steps_remaining", "result", "error"}
FIRST_OBS_KEYS = OBS_KEYS | {"task", "manifest", "policy"}
TRAIN_INFO_KEYS = {"metrics", "reward_shared", "reward_experimental", "inspection_steps",
                   "invalid_submission"}


# ------------------------------------------------------------------ helpers
def norm(t):
    return json.loads(json.dumps(list(t)))


def act(name, **args):
    return {"action": name, "args": args}


def calc(expr="1+1"):
    return act("calculator", expression=expr)


def empty_submission(ep_id):
    return act("submit", findings={"episode_id": ep_id, "findings": []})


def truth_findings(ep, only=None):
    out = []
    for it in ep.truth["issues"]:
        if only is not None and it["type"] != only:
            continue
        out.append({"type": it["type"], "affected_ids": list(it["affected_ids"]),
                    "claimed_amount_minor": it["expected_amount_minor"],
                    "currency": it["currency"], "evidence_ids": list(it["evidence_sets"][0]),
                    "explanation": "test"})
    return out


def submit_truth(ep):
    return act("submit", findings={"episode_id": ep.episode_id, "findings": truth_findings(ep)})


def step_ok(env, action):
    out = env.step(action)
    assert isinstance(out, tuple) and len(out) == 5
    return out


def assert_error(out, code, *, consumed_from=None, budget=MAX_STEPS):
    obs, reward, term, trunc, info = out
    assert obs["error"] is not None, obs
    assert obs["error"]["code"] == code, obs["error"]
    assert obs["error"]["code"] in ERROR_CODES
    assert set(obs["error"]) == {"code", "message"}
    msg = obs["error"]["message"]
    assert isinstance(msg, str) and 0 < len(msg) <= 120
    assert msg.isprintable()
    assert obs["result"] is None
    assert reward == 0.0
    assert (term, trunc) == (False, False)
    assert info == {}
    if consumed_from is not None:
        assert obs["step"] == consumed_from + 1
        assert obs["steps_remaining"] == budget - consumed_from - 1


def new_env(mode="train", seed=SEED, config=None, **kw):
    env = AuditEnv(mode=mode, **kw)
    obs, info = env.reset(seed=seed, config=config)
    return env, obs


def ids_by_table(ep):
    return {"ledger": [r["row_id"] for r in ep.ledger],
            "invoices": [r["invoice_id"] for r in ep.invoices],
            "payments": [r["payment_id"] for r in ep.payments],
            "fx": [fx_id(r["date"], r["source_currency"], r["target_currency"]) for r in ep.fx]}


def hidden_tokens(ep):
    toks = {"iss_", "expected_amount", "evidence_sets", "tolerance_minor", "scenario",
            "provenance", "injector", "root_cause", "clean_records", "records_added",
            "records_removed", "records_modified", '"truth"', "grader_data"}
    toks |= set(INTERACTIONS)
    toks |= {i["issue_id"] for i in ep.truth["issues"]}
    return toks


def assert_no_leak(blob, ep, seed=None):
    text = blob if isinstance(blob, str) else json.dumps(blob, sort_keys=True)
    low = text.lower()
    for t in hidden_tokens(ep):
        assert t.lower() not in low, f"leaked token {t!r}"
    assert '"seed"' not in low
    if seed is not None:
        assert ep.episode_id != f"ep_{seed}"


def random_action(rng, ids, ep_id):
    """Random mix of valid actions and assorted garbage (never contains hidden tokens)."""
    k = rng.random()
    tab = rng.choice(list(TABLES))
    if k < 0.15:
        return act("list_records", table=tab, limit=rng.choice([1, 5, 50]))
    if k < 0.25:
        return act("list_records", table=tab, limit=rng.choice([0, 51, -3, "7", None, 2.5, True, 10 ** 9]))
    if k < 0.32:
        return act("list_records", table=tab, cursor=rng.choice(["junk", "", "bGVkZ2VyOjA", 5, [], "A" * 400]))
    if k < 0.50:
        return act("get_record", table=tab, id=rng.choice(ids[tab]))
    if k < 0.58:
        return act("get_record", table=rng.choice([tab, "nope", "", 7, None]), id=rng.choice(["zzz", "R-00000000", "", 5, "x" * 300, "\n\x00‮"]))
    if k < 0.68:
        t = rng.choice(["invoices", "payments"])
        return act("get_document", doc_id=rng.choice([rng.choice(ids["_docs"]), "D-0000000000", "", 3, "y" * 500]))
    if k < 0.78:
        return calc(rng.choice(["1+1", "round(2.5)", "1/0", "x", "2**99", "__import__('os')", "'a'", "(" * 70, "10/4"]))
    if k < 0.84:
        return {"action": rng.choice(["drop", "LIST_RECORDS", "", None, 3, "z" * 400, "submit "]), "args": {}}
    if k < 0.88:
        return rng.choice([None, "list_records", [], 5, {}, {"args": {}}, {"action": "calculator", "args": []},
                           {"action": "list_records", "args": "x"}])
    if k < 0.94:
        return act("get_record", table=tab, id=rng.choice(ids[tab]), extra=1)
    return act("list_records", table=tab, cursor=None, limit=50)


def run_random(env, ep, steps, seed=0, with_submit=False):
    rng = random.Random(seed)
    ids = ids_by_table(ep)
    ids["_docs"] = [r["doc_id"] for r in ep.invoices] + [r["doc_id"] for r in ep.payments]
    outs = []
    for _ in range(steps):
        a = random_action(rng, ids, ep.episode_id)
        o = env.step(a)
        outs.append((a, o))
        if o[2] or o[3]:
            break
    return outs


# ------------------------------------------------------------------ reset
def test_reset_returns_obs_info_with_task_manifest_policy():
    env = AuditEnv()
    ret = env.reset(seed=SEED)
    assert isinstance(ret, tuple) and len(ret) == 2
    obs, info = ret
    assert set(obs) == FIRST_OBS_KEYS
    assert info == {}
    assert obs["step"] == 0
    assert obs["steps_remaining"] == MAX_STEPS == 100
    assert obs["result"] is None and obs["error"] is None
    assert isinstance(obs["task"], str) and len(obs["task"]) > 500
    m, p = obs["manifest"], obs["policy"]
    for k in ("schema_version", "env_version", "episode_id", "base_currency", "period_start",
              "period_end", "timezone", "counts", "rounding", "money"):
        assert k in m, k
    assert m["episode_id"] == obs["episode_id"] and obs["episode_id"].startswith("ep_")
    for k in ("revenue_recognition", "period_boundaries", "fx", "ledger", "postings",
              "duplicate_rule", "finding_amounts", "finding_affected_ids", "amount_tolerance", "evidence"):
        assert k in p, k
    for word in ("list_records", "get_record", "get_document", "calculator", "submit",
                 "duplicate_invoice", "payment_misallocation", "fx_error", "period_cutoff",
                 m["base_currency"]):
        assert word in obs["task"], word


def test_reset_obs_is_json_roundtrip_stable():
    obs, info = AuditEnv().reset(seed=SEED)
    assert json.loads(json.dumps(obs, allow_nan=False)) == obs
    assert json.loads(json.dumps(info, allow_nan=False)) == info


def test_reset_is_deterministic_and_seed_dependent():
    a, _ = AuditEnv().reset(seed=SEED)
    b, _ = AuditEnv().reset(seed=SEED)
    c, _ = AuditEnv().reset(seed=SEED + 1)
    assert a == b
    assert a["episode_id"] != c["episode_id"]


def test_manifest_counts_match_tables(make_env):
    env, obs = new_env(max_steps=10_000)
    counts = obs["manifest"]["counts"]
    totals = {}
    for t in TABLES:
        o = step_ok(env, act("list_records", table=t, limit=1))[0]
        totals[t] = o["result"]["total"]
    assert totals["ledger"] == counts["ledger_rows"]
    assert totals["invoices"] == counts["invoices"]
    assert totals["payments"] == counts["payments"]
    assert totals["fx"] == counts["fx_rates"]
    assert counts["documents"] == counts["invoices"] + counts["payments"]


def test_reset_rejects_bad_seed_type():
    with pytest.raises(Exception):
        AuditEnv().reset(seed="abc")


def test_constructor_validation():
    with pytest.raises(ValueError):
        AuditEnv(mode="nope")
    with pytest.raises(ValueError):
        AuditEnv(max_steps=0)


def test_reset_from_saved_episode_dirs(tmp_path):
    from auditgym.episode import save_episode
    ep = generate_episode(SEED, "train")
    pub, priv = save_episode(ep, str(tmp_path / "pub"), str(tmp_path / "priv"))
    env = AuditEnv()
    obs, info = env.reset(config={"episode_dir": pub, "grader_dir": priv})
    ref, _ = AuditEnv().reset(seed=SEED)
    assert obs["episode_id"] == ref["episode_id"] == ep.episode_id
    assert obs["manifest"] == ref["manifest"]
    # the loaded episode is playable and scored against the loaded truth
    out = env.step(submit_truth(ep))
    assert out[2] is True
    if ep.truth["issues"]:
        assert out[4]["metrics"]["tp"] == len(ep.truth["issues"])


# ------------------------------------------------------------------ step contract
def test_step_returns_five_tuple_with_correct_types():
    env, obs = new_env()
    o, r, term, trunc, info = step_ok(env, calc("1+1"))
    assert set(o) == OBS_KEYS
    assert o["episode_id"] == obs["episode_id"]
    assert o["step"] == 1 and o["steps_remaining"] == 99
    assert o["result"] == {"value": "2"} and o["error"] is None
    assert isinstance(r, float) and r == 0.0
    assert term is False and trunc is False and info == {}
    assert json.loads(json.dumps(o)) == o


def test_step_counter_and_budget_decrement_each_step():
    env, _ = new_env()
    for i in range(1, 11):
        o = step_ok(env, calc())[0]
        assert o["step"] == i and o["steps_remaining"] == 100 - i


# ------------------------------------------------------------------ happy paths
@pytest.fixture(scope="module")
def pub():
    return generate_episode(SEED, "train")


def test_list_records_each_table_happy_path(pub):
    env, _ = new_env(max_steps=500)
    for t in TABLES:
        o, r, term, trunc, info = step_ok(env, act("list_records", table=t))
        assert o["error"] is None and r == 0.0 and not term and not trunc and info == {}
        res = o["result"]
        assert set(res) == {"table", "rows", "next_cursor", "total"}
        assert res["table"] == t
        assert 0 < len(res["rows"]) <= PAGE_LIMIT_MAX
        assert res["total"] == len(getattr(pub, {"ledger": "ledger", "invoices": "invoices",
                                                 "payments": "payments", "fx": "fx"}[t]))
        assert (res["next_cursor"] is None) == (res["total"] <= len(res["rows"]))
        if res["next_cursor"] is not None:
            assert isinstance(res["next_cursor"], str) and res["next_cursor"]


def test_list_records_row_shapes(pub):
    env, _ = new_env(max_steps=500)
    led = step_ok(env, act("list_records", table="ledger", limit=5))[0]["result"]["rows"]
    assert led == pub.ledger[:5]
    inv = step_ok(env, act("list_records", table="invoices", limit=5))[0]["result"]["rows"]
    assert all("line_items" not in r for r in inv)
    assert all({"doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency", "total_minor"} <= set(r)
               for r in inv)
    pay = step_ok(env, act("list_records", table="payments", limit=5))[0]["result"]["rows"]
    assert pay == pub.payments[:5]
    assert all("invoice_refs" in r for r in pay)
    fx = step_ok(env, act("list_records", table="fx", limit=5))[0]["result"]["rows"]
    for r, ref in zip(fx, pub.fx[:5]):
        assert r["fx_id"] == fx_id(ref["date"], ref["source_currency"], ref["target_currency"])
        assert r["rate"] == ref["rate"] and isinstance(r["rate"], str)


def test_get_record_each_table_happy_path(pub):
    env, _ = new_env(max_steps=500)
    ids = ids_by_table(pub)
    for t in TABLES:
        for rid in ids[t][:3]:
            o, r, term, trunc, info = step_ok(env, act("get_record", table=t, id=rid))
            assert o["error"] is None and r == 0.0 and info == {}
            assert o["result"]["table"] == t
            rec = o["result"]["record"]
            assert rec[PRIMARY_KEY[t]] == rid
    inv = step_ok(env, act("get_record", table="invoices", id=ids["invoices"][0]))[0]["result"]["record"]
    assert inv == pub.invoices[0]
    assert isinstance(inv["line_items"], list) and inv["line_items"]
    assert {"line", "description", "quantity", "unit_price_minor", "vat_rate",
            "service_period", "po_ref"} <= set(inv["line_items"][0])
    pay = step_ok(env, act("get_record", table="payments", id=ids["payments"][0]))[0]["result"]["record"]
    assert pay == pub.payments[0]


def test_get_document_invoice_and_payment(pub):
    env, _ = new_env()
    inv = pub.invoices[0]
    o = step_ok(env, act("get_document", doc_id=inv["doc_id"]))[0]
    assert o["error"] is None
    assert o["result"]["doc_type"] == "invoice"
    assert o["result"]["document"] == inv
    pay = pub.payments[0]
    o = step_ok(env, act("get_document", doc_id=pay["doc_id"]))[0]
    assert o["result"]["doc_type"] == "payment"
    assert o["result"]["document"] == pay


def test_calculator_action_happy_path_and_errors():
    env, _ = new_env()
    o = step_ok(env, calc("0.1+0.2"))[0]
    assert o["result"] == {"value": "0.3"} and o["error"] is None
    assert_error(step_ok(env, calc("1/0")), ERR_CALC, consumed_from=1)
    assert_error(step_ok(env, calc("__import__('os').system('true')")), ERR_CALC, consumed_from=2)
    assert_error(step_ok(env, calc("1" * 600)), ERR_CALC, consumed_from=3)
    out = step_ok(env, calc("x" * 400))
    assert_error(out, ERR_CALC, consumed_from=4)
    assert "xxxxxxxxxxxx" not in out[0]["error"]["message"]
    assert_error(step_ok(env, act("calculator", expression=5)), ERR_INVALID_ARGS, consumed_from=5)
    assert_error(step_ok(env, act("calculator")), ERR_INVALID_ARGS, consumed_from=6)
    assert_error(step_ok(env, act("calculator", expression="1", more=1)), ERR_INVALID_ARGS, consumed_from=7)


def test_returned_data_is_a_copy_not_a_view(pub):
    env, _ = new_env(max_steps=500)
    a = step_ok(env, act("list_records", table="ledger", limit=3))[0]
    a["result"]["rows"][0]["account"] = "TAMPERED"
    a["result"]["rows"].clear()
    b = step_ok(env, act("list_records", table="ledger", limit=3))[0]
    assert b["result"]["rows"] == pub.ledger[:3]
    r = step_ok(env, act("get_record", table="invoices", id=pub.invoices[0]["invoice_id"]))[0]
    r["result"]["record"]["line_items"].append("junk")
    r2 = step_ok(env, act("get_record", table="invoices", id=pub.invoices[0]["invoice_id"]))[0]
    assert r2["result"]["record"] == pub.invoices[0]
    traj = env.trajectory()
    traj[0]["observation"]["manifest"]["counts"]["ledger_rows"] = -1
    assert env.trajectory()[0]["observation"]["manifest"]["counts"]["ledger_rows"] != -1


# ------------------------------------------------------------------ paging
def test_page_limit_50_ok_and_51_is_error():
    env, _ = new_env(max_steps=500)
    o = step_ok(env, act("list_records", table="ledger", limit=50))[0]
    assert o["error"] is None and len(o["result"]["rows"]) == 50
    o = step_ok(env, act("list_records", table="ledger", limit=1))[0]
    assert len(o["result"]["rows"]) == 1
    o = step_ok(env, act("list_records", table="ledger"))[0]       # default limit
    assert len(o["result"]["rows"]) == 50
    n = o["step"]
    out = step_ok(env, act("list_records", table="ledger", limit=51))
    assert_error(out, ERR_INVALID_ARGS, consumed_from=n, budget=500)   # step consumed
    for bad in (0, -1, 51, 1000, "10", None, 2.5, 5.0, True, False, [], {}):
        assert_error(step_ok(env, act("list_records", table="ledger", limit=bad)), ERR_INVALID_ARGS)


def test_paging_covers_all_rows_exactly_once_for_every_table(pub):
    env, _ = new_env(max_steps=10_000)
    ids = ids_by_table(pub)
    pk = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}
    for t in TABLES:
        for limit in (50, 7, 1):
            seen, cursor, pages = [], None, 0
            while True:
                o = step_ok(env, act("list_records", table=t, cursor=cursor, limit=limit))[0]
                assert o["error"] is None, o
                res = o["result"]
                assert res["total"] == len(ids[t])
                assert 1 <= len(res["rows"]) <= limit
                seen.extend(r[pk[t]] for r in res["rows"])
                pages += 1
                cursor = res["next_cursor"]
                if cursor is None:
                    break
                assert pages <= len(ids[t]) + 1
            assert seen == ids[t], (t, limit)                   # all rows, once each, same order
            assert len(set(seen)) == len(seen)
            assert pages == -(-len(ids[t]) // limit)


def test_same_cursor_twice_gives_same_page_and_is_stable():
    env, _ = new_env(max_steps=500)
    first = step_ok(env, act("list_records", table="ledger", limit=10))[0]["result"]
    cur = first["next_cursor"]
    p1 = step_ok(env, act("list_records", table="ledger", cursor=cur, limit=10))[0]["result"]
    step_ok(env, act("list_records", table="fx"))                       # unrelated calls in between
    step_ok(env, act("get_record", table="ledger", id=first["rows"][0]["row_id"]))
    p2 = step_ok(env, act("list_records", table="ledger", cursor=cur, limit=10))[0]["result"]
    assert p1 == p2
    assert p1["rows"] != first["rows"]
    # a different page size from the same cursor starts at the same offset
    p3 = step_ok(env, act("list_records", table="ledger", cursor=cur, limit=3))[0]["result"]
    assert p3["rows"] == p1["rows"][:3]
    # cursor itself is deterministic
    again = step_ok(env, act("list_records", table="ledger", limit=10))[0]["result"]
    assert again["next_cursor"] == cur


def test_cursor_valid_across_env_instances_same_episode():
    e1, _ = new_env(max_steps=500)
    cur = step_ok(e1, act("list_records", table="payments", limit=4))[0]["result"]["next_cursor"]
    e2, _ = new_env(max_steps=500)
    r1 = step_ok(e1, act("list_records", table="payments", cursor=cur, limit=4))[0]
    r2 = step_ok(e2, act("list_records", table="payments", cursor=cur, limit=4))[0]
    assert r1["error"] is None and r1["result"] == r2["result"]


def _good_cursor(env, table="ledger", limit=5):
    return step_ok(env, act("list_records", table=table, limit=limit))[0]["result"]["next_cursor"]


def test_garbled_cursors_are_invalid_cursor_and_consume_a_step():
    env, _ = new_env(max_steps=500)
    good = _good_cursor(env)
    raw = base64.urlsafe_b64decode(good + "=" * (-len(good) % 4)).decode()
    tbl, off, chk = raw.split(":")
    enc = lambda s: base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")
    flip = lambda c, i: c[:i] + ("B" if c[i] != "B" else "C") + c[i + 1:]   # change a char in the middle
    garbled = [
        "garbage", "!!!!", "====", " ", "A" * 500, good[:-3], good + "AAAA", good[::-1],
        good.upper() if good.upper() != good else "ZZZZ", flip(good, 5), flip(good, len(good) - 8), flip(good, 9),
        enc("ledger:0"), enc("ledger"), enc(f"ledger:{off}"), enc(f"ledger:{off}:{'0' * 10}"),
        enc(f"ledger:{int(off) + 1}:{chk}"),                    # forged offset, old checksum
        enc(f"ledger:0:{chk}"), enc(f"ledger:-1:{chk}"), enc(f"ledger:99999999999999:{chk}"),
        enc(f"ledger:1e2:{chk}"), enc(f"ledger:{off}:{chk}:extra"), enc(f"LEDGER:{off}:{chk}"),
        enc(f"fx:{off}:{chk}"), enc(f"nope:{off}:{chk}"), enc(f"ledger:{off}:{chk}\n"),
        "", 0, 5, 1.5, True, [], {}, [good], {"c": good},
    ]
    for g in garbled:
        n_before = env.trajectory()[-1]["step"]
        out = step_ok(env, act("list_records", table="ledger", cursor=g))
        assert_error(out, ERR_INVALID_CURSOR, consumed_from=n_before, budget=500)
        assert "A" * 100 not in out[0]["error"]["message"]
    # env still works afterwards and the good cursor is still good
    ok = step_ok(env, act("list_records", table="ledger", cursor=good))[0]
    assert ok["error"] is None and ok["result"]["rows"]


def test_cursor_from_another_table_is_invalid():
    env, _ = new_env(max_steps=500)
    cur = {t: _good_cursor(env, t) for t in ("ledger", "invoices", "payments", "fx")}
    for src, c in cur.items():
        for dst in cur:
            out = step_ok(env, act("list_records", table=dst, cursor=c))
            if src == dst:
                assert out[0]["error"] is None
            else:
                assert out[0]["error"]["code"] == ERR_INVALID_CURSOR, (src, dst)


def test_cursor_from_another_episode_is_invalid():
    e1, _ = new_env(seed=SEED, max_steps=500)
    e2, _ = new_env(seed=SEED + 1, max_steps=500)
    for t in TABLES:
        c1 = _good_cursor(e1, t, 3)
        out = step_ok(e2, act("list_records", table=t, cursor=c1))
        assert out[0]["error"]["code"] == ERR_INVALID_CURSOR, t


def test_cursor_does_not_reveal_data_only_position():
    env, _ = new_env()
    cur = _good_cursor(env)
    assert len(cur) < 100
    assert all(c.isalnum() or c in "-_=" for c in cur)             # opaque url-safe token


# ------------------------------------------------------------------ bounded errors
@pytest.mark.parametrize("bad_action,code", [
    (None, ERR_UNKNOWN_ACTION),
    ("list_records", ERR_UNKNOWN_ACTION),
    ([], ERR_UNKNOWN_ACTION),
    (5, ERR_UNKNOWN_ACTION),
    ({}, ERR_UNKNOWN_ACTION),
    ({"args": {}}, ERR_UNKNOWN_ACTION),
    ({"action": None}, ERR_UNKNOWN_ACTION),
    ({"action": 5}, ERR_UNKNOWN_ACTION),
    ({"action": ["list_records"]}, ERR_UNKNOWN_ACTION),
    ({"action": "drop_database", "args": {}}, ERR_UNKNOWN_ACTION),
    ({"action": "LIST_RECORDS", "args": {"table": "ledger"}}, ERR_UNKNOWN_ACTION),
    ({"action": " list_records", "args": {"table": "ledger"}}, ERR_UNKNOWN_ACTION),
    ({"action": "grader_report"}, ERR_UNKNOWN_ACTION),
    ({"action": "reset"}, ERR_UNKNOWN_ACTION),
    ({"action": "x" * 5000, "args": {}}, ERR_UNKNOWN_ACTION),
    ({"action": "\n\x00‮", "args": {}}, ERR_UNKNOWN_ACTION),
    ({"action": "list_records", "args": []}, ERR_INVALID_ARGS),
    ({"action": "list_records", "args": "table=ledger"}, ERR_INVALID_ARGS),
    ({"action": "list_records", "args": None}, ERR_INVALID_ARGS),       # no table
    ({"action": "list_records"}, ERR_INVALID_ARGS),
    ({"action": "list_records", "args": {"table": None}}, ERR_INVALID_ARGS),
    ({"action": "list_records", "args": {"table": 3}}, ERR_INVALID_ARGS),
    ({"action": "list_records", "args": {"table": ["ledger"]}}, ERR_INVALID_ARGS),
    ({"action": "list_records", "args": {"table": "ledger", "bogus": 1}}, ERR_INVALID_ARGS),
    ({"action": "list_records", "args": {"table": "nope"}}, ERR_UNKNOWN_TABLE),
    ({"action": "list_records", "args": {"table": ""}}, ERR_UNKNOWN_TABLE),
    ({"action": "list_records", "args": {"table": "LEDGER"}}, ERR_UNKNOWN_TABLE),
    ({"action": "list_records", "args": {"table": "ledger.csv"}}, ERR_UNKNOWN_TABLE),
    ({"action": "list_records", "args": {"table": "truth"}}, ERR_UNKNOWN_TABLE),
    ({"action": "list_records", "args": {"table": "provenance"}}, ERR_UNKNOWN_TABLE),
    ({"action": "list_records", "args": {"table": "t" * 500}}, ERR_UNKNOWN_TABLE),
    ({"action": "get_record", "args": {"table": "nope", "id": "x"}}, ERR_UNKNOWN_TABLE),
    ({"action": "get_record", "args": {"table": "ledger"}}, ERR_INVALID_ARGS),
    ({"action": "get_record", "args": {"id": "x"}}, ERR_INVALID_ARGS),
    ({"action": "get_record", "args": {"table": "ledger", "id": 5}}, ERR_INVALID_ARGS),
    ({"action": "get_record", "args": {"table": "ledger", "id": None}}, ERR_INVALID_ARGS),
    ({"action": "get_record", "args": {"table": "ledger", "id": "R-nope"}}, ERR_UNKNOWN_ID),
    ({"action": "get_record", "args": {"table": "ledger", "id": ""}}, ERR_UNKNOWN_ID),
    ({"action": "get_record", "args": {"table": "invoices", "id": "INV-0"}}, ERR_UNKNOWN_ID),
    ({"action": "get_record", "args": {"table": "payments", "id": "nope"}}, ERR_UNKNOWN_ID),
    ({"action": "get_record", "args": {"table": "fx", "id": "fx:1999-01-01:EUR:KZT"}}, ERR_UNKNOWN_ID),
    ({"action": "get_record", "args": {"table": "fx", "id": "2026-03-02"}}, ERR_UNKNOWN_ID),
    ({"action": "get_record", "args": {"table": "ledger", "id": "i" * 1000}}, ERR_UNKNOWN_ID),
    ({"action": "get_record", "args": {"table": "ledger", "id": "\n\x00‮\U0001f600"}}, ERR_UNKNOWN_ID),
    ({"action": "get_record", "args": {"table": "ledger", "id": "x", "extra": 1}}, ERR_INVALID_ARGS),
    ({"action": "get_document", "args": {}}, ERR_INVALID_ARGS),
    ({"action": "get_document"}, ERR_INVALID_ARGS),
    ({"action": "get_document", "args": {"doc_id": 5}}, ERR_INVALID_ARGS),
    ({"action": "get_document", "args": {"doc_id": None}}, ERR_INVALID_ARGS),
    ({"action": "get_document", "args": {"doc_id": "D-0000000000"}}, ERR_UNKNOWN_ID),
    ({"action": "get_document", "args": {"doc_id": ""}}, ERR_UNKNOWN_ID),
    ({"action": "get_document", "args": {"doc_id": "d" * 1000}}, ERR_UNKNOWN_ID),
    ({"action": "get_document", "args": {"doc_id": "x", "table": "invoices"}}, ERR_INVALID_ARGS),
    ({"action": "calculator", "args": {}}, ERR_INVALID_ARGS),
    ({"action": "calculator", "args": {"expression": "1/0"}}, ERR_CALC),
])
def test_bad_actions_give_bounded_error_consume_step_and_zero_reward(bad_action, code):
    env, _ = new_env()
    step_ok(env, calc())                                              # step 1
    out = step_ok(env, copy.deepcopy(bad_action))
    assert_error(out, code, consumed_from=1)
    # the episode continues normally afterwards
    o = step_ok(env, calc("2+2"))[0]
    assert o["step"] == 3 and o["result"] == {"value": "4"}


def test_non_json_action_object_is_unknown_action():
    env, _ = new_env()
    assert_error(step_ok(env, {"action": object(), "args": {}}), ERR_UNKNOWN_ACTION, consumed_from=0)
    assert_error(step_ok(env, {"action": "list_records", "args": {"table": object()}}), ERR_INVALID_ARGS,
                 consumed_from=1)


def test_error_message_echo_is_truncated_to_64_chars(pub):
    env, _ = new_env()
    long_id = "q" * 300
    for a in (act("get_record", table="ledger", id=long_id), act("get_document", doc_id=long_id),
              act("list_records", table=long_id), {"action": long_id, "args": {}}):
        out = step_ok(env, a)
        msg = out[0]["error"]["message"]
        assert len(msg) <= 120
        assert "q" * 65 not in msg


def test_error_for_hidden_removed_ids_is_indistinguishable_from_unknown_id():
    ep = generate_episode(SEED + 3, "train")
    public = ep.public_ids()
    removed = []
    for ch in ep.provenance.get("changes", []):
        removed.extend(ch.get("records_removed", []))
    removed = [r for r in removed if isinstance(r, str) and r not in public]
    env, _ = new_env(seed=SEED + 3)
    probes = removed[:6] + ["R-ffffffff", "D-ffffffffff"]
    shapes = set()
    for pid in probes:
        for a in (act("get_record", table="ledger", id=pid), act("get_record", table="invoices", id=pid),
                  act("get_record", table="payments", id=pid), act("get_document", doc_id=pid)):
            if env.ended():
                break
            out = step_ok(env, a)
            e = out[0]["error"]
            assert e["code"] == ERR_UNKNOWN_ID
            shapes.add((a["action"], e["message"].replace(pid, "<ID>")))
    # one message template per action/table kind, regardless of whether the id is "hidden" or random
    assert len({s for s in shapes if s[0] == "get_document"}) == 1


# ------------------------------------------------------------------ rewards of non-terminal steps
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_all_non_terminal_steps_have_zero_reward_and_empty_info(mode, pub):
    env, _ = new_env(mode=mode)
    outs = run_random(env, pub, 60, seed=5)
    assert len(outs) == 60
    for a, (o, r, term, trunc, info) in outs:
        assert r == 0.0 and isinstance(r, float)
        assert (term, trunc) == (False, False)
        assert info == {}


# ------------------------------------------------------------------ submit
def test_submit_terminates_and_ends_episode_train_mode(issue_config):
    ep = generate_episode(SEED, issue_config)
    env, obs = new_env(config=issue_config)
    for _ in range(3):
        step_ok(env, calc())
    o, r, term, trunc, info = step_ok(env, submit_truth(ep))
    assert term is True and trunc is False
    assert o["error"] is None and o["step"] == 4
    assert set(info) == TRAIN_INFO_KEYS
    assert info["inspection_steps"] == 3
    assert info["invalid_submission"] is False
    assert info["metrics"]["tp"] == len(ep.truth["issues"]) and info["metrics"]["fp"] == 0
    assert info["metrics"]["fn"] == 0 and info["metrics"]["f1"] == 1.0
    assert r == pytest.approx(1.0 - STEP_COST * 3)
    assert info["reward_shared"] == r
    assert -1.0 <= info["reward_experimental"] <= 1.0
    assert json.loads(json.dumps(info, allow_nan=False)) == info
    assert all(isinstance(v, (int, float, bool, dict)) for v in info.values())     # counts only, no ids
    assert all(isinstance(v, (int, float)) for v in info["metrics"].values())


def test_submit_empty_on_clean_books_vs_issues_vs_false_accusation(clean_config, issue_config):
    # clean + silence -> best
    env, obs = new_env(config=clean_config)
    r_clean = step_ok(env, empty_submission(obs["episode_id"]))[1]
    assert r_clean == pytest.approx(1.0)
    # issues + silence -> -1 (clip)
    env, obs = new_env(config=issue_config)
    r_silent = step_ok(env, empty_submission(obs["episode_id"]))[1]
    assert r_silent == -1.0
    # issues + correct answer beats silence; clean + false accusation loses to silence
    ep = generate_episode(SEED, issue_config)
    env, obs = new_env(config=issue_config)
    r_correct = step_ok(env, submit_truth(ep))[1]
    assert r_correct > r_silent
    cep = generate_episode(SEED, clean_config)
    env, obs = new_env(config=clean_config)
    bogus = {"type": "fx_error", "affected_ids": [cep.invoices[0]["invoice_id"]],
             "claimed_amount_minor": 1000, "currency": cep.manifest["base_currency"],
             "evidence_ids": [cep.invoices[0]["doc_id"]], "explanation": "x"}
    r_false = step_ok(env, act("submit", findings={"episode_id": obs["episode_id"], "findings": [bogus]}))[1]
    assert r_false < r_clean and r_false == -1.0


def test_duplicate_of_correct_finding_gives_no_gain(issue_config):
    ep = generate_episode(SEED, issue_config)
    fs = truth_findings(ep)
    env, obs = new_env(config=issue_config)
    r1 = step_ok(env, act("submit", findings={"episode_id": obs["episode_id"], "findings": fs}))[1]
    env, obs = new_env(config=issue_config)
    r2 = step_ok(env, act("submit", findings={"episode_id": obs["episode_id"], "findings": fs + [fs[0], fs[0]]}))[1]
    assert r2 < r1


def test_submit_without_findings_key_is_an_invalid_submission():
    # SPEC section 4 (arbiter): invalid submission structure => reward -1 and the episode ends
    env, obs = new_env()
    out = step_ok(env, {"action": "submit", "args": {}})
    assert out[0]["error"]["code"] == ERR_INVALID_ARGS
    assert len(out[0]["error"]["message"]) <= 120
    assert out[1] == -1.0 and out[2] is True
    assert env.grader_report()["errors"]


def _bad_submissions(eid, base, fid):
    good = {"type": "fx_error", "affected_ids": [fid], "claimed_amount_minor": 1, "currency": base,
            "evidence_ids": [fid], "explanation": ""}
    def f(**kw):
        d = dict(good); d.update(kw); return d
    def sub(findings): return {"episode_id": eid, "findings": findings}
    missing = dict(good); del missing["explanation"]
    extra = dict(good); extra["note"] = "x"
    return {
        "not_dict": "nope", "none": None, "list": [], "int": 5,
        "missing_episode_id": {"findings": []},
        "wrong_episode_id": {"episode_id": "ep_000000000000", "findings": []},
        "missing_findings": {"episode_id": eid},
        "extra_top_key": {"episode_id": eid, "findings": [], "x": 1},
        "findings_str": sub("none"), "findings_none": sub(None), "findings_dict": sub({}),
        "finding_not_dict": sub(["x"]), "finding_missing_key": sub([missing]), "finding_extra_key": sub([extra]),
        "bad_type": sub([f(type="made_up")]), "type_int": sub([f(type=3)]),
        "affected_empty": sub([f(affected_ids=[])]), "affected_str": sub([f(affected_ids="a")]),
        "affected_nonstr": sub([f(affected_ids=[1])]), "affected_11": sub([f(affected_ids=[f"a{i}" for i in range(11)])]),
        "amount_bool": sub([f(claimed_amount_minor=True)]), "amount_negative": sub([f(claimed_amount_minor=-1)]),
        "amount_float": sub([f(claimed_amount_minor=1.5)]), "amount_str": sub([f(claimed_amount_minor="1")]),
        "amount_none": sub([f(claimed_amount_minor=None)]), "currency_int": sub([f(currency=5)]),
        "evidence_11": sub([f(evidence_ids=[f"e{i}" for i in range(11)])]),
        "evidence_str": sub([f(evidence_ids="e")]), "explanation_none": sub([f(explanation=None)]),
        "explanation_1001": sub([f(explanation="x" * 1001)]),
        "findings_31": sub([good] * 31),
    }


def test_invalid_submissions_train_mode_reward_minus_one_terminated(pub):
    base = pub.manifest["base_currency"]
    fid = pub.invoices[0]["doc_id"]
    for name, sub in _bad_submissions(pub.episode_id, base, fid).items():
        env, obs = new_env()
        o, r, term, trunc, info = step_ok(env, act("submit", findings=sub))
        assert term is True and trunc is False, name
        assert r == -1.0, name
        assert o["error"]["code"] == ERR_INVALID_ARGS, name
        assert o["error"]["message"].startswith("submission rejected") and len(o["error"]["message"]) <= 120, name
        assert info["invalid_submission"] is True and info["reward_shared"] == -1.0, name
        assert_no_leak(o, pub)
        assert_no_leak(info, pub)
        rep = env.grader_report()
        assert rep["invalid_submission"] is True and rep["errors"], name


def test_invalid_submissions_eval_mode_hide_reason_but_grader_keeps_it(pub):
    base = pub.manifest["base_currency"]
    fid = pub.invoices[0]["doc_id"]
    for name, sub in list(_bad_submissions(pub.episode_id, base, fid).items())[::3]:
        env, obs = new_env(mode="eval")
        o, r, term, trunc, info = step_ok(env, act("submit", findings=sub))
        assert term is True and trunc is False
        assert r == 0.0 and info == {}
        # brief: eval mode must not leak 1 bit -> the terminal observation of an invalid submit is
        # identical to that of a valid one (error None); reason and -1 live only in the grader report
        # (this replaces the earlier expectation of an explicit 'submission rejected' error in eval)
        env_ok, _ = new_env(mode="eval")
        o_ok, r_ok, t_ok, tr_ok, i_ok = step_ok(env_ok, empty_submission(obs["episode_id"]))
        assert o["error"] is None and o == o_ok and (r, term, trunc, info) == (r_ok, t_ok, tr_ok, i_ok), name
        rep = env.grader_report()
        assert rep["invalid_submission"] is True
        assert rep["reward_shared"] == -1.0
        assert rep["errors"] and any(e["message"] and e["message"] != "submission rejected" for e in rep["errors"]), name
        # the same rejection reason is listed once, not twice (invalid_args + submission_invalid)
        msgs = [e["message"] for e in rep["errors"]]
        assert len(msgs) == len(set(msgs)), name


# ------------------------------------------------------------------ step budget
def test_hundred_step_limit_truncates_with_empty_answer_scored(clean_config, issue_config):
    # clean books: empty answer has F1 = 1 but all 100 steps are inspection steps
    env, obs = new_env(config=clean_config)
    for i in range(99):
        o, r, term, trunc, info = step_ok(env, calc())
        assert (term, trunc) == (False, False) and r == 0.0 and o["steps_remaining"] == 99 - i
    o, r, term, trunc, info = step_ok(env, calc())
    assert (term, trunc) == (False, True)
    assert o["step"] == 100 and o["steps_remaining"] == 0
    assert r == pytest.approx(2 * 1.0 - 1 - STEP_COST * 100)
    assert info["inspection_steps"] == 100 and info["invalid_submission"] is False
    assert info["metrics"]["tp"] == 0 and info["metrics"]["fp"] == 0
    # books with issues: empty answer -> recall 0 -> -1
    ep = generate_episode(SEED, issue_config)
    env, obs = new_env(config=issue_config)
    for _ in range(99):
        step_ok(env, calc())
    o, r, term, trunc, info = step_ok(env, calc())
    assert (term, trunc) == (False, True)
    assert r == -1.0
    assert info["metrics"]["tp"] == 0 and info["metrics"]["fn"] == len(ep.truth["issues"])
    rep = env.grader_report()
    assert rep["truncated"] is True and rep["terminated"] is False
    assert rep["metrics"]["fn"] == len(ep.truth["issues"])


@pytest.mark.parametrize("last_action", [
    {"action": "bogus", "args": {}},
    {"action": "get_record", "args": {"table": "ledger", "id": "nope"}},
    {"action": "list_records", "args": {"table": "ledger"}},
    {"action": "calculator", "args": {"expression": "1/0"}},
])
def test_hundredth_step_truncates_even_if_it_was_an_error(last_action):
    env, obs = new_env()
    for _ in range(99):
        step_ok(env, calc())
    o, r, term, trunc, info = step_ok(env, last_action)
    assert (term, trunc) == (False, True)
    assert o["step"] == 100 and o["steps_remaining"] == 0
    assert isinstance(r, float) and -1.0 <= r <= 1.0
    assert info["invalid_submission"] is False


def test_submit_as_hundredth_step_is_terminated_not_truncated(issue_config):
    ep = generate_episode(SEED, issue_config)
    env, obs = new_env(config=issue_config)
    for _ in range(99):
        step_ok(env, calc())
    o, r, term, trunc, info = step_ok(env, submit_truth(ep))
    assert term is True and trunc is False
    assert info["inspection_steps"] == 99
    assert info["metrics"]["tp"] == len(ep.truth["issues"])
    assert r == pytest.approx(max(-1.0, 1.0 - STEP_COST * 99))


def test_custom_small_step_budget(clean_config):
    env, obs = new_env(max_steps=3, config=clean_config)
    assert obs["steps_remaining"] == 3
    step_ok(env, calc()); step_ok(env, calc())
    o, r, term, trunc, info = step_ok(env, calc())
    assert trunc is True and not term and o["steps_remaining"] == 0


def test_truncated_in_eval_mode_hides_reward_and_info_but_grader_has_it(issue_config):
    ep = generate_episode(SEED, issue_config)
    env, obs = new_env(mode="eval", config=issue_config)
    for _ in range(99):
        step_ok(env, calc())
    o, r, term, trunc, info = step_ok(env, calc())
    assert trunc is True and not term
    assert r == 0.0 and info == {}
    rep = env.grader_report()
    assert rep["reward_shared"] == -1.0 and rep["ended_by"] == "steps"
    assert rep["metrics"]["fn"] == len(ep.truth["issues"])


# ------------------------------------------------------------------ time limit
def test_time_limit_truncates_with_empty_answer(issue_config):
    ep = generate_episode(SEED, issue_config)
    env, obs = new_env(time_limit_s=0.2, config=issue_config)
    step_ok(env, calc())
    time.sleep(0.4)
    o, r, term, trunc, info = step_ok(env, submit_truth(ep))          # a late (correct!) submit does not count
    assert (term, trunc) == (False, True)
    assert r == -1.0
    assert info["metrics"]["tp"] == 0 and info["metrics"]["fn"] == len(ep.truth["issues"])
    assert info["invalid_submission"] is False
    rep = env.grader_report()
    assert rep["truncated"] is True and rep["terminated"] is False and rep["ended_by"] == "time"
    out = step_ok(env, calc())                                        # over afterwards
    assert out[0]["error"]["code"] == ERR_EPISODE_OVER and out[2:4] == (False, True)


def test_time_limit_on_clean_books_scores_empty_answer(clean_config):
    env, obs = new_env(time_limit_s=0.2, config=clean_config)
    time.sleep(0.4)
    o, r, term, trunc, info = step_ok(env, calc())
    assert trunc is True and not term
    assert r >= 0.99                                                   # F1=1 minus at most a step cost


def test_time_limit_eval_mode_hides_result():
    env, obs = new_env(mode="eval", time_limit_s=0.2)
    time.sleep(0.4)
    o, r, term, trunc, info = step_ok(env, calc())
    assert trunc is True and r == 0.0 and info == {}
    assert env.grader_report()["truncated"] is True


def test_time_limit_is_measured_from_reset_not_construction():
    env = AuditEnv(time_limit_s=0.5)
    time.sleep(0.6)
    env.reset(seed=SEED)
    o, r, term, trunc, info = step_ok(env, calc())
    assert (term, trunc) == (False, False) and o["result"] == {"value": "2"}
    time.sleep(0.6)
    env.reset(seed=SEED)                                               # reset restarts the clock
    assert step_ok(env, calc())[3] is False


# ------------------------------------------------------------------ after the end / before reset
@pytest.mark.parametrize("how", ["submit", "steps", "time"])
def test_steps_after_end_are_episode_over_with_no_state_change(how, issue_config):
    ep = generate_episode(SEED, issue_config)
    kw = {"time_limit_s": 0.2} if how == "time" else {}
    env, obs = new_env(config=issue_config, **kw)
    if how == "submit":
        end = step_ok(env, submit_truth(ep))
    elif how == "steps":
        for _ in range(99):
            step_ok(env, calc())
        end = step_ok(env, calc())
    else:
        time.sleep(0.4)
        end = step_ok(env, calc())
    term, trunc = end[2], end[3]
    assert term or trunc
    rep_before = json.dumps(env.grader_report(), sort_keys=True)
    traj_before = json.dumps(env.trajectory(), sort_keys=True)
    last_step = end[0]["step"]
    for a in (calc(), submit_truth(ep), act("list_records", table="ledger"), {"action": "bogus"}, None,
              empty_submission(obs["episode_id"])):
        o, r, t2, tr2, info = step_ok(env, a)
        assert o["error"]["code"] == ERR_EPISODE_OVER
        assert len(o["error"]["message"]) <= 120
        assert r == 0.0 and info == {}
        assert (t2, tr2) == (term, trunc)                              # flags as they were
        assert o["step"] == last_step and o["result"] is None
    assert json.dumps(env.grader_report(), sort_keys=True) == rep_before
    assert json.dumps(env.trajectory(), sort_keys=True) == traj_before


def test_resubmit_after_end_cannot_change_score(clean_config, issue_config):
    ep = generate_episode(SEED, issue_config)
    env, obs = new_env(mode="eval", config=issue_config)
    step_ok(env, empty_submission(obs["episode_id"]))
    before = env.grader_report()["reward_shared"]
    step_ok(env, submit_truth(ep))
    assert env.grader_report()["reward_shared"] == before == -1.0


def test_step_before_reset_is_not_reset_error():
    env = AuditEnv()
    for a in (calc(), empty_submission("ep_x"), None, {"action": "bogus"}):
        o, r, term, trunc, info = env.step(a)
        assert o["error"]["code"] == ERR_NOT_RESET
        assert set(o) == OBS_KEYS
        assert o["step"] == 0 and r == 0.0 and (term, trunc) == (False, False) and info == {}
        assert json.loads(json.dumps(o)) == o
    assert env.trajectory() == []
    with pytest.raises(GraderReportNotReady):                           # sealed until the episode ended
        env.grader_report()
    obs, _ = env.reset(seed=SEED)                                       # then it works
    assert step_ok(env, calc())[0]["result"] == {"value": "2"}


def test_failed_reset_leaves_env_not_reset():
    env, _ = new_env()
    with pytest.raises(Exception):
        env.reset(seed=1, config="no_such_config_anywhere")
    assert env.step(calc())[0]["error"]["code"] == ERR_NOT_RESET


# ------------------------------------------------------------------ reset isolation
def test_reset_isolation_no_state_carry_over(issue_config):
    ep = generate_episode(SEED, issue_config)
    env, obs1 = new_env(config=issue_config)
    for a in (calc(), act("get_record", table="ledger", id="nope"), act("list_records", table="fx", limit=3),
              submit_truth(ep)):
        step_ok(env, a)
    assert env.ended()
    obs2, info2 = env.reset(seed=SEED, config=issue_config)
    assert obs2 == obs1 and info2 == {}
    assert obs2["step"] == 0 and obs2["steps_remaining"] == 100
    assert env.ended() is False
    traj = env.trajectory()
    assert len(traj) == 1 and traj[0]["step"] == 0 and traj[0]["action"] is None
    with pytest.raises(GraderReportNotReady):            # a fresh episode has no report until it ends
        env.grader_report()
    o = step_ok(env, calc("3+4"))[0]
    assert o["step"] == 1 and o["result"] == {"value": "7"}
    # a different episode after a finished one
    obs3, _ = env.reset(seed=SEED + 5)
    assert obs3["episode_id"] != obs1["episode_id"]
    assert env._errors == []
    out = step_ok(env, act("get_record", table="ledger", id=ep.ledger[0]["row_id"]))   # old ids are not valid in new episode
    assert out[0]["error"]["code"] == ERR_UNKNOWN_ID
    assert len(env.trajectory()) == 2


def test_reset_after_truncation_and_after_invalid_submission(clean_config):
    env, obs = new_env(max_steps=2, config=clean_config)
    step_ok(env, calc()); step_ok(env, calc())
    assert env.ended()
    env.reset(seed=SEED, config=clean_config, )
    assert not env.ended()
    assert step_ok(env, calc())[3] is False                              # max_steps persists, step count fresh
    env.reset(seed=SEED)
    step_ok(env, act("submit", findings="junk"))
    assert env.ended() and env.grader_report()["errors"]
    env.reset(seed=SEED)
    assert env._errors == []


def test_two_envs_do_not_share_state():
    a, _ = new_env(seed=SEED)
    b, _ = new_env(seed=SEED + 1)
    step_ok(a, calc()); step_ok(a, calc())
    ob = step_ok(b, calc())[0]
    assert ob["step"] == 1
    step_ok(a, empty_submission(a.trajectory()[0]["episode_id"]))
    assert a.ended() and not b.ended()
    assert step_ok(b, calc())[0]["step"] == 2


def test_reset_mutation_of_returned_first_obs_does_not_leak_into_env():
    env = AuditEnv()
    obs, _ = env.reset(seed=SEED)
    obs["manifest"]["counts"]["ledger_rows"] = -1
    obs["policy"]["fx"] = "TAMPERED"
    obs["task"] = ""
    obs2, _ = env.reset(seed=SEED)
    assert obs2["manifest"]["counts"]["ledger_rows"] > 0
    assert obs2["policy"]["fx"] != "TAMPERED" and obs2["task"]


def test_reset_options_transform_changes_ids_but_keeps_playable():
    env = AuditEnv()
    base, _ = env.reset(seed=SEED)
    obs, _ = env.reset(seed=SEED, options={"transform": {"rename_salt": 7, "permute_salt": 9}})
    assert obs["manifest"]["counts"] == base["manifest"]["counts"]
    o = step_ok(env, act("list_records", table="ledger", limit=2))[0]
    assert o["error"] is None
    r = step_ok(env, empty_submission(obs["episode_id"]))
    assert r[2] is True


# ------------------------------------------------------------------ modes
@pytest.mark.parametrize("how", ["submit", "invalid", "steps"])
def test_eval_mode_returns_reward_zero_info_empty_but_grader_has_real_result(how, issue_config):
    ep = generate_episode(SEED, issue_config)
    n_issues = len(ep.truth["issues"])
    env, obs = new_env(mode="eval", config=issue_config)
    if how == "submit":
        for _ in range(2):
            step_ok(env, calc())
        o, r, term, trunc, info = step_ok(env, submit_truth(ep))
        expect = 1.0 - STEP_COST * 2
    elif how == "invalid":
        o, r, term, trunc, info = step_ok(env, act("submit", findings=123))
        expect = -1.0
    else:
        for _ in range(99):
            step_ok(env, calc())
        o, r, term, trunc, info = step_ok(env, calc())
        expect = -1.0
    assert term or trunc
    assert r == 0.0 and isinstance(r, float)
    assert info == {}
    rep = env.grader_report()
    assert rep["ended"] is True and rep["mode"] == "eval"
    assert rep["reward_shared"] == pytest.approx(expect)
    assert rep["n_truth_issues"] == n_issues
    if how == "submit":
        assert rep["metrics"]["tp"] == n_issues and rep["metrics"]["f1"] == 1.0
        assert rep["pairs"] and rep["inspection_steps"] == 2
    # nothing the policy sees reveals the score
    assert_no_leak(env.trajectory(), ep)
    assert all(rec["reward"] == 0.0 for rec in env.trajectory())


def test_train_and_eval_grader_reports_agree(issue_config):
    ep = generate_episode(SEED, issue_config)
    reps = []
    for mode in ("train", "eval"):
        env, _ = new_env(mode=mode, config=issue_config)
        step_ok(env, calc())
        step_ok(env, submit_truth(ep))
        reps.append(env.grader_report())
    for k in ("reward_shared", "reward_experimental", "metrics", "inspection_steps", "n_truth_issues", "episode_id"):
        assert reps[0][k] == reps[1][k], k


def test_grader_report_before_end_is_refused():
    env, _ = new_env()
    with pytest.raises(GraderReportNotReady):
        env.grader_report()
    step_ok(env, calc())
    with pytest.raises(GraderReportNotReady):          # mid-episode: no truth-derived data at all
        env.grader_report()
    step_ok(env, empty_submission(env._ep.episode_id))
    assert env.grader_report()["ended"] is True
    # and nothing truth-bearing is held by the Episode object the env keeps
    assert env._ep.truth == {} and env._ep.provenance == {}


# ------------------------------------------------------------------ trajectory
def test_trajectory_records(issue_config):
    ep = generate_episode(SEED, issue_config)
    env, obs = new_env(config=issue_config)
    sent = [calc(), act("get_record", table="ledger", id="nope"), act("list_records", table="fx", limit=2), {"action": "zzz"}]
    outs = [step_ok(env, a) for a in sent]
    end = step_ok(env, submit_truth(ep))
    traj = env.trajectory()
    assert len(traj) == 1 + len(sent) + 1
    assert traj[0] == {"episode_id": obs["episode_id"], "step": 0, "action": None, "observation": obs,
                       "reward": 0.0, "terminated": False, "truncated": False}
    for i, (a, o) in enumerate(zip(sent + [submit_truth(ep)], outs + [end]), start=1):
        rec = traj[i]
        assert set(rec) == {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"}
        assert rec["step"] == i and rec["episode_id"] == obs["episode_id"]
        assert rec["action"] == a
        assert rec["observation"] == o[0] and rec["reward"] == o[1]
        assert (rec["terminated"], rec["truncated"]) == (o[2], o[3])
    assert traj[-1]["terminated"] is True
    json.dumps(traj, allow_nan=False)
    step_ok(env, calc())                                                 # episode_over: nothing recorded
    assert len(env.trajectory()) == len(traj)


def test_trajectory_is_a_copy():
    env, _ = new_env()
    step_ok(env, calc())
    t = env.trajectory()
    t.append("junk")
    t[1]["observation"]["result"] = "tampered"
    assert len(env.trajectory()) == 2 and env.trajectory()[1]["observation"]["result"] == {"value": "2"}


def test_same_seed_same_actions_gives_identical_trajectory(pub):
    t = []
    for _ in range(2):
        env, _ = new_env()
        run_random(env, pub, 70, seed=21)
        t.append(json.dumps(env.trajectory(), sort_keys=True))
    assert t[0] == t[1]
    env, _ = new_env(seed=SEED + 1)
    run_random(env, generate_episode(SEED + 1, "train"), 10, seed=21)
    assert json.dumps(env.trajectory(), sort_keys=True) != t[0]


# ------------------------------------------------------------------ no truth leakage
@pytest.mark.parametrize("seed,mode", [(SEED, "train"), (SEED + 1, "eval"), (SEED + 2, "train"), (SEED + 7, "eval")])
def test_no_truth_tokens_in_any_observation_info_or_error(seed, mode, issue_config):
    ep = generate_episode(seed, issue_config)
    env, first = new_env(mode=mode, seed=seed, config=issue_config)
    assert_no_leak(first, ep, seed)
    outs = run_random(env, ep, 99, seed=seed)
    blob = [first] + [o for _, o in outs]
    n_err = sum(1 for _, o in outs if o[0]["error"])
    assert n_err >= 15                                                   # many error paths exercised
    assert_no_leak(blob, ep, seed)
    assert_no_leak(env.trajectory(), ep, seed)
    for o in [x for _, x in outs]:
        assert json.loads(json.dumps(o[0], allow_nan=False)) == o[0]     # observations are plain JSON
        assert json.loads(json.dumps(o[4], allow_nan=False)) == o[4]
    # terminal outputs too: correct, partially wrong and malformed submissions
    for sub in (submit_truth(ep), empty_submission(ep.episode_id), act("submit", findings={"episode_id": "ep_x"}),
                act("submit", findings=[1, 2, 3])):
        e2, _ = new_env(mode=mode, seed=seed, config=issue_config)
        out = step_ok(e2, sub)
        assert_no_leak([out[0], out[4]], ep, seed)
        assert (out[2] or out[3]) is True


def test_first_observation_has_no_private_file_names_or_paths(tmp_path, issue_config):
    from auditgym.episode import save_episode
    ep = generate_episode(SEED, issue_config)
    pub_dir, priv_dir = save_episode(ep, str(tmp_path / "ep"), str(tmp_path / "gd"))
    env = AuditEnv()
    obs, _ = env.reset(config={"episode_dir": pub_dir, "grader_dir": priv_dir})
    text = json.dumps(obs)
    assert "truth.json" not in text and "provenance.json" not in text and priv_dir not in text and str(tmp_path) not in text
    assert_no_leak(obs, ep)


def test_guessing_truth_style_ids_gets_plain_unknown_id(issue_config):
    ep = generate_episode(SEED, issue_config)
    env, _ = new_env(config=issue_config)
    # an agent asking for the hidden issue ids / provenance gets the same bounded errors as any junk
    for it in ep.truth["issues"][:3]:
        out = step_ok(env, act("get_record", table="ledger", id=it["issue_id"]))
        assert out[0]["error"]["code"] == ERR_UNKNOWN_ID
        out = step_ok(env, act("get_document", doc_id=it["issue_id"]))
        assert out[0]["error"]["code"] == ERR_UNKNOWN_ID
    out = step_ok(env, act("list_records", table="issues"))
    assert out[0]["error"]["code"] == ERR_UNKNOWN_TABLE
    # echoing the agent's own junk is allowed, but nothing beyond what it sent
    for o in env.trajectory()[1:]:
        msg = o["observation"]["error"]["message"]
        assert "expected" not in msg and "amount" not in msg


# ------------------------------------------------------------------ remote parity
def _scripted_actions(ep):
    ids = ids_by_table(ep)
    inv = ep.invoices[0]
    return [
        act("list_records", table="ledger", limit=50),
        act("list_records", table="invoices", limit=5),
        act("list_records", table="payments"),
        act("list_records", table="fx", limit=10),
        act("get_record", table="ledger", id=ids["ledger"][0]),
        act("get_record", table="invoices", id=inv["invoice_id"]),
        act("get_record", table="fx", id=ids["fx"][0]),
        act("get_document", doc_id=inv["doc_id"]),
        act("get_document", doc_id=ep.payments[0]["doc_id"]),
        calc("round(1234.5678, 2) * 3"),
        calc("1/0"),
        act("list_records", table="ledger", limit=51),
        act("list_records", table="ledger", cursor="garbled"),
        act("get_record", table="ledger", id="nope"),
        act("get_record", table="nope", id="x"),
        {"action": "bogus", "args": {}},
        {"action": None},
        None,
    ]


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_remote_env_parity_with_in_process_env(mode, issue_config, make_remote):
    ep = generate_episode(SEED, issue_config)
    local = AuditEnv(mode=mode)
    remote = make_remote(mode)
    lo, li = local.reset(seed=SEED, config=issue_config)
    ro, ri = remote.reset(seed=SEED, config=issue_config)
    assert lo == ro and li == ri
    actions = _scripted_actions(ep)
    # page one table to the end through cursors on both sides
    cur = None
    while True:
        a = act("list_records", table="payments", cursor=cur, limit=7)
        l, r = local.step(a), remote.step(a)
        assert norm(l) == norm(r)
        cur = l[0]["result"]["next_cursor"]
        if cur is None:
            break
    for a in actions + [submit_truth(ep)]:
        l, r = local.step(a), remote.step(a)
        assert norm(l) == norm(r), a
        assert type(r[1]) is float and isinstance(r[2], bool) and isinstance(r[3], bool)
    assert l[2] is True
    # after the end both answer episode_over identically
    l, r = local.step(calc()), remote.step(calc())
    assert norm(l) == norm(r) and l[0]["error"]["code"] == ERR_EPISODE_OVER
    if mode == "eval":
        assert r[1] == 0.0 and r[4] == {}


def test_remote_env_random_trajectory_parity_and_truncation(issue_config, make_remote):
    ep = generate_episode(SEED + 1, issue_config)
    local = AuditEnv(mode="train", max_steps=30)
    remote = make_remote("train", max_steps=30)
    assert local.reset(seed=SEED + 1, config=issue_config) == remote.reset(seed=SEED + 1, config=issue_config)
    rng = random.Random(3)
    ids = ids_by_table(ep)
    ids["_docs"] = [r["doc_id"] for r in ep.invoices] + [r["doc_id"] for r in ep.payments]
    last = None
    for _ in range(40):
        a = random_action(rng, ids, ep.episode_id)
        l, r = local.step(a), remote.step(a)
        assert norm(l) == norm(r), a
        last = l
    assert last[3] is True                                              # both truncated at step 30 and stay over
    assert last[0]["error"]["code"] == ERR_EPISODE_OVER


def test_remote_before_reset_and_reset_isolation(make_remote):
    remote = make_remote("eval")
    o, r, term, trunc, info = remote.step(calc())
    assert o["error"]["code"] == ERR_NOT_RESET and r == 0.0 and not term and not trunc and info == {}
    o1, _ = remote.reset(seed=SEED)
    remote.step(calc())
    remote.step(empty_submission(o1["episode_id"]))
    assert remote.step(calc())[0]["error"]["code"] == ERR_EPISODE_OVER
    o2, _ = remote.reset(seed=SEED)
    assert o2 == o1
    assert remote.step(calc())[0]["step"] == 1


def test_remote_time_limit_flag(make_remote):
    remote = make_remote("train", time_limit_s=0.3)
    remote.reset(seed=SEED)
    time.sleep(0.6)
    o, r, term, trunc, info = remote.step(calc())
    assert trunc is True and not term


def test_remote_grader_report_gating(tmp_path, issue_config, make_remote):
    ep = generate_episode(SEED, issue_config)
    # not enabled -> the agent-side client cannot read the report
    denied = make_remote("eval")
    denied.reset(seed=SEED, config=issue_config)
    denied.step(submit_truth(ep))
    with pytest.raises(RemoteEnvError):
        denied.grader_report()
    # enabled: refused before the episode ends, available afterwards and equal to the in-process report
    log_dir = tmp_path / "logs"
    allowed = make_remote("eval", log_dir=str(log_dir), allow_grader_report=True)
    allowed.reset(seed=SEED, config=issue_config)
    allowed.step(calc())
    with pytest.raises(RemoteEnvError):
        allowed.grader_report()
    out = allowed.step(submit_truth(ep))
    assert out[1] == 0.0 and out[4] == {}
    rep = allowed.grader_report()
    local = AuditEnv(mode="eval")
    local.reset(seed=SEED, config=issue_config)
    local.step(calc())
    local.step(submit_truth(ep))
    assert json.loads(json.dumps(local.grader_report())) == rep
    assert rep["reward_shared"] == pytest.approx(1.0 - STEP_COST) and rep["metrics"]["tp"] == len(ep.truth["issues"])
    # the per-episode log went to --log-dir, with the real result
    logs = sorted(log_dir.glob("report_*.json"))
    assert len(logs) == 1
    logged = json.loads(logs[0].read_text())
    assert logged["reward_shared"] == rep["reward_shared"]


def test_remote_never_leaks_truth(issue_config, make_remote):
    ep = generate_episode(SEED, issue_config)
    remote = make_remote("train")
    first, _ = remote.reset(seed=SEED, config=issue_config)
    ids = ids_by_table(ep)
    ids["_docs"] = [r["doc_id"] for r in ep.invoices] + [r["doc_id"] for r in ep.payments]
    rng = random.Random(8)
    seen = [first]
    for _ in range(60):
        o = remote.step(random_action(rng, ids, ep.episode_id))
        seen.append(o)
    seen.append(remote.step(submit_truth(ep)))
    assert_no_leak(seen, ep)


def test_remote_survives_malformed_actions(make_remote):
    remote = make_remote("train")
    remote.reset(seed=SEED)
    weird = [{"action": "list_records", "args": {"table": "ledger", "limit": float("inf")}},
             {"action": "calculator", "args": {"expression": "9" * 100000}},
             {"action": "get_record", "args": {"table": "ledger", "id": "\ud800"}} if False else {"action": "x"},
             {"action": object()}]
    for a in weird:
        o, r, term, trunc, info = remote.step(a)
        assert o["error"] is not None and r == 0.0 and not term
    assert remote.step(calc())[0]["result"] == {"value": "2"}


def test_client_module_does_not_import_generator_or_truth(repo_root):
    code = ("import sys, auditgym.client\n"
            "bad = [m for m in sys.modules if m.startswith('auditgym.generator') or m in "
            "('auditgym.validate', 'auditgym.rules', 'auditgym.matcher', 'auditgym.env')]\n"
            "print(','.join(bad))\n")
    env = dict(os.environ, PYTHONPATH=repo_root, PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", code], cwd=repo_root, env=env, capture_output=True,
                         text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_server_survives_garbage_lines(repo_root):
    env = dict(os.environ, PYTHONPATH=repo_root, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.Popen([sys.executable, "-m", "auditgym.server", "--mode", "eval", "--insecure-open"],
                         cwd=repo_root, env=env,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        replies = []
        for line in ("not json", "[]", '"x"', '{"cmd": "nope"}', '{"cmd": "step", "action": {"action": "calculator", "args": {"expression": "1"}}}',
                     '{"cmd": "grader_report"}', '{"cmd": "reset", "seed": "bad"}', '{"cmd": "reset", "seed": 5}', '{"cmd": "close"}'):
            p.stdin.write(line + "\n"); p.stdin.flush()
            replies.append(json.loads(p.stdout.readline()))
        assert [r["ok"] for r in replies] == [False, False, False, False, True, False, False, True, True]
        assert replies[4]["obs"]["error"]["code"] == ERR_NOT_RESET
        assert "Traceback" not in json.dumps(replies)
        p.wait(timeout=10)
        assert p.returncode == 0
    finally:
        if p.poll() is None:
            p.kill()
