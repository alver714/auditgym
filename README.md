# AuditGym

**EN.** A reproducible RL environment for an AI financial auditor. One episode is one fictional
company and one monthly period. The agent reads records through a narrow tool interface and
submits findings with evidence. A deterministic grader scores them. Python 3.11, stdlib only,
fully offline.

**RU.** Воспроизводимая RL-среда для ИИ-аудитора. Один эпизод: вымышленная компания и один
месячный период. Агент читает записи через узкий набор инструментов и сдаёт находки с
доказательствами; детерминированный грейдер выставляет оценку. Python 3.11, только стандартная
библиотека, без сети.

## Quickstart / Быстрый старт

```bash
# Docker (offline run; the mount keeps the report, `make docker` does the same)
docker build -t auditgym . && mkdir -p reports \
  && docker run --rm --network none -v "$PWD/reports:/app/reports" auditgym

# Local (Python 3.11: use `python3.11`, or `python3` if that is a 3.11). Run everything from the repo root.
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m auditgym check
# optional: `.venv/bin/pip install -e .` makes `import auditgym` work from any directory
# (`python -m auditgym ...` from the repo root needs no install)
```

**Network and runtime.** Network is needed once, to install the test dependency (`pip install -r
requirements-dev.txt`, or the `docker build` step). After that `run`, `check`, `demo` and `generate` are fully
offline (`docker run --network none` is the recorded way). Expected runtime, measured on a laptop-class
CPU: about 80 s for `check --quick` and about 110 s for the full `check` (both include the pytest suite; the
Linux/Docker run, about 112 s, is recorded in `docs/docker_check_result.md`; slower machines scale roughly linearly). The `wall s` column of the
Results table is the scripted policy's own time per episode (tens of milliseconds), not the check's runtime and
not LLM latency.

The runtime is stdlib-only; `requirements-dev.txt` pins pytest and its transitive packages, and the Docker
base image has an exact patch tag. `pyproject.toml` pins 3.11.*; public output was also verified byte-identical on CPython 3.12 and 3.14.
Continuous integration (`.github/workflows/ci.yml`) runs `python -m auditgym check --quick` on every push.

`check` is the single official verification command (exit code 0 iff all hard checks pass).
`check` это единственная официальная команда проверки (код 0, если все жёсткие проверки пройдены).

## Commands / Команды

| Command | EN | RU |
|---|---|---|
| `python -m auditgym generate` | Write `episodes/{train,dev}/`, private `grader_data/`, and `trajectories/{trajectories,oracle_reference,samples}.jsonl` (see "Reproducibility and trajectory format") | Записать эпизоды, приватные данные грейдера и траектории |
| `python -m auditgym check [--quick] [--out reports/] [--update-docs]` | Tests, reproducibility, validation, schemas, baselines, paired checks, isolation and leak scan, reward properties, failure analysis; writes `reports/check_report.{json,md}` and per-episode grader logs in `reports/grader_logs/<policy>/`. `--update-docs` (full run only) regenerates the results blocks below and `docs/check_report.*` | Тесты, воспроизводимость, валидация, схемы, базовые политики, парные проверки, изоляция, скан утечек, свойства награды, анализ ошибок |
| `python -m auditgym run --policy P --seed N --config C [--mode eval\|train] [--pretty] [--deterministic] [--rename-salt N --rename-mode shape\|aggressive --permute-salt M] [--log-dir D]` | Run ONE policy (any of the 4 baselines, `careful_auditor` or `careful_auditor_degraded`) on ONE episode through the subprocess grader. Prints the grader report as JSON, or with `--pretty` the same readable step log as `demo`. `--deterministic` runs in-process with the wall clock frozen | Один эпизод одной политикой; `--pretty` даёт читаемый лог |
| `python -m auditgym demo [--policy P] [--contrast Q\|none] [--seed N --config C] [--index I]` | Default: `careful_auditor` on the first dev episode with a readable step log, then `reconcile` on the same episode for contrast (what the naive baseline misses). `--seed/--config` pick any episode (config name or path to a JSON file) | Сначала сильная политика, затем наивная для контраста |
| `python -m auditgym golden [--write]` | Compare (or rewrite) the committed cross-platform digests in `tests/golden_hashes.json` | Сверка эталонных хэшей |

Makefile shortcuts: `make check | quick | test | generate | demo | docker`.

## RL interface / RL-интерфейс

```python
from auditgym.env import AuditEnv
env = AuditEnv(mode="train")
obs, info = env.reset(seed=7, config="train")      # obs has task, manifest, policy
obs, r, terminated, truncated, info = env.step(
    {"action": "list_records", "args": {"table": "invoices", "limit": 50}})
```

Observation / наблюдение:

```json
{"episode_id": "ep_3f9a1c22b7d4", "step": 1, "steps_remaining": 99,
 "result": {"table": "invoices", "rows": [], "next_cursor": null, "total": 71}, "error": null}
```

Action schemas / схемы действий (`{"action": name, "args": {...}}`):

| Action | Args |
|---|---|
| `list_records` | `table` (ledger, invoices, payments, fx), `cursor` (opaque, default null), `limit` 1..50 |
| `get_record` | `table`, `id` (primary key; fx uses `fx:<date>:<source>:<target>`) |
| `get_document` | `doc_id` |
| `calculator` | `expression` (safe Decimal arithmetic) |
| `submit` | `findings` (object below), ends the episode |

Budget: 100 steps including submit. Truncation or timeout means empty findings are scored.
Бюджет: 100 шагов с учётом submit; при усечении или таймауте находки считаются пустыми.

The 60 s wall-clock limit runs from `reset` (monotonic clock) and therefore **includes the policy's own thinking time**
(a slow LLM agent can be truncated); the step budget is the deterministic limit, and replay-determinism
claims are for step-bounded runs. It is enforced two ways: cooperatively on every `step`, and, in the server
process, by a watchdog timer that truncates an unfinished episode (empty answer, report logged, `ended_by=time`)
even while the agent is silent, so a policy that simply stops calling `step` cannot hold an episode open
forever. The step that hits either limit counts as an inspection step.
Machine-readable JSON Schemas of every file, action and observation live in [`schemas/`](schemas/)
(`python -m auditgym.schema_check <episode_dir> [<grader_dir>]`).

### Trust boundary and deployment / Граница доверия

**Only the server path is an isolation boundary:** `auditgym/server.py` as a separate process, driven by
`client.RemoteAuditEnv`. The truth, the generator state and the control token live in that process; the policy
side only sees JSON observations. Everything in-process is **trusted-code only**: `AuditEnv` (it holds the truth as
Python attributes) and the `AgentHandle` returned by `RemoteAuditEnv.agent_handle()`. The handle is a read-only
object whose only callable is `step`; it exists to keep a well-behaved policy from calling `reset` or
`grader_report` by accident, but closures and attributes of in-process Python are introspectable, so it is hygiene,
not a sandbox. A policy that must be treated as hostile has to run in another process or container and talk to the
server only through `step`.

What the server enforces:

* `step` is the only agent command. `reset` (seed / config / options), `grader_report`, `abandon` and `close` are
  harness commands that the server refuses without the per-process random token (`AUDITGYM_CONTROL_TOKEN`), which
  only `RemoteAuditEnv` holds; with no token configured they are refused outright (`--insecure-open` exists for
  local experiments only). File paths (`episode_dir`, `grader_dir`) are never accepted over the wire.
* No request can stop the server: a malformed token (non-string, over-long, lone surrogate such as `"\ud800"`),
  invalid UTF-8, duplicate JSON keys and garbage lines all get a bounded `{"ok": false}` reply. Only an authorised
  `close` or end of input ends the loop.
* An unfinished episode is never silently dropped: a re-reset (`ended_by=abandoned`), an `abandon` by the harness
  after an agent failure (`agent_failed`), an authorised `close` (`closed`) or a lost connection (`disconnected`)
  scores it as an empty truncated answer and logs the grader report.
* Malformed `submit` payloads (missing/non-object `args`, extra keys, bad findings, absurd nesting, claimed amounts
  outside `0..10**18`, lone surrogates) end the episode with reward -1. Deeply nested or huge actions give a
  bounded `invalid_args` error; integers wider than 64 bits are replaced by a placeholder in logs and
  trajectories. Repeated ids in `affected_ids` / `evidence_ids` are accepted (they are compared as sets) and
  neither help nor hurt.
* `list_records` cursors are keyed integrity checks: base64 of `table:offset:tag`, where `tag` is an HMAC-SHA256
  keyed with a per-episode secret derived from grader-side material (the private provenance), so a cursor cannot be
  forged from public values; a foreign or edited cursor returns `invalid_cursor`.
* In `eval` mode the terminal reward returned to the policy is `0.0` and `info` is `{}`; the real result is kept
  grader-side (see "Reproducibility and trajectory format").

**Closed (jury) run: `python -m auditgym isolate`.** One command starts the three parties as separate processes:

```
grader server  <-- JSON lines + control token -->  harness  <-- observations / actions only -->  agent
(truth, token, grader reports)                     (this process)                                 (own uid, per episode)
```

```bash
# inside the image (runs as root, so the agent can get its own uid); --network none keeps everything offline
docker run --rm --network none -v "$PWD/secrets:/secrets:ro" -v "$PWD/my_agent:/agent:ro" auditgym \
  python -m auditgym isolate --agent "python3 /agent/agent.py" --episodes /secrets/hidden_seeds.json \
                             --log-dir /app/reports/isolated_logs
python -m auditgym isolate --policy careful_auditor --split dev      # a built-in policy as the isolated agent
python -m auditgym isolate --self-test                               # hostile probe agent; exit 1 on any leak
```

* **Agent protocol.** The agent reads `{"type":"observation","obs":..,"reward":..,"terminated":..,"truncated":..,
  "info":..}` lines on stdin and answers each with one `{"action": {...}}` line on stdout. The harness builds every
  server message itself, so a line can only ever become the `action` of a `step`: `{"cmd":"close"}`, a forged
  token or invalid bytes from the agent are answered as an invalid action of the same episode. An agent that exits,
  stays silent longer than `--step-timeout` or sends a line longer than 1 MB is killed; its episode is scored empty
  (`ended_by=agent_failed`) and logged.
* **Sandbox `uid` (default; needs root, e.g. the Docker image).** A fresh agent process per episode runs under a
  separate unprivileged uid/gid (default 65534, use a dedicated one) with no supplementary groups and
  `no_new_privs`, in its own session, in an empty temporary working directory, with an environment rebuilt from
  scratch (no token, no harness variables) and no inherited file descriptors. The grader log directory, the episodes
  file and `grader_data/` are made unreadable to it for the run (mode 0700, restored afterwards), and a preflight
  probe running **as the agent uid** must fail to read each of them, otherwise the run is refused. The server and
  harness run as another user, so the kernel denies the agent their `/proc/<pid>/environ`, `/proc/<pid>/mem` and
  ptrace. After each episode every process of the agent uid is killed and the workdir is deleted. `--no-network`
  additionally gives the agent an empty network namespace (refused if unavailable).
* **Fail closed.** Without root, `--sandbox auto` refuses to run; `--sandbox none` (process split and scrubbed
  environment only, the agent runs as the harness user) must be asked for explicitly and every summary records
  `isolation_boundary: false`. The self-test demonstrates the difference: under `uid` the probe finds nothing, under
  `none` it reads the control token from the grader's `/proc/<pid>/environ` and the grader files.
* **Self-test.** `isolate --self-test` runs `auditgym/agents/escape_probe.py`, an agent that hunts for the token (own
  environment, every `/proc/<pid>/environ`), opens the grader's `/proc/<pid>/mem`, reads and writes the grader paths
  it finds in the (world-readable) process command lines and next to its own script, looks for inherited file
  descriptors and for a seed in the observations, and sends server commands, invalid UTF-8 and duplicate keys over
  the action channel. It passes only if nothing leaks, the episode survives every hostile line and ends with the
  probe's own `submit`. `check` runs it as a hard check when it runs as root (otherwise it is a SKIP).
* Holdout truth is **never written to disk**: the server generates each episode from `(seed, config)` in memory, so
  the files worth protecting are the jury's episodes file and the grader logs. Results land in
  `<log-dir>/isolated_summary.json`, grader reports in `<log-dir>/report_*.json`, agent stderr in
  `<log-dir>/agent_stderr/`.

This is process-level isolation on one kernel, not a VM: a kernel exploit, a world-readable copy of private data
elsewhere on the host or an agent uid shared with other processes are out of scope. Splitting grader and agent into
separate containers on top of this is still possible (the agent then only needs the JSON-lines pipe).

## Findings schema / Схема находок

```json
{"episode_id": "ep_...",
 "findings": [{"type": "fx_error",
               "affected_ids": ["INV-2026-0042"],
               "claimed_amount_minor": 12345,
               "currency": "KZT",
               "evidence_ids": ["D-1a2b3c4d5e", "R-9f8e7d6c", "fx:2026-03-12:EUR:KZT"],
               "explanation": "free text, never affects the score"}]}
```

Types: `duplicate_invoice`, `payment_misallocation`, `fx_error`, `period_cutoff`.
Missing, extra or wrongly typed fields make the submission invalid (reward -1). A finding
matches an issue iff type, affected id set and currency are equal, the amount is within
`max(1, expected // 1000)` minor units, every evidence id exists, and one full evidence set is
included. Matching is 1-1 (maximum bipartite matching).

## Reward / Награда

```
shared_reward = clip(2 * F1 - 1 - 0.002 * inspection_steps, -1, 1)
F1 = 2TP / (2TP + FP + FN);  empty findings on a clean book -> F1 = 1
P = TP/(TP+FP), R = TP/(TP+FN), zero denominators -> 0 (so a silent agent on a clean book has P = R = 0, F1 = 1)
```

Invalid submission: -1. Eval mode returns 0 and an empty info to the policy; the grader keeps the
real result. An experimental soft-F1 reward is reported separately and never replaces the shared one.

## Reproducibility and trajectory format / Воспроизводимость и формат траекторий

* **Eval mode hides the reward from the policy.** In `mode="eval"` the terminal `reward` returned by `step`
  (and stored in every trajectory row) is `0.0` and `info` is `{}` by design. The real result lives only on the
  grader side: **paired and regression checks must read `grader_report["reward_shared"]`** (and
  `grader_report["metrics"]`), never the trajectory reward. In `mode="train"` the two are equal. / В eval-режиме
  награда в траектории скрыта (0.0); сравнивайте `grader_report.reward_shared`.
* **Version stamp.** Every `grader_report` returned by `runner.run_episode` carries `env_version`, `config_id`,
  `config_version`, `config_hash` (sha256 of the RESOLVED config including its `force` block),
  `time_truncated` and `deterministic`. The same fields head every shipped trajectory and `check_report.json`
  (`config_ids`, `config_versions`, `config_hashes`, `config_file_sha256`; `check` also fails if
  `configs/CONFIG_VERSIONS.json` exists and disagrees with a config file).
* **Trajectory rows** (`trajectories/*.jsonl`, schema `schemas/trajectory_row.schema.json`): one JSON object per
  env step with `episode_id, step, action, observation, reward, terminated, truncated, policy` (`policy` = the name
  of the policy that produced the row). Step 0 is the reset observation (`action: null`, reward 0). The step-0 row also carries `header`: `env_version, config_id, config_version,
  config_hash, mode, policy, seed, split, steps, terminated, truncated, ended_by, time_truncated, deterministic,
  reward_visible_to_policy, grader_reward_shared, grader_f1, trajectory_sha256` (+ `label`). In an eval-mode run
  `reward_visible_to_policy` is false and `header.grader_reward_shared` is the real reward.
* **Which file is what.** `trajectories.jsonl`: **train-mode** runs of the non-oracle policies on the 20 open
  episodes, so the terminal row carries the real shared reward (an eval-mode trajectory would show 0.0).
  `oracle_reference.jsonl`: the grader-side oracle (it reads the truth, submits in one step; a **reward-ceiling
  demo, not agent behaviour**: do not train on it or read it as a policy), every run labelled. `samples.jsonl`: one eval-mode `careful_auditor` run, one scripted run truncated by the
  100-step budget (`truncated=true`, `time_truncated=false`) and one truncated by the wall clock
  (`time_truncated=true`, produced with a fake clock so it is reproducible).
* **Wall-clock truncation and deterministic mode.** The only non-deterministic input of an episode is the wall
  clock (a run is truncated after 60 s). `time_truncated` records whether it happened; `run_episode(...,
  deterministic=True)` / `run --deterministic` freeze the clock (in-process env only) so a replay can never differ
  because of it.
* **Cross-platform determinism.** `tests/golden_hashes.json` holds digests (episode id, public hash, truth and
  provenance sha256, sha256 of the `reconcile` and `careful_auditor` trajectories) for six episodes (train, dev,
  holdout). `test_golden_hashes_match_this_machine` and `check` (step 2) compare them on whatever OS/CPU/Python
  runs the suite; `python -m auditgym golden --write` regenerates them after an intended change. A recorded
  Docker/Linux run of the full `check` (digests committed from macOS match) is in `docs/docker_check_result.md`.
* **Id-format independence.** `check` pairs every holdout episode with two transformed twins: `shape`
  (ids keep prefix/length/charset) and `aggressive` (every opaque id replaced by a numeric-only or `z`+hex id of
  different length and sort order, e.g. `INV-2026-0042` -> `70419283356` / `z5d314310cf0`). The rename is a pure
  relabelling (record order untouched), so it commutes with the order shuffle. `fx:<date>:<src>:<dst>` keys are
  natural keys derived from the row's own columns and are intentionally kept. Aggressive ids deliberately
  violate the id patterns of `schemas/*.json`, which describe the generator's formats.
* **Paired checks are only as strong as the policy.** `random_explorer` and `flag_all` sit at the -1.0 floor
  on almost every episode, so "same score after transform" would be trivially true for them; the discriminating
  pairs are `reconcile` (above the floor on 14 of the 20 holdout pairs) and `careful_auditor`. Per-pair F1 is
  reported in `check_report.json` (`sections.paired.per_pair_f1`). The oracle is additionally cross-checked
  against the committed `grader_data/*/truth.json`, not only against a regeneration.

## Repo layout / Структура

```
auditgym/   constants, money, episode, rng, calculator, transforms, env, server, client,
            matcher, rules, validate, runner, check, __main__
  generator/  config, world, distractors, injectors, build
  policies/   random_explorer, reconcile, flag_all, oracle (grader only), careful_auditor and
              careful_auditor_degraded (extra diagnostics; the degraded one is the near-miss variant)
  schema_check.py   stdlib JSON-Schema checker for schemas/
configs/    train / dev / holdout / splits (versioned)
schemas/    JSON Schemas: manifest, policy, ledger row, invoice, payment, fx row, truth, provenance,
            findings, action, observation, trajectory_row
docs/       DESIGN.md, CFA_GROUNDING.md, CHALLENGE.md, check_report.{md,json} (last full run)
CHANGELOG.md  config / generator history (config 1.0.0 to the shipped version)
.github/workflows/ci.yml   minimal CI: `python -m auditgym check --quick`
tests/      pytest suite; golden_hashes.json = committed cross-platform digests
episodes/ grader_data/ trajectories/   committed open splits (10 train + 10 dev), their truth, and
            trajectories.jsonl (random_explorer, reconcile, flag_all, careful_auditor, careful_auditor_degraded
            on those 20 episodes, train mode), oracle_reference.jsonl (the grader-side oracle, kept apart and
            labelled), samples.jsonl (an eval-mode run and two truncated=true runs)
```

`grader_data/{train,dev}` is the truth of the OPEN splits and is shipped on purpose (the official check and
the Docker image read it). Holdout truth is never written to disk: `check` regenerates it in memory.

## Deliverables checklist (hackathon section 09) / Чек-лист

Status against the hackathon brief (each item is covered by a `check` step or a test).

- [x] Environment with reset/step interface and documented schemas (this README, `schemas/`, SPEC)
- [x] Reproducible generation, same seed gives identical output (`check`, step 2)
- [x] Hidden ground truth with provenance, validated by an independent rules engine
- [x] Baseline policies and measured results (see Results)
- [x] Anti-gaming measures and leak scan (docs/DESIGN.md)
- [x] One-command verification, offline, Docker (`docker run --network none`)
- [x] Domain grounding (docs/CFA_GROUNDING.md)
- [x] Honest limitations (below)

## Results / Результаты

Measured by `python -m auditgym check` on the 20-episode holdout (5 clean, 5 interaction, 10 mixed), eval
mode, remote grader. The `careful_auditor` row is an extra diagnostic policy, not one of the 4 required baselines.

> **Caveat: these are NOT hidden-seed results.** The holdout seeds and `configs/holdout.json` are public in this
> repository (`configs/splits.json`), so anyone can regenerate the holdout episodes and their truth. The table
> shows that the harness works and how the shipped policies behave on a fixed, reproducible set; it is not a
> secret-test score. A jury evaluation must use its **own hidden seeds** (a large, private seed range) and, ideally,
> new parameter combinations in its own config files; `python -m auditgym run --seed N --config C` and
> `check`'s paired transforms work unchanged for them. Small sample (20 episodes): intervals are wide.

<!-- RESULTS:BEGIN (generated by `python -m auditgym check --update-docs`; do not edit) -->

Generated by the latest **full** `python -m auditgym check` run: 74 PASS, 0 FAIL, 1 WARN, 0 SKIP; pytest: 1070 passed. Full per-check table: [docs/check_report.md](docs/check_report.md).


#### Baseline policies (eval mode, holdout, remote env)

Shared reward, F1, experimental reward, steps and wall time are means over episodes (95% percentile bootstrap CI, 2000 resamples, fixed seed). Precision and recall are POOLED over the episodes (sum TP / (sum TP + sum FP or FN)), because a silent agent on a clean book has P = R = 0 (0/0) while F1 = 1.

| policy | n | precision (pooled) | recall (pooled) | F1 [95% CI] | shared reward [95% CI] | experimental | steps | wall s |
|---|---|---|---|---|---|---|---|---|
| random_explorer | 20 | 0.000 | 0.000 | +0.100 [+0.000, +0.250] | -0.806 [-1.000, -0.514] | -0.806 | 29.0 | 0.11 |
| reconcile | 20 | 0.066 | 0.479 | +0.109 [+0.063, +0.155] | -0.807 [-0.891, -0.721] | -0.791 | 20.4 | 0.12 |
| flag_all | 20 | 0.003 | 0.028 | +0.006 [+0.000, +0.014] | -0.990 [-1.000, -0.974] | -0.984 | 6.0 | 0.10 |
| oracle | 20 | 1.000 | 1.000 | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 | 1.0 | 0.10 |
| careful_auditor (extra) | 20 | 1.000 | 1.000 | +1.000 [+1.000, +1.000] | +0.949 [+0.946, +0.953] | +0.949 | 26.4 | 0.12 |
| careful_auditor_degraded (extra) | 20 | 0.570 | 0.634 | +0.531 [+0.390, +0.663] | +0.030 [-0.246, +0.290] | +0.030 | 20.9 | 0.12 |

`oracle` reads the truth grader-side and only demonstrates the reward ceiling; it is not agent ability.

`careful_auditor (extra)` is an additional diagnostic row, not one of the 4 required baselines: an honest observation-only scripted agent (no truth, no generator, no rules engine) showing the environment is solvable from public evidence within the 100-step budget.

`careful_auditor_degraded (extra)` is the same agent with a 20-inspection-step cap and period membership decided on the wall-clock fields of `issued_at` (the time-zone rule ignored). It is the near-miss row: the reward and F1 gap to `careful_auditor` is measured, not asserted, and the failure analysis below lists which issue types it misses.

#### Failure analysis (measured on the same eval runs; from the grader reports)

Recall by issue type = matched truth issues / truth issues of that type (all episodes of the run).

| policy | duplicate_invoice | payment_misallocation | fx_error | period_cutoff | issues born of the 2 interaction scenarios |
|---|---|---|---|---|---|
| random_explorer | 0/13 | 0/15 | 0/26 | 0/17 | 0/20 |
| reconcile | 13/13 | 1/15 | 13/26 | 7/17 | 2/20 |
| flag_all | 0/13 | 0/15 | 0/26 | 2/17 | 0/20 |
| oracle | 13/13 | 15/15 | 26/26 | 17/17 | 20/20 |
| careful_auditor | 13/13 | 15/15 | 26/26 | 17/17 | 20/20 |
| careful_auditor_degraded | 3/13 | 14/15 | 21/26 | 7/17 | 12/20 |

Findings by type: true positives / findings submitted (the rest are false positives).

| policy | duplicate_invoice | payment_misallocation | fx_error | period_cutoff | FP total |
|---|---|---|---|---|---|
| random_explorer | 0/10 | 0/2 | 0/11 | 0/0 | 23 |
| reconcile | 13/69 | 1/148 | 13/261 | 7/34 | 478 |
| flag_all | 0/100 | 0/0 | 0/400 | 2/100 | 598 |
| oracle | 13/13 | 15/15 | 26/26 | 17/17 | 0 |
| careful_auditor | 13/13 | 15/15 | 26/26 | 17/17 | 0 |
| careful_auditor_degraded | 3/3 | 14/15 | 21/21 | 7/40 | 34 |

Why the false positives failed to match (matcher reason = furthest stage reached against any issue; `ids_mismatch` = no truth issue has this type + affected_ids, i.e. a plain false accusation), and which distractor kind they landed on:

| policy | FP reasons | FP by distractor kind |
|---|---|---|
| random_explorer | ids_mismatch: 13, unmatched: 6, wrong_type: 4 | (not a distractor): 18, realized_fx_gain_loss: 2, split_payment: 3 |
| reconcile | amount_out_of_tolerance: 12, evidence_insufficient: 1, ids_mismatch: 285, unmatched: 110, wrong_type: 70 | (not a distractor): 294, legit_recurring_amount: 58, overpayment_refund: 32, realized_fx_gain_loss: 21, split_payment: 20, timezone_boundary_ok: 27, unreferenced_prepayment: 26 |
| flag_all | amount_out_of_tolerance: 5, ids_mismatch: 353, unmatched: 150, wrong_type: 90 | (not a distractor): 413, legit_recurring_amount: 12, overpayment_refund: 55, realized_fx_gain_loss: 34, split_payment: 49, timezone_boundary_ok: 19, unreferenced_prepayment: 16 |
| oracle | - | - |
| careful_auditor | - | - |
| careful_auditor_degraded | ids_mismatch: 19, unmatched: 9, wrong_type: 6 | (not a distractor): 13, split_payment: 1, timezone_boundary_ok: 20 |

Near-miss analysis: what each policy misses (false negatives) and wrongly claims (false positives) per issue type, and what its inspection steps cost. `careful_auditor_degraded` is `careful_auditor` with a 20-step cap and period membership decided on wall-clock fields instead of instants; the gap to `careful_auditor` is the price of those two mistakes. Step cost = mean inspection steps x 0.002 (it is the only reward term a perfect answer can still lose).

| policy | missed (FN) per type: duplicate_invoice / payment_misallocation / fx_error / period_cutoff | FP per type | mean steps | step cost | mean shared reward |
|---|---|---|---|---|---|
| random_explorer | 13 / 15 / 26 / 17 | 10 / 2 / 11 / 0 | 28.0 | 0.056 | -0.806 |
| reconcile | 0 / 14 / 13 / 10 | 56 / 147 / 248 / 27 | 19.4 | 0.039 | -0.807 |
| flag_all | 13 / 15 / 26 / 15 | 100 / 0 / 400 / 98 | 5.0 | 0.010 | -0.990 |
| oracle | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0.0 | 0.000 | +1.000 |
| careful_auditor | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 25.4 | 0.051 | +0.949 |
| careful_auditor_degraded | 10 / 1 / 5 / 10 | 0 / 1 / 0 / 33 | 19.9 | 0.040 | +0.030 |

#### Paired invariance pairs (per-pair F1 of the untransformed run)

A pair only proves something when the policy is above the -1.0 floor, so the discriminating policies are `reconcile` and `careful_auditor`; `random_explorer` and `flag_all` are excluded (trivially invariant at the floor). Each cell is the F1 of one holdout episode (order: clean, interaction, mixed, as in `splits.json`); every listed pair gave identical metrics, reward and steps under the shape and the aggressive rename + permute.

| policy | pairs above floor | per-pair F1 |
|---|---|---|
| reconcile | 13/20 | 0.00 0.00 0.00 0.00 0.00 0.00 0.22 0.00 0.06 0.05 0.14 0.14 0.22 0.21 0.36 0.06 0.20 0.12 0.16 0.24 |
| careful_auditor | 20/20 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 |

<!-- RESULTS:END -->

### How to read the failure analysis / Как читать анализ ошибок

The tables above are measured by `check`, not written by hand, and the holdout is deterministic (only wall times
vary). The numbers below are quoted from the run shown in the generated block; if you re-run `check
--update-docs` the block is authoritative.

* **reconcile** (the "simple reconciliation" baseline) finds every duplicate invoice (13/13) but only 1/15
  payment misallocations, 13/26 FX errors, 7/17 period cut-offs and 2/20 of the issues born of the two interaction
  scenarios. Its precision is the real failure: 478 false positives over 20 episodes. Where the matcher can say why,
  most are plain accusations of records that are not issues at all (`ids_mismatch`), and the rest land on exactly the
  distractors the generator planted: legitimate recurring billing (from the "same entity and amount within 10 days"
  rule), unreferenced prepayments, overpayments with refunds, realised FX gain/loss, split payments and
  timezone-boundary invoices (from comparing date strings and ignoring offsets). The date-string shortcuts are also
  why it misses tricky-offset cut-offs.
* **random_explorer** and **flag_all** hit almost nothing (0 and 2 true positives): guessed ids fail
  `ids_mismatch`/`wrong_type` before amounts or evidence are even looked at. flag_all submits the maximum number of
  findings and ends near -1.0; random_explorer's mean F1 of 0.100 comes only from clean books on which it happened
  to stay silent (F1 = 1 there).
* **The one `check` WARN is the baseline ordering** (oracle > reconcile > random_explorer, flag_all): on the
  holdout the mean shared reward of `reconcile` (-0.807) is not above `random_explorer` (-0.806). Both sit near the
  floor for opposite reasons: reconcile finds many issues but drowns them in false positives, random_explorer finds
  none but stays silent on some clean books. It is reported as a warning, not hidden; F1 and the failure tables
  separate them.
* **careful_auditor** (extra, observation only) finds all 71 issues with 0 false positives, including every
  interaction issue and the clean books, in about 26 steps. It shows the environment is solvable within the
  100-step budget; see Limitations for what that implies about difficulty.
* **careful_auditor_degraded** (extra) is the **failure analysis of a near-miss agent**: the same agent with two
  deliberate defects, a 20-inspection-step cap and period membership decided on the wall-clock fields of `issued_at`
  (the time-zone rule ignored). It reaches F1 0.531 and +0.030 shared reward against 1.000 / +0.949 for the intact
  agent, and its errors have a clean per-type signature:
  * **Step cap**: it cannot finish paging plus line-item fetches, so it misses 10 of 13 duplicate invoices
    (3/13 found); this is a budget failure, not a reasoning one.
  * **Ignoring offsets**: it misses 10 of 17 period cut-offs and 5 of 26 FX errors, and it accuses 33 invoices that are in-period as instants; 20 of those land exactly on the
    `timezone_boundary_ok` distractors, i.e. the distractor design catches the shortcut.
  * **Untouched**: payment misallocation (14/15) does not depend on either defect.
  * Its inspection steps are cheaper (0.040 vs 0.051 of reward), but the reward lost to misses and false positives
    is about 0.9. The gap is measured per type in the failure tables, not asserted.
* Per-type recall of the naive policy is lowest exactly where the policy text needs offset-aware dates and
  evidence assembly (misallocation, interactions), which is the intended headroom for real agents.

`reports/` is a **snapshot of the last run** on whichever machine ran `check` last (it is git-ignored and
regenerated by every `check`); the committed, authoritative copies are `docs/check_report.{md,json}` and the
generated blocks in this README and `docs/DESIGN.md`, refreshed only by a full `check --update-docs`.

Raw material: `reports/grader_logs/<policy>/report_*.json` (one grader report per run, with the matcher reason
of every finding) is written by every `check`; `trajectories/trajectories.jsonl` holds a full competent
trajectory (careful_auditor) next to the four baselines on the 20 open episodes.

## Limitations / Ограничения

* Simplified fictional accounting policy, not IFRS or an audit standard. This is a programming
  task, not an audit opinion. / Упрощённая вымышленная политика; это задача по программированию,
  а не аудиторское заключение.
* Four issue types, monthly scope, no opening balances, fixed-offset time zones only.
* **Saturation.** `careful_auditor` (a scripted, observation-only policy that reads `policy.json`) reaches
  F1 = 1.000 on the holdout with about 26 steps. The environment is therefore *solvable* from public evidence,
  but it is **not hard for a scripted solver that knows the policy**: the benchmark's headroom is for agents
  that must discover the rules, not for a programmed one. The reward separates careful work from naive
  heuristics by about 1.7 shared-return points.
* **Period errors have one direction.** `period_cutoff` is always "invoice issued after period end, posted
  inside the period". There is no prior-period invoice, no in-period invoice missing from the ledger and no
  late posting variant (records outside the period never appear in the ledger by construction).
* Foreign-currency `overpayment_refund` distractors leave a small base-currency residual in
  `2300_unapplied_cash` (refund booked at the refund-date rate, excess at the payment-date rate, no
  `7900_fx_gain_loss` row). This is policy-conformant and stated in the policy text; it is not an error.
* The wall-clock limit includes policy latency (see above). The step budget is the deterministic limit.
* `episode_id` is content-addressed (sha256 of the public records without the id field), so it adds no
  information beyond the records. Threat model: the generator is open source and the configs are public, so
  secrecy of a hidden seed relies on the seed space being large/private and on the agent having no code execution
  (tools only). Anyone with the generator and the seed can regenerate the truth, as with any seeded benchmark.
* Forced issue counts are capped at 12 in total; per-type capacity is handled by the generator (it creates as
  many post-period invoices as the forced cutoffs need). Company sizes outside 14-105 invoices, 4-60
  customers or more than 1000 noise postings raise `ConfigError`; the ledger itself is capped at 1900 rows and
  the effective noise count is recorded in the (private) provenance spec as `n_noise_effective`.
* Order/id invariance (paired shape + aggressive rename + permute) is checked for the oracle, `reconcile` and
  `careful_auditor`; `random_explorer` and `flag_all` read only the first pages / follow row order by design and
  are not invariant (and sit at the -1.0 floor, so a pair would prove nothing); `careful_auditor_degraded` is
  order-dependent on purpose (its step cap truncates the ledger it sees).
* Small holdout (20 episodes), so confidence intervals are wide.
* Robustness is tested against the shipped baselines and a set of scripted attacks (reset oracle, forged
  truth paths, deep-nesting payloads, malformed submits), not against adaptive learning attackers.

### Known limitations (won't-fix, stated honestly)

These items were raised in review and deliberately not changed. Each says why.

* **Open dev/train truth and provenance files are world-readable and sit in the same repo tree as agent code.** The
  brief makes dev/train open on purpose and ships `grader_data` for those splits. Holdout truth is generated on the
  fly and never stored. OS-level separation (a separate user or container for the grader, agent code without read
  access to `grader_data/`) is enforced by `python -m auditgym isolate` (README "Trust boundary and deployment"):
  under its default sandbox the agent's uid cannot read `grader_data/` for the duration of the run. Plain `run` /
  `check` keep the single-trust-domain layout.
* **In-process policies share the harness interpreter.** A hostile policy run in the same Python process can reach
  private attributes of the harness. A hostile agent belongs in `python -m auditgym isolate` (separate process and
  uid, step-only pipe); the in-process path is for trusted code. The handle is hardened
  cheaply, and the assumption is stated: in-process policies are trusted code.
* **No wall-clock watchdog for blocked in-process `policy.act`.** Preempting arbitrary in-process Python without
  threads or signals is not reliable. A transport-level deadline is enforced in the server/client path where
  feasible; in the in-process path the time limit is cooperative.
* **Generated issues are single-id; no same-type, same-ids pairs.** The brief does not require multi-id issues. Each
  issue carries its canonical maximal evidence set plus the minimal set(s) where a smaller proof exists (for
  `payment_misallocation` without a posted AR credit, one minimal set per candidate row). Every set must cite all
  ledger rows the posted amount is summed from: `rules.check_evidence` rejects a citation that leaves one out
  (it would otherwise recompute a different amount). The remaining matcher paths are covered by synthetic matcher
  tests, not generated data.
* **Truncation on a clean book scores about +0.8, close to an honest empty submit.** This is what the brief mandates
  (truncation is scored as an empty answer: `2F1 - 1 - 0.002*steps`). Behaviour kept; a test pins it.
* **Fixed-offset time zones only (no DST); company names can collide.** DST needs a timezone database for little
  gain, and truth correctness is unaffected. Name collisions are likewise harmless for truth.
* **No hash-pinned requirements, no mypy/strict lint.** Optional hygiene with no bearing on the brief's criteria;
  only minimal CI and an optional ruff config are included.
* **The oracle is grader-side regeneration, so the oracle paired check is partly self-consistent.** That is inherent to
  what an oracle is. It is cross-checked against the committed `grader_data` `truth.json` and by the non-oracle
  `careful_auditor` / `reconcile` invariance checks. Documented, not redesigned.
* **Ordering warning in `check`.** The expected ordering oracle > reconcile > random_explorer, flag_all is reported as a
  WARN, not a failure: `random_explorer` (-0.806) and `reconcile` (-0.807) are statistically tied near the floor on
  this holdout.

License: MIT.
