# Changelog

Two version numbers matter and move independently:

* `ENV_VERSION` (`auditgym-1.0.0`, `auditgym/constants.py`, frozen contract): the env / matcher / reward semantics.
  It has not changed; none of the entries below alters the reward, the matching rule or the action interface.
* `config_version` of each file in `configs/` (`train`, `dev`, `holdout`): bumped whenever a config's *resolved
  content* changes. `configs/CONFIG_VERSIONS.json` binds `config_id@config_version` to the sha256 of the resolved
  config, and `python -m auditgym check` fails if a config changes without a bump (`python -m auditgym lock --write`
  regenerates the lock).

Generator output is a function of `(seed, config, ENV_VERSION)`; a generator change that alters output therefore
comes with a config bump and a regenerated `episodes/`, `grader_data/`, `trajectories/` and `tests/golden_hashes.json`.

## Harness and grader hardening (no ENV_VERSION or config change)

Nothing below alters generated data, the matcher, the reward or the action interface.

* Server: a malformed control token (lone surrogate such as `"\ud800"`, non-string, over-long) is refused like a wrong
  one instead of crashing the process with `UnicodeEncodeError`; invalid UTF-8 on stdin and any unexpected exception
  in a handler get a bounded `{"ok": false}` reply. Only an authorised `close` or end of input ends the loop.
* Server: `close` is a harness command (token required), and so is the new `abandon` (the harness scores a running
  episode empty after an agent failure). An episode still running when the server stops is scored as an empty
  truncated answer and its grader report is logged (`ended_by` = `closed` / `disconnected` / `agent_failed`).
* New `python -m auditgym isolate`: grader, harness and agent as separate processes, the agent under its own uid with
  a scrubbed environment, an empty workdir and the grader paths made unreadable (verified by a preflight probe as the
  agent uid, fail closed); `--self-test` runs a hostile escape-probe agent. `check` runs the self-test as a hard check
  when it runs as root.
* `check`: a missing pytest or an empty `tests/` is a FAIL, not a SKIP; `--allow-skip-tests` is the explicit opt-out
  (and never updates the docs).
* `rules.check_evidence` (the independent verifier used by `validate` and the tests, not by the scorer) requires
  every ledger row the posted amount is summed from; a citation that omits one is rejected instead of yielding a
  different amount. All generated evidence sets already satisfy this.
* `isolate`, second review: writes to the agent are non-blocking with a deadline (an agent that stops reading can no
  longer block the harness) and every episode has a hard wall-clock cap (`--episode-timeout`); `protect` leaves
  already-private paths untouched, refuses a readable path it cannot chmod (read-only mount) with a hint, and always
  restores modes it changed; `--seed` (visible in `/proc/<pid>/cmdline`) needs `--public-seeds`, private seeds go
  through `--episodes FILE` or `--episodes -`; the escape probe flags seeds on process command lines. The preflight
  treats a directory as reachable if the agent can list it OR traverse it (a `0711` directory hides its listing but
  not a known file inside), so `protect` closes such directories too. New CI job
  `isolation-docker` runs `scripts/docker_isolation_check.sh` (isolation tests, self-test and the README commands)
  as root inside the Python 3.11 image with `--network none`.
* New `python -m auditgym demo --story`: defense walk-through (clean book, interacting errors, cited evidence, score,
  false accusations, truth never in observations, identical rerun); exit code 1 if the last two checks fail.
* README: `period_cutoff` described as two-directional (issued after `period_end` or before `period_start`, posted
  inside the period), matching DESIGN, SPEC and the generator; commands table lists `isolate`, `demo --story` and
  `check --allow-skip-tests`. `docs/docker_check_result.md` records a full `check` in the image on Linux/x86-64
  (golden digests match; previously only arm64 was recorded).
* README / DESIGN: the closed-run recipe is now the `isolate` command; removed the outdated statements that duplicate
  JSON keys are accepted (the server rejects them) and that every issue has exactly one evidence set.

## Config 1.3.0 (current; `train`, `dev`, `holdout`)

Cumulative summary of the generator and config changes since 1.0.0. Only 1.3.0 is shipped and locked; the
intermediate 1.1.0 and 1.2.0 states were not kept as separate releases, so the changes are listed together.

Generator and truth
* `period_cutoff` is two-directional: besides "issued after `period_end`, posted inside the period" there is now the
  mirror case "issued before `period_start`, posted inside the period", with the exact boundary cases
  `period_end + 1 s` and `period_start - 1 s` and tricky-offset invoices whose date string still looks in-period.
* `payment_misallocation` evidence sets are canonical: one maximal set (payment document, all ledger rows of the
  payment, referenced invoice documents and fx rows for foreign currency) and the minimal set(s) the independent
  verifier accepts (one per candidate row when no AR credit was posted). The matcher accepts exactly the subsets
  `rules.check_evidence` accepts.
* Distractors and density: distractor counts and company density are drawn so the clean book always passes the
  rules engine with zero violations; the effective noise count is recorded in the private provenance spec
  (`n_noise_effective`) and the ledger is capped at 1900 rows.
* Every injected issue is checked against the independent rules engine (`rules.py`) and its evidence sets; an
  injection that cannot produce a positive, verifiable amount is retried or rejected (`InjectionError`).

Config schema and validation
* Strict validation (`ConfigError`): unknown top-level, `company` or `force` keys are rejected; `config_id` and
  `config_version` are required `X.Y.Z` strings and never defaulted; probabilities must lie in [0, 1]; company sizes
  outside 14-105 invoices, 4-60 customers or more than 1000 noise postings, and more than 12 forced issues in total,
  are rejected; at least one customer offset must differ from the company offset by 1 h or more.
* `force` is documented: listed issue/interaction keys are pinned and unlisted ones are 0; `clean_probability` is not
  applied to a forced episode.
* `schemas/config.schema.json` describes the shape; the public manifest carries no `config_id`/`config_version`/
  `config_hash` (they stay in `provenance.json`, grader reports and trajectory headers).

Identifiers and reproducibility
* `episode_id` is content-addressed (`"ep_" + sha256(public content without the id)[:12]`), so two configs never alias
  on one id.
* `rename_ids` (modes `shape` and `aggressive`) and `permute` commute and are covered by the paired checks;
  `fx:<date>:<src>:<dst>` natural keys are intentionally kept.

## Config 1.0.0

Initial release: four issue types (`duplicate_invoice`, `payment_misallocation`, `fx_error`, `period_cutoff`),
two interaction scenarios, six distractor kinds, clean companies, fixed-offset time zones, 10 train + 10 dev open
episodes and a 20-episode holdout.

## Non-config changes (same `ENV_VERSION`)

* Server and client hardening: harness commands (`reset`, `grader_report`) need a per-process token; bounded and
  validated actions; a watchdog enforces the 60 s limit while the agent is silent; HMAC-keyed cursors; huge
  integers and lone surrogates are handled; the `AgentHandle` exposes `step` only (hygiene, not a sandbox).
* Runner and CLI: `run`, `demo`, `golden`, `lock` commands, deterministic replay, version stamps
  (`env_version`, `config_id`, `config_version`, `config_hash`) in every grader report, trajectory header and
  `check_report.json`, per-episode grader logs, `check --update-docs`.
* Policies: extra `careful_auditor` and `careful_auditor_degraded` diagnostic policies and a measured failure analysis.
* Repo: pinned dev requirements, exact-patch Docker base image, a minimal CI workflow
  (`.github/workflows/ci.yml`, `check --quick`), `CHANGELOG.md`.
