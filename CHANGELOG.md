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
