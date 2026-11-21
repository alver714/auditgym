# AuditGym design

## Architecture

```
 configs/*.json ──► generator/config ──► EpisodeSpec (sampled with sub_rng(seed,"spec"))
                                              │
   hidden history (world.py) ─► distractors ─► clean postings ─► snapshot (clean_records)
                                                                     │
                                          injectors (4 issues + 2 interactions)
                                          emit: truth issues, evidence sets, provenance
                                                                     │
                                                      Episode (public files | private truth)
                                                                     │
   ┌────────────── agent process ──────────────┐     ┌──────── grader subprocess ────────┐
   │ policy ── JSON action ──► client.py ──────┼────►│ server.py ─► env.py (AuditEnv)     │
   │        ◄── observation ◄──────────────────┼─────│   tools, budget, matcher.py,       │
   └────────────────────────────────────────────┘     │   truth held here only            │
                                                      └───────────────────────────────────┘
 rules.py + validate.py: independent perfect checker, used offline to validate data and truth
```

## Episode lifecycle

1. `reset(seed, config)`: generate the episode, return the first observation (task text,
   manifest, policy).
2. Up to 100 steps of `list_records`, `get_record`, `get_document`, `calculator`. Each step
   costs 0.002 in the shared reward. Errors are bounded and consume a step.
3. `submit(findings)` ends the episode. The matcher does 1-1 matching and scores F1.
4. Budget exhausted or time limit reached: truncated, findings treated as empty.

## Data construction

Hidden history, then clean postings, then distractors, then injection. Customers, invoices,
payments and an FX random walk are built first. Distractors (split payments, overpayment and
refund, recurring amounts, timezone-boundary invoices, realised FX gain/loss, prepayments)
are applied in the clean world, so the clean books pass the rules engine with zero
violations. Injectors then mutate records. Each injector records the causal change, the truth
issue and its evidence sets in `provenance.json`. Because the amounts come from the final
records, every truth issue is independently verifiable by `rules.py`. IDs are drawn from the
same pools as normal ones and row ids are random hex, so injected records are not
distinguishable by format.

## Why each anti-gaming measure exists

| Measure | Attack or failure it prevents |
|---|---|
| Evidence sets (a finding needs a full sufficient set, all ids must exist) | Guessing ids and amounts from partial reading. Citing everything. |
| 1-1 matching (Kuhn) | Repeating a correct finding for extra credit. |
| Step cost (0.002 per step) | Unbounded exploration. Kept tiny so it only separates equal answers. |
| Flag-all / FP penalty (F1-based) | Spraying findings of every type over many ids. Flag-all scores below zero. |
| Distractors | Pattern-matching on "same amount", "partial payment", "date near boundary". |
| Paired transforms (rename ids, permute order) | Memorising ids or relying on record order. Oracle reward must be identical. |
| Truth never in observations, errors or info | Leak of answers. Enforced by a leak scan in `check`. |
| Eval mode returns reward 0 and `info={}` | Reward hacking via the terminal signal during evaluation. Consequence: the trajectory reward of an eval run is 0.0, so paired and regression checks read `grader_report.reward_shared`. |
| `reset` / `grader_report` are harness-only (per-process token; cursors are HMAC-keyed) | An agent re-resetting the same episode to use the grader as an adaptive truth oracle, pointing the grader at its own "truth" files, or reading the report. |
| Salted, config-bound `episode_id` | Brute-forcing the seed out of the manifest and regenerating the truth; two configs aliasing on one id. |
| Input bounds on every action (depth 24, 50 000 nodes) and malformed submit => -1 | Crashing `step` with hostile nesting; non-terminal "free retry" submits. |
| Grader log per episode (including abandoned ones) | Silent loss of an episode, or an invalid submission that leaves no trace. |

## Isolation model

**Only the server path is an isolation boundary:** `server.py` (a separate process holding the truth) driven
through `client.RemoteAuditEnv`, with the per-process token. In-process `AuditEnv` and the `AgentHandle` are
**trusted-code only**: the env keeps truth in Python attributes, and the handle (a read-only object whose only
callable is `step`) is hygiene against accidental calls to `reset` / `grader_report`, not a sandbox, because closures and
attributes are introspectable in-process. A hostile policy must run in another process or container.

The agent-side process speaks JSON lines through `client.py` and never imports truth. `step` is the only agent
command. `reset` (which chooses seed, config and options; file paths are never accepted over the wire) and
`grader_report` are *harness* commands: the server pops a random per-process token from its environment at start and
refuses these two commands without it (and refuses them outright when no token is configured, unless the operator
passes `--insecure-open`). `RemoteAuditEnv` creates the token and keeps it; its `agent_handle()` exposes `step`
only. `grader_report` additionally needs `--allow-grader-report` and a finished episode. A reset of an unfinished
episode scores it as truncated/empty and logs it, so a harness cannot lose an episode silently. Failed resets reply
with a fixed message (our own `ConfigError` / `GenerationError` text is passed on, everything else only goes to the
server log), so there is no path or exception-text oracle. The oracle policy exists only inside the grader context.

**Time limit.** The 60 s limit runs from `reset` on a monotonic clock and includes the policy's own latency. It is
enforced cooperatively on every `step` and, in the server, by a watchdog timer that truncates an unfinished episode
(empty answer, `ended_by=time`, report logged) even when the agent goes silent.

**Cursors** are integrity checks, not capabilities-by-secrecy alone: `base64("table:offset:tag")` with `tag` an
HMAC-SHA256 keyed by a per-episode secret derived from private provenance, so a cursor cannot be forged from public
values (a foreign or edited cursor gives `invalid_cursor`). The cursor only encodes a page offset in a table the
agent can already read in full, so forging one would reveal nothing; the key just makes tampering detectable.

**Hostile inputs.** Actions are bounded (depth 24, 50 000 nodes, 200 000 string chars). A submit with a missing or
non-object `args`, extra keys, bad findings, a claimed amount outside `0..10**18` or a lone surrogate ends the
episode with reward -1. Integers wider than 64 bits are replaced by a placeholder in grader records and
trajectories (so a 5000-digit integer cannot break serialisation). Repeated ids in `affected_ids` / `evidence_ids`
are accepted and compared as sets, so a repeat neither helps nor hurts.

**Deployment recipe for a closed run.** `python -m auditgym isolate` is the shipped closed-run harness:

* the grader server (truth, token, grader reports) and the harness run as one user; the agent runs as a fresh
  process per episode under a separate unprivileged uid with `no_new_privs`, an empty workdir, an environment rebuilt
  from scratch (the token lives only in the server's environment) and no inherited descriptors, and reaches the harness
  only through the JSON-lines agent protocol, whose lines can only become `step` actions;
* `grader_data/`, the grader log directory and the episodes file are made unreadable to the agent uid for the run,
  and a preflight probe running as that uid must fail to read them (fail closed); without root the command refuses
  to run unless `--sandbox none` (no boundary) is requested explicitly;
* **holdout truth is generated on the fly** from `(seed, config)` inside the grader process and is never stored;
  only the open `train` / `dev` truth is on disk, shipped on purpose, so a closed run must use the jury's own hidden
  seeds;
* reads from and writes to the agent are both bounded by the step timeout and a per-episode wall-clock cap, so an
  agent that stops reading cannot block the harness; private seeds enter only through a protected file or stdin
  (`--seed` on the command line is world-readable via `/proc` and needs `--public-seeds`);
* `isolate --self-test` runs a hostile probe agent (token hunting via `/proc`, seeds on command lines, grader paths, `/proc/<pid>/mem`,
  server commands and malformed bytes over the action channel); `check` runs it as a hard check when it runs as root.

Separate containers for grader and agent can be layered on top; the agent then needs only the pipe.

Per-episode grader reports are written to `--log-dir` (the runner and `check` pass
`reports/grader_logs/<policy>/`), including `errors` for an invalid submission and `ended_by`
(`submit`, `steps`, `time`, `abandoned`, `agent_failed`, `closed`, `disconnected`).

## Reproducibility

Every random choice uses `random.Random` seeded by `sub_rng(seed, *labels)`, derived from
sha256 over the seed and labels (no `hash()`, no wall-clock). Sub-streams are independent, so
adding a draw in one component does not shift another. Same (seed, config, ENV_VERSION) gives
byte-identical public output and truth. `episode_id` is content-addressed:
`"ep_" + sha256(canonical public content without the manifest's episode_id)[:12]`, so it is a pure function of
the public records and adds no information about the seed beyond them. Because content differs, two
different configs or force blocks never alias on one id (grader reports, trajectories, cursor checksums and
`episodes/<id>/` are all keyed by it). `rename_ids` and `permute` change the public content and therefore
recompute the id. **Threat model:** the generator is open source and configs are public, so hidden-seed secrecy
relies on the seed space (use a private/large seed range for hidden evaluation) and on the agent having no code
execution, only the tool interface. There is no salt that could be kept secret from someone who can run the
generator. `check` regenerates committed
episodes and compares hashes. Time limit and step limit: see README (the wall-clock limit includes policy latency).

## Limitations (honest)

* The accounting policy is a simplified, self-contained rule set, not IFRS or an audit
  standard. Success here does not imply audit competence.
* Four issue types, monthly period, one ledger, no opening balances, no multi-entity
  consolidation. Fixed-offset time zones only (no DST).
* **Saturation:** the extra `careful_auditor` policy solves every holdout episode (F1 = 1.000, about 26
  steps, no false positive on clean books) from public evidence and the policy text. The environment is
  solvable, which is what the brief requires, but a programmed solver that knows the rules sees no
  headroom; headroom exists for agents that must infer or apply the rules themselves.
* **Period errors are two-directional, but only about posting inside the period:** `period_cutoff` covers an
  invoice issued after `period_end` (including date strings that still show the last day and exactly `period_end +
  1 s`) and the mirror case of an invoice issued before `period_start` (including exactly `period_start - 1 s`),
  in both cases posted inside the period. There is still no in-period invoice missing from the ledger and no late
  posting variant (records outside the period never appear in the ledger by construction).
* **Fixed-offset time zones only, no DST.** Every company and customer offset is a constant UTC offset, and code uses
  `datetime.timezone(timedelta)` only. Ambiguous or non-existent local times never occur, which also keeps the
  generator free of tz-database dependence; a DST-aware variant is not implemented.
* **Single-id issues by design.** Every generated truth issue has exactly one affected id (an invoice id for
  duplicates, cut-offs and FX-on-invoice, a payment id for misallocations). The matcher supports multi-id sets (only
  one finding that carries the whole set matches; see below) but the generator does not produce them, so the
  affected-id-set logic is covered by unit tests, not by generated data.
* **`payment_misallocation` has canonical maximal and minimal evidence sets.** The maximal set cites the payment
  document, every ledger row of the payment, and (foreign currency) every referenced invoice document plus the fx
  row of its issue date; the minimal set(s) drop the rows that are not needed to prove the discrepancy (the payment
  document, every AR credit row, and the fx/invoice documents; with no posted AR credit any one other row of the
  payment suffices, one minimal set per candidate row). The matcher therefore accepts exactly the subsets the
  independent verifier `rules.py` accepts, and a finding does not need to cite everything.
* **`fx:<date>:<src>:<dst>` keys are natural keys.** They are a pure function of the row's own public columns, so
  `rename_ids` keeps them on purpose (renaming would not hide anything: the date and currencies are in the row).
  The aggressive rename therefore changes every opaque id but not these keys; check pairs still pass because the
  agent can only use them as the lookup keys they are.
* **Cross-platform determinism status.** Generator output and scripted trajectories are byte-identical on macOS
  and Linux/aarch64 on CPython 3.11.15 (`docs/docker_check_result.md`), and public output was also compared on CPython
  3.12 and 3.14. An x86-64 run has not been recorded; `python -m auditgym golden` compares the committed digests on any
  machine. Randomness comes only from `sha256`-seeded `random.Random`; money is integer minor units and rates are `Decimal`, never floats.
* **Truncation on a clean book costs only the step cost.** A run truncated by the step or time limit is scored as
  an empty answer, so on a clean book it keeps F1 = 1 and pays only 0.002 per inspection step (+0.800 at 100 steps),
  and an honest early silent submit always scores at least as well; on an erroneous book it is clipped to -1.0. Kept on purpose (the brief says truncation means an empty answer).
* **Open-split private files are world-readable by design.** `grader_data/{train,dev}/*/truth.json` and
  `provenance.json` are committed and readable by any user, because the official `check` and the Docker image read
  them and the open splits are meant to be inspectable. They are not secret. Holdout truth is never written to disk.
* Foreign-currency refund distractors leave a small FX residual in `2300_unapplied_cash` (stated in the
  policy text, not an error).
* The perfect checker (`rules.py`) is independent of the generator but written by the same
  team. Shared misreadings of the policy are possible.
* The 20-episode holdout is small. Intervals in the report are wide for that reason.
* The wall-clock limit includes policy latency (a slow LLM agent can be truncated); replay determinism holds for
  step-bounded runs, and `run --deterministic` freezes the clock for in-process replays.
* The `wall s` column of the results is scripted-policy time per episode, not LLM latency.
* Threat model: generator and configs are public, so a hidden seed is only as secret as the seed space and the
  agent's lack of code execution. `episode_id` is content-addressed and adds nothing beyond the records.
* Exploit resistance is tested against the four shipped baselines and scripted attacks (reset oracle,
  forged truth paths, nesting bombs, malformed submits), not against adaptive learning attackers.
* The experimental reward is a research variant and is reported separately. It is soft-F1,
  `2*(TP + 0.25*P) / (n_truth + n_findings)` clipped like the shared reward, where `P` counts one-to-one
  near misses (type + affected_ids right, amount/currency/evidence wrong) that still show work: they cite
  existing evidence ids or claim an amount within 10x the tolerance (a bare type + ids guess earns nothing).
  **Warning:** this reward is shaped; it must not be used for model selection or reported as a score. Only
  the shared reward is official.
* Matcher corner cases (covered by tests, not by generated data): an issue with two affected ids is only matched
  by one finding carrying the whole set (split findings score 0 TP); `affected_ids` is a set, so repeated ids
  are accepted and neither help nor hurt; claimed amounts above `10**18` and lone surrogates make the submission
  invalid. An honest early silent submit on a clean book (+0.802 after 99 steps) beats a run truncated at the limit
  (+0.800).

### Known limitations (won't-fix, stated honestly)

These items were raised in review and deliberately not changed. Each says why.

* **Open dev/train truth and provenance files are world-readable and sit in the same repo tree as agent code.** The
  brief makes dev/train open on purpose and ships `grader_data` for those splits. Holdout truth is generated on the
  fly and never stored. OS-level separation (a separate user or container for the grader, agent code without read
  access to `grader_data/`) is enforced by `python -m auditgym isolate` (DESIGN "Deployment recipe"): under its
  default sandbox the agent's uid cannot read `grader_data/` for the duration of the run. Plain `run` / `check` keep
  the single-trust-domain layout.
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
  ledger rows the posted amount is summed from: `rules.check_evidence` rejects a citation that leaves one out. The
  remaining matcher paths are covered by synthetic matcher tests, not generated data.
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

## Measured results

<!-- RESULTS:BEGIN (generated by `python -m auditgym check --update-docs`; do not edit) -->

Generated by the latest **full** `python -m auditgym check` run: 74 PASS, 0 FAIL, 1 WARN, 0 SKIP; pytest: 1077 passed. Full per-check table: [docs/check_report.md](check_report.md).


#### Baseline policies (eval mode, holdout, remote env)

Shared reward, F1, experimental reward, steps and wall time are means over episodes (95% percentile bootstrap CI, 2000 resamples, fixed seed). Precision and recall are POOLED over the episodes (sum TP / (sum TP + sum FP or FN)), because a silent agent on a clean book has P = R = 0 (0/0) while F1 = 1.

| policy | n | precision (pooled) | recall (pooled) | F1 [95% CI] | shared reward [95% CI] | experimental | steps | wall s |
|---|---|---|---|---|---|---|---|---|
| random_explorer | 20 | 0.000 | 0.000 | +0.100 [+0.000, +0.250] | -0.806 [-1.000, -0.514] | -0.806 | 29.0 | 0.11 |
| reconcile | 20 | 0.066 | 0.479 | +0.109 [+0.063, +0.155] | -0.807 [-0.891, -0.721] | -0.791 | 20.4 | 0.12 |
| flag_all | 20 | 0.003 | 0.028 | +0.006 [+0.000, +0.014] | -0.990 [-1.000, -0.974] | -0.984 | 6.0 | 0.09 |
| oracle | 20 | 1.000 | 1.000 | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 | 1.0 | 0.11 |
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

See the README section "How to read the failure analysis" for the interpretation of the tables above.
