# AuditGym check report

- env_version: `auditgym-1.0.0`  python: `3.13.18`  platform: `Linux`  mode: `full`
- configs: `dev` dev@1.3.0 (hash `fb29dd57ab25`), `holdout` holdout@1.3.0 (hash `3396914da9dc`), `train` train@1.3.0 (hash `b0cdce3de8ea`)
- result: **PASS**  (75 pass, 0 fail, 1 warn, 0 skip)


## Step 1

| status | check | detail |
|---|---|---|
| PASS | pytest | 1080 passed in 118.76s (0:01:58) (121.2s) |

## Step 2

| status | check | detail |
|---|---|---|
| PASS | train: regeneration is byte-identical (x2) | 10 episodes |
| PASS | train: matches committed episodes/train/ | 10 compared, 0 not committed |
| PASS | dev: regeneration is byte-identical (x2) | 10 episodes |
| PASS | dev: matches committed episodes/dev/ | 10 compared, 0 not committed |
| PASS | every shipped config has config_id, config_version and a resolvable config_hash | dev@1.3.0, holdout@1.3.0, train@1.3.0 |
| PASS | configs lock (CONFIG_VERSIONS.json) binds every config_id@config_version to its content hash (no content change without a version bump) | 3 configs locked |
| PASS | golden digests (tests/golden_hashes.json) match this machine (episodes, truth, trajectories) | 6 episodes, python 3.13.18 on Linux |
| PASS | every shipped trajectory row conforms to schemas/trajectory_row.schema.json | 2211 rows in 3 files |
| PASS | oracle rows are in oracle_reference.jsonl (labelled), not mixed into trajectories.jsonl | trajectories.jsonl policies ['careful_auditor', 'careful_auditor_degraded', 'flag_all', 'random_explorer', 'reconcile'] |
| PASS | samples.jsonl has an eval-mode run (reward hidden) and truncated=true runs (steps and wall clock) | truncated by ['steps', 'time'], eval-mode samples 1 |
| PASS | no private truth (truth/evidence/scenario tokens, seed key) in any observation of the shipped trajectories | 2211 observations in 3 files scanned for 14 tokens |
| PASS | every trajectory header carries env_version, config_id, config_version and config_hash | 2 distinct (env, config, version) combinations |

## Step 3

| status | check | detail |
|---|---|---|
| PASS | holdout has 20 episodes in splits.json | found 20 |
| PASS | all holdout episodes pass validate.py (identities, rules==truth, evidence, leak, counts) | 20/20 valid |
| PASS | every holdout episode (public files, truth, provenance) conforms to schemas/*.json | 20/20 |
| PASS | holdout composition 5 clean / 5 interaction / 10 mixed | found {'clean': 5, 'interaction': 5, 'mixed': 10} |
| PASS | declared splits.json category matches the category measured from truth | all consistent |

## Step 4

| status | check | detail |
|---|---|---|
| PASS | every (policy, episode) run completed | 80 runs |
| PASS | extra diagnostic policies completed (not one of the 4 required baselines) | 40 runs |
| PASS | all episodes ended with a grader report |  |
| PASS | all runs within 100 steps |  |
| PASS | the grader wrote one per-episode log file for every run (--log-dir, never sent to the agent) | 120 log files for 120 runs |
| PASS | no baseline submitted a malformed submission |  |
| PASS | reconcile obeys the budget (never truncated, submits) |  |
| PASS | env grader_report == independent matcher.score(recorded submission) | all runs agree |
| PASS | oracle: F1 = 1.0 and shared reward = +1.0 on every episode (0 inspection steps) | mean shared +1.000 |
| PASS | flag_all mean shared reward < 0 | -0.990 |
| WARN | ordering oracle > reconcile > random_explorer, flag_all (mean shared) | random_explorer -0.806, reconcile -0.807, flag_all -0.990, oracle +1.000, careful_auditor (extra) +0.949, careful_auditor_degraded (extra) +0.030 |
| PASS | careful_auditor_degraded is a measurable near miss (0 < mean F1 < careful_auditor's; reward below it) | degraded F1 0.531 / shared +0.030 vs careful_auditor 1.000 / +0.949 |
| PASS | reconcile earns partial credit (0 < mean F1 < 1) | mean F1 0.109 |

## Step 5

| status | check | detail |
|---|---|---|
| PASS | rename (shape-preserving ids) + permute: oracle reward identical and answer maps (20 pairs) | 20/20 |
| PASS | AGGRESSIVE rename (other prefix/length/charset/sort order) + permute: oracle reward identical and answer maps (20 pairs) | 20/20 |
| PASS | rename+permute: reconcile (a non-oracle policy) gets identical metrics, reward and steps (20 pairs) | 20/20 |
| PASS | aggressive rename+permute: reconcile (a non-oracle policy) gets identical metrics, reward and steps (20 pairs) | 20/20 |
| PASS | rename+permute: careful_auditor (a non-oracle policy) gets identical metrics, reward and steps (20 pairs) | 20/20 |
| PASS | aggressive rename+permute: careful_auditor (a non-oracle policy) gets identical metrics, reward and steps (20 pairs) | 20/20 |
| PASS | rename actually changed the ids the oracle cites | ids changed on every episode |
| PASS | aggressive rename: every cited opaque id is numeric-only or z+hex and differs from the original (fx:* natural keys are intentionally kept) | 20/20 |
| PASS | paired checks are discriminating: reconcile and careful_auditor score above the -1.0 floor on >= 4 of 20 pairs | reconcile: 13/20 above floor, careful_auditor: 20/20 above floor |
| PASS | replay of the same seed + actions gives an identical trajectory sha256 (20 pairs) | 20/20 |
| PASS | oracle submission == the one derived from the committed grader_data truth.json (open splits) | 20/20 episodes |
| PASS | oracle scored against the COMMITTED truth.json + committed public ids: F1 = 1.0 | 20/20 |
| PASS | careful_auditor env reward == independent score against the committed truth.json | 20/20 |

## Step 6

| status | check | detail |
|---|---|---|
| PASS | no hidden-data tokens in 4225 recorded observations | scanned 240 runs for 14 tokens |
| PASS | observation schema is exactly {episode_id, step, steps_remaining, result, error} (+task/manifest/policy at reset) |  |
| PASS | episode_id equals the content hash of the public records (carries no extra information about the seed) | 20 episodes |
| PASS | episode_id is not / does not embed the decimal seed (seeds of >= 6 digits searched as substring) | 20 episodes |
| PASS | reset and grader_report are harness-only: refused without the control token | refused=[True, True] |
| PASS | a deeply nested action gives a bounded invalid_args error, consumes one step, keeps the run alive | code=invalid_args, next step=2 |
| PASS | submit without args.findings: ends the episode; eval shows no error (validity is not leaked), grader log has reward -1 and the reason | terminated=True, grader reward -1.0 |
| PASS | grader log file written for the invalid submission and for an episode abandoned by re-reset | 3 logs, invalid-with-error=True, abandoned=1 |
| PASS | isolated agent run: escape probe finds no token, grader file or protocol hole | probe passed (separate uid, scrubbed env, grader paths unreadable) |

## Step 7

| status | check | detail |
|---|---|---|
| PASS | complete correct answer with evidence > silence on books with issues | +1.000 vs -1.000 (2 issues) |
| PASS | silence > false accusations on clean books | +1.000 vs -1.000 |
| PASS | duplicating a correct finding gives no gain (second copy is FP) | single +0.333, duplicated +0.000, tp=1 fp=1 |
| PASS | score ignores explanation text and the order of findings/ids | order, id order and explanation text do not change the score |
| PASS | step cost is tiny: <= 0.002 per inspection step | 0.002000 per step |
| PASS | env: perfect answer after 5 inspection steps earns 1 - 5*0.002 | reward +0.9900 (expected +0.9900) |
| PASS | unknown/non-existent evidence id invalidates the finding | evidence_unknown_id |
| PASS | guessed affected ids are false positives | reason ids_mismatch |
| PASS | wrong currency invalidates the finding | currency_mismatch |
| PASS | missing or incomplete evidence set invalidates the finding | empty evidence: evidence_insufficient; one id missing: evidence_insufficient |
| PASS | amount tolerance is inclusive max(1, expected//1000) | tolerance 1: +tol ok, +tol+1 rejected |
| PASS | malformed submissions score -1 (15 variants) | 15 malformed variants all score -1 |
| PASS | re-submit / any step after the end -> episode_over, no state change, reward 0 | first +1.000, re-submit reward 0.0, code episode_over |
| PASS | malformed submit: train -> reward -1 + reason; eval -> no error shown (no validity oracle), grader keeps -1 and error log | train reward -1.0, eval error None (no validity oracle), grader reward -1.0 |
| PASS | truncation by steps: 100th non-submit step -> truncated, findings treated as empty and scored | clean +0.800, issues -1.000 (truncated at step 100, scored as empty) |
| PASS | truncation by wall-clock limit -> truncated and scored as empty | truncated=True, reward +0.998 |
| PASS | step before reset -> not_reset | not_reset |
| PASS | eval mode: terminal reward 0.0 and empty info to the policy; grader keeps the real result | policy sees reward 0.0, info {}; grader keeps +1.000 |
| PASS | train-mode terminal info has only counts/metrics (no ids) | info keys ['inspection_steps', 'invalid_submission', 'metrics', 'reward_experimental', 'reward_shared'] |
| PASS | bad actions: bounded error code, message <= 120 chars, consume a step, reward 0 | codes ['unknown_action', 'unknown_id', 'invalid_args', 'invalid_args', 'invalid_cursor', 'unknown_table', 'calculator_error'] |
| PASS | pagination: stable cursors, every row exactly once, foreign cursor rejected | 457 ledger rows over 10 pages; foreign cursor rejected |
| PASS | flag-all scores badly: reward < 0 on the first 6 episodes (train mode, env) | rewards -1.00, -1.00, -1.00, -1.00, -1.00, -1.00 |
| PASS | agent-side policies import neither generator nor truth | policies (except oracle) have no truth/generator imports |

## Baseline policies (eval mode, holdout, remote env)

Shared reward, F1, experimental reward, steps and wall time are means over episodes (95% percentile bootstrap CI, 2000 resamples, fixed seed). Precision and recall are POOLED over the episodes (sum TP / (sum TP + sum FP or FN)), because a silent agent on a clean book has P = R = 0 (0/0) while F1 = 1.

| policy | n | precision (pooled) | recall (pooled) | F1 [95% CI] | shared reward [95% CI] | experimental | steps | wall s |
|---|---|---|---|---|---|---|---|---|
| random_explorer | 20 | 0.000 | 0.000 | +0.100 [+0.000, +0.250] | -0.806 [-1.000, -0.514] | -0.806 | 29.0 | 0.11 |
| reconcile | 20 | 0.066 | 0.479 | +0.109 [+0.063, +0.155] | -0.807 [-0.891, -0.721] | -0.791 | 20.4 | 0.12 |
| flag_all | 20 | 0.003 | 0.028 | +0.006 [+0.000, +0.014] | -0.990 [-1.000, -0.974] | -0.984 | 6.0 | 0.11 |
| oracle | 20 | 1.000 | 1.000 | +1.000 [+1.000, +1.000] | +1.000 [+1.000, +1.000] | +1.000 | 1.0 | 0.10 |
| careful_auditor (extra) | 20 | 1.000 | 1.000 | +1.000 [+1.000, +1.000] | +0.949 [+0.946, +0.953] | +0.949 | 26.4 | 0.12 |
| careful_auditor_degraded (extra) | 20 | 0.570 | 0.634 | +0.531 [+0.390, +0.663] | +0.030 [-0.246, +0.290] | +0.030 | 20.9 | 0.13 |

`oracle` reads the truth grader-side and only demonstrates the reward ceiling; it is not agent ability.

`careful_auditor (extra)` is an additional diagnostic row, not one of the 4 required baselines: an honest observation-only scripted agent (no truth, no generator, no rules engine) showing the environment is solvable from public evidence within the 100-step budget.

`careful_auditor_degraded (extra)` is the same agent with a 20-inspection-step cap and period membership decided on the wall-clock fields of `issued_at` (the time-zone rule ignored). It is the near-miss row: the reward and F1 gap to `careful_auditor` is measured, not asserted, and the failure analysis below lists which issue types it misses.

## Failure analysis (measured on the same eval runs; from the grader reports)

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

## Paired invariance pairs (per-pair F1 of the untransformed run)

A pair only proves something when the policy is above the -1.0 floor, so the discriminating policies are `reconcile` and `careful_auditor`; `random_explorer` and `flag_all` are excluded (trivially invariant at the floor). Each cell is the F1 of one holdout episode (order: clean, interaction, mixed, as in `splits.json`); every listed pair gave identical metrics, reward and steps under the shape and the aggressive rename + permute.

| policy | pairs above floor | per-pair F1 |
|---|---|---|
| reconcile | 13/20 | 0.00 0.00 0.00 0.00 0.00 0.00 0.22 0.00 0.06 0.05 0.14 0.14 0.22 0.21 0.36 0.06 0.20 0.12 0.16 0.24 |
| careful_auditor | 20/20 | 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 1.00 |
