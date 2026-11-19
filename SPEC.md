# AuditGym — implementation spec (source of truth for all modules)

AuditGym is a reproducible RL environment for an AI financial auditor. One episode = one
fictional company x one monthly reporting period. The agent reads records through a narrow
tool interface and submits findings with evidence; a grader scores them.

Runtime: **Python 3.11, stdlib only** (pytest only for tests). No network, no GPU, no floats
in money (ints in minor units, `decimal.Decimal` for rates). Deterministic: every random
choice comes from `random.Random` instances seeded via `auditgym.rng.sub_rng(seed, *labels)`
(sha256-derived; never `hash()`, never wall-clock). Already written and FROZEN (import, don't
edit): `auditgym/constants.py`, `auditgym/money.py`, `auditgym/episode.py`.

## 0. Package layout and ownership

```
auditgym/
  constants.py money.py episode.py        # frozen contracts (orchestrator)
  rng.py                                   # sub_rng(seed, *labels) -> random.Random   [generator owner]
  generator/
    config.py     # load/validate config JSON, sample concrete EpisodeSpec from ranges
    world.py      # hidden clean company history -> clean records
    distractors.py# correct-but-suspicious patterns (applied in clean world)
    injectors.py  # 4 issue types + 2 interactions; mutate records, emit truth+provenance
    build.py      # generate_episode(seed, config) -> Episode   (THE entry point)
  transforms.py   # rename_ids(ep, salt), permute(ep, salt)  (benign, truth-preserving)
  calculator.py   # safe Decimal arithmetic evaluator
  env.py          # AuditEnv: reset/step contract, tools, budget, trajectory recording
  server.py client.py   # out-of-process grader: env lives in a subprocess, JSON lines
  matcher.py      # submission schema, normalisation, 1-1 matching, metrics, rewards
  rules.py        # INDEPENDENT rules engine (perfect checker), used by validate + oracle sanity
  validate.py     # data/eval-truth validation: identities, discrepancy, evidence sufficiency
  policies/       # base.py random_explorer.py reconcile.py flag_all.py oracle.py
  runner.py       # run policies over episodes, record trajectories, collect grader reports
  check.py        # `python -m auditgym check` — single official verification command
  __main__.py     # CLI: generate | check | run | demo
configs/          # train.json dev.json holdout.json splits.json (versioned)
tests/
```

## 1. Episode data (public)

All timestamps ISO-8601 **with offset** (`2026-03-31T23:10:00+05:00`). Money = int minor units;
`CURRENCY_EXPONENT` gives decimals. FX rates = decimal strings with 6 places. Missing links are
`null` in JSON / empty field in CSV (loader turns empty into None). Never guess links.

**manifest.json**
```json
{"schema_version":"1.0.0","env_version":"auditgym-1.0.0","episode_id":"ep_3f9a1c22b7d4",
 "company_name":"Fictional Northwind Tiles LLP",
 "base_currency":"KZT","timezone":"Asia/Almaty","utc_offset":"+05:00",
 "period_start":"2026-03-01T00:00:00+05:00","period_end":"2026-03-31T23:59:59+05:00",
 "counts":{"ledger_rows":412,"invoices":71,"payments":58,"documents":129,"fx_rates":96},
 "rounding":{"mode":"half_even","unit":"base_minor","fx_rate_places":6},
 "money":{"amounts":"integer minor units","exponents":{"KZT":2,"EUR":2,...}}}
```
The public manifest carries **no** `config_id` / `config_version` / `config_hash`: they would tell the policy
which split (train / dev / holdout) it is on. They are kept in `provenance.json`, in the grader report and in the
trajectory headers; `configs/CONFIG_VERSIONS.json` maps `config_id@config_version` to the sha256 of the resolved
config, and `check` fails if a shipped config changes without a version bump (`python -m auditgym lock`).

`episode_id` = `"ep_" + sha256(<canonical public content>)[:12]`, content-addressed: the hash covers
`render_public` of manifest (with the `episode_id` field removed), policy, ledger, invoices, payments and fx
(see `auditgym/content_id.py`). It is a pure function of the public records, so it reveals nothing about the
seed beyond the records themselves and cannot alias across different content. There is no id salt. Truth,
provenance and the manifest are filled with the id after it is computed. `rename_ids` and `permute` recompute it.
**Threat model:** the generator is open source, so hidden-seed secrecy relies on the seed space being large/private
and on the agent having no code execution (tools only); `episode_id` adds no information beyond the records.
Company names, entity names are fictional (word lists).
Timezones: only fixed-offset, no-DST zones (Asia/Almaty +05:00, Asia/Tokyo +09:00, UTC +00:00,
Asia/Dubai +04:00, America/Bogota -05:00, Asia/Kolkata +05:30). Code uses `datetime.timezone(timedelta)` only.

**policy.json** — machine-readable + short text rules (the agent sees it in the first observation):
```json
{"revenue_recognition":{"rule":"AR and revenue are recognised at invoice issued_at. An invoice belongs to the period iff period_start <= issued_at (as an instant) <= period_end. Compare instants, not date strings."},
 "period_boundaries":{"period_start":"...","period_end":"...","inclusive":true,"timezone":"Asia/Almaty","utc_offset":"+05:00"},
 "fx":{"direction":"rate converts 1 unit of source_currency into target_currency (=base)",
       "invoice_rate_date":"calendar date of issued_at expressed in company utc_offset",
       "payment_rate_date":"calendar date of payment timestamp expressed in company utc_offset",
       "fallback":"if no rate for that date, use the most recent earlier date present in fx.csv",
       "conversion":"base_minor = round_half_even(amount_minor / 10**exp_src * rate * 10**exp_base)"},
 "ledger":{"currency":"all ledger rows are in base_currency","double_entry":"each posting group (same timestamp + same invoice_id/payment_id) balances"},
 "postings":{
   "invoice":"Dr 1200_accounts_receivable = conv(total); Cr 2200_vat_payable = conv(vat); Cr 4000_revenue = conv(total)-conv(vat)",
   "payment":"Dr 1000_cash = conv(amount, payment date rate); for each invoice_refs entry Cr 1200_accounts_receivable = conv(ref.amount_minor, that invoice's rate); amount not covered by invoice_refs -> Cr 2300_unapplied_cash (converted at payment rate); remaining difference -> 7900_fx_gain_loss (Cr if gain, Dr if loss)",
   "refund":"negative amount_minor; Dr 2300_unapplied_cash, Cr 1000_cash"},
 "duplicate_rule":"two invoices of the same entity with identical currency and identical line_items (description, quantity, unit_price_minor, vat_rate, service_period, po_ref) are duplicates; the later one (issued_at, then invoice_id) is the duplicate. Same amount with a different service_period is NOT a duplicate.",
 "finding_amounts":{
   "duplicate_invoice":"AR debit posted for the duplicate invoice (base minor)",
   "payment_misallocation":"sum over invoice_refs of conv(ref.amount_minor, invoice rate) minus AR credits actually posted with this payment_id (base minor)",
   "fx_error":"abs(posted - correct) on the AR debit row (invoice) or the cash debit row (payment), base minor",
   "period_cutoff":"AR debit posted inside the period for an invoice issued outside it (base minor)"},
 "finding_affected_ids":{"duplicate_invoice":"[duplicate invoice_id]","payment_misallocation":"[payment_id]","fx_error":"[invoice_id] or [payment_id]","period_cutoff":"[invoice_id]"},
 "finding_currency":"always base_currency",
 "amount_tolerance":"|claimed - expected| <= max(1, expected // 1000) minor units",
 "evidence":"evidence_ids are ledger row_id, document doc_id or fx ids 'fx:<date>:<source>:<target>'; every cited id must exist; a finding needs a full sufficient evidence set (see task text)"}
```

**ledger.csv** columns `LEDGER_FIELDS`. One row is one side of a posting (debit_minor xor credit_minor > 0,
the other 0). All rows currency = base. Rows exist only for timestamps inside the period. `entity_id`,
`invoice_id`, `payment_id` filled when applicable else null. Noise postings (rent, payroll, software:
`Dr 6xxx / Cr 2000_accounts_payable` or `Cr 1000_cash`) have entity_id null (or a vendor id) and no
invoice/payment ids; they make up the bulk to reach 200–2000 rows.

**invoices.jsonl** fields `INVOICE_FIELDS`. `line_items` = list of
`{"line":1,"description":"Ceramic tiles 30x30","quantity":12,"unit_price_minor":45000,"vat_rate":"0.12","service_period":"2026-03","po_ref":"PO-88213"}`;
`total_minor` = sum(round(q*unit_price*(1+vat))) — define per line: `net = q*unit_price_minor`, `vat = round_half_even(net*vat_rate)`, total = Σ(net+vat). Invoice file contains: all invoices issued in period, plus 3–8 issued in the first 1–10 days after period_end (needed for cutoff testing; correctly NOT in the ledger), plus 2–4 issued in the 7 days before period_start (prior-period documents, never posted in the clean ledger; no opening AR in v1). issued_at uses the customer's own offset (varied), due_at = issued_at + 15/30/45 days.

**payments.jsonl** fields `PAYMENT_FIELDS`. `invoice_refs` = list of `{"invoice_id":..., "amount_minor":...}`
(amounts in payment currency) or `null` (prepayment / unreferenced) . Payment currency always equals the
currency of every referenced invoice. Refunds have negative amount_minor and invoice_refs null. All payment
timestamps fall inside the period.

**fx.csv** `date,source_currency,target_currency,rate`: one row per foreign currency used per **business day**
(Mon–Fri) from period_start-7d to period_end+12d; target = base. Rates follow a seeded random walk (daily
moves 0.1%–1.5%, guaranteed to differ between consecutive dates). Weekend gaps make the fallback rule matter.

**Sizes** (defaults, configurable): 200–2000 ledger rows, 20–200 documents, 0–12 issues.

## 2. Private grader files

**truth.json**
```json
{"schema_version":"1.0.0","episode_id":"ep_...","base_currency":"KZT",
 "issues":[{"issue_id":"iss_01","type":"fx_error","affected_ids":["INV-..."],
            "expected_amount_minor":12345,"currency":"KZT","tolerance_minor":12,
            "evidence_sets":[["D-...","R-...","fx:2026-03-12:EUR:KZT"]],
            "scenario":null}]}
```
`scenario` = interaction name if the issue came from an interaction, else null. `issue_id`s are
`iss_01..` in a seeded-shuffled order (must not encode type). `tolerance_minor = max(1, expected//1000)`.

**provenance.json**: `{episode_id, env_version, config_id, config_version, seed, spec (sampled counts),
clean_records:{ledger,invoices,payments} (pre-injection snapshot), changes:[{issue_ids, injector, root_cause,
description, records_added:[ids], records_removed:[ids], records_modified:[{id, field, before, after}]}],
distractors:[{kind, record_ids}]}`. Truth/provenance never enter observations, file paths, errors or info.

## 3. Generator (`generator/`)

`generate_episode(seed: int, config: dict | str) -> Episode` (config may be a dict or a path / name in
`configs/`). Steps: (1) sample spec from config ranges with `sub_rng(seed,"spec")`; (2) build a coherent
hidden history (customers, invoices with line items, payments incl. split/partial-by-design, fx walk); (3)
apply distractors; (4) post the **clean** ledger by the policy rules; snapshot clean records; (5) apply
injectors (each records its causal change + truth issue + evidence sets); (6) shuffle-free canonical order
(ledger by timestamp then row_id, docs by issued_at/timestamp then id); (7) assemble manifest/policy.
Same (seed, config, ENV_VERSION) ⇒ byte-identical `render_public` output and truth.

ID styles vary per seed (pick one style per company): invoice ids e.g. `INV-2026-0042`, `SI/7781-K`,
`F-AB12C9`; payments `PAY-…`/`RCPT-…`/`BNK…`; entities `CUST-…`/`C…`; doc_id `D-<10 hex>`; row_id
`R-<8 hex>` (random hex, NOT sequential, so injected rows are indistinguishable). Numbering sequences
must not reveal injection (draw duplicate ids from the same pool/sequence as normal ones).

### Issue types (independently configurable counts)
1. **duplicate_invoice** — an existing posted invoice is re-issued as a new invoice (new invoice_id/doc_id,
   issued 0–5 days later inside period, identical entity/currency/line_items incl. service_period & po_ref)
   and posted to the ledger again (AR/VAT/revenue rows). affected=[dup invoice_id]; amount = dup AR debit;
   evidence_sets=[[orig_doc_id, dup_doc_id, dup_AR_row_id]].
2. **payment_misallocation** — a referenced payment's AR credit is posted partially (40–90% of the ref) or
   not at all (missing); the uncredited part goes to 2300_unapplied_cash (converted at payment rate; fx row
   adjusted so the group still balances). amount = Σ conv(refs, invoice rates) − posted AR credits for the
   payment. evidence_sets = the canonical maximal set [payment_doc_id, *all ledger row_ids with this payment_id,
   *referenced invoice doc_ids (only if foreign currency), *fx ids of those invoices' rate dates (only if foreign)]
   plus the minimal set [payment_doc_id, *AR credit rows of the payment, *invoice doc_ids, *fx ids] (when no AR credit
   was posted: one minimal set per other row of the payment) -- every set ≤10 ids, so the matcher accepts exactly
   the subsets `rules.check_evidence` accepts (any-of-sets).
3. **fx_error** — foreign invoice or payment converted with a wrong rate: variant `wrong_date` (rate of
   another date, e.g. due date, the UTC date, or ±1 business day) or `wrong_rate` (digits transposed / stale
   rate). Invoice: AR, VAT and revenue rows all use the wrong rate (still balanced). Payment: cash row wrong,
   AR credit correct, fx gain/loss row absorbs the difference. Must change the base amount by ≥ 2*tolerance+1.
   affected=[invoice_id] or [payment_id]; amount=|posted−correct| on AR/cash row;
   evidence_sets=[[doc_id, AR-or-cash row_id, fx id of the CORRECT rate date (after fallback)]].
4. **period_cutoff** — an invoice issued after period_end (as an instant) is posted inside the period
   (timestamp moved to the last 0–3 days of period); mirror case: an invoice issued before period_start
   (prior-period document, unposted in the clean ledger) posted in the first 0–3 days. Boundary cases: issued
   exactly `period_end + 1 s` / `period_start - 1 s` (out of period); `timezone_boundary_ok` distractors issued
   exactly at `period_end` / `period_start` are correct (the bounds are inclusive). Use tricky offsets half the time (e.g. issued_at
   `2026-03-31T22:00:00-03:00` = 2026-04-01 06:00 +05:00: date string looks in-period). It is one of the
   post-period invoices in invoices.jsonl. Rates: posted with correct rate of its issue date unless it is
   the interaction below. affected=[invoice_id]; amount = AR debit posted; evidence_sets=[[doc_id, AR_row_id]].

### Interactions (two causal scenarios, each yields 2 truth issues with the same `scenario`)
- **fx_partial_payment_at_boundary** — foreign payment whose timestamp sits near a day boundary such that
  its own-offset date ≠ company-offset date, booked with the rate of the wrong (own-offset) date AND
  allocated partially. Truth: payment_misallocation + fx_error, both affected=[payment_id].
- **cutoff_with_wrong_rate_date** — foreign invoice issued just after period_end (company offset) posted at
  period_end using the period_end date's rate. Truth: period_cutoff (amount = AR posted) + fx_error
  (|posted − conv(total, rate of real issue date)|), both affected=[invoice_id].
Amounts must be computed from the final records, so each truth issue is independently verifiable.

### Distractors (correct; must produce zero truth issues)
split_payment (one invoice paid by 2–3 payments, each ref partial, fully credited); overpayment_refund
(payment > ref, excess to unapplied, later negative refund payment clears it); legit_recurring_amount (same
entity & amount in consecutive months/weeks with different service_period); timezone_boundary_ok (invoice
issued_at like `2026-04-01T01:00:00+08:00` that is inside the period as an instant, correctly posted);
realized_fx_gain_loss (foreign invoice paid at a different rate, gain/loss row); unreferenced_prepayment
(invoice_refs null, all to unapplied). Clean companies: config `clean_probability` or explicit `issues` all 0.

### Config JSON (`configs/*.json`)
Illustrative example: it shows the shape and the shipped `train` values at config version 1.3.0; the files in
`configs/` are authoritative, and `CHANGELOG.md` lists how configs changed since 1.0.0.
```json
{"config_id":"train","config_version":"1.3.0",
 "company":{"n_customers":[6,18],"n_invoices":[30,90],"n_noise_postings":[60,250],
            "foreign_invoice_share":0.35,"base_currencies":["KZT","USD","EUR"],
            "foreign_currencies":["EUR","USD","GBP","JPY","CHF","AED"],
            "timezones":[["Asia/Almaty","+05:00"],["UTC","+00:00"]],
            "customer_offsets":["+05:00","+00:00","-03:00","+08:00","-05:00","+09:00"],
            "years":[2025,2026],"vat_rates":["0","0.12","0.20"]},
 "issues":{"duplicate_invoice":[0,2],"payment_misallocation":[0,2],"fx_error":[0,2],"period_cutoff":[0,2]},
 "interactions":{"fx_partial_payment_at_boundary":[0,1],"cutoff_with_wrong_rate_date":[0,1]},
 "distractors":{"split_payment":[1,3],"overpayment_refund":[0,2],"legit_recurring_amount":[1,3],
                "timezone_boundary_ok":[1,2],"realized_fx_gain_loss":[1,4],"unreferenced_prepayment":[0,2]},
 "clean_probability":0.2,
 "force":{}}
```
Validation is strict (`ConfigError`): unknown top-level / `company` / `force` keys, a missing or non-`X.Y.Z`
`config_id` / `config_version` (never defaulted), probabilities outside [0,1] and malformed values are rejected;
`schemas/config.schema.json` describes the shape. Natural (unforced) error counts are drawn as a TOTAL first
(skewed to small/medium, ≥1 for a non-clean episode) and then spread over the types within their ranges, so
the cap of 12 is rarely hit and interactions appear in roughly a third of episodes.
`force` may pin exact values (e.g. `{"clean":true}` or `{"issues":{"fx_error":3}}`) — used by tests and the
holdout suite. Total issues capped at 12.

## 4. Environment (`env.py`)

```python
class AuditEnv:
    def __init__(self, mode="train", max_steps=MAX_STEPS, time_limit_s=EPISODE_TIME_LIMIT_S): ...
    def reset(self, seed=None, config=None, options=None) -> tuple[dict, dict]
        # config: dict | config name | {"episode_dir": public_dir, "grader_dir": private_dir}
        # options: {"transform": {"rename_salt": int|None, "permute_salt": int|None}} (paired checks)
    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]
    def grader_report(self) -> dict    # grader side only (metrics, match table, errors) — never sent to policy in eval
    def trajectory(self) -> list[dict] # policy-visible trajectory records
```
Action JSON: `{"action": "<name>", "args": {...}}`.
- `list_records {table, cursor=null, limit=50}` — limit 1..50 (else invalid_args). Cursor is an opaque
  stable string (base64 of `"<table>:<offset>:<tag>"`, `tag` = HMAC-SHA256 keyed by a per-episode grader-side secret) ; foreign/garbled cursor
  → invalid_cursor. Rows for invoices/payments omit `line_items` / keep `invoice_refs`; ledger and fx rows
  full (fx rows get an extra `fx_id`). Result `{table, rows, next_cursor, total}`.
- `get_record {table, id}` — by PRIMARY_KEY (fx by fx_id). Full record (invoice includes line_items).
- `get_document {doc_id}` — full invoice or payment document.
- `calculator {expression}` — see §5.
- `submit {findings: <findings.json object>}` — ends episode, terminated=True.
Observation (always JSON-serialisable, deterministic, no wall-clock):
`{"episode_id", "step": n, "steps_remaining": k, "result": <obj|null>, "error": {"code","message"}|null}`.
First observation (reset) additionally has `"task"` (instructions text incl. types, amount definitions,
evidence-set rules, action schemas), `"manifest"`, `"policy"`; `result`/`error` null.
Unknown action / bad args / unknown table / unknown id → bounded error (code from ERROR_CODES, message
≤ 120 chars, never echoing hidden data, may echo the offending id truncated to 64 chars), consumes a step,
reward 0. Every non-submit step reward = 0.0. Budget: MAX_STEPS=100 including submit. When the 100th step
is a non-submit action, or wall-clock since reset > time_limit_s ⇒ truncated=True, findings treated as empty,
scored. Steps after the episode ended → error episode_over, no state change (returns terminated/truncated
as they were, reward 0). step before reset → not_reset.
Terminal reward: train mode → returned; eval mode → reward 0.0, info `{}`; the grader keeps the real result
(`grader_report()`). Train-mode terminal info: `{"metrics":{precision,recall,f1,tp,fp,fn}, "reward_shared":r,
"reward_experimental":r2, "inspection_steps":n, "invalid_submission": bool}` (counts only, no ids).
`inspection_steps` = number of steps excluding the submit (a step that hits the step/time limit counts). Invalid
submission structure (incl. missing / non-object `args`, args keys other than exactly `findings`, absurd nesting) ⇒ reward −1,
terminated, error kept in grader log (`grader_report()["errors"]`), the policy observation only gets
`{"code":"invalid_args","message":"submission rejected"}` in eval and a short reason in train.
Reset fully isolates state (new episode object, cleared trajectory). The env holds truth; it never puts it
in observations/info. Trajectory record per step: `{episode_id, step, action, observation, reward,
terminated, truncated}`; reset is step 0 with action null.

## 5. Calculator (`calculator.py`)
`evaluate(expr: str) -> str`: parse with `ast` (mode eval), allow only Expression, BinOp(+,-,*,/,//,%,**
with integer exponent |e|≤12), UnaryOp(+,-), numeric Constant (int/float literal is converted via its source
text to Decimal — never float math), parentheses. Function calls whitelist: `round(x, n)` (half_even), `abs`,
`min`, `max`. Everything else (names, attributes, strings, subscripts, lambdas, comprehensions, very long
input > 512 chars, nesting depth > 50) → `CalculatorError`. Decimal context prec=40; division by zero →
error. Result as normalized plain decimal string (no exponent notation).

## 6. Matcher & reward (`matcher.py`) — published, deterministic
```python
def validate_submission(sub, episode_id) -> tuple[bool, str|None, dict|None]
def normalize_finding(f) -> dict
def eligible(f_norm, issue, public_ids) -> tuple[bool, str]       # reason code for diagnostics
def match(findings_norm, issues, public_ids) -> dict              # {"pairs":[(fi, issue_id)], "tp","fp","fn","reasons":[...]}
def prf(tp, fp, fn, n_truth, n_findings) -> dict
def shared_reward(f1, inspection_steps) -> float                  # clip(2*F1 - 1 - 0.002*steps, -1, 1)
def experimental_reward(...) -> float                             # shown separately, never replaces shared
def score(submission, truth, public_ids, inspection_steps, episode_id) -> dict
```
Schema: object with exactly-known keys `episode_id` (must equal) and `findings` (list ≤30); each finding has
`type` (one of ISSUE_TYPES after normalisation), `affected_ids` (non-empty list of str, ≤10), 
`claimed_amount_minor` (int, not bool, ≥0), `currency` (str), `evidence_ids` (list of str ≤10),
`explanation` (str ≤1000). Missing/extra/wrongly typed field ⇒ invalid ⇒ reward −1.
Normalisation: strip whitespace; type lowercased; currency uppercased; ids as-is (case-sensitive) after strip;
affected/evidence compared as sets.
Eligibility (finding f vs issue t): type equal; set(affected)==set(t.affected_ids); currency==t.currency;
|amount−expected| ≤ t.tolerance_minor; every evidence id ∈ public_ids; ∃ set S in t.evidence_sets with S ⊆ evidence.
Matching: maximum-cardinality bipartite matching (Kuhn's augmenting paths), findings visited in submitted
order, issues tried in sorted issue_id order ⇒ deterministic. Unmatched findings (incl. repeats) = FP,
unmatched issues = FN. Metrics: P=TP/(TP+FP), R=TP/(TP+FN) (0 if denominator 0); F1=2TP/(2TP+FP+FN);
both empty ⇒ F1=1 (precision = recall = 0, i.e. 0/0 → 0); truth empty & findings non-empty ⇒ F1=0. Explanation text never affects score.
Experimental reward: soft-F1 where a finding matching type+affected_ids but failing amount or evidence counts
0.25 TP (still 1-1), minus same step cost; reported separately as `reward_experimental`.

## 7. Out-of-process grader (`server.py`, `client.py`)
`python -m auditgym.server --mode eval|train [--log-dir DIR]` reads JSON lines on stdin
(`{"cmd":"reset","seed":..,"config":..,"options":..}`, `{"cmd":"step","action":{..}}`,
`{"cmd":"grader_report"}` (allowed only after the episode ended, and only when started with
`--allow-grader-report`, which the runner uses AFTER the policy finished), `{"cmd":"close"}`). `reset` and
`grader_report` are harness commands requiring the per-process control token (env `AUDITGYM_CONTROL_TOKEN`)
held by `client.RemoteAuditEnv`; `step` is the only agent command, and the server writes one
JSON line per reply. Only this separate-process path is an isolation boundary; in-process `AuditEnv` and the
`AgentHandle` (which exposes `step` only) are trusted-code only. Holdout truth is generated on the fly and never
stored; a closed run mounts `grader_data/` only into the grader's container (see README, "Trust boundary and
deployment"). The 60 s limit is enforced cooperatively on each step and by a server-side watchdog. `client.RemoteAuditEnv(mode, ...)` mirrors `reset/step/close`, spawns
`[sys.executable, "-m", "auditgym.server", ...]`; the agent-side process never imports truth. Grader logs
(per-episode report JSON) go to `--log-dir`, not to the agent.

## 8. Rules engine & validator (`rules.py`, `validate.py`)
`rules.find_violations(public_episode) -> list[issue-like dicts]` — an INDEPENDENT implementation (must not
import generator modules) of the policy: recompute expected postings for every invoice/payment from docs +
fx + policy and compare with ledger; report duplicate_invoice / payment_misallocation / fx_error /
period_cutoff with type, affected_ids, amount (base minor). `rules.check_evidence(issue, evidence_ids,
public_episode)` recomputes the amount using ONLY records in the evidence set (+manifest/policy, + full fx
table for fallback lookup) and returns (ok, amount).
`validate.validate_episode(ep) -> dict` checks: (a) clean snapshot: Σdebit=Σcredit, every posting group
balances, rules engine finds 0 violations; (b) final records: still balanced, rules violations == truth
issues exactly (type, affected set, amount within tolerance); (c) every evidence set of every issue exists
in public ids, is ≤10 ids and passes `check_evidence`; (d) no truth/provenance tokens (issue ids, the word
"iss_", scenario names) appear in public files; (e) counts in manifest match; sizes in range.

## 9. Policies (`policies/`)
Interface: `class Policy: name; def reset(self, seed:int, first_obs:dict) -> None; def act(self, obs) -> dict`.
- `random_explorer` (fixed seed): ~15–30 random list/get/get_document/calculator actions using ids it has
  seen, then submits 0–3 random findings built from seen ids/types (random amounts, base currency).
- `reconcile` (simple sweep): pages through all tables, then naive checks: same entity+total+currency within
  10 days ⇒ duplicate (falls for recurring); payment AR credits < amount ⇒ misallocation (falls for
  prepayment/overpayment); recompute FX using naive issued_at date string (ignoring offset) ⇒ fx_error;
  cutoff by comparing date strings (ignoring offsets). Evidence: the ids it used. Gets partial credit.
- `flag_all` (attack): lists tables, then submits up to 30 findings covering every type over many ids with
  amounts copied from records and all-plausible evidence.
- `oracle` (GRADER ONLY): constructed with the truth by the runner inside the grader context; submits
  exactly the truth issues with their first evidence set; 0 inspection steps. Proves max reward
  (≈ +1.0) — not agent ability.

## 10. Runner & check (`runner.py`, `check.py`)
`runner.run_episode(policy, seed, config, mode, options, remote=True) -> (trajectory, grader_report, wall_s)`.
`python -m auditgym check [--quick] [--out reports/]` (the ONE official command, offline):
1. run pytest if importable (else skip with notice) ;
2. regenerate train/dev episodes from `configs/splits.json`, compare public hash with committed
   `episodes/` (if present) ⇒ reproducibility;
3. validate 20 holdout episodes (5 clean, 5 interaction, 10 mixed; seeds+config in splits.json) with
   `validate.py`;
4. run the 4 baselines on those 20 in eval mode via the remote env ⇒ table of P, R, F1, shared return,
   experimental return, steps, wall time per policy (mean + bootstrap 95% CI) ;
5. 20 paired checks: (rename+permute) ⇒ oracle reward identical and answer maps; replay same seed+actions
   ⇒ identical trajectory hash;
6. leak scan over all recorded observations (issue ids, "expected_amount", "evidence_sets", "truth",
   "provenance", scenario names, seed values) ;
7. reward property checks (§11);
8. write `reports/check_report.json` + `reports/check_report.md`; exit 0 iff all hard checks pass.
`python -m auditgym generate` writes episodes/{train,dev}/, grader_data/{train,dev}/ and
`trajectories/trajectories.jsonl` (4 policies on train+dev, train mode). `python -m auditgym demo` runs the
reconcile policy on one episode and prints a readable step log.

## 11. Required reward properties (tests + check)
complete correct answer with evidence > silence on books with issues; silence > false accusations on clean
books; duplicates of a correct finding give no gain; flag-all scores badly (< 0); step economy only slightly
separates equal answers (≤0.002/step); guessed ids, unsupported/non-existent evidence, wrong currency,
re-submit after end, malformed input, timeout/truncation — all handled as specified.
