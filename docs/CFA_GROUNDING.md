# CFA grounding

AuditGym's design is anchored in the CFA Program Level I curriculum (2025). Page numbers are
**book page numbers** printed in the volumes (PDF page = book page + 12 for Volume 4 FSA).

> **Scope note.** The grounding is conceptual. It explains why each issue type and distractor
> corresponds to a recognised area of financial statement analysis. AuditGym itself is a
> programming task with a deterministic grader. It is not an audit, not an audit opinion, and
> not investment or professional advice. The policy rules in `policy.json` are simplified,
> self-contained rules of a fictional company. They are not a reproduction of IFRS 15, IAS 21
> or any auditing standard.

## Mapping table

| AuditGym element | Curriculum source | What the curriculum gives us | How we use it |
|---|---|---|---|
| Revenue and AR recognised at `issued_at`; `period_cutoff` issue | Vol. 4 FSA, LM2 *Analyzing Income Statements*, "Revenue Recognition" (general principles and the five-step standard), pp. 39-45 | Revenue is recognised when the performance obligation is satisfied, not when cash moves. A receivable and revenue arise together when no contingency remains. | The policy fixes one recognition event (invoice `issued_at`). A period belongs to an invoice iff that instant lies in `[period_start, period_end]`. Posting an invoice from the next period inside this one is a cutoff error. |
| Same, why premature recognition is a risk | Vol. 4 FSA, LM10 *Financial Reporting Quality*, "Warning Signs: Pay Attention to Revenue", pp. 372-373; "Biased Accounting Choices", pp. 323-336 | Bias in recognition has a timing dimension. Early revenue recognition is a recurring manipulation channel. Receivables growing faster than revenue is a signal. | `period_cutoff` is the mechanical form of "revenue booked too early". Cutoff cases that look in-period as date strings but fall outside as instants test whether the agent reasons about instants. |
| AR roll-forward; AR subledger vs cash collection; `payment_misallocation` | Vol. 4 FSA, LM4 *Analyzing Statements of Cash Flows I*, "Linkages between the Financial Statements" and "Linkages Between Current Assets and Current Liabilities", pp. 125-131 (AR roll-forward: opening AR + revenue - cash collected = closing AR) | Statements are linked through working-capital accounts. A collection that does not reduce AR leaves the ledger internally inconsistent with the cash side. | Each payment must credit AR for exactly the invoice references it settles. A partial or missing AR credit parks cash in `2300_unapplied_cash`. |
| Receivable and credit management | Vol. 3 Corporate Issuers, LM4 *Working Capital and Liquidity*, "Cash Conversion Cycle" (pp. 101-111, receivables / DSO) and "Managing Working Capital and Liquidity" (pp. 123-127, receivables policy) | Receivable balances, collection discipline and delinquency are managed and monitored. | Motivates why unapplied or misapplied cash matters (it distorts DSO and collections reporting). Also motivates the legitimate distractors: split payments, overpayments, prepayments. |
| FX conversion at the transaction date; `fx_error` | Vol. 4 FSA, LM2 "Changes in Scope and Exchange Rates", p. 68 (exchange-rate effects must be separated from underlying performance); Vol. 4 FSA, LM5 *Analyzing Statements of Cash Flows II*, cash-flow statement line "effect of exchange rate changes on cash" (statement examples, pp. 159-160) | Level I only introduces that exchange rates change reported amounts and need isolating. | **Weak grounding, stated honestly.** Level I does not teach transaction-level translation or realised FX gain/loss mechanics (those belong to later levels). The AuditGym FX policy (rate date in company offset, fallback to earlier date, half-even rounding) is a self-contained engineering rule, not a curriculum rule. |
| Realised FX gain/loss distractor (`realized_fx_gain_loss`) | Same as above; Vol. 7 Derivatives, LM1 *Derivative Instruments and Derivative Market Features*, "Derivative Underlyings" (currency as an underlying; forward contracts) | Currency rates are a standard source of gains and losses, and a rate difference between invoice date and payment date is normal. | A foreign invoice paid at a different rate legitimately books the difference to `7900_fx_gain_loss`. A good agent must not flag it. |
| Financial reporting quality: spectrum and warning signs; issue and distractor design | Vol. 4 FSA, LM10, "Conceptual Overview" (quality spectrum from unbiased to fabricated) pp. 319-321; "Warning Signs" pp. 372-378; "Accounting Choices and Estimates" pp. 353-368 | Reporting spans a continuum. Analysts should treat anomalies as signals to investigate, not as proof. | Duplicate billing and misallocated cash are the "low-quality end" patterns (error or inflated revenue). Distractors are the "looks odd but is fine" patterns. A warning sign is not a finding until evidence confirms it. |
| Auditor role and evidence; evidence sets | Vol. 4 FSA, LM1 *Introduction to Financial Statement Analysis*, "Auditor's Reports", pp. 22-27; LM10 "Mechanisms That Discipline Financial Reporting Quality: Auditors", pp. 341-345 | An independent party examines records and expresses a conclusion. The conclusion has to be supported by what was examined. | Each finding must cite the specific ledger rows, documents and rates that support it. Evidence sets in `truth.json` list sufficient records (for `payment_misallocation` both a canonical maximal and the minimal sets), and the matcher requires one full set. |
| Sampling, estimation and confidence intervals; diagnostic CIs in the check report | Vol. 1 Quantitative Methods, LM7 *Estimation and Inference*, "Central Limit Theorem and Inference" and "Standard Error of the Sample Mean", pp. 205-209 | A sample statistic has a sampling distribution. Its spread tells how much to trust the estimate. | Policy scores over 20 holdout episodes are a small sample. Reported means carry intervals so differences between policies are not over-read. |
| Bootstrap | Vol. 1 Quantitative Methods, LM6 *Simulation Methods*, "Bootstrapping", pp. 189-193; LM7, "Bootstrapping and Empirical Sampling Distributions", pp. 209-213 | Resample the observed data with replacement, recompute the statistic many times, read an interval from the empirical distribution. No distributional assumption needed. | `check` bootstraps the per-episode returns (seeded, so the report is reproducible) to give a 95% interval for each policy's mean. |

## Notes on use

* **Accrual vs cash.** LM4 stresses that accrual accounting separates revenue from cash receipt.
  AuditGym keeps the two streams separate (invoices vs payments) and asks whether they were
  linked correctly.
* **Warning sign vs finding.** LM10 frames anomalies as prompts for deeper work. The reward
  design follows this. Flagging a distractor is a false positive, and flagging without
  evidence does not match.
* **What is not claimed.** We do not claim that any scenario reproduces a standard's detailed
  requirements, that the scoring rule reflects professional practice, or that performance here
  predicts audit quality.
