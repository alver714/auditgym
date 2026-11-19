"""Shared contracts for AuditGym. Every module imports names from here.

Do not change values without bumping SCHEMA_VERSION / ENV_VERSION.
"""

SCHEMA_VERSION = "1.0.0"
ENV_VERSION = "auditgym-1.0.0"

# ---------------------------------------------------------------- issue types
DUPLICATE_INVOICE = "duplicate_invoice"
PAYMENT_MISALLOCATION = "payment_misallocation"   # missing or partial allocation
FX_ERROR = "fx_error"                              # wrong rate date or wrong rate
PERIOD_CUTOFF = "period_cutoff"                    # posted to wrong reporting period
ISSUE_TYPES = (DUPLICATE_INVOICE, PAYMENT_MISALLOCATION, FX_ERROR, PERIOD_CUTOFF)

# interaction scenarios (each injects >=2 linked issues with one root cause)
INTERACTION_FX_PARTIAL_BOUNDARY = "fx_partial_payment_at_boundary"   # -> payment_misallocation + fx_error
INTERACTION_CUTOFF_FX_DATE = "cutoff_with_wrong_rate_date"            # -> period_cutoff + fx_error
INTERACTIONS = (INTERACTION_FX_PARTIAL_BOUNDARY, INTERACTION_CUTOFF_FX_DATE)

# distractor kinds (correct, but suspicious-looking)
DISTRACTOR_SPLIT_PAYMENT = "split_payment"
DISTRACTOR_REFUND = "overpayment_refund"
DISTRACTOR_RECURRING = "legit_recurring_amount"
DISTRACTOR_TZ_BOUNDARY = "timezone_boundary_ok"
DISTRACTOR_FX_GAIN_LOSS = "realized_fx_gain_loss"
DISTRACTOR_PREPAYMENT = "unreferenced_prepayment"
DISTRACTORS = (
    DISTRACTOR_SPLIT_PAYMENT, DISTRACTOR_REFUND, DISTRACTOR_RECURRING,
    DISTRACTOR_TZ_BOUNDARY, DISTRACTOR_FX_GAIN_LOSS, DISTRACTOR_PREPAYMENT,
)

# ---------------------------------------------------------------- tables
TABLE_LEDGER = "ledger"
TABLE_INVOICES = "invoices"
TABLE_PAYMENTS = "payments"
TABLE_FX = "fx"
TABLES = (TABLE_LEDGER, TABLE_INVOICES, TABLE_PAYMENTS, TABLE_FX)

LEDGER_FIELDS = ("row_id", "timestamp", "account", "debit_minor", "credit_minor",
                 "currency", "entity_id", "invoice_id", "payment_id")
INVOICE_FIELDS = ("doc_id", "invoice_id", "entity_id", "issued_at", "due_at",
                  "currency", "total_minor", "line_items")
PAYMENT_FIELDS = ("doc_id", "payment_id", "timestamp", "currency", "amount_minor",
                  "invoice_refs")
FX_FIELDS = ("date", "source_currency", "target_currency", "rate")

# primary key used by get_record(table, id)
PRIMARY_KEY = {
    TABLE_LEDGER: "row_id",
    TABLE_INVOICES: "invoice_id",
    TABLE_PAYMENTS: "payment_id",
    TABLE_FX: "fx_id",          # synthetic: "fx:<date>:<source>:<target>"
}

def fx_id(date: str, source: str, target: str) -> str:
    return f"fx:{date}:{source}:{target}"

# ---------------------------------------------------------------- accounts
ACC_AR = "1200_accounts_receivable"
ACC_CASH = "1000_cash"
ACC_REVENUE = "4000_revenue"
ACC_VAT = "2200_vat_payable"
ACC_UNAPPLIED = "2300_unapplied_cash"
ACC_FX = "7900_fx_gain_loss"
ACC_EXPENSE_RENT = "6100_rent_expense"
ACC_EXPENSE_PAYROLL = "6200_payroll_expense"
ACC_EXPENSE_SOFTWARE = "6300_software_expense"
ACC_AP = "2000_accounts_payable"
ACCOUNTS = (ACC_AR, ACC_CASH, ACC_REVENUE, ACC_VAT, ACC_UNAPPLIED, ACC_FX,
            ACC_EXPENSE_RENT, ACC_EXPENSE_PAYROLL, ACC_EXPENSE_SOFTWARE, ACC_AP)

# ---------------------------------------------------------------- currencies
# ISO-4217 minor-unit exponent
CURRENCY_EXPONENT = {"USD": 2, "EUR": 2, "GBP": 2, "KZT": 2, "CHF": 2, "JPY": 0, "AED": 2}

# ---------------------------------------------------------------- env limits
MAX_STEPS = 100            # includes submit
PAGE_LIMIT_MAX = 50
EPISODE_TIME_LIMIT_S = 60.0
MAX_FINDINGS = 30
MAX_EVIDENCE_IDS = 10
MAX_AFFECTED_IDS = 10
MAX_EXPLANATION_CHARS = 1000
MAX_CALC_EXPR_CHARS = 512
STEP_COST = 0.002

ACTIONS = ("list_records", "get_record", "get_document", "calculator", "submit")

# bounded error codes returned in observations (never contain hidden data)
ERR_UNKNOWN_ACTION = "unknown_action"
ERR_INVALID_ARGS = "invalid_args"
ERR_UNKNOWN_TABLE = "unknown_table"
ERR_UNKNOWN_ID = "unknown_id"
ERR_INVALID_CURSOR = "invalid_cursor"
ERR_CALC = "calculator_error"
ERR_EPISODE_OVER = "episode_over"
ERR_NOT_RESET = "not_reset"
ERROR_CODES = (ERR_UNKNOWN_ACTION, ERR_INVALID_ARGS, ERR_UNKNOWN_TABLE, ERR_UNKNOWN_ID,
               ERR_INVALID_CURSOR, ERR_CALC, ERR_EPISODE_OVER, ERR_NOT_RESET)

MODE_TRAIN = "train"   # terminal reward returned to the policy
MODE_EVAL = "eval"     # reward withheld (0.0) and diagnostics kept by the grader
