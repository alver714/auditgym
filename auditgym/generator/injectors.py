"""Issue injectors (4 types + 2 interactions). Each mutates the posted ledger / invoices, records its
causal change, and a draft truth issue.  Truth amounts and evidence sets are computed afterwards by
`finalize_truth` from the FINAL records (so every truth issue is independently verifiable).
"""
from __future__ import annotations

import copy
from datetime import timedelta
from decimal import Decimal

from ..constants import (ACC_AR, ACC_CASH, DUPLICATE_INVOICE, FX_ERROR, INTERACTION_CUTOFF_FX_DATE,
                         INTERACTION_FX_PARTIAL_BOUNDARY, INTERACTIONS, ISSUE_TYPES, PAYMENT_MISALLOCATION,
                         PERIOD_CUTOFF, MAX_EVIDENCE_IDS, fx_id)
from ..money import format_rate
from ..rng import sub_rng
from .world import World, is_bd, parse_ts


class InjectionError(Exception):
    """Requested injection infeasible in this attempt; the caller retries with a new attempt index."""


class Ctx:
    def __init__(self, w: World, seed: int):
        self.w = w
        self.r = sub_rng(seed, "inject", w.attempt)
        self.used: set[str] = set()
        self.drafts: list[dict] = []
        self.changes: list[dict] = []

    def add_issue(self, type_, affected, scenario=None, **info) -> int:
        self.drafts.append({"key": len(self.drafts), "type": type_, "affected_ids": list(affected),
                            "scenario": scenario, "info": info})
        return len(self.drafts) - 1

    def add_change(self, keys, injector, root_cause, description, added=(), removed=(), modified=()):
        self.changes.append({"issue_keys": list(keys), "injector": injector, "root_cause": root_cause,
                             "description": description, "records_added": list(added),
                             "records_removed": list(removed), "records_modified": list(modified)})


# ----------------------------------------------------------------------------- helpers
def _scale_ok(w: World, diff: int, correct: int) -> bool:
    tol = max(1, diff // 1000)
    return diff >= 2 * tol + 1 and diff >= 10 ** 2 and diff * 2000 >= correct


def _transpose(rate: Decimal, r):
    s = format_rate(rate)
    pairs = [i for i in range(len(s) - 1) if s[i].isdigit() and s[i + 1].isdigit() and s[i] != s[i + 1]]
    r.shuffle(pairs)
    for i in pairs:
        t = list(s)
        t[i], t[i + 1] = t[i + 1], t[i]
        v = Decimal("".join(t))
        if v != rate and v > 0:
            return v
    return None


def _wrong_rate_options(w: World, cur: str, correct_date: str, correct_rate: Decimal, inst,
                        cust, extra_dates, r):
    """-> list of (variant, label, rate). inst: aware instant of the event."""
    opts = []
    dates = {}
    cd = inst.astimezone(w.tz).date()
    own = inst.astimezone(cust["tz"]).date()
    utc = inst.astimezone(parse_utc()).date()
    dates["own_offset_date"] = own
    dates["utc_date"] = utc
    nxt = cd + timedelta(days=1)
    prv = cd - timedelta(days=1)
    while not is_bd(nxt):
        nxt += timedelta(days=1)
    while not is_bd(prv):
        prv -= timedelta(days=1)
    dates["next_business_day"] = nxt
    dates["previous_business_day"] = prv
    for label, d in extra_dates.items():
        dates[label] = d
    for label, d in dates.items():
        ds, rate = w.rate_for(cur, d.isoformat())
        if rate != correct_rate:
            opts.append(("wrong_date", label, rate))
    t = _transpose(correct_rate, r)
    if t is not None:
        opts.append(("wrong_rate", "transposed_digits", t))
    fdates = w.fx_dates[cur]
    i = fdates.index(correct_date)
    j = max(0, i - r.randint(3, 8))
    if fdates[j] != correct_date:
        opts.append(("wrong_rate", "stale_rate", w.fx[cur][fdates[j]]))
    return opts


def parse_utc():
    from datetime import timezone
    return timezone.utc


def _row_ids(w: World, kind, key):
    return w.group_row_ids(kind, key)


# ----------------------------------------------------------------------------- duplicate_invoice
DUP_PAID_SHARE = 0.35        # share of injected duplicates that receive a (partial) payment
DOC_SOFT_LIMIT = 190         # no extra payment document beyond this many documents (hard cap 200)


def _pay_duplicate(ctx: Ctx, dup: dict, cust: dict):
    """A correct payment (full or 30-90 %) for the duplicate invoice, after it was issued and inside the period."""
    w, r = ctx.w, ctx.r
    di = w.inv_instant(dup)
    room = int((w.period_end - di).total_seconds()) - 60
    if room < 7200:
        return None
    inst = di + timedelta(seconds=r.randint(3600, min(room, 20 * 86400)))
    amt = dup["total_minor"]
    if r.random() < 0.4:
        amt = max(1, amt * r.randint(30, 90) // 100)
    return w.new_payment(cust, inst, amt, [{"invoice_id": dup["invoice_id"], "amount_minor": amt}],
                         kind="dup_paid")


def inject_duplicate(ctx: Ctx):
    w, r = ctx.w, ctx.r
    cands = [i for i in w.invoices.values()
             if w.inv_meta[i["invoice_id"]]["kind"] == "ordinary" and i["invoice_id"] not in ctx.used
             and i["invoice_id"] not in w.reserved and w.in_period(w.inv_instant(i))
             and w.inv_instant(i) + timedelta(hours=2) <= w.period_end - timedelta(minutes=1)]
    if not cands:
        raise InjectionError("no duplicate candidate")
    orig = r.choice(cands)
    oi = w.inv_instant(orig)
    room = int((w.period_end - oi).total_seconds()) - 60
    delta = r.randint(3600, max(3600, min(5 * 86400, room)))
    cust = w.customers[orig["entity_id"]]
    dup = w.new_invoice(cust, oi + timedelta(seconds=delta), kind="dup", lines=copy.deepcopy(orig["line_items"]))
    # same payment terms as the original
    term = w.inv_meta[orig["invoice_id"]]["term"]
    di = w.inv_instant(dup)
    dup["due_at"] = (di + timedelta(days=term)).replace(microsecond=0).isoformat()
    w.inv_meta[dup["invoice_id"]]["term"] = term
    assert dup["total_minor"] == orig["total_minor"]
    # the duplicate is the LATER one by issued_at alone, so rules._dup_order_key never reaches its id tie-break
    if w.inv_instant(dup) <= w.inv_instant(orig):
        raise InjectionError("duplicate not strictly later than the original")
    added, removed, modified = w.post_invoice(dup)
    ctx.used.update([orig["invoice_id"], dup["invoice_id"]])
    paid_note = ""
    # a share of duplicates is (partly) paid by the customer; the evidence set stays {orig doc, dup doc, AR debit}
    if r.random() < DUP_PAID_SHARE and w.doc_count() < DOC_SOFT_LIMIT:
        pay = _pay_duplicate(ctx, dup, cust)
        if pay is not None:
            pa, pr_, pm = w.post_payment(pay)
            ctx.used.add(pay["payment_id"])
            added += [pay["doc_id"], pay["payment_id"]] + pa
            removed += pr_
            modified += pm
            paid_note = (f" The customer then {'partly ' if pay['amount_minor'] < dup['total_minor'] else ''}paid the "
                         f"duplicate with payment {pay['payment_id']}.")
    k = ctx.add_issue(DUPLICATE_INVOICE, [dup["invoice_id"]], orig_invoice_id=orig["invoice_id"])
    ctx.add_change([k], DUPLICATE_INVOICE, "invoice_reissued",
                   f"Invoice {orig['invoice_id']} was re-issued as {dup['invoice_id']} with identical "
                   f"entity, currency and line items and posted again." + paid_note,
                   added=[dup["invoice_id"], dup["doc_id"]] + added, removed=removed, modified=modified)


# ----------------------------------------------------------------------------- payment misallocation
def _misalloc_candidates(ctx: Ctx, foreign=None):
    w = ctx.w
    out = []
    for p in w.payments.values():
        pid = p["payment_id"]
        if pid in ctx.used or pid in w.reserved or w.pay_meta[pid]["kind"] not in ("ordinary", "boundary"):
            continue
        refs = p["invoice_refs"]
        if not refs or p["amount_minor"] <= 0 or sum(x["amount_minor"] for x in refs) != p["amount_minor"]:
            continue
        isf = p["currency"] != w.base
        if foreign is not None and isf != foreign:
            continue
        if len(refs) > (2 if isf else 4):
            continue
        out.append(p)
    return out


def _plan_misalloc(ctx: Ctx, pay):
    """-> (ar_credit_pcc override dict, extra_unapplied_pcc, mode, ref) or None if too small."""
    w, r = ctx.w, ctx.r
    refs = pay["invoice_refs"]
    big = []
    for ref in refs:
        inv = w.invoices[ref["invoice_id"]]
        _, ir = w.inv_rate(inv)
        if w.conv(ref["amount_minor"], pay["currency"], ir) >= 2000:
            big.append(ref)
    if not big:
        return None
    ref = r.choice(big)
    if r.random() < 0.5:
        pct = r.randint(40, 90)
        credited = ref["amount_minor"] * pct // 100
        mode = "partial"
    else:
        credited, mode = 0, "missing"
    return {ref["invoice_id"]: credited}, ref["amount_minor"] - credited, mode, ref


def inject_misallocation(ctx: Ctx):
    w, r = ctx.w, ctx.r
    cands = _misalloc_candidates(ctx)
    r.shuffle(cands)
    for pay in cands:
        plan = _plan_misalloc(ctx, pay)
        if plan is None:
            continue
        credits, extra, mode, ref = plan
        added, removed, modified = w.post_payment(pay, ar_credit_pcc=credits, extra_unapplied_pcc=extra)
        ctx.used.add(pay["payment_id"])
        k = ctx.add_issue(PAYMENT_MISALLOCATION, [pay["payment_id"]])
        ctx.add_change([k], PAYMENT_MISALLOCATION, f"payment_allocation_{mode}",
                       f"AR credit for invoice {ref['invoice_id']} from payment {pay['payment_id']} was "
                       f"posted {'partially' if mode == 'partial' else 'not at all'}; the uncredited part "
                       f"went to unapplied cash.", added=added, removed=removed, modified=modified)
        return
    raise InjectionError("no misallocation candidate")


# ----------------------------------------------------------------------------- fx_error
def _fx_invoice_plan(ctx: Ctx, inv, scale_ctx=True):
    """Pick a wrong rate for a foreign invoice -> (variant,label,rate) or None."""
    w, r = ctx.w, ctx.r
    cur = inv["currency"]
    cd, crate = w.inv_rate(inv)
    cust = w.customers[inv["entity_id"]]
    inst = w.inv_instant(inv)
    due = parse_ts(inv["due_at"]).astimezone(w.tz).date()
    correct = w.conv(inv["total_minor"], cur, crate)
    opts = _wrong_rate_options(w, cur, cd, crate, inst, cust, {"due_date": due}, r)
    r.shuffle(opts)
    want = "wrong_date" if r.random() < 0.5 else "wrong_rate"
    opts.sort(key=lambda o: o[0] != want)
    for var, label, rate in opts:
        diff = abs(w.conv(inv["total_minor"], cur, rate) - correct)
        if _scale_ok(w, diff, correct):
            return var, label, rate
    return None


def _fx_payment_plan(ctx: Ctx, pay, forced=None):
    w, r = ctx.w, ctx.r
    cur = pay["currency"]
    cd, crate = w.pay_rate(pay)
    inst = parse_ts(pay["timestamp"])
    own_tz = inst.tzinfo
    c = {"tz": own_tz}
    correct = w.conv(pay["amount_minor"], cur, crate)
    extra = {}
    refs = pay["invoice_refs"]
    if refs:
        extra["first_invoice_date"] = w.cdate(w.inv_instant(w.invoices[refs[0]["invoice_id"]]))
    opts = _wrong_rate_options(w, cur, cd, crate, inst, c, extra, r)
    r.shuffle(opts)
    if forced:
        opts = [o for o in opts if o[1] == forced]
    else:
        want = "wrong_date" if r.random() < 0.5 else "wrong_rate"
        opts.sort(key=lambda o: o[0] != want)
    for var, label, rate in opts:
        diff = abs(w.conv(pay["amount_minor"], cur, rate) - correct)
        if _scale_ok(w, diff, correct):
            return var, label, rate
    return None


def inject_fx_error(ctx: Ctx):
    w, r = ctx.w, ctx.r
    inv_c = [i for i in w.invoices.values()
             if w.inv_meta[i["invoice_id"]]["kind"] == "ordinary" and i["currency"] != w.base
             and i["invoice_id"] not in ctx.used and i["invoice_id"] not in w.reserved
             and w.in_period(w.inv_instant(i))]
    pay_c = [p for p in w.payments.values()
             if p["currency"] != w.base and p["amount_minor"] > 0 and p["payment_id"] not in ctx.used
             and p["payment_id"] not in w.reserved and w.pay_meta[p["payment_id"]]["kind"] in ("ordinary", "boundary")]
    r.shuffle(inv_c)
    r.shuffle(pay_c)
    order = ["inv", "pay"] if r.random() < 0.5 else ["pay", "inv"]
    for which in order:
        if which == "inv":
            for inv in inv_c:
                plan = _fx_invoice_plan(ctx, inv)
                if plan is None:
                    continue
                var, label, rate = plan
                added, removed, modified = w.post_invoice(inv, rate=rate)
                ctx.used.add(inv["invoice_id"])
                k = ctx.add_issue(FX_ERROR, [inv["invoice_id"]], kind="invoice")
                ctx.add_change([k], FX_ERROR, f"fx_{var}:{label}",
                               f"Invoice {inv['invoice_id']} converted with a wrong rate ({var}: {label}); "
                               f"AR, VAT and revenue rows all use it.",
                               added=added, removed=removed, modified=modified)
                return
        else:
            for pay in pay_c:
                plan = _fx_payment_plan(ctx, pay)
                if plan is None:
                    continue
                var, label, rate = plan
                added, removed, modified = w.post_payment(pay, cash_rate=rate)
                ctx.used.add(pay["payment_id"])
                k = ctx.add_issue(FX_ERROR, [pay["payment_id"]], kind="payment")
                ctx.add_change([k], FX_ERROR, f"fx_{var}:{label}",
                               f"Cash row of payment {pay['payment_id']} converted with a wrong rate "
                               f"({var}: {label}); the fx gain/loss row absorbs the difference.",
                               added=added, removed=removed, modified=modified)
                return
    raise InjectionError("no fx_error candidate")


# ----------------------------------------------------------------------------- period_cutoff
def _post_period(ctx: Ctx, foreign=None):
    w = ctx.w
    return [i for i in w.invoices.values()
            if w.inv_meta[i["invoice_id"]]["kind"] == "post_period" and i["invoice_id"] not in ctx.used
            and (foreign is None or (i["currency"] != w.base) == foreign)]


def _pre_period(ctx: Ctx):
    w = ctx.w
    return [i for i in w.invoices.values()
            if w.inv_meta[i["invoice_id"]]["kind"] == "pre_period" and i["invoice_id"] not in ctx.used]


def inject_cutoff(ctx: Ctx):
    """Post an out-of-period invoice inside the period.  Directions / variants:
      * after period_end (plain, `tricky` = date string still shows the last day, or exactly period_end + 1 s);
      * mirror: issued before period_start, posted inside (also a `tricky` and an exact period_start - 1 s case)."""
    w, r = ctx.w, ctx.r
    post = _post_period(ctx)
    pre = _pre_period(ctx)
    if not post and not pre:
        raise InjectionError("no out-of-period invoice left")
    pend = w.period_end.date().isoformat()
    edge_end = [i for i in post if w.inv_instant(i) == w.period_end + timedelta(seconds=1)]
    edge_start = [i for i in pre if w.inv_instant(i) == w.period_start - timedelta(seconds=1)]
    mirror = bool(pre) and (not post or r.random() < 0.3)
    if mirror:
        inv = r.choice(edge_start) if edge_start and r.random() < 0.4 else r.choice(pre)
        ts = w.cts(w.period_start + timedelta(seconds=r.randint(0, 3 * 86400 - 1)))
        where, how = "before period start", "at the start of the period"
    else:
        tricky = [i for i in post if i["issued_at"][:10] <= pend]
        if edge_end and r.random() < 0.3:
            inv = r.choice(edge_end)
        elif tricky and r.random() < 0.5:
            inv = r.choice(tricky)
        else:
            rest = [i for i in post if i not in tricky] or post
            inv = r.choice(rest)
        ts = w.cts(w.period_end - timedelta(seconds=r.randint(0, 3 * 86400 - 1)))
        where, how = "after period end", "inside the period"
    added, removed, modified = w.post_invoice(inv, ts=ts)
    w.inv_meta[inv["invoice_id"]]["kind"] = "cutoff"
    ctx.used.add(inv["invoice_id"])
    k = ctx.add_issue(PERIOD_CUTOFF, [inv["invoice_id"]])
    ctx.add_change([k], PERIOD_CUTOFF, "posted_in_wrong_period" if mirror else "posted_in_prior_period",
                   f"Invoice {inv['invoice_id']} issued {inv['issued_at']} ({where}) was posted "
                   f"at {ts}, {how}.", added=added, removed=removed, modified=modified)


# ----------------------------------------------------------------------------- interactions
def inject_fx_partial_boundary(ctx: Ctx):
    w, r = ctx.w, ctx.r
    cands = []
    for p in _misalloc_candidates(ctx, foreign=True):
        inst = parse_ts(p["timestamp"])
        company_d = inst.astimezone(w.tz).date()
        own_d = inst.date()      # date as written in the payment's own offset
        if company_d != own_d and is_bd(company_d) and is_bd(own_d):
            cands.append(p)
    r.shuffle(cands)
    for pay in cands:
        plan = _plan_misalloc(ctx, pay)
        if plan is None:
            continue
        inst = parse_ts(pay["timestamp"])
        cd, crate = w.pay_rate(pay)
        own_ds, own_rate = w.rate_for(pay["currency"], inst.date().isoformat())
        if own_rate == crate:
            continue
        correct = w.conv(pay["amount_minor"], pay["currency"], crate)
        diff = abs(w.conv(pay["amount_minor"], pay["currency"], own_rate) - correct)
        if not _scale_ok(w, diff, correct):
            continue
        credits, extra, mode, ref = plan
        added, removed, modified = w.post_payment(pay, cash_rate=own_rate, ar_credit_pcc=credits,
                                                  extra_unapplied_pcc=extra)
        ctx.used.add(pay["payment_id"])
        k1 = ctx.add_issue(PAYMENT_MISALLOCATION, [pay["payment_id"]], INTERACTION_FX_PARTIAL_BOUNDARY)
        k2 = ctx.add_issue(FX_ERROR, [pay["payment_id"]], INTERACTION_FX_PARTIAL_BOUNDARY, kind="payment")
        ctx.add_change([k1, k2], INTERACTION_FX_PARTIAL_BOUNDARY, "own_offset_date_rate_and_partial_allocation",
                       f"Payment {pay['payment_id']} ({pay['timestamp']}) sits near a day boundary: it was "
                       f"booked with the rate of its own-offset date {own_ds} instead of the company-offset "
                       f"date {cd}, and its AR credit for {ref['invoice_id']} was posted {mode}.",
                       added=added, removed=removed, modified=modified)
        return
    raise InjectionError("no boundary payment candidate")


def inject_cutoff_fx_date(ctx: Ctx):
    w, r = ctx.w, ctx.r
    pend = w.period_end.date()
    cands = []
    for inv in _post_period(ctx, foreign=True):
        inst = w.inv_instant(inv)
        d = w.cdate(inst)
        if (d - pend).days > 3:        # "just after period_end"
            continue
        _, true_rate = w.inv_rate(inv)
        _, end_rate = w.rate_for(inv["currency"], pend.isoformat())
        if true_rate == end_rate:
            continue
        correct = w.conv(inv["total_minor"], inv["currency"], true_rate)
        diff = abs(w.conv(inv["total_minor"], inv["currency"], end_rate) - correct)
        if _scale_ok(w, diff, correct):
            cands.append((inv, end_rate))
    if not cands:
        raise InjectionError("no cutoff+fx candidate")
    inv, end_rate = r.choice(cands)
    ts = w.cts(w.period_end - timedelta(seconds=r.randint(0, 7200)))
    added, removed, modified = w.post_invoice(inv, ts=ts, rate=end_rate)
    w.inv_meta[inv["invoice_id"]]["kind"] = "cutoff"
    ctx.used.add(inv["invoice_id"])
    k1 = ctx.add_issue(PERIOD_CUTOFF, [inv["invoice_id"]], INTERACTION_CUTOFF_FX_DATE)
    k2 = ctx.add_issue(FX_ERROR, [inv["invoice_id"]], INTERACTION_CUTOFF_FX_DATE, kind="invoice")
    ctx.add_change([k1, k2], INTERACTION_CUTOFF_FX_DATE, "posted_at_period_end_with_period_end_rate",
                   f"Invoice {inv['invoice_id']} issued {inv['issued_at']} (just after period end) was posted "
                   f"at {ts} using the rate of the period-end date.",
                   added=added, removed=removed, modified=modified)


# ----------------------------------------------------------------------------- truth from final records
def _ar_debit_row(w: World, invoice_id: str) -> dict:
    rows = [r for r in w.ledger.values()
            if r["invoice_id"] == invoice_id and r["account"] == ACC_AR and r["debit_minor"] > 0]
    if len(rows) != 1:
        raise InjectionError(f"expected exactly one AR debit row for {invoice_id}, found {len(rows)}")
    return rows[0]


def compute_truth(w: World, d: dict) -> dict:
    t, aff = d["type"], d["affected_ids"][0]
    base = w.base
    if t in (DUPLICATE_INVOICE, PERIOD_CUTOFF):
        inv = w.invoices[aff]
        row = _ar_debit_row(w, aff)
        amount = row["debit_minor"]
        ev = [w.invoices[d["info"]["orig_invoice_id"]]["doc_id"], inv["doc_id"], row["row_id"]] \
            if t == DUPLICATE_INVOICE else [inv["doc_id"], row["row_id"]]
    elif t == FX_ERROR:
        if d["info"]["kind"] == "invoice":
            inv = w.invoices[aff]
            row = _ar_debit_row(w, aff)
            fdate, rate = w.inv_rate(inv)
            correct = w.conv(inv["total_minor"], inv["currency"], rate)
            amount = abs(row["debit_minor"] - correct)
            ev = [inv["doc_id"], row["row_id"], fx_id(fdate, inv["currency"], base)]
        else:
            pay = w.payments[aff]
            rows = [r for r in w.ledger.values()
                    if r["payment_id"] == aff and r["account"] == ACC_CASH and r["debit_minor"] > 0]
            if len(rows) != 1:
                raise InjectionError("cash row missing")
            fdate, rate = w.pay_rate(pay)
            correct = w.conv(pay["amount_minor"], pay["currency"], rate)
            amount = abs(rows[0]["debit_minor"] - correct)
            ev = [pay["doc_id"], rows[0]["row_id"], fx_id(fdate, pay["currency"], base)]
    elif t == PAYMENT_MISALLOCATION:
        pay = w.payments[aff]
        rows = [r for r in w.ledger.values() if r["payment_id"] == aff]
        expected = 0
        fxids, docs = [], []
        for ref in pay["invoice_refs"]:
            inv = w.invoices[ref["invoice_id"]]
            fdate, rate = w.inv_rate(inv)
            expected += w.conv(ref["amount_minor"], pay["currency"], rate)
            if pay["currency"] != base:
                docs.append(inv["doc_id"])
                f = fx_id(fdate, pay["currency"], base)
                if f not in fxids:
                    fxids.append(f)
        posted = sum(r["credit_minor"] for r in rows if r["account"] == ACC_AR)
        amount = expected - posted
        rows_sorted = sorted(rows, key=lambda r: (parse_ts(r["timestamp"]), r["row_id"]))
        ev = [pay["doc_id"]] + [r["row_id"] for r in rows_sorted] + docs + fxids
        # What rules.check_evidence needs to prove the finding: the payment document, every AR credit row of the
        # payment (they enter the posted sum), at least one ledger row of the payment, and -- foreign currency
        # only -- every referenced invoice document plus the fx row of its issue date.  The canonical maximal set
        # above also cites the remaining rows of the payment; the minimal set(s) drop them.  Without a posted AR
        # credit (a fully missing allocation) any ONE other row of the payment is enough, so one minimal set is
        # emitted per candidate row: the matcher then accepts exactly the subsets the independent verifier does.
        ar_ids = [r["row_id"] for r in rows_sorted if r["account"] == ACC_AR and r["credit_minor"] > 0]
        rest = [r["row_id"] for r in rows_sorted if r["row_id"] not in set(ar_ids)]
        core = [pay["doc_id"]] + ar_ids + docs + fxids
        minimal = [core] if ar_ids else [[pay["doc_id"], x] + docs + fxids for x in rest]
        sets = []
        for cand in [ev] + minimal:
            if cand not in sets and len(cand) <= MAX_EVIDENCE_IDS:
                sets.append(cand)
        if not any(m in sets for m in minimal):
            raise InjectionError("misallocation evidence set too large")
        if amount <= 0:
            raise InjectionError(f"non-positive truth amount for {t}")
        return {"expected_amount_minor": int(amount), "evidence_sets": sets}
    else:
        raise InjectionError("unknown type " + t)
    if amount <= 0:
        raise InjectionError(f"non-positive truth amount for {t}")
    if len(ev) > MAX_EVIDENCE_IDS or len(set(ev)) != len(ev):
        raise InjectionError("evidence set too large or duplicated")
    return {"expected_amount_minor": int(amount), "evidence_sets": [ev]}


def finalize_truth(w: World, drafts: list[dict], seed: int) -> list[dict]:
    """Compute amounts/evidence from final records; assign seeded-shuffled ids iss_01.."""
    order = list(range(len(drafts)))
    sub_rng(seed, "issue_order", w.attempt).shuffle(order)
    issues = []
    for n, di in enumerate(order, 1):
        d = drafts[di]
        t = compute_truth(w, d)
        exp = t["expected_amount_minor"]
        issues.append({
            "issue_id": f"iss_{n:02d}", "type": d["type"], "affected_ids": list(d["affected_ids"]),
            "expected_amount_minor": exp, "currency": w.base, "tolerance_minor": max(1, exp // 1000),
            "evidence_sets": t["evidence_sets"], "scenario": d["scenario"], "_key": d["key"],
        })
    return issues


def apply_injections(w: World, seed: int) -> Ctx:
    ctx = Ctx(w, seed)
    sp = w.spec
    for name in INTERACTIONS:
        for _ in range(sp["interactions"].get(name, 0)):
            {INTERACTION_FX_PARTIAL_BOUNDARY: inject_fx_partial_boundary,
             INTERACTION_CUTOFF_FX_DATE: inject_cutoff_fx_date}[name](ctx)
    fns = {PERIOD_CUTOFF: inject_cutoff, FX_ERROR: inject_fx_error,
           PAYMENT_MISALLOCATION: inject_misallocation, DUPLICATE_INVOICE: inject_duplicate}
    for t in (PERIOD_CUTOFF, FX_ERROR, PAYMENT_MISALLOCATION, DUPLICATE_INVOICE):
        for _ in range(sp["issues"].get(t, 0)):
            fns[t](ctx)
    return ctx
