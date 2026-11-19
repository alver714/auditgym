"""Correct-but-suspicious patterns. They live in the clean world and must yield zero truth issues.

Each recipe creates/reserves its own invoices and payments (reserved ids are never used as injection
targets and never paid by ordinary payment generation).
"""
from __future__ import annotations

import copy
from datetime import timedelta

from ..constants import (DISTRACTOR_FX_GAIN_LOSS, DISTRACTOR_PREPAYMENT, DISTRACTOR_RECURRING,
                         DISTRACTOR_REFUND, DISTRACTOR_SPLIT_PAYMENT, DISTRACTOR_TZ_BOUNDARY,
                         DISTRACTORS)
from .world import World, WorldError, parse_ts


def _record(w: World, kind: str, ids: list[str], variant: str | None = None):
    d = {"kind": kind, "record_ids": list(ids)}
    if variant:
        d["variant"] = variant
    w.distractors.append(d)


def _pick_invoice(w: World, r, min_total=200, latest_days=8, foreign=None):
    """An existing unreserved, unpaid ordinary invoice with enough room before period end; else a new one."""
    paid = w.paid_invoice_ids()
    latest = w.period_end - timedelta(days=latest_days)
    cand = [i for i in w.invoices.values()
            if w.inv_meta[i["invoice_id"]]["kind"] == "ordinary"
            and i["invoice_id"] not in w.reserved and i["invoice_id"] not in paid
            and w.inv_instant(i) <= latest and i["total_minor"] >= min_total
            and (foreign is None or (i["currency"] != w.base) == foreign)]
    if cand:
        inv = r.choice(cand)
    else:
        custs = [c for c in w.customers.values() if foreign is None or (c["currency"] != w.base) == foreign]
        if not custs:
            raise WorldError("no customer for distractor")
        for _ in range(50):
            c = r.choice(custs)
            inst = w.period_start + timedelta(seconds=r.randint(3600, int(w.ndays * 86400 * 0.45)))
            inv = w.new_invoice(c, inst)
            if inv["total_minor"] >= min_total:
                break
        else:
            raise WorldError("cannot create invoice for distractor")
    w.reserved.add(inv["invoice_id"])
    return inv


def _later(w: World, r, after, min_s=3600, max_s=None, reserve_s=60):
    """A random instant in [after+min_s, period_end - reserve_s]."""
    hi = int((w.period_end - after).total_seconds()) - reserve_s
    if max_s is not None:
        hi = min(hi, max_s)
    if hi < min_s:
        raise WorldError("no room for later event")
    return after + timedelta(seconds=r.randint(min_s, hi))


def split_payment(w: World, r):
    inv = _pick_invoice(w, r, min_total=300, latest_days=8)
    cust = w.customers[inv["entity_id"]]
    n = r.choice([2, 2, 3])
    weights = [r.randint(2, 6) for _ in range(n)]
    total = inv["total_minor"]
    amounts = [total * wt // sum(weights) for wt in weights]
    amounts[-1] = total - sum(amounts[:-1])
    issue = w.inv_instant(inv)
    span = int((w.period_end - issue).total_seconds()) - 60
    cuts = sorted(r.sample(range(3600, max(3601 + n, span)), n))
    ids = [inv["invoice_id"]]
    for amt, cut in zip(amounts, cuts):
        pay = w.new_payment(cust, issue + timedelta(seconds=cut), amt,
                            [{"invoice_id": inv["invoice_id"], "amount_minor": amt}], kind="distractor")
        w.reserved.add(pay["payment_id"])
        ids.append(pay["payment_id"])
    _record(w, DISTRACTOR_SPLIT_PAYMENT, ids)


def overpayment_refund(w: World, r):
    inv = _pick_invoice(w, r, min_total=300, latest_days=10)
    cust = w.customers[inv["entity_id"]]
    total = inv["total_minor"]
    excess = max(1, total * r.randint(5, 25) // 100)
    issue = w.inv_instant(inv)
    t1 = _later(w, r, issue, 3600, 6 * 86400, reserve_s=3 * 86400)
    p1 = w.new_payment(cust, t1, total + excess, [{"invoice_id": inv["invoice_id"], "amount_minor": total}],
                       kind="distractor")
    t2 = _later(w, r, t1, 86400, 8 * 86400)
    p2 = w.new_payment(cust, t2, -excess, None, kind="distractor")
    for p in (p1, p2):
        w.reserved.add(p["payment_id"])
    _record(w, DISTRACTOR_REFUND, [inv["invoice_id"], p1["payment_id"], p2["payment_id"]])


def legit_recurring_amount(w: World, r):
    """A weekly service billed again at the same amount: identical price lines, a different ISO-week
    service_period label and a 7-day gap.  (A monthly label would need a 28-31 day gap, which does not fit
    inside a one-month period, so only the weekly pattern is generated.)"""
    custs = list(w.customers.values())
    cust = r.choice(custs)
    n = 2 if r.random() < 0.7 else 3
    gap_days = 7
    need = gap_days * (n - 1) + 1
    if need >= w.ndays - 2:
        n, need = 2, gap_days + 1
    t0 = w.period_start + timedelta(seconds=r.randint(3600, max(3601, (w.ndays - need) * 86400 - 3600)))
    insts = [t0 + timedelta(days=gap_days * k, seconds=r.randint(-7200, 7200) if k else 0) for k in range(n)]
    sps = []
    for k in range(n):
        iso_y, iso_w, _ = insts[k].astimezone(w.tz).isocalendar()
        sps.append(f"{iso_y}-W{iso_w:02d}")
    if len(set(sps)) < n:     # jitter pushed two invoices into one ISO week: label by the un-jittered dates
        sps = []
        for k in range(n):
            iso_y, iso_w, _ = (t0 + timedelta(days=gap_days * k)).astimezone(w.tz).isocalendar()
            sps.append(f"{iso_y}-W{iso_w:02d}")
    first = w.new_invoice(cust, insts[0], kind="ordinary")
    first["line_items"] = [dict(l, service_period=sps[0]) for l in first["line_items"]]
    w.reserved.add(first["invoice_id"])
    ids = [first["invoice_id"]]
    for k in range(1, n):
        lines = [dict(l, service_period=sps[k]) for l in copy.deepcopy(first["line_items"])]
        inv = w.new_invoice(cust, insts[k], kind="ordinary", lines=lines)
        w.reserved.add(inv["invoice_id"])
        ids.append(inv["invoice_id"])
    _record(w, DISTRACTOR_RECURRING, ids)


def timezone_boundary_ok(w: World, r):
    """Correct, in-period invoices that merely LOOK out of period.  Variants:
      * offset_edge  - customer offset makes the invoice's date string fall in the neighbouring month;
      * exact_end    - issued_at is exactly period_end (inclusive upper bound -> in period);
      * exact_start  - issued_at is exactly period_start (inclusive lower bound -> in period)."""
    cands = [c for c in w.customers.values() if abs(w.delta(c)) >= 3600]
    if not cands:
        raise WorldError("no customer with differing offset")
    c = r.choice(cands)
    delta = w.delta(c)
    variant = r.choice(["offset_edge", "offset_edge", "exact_end", "exact_start"])
    if variant == "exact_end":
        inst = w.period_end
    elif variant == "exact_start":
        inst = w.period_start
    elif delta > 0:   # customer ahead: late in the period -> customer's date string is the next month
        inst = w.period_end - timedelta(seconds=r.randint(300, delta - 300))
    else:             # customer behind: just after period start -> date string is the previous month
        inst = w.period_start + timedelta(seconds=r.randint(300, -delta - 300))
    inv = w.new_invoice(c, inst, kind="ordinary")
    w.reserved.add(inv["invoice_id"])
    _record(w, DISTRACTOR_TZ_BOUNDARY, [inv["invoice_id"]], variant)


def realized_fx_gain_loss(w: World, r):
    for _ in range(30):
        inv = _pick_invoice(w, r, min_total=2000, latest_days=8, foreign=True)
        cust = w.customers[inv["entity_id"]]
        issue = w.inv_instant(inv)
        for _try in range(20):
            t = _later(w, r, issue, 3 * 86400, 20 * 86400)
            pay_stub = {"timestamp": w.cts(t), "currency": inv["currency"]}
            _, pr = w.pay_rate(pay_stub)
            _, ir = w.inv_rate(inv)
            if w.conv(inv["total_minor"], inv["currency"], pr) != w.conv(inv["total_minor"], inv["currency"], ir):
                pay = w.new_payment(cust, t, inv["total_minor"],
                                    [{"invoice_id": inv["invoice_id"], "amount_minor": inv["total_minor"]}],
                                    kind="distractor")
                w.reserved.add(pay["payment_id"])
                _record(w, DISTRACTOR_FX_GAIN_LOSS, [inv["invoice_id"], pay["payment_id"]])
                return
    raise WorldError("no realized fx gain/loss scenario")


def unreferenced_prepayment(w: World, r):
    cust = r.choice(list(w.customers.values()))
    inst = w.period_start + timedelta(seconds=r.randint(3600, w.ndays * 86400 - 3600))
    w_amount = w.price_minor(cust["currency"], r) * r.choice([3, 5, 10, 20])
    pay = w.new_payment(cust, inst, w_amount, None, kind="distractor")
    w.reserved.add(pay["payment_id"])
    _record(w, DISTRACTOR_PREPAYMENT, [pay["payment_id"]])


RECIPES = {
    DISTRACTOR_SPLIT_PAYMENT: split_payment,
    DISTRACTOR_REFUND: overpayment_refund,
    DISTRACTOR_RECURRING: legit_recurring_amount,
    DISTRACTOR_TZ_BOUNDARY: timezone_boundary_ok,
    DISTRACTOR_FX_GAIN_LOSS: realized_fx_gain_loss,
    DISTRACTOR_PREPAYMENT: unreferenced_prepayment,
}


def apply_distractors(w: World) -> None:
    r = w.rng("distractors")
    for kind in DISTRACTORS:
        for _ in range(w.spec["distractors"].get(kind, 0)):
            RECIPES[kind](w, r)
