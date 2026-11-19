"""Hidden clean company history -> clean records (SPEC section 3, steps 2 and 4).

The World owns customers, invoices, payments, fx walk and the double-entry ledger.  All posting
follows policy.json exactly; injectors later re-post individual groups through `set_group`.
Internal metadata lives in side dicts (never in the public record dicts).
"""
from __future__ import annotations

import bisect
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_EVEN

from ..constants import (ACC_AP, ACC_AR, ACC_CASH, ACC_EXPENSE_PAYROLL, ACC_EXPENSE_RENT,
                         ACC_EXPENSE_SOFTWARE, ACC_FX, ACC_REVENUE, ACC_UNAPPLIED, ACC_VAT,
                         CURRENCY_EXPONENT)
from ..money import convert_minor, format_rate
from ..rng import sub_rng

DOC_CAP = 188          # soft cap on documents while generating ordinary payments (hard cap 200)
MIN_LEDGER_PAD = 225   # pad noise so that the clean ledger has at least this many rows


class WorldError(Exception):
    """Generation attempt cannot satisfy the spec; the caller retries with a new attempt index."""


USD_PER = {"USD": Decimal("1"), "EUR": Decimal("1.08"), "GBP": Decimal("1.27"), "JPY": Decimal("0.0067"),
           "CHF": Decimal("1.12"), "AED": Decimal("0.2723"), "KZT": Decimal("0.0020")}

DESCRIPTIONS = {
    "Tiles": ["Ceramic tiles 30x30", "Porcelain tiles 60x60", "Tile adhesive 25kg", "Grout white 5kg",
              "Mosaic sheet blue", "Floor trim aluminium", "Waterproof membrane roll",
              "Tile spacers 3mm box", "Marble border strip", "Delivery and unloading",
              "Pallet handling fee", "Installation labour hour"],
    "Logistics": ["Pallet storage per week", "Last-mile delivery", "Container handling",
                  "Customs paperwork", "Cold chain surcharge", "Freight forwarding fee",
                  "Warehouse picking hour", "Route planning service", "Fuel adjustment",
                  "Reverse logistics batch", "Insurance cover", "Express courier parcel"],
    "Analytics": ["Dashboard licence seat", "Data pipeline setup", "Consulting hour",
                  "Model retraining run", "API access tier", "Support retainer",
                  "Custom report build", "Data warehouse storage", "Onboarding workshop",
                  "Security review", "Training session", "Anomaly alert package"],
    "Provisions": ["Olive oil 5L", "Durum flour 25kg", "Dried apricots 10kg", "Black tea 1kg tin",
                   "Honey jars crate", "Roasted nuts mix 5kg", "Sunflower oil 20L",
                   "Rice basmati 25kg", "Spice assortment box", "Canned tomatoes carton",
                   "Cold storage fee", "Packaging and labelling"],
}
CUST_ADJ = ["Alderwick", "Brackenfell", "Corvane", "Dalmoor", "Embergate", "Fallowmere", "Greystone",
            "Hallowtree", "Ivybridge", "Jasperfield", "Kingsmere", "Lindenhoek", "Mistral",
            "Nettlebed", "Orchardby", "Pemberly", "Quarrydale", "Rosewarren", "Saltmarsh",
            "Tamarind", "Underhill", "Vexford", "Willowmere", "Yewdale", "Zinnmoor", "Applegarth",
            "Beechcombe", "Cinderwood", "Dovecote", "Eldergrove"]
CUST_NOUN = ["Traders", "Supplies", "Holdings", "Partners", "Industries", "Works", "Distribution",
             "Systems", "Enterprises", "Services", "Retail", "Foods"]
CUST_SUFFIX = ["LLP", "Ltd", "LLC", "GmbH", "Inc", "Co", "Group"]

# ----------------------------------------------------------------------------- time helpers


def parse_offset(s: str) -> timezone:
    sign = -1 if s[0] == "-" else 1
    hh, mm = s[1:].split(":")
    return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))


def offset_seconds(s: str) -> int:
    return int(parse_offset(s).utcoffset(None).total_seconds())


def iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def is_bd(d: date) -> bool:
    return d.weekday() < 5


def line_vat(net: int, rate: str) -> int:
    return int((Decimal(net) * Decimal(rate)).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))


def invoice_vat(lines: list[dict]) -> int:
    return sum(line_vat(l["quantity"] * l["unit_price_minor"], l["vat_rate"]) for l in lines)


def invoice_total(lines: list[dict]) -> int:
    return sum(l["quantity"] * l["unit_price_minor"] + line_vat(l["quantity"] * l["unit_price_minor"], l["vat_rate"])
               for l in lines)


# ----------------------------------------------------------------------------- id allocation
_ALNUM = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


class Ids:
    """Per-company id factory. Every id is a fresh random draw so that no ordering leaks injections."""

    def __init__(self, rng, styles: dict, year: int):
        self.r, self.styles, self.year = rng, styles, year
        self.used: set[str] = set()

    def _uniq(self, make):
        for _ in range(10000):
            v = make()
            if v not in self.used:
                self.used.add(v)
                return v
        raise WorldError("id space exhausted")

    def _digits(self, n):
        return "".join(self.r.choice("0123456789") for _ in range(n))

    def _alnum(self, n):
        return "".join(self.r.choice(_ALNUM) for _ in range(n))

    def invoice(self):
        s = self.styles["invoice"]
        if s == "year_seq":
            return self._uniq(lambda: f"INV-{self.year}-{self.r.randint(1, 9999):04d}")
        if s == "slash":
            return self._uniq(lambda: f"SI/{self.r.randint(1000, 9999)}-{self.r.choice('ABCDEFGHKLMNPRSTUVWXYZ')}")
        return self._uniq(lambda: f"F-{self._alnum(6)}")

    def payment(self):
        s = self.styles["payment"]
        if s == "pay":
            return self._uniq(lambda: f"PAY-{self.r.randint(1, 999999):06d}")
        if s == "rcpt":
            return self._uniq(lambda: f"RCPT-{self.r.randint(1, 99999):05d}")
        return self._uniq(lambda: f"BNK{self._alnum(8)}")

    def entity(self):
        s = self.styles["entity"]
        if s == "cust":
            return self._uniq(lambda: f"CUST-{self.r.randint(1, 9999):04d}")
        if s == "c":
            return self._uniq(lambda: f"C{self.r.randint(1, 99999):05d}")
        return self._uniq(lambda: f"CUST-{self._alnum(2)}{self._digits(2)}")

    def vendor(self):
        return self._uniq(lambda: f"VND-{self.r.randint(1, 999):03d}")

    def doc(self):
        return self._uniq(lambda: "D-" + "".join(self.r.choice("0123456789abcdef") for _ in range(10)))

    def row(self):
        return self._uniq(lambda: "R-" + "".join(self.r.choice("0123456789abcdef") for _ in range(8)))


# ----------------------------------------------------------------------------- world
class World:
    def __init__(self, seed: int, attempt: int, spec: dict, cfg: dict):
        self.seed, self.attempt, self.spec, self.cfg = seed, attempt, spec, cfg
        self._rngs: dict[str, object] = {}
        self.base = spec["base_currency"]
        self.tz = parse_offset(spec["utc_offset"])
        y, m = spec["year"], spec["month"]
        self.period_start = datetime(y, m, 1, tzinfo=self.tz)
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        self.period_end = datetime(ny, nm, 1, tzinfo=self.tz) - timedelta(seconds=1)
        self.ndays = (self.period_end.date() - self.period_start.date()).days + 1
        self.ids = Ids(self.rng("ids"), spec["id_styles"], y)
        self.customers: dict[str, dict] = {}
        self.invoices: dict[str, dict] = {}
        self.inv_meta: dict[str, dict] = {}
        self.payments: dict[str, dict] = {}
        self.pay_meta: dict[str, dict] = {}
        self.ledger: dict[str, dict] = {}
        self.row_meta: dict[str, tuple] = {}
        self.groups: dict[tuple, dict] = {}
        self.fx: dict[str, dict[str, Decimal]] = {}
        self.fx_dates: dict[str, list[str]] = {}
        self.reserved: set[str] = set()
        self.distractors: list[dict] = []

    # ------------------------------------------------------------------ rng
    def rng(self, label: str):
        if label not in self._rngs:
            self._rngs[label] = sub_rng(self.seed, "world", self.attempt, label)
        return self._rngs[label]

    # ------------------------------------------------------------------ small helpers
    def cdate(self, dt: datetime) -> date:
        """Calendar date of an instant in the company offset."""
        return dt.astimezone(self.tz).date()

    def cts(self, dt: datetime) -> str:
        return iso(dt.astimezone(self.tz))

    def in_period(self, dt: datetime) -> bool:
        return self.period_start <= dt <= self.period_end

    def inv_instant(self, inv: dict) -> datetime:
        return parse_ts(inv["issued_at"])

    def doc_count(self) -> int:
        return len(self.invoices) + len(self.payments)

    def paid_invoice_ids(self) -> set[str]:
        out = set()
        for p in self.payments.values():
            for ref in p["invoice_refs"] or []:
                out.add(ref["invoice_id"])
        return out

    def foreign_customers(self):
        return [c for c in self.customers.values() if c["currency"] != self.base]

    def delta(self, cust) -> int:
        """customer offset - company offset, seconds"""
        return cust["off_s"] - offset_seconds(self.spec["utc_offset"])

    def day_start(self, d: date) -> datetime:
        return datetime(d.year, d.month, d.day, tzinfo=self.tz)

    # ------------------------------------------------------------------ customers
    def build_customers(self):
        r = self.rng("customers")
        sp = self.spec
        n, offs = sp["n_customers"], sp["customer_offsets"]
        base_s = offset_seconds(sp["utc_offset"])
        differing = [o for o in offs if abs(offset_seconds(o) - base_s) >= 3600]
        fc = list(sp["foreign_currencies"])
        r.shuffle(fc)
        share = Decimal(sp["foreign_invoice_share"])
        jitter = Decimal(r.randint(-100, 100)) / Decimal(100)
        n_foreign = int((Decimal(n) * share + Decimal("0.5") + jitter).to_integral_value())
        n_foreign = max(2, min(n - 2, n_foreign))
        currencies = [fc[i % len(fc)] for i in range(n_foreign)] + [self.base] * (n - n_foreign)
        r.shuffle(currencies)
        offsets = [r.choice(offs) for _ in range(n)]
        if differing:
            idx_f = [i for i, c in enumerate(currencies) if c != self.base]
            idx_b = [i for i, c in enumerate(currencies) if c == self.base]
            for i in idx_f[:2] + idx_b[:1]:
                if offsets[i] not in differing:
                    offsets[i] = r.choice(differing)
        names = set()
        for i in range(n):
            for _ in range(200):
                nm = f"{r.choice(CUST_ADJ)} {r.choice(CUST_NOUN)} {r.choice(CUST_SUFFIX)}"
                if nm not in names:
                    names.add(nm)
                    break
            eid = self.ids.entity()
            self.customers[eid] = {
                "entity_id": eid, "name": nm, "currency": currencies[i], "offset": offsets[i],
                "tz": parse_offset(offsets[i]), "off_s": offset_seconds(offsets[i]),
                "vat": r.choice(sp["vat_rates"]),
            }

    # ------------------------------------------------------------------ fx
    def build_fx(self):
        r = self.rng("fx")
        used = sorted({c["currency"] for c in self.customers.values() if c["currency"] != self.base})
        start = self.period_start.date() - timedelta(days=10)
        end = self.period_end.date() + timedelta(days=12)
        step = Decimal("0.000001")
        for cur in used:
            rate = USD_PER[cur] / USD_PER[self.base]
            rate = rate * (Decimal(1) + Decimal(r.randint(-800, 800)) / Decimal(10000))
            rate = rate.quantize(step, rounding=ROUND_HALF_EVEN)
            rows = {}
            prev = None
            d = start
            while d <= end:
                if is_bd(d):
                    if prev is None:
                        new = rate
                    else:
                        pct = Decimal(r.randint(10, 150)) / Decimal(10000)
                        sign = 1 if r.random() < 0.5 else -1
                        new = (prev * (Decimal(1) + sign * pct)).quantize(step, rounding=ROUND_HALF_EVEN)
                        if new == prev:
                            new = prev + sign * step
                        if new <= 0:
                            new = prev + step
                    rows[d.isoformat()] = new
                    prev = new
                d += timedelta(days=1)
            self.fx[cur] = rows
            self.fx_dates[cur] = sorted(rows)

    def rate_for(self, cur: str, date_str: str) -> tuple[str | None, Decimal]:
        """(date actually used, rate): most recent date <= date_str present in the fx table."""
        if cur == self.base:
            return None, Decimal(1)
        dates = self.fx_dates[cur]
        i = bisect.bisect_right(dates, date_str) - 1
        if i < 0:
            raise WorldError("no fx rate on or before " + date_str)
        d = dates[i]
        return d, self.fx[cur][d]

    def conv(self, amount: int, cur: str, rate) -> int:
        return convert_minor(amount, cur, self.base, rate)

    def inv_rate(self, inv: dict) -> tuple[str | None, Decimal]:
        return self.rate_for(inv["currency"], self.cdate(self.inv_instant(inv)).isoformat())

    def pay_rate(self, pay: dict) -> tuple[str | None, Decimal]:
        return self.rate_for(pay["currency"], self.cdate(parse_ts(pay["timestamp"])).isoformat())

    def fx_rows(self) -> list[dict]:
        rows = []
        for cur in sorted(self.fx):
            for d in self.fx_dates[cur]:
                rows.append({"date": d, "source_currency": cur, "target_currency": self.base,
                             "rate": format_rate(self.fx[cur][d])})
        rows.sort(key=lambda x: (x["date"], x["source_currency"]))
        return rows

    # ------------------------------------------------------------------ invoices
    def price_minor(self, cur: str, r) -> int:
        usd = r.choice([5, 8, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 450, 600, 900]) + r.randint(0, 9)
        major = int(Decimal(usd) / USD_PER[cur])
        major = max(1, major)
        if major >= 1000:
            step = 10 ** (len(str(major)) - 3)
            major = major // step * step
        minor = major * 10 ** CURRENCY_EXPONENT[cur]
        if CURRENCY_EXPONENT[cur] > 0 and major < 50:
            minor += r.choice([0, 0, 25, 50, 75, 99])
        return minor

    def new_invoice(self, cust: dict, instant: datetime, kind: str = "ordinary",
                    lines: list | None = None) -> dict:
        r = self.rng("lines")
        sp = self.spec
        cur = cust["currency"]
        if lines is None:
            n = r.choices([1, 2, 3, 4], weights=[35, 35, 20, 10])[0]
            descs = r.sample(DESCRIPTIONS[sp["industry"]], n)
            y, m = sp["year"], sp["month"]
            py, pm = (y - 1, 12) if m == 1 else (y, m - 1)
            service = r.choices([f"{y:04d}-{m:02d}", f"{py:04d}-{pm:02d}"], weights=[70, 30])[0]
            po = f"PO-{r.randint(10000, 99999)}" if r.random() < 0.75 else None
            lines = []
            for i, d in enumerate(descs, 1):
                qty = r.choice([1, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 36, 48, 50, 60, 100])
                vat = cust["vat"] if r.random() < 0.9 else r.choice(sp["vat_rates"])
                lines.append({"line": i, "description": d, "quantity": qty,
                              "unit_price_minor": self.price_minor(cur, r), "vat_rate": vat,
                              "service_period": service, "po_ref": po})
        term = r.choice([15, 30, 45])
        issued = instant.astimezone(cust["tz"]).replace(microsecond=0)
        inv = {
            "doc_id": self.ids.doc(), "invoice_id": self.ids.invoice(), "entity_id": cust["entity_id"],
            "issued_at": iso(issued), "due_at": iso(issued + timedelta(days=term)),
            "currency": cur, "total_minor": invoice_total(lines), "line_items": lines,
        }
        self.invoices[inv["invoice_id"]] = inv
        self.inv_meta[inv["invoice_id"]] = {"kind": kind, "term": term}
        return inv

    def random_instant(self, r) -> datetime:
        while True:
            d = r.randrange(self.ndays)
            base = self.period_start + timedelta(days=d)
            if base.weekday() >= 5 and r.random() < 0.6:
                continue
            s = r.randint(7 * 3600, 19 * 3600) if r.random() < 0.9 else r.randint(0, 86399)
            return base + timedelta(seconds=s)

    def gen_ordinary_invoices(self):
        r = self.rng("invoices")
        custs = list(self.customers.values())
        weights = [1.0 / (i + 1) ** 0.5 for i in range(len(custs))]
        for _ in range(self.spec["n_invoices"]):
            c = r.choices(custs, weights)[0]
            self.new_invoice(c, self.random_instant(r))

    def gen_post_period_invoices(self):
        r = self.rng("postperiod")
        D = self.period_end.date()
        n = r.randint(3, 8)
        # capacity follows demand: every forced period_cutoff / cutoff interaction needs its own
        # post-period invoice (and the interaction needs a foreign one just after period end)
        k_inter = self.spec["interactions"].get("cutoff_with_wrong_rate_date", 0)
        need = self.spec["issues"].get("period_cutoff", 0) + k_inter
        if need:
            n = max(n, need + 1)
        n_foreign_after = min(n, max(2, 2 * k_inter))
        foreign = self.foreign_customers()
        behind = [c for c in self.customers.values() if self.delta(c) <= -3600]
        custs = list(self.customers.values())
        first_bd = D + timedelta(days=1)
        while not is_bd(first_bd):
            first_bd += timedelta(days=1)
        plans = []   # (customer, instant)
        # two foreign invoices just after period end (also the interaction candidates)
        fb = [c for c in foreign if c in behind] or foreign
        for i in range(n_foreign_after):
            c = r.choice(fb if i == 0 else foreign)
            plans.append((c, self.day_start(first_bd) + timedelta(seconds=r.randint(300, 12000))))
        # a "tricky" one: customer behind the company, date string still shows the period's last day
        if behind:
            c = r.choice(behind)
            hi = -self.delta(c) - 300
            plans.append((c, self.day_start(D + timedelta(days=1)) + timedelta(seconds=r.randint(300, hi))))
        # the very first instant after the (inclusive) period end: period_end + 1 s
        plans.append((r.choice(custs), self.period_end + timedelta(seconds=1)))
        while len(plans) < max(n, 3):
            k = r.randint(1, 10)
            plans.append((r.choice(custs), self.day_start(D + timedelta(days=k)) + timedelta(seconds=r.randint(0, 86399))))
        r.shuffle(plans)
        for c, inst in plans:
            if inst <= self.period_end:
                raise WorldError("post-period instant inside period")
            self.new_invoice(c, inst, kind="post_period")

    def gen_pre_period_invoices(self):
        """Mirror of the post-period set: invoices issued BEFORE period_start.  They belong to the prior period's
        books, so the clean ledger has no rows for them; `period_cutoff` may post one inside the period."""
        r = self.rng("preperiod")
        custs = list(self.customers.values())
        ahead = [c for c in custs if self.delta(c) >= 3600]
        plans = []   # (customer, instant)
        # the last instant before the period: period_start - 1 s
        plans.append((r.choice(custs), self.period_start - timedelta(seconds=1)))
        # "tricky": customer ahead of the company; the date written on the invoice already shows the
        # period's first day although the instant is still before period_start
        if ahead:
            c = r.choice(ahead)
            plans.append((c, self.period_start - timedelta(seconds=r.randint(300, self.delta(c) - 300))))
        n_pre = r.randint(2, 4)
        while len(plans) < n_pre:
            plans.append((r.choice(custs), self.period_start - timedelta(seconds=r.randint(86400, 7 * 86400))))
        r.shuffle(plans)
        for c, inst in plans:
            if inst >= self.period_start:
                raise WorldError("pre-period instant inside period")
            self.new_invoice(c, inst, kind="pre_period")

    def verify_no_duplicates(self):
        seen = {}
        for inv in self.invoices.values():
            key = (inv["entity_id"], inv["currency"], json.dumps(inv["line_items"], sort_keys=True))
            if key in seen:
                raise WorldError("accidental duplicate invoice")
            seen[key] = inv["invoice_id"]

    # ------------------------------------------------------------------ payments
    def new_payment(self, cust: dict, instant: datetime, amount: int, refs, kind="ordinary") -> dict:
        pay = {
            "doc_id": self.ids.doc(), "payment_id": self.ids.payment(),
            "timestamp": iso(instant.astimezone(cust["tz"])), "currency": cust["currency"],
            "amount_minor": amount, "invoice_refs": refs,
        }
        self.payments[pay["payment_id"]] = pay
        entity = cust["entity_id"] if refs else None
        self.pay_meta[pay["payment_id"]] = {"kind": kind, "entity_id": entity}
        return pay

    def gen_boundary_payments(self):
        """Natural (correctly posted) foreign payments whose own-offset date differs from the company date."""
        r = self.rng("boundary")
        need = self.spec["interactions"]["fx_partial_payment_at_boundary"]
        custs = [c for c in self.foreign_customers() if abs(self.delta(c)) >= 3600]
        if not custs:
            if need:
                raise WorldError("no foreign customer with differing offset")
            return
        n = max(r.randint(1, 3), need + r.randint(0, 1))
        for _ in range(n):
            done = False
            for _try in range(12):
                c = r.choice(custs)
                paid = self.paid_invoice_ids()
                cand = [i for i in self.invoices.values()
                        if self.inv_meta[i["invoice_id"]]["kind"] == "ordinary" and i["entity_id"] == c["entity_id"]
                        and i["invoice_id"] not in self.reserved and i["invoice_id"] not in paid
                        and self.inv_instant(i) <= self.period_end - timedelta(days=3)]
                if cand:
                    inv = r.choice(cand)
                else:
                    inst = self.period_start + timedelta(seconds=r.randint(3600, int(self.ndays * 86400 * 0.4)))
                    inv = self.new_invoice(c, inst)
                delta = self.delta(c)
                issue = self.inv_instant(inv)
                days = [self.period_start.date() + timedelta(days=k) for k in range(self.ndays)]
                r.shuffle(days)
                for D in days:
                    own = D + timedelta(days=1 if delta > 0 else -1)
                    if not (is_bd(D) and is_bd(own)):
                        continue
                    lo, hi = (86400 - delta + 300, 86400 - 300) if delta > 0 else (300, -delta - 300)
                    inst = self.day_start(D) + timedelta(seconds=r.randint(lo, hi))
                    if inst < issue + timedelta(hours=1) or inst > self.period_end - timedelta(minutes=1):
                        continue
                    refs = [{"invoice_id": inv["invoice_id"], "amount_minor": inv["total_minor"]}]
                    self.new_payment(c, inst, inv["total_minor"], refs, kind="boundary")
                    done = True
                    break
                if done:
                    break
            if not done:
                raise WorldError("could not place boundary payment")

    def gen_ordinary_payments(self):
        r = self.rng("payments")
        paid = self.paid_invoice_ids()
        cand = [i for i in self.invoices.values()
                if self.inv_meta[i["invoice_id"]]["kind"] == "ordinary"
                and i["invoice_id"] not in self.reserved and i["invoice_id"] not in paid]
        r.shuffle(cand)
        p = r.randint(35, 70) / 100
        for inv in cand:
            iid = inv["invoice_id"]
            if iid in paid:
                continue
            if self.doc_count() >= DOC_CAP:
                break
            if r.random() > p:
                continue
            group = [inv]
            if r.random() < 0.25:
                others = [o for o in cand if o["invoice_id"] != iid and o["invoice_id"] not in paid
                          and o["entity_id"] == inv["entity_id"]]
                r.shuffle(others)
                group += others[:r.randint(1, 2)]
            earliest = max(self.inv_instant(g) for g in group)
            remaining = int((self.period_end - earliest).total_seconds())
            if remaining < 7200:
                continue
            span = r.choice([5 * 86400, 20 * 86400, 28 * 86400])
            delta = r.randint(3600, max(3600, min(span, remaining - 60)))
            inst = earliest + timedelta(seconds=delta)
            refs = []
            for g in group:
                amt = g["total_minor"]
                if len(group) == 1 and r.random() < 0.10:
                    amt = max(1, amt * r.randint(30, 90) // 100)
                refs.append({"invoice_id": g["invoice_id"], "amount_minor": amt})
            cust = self.customers[inv["entity_id"]]
            self.new_payment(cust, inst, sum(x["amount_minor"] for x in refs), refs)
            for g in group:
                paid.add(g["invoice_id"])

    # ------------------------------------------------------------------ posting
    def invoice_slots(self, inv: dict, rate=None) -> dict:
        """slot -> (account, debit, credit, entity, invoice_id, payment_id)"""
        cur = inv["currency"]
        if rate is None:
            _, rate = self.inv_rate(inv)
        ar = self.conv(inv["total_minor"], cur, rate)
        vat = self.conv(invoice_vat(inv["line_items"]), cur, rate)
        e, i = inv["entity_id"], inv["invoice_id"]
        slots = {"ar": (ACC_AR, ar, 0, e, i, None)}
        if vat > 0:
            slots["vat"] = (ACC_VAT, 0, vat, e, i, None)
        slots["rev"] = (ACC_REVENUE, 0, ar - vat, e, i, None)
        return slots

    def payment_slots(self, pay: dict, cash_rate=None, ar_credit_pcc: dict | None = None,
                      extra_unapplied_pcc: int = 0) -> dict:
        cur, amt, pid = pay["currency"], pay["amount_minor"], pay["payment_id"]
        entity = self.pay_meta[pid]["entity_id"]
        _, prate = self.pay_rate(pay)
        slots = {}
        if amt < 0:
            v = self.conv(-amt, cur, prate)
            slots["unapplied"] = (ACC_UNAPPLIED, v, 0, entity, None, pid)
            slots["cash"] = (ACC_CASH, 0, v, entity, None, pid)
            return slots
        cash = self.conv(amt, cur, cash_rate if cash_rate is not None else prate)
        ar_rows = {}
        ar_total = 0
        ref_sum = 0
        for ref in pay["invoice_refs"] or []:
            ref_sum += ref["amount_minor"]
            credited = ref["amount_minor"]
            if ar_credit_pcc is not None and ref["invoice_id"] in ar_credit_pcc:
                credited = ar_credit_pcc[ref["invoice_id"]]
            if credited > 0:
                inv = self.invoices[ref["invoice_id"]]
                _, ir = self.inv_rate(inv)
                a = self.conv(credited, cur, ir)
                ar_rows[f"ar:{ref['invoice_id']}"] = (ACC_AR, 0, a, entity, None, pid)
                ar_total += a
        unapplied_pcc = amt - ref_sum + extra_unapplied_pcc
        u = self.conv(unapplied_pcc, cur, prate) if unapplied_pcc > 0 else 0
        slots["cash"] = (ACC_CASH, cash, 0, entity, None, pid)
        slots.update(ar_rows)
        if u > 0:
            slots["unapplied"] = (ACC_UNAPPLIED, 0, u, entity, None, pid)
        diff = cash - ar_total - u
        if diff > 0:
            slots["fx"] = (ACC_FX, 0, diff, entity, None, pid)
        elif diff < 0:
            slots["fx"] = (ACC_FX, -diff, 0, entity, None, pid)
        return slots

    def _new_row(self, ts, slot_t) -> dict:
        acc, dr, cr, ent, inv, pay = slot_t
        return {"row_id": self.ids.row(), "timestamp": ts, "account": acc, "debit_minor": dr,
                "credit_minor": cr, "currency": self.base, "entity_id": ent, "invoice_id": inv,
                "payment_id": pay}

    def set_group(self, kind: str, key: str, ts: str, slots: dict):
        """(Re)post a group; returns (added_row_ids, removed_row_ids, modified[{id,field,before,after}])."""
        old = self.groups.get((kind, key), {})
        new_map = {}
        added, removed, modified = [], [], []
        for slot, st in slots.items():
            if st[1] == 0 and st[2] == 0:
                continue
            if slot in old:
                row = self.ledger[old[slot]]
                acc, dr, cr, ent, inv, pay = st
                want = {"timestamp": ts, "account": acc, "debit_minor": dr, "credit_minor": cr,
                        "entity_id": ent, "invoice_id": inv, "payment_id": pay}
                for f, v in want.items():
                    if row[f] != v:
                        modified.append({"id": row["row_id"], "field": f, "before": row[f], "after": v})
                        row[f] = v
                new_map[slot] = old[slot]
            else:
                row = self._new_row(ts, st)
                self.ledger[row["row_id"]] = row
                self.row_meta[row["row_id"]] = (kind, key, slot)
                new_map[slot] = row["row_id"]
                added.append(row["row_id"])
        for slot, rid in old.items():
            if slot not in new_map:
                removed.append(rid)
                del self.ledger[rid]
                del self.row_meta[rid]
        self.groups[(kind, key)] = new_map
        return added, removed, modified

    def post_invoice(self, inv: dict, ts: str | None = None, rate=None):
        ts = ts or self.cts(self.inv_instant(inv))
        return self.set_group("inv", inv["invoice_id"], ts, self.invoice_slots(inv, rate))

    def post_payment(self, pay: dict, **kw):
        ts = self.cts(parse_ts(pay["timestamp"]))
        return self.set_group("pay", pay["payment_id"], ts, self.payment_slots(pay, **kw))

    def post_clean(self):
        for inv in self.invoices.values():
            if self.inv_meta[inv["invoice_id"]]["kind"] in ("post_period", "pre_period"):
                continue
            self.post_invoice(inv)
        for pay in self.payments.values():
            self.post_payment(pay)
        self.post_noise()

    def post_noise(self):
        r = self.rng("noise")
        n = self.spec["n_noise_postings"]
        rows = len(self.ledger)
        if rows + 2 * n < MIN_LEDGER_PAD + r.randint(0, 40):
            n = (MIN_LEDGER_PAD + r.randint(0, 40) - rows + 1) // 2
        n = max(0, min(n, (1900 - rows) // 2))
        self.spec["n_noise_effective"] = n      # recorded in provenance: requested vs. generated
        vendors = [self.ids.vendor() for _ in range(r.randint(3, 6))]
        total_secs = int((self.period_end - self.period_start).total_seconds())
        exp = CURRENCY_EXPONENT[self.base]
        for _ in range(n):
            kind = r.choices(["rent", "payroll", "software"], weights=[15, 35, 50])[0]
            acc = {"rent": ACC_EXPENSE_RENT, "payroll": ACC_EXPENSE_PAYROLL, "software": ACC_EXPENSE_SOFTWARE}[kind]
            usd = {"rent": r.randint(800, 4000), "payroll": r.randint(1500, 9000), "software": r.randint(15, 900)}[kind]
            major = max(1, int(Decimal(usd) / USD_PER[self.base]))
            if major >= 1000:
                step = 10 ** (len(str(major)) - 3)
                major = major // step * step
            amount = major * 10 ** exp + (r.randint(0, 99) if exp else 0)
            ts = self.cts(self.period_start + timedelta(seconds=r.randint(0, total_secs)))
            ent = r.choice(vendors) if r.random() < 0.3 else None
            credit_acc = ACC_AP if r.random() < 0.6 else ACC_CASH
            for st in ((acc, amount, 0, ent, None, None), (credit_acc, 0, amount, ent, None, None)):
                row = self._new_row(ts, st)
                self.ledger[row["row_id"]] = row
                self.row_meta[row["row_id"]] = ("noise", row["row_id"], "n")

    # ------------------------------------------------------------------ canonical output
    def sorted_ledger(self) -> list[dict]:
        return sorted(self.ledger.values(), key=lambda x: (parse_ts(x["timestamp"]), x["row_id"]))

    def sorted_invoices(self) -> list[dict]:
        return sorted(self.invoices.values(), key=lambda x: (parse_ts(x["issued_at"]), x["invoice_id"]))

    def sorted_payments(self) -> list[dict]:
        return sorted(self.payments.values(), key=lambda x: (parse_ts(x["timestamp"]), x["payment_id"]))

    def group_row_ids(self, kind: str, key: str) -> list[str]:
        return list(self.groups.get((kind, key), {}).values())


def check_balanced(rows: list[dict]) -> None:
    """Every posting group (timestamp + invoice_id/payment_id) balances; each row one-sided."""
    sums: dict[tuple, int] = {}
    for r in rows:
        dr, cr = r["debit_minor"], r["credit_minor"]
        if (dr > 0) == (cr > 0) or dr < 0 or cr < 0:
            raise WorldError("row not one-sided: " + r["row_id"])
        k = (r["timestamp"], r["invoice_id"], r["payment_id"])
        sums[k] = sums.get(k, 0) + dr - cr
    bad = [k for k, v in sums.items() if v != 0]
    if bad:
        raise WorldError(f"unbalanced groups: {bad[:3]}")
