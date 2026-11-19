"""Config loading / validation and sampling of a concrete episode spec (SPEC section 3, step 1).

`force` semantics (documented deviation-free interpretation of the SPEC):
  * {"clean": true}                      -> no issues, no interactions (distractors still apply)
  * {"issues": {...}} / {"interactions": {...}}
        -> the episode is fully specified: listed keys are pinned, unlisted keys of EITHER dict
           (and a wholly absent dict) are 0.  `clean_probability` is not applied.
  * {"distractors": {...}}               -> pins listed distractor counts (others sampled)
  * {"company": {...}}                   -> pins company parameters (base_currency, timezone,
        year, month, n_customers, n_invoices, n_noise_postings, foreign_currencies,
        customer_offsets, foreign_invoice_share)
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from decimal import Decimal, InvalidOperation

from ..episode import dumps
from ..constants import CURRENCY_EXPONENT, DISTRACTORS, INTERACTIONS, ISSUE_TYPES
from ..rng import sub_rng

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "configs")
MAX_TOTAL_ISSUES = 12
MAX_INVOICES = 105           # keeps documents <= 200 and ledger rows <= 2000
MIN_INVOICES = 14
MAX_NOISE_POSTINGS = 1000    # pairs of rows; ledger rows are capped at 1900 overall, the effective count is
                             # recorded in provenance spec as n_noise_effective
MIN_CUSTOMERS = 4
MAX_CUSTOMERS = 60


class ConfigError(ValueError):
    pass


# `config_id` / `config_version` are listed (None) only so that DEFAULT_CONFIG documents the full key set that
# `validate_config` whitelists; a config must state both itself, they are never filled in from here.
DEFAULT_CONFIG = {
    "config_id": None,
    "config_version": None,
    "company": {
        "n_customers": [6, 18], "n_invoices": [30, 90], "n_noise_postings": [60, 250],
        "foreign_invoice_share": 0.35,
        "base_currencies": ["KZT", "USD", "EUR"],
        "foreign_currencies": ["EUR", "USD", "GBP", "JPY", "CHF", "AED"],
        "timezones": [["Asia/Almaty", "+05:00"], ["UTC", "+00:00"]],
        "customer_offsets": ["+05:00", "+00:00", "-03:00", "+08:00", "-05:00", "+09:00"],
        "years": [2025, 2026], "vat_rates": ["0", "0.12", "0.20"],
    },
    "issues": {t: [0, 2] for t in ISSUE_TYPES},
    "interactions": {t: [0, 1] for t in INTERACTIONS},
    "distractors": {"split_payment": [1, 3], "overpayment_refund": [0, 2],
                    "legit_recurring_amount": [1, 3], "timezone_boundary_ok": [1, 2],
                    "realized_fx_gain_loss": [1, 4], "unreferenced_prepayment": [0, 2]},
    "clean_probability": 0.2,
    "force": {},
}

COMPANY_ADJ = ["Aldermoor", "Brightvale", "Cobalt Ridge", "Dunmere", "Eastgate", "Fernhollow",
               "Granite Bay", "Harrowfield", "Ironwood", "Juniper", "Kestrel", "Larkspur",
               "Marlowe", "Northwind", "Oakhaven", "Pinecrest", "Quillfeather", "Redwater",
               "Silverbrook", "Thornfield", "Umberlea", "Valewood", "Westmarch", "Yarrowby",
               "Zephyrine", "Ambergate", "Blackthorn", "Copperfield", "Driftwood", "Elmstead",
               "Foxglove", "Gilderoy", "Hartwell", "Inglenook", "Jadeport", "Kingfisher", "Lanternside",
               "Mossbank", "Nightingale", "Olivehurst", "Pennywhistle", "Quincefield", "Ravenscar",
               "Stonebridge", "Tidewater", "Upland", "Verdant", "Whitlock", "Yellowstone", "Ashdown",
               "Briarwood", "Cloverleaf", "Dovetail", "Emberly", "Fairhaven", "Goldcrest", "Heatherwick",
               "Ivorydale", "Jetstream", "Knotwood", "Lakemont", "Meadowlark", "Nutfield", "Orrinshaw",
               "Primrose", "Quarterdeck", "Rushmoor", "Sandpiper", "Tallowfield", "Ullswater",
               "Vineyard", "Windermoor", "Yeoman", "Zenith Hollow"]
INDUSTRIES = ["Tiles", "Logistics", "Analytics", "Provisions"]
SUFFIXES = ["LLP", "Ltd", "LLC", "GmbH", "Inc", "Co", "Group", "Trading"]
INVOICE_STYLES = ["year_seq", "slash", "alnum"]
PAYMENT_STYLES = ["pay", "rcpt", "bnk"]
ENTITY_STYLES = ["cust", "c", "cust_alpha"]


def _rng_pair(name, v, lo_min=0):
    if (not isinstance(v, (list, tuple)) or len(v) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in v)
            or v[0] < lo_min or v[0] > v[1]):
        raise ConfigError(f"{name}: expected [lo, hi] ints with lo>={lo_min}, got {v!r}")
    return [int(v[0]), int(v[1])]


def _check_offset(s):
    import re
    if not isinstance(s, str) or not re.fullmatch(r"[+-]\d{2}:\d{2}", s):
        raise ConfigError(f"bad utc offset {s!r}")
    if int(s[1:3]) > 14 or int(s[4:6]) > 59:
        raise ConfigError(f"bad utc offset {s!r}")


def _offset_secs(s: str) -> int:
    sign = -1 if s[0] == "-" else 1
    return sign * (int(s[1:3]) * 3600 + int(s[4:6]) * 60)


def _require_differing_offset(company_off: str, customer_offsets) -> None:
    """The generator needs at least one customer offset >= 1h away from the company offset
    (timezone-boundary distractors, boundary payments, tricky post-period invoices)."""
    if not any(abs(_offset_secs(o) - _offset_secs(company_off)) >= 3600 for o in customer_offsets):
        raise ConfigError(
            f"company.customer_offsets must contain at least one offset differing by >= 1 hour from the "
            f"company utc offset {company_off}")


_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
MIN_YEAR, MAX_YEAR = 1990, 2100
# upper bounds of the per-type [lo, hi] ranges (the total error cap of MAX_TOTAL_ISSUES is applied afterwards)
MAX_RANGE_HI = {"issues": MAX_TOTAL_ISSUES, "interactions": MAX_TOTAL_ISSUES // 2, "distractors": 20}
# keys a `force.company` block may pin (see module docstring)
FORCE_COMPANY_KEYS = frozenset({"base_currency", "timezone", "year", "month", "n_customers", "n_invoices",
                                "n_noise_postings", "foreign_currencies", "customer_offsets",
                                "foreign_invoice_share"})


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _unit_number(name, v) -> float:
    """A probability / share: a real number (not bool, not NaN/inf) in [0, 1]."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
        raise ConfigError(f"{name}: expected a number in [0, 1], got {v!r}")
    return float(v) if isinstance(v, float) else v


def _currency_list(name, v, allow_empty=False):
    if (not isinstance(v, (list, tuple)) or (not v and not allow_empty)
            or any(not isinstance(x, str) or x not in CURRENCY_EXPONENT for x in v)):
        raise ConfigError(f"{name}: expected a {'possibly empty ' if allow_empty else 'non-empty '}"
                          f"list of known currencies, got {v!r}")


def _offset_list(name, v):
    if not isinstance(v, (list, tuple)) or not v:
        raise ConfigError(f"{name}: expected a non-empty list of utc offsets")
    for o in v:
        _check_offset(o)


def _tz_pair(name, v):
    if not isinstance(v, (list, tuple)) or len(v) != 2 or not isinstance(v[0], str) or not v[0]:
        raise ConfigError(f"{name}: expected [name, utc_offset], got {v!r}")
    _check_offset(v[1])


def _vat_list(name, v):
    if not isinstance(v, (list, tuple)) or not v:
        raise ConfigError(f"{name}: expected a non-empty list of rates")
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (str, int, float)):
            raise ConfigError(f"{name}: bad rate {x!r}")
        try:
            d = Decimal(str(x))
        except InvalidOperation:
            raise ConfigError(f"{name}: bad rate {x!r}") from None
        if not d.is_finite() or not 0 <= d < 1:
            raise ConfigError(f"{name}: rate {x!r} must lie in [0, 1)")


def _validate_force_company(fc) -> None:
    if not isinstance(fc, dict):
        raise ConfigError("force.company must be an object")
    for k in fc:
        if k not in FORCE_COMPANY_KEYS:
            raise ConfigError(f"unknown force.company key {k!r}")
    if "base_currency" in fc and fc["base_currency"] not in CURRENCY_EXPONENT:
        raise ConfigError("force.company.base_currency: unknown currency")
    if "timezone" in fc:
        _tz_pair("force.company.timezone", fc["timezone"])
    if "year" in fc and not (_is_int(fc["year"]) and MIN_YEAR <= fc["year"] <= MAX_YEAR):
        raise ConfigError(f"force.company.year must be an int in [{MIN_YEAR}, {MAX_YEAR}]")
    if "month" in fc and not (_is_int(fc["month"]) and 1 <= fc["month"] <= 12):
        raise ConfigError("force.company.month must be an int in [1, 12]")
    for k in ("n_customers", "n_invoices", "n_noise_postings"):
        if k in fc and not _is_int(fc[k]):
            raise ConfigError(f"force.company.{k} must be an int")
    if "foreign_currencies" in fc:
        _currency_list("force.company.foreign_currencies", fc["foreign_currencies"])
    if "customer_offsets" in fc:
        _offset_list("force.company.customer_offsets", fc["customer_offsets"])
    if "foreign_invoice_share" in fc:
        _unit_number("force.company.foreign_invoice_share", fc["foreign_invoice_share"])


def validate_config(cfg: dict) -> dict:
    """Strict validation: unknown top-level / company keys, a missing or malformed `config_id` /
    `config_version` (semver `X.Y.Z`; never defaulted), out-of-range probabilities and malformed values all
    raise ConfigError. Returns a defaults-filled deep copy."""
    if not isinstance(cfg, dict):
        raise ConfigError("config must be an object")
    for k in cfg:
        if k not in DEFAULT_CONFIG:
            raise ConfigError(f"unknown config key {k!r}")
    cid, cver = cfg.get("config_id"), cfg.get("config_version")
    if not isinstance(cid, str) or not _ID_RE.fullmatch(cid):
        raise ConfigError("config_id is required: a string matching [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
    if not isinstance(cver, str) or not _VERSION_RE.fullmatch(cver):
        raise ConfigError("config_version is required and must match ^\\d+\\.\\d+\\.\\d+$ (e.g. '1.2.0')")
    out = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in cfg.items():
        if k in ("company", "issues", "interactions", "distractors"):
            if not isinstance(v, dict):
                raise ConfigError(f"{k} must be an object")
            if k == "company":
                for ck in v:
                    if ck not in DEFAULT_CONFIG["company"]:
                        raise ConfigError(f"unknown company key {ck!r}")
            out[k].update(copy.deepcopy(v))
        else:
            out[k] = copy.deepcopy(v)
    c = out["company"]
    for k in ("n_customers", "n_invoices", "n_noise_postings"):
        c[k] = _rng_pair("company." + k, c[k])
    if c["n_invoices"][0] < MIN_INVOICES or c["n_invoices"][1] > MAX_INVOICES:
        raise ConfigError(f"company.n_invoices must lie within [{MIN_INVOICES}, {MAX_INVOICES}] "
                          "(supported episode size: 20-200 documents)")
    if c["n_customers"][0] < MIN_CUSTOMERS or c["n_customers"][1] > MAX_CUSTOMERS:
        raise ConfigError(f"company.n_customers must lie within [{MIN_CUSTOMERS}, {MAX_CUSTOMERS}]")
    if c["n_noise_postings"][1] > MAX_NOISE_POSTINGS:
        raise ConfigError(f"company.n_noise_postings upper bound must be <= {MAX_NOISE_POSTINGS} "
                          "(supported episode size: 200-2000 ledger rows)")
    _currency_list("company.base_currencies", c["base_currencies"])
    _currency_list("company.foreign_currencies", c["foreign_currencies"])
    if not isinstance(c["timezones"], (list, tuple)) or not c["timezones"]:
        raise ConfigError("company.timezones must be a non-empty list")
    for tz in c["timezones"]:
        _tz_pair("company.timezones entry", tz)
    _offset_list("company.customer_offsets", c["customer_offsets"])
    for name, off in c["timezones"]:
        _require_differing_offset(off, c["customer_offsets"])
    if (not isinstance(c["years"], (list, tuple)) or not c["years"]
            or not all(_is_int(y) and MIN_YEAR <= y <= MAX_YEAR for y in c["years"])):
        raise ConfigError(f"company.years: expected a non-empty list of ints in [{MIN_YEAR}, {MAX_YEAR}]")
    c["foreign_invoice_share"] = _unit_number("company.foreign_invoice_share", c["foreign_invoice_share"])
    _vat_list("company.vat_rates", c["vat_rates"])
    for sec, names in (("issues", ISSUE_TYPES), ("interactions", INTERACTIONS), ("distractors", DISTRACTORS)):
        for n in out[sec]:
            if n not in names:
                raise ConfigError(f"unknown {sec} key {n!r}")
        for n in names:
            out[sec].setdefault(n, [0, 0])
            out[sec][n] = _rng_pair(f"{sec}.{n}", out[sec][n])
            if out[sec][n][1] > MAX_RANGE_HI[sec]:
                raise ConfigError(f"{sec}.{n}: upper bound must be <= {MAX_RANGE_HI[sec]}")
    out["clean_probability"] = _unit_number("clean_probability", out["clean_probability"])
    out["force"] = out.get("force") or {}
    if not isinstance(out["force"], dict):
        raise ConfigError("force must be an object")
    for k in out["force"]:
        if k not in ("clean", "issues", "interactions", "distractors", "company"):
            raise ConfigError(f"unknown force key {k!r}")
    if "company" in out["force"]:
        _validate_force_company(out["force"]["company"])
    for k in ("issues", "interactions", "distractors"):
        if k in out["force"] and not isinstance(out["force"][k], dict):
            raise ConfigError(f"force.{k} must be an object")
    if "clean" in out["force"] and not isinstance(out["force"]["clean"], bool):
        raise ConfigError("force.clean must be a boolean")
    return out


def config_hash(cfg: dict) -> str:
    """Canonical content hash (sha256 hex) of a RESOLVED config, `force` block included. This is what the
    version lock (`configs/CONFIG_VERSIONS.json`) pins to a `config_id` + `config_version`."""
    return hashlib.sha256(dumps(cfg).encode("utf-8")).hexdigest()


LOCK_FILE = "CONFIG_VERSIONS.json"
_NON_CONFIG_FILES = ("splits.json", LOCK_FILE)


def shipped_config_hashes(config_dir: str | None = None) -> dict:
    """{ "<config_id>@<config_version>": {"file": name, "config_hash": ...} } for every configs/*.json
    (splits.json and the lock file excluded)."""
    d = config_dir or CONFIG_DIR
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json") or fn in _NON_CONFIG_FILES:
            continue
        with open(os.path.join(d, fn), encoding="utf-8") as f:
            cfg = validate_config(json.load(f))
        out[f"{cfg['config_id']}@{cfg['config_version']}"] = {"file": fn, "config_hash": config_hash(cfg)}
    return out


def build_config_lock(config_dir: str | None = None) -> dict:
    """Content of `configs/CONFIG_VERSIONS.json`: maps `config_id@config_version` -> config_hash."""
    return {"_comment": "Maps config_id@config_version to the sha256 of the resolved config. Bump config_version "
                        "whenever a config's content changes, then regenerate this file "
                        "(python -m auditgym lock --write).",
            "versions": {k: v["config_hash"] for k, v in shipped_config_hashes(config_dir).items()}}


def verify_config_lock(config_dir: str | None = None) -> list[str]:
    """Problems found comparing the shipped configs with the lock file (empty list = consistent).
    A shipped config whose content hash changed without a version bump, a shipped config missing from the
    lock and a lock entry without a config are all reported."""
    d = config_dir or CONFIG_DIR
    path = os.path.join(d, LOCK_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            lock = json.load(f).get("versions")
    except (OSError, ValueError, AttributeError):
        return [f"{LOCK_FILE} missing or unreadable"]
    if not isinstance(lock, dict):
        return [f"{LOCK_FILE}: no 'versions' object"]
    problems = []
    cur = shipped_config_hashes(d)
    for key, v in cur.items():
        if key not in lock:
            problems.append(f"{key} ({v['file']}): not in {LOCK_FILE}")
        elif lock[key] != v["config_hash"]:
            problems.append(f"{key} ({v['file']}): content changed (hash {v['config_hash'][:12]} != locked "
                            f"{str(lock[key])[:12]}) without a config_version bump")
    for key in lock:
        if key not in cur:
            problems.append(f"{key}: locked but no shipped config has this id@version")
    return problems


def resolve_config(config) -> dict:
    """dict | path | name in configs/  ->  validated config dict (deep copy)."""
    if isinstance(config, dict):
        return validate_config(config)
    if isinstance(config, str):
        cands = [config, os.path.join(CONFIG_DIR, config), os.path.join(CONFIG_DIR, config + ".json")]
        for p in cands:
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    return validate_config(json.load(f))
        raise ConfigError(f"config not found: {config!r}")
    raise ConfigError("config must be a dict, path or name")


def config_with_force(config, force: dict | None) -> dict:
    """Load a config (name/path/dict) and set its `force` block (used by splits.json holdout entries)."""
    cfg = resolve_config(config)
    cfg["force"] = copy.deepcopy(force or {})
    return validate_config(cfg)


def _force_count(v, name):
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ConfigError(f"force.{name} must be a non-negative int")
    return v


INTERACTION_WEIGHT = 0.3     # relative chance (per added unit) of growing an interaction vs. a plain issue


def _sample_error_counts(r, cfg: dict) -> tuple[dict, dict]:
    """Natural (unforced) error counts.  The TOTAL error count (an interaction counts 2) is drawn first, skewed to
    small/medium values, in [sum of lower bounds, min(MAX_TOTAL_ISSUES, sum of upper bounds)], and then distributed over the types
    that still have room under their own [lo, hi] range.  Drawing every type independently and trimming
    afterwards (the old scheme) piled up at the cap of 12 and put an interaction in ~3/4 of all episodes."""
    ranges = {("i", t): cfg["issues"][t] for t in ISSUE_TYPES}
    ranges.update({("x", t): cfg["interactions"][t] for t in INTERACTIONS})
    cost = {"i": 1, "x": 2}
    counts = {k: lo for k, (lo, hi) in ranges.items()}

    def total():
        return sum(cost[k[0]] * v for k, v in counts.items())

    if total() > MAX_TOTAL_ISSUES:           # lower bounds alone exceed the cap: trim randomly
        keys = sorted(ranges)
        while total() > MAX_TOTAL_ISSUES:
            nz = [k for k in keys if counts[k] > 0]
            counts[nz[r.randrange(len(nz))]] -= 1
    else:
        hi_total = min(MAX_TOTAL_ISSUES, sum(cost[k[0]] * hi for k, (lo, hi) in ranges.items()))
        lo_total = total()
        # min of two uniform draws: favours small/medium totals so the cap of 12 is rarely reached
        # (a non-clean episode carries at least one error whenever the ranges allow it)
        lo_draw = max(lo_total, 1) if hi_total >= 1 else lo_total
        target = min(r.randint(lo_draw, hi_total), r.randint(lo_draw, hi_total))
        keys = sorted(ranges)
        while total() < target:
            room = [k for k in keys if counts[k] < ranges[k][1] and total() + cost[k[0]] <= target]
            if not room:
                break
            w = [INTERACTION_WEIGHT if k[0] == "x" else 1.0 for k in room]
            counts[r.choices(room, w)[0]] += 1
    return ({t: counts[("i", t)] for t in ISSUE_TYPES}, {t: counts[("x", t)] for t in INTERACTIONS})


def sample_spec(seed: int, cfg: dict) -> dict:
    """Sample a concrete, JSON-serialisable EpisodeSpec from config ranges (sub_rng(seed,'spec'))."""
    r = sub_rng(seed, "spec")
    force = cfg.get("force") or {}
    comp = cfg["company"]
    fcomp = force.get("company") or {}

    base = r.choice(comp["base_currencies"])
    tzname, off = r.choice(comp["timezones"])
    year = r.choice(comp["years"])
    month = r.randint(1, 12)
    n_customers = r.randint(*comp["n_customers"])
    n_invoices = r.randint(*comp["n_invoices"])
    n_noise = r.randint(*comp["n_noise_postings"])
    pool = [c for c in comp["foreign_currencies"] if c != base]
    if not pool:
        raise ConfigError("no foreign currency available for base " + base)
    k = r.randint(1, min(3, len(pool)))
    foreign = r.sample(pool, k)
    offsets = list(comp["customer_offsets"])
    share = str(comp["foreign_invoice_share"])

    # company-level pins
    if "base_currency" in fcomp:
        base = fcomp["base_currency"]
        foreign = [c for c in foreign if c != base] or [c for c in comp["foreign_currencies"] if c != base][:2]
    if "timezone" in fcomp:
        tzname, off = fcomp["timezone"]
    if "year" in fcomp:
        year = int(fcomp["year"])
    if "month" in fcomp:
        month = int(fcomp["month"])
    if "n_customers" in fcomp:
        n_customers = int(fcomp["n_customers"])
    if "n_invoices" in fcomp:
        n_invoices = int(fcomp["n_invoices"])
    if "n_noise_postings" in fcomp:
        n_noise = int(fcomp["n_noise_postings"])
    if "foreign_currencies" in fcomp:
        foreign = [c for c in fcomp["foreign_currencies"] if c != base]
    if "customer_offsets" in fcomp:
        offsets = list(fcomp["customer_offsets"])
    if "foreign_invoice_share" in fcomp:
        share = str(fcomp["foreign_invoice_share"])
    if not MIN_INVOICES <= n_invoices <= MAX_INVOICES:
        raise ConfigError(f"company.n_invoices must lie within [{MIN_INVOICES}, {MAX_INVOICES}]")
    if not MIN_CUSTOMERS <= n_customers <= MAX_CUSTOMERS:
        raise ConfigError(f"company.n_customers must lie within [{MIN_CUSTOMERS}, {MAX_CUSTOMERS}]")
    if not 0 <= n_noise <= MAX_NOISE_POSTINGS:
        raise ConfigError(f"company.n_noise_postings must lie within [0, {MAX_NOISE_POSTINGS}]")
    if not foreign:
        raise ConfigError("no foreign currency available (check company.foreign_currencies)")
    for o in offsets:
        _check_offset(o)
    _check_offset(off)
    _require_differing_offset(off, offsets)

    cp = r.random()
    issues, inter = _sample_error_counts(r, cfg)
    dis = {t: r.randint(*cfg["distractors"][t]) for t in DISTRACTORS}
    for t, v in (force.get("distractors") or {}).items():
        if t not in DISTRACTORS:
            raise ConfigError(f"unknown distractor {t!r}")
        dis[t] = _force_count(v, "distractors." + t)

    forced_issues = "issues" in force or "interactions" in force
    if force.get("clean") is True:
        clean = True
    elif forced_issues:
        clean = False
    else:
        clean = cp < float(cfg["clean_probability"])
    if clean:
        issues = {t: 0 for t in ISSUE_TYPES}
        inter = {t: 0 for t in INTERACTIONS}
    elif forced_issues:
        fi, fn = force.get("issues") or {}, force.get("interactions") or {}
        for t in fi:
            if t not in ISSUE_TYPES:
                raise ConfigError(f"unknown issue type {t!r}")
        for t in fn:
            if t not in INTERACTIONS:
                raise ConfigError(f"unknown interaction {t!r}")
        issues = {t: _force_count(fi.get(t, 0), "issues." + t) for t in ISSUE_TYPES}
        inter = {t: _force_count(fn.get(t, 0), "interactions." + t) for t in INTERACTIONS}
        if sum(issues.values()) + 2 * sum(inter.values()) > MAX_TOTAL_ISSUES:
            raise ConfigError(f"forced issues exceed cap of {MAX_TOTAL_ISSUES}")

    industry = r.choice(INDUSTRIES)
    company_name = f"Fictional {r.choice(COMPANY_ADJ)} {industry} {r.choice(SUFFIXES)}"
    spec = {
        "clean": bool(clean),
        "company_name": company_name,
        "industry": industry,
        "base_currency": base,
        "timezone": tzname,
        "utc_offset": off,
        "year": int(year),
        "month": int(month),
        "n_customers": n_customers,
        "n_invoices": n_invoices,
        "n_noise_postings": n_noise,
        "foreign_currencies": list(foreign),
        "foreign_invoice_share": share,
        "customer_offsets": offsets,
        "vat_rates": [str(v) for v in comp["vat_rates"]],
        "issues": issues,
        "interactions": inter,
        "distractors": dis,
        "id_styles": {"invoice": r.choice(INVOICE_STYLES), "payment": r.choice(PAYMENT_STYLES),
                      "entity": r.choice(ENTITY_STYLES)},
    }
    return spec
