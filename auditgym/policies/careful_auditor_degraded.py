"""careful_auditor_degraded: the strong auditor with two realistic handicaps (EXTRA diagnostic policy).

``careful_auditor`` is saturated (F1 = 1.0), so by itself it cannot show how the reward behaves for an agent
that is *almost* right. This variant makes the two classic near-miss mistakes, measurably and reproducibly:

* ``step_cap``   - the agent stops inspecting after ``step_cap`` inspection steps (default 20; the full
  analysis needs ~24-26 on average and more on big episodes) and submits whatever its partial view supports
  (the "ran out of budget / gave up early" mistake). The cap bites on the per-invoice ``get_record`` fetches
  that come after the four full pagings, so it costs recall mostly on duplicate_invoice (measured: 10 of 13
  duplicates missed on the holdout).
* ``ignore_tz``  - the time-zone rule of the policy text is ignored. ``"period"`` (default): period
  membership is decided on the wall-clock fields of ``issued_at`` instead of on the instant (misses and
  fabricates ``period_cutoff`` findings exactly on the timezone-boundary distractors and the cutoff
  interaction). ``"all"`` additionally reads fx rate dates as written in the record's own offset instead of
  the company offset (false ``fx_error`` / ``payment_misallocation``). ``False``: honest.

Same honesty contract as ``careful_auditor``: observation-only, no truth, no generator.
"""
from __future__ import annotations

from .careful_auditor import CarefulAuditor, _parse_ts

DEFAULT_STEP_CAP = 20


class CarefulAuditorDegraded(CarefulAuditor):
    name = "careful_auditor_degraded"

    def __init__(self, seed: int = 0, step_cap: int | None = DEFAULT_STEP_CAP,
                 ignore_tz: bool | str = "period"):
        super().__init__(seed)
        if ignore_tz not in (False, "period", "all", True):
            raise ValueError("ignore_tz must be False, 'period' or 'all'")
        self.step_cap = step_cap
        self.ignore_tz = "all" if ignore_tz is True else ignore_tz
        self._budget_at_start = 0

    def on_reset(self) -> None:
        super().on_reset()
        self._budget_at_start = int(self.first_obs.get("steps_remaining", 0))

    # -- handicap: offsets ignored
    def company_date(self, ts: str) -> str:
        if self.ignore_tz != "all":
            return super().company_date(ts)
        return _parse_ts(ts).date().isoformat()          # the date as written, in the record's own offset

    def in_period(self, issued_at: str) -> bool:
        if not self.ignore_tz or not (self.period_start and self.period_end):
            return super().in_period(issued_at)
        naive = lambda d: d.replace(tzinfo=None)         # wall-clock fields, offsets dropped
        return naive(self.period_start) <= naive(_parse_ts(issued_at)) <= naive(self.period_end)

    # -- handicap: step cap
    def can_inspect(self, reserve: int = 1) -> bool:
        if self.step_cap is not None and self._budget_at_start - self.remaining >= self.step_cap:
            return False
        return super().can_inspect(reserve)
