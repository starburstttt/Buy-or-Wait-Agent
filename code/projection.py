"""Project a daily balance timeline and answer the two safety questions.

HORIZON ANCHORING
-----------------
The window is fixed at [request_date, request_date + 90] for every question asked
about a request. It does NOT slide forward with each candidate payment date.

Reasoning: the spec says "forecast the user's balance for the next 90 days" and
judges a plan by whether it "cannot be completed safely within 90 days" - both
phrased relative to the request, not relative to a payment. A sliding window would
also make the answer self-referential: a later candidate payment date would buy
itself extra runway by pushing the horizon out, so a plan could pass simply by
starting later. Anchoring at request_date keeps every candidate scored against the
same stretch of time, which is what makes them comparable.
"""

import datetime as dt
from calendar import monthrange
from dataclasses import dataclass
from decimal import Decimal

from entities import Currency, Dataset, Request
from netting import future_cash_events, signed_home_amount
from recurrence import SpendingProfile

FORECAST_HORIZON_DAYS = 90
DAYS_PER_MONTH = Decimal(30)

# A projected recurring occurrence within this many days of an explicitly dated
# future event in the same category is assumed to BE that event, not a second one.
DEDUPE_WINDOW_DAYS = 6


@dataclass(frozen=True, slots=True)
class Timeline:
    start: dt.date
    balances: list[Decimal]

    @property
    def end(self) -> dt.date:
        return self.start + dt.timedelta(days=len(self.balances) - 1)

    def balance_on(self, day: dt.date) -> Decimal:
        return self.balances[(day - self.start).days]


def _add_months(anchor: dt.date, months: int) -> dt.date:
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    return dt.date(year, month, min(anchor.day, monthrange(year, month)[1]))


def build_timeline(data: Dataset, request: Request, spending: SpendingProfile) -> Timeline:
    profile = data.profiles[request.user_id]
    home: Currency = profile.home_currency
    start = request.request_date
    span = FORECAST_HORIZON_DAYS + 1
    deltas = [Decimal(0)] * span
    dated: list[tuple[str, dt.date]] = []

    for event in future_cash_events(data.events_by_user[request.user_id], start):
        offset = (event.settlement_date - start).days
        if not 0 <= offset < span:
            continue
        amount = signed_home_amount(data, event, home)
        if amount is None:
            continue
        deltas[offset] += amount
        dated.append((event.category, event.settlement_date))

    horizon_end = start + dt.timedelta(days=FORECAST_HORIZON_DAYS)
    for series in spending.series:
        step = 1
        while True:
            # Monthly series step by calendar month to preserve day-of-month. Averaging
            # 28/30/31-day gaps into a fixed stride drifts a salary off the 15th within
            # two cycles, which moves the trough and the earliest-safe date with it.
            if series.is_monthly:
                occurrence = _add_months(series.last_date, step)
            else:
                occurrence = series.last_date + dt.timedelta(days=series.cadence_days * step)
            if occurrence > horizon_end:
                break
            step += 1
            if occurrence < start:
                continue
            if any(
                category == series.category and abs((occurrence - when).days) <= DEDUPE_WINDOW_DAYS
                for category, when in dated
            ):
                continue
            deltas[(occurrence - start).days] += series.signed_amount

    daily_variable = sum(
        (flow.monthly_amount_home for flow in spending.variable), Decimal(0)
    ) / DAYS_PER_MONTH
    # Irregular salary is dripped rather than dated for the same reason variable
    # spend is: the rate is well supported by history but the individual dates are
    # not, and inventing pay dates would put fake precision into the trough.
    daily_income = spending.monthly_income_home / DAYS_PER_MONTH
    for offset in range(1, span):
        deltas[offset] += daily_income - daily_variable

    balances: list[Decimal] = []
    running = profile.current_available_balance
    for delta in deltas:
        running += delta
        balances.append(running)
    return Timeline(start=start, balances=balances)


def amount_safe_to_pay(timeline: Timeline, minimum: Decimal, requested: Decimal) -> Decimal:
    """Most that can leave the account today without the window ever breaching minimum.

    Paying X on request_date shifts every later balance down by exactly X, so the
    binding constraint is simply the trough of the unpaid curve.
    """
    headroom = min(timeline.balances) - minimum
    return max(Decimal(0), min(headroom, requested))


def earliest_full_payment_date(
    timeline: Timeline, minimum: Decimal, requested: Decimal
) -> dt.date | None:
    """First day the whole amount could leave as ONE payment and still stay safe.

    Capacity only: ignores the user's accepted payment methods and any optional
    spending changes, per the spec's note that this field measures financial
    capacity independently of preference.
    """
    balances = timeline.balances
    suffix_min = [Decimal(0)] * len(balances)
    running = balances[-1]
    for i in range(len(balances) - 1, -1, -1):
        running = min(running, balances[i])
        suffix_min[i] = running
    threshold = minimum + requested
    for offset, value in enumerate(suffix_min):
        if value >= threshold:
            return timeline.start + dt.timedelta(days=offset)
    return None
