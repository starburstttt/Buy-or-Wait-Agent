"""Detect recurring commitments and estimate variable essential spending.

RECURRENCE RULE
---------------
A (category, description) group is a recurring series when:

  1. it has at least MIN_OCCURRENCES_FOR_RECURRENCE occurrences, and
  2. every consecutive gap fits ONE cadence band (weekly / fortnightly / monthly).

Amount stability is deliberately NOT part of the test. Cadence is what makes
something recurring; the amount is a separate question about what to project.
Requiring identical amounts silently deletes real income: user_06 and user_08
both have five consecutive monthly payroll credits where the amount drops
part-way through (1441 -> 1037.52, 1422.85 -> 782.57). Treating those as
non-recurring forecasts zero income for 90 days and collapses the balance.

The projected amount is the LATEST observed amount, which follows the spec's
conflict rule that a newer record from the same source wins - and is also the
financially safer reading of a pay cut.

THIN SERIES ARE NOT A RECURRENCE PROBLEM
----------------------------------------
user_11's 'Weekend food delivery' appears twice, 126 days apart. That is not a
cadence and must not be projected as one. It still matters, because the spending
-change search needs a handle on it, but that is a question about flexibility.
Variable spend is forecast at CATEGORY level precisely because these descriptions
rotate across a pool of labels ('Bulk pantry shop', 'Fresh food shop', ...) and
never form a per-description series.

INCOME IS NOT SYMMETRIC WITH SPEND
----------------------------------
Variable flows are debit-only. Irregular credits are not projected forward at all,
because the spec forbids inventing unsupported future income - user_10's gig
payouts rotate across four descriptions at irregular gaps and are explicitly
flagged as not-yet-withdrawable. A documented monthly salary is different, and
gets the narrow fallback in _salary_fallback().
"""

import collections
import datetime as dt
import statistics
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from entities import Currency, Dataset, Direction, Event, Flexibility
from netting import SALARY_CATEGORY, clean_history, future_cash_events, to_home

MIN_OCCURRENCES_FOR_RECURRENCE = 3

# (label, low, high, is_monthly) - a series must sit entirely inside one band.
CADENCE_BANDS = (
    ("weekly", 5, 9, False),
    ("fortnightly", 12, 16, False),
    ("monthly", 26, 32, True),
)

# How to turn several months of observed variable spend into one forecast rate.
# The spec only says "conservatively", which is not a number. This is the knob
# calibrate.py fits: "mean" trusts the average, "max" assumes the worst month
# repeats, "p75" sits between them.
VariableAggregation = Literal["mean", "p75", "max"]
# Fitted against the 25 sample rows: "mean" centres the error distribution exactly
# (median actual/expected = 1.000); p75 and max both overstate spend. The spec's
# "conservatively" turns out to be about not counting unconfirmed INCOME, not about
# padding the spend estimate.
VARIABLE_SPEND_AGGREGATION: VariableAggregation = "mean"

VARIABLE_LOOKBACK_DAYS = 180

# A series whose last occurrence is older than this many cadences has lapsed and is
# not projected forward. Without it, user_05's payroll - which stops on 2025-09-15,
# 52 days before the request - keeps paying out for the whole horizon and reports
# the full requested amount as safe when the true answer is almost nothing.
STALE_CADENCE_MULTIPLE = Decimal("1.5")

# Irregular salary (freelance/gig) projected at its observed monthly rate when the
# stream is demonstrably still running. See _irregular_income_rate for why recency
# rather than a scheduled future row is the right guard against inventing income.
PROJECT_IRREGULAR_INCOME = True
IRREGULAR_INCOME_MIN_OCCURRENCES = 3
IRREGULAR_INCOME_STALE_MULTIPLE = Decimal("1.5")


@dataclass(frozen=True, slots=True)
class RecurringSeries:
    category: str
    description: str
    amount_home: Decimal
    direction: Direction
    cadence_days: int
    is_monthly: bool
    last_date: dt.date
    latest_event_id: str
    flexibility: Flexibility
    minimum_allowed_home: Decimal | None

    @property
    def signed_amount(self) -> Decimal:
        return self.amount_home if self.direction is Direction.CREDIT else -self.amount_home


@dataclass(frozen=True, slots=True)
class VariableFlow:
    category: str
    monthly_amount_home: Decimal
    latest_event_id: str
    months_observed: int


@dataclass(frozen=True, slots=True)
class SpendingProfile:
    series: tuple[RecurringSeries, ...]
    variable: tuple[VariableFlow, ...]
    unresolved_amount_events: tuple[str, ...]
    monthly_income_home: Decimal = Decimal(0)


def _gaps(dates: list[dt.date]) -> list[int]:
    return [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]


def _classify_cadence(gaps: list[int]) -> tuple[int, bool] | None:
    if not gaps:
        return None
    for _, low, high, is_monthly in CADENCE_BANDS:
        if all(low <= gap <= high for gap in gaps):
            return int(round(statistics.fmean(gaps))), is_monthly
    return None


def _is_stale(series: RecurringSeries, as_of: dt.date) -> bool:
    if series.last_date >= as_of:
        return False
    allowed = int(Decimal(series.cadence_days) * STALE_CADENCE_MULTIPLE)
    return (as_of - series.last_date).days > allowed


def _aggregate(values: list[Decimal], how: VariableAggregation) -> Decimal:
    if not values:
        return Decimal(0)
    ordered = sorted(values)
    match how:
        case "mean":
            return sum(ordered) / len(ordered)
        case "max":
            return ordered[-1]
        case "p75":
            if len(ordered) == 1:
                return ordered[0]
            idx = min(len(ordered) - 1, int(0.75 * (len(ordered) - 1) + 0.5))
            return ordered[idx]


def _to_series(
    data: Dataset, events: list[Event], home: Currency, cadence: tuple[int, bool]
) -> RecurringSeries:
    last = events[-1]
    on = last.settlement_date or last.event_date
    minimum_home = (
        to_home(data, last.minimum_allowed_amount, last.currency, home, on)
        if last.minimum_allowed_amount is not None
        else None
    )
    return RecurringSeries(
        category=last.category,
        description=last.description,
        # Freeze the converted amount at the last observed occurrence rather than
        # projecting an FX path: exchange_rates.csv only covers dates that actually
        # occur, and inventing future rates would be inventing financial facts.
        amount_home=to_home(data, last.amount, last.currency, home, on),
        direction=last.direction,
        cadence_days=cadence[0],
        is_monthly=cadence[1],
        last_date=last.event_date,
        latest_event_id=last.event_id,
        flexibility=last.flexibility,
        minimum_allowed_home=minimum_home,
    )


def _salary_fallback(
    data: Dataset, salary_events: list[Event], home: Currency, as_of: dt.date
) -> RecurringSeries | None:
    """Project salary that is documented but split across descriptions.

    user_01 has exactly two salary rows - 'Prorated first salary' (settled) and
    'Next confirmed salary' (scheduled) - 31 days apart. No per-description series
    exists, yet the salary is plainly monthly and confirmed. Grouping by category
    recovers it.

    Bridging descriptions is only safe when the salary is confirmed GOING FORWARD,
    so a scheduled future row is required. Inferring continuation from past records
    alone invents income the data does not support, and the dataset punishes it
    directly: user_05's five monthly payroll rows end with one described 'Final
    employer payroll', and user_12's contract income simply stops 80 days before the
    request. Both must forecast zero future salary. Requiring a scheduled row also
    excludes gig earners such as user_10, whose 21 payouts are all historical.
    """
    if not any(
        e.settlement_date is not None and e.settlement_date >= as_of for e in salary_events
    ):
        return None
    usable = sorted(
        (e for e in salary_events if e.amount is not None), key=lambda e: e.event_date
    )
    if len(usable) < 2:
        return None
    cadence = _classify_cadence(_gaps([e.event_date for e in usable]))
    if cadence is None or not cadence[1]:
        return None
    return _to_series(data, usable, home, cadence)


def _irregular_income_rate(
    data: Dataset,
    events: tuple[Event, ...],
    home: Currency,
    as_of: dt.date,
    aggregation: VariableAggregation,
) -> Decimal:
    """Monthly rate for salary that is real but too irregular to form a series.

    Freelance and gig income does not fit CADENCE_BANDS and never has a scheduled
    future row for _salary_fallback to lean on. user_09 is the clean case: ten
    salary credits landing on the 7th and the 20th, so the gaps alternate
    13/15/17/18 and no single band covers them, and every row is settled history.
    The existing rules forecast that user zero income for 90 days while ~1,000 EUR
    a month keeps arriving, which drags the projected trough far below the truth.

    The guard against inventing income is RECENCY rather than a scheduled row: the
    stream has to still be running as of request_date, measured against its own
    typical gap. That still excludes the cases the dataset punishes for optimism -
    user_05's payroll stops 52 days before the request on a row described 'Final
    employer payroll', and user_12's contract income stops 80 days out. Both are
    stale against their own cadence and keep forecasting zero.
    """
    history = [
        event
        for event in clean_history(events, as_of)
        if event.category == SALARY_CATEGORY
        and event.direction is Direction.CREDIT
        and event.amount is not None
    ]
    if len(history) < IRREGULAR_INCOME_MIN_OCCURRENCES:
        return Decimal(0)

    history.sort(key=lambda event: event.event_date)
    typical_gap = Decimal(str(statistics.median(_gaps([e.event_date for e in history]))))
    if typical_gap <= 0:
        return Decimal(0)
    since = Decimal((as_of - history[-1].event_date).days)
    if since > typical_gap * IRREGULAR_INCOME_STALE_MULTIPLE:
        return Decimal(0)

    monthly: dict[tuple[int, int], Decimal] = collections.defaultdict(Decimal)
    for event in history:
        on = event.settlement_date or event.event_date
        monthly[(on.year, on.month)] += to_home(data, event.amount, event.currency, home, on)
    buckets = [monthly[key] for key in sorted(monthly)]
    # Same partial-month trim as variable spend: the first and last calendar months
    # of a window are usually incomplete and understate the rate.
    if len(buckets) >= 4:
        buckets = buckets[1:-1]
    return _aggregate(buckets, aggregation)


def build_spending_profile(
    data: Dataset,
    events: tuple[Event, ...],
    home: Currency,
    as_of: dt.date,
    *,
    aggregation: VariableAggregation | None = None,
) -> SpendingProfile:
    aggregation = aggregation or VARIABLE_SPEND_AGGREGATION
    history = clean_history(events, as_of)
    unresolved = tuple(e.event_id for e in history if e.amount is None)
    usable = [e for e in history if e.amount is not None]

    grouped: dict[tuple[str, str], list[Event]] = collections.defaultdict(list)
    for event in usable:
        grouped[(event.category, event.description)].append(event)

    series: list[RecurringSeries] = []
    consumed: set[str] = set()
    for events_in_group in grouped.values():
        events_in_group.sort(key=lambda e: e.event_date)
        if len(events_in_group) < MIN_OCCURRENCES_FOR_RECURRENCE:
            continue
        cadence = _classify_cadence(_gaps([e.event_date for e in events_in_group]))
        if cadence is None:
            continue
        candidate = _to_series(data, events_in_group, home, cadence)
        if _is_stale(candidate, as_of):
            continue
        series.append(candidate)
        consumed.update(e.event_id for e in events_in_group)

    if not any(s.category == SALARY_CATEGORY for s in series):
        # Include scheduled future salary so the anchor carries the confirmed amount
        # rather than a stale historical one.
        salary_events = [e for e in usable if e.category == SALARY_CATEGORY]
        salary_events += [
            e
            for e in future_cash_events(events, as_of)
            if e.category == SALARY_CATEGORY and e.amount is not None
        ]
        fallback = _salary_fallback(data, salary_events, home, as_of)
        if fallback is not None and not _is_stale(fallback, as_of):
            series.append(fallback)

    window_start = as_of - dt.timedelta(days=VARIABLE_LOOKBACK_DAYS)
    leftovers = [
        e
        for e in usable
        if e.event_id not in consumed
        and e.direction is Direction.DEBIT
        and e.event_date >= window_start
    ]

    by_category: dict[str, list[Event]] = collections.defaultdict(list)
    for event in leftovers:
        by_category[event.category].append(event)

    variable: list[VariableFlow] = []
    for category, events_in_category in by_category.items():
        monthly: dict[tuple[int, int], Decimal] = collections.defaultdict(Decimal)
        for event in events_in_category:
            on = event.settlement_date or event.event_date
            monthly[(on.year, on.month)] += to_home(data, event.amount, event.currency, home, on)
        buckets = [monthly[k] for k in sorted(monthly)]
        # First and last calendar months are usually partial and bias the rate
        # downward; drop them once there is enough history to afford it.
        if len(buckets) >= 4:
            buckets = buckets[1:-1]
        latest = max(events_in_category, key=lambda e: e.event_date)
        variable.append(
            VariableFlow(
                category=category,
                monthly_amount_home=_aggregate(buckets, aggregation),
                latest_event_id=latest.event_id,
                months_observed=len(buckets),
            )
        )

    monthly_income = Decimal(0)
    if PROJECT_IRREGULAR_INCOME and not any(s.category == SALARY_CATEGORY for s in series):
        # A scheduled future salary row is already applied to the timeline as an
        # explicit event, so dripping a rate on top of it would pay the user twice.
        scheduled_ahead = any(
            event.category == SALARY_CATEGORY and event.direction is Direction.CREDIT
            for event in future_cash_events(events, as_of)
        )
        if not scheduled_ahead:
            monthly_income = _irregular_income_rate(data, events, home, as_of, aggregation)

    return SpendingProfile(
        series=tuple(sorted(series, key=lambda s: (s.category, s.description))),
        variable=tuple(sorted(variable, key=lambda v: v.category)),
        unresolved_amount_events=unresolved,
        monthly_income_home=monthly_income,
    )
