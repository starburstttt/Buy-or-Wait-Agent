"""Search the permitted spending changes that make an unsafe plan safe.

WHAT MAY BE CHANGED: ONE ACTION PER CATEGORY
---------------------------------------------
Spending changes are category-level. The user's permissions are stated per category
(expense_categories_user_is_willing_to_reduce / ..._to_stop), the forecast carries
variable spend per category, and the dataset prices a reduction per category - all
six of user_11's dining descriptions share the same minimum_allowed_amount of
665950, which is a monthly dining budget, not six separate subscription limits.

So each (category, kind) yields at most one action, referencing the most recent
occurrence of that CATEGORY on or before request_date. A category is changeable
when:

  - it is NOT in expense_categories_to_protect
  - it is in the matching permission list for the action
  - the referenced event's flexibility allows it: `stop` needs stoppable,
    `reduce_to` needs reducible (reducible_or_stoppable allows either)
  - the category occurs more than once on or before request_date

Per-DESCRIPTION enumeration is what the naive reading gives, and it is wrong:
user_11 has three reducible dining descriptions, and picking the cheapest
sufficient one selects event_985 ('Neighbourhood restaurant') where ground truth
expects event_989 ('Weekend food delivery') - the latest dining event of any
description. Category-level targeting reproduces all four sample targets
(event_476, event_989, event_1815, event_1816); each is the most recent event in
its category.

NOT "a detected recurring series" either. event_989's own description group has
exactly two occurrences 126 days apart, which recurrence.py deliberately refuses to
project as a series because two points that far apart are not a cadence. The
dataset still expects it to be reducible, so the ">1 occurrence in the category"
test above is what "recurring expense" means here.

THE RESOLUTIONS THE SAMPLES PIN DOWN
-------------------------------------
target  - the most recent occurrence of the category on or before request_date.
amount  - reduce_to always goes to exactly minimum_allowed_amount, in the event's
          own currency (665950 for event_989, 23.50 for event_1816 - both are the
          event's minimum_allowed_amount verbatim).

HOW A CHANGE MOVES THE FORECAST
-------------------------------
Whichever component of the spending profile carries the event:

  a projected series  - `stop` drops the series, `reduce_to` re-prices it.
  a variable category - the category's monthly rate falls by the FULL per-occurrence
                        delta (old - new), floored at zero.

The variable case is a real modelling choice. The alternative - amortising the
delta across the months observed, which is what removing the event from a
mean-of-buckets rate would literally do - is too weak to reproduce the data:
request_11 is short 599,355 IDR and the reduce_to saves 497,580.49 per occurrence,
so the amortised reading (~124k/month over ~4 buckets, ~373k across the horizon)
leaves it unaffordable, while the full-delta reading (~1.49M across the horizon)
matches the expected affordable_with_plan. Reading it as "from now on this costs
the new amount per month" is also the plainer reading of what the user is agreeing
to when they accept a reduction.

SUFFICIENCY IS MEASURED, NOT ESTIMATED
--------------------------------------
Whether a change set rescues a plan is decided by re-running the safety check with
the change applied, never by comparing monthly savings to the shortfall. The
number of future occurrences that land before the binding trough is what actually
matters, and that is not recoverable from a monthly rate: request_21 needs two
changes even though one change's monthly saving looks like it should cover the
31.05 gap on its own.

Monthly saving is used only to ORDER equally-sized candidate sets, implementing
"prefer minimal sufficient changes, then the smallest total saving that still
passes" - the smallest change that does the job, not the biggest hammer available.
request_21 shows why that matters: event_1816 is reducible_or_stoppable and
streaming sits in both of that user's permission lists, and the expected answer
reduces it to 23.50 rather than stopping it outright.
"""

import dataclasses
import datetime as dt
import itertools
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

from entities import (
    Currency,
    Dataset,
    Direction,
    Event,
    Profile,
    Request,
    SpendingChange,
    SpendingChangeKind,
)
from netting import is_dead, to_home
from projection import Timeline, build_timeline
from recurrence import SpendingProfile

MAX_CHANGES = 3

# A group must repeat to count as a recurring expense. Two occurrences is enough -
# see the module docstring on event_989.
MIN_OCCURRENCES = 2


@dataclass(frozen=True, slots=True)
class Action:
    """One permitted change, with the balance improvement it buys across the window."""

    change: SpendingChange
    savings: tuple[Decimal, ...]
    monthly_saving: Decimal


def _changeable_events(events: tuple[Event, ...], as_of: dt.date) -> dict[str, Event]:
    """Latest debit occurrence, on or before as_of, of every repeating category.

    `events` arrives sorted by (event_date, event_id), so taking the last match
    per category resolves same-day ties deterministically on event_id.
    """
    counts: Counter[str] = Counter()
    latest: dict[str, Event] = {}

    for event in events:
        if is_dead(event) or event.direction is not Direction.DEBIT or event.amount is None:
            continue
        if event.event_date > as_of:
            continue
        counts[event.category] += 1
        current = latest.get(event.category)
        if current is None or event.event_date >= current.event_date:
            latest[event.category] = event

    return {
        event.event_id: event
        for category, event in latest.items()
        if counts[category] >= MIN_OCCURRENCES
    }


def permitted_changes(events: tuple[Event, ...], profile: Profile, as_of: dt.date) -> list[SpendingChange]:
    out: list[SpendingChange] = []
    for event in _changeable_events(events, as_of).values():
        if event.category in profile.categories_to_protect:
            continue
        if event.can_stop and event.category in profile.categories_willing_to_stop:
            out.append(SpendingChange(SpendingChangeKind.STOP, event.event_id, None))
        if (
            event.can_reduce
            and event.category in profile.categories_willing_to_reduce
            and event.minimum_allowed_amount is not None
            and event.minimum_allowed_amount < event.amount
        ):
            out.append(
                SpendingChange(
                    SpendingChangeKind.REDUCE_TO, event.event_id, event.minimum_allowed_amount
                )
            )
    return out


def _home_delta(data: Dataset, event: Event, change: SpendingChange, home: Currency) -> Decimal:
    """How much one occurrence of this event stops costing, in home currency."""
    on = event.settlement_date or event.event_date
    current = to_home(data, event.amount, event.currency, home, on)
    if change.kind is SpendingChangeKind.STOP:
        return current
    return current - to_home(data, change.new_amount, event.currency, home, on)


def _with_change(
    data: Dataset,
    spending: SpendingProfile,
    event: Event,
    change: SpendingChange,
    home: Currency,
) -> SpendingProfile | None:
    """Re-price the forecast component that carries this event, or None if nothing does."""
    delta = _home_delta(data, event, change, home)
    if delta <= 0:
        return None

    for index, series in enumerate(spending.series):
        if series.category != event.category or series.description != event.description:
            continue
        if change.kind is SpendingChangeKind.STOP:
            updated = spending.series[:index] + spending.series[index + 1 :]
        else:
            repriced = dataclasses.replace(
                series, amount_home=max(Decimal(0), series.amount_home - delta)
            )
            updated = spending.series[:index] + (repriced,) + spending.series[index + 1 :]
        return dataclasses.replace(spending, series=updated)

    for index, flow in enumerate(spending.variable):
        if flow.category != event.category:
            continue
        reduced = max(Decimal(0), flow.monthly_amount_home - delta)
        if reduced == flow.monthly_amount_home:
            return None
        repriced = dataclasses.replace(flow, monthly_amount_home=reduced)
        updated = spending.variable[:index] + (repriced,) + spending.variable[index + 1 :]
        return dataclasses.replace(spending, variable=updated)

    return None


def _apply_all(
    data: Dataset,
    request: Request,
    spending: SpendingProfile,
    by_id: dict[str, Event],
    changes: tuple[SpendingChange, ...],
    home: Currency,
) -> list[Decimal] | None:
    modified = spending
    for change in changes:
        result = _with_change(data, modified, by_id[change.event_id], change, home)
        if result is None:
            return None
        modified = result
    return build_timeline(data, request, modified).balances


def build_actions(
    data: Dataset,
    request: Request,
    profile: Profile,
    events: tuple[Event, ...],
    spending: SpendingProfile,
    baseline: Timeline,
) -> list[Action]:
    """Price every permitted change once, as a per-day balance improvement."""
    home = profile.home_currency
    by_id = {event.event_id: event for event in events}
    actions: list[Action] = []

    for change in permitted_changes(events, profile, request.request_date):
        event = by_id[change.event_id]
        modified = _with_change(data, spending, event, change, home)
        if modified is None:
            continue
        balances = build_timeline(data, request, modified).balances
        savings = tuple(new - old for new, old in zip(balances, baseline.balances))
        if savings[-1] <= 0:
            # Changing something the forecast never spends again saves nothing, and a
            # no-op change must not dilute the "fewest changes" preference.
            continue
        actions.append(
            Action(
                change=change,
                savings=savings,
                monthly_saving=_home_delta(data, event, change, home),
            )
        )
    return actions


def search(
    actions: list[Action],
    balances: list[Decimal],
    outflow: list[Decimal],
    minimum: Decimal,
    verify: Callable[[tuple[SpendingChange, ...]], bool],
) -> tuple[SpendingChange, ...] | None:
    """Smallest sufficient change set, or None if at most MAX_CHANGES cannot rescue the plan.

    Sets are tried fewest-first, and within a size cheapest-first by total monthly
    saving, so the first sufficient set found is the preferred one.

    Savings curves are summed to screen combinations, which is exact whenever the
    actions touch different forecast components but can overstate two actions that
    both draw down the same variable category (the rate floors at zero). The
    surviving set is therefore re-checked against a real rebuilt timeline via
    `verify` before it is accepted.
    """
    deficits = [minimum - (balance - paid) for balance, paid in zip(balances, outflow)]
    worst = max(range(len(deficits)), key=deficits.__getitem__)
    if deficits[worst] <= 0:
        return ()
    if not actions:
        return None

    ordered = sorted(actions, key=lambda action: action.monthly_saving)
    shortfalls = [(index, need) for index, need in enumerate(deficits) if need > 0]

    for size in range(1, MAX_CHANGES + 1):
        combos = [
            (sum((a.monthly_saving for a in combo), Decimal(0)), combo)
            for combo in itertools.combinations(ordered, size)
            # One action per event: stopping and reducing the same event are
            # mutually exclusive (problem_statement.md).
            if len({a.change.event_id for a in combo}) == size
        ]
        combos.sort(key=lambda pair: pair[0])

        for _, combo in combos:
            if sum(a.savings[worst] for a in combo) < deficits[worst]:
                continue
            if any(sum(a.savings[index] for a in combo) < need for index, need in shortfalls):
                continue
            changes = tuple(a.change for a in combo)
            if verify(changes):
                return changes

    return None


def verifier(
    data: Dataset,
    request: Request,
    profile: Profile,
    events: tuple[Event, ...],
    spending: SpendingProfile,
    outflow: list[Decimal],
    is_safe: Callable[[list[Decimal], list[Decimal], Decimal], bool],
) -> Callable[[tuple[SpendingChange, ...]], bool]:
    by_id = {event.event_id: event for event in events}

    def verify(changes: tuple[SpendingChange, ...]) -> bool:
        balances = _apply_all(
            data, request, spending, by_id, changes, profile.home_currency
        )
        if balances is None:
            return False
        return is_safe(balances, outflow, profile.minimum_balance_to_keep)

    return verify
