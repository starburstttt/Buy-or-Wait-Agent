"""Event-lifecycle netting: decide which financial_events rows represent real cash.

Two different consumers need two different views of history:

  clean_history()      - settled rows BEFORE request_date, used to infer recurrence
                         and variable-spend rates. Must exclude anything that never
                         actually moved money, or the projected rate is inflated.

  future_cash_events() - rows settling ON OR AFTER request_date, used to build the
                         forecast timeline. current_available_balance is stated as of
                         request_date, so everything settled before it is already
                         inside that number and must NOT be applied again.
"""

import datetime as dt
from decimal import Decimal

from entities import Currency, Dataset, Direction, Event, EventStatus, EventType

DEAD_STATUSES = frozenset({EventStatus.CANCELLED, EventStatus.FAILED, EventStatus.UNREALIZED})

# Credits that are not cash until they settle: bonuses, commissions, prize money,
# refunds in flight. Confirmed salary is the documented exception - it counts on
# its settlement date even while still 'scheduled'.
SALARY_CATEGORY = "salary"


def is_dead(event: Event) -> bool:
    return event.status in DEAD_STATUSES or event.direction is Direction.NON_CASH


def reversed_pairs(events: tuple[Event, ...]) -> set[str]:
    """Event ids neutralised by a linked reversal (refund credit, or re-settled auth).

    Both sides of the pair are returned, so neither inflates a projected rate.
    """
    by_id = {e.event_id: e for e in events}
    neutralised: set[str] = set()
    for event in events:
        target_id = event.linked_event_id
        if target_id is None:
            continue
        target = by_id.get(target_id)
        if target is None:
            continue
        if event.event_type is EventType.REFUND and event.direction is Direction.CREDIT:
            if target.amount is not None and event.amount == target.amount:
                neutralised.add(event.event_id)
                neutralised.add(target.event_id)
        # A cancelled authorisation superseded by a settled charge: the cancelled
        # row is already dropped by is_dead(); the settled row is the real cash.
    return neutralised


def counts_as_cash(event: Event) -> bool:
    if is_dead(event):
        return False
    if event.direction is Direction.DEBIT:
        return event.status in (EventStatus.SETTLED, EventStatus.PENDING, EventStatus.SCHEDULED)
    # Credits: settled always; scheduled only for confirmed salary; pending never.
    if event.status is EventStatus.SETTLED:
        return True
    if event.status is EventStatus.SCHEDULED:
        return event.category == SALARY_CATEGORY
    return False


def clean_history(events: tuple[Event, ...], as_of: dt.date) -> list[Event]:
    neutralised = reversed_pairs(events)
    return [
        e
        for e in events
        if not is_dead(e)
        and e.event_id not in neutralised
        and e.settlement_date is not None
        and e.settlement_date < as_of
        and e.status is EventStatus.SETTLED
    ]


def future_cash_events(events: tuple[Event, ...], as_of: dt.date) -> list[Event]:
    neutralised = reversed_pairs(events)
    return [
        e
        for e in events
        if counts_as_cash(e)
        and e.event_id not in neutralised
        and e.settlement_date is not None
        and e.settlement_date >= as_of
    ]


def to_home(
    data: Dataset, amount: Decimal, currency: Currency, home: Currency, on: dt.date
) -> Decimal:
    if currency == home:
        return amount
    return amount * data.rate(on, currency, home)


def signed_home_amount(data: Dataset, event: Event, home: Currency) -> Decimal | None:
    """Event amount in home currency, positive for credit and negative for debit.

    Returns None for the 16 blank-amount rows whose value lives in an image; those
    are recoverable only once the vision step exists.
    """
    if event.amount is None:
        return None
    on = event.settlement_date or event.event_date
    value = to_home(data, event.amount, event.currency, home, on)
    return value if event.direction is Direction.CREDIT else -value
