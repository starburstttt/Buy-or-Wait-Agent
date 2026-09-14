"""Render decision_explanation from the structured decision.

FIVE TEMPLATES, ONE PER recommended_payment_method
---------------------------------------------------
The 25 solved samples use one sentence pair per method, and this module reproduces
them literally rather than inventing prose. No model is involved: every figure in
the output is a field that has already been decided, so the explanation cannot
disagree with the row it explains.

  full_payment     Pay X today. This leaves at least M available over the next 90 days.
  installments     Use N installments of X, starting D. This leaves at least M available.
  partial_payment  Pay X today and the remaining Y on D. This completes the full
                   request and keeps the M minimum protected.
  wait             Pay X in full on D. Paying earlier would take the balance below
                   the M minimum.
  not_recommended  Do not make this payment by D. None of the available options keeps
                   the M minimum protected.

M IS QUOTED VERBATIM
--------------------
"leaves at least M available" / "the M minimum" is always the profile's
minimum_balance_to_keep, not the projected trough. The samples are unambiguous on
this: request_01 leaves a trough well above its 18,000 floor and still says "at
least ZAR 18,000". The sentence is a statement about the floor being respected, not
a forecast readout.

TWO VARIANTS THIS DELIBERATELY DOES NOT REPRODUCE
--------------------------------------------------
request_09 phrases full_payment as "This keeps the EUR 600 minimum available..."
and request_04 phrases wait as "Wait until D, then pay X in full. Paying sooner
would put the M minimum at risk." Each is a one-off against 2 and 5 siblings
respectively using the forms above, so the majority form is emitted for every row.
That costs exactly those two sample rows and keeps all 250 outputs consistent.

not_recommended HAS TWO FORMS, AND THE SPLIT IS NOT COSMETIC
--------------------------------------------------------------
Five samples say "None of the available options keeps the M minimum protected".
Two - request_14 and request_24 - instead say "Do not proceed with the R request.
Although X is available today, the full amount cannot be completed safely within
90 days." Both of those are users whose ONLY accepted method is partial_payment on
a request that allows it: a partial payment today was genuinely on the table and
something is safe to pay, but the remainder cannot be completed in time. The other
five all had some other structure available that failed on the minimum balance,
which is what "none of the available options" describes. See _cannot_complete().
"""

import datetime as dt
from decimal import ROUND_HALF_UP, Decimal

from entities import (
    Dataset,
    Decision,
    PaymentMethod,
    Profile,
    RecommendedMethod,
    Request,
    SpendingChange,
    SpendingChangeKind,
)

FORECAST_DAYS = 90


def _money(currency: object, amount: Decimal) -> str:
    """'ZAR 25,256' / 'EUR 620.40' - thousands separated, decimals only when needed.

    The samples drop the decimal part on whole amounts and always show exactly two
    places otherwise, including trailing zeros ('EUR 620.40').
    """
    value = Decimal(amount).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if value == value.to_integral_value():
        return f"{currency} {value.to_integral_value():,}"
    return f"{currency} {value:,.2f}"


def _long_date(day: dt.date) -> str:
    """'8 August 2025' - no leading zero on the day, full month name."""
    return f"{day.day} {day:%B} {day.year}"


def _join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def _changes_clause(data: Dataset, currency: object, changes: tuple[SpendingChange, ...]) -> str:
    """'Stop the online backup subscription and reduce the streaming subscription to USD 23.50'.

    Phrased from the event's own description, lower-cased, in the order the plan
    lists the changes - which is the order request_21's ground truth uses.
    """
    parts: list[str] = []
    for change in changes:
        event = data.events.get(change.event_id)
        name = event.description.lower() if event is not None else change.event_id
        if change.kind is SpendingChangeKind.STOP:
            parts.append(f"stop the {name}")
        else:
            parts.append(f"reduce the {name} to {_money(currency, change.new_amount)}")
    clause = _join(parts)
    return clause[0].upper() + clause[1:]


def _cannot_complete(request: Request, profile: Profile, decision: Decision) -> bool:
    """True when partial payment was the only structure ever on the table.

    Reproduces the request_14 / request_24 wording. All three conditions matter: the
    request has to permit a partial payment, the user has to accept it and nothing
    else (otherwise some other option was available and failed, which is the other
    sentence), and something has to actually be safe to pay today for "Although X is
    available today" to be true.
    """
    return (
        request.allows_partial_payment
        and profile.payment_methods == frozenset({PaymentMethod.PARTIAL_PAYMENT})
        and decision.amount_safe_to_pay > 0
    )


def explain(data: Dataset, request: Request, profile: Profile, decision: Decision) -> str:
    currency = profile.home_currency
    minimum = _money(currency, profile.minimum_balance_to_keep)
    payments = decision.payment_plan
    method = decision.recommended_payment_method

    if method is RecommendedMethod.NOT_RECOMMENDED:
        if _cannot_complete(request, profile, decision):
            return (
                f"Do not proceed with the {_money(currency, request.requested_amount)} "
                f"request. Although {_money(currency, decision.amount_safe_to_pay)} is "
                f"available today, the full amount cannot be completed safely within "
                f"{FORECAST_DAYS} days."
            )
        return (
            f"Do not make this payment by {_long_date(request.desired_completion_date)}. "
            f"None of the available options keeps the {minimum} minimum protected."
        )

    if method is RecommendedMethod.FULL_PAYMENT:
        core = f"pay {_money(currency, payments[0].amount)} today"
        if decision.spending_changes:
            clause = _changes_clause(data, currency, decision.spending_changes)
            return f"{clause}, then {core}. This leaves at least {minimum} available."
        return (
            f"{core[0].upper()}{core[1:]}. This leaves at least {minimum} available "
            f"over the next {FORECAST_DAYS} days."
        )

    if method is RecommendedMethod.INSTALLMENTS:
        core = (
            f"use {len(payments)} installments of {_money(currency, payments[0].amount)}, "
            f"starting {_long_date(payments[0].due_date)}"
        )
        sentence = f"This leaves at least {minimum} available."
    elif method is RecommendedMethod.PARTIAL_PAYMENT:
        core = (
            f"pay {_money(currency, payments[0].amount)} today and the remaining "
            f"{_money(currency, payments[1].amount)} on {_long_date(payments[1].due_date)}"
        )
        sentence = f"This completes the full request and keeps the {minimum} minimum protected."
    else:  # WAIT
        core = (
            f"pay {_money(currency, payments[0].amount)} in full on "
            f"{_long_date(payments[0].due_date)}"
        )
        sentence = (
            f"Paying earlier would take the balance below the {minimum} minimum."
        )

    if decision.spending_changes:
        clause = _changes_clause(data, currency, decision.spending_changes)
        return f"{clause}, then {core}. {sentence}"
    return f"{core[0].upper()}{core[1:]}. {sentence}"
