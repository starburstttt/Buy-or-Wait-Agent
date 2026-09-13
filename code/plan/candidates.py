"""Enumerate every payment structure available for one request.

A candidate is one concrete way to pay: which payments, on which dates, at what
total cost. Generation is deliberately separate from safety and from ranking -
this module answers "what could they do", changes.py answers "what would make an
unsafe structure safe", and ranking.py answers "which of the safe ones wins".

THREE FILTERS, IN THIS ORDER
----------------------------
eligibility - the structure is available at all: the user accepts the method
              (payment_methods_user_will_consider), the request permits it
              (allows_partial_payment), and an installment option fits inside
              max_installment_months.
deadline    - the LAST payment lands on or before desired_completion_date.
              problem_statement.md lists "complete the full request by
              desired_completion_date" as ranking criterion 1, but the 25 solved
              samples treat it as elimination, not ranking: no option whose
              schedule runs past the deadline is ever selected, and request_06,
              request_11 and request_21 all spend permitted spending changes
              rather than emit a `wait` that lands past the deadline (by 1, 33
              and 1 days respectively). Applied here so ranking.py only ever sees
              plans that already complete in time.
safety      - the 90-day balance never dips below minimum_balance_to_keep once
              the payments are applied. Checked by the caller, which may first
              apply spending changes from changes.py.

WHY EVERY CANDIDATE IS SAFETY-CHECKED THE SAME WAY
--------------------------------------------------
full_payment-today and wait could each be decided from the scalars
amount_safe_to_pay / earliest_full_payment_date alone, but they are run through
the same is_safe() as installments anyway. One code path means one definition of
"safe", and it keeps the degenerate cases honest - a baseline that already
breaches the minimum before any payment makes every candidate unsafe, which the
scalar shortcuts would not have noticed.
"""

import datetime as dt
import sys
from dataclasses import dataclass
from decimal import Decimal

from entities import (
    Dataset,
    Payment,
    PaymentMethod,
    Profile,
    RecommendedMethod,
    Request,
)

# Sort value for candidates that carry no payment_option_id. Criterion 6 is the
# final tie-breaker, so a candidate that cannot supply an id must never win it.
NO_OPTION_RANK = sys.maxsize


@dataclass(frozen=True, slots=True)
class Candidate:
    method: RecommendedMethod
    payments: tuple[Payment, ...]
    total_paid: Decimal
    option_id: str | None = None

    @property
    def first_payment_date(self) -> dt.date:
        return self.payments[0].due_date

    @property
    def last_payment_date(self) -> dt.date:
        return self.payments[-1].due_date

    @property
    def option_rank(self) -> int:
        if self.option_id is None:
            return NO_OPTION_RANK
        return int(self.option_id.rsplit("_", 1)[-1])


def outflow_curve(start: dt.date, payments: tuple[Payment, ...], span: int) -> list[Decimal]:
    """Cumulative amount paid out by each day offset in the forecast window.

    Payments landing beyond the horizon are dropped rather than clamped: the
    90-day window cannot say anything about a balance after it ends, so pretending
    the money leaves on the final day would invent a breach that the forecast has
    no evidence for. The deadline filter keeps this from mattering in practice -
    every desired_completion_date in this dataset sits inside the window.
    """
    deltas = [Decimal(0)] * span
    for payment in payments:
        offset = (payment.due_date - start).days
        if offset >= span:
            continue
        deltas[max(0, offset)] += payment.amount

    curve: list[Decimal] = []
    running = Decimal(0)
    for delta in deltas:
        running += delta
        curve.append(running)
    return curve


def is_safe(balances: list[Decimal], outflow: list[Decimal], minimum: Decimal) -> bool:
    return all(balance - paid >= minimum for balance, paid in zip(balances, outflow))


def generate(
    data: Dataset,
    request: Request,
    profile: Profile,
    safe_today: Decimal,
    earliest: dt.date | None,
) -> list[Candidate]:
    """Every eligible structure that completes the request by its deadline.

    safe_today and earliest are the change-free capacity scalars; partial_payment
    is built from them directly because problem_statement.md fixes its shape: pay
    amount_safe_to_pay on request_date, then the remainder on
    earliest_date_for_full_payment.
    """
    deadline = request.desired_completion_date
    requested = request.requested_amount
    out: list[Candidate] = []

    accepts_full = profile.accepts(PaymentMethod.FULL_PAYMENT)

    if accepts_full and request.request_date <= deadline:
        out.append(
            Candidate(
                method=RecommendedMethod.FULL_PAYMENT,
                payments=(Payment(request.request_date, requested),),
                total_paid=requested,
            )
        )

    if (
        profile.accepts(PaymentMethod.PARTIAL_PAYMENT)
        and request.allows_partial_payment
        and Decimal(0) < safe_today < requested
        and earliest is not None
        and earliest <= deadline
    ):
        out.append(
            Candidate(
                method=RecommendedMethod.PARTIAL_PAYMENT,
                payments=(
                    Payment(request.request_date, safe_today),
                    Payment(earliest, requested - safe_today),
                ),
                total_paid=requested,
            )
        )

    if profile.accepts(PaymentMethod.INSTALLMENTS):
        cap = profile.max_installment_months
        for option in data.payment_options.get(request.request_id, ()):
            if option.payment_method is not PaymentMethod.INSTALLMENTS:
                continue
            if cap is not None and option.number_of_payments > cap:
                continue
            schedule = option.schedule()
            if not schedule or schedule[-1].due_date > deadline:
                continue
            out.append(
                Candidate(
                    method=RecommendedMethod.INSTALLMENTS,
                    payments=schedule,
                    total_paid=option.total_payable_amount,
                    option_id=option.payment_option_id,
                )
            )

    if accepts_full and earliest is not None and request.request_date < earliest <= deadline:
        out.append(
            Candidate(
                method=RecommendedMethod.WAIT,
                payments=(Payment(earliest, requested),),
                total_paid=requested,
            )
        )

    return out
