"""Per-request decision solver.

Current scope: the forecast is real, the PLAN SELECTION IS NOT. Only full_payment
and wait are considered, so installments, partial_payment and spending changes are
structurally unreachable. That is deliberate - it isolates the forecast so
calibrate.py measures the timeline rather than the ranking rules on top of it.
"""

from decimal import Decimal

from entities import (
    AffordabilityStatus,
    Dataset,
    Decision,
    Payment,
    PaymentMethod,
    RecommendedMethod,
    Request,
)
from projection import amount_safe_to_pay, build_timeline, earliest_full_payment_date
from recurrence import build_spending_profile


def solve(data: Dataset, request: Request) -> Decision:
    profile = data.profiles[request.user_id]
    events = data.events_by_user[request.user_id]

    spending = build_spending_profile(data, events, profile.home_currency, request.request_date)
    timeline = build_timeline(data, request, spending)

    safe_today = amount_safe_to_pay(
        timeline, profile.minimum_balance_to_keep, request.requested_amount
    )
    earliest = earliest_full_payment_date(
        timeline, profile.minimum_balance_to_keep, request.requested_amount
    )

    accepts_full = profile.accepts(PaymentMethod.FULL_PAYMENT)
    status = AffordabilityStatus.NOT_AFFORDABLE
    method = RecommendedMethod.NOT_RECOMMENDED
    plan: tuple[Payment, ...] = ()

    if earliest is not None and accepts_full:
        if earliest == request.request_date:
            status = AffordabilityStatus.AFFORDABLE_NOW
            method = RecommendedMethod.FULL_PAYMENT
            plan = (Payment(request.request_date, request.requested_amount),)
        elif earliest <= request.desired_completion_date:
            status = AffordabilityStatus.AFFORDABLE_LATER
            method = RecommendedMethod.WAIT
            plan = (Payment(earliest, request.requested_amount),)

    return Decision(
        request_id=request.request_id,
        amount_safe_to_pay=safe_today,
        affordability_status=status,
        recommended_payment_method=method,
        payment_plan=plan,
        earliest_date_for_full_payment=earliest,
        spending_changes=(),
        decision_explanation="",
    )
