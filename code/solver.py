"""Per-request decision solver.

Current scope: the forecast is real, the PLAN SELECTION IS NOT. Only full_payment
and wait are considered, so installments, partial_payment and spending changes are
structurally unreachable. That is deliberate - it isolates the forecast so
calibrate.py measures the timeline rather than the ranking rules on top of it.
"""

import dataclasses
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
from enrich.apply import enrich_user_events
from enrich.llm import LLMClient
from enrich.messages import extract_many
from projection import amount_safe_to_pay, build_timeline, earliest_full_payment_date
from recurrence import build_spending_profile


def enrich_dataset(data: Dataset, llm: LLMClient, *, model: str) -> Dataset:
    """Fold every user's message-derived facts into their events before solving.

    Extraction is cached on message content (enrich/cache.py), so calling this once
    per pipeline run costs API tokens only for messages never seen before; a rerun
    over an unchanged dataset is all cache hits. Only events_by_user is replaced -
    that is the only field solve() (via recurrence/projection/netting) reads events
    from.
    """
    events_by_user = dict(data.events_by_user)
    for user_id, messages in data.messages.items():
        extractions = extract_many(data, list(messages), llm, model=model)
        messages_by_id = {m.message_id: m for m in messages}
        merged, _log = enrich_user_events(data, user_id, extractions, messages_by_id)
        events_by_user[user_id] = merged
    return dataclasses.replace(data, events_by_user=events_by_user)


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
