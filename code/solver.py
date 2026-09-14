"""Per-request decision solver.

Pipeline per request: reconstruct spending -> forecast 90 days -> enumerate
payment structures -> rescue unsafe ones with permitted spending changes -> rank.
The forecast (projection.py, recurrence.py, netting.py) answers what the user can
afford; plan/ answers what they should do about it.

CAPACITY IS REPORTED CHANGE-FREE
--------------------------------
amount_safe_to_pay and earliest_date_for_full_payment are computed from the
untouched forecast and are never affected by the spending changes in the chosen
plan. That is the spec's wording ("before optional spending changes" / "without
optional spending changes") and the samples confirm the two can disagree:
request_06 reports 603.30 safe while the recommendation pays 620.40 in full by
stopping event_476.
"""

import dataclasses

from entities import (
    AffordabilityStatus,
    Dataset,
    Decision,
    RecommendedMethod,
    Request,
)
from enrich.apply import enrich_user_events
from enrich.images import apply_amounts, resolve_all
from enrich.llm import LLMClient
from enrich.messages import extract_many
from explain import explain
from plan.candidates import generate, is_safe, outflow_curve
from plan.changes import build_actions, search, verifier
from plan.ranking import Plan, best
from projection import amount_safe_to_pay, build_timeline, earliest_full_payment_date
from recurrence import build_spending_profile

DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"

# No solved sample pairs a not_affordable row with a non-empty
# earliest_date_for_full_payment, which argues for blanking the field there. Held
# at False anyway because calibration disagrees: blanking scores 60% on that field
# against 64% for reporting it. The reason is request_06 - ground truth calls it
# affordable_with_plan and reports 2026-01-15, and while the forecast is still
# pessimistic enough to call that row not_affordable, reporting the computed date
# keeps the field right even when the status is wrong. Worth re-testing once the
# forecast stops under-reporting amount_safe_to_pay.
BLANK_EARLIEST_WHEN_NOT_AFFORDABLE = False


def _reindex(data: Dataset, events_by_user: dict[str, tuple]) -> Dataset:
    """Rebuild both event views from one authority so they cannot drift apart.

    events_by_user is what solve() reads through recurrence/projection/netting;
    data.events is the by-id lookup that enrich/ and explain/ use. Replacing only
    the first leaves the second holding pre-enrichment rows - including the blank
    amounts the image pass exists to fill - so both are rebuilt together, and
    events synthesized by apply.py land in the index too.
    """
    events = {event.event_id: event for evs in events_by_user.values() for event in evs}
    return dataclasses.replace(data, events_by_user=events_by_user, events=events)


def enrich_dataset(
    data: Dataset, llm: LLMClient, *, model: str, vision_model: str = DEFAULT_VISION_MODEL
) -> Dataset:
    """Fold message-derived facts and image-recovered amounts into the event set.

    Both passes are cached on content (enrich/cache.py), so calling this once per
    pipeline run costs API tokens only for items never seen before; a rerun over an
    unchanged dataset is all cache hits.

    ORDER IS LOAD-BEARING. Messages run first, for two independent reasons:

    1. Cache validity. A message prompt embeds a snapshot of its related event, and
       three of the sixteen blank-amount events are referenced by a message. Filling
       those amounts first would change the prompt text, change the content hash, and
       silently re-bill all 215 message extractions.
    2. Precedence. problem_statement.md resolves conflicts with an explicit amendment
       first. A message that states an amount IS such an amendment; the image is the
       original document behind a field the CSV left blank. So apply_amounts only
       fills events whose amount is still None, and a message-set amount wins.
    """
    events_by_user = dict(data.events_by_user)
    for user_id, messages in data.messages.items():
        extractions = extract_many(data, list(messages), llm, model=model)
        messages_by_id = {m.message_id: m for m in messages}
        merged, _log = enrich_user_events(data, user_id, extractions, messages_by_id)
        events_by_user[user_id] = merged
    data = _reindex(data, events_by_user)

    # resolve_all raises rather than returning a partial result: a blank amount that
    # silently stays blank is the "treat it as zero" the spec forbids, and it biases
    # every affected forecast optimistic because the blanks are nearly all expenses.
    amounts = resolve_all(data, llm, model=vision_model)
    filled = {
        user_id: apply_amounts(events, amounts) for user_id, events in data.events_by_user.items()
    }
    return _reindex(data, filled)


def _status_for(plan: Plan) -> AffordabilityStatus:
    """affordability_status follows from the chosen structure.

    full_payment is affordable_now only when it needs no spending changes - with
    changes the full amount was NOT safe on request_date, which is exactly the
    affordable_with_plan case ("completed through ... permitted spending changes").
    request_12 pins the other direction: full payment is financially safe on
    request_date but the user does not accept full_payment, so the installment plan
    it falls back to reports affordable_with_plan rather than affordable_now.
    """
    match plan.candidate.method:
        case RecommendedMethod.FULL_PAYMENT:
            return (
                AffordabilityStatus.AFFORDABLE_WITH_PLAN
                if plan.changes
                else AffordabilityStatus.AFFORDABLE_NOW
            )
        case RecommendedMethod.WAIT:
            return AffordabilityStatus.AFFORDABLE_LATER
        case _:
            return AffordabilityStatus.AFFORDABLE_WITH_PLAN


def solve(data: Dataset, request: Request) -> Decision:
    profile = data.profiles[request.user_id]
    events = data.events_by_user[request.user_id]
    minimum = profile.minimum_balance_to_keep

    spending = build_spending_profile(data, events, profile.home_currency, request.request_date)
    timeline = build_timeline(data, request, spending)

    safe_today = amount_safe_to_pay(timeline, minimum, request.requested_amount)
    earliest = earliest_full_payment_date(timeline, minimum, request.requested_amount)

    span = len(timeline.balances)
    actions = None
    plans: list[Plan] = []

    for candidate in generate(data, request, profile, safe_today, earliest):
        outflow = outflow_curve(timeline.start, candidate.payments, span)
        if is_safe(timeline.balances, outflow, minimum):
            plans.append(Plan(candidate, ()))
            continue

        if actions is None:
            actions = build_actions(data, request, profile, events, spending, timeline)
        if not actions:
            continue
        changes = search(
            actions,
            timeline.balances,
            outflow,
            minimum,
            verifier(data, request, profile, events, spending, outflow, is_safe),
        )
        if changes:
            plans.append(Plan(candidate, changes))

    chosen = best(plans)

    if chosen is None:
        decision = Decision(
            request_id=request.request_id,
            amount_safe_to_pay=safe_today,
            affordability_status=AffordabilityStatus.NOT_AFFORDABLE,
            recommended_payment_method=RecommendedMethod.NOT_RECOMMENDED,
            payment_plan=(),
            earliest_date_for_full_payment=(
                None if BLANK_EARLIEST_WHEN_NOT_AFFORDABLE else earliest
            ),
            spending_changes=(),
            decision_explanation="",
        )
        return dataclasses.replace(
            decision, decision_explanation=explain(data, request, profile, decision)
        )

    decision = Decision(
        request_id=request.request_id,
        amount_safe_to_pay=safe_today,
        affordability_status=_status_for(chosen),
        recommended_payment_method=chosen.candidate.method,
        payment_plan=chosen.candidate.payments,
        earliest_date_for_full_payment=earliest,
        spending_changes=chosen.changes,
        decision_explanation="",
    )
    return dataclasses.replace(
        decision, decision_explanation=explain(data, request, profile, decision)
    )
