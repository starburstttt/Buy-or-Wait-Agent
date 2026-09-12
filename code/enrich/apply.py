"""Turn validated MessageFacts into event-set mutations. Nothing else may.

This is the ONLY place a message is allowed to change what the solver sees. By the
time a MessageFact reaches here it has already survived two independent checks: the
model's raw output had to parse as JSON (enrich/llm.py), and every field had to pass
closed-schema validation - real event id, owned by this user, a real currency, a
real category, a parseable decimal/date (enrich/messages.py:validate_fact). What's
left is narrow, typed data. apply_facts() below whitelists exactly what each fact
kind is allowed to do to the event set; there is no path from a fact's `note` field,
or from the message text itself, into any of these mutations.

CONFLICT RULE (AGENTS.md 6.3): "Resolve conflicts using an explicit cancellation,
settlement, or amendment first; then newer records from the same source; then a
settled event; then the financially safer interpretation."

This is applied as: facts are folded in ascending message.sent_at order, so a later
message's amend/cancel of the SAME event overwrites an earlier one ("newer records
from the same source" - here, the same target). But an amend/cancel is dropped
outright, regardless of timing, if the target event is already SETTLED: a message is
never allowed to rewrite realized cash ("then a settled event" outranks an
unsettled message-derived claim). Every drop is logged with a reason instead of
silently disappearing.

CANCEL SCOPE
------------
cancel requires a target_event_id (validate_fact already enforces this). There is no
mechanism here for "cancel this category's recurring series in the abstract" -
every "contract ended" / "no renewal confirmed" message actually observed in this
dataset describes a series with NO future scheduled row in financial_events.csv,
which recurrence.py's own staleness/fallback rules already treat as zero future
income without any help from the message layer. Inventing a category-wide
suppression mechanism for a case that doesn't occur would be adding machinery the
data never exercises - if a future dataset variant needs it, extend here, not by
smuggling scope into the schema.
"""

import dataclasses
import datetime as dt
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entities import Dataset, Event, EventStatus, Message  # noqa: E402

from .messages import MessageExtraction, MessageFact

ApplyOutcome = Literal["applied", "dropped"]


def _inherit_template(events: tuple[Event, ...], category: str) -> Event | None:
    candidates = [e for e in events if e.category == category]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.event_date)


def apply_facts(
    events: tuple[Event, ...], facts: list[tuple[Message, MessageFact]]
) -> tuple[tuple[Event, ...], tuple[str, ...]]:
    """Fold a user's message-derived facts into their event tuple.

    `facts` must all belong to the SAME user as `events` (apply_facts does not
    check user_id itself - validate_fact already rejected any fact whose
    target_event_id pointed at a different user, and target_category facts carry no
    user at all, so the caller is expected to have already scoped both lists to one
    user, matching how netting.py/recurrence.py/projection.py all operate per-user).

    Returns the new event tuple (original events, amended/cancelled ones replaced,
    message-derived synthetic ones appended) plus a human-readable log of what
    happened to every fact - applied or dropped, and why. The log is meant for
    decision_explanation / debugging, never for re-parsing.
    """
    by_id: dict[str, Event] = {e.event_id: e for e in events}
    overlay: dict[str, Event] = {}
    synthetic: dict[tuple[str, dt.date], Event] = {}
    log: list[str] = []

    ordered = sorted(facts, key=lambda pair: pair[0].sent_at)

    for message, fact in ordered:
        tag = f"[{message.message_id}]"

        if fact.kind == "no_fact":
            log.append(f"{tag} no_fact: {fact.note}".rstrip())
            continue

        if fact.kind == "confirm":
            log.append(f"{tag} confirm: {fact.note}".rstrip())
            continue

        if fact.kind == "cancel":
            assert fact.target_event_id is not None  # validate_fact enforces this
            current = overlay.get(fact.target_event_id, by_id.get(fact.target_event_id))
            if current is None:
                log.append(f"{tag} cancel dropped: {fact.target_event_id} not found for this user")
                continue
            if current.status is EventStatus.SETTLED:
                log.append(
                    f"{tag} cancel dropped: {fact.target_event_id} is already settled "
                    "(a message cannot override realized cash)"
                )
                continue
            overlay[fact.target_event_id] = dataclasses.replace(current, status=EventStatus.CANCELLED)
            log.append(f"{tag} cancel applied: {fact.target_event_id} -> cancelled")
            continue

        # fact.kind == "amend"
        if fact.target_event_id is not None:
            current = overlay.get(fact.target_event_id, by_id.get(fact.target_event_id))
            if current is None:
                log.append(f"{tag} amend dropped: {fact.target_event_id} not found for this user")
                continue
            if current.status is EventStatus.SETTLED:
                log.append(
                    f"{tag} amend dropped: {fact.target_event_id} is already settled "
                    "(a message cannot override realized cash)"
                )
                continue
            changes: dict[str, object] = {}
            if fact.new_amount is not None:
                changes["amount"] = fact.new_amount
            if fact.new_currency is not None:
                changes["currency"] = fact.new_currency
            if fact.effective_date is not None:
                changes["event_date"] = fact.effective_date
                changes["settlement_date"] = fact.effective_date
            if not changes:
                log.append(f"{tag} amend dropped: {fact.target_event_id} - no new amount or date stated")
                continue
            overlay[fact.target_event_id] = dataclasses.replace(current, **changes)
            log.append(
                f"{tag} amend applied: {fact.target_event_id} -> "
                + ", ".join(f"{k}={v}" for k, v in changes.items())
            )
            continue

        # target_category only: synthesize a future occurrence.
        assert fact.target_category is not None  # validate_fact enforces one of the two
        if fact.effective_date is None:
            log.append(
                f"{tag} amend dropped: category={fact.target_category} has no effective_date "
                "to schedule it on"
            )
            continue
        template = _inherit_template(events, fact.target_category)
        if template is None:
            log.append(
                f"{tag} amend dropped: category={fact.target_category} has no prior event to "
                "model a new one on (event_type/direction/flexibility cannot be inferred safely)"
            )
            continue
        if fact.new_amount is None and template.amount is None:
            log.append(
                f"{tag} amend dropped: category={fact.target_category} states no new_amount and "
                "the template event has none to inherit"
            )
            continue
        # A date-only fact (new_amount unset) inherits the template's amount rather
        # than being dropped: e.g. user_07's payroll settles ~8 days after the
        # naive monthly date every cycle, and a message can confirm just the next
        # settlement date without restating an unchanged amount.
        new_amount = fact.new_amount if fact.new_amount is not None else template.amount
        new_currency = fact.new_currency or template.currency
        synthetic_id = f"msg_{message.message_id}"
        new_event = Event(
            event_id=synthetic_id,
            user_id=template.user_id,
            event_type=template.event_type,
            description=f"{template.description} (updated per {message.message_id})",
            category=fact.target_category,
            direction=template.direction,
            amount=new_amount,
            currency=new_currency,
            event_date=fact.effective_date,
            settlement_date=fact.effective_date,
            status=EventStatus.SCHEDULED,
            linked_event_id=None,
            flexibility=template.flexibility,
            minimum_allowed_amount=None,
        )
        synthetic[(fact.target_category, fact.effective_date)] = new_event
        log.append(
            f"{tag} amend applied (synthetic {synthetic_id}): category={fact.target_category} "
            f"amount={new_amount} {new_currency} effective={fact.effective_date}"
        )

    merged = tuple(overlay.get(e.event_id, e) for e in events) + tuple(synthetic.values())
    merged = tuple(sorted(merged, key=lambda e: (e.event_date, e.event_id)))
    return merged, tuple(log)


def enrich_user_events(
    data: Dataset,
    user_id: str,
    extractions: list[MessageExtraction],
    messages_by_id: dict[str, Message],
) -> tuple[tuple[Event, ...], tuple[str, ...]]:
    """Convenience wrapper: extractions -> (message, fact) pairs -> apply_facts().

    Extractions with no fact (no_fact/confirm carry a fact with that kind and DO
    reach apply_facts for logging; unparseable/schema_invalid/budget_exceeded carry
    fact=None and are skipped here, not silently - messages.py's own per-message
    status already records why).
    """
    events = data.events_by_user.get(user_id, ())
    pairs = [
        (messages_by_id[extraction.message_id], extraction.fact)
        for extraction in extractions
        if extraction.fact is not None
    ]
    return apply_facts(events, pairs)
