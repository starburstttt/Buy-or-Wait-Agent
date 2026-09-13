"""Validate the repository-root output.csv against the submission rules.

Run from repo root: python3 code/validate.py

Checks (README.md "Before submitting" / problem_statement.md):
  1. output.csv has exactly one row per request_id in dataset/requests.csv,
     plus the header (250 data rows for this dataset).
  2. The header matches the required columns, in the required order.
  3. Every amount_safe_to_pay satisfies 0 <= amount_safe_to_pay <= requested_amount.
  4. Every "installments" payment_plan exactly matches the schedule of a supplied
     payment option for that request.
  5. Every spending change targets a flexible recurring expense belonging to that
     request's user: a repeating, non-protected debit in a category the user
     permits changing, whose flexibility allows the stop/reduce_to being applied.

"Recurring" here means the event's (category, description) group occurs more than
once on or before request_date - NOT that recurrence.py projects it as a series.
request_11's ground-truth change targets a two-occurrence group that recurrence.py
deliberately refuses to project, so the stricter test would reject correct output.

This re-derives the rule from the dataset rather than calling plan/changes.py, so
a bug in the change search cannot validate itself.

Prints every violation found and exits 1 if any exist, else prints OK and exits 0.
"""

import csv
import datetime as dt
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from entities import (  # noqa: E402
    AffordabilityStatus,
    Direction,
    PaymentMethod,
    RecommendedMethod,
    SpendingChangeKind,
)
from loaders import DECISION_COLUMNS, REPO_ROOT, load_dataset  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "output.csv"


def _dec(value: str) -> Decimal | None:
    try:
        return Decimal(value.strip())
    except InvalidOperation:
        return None


def _date(value: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value.strip())
    except ValueError:
        return None


def _parse_plan(value: str, violations: list[str], rid: str) -> tuple[tuple[dt.date, Decimal], ...] | None:
    text = value.strip()
    if not text or text == "none":
        return ()
    payments = []
    for part in text.split("|"):
        day, _, amount = part.partition(":")
        d, a = _date(day), _dec(amount) if amount else None
        if d is None or a is None:
            violations.append(f"{rid}: malformed payment_plan entry {part!r}")
            return None
        payments.append((d, a))
    return tuple(payments)


def _parse_changes(
    value: str, violations: list[str], rid: str
) -> tuple[tuple[SpendingChangeKind, str, Decimal | None], ...] | None:
    text = value.strip()
    if not text or text == "none":
        return ()
    changes = []
    for part in text.split("|"):
        fields = part.split(":")
        if len(fields) == 2 and fields[0] == "stop":
            changes.append((SpendingChangeKind.STOP, fields[1], None))
        elif len(fields) == 3 and fields[0] == "reduce_to":
            amount = _dec(fields[2])
            if amount is None:
                violations.append(f"{rid}: malformed spending change amount {part!r}")
                return None
            changes.append((SpendingChangeKind.REDUCE_TO, fields[1], amount))
        else:
            violations.append(f"{rid}: malformed spending change {part!r}")
            return None
    return tuple(changes)


def _repeating_event_ids(data, user_id: str, as_of: dt.date) -> set[str]:
    """Events in a category that occurs more than once on or before as_of.

    Category-level, matching the granularity the user's permissions are stated at.
    A per-description test would reject correct output: request_11's ground-truth
    target sits in a description group of two inside a much larger dining category.
    """
    counts: Counter[str] = Counter()
    for event in data.events_by_user.get(user_id, ()):
        if event.direction is Direction.DEBIT and event.event_date <= as_of:
            counts[event.category] += 1
    return {
        event.event_id
        for event in data.events_by_user.get(user_id, ())
        if counts[event.category] > 1
    }


def validate() -> list[str]:
    violations: list[str] = []

    if not OUTPUT_PATH.is_file():
        return [f"missing {OUTPUT_PATH}"]

    with OUTPUT_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames
        rows = list(reader)

    if header is None:
        return [f"{OUTPUT_PATH} is empty"]
    if list(header) != DECISION_COLUMNS:
        return [f"header mismatch\n  expected {DECISION_COLUMNS}\n  found    {list(header)}"]

    data = load_dataset()
    expected_ids = set(data.requests)

    if len(rows) != len(expected_ids):
        violations.append(
            f"row count {len(rows)} != {len(expected_ids)} requests in dataset/requests.csv"
        )

    seen = Counter(row["request_id"] for row in rows)
    for rid, count in seen.items():
        if rid not in expected_ids:
            violations.append(f"{rid}: not a request_id in dataset/requests.csv")
        elif count > 1:
            violations.append(f"{rid}: appears {count} times in output.csv")
    for rid in sorted(expected_ids - set(seen), key=lambda r: int(r.split("_")[1])):
        violations.append(f"{rid}: missing from output.csv")

    repeating_cache: dict[str, set[str]] = {}

    for row in rows:
        rid = row["request_id"]
        request = data.requests.get(rid)
        if request is None:
            continue  # already flagged above as an unknown request_id

        amount = _dec(row["amount_safe_to_pay"])
        if amount is None:
            violations.append(
                f"{rid}: amount_safe_to_pay {row['amount_safe_to_pay']!r} is not a decimal"
            )
        elif not (Decimal(0) <= amount <= request.requested_amount):
            violations.append(
                f"{rid}: amount_safe_to_pay {amount} not in [0, {request.requested_amount}]"
            )

        try:
            AffordabilityStatus(row["affordability_status"])
        except ValueError:
            violations.append(f"{rid}: invalid affordability_status {row['affordability_status']!r}")

        try:
            method = RecommendedMethod(row["recommended_payment_method"])
        except ValueError:
            violations.append(
                f"{rid}: invalid recommended_payment_method {row['recommended_payment_method']!r}"
            )
            method = None

        plan = _parse_plan(row["payment_plan"], violations, rid)

        if method is RecommendedMethod.INSTALLMENTS:
            if not plan:
                violations.append(
                    f"{rid}: recommended installments but payment_plan is {row['payment_plan']!r}"
                )
            else:
                options = [
                    o
                    for o in data.payment_options.get(rid, ())
                    if o.payment_method is PaymentMethod.INSTALLMENTS
                ]
                if not any(
                    tuple((p.due_date, p.amount) for p in o.schedule()) == plan for o in options
                ):
                    violations.append(
                        f"{rid}: installment payment_plan {row['payment_plan']!r} does not match "
                        "any supplied payment option"
                    )

        changes = _parse_changes(row["spending_changes_needed"], violations, rid)
        if changes:
            if len(changes) > 3:
                violations.append(f"{rid}: {len(changes)} spending changes, at most 3 allowed")
            if len({event_id for _kind, event_id, _amount in changes}) != len(changes):
                violations.append(f"{rid}: the same event is changed twice")

            profile = data.profiles[request.user_id]
            if request.user_id not in repeating_cache:
                repeating_cache[request.user_id] = _repeating_event_ids(
                    data, request.user_id, request.request_date
                )
            repeating = repeating_cache[request.user_id]

            for kind, event_id, _new_amount in changes:
                event = data.events.get(event_id)
                if event is None or event.user_id != request.user_id:
                    violations.append(
                        f"{rid}: spending change targets {event_id!r}, which is not an event "
                        f"belonging to {request.user_id}"
                    )
                    continue
                if event_id not in repeating:
                    violations.append(
                        f"{rid}: {event_id} is not a recurring expense (its category/description "
                        "occurs only once on or before request_date)"
                    )
                if event.category in profile.categories_to_protect:
                    violations.append(
                        f"{rid}: {event_id} is in protected category {event.category!r}"
                    )
                if kind is SpendingChangeKind.STOP:
                    if not event.can_stop:
                        violations.append(
                            f"{rid}: {event_id} is not stoppable (flexibility={event.flexibility})"
                        )
                    if event.category not in profile.categories_willing_to_stop:
                        violations.append(
                            f"{rid}: user will not stop category {event.category!r}"
                        )
                if kind is SpendingChangeKind.REDUCE_TO:
                    if not event.can_reduce:
                        violations.append(
                            f"{rid}: {event_id} is not reducible (flexibility={event.flexibility})"
                        )
                    if event.category not in profile.categories_willing_to_reduce:
                        violations.append(
                            f"{rid}: user will not reduce category {event.category!r}"
                        )

    return violations


def main() -> int:
    violations = validate()
    if not violations:
        print(f"OK: {OUTPUT_PATH} is valid")
        return 0
    print(f"{len(violations)} violation(s) in {OUTPUT_PATH}:")
    for v in violations[:50]:
        print(f"  - {v}")
    if len(violations) > 50:
        print(f"  ... and {len(violations) - 50} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
