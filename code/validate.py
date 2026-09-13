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
     request's user (a detected RecurringSeries whose flexibility permits the
     stop/reduce_to being applied).

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
    Flexibility,
    PaymentMethod,
    RecommendedMethod,
    SpendingChangeKind,
)
from loaders import DECISION_COLUMNS, REPO_ROOT, load_dataset  # noqa: E402
from recurrence import build_spending_profile  # noqa: E402

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


def _recurring_flexibility_by_event(data, user_id: str, as_of: dt.date) -> dict[str, Flexibility]:
    profile = data.profiles[user_id]
    events = data.events_by_user.get(user_id, ())
    spending = build_spending_profile(data, events, profile.home_currency, as_of)
    return {series.latest_event_id: series.flexibility for series in spending.series}


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

    recurring_cache: dict[str, dict[str, Flexibility]] = {}

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
            if request.user_id not in recurring_cache:
                recurring_cache[request.user_id] = _recurring_flexibility_by_event(
                    data, request.user_id, request.request_date
                )
            recurring = recurring_cache[request.user_id]
            for kind, event_id, _new_amount in changes:
                flexibility = recurring.get(event_id)
                if flexibility is None:
                    violations.append(
                        f"{rid}: spending change targets {event_id!r}, which is not a detected "
                        f"recurring expense for {request.user_id}"
                    )
                    continue
                can_stop = flexibility in (Flexibility.STOPPABLE, Flexibility.REDUCIBLE_OR_STOPPABLE)
                can_reduce = flexibility in (Flexibility.REDUCIBLE, Flexibility.REDUCIBLE_OR_STOPPABLE)
                if kind is SpendingChangeKind.STOP and not can_stop:
                    violations.append(f"{rid}: {event_id} is not stoppable (flexibility={flexibility})")
                if kind is SpendingChangeKind.REDUCE_TO and not can_reduce:
                    violations.append(f"{rid}: {event_id} is not reducible (flexibility={flexibility})")

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
