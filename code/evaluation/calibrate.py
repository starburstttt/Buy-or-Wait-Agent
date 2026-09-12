"""Calibration harness: score the solver against the 25 sample requests.

Run from code/: python evaluation/calibrate.py
Loads sample_requests.csv (input + reference output), runs the solver on each
of the 25 requests, and reports per-field exact-match accuracy plus a
tolerance-based check on amount_safe_to_pay. This is the instrument the
forecast/plan logic gets built against, not a test suite bolted on after.
"""

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entities import Decision, SpendingChange  # noqa: E402
from loaders import load_dataset  # noqa: E402
from solver import solve  # noqa: E402

AMOUNT_REL_TOLERANCE = Decimal("0.01")  # 1% of the expected amount
AMOUNT_ABS_TOLERANCE = Decimal("0.01")  # floor, so near-zero amounts aren't absurdly strict

EXACT_MATCH_FIELDS = [
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]


def _changes_set(changes: tuple[SpendingChange, ...]) -> frozenset[SpendingChange]:
    # Order isn't semantically meaningful for a set of spending changes; a
    # change is identified by (kind, event_id, new_amount).
    return frozenset(changes)


def _fmt(field: str, decision: Decision) -> str:
    if field == "payment_plan":
        return "|".join(f"{p.due_date}:{p.amount}" for p in decision.payment_plan) or "none"
    if field == "spending_changes_needed":
        parts = []
        for c in decision.spending_changes:
            if c.new_amount is None:
                parts.append(f"{c.kind}:{c.event_id}")
            else:
                parts.append(f"{c.kind}:{c.event_id}:{c.new_amount}")
        return "|".join(parts) or "none"
    value = getattr(decision, field)
    return str(value) if value is not None else ""


def field_match(field: str, expected: Decision, actual: Decision) -> bool:
    if field == "spending_changes_needed":
        return _changes_set(expected.spending_changes) == _changes_set(actual.spending_changes)
    if field == "payment_plan":
        return expected.payment_plan == actual.payment_plan
    return getattr(expected, field) == getattr(actual, field)


def amount_within_tolerance(expected: Decimal, actual: Decimal) -> bool:
    threshold = max(AMOUNT_ABS_TOLERANCE, AMOUNT_REL_TOLERANCE * abs(expected))
    return abs(actual - expected) <= threshold


def run() -> int:
    data = load_dataset()
    sample_ids = sorted(data.sample_requests, key=lambda rid: int(rid.split("_")[1]))

    misses: list[tuple[str, str, str, str]] = []
    field_correct = {field: 0 for field in EXACT_MATCH_FIELDS}
    amount_exact = 0
    amount_tolerant = 0
    fully_correct = 0

    for rid in sample_ids:
        request = data.sample_requests[rid]
        expected = data.sample_decisions[rid]
        actual = solve(data, request)
        if actual.request_id != rid:
            raise AssertionError(f"solver returned request_id {actual.request_id!r} for {rid!r}")

        row_ok = True

        for field in EXACT_MATCH_FIELDS:
            if field_match(field, expected, actual):
                field_correct[field] += 1
            else:
                row_ok = False
                misses.append((rid, field, _fmt(field, expected), _fmt(field, actual)))

        exact = expected.amount_safe_to_pay == actual.amount_safe_to_pay
        tolerant = amount_within_tolerance(expected.amount_safe_to_pay, actual.amount_safe_to_pay)
        amount_exact += exact
        amount_tolerant += tolerant
        if not exact:
            row_ok = False
            misses.append(
                (
                    rid,
                    "amount_safe_to_pay",
                    str(expected.amount_safe_to_pay),
                    str(actual.amount_safe_to_pay),
                )
            )

        fully_correct += row_ok

    n = len(sample_ids)
    _print_misses(misses)
    _print_totals(field_correct, amount_exact, amount_tolerant, fully_correct, n)
    return 0


def _print_misses(misses: list[tuple[str, str, str, str]]) -> None:
    w_rid, w_field, w_val = 12, 26, 30
    header = f"{'request_id':<{w_rid}} {'field':<{w_field}} {'expected':<{w_val}} {'actual':<{w_val}}"
    print("=" * len(header))
    print("MISSES (expected vs actual)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    if not misses:
        print("  (none)")
    for rid, field, expected, actual in misses:
        print(f"{rid:<{w_rid}} {field:<{w_field}} {expected:<{w_val}} {actual:<{w_val}}")
    print()


def _print_totals(
    field_correct: dict[str, int],
    amount_exact: int,
    amount_tolerant: int,
    fully_correct: int,
    n: int,
) -> None:
    header = f"{'field':<28} {'correct':>8} {'total':>8} {'accuracy':>10}"
    print("=" * len(header))
    print(f"TOTALS  (n={n} sample requests)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for field, count in field_correct.items():
        print(f"{field:<28} {count:>8} {n:>8} {count / n:>9.1%}")
    print(f"{'amount_safe_to_pay (exact)':<28} {amount_exact:>8} {n:>8} {amount_exact / n:>9.1%}")
    print(
        f"{'amount_safe_to_pay (tol 1%)':<28} {amount_tolerant:>8} {n:>8} "
        f"{amount_tolerant / n:>9.1%}"
    )
    print("-" * len(header))
    print(f"{'ALL FIELDS CORRECT':<28} {fully_correct:>8} {n:>8} {fully_correct / n:>9.1%}")


if __name__ == "__main__":
    raise SystemExit(run())
