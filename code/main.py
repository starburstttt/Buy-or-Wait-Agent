"""Full pipeline entry point.

Run from repo root: python3 code/main.py

Loads dataset/, folds every user's message-derived facts into their events
(enrich/), solves every request in dataset/requests.csv, and writes the
repository-root output.csv (not dataset/output.csv - that file is only the
blank submission template).
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from entities import Decision  # noqa: E402
from enrich.llm import LLMClient  # noqa: E402
from loaders import DECISION_COLUMNS, REPO_ROOT, load_dataset  # noqa: E402
from solver import enrich_dataset, solve  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "output.csv"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def format_decision(decision: Decision) -> dict[str, str]:
    payment_plan = (
        "|".join(f"{p.due_date.isoformat()}:{p.amount}" for p in decision.payment_plan) or "none"
    )
    spending_changes = (
        "|".join(
            f"{c.kind}:{c.event_id}"
            if c.new_amount is None
            else f"{c.kind}:{c.event_id}:{c.new_amount}"
            for c in decision.spending_changes
        )
        or "none"
    )
    return {
        "request_id": decision.request_id,
        "amount_safe_to_pay": str(decision.amount_safe_to_pay),
        "affordability_status": decision.affordability_status.value,
        "recommended_payment_method": decision.recommended_payment_method.value,
        "payment_plan": payment_plan,
        "earliest_date_for_full_payment": (
            decision.earliest_date_for_full_payment.isoformat()
            if decision.earliest_date_for_full_payment is not None
            else ""
        ),
        "spending_changes_needed": spending_changes,
        "decision_explanation": decision.decision_explanation,
    }


def main() -> int:
    data = load_dataset()

    model = os.environ.get("MODEL", DEFAULT_MODEL)
    llm = LLMClient()
    data = enrich_dataset(data, llm, model=model)

    request_ids = sorted(data.requests, key=lambda rid: int(rid.split("_")[1]))
    rows = [format_decision(solve(data, data.requests[rid])) for rid in request_ids]

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DECISION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"wrote {len(rows)} rows to {OUTPUT_PATH}\n"
        f"llm_calls={llm.usage.total_calls()} llm_tokens={llm.usage.total_tokens()} "
        f"est_cost_usd={llm.usage.total_cost_usd():.5f}"
    )
    for model_name, usage in llm.usage.by_model.items():
        print(
            f"  {model_name}: calls={usage.calls} input_tokens={usage.input_tokens} "
            f"output_tokens={usage.output_tokens}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
