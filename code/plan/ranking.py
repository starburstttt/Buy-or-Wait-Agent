"""Pick the winner among safe plans, strictly lexicographically.

problem_statement.md's order, and where each criterion is enforced:

  1. Complete the full request by desired_completion_date  -> candidates.py (filter)
  2. Require no spending changes                           -> key position 0
  3. Minimize the total amount paid                        -> key position 1
  4. Start payment earlier                                 -> key position 2
  5. Use fewer payments                                    -> key position 3
  6. Lowest payment_option_id                              -> key position 4

Criterion 1 is a filter rather than a key, so every plan reaching this module
already completes in time - see candidates.py for the sample evidence.

CRITERION 2 IS BINARY
---------------------
"Require no spending changes" separates change-free plans from plans that need
any change at all; it does not rank one change against two. Among plans that all
need changes, criterion 3 (total paid) decides, exactly as written. changes.py has
already minimised the change count within each individual plan, so the only thing
a count-based key could do here is reorder two DIFFERENT payment structures by
change count ahead of cost, which the spec's ordering does not ask for.

No sample exercises criterion 2 against 3-5: in all three change-using samples
(request_06, request_11, request_21) no change-free candidate completes by the
deadline at all. The literal reading is taken - a change-free plan beats a
with-change plan even when the with-change plan is cheaper and starts earlier,
which makes spending changes a genuine last resort.

CRITERION 3 INCLUDES FINANCING FEES
-----------------------------------
total_paid is the option's total_payable_amount, not the requested amount.
request_19 is the proof: payment_option_53 is an eligible, in-cap, by-deadline
2-payment installment plan, and the expected answer is partial_payment instead,
because 39,660 < 41,246.40. A fee-bearing installment plan therefore never
outranks full_payment/partial_payment/wait when those are eligible and safe -
installments win when the cheaper structures are ineligible or unsafe.
"""

from dataclasses import dataclass

from entities import SpendingChange

from .candidates import Candidate


@dataclass(frozen=True, slots=True)
class Plan:
    candidate: Candidate
    changes: tuple[SpendingChange, ...]


def rank_key(plan: Plan) -> tuple:
    candidate = plan.candidate
    return (
        bool(plan.changes),
        candidate.total_paid,
        candidate.first_payment_date,
        len(candidate.payments),
        candidate.option_rank,
    )


def best(plans: list[Plan]) -> Plan | None:
    return min(plans, key=rank_key) if plans else None
