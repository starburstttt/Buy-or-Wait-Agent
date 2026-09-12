# Buy or Wait, state at 9:30pm Sep 12

## FIRST THING TOMORROW
git merge worktree-message-enrichment
(the enrich/ layer lives on that branch, not in main yet)

## Then
1. Wire solver.py to the enrichment layer
2. Run message extraction on all 215 (cached, one-time cost)
3. Run calibrate.py
4. Build the plan layer — installments / partial_payment / affordable_with_plan

## Where the score is
8% all-fields, 48% status/method/plan, 16% amount within 1%
23 misses split: 9 need plan layer, 12 need messages wired, 5 real forecast error
Bias ruled out (10 over, 13 under). Horizon and aggregation constant ruled out too.

## Architecture
Python owns forecast, safety check, plan ranking. LLM only extracts facts
from 215 messages + 16 images, cached to disk, zero tokens on rerun.
apply.py is the only place a fact touches the event set.

## Bugs found and fixed
- gpt-oss-120b hidden reasoning tokens ate the completion budget before JSON
  → reasoning_effort="low" + more headroom
- Model invented target_event_id from "Ref payroll EMP-0001" in message text
  → validate_fact rejected it (event not owned by that user). This is the
  prompt-injection defense working. Use this in the interview.
- user_05 "Final employer payroll" was projected indefinitely, 21x overshoot
  → income now requires a scheduled future row

## Groq limits
8K tokens/min, 30 req/min, 200K tokens/day. Sleep 2.5s.