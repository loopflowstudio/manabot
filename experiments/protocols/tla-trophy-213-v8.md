# TLA trophy pilot with explicit first-64 legal support

Jack Heart selected 64 slots with logging and cutoff on 2026-11-04
(comment 3b0f8826). This supersedes unsubmitted v7 Run `af3661790ac5` at
`b5406bb33`; its immutable source and protocol remain evidence, not execution
authority for this successor. V8 starts fresh training and evaluation.

The observation configuration declares `max_actions=64` and
`action_overflow=first-legal`. Both the native buffered training encoder and
Python evaluation encoder retain the first 64 actions in native order after
legality/payability filtering. Fewer actions are unchanged. Retained row i is
native action i and maps to the original Command; padding is masked and one
categorical distribution covers the retained support. There is no page action.
The complete legal set remains at the native authority and in Academy.

This is a deliberate approximation: later legal actions have zero policy
probability at overflow decisions. Ordering can affect outcomes. Structured
diagnostics count consumed decisions and overflows by kind, dropped actions,
and retain up to four complete seed/deck/action prefixes with original,
retained and dropped counts. Console summaries occur at powers of two;
collector/game close retains exact aggregate counts, including zero overflow.
Evaluation game receipts include the same report. The fixed heuristic anchor
keeps its existing full native action semantics; learned candidates and their
initialization anchor use the same 64-slot configuration as training.

The selected-cost X repair and unchanged-state rejection remain required:
ordinary Wan Shi Tong with seven basics offers X0..5; airbend's fixed {2}
cost fixes X=0. A cutoff never legalizes an unavailable choice.

All other v7 choices remain: 134 frozen decks, input-v7 shared effects, 128
cards per seat, recipe, seeds, paired schedule, missing-game accounting,
<=USD1 per allocation and <=USD10 cumulative. Frozen v1–v7 evidence is unchanged.
The JSON protocol binds this action policy independently of tensor shape and
hashes the regime; checkpoint configuration carries the same explicit choice.
