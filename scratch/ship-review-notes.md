# Ship review response

Jack Heart (verbatim):

> seems fine ship it 2a and start it on auto mode 3a

## Reading of decisions

- **Ship MTG-501 (PR #636):** advance to `deliver` (`reflect` → `land --wait-and-fix`, which will also refresh the stale `test_checked_authored_match_parity_receipt` caused by stripping `SpendingObject` / `SpendingContext::Activate` from `decision_memory::snapshot` in `68c414a1b6`).
- **2a (start on auto mode):** file one follow-up `0.3.0` Task to untangle the ~360 synthetic `execution/*` rows and 44 phantom `DecisionMemory::*` tokens from `record_vocabulary()` (and review `catalog_tokens()` naming alongside it), enable `auto`, and start its Graph run.
- **3a:** once PR #636 merges, comment on `loopflowstudio/manabot#629` noting `8904bf12a5` + PR #636 for `0.2.7` and close `#629`.
