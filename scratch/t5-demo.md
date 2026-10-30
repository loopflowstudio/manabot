# T5 demo (headless, 2026-10-30)

## First pass at `edc1fabd8`: iterate

Joo Dee's ability was run through the built extension and the LLM
text-decision path (`capture_text_decision` / `submit_response`). Capture
failed at the Surveil obligation (`decision: 20`):
`parse_continuation_frames_json` rejected any `frame.decision > 13`, although
main already emitted 14–18 and this branch adds 19 (`DistributeCounters`) and
20 (`Surveil`). Every `DecisionInput` consumer was affected: the LLM runner and
session, `verify/llm_game.py`, and Academy's `study_protocol.py`. The tensor
path was unaffected. Debug `cargo test --test cards tla::` passed 307/307.
The decision to iterate was the agent's, made in Jack Heart's place.

## Fix

- `continuation::decision_code` is the only mapping from obligation to code,
  with `MAX_DECISION_CODE` (20) beside it. The validator uses that bound.
  `every_decision_code_is_within_the_validated_bound` builds every variant.
- The text renderer labels `decision` according to the frame kind
  (`_frame_decision`). Its obligation table now covers codes 14–20.
- Regression test:
  `test_joo_dee_surveil_and_copy_play_through_the_text_decision`.

## Second pass (rebuilt extension, text path)

- Joo Dee (`scratch/tools/demo_t5.py`): reaches "Stage: surveil", resolves,
  and the sacrifice choice follows. Sacrificing the original leaves the token
  with "copy of D1", and the turn line reads "This turn: sacrificed 1
  permanents."
- Ran and Shaw (`scratch/tools/demo_t5_more.py`): four were forced onto the
  battlefield, and the legend rule put three Dragons into the graveyard. Casting
  a fifth then made a token stated as "copy of D2, except isn't legendary".
  The forced copies made no tokens, so the trigger is cast-only.
- Phoenix Fleet Airship (same script): Joo Dee's sacrifice set the turn
  history. At the end step, a second Airship appears with "copy of D2".

Checks: debug `cargo test --lib agent::continuation` passed 5/5; `cargo clippy
--all-targets` was clean; `tests/semantic/test_text_observation.py` passed
37/37; continuation-input and trigger-order tests passed 20/20.
