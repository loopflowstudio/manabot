# Five-card rules and admission demo — 2026-11-03

MTG-239, code Graph demo, decision pass 1. **Advance.** The repaired behavior
matches the accepted design; no concrete implementation gap was observed.
This is delegated headless judgment under the demo invocation, not new human
feedback, visual approval, a merge, or Task completion.

Current design: [MTG-239 plan](mtg-239-complete-five-defined.md).
Accepted interaction decisions: [concept review](concept-review.html).
Prior failure and repair: [CI recovery](ci-recovery.md).
Published change: [PR #415](https://github.com/loopflowstudio/manabot/pull/415),
head `5844fc16c8c221320a08722d476bce47ddf828b1`. Its runtime is identical to
`e5627dd9832e89386a2a1ee69e18d8167c242a7f`; the intervening diff contains
only review notes and Engine memory. This demo changes no runtime source.

## Experience observed

The walkthrough used the real Academy ASGI application at `/ws/play` through
FastAPI TestClient and the installed release native extension. It replayed
the existing LEARN navigation prefix with ordinary revision-bound Commands;
it did not inject a pending ordering decision. Temporary trace storage was
removed when the client closed. [Raw transcript](five-card-demo-live.txt).

1. Start the configured UR Lessons/GW Allies navigation game. After 16 accepted
   Commands, Accumulate Wisdom offers two legal remainder choices:
   `Put Firebending Lesson next on the bottom` and
   `Put Mountain next on the bottom`.
2. Read the native prompt: “Choose which remaining card to draw first. It goes
   above the other remaining cards on the bottom. Each choice commits
   immediately; the last card is placed automatically.” The title is
   `Order library bottom`.
3. Disconnect and resume the retained session. Both offers and the native state
   digest are unchanged. Select Mountain. The server accepts the Command and
   returns to Priority, with the last remaining placement automatic.
4. Submit the same Command again: `duplicate`, unchanged digest. Submit a new
   Command ID with the old revision: `rejected`, unchanged digest. Neither
   rewrites the committed choice.

Focused checks then exercised the following outcomes, with **14 Python tests
and 10 native debug tests passing**. Logs:
[Python](five-card-demo-python.txt), [Rust](five-card-demo-rust.txt).

| Action / counterexample | Observed outcome |
| --- | --- |
| Import whole lists containing all five repaired cards, in main and sideboard | Both admit through Decklist/admit_decklists and reset the native environment with the current content identity. |
| Import unknown cards, an Ally token, or a rules fixture | Line-specific unknown/unsupported rejection remains. |
| Validate training sources, checkpoint setups and replacement decks | Repaired cards admit; invalid replacements reject without changing existing roots. |
| Choose either Accumulate Wisdom remainder first | Both orders are offered and produce that draw order; private library cards remain absent from the opponent observation. Three Lessons take the alternate all-three path. |
| Read ordering through LLM text and policy inputs | Draw-order wording and continuation code 23 are present; native/Python v5 encoders agree. Command forks replay identically, isolate their roots and reject stale submissions. |
| Cast Lightning Strike at each any-target category | Creatures take marked damage, players lose life, planeswalkers lose loyalty and battles lose defense. Zero-counter shells go to the graveyard; ordinary lands/artifacts are not offered. Hexproof and multiple current types are respected. |
| Resolve Cat-Gator with two Swamps | Planeswalker/battle counters fall from five to three; the controller gains two life. Existing creature/player behavior also passes. |
| Inspect Suki across zones, change spell controller, lose abilities | Power follows owner off the stack/battlefield, controller on the stack and layered abilities on the battlefield. Hand text and both encoders show the contextual power. Returning to hand restores the CDA. |
| Activate Rallier with Suki in the library at power three versus four | Three is selectable and reaches hand; four is excluded. The inspection reports the same contextual power used by the filter. |

## Intended progress versus results

The preceding repair pass intended to replace obsolete negative admission
fixtures, preserve genuine rejection and repaired-card coverage, reconcile the
changed benchmark digests using the PR's own artifact, and publish a validated
head before demo. The original failure remains recorded in CI recovery.

Live GitHub inspection confirmed
[CI 41734880466](https://github.com/loopflowstudio/manabot/actions/runs/41734880466)
completed successfully for runtime commit `e5627dd98`: all 11 jobs plus the
aggregate passed, including full debug Rust, main integration's previously
skipped evaluation/report/smoke steps, Academy browser interactions and
clean-machine play. [Package run 41734880090](https://github.com/loopflowstudio/manabot/actions/runs/41734880090)
also passed. The fresh Engine Budget comparison has matching digests/counts
and passes the unchanged 10% timing tolerance; exact measurements are in
CI recovery. This establishes repair progress rather than merely repeating
the earlier green subset. The current demo supplies new server-path evidence.

The review-note head started
[CI 41736834519](https://github.com/loopflowstudio/manabot/actions/runs/41736834519)
and [package run 41736834476](https://github.com/loopflowstudio/manabot/actions/runs/41736834476).
They were still running at inspection, with no reported failure. Their success
is not inferred from the prior run. Delivery must check their final status.

## Judgment, feedback and limits

Jack Heart previously accepted removal of bug-based exclusion while preserving
unknown/token/fixture restrictions, and immediate ordering with clearer
draw-order wording and automatic final placement (comment `6b64d212`). Jack
also required the CI repairs before demo (comment `d35e6e94`). No person
participated in this headless walkthrough. The advance decision is the agent's
delegated interpretation: the observed results meet those explicit choices,
the repair closes the known failures, and no further design change is needed.

There are no unresolved product questions and no agreed new design changes.
No screenshots or local browser visual review were produced on this headless,
non-rendering surface; existing CI browser proof remains supporting evidence.
Reconnect covers a retained server session, not crash durability. Battle and
planeswalker damage uses controlled native shells, not newly shipped cards or
a complete battle combat/defeat system. MTG-213's frozen population and the
134-to-306 eligibility projection remain unchanged and unmeasured here.

Recommended next action: advance to the Graph's next delivery step, retaining
the final-head CI check and ordinary delivery requirements. No additional
implementation pass is justified by this evidence. No Graph was restarted,
Session closed, merge requested or long-lived server left running by this demo.

## Reproduction

The server walkthrough follows
`tests/academy/test_choice_navigation.py::test_accumulate_wisdom_order_survives_reconnect_and_commits_once`,
with the additional new-ID stale submission described above. The focused
Python invocation was:

```sh
uv run --extra dev --extra artifacts pytest \
  tests/academy/test_choice_navigation.py::test_accumulate_wisdom_order_survives_reconnect_and_commits_once \
  tests/semantic/test_decklist_import.py::test_five_completed_cards_admit_as_whole_decklists \
  tests/semantic/test_decklist_import.py::test_decklist_errors_carry_each_failing_seat_with_line_issues \
  tests/semantic/test_text_observation.py::test_accumulate_wisdom_ordering_reaches_text_commands_and_both_encoders \
  tests/semantic/test_text_observation.py::test_suki_hand_power_reaches_text_and_both_encoders \
  tests/training/test_training_decks.py -q
```

Native scenarios ran sequentially with debug assertions enabled:

```sh
for scenario in accumulate_wisdom lightning_strike suki_cda_power suki_loses_its_cda water_tribe_rallier_filters cat_gator; do
  CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -p managym-cards --test cards -j 2 -- "$scenario" || exit $?
done
```
