# MTG-239 — five defined cards

Implementation plan, 2026-11-03. Jack Heart's Task directive is the accepted
scope; internal choices here are headless implementation decisions. One PR,
with publication and broader acceptance owned by later code Graph steps.

## Outcome and design

Make Accumulate Wisdom, Cat-Gator, Lightning Strike, Suki, Kyoshi Warrior and
Water Tribe Rallier faithful through the existing rules and deck admission.
The official September 25, 2026 [Comprehensive Rules](https://media.wizards.com/2026/downloads/MagicCompRules%2020260925.txt)
confirm owner-chosen library order (401.4), four any-target categories (115.4),
CDA evaluation in all zones (604.3), and loyalty/defense removal from damage
(120.3c/h). No new planeswalker/battle cards, battle combat system, T6 content,
training run or corpus revision is included.

- Give LookAndSelect an explicit remainder-order rule. Random-order cards
  retain it; Accumulate Wisdom suspends on an uncapped native bottom-order
  choice. Each SelectCard commits the next card, nearest the top first. Use
  existing Commands, card focus, private inspection and forced-decision skip.
  Append a continuation code and text stage; keep frozen widths/codes stable.
- Replace the misleading CreatureOrPlayer specification with AnyTarget across
  its existing callers (all printed any-target cards). Legal and resolution
  checks share layered types and hexproof. Damage marks creatures and removes
  loyalty/defense counters as applicable, with one damage event and lifelink.
  Zero-counter unsupported-shell fixtures exercise ordinary graveyard SBAs;
  no new catalog card is implied.
- Add a contextual card-power reader: battlefield uses the layered view;
  stack uses controller; other zones/outside use owner. Reuse the existing CDA
  evaluator. Route zone power filters and visible card rows through it.
  Ability loss removes the battlefield CDA; leaving restores the off-field CDA.
- Remove the five incomplete entries only with directed behavior and whole
  Decklist/admit_decklists coverage. Unknown and unsupported rows still fail.

## Delete — do not maintain

Completed; no remaining deletion targets:

- TargetSpec::CreatureOrPlayer and its old text/fast-path assumptions: replace
  with AnyTarget, preserving ordering, exact target identity and ward handling.
- Unconditional random remainder in LookAndSelect and the Accumulate Wisdom
  test that accepts either order: preserve random semantics on other cards.
- Printed-power fallbacks in contextual zone predicates/observations: preserve
  printed-only structural matching where no game context is involved.
- Five TLA INCOMPLETE entries after their directed checks pass.
- `Game::bottom_in_random_order`: its sole caller now owns shuffle and
  placement together, sharing placement with a forced singleton. Both
  LookAndSelect exits reuse `continue_resolution` for suspension/resumption.

## Implementation outcome — 2026-11-03

The deletion cuts are implemented. Directed native and Python tests cover
the five cards, whole lists in main and sideboard, actual ordering Commands,
private inspection, stale rejection, replay/fork identity, native/Python/text
observations, definition tokens and loyalty/defense fact rows. The rebuilt
cp312 extension was installed atomically. Academy reconnect preserves the
ordering decision and duplicate submission commits it only once.

State-based deaths group non-destruction graveyard moves before checking lethal
damage, so candidate admission and the commit path share one precedence rule.
Counter descriptions reuse canonical names except the two display spellings.
The ordering encoder proof reads the native observation: captured viewer JSON
omits continuation frames, which the proof must exercise.

Jack Heart’s November 3 steer keeps this focused: known bugs in previously
playable cards belong in bug Tasks, not new play restrictions. AGENTS.md records
that direction. No broader admission-policy redesign or new Task was made.

Current open-w17 evidence was regenerated through its producers. Rallier’s
correct CDA filter excludes Suki above power three at authored decision 90;
the resulting game has 135 Commands and 66 commitments instead of 142/69.
The old trace remains at `22e4aab71:conformance/authored-match-w17-v1/`. Three
navigation prefixes now explicitly order Accumulate Wisdom’s remainder; all
prior deliberate choices retain their IDs/labels and reach the intended
positions. Demo/trigger fixtures and current belief hashes bind the new content;
historical hashes, frozen receipts and MTG-213 corpus bytes are unchanged.

Check result: focused debug `cargo test --test cards <card/observability filter>` and continuation/admission filters passed; root release `maturin build` passed; focused `uv run --extra dev --extra play pytest` admission, text/native encoders, tokens, heuristic, navigation/reconnect (6), trigger-order (2), demo (3) and belief receipt (1) checks passed; kernel record and both parity producers passed (135 Commands, 816 viewer checks, 66 commitments, zero gaps); changed-Python Ruff, Cargo fmt and `git diff --check` passed. Final reduction checks: `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test --manifest-path managym/Cargo.toml -j 2 --test cards -- accumulate_wisdom rallier piandao guru_pathik seismic_sense ths:: cat_gator suki_cda suki_loses off_battlefield_cda library_order` (17 passed); `cargo test --test rules_tests -- cr_704_sba cr_704_4` with the same profile/manifest (11 passed); `uv run --extra dev --extra play pytest` for the two new text/encoder tests and choice-navigation module (10 passed); root release wheel rebuilt and atomically installed; authored parity producer refreshed only source hashes (135 Commands, 816 viewer checks, no divergence). Gate retains the broader checks below.

## Integration and remaining checks

Base 22e4aab71. PR #414 remains open at 1bb1219053 (inspected 2026-11-03);
its card-crate move and sparse counters are not delivered here. Implement
focused rule changes on this supplied checkout, without copying or rewriting
its architecture. Integration must map card paths and the two added counter
kinds onto its surviving owners if it lands first.

Implementation remaining: none. Gate owns full debug Cargo/clippy, affected
Python/model suites, world-admission/monitor checks in AGENTS.md/development.md
and frontend generation/check/build. The focused results above remain usable;
full acceptance and UI review are still pending. Demo owns the final real-path
walkthrough; the headless reconnect/shared-Command proof is already executable.
Publication belongs to the selected Graph, not this implementation step.
