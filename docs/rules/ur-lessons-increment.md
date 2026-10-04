# Gran-Gran, Proft's Eidetic Memory and Combustion Technique

Jack Heart requested these three additions in ETU-88 on 2026-09-29, then
accepted a new world version for their complete behavior. They are registered
for custom `MatchHypers` decklists. The authored 41-card UR Lessons and 40-card
GW Allies decks and their sideboards remain unchanged.

Jack's final candidate list has 40 cards: remove two It'll Quench Ya!, two
First-Time Flyer, one Pop Quiz and one Igneous Inspiration; add two Combustion
Technique, one Proft's Eidetic Memory, one Gran-Gran and one Accumulate Wisdom.
The initially proposed extra Otter-Penguin is omitted. Adoption and matchup
measurement remain ETU-87.

## Oracle and implementation

Exact text and characteristics were retrieved from Scryfall's named-card API
on 2026-09-29, before implementation:

- [Gran-Gran](https://scryfall.com/card/tla/54/gran-gran): {U}, legendary
  1/2 Human Peasant Ally. Becoming tapped draws, then requires a discard.
  Three Lessons in its controller's graveyard enable a generic cost reduction
  on that controller's noncreature spells.
- [Proft's Eidetic Memory](https://scryfall.com/card/mkm/67/profts-eidetic-memory):
  {1}{U}, legendary enchantment. Entering draws a card; its controller has no
  maximum hand size. Its own-turn combat trigger requires at least two draws,
  targets a controlled creature, and computes counters from draws at resolution.
- [Combustion Technique](https://scryfall.com/card/tla/128/combustion-technique):
  {1}{R}, instant Lesson. Damage counts the caster's graveyard Lessons at
  resolution, excluding the resolving spell. Its temporary replacement exiles
  the targeted incarnation instead of a battlefield-to-graveyard move,
  including a later sacrifice. Cleanup or leaving the battlefield ends it.

The definitions use the ordinary typed card/effect interpreter. Reusable
extensions cover resolution-time values, mandatory discard, conditional
spell-cost reduction, own-turn combat triggers, unlimited hand size, and
object-local death replacement. The legend rule collects all players' choices
before applying the simultaneous SBA batch. Cleanup discards before clearing
damage and temporary effects, and repeats after exceptional cleanup priority.

Registered rules text reaches hand, battlefield and outside-candidate
projections and the card hover preview. Discard and legend prompts use distinct
instructions and labels. The closed experience, replay, Study and advice
schemas accept these fields without rewriting historical fixtures.

## World and compatibility

This is **w4**, also exported as `managym.WORLD_VERSION` and recorded in native
content manifests. The tensor shape is **28 / 39 / 25 / 16** for player, card,
permanent and action types. The new permanent flag occupies index 23; validity
moves to 24. Decision kinds add Discard (11) and LegendRule (12), using SelectCard.
Match-state hashes are v3; semantic decision/observation schema is v6.

**Earlier checkpoints, including w3, are not comparable to w4.** Existing
win-rate reports—including the earlier 38% Search-64 measurement—are not w4
baselines. Runtime fingerprint registration rejects older world labels; the
local challenger and selected-match arena use the native current identity.
Frozen experiment IDs, receipts and historical evidence retain their identities.

The frozen authored compiled semantic packs are unchanged. These additions
are registered native content, not new admissions to those packs' semantic
program vocabulary. A semantic-program checkpoint requiring that vocabulary
must reject the additions until separately compiled and bound. The native
random/search paths execute their full rules. No checkpoint training, strength
comparison, performance budget or semantic-transfer claim follows from ETU-88.

## Executed evidence — 2026-09-29

- Debug directed tests cover each ability, both legendary card types,
  simultaneous legend/lethal SBAs, Learn retrieval of Combustion, resolution-time
  draw counting, expired and stale incarnation replacements, illegal targets,
  viewer visibility, and fork/rollback isolation.
- Eight complete native games, seeds 0–3 in both seat orders, use the exact
  revised 40-card candidate. Each recorded semantic Command is applied to an
  equivalent replay root and compared for full state hash, ordered events and
  observations after every transition. All reach terminal.
- Four Python `MatchHypers` games, seeds 0–1 in both seat orders, finish with
  ordinary observation encoding and no truncation. Native search evaluates
  both mandatory-discard and legend choices with complete rollouts and an
  unchanged authoritative root.
- Play adapter tests expose cast offers and registered text for all three cards,
  and distinguish discard from legend retention. Svelte checking passes;
  server-rendered preview and TypeScript protocol conformance tests pass.
  This is adapter/component proof, not an interactive browser acceptance claim.
- The selected arena's four-game block and current checkpoint-shape replay
  checks pass under w4. Corrupted roots/invalid commands return a failing
  replay receipt without executing dependent actions.

Focused reproduction commands (from the repository root unless noted):

```sh
cargo test --manifest-path managym/Cargo.toml --test rules_tests ur_lessons_increment
cargo test --manifest-path managym/Cargo.toml --test rules_tests cr_514
cargo test --manifest-path managym/Cargo.toml --lib observation_encoder
uv run pytest tests/etude/test_ur_lessons_increment.py tests/semantic/test_decision_contract.py tests/env/test_observation.py -q
uv run pytest tests/etude/test_experience_protocol.py tests/etude/test_study_protocol.py -q
uv run pytest tests/arena/test_match.py -k selected -q
uv run pytest tests/sim/test_train_challenger.py -q
# In frontend/:
npm run check
npm test -- src/lib/protocol-conformance.test.ts src/lib/components/HoverPreview.svelte.test.ts src/lib/prompt-instructions.test.ts
```

The cp312 extension was rebuilt from the repository root and extracted from
the wheel. Local Homebrew Node could not load `libllhttp.9.3.dylib`; frontend
checks used the already installed Node 24.12.0 instead.

## Gate review — 2026-09-29

Gate found and repaired missing Scryfall snapshots for all three cards, a
schema-rejection test that still treated v6 as invalid, and old seeded
benchmark/conformance expectations invalidated by mandatory cleanup.
Review also found that Suki's existing lowercase `legendary` supertype was
missed by the new legend rule. Matching is now case-insensitive, with a
directed duplicate-Suki regression; registered content bytes are unchanged.

The benchmark roots now use `interactive-midgame-48-w4-v1` and
`interactive-heavy-80-w4-v1`. With the same seeds and action counts, the former
has two root actions instead of six; the latter has 24 allocated permanent
slots and 505 committed events instead of 28 and 498. Historical fixture IDs
are rejected. These are fixture observations, not performance measurements.
The direct mutation rollback probes now reach settled priority first, since
normal Command application owns transition-queue journaling. The Command-based
deep rollback and cross-driver comparisons retain their original checks.

CI and the native conformance tests now use the separate
[w4 corpus](../../conformance/semantic-kernel-w4-v1/). Its four games finish
with 567 commands and reproduce exactly. Bounded fuzzing passes 32 cases and
5,078 commands. Both historical corpora, the authored decks and compiled IR
remain unchanged. This rebinds the scheduling conformance checks only; it does
not certify the wider Learn, checkpoint, live-play or strength evidence.

The additional cleanup regression proves that a creature dying when a
temporary toughness buff expires grants cleanup priority, and that cards
drawn during that priority round cause another mandatory-discard cleanup
before the next turn.

Validation selected the debug Rust suite because the changes affect shared
turn progression, plus the Python card/play, observation, decision, arena and
challenger suites; further checks cover world readers, evidence consumers,
semantic coverage and encoding parity. Rust formatting/Clippy, changed-file
Ruff lint/format, Svelte checking and seven frontend component/protocol tests
pass. The extension was rebuilt again after gate edits. Passing unaffected
tests were reused within this gate after focused repairs; no persistent test
cache was used.

The debug Rust run plus focused repair reruns cover 377 passing tests; the
final rules suite passes all 207, including the eight complete candidate-deck
games. The focused Python batch passes 149 tests. Additional world/evidence consumer
checks pass 45 tests, and the semantic coverage gate passes 41 tests plus the
coverage-gap artifact check. No local check failures remain after repairs.

Cross-platform integration, browser visual references and clean-machine launch
remain for CI. This headless run has no rendering environment, so it adds no
interactive browser acceptance or screenshot claim. No publication or landing
was performed.

CI then showed that the hand-size limit is reachable in the selected matchup:
the pinned GW Allies release scenario (seed 62) holds eight cards at cleanup on
turns 9, 11, 13 and 15, so DISCARD is a reachable prompt family rather than an
excluded one. That scenario was re-recorded (58 commands, turn-35 hero win) and
the release visual references moved to v5, freezing v4 with its matrix. The
checked Learn demo fixture was re-recorded for the w4 manifest along the same
13-command path to the same revision-29 Learn decision.
