# T2 — TLA casting, costs, zone permissions and private zones (Task 87e012)

T1 (892c47, PR #394) merged to main as `fa5e13b23`; T2 is PR #398, based on main.
Task directive lists the 51 cards and obligations; allocation in
`docs/rules/tla/allocation.jsonl` (`planned_pr: T2`). Every merge needs Jack
Heart's review. Status: implementation complete on the open world w17 (T2
extends it in place); concept review 4 changes implemented, gate pending.

## Decisions (reversible operator choices: questions.md)

- Cards are engine `CardDefinition`s in `cardsets/<set>.rs` (one
  representation, Jack Heart 2026-10-29); no authoring `Op` layer. T2's cards
  were regenerated from its prior lowering with T1's generator and proven
  identical.
- Activated abilities carry an effect sequence (`ActivatedAbilityDefinition.effects`)
  and a `zone` (battlefield, hand, graveyard); hand/graveyard abilities are
  `Action::ActivateCardAbility { card, ability_index }`.
- `Cost`-level features are definition fields: `flashback`, `foretell`,
  `flash_if`, `cost_reduction_per`; ability flags `exhaust`/`once_each_turn`,
  owned by `Permanent::{activation_used, record_activation}`.
- Restricted mana is keyed by `ManaUse` (spending subtype masks over Lesson,
  Shrine, Ally for spells and ability sources); `Mana::restricted_to` in
  definitions; narrowest usable class spends first.
- Casting outside hand: `flow/cast_permission.rs` owns
  `GameState.cast_permissions` — exact-`ObjectRef` exile grants and the
  `HandOnly` lock; `Game::zone_cast_cost` is the one non-hand cost authority.
- Cast-time announcements are one `CastChoices { kicked, x, free }`.
- Foretell: `Action::Foretell` exiles face down; the `foretold` grant is the
  face-down authority; viewers other than the owner see only an exile count.
- Control: a resolving permanent spell enters under its controller; each
  battlefield lists the permanents its player controls.
- Policy input v5 (default; v1–v4 widths unchanged): player 60, card 55,
  permanent 32; new terms start at the old validity column, validity last.
- World: T2 extends the open world w17 in place (Jack Heart, 2026-10-30: a
  world number names a deliberate freeze). w17's receipts and fixtures were
  replaced by T2's re-recorded ones under the w17 names; WORLDS.md's opening
  paragraph is the steer's text.
- Belief tracker (concept review 4, 1b): the foretold slot is public as a
  count. A foretell moves one unrevealed hand card into the slot (hand −1, no
  draw); a cast from the slot is a revealed pool exit that never was in hand.
  `BeliefState.transport(foretold_exits=, hidden_foretells=)`: the foretold
  card is removed from each hand row uniformly by count and rows the next space
  excludes are conditioned away; the real prior/posterior over slots is MTG-215.

## Done

All 51 T2 cards defined and supported (none in `tla::INCOMPLETE`; T1's
"Unreviewed until T2" entries removed), each with a Scryfall id and directed
tests in `tests/cards/tla.rs` (flat, card order). Observability: T1's guard
covers T2's 20 `Effect` variants, `AttacksAlone`, and T2's permanent/player
status fields; T2's observation-row guard and viewer-safety tests sit beside it
in `tests/cards/observability.rs`; Python text/encoder parity test for the new
permanent column. JSON protocol schemas regenerated. WORLDS.md w17 (open), one section.
Restricted-mana text names the class, not Lesson-only (`ee9e32383`).
Possible worlds (concept review 5): the unseen pool is Hand ∪ Library ∪ the
opponent's face-down foretold cards; the public `foretold` count rides on
`HiddenHandConstraints` and the space projection. A world's weight is
`Π C(n_i, k_i) · P(m, F)` (ordered fills of the F distinguishable slots from
the m foretell-bearing cards the hand leaves); zero-weight hands are dropped
and `Has`/`Lacks`/`Q` bounds come from the support. Materialization refills
the slots from the seeded remainder through `Game::fill_foretold_slots` (a
chosen card already foretold keeps its slot), which `Game::determinize`
shares. `POSSIBLE_WORLD_SPACE_VERSION` 3 in Rust and
Python (`managym.possible-world-space/v3`); Python `library_size` excludes
foretold cards.
Academy's card face prints and names the three T2 permanent statuses (used
exhaust, used once-each-turn, granted firebending) in text-5 wording
(`Card.svelte` via `PermanentRow.svelte`, `61d7e1d89`).

## Remaining

- Gate on `cfe477a7b` (debug `cargo test`, CI conformance, arena admission,
  Academy vitest/e2e); #398 CI was queued there at realign (main unmoved since
  `fa5e13b23`). Then Jack Heart reviews and lands #398; T3 (6875f0) syncs onto it
  and inherits the w17 rename.

## Check

Through 2026-10-30 (`601ed8f82`, then named w18): full gate passed (debug
`cargo test` all binaries, fmt/clippy, CI conformance + fuzz, arena admission
66, Academy/belief/model 320, vitest 109, Playwright and clean-machine play in
CI); headless demo of 60 T2-deck games, no errors, no foretell leak
(`scratch/t2-demo.md`). Detail: `git log -p -- scratch/87e012-tla-t2-casting.md`.

2026-10-30 implement (tracker slot + w17 rename): authored-match receipt
regenerates byte-identical under w17 (`--check`); authored-match and
public-commitment parity re-written (142 commands, 69 commitments, 0 gaps);
debug `semantic_conformance_tests` 6 pass on `semantic-kernel-w17-v1`; belief,
learn contract, Academy fixtures/parity 142 pass; arena admission 66 pass;
`npm run check` 0 errors. Full debug `cargo test`, vitest and e2e: gate.

2026-10-30 compress: tracker `_TransportFacts` derives `known_exits` from its
slot/hand split instead of storing it; `pytest tests/belief` 93 pass. Gate owns the rest.

## Concept review 4, with Jack Heart (2026-10-30)

Jack Heart accepted 1b: T2 fixes only the tracker's foretold-slot accounting
(done; test `test_foretell_moves_a_hidden_hand_card_to_the_slot_and_cast_leaves_it`);
the prior and posterior over foretold slots are MTG-215. Analysis and options:
`scratch/concept-review.html`.

## Concept review (headless, 2026-10-30, at `67077c5e9`)

Nothing to change and no open product choice. The accepted steers are all
reflected: one card representation with Scryfall ids, per-set card tests,
observability in both tensor and tokens (v5), exile permissions and the
hand-only lock visible to every viewer, restricted mana split by class, the
open world w17 with WORLDS.md's freeze paragraph, and the tracker counting the
foretold slot. The remaining foretell-unaware belief pieces (the hand prior,
`draw_unknown`, and the library row `pool − hand` in `belief/encoding.py`) need
a distribution over slot contents, so they are MTG-215 under Jack Heart's 1b,
not T2. Next: the gate.
