# T2 — TLA casting, costs, zone permissions and private zones (Task 87e012)

Stacked on T1 (892c47, PR #394, synced at its head `01e8f9975` via `4d3b9b066`); T2 is PR #398.
Task directive lists the 51 cards and obligations; allocation in
`docs/rules/tla/allocation.jsonl` (`planned_pr: T2`). Every merge needs Jack
Heart's review; don't arm T2's merge while #394 is open. Status: implementation
complete on w18; gate pending.

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

## Done

All 51 T2 cards defined and supported (none in `tla::INCOMPLETE`; T1's
"Unreviewed until T2" entries removed), each with a Scryfall id and directed
tests in `tests/cards/tla.rs` (flat, card order). Observability: T1's guard
covers T2's 20 `Effect` variants, `AttacksAlone`, and T2's permanent/player
status fields; T2's observation-row guard and viewer-safety tests sit beside it
in `tests/cards/observability.rs`; Python text/encoder parity test for the new
permanent column. JSON protocol schemas regenerated. WORLDS.md w18.
Restricted-mana text names the class, not Lesson-only (`ee9e32383`).

## Remaining

- Gate: CLAUDE.md arena admission/monitoring suite; Academy vitest suite; world-bound receipts T1 already flagged.
- Land only after #394 lands; re-sync then.

## Check

2026-10-29 compress (after Academy statuses): nothing to reduce in `Card.svelte`/`PermanentRow.svelte` and its test; scratch 9518/12000 tokens; no checks rerun.

2026-10-29 implement (concept review 4): `Card.svelte` prints and names exhaust/once-each-turn/granted-firebending statuses in text-5 wording, fed from `PermanentRow.svelte`; `PermanentRow.svelte.test.ts` 2 pass; Academy `npm run check` 0 errors and `npm run build` pass (after `npm ci`, node 26 runs).

2026-10-29 realign: plan matches code (51 T2 cards allocated; none in `tla::INCOMPLETE`); T1 head unchanged since `01e8f9975`; no checks rerun.

2026-10-29 compress: removed orphaned `tests/support/casting.rs`; TLA any-color/two-color mana modes use `one_mana_of_any_color`/`two_mana_in_any_combination` (same mode order). Debug `cargo test --test cards` 303, rules 131 + 2 ignored, content_pack 3; clippy `-D warnings` clean.

2026-10-29 after syncing T1 `8790821da`: debug `cargo test --no-fail-fast` passes except the known semantic-conformance pair (cards 303, rules 131 + 2 ignored, lib 121); clippy `-D warnings` with and without `python` and fmt clean; pytest env/observation, continuation input, experience protocol, play modes, optional payments, text observation 123 pass. Arena admission suite left for gate.
