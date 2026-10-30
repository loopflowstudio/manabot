# T2 — TLA casting, costs, zone permissions and private zones (Task 87e012)

Stacked on T1 (892c47, PR #394, open; synced at its head `2c50c8793` via `1d650064e`); T2 is PR #398.
Task directive lists the 51 cards and obligations; allocation in
`docs/rules/tla/allocation.jsonl` (`planned_pr: T2`). Every merge needs Jack
Heart's review; don't arm T2's merge while #394 is open. Status: implementation
complete on w18; gated.

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

- Land only after #394 lands; re-sync then (re-record the w18 evidence below if T1's world moves).

## Check

2026-10-29 gate: w18 world-bound evidence re-recorded beside w17 (kept): `semantic-kernel-w18-v1` (same 14 cases, 3401 commands; test, CI check/fuzz and README roots moved), `authored-match-w18-v1` + `authored-match-parity-w18-v1` (same 142-Command tape, same labels and terminal board, new digests), `public-commitment-parity-w18-v1` (142 commands, 69 commitments, no gaps; needs `manabot.data fetch` of the RUL-9 measurement), Academy `*-w18.json` learn/precision/precision-mana demos, choice-navigation and trigger-order (same decision revisions as w17; tools `scratch/tools/regen_*_w18.py`); belief receipts move to possible-world v3 (w7 kept as excluded history); belief fixture gains its `relations` buffer (pre-existing KeyError, default policy input ≥ v2). Results: debug `cargo test` (opt-level 1) all binaries pass (lib 121, cards 304, rules 131 + 2 ignored, semantic_conformance 6, search_state_contract 12, …); `cargo fmt --check` and `clippy --all-targets -D warnings` clean (`--all-features` blocked locally by the PyO3 interpreter-version check, CI runs no clippy); CI conformance check + fuzz (32 cases) pass on w18. Arena admission/monitoring 66 pass; academy world-bound + belief + batched_eval + `tests/model/` 320 pass, 2 skipped; earlier wider Python pass 259 pass (its 12 failures were the fixtures above plus `test_batched_eval` hung-service timing, which failed only while cargo compiled alongside and passes alone). Academy vitest 109 pass, `npm run check` 0 errors. Ruff check/format clean on changed Python. Playwright e2e and clean-machine play ran in CI (pass, below).

2026-10-29 later passes (gate re-runs ×2, concept reviews 7–9, implement, compress, realign ×2, demo re-runs) at T1 `2c50c8793`, main unmoved: no tracked change, no rerun; plan matches code (`lesson_mana` stays the frozen total; per-class split is `restricted_mana`); cards 304 after `912a5d0e9`. PR #398 CI at `a5587df5a` (2026-10-30): all checks pass, including the CI-deferred Playwright (Academy UI) and Clean-machine Play; later commits are scratch-only. Older results: `git log -p -- scratch/87e012-tla-t2-casting.md`.

2026-10-29 demo (headless, decision pass 1): 60 random T2-deck games through `managym.Env` with no errors; flashback, Zuko impulse and foretell casts offered and taken; the opponent never sees the foretold card. Advance to gate. Record: `scratch/t2-demo.md`.
