# MTG-430 — Planeswalkers, loyalty, empowerment and emblems (FDN + FRA)

Base `cf73d9778f`; synced main through MTG-420 (#593, `45ddf3bca7`) and
later main (`964699a20b`). MTG-420 supplies exact object identity for stacked
abilities. Live layers and triggers also resolve exact command-zone sources.

Done (detail in commits):
- Slice 1, loyalty abilities and quota (`a5534ee7bb`, `aa35e1a2b6`).
- Slice 2, attack targets and Tomik (`0674982740`, fixtures `5ecb50ceee`).
- Slice 3, command-zone emblems (`833f67e3ed`, `52b05b2fa8`, `eb842d8d57`).
- Slice 4, empower and `token.jace` (`4966204033`, `2327a41b94`).
- Slices 5–6, Way grants, zero-loyalty SBA exception, loyalty activation
  events and turn-state column (`6429f0f54a`, `553ce2b149`, `d83da6e53b`).
  Lesson: positions never name clauses (a declined up-to clause renumbered
  Warlord's target clause).
- Slice 7, divided damage, keep-one-each-type, delayed attack triggers and
  seven planeswalkers; `bench_states` stress states (`580ff7d36e`,
  `a256b00666`, test moves and fmt `75ac57e28e`).
- Slice 8, FRA lands (`b17ed61f62`, `55430f5c91`), primitives
  (`fd549a4339`), 28 adjacent cards with directed and integrated tests and
  closed `fdn.jsonl`/`fra.jsonl` (`09fb662747`), fixtures (`b1649e3a8c`).
  Recorded: 68/68 `fra::` tests, conformance tests, four protocol schema
  tests (no drift) and `cargo fmt --check` passed in debug; cp313 refresh
  kept every offer label and Command.

Pending on other authorities (ledger rows carry them): Countersculpt and
Theorist's Sanctum (behold), Tam (resumable proliferate), Gideon's Memorial
and Chandra, Chill of Compliance (`ManaUse` by card type), Return to the
Light Realms (CR 303.4f non-cast Aura attachment), Loot (negative-power
damage assignment). Open defect: no replacement diverts a
battlefield-to-exile move, so Overwrite's diverted case is untested.

Already red on CI at `d1ef91d814` (run 45568225341), before slices 5–6, and
left for gate: `observability::` legacy card-row parity (four tests,
card column 8 vs permanent column 6),
`observation_layout::named_layout_matches_every_encoder_width`
(agent_permanents), `content_pack_contract_covers_env_roots_siblings_and_rollout_slots`,
`definitions::every_authored_subtype_has_a_category` and the frozen
semantic-conformance cases.

## Passes remaining

No implementation slice remains. Compress resolved the shape findings
(`FightLastTarget` folded into `Effect::Fight`, `RevealUntilToHand` renamed
to `Effect::RevealUntil { predicate, to }`, with `token_renames.rs` entries),
simplified target-division and defender paths, and registered `GrantPlay` /
`SacrificedBy` / `"target divisions"`. Gate:
- Debug nextest (`CARGO_BUILD_JOBS=2 cargo nextest run --profile ci
  --test-threads 2`), Engine Budget, the pinned pilot (record with
  `--reason` if the attack row or Way/loyalty tokens moved), and the
  pre-existing CI failures above. Keep main's MTG-420 receipts at their
  original revision.

## Consumer invariants (every slice)

Offers complete under `skip_trivial`; replay, forks and search clones carry
loyalty ledgers, attack targets and emblems; viewer privacy unchanged (all new
state is public); v11 columns appended, never reordered; frozen v9/v10 inputs
untouched; `protocol/*.schema.json` patched with each wire type change;
`tests/model/test_pinned_pilot.py` — declare the break with
`scripts/pinned_pilot.py record --reason` if the attack action row moves.
Engine Budget within 10%; rebuild the extension and run debug nextest
(`CARGO_BUILD_JOBS=2 cargo nextest run --profile ci --test-threads 2`) before
publishing.

## Ledger

Update `fdn.jsonl` (`implementation.status`) and `fra.jsonl` rows as each
slice lands; pending rows carry the missing authority now.
