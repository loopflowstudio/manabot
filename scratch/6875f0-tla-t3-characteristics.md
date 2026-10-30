# T3 — characteristics, counters, replacements, attachments (Task 6875f0)

Brief: the Task directive (49 cards, obligations in `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T3"`). Stacked on T1 (892c47, PR #394; last synced at head `01e8f9975`)
and merged with T2 at PR #398 head `a5587df5a`. Every merge needs Jack Heart's review.
Card tests live in `managym/tests/cards/tla.rs` (steer, Jack Heart 2026-10-29).

## Done (49 of 49)

Per-item detail for items 1–22 is at
`d0c3e3a8e:scratch/6875f0-tla-t3-characteristics.md`; per-slice code in
`git log 86ebe9ccf..`.

1–11. Conditional entry, static grants, counters by kind (`CounterKind`, stun
   in `untap`, exhaust), 7b base P/T, Equipment and Aura attachment with SBAs
   704.5m/n, Vehicles and crew, state-dependent mana, `ScopeView` layered
   characteristics, gained land types and red-mana replacement, X costs.
12–16. Excess damage and fight, kept red mana, tutor and library-top casting,
   granted abilities from attachments (`activated_abilities_for`,
   `AbilityGrant`), Honest Work's `set_characteristics` (one
   `has_lost_abilities` reader), one tap authority (`permanent_can_tap`),
   shared-creature-type targets.
17. T2 sync onto PR #398 head `16dfd3dbb`: one X announcement, exhaust gate,
   `ZoneEntry`; world w17. 18. Glider Staff, Crashing Wave.
19–20. Fire Nation Palace; observation-only fields for counters by kind,
   attachment, set name/types, granted land, firebending, stack X and spell
   grants (PyO3, protocol schemas, Academy types), then Academy and LLM text.
21–22. Card tests in `tests/cards/<set>.rs` with shared `cards/support.rs`.


## Accepted model (concept reviews, 2026-10-29)

A permanent's characteristics are what `Game` derives from them (`ScopeView`
layers, `has_lost_abilities`, `permanent_name`, `permanent_has_subtype`,
`activated_abilities_for`). `Permanent` and `Card` fields are inputs, never
answers. Observations stay complete (acceptance; Jack Heart's 2026-10-21
shared-decision-surface direction), so item 19 added counters by kind,
`attached_to`, effective name and creature types, granted land, stack X, spell
keyword grants and granted firebending, following T2's `CardData.face_down`
precedent. Frozen widths are unchanged.

Item 20 (`a1d67b416`) gives people and LLM bots the same fields as labels only:
`permanent-notes.ts` (counters by kind, "attached to X" / "X attached" across
seats via `tablePermanents`, set creature types, granted land, firebending) feeds
`Card`'s accessible name and margin chips, and the ordered stack shows "X = n".
`text_observation.py` appends the same fields only when they are present, so
existing text is byte-identical. `_characteristics` in the projection omits on
`None`, not falsiness. Academy checks use nvm node 24 (Homebrew node 25.8 lacks llhttp). TLA card
names left in `tests/rules/` are CR-rule witnesses or deck filler, not
single-card tests (concept review after item 22).

## Delete — do not maintain

Qualification entries, `Permanent::local_power`/`local_toughness`,
`Permanent::can_tap`/`producible_mana`,
`StaticScope::Equipped`/`YouControlWithCounters`, play.rs `MAX_X`,
`Game::equipped_creature`, `Permanent::can_attack` and
`CounterKind::parse` (authoring-era) are gone; do not restore them.

## Accepted: the policy input shows everything (Jack Heart, 2026-10-29)

Jack Heart, at demo: "clearly the observation (and the tokenized representation)
need to show everything." This supersedes item 19's "no policy tensor column"
choice and the open PR-review question (concept review page
`scratch/concept-review.html`, options 1a and 2b).

Contract: the `Observation` is complete, and so is the policy input. New engine
state reaches the native and Python policy encoders in the change that adds it.
A frozen version never silently falls behind; older versions stay byte-frozen
for their checkpoints.

Jack Heart's later 2026-10-29 steers widen it: every mechanic is observable in
all three encodings (policy tensor, learning program tokens, LLM text), with an
exhaustive guard, a directed test per mechanic and viewer-safety tests; cards
become engine-type definitions (no authoring layer) on T1's sync, each with a
Scryfall id.

23–25. Policy input v5 (2026-10-29), the new default; v1–v4 byte-frozen.
   Layout constants live in `observation_encoder.rs` (Python mirrors them;
   WORLDS.md w18 lists widths); `ContinuationFrame` gains `x` and
   `granted_keywords`. The model joins attachment and blocking columns
   (`PERMANENT_LINK_INDEXES`), not numeric features. Item 24 added the
   state the `Observation` lacked (attacking, blocking, abilities lost, used
   abilities, kept red mana, no maximum hand size); item 25 shows it in
   Academy (`permanent-notes.ts`, PlayerArea) and the LLM text with the
   same words, and adds directed LLM-text tests (Foretell, foretold casts,
   Hakoda's library-top casts).

26. Sync onto T1's one card representation (`01e8f9975`, 2026-10-29; Jack
   Heart's steer). Every T2+T3 card and authoring `Op` is now an engine
   `CardDefinition`/`Ability`/named `Effect` value: set files regenerated with
   T1's `scratch/conversion/generate.py` (extended for `ManaUse`, restricted
   mana, `{X}` and all `state` modules) from T3's lowering; all 254
   definitions Debug-identical to it. 93 new Scryfall ids fetched. World w18
   (T1's w17 is the representation). v1–v4 stay 28 wide; T1's v4 `attacking`
   column moves to v5. Token closure walks serialized `token_name`s.
   `SetCharacteristics.name` became `named` (tokens never carry a `name`
   field). Tests now use T1's `tests/support/{game,play,inspect}.rs` (added:
   `five_color_unfunded_setup`, `move_to_hand`, `cast_card`, `castable`,
   `play_land`, `can_activate*`, `activate_card`, `has_action`,
   `offered_targets`, `counters_of`); `observability.rs` joins T1's token
   guards (106 `Effect`, 16 `TriggerCondition` witnesses, `Permanent`/`Player`
   fields at v5 columns) with T3's v5 tensor guard.

27. Merge of T2's PR #398 head `d218e1dc2` (2026-10-29). T2's card conversion,
   tests and v5 rows are the base; T3's columns extend T2's unfrozen v5 and w18
   in place (assumption recorded in `scratch/questions.md` for Jack Heart to
   overrule; the alternative is v6/w19). Widths: player 63 (T2's 60 plus turn,
   seat, kept red mana), card 55 (T2's; the foretold term marks face-down),
   permanent 52 (T2's 32 plus counters by kind, attachment join, set
   characteristics, granted land, six temporary subtypes and a count, blocking
   join, lost abilities, four used-ability flags and a count), relation 48 (T3's
   frame X and granted spell keywords); v1–v4 keep 40 / 41 / 29 with
   `attacking` at permanent column 27. `granted_firebending` replaces T3's
   `firebending`. T2's shared `check_activation_terms` now also checks sorcery
   timing; `ActivatedAbilityDefinition::requires_target` reads
   `target_requirement`. T3's refreshing index helper is `offers_ability`.
   Compress after the merge: both encoders write player columns in column
   order; the native one-use `shows_everything` gate is inlined.

28. Second sync (2026-10-29): T2 head `89e320c48` (possible worlds v3, Academy
   ability-use statuses) and T1 head `2c50c8793` (`world` pool key, card
   catalog under `cards/`). T2's exhaust / once-each-turn / firebending card
   props became `permanentNotes` entries (one Academy note authority; wording
   matches the LLM text, "firebending N until end of turn").

29. Third sync (2026-10-30): T2 head `a5587df5a` (w18 world-bound evidence,
   foretold-slot determinization). T3 keeps its in-place v5/w18 extension, so
   the w18 evidence was re-recorded at T3's widths (63 / 55 / 52, relation 48):
   `semantic-kernel-w18-v1` via `semantic_conformance record` (same 14 cases /
   3401 commands), authored-match receipt (digests only), authored-match and
   public-commitment parity (142 commands, 69 commitments, no gaps; pinned
   `AUTHORITY_SHA256` / `TERMINAL_STATE` updated), Academy demo and
   choice-navigation fixtures via T2's `scratch/tools/regen_*_w18.py` (same
   decision revisions as w17), trigger-order root/content digests, and the
   four possible-world v3 belief receipts (the schema binds the content digest).

## Remaining

- If T2 lands on main first, its w18 evidence is frozen and T3 takes w19 (and
  v6) instead; the open in-place assumption is in `scratch/questions.md`.
- Browser proof (Playwright e2e) is CI's.

Check (2026-10-30, at the third sync): debug `cargo test` all binaries pass
(lib 122, cards 384, rules 131 + 2 ignored, semantic_conformance 6, …); fmt,
clippy `--all-targets -D warnings`, conformance check and fuzz (seed 24301, 32
cases) clean; `tests/academy` 261 pass; world/observation-shape set plus
`tests/belief`, `tests/env`, text observation 301 pass; ruff clean. Realign (2026-10-30): PR #398 `a5587df5a` and #394 `2c50c8793` heads unchanged and contained; Academy shared-table/study/experience protocol and CI-selection tests 74 pass (`tests/eval/test_arena_log_conformance.py` needs undeclared `duckdb`, unchanged from main).

Concept review (2026-10-30, after the third sync): no change. The implemented
change matches the accepted model (derived characteristics, complete
observation, all three encodings, one card representation). Two recorded items
are not T3 product choices: the in-place v5/w18 extension is a reversible
versioning assumption (no checkpoint uses either; `scratch/questions.md`), and
the opponent's sideboard counts in the `Observation` and LLM text predate T3
(on main), so that visibility question stays Jack Heart's, outside this Task.
