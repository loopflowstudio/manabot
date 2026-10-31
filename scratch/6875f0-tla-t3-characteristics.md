# T3 — characteristics, counters, replacements, attachments (Task 6875f0)

Brief: the Task directive (49 cards, obligations in `docs/rules/tla/allocation.jsonl`,
`planned_pr == "T3"`). T1 (#394) and T2 (#398) are on main; PR #399 targets main
and carries main `10eeb1f55`. Every merge needs Jack Heart's review.
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

30. Continuous effects as records (2026-10-30, Jack Heart's direction below).
   `state/effect.rs` (`ContinuousEffect`, `Modification<N>`, `EffectDuration`,
   `Affected`) and `flow/effects.rs` (apply, queries, expiry) replace every
   per-object until-EOT field; `GameState.effects` is journaled whole and
   hashed (`MATCH_STATE_HASH_VERSION` 11). `statics.rs` applies statics and
   records by layer, then timestamp; `Permanent.timestamp` (renewed on attach,
   CR 613.7e) orders statics. 23 one-off `Effect` variants became
   `Effect::ApplyContinuous { affected: AffectedSpec, modification, duration }`
   (+ `PutCountersOnTriggeringPermanent`); set files converted with
   `scratch/tools/convert_continuous.py`. Day of Black Sun's loss is a record.
31. Counters map and v5 facts (Jack Heart's "1a 2a" review, below).
   `Counters` is one fixed `CounterKind` → count map (+1/+1 included,
   serialized as a map of present kinds). v5 permanent rows drop per-kind
   columns (52 → 37, adding a timestamp rank); a `facts` buffer (128 × 12,
   `FactKind`) carries counters, added subtypes, set name/creature types,
   used abilities and effect records, with learning-token ids. Native and
   Python encoders, PyO3 (`facts` buffer, `ContinuousEffect`), model
   (`manabot/model/facts.py`, joined into object embeddings), LLM text
   (`effect_phrase`), Academy notes (`effectNote`) and the experience
   protocol carry the records.

32. Merge of main `c7979d310` (T2 #398 squashed as w17, 2026-10-30) as
   `eecbc465f`: T3's tree plus T2's post-`601ed8f82` delta (w18 → w17 rename,
   belief range/tracker); T3's `*-w18*` evidence and fixtures moved over the
   older w17 copies and renamed; WORLDS.md's w17 row and section carry T3.
   Permanent observations omit `timestamp` unless an attachment renewed it.
   Compress after it: one Aura-timestamp reader
   (`set_characteristics_timestamp`) and `ContinuousEffect::follows` replace
   three inline timestamp comparisons. `PermanentData.plus1_counters` stays
   (pre-T3 wire field in every retained fixture).

33. Merge of main `10eeb1f55` (afbd6a: statics and additional costs as
   `StaticAbility`/`AdditionalCost` variants, old-world machinery deleted) as
   `3cef39b8e`. T3's one-off `CardDefinition` fields became variants (list in
   `scratch/questions.md`). Realign after it re-recorded what the merge left
   stale (authored-match parity receipt; belief v3 receipt pins), repointed
   the 400.7j / 608.2k audit anchors from the removed
   `ModifyTriggeringPermanent` to `AffectedSpec::TriggeringPermanent`, allowed
   `clippy::large_enum_variant` on `AdditionalCost` (T3's wider
   `CardPredicate` grew `SacrificeCost`), and gave `ContentPack` a `Debug`
   that omits its lazy `OnceLock` caches: the shared default pack filled them
   from other test threads, which was the intermittent "rejection must not
   mutate state" failure in `structured_offers` / `structured_attacker_offers`.

## Concept review after item 33 (2026-10-30)

The accepted models (continuous-effect records, counter map, open vocabularies
as fact rows, one open world w17) match the code: no `temp_*`,
`keeps_red_mana`, `no_maximum_hand_size` or `Permanent.abilities_lost` state
remains; WORLDS.md's w17 row and v5 paragraph state the shipped widths. No open
product choice.

Next implement direction: the guidance this branch added to AGENTS.md
(`CLAUDE.md` links to it) predates items 30–31 and teaches the old model to
every later agent. Make it state what the code does:
- v5 widths: player 63, card 55, permanent **37** (not 52), relation 48,
  plus the `facts` block (128 × 12, `FactKind`) for open-ended vocabularies.
  New open-vocabulary state is a fact kind, not a column (Jack Heart's 2a).
- `observability.rs` names each `Observation` field's tensor columns **or fact
  kind**.
- `Game::has_lost_abilities` reads continuous-effect records and an Aura's
  `SetCharacteristics`; drop "never read `Permanent.abilities_lost` alone"
  (that field no longer exists). Add one line: an until-end-of-turn or
  other-duration change is an `Effect::ApplyContinuous` record
  (`state/effect.rs`, `flow/effects.rs`), never a new `Permanent`/`Player`
  field.
Docs only; no code, receipt or world change. Done 2026-10-30: widths checked
against `ObservationSpace` (player 63, card 55, permanent 37, relation 48,
facts 128 × 12); grep finds no stale wording.

Concept review after the AGENTS.md pass (2026-10-30, HEAD `09e6babe5`, main
`6f48f5ca0` merged and pushed): nothing to change. The removed per-object
fields survive only as derived observation views (`agent/observation.rs`,
`experience.rs` `Legacy*View`, PyO3) computed from `Game` queries such as
`exiled_if_it_would_die`, which the accepted model keeps; no open product
choice.

## Remaining

- Characteristics consolidation (Jack Heart, 2026-10-30, steer 9bf48ffa):
  one `Game::characteristics(id)` view replaces the separate public readers;
  done. Check: debug `cargo test --no-fail-fast` 755 passed, clippy
  `-D warnings` clean, w17 fixtures unchanged; Python world-change set,
  semantic, belief, env and Academy parity tests 444 passed. CI fix for
  `test_action_identity` landed at `a18902e0e` (see questions.md).
- Browser proof (Playwright e2e) is CI's.
- Jack Heart's review of PR #399.
- Outside T3 (Jack Heart's call): the opponent's sideboard counts in the
  `Observation` and LLM text predate T3 (`scratch/questions.md`, item 23).

Check (2026-10-30, after the `10eeb1f55` merge and realign repairs): debug
`cargo test` all binaries pass (lib 122, cards 394, rules 136 + 2 ignored; one
run hit the now-fixed `Debug` flake); fmt and clippy `--all-targets -D warnings`
clean; extension rebuilt; world/observation-shape set plus `tests/academy`,
`tests/belief`, `tests/env`, `tests/verify`, `tests/semantic`,
`tests/infra` pass after re-recording (746 + 61).

## Accepted: world numbers name freezes (Jack Heart, 2026-10-30)

Proposal page `scratch/concept-review.html`. Safety does not depend on the
number: `checkpoint_incompatibility` (`manabot/model/catalog.py`) rejects any
checkpoint whose whole `world_binding` differs.

Accepted, Jack Heart 2026-10-30 ("1a, 2a"): a world number names a deliberate
freeze, not a merge. Main carries one open world; merged rules or shape changes
extend it in place and update its receipts; freezing re-runs the headline
baselines, records receipts, marks the WORLDS.md row frozen and opens the next
number. Policy input versions follow the same rule. The TLA batch's open world
is w17 (main's): T2 and T3 rename their w18 back to w17 and extend v5 in place;
T3 never takes w19/v6. T2 lands the WORLDS.md opening-paragraph change (page
wording) and the rename first; T3 follows on its next sync.

## Accepted: continuous effects as records (Jack Heart, 2026-10-30)

Jack Heart, reviewing `Player.keeps_red_mana`: "Redo this to be the architecture
that will scale to 10s of thousands of cards. IN this diff." That replaces the
per-object until-end-of-turn fields with one model of continuous effects
(CR 611, 613), in T3.

Principle: adding cards adds data, never engine fields. A card applies a
continuous effect by composing a closed vocabulary (what it modifies × whom ×
how long). A new mechanic adds a variant to that vocabulary, and the exhaustive
observability guard then forces its tensor, token and LLM encoding.

Model (names are direction; implement may refine them):
- `Game.effects: Vec<ContinuousEffect>`, kept in timestamp order and journaled
  for undo like players and permanents. One record holds:
  - `id`, `timestamp` (a monotonic `Game` counter; CR 613.7)
  - `source` (`ObjectRef` + definition; last-known information if it left)
  - `controller`
  - `affected`:
    - `Player(PlayerId)`
    - `Object(ObjectRef)` (one incarnation; CR 400.7 ends it when the object
      changes zones)
    - `Locked(Vec<ObjectRef>)`: the set fixed at resolution (CR 611.2c)
  - `modification`: one typed enum grouped by layer, covering:
    - layer 4: add/set types and subtypes
    - layer 6: add keywords, lose all abilities, grant ability, firebending N
    - layer 7b: set base P/T
    - layer 7c: modify P/T
    - rules effects: can't be blocked, exile if it would die, keep unspent mana
      of a color, no maximum hand size, becomes a creature
  - `duration`: `EndOfTurn` (CR 514.2), `EndOfCombat`, `WhileSourceOnBattlefield`,
    `UntilYourNextTurn`, `Game`.
- Static abilities stay derived from their permanents. They feed the same layer
  application with the permanent's own timestamp (CR 613.7a/d), so `ScopeView`
  gathers static and recorded modifications and applies them by layer, then by
  timestamp. Honest Work's "no timestamps" limitation and ability loss ignoring
  timestamps (`questions.md`) go away; a later grant now survives an earlier
  loss. Dependency order (CR 613.8) stays out of scope and is recorded as a
  limitation unless a T3 card needs it.
- One read path: every characteristic reader (`effective_keywords`,
  `has_lost_abilities`, P/T, subtypes, `permanent_is_land`,
  `activated_abilities_for`) and every rules query (blockability, death
  replacement, mana emptying, hand size) reads `Game`, never a `temp_*` field.
  For scale, reads index records by affected object rather than scanning, and
  characteristics may be cached per state revision if profiling asks for it.
- Expiry is one function per boundary:
  - cleanup removes `EndOfTurn`; the end of combat removes `EndOfCombat`
  - a source leaving removes `WhileSourceOnBattlefield`
  - an affected object changing zones removes `Object` records for it
  - `clear_temporary_modifiers` goes away
- Card authoring: one engine `Effect::ApplyContinuous { affected, modification,
  duration }`, plus constructor helpers. It replaces `GrantKeywords*`,
  `set_base_pt`, `BecomeCreatureTarget`, `SetBasePtTarget`,
  `GrantFirebendingToTarget`, `keep_unspent_red_mana`,
  `destroy_creatures_losing_abilities`' loss and similar one-offs.
- Remove: `Permanent.{temp_power, temp_toughness, temp_base_power,
  temp_base_toughness, temp_keywords, temp_subtypes, abilities_lost,
  cant_be_blocked_this_turn, firebending_this_turn, exile_if_dies_this_turn}`,
  `BecomesCreature::EndOfTurn` and `Player.{keeps_red_mana,
  no_maximum_hand_size}`. `activated_this_turn` and turn counters are turn
  history, not effects, and stay.
- Observation: each record visible to its viewers as a row of (affected,
  modification kind and amount, source definition, duration, timestamp order).
  It goes in the `Observation`, the policy tensor (v5 extended in place: an
  effect row block like the relation rows, with a join to the affected object),
  the learning tokens (modification and duration enums) and the LLM text
  ("until end of turn: +2/+0, flying (Iroh's Demonstration)"). The derived
  effective values stay too. The observability guard matches `Modification` and
  `Duration` exhaustively. Academy shows the records as permanent and player
  notes.
- World: w17, extended in place (1a/2a).

Proof:
- Every existing card test passes unchanged; assertions change only where
  timestamp order is now correct, and each such change is named.
- New tests:
  - timestamp order (a loss, then a later grant; Honest Work, then a pump)
  - each expiry boundary
  - the source leaving
  - the affected object changing zones and returning as a new object
  - undo and rollback of effect records
  - viewer safety
  - hash and determinism stability
- Native/Python v5 parity; guard and directed observability tests for each
  `Modification` and `Duration`; debug `cargo test`; the AGENTS.md world suite.

## Accepted: open vocabularies as data (Jack Heart, 2026-10-30, PR review "1a 2a")

Page: `scratch/pr-review.html`. Both changes land in this diff, together with
the continuous-effects records.
- Counters (1a): `Permanent.counters` becomes one `CounterKind → count` map,
  with +1/+1 included. `plus1_counters` and the named fields in `Counters` go
  away. A new counter kind is one enum variant. Readers use one accessor
  (`counter_count(kind)`), and serialization stays omitted-when-empty with a
  stable key order for hashes.
- Encoding (2a): v5 (still open, so it extends in place) encodes open-ended
  vocabularies as rows of (object index, vocabulary kind, vocabulary id, value),
  joined to objects the way relations and the attachment link are. The rows
  cover:
  - counter kinds
  - subtypes gained or set by an effect
  - a set name, as a definition or name id
  - set creature types
  - used or exhausted ability indices
  - continuous-effect records

  The per-kind columns go away: stun, fire and double-strike counters,
  `TEMP_SUBTYPES` and its "other" count, the used-ability flags and their
  "other" count, and the characteristics-set bool. Generic columns stay, such as
  tapped, damage, the effective P/T derived values and the keyword block. Row
  capacity is fixed in `ObservationSpace`, and overflow fails explicitly like
  the other blocks. Vocabulary ids come from the same Rust enums and tables as
  the learning tokens, so the tensor and the tokens share one vocabulary. The
  guard test claims each row kind exhaustively, and the directed tests assert on
  rows, not columns. Native/Python parity holds.
