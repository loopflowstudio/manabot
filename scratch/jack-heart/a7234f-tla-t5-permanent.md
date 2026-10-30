# T5 (a7234f): permanent copying — plan and status

Status: first slice implemented on main (synced to 10eeb1f55, afbd6a's
delete-old-world), 2026-10-30, per Jack Heart's steer (start from main; sync
onto T4 488a40 and T3 #399 once they publish). Stop at demo.

Blocked on publication (checked 2026-10-30): T4 488a40 is still implementing
with no PR; T3 #399 is open, unmerged. Phoenix Fleet Airship, and therefore
the Task's acceptance, waits for that sync. #399's title names world w18;
Jack Heart's 2026-10-30 direction keeps one open world (w17), so the sync
extends w17 unless that direction changes.

## Cards

| Card | State |
| --- | --- |
| Joo Dee, One of Many | defined, supported; 3 directed tests |
| Ran and Shaw | reviewed, removed from `tla::INCOMPLETE`; 2 new directed tests (copiable values with source present; firebending + Dragon anthem scope/duration) beside the 2 existing |
| Phoenix Fleet Airship | **not defined yet** — needs the T3/T4 sync |

## Primitives in this slice

- `Effect::CreateTokenCopy { nonlegendary }` replaces `CreateNonlegendaryCopy`
  (deleted). One authority: `Game::create_token_copy` in `flow/zones.rs`.
  Copiable values = the card's definition (play never mutates definitions;
  status/counters/non-copy effects live on `Permanent`); a copy token's
  definition already carries its copy effect, so copies of copies work; a
  departed source is copied from its retained card (LKI).
- `Effect::Surveil { count }` + `Decision::Surveil` on the existing
  LookAndSelect surface (SelectCard → graveyard, Decline → keep). No new
  `ActionType`/`ActionSpaceKind`, so `ACTION_TYPE_DIM` and checkpoints are
  unchanged. Continuation decision kind 19; stage `surveil` (Academy label,
  text-observation stage list).
- `ActivatedAbilityDefinition.sorcery_timing` (same name/semantics as T3/T4),
  enforced in `check_activation_terms`, so offers and submissions agree.

## Concept review (headless, 2026-10-30)

Settled by accepted direction; implemented 2026-10-30 (labels in
`Game::action_description`, `flow/action.rs`; witnesses
`joo_dee_surveil_choices_name_what_they_do`,
`ran_and_shaw_copy_shares_its_definition_and_its_rows_show_what_differs`):

- **Copy identity.** A token copy keeps its source's `definition_id` /
  `registry_key`: learning tokens and the text definition line describe the
  copied printed definition; the object's card row carries what differs
  (`is_token`, `is_legendary`, later the Airship's granted creature type);
  permanent rows share the key and link to that card row.
  No per-copy definition ids or catalog entries.
- **Surveil choices name what they do, at the shared authority** (Jack
  Heart's 2026-10-21 shared-decision-surface direction): under
  `Decision::Surveil`, `SelectCard` reads "Put <card> into your graveyard"
  and `Decline` "Keep on top". No new `ActionType`, so `ACTION_TYPE_DIM` and
  checkpoints are unchanged.
- `Effect::Surveil { count }` keeps kept cards in order, exact only for
  count 1 (the only printed amount). Leave as documented; a count > 1 card
  must add reordering in its own change.

No open product choice. Acceptance still waits on T3/T4 publication
(Phoenix Fleet Airship); that is a dependency, not a decision.

Re-review 2026-10-30 against the diff at 0ecacebc0: the code matches the
model above (one `create_token_copy` authority, surveil labels at
`action_description`, `sorcery_timing` in `check_activation_terms`); nothing
to change. T4 488a40 still has no PR (running compress); #399 still open.

## Design review (Jack Heart, 2026-10-30, comment 10a31948) — accepted

1. Stack on T3 #399 now; drop this branch's duplicate
   `ActivatedAbilityDefinition.sorcery_timing` for T3's (same name, already
   checked in T3's `check_activation_terms`).
2. Replace tensor-only copy identity (above, "Copy identity") with generic,
   composable modified-characteristics observation: copy + general
   exceptions + layered effects, in tensor and tokens. Shape proposed below
   first.
3. Sacrifice history on `TurnState`, shared with T4's raid history.
4. Then define Phoenix Fleet Airship.

## Proposal: modified characteristics (2026-10-30, draft; implemented as written)

A permanent's characteristics are its copiable values (layer 1) modified by
T3's layered effects (layers 2–7). Observation states each input, never only
the result:

- **Copy effect as data.** `Card.copy: Option<CopyEffect>` with
  `CopyEffect { of: CardDefId, exceptions: Vec<Modification> }`: `of` is the
  printed definition at the root of a copy chain; `exceptions` accumulate
  down the chain (CR 707.9b: a copy of a copy inherits its exceptions). The
  card's definition stays the materialized copiable values (every layer
  reader, legend rule and mana reader already starts there); `create_token_copy`
  derives it once from `(of, exceptions)` through `Modification::apply_copiable`.
- **Exceptions reuse T3's `Modification`** (one vocabulary for layer-1
  exceptions and layered records, one token id per variant). New variant
  `Modification::Nonlegendary` (loses the legendary supertype). Only
  characteristic-defining variants are copiable (`apply_copiable` returns
  false for rules effects; definitions never author those as exceptions,
  checked by a test over every defined card).
- **Card authoring:** `Effect::CreateTokenCopy { exceptions: Vec<Modification<EffectValue>> }`
  replaces `{ nonlegendary }`. Ran and Shaw: `[Nonlegendary]`; Joo Dee and the
  Airship: `[]`.
- **Observation:** `PermanentData.copy: Option<CopyData { of: registry_key,
  exceptions: Vec<ContinuousEffectData-shaped modification> }>` (omitted when
  none). v5 facts gain `FactKind::Copy` (id: the copied definition's learning
  token value, source: its registry key + 1) and `FactKind::CopyException`
  (id/detail/value exactly as `FactKind::Effect` rows, via one shared
  modification-row helper; no duration or order). Layered effects keep
  T3's `FactKind::Effect` rows. The permanent row's definition index stays
  the copied definition (its effective printed characteristics).
- **Tokens:** `CreateTokenCopy`'s exceptions tokenize as the same
  `Modification::*` values `ApplyContinuous` produces.
- **LLM text / Academy:** "copy of Ran and Shaw, except not legendary" as a
  permanent note, the same words in both.
- **Guard:** `FactKind` and `Modification` stay exhaustive; witnesses for
  `Copy`, `CopyException` and `Nonlegendary` (Ran and Shaw).

## Remaining (after syncing onto T3/T4)

Phoenix Fleet Airship needs, all owned by T3/T4 constructs:
- Vehicle + `crew: Some(1)` and `PendingChoice::Crew` (T3).
- "As long as you control eight or more permanents named Phoenix Fleet
  Airship, this Vehicle is an artifact creature": T3 `StaticTypeGrant` with a
  condition counting permanents sharing this source's name (names must not
  enter learning tokens — express it relative to the source, not as a name
  string).
- `TriggerCondition::BeginningOfYourEndStep` (T3/T4) with an intervening-if
  "you sacrificed a permanent this turn": per-player turn history, incremented
  in `sacrifice_permanent`, observed through T3's v5 fact rows (not a new
  fixed player column).
- `CreateTokenCopy { nonlegendary: false }` already exists.
- Tests: copy on own end step only after a sacrifice; LKI copy if the Airship
  left before resolution; eighth Airship animates all of them; crew 1.

## Delete — do not maintain

- `Effect::CreateNonlegendaryCopy` and its resolution body (done).

## Checks

- 2026-10-30 at 0ecacebc0: debug `cargo test --test cards` 318/318 pass (both refinement witnesses included); extension rebuilt 12:08 with the label change.

## pr-publish step, 2026-10-30 (loop 2 pass 2) — not published

Blocked; no `lf sync`/`lf pr publish` run. Jack Heart's launch steer says
"Stop at demo", and the accepted outcome (Phoenix Fleet Airship) is not
met: T4 488a40 still has no PR (running compress), T3 #399 still open.
Publish after syncing onto T3/T4 and defining the Airship.
