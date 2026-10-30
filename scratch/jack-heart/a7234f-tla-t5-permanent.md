# T5 (a7234f): permanent copying — plan and status

Status: first slice implemented from main (c7979d310), 2026-10-30, per Jack
Heart's steer (start from main; sync onto T4 488a40 and T3 #399 once they
publish). Stop at demo.

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

- Debug: `cargo test --test cards` 310/310 pass; `conformance_tests`, `content_pack_tests`,
  `semantic_conformance_tests`, `authored_match_tests` pass; clippy clean.
