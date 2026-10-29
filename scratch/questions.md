# TLA direct-Rust decisions — 2026-10-28

Reversible operator choices made headless; each is noted in the PR.

- Old worlds run from their recorded git revision; no in-process rules shims.
- No w17: main is w15 and w16 is this PR's unreleased world, so key renames and
  the default-pack change stay in w16 (one unreleased world per Task).
- IR keeps the retired compiler's JSON shape, built in Rust on `serde_json::Value`
  from serialized typed definitions. This made byte-identity provable; lowering
  typed definitions straight to `Step` is a possible later simplification.
- Learning tokens still come from `learning.py` over that IR (now native-sourced);
  schema 8 unchanged.
- Keys renamed by real set; qualification hashes refreshed only after a proof of
  rename-equivalence, not by review of behavior (behavior did not change).
- Research cards (Lightning Bolt, Hill Giant, Savannah Lions, Glory Seeker) now sit
  in `tla-pool-v1` as unsupported definitions and in the default pack.
- World support = current qualification in `tla-pool-v1`; native legacy cards are
  unsupported. Construction policy stays with D1/f871a8.
- Not rewritten (dated evidence): `docs/rules/tla/t1-clause-review.json` and the
  Academy curated pack `local_identity` values keep pre-rename keys.
- Full TLA completion remains the eight-PR plan; no merge or new Task authorized.

Main's MTG-193 notes at the sync live at `6d069d6e5:scratch/questions.md`.

## T2 (87e012) — 2026-10-29 operator choices

- When printed flashback and Iroh's grant both apply, the engine uses the
  grant's cost instead of offering both. True dominance holds for every TLA
  card; a non-dominated alternative would need a cost choice.
- Learning schema 8 extended append-only (Plant, Rogue, `flashback`) following
  `1e0d3c998`. The first new opcode needs learning schema 9; not yet taken.
- Slice 1 avoided a new opcode: landcycling is `search_basic_land` with
  `destination: hand` (searching any card of the subtype, not only basics),
  graveyard self-return is `return_to_hand` targeting `source`. If reviewers
  prefer a distinct `search_library` opcode, that is schema 9.
- Card abilities get their own action variant (`ActivateCardAbility`) sharing
  the `PriorityActivateAbility` type, rather than widening `ActivateAbility`
  to an `ActivationSource`; internal payment/target state uses the shared
  `ActivationSource`.
- A discard-cost ability refers to its card's graveyard object after costs
  (Team Avatar's damage source), standing in for the hand object's LKI.
- World stays w16 while T2 only adds cards and flashback costs; bump when a
  slice changes existing rules meaning (one unreleased world per Task).
