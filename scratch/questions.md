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

## T3 (6875f0) assumptions — 2026-10-29

- Support at this base still reads `qualifications.json`; the four new lands get
  entries there so the world supports them. Jack Heart's steer says tests never
  gate support; those entries go away when T1's gate removal is synced.
- Per the same steer, only this batch's new tests went to `managym/tests/cards/`;
  moving T1's existing per-card tests is left to T1/PR #394 to avoid conflicts.
- Conditional entry reads the card owner's battlefield (cards enter under owner
  control in every current path).
- Learning schema 8 is unreleased (new in T1), so new vocabulary is appended in place.
- `semantic_conformance_tests` fails at this base (WORLD_VERSION w16, evidence w15);
  left to T1's world release rather than regenerated here.
- Zookeeper's creature-type count excludes a fixed noncreature-subtype list rather
  than a creature-type registry (none exists); revisit if a pool card breaks it.
