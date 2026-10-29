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
