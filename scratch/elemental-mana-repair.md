# MTG-174 shared mana repair

Accepted direction: shared payment authority with visible choices and exact
continuations. Scope: bounded repair; no paid/scientific execution or publication.

Earlier diagnosis, full scenario matrix, exact hashes and validation are retained
at `863938740:scratch/elemental-mana-repair.md`. The original Professor root and
later priority Prism/Treasure counterexample do not establish the unretained GPU
failure's scalar cause. Actual outcome: `experiments/elemental-uniform-baseline.md`.

## Accepted implementation at 863938740

`PaymentModel` in `flow/mana.rs` owns offers and executable witnesses: exact
sources, pool remainders/restrictions, spend purpose, life, waterbend, and
per-use tap/sacrifice reservations. `Decision::ManaUpfront` carries payer,
source, ability, recursive parent and its effect frame. Upfront payment precedes
source costs and visible output. Mana abilities create no stack object.
Fixed production preserves source order; only nested automatic prefixes minimize
commitments while proving the complete parent continuation.

Delete — do not maintain: grouped boolean `can_cover_cost`, separate fixed-only
planner, greedy production loop, recursive waterbend feasibility, whole-object
`reserved_payment_source` / `pending_mana_source` / `produce_mana_without`, and
obsolete duplicate frontend/demo authorities. Preserve the current upstream
Academy surface and one native payment authority.

Prior acceptance and exact receipts: `863938740:scratch/elemental-mana-repair.md`
(423 debug Rust, 14 corpus replays, 51 Python). Frozen native sources remain intact.

## Integration regression repair — 2026-10-22

Fetched current main `29ab4fa0ddcc759ddb2b699ce3d08ccfe51ad757` and inspected
supported `lf sync origin/main --plan`. Main has five later commits, including
experiment and engine packaging reorganizations. The caller owns the next sync;
this repair restores upstream semantics against the already merged `20194feb7`
instead of partially importing the later package move.

The previous conflict resolution reverted the shared Academy surface, typed
presentation subjects, demo admission/provenance, explicit `skip_trivial=False`,
Study exact-offer execution/privacy checks, replay persistence optimizations,
Academy UI CI, clean-machine timing receipts, deterministic profiler tests,
named training recipes, docs and launcher/deployment/package paths. Restored
those upstream paths and removed the obsolete duplicate `frontend/` tree and
learn-only demo owner/tests. The repaired corpus remains the CI conformance root.
The optional reconnect fixture retains six lands so both payment outcomes remain
available after the filter fee. No Rust source changed in this integration pass.

Native shared-payment/diagnostic sources, uniform sampler, manifests, scientific
protocols, outcome report and frozen conformance evidence remain unchanged.
Remaining caller boundary: compress, sync/realign, concept review, publication
and requested HTML pr-review walkthrough. No further Flow was launched here.

Validation: `cargo test --locked --manifest-path managym/Cargo.toml --test rules_tests temur_cards -- --test-threads=1`: 33 passed. Focused uv pytest groups: 140 Study/reconnect/runtime checks, 41 surface/sampler checks, 4 decision-surface checks, 2 artifact/recipe checks passed. Ruff and changed-file formatting/diff checks pass. Logs: `.runs/mtg174-integration-repair/`. No native rebuild required.
