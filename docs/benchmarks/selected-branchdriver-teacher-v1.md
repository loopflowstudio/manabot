# Selected BranchDriver teacher evidence v1

Decision: **remain on `full_clone/current_game_v1` for production PUCT**.

RUL-1 retained compact full clone as the simplest correct representation. This
receipt runs that exact driver through the real determinized visit teacher on
the compiled UR Lessons versus GW Allies matchup. It does not reopen the
clone-plus-undo or page-COW benchmark matrix.

## Exactness

The selected and legacy full-clone paths completed the fixed authored audit at
seed `1197` with the same 421 decision roots, winner, legal offers, typed
Commands, visit/Q/value outputs, authority and legal hashes, fixed-viewer
projections, event boundaries, RNG continuations, and terminal witness. Their
logical trace digest is
`bf147a47eaa04b6d5feebecf238f061a3dc87c4fa37ef7836729648bcf708935`;
there were zero mismatches, source mutations, viewer exposures, or fallbacks.

Every selected search mutation crossed the structured Command seam. The audit
recorded and reconciled 421 `world` and 7,772 `leaf` applies; this one-traversal
audit did not reach a retained `child` edge, while the focused eight-traversal
test covers and reconciles all three sites. The selected backend rejects direct
`Env.step(index)`, cloning, legacy submission, determinization, and random
playout on guarded branches. Stale policy keys, offers, prompts, revisions,
authority hashes, and legal hashes fail before mutation with unchanged witness
and native apply count.

## Consumer measurements

Fresh spawned workers ran complete authored games. Import, pack compilation,
and one warmup root were excluded; all game decisions and searches were
included. RSS is summed process RSS sampled every 5 ms, so shared pages may be
double-counted; only matched fresh-process rows are compared.

| Cell / driver | decisions/s | traversals/s | p50 | p95 | peak RSS | CPU ms/label | caps / fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| interactive selected | 35.76 | 286.07 | 21.88 ms | 65.99 ms | 228.1 MiB | 27.96 | 0 / 0 |
| interactive reference | 21.81 | 174.50 | 40.63 ms | 102.35 ms | 228.3 MiB | 45.07 | 0 / 0 |
| saturated selected | 4.70 | 601.04 | 607.41 ms | 1,343.29 ms | 951.5 MiB | 572.14 | 0 / 0 |
| saturated reference | 3.40 | 435.50 | 841.48 ms | 1,874.13 ms | 951.2 MiB | 777.56 | 0 / 0 |

The selected path clears every pre-registered absolute and matched-reference
gate. Its interactive and saturated decision throughput are 1.64× and 1.38×
the reference; p95 latency is 0.64× and 0.72× reference; matched peak RSS is
effectively unchanged. Marks and rollbacks are zero by design because PUCT
retains independent full-clone siblings.

The performance difference comes from the explicit native seam: once a typed
Command's match, prompt, revision, offer, and optional audit hashes are
validated, its exact action-aligned offer commits directly through
`Game::apply_offer_submission`. It does not perform the legacy Python JSON
lookup plus indexed step, and it does not take a redundant rollback clone after
all fallible checks have passed.

## Provenance and decision

- Contract: `experiments/contracts/rul-2-selected-branchdriver-teacher-v1.json`
- Raw receipt: `experiments/data/rul-2-selected-branchdriver-teacher-v1.json`
- Contract SHA-256: `fea38c3e5cb3fb6c72813ec2a8680b7581b56f61355698aee78c2ed58d5b68bd`
- Receipt SHA-256: `e5b30be3f0162a134c4d3287aa2105cfed152629d7cfafab40830db2c808f1b5`
- Source closure SHA-256: `24d301266db8e48c5577026474e06bd648db7441c64e6e5051e186d1398428ec`
- Compiled IR SHA-256: `c8bfe15eab35e5953c7a55ba69d83753fa8878b86f9f74e371ec66a233606337`
- Compiled source SHA-256: `f712ff02334d257b792523297f369e728ca3914524304056b029fb46b1b2c290`

The result is `remain`. A split is not justified: both consumers compared here
use full clone, and no second representation ran this contract. The rejected
RUL-1 drivers remain diagnostics only.

Verify from another checkout with:

```bash
uv run --extra dev experiments/runners/run_selected_branchdriver_teacher.py \
  --contract experiments/contracts/rul-2-selected-branchdriver-teacher-v1.json \
  --out experiments/data/rul-2-selected-branchdriver-teacher-v1.json \
  --verify
```
