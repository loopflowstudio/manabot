# Prove Exact Study Source Return and Sibling Isolation

## Problem

The first RUL-4 serial PR made a historical, viewer-safe replay decision
executable through an authority-private `StudyForkProvider`, but its return
value proves only the recorded replay row. It does not identify the retained
managym root from which the ephemeral branch was cloned. Study therefore
cannot demonstrate in one return action that branch work left the exact source
authority unchanged.

This serial PR closes that seam for the human-facing Study consumer without
reopening the representation decision. It keeps compact full clone as the one
Study branch driver, binds every retained root to a canonical source digest,
and proves that one branch can execute a normal structured command while its
source and sibling remain exact. This advances the Rules measure that a
historical `StudyIdentity` can execute normal commands and return to the
identical source hash, cursor, offer, and viewer projection.

## The demo

Run `uv run pytest tests/etude/test_study_branch.py -q`. The focused proof
forks two branches at one completed-match decision, executes a structured
priority command on one, shows the sibling and recorded source did not move,
then returns both branches with the same 64-character source digest and exact
recorded frame, offer, command, cursor, and continuation.

## Approach

Add an authority-private `StudyReturnReceipt` that extends the existing
`RestoredReplayDecision` with `source_digest`. When `StudyForkProvider` is
built, deep-copy the canonical replay as before and retain each root together
with its then-current `Env.state_digest()`. Keep `root.clone_env()` as the only
fork operation.

Before every fork, compare the live retained root to its captured digest and
viewer. Before every return, repeat the digest comparison against the retained
source—not the now-mutated branch. A mismatch fails closed. Return constructs
the receipt from a deep copy of the canonical restored decision, adds the
captured digest, and consumes all branch references whether return succeeds or
detects drift.

Strengthen the completed-match integration proof around one historical hero
decision:

1. Capture a baseline return receipt, canonical replay bytes, and presentation
   events.
2. Fork two siblings from the same retained root and compare their complete
   structured-offer projections.
3. Submit one legal `pass_priority` command through the normal structured
   executor on one branch, then prove the other sibling's offer projection is
   unchanged and remains viewer-safe.
4. Return both branches and a fresh post-return fork. Compare their source
   digests and every recorded return component with `restore_decision`.
5. Prove return is consuming and the saved canonical replay and presentation
   events did not change.
6. In a separate authority-private test, mutate the retained root after a
   branch has forked. Prove the open branch's return and every later fork fail
   closed, the failed return emits no receipt, and the branch is consumed.

## De-risking

| Question | Finding | Impact on design |
|----------|---------|------------------|
| Does the selected representation need to change for Study? | No. `Env.clone_env()` delegates to the existing full-clone fork, which independently copies mutable game state while immutable content definitions remain shared. The approved RUL-4 review retained this representation. | Keep `StudyForkProvider -> root.clone_env()` as the only branch path; add no page-COW, undo, or second driver. |
| Is there already a canonical source witness available to Python? | Yes. `Env.state_digest()` hashes canonical mutable `GameState` facts—including RNG, events, object incarnations, allocation watermark, and content digest—plus the current action space, pending choice, and skip-trivial state. | Capture this digest once per retained root and return it; do not invent a replay hash or duplicate serialization in Etude. |
| Is the Study digest interchangeable with the search BranchDriver admission witness? | No. The search authority witness additionally binds its private `decision_epoch`; `state_digest()` deliberately omits that epoch so structured and legacy paths can compare semantic states. Study's external frame revision/prompt/offer identity and the cloned root retain the structured-command binding, while the source receipt detects mutations available through the retained Python authority boundary. | Name the field `source_digest`, keep it local to `etude.study_branch`, and do not claim it replaces search precondition evidence or add a new Rust binding in this serial slice. |
| Can returning accidentally report the mutated branch as the source? | The branch and retained root are distinct `managym.Env` objects. The provider can retain the root reference and its captured digest separately from the clone. | Hash `_source` at return, never `_env`; clear both references after return. |
| Can a failed return leak a plausible receipt or leave a usable branch? | Without explicit close-on-error, drift could be retryable and the branch would remain live. The retained root is authority-private but available to a focused integration test, so this behavior can be proved without a production mutation API. | Mutate the retained test root after forking; on source drift, close first and then raise `StudyBranchUnavailableError`. Assert no receipt, no later fork, and no branch reuse. |
| Does the return receipt require a public protocol or persisted replay migration? | No. `RestoredReplayDecision` is already the authority-private return object, and canonical replay rows remain the durable record. | Subclass it inside `etude.study_branch`; leave Pydantic protocol schemas, replay storage, and frontend types unchanged. |
| Does sibling equality prove source isolation by itself? | No. Equal sibling offers could coexist with mutation of the retained replay or presentation trace. | Assert sibling offer stability, identical source digests across baseline/branch/sibling/fresh returns, and byte-for-byte canonical replay plus presentation-event stability. |
| Does the proposed validation exercise the debug configuration CI uses? | Yes. `cargo test --manifest-path managym/Cargo.toml --test search_state_contract full_clone_passes_the_representation_neutral_branch_contract -- --exact` runs the existing representation-neutral full-clone contract in Rust's debug profile. The current branch also passes the focused Etude test. | Make both commands landing gates; no extension rebuild is needed because this serial PR changes no Rust. |

## Alternatives considered

| Approach | Tradeoff | Why not |
|----------|----------|---------|
| Store the authority digest in every canonical replay row | Makes the digest durable outside the process, but changes the replay schema and creates a second persisted authority claim that must be migrated and verified. | The requested return receipt is runtime authority-private; persisted replay remains presentation/recovery evidence, not restorable managym truth. |
| Expose the full search-state witness through PyO3 | Would include the private decision epoch and several diagnostic projections, but requires Rust/API work and couples an interactive return receipt to benchmark/search admission machinery. | The retained Study root is not a generic search state. The existing semantic digest covers the mutations possible at this boundary, and the approved review explicitly asks for the smallest source-digest receipt. |
| Hash the branch on return | Provides evidence about the exploratory state after commands. | Study return must identify the untouched recorded source; a changed branch digest is expected and answers the wrong question. |
| Reconstruct authority from canonical replay on every fork | Avoids retaining roots but would require replay to become rules truth and re-execute hidden state, RNG, and engine events. | This violates the existing authority split and the directive not to create replay truth or a second branch representation. |

## Key decisions

- Compact full clone remains the production Study representation. The
  successful shape is intentionally boring: retained root, independent clone,
  normal structured execution, consuming return.
- `source_digest` is a lowercase 64-hex managym semantic digest. It names the
  exact retained Study source closure used by this adapter, not the replay
  payload and not a portable search witness.
- Root drift is checked at provider fork and branch return. There is no useful
  recovery from a mutated retained authority; failure is explicit and the
  branch is consumed.
- Exact return means structural equality for the canonical viewer-safe frame,
  selected offer, revision-bound command, presentation cursor, and semantic
  continuation, plus equality of the retained source digest.
- The integration test uses one normal structured command and one concurrently
  retained sibling. A fresh post-return fork demonstrates that neither branch
  poisoned the provider's root.
- The likely successful product behavior is instant, trustworthy exploration:
  a player can try a line and return without a reload or replay reconstruction.
  The failure mode to prevent is subtler than a crash—returning a visually
  familiar frame whose hidden authority or sibling was mutated. Digest and
  isolation assertions make that silent failure impossible at this boundary.

## Scope

- In scope: authority-private Etude adapter, canonical managym source digest,
  fork-time and return-time root drift checks, consuming exact return receipt,
  one executed structured command, one sibling, one fresh post-return fork,
  a test-only retained-root drift proof, viewer-safety assertions, and retained
  replay/presentation immutability.
- Out of scope: page COW, undo, new Rust or PyO3 APIs, frontend/UI, public
  protocol or replay-schema changes, landmark ranking, saved branches,
  replay-as-rules truth, alternate viewers, production root-corruption hooks,
  stale/pack/nested stress outside this approved serial review fix, and new
  latency/RSS evidence.

## Done when

- `StudyReturnReceipt.source_digest` is present and schema-validated as 64
  lowercase hex characters.
- Fork and return fail closed when the retained root no longer matches its
  captured digest; a focused mutation test proves that a failed return emits
  no receipt and that failed or successful returns cannot be reused.
- The focused integration proof shows identical source digest before branch
  work, after executing one sibling, after both sibling returns, and from a
  fresh fork.
- The executed sibling cannot change the other sibling's structured offers,
  the recorded viewer-safe frame, offer, revision-bound command, presentation
  cursor, continuation, canonical replay, or presentation events.
- `uv run pytest tests/etude/test_study_branch.py` passes.
- `cargo test --manifest-path managym/Cargo.toml --test search_state_contract full_clone_passes_the_representation_neutral_branch_contract -- --exact`
  passes in debug.

## Measure

This serial proof does not select or tune a representation, so it introduces
no new quantitative target. Full-clone latency/RSS was measured during RUL-1;
interactive fork/apply/return budgets remain a later production-integration
receipt. The outcome here is binary: zero source, sibling, replay, projection,
or cursor drift across the focused completed-match proof.
