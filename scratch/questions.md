# MTG-184 unresolved choices — 2026-10-24

Current design and accepted direction: `scratch/llm-finetune.md`. This checkout
owns representation; MTG-185 owns the harness. Jack Heart requested both Task
sessions in cmux, then “launch pursue” after concept review on 2026-10-24.
No model or Wave is selected. The separately authorized Runpod performance test
is complete; its receipts and revised delivery acceptance are recorded below.

- **Accepted revision, 2026-10-25:** Jack Heart accepted fixing the ordinary
  policy's missing continuation input through a shared native observation,
  requiring no observation-encoding performance regression and moving work to
  Rust as needed. The local v2 revision implements the bounded native/model
  path, removes duplicate observation models and preserves legacy admission.
  Focused collision/parity/model and checkpoint proofs pass. Native and vector
  benchmark receipts under `.runs/continuation-revision/` are historical only.
  Gate completed the 200-game soak and affected correctness suites, repaired
  full-vector attribution/coverage and collected three fresh serial pairs.
  The local performance attempt was blocked: measured host load was
  84–113 on 16 CPUs, with unresolved batch-64 large-action latency signals.
  The completed Runpod comparison supersedes that acceptance blocker under
  Jack Heart's 2026-10-26 shipping direction. Exact results and identities are
  in the design's final gate evidence; small measured overheads remain disclosed.

- **Shared input contract:** after Jack Heart requested getting this ready to
  demo, the implementation selects a separately versioned native capture of
  observation, continuation, definitions and optional public history, taken
  atomically and hashed with world/content identity. Preserve schema-6 evidence;
  reuse existing fact owners. This resolves the engineering choice exposed by
  the collision proof (`e929d4c7c:scratch/observation_collision.py`); the counterexample now passes with distinct capture digests and text. The demo uses declared public history v1 up to 32 records and no writable
  model memory. Renderer and local demo are implemented; broader verification belongs to parent gate.
- **Comparison configuration:** select the trained comparator and history
  treatment. Jack Heart directed matching non-LLM input and excluded writable
  memory for the untrained model. The proposed bounded public-history treatment
  and previous-state alternative remain unselected; neither implies full parity.
- **Model and measurements:** model/tokenizer, prompt budget, corpus/world,
  comparator artifact, hardware, sample sizes, performance targets and allocation
  remain open. Qwen3-4B-Instruct-2507 is a candidate, not measured single-L4 fit.
  Jack Heart requested performance and size comparisons; no model-inference or playing-strength benchmark ran.
- **Interface fitting:** measure the untouched model first. The fitting objective
  is open: random choices teach randomness and expert choices teach strategy.
  The accepted exact-reference, single-repair, seeded-uniform fallback protocol
  is specified once in the design, including its reporting and failure limits.

Historical notes and uncommitted-input preservation paths are recorded once in
the design's **Delete — do not maintain** section. Resolving the sync conflict
preserved these MTG-184 questions; the shared-input engineering choice is resolved.

- **Reversible implementation bounds:** capture defaults to history off;
  demo explicitly uses public history v1 with 32 rows. The renderer has a 262,144
  byte admission ceiling and a complete-prompt tokenizer callback; this byte
  ceiling is an engineering guard, not a selected model context budget. Prompt
  token counting must include the harness chat template before inference.

- **Implementation layout assumption, 2026-10-25:** policy input v2 adds a
  bounded 64-row native relation table (32 float32 columns). Frames retain stack
  order and interrupted-payment parents; relation rows retain target order and
  clauses, exact visible object links, definition identity, mode/ability, kicker,
  sacrifice and typed remaining costs. Capacity overflow rejects input. Existing
  checkpoint configs missing a version explicitly load v1; fresh construction
  defaults to v2. The model embeds relation selectors/references and combines
  their context with both object and action embeddings. Frozen semantic schema
  6 and world identity remain unchanged. Runpod performance meets the revised
  shipping acceptance; this does not establish zero overhead.

- **Audit limit, 2026-10-25:** native `Decision::ManaChoice` retains the exact
  output menu and resumed payment, but discards its activation source after
  costs commit. The shared frames expose the outputs and mark the source
  unavailable; they do not infer it from tapped permanents or history. Retaining
  a new rules-state source identity would need separate frozen-world design.

- **Gate measurement protocol, 2026-10-26:** use 128 independently restored,
  real one-command vector steps per family/batch/arm, after 20 warmups, in three
  alternating serial pairs. This replaces the passive/action-zero trajectory
  as the acceptance corpus so priority, target, additional-cost and 33-action
  roots are provably matched. Root restoration, receipt hashing and allocation
  profiling are untimed. No model inference or whole-game throughput claim.
  Allocation scopes are explicit: Rust requests, Torch CPU tensor events and
  Python traced peak/live bytes; byte attribution is logical output/copy size,
  not physical memory traffic. Ordinary wheels omit benchmark instrumentation.
- The final pre-timing corpus audit adds a fifth vector case: the last combat
  priority pass returns a 33-action postcombat-main observation. The existing
  33-action input case remains. This distinguishes large offer/capture overhead
  from actual large-output encoding; all other sample/warmup/pair bounds remain.

- **Delivery direction, 2026-10-26:** Jack Heart approved shipping, authorized up to $2 for a Runpod performance test because this host remains overloaded, and said to ship if it looks normal-ish. Use a stable matched comparison to detect clear repeatable regression; exact timing equality is not required. Retain measurements and terminate the rental.
