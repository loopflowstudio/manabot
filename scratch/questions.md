# MTG-184 unresolved choices — 2026-10-24

Current design and accepted direction: `scratch/llm-finetune.md`. This checkout
owns representation; MTG-185 owns the harness. Jack Heart requested both Task
sessions in cmux, then “launch pursue” after concept review on 2026-10-24.
No model, Wave or paid allocation is selected.

- **Accepted revision, 2026-10-25:** Jack Heart accepted fixing the ordinary
  policy's missing continuation input through a shared native observation,
  requiring no observation-encoding performance regression and moving work to
  Rust as needed. The local v2 revision implements the bounded native/model
  path, removes duplicate observation models and preserves legacy admission.
  Focused collision/parity/model and checkpoint proofs pass. Native and vector
  benchmark receipts live under `.runs/continuation-revision/`; the heavily
  loaded host cannot establish nonregression. Gate and quiet-host performance
  acceptance remain before publication.

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
  6 and world identity remain unchanged. Performance acceptance remains open.

- **Audit limit, 2026-10-25:** native `Decision::ManaChoice` retains the exact
  output menu and resumed payment, but discards its activation source after
  costs commit. The shared frames expose the outputs and mark the source
  unavailable; they do not infer it from tapped permanents or history. Retaining
  a new rules-state source identity would need separate frozen-world design.
