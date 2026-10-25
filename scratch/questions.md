# MTG-184 unresolved choices — 2026-10-24

Current design and accepted direction: `scratch/llm-finetune.md`. This checkout
owns representation; MTG-185 owns the harness. Jack Heart requested both Task
sessions in cmux, then “launch pursue” after concept review on 2026-10-24.
No model, Wave or paid allocation is selected.

- **Shared input contract:** after Jack Heart requested getting this ready to
  demo, the implementation selects a separately versioned native capture of
  observation, continuation, definitions and optional public history, taken
  atomically and hashed with world/content identity. Preserve schema-6 evidence;
  reuse existing fact owners. This resolves the engineering choice exposed by
  `scratch/observation_collision.py`; the counterexample now passes with distinct capture digests and text. The demo uses declared public history v1 up to 32 records and no writable
  model memory. Renderer and local demo are implemented; broader verification belongs to parent gate.
- **Comparison configuration:** select the trained comparator and history
  treatment. Jack Heart directed matching non-LLM input and excluded writable
  memory for the untrained model. The proposed bounded public-history treatment
  and previous-state alternative remain unselected; neither implies full parity.
- **Model and measurements:** model/tokenizer, prompt budget, corpus/world,
  comparator artifact, hardware, sample sizes, performance targets and allocation
  remain open. Qwen3-4B-Instruct-2507 is a candidate, not measured single-L4 fit.
  Jack Heart requested performance and size comparisons; no benchmark ran.
- **Interface fitting:** measure the untouched model first. The fitting objective
  is open: random choices teach randomness and expert choices teach strategy.
  The accepted exact-reference, single-repair, seeded-uniform fallback protocol
  is specified once in the design, including its reporting and failure limits.

Historical notes and uncommitted-input preservation paths are recorded once in
the design's **Delete — do not maintain** section. Resolving the sync conflict
preserved these MTG-184 questions; the shared-input engineering choice is resolved.

- **Reversible implementation bounds:** capture v1 defaults to history off;
  demo explicitly uses public history v1 with 32 rows. The renderer has a 262,144
  byte admission ceiling and a complete-prompt tokenizer callback; this byte
  ceiling is an engineering guard, not a selected model context budget. Prompt
  token counting must include the harness chat template before inference.
