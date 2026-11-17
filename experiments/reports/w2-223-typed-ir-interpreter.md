# W2-223 two-deck typed-IR admission and interpreter proof

Closes the Semantic Programs and Choice ABI KR1 on top of the admitted IR from
W2-187 (`content/semantic/v1`) and the learning projection from W2-215. This
task adds the missing runtime piece: a generic interpreter that **consumes** the
checked-in typed programs by opcode, never by card name.

## What was already true (W2-187)

- `content/semantic/v1/two_deck.source.json` admits all 31 reviewed definitions
  (26 distinct nonland cards + 4 basics + Ally/Clue tokens) for the exact UR
  Lessons versus GW Allies acceptance slice.
- The offline compiler lowers named operations to 23 stable numeric opcodes and
  writes canonical `generated/two_deck.ir.json` plus a machine-checked coverage
  report (`generated/two_deck.coverage.json`, `admission_closure_complete:
  true`, `no_card_name_dispatch: true`).
- `managym/tests/semantic_ir_tests.rs` validates the IR *document*: binding,
  characteristic parity, and the absence of name-based dispatch fields.

## What W2-223 adds

A generic Rust interpreter, `managym::semantic`, that:

1. Parses the versioned checked-in IR into typed `Step`/`Condition`/`Predicate`
   values. Parsing is fail-closed: any instruction carrying a `card_name`,
   `registry_name`, or `definition_ref` field is rejected as name-based
   dispatch, and any unknown opcode stops the load.
2. Binds every semantic definition **once** to a real `ContentPack`
   `CardDefId`. A definition that does not resolve fails binding
   (`IrError::UnadmittedDefinition`), so a dropped admission cannot pass
   silently. After binding, execution carries typed ids and numeric opcodes
   only.
3. Executes admitted programs with a pure `match` over typed steps keyed on the
   numeric opcode, resolving `create_token` definition indexes through the
   bound ids and evaluating branches / per-target expansion through an
   injected `InterpreterContext`. The result is a deterministic `TraceEvent`
   sequence.

The interpreter is additive. It does not replace the live effect resolver in
`managym::flow`, so ContentPack binding, rules behaviour, the learning
projection, and the legacy/structured choice ABIs are all preserved. Runtime
differential parity between this trace and the legacy effect path remains a
documented future step (`content/semantic/README.md`).

## Proof (`managym/tests/semantic_interpreter_tests.rs`)

Source checks (fail closed):

- `source_admits_every_deck_definition` — all 31 definitions and every
  deck-referenced index bind to a real card.
- `unadmitted_definition_fails_binding_closed` — a definition bound to a
  nonexistent card fails.
- `name_based_instruction_is_rejected` — an instruction with a `card_name`
  field fails to parse.
- `checked_in_ir_carries_only_typed_numeric_opcodes` — the whole checked-in IR
  lowers to typed steps (37 programs, typed branch and for-each arms present).
- `every_admitted_program_runs_by_opcode` — all 37 programs execute without a
  card-name fallback.

Happy-path and interaction traces (pinned effect sequences):

- basic land mana, Divide by Zero (bounce → learn), Forecasting Fortune Teller
  (bound Clue token), Firebending Lesson kicker branch (2 vs 5), Accumulate
  Wisdom graveyard branch, Fancy Footwork per-target expansion, South Pole
  Voyager second-arrival draw, Yip Yip! ally-only flying grant, and the
  canonical Water Tribe Rallier waterbend → Badgermole Cub `{G}` interaction.

## Validation

Headless, proportional to the change surface:

```bash
cd managym
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test                       # 263 tests, incl. 14 new interpreter tests

# offline artifacts unchanged and still consistent
uv run scripts/compile_semantic_content.py --check
uv run scripts/generate_coverage_gaps.py --check
```

All green. The IR, coverage report, and content pack are unchanged by this task;
only the Rust interpreter and its tests are new.
