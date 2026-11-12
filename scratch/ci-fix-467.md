# PR 467 CI repair, 2026-11-12 UTC

Failed published head: `7e2fef9dd4c7d80bb78a6e3615445e19c3357b03`.
`lf sync` completed without changing HEAD. Main's evaluation check passed in
run `43142920822`; no layer-risk runs exist on main.

Evaluation job `130934875244` in run `43143144527` failed
`tests/eval/test_inference_receipts.py::test_cli_never_imports_model_or_native_runtime`:
`manabot/__init__.py` called `find_spec("managym")`, triggering the test's import
blocker. Reproduced locally before editing. Moved the upgrade diagnostic to the
engine compatibility import. Added a real namespace-package failure fixture and
initializer validation guidance. Ruff and 92 tests passed:

```sh
uv run --extra dev --extra artifacts --extra evaluation pytest tests/test_package_boundaries.py tests/eval/test_inference_receipts.py tests/infra/test_layer_risk_receipts.py -q
```

Measurement job `130934692562` in run `43143144534` fails compiling its `before`
engine, pinned to `fb3555ffbd36bb797908110fc549999fe8e1a381`. Preparation copies
current `managym/cards/src/tla.rs` into that historical tree. CI reports missing
`LibraryEnd`, `Effect::search_library`, `CardPredicate::basic_land` and
`ObjectSelector::Selected`. The copied card file and attribution patch are
identical to main. The PR's only harness change locates the renamed Python
extension inside wheels; that executes later and cannot cause this build error.

Local reproduction:

```sh
uv run --no-sync python scripts/measure_layer_risk.py prepare --out .cache/ci467-layer-risk
cargo check --locked -j 2 --manifest-path .cache/ci467-layer-risk/sources/before/managym/Cargo.toml --example bench_steps --example bench_search
```

This exits 101 with two E0433 errors for `LibraryEnd`. Logs are
`/tmp/manabot-467-measure-local.log` and `/tmp/manabot-467-measure.log`.
No timing workload was launched. The prepared archive is only a measurement
input, not a Task worktree. No processes remain running.

Unresolved: the historical comparison requires identical definitions and digests
while retaining the old engine's unmitigated behavior. Replacing BEFORE with
main, omitting failing cards, or disabling the check would change the comparison
or hide it. It needs a compatible control or an explicit revised measurement
scope. Do not rerun this deterministic compiler failure as a flake.

The verified evaluation repair is checkpointed locally, not published or armed.
The existing published head still has auto-merge enabled. This attempt is blocked,
not a completed CI repair.
