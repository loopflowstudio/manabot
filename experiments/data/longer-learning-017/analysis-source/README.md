# Protocol017 analysis source archive

This is a code archive, not a complete study dataset or a standalone executable.
The manifest records each file's original and archived paths, byte count and SHA256. The final
analysis receipts separately bind the programs actually used to produce results.
Snapshots are immutable; changed programs receive another contents-derived path.
Archived paths omit the leading `.runs` directory so normal Git ignore rules do
not discard the source when a report package is published. Restore the original
path from each manifest entry when reproducing the analysis.

## Required inputs

Use the pinned repository runtime from source
`3c3681d2b26a7fc8167928fa63fe24a6cd3d058a` and its locked environment. Restore the
archived files to their original relative paths in an isolated checkout, preserving
any newer work. Restore the retained data at the paths named in the analysis and
allocation: protocol017 training directories, allocation/frozen inputs, complete
evaluation tree and manifest, resolved recipes and resource receipts.

The verifier also reads the locked-wheel upload receipt. Runtime admission reads
the retained route012 baseline monitor and the short-screen small-four-stream
initial checkpoint, plus `learning-017/evaluation-runtime.json`. The exact CPU
binary/source/content identities and Python/Torch/NumPy versions must match;
rebuilding from a commit alone does not prove byte-identical native runtime.
The readiness scripts additionally read the named earlier screen013 datasets.

## Ordinary offline checks

From that restored repository root, the selector/manifest invariants are checked by:

```sh
uv run --no-sync pytest .runs/grid-007/test_learning_analysis.py -q
```

To audit existing retained game evidence and apply the frozen selection rule:

```sh
uv run --no-sync python .runs/grid-007/analyze_learning.py
```

The analysis consumes saved games; it does not launch training or replay missing
panels. The report renderer consumes the verified analysis and resources. Rendering
also requires a matching current source-archive receipt; it refuses to overwrite
an existing report directory. Inspect any existing report before replacing it.

`evaluate_learning.py` executes the declared CPU arena panel and is not needed to
inspect already retained results. It rejects an existing admission; do not remove
that guard to rerun a failed panel. `verify_learning.py` calls the earlier numerical
audit through `verify_grouped.py`, which imports definitions with AST selection.
Do not execute `audit_completed_numerics.py` directly: its top-level historical
driver writes an older aggregate.

`learning_resources.py` reads existing SDK rental records only after full retention
and deletion settlement. Its configuration helper expects the operator's existing
credential environment and launcher history, which are intentionally absent from
this source archive. It is unnecessary for rendering an already retained resource
receipt. No credential file, SDK state, wheel archive or paid-run launcher is copied.

The full result still needs retained data, successful checks and scientific/visual
review. Merely archiving these files does not establish any training outcome.
