# Retain GPU creation diagnostics

MTG-121's first custom-image attempt has an unresolved provider-creation outcome.
The Etude SDK discarded its creation exception, so the retained evidence cannot
distinguish an HTTP failure from a transport failure. This slice preserves the
exception type and an allowlisted provider summary before rethrowing the original
exception. It neither retries creation nor changes rental ownership or deadlines.

This independent slice comes from MTG-121's existing implementation and tests.
The runtime image and matched GPU comparison remain in that Task's checkout;
no startup improvement or reconstruction of the missing error is claimed here.
The source checkout and live recovery controller remain untouched.

Done when the standalone launch-guard tests establish safe error retention,
single-call behavior, original-exception propagation and restored API ownership,
then the focused PR passes required CI and is verified merged. The user has
authorized landing necessary MLops fixes. This PR does not complete MTG-121.

Standalone verification: a fresh worktree and its own Python 3.12 environment
passed all 12 launch-guard tests with `uv run --frozen --extra dev --extra reports
python -m pytest tests/training/test_etude_launch_guard.py -q`. Changed-file Ruff
and `git diff --check` also passed. Required CI, including debug Rust tests,
remains to be verified on the published commit.
