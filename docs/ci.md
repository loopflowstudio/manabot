# CI selection

The CI workflow always reports `Tests Result`. Its selector reads the complete
local Git diff with NUL-delimited paths, including both sides of renames, and
selects jobs independently on pull requests and pushes to main. Manual dispatch
runs every job. An empty diff or unknown path also runs everything.

Typical selections:

| Change | Selected checks |
| --- | --- |
| Academy component or browser test | UI unit, type/build and focused browser checks |
| UI dependency, build config or asset script | UI checks and clean-machine launch |
| Academy server, presentation or demo | Academy integration, protocol and UI checks |
| Training, remote worker or experiment | Manabot integration |
| Semantic compiler | Manabot integration and semantic coverage |
| Website | Python unit and website routing checks |
| Documentation | Detection and aggregate only |
| Native engine, shared Python dependencies or CI selector | All checks |

Rules, protocol evidence and immutable packs select their downstream consumers.
Runtime inputs such as experiment Markdown and benchmark data are not treated
as prose-only changes. The dependency table lives in
`.github/ci/select-tests.sh`; `tests/infra/test_ci_selection.py` exercises its
boundaries and the aggregate's rejection of missing selections or skipped
required jobs. Update those cases when adding a package or changing a consumer.

The full-match screenshot matrix has been removed. Focused release browser
checks run with `npm --prefix academy/ui run test:e2e:release`; no reference-image
regeneration is required. Nightly macOS integration remains independent.
