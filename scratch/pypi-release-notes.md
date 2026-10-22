# MTG-178 release preparation

Scope: versioned PyPI release workflow and reviewable PR in this Task checkout.
No merge, tag, production upload or running experiment/controller changes.

- Synced main through `lf sync --manual`: #349 is merged (845a1e350), using Etude SDK 0.2.0. No migration code was duplicated.
- Planned first pair: manabot/managym 0.2.0, CPython 3.12; Linux x86-64 manylinux_2_28, macOS Apple Silicon.
- PyPI JSON endpoints both returned 404 on 2026-10-21; no published versions visible. Account ownership/pending publishers remain unverified. GitHub `pypi` environment API returned 404; owner must configure both publishers and a protected environment before tagging.
- Release staging preserves managym/content relative paths, adds public Torch metadata while retaining source development CPU/CUDA groups, and excludes experiment drivers. Published source archives rebuild directly.
- Native bindings explicitly use pyo3; automatic pyo3/bin detection previously produced a wheel without the extension. Clean import smoke caught this before publication.
- Report import smoke exposed historical experiment imports through shared statistics; extracted identical bootstrap/valid-game logic to study_statistics, preserving old analysis imports/reexports and frozen experiment protocols.
- Local builds produced both sdists and Apple Silicon wheels from those sdists. Strict twine and archive checks pass. Clean external installs load semantic catalogs and perform a CPU update with step-0 and step-32 checkpoints. Optional Academy/reports/artifacts/evaluation imports pass after the stats extraction.
- Isolated remote regression suite: 179 passed, 2 skipped. Full cargo test with debug assertions passed (opt-level 1). Actionlint passed. Release tooling tests: 6 passed.
- Linux wheel/installation and the full workflow remain CI proof, not local GPU or Linux claims. First upload and post-upload registry-only installation remain after review/merge and publisher setup.
