# MTG-271 direction and implementation choices

The Task brief preserves the originating direction: "we should make etude-sdk a >=" and "can wem akea torch 2.10 a >= also?" Public requirements remain minimums without speculative upper bounds.

Jack Heart, comment 3a7bd499 (2026-11-11): "i dont actually want this to do 2.1, i will do that in a future task". Correction in comment 068206ea: "i am fine with putting this stuff in here though -- prepping for us to 2.1 well and soon is still good". Current scope keeps version 0.2.0 and version-generic docs, and includes GitHub completion plus digest-checked retries. No tagging, publication or backfill.

- Preserve nbformat in core: ExperimentExecution.report creates a notebook during ordinary execution. Remove unused direct pandas; no source consumers were found. Notebook execution stays in its existing extra.
- Preserve frozen experiment/setup scripts, historical receipts and notebook metadata. Fresh shared machine setup and active development/CI use 3.13. Deployed 3.12 images remain supported consumers.
- uv 0.9.22: an unconditional core Torch requirement conflicts with the existing per-group indexes. Removing the non-Linux source let lock succeed but selected CUDA on macOS. Splitting identical public minimums into Linux/non-Linux markers resolves and installs correctly. Compression keeps this split in core; `llm` inherits it, verified by isolated locked syncs.
- Inherited b015d5/C2 notes remain in 757bb6257ef07cf2536ad7f7ec24f0f7309628c6:scratch/questions.md; they do not authorize experiments here.
