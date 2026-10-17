# Prebuilt Etude client startup canary

Frozen before admission, 2026-10-17. This is infrastructure validation, not a
model-size or playing-strength result. The five-shape batching grid stays intact.

Replacement `6bef7648d5e4` failed inside the SDK's 120-second client installer;
retained stderr ended at building the local `etude-sdk` package. Replacement
`394601fdf1a5` reached the job, then exhausted the separate 600-second project
setup deadline while downloading CUDA dependencies. Neither reached training.

Use the exact installed Etude SDK 0.1.1 source, packaged as a platform-independent
wheel. Verify every Python file and the forwarding pyproject/lock byte-for-byte.
Install runtime dependencies from the unchanged lock with `--no-install-project`,
then install the wheel offline without resolving dependencies. Both commands
share the original 120-second total. Guardian bytes, its financial stop, and
the five-minute pre-start cutoff remain unchanged. Enable this process-local
adapter before the existing startup diagnostics.

Retained local artifact identities:

- Original SDK archive SHA256: `48b54dfe2211c2dec606c108fb2547e268e6b46bd33e00ccce2aeab404cf5b52`.
- Pure Python wheel SHA256: `313bb3f4ff214f970e4b23985fa1a2eb1749613580a4ef599a526ce9338a82f1`.
- Prepared bundle SHA256: `f51c21b3eaf1d2ef290ff146cb92663066b4bdbeabd8e45889b1593421f9eeaa`.

Admit at most one RTX 5090, one attempt, $1 cap, 30-minute job limit and
40-minute rental bound, quoted at no more than $1.25/hour. Reserve it from the
existing inclusive $50 ledger; retain the $2 ancillary reserve. No automatic
retry or scientific-cell admission follows this canary.

Keep scientific source `4d65d477d8e7bc47a327464feea7ace08aae07da`, using its
already verified history-excluded source projection. Embed
`experiments/runners/client_install_canary.py` in the immutable job command and
retain its SHA256 and launcher commit separately. The wrapper uses the original
Worker and two-update small-model canary, seed 13599. Only project setup receives
1,200 seconds instead of 600; CUDA admission gets 60, training 180, and retained
outputs keep the existing 240-second reserve inside the outer deadline.

Capture provider logs continuously before deletion. Success requires both
installer phases, CUDA admission, exactly two completed updates, all study
manifest hashes, a loadable finite checkpoint, and provider-confirmed deletion.
Record failed phases and settle conservative cost even on failure. A successful
installer alone does not establish complete GPU readiness. This experiment
cannot establish the cause or resolution of earlier CUDA/NVML failures.
