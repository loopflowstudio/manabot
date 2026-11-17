# Etude execution comparison

**Not yet measured.** Two rentals on 2026-11-17 ended before any scored GPU
workload (a Vast host went offline; the pinned Etude client could not read
Lium's pod). Both were terminated and confirmed absent; launcher estimates
total about USD 0.13 of the USD 10 cap. The method below stands for a repeat.

Exploratory, not a playing-strength claim. Does execution adoption change
Elemental update throughput on the same GPU, do managed Manabot players batch
on CUDA, and how does a stopped/resumed baseline compare with uninterrupted
learning? Compare main 3afd92d8f3 with this branch, alternating three-update
Elemental runs at seed 197, the recipe's 256 streams, 64 transitions, float32, width 64,
two attention layers and the same resolved current-world settings. Retain
collection/learning seconds and real branch save-call durations. Run a six-update
branch baseline uninterrupted and as two three-update attempts, keeping its
six-update schedule; compare curves and replayed paired-deal scores. Resume
starts fresh games and RNG streams, so bitwise equality is not expected.
Managed evaluation uses each endpoint's immutable checkpoint and 16 paired deals,
retaining actual CUDA forward batch sizes and Etude performance records.
Small samples describe execution, not strength equivalence. Total approved
rental cap USD 10, one reasonable GPU, no synthetic load. Stop and confirm
termination even if setup or measurement fails.

## Attempt — 2026-11-17

The source archives were main
`3afd92d8f3bf48b87d873693580f0d5f8ccf3612` and branch
`ee370d9757b86422db0ff0140d651c37a911de3f`, with this measurement module
overlaid on each. Python source manifests bound the archive bytes; the engine
implementation is unchanged between the revisions (the Rust diff names test
files only). Each archive kept its own locked Etude SDK. Setup used CPython
3.13, Rust 1.98.1, the locked CUDA Torch and a shared Cargo build directory,
with only one environment build at a time and two Cargo build jobs.

Etude rented Vast instance `61090179`, machine
`mtg329-speed-4090-vast-ffe7d7083bd2`, at 15:36 UTC with:

```sh
ETUDE_HOME=/tmp/mtg329-performance-etude doppler run --project etude --config prd -- doppler run --project manabot --config prd -- uv run --extra artifacts etude rent mtg329-speed-4090-vast --hours 2 --max-usd 10
```

The provider confirmed creation at 15:37:03 UTC. SSH became usable after
the image's startup package installation. `nvidia-smi` reported an NVIDIA
GeForce RTX 4090, 24,564 MiB, driver 595.71.05. Source upload, Rust installation
and CUDA dependency downloads completed; native compilation began. At
15:50 UTC the Cargo target held 181 MB, with no completed environment or
training log. At 15:52 UTC SSH disconnected and Vast reported
`actual_status=offline`, `cur_state=running`. The cause is unknown. No Torch
CUDA check, training timing, policy export or managed CUDA forward was
observed. Etude received no bootstrap/worker output; direct SSH observations
do not demonstrate launcher job delivery.

Terminated with `uv run --extra artifacts etude terminate
mtg329-speed-4090-vast-ffe7d7083bd2` in the same isolated/Doppler environment.
At 15:53 UTC Etude listed the machine deleted and Vast's instance lookup
returned absent. The launcher's final retained cost estimate was **~USD 0.12**;
this is not a reconstructed provider bill. No machine remained rented and no
remote workload survived deletion. A fallback query for RunPod NVIDIA RTX
A6000 (4 vCPUs, 24 GB, USD 1/hour ceiling) at 15:53:11 UTC returned no capacity;
no second rental was made. The shared launcher database was untouched.

## Harness checks and limits

The measurement module's actual Etude CLI path was checked locally before
remote training: a two-update CPU baseline at seed 197 stopped after update
one (exit 75) and resumed to update two (exit 0). Both attempts timed their
actual saves. The production managed callback completed two paired deals
(16 replay-checked games) on the existing three-update CPU checkpoint in
7.72 seconds, recording 645 CPU forward calls, all batch size one. This was
a harness preflight, not CUDA batching or a speed comparison. The remote
cohort was increased to 16 paired deals before any remote scored run.

Ruff check/format and `git diff --check` pass. No experiment-specific test
suite or saved schedule was added. Working setup script and source archives
remain in `/tmp/mtg329-gpu`; those paths are temporary, not archival evidence.

## Retry — 2026-11-17, 16:00–16:03 UTC

At branch `f962de12a174a17515692cd69a42a759feaa7b78`, fresh capacity checks
reported no matching RunPod L4, 4090, A6000 or A5000 and no Vast 4090 or
3090. Lowering Vast's minimum to four vCPUs and 24 GB did not find capacity.
Lium's previously unreadable inventory now offered a 4090, 23 vCPUs and
58 GB at an admitted USD 0.41/hour. Requested the same two-hour window,
with the retry cap reduced to USD 9:

```sh
ETUDE_HOME=/tmp/mtg329-performance-etude doppler run --project etude --config prd -- doppler run --project manabot --config prd -- uv run --extra artifacts etude rent mtg329-speed-4090-lium --hours 2 --max-usd 9
```

Lium created pod `55297a92-fa2b-47dd-9c73-b327672e74ca`, named
`etude-7b0c01e641d4`, at `2026-11-17T16:01:32.886331` (the provider's exact
timezone-naive value). Etude's pinned `Pod.created_at: AwareDatetime` rejected
that response (`timezone_aware`). Its next inventory read rejected the same
field. The launcher therefore retained machine
`mtg329-speed-4090-lium-7b0c01e641d4` as starting, with no provider id or
receipt; a subsequent `tick()` could not reconcile it. No SSH address,
bootstrap, uploaded source, CUDA check or scored workload was observed.
No Manabot or SDK workaround was added.

`etude terminate mtg329-speed-4090-lium-7b0c01e641d4` failed with
`Lium create already attempted; awaiting reconciliation`. The authenticated
Lium API accepted deletion of this exact pod. Its inventory then excluded
the pod; its individual lookup briefly reported `DELETING`, then returned
404/absent by 16:03 UTC. Termination is confirmed at the provider. The isolated
launcher still has the unresolved intent and must not be treated as a live
pod or a reconciled deletion. No launcher/server loop was left running.

The launcher's retained estimate at 16:03 UTC was **~USD 0.01**, not a final
settlement: it cannot reconcile this deleted pod with the pinned client.
A retry needs a provider allocation whose lifecycle the client can read, or an
Etude fix for the actual Lium timestamps and interrupted-creation recovery.
