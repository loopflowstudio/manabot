# MTG-121 bounded runtime startup comparison

Prepared 2026-10-17, before any MTG-121 rental. The separate authorization is
$5 total including failures. Reserve $1 for uncertainty and at most four $1
attempts, one at a time. Use an isolated Etude state directory, storage prefix,
controller and budget ledger. MTG-135/137 resources and controllers are excluded.

Compare the existing verified offline-bundle setup on the stock image against
[the lock-bound runtime image](../../docs/gpu-runtime-image.md). Preserve scientific
source `4d65d477d8e7bc47a327464feea7ace08aae07da`, seed 13599, the small-model
size-frontier two-update canary, source projection, all 94 wheel identities and
the source-identical prebuilt Etude client. The original bundle is SHA256
`a53386019a7f1bf557a1752b7492ca825114c526f4c16e1f904d78e0d10aa276`.
Do not change W&B or remove dependencies in this comparison.

Freeze each launcher command, image digest, runtime manifest hash, source,
client and helper identities before admission. Alternate baseline/candidate,
then repeat only after diagnosing any failure. Inspect complete fresh provider
inventory and require fewer than two existing pods. Require an available RTX 5090
quote at most $1.25/hour; reserve the attempt before queueing. Preserve the launch
freshness/idle-sleep guard, 300-second pre-start cutoff, 30-minute job, 36-minute
rental, 1,200-second setup, 60-second CUDA admission, 180-second training and
240-second retention reserve. Never extend deadlines after admission.

Record allocation-to-entrypoint (including acquisition), client installation,
source delivery, dependency preparation, source installation/native build, CUDA
admission, first update and recovered-checkpoint time. Retain provider logs before
deleting the pod and all failures. Each attempt needs provider-confirmed absence,
a conservative cost settlement, complete output manifests, two completed updates,
a finite checkpoint loaded through the ordinary checkpoint loader, and source,
lock, Python/CUDA/toolchain and checkpoint identities.

Classify image acquisition as cached, demonstrably uncached, or unknown from
provider evidence. Compare total allocation-to-first-update and allocation-to-
recovered-checkpoint times; setup-only savings do not establish an improvement.
Two repetitions are operational evidence, not a population-level performance
claim. Historical canary016 remains MTG-135's paid evidence, never a charge on
this ledger. Historical CUDA/NVML and numerical failures remain unresolved.
