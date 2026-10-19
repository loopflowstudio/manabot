Jack Heart renewed this request on 2026-10-17 in MTG-135: optimize startup time for GPU training, considering a maintained custom image and/or fewer dependencies, and launch an autonomous Task worker now. He authorized a separate maximum $5 of GPU testing, including failed attempts. This is the existing prebuilt-image Task, reused rather than duplicated.

Problem and beneficiary
Researchers renting GPUs repeatedly spend paid time acquiring the training runtime before the first update. Historical MTG-114 deployments installed about 7 GB of packages and sometimes spent about ten minutes in setup. Current MTG-135 evidence identifies several separate startup phases, so measure the whole path and do not attribute every startup failure to dependencies.

Desired outcome
A reproducible, materially faster cold-start path from rental allocation to useful manabot training, with measured costs and one documented way to rebuild and deploy the runtime. Evaluate a lock-bound prebuilt image and reducing the runtime dependency set; choose the simplest effective implementation from evidence. These are candidate mechanisms, not a requirement to ship both. Preserve scientific behavior and honest negative results.

Acceptance
- Establish a phase-timed baseline and an improved run on real rented hardware: allocation/image pull, client bootstrap, source transfer, dependency preparation, native build, CUDA admission, first update, and a recovered valid checkpoint. Include failure time and total paid cost. Distinguish fresh-host image pulls from cached starts; do not claim an uncached test without evidence.
- Pin source, Python, CUDA/runtime, dependency lock or its verified training projection, image digest and checkpoint identities. Refuse incompatible images or missing dependencies before training. Validate isolated installation so the local developer environment cannot hide missing packages.
- A prebuilt-image path, if selected, does not download training dependencies at every start. Provide one documented rebuild/publish command, an explicit invalidation rule, and no stored registry credentials. If dependency reduction wins, prove the full training and checkpoint-loading path still works and record exactly which dependencies remain necessary.
- Run appropriate regression tests, retain before/after evidence including inconclusive or slower results, publish focused PRs, and land necessary MLops fixes once checks pass under Jack's existing authorization. Do not complete merely because a Dockerfile or benchmark report exists.
- Track every GPU attempt in this Task's separate ledger with conservative reservation before admission, a hard job/rental deadline, progress reporting, checkpoint retention and provider-confirmed deletion. Maximum total GPU spend is $5; no borrowing MTG-135's $50 budget or MTG-137's resources. Do local/static work before paid tests and retain enough budget for verification.

Coordination and constraints
MTG-114 is completed. MTG-135 owns the active batching/model-size sweep; do not change or restart its frozen sources, learning rules, jobs or shared scheduler. MTG-150 separately owns removing W&B; do not silently absorb or duplicate that migration. Coordinate dependency boundaries through current code and document overlap.
Use a separate checkout, Etude state directory, controller and ledger. Never tick the MTG-135 shared scheduler. Existing provider capacity is two GPUs: launch at most one startup-test rental and only after a fresh provider inventory shows room; wait on capacity without terminating other Tasks' pods. Existing credentials may be used through the configured environment; no new account or paid registry is authorized.
Reuse the existing deadline, progress, artifact, startup diagnostics and launch guard mechanisms. The confirmed canary015 failure was a 772-second local host sleep after claiming work and before creating the pod, not a proven dependency-install or CUDA failure. Guarded canary016 is a baseline attempt already owned by MTG-135; observe its saved evidence only. Earlier CUDA/NVML failures and the numerical anomaly remain unresolved and must not be claimed fixed by this work.

Decision authority
Jack explicitly authorized autonomous implementation and bounded GPU testing, and said either a delegated worker or the current operator may manage it. Resolve routine image/dependency choices from measurements without another design approval round. Ask in the existing conversation only for a consequential scope change, unavailable credentials/account change, or spending beyond $5. Keep accepted direction and results in this Task's own checkout.


Preparation and launch disposition — 2026-10-17 23:07 UTC

A delegated read-only worker completed preparation with $0 GPU spend and no repository edits or image publication. This is preparation evidence, not a launched Loopflow Flow.

Both `lf task checkout MTG-121` and `lf task run MTG-121 pursue` refused before preparing work: "shared Task first-start admission is unavailable; no recorded execution owner was observed". Reading admission and claiming for local Machine home_39860354aaca640c2ccb50bf6ca609d8 confirmed no established first-start authority. Fresh records contain no checkout, Session, Flow or unfinished Process for MTG-121; location is unrecorded. No supported installed recovery was found. No ownership records were bypassed or directly edited. LOO-414 already owns recovery for historical Tasks missing first-start authority; its observed state was unstarted with no checkout or PR. Disposition: waiting on that coordination dependency. The requested $5 allowance remains entirely unspent.

Prepared implementation

Begin with a digest-pinned minimal Linux/Python 3.12 runtime containing the exact locked training wheels, uv and pinned native build tools. Initially preserve the dependency selection and source-specific managym build to isolate packaging from scientific/runtime changes. Verify image, lock, Python/architecture and installed package identities before training. Mismatches must fail rather than silently download replacements. Keep the existing setup path for comparison; key rebuilds to the base digest, lock, toolchain and build recipe. This is a proposed implementation for autonomous execution once supported admission is available.

Evidence

Torch, CUDA and Triton account for 4,119,325,672 of 4,221,160,784 wheel bytes (97.59%). The remaining 75 packages total 101,835,112 bytes. The stock image runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851 contains 10,564,340,940 compressed layer bytes, with build metadata for Torch 2.8.0, torchvision 0.23.0 and torchaudio 2.8.0. The frozen training lock requires Torch 2.10.0+cu128, Triton 3.6.0, cuDNN 9.10.2.21 and NCCL 2.27.5. Reusing the stock environment would change the experiment; adding the required runtime on top risks duplicating large libraries.

FastAPI currently enters the training-profile import path through academy.server for content identities. W&B is also imported unconditionally. Deleting requirements alone would break imports; W&B removal stays with MTG-150.

Canary016, owned and paid by MTG-135, is now fully verified: run 56ac402f2a9f, source 4d65d477d8e7bc47a327464feea7ace08aae07da, two CUDA updates, finite checkpoint reloaded, all manifest hashes verified, provider deleted. Setup took 625.2284 seconds, CUDA admission 4.2051 seconds, and training 7.02085 seconds. Its estimated rental cost of $0.24654 and conservative charge of $0.45 belong solely to MTG-135. The stock image was already cached, so this does not establish uncached image-pull time. The 4,221,296,640-byte dependency bundle has SHA256 a53386019a7f1bf557a1752b7492ca825114c526f4c16e1f904d78e0d10aa276.

Prepared GPU test proposal

Reserve at most four attempts at $1 each, plus $1 for uncertainty. Use one GPU at a time after checking fresh capacity. Alternate two baseline and two candidate runs with the same GPU class and frozen two-update recipe. At the existing admission ceiling of $1.25/hour, a 36-minute rental bounds compute at $0.75 before the conservative reserve. Include allocation and image pull in both time-to-first-update and time-to-recovered-checkpoint. Classify acquisition as cached, demonstrably uncached, or unknown. Do not retry without diagnosing failure. A custom image could move download time into the image pull; faster setup inside the container alone is insufficient.
