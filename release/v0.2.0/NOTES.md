# v0.2.0

<!-- loopflow:release-notes=narrative;gate=safe -->

Manabot 0.2.0 brings training, the managym rules engine, and Academy’s play and study tools into a more coherent stack. Across this cycle, training gained structured decisions, recoverable experiments and two-seat self-play, while Academy made the agent’s legal choices easier to inspect and follow. The release also prepares a single versioned Python distribution with the native engine included; publication and installation from PyPI remain release steps to verify.

## Install one training stack

Installing Manabot previously required a source checkout and a separate engine build. The new packaging brings Python code, the Rust engine and runtime data together so supported machines can use a pinned wheel without building Rust after publication. ([#350](https://github.com/loopflowstudio/manabot/pull/350))

- One `manabot` distribution contains the engine at `manabot.managym`; `import managym` remains an alias to the same native module and protocol types.
- The release workflow builds Linux x86-64 and Apple Silicon wheels from the source archive. Optional extras provide the Academy backend, reports, artifacts and evaluation dependencies; Torch is declared at 2.10.0.
- A locally rebuilt wheel installed outside the checkout and completed a CPU training update with saved checkpoints. GPU execution, Academy’s browser frontend and repository experiment drivers are outside that wheel-only proof.

## Train both seats and keep experiments reproducible

Self-play now learns from both seats by default, bringing more of each completed game into training. The experiment layout also makes the recipe for a current study easier to locate, select and reuse, while preserving historical evidence separately. ([#330](https://github.com/loopflowstudio/manabot/pull/330), [#343](https://github.com/loopflowstudio/manabot/pull/343))

- `baseline-v3` combines the FiLM default with two-seat learning. Saved schema-1 recipes retain their one-seat meaning, and frozen or random opponents remain one-seat. Paired models use independent optimizers.
- Tracking records actual learner and environment decisions and complete terminal seat trajectories. Continuation/adoption and replay-checked CPU evaluation support the existing two-seat experiment; reporting cannot rent machines or change training.
- Current experiments live in flat `experiments/<experiment>.py` modules beside their write-ups. Select canonical recipes with `manabot train --regime-name <experiment>` and variants with `<experiment>:<variant>`.
- Shared builders, study execution and bounded matchup evaluation live in the training and arena library modules. Unused runners, their exclusive configs and fixtures have been removed; Git history retains their code.

The retained early two-seat comparison uses one seed, equal update counts and unequal training cost. It does not demonstrate a strength-per-dollar gain. Landing the training changes does not restart or upgrade the running experiment or its pinned services. Continuation segments restart collector, game and RNG streams, and the paired API does not supply a remote dual-model executor. ([#330](https://github.com/loopflowstudio/manabot/pull/330))

## Understand and commit Academy decisions explicitly

Academy now connects each legal choice to the card, target or payment step it affects. The battlefield and complete legal choices take priority across Play, Watch, Replay and Study, with inspection available before committing an action. ([#342](https://github.com/loopflowstudio/manabot/pull/342))

- Decision context shows who acts, source objects and targets, unpaid reservations, committed costs and mana pools using viewer-safe native information.
- Focusing an offer reveals its objects without submitting it. Sole choices still require explicit commitment; stale rejection, pending decisions on reconnect and exact Study forks remain part of the Command contract.
- Bounded battlefield lanes group permanents and page dense groups. Logs, advice and secondary controls start collapsed, and one pending-aware New Game control replaces competing controls.
- Cached Scryfall artwork includes artist/source credits and a procedural fallback. The complete frontend now lives in `academy/ui`.

Focused browser interactions, keyboard play, reconnect behavior and automated accessibility checks passed for the redesign. The Learn and Precision demos are automated validation prefixes, not evidence of human usability. The slow full-match screenshot suite was retired; older visual corpora remain historical evidence. ([#342](https://github.com/loopflowstudio/manabot/pull/342))

## Follow payments through nested choices

Optional mana costs and waterbend costs now use one payment flow with explicit paid and declined outcomes. This makes pending payments understandable to both policy input and Academy while preserving costs already committed during nested mana choices. ([#344](https://github.com/loopflowstudio/manabot/pull/344))

- Waterbending Lesson still draws three before payment and requires a discard when payment is declined. A waterbend contribution commits the payment and removes the decline option.
- Counter-unless-paid effects share the runtime. Both outcome branches are projected into policy input.
- Mana-filter activation planning preserves colors needed by the announced spell. The repair covers the case where Prophetic Prism consumed the only green source and left an offered Gnarled Professor cast unable to complete. ([#348](https://github.com/loopflowstudio/manabot/pull/348))

## Operational notes

- **Current content:** play and training use world w11, learning schema 7 and `elemental-world-v2`. Earlier checkpoints, including w10, are rejected by current-world admission. The ten deck setups and observation tensor dimensions remain unchanged; historical content and receipts are retained. The payment work establishes neither a strength baseline nor balance certification. ([#344](https://github.com/loopflowstudio/manabot/pull/344))
- **Publishing:** 0.2.0 was not published in the packaging PR’s release record. Before tagging reviewed main, verify the pending PyPI trusted publisher and GitHub environment reviewers/tag restrictions. The workflow publishes through OIDC with attestations. After upload, verify installation directly from PyPI on both supported targets and retain version and attestation results. The revised single-distribution workflow still required its final CI run in that record. ([#350](https://github.com/loopflowstudio/manabot/pull/350))
- **Etude:** integrations now use `etude-sdk==0.2.0` and its `etude.run` / `etude.train` modules. An existing launcher home requires a fresh `ETUDE_HOME` for schema 8; no database migration is provided. Training tracking keeps the `~/.etude/track` layout and CLI output. The removed saved-evaluation publisher used the deleted `Run.evaluation` API; new evaluation recording uses Etude eval runs, while retained `monitor.json` files remain the historical record. ([#349](https://github.com/loopflowstudio/manabot/pull/349))
- **Experiment entry points:** `experiments/recipes.py`, `experiments/runners/`, the `REGIMES` registry and its JSON-writing `main()` are removed without a shim. Update callers to the current experiment modules and library entry points. Shared builders now live in `manabot/training/builders.py`. ([#343](https://github.com/loopflowstudio/manabot/pull/343))
- **Clean-machine launch:** CI still requires a clean checkout to reach playable state, survive offline reload and accept a later action. The 90-second budget is measured and reported, with a warning when exceeded; it no longer fails the job based on shared-runner speed. A launch that never becomes ready fails at the 600-second hang limit. ([#347](https://github.com/loopflowstudio/manabot/pull/347))
- **Proof limits:** the packaging record retains a failing historical study benchmark caused by a missing replay prompt and stale frozen source binding. Historical public-commitment tape regeneration also remains uncertified by current-runtime checks. Neither set of frozen evidence was regenerated to conceal the mismatch. ([#350](https://github.com/loopflowstudio/manabot/pull/350), [#342](https://github.com/loopflowstudio/manabot/pull/342))

## Small changes

- Profiler tests use a controlled clock instead of sleeps and timing tolerances. The distributed benchmark’s hidden three-second cap on local metadata reads was removed. ([#346](https://github.com/loopflowstudio/manabot/pull/346))
- CI selects jobs by changed paths while retaining a required aggregate result that rejects unexpected skips or missing selector output. Manual dispatch still runs every job. ([#342](https://github.com/loopflowstudio/manabot/pull/342))
- Etude 0.2.0 replaces local launcher diagnostics, startup guards, storage-read retries and prebuilt-client workarounds with upstream behavior. ([#349](https://github.com/loopflowstudio/manabot/pull/349))