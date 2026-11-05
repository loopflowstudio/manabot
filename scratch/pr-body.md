Naming a card (The Rise of Sozin) used to offer every catalog name in one decision, which tied the catalog's growth to action capacity. Naming is now a short sequence of ordinary native decisions: choose a group of names, then a name. Each prompt offers at most 64 options however large the catalog grows, every name stays reachable exactly once, and only the final name choice does anything. The policy scores each prompt with its regular head and learns from it through normal rollouts. The experimental neural compound decoder is removed.

<!-- loopflow:task-pr-context:start -->
> [!NOTE]
> **Task:** [Make catalog-wide naming decisions complete and scalable for policies · MTG-243](https://linear.app/loopflow/issue/MTG-243/make-catalog-wide-naming-decisions-complete-and-scalable-for-policies)
> **PR lifecycle:** The pull request is published for review; no Task settlement is requested.
<!-- loopflow:task-pr-context:end -->

## What changes

- **Engine:** `Decision::NameCard` holds a `NameScope` over one shared, immutable, sorted `NameVocabulary`. This is the real non-token catalog, including back faces. A scope larger than `NAME_BRANCHING` (64) offers a deterministic, balanced, complete partition into at most 64 contiguous groups, applied recursively. Choosing a group only narrows the pending scope and re-suspends. Choosing a name pushes one `ExileNamed`, whose private `SearchNamed` continuation keeps optional stopping and fail-to-find. With 928 names that is 15 groups of 61–62, then a name; the installed 344-name catalog shows 6 groups. Single-option prompts follow `skip_trivial`.
- **Shared surfaces:** Group and name prompts are ordinary `ActionOption`s ("Names A through B", "Name X") with a machine-readable `name_group` span through JSON, PyO3, Commands, text observations and Academy. Reconnect, replay and forks restore the actual current prompt, and a Command bound to an earlier prompt is stale.
- **Policy input v9 (explicit opt-in):** `name_candidates[A,4]` at ordinary action capacity `A = max_actions` holds the name row, an optional definition reference and the group's first and last name rows, all one-based. The sorted span identifies every member without revealing anything hidden. Live observation bytes are 155,120 whatever the vocabulary size.
- **Model:** A name scores from a trainable name vector plus its definition's learning program where one exists. A group is the exact prefix-sum mean of its members' name vectors, so every member receives the gradient and no member programs are encoded. Name identities also join viewer-visible card, known-hand and decklist inputs. Closed decklists contribute nothing. Checkpoints bind the ordered vocabulary, the definition joins, `public-groups-v1` and the branching bound, and any mismatch fails before weights load.
- **Compatibility:** V1–v7 never carried name identity and see group and name prompts as anonymous modal choices within their capacities. Explicit v8 (`[A,2]`, a single choice over the whole vocabulary) cannot represent groups and rejects them. Fresh v6 defaults and missing-checkpoint defaults are unchanged.
- **Decoder removal:** `CompoundDecoder`, `ActorCritic.compound`, `compound_decisions`, `TrainCompound`, and the compound rollout, learner, search and action modules are deleted, along with their study recipe and exclusive tests. Historical compound configs, checkpoints and `train_compound` regimes fail clearly, pointing to `ba3cd58de` for reproduction. Retained evidence is unchanged. The native compound-offer lowering stays because it belongs to the live structured-offer family and has Rust rule tests. Validating compound learning, including mana payment, is a separate experiments Task.
- **Training/speed (carried from the earlier pass):** Rollouts pack legal action rows and full behavior distributions. Program encoding runs the same GRU in chunks of at most 64 steps, carrying full gradients across chunks.
- **Engine Budget job:** Builds both engines before timing, runs candidate/reference/reference/candidate, compares medians and retains every sample with CPU telemetry. The 10% tolerance and the `ea316ff14` reference are unchanged.

## Why it matters

The Standard roadmap grows the catalog well past today's cards. Ordinary action capacity no longer depends on vocabulary size. People, LLMs and policies all see the same bounded prompts through existing controls, without a separate decoding architecture for policies alone.

## Risks and limitations

- **Learning:** A group prompt adds a real decision with the standard discount and credit, and no reward is assigned to group choices. Learning is not claimed to equal a single flat choice. Tests show learnable group and name distinctions with gradients, and a self-play rollout plus PPO update through both transitions. Those witnesses do not show context-dependent routing between different groups or compare learning quality; a paired-context group witness was proposed and has not been run. No strength or efficiency result is claimed, and no training was run.
- **Contracts:** Semantic Observation 8, capture 5 and state hash 16 are this change's single transition. Receipts and Academy fixtures are refreshed, and historical captures stay readable through schema 7 for validation only. PUCT and local-search admission source hashes change.
- **Speed:** Earlier CPU batch measurements against `b31998cc3` (width-64 mixed or naming batches about 5.8–6.6× lower CPU, width-16 ordinary batches slower) predate sequential naming and were not retaken. No native or end-to-end speedup is claimed. A local control on byte-identical binaries failed worst reply by more than 10%, which is measurement noise rather than a code regression. That failure stays on record. Group means still build a vocabulary-wide prefix sum on every forward pass.
- **Fixture:** The 928-name fixture mostly adds names with no executable program. It is not 928 playable definitions and does not broaden deck admission. It also makes no claim that the catalog covers every name legal under Magic's rules.
- **Not included:** Adopting v9 as the fresh default, adding production names and any training run. The separate study of compound decisions, including mana payment (acc0fe), is filed and has not started.

## Checks

At reviewed head `833ef4120`, every CI check passed: Rust debug tests, Python unit tests, main and long integration, Academy integration and UI, protocol and kernel conformance, clean-machine Play, packaging and Engine Budget at the unchanged tolerance and reference. Later commits add only review notes. Locally before the earlier sync: debug `cargo test --workspace` (972 passed), clippy and rustfmt clean, and focused Python suites covering model, naming, Academy recovery, text, verify, pilot, current-world admission and package boundaries (611 passed, 1 skipped).

Main has since merged MTG-244, which also defines a policy input v8 (layer state). This branch must sync before landing, and CI must pass again on the merged head. The naming input's version number may change in that sync.
