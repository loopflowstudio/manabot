# Effect vocabulary demo — MTG-235, 2026-11-02

Decision: **advance** from demo. The accepted selector amendment holds on
published head `1dffa1ec1af5d4b6140064875e292bfad79186a6`, whose implementation
is the verified cache-integration source `d22b55757`. No repairable gap was
found in this demonstration. This is delegated headless judgment under the
caller's instruction to stand in for Jack Heart, not a claim that he observed
or approved this run personally.

Current design: [Effect verbs over an ObjectSelector](mtg-235-effect-vocabulary-verbs.md).
Delivery: [PR #412](https://github.com/loopflowstudio/manabot/pull/412).
The experience is authoring one verb with a selector, then resolving its
complete legal choice through the native engine and its Academy/LLM consumers.
No browser layout change or new interaction design is proposed.

## Authority and progress from the preceding pass

Jack Heart accepted decisions 1a/2a/3a/4b: shared object selectors, removal of
AffectedSpec, one EffectValue vocabulary, and Chosen for supported object verbs.
The preceding publication pass intended to preserve those decisions while
integrating main's caches, revalidate the combined source and publish it.
Its retained logs in `.runs/publish-cache-mtg-235/` record 885 debug Rust tests
(including 502 card tests), 402 Python tests, 9 Vitest tests and 10 browser
checks passing; 2 Rust tests were ignored and 1 Python test skipped. These
are prior evidence, not tests all rerun at demo.

The published head matches the local head. Hosted CI run
[41564801622](https://github.com/loopflowstudio/manabot/actions/runs/41564801622)
now passes, including Rust, Python, integration, Academy UI, Clean-machine Play,
semantic conformance, Engine Budget and Tests Result. This resolves the prior
pass's pending hosted-CI evidence. Release-package build/install checks also
passed; its publish job was skipped. No local performance claim is inferred.

During this demo, Jack Heart's comment
`2e1c66f1-d00a-4319-b5f4-c631bcf608c9` explicitly corrected delivery order:
land MTG-235 before T6. **All earlier statements making T6 a prerequisite to
MTG-235 landing or completion are superseded.** T6 adapts its saga triggers
and new effects in its own work. The PR body, design status and current
`scratch/questions.md` guidance now reflect this correction. A fresh Task
read still found no T6 PR (realign running); that is informational only.

## Executed demonstration and observed results

Bounded scenarios exercise the existing native resolution/action path; the
parity CLIs exercise the rebuilt native extension through Academy's real
configured live/headless/replay adapters. No mock resolver or replacement
receipt was used. Logs are retained in `.runs/demo-mtg-235/`.

| Action | Expected result | Observed proof |
| --- | --- | --- |
| Resolve `tap(chosen(creature))` with two creatures; choose the second | A mandatory legal selection appears; only the selected creature taps | `selectors-rerun.log`: passed; the first stays untapped and the stack clears |
| Stop counter removal, then resolve another chosen tap | The next instruction offers a fresh choice | `selectors-rerun.log`: passed; the second creature can be selected independently |
| Exile the source, then attempt a chosen exile with no candidates, then “if you do” gain life | No stale move result grants life | `selectors-rerun.log`: passed; source exiled, life unchanged |
| Earthbend every controlled land | Both lands become 2/2 creatures with haste | `selectors-rerun.log`: passed via layered characteristics |
| Activate Joo Dee, surveil, create its copy, and select the original for sacrifice | Both copies are legal sacrifice choices; private surveil stays private | `joo-dee.log`: passed; original leaves, untapped 2/2 token remains without copied counters |
| Encode every object/player selector | Each selector reaches the definition-derived learning tokens | `tokens.log`: passed |
| Run the LLM text-observation suite | Native decisions render, retain alternatives and enforce binding | `text-observation.log`: 40 passed |
| Verify authored live/headless/replay parity | Commands, viewer projections and identity agree | `authored-parity.log`: 142 commands, 143 checkpoints, 858 viewer checks, no divergence; object 96@3 rejects as `stale_object`, retained address as `stale_revision` |
| Verify public-commitment parity | Retained commitments reproduce without missing engine support | `public-parity.log`: 142 commands, 69 commitments, zero gaps |

Reproduction commands (run from repository root):

```sh
CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml --test rules_tests selector
CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml --test cards selector
CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml --test cards joo_dee_surveils
uv run --extra dev --extra play pytest tests/semantic/test_text_observation.py -q
uv run --extra play python -m academy.authored_match_parity verify
uv run --extra play python -m academy.public_commitment_parity verify
```

The first Rust invocation mistakenly named test target `rules`; Cargo rejected
it before running tests. That failure remains in `selectors.log`. The corrected
`rules_tests` invocation passed all four selected cases. No source repair was
needed. Six focused Rust tests and 40 Python tests passed in this demo.

Supporting source: `managym/tests/rules/cr_608_resolution.rs`,
`managym/tests/cards/tla.rs`, `managym/tests/cards/observability.rs`,
`tests/semantic/test_text_observation.py`, `academy/authored_match_parity.py`,
and `academy/public_commitment_parity.py`. Retained browser evidence is
`.runs/publish-cache-mtg-235/browser-tests.log`; this headless pass adds no
screenshots or new visual-review claim.

## Judgment and next action

The observed behavior matches the accepted design: authors reuse verbs without
losing legal alternatives, instruction-local choice state or replay identity.
The current combined vocabulary has 71 variants (70 in the amendment plus
upstream Surveil), with 14 specialized object-named rules remaining outside the
cut. Prior complete acceptance plus today's direct resolution and consumer
proof supports advancing, with no further implementation iteration or design
change. No unresolved product question requires an absent person.

Continue the already-authorized delivery process for PR #412, then verify the
landed result and complete MTG-235. This demo itself neither merges nor marks
the Task complete, and starts no other Graph. T6 follows this landing.
No servers, watchers or background commands were started or left running.
