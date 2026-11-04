# TLA pilot card-capacity demonstration — 2026-11-04 UTC

Decision: **advance to delivery**. This headless demo accepts the implemented
card-capacity repair for PR [#420](https://github.com/loopflowstudio/manabot/pull/420).
It does not establish paid GPU learner progress, learning improvement or Task
completion. The code Graph's next step is delivery, followed by verification.

## Reviewed experience and intended progress

Source: `51566c9f71b01cbef3b40993a62c1e459c10a470`. Current design:
[v5 protocol](../experiments/protocols/tla-trophy-213-v5.md),
[frozen shapes and schedule](../experiments/protocols/tla-trophy-213-v5.json),
[continuation plan](tla-pilot-verify.md), and
[accepted direction](questions.md).

The preceding implementation pass intended to remove native encoding's
hard-coded 60-card allocation and bind the prospective pilot to 128 rows,
preserving the 134-deck population, input v7, frozen failures and cumulative
$10 cap. V4 had passed CUDA admission but failed before its first update:
40 original cards, five generated tokens and 16 sideboard cards require 61
rows. That exact failure supplies a direct before/after witness.

## Observed demonstration

Executed through the existing production-backed proof paths:

```bash
uv run --extra dev --extra artifacts --extra evaluation pytest tests/eval/test_tla_pilot.py::test_pilot_retains_generated_cards_and_sideboard_in_all_encoders 'tests/model/test_shared_effects.py::test_shared_effect_policy_uses_ordinary_learner_and_export[128]' tests/eval/test_tla_pilot.py::test_pilot_cli_discovery_keeps_evaluation_callback_importable -q --basetemp=.runs/mtg213/demo-20261104
```

Result: **4 passed in 8.16 seconds**. Log:
`.runs/mtg213/demo-20261104.log`.

- The retained 300-action prefix reaches its recorded pre-command frame in
  scalar and vector environments. Its resulting observation contains 45 card
  objects and 16 sideboard cards. Python, native scalar and native vector
  encoding agree on every tensor at both 61 and 128 rows. A 60-row allocation
  still fails explicitly; nothing is dropped to manufacture success.
- The ordinary `execute_regime` CPU path completes one optimizer update with
  290 learner transitions, exports its checkpoint, and reloads with input v7
  and card capacity 128. Receipt:
  `.runs/mtg213/demo-20261104/test_shared_effect_policy_uses0/training/run.json`;
  raw checkpoint: the sibling `self-play-raw.pt`. This is the bounded local
  smoke matchup, not random draws from the pilot population or CUDA admission.
- Etude discovery loads the current `tla_pilot_v5.py` runner and resolves its
  evaluation callback from the canonical importable module. This proves
  discovery/callback wiring, not completion of the paired evaluation schedule.

The earlier publication checks remain supporting evidence: 391 Python tests
passed with one skip, and full debug Rust tests passed. Logs:
`.runs/mtg213/publish-python-tests.log` and
`.runs/mtg213/publish-cargo-test.log`. No production code changed during demo.
Fresh GitHub reads show PR #420 open at the reviewed HEAD, with required CI
still pending; this review is not merge authorization to bypass those checks.

## Human direction and operator judgment

No human attended this headless demonstration and no new human acceptance is
claimed. Jack Heart's recorded direction authorizes shipping enabling repairs,
preserving evidence, and continuing the GPU pilot within $10; it requires the
launch milestone only after actual GPU learner updates. Acting under that
direction, the operator accepts this bounded repair because its exact original
failure now passes without exclusions and the normal learner/export path works.
Returning to implementation would have no concrete newly observed defect to fix.
No design change is proposed or agreed in this review.

## Remaining work and next proof

Deliver PR #420 after its checks resolve. Then continue the existing prepared
v5 Run `e8c76bc2c06f`, bound to its own frozen source
`97a2cfb963d7e3a3b2816f31c45dda72f34cdddd`, through fresh capacity, inventory,
billing and bounded quote checks. Its own CUDA admission is still required;
this later-HEAD CPU demo does not retarget or admit that Run. Preserve the
reserved $2.95 and next-allocation $1 ceiling under the cumulative $10 cap.

The last retained capacity observation (06:59:30–33 UTC) reported all four
configured options unavailable. No fresh provider read, paid allocation or
GPU launch occurred in this demo. Capacity is an execution dependency, not an
unanswered product choice. Report the launch milestone only after GPU updates;
admit trained checkpoints, finish the full paired evaluation, and report cost
and learning limits relative to Elemental. Those remain unfinished outcomes.

The 128-row allocation is not a universal bound, and the CPU update is not
strength evidence. Historical missing evaluations stay missing. No presentation
page or absent-user checkpoint was created on this headless surface. This note
is the review record; the live continuation plan remains authoritative.
