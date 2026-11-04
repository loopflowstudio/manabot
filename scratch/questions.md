# MTG-244 design decision — 2026-11-04

Jack Heart additionally requested measurement and mitigation of performance and
learning risks on 2026-11-04. [mtg-244-risk-validation.md](mtg-244-risk-validation.md)
records the required engine/search, observation/model, representation and
compatibility evidence for this delivery. This is accepted work, not an optional
future recommendation; retain the existing no-merge/no-experiment boundary.

Jack Heart resolved the input-version blocker in this Task conversation on
2026-11-04: "new input version is totally fine". The implemented successor is
policy input v8. The operator's rollout choice is
explicit selection, preserving the standing v6 fresh default and saved input
versions; the user did not request a default migration.

Concrete row types, capacities, program joins and encoder dispatch are routine
implementation work within this approval. Older inputs retain their contracts:
representable states continue to work; unrepresentable new layer payloads must
fail at the shared observation/input boundary before model inference, with a
diagnostic naming the selected version and required successor. The shared
validation boundary and focused native/Python/model-forward proofs are
implemented. No silent
omission, implicit upgrade, checkpoint reshaping or card exclusion list.

The archived draft has been recovered and the accepted interaction witnesses
implemented. The earlier blocked verdict is superseded; risk evidence remains
open before review readiness. Delivery stays publication for review at the
interactive boundary, with no merge or experiment launch.

[mtg-244-layers.md](mtg-244-layers.md) owns scope, the counterexample, implemented
design and remaining work. The unrelated parent T7 note remains at
`4463e57f5a92ebd3b60e2fb50c046b84ac2c0372:scratch/questions.md`.

Implementation assumption (2026-11-04): `WORLDS.md` keeps w17 open and
extends its exact bindings in place; this Task does not freeze a world. A new
`semantic-kernel-w17-v2` corpus preserves the earlier v1 evidence.

Sync resolution (2026-11-04): retain MTG-244's direction above. Incoming
MTG-213 pilot decisions and evidence remain at
`1546aae3fa7b207b2c5e8ac92a6571af036ddf83:scratch/questions.md`;
they do not authorize a pilot launch in this Task. Keep shared input-version
validation before the explicitly configured native action-prefix selection,
and retain both native API declarations. The default overflow policy remains
error, and the fresh policy-input default remains v6.


Risk implementation assumption (2026-11-04): after discovering severe unrelated
host contention (load average >500), stop long timing runs and retain only
exploratory timings, deterministic operation/allocation counts and focused proofs.
Clean parent/change and full-policy latency comparisons remain unavailable locally;
the existing isolated Gate/Engine Budget path is the safe producer. This is an
evidence limitation, not approval of a regression, a changed tolerance or review
readiness. `mtg-244-risk-validation.md` retains exact results and remaining checks.
