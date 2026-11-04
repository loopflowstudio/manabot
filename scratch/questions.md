# MTG-244 design decision — 2026-11-04

Jack Heart additionally requested measurement and mitigation of performance and
learning risks on 2026-11-04. [mtg-244-risk-validation.md](mtg-244-risk-validation.md)
records the required engine/search, observation/model, representation and
compatibility evidence for this delivery. This is accepted work, not an optional
future recommendation; retain the existing no-merge/no-experiment boundary.

Jack Heart resolved the input-version blocker in this Task conversation on
2026-11-04: "new input version is totally fine". Continue implementation with
a successor policy input (provisionally v8). The operator's rollout choice is
explicit selection, preserving the standing v6 fresh default and saved input
versions; the user did not request a default migration.

Concrete row types, capacities, program joins and encoder dispatch are routine
implementation work within this approval. Older inputs retain their contracts:
representable states continue to work; unrepresentable new layer payloads must
fail at the shared observation/input boundary before model inference, with a
diagnostic naming the selected version and required successor. Choose the exact
boundary and prove native/Python parity and model-forward behavior. No silent
omission, implicit upgrade, checkpoint reshaping or card exclusion list.

Recover useful parts of the archived draft and finish the accepted witnesses.
The earlier blocked verdict is superseded by this direction. Publish for review
and stop at the interactive boundary; no merge or experiment launch.

[mtg-244-layers.md](mtg-244-layers.md) owns scope, the counterexample, proposed
design and remaining work. The unrelated parent T7 note remains at
`4463e57f5a92ebd3b60e2fb50c046b84ac2c0372:scratch/questions.md`.

Implementation assumption (2026-11-04): `WORLDS.md` keeps w17 open and
extends its exact bindings in place; this Task does not freeze a world. A new
`semantic-kernel-w17-v2` corpus preserves the earlier v1 evidence.
