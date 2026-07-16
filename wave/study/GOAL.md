---
pm:
  provider: linear
  linear_initiative: 1f18f754-dbad-44c0-a90a-4a51941aed88
  linear_team: 60558c53-2169-49f8-a76a-1f4586705aa9
---

# Study

## Objective

Turn completed manabot games into a beautiful, honest engine-study experience:
restore consequential decisions as the player understood them, show what else
was worth considering, and let the player retry, inspect, and inhabit stronger
lines. Study should make human learning intrinsically enjoyable while producing
versioned, attributable evidence for the research program toward superhuman
Magic play.

## Measures

- A completed creator-selected matchup opens directly into a guided review that
  identifies a small number of meaningful decision landmarks rather than
  replaying raw engine actions.
- Every policy, search, uncertainty, and robustness claim records the viewer's
  information boundary plus content-pack, engine, model, state, and analysis
  identities; hindsight is separate and unmistakable.
- At a landmark, the table makes the played command and credible alternatives
  visually legible, supports retry before reveal, and can play semantic
  continuations without inventing rules meaning in the client.
- A player can enter an exact ephemeral branch, compare it with the recorded
  line, and return to the original match in one action; branches persist only
  when explicitly saved.
- Bookmarks, annotations, shared decisions, and model-version comparisons are
  reproducible and viewer-safe, and any research use of human study behavior is
  transparent and attributable.
- Desktop and mobile study flows keep the Magic table primary and pass
  deterministic release-stack interaction, accessibility, latency, and visual
  evidence gates.

## Process

Study consumes the versioned `ExperienceFrame`, `InteractionOffer`/`Command`,
`PresentationEvent`, replay, search, and safe-fork contracts owned by Game and
Rules. It does not create a second rules engine, client-side legality model,
search implementation, or replay truth.

Default analysis answers "what was reasonable from what the player knew then."
Actual hidden cards and future randomness may appear only through an explicitly
labelled hindsight lens and never contaminate the default evidence or research
trace. Policy mass, search value, visits, robustness across sampled worlds, and
uncertainty remain distinct quantities.

The design standard is: triage like Chess.com, explore on the board like OGS,
retain 17Lands-style replay and sharing ergonomics, and explain hidden-information
Magic decisions in a way those perfect-information games cannot. Prefer a few
grounded plans and short contingent continuations over an omniscient evaluation
bar, raw event log, generic coach prose, or persistent variation-tree clutter.

Concrete repository changes begin as Linear Tasks under one Study Project.
Projects must prove player-visible improvements or a durable evidence frontier;
infrastructure without a demonstrable study moment belongs in the providing
Game or Rules wave.

