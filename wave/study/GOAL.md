---
pm:
  provider: linear
  linear_initiative: 1f18f754-dbad-44c0-a90a-4a51941aed88
  linear_team: feadac43-5d63-412e-b6f6-39424a13f45a
---

# Study

## Objective

Turn completed Etude Fantasia games into a beautiful, honest engine-study experience:
restore consequential decisions as the player understood them, show what else
was worth considering, and let the player retry, inspect, and inhabit stronger
lines. Study should make human learning intrinsically enjoyable while producing
versioned, attributable evidence for the research program toward superhuman
Magic play.

## Measures

- A completed creator-selected matchup opens into a deterministic semantic
  timeline where every historical player decision is directly addressable;
  3–7 ranked landmarks guide the first review without gating free navigation.
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
- Self-study keeps engine judgment sealed until reveal while preserving the
  historical command as subtle context; a shared question can seal both the
  historical command and engine evidence until the recipient responds or
  explicitly reveals them.
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

The design standard is: triage like Chess.com, navigate and explore trees like
chess and OGS analysis boards, retain 17Lands-style replay and shareable-position
ergonomics, and grow toward poker-solver-quality range and counterfactual study
only as the evidence supports it. Prefer a few grounded plans and short
contingent continuations over an omniscient evaluation bar, raw event log,
generic coach prose, or persistent variation-tree clutter.

Poker solvers are a north star, not a vocabulary shortcut. The evidence ladder
is exact historical positions; policy and search comparisons; robustness across
sampled hidden worlds; learned beliefs and ranges; range-conditioned
counterfactuals; then approximate subgame equilibrium, regret, or
exploitability. Do not ship an empty Ranges surface or use equilibrium language
before the corresponding objects and measurements exist.

Concrete repository changes begin as Linear Tasks under one Study Project.
Projects must prove player-visible improvements or a durable evidence frontier;
infrastructure without a demonstrable study moment belongs in the providing
Game or Rules wave.
