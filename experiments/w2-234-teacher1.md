# W2-234 Teacher-1: determinized PUCT substrate

## Verdict

The first real tree-search path is executable end to end. It adaptively reuses
tree nodes, selects edges with PUCT, backs up alternating-player values, emits
root visit counts and root values, generates self-play shards, and runs a
matched chosen-action versus visit-distribution supervised comparison through
the same checkpoint and gameplay evaluation machinery as Teacher-0.

This is a substrate pass, not a strength result. Teacher-1 uses uniform priors
and seeded random playouts at leaves. It builds a separate tree for every
hidden-information determinization, so it retains classic strategy-fusion
limitations. Decision shards preserve the acting viewer observation and legal
mask but are not yet replayable `InteractionOffer`/`Command` trajectories.

## Budget semantics

Teacher-0's `sims` means independent playouts **per legal root action**.
Teacher-1's `sims` means total adaptive tree traversals **per decision**, split
across a declared number of determinizations. Root visits therefore sum exactly
to `sims`; the manifest records this distinction so matched-cost experiments do
not compare identically named but unequal budgets.

## Smoke evidence

The deterministic smoke used two self-play games, two PUCT traversals per
decision, one hidden world, and one epoch per arm. It generated 133 decisions
with zero illegal teacher actions, finite root values, and visit mass confined
to the encoded legal mask. The teacher probe used 184 tree traversals over 92
decisions with zero caps. The full runner completed in 49 seconds and produced
both checkpoints plus an atomic manifest.

Both training arms improved held-out policy loss and exceeded
uniform-over-legal top-1 accuracy. Their final squared error against the
teacher's root-value labels was 0.189. This checks value-label learnability; it
does **not** establish calibration against terminal outcomes, and the two-game
sample is too small for either claim. The overall experiment correctly
reported `completed_diagnostic_failure` because the two-game gameplay
comparison was maximally noisy and the visit arm did not clear the
noninferiority gate. These numbers validate plumbing only.

## Controls now available

- identical PUCT data, initialization, game-level split, optimizer, capacity,
  root-value target, and evaluation path;
- one-hot chosen-command policy supervision versus full root-visit
  distribution supervision;
- held-out policy KL as well as cross-entropy, top-1 agreement, and target
  entropy, so the differently entropic labels are not compared by raw CE alone;
- exact source/content/observation hashes, teacher spec, world and simulation
  budgets, seed, hardware, caps, dataset/checkpoint hashes, and failure branch;
- deterministic repeated search with exact visit, Q, value, and action output;
- aggregate tree nodes, sampled worlds, mean/max depth, and root traversals in
  matchup and dataset manifests, making root-only degeneration visible;
- authoritative-root isolation: no search clone can mutate the live match;
- fail-fast rejection if the raw teacher surface exceeds the encoded action
  ABI instead of silently truncating labels.

## Next bounded experiment

Run three declared Teacher-1 budgets and compare against Teacher-0 at matched
**total playouts**, not the existing per-action `sims` label. If Python tree
bookkeeping dominates, keep this implementation as the readable differential
oracle and move the algorithm behind `BranchDriver`. Only after uniform-prior,
random-leaf PUCT has a measured baseline should policy priors and learned value
leaves be added as separate ablations.

The next trajectory change is orthogonal: record replayable viewer-safe
offers, chosen commands, state identities, and provenance rather than treating
encoded observations as sufficient replay evidence.
