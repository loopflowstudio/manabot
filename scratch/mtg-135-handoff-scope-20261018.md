# MTG-135 handoff scope clarification

Observed 2026-10-18T01:43:58.139488+00:00. This records the destination's saved direction while the longer
size/batching comparison is still running; it is not a final recipe or launch.

MTG-122's current protocol selects 64-stream GPU training. The later split-placement
direction in `scratch/questions.md` selects Mini evaluation, removes inline
diagnostics, preserves the full 36,000 scientific games and frozen controls, and
keeps the separate $35 GPU ceiling. That later direction supersedes the proposed
104-game cohort reduction and combined GPU training/evaluation placement. The
MTG-135 recommendation must address this configuration and its common research
catalog, rather than treating product-world checkpoints or our inline-diagnostic
timings as directly transferable.

The earlier interim handoff's statement about an execution hold was an account
of this conversation's understanding, not a new user instruction or an additional
approval requirement imposed by MTG-135. MTG-122's own current accepted user
direction governs execution. This clarification neither launches work nor changes
any existing hold, budget, source, recipe or cohort. MTG-135 still owes its exact
long-comparison recommendation; no final playing-strength result exists yet.

The frozen protocol017 selection compares 64 and 256 streams. A possible
256-stream winner would be presented as an alternative to MTG-122's selected
64-stream configuration, with the strongest supported 64-stream option reported
explicitly. Any adoption decision remains with MTG-122; no recipe is selected
before the retained full comparison and evaluation.

Read inputs at this observation:

- `experiments/environment-complexity.md`: SHA256 `78a0e81e2c0f25fc532405b71d3ea7f98c835bbcba04ae7842cd8161878d78fc`.
- `scratch/questions.md`: SHA256 `a0606132a6315657eb69ece9ce1b2e34be23199182bb8c4e1cff4aa23c3612bd`.
