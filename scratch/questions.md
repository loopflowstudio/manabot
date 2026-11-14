# Assumptions made without an answer (MTG-344 one-token layout)

Each is reversible before the PR lands.

1. The earlier object-record rename (`agent_objects`, `managym.Object` and
   their deprecated aliases) is dropped entirely: the branch tree returned to
   origin/main 452d7b5357 before the layout work.
2. The legacy fork is the parts that differ (`legacy_input` modules plus
   `InputLayout::Legacy` branches for the card source, permanent width, stack
   and history), not a full copy of the encoders: facts, effects, relations
   and actions encode identically in both layouts. Byte identity with
   452d7b5357 was checked on 12,403 curated observations.
3. v11 includes v10's turn-state buffers, so the comparison is v10 vs v11
   (same information, different rows), not v9 vs v11.
4. Learning-token projection keeps its roles: `content/semantic/current/
   learning_schema.json` is bound into the world (`learning_schema_sha256`),
   so new permanent/stack roles would refuse v9/v10 checkpoints. Battlefield
   objects and spells reach it under the card roles in v9/v10 row order. New
   roles belong to the next world freeze.
5. Framework backends (MLX, `manabot.policy.prepare`) stay v9-only;
   `require_supported` refuses v11 by name. A fresh default config therefore
   does not run on MLX until it is ported.
6. Frozen regime files that omit `policy_input_version` now resolve to v11
   when validated directly (the default moved); checkpoints go through
   `from_checkpoint`, which keeps their bound version.
7. Retained authored-match parity receipt regenerated: only
   `viewer_observation_sha256`/`viewer_state_sha256` (270 checkpoints × 3
   surfaces each) and the source manifest changed.
8. Retained public-commitment receipt regenerated: only tracker history/prior/posterior digests and the source manifest changed; commands, commitments, gaps and consumed counts are identical.
9. The comparison runs on RunPod RTX A5000 (`mtg344-a5000`, $0.27/h at
   launch): L4 and RTX 4000 Ada had no capacity at 14:04 UTC. All four runs
   share the GPU type, so the arms stay matched.
