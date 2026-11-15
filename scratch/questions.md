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
6. Saved regime observation settings that omit `policy_input_version` retain
   v9 through `TrainingRegime.read_saved_stages`; fresh configs default to v11.
   Checkpoints go through `from_checkpoint`, keeping their bound version.
7. Retained authored-match parity receipt regenerated: only
   `viewer_observation_sha256`/`viewer_state_sha256` (270 checkpoints × 3
   surfaces each) and the source manifest changed.
8. Retained public-commitment receipt regenerated: only tracker history/prior/posterior digests and the source manifest changed; commands, commitments, gaps and consumed counts are identical.
9. The comparison runs on RunPod RTX A5000 (`mtg344-a5000`, $0.27/h at
   launch): L4 and RTX 4000 Ada had no capacity at 14:04 UTC. All four runs
   share the GPU type, so the arms stay matched.
10. The four runs stopped at their wall-clock watchdog at 115-156 updates
    (A5000 ran ~50-60 s per update against the 40 s budget). Accepted rather
    than continued: every run has the 20/40/75 checkpoints, so the arms
    compare at matched updates.
11. The v10-v11 head-to-heads were dropped: an arena game encodes both seats
    with one `ObservationSpace` (`manabot/sim/native_game.py`, "players have
    different observation ABIs"), so cross-layout checkpoints cannot meet.
    The arms compare through the heuristic only. Per-seat encoding in the
    arena is separate work.
12. Heuristic play surfaces a ChooseMode whose focus names an object with no
    row (Food activation, then a modal choice; 1 in ~12k observations). It
    happens identically in v10, whose encodings match main, so it predates
    this change; filed as a bug Task rather than fixed here.

13. The characteristics aliases use removal release 0.4.0, matching this
    Task's already reviewed v9/v10 deprecation. Frozen viewer JSON keeps
    its old field names through boundary-only wire records; those are not
    canonical live storage.
