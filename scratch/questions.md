# MTG-155 assumptions (headless run, 2026-10-20)

- Catalog checkpoints play **sampled** (`deterministic=False`), matching
  `CheckpointVillain`'s default and how policies act during training; the
  setting is recorded on the trace. Candidate bundles keep their own setting.
- "Either deck and seat": the hero is always engine player 0; who plays first
  is the deal. The checkpoint plays whichever of its two bound decks the table
  gives it, so both deck assignments are supported. No separate seat control
  was added.
- Tournament entrant files resolve beside the Tournament JSON or in any
  directory listed in `ACADEMY_PLAY_CHECKPOINTS`, admitted only on the pinned
  SHA-256. The schema owner remains `experiments.runners.head_to_head`;
  the catalog reads only entrant identity fields.
- The configured candidate option changed value from `checkpoint` to
  `checkpoint:<sha256>` like every trained entry; the env-gated
  trained-challenger e2e spec was updated accordingly.
