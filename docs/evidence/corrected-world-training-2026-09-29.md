# Corrected-world training runs — 2026-09-29

This record preserves the decisions and dated evidence for
[Bind training to the corrected Learn/Lesson world · ETU-81](https://linear.app/loopflow/issue/ETU-81).
Jack Heart requested the Task on 2026-09-29 and asked for it to run without
his review; he reviews the combined result in the comparison Task's design
review. The [operator guide](../local-trained-challenger.md) owns the
repeatable commands. Local `.runs/` paths locate original evidence; a
repository clone does not contain them.

This is a pipeline result. It is not a strength result, a human-play result or
a registered world freeze.

## What changed

- The runner builds every teacher game from the authored setup
  (`MatchHypers.authored`), so both decks carry their sideboards and Learn
  offers Take a Lesson, Discard and draw, and Decline. It previously passed
  main decks only, which left Learn without a Lesson to take.
- Teacher shards record the rules engine's own legal-offer count and decision
  kind at every decision. Admission compares them with the encoded rows.
- `train_search_supervised` accepts the observation configuration the shards
  were encoded with and rejects any other shape. It previously built a default
  observation space regardless of the data.
- The runner records world identities before work starts and binds them into
  the exported candidate.
- After export the runner loads the candidate through the play server's
  configured-opponent path and plays one complete game per deck assignment.

## Decisions made without review

1. **Recipe.** The runner now follows the distillation starting point in the
   [approved plan](../plans/strongest-manabot-training.md): determinized PUCT
   with 64 simulations over four worlds, visit-distribution targets, policy
   weight 1, value weight 0, width-64 policy, batch 128, 10% whole-game
   validation. It replaces ETU-79's flat search-64 behavior cloning. The two
   ETU-79 checkpoints stay frozen as old-world evidence.
2. **Shard writer.** Teacher games run through `generate_selfplay_shard`, which
   calls the same `determinized_puct` search as `run_teacher_game` and already
   writes observations, visits and provenance. Shards carry positional engine
   order. They do not carry semantic offer IDs or Commands.
3. **World version.** Runs record `w3` and the dims tuple 28 / 39 / 24 / 16
   (player / card / permanent / action types). [WORLDS.md](../../WORLDS.md)
   still lists w2 as current. Registering the freeze stays with ETU-75.
4. **Lesson pool.** Recorded as both decks' authored sideboards.
5. **Learn coverage.** Admission fails when no Learn decision offered a Lesson.
6. **Demo check.** It drives `GameSession` directly against a seeded random
   player, without a browser or websocket.

## Runs

All runs used one CPU thread on the M4 Max, the default 600-second and 4 GiB
caps. The first three ran on commit `48910ab` plus this Task's then-uncommitted
changes; `etu81-gate-1` repeated the smoke on commit `571093a` after the
trainers, runner and tests came to share one shard shape check and
teacher-game call. New cloud spend was $0; electricity cost is unknown. Every
attempt is listed.

| Run directory under `.runs/` | Games / epochs / seed | Result | Wall seconds | Peak RSS bytes | Checkpoint SHA-256 |
| --- | --- | --- | ---: | ---: | --- |
| `etu81-smoke-1` | 2 / 2 / 81 | complete | 73.743 | 2522202112 | `769df5b73b67d78b8f192901c509c151307f75b39988420ec4f1dec0c2817f2f` |
| `etu81-corrected-1` | 8 / 8 / 81 | complete | 200.169 | 2842673152 | `566a134d45f0c1d336993d1181aa3d1ed03632b956dbeed5739f54cd0c1925b7` |
| `etu81-corrected-2` | 8 / 8 / 82 | complete | 138.692 | 3395928064 | `b1f7a7aadc667099a9cc70703cb0ddca4c1e777a6341a6f59ccce7da9dbab4f9` |
| `etu81-gate-1` | 2 / 2 / 81 | complete | 58.349 | 2502393856 | `7d2b3cf31496978832dfde97b3f1c2a4a823d3c2ca0f4e1c874fe5e1b552737a` |

The smoke run preceded the receipt's `demo_check_wall_seconds` field; its
training and admission code matched the later runs. Timings are workflow
costs on a machine doing other work, not controlled measurements. Peak RSS
includes the demo check's server process and is higher than ETU-79's 1.5 GiB.

| Phase | `etu81-corrected-1` | `etu81-corrected-2` |
| --- | ---: | ---: |
| Teacher seconds | 166.27 | 117.90 |
| Teacher decisions | 993 | 824 |
| Teacher decisions per second | 5.97 | 6.99 |
| Training seconds | 17.55 | 14.08 |
| Demo check seconds | 15.35 | 5.56 |

## Admission

| Measure | `etu81-corrected-1` | `etu81-corrected-2` |
| --- | ---: | ---: |
| Decisions | 993 | 824 |
| Engine-legal choices | 3148 | 2450 |
| Omitted legal choices | 0 | 0 |
| Decisions differing from the engine | 0 | 0 |
| Largest decision (128 rows available) | 13 | 14 |
| Learn decisions | 18 | 17 |
| Learn decisions offering a Lesson | 18 | 17 |
| Lesson offers | 52 | 47 |
| Omitted Learn choices | 0 | 0 |
| Search continuation cap hits | 0 of 63552 | 0 of 52736 |

`etu81-gate-1` admitted 201 decisions and 605 engine-legal choices with none
omitted; its 4 Learn decisions all offered a Lesson (12 offers), and 0 of
12864 search continuations hit the cap.

All 16 teacher games of the two eight-game runs reached an authoritative
winner. Training rows contain
the acting player's observation, the legal mask, visits, the chosen action,
seat and game outcome. They contain no opponent hand or library contents.

## Identities

Identical across all four runs:

| Identity | SHA-256 |
| --- | --- |
| Rules runtime (`_managym.cpython-312-darwin.so`) | `0f41ceca339a1d54c0ca7e6eba07b5ec6a32793d038762b723fc67e987f733f2` |
| Content digest | `dca9ab421ff18dc33546a35a0555c9817c302267f8fe21ff75d247e1600ce124` |
| Compiled semantics IR | `7a70679a4f4952976a286c4a31ffa90318bd0569745aace26353348fe6ee1edd` |
| Compiled semantics source | `a8a8bdf74177b50afb8275e3a4de29ac1c8f4f96a10cda6cc8d3c1f8f2a65a2c` |
| Content manifest | `c43f0220ca2ad8f90e53ad6c290ad14af5f9d96a56e765ee40f80771388e041b` |
| Deck and sideboard setup | `445a01c2f5b83d6bcf43a96e6482ef53c97c04dfe4ebb24cef888546a6a84a2f` |
| Lesson pool | `47753949c945aa3ed12949e4e1d4ab1105a4ff642d68016858eb3ead51d560d2` |
| Observation/action ABI | `25fe494fd9e09249bde027c0c8265bb4611ed152bed7685eb70f9cc2df33461f` |

The Lesson pool is Accumulate Wisdom, Firebending Lesson and It'll Quench Ya!
for UR Lessons, and Fancy Footwork and Yip Yip! for GW Allies, one copy each.
The two eight-game runs archived the same 196 source files with equal
per-file digests. Their `sources.zip` digests differ because the archive
stores write times.

## Demo-path load

Each run loaded its candidate through `etude.opponent.configured_opponent`,
which checks the receipt, recipe digest, rules runtime and checkpoint digest.
`GameSession.new_game` then checked the content manifest and decks and built
the checkpoint opponent. All eight games completed.

| Run | Candidate deck | Attempt | Random player moves | Winner seat |
| --- | --- | --- | ---: | ---: |
| `etu81-smoke-1` | GW Allies | `TK6HuqYdplCckudQz-r5fQ` | 132 | 1 |
| `etu81-smoke-1` | UR Lessons | `nBLdNpUYCWDiyVKEHjUiJw` | 51 | 0 |
| `etu81-corrected-1` | GW Allies | `Bilxw5XFAnJWr0RE6vAvGg` | 63 | 1 |
| `etu81-corrected-1` | UR Lessons | `Bsbi7L9gN9qERUyOSHVq4A` | 82 | 0 |
| `etu81-corrected-2` | GW Allies | `JuIpZ1e62cN7_Ybe9Q_E6Q` | 51 | 1 |
| `etu81-corrected-2` | UR Lessons | `tdEQY3C3TH-bfGiqpPG0mg` | 39 | 0 |
| `etu81-gate-1` | GW Allies | `4XwAx7E-rRoRY4ZedTHLrA` | 51 | 1 |
| `etu81-gate-1` | UR Lessons | `lYaD3rTog1yd8KXzpLg2XQ` | 57 | 0 |

The candidate sits in seat 1. It won four games and the random player won
four. Eight games against a random player measure nothing about strength.

## The student did not measurably learn

Held-out policy KL did not fall below its untrained value in either
eight-game run.

| Measure | `etu81-corrected-1` | `etu81-corrected-2` |
| --- | ---: | ---: |
| Validation policy KL before training | 0.0218 | 0.0197 |
| Validation policy KL after epoch 8 | 0.0222 | 0.0212 |
| Training policy loss, epoch 1 → 8 | 1.0357 → 1.0345 | 0.9897 → 0.9883 |
| Validation top-choice agreement, before → after | 0.574 → 0.519 | 0.443 → 0.475 |
| Validation decisions (one held-out game) | 54 | 122 |

In `etu81-corrected-1`, the teacher's most-visited choice received a mean
47% of visits where a uniform split would give 39%. Mean target entropy was
0.997 nats against 1.032 for uniform. Each run took about 56 optimizer steps.

Two explanations are untested: the 64-simulation visit targets are too close
to uniform to teach from, or 56 steps on under a thousand rows is too little
training. These checkpoints prove the pipeline runs. They should be treated
as near-initialization policies until a comparison shows otherwise.

## Checks

- Focused: `uv run pytest tests/sim/test_train_challenger.py
  tests/sim/test_search_supervised.py tests/sim/test_distill.py
  tests/sim/test_mcts.py -q` reported **27 passed** in 14.32 seconds.
- Affected suites: `uv run pytest tests/sim tests/env tests/etude
  tests/planning -q` reported **564 passed, 16 failed** in 308.10 seconds.
  The same 16 tests fail on unmodified `48910ab`; this Task added none.
  Fifteen remain: frozen-contract and receipt drift guards in
  `tests/sim` (5) and checkpoint-advice, authored-match, public-commitment
  and combat-tape parity checks in `tests/etude` (10). Frozen evidence was
  not changed.
- The sixteenth, `test_full_game_vs_checkpoint_villain`, failed because it
  bound its opponent to decks without sideboards and the server answered
  "Trained opponent is incompatible with the current world." The test now
  uses the authored setup and passes. That rejection is the server refusing
  an old-setup candidate, which is the intended behavior.
- On commit `571093a` plus the receipt change below, the focused files and
  `tests/etude/test_play_modes.py` reported **38 passed, 1 failed**; the
  failure is the combat-tape parity check already counted above.
- If the supervising process fails while the demo check runs, the runner now
  replaces the provisional complete receipt with a failed one before exiting.
- Ruff lint and format pass on the six changed Python files.
- Rust was not changed, so no debug `cargo test` was run.

## Open

- ETU-75 owns registering w3 in WORLDS.md and ordinary checkpoint loaders
  that reject old-world checkpoints by world identity. The play server's
  content-manifest and rules-runtime checks are what reject a mismatched
  candidate today.
- Shards do not carry semantic offer IDs or Commands.
- `sources.zip` is not byte-reproducible.
- No strength, improvement or human-play claim follows from these runs.
