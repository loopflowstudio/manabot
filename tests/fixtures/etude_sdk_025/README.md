# Retained Etude 0.2.5 evaluation fixture

Written on 2026-10-27 by the actual PyPI `etude-sdk[eval]==0.2.5` writer.
`input.json` freezes synthetic checkpoint references and production schedule
conditions for Elemental (heuristic full/subset grids and a checkpoint pair),
MTG-186 cards (one complete match), and the training benchmark (two deals).
Inputs use the pre-migration Agent schema. Checkpoint bytes do not exist.

Every participant 0 wins; both native seat assignments occur. Elemental and
MTG-186 retain both `player` (participant) and `seat` (native). The benchmark
retains `player` and `on_play`, matching its historical producer. No native
games, training, downloads or remote operations were performed.

`write_fixture.py` is an archival producer, not a current API consumer. Its
inline dependency pins the writer independently of the project's SDK. Tests
copy the existing run records and Parquet tables; they never regenerate them.
For an intentional new fixture, run the script in a fresh output directory:

```sh
uv run --script tests/fixtures/etude_sdk_025/write_fixture.py
```

The test recreates matches through current Manabot adapters and checks the
retained filename keys, zero newly played matches, unchanged match/table bytes,
participant scores, native-seat grouping and the real report/collector readers.
This is schema compatibility evidence, not a claim about policy strength.
