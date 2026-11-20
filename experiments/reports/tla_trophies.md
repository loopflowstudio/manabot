# TLA trophies

## Population audit, 2026-11-20

The retained 17Lands Premier Draft dump
contains 562,924 distinct draft/match pairs from 101,496 drafts. Of these
drafts, 8,763 have at least seven observed wins. The source SHA-256 is
`6bb12e5cf17e57a2624ca8ac83f15fd4415ea347aa58eac0ed82feb1b17d536b`.

| Observed losses through the seventh win | Drafts |
| --- | ---: |
| 0 | 1,135 |
| 1 | 3,240 |
| 2 | 4,383 |
| 3 | 5 |

The initial first-N derivation stopped at draft
`163f0b88025e494a8ab91cb30fa8bf37` (source data rows 134079–134088).
Its third loss was match 8 and its seventh win was match 10.
Scanning each draft to its final match also found 12 drafts with eight wins,
and 19 with exactly seven wins and a tenth match. These counts describe
different conditions and can overlap with the five above. First-N selection stops
reading a draft once it selects the seventh win, so excluding only those
five would still admit anomalous records.

A full-draft selection requiring exactly seven wins, at most two observed
losses, a final win and a final match number of at most nine yields 8,732
drafts, excluding 31 of the 8,763 candidates.

## Frozen training and evaluation inputs

The full-source builder selected 8,732 trophies from all 562,924 rows.
Each trophy retains the main deck and sideboard from its final winning
match. The pool split uses the existing salted draft-ID ranking across all
8,732 drafts. Playability was checked after assignment against every card in
each pool and human build.

| Split | Drafts | Playable pools |
| --- | ---: | ---: |
| Training | 7,859 | 7,859 |
| Held out | 873 | 873 |

A fresh checkout can reproduce both files from the retained source.

```bash
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg213/corpus-v1/games.csv.gz experiments/data/mtg213/corpus-v1/source.json
uv run python -m experiments.tla_corpus experiments/data/mtg409/corpus-v1 --source-dir experiments/data/mtg213/corpus-v1 --all-trophies
uv run python -m experiments.tla_pools experiments/data/mtg409/corpus-v1/corpus.json experiments/data/mtg409/pools-v2/corpus.json --held-out 873
```

GPU calibration, training and held-out evaluation remain pending.
