# TLA trophies

## Population audit, 2026-11-20

Derivation stopped before training. The retained 17Lands Premier Draft dump
contains 562,924 distinct draft/match pairs from 101,496 drafts. Of these
drafts, 8,763 have at least seven observed wins. The source SHA-256 is
`6bb12e5cf17e57a2624ca8ac83f15fd4415ea347aa58eac0ed82feb1b17d536b`.

| Observed losses through the seventh win | Drafts |
| --- | ---: |
| 0 | 1,135 |
| 1 | 3,240 |
| 2 | 4,383 |
| 3 | 5 |

The existing corpus builder stopped at draft
`163f0b88025e494a8ab91cb30fa8bf37` (source data rows 134079–134088).
Its third loss was match 8 and its seventh win was match 10.
Scanning each draft to its final match also found 12 drafts with eight wins,
and 19 with exactly seven wins and a tenth match. These counts describe
different conditions and can overlap with the five above. The builder stops
reading a draft once it selects the seventh win, so excluding only those
five would still admit anomalous records.

A full-draft selection requiring exactly seven wins, at most two observed
losses, a final win and a final match number of at most nine yields 8,732
drafts. A 10% holdout rounded to 873 would leave 7,859 before checking card
playability. These are audit counts. The planned corpus has not been published. GPU
calibration, training and held-out evaluation have no runs or costs to report.
