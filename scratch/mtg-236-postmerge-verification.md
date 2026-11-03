# MTG-236 post-merge verification — 2026-11-02

**Return acceptance to review; MTG-236 stays open.** The merged candidate's
full-score digests differ from pre-T3 in all three seeds and match a later
pre-optimization reference. Different runner CPUs prevent a timing comparison.
No cache defect or comparable timing regression is demonstrated. The original
<=4.34 s worst reply / <=128.7 s total acceptance remains unverified.

## Delivered source and applicable checks

PR #411 merged as `51fbacbbf82ad9fee5d6ed078d32ddfe7c3b0d51`.
At verification, fetched main and checkout `78129b059f934b329567ff3cc5f933526c79339d`
had the same complete tree, `827c696cd5e4ea5ce8a81a52652cb0fa2b20cebd`.
`WORLDS.md` is unchanged. Fresh checks on that source:

- `CARGO_PROFILE_DEV_OPT_LEVEL=1 cargo test -j 2 --manifest-path managym/Cargo.toml --test search_state_contract` — 4 passed: clone/event-drain Arc reuse, fork-local invalidation, stale references, privacy and witnesses.
- `uv run --extra dev --extra play pytest tests/academy/test_authored_match_parity.py tests/academy/test_public_commitment_parity.py -q` — 7 passed in 31.31 s through live, headless, replay and hypothesis producers; no receipt edits.

Final PR CI 41551039990 and package/install run 41551039837 passed;
publication was skipped. Main-push CI 41554039902 was still running at the
verification's last inspection; no later status is asserted here.
The historical design at `1e0343b25:scratch/mtg-236-derived-state-revisions.md`
contains the pre-T5 full debug/release passes and post-T5 focused passes.
Gate still owns the current full release suite. Shipping approval and subtype
deferral are retained in [decisions](questions.md).

The historical 31% user-CPU reduction and projected ~3.4 s / ~105 s predate
T5 integration and were measured under contention. They are not measured
acceptance. Local load at verification was 17.46 / 9.99 / 11.63; measurement
used CI, as the brief requires a quiet host. Ordinary main CI did not yet have
the Engine Budget job; MTG-237 / PR #413 owns it.

## Measurement contract

Both completed runs use MTG-237 workflow revision
`88dc8c1ad25cd99cca6026581a569f5c2f63ee1b`. Its `measure=<commit>` input
copies the benchmark into the exact selected source, builds release with
`--locked`, records source/machine/date, and uploads `engine-budget`.
It measures one candidate and skips reference comparison; separate runs
cannot establish a paired speedup, even with matching CPU models.
MTG-237's live Task/Graph and code were not changed.

Workload: original UR Lessons versus GW Allies main decks, both seats
Search-64 (16 worlds × 4 rollouts), 2,000-step cap, seeds 1–3, 40 decisions
each, native trivial-decision skipping enabled. The producer hashes every
ordered IEEE-754 action score and decision boundary. The original
`bench_search` digest hashes only the winning action and simulation count;
its agreement does not establish equality of losing scores.
The workflow also runs 512 uncapped games in `bench_steps`; that throughput
measure does not replace reply-latency acceptance.

Both profiles use GitHub-hosted Ubuntu 24.04, 4 vCPU, default Cargo release,
locked dependencies and Rust 1.99.1 (`58cf8f9c3`, 2026-10-13).
Receipts are dated 2026-11-03 UTC (2026-11-02 local).
The pre-T3 run started after the candidate completed.

```sh
gh workflow run ci.yml --ref jack/mtg-237-engine-performance-budget-in-ci -f measure=51fbacbbf82ad9fee5d6ed078d32ddfe7c3b0d51
gh workflow run ci.yml --ref jack/mtg-237-engine-performance-budget-in-ci -f measure=b73a6b5db33bd595fe53f75cea893bcb7c42e742
```

## Candidate receipt

[Run 41554442241](https://github.com/loopflowstudio/manabot/actions/runs/41554442241)
succeeded; `measurement.json` names merged source
`51fbacbbf82ad9fee5d6ed078d32ddfe7c3b0d51`.
CPU: Intel Xeon Platinum 8370C @ 2.80 GHz.
Artifact `engine-budget` ID `13193504640`, ZIP SHA256
`3a410d8e5359174f29fa607bcde9e64650589177fb952c5ba0927092a4f03a6b`.
Original files and job log: `.cache/mtg-236/candidate-run-41554442241/`.

| Seed | Decisions | Worst (s) | Median (s) | Total (s) | Full-score digest |
| --- | --- | --- | --- | --- | --- |
| 1 | 40 | 9.671 | 2.629 | 126.985 | 5192957023818819312 |
| 2 | 40 | 6.424 | 1.593 | 80.918 | 14020561159006433540 |
| 3 | 40 | 8.772 | 1.881 | 87.264 | 12455483057923843581 |

Overall worst **9.671 s**, total **295.167 s**. These exceed the numerical
target on this runner, whose hardware cannot be equated with the unspecified
original profile. Step receipt: 512 games, 99,923 steps, 9,592.9 steps/s,
digest `12247096877573042301`.

All three search digests match later pre-optimization reference
`e5615edcc7f6a7f1ef7adf83ce1a15f8302ac0b9` from
[run 41552162812](https://github.com/loopflowstudio/manabot/actions/runs/41552162812),
which ran separately on AMD EPYC 7763. This supports preserved search results
across the optimization; it is neither pre-T3 equality nor a timing comparison.

## Pre-T3 receipt

[Run 41555212674](https://github.com/loopflowstudio/manabot/actions/runs/41555212674)
succeeded; `measurement.json` names
`b73a6b5db33bd595fe53f75cea893bcb7c42e742`.
CPU: AMD EPYC 7763. Artifact `engine-budget` ID `13194686322`, ZIP SHA256
`099870bed79ac77e41e96f1a89cfb54065f33da08181a8cda7bf145b9710eb21`.
Original files and log: `.cache/mtg-236/pre-t3-run-41555212674/`.

| Seed | Decisions | Worst (s) | Median (s) | Total (s) | Full-score digest |
| --- | --- | --- | --- | --- | --- |
| 1 | 40 | 8.628 | 2.275 | 112.700 | 16088017085697099832 |
| 2 | 40 | 5.029 | 1.247 | 59.766 | 5775908583067295463 |
| 3 | 40 | 8.128 | 1.648 | 80.557 | 724794524755790472 |

Overall worst **8.628 s**, total **253.023 s**. Step receipt: 512 games,
100,523 steps, 12,733.3 steps/s, digest `6515799885026844005`.
Both receipts' exact commits, seed/decision counts and summary arithmetic
were checked. All three pre-T3 full-score digests differ from the candidate;
the step workload's count and trace also differ. This limits the historical
comparison; it does not attribute a regression to MTG-236.

## Remaining acceptance and ownership

Review must settle the equality reference and machine/build profile before a
qualifying comparison. Proposed distinction: use same-world pre-optimization
source `a8abf01b1a7014d5a59ed85f3223544d5140c172` for current-score preservation,
while retaining pre-T3 as the historical latency goal. **This proposal and any
replacement machine-specific bounds are not accepted.** Do not change game
results to match the older world.

The brief, PR #410 table and Task comments do not precisely identify the
original machine/build profile. Recover that profile or obtain prospective
acceptance of a replacement, then measure on one idle host using MTG-237's
producer. Optimize only if a valid comparison reveals a shortfall. Gate owns
the remaining current full release suite. PR #411 remains delivered history;
merge does not complete MTG-236.

## Delete — do not maintain

The shipped design's removals are complete: the single `TrackedState::revision`,
bare effect vector, linear LKI scans, cold clone cache and separate
`TriggerIndexCache`. No engine symbol or exclusive fixture remains to delete
in this evidence pass. Do not create a second benchmark implementation.
Repeated measurement conclusions now live only here; inherited T5/disclosure
instructions remain at `78129b059:scratch/questions.md`.

Before compression, uncommitted notes were saved under
`.cache/mtg-236/pre-compress/`; this note's original Git blob is
`2cfe9d57057499437db1235f32d9adf1ec113531`.
Validation: `git diff --check` passed; `lf context` is within both budgets.
Only notes changed, so applicable code checks above were reused. Both benchmark
runs are complete; this compression started no background processes.
