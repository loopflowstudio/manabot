# Trained policy self-play on RTX A6000 — 2026-10-27

Four concurrent games with a 5 ms batch collection window completed the
same eight-game panel in a median **27.21 s**, versus **68.75 s** for one active
game: **2.53× faster**. Fixed-input batches of four processed 64 retained
observations in **80.53 ms**, versus **279.27 ms** at batch one: **3.47× faster**.
Live dispatches averaged only **1.13 rows**, so the full-game gain combines
concurrency and occasional batching; it does not establish sustained batch-four
throughput or GPU saturation.

All **102 games** across two experiments completed and passed native replay.
All 12 retained-input cells completed. Both provider pods were verified absent
after cleanup. No further GPU run is reserved.

## Protocol and scope

Both seats use MTG-183 final raw u5950, SHA-256
`4f2ad656ba036a33d527212a7ac2badf2c5b9db136916f730846b07eef2fbea4`
(1,071,166 bytes), with separate player identities and policy RNG. Matchup:
Elemental world v2 UR versus GW, deal seeds 18801–18804, both legs. Each
experiment rotates three repeats of one single-game cell, eight games with one
active game, and eight games with four active games: 51 game opportunities.

NVIDIA RTX A6000, 49,140 MiB reported memory, driver 595.91.07, PyTorch
2.10.0+cu128, FP32, one Torch thread per worker, TF32 disabled. The Linux
native input is the digest-verified release artifact recorded by the launcher.
Protocols bind exact source, native, checkpoint, request, compute and runtime
bytes/facts. Each phase has a 900 s ceiling; each cell 300 s; each game 120 s;
process-tree RSS ceiling 8 GiB. These are finite real evaluation workloads,
without artificial resource pressure.

The existing resident inference service admits both seats against one model
load per cell. This experiment adds orchestration and permits the existing
5 ms collection window in the bounded runner; it does not replace serving.

## Actual batching treatment

Run `0e85d66534f3`, source `563c77f93882`, on one A6000 rental. Batch-one cells
have no collection delay; four-game cells allow at most four rows and 5 ms.
Whole-attempt wall includes preparation, execution, replay/evidence collection
and child cleanup, excluding only final ledger writes. It excludes rental
setup and delivery. Request latency is submission through supervisor reply
validation, excluding return to the game worker. Quantiles below are medians
of the three per-cell quantiles, not pooled quantiles.

| Treatment | Games across repeats | Median cell wall | Games/minute over summed whole-attempt wall | Request p50 | Request p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| One isolated game | 3 | 14.23 s | 4.23 | 15.99 ms | 18.88 ms |
| Eight games, one active | 24 | 68.75 s | 7.00 | 15.56 ms | 19.16 ms |
| Eight games, four active, 5 ms window | 24 | 27.21 s | 18.08 | 13.58 ms | 23.84 ms |

The four-game cells served 3,801 decisions in 3,351 forwards:
2,926 batches of one, 401 of two, 23 of three and one of four. Only 12.7% of
forwards contained multiple rows (23.0% of decisions). More concurrency or a
longer window is unmeasured; this is not a recommendation to change production
defaults. Tail request latency increased even while game throughput improved.

The first 64 candidate decision roots from a completed many-game cell were
replayed with exact observations, masks and row seeds. Each cell retained two
warmup passes and one measured pass, three repeats per batch size. Median
measured-pass wall was 279.27 ms at one row and 80.53 ms at four. That timer
includes output-witness writes, so it is not isolated GPU kernel time. The
measured fixed-input pass excludes admission and root reconstruction; their
cost remains in the retained whole-attempt wall.

All sampled actions agreed; maximum absolute legal-logit difference was
7.6294e-6, within declared atol 1e-5 / rtol 1e-4. Repeated passes within each
cell were exactly stable. All six many-game cells also have identical ordered
Command hashes, post-state digests and policy RNG seeds for each deal/leg.
This verifies these trajectories, not universal equivalence or playing strength.

## Zero-delay observation

Earlier run `4f8f34d188a7`, source `bb47aafe0d3c`, also completed 51 games and
six retained-input cells. It used four active games with **zero** collection
delay. Every live forward was batch one. Its eight-game cell medians were
47.85 s serially and 19.37 s with four active games; its fixed-input pass
medians were 267.68 ms and 76.07 ms (3.52×).

The two runs occupied different rentals, and their serial baselines differ
substantially. Compare treatments within a rental. The absolute difference
between the two rentals cannot isolate the cost of the 5 ms window. Preserve
the zero-delay result as a useful concurrency measurement, not a live batching
speedup.

## Startup, memory and spending

Dependency installation took 80 s in the first rental and 35.31 s in the
second; those are setup observations, not policy inference. In the first run,
many-game service readiness was about 1.44–1.49 s and model load 0.29–0.42 s;
readiness precedes model admission and overlaps its round trip. Neither is a
whole-rental startup measure. Job entry through admission, both phases and
archive creation was 275.73 s / 369.98 s respectively, excluding Etude setup,
delivery and its final self-write. Detailed separate spans are in Perf.

One-Hz whole-device memory samples peaked at 638 MiB / 370 MiB for the two
runs. These are sampled observations, not guaranteed maxima or utilization.
Torch allocator peaks remain separately recorded; no CUDA kernel profiler or
GPU saturation measurement was performed.

Rental estimates at verified deletion: **$0.062657 + $0.070798 = $0.133454**.
Both SDK receipts still mark these estimates. Together with $2.917706 finalized
prior Qwen rentals, cumulative rental spending is approximately **$3.05116**
within the original $10 cap. A $0.50 ancillary reserve remains; final inclusive
provider/storage billing is unavailable, so there is no finalized games/dollar
claim. No Qwen work ran in these measurements. Its continuation is MTG-191.

## Retained evidence

Immutable raw archives, original attempt manifests, GPU facts, job walls,
cleanup/cost observations, summaries and portable Perf reports are published
through `experiments/data.manifest.json` under:

- `experiments/data/inference-policy-gpu/4f8f34d188a7/`
- `experiments/data/inference-policy-gpu/0e85d66534f3/`

Fetch without executing a model:

```sh
uv run --extra artifacts python -m manabot.data fetch experiments/data/inference-policy-gpu
```

`policy-inference.tar.gz` contains the frozen protocols, all game/Command
traces, original manifests, individual repeats, batch histograms and warmup/
measured witnesses. `perf-report.tar.gz` contains `report/report.html` and its
relative evidence links. Display copies relocate the remote absolute serving
path and regenerate only their inventory; `report/relocation.json` records
those edits. Raw archive bytes and measured fields remain unchanged, and every
raw attempt inventory was reverified after report construction.

Software checks: 18 existing bounded-runner cases passed; the new trace fixture
passed after its missing required legs field was corrected. Three focused
trace/batch-window cases then passed. Ruff and diff checks passed. The artifact
untracked check passed. These checks are separate from the actual GPU results.

## Landing integration

The landing sync incorporates main's newer engine/observation changes and hosted
LLM provider support. The measurements above remain bound to their recorded
pre-sync source/native identities; they do not certify current-world checkpoint
admission or performance. No checkpoint or historical receipt is upgraded.
Current-world software validation is separate from these empirical results.
Further CPU/GPU comparisons, larger concurrency sweeps and Qwen efficiency
remain later work; the two displayed reports are the reviewed GPU baseline.
