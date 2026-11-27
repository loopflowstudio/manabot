# TLA trophy self-play on Google Colab (w64 and w160l8)

Exploratory. Demonstrate Etude 0.4.5's Google Colab provider by training one
seed each of the small (`w64`) and large (`w160l8`) TLA trophy models together
on a single two-slot Colab L4 runtime (`colab-l4-s2`) to ~500k games per model,
and compare Colab throughput and cost against historical RunPod runs.

## Setup

- **Recipe.** Weekend trophy default (`experiments/reports/tla_trophies_floor.md`):
  policy input v11, complete 7,859-pool TLA trophy training split with closed
  decklists, `focus_film=True`, `stream_norm=True`, damped policy learning at
  `tau_scale=0.1` (no forward-KL floor), `streams=256`, `transitions=64`,
  `rows=512`, `workers=4`, `threads=2`, float32, bucketed shapes and live-token
  attention, raw checkpoints every 100 updates.
- **Horizon.** 2,000 updates (~257 complete games/update ≈ 514k games) mapped
  onto the 10,000-step reference damping curve (`IterationSchedule(steps=2000)`).
- **Models.** One seed (`40902`) each of `w64` (width 64, 2 layers, 4 heads)
  and `w160l8` (width 160, 8 layers, 8 heads), packed onto one `colab-l4-s2`
  runtime (`etude provider set colab --kind colab --storage r2 --usd-per-unit 0.10`,
  `etude ondemand add colab-l4-s2 --provider colab --gpu L4 --max 1 --slots 2 --startup-seconds 900 --delivery-seconds 600`).

## Colab GPU capacity & throughput benchmark (2026-11-23)

Direct `POST /tun/m/assign` probes at 19:35–19:39 UTC:
- **L4** (12 vCPU / 53 GB RAM / 24 GB VRAM): available in 2.9s, **`1.94 CU/hr`** (`$0.194/hr`).
- **T4** (std 2 vCPU / 13 GB RAM / 16 GB VRAM): available in 2.8s, `1.15 CU/hr` (`$0.115/hr`) — insufficient RAM for 15.5 GiB/process RSS.
- **A100** (std & High-RAM), **G4** (RTX PRO 6000), **H100**: all returned HTTP `503` (zero Colab capacity at probe time).

Four-update `w64 + w160l8` packed L4 comparison (`tla_trophies_colab` at
`workers=1, threads=4` vs `tla-colab-bench-l4-w4` at `workers=4, threads=2`,
updates 2–4 post-warmup, `$0.194/hr` = `1.94 CU/hr`):

| Config | Model | s/update (u2–u4) | `prep` | `step` | `inf + learn` | Games/hr | Dec/s | RSS | Peak VRAM |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `workers=1, threads=4` | `w64` | 122–140 s *(104s by u8)* | 36–41 s | 20–23 s | 53–62 s | 6,607–7,565 *(8,911 by u8)* | 427–490 | 15.4 GiB | 0.34 GiB |
| `workers=1, threads=4` | `w160l8` | 153–167 s | 35–38 s | 21–22 s | 84–94 s | 5,537–6,133 | 385–410 | 16.2 GiB | 1.46 GiB |
| `workers=4, threads=2` | `w64` | 109–138 s (mean 126s) | 36–45 s | **9.8–12.1 s** | 51–67 s | 6,699–8,458 | 451–539 | 15.6 GiB | 0.34 GiB |
| `workers=4, threads=2` | `w160l8` | 143–187 s *(143–145s solo u3–u4)* | 36–45 s | **9.6–12.0 s** | 83–116 s | 4,940–6,454 | 343–422 | 15.8 GiB | 1.46 GiB |

### Comparison vs historical RunPod throughputs

| Machine / setup | Rate | Aggregate dec/s | M decisions / USD |
| --- | ---: | ---: | ---: |
| **Colab L4 (`w64` + `w160l8`, `workers=4`, `slots=2`)** | **$0.194/hr** | ~850–960 | **~15.8–17.8 M** *(rising ~20% as games shorten)* |
| RunPod L4 (2× `w64`, `tla_trophies.md` Arm B) | $0.69/hr | 1,307 | 6.82 M |
| RunPod L4 (3× `w64`, `tla_trophies.md` Arm B) | $0.69/hr | 1,979 | 10.33 M |
| RunPod A40 (2× `w160l8`, `training-update-time.md` C2) | $0.60/hr | 1,064 | 6.49 M |

- `workers=4` cuts native `environment_step_and_encoding` in half (**21–23s → 10–12s**) with zero extra VRAM or RSS.
- `observation_preparation` (~36–45s/update on single-threaded Python) remains the largest bottleneck (filed as Task `b75cdb` in `engine-performance`).
- Combined RSS is ~31.5 / 53 GiB (`slots=2` is the memory ceiling on Colab L4).

## Etude 0.4.5 Colab findings & patches

Testing live on Colab exposed three issues patched and merged upstream in
`loopflowstudio/etude#142` (on `2026-11-23`, off `v0.4.5`):
1. **21-minute disconnected-runtime timeout:** Google Colab drops runtimes with no open client WebSocket after ~21 minutes even when a kernel cell is `busy` (session 1 unpatched dropped at `21m 33s`; session 2 with `GET /tun/m/<endpoint>/keep-alive/` dropped at `21m 17s` — `colab-vscode` PR #718 confirms that TFE endpoint is now a no-op; session 3 adding `GET <proxy>/api/colab/resources` still dropped at `21m 34s`). Keeping a lightweight `KernelChannel` WebSocket open to `/api/kernels/<id>/channels` (30s pings, 45m proxy-token rotation, lazy re-attachment across restarts) plus a 30s kernel `last_activity` heartbeat cleared both the 21-minute wall and the 1-hour proxy token expiry cleanly across two consecutive full 24-hour sessions.
2. **HTTP 503 reconciliation wedge:** `ColabAPI.assign()` raised `ProviderUnavailable` on `503`, leaving unconfirmed sessions reconciling until `rental.deadline`. On `503`, `assign()` now verifies absence via deterministic `nbh` `lookup()` and raises `Refused` (verified live during both 24-hour failovers).
3. **Dead-startup idle VMs & teardown flush race:** When Google dropped a session from `/tun/m/assignments`, the dying VM flushed one last append to `out/game-attempts/*.jsonl` after the replacement session's `bootstrap.json` was snapshotted, causing restore to exit with status 1 while the Colab runtime sat idle until `startup_seconds` (15m). `ColabMachines.alive()` now checks `GET /api/kernels` and reports a booted runtime dead as soon as no active (non-`idle`) kernel remains.

## Full ~500k-game run (`tla_trophies_colab_500k_v2`, 2026-11-24 – 2026-11-26)

- **Freeze:** commit `82c5d1d1b7cf`, Etude experiment `tla_trophies_colab_500k_v2`
  (`r2://manabot/jobs/tla_trophies_colab_500k_v2`), `--max-usd 22`.
- **Models:** `w64` (552,770 learnable parameters; `d=64`, 2 layers, 4 heads)
  and `w160l8` (6,033,122 learnable parameters, 10.9×; `d=160`, 8 layers, 8 heads),
  seed `40902`, `workers=4, threads=2`, packed onto `colab-l4-s2`.

### Training summary, reliability, and cost

Both runs completed all 2,000 updates with zero dropouts and zero manual
restarts across three consecutive Colab L4 sessions (with automatic failover
at Colab's 24-hour runtime cap):

| Run | Updates | Learner transitions | Active train time | Running hours | Colab sessions | Cost (CU / USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `w64@40902` | 2,000 | 95.73M | 42.55 h | 43.32 h | 2 | ~51.25 CU (**$5.13**) |
| `w160l8@40902` | 2,000 | 95.49M | 57.46 h | 58.90 h | 3 | ~77.60 CU (**$7.76**) |
| **Total** | **4,000** | **191.22M** | **100.01 h** | **102.22 h** | **3** | **~128.85 CU ($12.89)** |

- **Attempt 1 (`colab-l4-s2-0f8f46e6b4ee`):** `2026-11-24 05:58:13Z` → `2026-11-25 05:58:16Z` (**24h 00m 03s** uninterrupted), reaching `w64 u1183` / `w160l8 u893`.
- **Attempt 2 (`colab-l4-s2-bc7f770aeb2a`):** booted 6m after attempt 1 hit the 24h cap (after 3 fast `capacity_refused` probes); `w64` finished `u2000` at `2026-11-26 01:37:32Z`; `w160l8` ran the full **23h 57m** to `2026-11-26 06:01:51Z` (`u1643`).
- **Attempt 3 (`colab-l4-s2-53f3ce9fc7af`):** booted 2m 42s after attempt 2 hit the 24h cap (after 1 `capacity_refused` probe); `w160l8` ran solo on L4 for **11h 17m** (~32–36 updates/hr solo vs ~26–29 packed) and finished `u2000` at `2026-11-26 17:21:16Z`.
- **Cost efficiency:** **14.8M learner transitions / USD** (**~$6.45** per 500k-game run on average; `$12.89` total vs `$22` cap). All **68,836 held-out evaluation games** (`137,672` play rows in `statsdb`) ran on owned `mini-r2` at `$0.00`.

### Held-out TLA trophy evaluation (873 disjoint held-out pools)

Evaluated every 100 updates on disjoint held-out trophy deck pairs (`matchup_mode="cross"`, 4 games/pair) across five tracks:
- `u0–u400`: 32 pairs (128 games/point) on all tracks.
- `u500–u2000`: 160 pairs (634–640 games/point, ±3.8pp 95% CI at 50%) on heuristic, half-training, equal-updates, and equal-training-time; `u1000–u2000` at 160 pairs (636–640 games/point) on `-200 updates` (`u300–u900` at 32 pairs / 128 games).

#### Headline findings

1. **`w160l8` breaks 90% against the heuristic and leads `w64` by +5.4pp at the end.** `w160l8` peaks at **90.3%** (`u1900`, 639g) and averages **89.2%** across `u1600–u2000` (finishing **89.0%** at `u2000`), compared to `w64`'s **83.8%** average across `u1600–u2000` (finishing **84.7%** at `u2000`).
2. **Clear wall-clock crossover at 15–18 hours on Colab L4.** Despite running ~1.35× slower per update when packed, `w160l8` crosses over `w64` at equal active training time between `u600` (15.3h, **50.3%**) and `u700` (18.1h, **57.7%**). At virtually identical active training time (`w160l8 u1500` at 42.36h vs `w64 u2000` at 42.55h), `w160l8` wins **58.1%** head-to-head (639g) and leads by **+3.7pp** vs heuristic (`88.4%` vs `84.7%`). At equal updates (`u2000`), `w160l8` wins **61.2%** (639g; peaking at **62.3%** at `u1800`).
3. **`w64` essentially saturates past `u1000`, while `w160l8` is still gaining strongly at `u2000`.** Across the 11 checkpoints `u1000–u2000` (~7,032 held-out games each on fixed-lag `-200 updates`), `w64` averages **50.8%** per 200 updates (and decays to **53.4%** on `u2000 vs u1000` half-training), whereas `w160l8` averages **52.9%** per 200 updates — still posting **53.9%** (`u1900 vs u1700`) and **53.1%** (`u2000 vs u1800`) at the end, and **57.8%** on `u2000 vs u1000` half-training.

#### Full checkpoint progression (`u0–u2000`)

| Updates | `w64` train h | `w64` vs heur | `w64` vs ½u | `w64` vs −200u | `w160l8` train h | `w160l8` vs heur | `w160l8` vs ½u | `w160l8` vs −200u | `w160l8` vs `w64` (=upd) | `w160l8` vs `w64` (=time) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00 | 9.4% | — | — | 0.00 | 10.9% | — | — | — | — |
| 100 | 1.87 | 60.2% | — | — | 2.26 | 60.9% | — | — | 54.7% | 54.7% *(u100)* |
| 200 | 3.48 | 68.0% | 56.2% | — | 4.64 | 73.4% | 55.5% | — | 58.6% | 49.2% *(u300)* |
| 300 | 5.19 | 69.5% | — | 61.7% | 7.16 | 75.0% | — | 60.2% | 57.0% | 51.6% *(u400)* |
| 400 | 7.09 | 72.7% | 58.6% | 58.6% | 9.84 | 79.7% | 57.0% | 57.0% | 56.2% | 56.2% *(u500)* |
| 500 | 9.08 | 75.2% | — | 61.7% | 12.48 | 78.3% | — | 58.6% | 55.3% | 49.8% *(u700)* |
| 600 | 11.12 | 78.2% | 61.4% | 57.0% | 15.30 | 79.1% | 56.7% | 55.5% | 55.0% | 50.3% *(u800)* |
| 700 | 13.13 | 77.2% | — | 55.5% | 18.12 | 83.9% | — | 57.0% | 59.4% | 57.7% *(u900)* |
| 800 | 15.28 | 80.8% | 58.1% | 56.2% | 21.04 | 84.4% | 60.0% | 52.3% | 57.5% | 53.3% *(u1100)* |
| 900 | 17.54 | 78.7% | — | 54.7% | 23.98 | 84.0% | — | 50.8% | 56.6% | 55.6% *(u1200)* |
| 1000 | 19.78 | 80.1% | 58.4% | 50.3% | 26.84 | 87.8% | 62.0% | 54.8% | 57.3% | 59.5% *(u1300)* |
| 1100 | 21.90 | 82.0% | — | 51.7% | 29.73 | 87.6% | — | 53.4% | 60.3% | 58.9% *(u1400)* |
| 1200 | 24.12 | 83.1% | 55.9% | 50.9% | 32.69 | 87.9% | 63.2% | 52.8% | 60.5% | 55.3% *(u1600)* |
| 1300 | 26.30 | 82.2% | — | 50.0% | 35.99 | 87.8% | — | 55.2% | 61.1% | 57.9% *(u1700)* |
| 1400 | 28.61 | 82.9% | 57.9% | 50.5% | 39.14 | 87.9% | 60.0% | 53.6% | 61.3% | 59.5% *(u1900)* |
| 1500 | 30.85 | 83.4% | — | 53.0% | 42.36 | 88.4% | — | 50.3% | 59.2% | 58.1% *(u2000)* |
| 1600 | 33.22 | 83.7% | 55.1% | 50.6% | 45.76 | 89.0% | 60.8% | 51.3% | 60.6% | 56.7% *(u2000)* |
| 1700 | 35.55 | 82.9% | — | 50.2% | 48.75 | 89.0% | — | 52.7% | 59.8% | 57.7% *(u2000)* |
| 1800 | 37.85 | 84.0% | 54.6% | 51.1% | 51.49 | 88.9% | 58.6% | 50.5% | **62.3%** | 57.6% *(u2000)* |
| 1900 | 40.16 | 83.6% | — | 49.8% | 54.36 | **90.3%** | — | 53.9% | 59.4% | 60.1% *(u2000)* |
| 2000 | 42.55 | 84.7% | 53.4% | 50.8% | 57.46 | 89.0% | 57.8% | 53.1% | 61.2% | 61.2% *(u2000)* |

*(Parenthetical updates in the final column show the nearest `w64` checkpoint in `training_seconds`; once `w64` ends at `u2000` / 42.55h, `w160l8 u1500–u2000` all pair against `w64 u2000`.)*

### Operational & engine follow-ups filed during the run

- **`loopflowstudio/etude#142` (merged):** Colab `KernelChannel` WebSocket keep-alive, `503 -> Refused`, dead-startup detection.
- **`loopflowstudio/etude#146` (merged, Task `89b394`):** First batch of Colab & operational follow-ups.
- **`etude` Task `5e038f` (in progress):** Seated curve opponent-coverage breaks, 160-seating subtitle truncation, SSH liveness timeouts, chunked concurrent `statsdb._catch_up`, `Scheduler.wait()` adaptive cadence, and owned-machine `~/.etude/sessions` disk cleanup + ENOSPC recovery.
- **`manabot` Task `MTG-445` (`7dc919`):** `Decision::EnterAsCopy` stale object appearance panic at `managym/src/flow/decision.rs:1612` (`Sakashima of a Thousand Faces`, caught cleanly by `GameErrorPolicy` during `w64 u1311`; deterministic <1s local reproducer included).

## Steady-state training throughput (`tla_trophies_colab_2m`, 2026-11-23)

Over 40.7 minutes on session 4 (`colab-l4-s2-a066620b9e72`, `workers=4, threads=2`, `$0.194/hr`):
- **`w64@40902`** (`u41` → `u71`): **`81.4 s/update`** (~11,900 games/hr)
- **`w160l8@40902`** (`u25` → `u49`): **`101.8 s/update`** (~9,500 games/hr; 1.6× faster than init as games shorten)
- The initial `tla_trophies_colab_2m` pilot ran 91 / 64 updates for ~$0.47 before switching the horizon to 500k games (`tla_trophies_colab_500k_v2`) with `IterationSchedule(steps=2000)`.
