# TLA trophy self-play on Google Colab (w64 and w160l8)

Exploratory. Demonstrate Etude 0.4.5's Google Colab provider by training one
seed each of the small (`w64`) and large (`w160l8`) TLA trophy models together
on a single two-slot Colab L4 runtime (`colab-l4-s2`) toward ~2 million games
per model, and compare Colab throughput and cost against historical RunPod runs.

## Setup

- **Recipe.** Weekend trophy default (`experiments/reports/tla_trophies_floor.md`):
  policy input v11, complete 7,859-pool TLA trophy training split with closed
  decklists, `focus_film=True`, `stream_norm=True`, damped policy learning at
  `tau_scale=0.1` (no forward-KL floor), `streams=256`, `transitions=64`,
  `rows=512`, `workers=4`, `threads=2`, float32, bucketed shapes and live-token
  attention, raw checkpoints every 250 updates.
- **Horizon.** 7,500 updates (~269 complete games/update ≈ 2.02M games) mapped
  onto the 10,000-step reference damping curve (`IterationSchedule(steps=7500)`).
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

Testing live on Colab exposed three issues patched locally in `/Users/jack/src/etude.colab-keepalive` (`jack-heart/colab-keepalive` off `v0.4.5`):
1. **Missing keep-alive:** `ColabMachines.alive()` never sent `GET /tun/m/<endpoint>/keep-alive/` (`X-Colab-Tunnel: Google`), causing Google Colab to idle-prune session 1 (`5c95dd06d627`) at 21m 49s. Added 60s-throttled keep-alive inside `alive()`.
2. **HTTP 503 reconciliation wedge:** `ColabAPI.assign()` raised `ProviderUnavailable` on `503`, leaving unconfirmed sessions reconciling until `rental.deadline`. On `503`, `assign()` now verifies absence via deterministic `nbh` `lookup()` and raises `Refused`, and `loop.py` bounds unconfirmed-create reconciliation by `min(deadline, startup_deadline)`.
3. **Teardown flush race:** When Google dropped session 1 from `/tun/m/assignments`, the dying VM flushed one last append to `out/game-attempts/0000.jsonl` after session 2's `bootstrap.json` was snapshotted, causing restore to fail on changed file length.
