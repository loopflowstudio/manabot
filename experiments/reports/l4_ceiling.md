# L4 model-size ceiling study on TLA trophies

Exploratory. Find the largest BF16 model architecture that clearly trains to a
higher playing-strength ceiling on Google Colab NVIDIA L4 runtimes (24 GB VRAM,
53 GB RAM, 12 vCPU, ~2.33 CU/hr) compared against the completed `w160l8` (6.0M)
and `w64` (0.55M) 2,000-update runs from `tla_trophies_colab_500k_v2`.

## Setup

- **Recipe.** Trophy default (`experiments/tla_trophies_colab.py`): policy input
  v11, complete 7,859-pool TLA trophy split with closed decklists,
  `focus_film=True`, `stream_norm=True`, `tau_scale=0.1`,
  `IterationSchedule(steps=2000)`, `streams=256`, `transitions=64`, `rows=512`,
  `workers=4`, `threads=2`, `bucketed_shapes=True`, `live_tokens=True`, `bf16`
  on CUDA, raw checkpoints every 100 updates.
- **L4 size ladder (`experiments/l4_ceiling.py`).** All four scaled sizes use
  `head_dim=64`, explicit 4× FFN (`attention_feedforward_dim = 4 * hidden_dim`),
  and `bf16`:

| Size | `hidden_dim` | Layers | Heads (`d/H`) | `ff_dim` | Params | Precision | Role |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `w384l8` | 384 | 8 | 6 (64) | 1536 | 23.92M | `bf16` | ~4× `w160l8`, validated BF16 width |
| `w512l8` | 512 | 8 | 8 (64) | 2048 | 41.53M | `bf16` | ~7× `w160l8`, wider 8-layer |
| `w512l12` | 512 | 12 | 8 (64) | 2048 | 54.14M | `bf16` | ~9× `w160l8`, deeper 12-layer |
| `w768l8` | 768 | 8 | 12 (64) | 3072 | 91.19M | `bf16` | ~15× `w160l8`, upper L4 VRAM/compute probe |

## 6-update 4-L4 VRAM & throughput probe (`l4_ceiling_probe`, 2026-11-27)

All four runs succeeded at `e5a5069930` on solo Colab L4s (12 vCPU / 53.0 GiB
RAM / 24 GiB VRAM, `2.33 CU/hr`) for **3.06 CU total (~$0.31)**, with zero
non-finite values and zero rejected steps. Colab granted 2 concurrent L4s
immediately and queued the remaining 2 behind HTTP 503 until the first pair
completed.

Medians over post-warmup updates 2–6 (~62.4k–64.8k rows/update):

| Size | Params | Med s/upd | `step` | `prep` | `inf` | `learn` | GPU % | RSS | Peak VRAM (alloc / res) | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `w160l8` *(prior fp32 solo ref)* | 6.03M | 143–145 s | 10–12 s | 36–45 s | ~83–116 s *(inf+learn)* | | ~66% | 15.8 GiB | 1.46 GiB | — |
| `w384l8` (`bf16`) | 23.92M | **127.7 s** | 6.9 s | 15.1 s | 36.8 s | 63.8 s | 79% | 3.44 GiB | 2.40 / 3.11 GiB | 0.67 CU |
| `w512l8` (`bf16`) | 41.53M | **136.4 s** | 6.8 s | 15.3 s | 39.9 s | 70.0 s | 81% | 3.45 GiB | 3.19 / 3.34 GiB | 0.69 CU |
| `w512l12` (`bf16`) | 54.14M | **169.5 s** | 8.1 s | 17.0 s | 50.0 s | 88.5 s | 82% | 3.49 GiB | 3.89 / 4.26 GiB | 0.86 CU |
| `w768l8` (`bf16`) | 91.19M | **180.8 s** | 7.5 s | 16.5 s | 56.4 s | 94.7 s | 84% | 3.49 GiB | 5.29 / 6.08 GiB | 0.84 CU |

Key observations:
- Scaling 3.8× in parameters from `w384l8` (23.9M) to `w768l8` (91.2M) costs
  only **1.42×** more wall time per update and peaks at **6.08 GiB / 24 GiB**
  reserved VRAM.
- **Width is much cheaper than depth on L4:** `w512l8` (41.5M) is only +6.8%
  slower than `w384l8` (23.9M), and `w768l8` (91.2M, 8L) is only +6.7% slower
  than `w512l12` (54.1M, 12L) with +68% more parameters, because extra depth
  adds sequential per-layer launches on under-filled collection tail batches.
- Process RSS is ~3.5 GiB across all four sizes (down from ~15.8 GiB in Nov 23
  `w160l8` runs), leaving plenty of host RAM and VRAM headroom for Stage C
  multi-slot packing.

## Full ceiling study (`l4_ceiling`)

Launched across `colab-l4-4x` (`--max-usd 25`, 25 wall-hours / $6.25 ceiling per
size, `updates=1200`, `schedule_steps=2000` matching `tla_trophies_colab_500k_v2`
so winners can continue cleanly to u2000, `checkpoint_every=200`).

### Separating representational depth from throughput

Evaluation compares models **at equal updates** (pure learning capability,
independent of L4 wall speed) in addition to equal wall/CU budget:
- **Pure depth at fixed width:** `w512l12` (54.1M, 12L) vs `w512l8` (41.5M, 8L)
  at matched updates.
- **Width vs. depth at matched speed:** `w768l8` (91.2M, 8L) vs `w512l12`
  (54.1M, 12L) at matched updates.
- **Pure width at fixed depth:** `w384l8` → `w512l8` → `w768l8` (all 8L) at
  matched updates, plus behavior curves (combat selectivity, chump rates,
  off-turn play, mana use, per-card GIH WR) to see *what* deeper vs wider
  models learn differently. If 12L shows a per-update gain over 8L regardless of
  L4 step cost, follow up with a deeper arm (e.g. `w384l16` or `w512l16`).
