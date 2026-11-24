# Baseline self-play throughput by backend on an M1 Mac mini

Which framework, device and precision runs the current baseline recipe's
self-play fastest on an Apple M1, and does that change with model width or
shape layout? This is an exploratory throughput measurement: no prediction,
no adoption threshold, no playing-strength claim, and no change to defaults.

## Answer

**Padded Torch CPU float32 delivers the most learner decisions per second of
active self-play at every measured width** among strictly admitted arms: 90.2
at w64l2, 39.5 at w256l2 and 25.3 at w384l2. Torch MPS float32 narrows the
gap from 0.48× of CPU at w64l2 to 0.81× at w256l2 and w384l2. At w384l2 Torch MPS bf16
passes strict parity in both layouts and reaches 22.8 padded (0.90×).

**With padded shapes, MLX float32 collects fastest at w256l2 and w384l2.** Its
collection rate is 202.4 decisions/s at w256l2 and 147.6 at w384l2, against
Torch CPU's 183.0 and 143.3. Its learning rate falls to 10.3 and 6.7
decisions/s, against Torch's 50–53 and 30–37, so MLX ends at 9.8 and 6.4
active, roughly a quarter of Torch CPU. MLX bf16 learns faster than MLX
float32 at w256l2 and w384l2, but every MLX bf16 cell failed the 1% relative
learner-loss gate (1.65%–10.03%) and is only approximate.

**On this M1, padded Torch learns 3–12× faster than bucketed Torch.** At
w64l2, Torch CPU spent 18.1 s in minibatches with padding and 177.8 s with
bucketing for the same 349 Adam steps. MLX's padded and bucketed
learning rates agree within about 10% at every width. Under bucketing, MLX float32
is the fastest strict arm (23.7, 9.8 and 6.4 active). The baseline enables
bucketing only on CUDA, which this result supports keeping.

**Observation preparation, policy inference and action selection account for
84.2%–95.3% of collection time across the 30 cells.** The Rust engine spends
1.0%–2.8% of collection time stepping games and encoding observations.

| Width | Best strict active (padded) | Best strict collection | Best strict learning |
|---|---|---|---|
| w64l2 | Torch CPU f32, 90.2 | Torch CPU f32 padded, 274.8 | Torch CPU f32 padded, 134.3 |
| w256l2 | Torch CPU f32, 39.5 | MLX f32 padded, 202.4 | Torch MPS f32 padded, 53.4 |
| w384l2 | Torch CPU f32, 25.3 | MLX f32 padded, 147.6 | Torch MPS bf16 padded, 37.1 |

All rates count learner decisions per second, pooled over updates 2–4.

## Method

`experiments/mini_baseline_backends.py` ran on `mini-heart` (Apple M1, 4
performance + 4 efficiency cores, 16 GiB, macOS 15.5, CPython 3.13.7, Torch
2.15.0, MLX 0.32.4) at revision `d7113fb78a76d06f0e355ff528f2b1dfca9a57fa`,
with nothing else scheduled on the host. For each width it captured one
ordinary CPU float32 padded baseline update at seed 25201, then checked
parity and timed each of the ten (layout, arm) cells serially in fresh
subprocesses, rotating arm order between blocks.

- **Recipe:** `manabot.recipes.baseline.settings` (Lessons/Allies, input v11,
  four streams, both seats, complete games, 128-update schedule), with
  `threads=4`, explicit `device`, `precision` and `bucketed_shapes`, and every
  arm starting from the same width's initial checkpoint. Width is the only
  architectural change (two layers, four heads).
- **Parity gate:** `manabot.backends.parity` on the first eight minibatches and
  eight spread-out collection calls of that capture. Float32 arms must pass
  every unchanged tolerance. A bf16 arm that passes the hard execution gate
  (verified device and dtype, finite values, gradients and step) but misses a
  tolerance is timed and marked `approximate (parity FAIL)`. No cell was
  rejected.
- **Timing:** tracked `self_play(updates=4)`; update 1 warms up and updates
  2–4 are read from Etude's `self_play.step` rows. Active rate is decisions
  over collection plus learning time; wall rate covers the whole worker.
  Live play diverges between backends, so each cell's games, decisions and
  Adam steps are listed beside its rates.

## Cost and limits

The run used about 2.9 hours of the owned mini (live timing 2.8 h, parity
0.1 h, captures 1.2 min) and no rented compute.

- Each cell has three measured updates (about 2,100–3,500 learner decisions),
  one attempt and one host. Per-update active ranges are shown; differences
  under about 10% are not distinguishable here.
- Workloads differ between cells because live games diverge, and learning
  time scales with retained rows and Adam steps (279–477).
- The host had about 6 GiB available and 3.5 GiB of swap in use before the
  first cell. Peak worker RSS stayed under 3 GiB.
- The w384l2 runner's parent process exited after cell 7, when its stdout
  pipe closed. Cells 8–10 ran from `w384l2/resume_w384.py`, which repeats
  `run_width`'s loop body by importing the same revision's module; its log is
  `w384l2/resume_w384.log`.
- Parity covers one capture per width. bf16 results marked approximate are
  speed readings only and establish nothing about optimization equivalence.

## Evidence

Receipts, Etude tracking rows and the three captures are in
`experiments/data/mini_baseline_backends/` (fetch with
`uv run --extra artifacts python -m manabot.data fetch experiments/data/mini_baseline_backends`).
The archive omits the per-cell `training/` checkpoints. To reproduce the
tables below, run

```sh
uv run python experiments/mini_baseline_backends.py report --out experiments/data/mini_baseline_backends
```

The command checks each cell against its parity receipt and Etude run
declaration. It also verifies the digest of each capture's file list.

Rates pool updates 2–4. Wall rate includes startup, warmup and shutdown. Policy rate counts collection preparation and inference. Rates use learner decisions per second except exp/s, which counts optimizer exposures.

## w64l2 on mini-heart

Apple M1, macOS-15.5-arm64-arm-64bit-Mach-O, Python 3.13.7, Torch 2.15.0, MLX 0.32.4. Four threads. Revision `d7113fb78a76d06f0e355ff528f2b1dfca9a57fa`. Capture run `0d0782f05fa64b77`, checkpoint `121d89ac4ca91d8325e8112b2c0c505c50010e03aed867e82c14a9df90ca227f`.

### Padded

| Arm | Parity | Collect | Policy | Engine | Learn | exp/s | Active (range) | Wall | Etude run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| torch-cpu-f32 | admitted | 274.8 | 299.3 | 2.8% | 134.3 | 33.6 | 90.2 (88.2–91.2) | 86.1 | 09171fac134a4122 |
| torch-mps-f32 | admitted | 127.1 | 141.4 | 1.5% | 65.3 | 16.4 | 43.1 (40.0–47.0) | 38.0 | fc60b0a9303d41d6 |
| torch-mps-bf16 | approximate (parity FAIL) learner loss 3.63% on batch 1 (0.0056568436 vs 0.0054512308) | 123.3 | 135.8 | 1.4% | 61.3 | 15.4 | 40.9 (38.6–43.6) | 35.6 | a072ef9d88aa40be |
| mlx-gpu-f32 | admitted | 189.4 | 225.0 | 2.3% | 27.3 | 6.8 | 23.9 (23.7–24.0) | 24.0 | 3613124134fe437f |
| mlx-gpu-bf16 | approximate (parity FAIL) learner loss 10.02% on batch 1 (0.0056568436 vs 0.0050897617) | 179.8 | 211.2 | 2.2% | 27.1 | 6.8 | 23.6 (23.2–24.6) | 23.7 | dcb5b33c220248a8 |

### Bucketed

| Arm | Parity | Collect | Policy | Engine | Learn | exp/s | Active (range) | Wall | Etude run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| torch-mps-f32 | admitted | 141.2 | 158.3 | 1.6% | 21.4 | 5.4 | 18.6 (17.3–20.1) | 18.0 | fa2b9456e51c463f |
| torch-mps-bf16 | approximate (parity FAIL) learner loss 3.63% on batch 1 (0.0056568436 vs 0.0054512308) | 136.0 | 150.4 | 1.5% | 20.3 | 5.1 | 17.6 (17.0–19.0) | 16.9 | 8f0f5fb816d04662 |
| mlx-gpu-f32 | admitted | 177.5 | 207.4 | 2.1% | 27.4 | 6.9 | 23.7 (23.5–23.9) | 24.0 | 265d45056ec44f4d |
| mlx-gpu-bf16 | approximate (parity FAIL) learner loss 10.03% on batch 1 (0.0056568436 vs 0.0050893519) | 162.3 | 188.6 | 1.9% | 27.3 | 6.8 | 23.4 (22.5–24.2) | 23.6 | 06091a674b104e2a |
| torch-cpu-f32 | admitted | 273.8 | 296.3 | 2.6% | 13.8 | 3.5 | 13.2 (12.5–13.7) | 13.5 | 5e588c1b8c4e4850 |

### Measured work

Games and learner decisions are listed in update order (2, 3, 4). Targets and minibatches are portions of learning time. Peak RSS covers the worker's entire lifetime.

| Layout | Arm | Games | Decisions | Collect range | Retained | Adam steps | Targets (s) | Minibatches (s) | Peak RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| padded | torch-cpu-f32 | 4, 4, 4 | 1111, 690, 660 | 255.2–303.1 | 25.0% | 349 | 0.19 | 18.08 | 921 |
| padded | torch-mps-f32 | 4, 4, 4 | 1077, 583, 850 | 123.0–135.8 | 25.1% | 361 | 0.32 | 38.07 | 1446 |
| padded | torch-mps-bf16 | 4, 4, 4 | 952, 586, 565 | 105.4–146.3 | 25.1% | 298 | 0.37 | 33.89 | 1410 |
| padded | mlx-gpu-f32 | 4, 4, 4 | 1052, 1167, 835 | 158.9–212.9 | 25.0% | 430 | 0.14 | 111.49 | 876 |
| padded | mlx-gpu-bf16 | 4, 4, 4 | 1052, 1167, 835 | 152.4–202.3 | 25.0% | 425 | 0.16 | 112.14 | 908 |
| bucketed | torch-mps-f32 | 4, 4, 4 | 1077, 583, 850 | 136.9–152.2 | 25.1% | 367 | 0.31 | 116.84 | 1887 |
| bucketed | torch-mps-bf16 | 4, 4, 4 | 1410, 633, 580 | 127.3–157.1 | 25.0% | 362 | 0.37 | 129.04 | 1896 |
| bucketed | mlx-gpu-f32 | 4, 4, 4 | 1052, 1167, 835 | 149.2–198.0 | 25.0% | 430 | 0.14 | 111.05 | 779 |
| bucketed | mlx-gpu-bf16 | 4, 4, 4 | 1052, 935, 736 | 143.0–192.8 | 25.0% | 381 | 0.16 | 99.46 | 721 |
| bucketed | torch-cpu-f32 | 4, 4, 4 | 1111, 690, 660 | 245.6–318.9 | 25.0% | 349 | 0.18 | 177.82 | 1122 |

## w256l2 on mini-heart

Apple M1, macOS-15.5-arm64-arm-64bit-Mach-O, Python 3.13.7, Torch 2.15.0, MLX 0.32.4. Four threads. Revision `d7113fb78a76d06f0e355ff528f2b1dfca9a57fa`. Capture run `742ac9b8ef2a4f5d`, checkpoint `848479b3c904048febf5e9f0e8848d7fd6a0322a4118dceef4996cc2b4ad4115`.

### Padded

| Arm | Parity | Collect | Policy | Engine | Learn | exp/s | Active (range) | Wall | Etude run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| torch-mps-bf16 | approximate (parity FAIL) full gradient | 77.2 | 82.6 | 1.1% | 51.8 | 13.0 | 31.0 (29.1–33.2) | 26.6 | 9decd8f8816d4ea3 |
| mlx-gpu-f32 | admitted | 202.4 | 237.5 | 2.2% | 10.3 | 2.6 | 9.8 (9.2–10.5) | 9.8 | aa7543ae4a8540b3 |
| mlx-gpu-bf16 | approximate (parity FAIL) learner loss 2.42% on batch 4 (-0.047226862 vs -0.04837136) | 183.1 | 214.2 | 2.1% | 14.9 | 3.7 | 13.8 (13.1–15.6) | 13.7 | d252942fabaa4c93 |
| torch-cpu-f32 | admitted | 183.0 | 195.5 | 2.1% | 50.4 | 12.6 | 39.5 (38.8–40.3) | 37.6 | f6f5ae47b7e64ee9 |
| torch-mps-f32 | admitted | 78.5 | 84.8 | 1.2% | 53.4 | 13.4 | 31.8 (30.6–33.3) | 27.8 | 2a3fdb6f9c644fd1 |

### Bucketed

| Arm | Parity | Collect | Policy | Engine | Learn | exp/s | Active (range) | Wall | Etude run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlx-gpu-f32 | admitted | 184.7 | 215.6 | 2.2% | 10.3 | 2.6 | 9.8 (9.2–10.8) | 9.8 | 4c630c2c159a4a8c |
| mlx-gpu-bf16 | approximate (parity FAIL) learner loss 2.08% on batch 6 (-0.20628426 vs -0.21057236) | 174.2 | 202.4 | 2.0% | 15.5 | 3.9 | 14.3 (13.7–14.7) | 14.2 | e63756369b2f432a |
| torch-cpu-f32 | admitted | 180.4 | 192.7 | 2.2% | 4.2 | 1.1 | 4.2 (3.9–4.4) | 4.1 | b2995d787f274c70 |
| torch-mps-f32 | admitted | 115.1 | 126.6 | 1.3% | 8.3 | 2.1 | 7.8 (7.2–8.4) | 7.6 | 1408436b08f442ae |
| torch-mps-bf16 | approximate (parity FAIL) full gradient | 86.8 | 94.2 | 1.1% | 8.4 | 2.1 | 7.6 (7.1–8.1) | 7.3 | 0132725071724ff1 |

### Measured work

Games and learner decisions are listed in update order (2, 3, 4). Targets and minibatches are portions of learning time. Peak RSS covers the worker's entire lifetime.

| Layout | Arm | Games | Decisions | Collect range | Retained | Adam steps | Targets (s) | Minibatches (s) | Peak RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| padded | torch-mps-bf16 | 4, 4, 4 | 924, 533, 590 | 72.6–82.7 | 25.1% | 279 | 0.72 | 38.70 | 1382 |
| padded | mlx-gpu-f32 | 4, 4, 4 | 1079, 1007, 645 | 186.1–217.5 | 25.0% | 379 | 0.29 | 263.60 | 1046 |
| padded | mlx-gpu-bf16 | 4, 4, 4 | 1130, 516, 612 | 165.4–244.2 | 25.0% | 311 | 0.30 | 151.04 | 968 |
| padded | torch-cpu-f32 | 4, 4, 4 | 716, 921, 498 | 176.8–195.7 | 25.1% | 299 | 0.53 | 41.77 | 1297 |
| padded | torch-mps-f32 | 4, 4, 4 | 812, 909, 595 | 70.4–84.9 | 25.0% | 305 | 0.60 | 42.73 | 1496 |
| bucketed | mlx-gpu-f32 | 4, 4, 4 | 1079, 1007, 861 | 166.3–205.8 | 25.0% | 410 | 0.29 | 284.26 | 989 |
| bucketed | mlx-gpu-bf16 | 4, 4, 5 | 1079, 790, 815 | 136.8–202.7 | 25.0% | 359 | 0.29 | 172.14 | 903 |
| bucketed | torch-cpu-f32 | 4, 4, 4 | 716, 810, 701 | 163.3–194.6 | 25.1% | 292 | 0.54 | 523.42 | 2083 |
| bucketed | torch-mps-f32 | 4, 4, 4 | 812, 945, 836 | 97.7–127.3 | 25.0% | 373 | 0.58 | 309.92 | 1776 |
| bucketed | torch-mps-bf16 | 4, 4, 4 | 1009, 909, 582 | 75.8–94.7 | 25.1% | 326 | 0.69 | 297.67 | 1795 |

## w384l2 on mini-heart

Apple M1, macOS-15.5-arm64-arm-64bit-Mach-O, Python 3.13.7, Torch 2.15.0, MLX 0.32.4. Four threads. Revision `d7113fb78a76d06f0e355ff528f2b1dfca9a57fa`. Capture run `00b66c343eeb4e00`, checkpoint `db25062d5717196c8618607d451a1c8f26a34b2c0a3cf703b487b26aafc29dfd`.

### Padded

| Arm | Parity | Collect | Policy | Engine | Learn | exp/s | Active (range) | Wall | Etude run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| mlx-gpu-bf16 | approximate (parity FAIL) learner loss 1.65% on batch 1 (-0.097998871 vs -0.096379526) | 140.7 | 161.1 | 1.8% | 8.8 | 2.2 | 8.3 (7.8–9.3) | 8.2 | 9f7d2d6f26f84416 |
| torch-cpu-f32 | admitted | 143.3 | 150.8 | 1.8% | 30.7 | 7.7 | 25.3 (23.9–27.0) | 24.3 | 8e9186a1e3154d0e |
| torch-mps-f32 | admitted | 64.4 | 68.8 | 1.1% | 29.9 | 7.5 | 20.4 (19.7–20.8) | 18.8 | 5899db6ec72047f0 |
| torch-mps-bf16 | admitted | 58.9 | 62.5 | 1.0% | 37.1 | 9.3 | 22.8 (22.3–23.7) | 20.0 | fcf3d729864648d3 |
| mlx-gpu-f32 | admitted | 147.6 | 172.0 | 1.8% | 6.7 | 1.7 | 6.4 (6.1–6.7) | 6.2 | 965b4064db494ffb |

### Bucketed

| Arm | Parity | Collect | Policy | Engine | Learn | exp/s | Active (range) | Wall | Etude run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| torch-cpu-f32 | admitted | 131.1 | 137.6 | 1.7% | 2.6 | 0.7 | 2.6 (2.0–2.8) | 2.5 | 4753bbe370b14775 |
| torch-mps-f32 | admitted | 92.7 | 100.6 | 1.1% | 5.0 | 1.2 | 4.7 (4.4–5.4) | 4.7 | 37e5981873d8408a |
| torch-mps-bf16 | admitted | 86.4 | 92.6 | 1.1% | 5.1 | 1.3 | 4.8 (4.7–4.8) | 4.6 | d181508df28b4569 |
| mlx-gpu-f32 | admitted | 132.0 | 151.5 | 1.7% | 6.8 | 1.7 | 6.4 (6.0–6.7) | 6.3 | 346d762f15634197 |
| mlx-gpu-bf16 | approximate (parity FAIL) learner loss 1.65% on batch 1 (-0.097998871 vs -0.096379526) | 145.6 | 168.1 | 1.7% | 9.7 | 2.4 | 9.1 (8.6–9.9) | 8.7 | 1ff8e062d3104d38 |

### Measured work

Games and learner decisions are listed in update order (2, 3, 4). Targets and minibatches are portions of learning time. Peak RSS covers the worker's entire lifetime.

| Layout | Arm | Games | Decisions | Collect range | Retained | Adam steps | Targets (s) | Minibatches (s) | Peak RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| padded | mlx-gpu-bf16 | 4, 4, 4 | 1363, 1040, 1098 | 129.4–150.7 | 25.0% | 477 | 0.46 | 396.15 | 1356 |
| padded | torch-cpu-f32 | 4, 4, 4 | 1481, 882, 958 | 134.4–154.9 | 25.1% | 432 | 0.86 | 107.35 | 1830 |
| padded | torch-mps-f32 | 4, 4, 4 | 924, 697, 873 | 57.0–72.0 | 25.1% | 361 | 0.76 | 82.69 | 1861 |
| padded | torch-mps-bf16 | 4, 4, 4 | 877, 970, 607 | 55.7–65.1 | 25.1% | 315 | 0.91 | 65.19 | 1765 |
| padded | mlx-gpu-f32 | 4, 4, 4 | 1149, 636, 520 | 121.7–191.8 | 25.0% | 308 | 0.49 | 344.59 | 1048 |
| bucketed | torch-cpu-f32 | 4, 4, 4 | 1481, 449, 547 | 113.8–169.4 | 25.1% | 310 | 0.97 | 942.26 | 2951 |
| bucketed | torch-mps-f32 | 4, 4, 4 | 936, 900, 827 | 86.5–98.9 | 25.0% | 395 | 0.77 | 533.12 | 2028 |
| bucketed | torch-mps-bf16 | 4, 4, 4 | 936, 821, 613 | 84.9–88.5 | 25.1% | 344 | 0.92 | 468.06 | 1871 |
| bucketed | mlx-gpu-f32 | 4, 4, 4 | 1149, 1126, 590 | 122.1–170.0 | 25.1% | 379 | 0.50 | 421.98 | 988 |
| bucketed | mlx-gpu-bf16 | 4, 4, 4 | 1119, 643, 822 | 124.0–193.4 | 25.0% | 322 | 0.47 | 265.53 | 1298 |

