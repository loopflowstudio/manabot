# Paired evaluation throughput

Exploratory calibration for MTG-497. The comparison measures completed,
replay-verified games per hour through the full paired arena. The first
measurements ran at `08a854747f` on 2026-11-27 on an Apple M4 Max
(16 CPU cores, 40-core GPU, 128 GiB unified memory, macOS 26.7.4, PyTorch
2.15.0).

## Workload

- The first 32 held-out TLA trophy deals from `tla_trophies`, deal seeds
  40900–40931. Each deal crosses seats and decks for four games against the
  heuristic under closed decklists. Every arm must finish all 128 games.
- The trained `w160l8` (6,490,722 parameters, sha256 `26ff27b76cff...`) is the
  raw update-2000 checkpoint from `tla_trophies_colab_500k_v2/w160l8@40902`.
  It scores 113–0–15 against the heuristic across 26,601 total game decisions
  (11,774 checkpoint decisions, 44.3%).
- The seeded `w384l8` (33,752,450 parameters, sha256 `d89c4ee7c0cb...`) uses
  the same eight-layer frontier model with width 384 and random seed 1. As an
  untrained policy it loses quickly (13–0–115 against the heuristic) across
  25,609 total game decisions (7,861 checkpoint decisions, 30.7%), so its
  games/hour reflects shorter policy-acting sequences rather than trained game
  length.
- Each arm runs sequentially in one process with one Torch CPU thread and the
  real evaluation loader. The Apple arms use float32. Model loading and deal
  construction are outside the timer. Timing includes arena admission,
  decision preparation, gameplay, recording, replay verification, and writing
  retained evidence.
- Forward time includes host-to-device transfer and synchronous readback of
  logits. Replay time comes from the arena's replay receipts. Other arena time
  is the remaining wall time, including observation preparation, viewer frames,
  and native steps. Forward batch size counts only decisions served by the
  checkpoint. A 256-slot arm on this 32-deal corpus has at most 128 live games.

## Apple Silicon results

Every Apple arm finished 128/128 games with 128/128 replay passes and exact
leg-for-leg parity (identical winners, scores, decision counts, and checkpoint
forward row totals) against CPU scalar.

### Trained `w160l8` (6.49M parameters, 11,774 policy rows)

| Arm | Slots | Games/hr | vs CPU-1 | Wall (s) | Forward (s) | Fwd rows/s | Other arena (s) | Replay (s) | Fwd calls | Mean / max rows | Limiting stage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CPU scalar | 1 | 749 | 1.00× | 615.2 | 362.5 (58.9%) | 32.5 | 152.8 (24.8%) | 99.9 (16.2%) | 11,774 | 1.00 / 1 | `forward` |
| CPU vector | 64 | 857 | 1.14× | 537.4 | 346.9 (64.5%) | 33.9 | 96.0 (17.9%) | 94.5 (17.6%) | 746 | 15.78 / 39 | `forward` |
| MPS scalar | 1 | 1,256 | 1.68× | 366.9 | 119.1 (32.5%) | 98.9 | 149.1 (40.6%) | 98.7 (26.9%) | 11,774 | 1.00 / 1 | `other_arena` |
| MPS vector | 16 | 1,942 | 2.59× | 237.3 | 44.7 (18.8%) | 263.4 | 99.1 (41.7%) | 93.5 (39.4%) | 1,873 | 6.29 / 13 | `other_arena` |
| MPS vector | 64 | 1,989 | 2.66× | 231.7 | 39.6 (17.1%) | 297.3 | 97.3 (42.0%) | 94.8 (40.9%) | 746 | 15.78 / 39 | `other_arena` |
| MPS vector | 256 | 1,978 | 2.64× | 233.0 | 39.8 (17.1%) | 295.9 | 97.0 (41.6%) | 96.2 (41.3%) | 629 | 18.72 / 69 | `other_arena` |

### Seeded `w384l8` (33.75M parameters, 7,861 policy rows)

| Arm | Slots | Games/hr | vs CPU-1 | Wall (s) | Forward (s) | Fwd rows/s | Other arena (s) | Replay (s) | Fwd calls | Mean / max rows | Limiting stage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CPU scalar | 1 | 782 | 1.00× | 589.2 | 346.8 (58.9%) | 22.7 | 144.7 (24.6%) | 97.7 (16.6%) | 7,861 | 1.00 / 1 | `forward` |
| CPU vector | 64 | 873 | 1.12× | 528.1 | 331.0 (62.7%) | 23.7 | 101.7 (19.3%) | 95.3 (18.1%) | 901 | 8.72 / 51 | `forward` |
| MPS scalar | 1 | 1,344 | 1.72× | 342.8 | 96.1 (28.0%) | 81.8 | 146.3 (42.7%) | 100.4 (29.3%) | 7,861 | 1.00 / 1 | `other_arena` |
| MPS vector | 16 | 1,747 | 2.23× | 263.7 | 61.9 (23.5%) | 126.9 | 106.0 (40.2%) | 95.8 (36.3%) | 1,934 | 4.06 / 14 | `other_arena` |
| MPS vector | 64 | 1,882 | 2.41× | 244.9 | 52.0 (21.3%) | 151.1 | 99.7 (40.7%) | 93.1 (38.0%) | 901 | 8.72 / 51 | `other_arena` |
| MPS vector | 256 | 1,841 | 2.35× | 250.3 | 50.9 (20.3%) | 154.5 | 102.6 (41.0%) | 96.8 (38.7%) | 734 | 10.71 / 96 | `other_arena` |

## Findings

- **Vector stepping on MPS moves end-to-end throughput by 2.41–2.66× over CPU
  batch-1** (and 1.40–1.58× over MPS batch-1): `w160l8` rises from 749 to
  1,989 games/hr and `w384l8` rises from 782 to 1,882 games/hr at 64 slots.
- **Batched MPS cuts forward time 6.7–9.2×.** At 64 slots, `w160l8` forward
  drops from 362.5 s (32.5 rows/s) to 39.6 s (297.3 rows/s, 17.1% of wall
  time), and `w384l8` forward drops from 346.8 s (22.7 rows/s) to 52.0 s
  (151.1 rows/s, 21.3%). On single-threaded CPU, vector batching barely changes
  forward throughput (32.5 → 33.9 rows/s at `w160l8`; 22.7 → 23.7 rows/s at
  `w384l8`), so CPU vector-64 gains only 12–14% overall.
- **Vector stepping cuts non-forward arena host time by ~31–37%** even on CPU
  (`other_arena` drops from 152.8 s to 96.0 s at `w160l8`, and 144.7 s to
  101.7 s at `w384l8`) by preparing collated v11 CSR `ObservationBatch` rows
  rather than encoding and admitting dense observation dicts one decision at a
  time.
- **Host arena work and replay verification now dominate (~79–83% of wall
  time).** Once MPS vector stepping reduces forward to 40–52 s, `other_arena`
  (97–103 s, ~41%) and sequential `replay` verification (93–97 s, ~38–41%)
  account for ~192–200 s of the ~232–250 s run. In a single process, further
  accelerator speedups above ~2,000 games/hr are capped by host recording and
  replay cost; multi-process game workers sharing one GPU agent process (Etude
  EXP-77 `play_many`) are needed to scale past that host ceiling.
- **64 slots gave the highest measured throughput on MPS.** Going
  from 16 to 64 slots increases mean rows per forward (6.29 → 15.78 at
  `w160l8`; 4.06 → 8.72 at `w384l8`) and trims forward time by 5–10 s. Going
  from 64 to 256 slots (capped at 128 active games by the 32-deal corpus)
  leaves wall time flat within run noise. Mean batch size stays below slot
  count because the game process picks heuristic actions separately (leaving
  44.3% of `w160l8` decisions and 30.7% of `w384l8` decisions on the checkpoint)
  and active slots drain at the end of the 128-game batch.

## MLX on Apple Silicon

At `d2012147d4` (2026-11-27), MLX 0.32.4 ran both checkpoints in float32
with 64 vector slots on the same M4 Max. The comparison reuses the CPU
scalar and MPS vector-64 receipts above, with the same checkpoint bytes and
32 deals per arm. Each arm ran once. The CPU and MPS timings come from an
earlier session on the same host.

| Checkpoint | CPU scalar games/hr | MPS vector-64 games/hr | MLX vector-64 games/hr | MLX vs CPU | MLX vs MPS | MLX wall (s) | MLX forward (s) | MLX other arena (s) | MLX replay (s) | MLX mean / max rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Trained `w160l8` | 749 | 1,989 | 2,033 | 2.71× | 1.022× | 226.7 | 32.7 | 98.6 | 95.4 | 15.78 / 39 |
| Seeded `w384l8` | 782 | 1,882 | 1,915 | 2.45× | 1.018× | 240.6 | 39.9 | 102.8 | 97.9 | 8.72 / 51 |

All 128 games per MLX arm passed replay. Every leg matched CPU scalar and
MPS vector-64 on deal and player seeds, actions, selected offers, frame and
state digests, winner, score, and decision count. The checkpoint forward row
counts also matched (11,774 and 7,861).

MLX spent 17.5% less time in forwards than MPS on `w160l8` and 23.4% less on
`w384l8`, reaching 360 and 197 policy rows per second, respectively. Arena
work outside the forward was the largest stage for both models. Together,
that work and replay took 83–86% of wall time, leaving a measured 1.8–2.2%
end-to-end gain over MPS across these single runs.

## RunPod L4

At `966e8ff296` (2026-11-27), the benchmark ran on one RunPod NVIDIA L4
with an AMD EPYC 9254 host, Linux 6.8, driver 595.112.07, and PyTorch
2.15.0+cu132. Both models used the checkpoint bytes from the Apple arms.
The on-pod CPU baselines and the four CUDA arms ran sequentially with one
Torch CPU thread. Comparisons in this section use the on-pod baseline for
each model at the same revision.

| Checkpoint | Arm | Games/hr | Wall (s) | Forward (s) | Other arena (s) | Replay (s) | Forward rows | Mean / max rows | Limiting stage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Trained `w160l8` | CPU scalar float32 | 293 | 1,570.7 | 1,353.8 | 164.8 | 52.1 | 11,209 | 1.00 / 1 | `forward` |
| Trained `w160l8` | CUDA vector-64 float32 | 3,648 | 126.3 | 24.0 | 58.5 | 43.9 | 11,209 | 18.59 / 36 | `other_arena` |
| Trained `w160l8` | CUDA vector-64 bf16 | 3,192 | 144.4 | 40.5 | 59.0 | 44.8 | 11,252 | 18.66 / 38 | `other_arena` |
| Seeded `w384l8` | CPU scalar float32 | 195 | 2,359.0 | 2,155.3 | 152.9 | 50.8 | 6,712 | 1.00 / 1 | `forward` |
| Seeded `w384l8` | CUDA vector-64 float32 | 3,380 | 136.3 | 29.5 | 64.9 | 42.0 | 6,712 | 9.91 / 54 | `other_arena` |
| Seeded `w384l8` | CUDA vector-64 bf16 | 3,908 | 117.9 | 12.6 | 63.6 | 41.7 | 6,712 | 9.91 / 54 | `other_arena` |

Float32 CUDA delivered 12.4× the CPU baseline throughput on `w160l8`
and 17.3× on `w384l8`. Forwards were the limiting stage in both CPU arms.
Arena work and replay took 72–89% of time across the four CUDA arms.
Bfloat16 forwards took 43% as long on `w384l8`, raising whole-run throughput
by 16%. On `w160l8`, bfloat16 forwards took 1.69× as long and throughput fell
by 12%. Each figure comes from one arm on one worker.
The [same-batch L4 profile](eval_precision_profile.md) identifies the attention
backend crossover and measures the spread across three repeats per arm.

All 768 games passed Command replay. Both float32 CUDA arms matched their
CPU baselines on all 128 action traces and scores, as did `w384l8` bf16.
The trained bf16 arm kept the same seeds and deck pairings and matched
117 traces and 127 scores. Both bf16 arms declared
`etude-policy-bf16-auto-v1` in their retained registrations.

Etude rental `mtg497-l4-d7fe257a1d5b` used `--max-usd 5` and ended at 18:15
UTC. Its launcher recorded USD 0.6441 billed plus approximately USD 0.1802
pending. RunPod confirmed the pod was gone. The local receipts and per-leg
comparison are under `.runs/eval-throughput/l4-966e8ff296/`.

## Gate 1 on Apple Silicon

The managed TLA trophy callback ran at `92a578b417` on the M4 Max with a fresh
native extension (sha256 `2dbae64fbde0...`). Each float32 arm used 64 slots,
one game process, and 15 inference threads. The panel used the same checkpoint
bytes and held-out deck pairings as above, under the managed `tla-trophies`
arena's seed derivation. The timer includes the Etude run and result delivery.
EXP-77 `c5923cf` used its shared socket with placement set by the harness because
its automatic accelerator detection is NVIDIA-only. 6db832 `49b085a` selected
its direct inference thread through ordinary Etude placement.

| Checkpoint | Transport | Backend | Games/hr | Wall (s) | Forward calls | Mean rows/forward | Forward round trips (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Trained `w160l8` | EXP-77 socket | MLX | 4,749 | 97.0 | 731 | 16.61 | 31.5 |
| Trained `w160l8` | EXP-77 socket | MPS | 4,009 | 115.0 | 731 | 16.61 | 46.7 |
| Trained `w160l8` | 6db832 direct | MLX | 5,091 | 90.5 | 731 | 16.61 | 26.0 |
| Trained `w160l8` | 6db832 direct | MPS | 4,344 | 106.1 | 731 | 16.61 | 41.2 |
| Seeded `w384l8` | EXP-77 socket | MLX | 4,930 | 93.5 | 559 | 12.11 | 34.6 |
| Seeded `w384l8` | EXP-77 socket | MPS | 4,377 | 105.3 | 559 | 12.11 | 46.9 |
| Seeded `w384l8` | 6db832 direct | MLX | 5,096 | 90.4 | 559 | 12.11 | 30.2 |
| Seeded `w384l8` | 6db832 direct | MPS | 4,607 | 100.0 | 559 | 12.11 | 40.8 |

All 1,024 games passed replay. Each model's four arms agreed on every deal and
player seed, selected action, offer, frame digest, state digest and final
score. The trained model won 115 of 128 games and used 12,140 policy rows in
each arm. The seeded model won 9 of 128 and used 6,770 rows.

Direct mode gained 3.4–8.4% over the socket at the same backend. At
`49b085a`, Etude pickled the `Prepared` dataclass and its arrays in-band,
totaling 4.13 GB per trained-model arm and 2.44 GB per seeded-model arm.

An initial run on the older local engine build produced 23 wall-clock
timeouts while other deals were being replayed. The regression test and
`92a578b417` move the slot budget to live decision rounds. The reported
measurements all use the rebuilt engine and the corrected clock.

### Gate 1 with zero-copy inputs

The M4 Max rerun used Etude 6db832 at `d876a7b`, Manabot's contiguous
preparation change `f3b00a846`, and the local engine extension (SHA256
`e2bab7ce2570…`). It kept the same checkpoint bytes, deal and environment
identity, 64 slots, and 15 inference threads. The first run with five field
fixes still copied `effects.detail_kinds` on rounds with live effects. The
measurements below include its fix.

| Checkpoint | Backend | Games/hr | Wall (s) | Mean rows/forward | Forward round trips (s) | Array transfers by pointer | In-band MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Trained `w160l8` | MLX | 4,818 | 95.6 | 16.61 | 31.4 | 56,979 | 14.23 |
| Trained `w160l8` | MPS | 4,522 | 101.9 | 16.61 | 41.3 | 56,979 | 14.23 |
| Seeded `w384l8` | MLX | 5,486 | 84.0 | 12.11 | 30.2 | 45,563 | 11.79 |
| Seeded `w384l8` | MPS | 4,865 | 94.7 | 12.11 | 42.0 | 45,563 | 11.79 |

All 512 games replayed and matched their original direct arms leg for leg
on seeds, actions, offers, frame and state digests, and scores. Every arm
recorded zero copied input arrays and zero socket bytes. A separate 64-row
native compact-attention input preserved the memory addresses of all 68
array leaves through the codec. In-band bytes carry the remaining Python
structure. Work outside forward round trips still took 56–67% of wall time.
The timings came from separate runs and do not isolate the effect of
transport on games/hour.

### Torch compilation on MPS

A Torch 2.15 probe used the trained `w160l8` checkpoint and prepared opening
observations with 12, 17 and 16 rows. Full-graph compilation failed at the
policy's binding check. Allowing graph breaks matched eager logits within
`1e-5`. The first compiled call took 19.0 s from a cold cache. Over 36 warm
calls with synchronous logits readback, eager took 1.315 s and compiled took
1.620 s. The managed loader uses eager execution.

## Reproduce

Install the evaluation and artifact dependencies with
`uv sync --extra evaluation --extra artifacts`. Add `--extra mlx` on Apple
Silicon for the MLX arm. While the lock uses Etude SDK 0.4.7, install an
editable Etude checkout that supports `play_many` with
`uv pip install --no-deps -e PATH`. Use `uv run --no-sync` to keep that
checkout in use.

Fetch the corpus and the trained policy into local files.

```bash
uv run --no-sync python -m manabot.data fetch experiments/data/mtg409/pools-v2/corpus.json
uv run --no-sync etude fetch 'tla_trophies_colab_500k_v2/w160l8@40902' \
  policies/00002000-raw.safetensors .runs/eval-throughput/policy
```

Use a fresh output directory for each arm. The default creates the seeded
`w384l8` checkpoint in that arm's directory. Reuse that file with
`--checkpoint` to hold the bytes fixed across arms. The driver uses the
trophy runner's 120-second game budget.

```bash
uv run --no-sync python -m experiments.eval_throughput \
  --output-dir .runs/eval-throughput/w384-mps-64 --device mps --runner vector --slots 64
uv run --no-sync python -m experiments.eval_throughput \
  --output-dir .runs/eval-throughput/w160-cpu-scalar --device cpu --runner scalar \
  --checkpoint .runs/eval-throughput/policy/00002000-raw.safetensors
uv run --no-sync python -m experiments.eval_throughput \
  --output-dir .runs/eval-throughput/w160-mlx-64 --device mlx --runner vector --slots 64 \
  --checkpoint .runs/eval-throughput/policy/00002000-raw.safetensors
uv run --no-sync python -m experiments.eval_throughput \
  --output-dir .runs/eval-throughput/w160-l4-bf16 --device cuda --runner vector --slots 64 \
  --precision bf16 --checkpoint .runs/eval-throughput/policy/00002000-raw.safetensors
```

On the L4 host, run the scalar CPU baseline and both CUDA precision arms with
each checkpoint, keeping the same checkpoint file and 32 deals. Use
`--precision float32` and a fresh output directory for the other CUDA arm.

The CLI writes `summary.json` alongside the per-deal diagnostics in `arena/`.
For the managed comparison, use `EvalRun.play_many` with
`experiments.tla_trophies.play_many` and `AGENTS`. The Gate 1 harness, receipts,
and per-leg comparison are under `.runs/eval-throughput/gate1-92a578b417/`.
The zero-copy rerun uses the same harness with the original environment digest
held fixed and an editable 6db832 at `d876a7b`. Its receipts and comparison
are under `.runs/eval-throughput/gate1-all-arrays/`.
