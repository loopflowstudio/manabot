# Paired evaluation throughput on Apple Silicon

Exploratory calibration for MTG-497 at commit `08a854747f` (2026-11-27). The
comparison measures completed, replay-verified games per hour through the full
paired arena on one Apple M4 Max host (16 CPU cores, 40-core GPU, 128 GiB
unified memory, macOS 26.7.4, PyTorch 2.15.0).

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
- Each arm runs sequentially in one process with one Torch CPU thread, float32,
  and the real evaluation loader. Model loading and deal construction are
  outside the timer. Timing includes arena admission, decision preparation,
  gameplay, recording, replay verification, and writing retained evidence.
- Forward time includes host-to-device transfer and synchronous readback of
  logits. Replay time comes from the arena's replay receipts. Other arena time
  is the remaining wall time, including observation preparation, viewer frames,
  and native steps. Forward batch size counts only decisions served by the
  checkpoint. A 256-slot arm on this 32-deal corpus has at most 128 live games.

## Results

Every arm finished 128/128 games with 128/128 replay passes and exact
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
- **64 slots saturates single-process MPS on this 128-game workload.** Going
  from 16 to 64 slots increases mean rows per forward (6.29 → 15.78 at
  `w160l8`; 4.06 → 8.72 at `w384l8`) and trims forward time by 5–10 s. Going
  from 64 to 256 slots (capped at 128 active games by the 32-deal corpus)
  leaves wall time flat within run noise. Mean batch size stays below slot
  count because heuristic turns run natively without a forward (leaving 44.3%
  of `w160l8` decisions and 30.7% of `w384l8` decisions on the checkpoint) and
  active slots drain at the end of the 128-game batch.
- **CUDA L4 float32/bf16 arms** remain for pass 4 once `etude-sdk > 0.4.7`
  ships `play_many`.

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

## Reproduce

Fetch the corpus and the trained policy into local files.

```bash
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg409/pools-v2/corpus.json
uv run --extra evaluation etude fetch 'tla_trophies_colab_500k_v2/w160l8@40902' \
  policies/00002000-raw.safetensors .runs/eval-throughput/policy
```

Use a fresh output directory for each arm. The default creates the seeded
`w384l8` checkpoint in that arm's directory. Reuse that file with
`--checkpoint` to hold the bytes fixed across arms.

```bash
uv run --extra evaluation python -m experiments.eval_throughput \
  --output-dir .runs/eval-throughput/w384-mps-64 --device mps --runner vector --slots 64
uv run --extra evaluation python -m experiments.eval_throughput \
  --output-dir .runs/eval-throughput/w160-cpu-scalar --device cpu --runner scalar \
  --checkpoint .runs/eval-throughput/policy/00002000-raw.safetensors
uv run --extra evaluation --extra mlx python -m experiments.eval_throughput \
  --output-dir .runs/eval-throughput/w160-mlx-64 --device mlx --runner vector --slots 64 \
  --checkpoint .runs/eval-throughput/policy/00002000-raw.safetensors
```

The CLI writes `summary.json` alongside the per-deal diagnostics in `arena/`.
