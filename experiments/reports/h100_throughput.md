# H100 throughput, bottleneck, and packing analysis on TLA trophies

Exploratory. Measure self-play throughput, phase bottlenecks, memory footprints,
and multi-run packing efficiency on Google Colab NVIDIA H100 (80 GB HBM3) across
an H100-appropriate model size ladder, to size subsequent H100 learning and
kickstarting experiments within a 20% credit budget (~488 CU of 2,440 CU).

## Setup

- **Recipe.** Same TLA trophy self-play setup as
  [`tla_trophies_colab.py`](../tla_trophies_colab.py): policy input v11,
  complete 7,859-pool TLA trophy split with closed decklists, `focus_film=True`,
  `stream_norm=True`, damped policy learning (`tau_scale=0.1`,
  `IterationSchedule(steps=2000)`), `streams=256`, `transitions=64`, `rows=512`,
  `workers=4`, `threads=2`, `bucketed_shapes=True`, `live_tokens=True`.
- **H100 size ladder (`experiments/h100_throughput.py`).** All scaled models use
  `head_dim=64`, explicit 4× FFN (`attention_feedforward_dim = 4 * hidden_dim`),
  and `bf16` (activating CUDA jagged FlashAttention and Hopper BF16 Tensor
  Cores). `w160l8` keeps its exact L4 frontier config (`fp32`, default 8× FFN)
  as the cross-GPU anchor.

| Size | `hidden_dim` | Layers | Heads (`d/H`) | `ff_dim` | Params | Precision | Role |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `w160l8` | 160 | 8 | 8 (20) | 1280 *(8×)* | 6.03M | `float32` | Colab L4 frontier anchor |
| `w384l8` | 384 | 8 | 6 (64) | 1536 *(4×)* | 23.18M | `bf16` | Validated BF16 bridge (~4× anchor) |
| `w512l12` | 512 | 12 | 8 (64) | 2048 *(4×)* | 53.15M | `bf16` | Mid-scale H100 workhorse (~9× anchor) |
| `w768l12` | 768 | 12 | 12 (64) | 3072 *(4×)* | 118.06M | `bf16` | Large H100 scale (~20× anchor) |
| `w1024l16` | 1024 | 16 | 16 (64) | 4096 *(4×)* | 258.92M | `bf16` | Upper-envelope H100 stress test (~43× anchor) |

## Colab account & H100 availability probes (2026-11-25)

- **Account balance & entitlements (`GET /tun/m/ccu-info` at 16:12 UTC):**
  - Balance: **2,439.25 CU** (20% = ~488 CU).
  - Active usage: 1 Colab L4 (`tla_trophies_colab_500k_v2`) at **2.33 CU/hr**.
  - `eligibleGpus`: `["H100", "G4", "A100", "L4", "T4"]`.
- **H100 assignment probes (16:07–17:42 UTC):**
  - `POST /tun/m/assign?variant=GPU&accelerator=H100` (standard shape):
    **HTTP 503 Service Unavailable** at 16:07, 16:17, 16:50, and 17:42 UTC.
  - `POST /tun/m/assign?variant=GPU&accelerator=H100&shape=hm` (`--high-mem`):
    **HTTP 400** (`Backend rejected accelerator 'H100'`) — H100 is single-shape
    on Colab, so Etude templates must omit `--memory-gb` (keep `high_mem=False`).
  - **Broader Colab GPU outage (16:50–16:52 UTC):** `H100`, `A100`, `G4`, `L4`,
    and `T4` all returned HTTP 503 on fresh assignment while the already-running
    L4 (`tla_trophies_colab_500k_v2`) stayed healthy and kept advancing.

## Results

*Queued on `colab-h100-s1` (`--max-usd 15`); waiting for Colab H100 capacity to
return.*
