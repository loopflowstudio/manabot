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

## 6-update 4-L4 VRAM & throughput probe (`l4_ceiling_probe`)

*Launching across `colab-l4-4x` (up to 4 concurrent solo L4s, 6 updates each).*

## Full ceiling study

*Pending 6-update probe results.*
