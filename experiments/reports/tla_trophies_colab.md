# TLA trophy self-play on Google Colab (w64 and w160l8)

Exploratory. Demonstrate Etude 0.4.5's Google Colab provider by training one
seed each of the small (`w64`) and large (`w160l8`) TLA trophy models together
on a single two-slot Colab L4 runtime (`colab-l4-s2`) toward ~2 million games
per model.

## Setup

- **Recipe.** Weekend trophy default (`experiments/reports/tla_trophies_floor.md`):
  policy input v11, complete 7,859-pool TLA trophy training split with closed
  decklists, `focus_film=True`, `stream_norm=True`, damped policy learning at
  `tau_scale=0.1` (no forward-KL floor), `streams=256`, `transitions=64`,
  `rows=512`, `threads=4`, float32, bucketed shapes and live-token attention,
  raw checkpoints every 250 updates.
- **Horizon.** 7,500 updates (~269 complete games/update ≈ 2.02M games) mapped
  onto the 10,000-step reference damping curve (`IterationSchedule(steps=7500)`).
- **Models.** One seed (`40902`) each of `w64` (width 64, 2 layers, 4 heads)
  and `w160l8` (width 160, 8 layers, 8 heads), packed onto one `colab-l4-s2`
  runtime (`etude provider set colab --kind colab --storage r2 --usd-per-unit 0.10`,
  `etude ondemand add colab-l4-s2 --provider colab --gpu L4 --max 1 --slots 2 --startup-seconds 900 --delivery-seconds 600`).
- **Budget.** Initial launch capped at USD24 (`USD12` per job, 140 hours,
  up to 12 automatic attempts across Colab runtime resets).
