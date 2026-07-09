# E0b — Cost Basis

Establishes the cost denominator for the north-star chart (strength vs
cumulative training cost). Three numbers: parameter count, measured training
throughput, and $/1M environment steps at the reference cloud rate.

**Pre-registered expectation:** training SPS (with learning) ≤ 2.0k, per
`reports/sps-closeout.md` (2,035 SPS with inference only — the backward pass
and PPO epochs can only lower it). **Result:** 637 SPS, ~3.2x below the
inference-only bound. Expectation held.

## 1. Parameter count

Method: instantiate `manabot.model.agent.Agent` with default `AgentHypers`
(`hidden_dim=64`, `num_attention_heads=4`, `attention_on=True`) and the default
`ObservationSpace` (`ObservationSpaceHypers` defaults), sum `p.numel()` over
`agent.parameters()`. Encoder dims at these defaults: player=26, card=29,
permanent=5, action=8, max_focus_objects=2.

| Config | Parameters |
| --- | ---: |
| Default (attention on) | **100,354** |
| Attention off | 50,306 |

All parameters trainable. Breakdown (attention on): attention block 50,048
(49.9%), action_layer 16,512, value_head 8,385, card_embedding 6,080,
player_embedding 5,888, action_embedding 4,672, perm_embedding 4,544,
policy_head 4,225. FLOPs/step was not measured in this pass (the wave doc
lists it; deferred — at 100k params it will not change any dollar figure
below).

## 2. Measured training throughput

Method: real training entrypoint (`python -m manabot.cli train --preset local`)
with full PPO learning enabled (rollout + GAE + 4 epochs x 4 minibatches per
update), default deck, `opponent_policy=passive`, seed 1, wandb disabled
(`experiment.wandb=false` + `WANDB_MODE=disabled`), profiler disabled. 12
updates of `num_steps=128` per run; steady-state SPS computed from the
wall-clock of updates 2–12 (update 1 excluded: warmup/compile — on MPS it is
~3.5x slower than steady state). No periodic eval fired (`eval_interval=100`
updates > 12).

| Shape | Device | Steady-state SPS |
| --- | --- | ---: |
| 16 envs x 128 steps (default) | **CPU** | **637** (per-update spread 632–642) |
| 16 envs x 128 steps | MPS | 584 |
| 8 envs x 128 steps (decision preset shape) | CPU | 621 |
| 4 envs x 128 steps (dev preset shape) | CPU | 520 |

Machine: Apple M4 Max, macOS 26.0.1, Python 3.12.12, torch 2.10.0. The
default device (`ExperimentHypers.device="cpu"`) is also the *faster* device
on this machine at this model size — MPS loses to CPU (584 vs 637 SPS),
consistent with kernel-launch overhead dominating a 100k-param model. Headline
number used below: **637 SPS (CPU, 16 envs)**.

Cross-check vs `reports/sps-closeout.md`: env-only 183k SPS, inference-only
2.0k SPS. Learning costs ~3.2x on top of inference; the env remains <1% of
the budget. Torch (inference + backward) is where all future throughput work
lives.

## 3. $/1M environment steps

Reference rate: AWS g5.xlarge on-demand, us-west-2, **$1.006/hr** (published
base on-demand rate per [Vantage](https://instances.vantage.sh/aws/ec2/g5.xlarge)
and [aws-pricing.com](https://aws-pricing.com/g5.xlarge.html), checked
2026-07-09; those listings quote us-east-1 — us-west-2/Oregon historically
matches it, assumed equal here). `ops/specs/job.yaml` caps spot at $1.50/hr;
that bound is also shown.

Accounting policy (decided with this report): **all runs stay local; the
ledger charges local wall-clock hours at the g5.xlarge on-demand rate.** No
instances are launched. Actual marginal cost of local runs is electricity —
roughly 40–60 W package power for ~0.44 kWh over the 20M-step run, i.e.
~$0.10–0.15 at $0.30/kWh, ~60x below the booked figure. The booked figure is
the honest market price of the compute, not the out-of-pocket cost.

At 637 SPS: 1M steps = 1,570 s = 0.436 hr.

| | Steps | Shape / SPS used | Wall-clock | @$1.006/hr (on-demand) | @$1.50/hr (spot cap) |
| --- | ---: | --- | ---: | ---: | ---: |
| **Per 1M steps** | 1,000,000 | 16 env / 637 | 0.44 hr | **$0.44** | $0.65 |
| first-light dev preset | 262,144 | 4 env / 520 | 0.14 hr | $0.14 | $0.21 |
| first-light decision preset | 1,048,576 | 8 env / 621 | 0.47 hr | $0.47 | $0.70 |
| hypothetical default run | 20,000,000 | 16 env / 637 | 8.72 hr | $8.77 | $13.08 |

Even the "big" default run books under $9 — the $100 wow-milestone budget is
~11 such runs, or ~200 decision presets. Training cost is not the constraint
today; measurement (eval games, search baselines) will dominate spend.

## Caveats

- **MPS/CPU throughput is not comparable to CUDA.** The g5 conversion assumes
  the A10G instance sustains the same 637 SPS. It has not been measured there;
  it could be several-x faster (GPU inference at larger batch) or slower
  (g5.xlarge has 4 vCPUs — the single-threaded Rust env and Python loop may
  lag the M4 Max). Re-measure on the actual instance before booking any cloud
  run; until then every dollar figure inherits this assumption.
- Preset costs exclude eval time. The first-light harness runs baseline /
  checkpoint / final evaluations (50–200 games each); those add wall-clock not
  counted in the steps denominator. Dev-preset dollar cost is therefore a
  floor.
- Single measurement day, one seed per shape; per-update spread at 16 envs was
  632–642 SPS (tight), but games/step mix can drift as the policy trains.
- SPS logged by the trainer is cumulative (includes warmup); steady-state here
  is recomputed from per-update wall-clock deltas.
- The repo's prebuilt `managym/_managym.cpython-312-darwin.so` was stale
  (predates `eb42cd1`, which added `EventData` — imports failed at HEAD). It
  was rebuilt from HEAD Rust source via `maturin build --release -i
  .venv/bin/python` and the wheel's `.so` copied into place (gitignored
  artifact; no source changes). The site-packages `managym.pth` still points
  at a deleted `/Users/jack/src/manabot.sps` — worth cleaning up.

## Provenance

- Git commit: `e38f20b` (main, tree clean apart from rebuilt gitignored
  extension artifact and this report)
- Machine: Apple M4 Max, macOS 26.0.1
- Python 3.12.12, torch 2.10.0, managym 0.2.0 (rebuilt from HEAD source)
- Date: 2026-07-09
- Runs: `manabot.cli train --preset local` with overrides
  (`train.num_envs`/`num_steps=128`/`total_timesteps` = 12 updates,
  `experiment.wandb=false`, `experiment.profiler_enabled=false`,
  `experiment.device` ∈ {cpu, mps}); param count script in session scratchpad.

## Next question

E0a/E0c remain in C0: what is the real surfaced-decision horizon per game, and
what are the clean baseline win rates (random-vs-random, untrained-vs-random,
untrained-vs-passive) with CIs — the denominators' counterpart numerators —
before C1 changes the game.
