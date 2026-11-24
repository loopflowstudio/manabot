# Exploratory TLA trophy w64 learnability trial: PLR + Vine-RLOO snapshots vs stored `tau-x2`

## Exploratory

**Four 400-update snapshot self-play runs completed cleanly on Colab L4 (`~$3.62` training; `~$7.01` total including all 1,888 CPU32 eval matches and one cancelled RunPod trial, well inside the USD 25 exploratory cap), and none of the three tested configurations beat pure `tau-x2` self-play head-to-head.** From scratch, `H=20` (80/20 snapshot/fresh starts) learned reliably (`0.055 -> 0.684` mean vs the first-64 heuristic across 2 seeds; `+5.5` to `+10.5 pts` vs itself at half updates) but consistently trailed retained `tla_trophies_floor/tau-x2` (`0.727` vs heuristic at u400; `0.449` mean H2H across 2,048 games at u100–u400: `0.447`, `0.455`, `0.451`, `0.443`). Warm-starting from `tau-x2@40902` update 400 split heuristic vs head-to-head behavior — `warm-h20` (`H=20`, 80/20) reached `0.813` vs heuristic at effective u800 vs `warm-full` (`H=None` terminal, 65/35) at `0.734` — yet both warm-start arms lost head-to-head to retained `tau-x2@40902` across all four u500–u800 milestones (`warm-h20`: `0.438`, `0.492`, `0.414`, `0.461`, mean `0.451` over 512 games; `warm-full`: `0.461`, `0.461`, `0.461`, `0.484`, mean `0.467` over 512 games).

This is an exploratory report. Its observations are operational and directional only and are not cited as findings.

## Question

On the complete 7,859-pool TLA trophy `w64` stack (`tla_trophies` + `tau_scale=0.1`, policy input v11, `focus_film=True`, `stream_norm=True`), can PLR + 4-branch Vine-RLOO snapshot self-play beat ordinary fresh self-play (`tla_trophies_floor/tau-x2` retained in R2) at matched updates — either from scratch or warm-started from a calibrated u400 critic — and how does truncation horizon (`H=20` bootstrapped vs `H=None` full terminal rollouts) affect critic calibration and playing strength?

Because this is an exploratory trial, it makes no committed prediction.

## Method

All arms are declared in [`experiments/tla_frontier_snapshots.py`](../tla_frontier_snapshots.py) and share the `tla_trophies` + `tau_scale=0.1` recipe (7,859 train / 873 held-out TLA trophy pools from `pools-v2`, closed decklists, policy input v11, `max_actions=64`, `action_overflow="first-legal"`, `focus_film=True`, `stream_norm=True`, `streams=256`, `transitions=64`, `rows=512`, `workers=4`, `threads=2`, float32 CUDA, bucketed shapes, live tokens, `checkpoint_every=50`, raw checkpoint evaluation, `root_actor_weight=2.0`, 512-slot PLR bank prioritized by root `critic_residual` with `score_half_life=8`, fresh-only capture at `min_legal_actions=3`, `probability=0.05`, `max_captures_per_batch=64`, `group_size=4`). No fresh baseline was trained; all comparisons reuse retained `tla_trophies_floor/tau-x2` checkpoints (`40901` u100–u400, `40902` u100–u1000) stored in R2.

| Arm | Start | Seeds | Horizon (`H`) | Start mix (snapshot / fresh) | Updates |
| --- | --- | --- | --- | --- | ---: |
| `plr-vine-rloo` | From scratch | `40901`, `40902` | `20` decisions | `80% / 20%` | 400 |
| `warm-h20` | `tau-x2@40902` u400 | `40902` | `20` decisions | `80% / 20%` | +400 (eff. u400–u800) |
| `warm-full` | `tau-x2@40902` u400 | `40902` | `None` (terminal) | `65% / 35%` | +400 (eff. u400–u800) |
| `warm-h60` *(cancelled)* | `tau-x2@40902` u400 | `40902` | `60` decisions | `75% / 25%` | 16 / 400 (stopped on RunPod per Colab-only GPU rule) |

Evaluation scores policy-only play on `tla_trophies`'s 32 held-out deals crossed across all 4 seat/deck legs (128 games per seed per checkpoint):
- Against the first-64 heuristic (`tla-snapshots-held-out-32`, `-warm-h20`, `-warm-full`, `-warm-h60`)
- Against each from-scratch line at half its updates (`tla-snapshots-half-training-32`)
- Head-to-head at matched effective updates against retained `tla_trophies_floor/tau-x2` checkpoints (`tla-snapshots-vs-tau-x2-32-v2` at u100–u400; `tla-snapshots-warm-vs-tau-x2-32` at effective u500–u800).

Library-shuffle common random numbers across sibling branches were not enabled in this pass.

## Results

### 1. From-scratch `H=20` (`plr-vine-rloo`, 80/20 starts) vs retained `tau-x2` (`u0 -> u400`)

All 128-game heuristic and half-training points completed for both seeds (`40901`, `40902`). From-scratch `H=20` climbed steadily (`0.055 -> 0.684` mean vs heuristic; `+5.5` to `+10.5 pts` vs itself at half updates) with zero failed or discarded snapshot branches, but trailed `tau-x2` at every milestone by 4–7 pts on heuristic and 4.5–8 pts head-to-head.

| Update | `tau-x2` heur mean (`40901` / `40902`) | `plr-vine-rloo` heur mean (`40901` / `40902`) | Delta (heur) | H2H vs `tau-x2` (512g/pt) | Half-training self |
| --- | ---: | ---: | ---: | ---: | ---: |
| u0 | `0.055` | `0.055` (`0.055` / `0.055`) | `0.0 pts` | — | — |
| u50 | — | `0.457` (`0.461` / `0.453`) | — | — | — |
| u100 | `0.586` (`0.570` / `0.602`) | `0.516` (`0.500` / `0.531`) | `-7.0 pts` | **`0.447`** (`40901`: `0.500`/`0.430`; `40902`: `0.438`/`0.422`) | `0.559` (`0.547` / `0.570`) |
| u150 | — | `0.566` (`0.609` / `0.523`) | — | — | — |
| u200 | `0.648` (`0.641` / `0.656`) | `0.609` (`0.633` / `0.586`) | `-3.9 pts` | **`0.455`** (`40901`: `0.453`/`0.477`; `40902`: `0.438`/`0.453`) | `0.559` (`0.555` / `0.563`) |
| u250 | — | `0.637` (`0.656` / `0.617`) | — | — | — |
| u300 | `0.695` (`0.719` / `0.672`) | `0.645` (`0.641` / `0.648`) | `-5.0 pts` | **`0.451`** (`40901`: `0.469`/`0.484`; `40902`: `0.430`/`0.422`) | `0.605` (`0.633` / `0.578`) |
| u350 | — | `0.680` (`0.664` / `0.695`) | — | — | — |
| **u400** | **`0.727`** (`0.734` / `0.719`) | **`0.684`** (`0.672` / `0.695`) | **`-4.3 pts`** | **`0.443`** (`40901`: `0.469`/`0.430`; `40902`: `0.430`/`0.445`) | `0.574` (`0.555` / `0.594`) |

### 2. Warm-start from `tau-x2@40902` u400 (`eff. u400 -> u800`, 128g/pt)

Starting from a calibrated u400 critic (`explained_variance ~ 0.93`, `0.719` vs heuristic) showed a split between heuristic and head-to-head strength: `warm-h20` tracked `tau-x2@40902` closely against the heuristic (`0.813` vs `0.824` at u800) and beat `warm-full` by `+7.8 pts` on heuristic (`0.813` vs `0.734`), while `warm-full` edged `warm-h20` slightly in head-to-head vs `tau-x2@40902` (`0.467` vs `0.451` mean over 512 games). Every single one of the eight 128-game warm-start H2H matchups against retained `tau-x2@40902` finished below `0.500`.

| Effective update | `tau-x2@40902` (heur) | `warm-h20` (heur) | `warm-full` (heur) | `warm-h20` H2H vs `tau-x2@40902` | `warm-full` H2H vs `tau-x2@40902` |
| --- | ---: | ---: | ---: | ---: | ---: |
| u400 (`+0`) | `0.719` | `0.719` | `0.719` | — | — |
| u450 (`+50`) | — | `0.750` | `0.664` | — | — |
| **u500 (`+100`)** | `0.776` | `0.758` | `0.719` | **`0.438`** (128g) | **`0.461`** (128g) |
| u550 (`+150`) | — | `0.734` | `0.766` | — | — |
| **u600 (`+200`)** | `0.771` | `0.750` | `0.758` | **`0.492`** (128g) | **`0.461`** (128g) |
| u650 (`+250`) | — | `0.781` | `0.719` | — | — |
| **u700 (`+300`)** | `0.735` | `0.695` | `0.719` | **`0.414`** (128g) | **`0.461`** (128g) |
| u750 (`+350`) | — | `0.789` | `0.727` | — | — |
| **u800 (`+400`)** | **`0.824`** | **`0.813`** | `0.734` | **`0.461`** (128g) | **`0.484`** (128g) |
| **Mean H2H (512g)** | — | — | — | **`0.451`** | **`0.467`** |

### 3. Training dynamics and the `H=20` vs `H=None` mechanistic split

All four completed runs had **0 failed branches and 0 discarded groups** across 400 updates on the native 4-worker `SnapshotVector` path.

| Run (at u400) | Fresh `decisions/game` | Critic `explained_var` | Critic `pred_std` | `their_9_9` shift | `my_9_9` shift | `block_gap` | Terminal branches / update | Completed groups / update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `plr-vine-rloo@40901` | `183.2` | `0.927` | `0.622` | `-2.88` | `+2.96` | `0.36` | `0–1 / 608` | `~152` |
| `plr-vine-rloo@40902` | `193.1` | `0.920` | `0.598` | `-2.55` | `+5.37` | `0.28` | `0–1 / 608` | `~152` |
| `warm-h20@40902` | `184.2` | `0.924` | `0.560` | `-4.27` | `+5.63` | `1.18` | `0–1 / 608` | `~152` |
| `warm-full@40902` | `166.8` | `0.932` | `0.645` | `-3.98` | `+6.03` | `-0.50` | `184 / 184` (100%) | `~45` |

Two structural mechanisms stand out:
1. **Horizon `H=20` is 100% critic-bootstrapped (`0–1 / 608` terminals) because 20 surfaced decisions across both seats is ~1.5–2 turns in a ~170-decision TLA game.** From scratch, bootstrapping off an uncalibrated early critic plus losing ~45% of fresh full-game rows explains the `-7.0 pt` u100 deficit and steady `~0.449` H2H lag. Once warm-started from a calibrated u400 critic, `H=20` climbed much higher against the heuristic (`0.813` vs `0.734` for `H=None`) by generating **~3.3x more Vine-RLOO groups/update (`152` vs `45`)** with low-variance `V(s_20)` returns, whereas `H=None` preserved sharper critic calibration (`pred_std=0.645` vs `0.560`, `ev=0.932` vs `0.924`) and was slightly steadier head-to-head against `tau-x2` (`0.467` vs `0.451`).
2. **Even warm-started, 65–80% snapshot replay starved fresh full-game diversity without enough variance reduction on siblings.** Without library-shuffle common random numbers across sibling branches, 4-branch rollouts diverge via random draws as well as root actions, so paying ~45–60% of the stream budget for snapshot continuations did not outperform 100% fresh `tau-x2` self-play at matched updates.

## What this supports

- **Operational readiness:** The native 4-worker `GameSnapshot` vector path (`6b4a1a`) is stable and fast on Colab L4 across 1,600 total training updates (`0` failed branches, `0` discarded groups, ~5s/update step overhead vs ~4.3s on `H=20`, ~$0.76–$1.22 per 400-update run).
- **Learnability direction (exploratory):** At high `replay_fraction` (`0.65–0.80`) without sibling CRN, PLR + Vine-RLOO does not beat `tau-x2` from scratch (`0.449` H2H across 2,048 games) or warm-started from u400 (`0.451` for `H=20`, `0.467` for `H=None` across 512 games each).

## Recommendation

Before investing in a full multi-seed finding trial, test three targeted changes on the warm-start u400 setup (where a 400-update Colab L4 run costs `~$1`):
1. **Lower `replay_fraction` sharply (`0.15–0.30`, keeping `70–85%` fresh full-game rows)** so snapshot self-play supplements rather than starves fresh self-play and critic calibration.
2. **Add Common Random Numbers (`ShuffleLibraries` / shared RNG seed across sibling branches)** so 4-branch Vine-RLOO isolates root action quality from library-draw noise.
3. **Test intermediate horizon (`H=40–60`) or root-only snapshot actor loss (`snapshot_weight=0` on non-root continuation rows)** so the policy only learns from RLOO-advantaged root forks while fresh games own full-trajectory GAE and critic calibration.

## Caveats

- **Exploratory single-seed warm-start arms:** `warm-h20` and `warm-full` ran on one seed (`40902`) to match retained `tau-x2@40902` u500–u800; replicated method claims require multiple training seeds.
- **Sequential eval upload overhead (`etude` `courier._track`):** All 1,888 / 1,888 matches completed and sealed in R2, but single-threaded sequential R2 PUTs (~0.7s/file across ~17.9k files) stretched delivery time to ~3.5 hours (`~$3.25`).
- **Engine panic observed once in training:** `warm-h20@40902` logged one Rust panic around update 113 (`managym/src/flow/resolution.rs:1066:44: internal error: entered unreachable code: clauses are bound before resolution`, `Effect::TargetClause`). `GameErrorPolicy` isolated and recovered the stream cleanly and the run finished all 400 updates.

## Appendix: runs, changes, and admission state

### Run ledger

Launcher costs recorded once from `etude status --experiment tla_frontier_snapshots`:

| Run | Commit | Machine | Updates / matches | Launcher cost (USD) | State |
| --- | --- | --- | --- | ---: | --- |
| `tla_frontier_snapshots/plr-vine-rloo@40901` | `c320c1ed9a6e` | `colab-l4-bench-ca123a39d363` (L4) | 400 / 400 updates | ~0.76 (~7.60 CU) | Succeeded |
| `tla_frontier_snapshots/plr-vine-rloo@40902` | `c320c1ed9a6e` | `colab-l4-bench-ca123a39d363` (L4) | 400 / 400 updates | ~0.76 (~7.60 CU) | Succeeded |
| `tla_frontier_snapshots/evals-c384ca093b89` | `c320c1ed9a6e` | `mtg344-cpu32-611f1e9b0dd6` (32 vCPU) | 320 / 448 matches (10 failed) | 0.03 | Failed (`games=1` pairing bug on fixed H2H) |
| `tla_frontier_snapshots/warm-h20@40902` | `96b53172c098` | `colab-l4-warm-073bd866e78b` (L4) | 400 / 400 updates | ~1.22 (~12.25 CU) | Succeeded |
| `tla_frontier_snapshots/warm-h60@40902` | `96b53172c098` | `mtg344-3090-e355dfc86b3b` (RTX 3090) | 16 / 400 updates | 0.11 | Cancelled (switched GPU training to Colab only) |
| `tla_frontier_snapshots/warm-full@40902` | `a554e2ea9275` | `colab-l4-warm-073bd866e78b` (L4) | 400 / 400 updates | ~0.88 (~8.81 CU) | Succeeded |
| `tla_frontier_snapshots/evals-6f4fbf2aa69a` | `a554e2ea9275` | `mtg344-cpu32-d4b93f31940b` (32 vCPU) | 1,888 / 1,888 sealed | ~3.25 | All 1,888 matches sealed (stopped after delivery) |
| **Total** | | | | **~7.01** | Well inside USD 25.00 cap |

### Dated change log

- 2026-11-23 (`c61219fba3`) — Added [`experiments/tla_frontier_snapshots.py`](../tla_frontier_snapshots.py), initial report notes, and discovery registration in `tests/eval/test_experiment_files.py`.
- 2026-11-23 (`e91265b012`) — Initial pre-rental check blocked on `6b4a1a` native `SnapshotVector` wiring (`ValueError: snapshot bank requires a snapshot vector`); USD 0.00 spent.
- 2026-11-23 (`c320c1ed9a`) — Synced `6b4a1a` (`4dba133cd9`), rebased `experiments/tla_frontier_snapshots.py` onto `tla_trophies` + `tau_scale=0.1` comparing against retained R2 `tla_trophies_floor/tau-x2` checkpoints (no fresh baseline), passed 4-worker `etude smoke`, and launched `plr-vine-rloo` (`40901`, `40902`) on Colab L4.
- 2026-11-24 (`9e83e3fd29`, `96b53172c0`, `a554e2ea92`) — Fixed fixed-pair H2H `games=4` (`tla-snapshots-vs-tau-x2-32-v2`), added warm-start u400 horizon arms (`warm-h20`, `warm-h60`, `warm-full`) with matched `tau-x2@40902` u500–u800 H2H, completed all 4 Colab L4 runs (400/400 updates each), and scored on `mtg344-cpu32`.
- 2026-11-24 — Added optional cross-experiment `tau-x2` baseline overlay and `start`-checkpoint clock displacement in `report()` (guarded for stock `etude-sdk`) and finalized this exploratory report.
