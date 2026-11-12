# One pooled typed-token stream against the v9 object tables — MTG-306

## Question

Exploratory pilot (USD 15 cap; USD 25 until 2026-11-12). Does a policy that
attends over one pooled, typed token sequence per observation learn as well as
the current v9 table model at matched updates while costing less GPU time per
update? No prediction: the numbers open or close a design Task for an
accelerator-first token architecture.

Decision rule (from the Task): open that design Task if the pooled arm is
within noise of tables at matched updates and at least 1.5× faster per update
on the L4, or if fill shows pooled budgets beat per-axis bucketing by more than
2× in padded work. Otherwise report the numbers and recommend MTG-302-style
bucketing.

## Method

Module: `experiments/token_stream.py`. Model option:
`ModelConfig.token_stream="pooled"` (`manabot/model/token_stream.py`); omitted,
it is `tables` and every existing recipe, checkpoint and architecture identity
is unchanged.

### Arms

Both arms share every embedding and join of the current model: typed player,
card and permanent projections with catalog card programs, permanent links,
fact rows, shared-effect records and continuation context added onto their
objects, then the same attention blocks, policy head and `masked_mean` critic.

- **tables** (control): attention over the fixed object tables, 2 + 2·cards +
  2·permanents slots (202 on Elemental, 450 on TLA); each legal action gathers
  its post-attention focus objects.
- **pooled**: one sequence of each observation's valid players, cards,
  permanents, continuation relation rows and legal actions, each with a learned
  type embedding. Valid tokens move to the front in table order and the batch
  pads to the smallest multiple of 32 at or above its longest observation (one
  host read per forward). Action tokens join their focus objects' pre-attention
  embeddings through the existing action layer; logits read the attended action
  token. Objects scatter back to table coordinates for the critic. Parameters:
  +5·width (449,474 against 449,154 at w64l2 on TLA inputs). The exact
  pack/unpack round trip (`tests/model/test_token_stream.py`) checks that the
  stream loses nothing.

### Stages

1. **Fill** (laptop, counts only). Per-observation entity rows, and padded input
   elements under (a) the declared capacities, (b) MTG-302's per-axis
   power-of-two buckets (minimum 8, capped at capacity, sized to the batch's
   last used row), (c) one pooled token budget per batch at a common row width
   (widest row + a type id), and (d) ragged rows at their own widths, the
   floor. Attention lengths: table slots, MTG-302's bucketed object slots, and
   the pooled stream.
2. **L4 timing** (`--stage timing`, one run, cap USD .75). Two in-situ CUDA
   updates of each arm from a fresh initialization give end-to-end seconds per
   update; the tables arm keeps 48 collection batches and 24 learner
   minibatches. Those replay through both arms at w64l2 and w160l8 (width 160,
   8 layers, 8 heads), alternating arms within each pass: collection
   `PolicyInference.prepare` plus forward, and one learner step. Recorded:
   CUDA-event wall time, profiled kernel time and launches, synchronizing calls
   (`set_sync_debug_mode("warn")`), peak memory. Fill also runs on these
   Elemental captures.
3. **Learning** (`--stage learning`, planned, not run). Elemental recipe
   (`elemental_recipe`: ten decks, both seats, 256 streams, w64l2, one
   producer), seeds 30601 and 30602 shared by both arms, 400 updates, scored
   at 50/100/200/400 against the heuristic on the 50 held-out closure-lambda
   Elemental pairings and pooled against tables at 200 and 400 updates.

2026-11-12: the cap is now USD 15 in total, including rental idle, evaluation
and cleanup. Learning would be sized from the timing run's in-situ seconds per
update, and the pilot stops after timing if that cannot cover both arms at w64.

### Runs

| Stage | Commit | Run | Cost |
|---|---|---|---:|
| Fill, TLA captures | `7ac8c9b6f` | laptop, no run ID | $0 |
| L4 timing (+ Elemental fill) | `7ac8c9b6f` | Etude `a1bcb3bf17a2` | ~$0.16 (launcher) |
| Learning | — | not run | — |

The timing machine `mtg306-l4-31e6e7676515` ended with the run. Data:
`experiments/data/token_stream/fill-tla-e1.json` and
`timing-a1bcb3bf17a2.json` (artifact store; manifest in git).

## Results

### Fill (TLA w64/w160l8 collection captures, laptop, USD 0)

Inputs: the MTG-267 TLA regime's captured collection batches (300 per size) and
learner minibatches (8 per size), fresh-initialization update 1, v9 input
(cards 128, permanents 96 per seat). Padded elements in millions of floats over
all batches of the set; token counts per observation are median / p90 / max.

| Set | Batches | Obs | Batch | Rows per obs | Capacity | Pow2 buckets | Pooled | Ragged | Pow2 / pooled | Stream tokens | Stream length | Pow2 object slots | Pairs, tables / pooled | Pairs, pow2 / pooled |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w64 collection | 300 | 46,870 | 160 | 154 / 200 / 466 | 2,522 | 571 | 955 | 206 | 0.60 | 76 | 128 | 194 | 12.4 | 1.47 |
| w160l8 collection | 300 | 46,294 | 158 | 153 / 198 / 453 | 2,491 | 557 | 944 | 201 | 0.59 | 76 | 160 | 194 | 7.9 | 1.47 |
| w64 learner | 8 | 498 | 60 | 150 / 224 / 309 | 26.8 | 5.8 | 9.8 | 2.2 | 0.59 | 74 | 144 | 130 | 10.1 | 1.03 |
| w160l8 learner | 8 | 510 | 58 | 153 / 197 / 269 | 27.4 | 5.9 | 10.2 | 2.2 | 0.58 | 76 | 128 | 130 | 12.4 | 1.25 |

Columns from "Stream tokens" on are medians per batch: valid tokens the pooled
model attends over per observation, its padded sequence length, MTG-302's
bucketed object slots, and query–key pairs per batch relative to the pooled
stream (B·N² / B·L²).

Rows per observation (w64 collection, median / p90 / max): cards 35/46/60 (own)
and 15/27/43 (opponent's); permanents 9/15/31 each; actions 2/6/32; relations
0/9/39; facts 1/5/147; effects 0/6/28; effect details 0/2/24; members 0/6/228;
events 0/4/15; known hand 0/1/6; decklist rows 72/75/80; one player each.

Read: one pooled budget at a common row width pads **1.7× more** input elements
than per-axis power-of-two buckets (pow2 / pooled 0.58–0.60). Row widths range
from 2 to 73 floats, so pooling narrow rows (effect members, details, known hand)
at the widest width costs more than it saves in slot variance; the pooled layout
is 4.6× the ragged floor. The fill arm of the decision rule fails. For
attention, the pooled sequence (median 76 tokens, padded to 128–160) has 8–12×
fewer query–key pairs than the 450 table slots, but only 1.0–1.5× fewer than
MTG-302's bucketed objects (194 collection, 130 learner median), while also
carrying relations and actions.

### L4 timing (Etude run `a1bcb3bf17a2`)

One RunPod L4 (Torch 2.15.0+cu132), 2026-11-12.

End to end, two in-situ Elemental updates per arm from a fresh initialization
(256 streams, w64l2):

| Arm | Collection s | Learning s | s / update | Games |
|---|---:|---:|---:|---:|
| tables | 74.6 | 44.3 | 59.5 | 516 |
| pooled | 67.6 | 43.2 | 55.4 | 513 |

Pooled is **1.07×** faster per update. Collection, mostly the CPU
environment, is about 60% of an update in both arms.

Replays of the tables arm's 48 collection batches and 24 learner minibatches,
both arms interleaved (median CUDA-event wall time per call):

| Size | Call | Tables ms | Pooled ms | Speedup | Peak MB, tables / pooled |
|---|---|---:|---:|---:|---:|
| w64l2 | collection forward | 39.8 | 36.2 | 1.10× | 712 / 547 |
| w64l2 | learner step | 97.7 | 98.4 | 0.99× | 529 / 475 |
| w160l8 | collection forward | 138.8 | 92.6 | 1.50× | 1,202 / 906 |
| w160l8 | learner step | 187.9 | 142.7 | 1.32× | 2,632 / 1,900 |

Kernel launches (≈1,030–1,170 per forward, ≈4,130–4,330 per learner step) and
synchronizing calls (107–109 and 159–160) are nearly equal between arms. At
w64 both arms are launch- and sync-bound: pooled cuts kernel time by 25% on the
forward and 6% on the learner step, which barely moves wall time. At w160l8,
pooled reduces attention work over the 202 Elemental table slots enough to
cut wall time. Profiled kernel time exceeds wall time at w160l8, so the
profiler counts overlapping kernels; compare wall times.

Fill on these Elemental captures (8,252 collection and 1,505 learner
observations) repeats the TLA result: pow2 / pooled 0.58 and 0.61.

### Learning

Not run. Neither arm of the decision rule can pass: fill already failed, and
the speed arm needs ≥1.5× per update on the L4, while the w64 recipe measured
1.07× end to end. Even at w160l8, whose GPU calls are 1.3–1.5× faster, the
CPU collection share (about 60% of an update) keeps the per-update gain well
under 1.5×. A strength result therefore cannot change the decision. At the
measured ~57 s per update, the planned comparison (two seeds × two arms × 400
updates) needs four ~6.5 h L4 windows, about $15.50, plus about $4.50 of
evaluation: roughly $20, over the $15 cap. One seed per arm (about $11) would
fit but has no power to detect "within noise".

## Verdict

**Exploratory.** Recommend MTG-302-style per-axis power-of-two bucketing; no
design Task for a token architecture. What was learned:

- One pooled budget at a common row width pads **1.7× more** input than
  per-axis buckets (pow2 / pooled 0.58–0.61 on TLA and Elemental captures),
  because row widths range from 2 to 73 floats. The fill arm fails.
- The pooled attention stream is **1.07×** faster per update on the w64
  Elemental recipe, whose cost is CPU collection plus launch-bound GPU calls.
  The speed arm fails.
- What changes next: MTG-302's bucketing is the padding path. The stream is
  worth revisiting only for wider models (w160l8: 1.3–1.5× on GPU calls,
  25–28% less peak memory), and then with ragged or bucketed rows rather than
  one common width.

No strength claim is made for either arm; learning was not run.

## Caveats

- Strongest confound for the speed numbers: the end-to-end figure is two
  updates per arm in one run on one L4, from a fresh initialization, so it
  carries rental-machine noise and early-training observation sizes (short
  games, few permanents). Later-training boards are larger, which fills more
  of the fixed table slots and narrows the pooled stream's attention saving;
  the w64 conclusion rests on collection being CPU-bound, which board size
  does not change.
- Tokens pack on the device after per-type embedding rather than through a
  host conversion of the v9 dictionaries, departing from the Task's design.
  References (links, facts, effects, relations, action focus) are resolved by
  the existing joins before packing, so no host remapping exists and
  collection, admission and the native encoder are untouched. There is no
  host tokens→tables inverse; the pack/unpack round trip is the check.
- Facts and shared effects stay joins onto their objects in both arms; their
  encoders run unchanged. Events, known hands and decklists are counted by fill
  but do not enter the pooled sequence (events are off in this recipe; known
  hands and decklists reach the players through the card encoder in both arms).
- Learner-step timing uses a stand-in loss (legal-action log-likelihood plus
  value MSE) with the damped loss's gradient support; the model's forward and
  backward are the measured cost, not the full damped objective.
- "Per update" in the decision rule is read from the in-situ update wall time
  on the L4; replay timings separate the GPU forward and learner step from
  collection's CPU environment work.
- The 187 captured learner minibatches the Task names were not found; TLA fill
  uses the 8 + 8 retained from the same captures, so its learner rows rest on
  few batches. The Elemental captures (24 learner minibatches) agree.
