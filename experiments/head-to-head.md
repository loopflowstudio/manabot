# Head-to-head play between trained policies

Status: queued on the M1 mini on 2026-10-09 to start after the early
learning-rate screen releases the host. No results yet. This is **exploratory**:
one checkpoint per model, one training seed behind each, and no result here
changes a recipe by itself.

## Question

Every experiment this week scored a policy against scripted greedy. That score
is dominated by which deck the policy holds, and several checkpoints that
differ by tens of thousands of updates landed in the same band. Does playing
trained policies directly against each other, on the same deals, separate
models that the greedy score does not?

## Entrants

All checkpoints were loaded through the ordinary loader on the current world
(`w4`, pack `ur-lessons-vs-gw-allies`), which checks each checkpoint's content
manifest, deck setups and input schema against the runtime. The runner repeats
that admission, plus a byte check against the hashes below, before any game.

| id | what it is | parameters | SHA-256 |
| --- | --- | --- | --- |
| `scripted-greedy` | code anchor, the familiar opponent | - | - |
| `small-0` | untrained small model, GPU cohort seed 10351 (`etu103-recovery-v3-science-0` initial weights) | 188,482 | `dc5cdbb7da7af4aacaf3b4d3f6ba36756c2ff879c38fef8c7b4df5142419e9c2` |
| `small-26k` | same run at 26,000 updates (`etu103-recovery-v3-science-0`, raw) | 188,482 | `3a3c65cbbc90231d9f96c5619b9bbc1b8ea88ac4711e1b1752fe2e99e33cb0da` |
| `small-63k` | continued to 63,000 (`etu126-small-63000`, raw) | 188,482 | `4c7efdc71c4068691e4bae4d3b42e5b45a4854e98a7a0c87bee1468206ab3f2d` |
| `small-100k` | continued to 100,000 (`etu126-small-100000`, raw) | 188,482 | `23e295c3873a0eb251c28ea12577073415ca6120e72727f339b9afbb7d85af82` |
| `small-100k-ema` | the same endpoint, averaged weights | 188,482 | `060fa8286c26ea71e0f44373a0dc35d87e9e9c469f8c9f651965efc7f3e930c7` |
| `large-15k6` | large model (width 384, depth 8) at 15,600 updates (`etu103-recovery-v3-science-1`, raw) | 16,815,746 | `b7a52466b1d613ae3daa2c243bc0dc24bb71289d039717fb1cbde9fba07638d6` |
| `etu118-10k` | ETU-118 mini baseline, seed 11851, 10,000 updates, raw | 188,418 | `2475f3e28d0097ef79c8833cc90ea81eb6808b1727b1d68765dc16f72eae0fe4` |
| `etu125-cross-10k` | ETU-125 cross-deck-only arm, seed 12551, 10,000 updates, raw | 188,418 | `19e548468283880fa5d7155b1675850c4c75941b74f7ccb445e7d89953e4408c` |
| `etu125-mirrors-10k` | ETU-125 mirror-inclusive arm, seed 12551, 10,000 updates, raw | 188,418 | `99fff198fe96cede65500b66795b8ef10b7809c0745103639b4fd6d9ec2568f4` |

The three small-ladder checkpoints are one training run and its continuations,
so their ordering is a statement about that run. The update counts are the job
names' and were not re-derived from the run records. The roster in
[`head_to_head.py`](runners/head_to_head.py) is the authority; `roster` prints it.

Left out, with reasons:

- **Ataraxos campaign policies** (2026-10-04, self-play and search-distillation,
  seed 5601). They load and bind to the current world, so this is a budget
  choice, not an exclusion for incompatibility: they are a smaller, shallower
  model from short runs, and the untrained and 10,000-update entrants already
  cover the weak end.
- **ETU-118 seeds 11852 and 11853**, and the 26,000 and 63,000 averaged weights
  and the large model's averaged weights. One mini-baseline seed and one
  raw-against-averaged comparison, at the best checkpoint, are enough to ask
  the question.
- **ETU-125 seed 12552.** Its cross-deck arm was still training when this was
  frozen, so only seed 12551 has both arms.
- **Intermediate monitoring checkpoints** of the GPU runs. Retrievable, not
  needed for a first look.

## Schedule

Each pairing plays 100 deals. A deal is four games: both deck assignments, and
both seats within each, so each side holds each deck on the play and on the
draw once. That is **400 games per pairing**, on deal seeds
1,913,260,000-099, a namespace no other cohort uses. Every pairing uses the
same deals. Action sampling is seeded per pairing, so only the deal is shared.

25 pairings, 10,000 games:

- every entrant against scripted greedy (9);
- a full round robin among `small-26k`, `small-63k`, `small-100k`,
  `large-15k6` and `etu118-10k` (10);
- `small-100k-ema`, `small-0`, `etu125-cross-10k` and `etu125-mirrors-10k`
  each against `small-100k`, the candidate frozen opponent (4);
- `etu125-mirrors-10k` against `etu125-cross-10k`, and `etu118-10k` against
  `etu125-cross-10k` (2).

Policies sample from their action distribution, one CPU thread, as in
monitoring. Units are played deal by deal across all pairings, so a run that
stops early has every pairing at the same depth.

## What is reported

`uv run python -m experiments.runners.head_to_head report <run>` prints, from
retained rows, and is safe to run while games continue:

- per pairing: the first-named side's score (a draw counts half), split by the
  deck it held and by play or draw;
- nine named contrasts, each giving two numbers for the same two entrants on
  the same deals: the difference of their greedy scores, and their direct
  score minus 50;
- a rating per entrant on an Elo scale with greedy at zero (the arena's
  existing Bradley-Terry fit with a seat term).

Every interval is 95% and resamples whole deals, keeping a deal's four games
together, and for the contrasts keeping the same deal across the pairings
being compared. A pairing with any invalid game (timeout, crash, failed
replay) gets no score rather than a score over the survivors.

## What 400 games can resolve

If games were independent, a pairing near 50% has a standard error of 2.5
points: a 95% half-width of about 5, enough to detect a true 57-43 edge about
four times in five. The four-game deal block usually does better because it
cancels deal and deck luck: ETU-118's final evaluations of this design (100
deals against greedy) had half-widths of 2.4 to 3.3 points. A difference of
two greedy scores is about 1.4 times wider than one. The deck and seat splits
are 200 games each, about 7 points either way at worst: descriptive only.

## Reading rule

Fixed before any result exists. For each of the nine contrasts, call the
direct result *decided* if its interval excludes 50, and the greedy result
*decided* if the interval of the greedy-score difference excludes zero.

- **Head-to-head adds information** if at least three contrasts are decided
  directly and undecided by greedy, or if any contrast is decided both ways
  with opposite signs. Nine intervals at 95% produce about one false decision
  in two runs by chance, which is why one or two do not count.
- **Head-to-head adds nothing** if no contrast is decided directly and
  undecided by greedy, and the rating order matches the greedy-score order
  wherever adjacent entrants' intervals do not overlap.
- Otherwise **unresolved**.

Also report, without a threshold, the ratio of each contrast's direct effect
to its half-width beside the same ratio for the greedy difference. That is the
plain measure of which instrument is more sensitive per game.

What would be surprising:

- the 26,000 / 63,000 / 100,000 ladder separating directly by 10 points or
  more per step when their greedy scores do not: training kept improving play
  that greedy could not see;
- an inversion: the entrant with the clearly higher greedy score losing the
  direct meeting. That would mean the greedy score rewards exploiting greedy;
- `large-15k6` beating `small-100k` directly on a sixth of the updates;
- the averaged weights differing from the raw ones by more than 5 points
  either way;
- `small-0` scoring above 20% against `small-100k`. Untrained policies score
  about 28 to 30% against greedy; if a trained policy cannot do much better than
  that against an untrained one, head-to-head has little range at the bottom.

## Decision it informs

Whether the next training comparison should score checkpoints against a frozen
trained opponent, with `small-100k` as the first candidate, alongside or
instead of scripted greedy. "Adds information" argues for adding the frozen
opponent to monitoring and keeping greedy as the anchor to earlier work.
"Adds nothing" argues for leaving monitoring as it is. Secondary: whether the
averaged export is worth evaluating, and whether the ETU-125 arms look
different when they meet.

## What this cannot show

One checkpoint per model and one seed behind each: a difference between two
entrants is about those two files, not their recipes. The entrants come from
several source revisions and two value-head variants. A frozen opponent is itself
a single point; beating `small-100k` is not general strength, and a policy
could learn to exploit it as readily as greedy. The ratings come from an
incomplete schedule and lean on the model for pairs that never met.

## Execution

- Host: M1 mini (4 performance and 4 efficiency cores), CPU inference only,
  four worker processes, one game process per worker. No cloud calls.
- Why it queues instead of running alongside: on 2026-10-09 08:06 PDT the
  host's load average was 3.9 with the ETU-118 final evaluation running, so its
  performance cores were already taken; a two-worker smoke at the lowest
  priority raised the one-minute load to about 6. The learning-rate screen's
  waiter also refuses to start while any experiment runner is alive, so
  overlapping its handoff would delay it. This run starts only after that
  screen reaches a terminal status and no experiment process remains.
- Expected cost: the mini smoke measured 3.2 to 4.1 s per game between small
  models and 7.2 s with the large model, at the lowest priority on a busy
  host. At those rates the full schedule is about 3.5 hours on four workers;
  an idle host should be faster.
- Bounds: the runner stops admitting games at 6 hours and reports the run
  incomplete; the waiter kills it at 6.5 hours. Each game is capped at 120 s
  and 10,000 commands. The run stops if free disk falls below 40 GiB;
  projected use is about 0.3 GiB.
- Failure: admission is all-or-nothing before the first game. A unit whose
  worker fails is retained as failed and never replayed. The exit status is
  non-zero unless every unit finished with every game valid.
- Source: a clone on the mini at this document's commit, `main` at `73e73aa7`
  plus this experiment. The native library is the one built for that engine
  source (SHA-256 `9e3be349...0da677`).
- The waiter is a shell loop started with `nohup`; it does not survive a
  reboot, and neither does the run. It reads the learning-rate screen's status
  file and the process table and touches nothing else outside its own
  directory.

## Smoke

Not findings. Two deals (8 games) for each of four pairings, to prove loading,
pairing, replay and the report. Run on the laptop and again on the mini; the
two hosts produced identical score tables, as they should from identical seeds.
The starting player's opening hand was identical across all four pairings on
each deal and deck assignment, which is the shared-deal claim checked directly.
All 32 games were valid and replayed.
