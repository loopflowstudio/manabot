# Training decisions from primary evidence

Research date: 2026-09-24 PDT / 2026-09-25 UTC. Read original papers, methods,
ablation sections and official implementation documentation. Statements labelled
**Transfer** are design inferences, not results measured on manabot. No reported
Go, StarCraft, Atari or poker score is converted into Magic Elo or dollars.

## KataGo: buy useful targets, not uniform search depth

**Public evidence.** Wu's main run used approximately 27 V100s for 19 days,
producing 4.2 million games and 241 million samples. Full-search turns initially
used 600 visits with probability .25; other turns used 100 and were excluded
from training. Later budgets were 1,000/200. The approximately two-day ablations
estimated speed factors of 1.37 for playout randomization, 1.25 for forced
playouts/target pruning, 1.60 for global pooling, 1.30 for auxiliary policy,
1.65 for ownership/score, and 1.55 for game features. The paper cautions that
the comparisons are approximate; these factors are not independent causal
multipliers for another game. Larger fixed searches could overfit scarce data.
Networks grew from 6x96 to 20x256; the data window grew too.
[Paper, §§2–5 and appendices C/E](https://arxiv.org/html/1902.10565v5).

**Transfer.** Measure distinct complete games, labelled decisions and repeated
optimizer exposures separately. More simulations can reduce useful data
diversity at fixed cost. Preserve a small-model control when increasing model
size. Candidate intervention: spend stronger search on a preselected fraction
of roots, with complete provenance and separate full-search targets; compare
against uniform search at equal total label cost. Do not substitute hidden
truth for the acting viewer's belief, import Go symmetries, or make Magic life
total a shaped objective. Auxiliary losses need an ablation against policy-only
BC: the repository already has value-calibration improvements that hurt play.

**Implementation evidence.** The official training guide separates self-play,
shuffle, train, export and optional gatekeeping. It describes a bounded,
synchronous single-machine loop, warns its defaults are not the main-run
configuration, and recommends balancing self-play against learner throughput
(roughly 4–40 times the GPU power in its Go setting). Its train bucket limits
updates relative to fresh data.
[Official training guide](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md).
The current `selfplay1.cfg` has `cheapSearchProb=.75`, visits 100/600,
`cheapSearchTargetWeight=0`, 128 game threads, one search thread and inference
batch cap 128. These are implementation examples, not historical immutable
configs or manabot settings.
[Config](https://github.com/lightvector/KataGo/blob/master/cpp/configs/training/selfplay1.cfg),
[config lineage](https://github.com/lightvector/KataGo/blob/master/cpp/configs/training/README.md).

**Decision.** Start with serial generate/train/export/evaluate stages and durable
shards. Rent a learner only while it has work. Do not build a distributed actor
service to reproduce a ratio whose denominator differs here: manabot's
uniform-prior random-rollout search is predominantly CPU work.

## AlphaStar: a useful curriculum is not free

**Public evidence.** The final system initialized policies from 971,000 human
replays, then fine-tuned on 16,000 high-rated winning games. Human-data
regularization and strategy conditioning mattered in its ablations. League
training used main agents, main exploiters, league exploiters and prioritized
fictitious self-play; PFSP beat FSP in the reported population comparisons.
Each agent trained on 32 TPU v3 devices over 44 days. Per-agent infrastructure
included 16,000 concurrent matches, batched actors, a learner processing about
50,000 steps/s, and twice-reused trajectories. There were twelve actor-learner
setups. The 139M-weight network needed 55M weights at inference. V-trace,
TD(lambda), UPGO and supervised KL served different purposes; this was not
ordinary PPO with an opponent list. Its training-only critic used opponent
observations.
[Author-hosted final paper, Methods and extended data 5/6](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf).

**Transfer.** Use frozen opponents and retain weaknesses, but start with a
handful of checkpoints. A scarce $100 corpus cannot replace that human
curriculum by assertion. Keep the repository's viewer-safe value contract;
do not silently copy the privileged critic. Use terminal rewards first,
independent seeds and both deck assignments. Spending on twelve concurrent
learners before proving one improvement loop is a poor supported bet here.

**Implementation evidence.** The official repository explicitly does **not**
ship online RL training; it offers architectures and offline BC tooling.
Its offline config has batch 1,024, unroll 1, LR 5e-4 and a 1e10-frame limit.
The unplugged guide releases BC/FT-BC as examples, not the entire paper's
algorithm suite.
[README](https://github.com/google-deepmind/alphastar/blob/main/README.md),
[config](https://github.com/google-deepmind/alphastar/blob/main/alphastar/unplugged/configs/alphastar_supervised.py),
[offline guide](https://github.com/google-deepmind/alphastar/blob/main/alphastar/unplugged/README.md).

**Decision.** Budget a small sequential self-play/PPO competitor. A later
historical pool can use existing immutable registrations, but matchmaking
integration is an explicit gap. No plan claims that cloning AlphaStar code
provides a working manabot league or pretrained Magic policy.

## AlphaZero, MuZero and modern search

**Public evidence.** AlphaZero alternates search-generated policy targets and
outcomes with policy/value learning, using known rules. Its tabula-rasa result
does not imply cheap training: the original account used 5,000 first-generation
TPUs for self-play and 64 second-generation TPUs for learning, with 800
simulations per move.
[Original paper, Methods](https://arxiv.org/abs/1712.01815).
**Transfer.** managym already supplies exact transitions. Reuse them; do not
learn a replacement rules model merely to imitate the algorithm family.

**Public evidence.** MuZero trains representation, dynamics and prediction
jointly from unrolled reward/value/policy losses. Its board-game setup used
16 TPU devices for learning and 1,000 for self-play; Atari used 8/32. Reanalyse
refreshes targets on existing trajectories using the current model.
[Paper, training/infrastructure and Reanalyse](https://arxiv.org/abs/1911.08265).
**Transfer.** Relabelling viewer-safe archived roots may be cheaper than new
full teacher games, but costs real searches and cannot create independent
game outcomes. A learned dynamics model adds approximation error to an engine
we already own. Admit it only after measuring an actual engine bottleneck.

**Public evidence.** Gumbel AlphaZero uses sampled actions and sequential
halving to improve low-simulation policy improvement; its guarantee assumes
correct action values. This addresses a different issue from imperfect-
information continuation honesty.
[Paper](https://openreview.net/pdf?id=bERaNdoegnO),
[official Mctx implementation](https://github.com/google-deepmind/mctx).
**Transfer.** A promising later teacher pilot if shallow PUCT produces poor
targets, not permission to replace the current planner before a baseline.
Keep the whole semantic legal action set; sampling must have explicit support
and recorded probabilities, never encoder-tail truncation.

**Public evidence.** EfficientZero's self-supervised consistency, value-prefix
prediction and off-policy correction improve Atari-100k sample efficiency.
Its two hours of game experience is an environment-data quantity, not two
hours of paid training.
[Paper](https://arxiv.org/abs/2111.00210).
DreamerV3 demonstrates a reusable learned-world-model recipe across many
domains, not a low-cash Magic result.
[2025 paper](https://www.nature.com/articles/s41586-025-08744-2),
[official code](https://github.com/danijar/dreamerv3).
**Decision.** Neither learned-model family displaces exact-engine BC/PPO in
the first allocation. Reconsider if retained end-to-end measurements show
simulation, rather than inference, search depth or data quality, dominates.

## Hidden information: do not buy a false theorem

**Public evidence.** DeepNash's R-NaD changes learning dynamics with evolving
regularization and trains without search or explicit opponent modelling.
The full system includes extra components beyond the core dynamics and
reported 768 learner TPU nodes plus 256 actor TPU nodes. Equilibrium theory
and the scaled neural implementation are distinct claims.
[Paper, R-NaD, Table 2 and infrastructure](https://arxiv.org/abs/2206.15378).
**Transfer.** It is evidence that model-free play can learn hidden-information
strategy, not that adding an entropy term to PPO recreates DeepNash. The
candidate becomes attractive if teacher/search cost dominates and teacher
exploitation survives stronger continuations. It remains a separate,
falsifiable prototype, not the default $10,000 recipe.

**Public evidence.** Deep CFR learns regret and average-strategy networks
from sampled traversals. External sampling traverses **every traverser
action**, samples opponent actions, uses reservoir memories, and retrains
regret networks; its ablations favor fresh regret-network training.
[Paper, algorithms 1/2 and §6](https://proceedings.mlr.press/v97/brown19b/brown19b.pdf).
**Transfer.** Large declarations and long games make traversal cost a first
measurement. The current teacher dataset is not a counterfactual-regret
dataset; bootstrapping an average strategy from it would need new semantics.

ReBeL uses public beliefs and solving; Student of Games combines belief-aware
search with game-theoretic learning. They motivate an honest information-set
boundary, not the claim that determinized PUCT already inherits safe resolving.
[ReBeL](https://arxiv.org/abs/2007.13544),
[Student of Games](https://arxiv.org/abs/2112.03178).
**Decision.** Retain uniform-prior determinization as an explicitly approximate
baseline. No stronger equilibrium claim. New belief, regret or resolving
machinery is blocked on a separate bounded strategic/cost discriminator.

## Offline learning, replay and warm starts

**Public evidence.** AlphaStar Unplugged studies offline learning over about
1.4M games / 2.8M player episodes. It compares BC, fine-tuned BC and value-based
improvement methods; naive offline actor-critic can improve and then diverge.
The learned behavior policy/value and constrained improvement are central.
[Paper, dataset and algorithms](https://arxiv.org/abs/2308.03526).
**Transfer.** Preserve a BC control. Magic has no admitted comparable replay
corpus here; data collection is a budget line, not a free prior.

IQL avoids querying unseen actions during value fitting and extracts a policy
by advantage-weighted BC. Its D4RL result is not evidence that weak self-play
data become an expert teacher.
[Original paper](https://arxiv.org/abs/2110.06169).
DAgger motivates collecting labels on the learner's visited states to address
the distribution shift left by one-shot imitation.
[Original paper](https://proceedings.mlr.press/v15/ross11a.html).
**Decision.** First reuse immutable teacher data with whole-game validation;
then spend a bounded fraction on fresh learner-visited roots, if label quality
holds. PPO reuses each current rollout for its configured epochs, not an
arbitrary stale replay buffer. BC epochs do not increase unique game count.

The January 2026 QZero preprint reports search-free off-policy Go learning,
but its retained account is five months on seven GPUs. It makes replay-based
self-play worth monitoring; it supplies no small-budget or hidden-information
transfer guarantee.
[Primary preprint](https://arxiv.org/abs/2601.03306).

Pretrained warm start means a current-content, current-ABI, admitted manabot
checkpoint. Historical ports and foundation-model text weights are excluded
from the primary comparison. No verified transferable pretrained action model
was found in this repository. A new BC warm start is therefore the default.

## What repository evidence outranks analogy

At base `1cd60fdaabf783788f6ee94ce4c4508c9435785c`:

- `manabot/config/presets.py` plus `infra/hypers.py`: local/default training
  uses a vanilla mirror and passive opponent. It is a loadability smoke, not
  the selected-match strength baseline.
- `sim/distill.py:generate_selfplay_shard` and
  `verify/run_exp11_train.py` hardcode INTERACTIVE_DECK mirrors. The reusable
  `train_bc`, `SeatRoutedCollector`, `NetOpponentTrainer`, and checkpoint
  exporter exist; the selected-world orchestration still needs binding.
- `sim/selected_branchdriver_teacher.py` explicitly creates Allies/Lessons
  roots and binds extension/content/source receipts. It is an existing
  teacher verification consumer, not a production shard exporter by itself.
- `env/observation.py` warns and truncates legal actions beyond `max_actions`.
  Full-game admission must detect every overflow; merely zero illegal chosen
  actions does not prove all legal choices were represented.
- `Trainer.save` saves weights/optimizer/step, but there is no complete
  environment/RNG/collector continuation interface. Treat restarts as new
  attempts, not exact resumes. Prefer on-demand capacity initially.
- `etude/villain.py:CheckpointVillain` is the actual demo load consumer.
  Closer inspection of `arena/match.py:play_cell` found another hardcoded
  INTERACTIVE_DECK mirror. Its registration/replay machinery is reusable, but
  its current game loop does not evaluate Allies/Lessons. This counterexample
  rules out presenting the existing arena CLI as an accepted selected-match
  evaluation command without ETU-79's binding or a separately tested change.
- `experiments/exp-03-distillation.md` supports a BC hypothesis but its
  addendum admits the PPO comparison used harmful shaping; the deal bug and
  old world further limit transport. `exp-07-expert-iteration.md` records a
  ~30x policy-rollout decision premium and negative matched-time result.
  `exp-10-value-gate.md` records better-than-value-greedy search but worse
  economics than random rollouts. Wave memory records INT-7/8 adverse player
  results. None predicts current-world strength quantitatively.
- At the 2026-09-25T00:25:43Z read, ETU-78 was unstarted. It was subsequently
  superseded by ETU-79, which now owns corrected pipeline receipts. No accepted
  full-pipeline receipt was present in this implementation checkout. Missing observations stay null.

Evaluation will use independent training seeds and untouched full-game paired
deals. The literature also warns how uncertain few-run RL comparisons are.
[Statistical Precipice](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf).
