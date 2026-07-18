---
pm:
  provider: linear
  linear_initiative: 427144c1-6896-40e1-a23e-6e7fe9bc9fc4
  linear_team: 49e062b0-1645-42c3-9bed-6c6c785cafcc
---

# Intelligence

## Objective

Build increasingly strong manabots as pilots, opponents, teammates, sparring
partners, and advisors in Etude Fantasia's AI-assisted Pro Tour testing house.
Make runnable agents and search systems, play them in authoritative Magic
worlds, measure what they actually do, and iterate toward superhuman
performance. Typed card programs, structured commands, search, self-play, and
Study evidence matter when they improve a working agent—not as prerequisites
that must be proven in isolation before one may be built.

The first player-facing AI loop starts from one canonical viewer decision and
an explicit viewer-safe belief. One pinned advisor and compute class return the
complete aligned strategy distribution, values, robustness, uncertainty, and
provenance; changing only the belief makes the delta inspectable. The same
decision, belief, advisor, compute, and provenance identities yield the same
advice live or in Study. Intelligence owns policies, priors, model-inferred
beliefs, search, training, evaluation, and attributable evidence. It does not
own match facts, player-authored beliefs, rules authority, or Game's surface.

The immediate architecture program is defined in
[docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md): consume managym's canonical
Command/Observation/world-query authority, ship conditional strategy search,
train belief-conditioned policy/value students, learn calibrated beliefs from
viewer history, and make the results available to Study and the world-pinned
arena. These are the highest-priority next Intelligence tasks.

The product north star places manabots in Avatar Cube Team Sealed as pilots,
teammates, and opponents. The first robot team may use fixed authored decks,
and manabots need not sideboard. Once a manabot can play the selected world,
constructing three legal decks from a shared sealed pool becomes an important
later Intelligence capability. A versioned, world-pinned Elo arena measures
the resulting players at declared compute classes. Drafting is separate and is
not a prerequisite.

## Measures

- Every primary Project produces a runnable manabot, teacher, search system, or
  training loop that executes against real `managym` positions and can be
  exercised with one documented command.
- Search teachers and students are compared in actual selected matchups at
  explicit compute budgets, with legality, competencies, seat-balanced
  strength, calibration, latency, throughput, label cost, and uncertainty.
- One historical/root Observation can be evaluated under the compatible-deal
  prior and typed conditions such as `Has(Bolt)` and `NoLands`, returning
  aligned complete action distributions, values, condition mass, uncertainty,
  and exact provenance without exposing actual hidden truth.
- Advice for one decision is identity-pinned to its viewer-safe belief,
  advisor, planner/evaluator, compute class, seed plan, and evidence bytes. The
  same identity is reproducible through live play and Study, while a mismatch
  returns typed unavailability rather than adapted or invented evidence.
- A supervised belief head maps lossless viewer history to a calibrated
  normalized distribution over managym's world hypotheses. Both policy and
  value are conditioned on that `BeliefState`; actual hidden worlds remain
  calibration targets rather than inference inputs.
- Conditional teacher trajectories, shards, and checkpoints bind world/query,
  belief, history, target, source, seed, and exact byte identities and replay
  through the same semantic Commands as live play.
- Every admitted candidate enters a versioned, world-pinned skill arena. The
  primary hill-climbing signal is a population Elo rating at a declared
  compute class, reported with paired-deal uncertainty and the underlying
  matchup matrix; ratings never cross world or arena-version boundaries.
- A semantic policy consumes viewer-safe runtime facts, typed ability programs,
  and structured legal offers, emits atomic `Command` values, and is evaluated
  on real play—including held-out cards or compositions of known operations.
- Ablations remove card identity, semantic structure, structured decoding, or
  search at the boundary of a working prototype so their effects on learning,
  transfer, strength, and systems cost are directly measurable.
- Policy, search, robustness, and uncertainty evidence can be replayed through
  the versioned Study contract without hidden-information leakage or invented
  client-side meaning.
- Any superhuman claim names the matchup and content boundary, information
  boundary, model and opponent cohort, compute budget, seeds, competencies,
  exploitability evidence where available, and uncertainty.

## Operating loop

Lead with building:

1. make the thinnest end-to-end prototype that can act in the real engine;
2. place it on the common skill-and-cost scoreboard and measure behavior,
   learning, strength, and cost;
3. identify a surprising or confounded result;
4. run the smallest diagnostic kata or ablation that separates the live
   explanations;
5. change the prototype and measure again.

Katas are diagnostic instruments, not admission exams. A new kata must name the
prototype ambiguity it resolves and the decision its result will change. No
chain of diagnostic work proceeds without returning to an end-to-end agent.
Pre-registration remains useful for expensive comparisons and for preventing
post-result threshold changes; it is not a burden of proof charged before a
first prototype may exist.

Prefer plausible architectures early. A small Transformer, graph Transformer,
or tree-aware encoder that can represent sequence, hierarchy, field roles, and
binding is a better prototype than extending an intentionally weak pooling
architecture through a proof ladder. Keep simple baselines and destructive
ablations beside it so improvements remain interpretable.

## Dependencies and bounds

Rules owns typed programs, structured offers and commands, viewer-safe state,
identity, exact forks, possible-world/query meaning, the reference
compatible-deal measure, and legal world materialization. Intelligence owns
memory, priors and learned beliefs, sampling policy, planning, and learning over
those interfaces. It reports pressure back through real workloads and does not
delay prototypes until every representation is settled. A proven full-clone
path and exact small world support are acceptable first backends when they fit
the measured budget.

Game owns player-authored belief input, Study decision navigation, reveal,
comparison, explanation, roles, and human research consent. Intelligence emits
model-inferred beliefs and attributable advice; it does not build a second
replay, legality, presentation, or hidden-information system. Etude or an LLM
may construct a typed `WorldQuery`; neither can inspect actual hidden authority
or introduce arbitrary query semantics.

An external LLM may be a teacher, baseline, or grounded narrator. It is not the
inner-loop rules oracle or source of legal actions. Open-ended card coverage,
deck building, format legality, Commander breadth, and runtime natural-language
card parsing remain out of scope.

## Evidence discipline

Prototype measurements begin with the first runnable version. Promote claims
only after matched controls and multiple seeds exist. Pin content, engine,
observation, action, model, opponent, and compute identities; retain raw
rerunnable results. Win rate alone is insufficient: legality, competencies,
information safety, calibration, throughput, transfer, and matched cost remain
separate evidence.

Concrete repository changes begin as Linear Tasks under an Intelligence
Project. Tasks that change Rules or Game authority stay in their providing
waves and are represented here as dependencies exercised by a running system.

Do not create a deck-construction Project until the runnable policy and search
systems can play a meaningful content boundary. Record the Team Sealed goal in
the portfolio now; earn the Task later from an executable agent.
