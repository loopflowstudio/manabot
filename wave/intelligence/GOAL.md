---
pm:
  provider: linear
  linear_initiative: 427144c1-6896-40e1-a23e-6e7fe9bc9fc4
  linear_team: 60558c53-2169-49f8-a76a-1f4586705aa9
---

# Intelligence

## Objective

Build learned and search-augmented Magic intelligence that composes over typed
card programs and structured legal choices, transfers beyond memorized card
identities, remains information-set honest, and advances toward measured
superhuman play at practical compute cost.

The authored game and Study experience are part of the research loop: they make
strong play enjoyable, expose what the agent considered, and produce versioned
human and engine evidence. Intelligence owns the policy, search, training, and
evaluation machinery behind those moments. It does not own rules authority or
the Study product.

## Measures

- A KataGo-shaped search teacher runs on the existing robust observation and
  action ABI without waiting for semantic inputs, emits visit-distribution
  policy targets and value or outcome targets with complete provenance, and is
  evaluated against policy-only and frozen controls at matched compute.
- A bounded overnight supervised experiment determines whether a student can
  distill the search teacher: label quality, policy-target formulation,
  optimization, capacity, and student inference cost are separated before a
  failure is assigned to the architecture.
- Tiny semantic representation katas distinguish equal-token programs whose
  order, hierarchy, field roles, argument binding, or target and choice roles
  differ; a structural encoder is compared with the current order-invariant
  encoder at matched capacity and cost.
- Semantic katas hold out compositions and card identities to distinguish
  memorization from recombination of known operations, with CardDefId and name
  ablations, zero-shot and limited-retraining evidence, and explicit failure on
  unseen opcodes.
- A semantic-conditioned gameplay integration Project is created only if the
  teacher/distillation and semantic-kata gates both produce useful evidence.
  That later Project runs the four-arm ablation: Card-ID plus legacy head,
  Card-ID plus structured decoder, structural semantics plus structured
  decoder, and held-out structural semantics plus structured decoder.
- Later strength claims combine exact or approximate exploitability,
  mechanism-level competencies, held-out transfer, and matched-compute
  evaluation. A claim of superhuman play names its matchup/content boundary,
  information boundary, compute budget, opponent cohort, seeds, and
  uncertainty.

## Diagnostic program

Intelligence advances through bounded, pre-registered experiments, not a linear
feature ladder. Every experiment states its prediction, matched controls,
training seeds, wall-clock or compute cap, success and kill criteria, and the
next branch for each plausible result. Overnight supervised training against a
search teacher is one evidence-producing experiment, not a milestone by itself.

The first two Projects run largely in parallel:

1. **Search Teacher & Distillation Loop.** Establish teacher quality on the
   current ABI, generate attributable visit and value targets, and run one
   cost-capped student experiment. Separate teacher weakness, target choice,
   decoder legality, optimization, capacity, and inference cost.
2. **Semantic Representation Katas.** Run tiny paired-program and held-out
   recombination probes. Prove or falsify structural understanding before
   paying for full gameplay integration; retain the order-invariant encoder as
   a required negative control.

Only their joint evidence can promote **Semantic-Conditioned Gameplay Policy**
as a later integration Project. Its four arms isolate choice grammar,
structural semantics, and held-out transfer. A failed arm branches to the
diagnosed component rather than automatically scaling model size or training.

Planning follows only where those results point. Retain the information ×
continuation 2×2 and exactly solvable microgame as diagnostic instruments:
separate rollout/value failure, belief estimation, information-set inconsistency,
and implementation defects before choosing tree search, weighted
determinization, or public-belief/CFR machinery. Compute-conditioned search and
opponent-league training are tested against policy-only and frozen controls, not
assumed to be improvements.

## Dependencies and bounds

Rules owns typed semantic programs, `InteractionOffer`/`Command`, viewer-safe
state, identity, exact fork/rollback, and branching representation. Intelligence
may prototype against those versioned interfaces but does not close their KRs.
Semantic katas and student scaffolding may start before the final branching
representation is selected. The first bounded teacher may use the proven full
clone baseline if its pre-registered cost cap fits; rollout-heavy search,
self-play scaling, and league training wait for W2-198/W2-199 or an explicit
benchmark decision to retain full clone.

Study owns landmark selection, retry, comparison, explanation, and human-facing
research consent. Intelligence emits versioned policy/search evidence through
the merged `StudyArtifact`/`DecisionEvidence` boundary; it does not build a
second replay, legality, presentation, or hidden-information system.

An external LLM may later serve as a teacher, baseline, or narrator over
authoritative artifacts. It is not the inner-loop policy, rules oracle, or
source of legal actions. Open-ended card coverage, deck building, format
legality, Commander breadth, and generic natural-language card parsing remain
out of scope.

## Evidence discipline

Training seeds are the experimental unit. Comparisons pin content pack, engine,
observation, action, model, opponent, and compute identities; headline claims
have rerunnable scripts and raw results. Win rate alone is never sufficient:
competency, legality, exploitability, information safety, matched cost, and
transfer are reported separately.

Concrete repository changes begin as Linear Tasks under one Intelligence
Project. Tasks that change Rules or Study contracts stay in their providing
waves and are represented here as explicit dependencies.
