# Belief-forming manabot architecture

## Intent

> “I want to focus more on being able to show that a trained bot *WOULD ACT
> DIFFERNTLY* based on conditional belief has bolt = 1 vs has bolt = 0”

> “I would think that we should make generating a prior a fundamental part of
> the agent, though we should be abel to "override it" during training with
> teacher values etc”

> “theres also the language of beliefs right? the belief model i think produces
> per card beliefs or something? and then qe have a query language on top”

The architecture must make belief formation part of an ordinary manabot, not a
teacher-specific feature. Search, teaching, training interventions, and Study
all use the same explicit belief boundary that autonomous play uses.

## What to build

Build a manabot that forms and updates a normalized belief over authoritative
managym worlds at every decision, conditions that belief through managym's
typed query language, and feeds an explicit generated or supplied belief into
one shared policy/value/search decision core.

## Placement

- Wave: Intelligence
- Project: Search Teacher & Distillation Loop

This Project is where the boundary is first exercised because conditional
search and distillation need it. The boundary itself belongs to every manabot;
the teacher does not own belief formation.

managym owns viewer Observation and history, legal Commands, possible-world
meaning, `WorldQuery` semantics, the reference compatible-deal measure, and
world materialization. manabot owns agent memory, belief-model weights,
belief updates over that domain, model projections, policy/value, search,
training, and evaluation.

## Laws of physics

There are three different semantic layers:

```text
PossibleWorldSpace Ω(H)   authoritative hidden-world language      managym
BeliefState b(w)          normalized weights over Ω(H)             manabot
WorldQuery Q(w)           typed predicate selecting worlds         managym
```

For a belief `b` and query `Q`:

```text
P_b(Q)   = sum_w b(w) * 1[Q(w)]
b_Q(w)   = b(w) * 1[Q(w)] / P_b(Q)
```

Empty support is an explicit failure. Query evaluation never consults the
actual hidden world. Equivalent queries that select the same worlds produce
the same conditioned belief.

Per-card count probabilities are a derived model view of `b`, not the belief's
canonical meaning. Marginals are useful policy tokens, but they lose
correlations and therefore cannot authoritatively answer conjunctions such as
`Has(Bolt) AND Has(Counterspell)`. Query probability, conditioning, search
sampling, and world materialization always use the full distribution or
scorer over `Ω(H)`, never a reconstruction from marginal tensors.

Actual hidden truth may supervise belief calibration in access-controlled
training evidence. It is never a policy feature, never selects a condition,
and never enters the shared encoder on a live decision path.

## Agent loop

At decision `t`, belief formation precedes action selection:

```text
b(t-1) + new viewer-safe history + current Observation
                         |
                    belief update
                         |
                       b(t)
                         |
              generated-or-supplied selector
                         |
          policy/value -------- optional search
                         |
                     Command(t)
```

Conceptually, the update may perform a prediction through intervening public
actions and then correct on newly visible evidence. The first reference model
may instead recompute the exact compatible-deal measure from complete canonical
viewer history; both satisfy the same state transition contract.

In search language this generated `b(t)` may be called the prior: it is the
agent's baseline distribution before a query intervention or rollout. It is
not prior to the game history; all viewer-safe evidence through decision `t`
has already been incorporated.

## Supervised belief-learning contract

The learned belief model begins from managym's exact, policy-free
compatible-world measure. Call that state-dependent base measure `p0`. It is
not a fixed vector and not merely an optional input feature: managym recomputes
its support and combinatorial weights from the current viewer-visible hard
facts, including hidden-zone sizes, reveals, known deck contents, and the
authoritative zone transition rules.

For a known unseen deck with `ci` remaining copies of card definition `i`,
`N` total unseen cards, and a hidden hand of size `k`, the count-vector form is
the multivariate hypergeometric distribution:

```text
p0(h | k) = product_i choose(ci, hi) / choose(N, k)
```

The current canonical encoding is ragged by decision. `PossibleWorldSpace`
enumerates `W` compatible opponent-hand name-multisets in deterministic order;
world `j` stores its sparse positive counts and exact integer physical-deal
weight. `p0.probabilities` is `float64[W]`, and an exact learned
scorer produces one correction logit in that same canonical order. `W` is the
number of bounded count vectors satisfying `sum_i hi == k`, not necessarily
`choose(N, k)`, because interchangeable copies are collapsed into one world:

```text
worlds[j] = {card_name: positive_count, ...}
weight[j] = product_i choose(ci, hi)
p0[j]     = weight[j] / choose(N, k)
```

For example, an unseen pool `{A: 2, B: 1, C: 1}` and hidden hand size two has
four worlds: `{A: 2}`, `{A: 1, B: 1}`, `{A: 1, C: 1}`, and `{B: 1, C: 1}`,
with weights `1, 2, 2, 1` and probabilities `1/6, 2/6, 2/6, 1/6` in the
engine's canonical ordering. The current world domain represents opponent hand
counts; the opponent library multiset is inferred as `pool - hand`. It does not
yet represent hidden library order or arbitrary additional hidden state.

There is no global world index across decisions. For the current hand-only
domain, each decision first writes its candidates in the content manifest's
shared card-definition vocabulary:

```text
world_counts: int[W, C]   # candidate hand count vectors
log_p0:       float[W]    # exact combinatorial log probabilities
```

Across a batch, `W` varies with the compatible world space. Exact-world
training therefore packs the ragged candidate sets into common tensors:

```text
world_counts: int[sum_b W_b, C]
log_p0:       float[sum_b W_b]
world_batch:  int[sum_b W_b]     # candidate -> decision
offsets:      int[B + 1]         # decision -> candidate range
target_world: int[B]             # supervision-only local candidate index
```

The scorer joins each candidate with its decision's viewer-history embedding,
emits `s_theta` with shape `[sum_b W_b]`, and applies a segment log-softmax to
`log_p0 + s_theta` independently within every offset range. A padded
`[B, W_max, C]` tensor with a world mask is semantically equivalent but is not
the preferred storage because support sizes can differ sharply. The
policy-facing `BeliefTensorView[rows, count_buckets]` is a derived marginal
projection and is not the `p0` representation or the belief-training target.

Thus a seven-card hand and a five-card hand have different world spaces and
different base distributions. If a future mulligan rule changes the hand from
seven to five, managym owns the exact redraw, return, or bottoming transition;
manabot receives the resulting compatible five-card measure rather than
inventing mulligan semantics.

The deployed updater does not require the acting policy. It learns a normalized
likelihood-ratio correction from viewer-safe history `x`:

```text
q_theta(w | x) = p0(w | x_hard) * exp(s_theta(w, x))
                  / sum_w' p0(w' | x_hard) * exp(s_theta(w', x))
```

Before the first viewer-visible, policy-dependent event, the learned correction
is disabled and `q_theta == p0` exactly. For opening hands, the first useful
update is `Keep(k)`: `p0` already accounts for the publicly known hand size,
while `s_theta` learns what choosing to keep that many cards implies about the
kept hand. Hidden intermediate mulligan decisions are not inputs unless they
are actually present in the viewer-safe history. Mulligans are not implemented
in the current managym rules surface, so this is a pinned boundary for that
future engine transition, not a claim about the present demo.

Training trajectories may retain the authoritative materialized world behind
an access-controlled supervision boundary. Each realized world is one sample
from the conditional distribution, not a target asserting that all probability
must collapse onto that world at inference time. The primary loss is whole-
world negative log likelihood:

```text
L_belief = -log q_theta(w_materialized | x_viewer_safe)
```

Independent card marginals may be auxiliary losses or evaluation views, but
they do not replace the joint-world objective. Start without label smoothing
or an entropy bonus; a proper scoring rule over many independently
materialized trajectories supplies the calibration pressure. Acting-policy or
opponent-checkpoint identity is provenance for population-shift and
leave-one-opponent-out evaluation, not a required inference input. A
known-policy Bayesian update may be retained only as a frozen diagnostic.

The first learned-belief continuation gate requires held-out joint NLL better
than `p0` after policy-dependent events, calibrated query probabilities and
credible-set coverage, zero mass on incompatible worlds, exact equality to
`p0` before the first informative event, viewer-equivalent hidden-truth swap
invariance, and no unacceptable latency. Dataset splits are by whole match and
opponent version, never by adjacent decisions from the same trajectory.

An ordinary autonomous decision always calls the belief model. A training,
evaluation, or Study intervention may explicitly supply another valid
`BeliefState` for the decision core. The supplied belief replaces `b(t)` for
that evaluation only and does not silently mutate autonomous agent memory.

## Shape and delivery

This is an additive research series. This design fully specifies the keystone;
later increments receive their own exact designs when launched.

1. **Keystone — fundamental belief-forming agent.** Add the belief-model
   lifecycle, exact belief/query operations, model projection, explicit
   intervention seam, and belief-conditioned policy/value input. Exercise it
   with a runnable agent path.
2. **Conditional teacher evidence.** Search generated and query-conditioned
   beliefs at paired real roots and produce a teacher sensitivity atlas.
3. **Conditional distillation.** Train policy/value students under supplied
   beliefs and show a held-out `Has(Bolt)`/`Lacks(Bolt)` action change.
4. **Learned belief formation.** Replace or reweight the reference prior from
   viewer history, measure calibration separately, and then evaluate the whole
   autonomous loop.

The teacher and student increments depend on the keystone, but the keystone is
not a teacher API. It lands only with an ordinary autonomous consumer and an
intervention proof.

Within this branch the keystone is an indivisible architectural change. Types,
the reference belief model, query operations, tensor projection, agent
integration, and the demo may be implemented in internal slices, but they ship
together; a contract-only or teacher-only subset is not independently useful.

## Keystone data structures

The canonical source belief remains the architecture's existing contract:

```python
@dataclass(frozen=True)
class BeliefState:
    space: PossibleWorldSpaceIdentity
    model: BeliefModelIdentity
    normalized_distribution: WorldDistribution

@dataclass(frozen=True)
class BeliefUpdate:
    previous_belief: BeliefStateIdentity | None
    viewer_history: ViewerHistoryIdentity
    belief: BeliefState
    update_receipt: BeliefUpdateReceipt
```

The world-space identity binds the current viewer Observation, history, and
world schema. `BeliefUpdateReceipt` binds the producing model, previous belief
when present, consumed history range, normalization result, and output digest.

The first policy-facing projection is deliberately cheaper than the source
belief:

```python
@dataclass(frozen=True)
class BeliefTensorView:
    schema_identity: str
    card_def_ids: Array              # [rows]
    owner_role_ids: Array            # [rows]
    hidden_zone_ids: Array           # [rows]
    count_probabilities: Array       # [rows, count_buckets]
    validity: Array                  # [rows]
    entropy: float
    effective_support: float
    encoding_receipt: str
```

Rows are canonically keyed by `(viewer-relative owner, hidden zone,
CardDefId)`. Count buckets and padding are schema-bound; the content manifest
binds the card vocabulary. `encoding_receipt` hashes the schema and canonical
tensor values. Query text, positional condition index, belief source, and
artifact path are provenance only and cannot become model features.

The initial view preserves each row's complete count marginal but not
cross-row correlation. The exact `BeliefState` remains in trajectories and
datasets so a later hypothesis-set or correlation-aware encoder does not
change evidence meaning.

## Shared model structure

Belief formation and decision-making should share semantic understanding
without hiding the intervention boundary:

```text
Observation/history ──> shared semantic embeddings/encoder E
                                  |                |
                       belief updater B       current-state features
                                  |                |
                             BeliefState           |
                                  |                |
                       BeliefTensorView --------> decision core D
                                                   |
                                             policy and value
```

The keystone shares card, zone, player, event, and other semantic embeddings,
plus current visible-state encoding where shapes permit. The belief updater
and decision heads remain distinct. The decision core receives no private
recurrent activation from the belief updater; all belief-dependent information
crosses the explicit `BeliefState`/`BeliefTensorView` boundary. This makes an
override causal even when parameters are shared.

Belief calibration loss and policy/value loss are separate. In the keystone,
policy loss does not train a learned belief updater implicitly. Later joint
gradient flow is an explicit, checkpointed experiment rather than an invisible
default.

## Belief training contract

The initial distribution is managym's exact pre-mulligan combinatorial prior.
The deployed belief updater does not receive or require the acting policy. It
learns the likelihood-ratio correction implied by viewer-safe actions across a
training population. A known-policy normalized Bayes update is diagnostic only,
used to check whether an observed update is explainable when that policy is
available. Learned belief models are supervised against materialized hidden
hands with whole-world negative log likelihood; when exact categorical support
is too large, an autoregressive factorization may represent the same normalized
joint distribution.

Independent card or zone marginals are never the primary belief loss because
they erase joint correlations. They remain derived diagnostics and cheap
policy/value tokens. Training examples and checkpoints bind the opponent
policy/version plus provenance because changing the opponent induces covariate
shift in action-conditioned belief updates. Actual hidden truth is available
only to access-controlled supervision and audit code, never the inference
path.

## Keystone functions

```python
class BeliefModel(Protocol):
    identity: BeliefModelIdentity

    def update(
        self,
        *,
        previous: BeliefState | None,
        world_space: PossibleWorldSpace,
        viewer_history: ViewerHistory,
    ) -> BeliefUpdate:
        """Form the current viewer-safe belief without private authority."""


def query_mass(
    belief: BeliefState,
    query: WorldQuery,
) -> float:
    """Measure a managym query under the canonical belief distribution."""


def condition_belief(
    belief: BeliefState,
    query: WorldQuery,
) -> BeliefState | EmptyBeliefSupport:
    """Restrict and normalize belief without consulting actual truth."""


def encode_belief(
    belief: BeliefState,
    schema: BeliefEncodingSchema,
) -> BeliefTensorView:
    """Create the deterministic, receipt-bound policy projection."""


class Manabot:
    def decide(
        self,
        decision: ViewerDecision,
        memory: AgentMemory,
    ) -> AgentStep:
        """Update belief, score legal actions, choose a Command, update memory."""

    def evaluate_under_belief(
        self,
        decision: ViewerDecision,
        belief: BeliefState,
    ) -> PolicyValueResult:
        """Evaluate one explicit intervention without mutating agent memory."""
```

Both public paths call the same internal decision core after selecting the
belief. Supplying the exact belief generated by `decide` must reproduce its
policy/value bytes under deterministic execution.

## Query interventions and the teacher role

Queries create experimental beliefs; they are not policy features:

```python
base = belief_model.update(...).belief
has_bolt = condition_belief(base, Has(opponent_hand, LIGHTNING_BOLT))
lacks_bolt = condition_belief(base, Not(Has(opponent_hand, LIGHTNING_BOLT)))

agent.evaluate_under_belief(decision, has_bolt)
agent.evaluate_under_belief(decision, lacks_bolt)
```

Search consumes the same explicit belief. When search is used as a teacher, it
simply records its visit distribution, Q values, and root value as training
targets for the same `(Observation, BeliefState, legal actions)` input. Those
targets do not override belief. A teacher-forced curriculum overrides the
belief input and separately supplies policy/value labels produced under that
same belief.

For intermediate Bolt probability `p`, the supplied belief is
`p * b_Has + (1 - p) * b_Lacks`. This changes belief, not a scalar condition
tag, and remains defined only when both endpoint supports are non-empty.

## Failure behavior

- A belief whose world-space identity does not match the decision fails before
  inference.
- Missing, negative, non-finite, or unnormalized weights fail closed.
- Empty query support returns `EmptyBeliefSupport`; it never falls back to the
  base belief.
- An unsupported tensor schema or content vocabulary mismatch fails rather
  than dropping belief rows.
- An intervention is explicit in provenance and is not committed into agent
  memory unless a separate stateful simulation API requests that behavior.
- Search cannot reconstruct worlds independently from per-card marginals.

## The keystone demo

One documented `uv run ...` command loads one retained viewer decision and:

1. runs the ordinary agent path, showing the generated belief identity,
   `P(Has(Bolt))`, complete legal-action distribution, value, and Command;
2. re-evaluates the same decision under the exact generated belief and proves
   byte-identical policy/value output;
3. derives `Has(Bolt)` and `Lacks(Bolt)` beliefs through typed queries and
   evaluates both while holding Observation, legal actions, weights, and
   current-state encoding fixed;
4. prints the changed semantic Bolt count token and complete policy delta;
5. swaps viewer-equivalent actual hidden worlds and proves the generated or
   supplied belief path cannot observe the swap; and
6. reports belief-update, belief-encoding, and policy inference latency.

A small trained overfit probe may show that the policy has capacity to react to
the two belief views. That is wiring evidence, not the later strategic action
change claim.

## Keystone done when

- Every ordinary agent decision forms a valid belief before policy/value or
  search runs; there is no silent neutral `condition_index` fallback.
- The reference compatible-deal model can recompute `b(t)` from canonical
  viewer history, and its receipt binds the relevant identities.
- Autonomous and explicit-belief paths converge on one decision core and agree
  exactly when given the same belief.
- `WorldQuery` measurement and conditioning operate on the canonical world
  distribution, not marginal tensors.
- Equivalent distributions encode identically regardless of query spelling or
  hypothesis order.
- Policy/value receive canonical per-card belief tokens with no query or
  positional-condition feature.
- Empty support, identity mismatch, normalization failure, and schema mismatch
  fail explicitly.
- Viewer-equivalent hidden-truth swaps cannot affect output with Observation,
  history, and belief held fixed.
- The runnable demo and focused tests exercise the real configured agent path.

## Follow-up evidence

The paired teacher increment scans a bounded real workload before student
training. It reports conditional policy distance, stable top-action flip rate,
decision types, and label cost, then freezes showcase and held-out roots. If no
stable teacher action flip exists under the declared compute cap, it records
`KILL_NO_CONDITIONAL_TEACHER_SIGNAL` instead of training on identical labels.

The distillation increment trains multiple seeds under paired supplied beliefs
and compares baseline-belief and shuffled-belief controls. Success requires a
preregistered held-out action flip, agreement with both condition-specific
teacher actions, suite-level conditional-policy fidelity, zero viewer-safety
failures, and declared teacher plus student cost. Search is off during the
policy-only demonstration.

The learned-belief increment supervises calibration against access-controlled
hidden-world labels, then evaluates autonomous play using generated beliefs.
It does not weaken or replace the explicit intervention interface.

## Measures

Keystone systems measures:

- belief update and marginal-projection latency;
- policy inference overhead from belief tokens;
- exact autonomous/intervention agreement under the same belief;
- query/conditioning invariance and viewer-safety failures; and
- model capacity to distinguish intentionally different belief views.

Later research measures:

- belief calibration and query-probability calibration;
- stable teacher action-flip prevalence and label cost;
- student reproduction of teacher conditional policy deltas;
- degradation under baseline and shuffled-belief controls;
- policy/value/search strength at matched compute; and
- variation across belief, search, and training seeds.

## Constraints

- Belief formation is fundamental agent behavior; search teachers and students
  do not get separate belief semantics.
- managym remains the only authority for worlds and typed query meaning.
- manabot assigns and learns weights; it does not invent a parallel hidden
  ontology.
- The canonical belief is a distribution or scorer over worlds. Per-card
  marginals are derived policy inputs, never authoritative query semantics.
- Shared parameters may improve representation learning, but no opaque
  belief-carrying activation may bypass the override seam.
- Generated and supplied beliefs use the same decision core. Their origin is
  provenance, not a model feature.
- Policy/value teacher targets and belief overrides are separate channels.
- Actual hidden truth is a calibration target only.
- Current INT-13 and INT-14 artifacts remain plumbing evidence: INT-13 has no
  Bolt top-action flip, and INT-14 repeats identical labels across conditions.
- The keystone must immediately support real conditional search measurement;
  it cannot grow into an unexercised general belief framework.
