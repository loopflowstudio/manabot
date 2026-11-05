# MTG-243 — complete catalog naming

## Shipping authorization — 2026-11-05

Jack Heart reviewed the sequential naming design and implementation and said
“ok lg2m ship it”. This authorizes shipping PR #424 through the remaining Task
steps, merge, verification and completion, superseding earlier review-only
boundaries below. It does not authorize paid or strength experiments. Full
end-to-end throughput measurement remains unnecessary before shipping.

Preserve the interactive review's evidence limits in durable PR copy before
scratch cleanup: the group-gradient and rollout witnesses do not demonstrate
context-dependent routing between different groups or comparative learning
quality; the proposed paired-context group witness remains unrun. Group means
still build a vocabulary-wide prefix sum per forward. No end-to-end speedup is
claimed. The separate compound/mana-payment learning study acc0fe remains filed
and unstarted. Reviewed head 833ef4120 has all CI checks passing, including main
integration and Engine Budget; update the stale pending-CI wording in PR copy.


## CURRENT ACCEPTED REVISION — 2026-11-05, sequential naming and decoder removal

This section supersedes earlier same-day proposals for bot-internal hierarchical
decoding, a flat/compound hybrid, and one-transition route credit. Those remain
historical discussion below, not current implementation direction.

Jack Heart directed: delete the active neural compound decoder; introduce name
selection as simple serialized hierarchical decisions; file a separate Task to
experimentally validate compound decision-making, explicitly including mana
payment. He confirmed the exact meaning: **ordinary sequential engine decisions**
for choose-a-group then choose-a-name, each using the regular policy head and
normal rollout. Only the final leaf commits the selected name. Continue MTG-243
and PR #424 through implementation and publication to interactive review. No
merge/Auto-Ship, paid execution or strength campaign is authorized. Full end-to-end
throughput measurement is still not a shipping prerequisite.

### Implementation outcome

- Remove the neural CompoundDecoder and its active model-wide alternate policy,
  compound-only learner/rollout/search glue, runnable study recipes and obsolete
  feature/configuration branches that exist solely to support it. Trace consumers
  before deleting; remove dead parallel paths instead of retaining disabled copies.
  Keep shared flat-policy encoders, naming identity, packed-rollout improvements,
  shared native legality/offer/Command APIs needed by real remaining consumers,
  and the broader Task's completed correctness/performance repairs.
- This is retirement of an unvalidated alternate learning architecture, not a
  claim it never ran or that compound choices are worse. ETU-94's retained 56-game
  proof establishes wiring only; Elemental 10x10 MTG-181/183 uses the flat policy.
  Retain historical evidence byte-for-byte and point to historical source
  ba3cd58de25bcc5d0b32e0fa3c546ee3d27be52b for reproduction. Historical compound
  checkpoints must fail clearly if execution is no longer supported, never be
  loaded as flat policies. Preserve ordinary/frozen configuration omissions and
  offline evidence readers; do not retain the full decoder just for metadata.
- Native NameCard becomes a complete bounded hierarchy of ordinary decisions:
  publish legal groups; selecting one advances the pending naming scope; publish
  its children until a leaf set fits; selecting a leaf resumes the original
  effect. Use deterministic balanced complete grouping, initially at most 64
  options per prompt, recursively for larger vocabularies. No reject/next paging,
  hidden auto-selection or missing names. Skip single-choice decisions according
  to the existing native skip_trivial contract.
- Group navigation changes pending decision state but commits no naming effect,
  resource cost or private inspection. Only selecting the actual name starts the
  existing private SearchNamed continuation. Preserve committed exiles, optional
  stopping and fail-to-find semantics. Reconnect/fork/replay restore the actual
  current group or leaf prompt; stale earlier-group Commands reject. No undo or
  cancel lifecycle is newly authorized.
- Every exposed hierarchy stage uses the ordinary policy forward, categorical
  distribution, rollout, value estimate and existing learner. No separate GRU,
  internal token tape, joint-path PPO ratio or special compound KL. Standard
  temporal discount/credit applies: document the extra decisions and do not claim
  learning equivalence to an atomic flat choice. No intermediate reward is invented.
- Group choices need viewer-safe, machine-readable identity/features that let the
  regular policy distinguish their candidate sets; presentation labels alone are
  insufficient. Bind grouping/membership to the vocabulary/decision identity.
  Preserve useful public name/definition joins, including names without definitions,
  and never infer hidden membership. Avoid expanding every ordinary live buffer
  or learner minibatch with the catalog. Ordinary action capacity must be independent
  of vocabulary size; shared immutable catalog storage can remain linear.
- The new native decision sequence is shared authority. Academy and LLM clients
  should work through their existing ordinary-offer controls/text projections;
  no bespoke Academy redesign or new LLM tool interface is requested. Update the
  normal native/Python/scalar/vector/model/text/Command consumers and focused
  shared-surface proofs needed for the changed contract.
- Preserve frozen v1–v8 input layouts/semantics and saved evidence. If bounded
  group representation/transport needs a new input version or live world/capture
  transition, make it explicit; never reinterpret v8 name rows as groups. Fresh
  default adoption remains a separate decision unless explicitly necessary to
  expose this Task's opt-in feature. Unsupported old configurations must reject
  explicitly instead of truncating choices or pretending to understand groups.

### Required proof and review

Use one real native vocabulary beyond 512 (existing 928-name fixture) through
scalar/vector/ordinary policy/Command and current shared clients. Prove every name
is reachable exactly once, balanced bounded branching including deeper trees,
learnable group and leaf distinctions with gradients, correct final effect and
no effect before leaf selection, private visibility, stale/replay/reconnect,
trivial-decision skipping and fail-to-find. Exercise ordinary flat training through
both group and leaf transitions. Keep tests meaningful rather than mirroring a
partition function. Measure bounded buffer sizes and ordinary/naming costs with
controlled fixtures; no artificial load. Apply docs/development.md checks, rebuild
Rust extension atomically, run debug Rust, and retain the unchanged Engine Budget
reference/tolerance unless actual semantic digests require documented adoption.
Published old-head CI is not evidence for this new architecture.

The implementation may settle routine data structures, grouping and deletion
mechanics within these constraints. If a real product/compatibility conflict
cannot be resolved without changing accepted behavior, bring that precise issue
back to this conversation. Do not restart design merely to re-ask the accepted
choice of ordinary sequential decisions.

### Coordination

Open PRs at this decision: #433 / MTG-241 owns execution/payment continuations;
#427 / MTG-244 owns layers and shared observations; #434 / MTG-252 owns critic
readout changes; #383 / MTG-175 owns curriculum/learner work; #428 / MTG-245 owns
league protocols. MTG-249 / #421 owns effective-state/agent-memory design. Sync
through lf as appropriate, preserve their work and do not copy their active
implementations. Keep deletion narrowly about the neural compound architecture;
engine APIs with existing non-neural consumers have separate responsibilities.

A separate experiments-Wave Task is being filed for controlled learning validation
of sequential versus compound choices including mana payment; filing is not an
experiment launch. Its brief pins this source and separates prior wiring proofs
from scientific validation. No compound architecture is to be reintroduced here
on speculative benefit.

### Implementation — 2026-11-05, committed `acd639aa4`, synced `9a4eca36d`

Native: `Decision::NameCard` holds `scope: NameScope` (`cards/names.rs`). A scope
over `NAME_BRANCHING` = 64 offers `ceil(len / 64^(d-1))` balanced contiguous
groups (sizes differ by ≤1); choosing one only narrows the scope and re-suspends;
a leaf name choice pushes `ExileNamed`, whose `SearchNamed` continuation keeps
private inspection, optional stopping and fail-to-find. Labels: "Names A through
B" / "Name X"; the instruction says choosing a group commits nothing.
`ActionOption` gains `name_group: [first, last]` (JSON/PyO3/stub). 928 names →
15 groups then a leaf; installed 344 → 6 groups. Live schema 8 / capture 5 /
hash 16 (this Task's single unreleased transition, edited in place); historical
captures 3/4 still validate read-only under schema 7.

Input: explicit v9 `name_candidates[A,4]` (name, definition, group first, last;
one-based) at `A = max_actions`; native vector infers v8/v9 from width. V8 rejects
any group prompt; v1–v7 see anonymous modal choices. Model: a group is the exact
prefix-sum mean of member name vectors (padding excluded); name identity joins
viewer-visible program references before card/known-hand/decklist pooling;
closed decklists contribute nothing. Checkpoint binding `public-groups-v1` plus
`naming_branching`; no remapping. Live observation bytes at default capacity:
v9 155,120 independent of vocabulary; v8 151,024 (344) / 195,952 (928). Native
single-sample release controls at the 928-name root: action space 600 B / 1
allocation, observation 9.3 KB / 58, `encode_into` 792 B / 5 (~3 µs). Not
throughput claims.

Deletion: neural `CompoundDecoder`, `ActorCritic.compound`, `compound_decisions`,
`TrainCompound`, compound rollout/learner/search/action modules, study recipe and
exclusive tests are removed. Historical `compound_decisions: true` configs,
checkpoints and `train_compound` regimes fail with a pointer to `ba3cd58de`
(`docs/training.md` "Compound decisions (removed)"); local-search receipts keep
their fields (None/0) and reject a compound prefix. Kept deliberately: native
compound-offer lowering and bindings (questions.md), `CompoundPrefixReceipt` /
`decoder_factors` for historical receipts and Academy arena display, report
readers of retained compound runs. No further deletion target remains.
PUCT/local-search admission source hashes change.

Carried forward unchanged from earlier passes (detail in
`.lf/logs/naming-sequential/plan-before-realign.md` and
`b31998cc3:scratch/mtg-243-naming.md`): `PackedActions` legal-row rollouts with
complete behavior probabilities and original coordinates; bounded CPU behavior
validation; numerical incidents schema 2; the speed pass's chunked packed GRU
(`SemanticCardEncoder._encode_programs`, ≤64 steps, full cross-chunk gradients),
measured ~5.8–6.6x lower width-64 mixed/naming batch CPU vs `b31998cc3` and 91%
less naming slice-backward allocation traffic, with width-16 ordinary PPO/damped
regressions and a failed identical-binary native worst-reply control kept beside
them. Those measurements predate sequential naming (whose group prompts encode no
member programs) and are not retimed. Jack Heart (2026-11-05): an end-to-end
throughput measurement is not a shipping prerequisite; no end-to-end training or
native speedup is claimed. Engine Budget reference `ea316ff14` and 10% tolerance
are unchanged; earlier Linux runs passed at +4–8% aggregate cost that includes
intervening engine changes, and the local paired worst-reply failure remains
recorded, not cleared by retry (`.lf/logs/naming-budget-*`).

Tests: native hierarchy/partition (depths with branching 2/4/7/64), Sozin card
tests through groups, v9 scalar/vector/Command parity across group and leaf,
group/leaf gradients, stale/replay/reconnect in Academy at both prompts, text
policy, and a real self-play rollout + PPO update through group and name
transitions (`tests/training/test_naming_transitions.py`).

### Status and remaining

- Pre-sync checks (source before `acd639aa4`): debug `cargo test --workspace` 972
  passed; clippy/rustfmt clean; focused Python (model, naming, Academy recovery,
  text, verify, pilot, current-world admission, boundaries) 611 passed / 1
  skipped; deletion batches 394 passed with one pre-existing missing-data failure
  (`int-13` fixture absent in this worktree).
- Sync onto main `264041f99` (`9a4eca36d`): main deleted docs/ARCHITECTURE.md and
  docs/search-and-beliefs.md and moved policy inputs to
  docs/observation-and-model.md. The catalog naming section is relocated there
  and the managym README link retargeted — **uncommitted** at this
  reconciliation; it must be committed before publication.
- PR #424 head is `9a4eca36d`; its Linux CI (debug Rust, Python, integration,
  Academy, conformance, Engine Budget at the unchanged 10%) was pending at this
  reconciliation. Published old-head CI is not evidence for this architecture.
- Publication: PR copy and review artifacts (`scratch/pr-review.html`,
  `scratch/concept-review.html`, `scratch/naming-risks.html`) still describe the
  atomic v8 naming experience and must be refreshed for group → name prompts,
  v9 and the decoder removal. Then demo and interactive review. No
  merge/Auto-Ship, paid execution or strength campaign.

### Concept review — 2026-11-05, against `9a4eca36d` plus the uncommitted docs relocation

No design change and no open product choice. Actions → concepts → APIs agree:
a person or policy sees "Names A through B" group offers, then "Name X" offers,
all ordinary `ActionOption`s (`name_group` span) through the shared offer,
Command and text paths; v9 `name_candidates` carries the same span; only the
leaf pushes `ExileNamed`. Names line up across layers (`NameScope`,
`NAME_BRANCHING`, `name_group`, `public-groups-v1`). Checked deliberately:
- Retained `compound_offers` / `compound_commands_json` have no Python caller,
  but they are members of the live structured-offer family (`structured_offers`
  is used by Academy studies and `manabot/sim/action/benchmark.py`), have Rust
  rule tests, sit in MTG-241's payment domain and do not touch naming. Keeping
  them follows "deletion narrowly about the neural compound architecture".
- No back/cancel from a group prompt is consistent with "no undo or cancel
  lifecycle newly authorized"; every name stays reachable exactly once.
- V1–v7 anonymous group choices do not pretend to understand names (they never
  carried identity); v8 rejects. Reversible assumption stays in questions.md.
Remaining before demo is delivery, not design: commit the docs relocation,
refresh `scratch/pr-review.html` and `scratch/naming-risks.html`, read PR #424
CI on the new head.

### Proposed Wave memory update (not applied; standalone realign)

`.lf/wave/coverage/MEMORY.md` "Standard direction and catalog naming" still
states the atomic v8 design as review-pending. Replace with: Jack Heart
(2026-11-05) accepted ordinary sequential native group → name decisions
(`NameScope`, branching 64, only the leaf commits) through the regular policy
and rollout, explicit v9 `name_candidates[A,4]` at ordinary action capacity,
and removal of the neural compound decoder (reproduce from `ba3cd58de`);
compound/mana-payment learning validation belongs to study acc0fe. Keep the
v8-era learnability lesson (identity joins needed for equal-rule names) and
the budget-evidence lessons.

## Coordination and integration

Open PRs at the accepted revision: #433 / MTG-241 owns execution/payment
continuations (it reported `FrameCompletion`, obligation codes 27–29; this
checkout still has `FrameFinalize`, max code 26, naming code 10 — ordinary sync
reconciles them, no collision known); #427 / MTG-244 layers and shared
observations; #434 / MTG-252 critic readout; #383 / MTG-175 curriculum/learner;
#428 / MTG-245 league protocols; #421 / MTG-249 effective-state design. Native
hotspots from the speed profile (stabilization, state-based actions, action
generation) sit in MTG-241/242/244 domains and stay with them. Sync through lf;
do not copy their implementations.

## Superseded history

Atomic v8 whole-vocabulary naming, its concept reviews, the pagination /
bot-internal group→name / compound-decoder investigations and the full speed and
budget record are retained verbatim in
`.lf/logs/naming-sequential/plan-before-realign.md`,
`.lf/logs/naming-sequential/plan-before-curation.md` and
`.lf/logs/naming-hierarchy/design-before-curation.md`. They do not direct
implementation.

Check: `lf context` and `git diff --check` after this reconciliation; behavioral checks reused, not rerun.


### Interactive concept review — group learnability, 2026-11-05 at 833ef4120

Jack Heart asked how the groups are made learnable. Inspection of
NameCandidateEncoder._groups, ActorCritic._gather_informed_actions and the naming
tests finds a shared identity mechanism: balanced alphabetical native spans are
embedded as the arithmetic mean of their trainable member-name vectors. Those
same name vectors join viewer-visible definitions before card/known-hand/decklist
pooling. The regular action head can therefore relate group membership to public
state. Group labels/endpoint magnitudes are not the model's learned meaning, and
no separate group embedding, learned partition or compound decoder is present.
Group vectors contain name identities, not pooled executable member programs;
leaf scoring adds the selected name's optional definition/program vector.

Each member's derivative through the group mean is 1/member_count times the
group-vector derivative. This shares signal and avoids dead members, but it is
not targeted credit identifying which member caused a good outcome. Mean pooling
can dilute a rare useful member or collapse distinct sets; distinct means in one
seeded test do not prove universal distinguishability. Alphabetical grouping
supplies no strategic similarity prior. This is a plausible simple baseline,
not demonstrated learning quality or a reason to reintroduce compound machinery.

Existing witnesses: test_catalog_naming_large_support_learns_and_rejects_wrong_binding
checks distinguishable seeded group vectors, positive support, one supervised
SGD step reducing a target group's loss, all member gradients and selected native
execution. test_equal_rule_names_learn_opposite_public_preferences is explicitly
a within-leaf, two-state linear-probe representability witness (nonzero FiLM), not
SGD convergence. test_naming_transitions exercises ordinary rollout/PPO and changed
name weights, not correct context-dependent group selection. No analogous paired
public-context witness for different groups was found. Proposed next proof: use
real native group prompts and two public states whose intended names lie in
different groups; verify routing preferences can reverse, remain independent of
action ordering, and lose hidden-information distinctions under closed disclosure.
A bounded optimization witness would strengthen this beyond a fitted final head.
This is a recommended engineering proof, not a new strength campaign or permission
to rewrite implementation/restart a Graph in this concept-review turn.

Performance caveat: _groups builds a full name-table prefix sum once per encoder
forward, even when all supplied spans are zero. This retains O(vocabulary*width)
additions per forward; it does not expand every environment's action buffer or
encode every member program. No new timing or claim of end-to-end speed is made.

Experience/design unchanged: ordinary public group prompt -> leaf prompt ->
private SearchNamed, final leaf alone commits; recovery retains current scope,
stale previous-prompt Commands reject, committed exiles survive. Proposed usage
wording: “Choose a public alphabetical range, then the exact name. Selecting a
range narrows the pending choice; it does not inspect or exile cards.” Existing
docs accurately describe that contract; no product docs/skill edits are required.
Concept outcome: retain the accepted simple sequential design, with the evidence
limit above visible in review. No new product choice or shipping approval inferred.

Observed delivery: PR #424 is open at 833ef4120bcc71062c33e52c48a8e4a60cee6ff2,
merge state CLEAN and required checks green (release publish skipped); Graph
aba71092 completed pr-publish and Task stands at demo. Earlier reconciliation's
uncommitted-docs/pending-publication statements are historical, superseded by this
read. CI passing does not supply the missing contextual group-learning witness.
