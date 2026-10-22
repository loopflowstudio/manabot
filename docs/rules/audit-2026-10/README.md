# Rules audit: proposed next work

Use the audit to find rules work that is falling between existing efforts.
Start with simultaneous trigger ordering, then examine target validity, zone
changes and replacement interactions. Payment repairs already have owners and
should stay with them.

This is a proposed sequence for review, not a launch of those changes. The
first draft built a complete inventory, but most entries received only broad
triage: **2,974 remain unknown**. That gives us a map for investigation, not
enough evidence to rank every missing mechanic or claim a complete audit.

## 1. Establish the trigger-ordering gap

CR 603.3b lets a player choose the order of simultaneous triggers they control.
The inspected implementation instead orders pending triggers by APNAP rank
and enqueue order. It offers target choices, but has no trigger-order decision.
This is the strongest independent implementation gap identified so far; a
small behavioral example should establish its effect before a redesign.

Start with two distinguishable triggers controlled by one player whose order
changes the result. Specify both legal orders and their expected outcomes from
the rules, then check what the shared action surface can express. Add a second
case with triggers controlled by both players to separate APNAP ordering from
each player's choice. First check current Task ownership to avoid duplicating
a newer effort.

**Checkpoint:** a tiny failing witness, exact native owner, and a bounded repair
proposal. If confirmed, implement the choice in the shared native decision
model and exercise human and bot actions, target selection, stale rejection,
replay and fork restoration. Academy should expose that same decision.

## 2. Review target validity across zone changes

Work through CR 115, 400.7 and 608.2b together. These rules meet at a common
failure boundary: an object can leave and return while a spell or ability is
waiting to resolve. Passing replay tests alone cannot prove the MTG result.
The current audit has not established a defect here.

Use small cases for one invalid target, all targets invalid, mixed valid and
invalid targets, and a target leaving and returning before resolution. Read
existing tests first and add only missing distinctions. Write expected effects
and object identities before checking engine output.

**Checkpoint:** each reviewed rule has an exact owner and behavioral evidence,
or a specific unresolved question. Any failure gets a minimal regression and
a focused fix; passing cases close evidence gaps without prompting a refactor.

## 3. Check state-based actions against replacements

Extend the draft's cheap toughness/damage/indestructible matrix into selected
CR 704 and 614–616 interactions supported by authored cards. Prioritize a
simultaneous death batch, a replacement affecting a zone move, and competing
applicable replacements where current content permits them. The existing
matrix does not establish these interactions.

Keep ownership visible: stabilization determines the batch, proposed events
apply replacements, and zones commit movement. Document the invariant at the
authority that enforces it. Propose consolidation only when a failing example
shows that dispersed ownership is causing trouble.

**Checkpoint:** independent expected outcomes for each selected interaction,
bounded fixtures, and an explicit distinction between supported-play defects
and mechanics the admitted world cannot yet express.

## 4. Continue the audit in useful batches

After those boundaries, review supported combat restrictions and P/T effects
under CR 508–510 and 613. The inspected P/T implementation is additive and
does not establish general timestamp/dependency behavior. Determine which
current cards require more before proposing a general layer engine.

Keep every numbered rule in the inventory, including unsupported formats.
Upgrade entries individually as code, intent comments and tests are reviewed.
For each batch, produce a short account of confirmed defects, closed evidence
gaps and remaining uncertainty. Missing mulligans, poison, ante and variants
remain admission/product decisions; their absence alone does not select them
for implementation.

## Existing work and validation budget

MTG-174 owns the shared payment model for offers and execution. Optional-payment
Task `4ba5dd` owns composable costs and continuations. Retain the Prism/Treasure
counterexample as evidence for that work; do not create another payment rewrite
here. The original audit's ownership observations are historical snapshots and
must be refreshed before any implementation starts.

Keep fixtures deterministic and small. Use debug Rust tests, uv-managed Python
checks, two Cargo build jobs and the recorded per-command time limits. No paid
compute, stress loops or coverage removal. Keep CI topology unchanged while
the added duration reporting gathers evidence. The sampled runs spent
35–46 runner-minutes for 9–11 minutes of wall time; more shards are not yet a
demonstrated improvement. [CI analysis](ci.md) explains the tradeoffs.

## Draft already available

Commit `2b6e2b1b7` adds bounded mana and state-based-action tests, corrected rule
comments, inventory tooling and CI duration reporting. Retained validation has
300 passing debug rules tests, one new ignored payment probe separately
reproduced as failing, and 44 passing final Python checks. Production edits
change comments only. The original draft accounts for 621 lines including
local helpers, separately from report/index artifacts.

The [detailed evidence report](evidence.md) preserves the original findings,
source pin and validation limits. The [rule index](index.jsonl),
[authored reviews](reviews.json), [validation receipts](validation.json) and
[line accounting](line-accounting.json) are reference material for the work
above. No additional implementation Flow, publication or merge is selected
by this plan.
