# MTG-177: supporting audit evidence — original draft

The [working plan](README.md) supersedes this report's investment ordering.
Payment findings below remain evidence for existing owners, not proposed new
work for this audit. The findings and validation describe the original draft.

This draft delivers a complete **numbered inventory and conservative triage**,
concrete deterministic coverage, corrected rule comments, and measured CI advice.
It does **not** establish full MTG conformance: 2,974 entries remain unknown.
That is the principal result and investment gap, not a claim that those rules
are absent or correct. The narrow positive findings below concern stated
subsets of ordinary two-player authored play. No world, deck, checkpoint or
frozen receipt changes accompany this draft.

## Pins and completeness

- Audited source revision: `20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f`.
  Draft changes are the local commits containing this directory; findings about
  engine behavior refer to that base unless explicitly identified as new tests.
- Official discovery: [Wizards rules page](https://magic.wizards.com/en/rules).
  Its TXT link resolved to [September 25, 2026 Comprehensive Rules](https://media.wizards.com/2026/downloads/MagicCompRules%2020260925.txt),
  retrieved October 22 UTC (October 21 locally). This pins the published text;
  it does not assume a rules update from the session date.
- Exact downloaded-byte SHA-256:
  `8d860e451f20f38865b725b42d82feb714c725373dd8f3b32b8652b3eeb070ca`.
  [source.json](source.json) records the ordered identifier digest too.
- **3,322 entries: 9 chapter headings, 147 section headings, 3,166 numbered
  rules/subrules.** The body is inventoried once, excluding duplicate contents,
  unnumbered examples, glossary and credits. Multi-letter suffixes are retained,
  including 704.5aa; skipped letters are not invented as gaps.
- [index.jsonl](index.jsonl) contains one row per identifier, in official order,
  with source spans, per-entry content digest, status, scope, authority,
  dispersion, comment assessment, test references and explicit gap. Parent
  status never propagates to children. Source spans allow direct inspection
  without republishing the full rules text.

| Chapter | Entries | Implemented | Partial | Missing | Outside admitted formats | Unknown |
|---|---:|---:|---:|---:|---:|---:|
| 1 Game concepts | 531 | 1 | 7 | 0 | 0 | 523 |
| 2 Card parts | 135 | 0 | 0 | 0 | 0 | 135 |
| 3 Card types | 181 | 0 | 1 | 0 | 0 | 180 |
| 4 Zones | 88 | 0 | 1 | 1 | 0 | 86 |
| 5 Turn structure | 165 | 0 | 1 | 0 | 0 | 164 |
| 6 Spells, abilities, effects | 338 | 0 | 6 | 0 | 0 | 332 |
| 7 Additional rules | 1,562 | 2 | 5 | 1 | 0 | 1,554 |
| 8 Multiplayer | 184 | 0 | 0 | 0 | 184 | 0 |
| 9 Casual variants | 138 | 0 | 0 | 0 | 138 | 0 |
| **Total** | **3,322** | **3** | **21** | **2** | **322** | **2,974** |

“Implemented” still carries the row's explicit scope and evidence limits; it is
not exhaustive proof. “Missing” identifies an inspected absent mechanic (ante
zone, poison loss), not an observed failure of an admitted card. Chapters 8/9
are outside current format admission: setup requires exactly two players,
ordinary starting life, and has no variant selection/state machine. Two-player
Commander is also unsupported; player count alone does not exclude it.

## Evidence method and limits

[reviews.json](reviews.json) separates authored per-entry findings from shared
family reviews. Shared reviews list inspected candidate authorities, dispersion
and remaining obligations; they **do not certify their subrules**. Where exact
individual ownership, comment adequacy or behavior is unresolved, the row says
unknown. Each unknown has a rule-specific verification obligation and source
span. Most entries received family triage rather than a completed independent
behavioral adjudication. A future review should upgrade individual rows with
specific tests and findings, not relabel whole families green.

Before adding the inventory tool, this audit inspected:

- [Historical rules map](../../rules_coverage.yaml): broad family claims and
  useful test pointers, but no complete numbered inventory. Its single-block
  combat and no-choice mana prose is stale; the draft adds an explicit scope
  notice instead of treating it as current certification.
- [Card coverage generator](../../../scripts/generate_coverage_gaps.py) and
  [coverage authority](../../../manabot/semantic/coverage.py): authored-card
  compiler/evidence coverage, not a whole-CR inventory.
- [Semantic conformance authority](../../../managym/src/conformance.rs) and
  [w11 corpus](../../../conformance/semantic-kernel-w11-v1/): replay,
  optimization, metamorphic and property evidence. Reference and optimized
  paths share low-level rules primitives. Agreement is valuable and cannot
  independently prove an MTG outcome.
- Registered-card/Scryfall shell tests, directed rules tests and the actual
  setup, mana, zone/identity, damage, SBA, trigger, priority, casting and static
  query authorities. Tests were read alongside the relevant implementation.

[citation-locations.json](citation-locations.json) is an additional mechanical
retrieval aid (110 inventoried identifiers have exact-number hits). It includes
stale citations and is never used to infer implementation or correctness.
Locations describe the draft checkout; authoritative behavior findings pin the
base SHA. No claim is made that every existing test has been independently
validated or that every unknown entry has been individually proved absent.

## Prioritized rules investments

| Priority | Finding and consequence | Concrete evidence / owner | Next investment |
|---|---|---|---|
| P0 | An offered Prism activation with an empty pool and one Treasure fails execution. This violates the shared action contract in supported play. | `flow/mana.rs`, `flow/play.rs`; new `audit_prism_treasure_empty_pool_activation` explicitly reproduces `illegal mana activation`. | MTG-174 owns one shared payment model for offers and execution. Retain this tiny counterexample alongside existing colored-preservation regressions; extend to bounded source-resource combinations after that repair. |
| P1 | CR 603.3b's same-controller simultaneous trigger ordering is fixed by enqueue order; a player cannot choose the order. The chosen deterministic outcome can affect strategy. | `triggers.rs::pop_next_pending_trigger` minimizes `(apnap_rank, enqueue_order)`; `flush_triggers` offers targets, not order, and `Decision` has no trigger-order state. Existing trigger tests do not prove legal choice completeness. | Add one shared native ordering decision, then deterministic tests for two distinguishable triggers, both controllers, APNAP, required targets and suspended replay/rollback. This is a proposal, not a browser-only workaround or a claim of a newly reproduced game failure. |
| P1 | Most numbered obligations lack reviewed independent outcome evidence. Current passing corpora can agree on a common rules defect. | 2,974 unknown rows, including most symbols, card-part exceptions, target rechecks, combat restrictions and keyword interactions. | Review supported-rule islands first: 601/602/605 payments, 115/608 targets and resolution, 603 triggers, 704/614 interactions. Specify outcomes before consulting engine results. |
| P2 | P/T queries implement a limited additive layer model, not general timestamps/dependencies or P/T switches. Old comments incorrectly put +1/+1 counters in 7d. | `statics.rs::effective_pt`; CR 613.4c/613.4d. Counters and additive deltas currently commute. | Corrected comments in draft. Add bounded CDA/animation/anthem/counter/temporary-buff cases before admitting noncommuting effects. General layer redesign is deferred. |
| P2 | Ordinary setup draws seven but has no mulligan procedure; fixed start and short/custom decks are setup capabilities, not tournament enforcement. | `setup.rs::new_with_content`; CR 103.5; setup tests. | Make format/setup limitations explicit in admission; implement mulligans only as a separately accepted rules change. |
| P2 | Ante, poison, multiplayer and casual variants are unsupported. Plane/battle tags or a Command zone are not evidence of their complete rules. | `ZoneType`, `Player`, setup and SBA branches; missing vs not-applicable rows are distinct. | Keep unadmitted mechanics separate from supported-play defects. Require explicit admission plus representation, rules and behavioral tests for expansions. |

The GPU failure's exact failing state was not retained. The reproduced local
Prism failure is **not proof** of the GPU failure's exact cause.

## Coordination and ownership

Fresh `lf task status` reads identified MTG-174 and optional-payment Task
`4ba5dd` as started. The latter's reusable optional-cost/continuation work is
already present in this audited base through #344; its accepted contract is
[optional-payments.md](../optional-payments.md). This audit does not launch their
Flows, rewrite their payment model or alter their training source/budget.
It adds the isolated payment review case and identifies the shared authority.
Task snapshots are retained locally under `scratch/`.

Current ownership is usable but dispersed: `Mana` handles typed pool arithmetic;
`flow/mana` handles production/feasibility and nested payment decisions;
`play` handles announcement and cost locking; `decision` resumes resolution.
`zones` commits movements, `identity` resolves incarnations/LKI, `proposed_event`
handles replacements, and `stabilization` owns the pre-priority fixed point.
Keep those commit and continuation boundaries explicit. Larger consolidations
should follow failing behavioral examples; a universal rules rewrite is not
justified by this audit.

## Concrete draft changes

- **1,152** finite generic/specific mana cases: all 0/1 six-type pools, one fixed
  pip, 0..2 generic; independent resource-counting oracle, colorless distinction,
  conservation and noncreation assertions. **12** restricted-mana contexts check
  ineligible spending and pool clearing.
- **192** SBA cases: both controllers; toughness -1/0/1/2; damage 0/1/2;
  positive deathtouch damage; indestructible. Expected death derives from printed
  Gray Ogre 2/2 and rule distinctions, not an engine P/T or legality query.
  Nine-card setup decks keep this matrix small.
- One explicit **new ignored known-failure probe** for Prism/Treasure, run
  separately to preserve its red result. No existing test was ignored or
  weakened. Its assertions specify final blue mana, Treasure consumption,
  tapped Prism and empty stack. Its final fixture uses nine cards per player,
  bypassing the unrelated ETB draw. This is draft diagnosis, not a passing fix.
- Corrected setup/priority/sacrifice/layer citations and comments explaining
  the implemented subset. No executable production behavior changed.
- Offline inventory/review validation, source-pinned regeneration and five
  Python integrity tests. The CI selector routes inventory/tool changes to the
  existing Python unit job; integration pytest now reports its 20 slowest tests.
  Coverage selection and required-result semantics are otherwise preserved.

## Validation and cost

[validation.json](validation.json) retains exact commands, limits, elapsed time,
results and failure logs. Per-command ceilings were 900 seconds for cold builds,
180 for warm Rust checks and 120 for Python; Cargo used two build jobs per
command. Directed work is bounded to the cases above; no stochastic load search,
paid compute or new CI run was launched.

The debug rules binary passed **300 tests with one explicitly new ignored probe**
(15.47 seconds execution; 92.623 seconds including rebuild). The first focused
new-test invocation passed three tests (0.02 seconds execution; 140.487 seconds
cold build). These are local measurements, not Linux CI forecasts.
The isolated known-failure invocation failed as expected in 0.01 seconds of test
execution. An earlier exact-name filter selected zero tests; that selection
mistake is retained and excluded from positive evidence.

Five inventory tests plus CI-selection/aggregation tests passed (**42 total**,
0.74 seconds execution; 2.953 seconds command time). A final check including package-boundary tests passed **44 tests** in 5.20
seconds (7.456 seconds command time). Initial wheel rebuilding
failed because `maturin` was not installed; enabling the repository's `play`
extra repaired provisioning. The release wheel was built and its cp312 native
extension installed atomically, followed by a successful native import. Release
rebuilding is not a substitute for the debug checks.

No full learnability diagnostic, paid training, arena cohort, broad Python
integration, repeated fuzzing or whole Rust suite was run locally. Changes to
Rust production files are comments only: there is no world/observation shape
change requiring a new admission or checkpoint identity. The focused rules
binary covers the changed tests. Formatting and lint checks accompany the
recorded receipts; earlier formatting failures and their repair are noted.

See [CI analysis](ci.md) for measured wall latency, total runner-minutes,
sharding economics and retained required-check semantics. See
[line accounting](line-accounting.json) for the separate draft budget. This is
local **draft delivery**: no PR publication, merge or auto-merge is performed.

To reproduce the inventory against the official bytes (from repository root):

```bash
curl -fL --max-time 45 'https://media.wizards.com/2026/downloads/MagicCompRules%2020260925.txt' -o scratch/MagicCompRules-20260925.txt
uv run python scripts/audit_rules.py --source scratch/MagicCompRules-20260925.txt
uv run --extra dev pytest tests/infra/test_rules_inventory.py tests/infra/test_ci_selection.py -q
```

Regeneration uses the same command with `--write`; it fails closed on a changed
source digest or an authored rule identifier absent from that source. A rules
edition update should get a new pin and review, rather than silently overwriting
this audit's meaning.
