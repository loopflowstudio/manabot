Generalize mana spending restrictions and payment resources

Jack Heart authorized this implementation Task on 2026-11-04 as one of three
engine improvements for broad Standard support. Launch the code Graph, produce
a reviewable PR and stop at the interactive review boundary; no merge or training
experiment launch is authorized.

Outcome: mana offer generation and execution share typed spending restrictions
and resource transitions that compose beyond the original Lesson/Shrine/Ally
cards and single-use source assumptions. Real card combinations remain payable
exactly as offered, including nested upfront costs.

Source findings at 5dd11cec4, to recheck on current main:
- state/src/mana.rs SPEND_SUBTYPES is [Lesson, Shrine, Ally]. ManaUse uses a
  fixed bit mask, and unrecognized subtype strings disappear in filter_map.
- flow/mana.rs finite witness search reserves each source once. The current
  registry guard accepts tap_source OR once_each_turn, not just tapping.
  Do not falsely classify all non-tapping abilities as unsupported.
- Variable production and additional cost resources require explicit modeling;
  do not add card-specific affordability bypasses or automatic approximations.

Replace the small hard-coded restriction vocabulary with typed predicates and
explicit spending context. Unknown unsupported restrictions must not silently
become unrestricted. Preserve old observation meanings and compatibility
surfaces while moving all executable consumers to the shared representation;
do not retain a parallel restriction system. Treat cost purpose, announced
spell characteristics and activation source/zone explicitly.

Extend resource transitions and executable witnesses for representative
repeatable abilities with bounded consumed resources, upfront costs and
variable mana outputs. Use real cards from the reviewed Standard pool to prove
distinct compositions, selected after inspecting current definitions and Oracle
text (e.g. Three Tree Mascot for non-tapping once-turn filtering; Selvala, Eager
Trailblazer for variable chosen-color production). Do not promise arbitrary
infinite-loop solving or hide unmodeled sources behind an arbitrary iteration
cap. State the supported resource domain and remaining limits explicitly.

Retain the Prism/Treasure lesson: the child payment must preserve a completion
for its parent, and production exists only after upstream costs commit. Preserve
life/tap/sacrifice semantics, exact sources, restrictions and once-turn histories.
Offered implies executable must hold across real Commands, nested choices and
forks. Distinguish tapping for mana from other tap costs.

Acceptance includes restrictions on previously unrepresented subtypes, legal
and illegal spending purposes, multiple sources sharing resources, a constrained
repeatable source, variable outputs, and nested Prism/Treasure payment. Check
both successful payment and final resource remainders. Projection tests must
distinguish strategically different restrictions and outcomes. Keep current
native/Python encoders, learning tokens, LLM text, Commands/replay and Academy
aligned, with explicit version handling for frozen input schemas.

This Task owns mana types, feasibility witnesses and production/resource
semantics. The composable-execution Task owns the general frame protocol;
coordinate the narrow interface rather than implementing a competing stack.
The layers Task owns characteristic derivation; consume its public view.
The naming Task owns catalog-scale choices. Work in this Task's checkout only.

Performance: investigate reusable source inventories within one decision, with
invalidation including mana, taps, life, usage, turn and pending payments; a
characteristics revision alone is insufficient. Optimize only demonstrated
repeated work. Preserve debug witness oracles, ordinary step throughput and
search total/tail evidence under the same-runner Engine Budget. No stress/load
generation, no full storage rewrite and no end-to-end RL claims from engine
timings. Follow AGENTS.md and docs/development.md, including debug Rust checks.

Bugs are expected; useful training need not wait for perfect rules. Prioritize
failures or distortions that could invalidate learning conclusions. Preserve
historical evidence and scope changed semantics; do not revoke playable cards.
