Make catalog-wide naming decisions complete and scalable for policies

Jack Heart authorized this implementation Task on 2026-11-04 as one of three
engine improvements before broad Standard authoring. Run the code Graph, publish
a reviewable PR and stop at interactive review; no merge or training experiments
are authorized.

Outcome: The Rise of Sozin's naming choice remains fully available and learnable
as the installed catalog grows beyond 512 names, without rebuilding one owned
effect program for every name at each resolution or silently restricting choices.

At inspected source 5dd11cec4, flow/resolution.rs NameAndExileFromTargetPlayer
constructs a BTreeSet<String> from every non-token catalog name (including real
back-face names) and a Vec<Effect> for each Decision::Modal branch. Fresh inputs
have 512 action rows; Python encoding fails explicitly beyond capacity. Recheck
current main rather than assuming line numbers or limits are unchanged.

Design and implement immutable shared name candidates and a scalable semantic
choice representation with complete policy scoring. Compare candidate-based
encoding/scoring against a shared structured choice if needed; choose based on
decision semantics, observability and measured costs. Merely increasing the
constant, truncating options, choosing automatically, or browser-only filtering
does not meet the outcome. Every meaningful choice remains accessible to the
policy, search, Command authority and human interface. Any multi-step protocol
must have an explicit common meaning, not human-only semantics.

Preserve the currently accepted real-catalog naming universe, including back
faces, without inventing unsupported definitions. Explicitly identify the rules
question of names outside the supported catalog rather than quietly broadening
card admission or pretending an engine enum establishes all legal Magic names.
Choose a representation that can admit a larger validated vocabulary later.

Current and historical policy inputs need deliberate compatibility boundaries.
Preserve frozen shapes and omitted checkpoint defaults; use a new explicit
version if required. Integrate scalar/vector native encoding, Python parity,
learning tokens/definition references, model action scoring, LLM text, Academy,
stale rejection, replay and reconnect. Preserve private search visibility and
fail-to-find semantics after the name is selected. Do not fake outside candidates
as in-game cards or leak unseen opponent definitions under closed disclosure.

Acceptance uses a controlled catalog with more than 512 actual validated names,
including choices on both sides of that boundary and back-face names. Prove
both native legality and policy reachability, distinct encodings/learnable
preferences, correct named search continuation, replay/fork/stale rejection and
viewer privacy. Capacity failures remain explicit; no random-policy fallback.
Measure catalog setup separately from per-decision allocations, observation
encoding/scoring and ordinary step/search cost. Avoid requiring every unrelated
card program to be encoded on every learner step.

Own the naming decision, candidate representation and its policy/client consumers.
The composable-execution Task owns general frames and search composition; coordinate
its continuation interface. Mana and layers Tasks own their separate rule domains.
Do not copy their implementations into this branch; use normal Task sync.

Follow AGENTS.md and docs/development.md required checks; retain debug guards,
same-runner Engine Budget and exact digests for unchanged semantics. Explain
changed decision traces. No artificial load, paid experiments or rewrite of
frozen evidence. A generally working trainable engine is valuable despite bugs;
prioritize defects that threaten training or the validity of learning claims.
