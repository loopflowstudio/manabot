# MTG-243 — catalog naming (2026-11-04)

Jack Heart authorized this outcome on 2026-11-04: complete, learnable catalog-wide
naming beyond 512 names, one reviewable PR, no merge or training experiments.
The Task directive is accepted scope. The atomic candidate design below is the
implemented approach selected under that directive; interactive review is pending.
The inherited T7 note is historical. This Task has no Wave.

## Design

The catalog's immutable, sorted name candidates and one `Decision::NameCard`
replace per-resolution `BTreeSet<String>` and `Vec<Vec<Effect>>` construction.
Only the selected name constructs `ExileNamed`; private search and its permission
lifetime stay at that existing continuation authority. The published modal
Command ordering and labels remain, so humans, LLMs, replay and search choose the same
complete name in one step. Candidates are vocabulary entries, never fake objects.

Explicit policy input v8 has a public name row and optional candidate-definition join per action and
catalog-derived action capacity. This is linear scoring of the complete support;
it avoids sequential rejection or extra game decisions. Candidate identity and
its existing learning definition both reach scoring. Only naming steps request
all candidate programs. Existing v1–v7 widths and omitted checkpoint defaults stay
frozen; fresh default remains v6 pending explicit adoption. Complete naming has
no truncation or random fallback. The action axis is the larger of configured capacity and vocabulary size; scoring
trims unused rows so ordinary decisions do not run the larger action network.

Current naming universe is exactly the existing non-token real catalog,
including back faces. Names outside supported definitions remain an unresolved
rules coverage question: this work does not claim all legal Magic names or
expand deck admission. `NameVocabulary` can hold validated names without executable definitions; its
digest enters state identity, and v8 checkpoints bind its ordered names and
optional definition joins. Only the test fixture extends it today.

## Reconciliation — 2026-11-04

The completed simplification removed owned modal branches, duplicate sorting and
program encoding, and separate action traversals for candidate joins. Name and
object definitions share `EmbeddedObservation`; retaining the original prepared
observation preserves its inference cache. Scalar/vector parity replays one
recorded trace because cloned observations have already drained per-step events.
The obsolete deletion checklist is retained at `5ac29cc12:scratch/mtg-243-naming.md`.

Main remains `8e8d702f8` (remote checked during reconciliation). Its supplied-axis
binding and explicit `first-legal` overflow policy are incorporated. Default
overflow still rejects. V8 reserves the entire installed naming vocabulary even
with a smaller configured `max_actions`; a legacy input with explicit cutoff can
truncate names and is not complete naming support. Documentation now states
that exception. Native validation and encoding use the same retained prefix.

The merged capacity regression test incorrectly required a schema-7 retained
frame to equal schema 8 byte-for-byte. The receipt remains untouched; the test
now requires the declared schema transition, changed fingerprint, exact remaining
decision fields/offer identities, and exact current scalar/vector frame equality.
Nondefault-buffer parity now includes v8 with both history layouts.

MTG-241's Task comment at 2026-11-04 20:45 UTC reports a new `FrameCompletion`
owner and top-ordering obligation 27, with no policy version reservation yet.
That work is not on main or incorporated here. Naming resumes the existing
suspended frame and retains modal obligation 10, so no obligation-code collision
is evident. Compatibility with the new completion owner remains unverified
until ordinary Task sync; no parallel continuation implementation is needed.

## Implemented behavior and evidence limits

Implemented shared candidates/decision; v8 native scalar/vector/Python parity;
complete name identity and learning-definition scoring; typed immutable
`NameCandidate` exports; catalog-derived capacity;
LLM/Academy stage presentation and reconnect; explicit checkpoint vocabulary
binding. The controlled fixture combines the installed universe with 600
Scryfall-identified names (928 distinct choices), without adding definitions.
Native proofs cover rows 511/512, back faces, legality, exact forks, replay,
stale rejection and fail-to-find. The policy proof gives every candidate positive
support, distinguishes identities and improves the selected log probability in
one bounded supervised gradient step. This is not a strength experiment.

Semantic Observation is schema 8, capture 5, state hash 16: public input joins
and a shared vocabulary digest replace owned effect branches. Current w17
conformance, authority, both parity receipts and Academy fixtures were refreshed;
the authority's 135 Commands are byte-for-byte unchanged. Frozen evidence was
untouched; the RUL-9 dependency was read from its artifact cache. Old semantic
world bindings fail explicitly; historical tensor shapes/omissions remain fixed.

Prior focused model/client, native debug and rebuild results remain recorded at
`5ac29cc12:scratch/mtg-243-naming.md` and `1e7dc47d4:scratch/mtg-243-naming.md`.
The large-vocabulary native test injects a shared vocabulary into a real naming
decision; Python scoring uses its corresponding observation projection. Vector
stepping and Academy reconnect exercise the installed catalog. These are
complementary checks, not an end-to-end 928-name Academy or vector-game proof.

Pre-reduction debug allocation attribution (not a speed threshold): catalog setup
5,232 allocations / 1,863,512 bytes; naming decision 0 / 0; 928-action vector
1 / 37,120; observation projection 1,880 / 154,583; encode into existing buffers
5 / 792; naming fork 15 / 47,012; ordinary fork 15 / 10,404; ordinary fork+step
393 / 41,980. One current-catalog CPU semantic forward (344 candidates, hidden
width 8) took 19.5 ms before sharing its program table. Setup, projection, encoding, scoring and ordinary work are
reported separately; no cross-runner performance claim. The focused
`catalog_naming_cost_attribution` test with `benchmark` and one test thread
provides bounded attribution. These pre-reduction numbers do not establish final
performance; final attribution and the same-runner CI comparison remain pending.

## Remaining work and review

Gate owns the complete debug Rust suite, Clippy/format checks, affected Python
model/checkpoint and active-training suites, current-world arena admission and
monitoring, Academy world/fixture checks and headless browser coverage. The
full checkpoint path still needs explicit v8 vocabulary mismatch coverage;
the current naming unit test checks binding presence and model-state restoration.
Final parity source fingerprints and Engine Budget remain unverified. The changed
hash contract needs an explained CI digest adoption while unchanged Commands stay
exact. MTG-241 continuation compatibility becomes relevant when its code is synced.

No product decision blocks the current catalog scope. Production admission of
validated names without definitions and default adoption of v8 remain unaccepted
future choices; neither is required to deliver this explicit-version implementation.

Check: `uv run --extra dev pytest tests/training/test_action_cutoff.py tests/verify/test_native_buffer_capacities.py tests/model/test_catalog_naming.py -q` — 71 passed; Ruff and `git diff --check` passed. Full gate/CI validation remains pending.

Publication remains pending at the Graph's delivery step, followed by interactive
review. No merge, auto-merge or training experiment was performed.

## Concept review — 2026-11-04

Outcome: nothing to change in the design and no open product choice within the
accepted Task scope. This is a headless review against Jack Heart's supplied
directive, not new approval or a delivery decision. No next implementation
direction is required. The review page is `scratch/concept-review.html`.

The experience is one deliberate name selection followed by the existing private
search. For The Rise of Sozin, choose `Name Gray Ogre`, select matching cards as
offered, then use `Stop searching and shuffle` if stopping early. Academy's
existing `Find an action` filter narrows the displayed offers and can be cleared;
it neither submits a name nor changes the legal support. The LLM and policy
choose from that same native support. A name denotes public vocabulary, not an
opponent card, a deck entry, or evidence that the opponent owns that card.

The consequential former mismatch was a simple name choice represented as a
separate owned effect program for every candidate, with fixed policy capacity.
The implemented shared vocabulary and explicit v8 resolve that mismatch without
adding a search/confirmation protocol to the game. Retain `ChooseMode` as the
compatible Command representation; the native `NameCard` decision and visible
`Name a card` stage already give it the specific meaning people need. Replacing
the wire verb would create migration work without improving this interaction.

Across layers, `ContentPack::name_candidates` owns the immutable sorted names;
`Decision::NameCard` retains their shared vocabulary, chooser and count. A bound
Command selects a candidate and resumes the suspended frame with one
`ExileNamed`. `SearchNamed` continues to own remaining selections and inspection
permission. V8 carries categorical name identity plus an optional learning
definition join; neither consumes a physical card slot. State identity records
the vocabulary digest, and checkpoint admission binds its ordering and joins.
The existing server session and Command recovery own reconnect; no browser name
draft or second recovery lifecycle is needed. Public catalog knowledge must
remain distinct from private zone membership under closed disclosure.

Failure and recovery preserve the committed boundary: disconnect before naming
returns the pending naming offers; disconnect during search returns its remaining
choices and authorized inspection. Retrying an accepted Command is idempotent;
a newly identified stale submission is rejected without changing state. Stopping
search ends inspection but retains already exiled cards. It is not undo or a
return to naming. A mismatched vocabulary checkpoint must fail admission; it
cannot silently reinterpret row identities. Explicit v8 adoption supplies complete
policy support; keeping fresh v6 defaults is an accepted compatibility boundary,
not an assertion that v6 supplies complete naming beyond its capacity.

Counterexamples that would invalidate this conclusion: a candidate beyond row
511 has no legal policy score; filtering/reconnect loses its original Command
binding; public candidate definitions reveal hidden card membership; stop/retry
reverses an exile or repeats it; or a changed vocabulary loads an old name-vector
mapping. The existing directed tests cover parts of these contracts, not a
single end-to-end large-catalog proof. Keep the already-listed gate obligations,
especially full checkpoint vocabulary-mismatch admission, headless Academy
coverage, final allocation attribution and same-runner Engine Budget. This review
read code and tests; it did not rerun them or establish final performance.

Existing usage guidance in `docs/training.md#catalog-naming-input` and the package
READMEs already describes the selected experience and compatibility boundaries;
no usage or skill rewrite is needed. No proposed behavior is being promoted to
implemented behavior. Future questions remain outside this Task: which validated
names without definitions should production admit, and when should v8 become
the fresh default? Neither choice is needed for the authorized explicit-version,
real-catalog outcome. No publication, merge or training authorization is added.
