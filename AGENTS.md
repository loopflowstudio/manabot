Never add "Co-Authored by:" tags in commits

## Naming

- **Manabot** — the whole project and public brand at `manabot.gg`.
  The repository is `loopflowstudio/manabot` (renamed at Jack Heart's request
  on 2026-10-15).
- **Academy** — the browser UI for inspecting bots and replays: the `managym.academy`
  package, its Svelte frontend in `academy/ui`, and their service namespaces. It is distributed with managym and uses optional training bot capabilities. The project website uses Manabot.
- **manabot** — the bot, and the Python training library/CLI. Indefinite
  noun: you train *a* manabot.
- **managym** — the rules environment the agent lives in.
- **Etude** — the separate launching and experiment system. Its public client
  is `loopflowstudio/etude`; the private research service is
  `loopflowstudio/etude-studio` (repository split verified 2026-10-15).
  The import name `etude` and the service integration variables
  `ETUDE_URL` / `ETUDE_TOKEN_FILE` belong to that system, never to the app.
  Always ASCII (never "Étude").

Jack Heart selected Manabot and `manabot.gg` on 2026-10-14, replacing the
provisional Fantasia project name.

Boundary rule: browser UI is Academy; training and evaluation is manabot; the
rules engine is managym; experiment launching and recording is Etude. So
the play server, experience protocol, curated packs, and their env/error
namespaces (`ACADEMY_PLAY_*`) take Academy names, while training
infrastructure (wandb project, presets, ops jobs, `MANABOT_*` config vars)
keeps manabot names. Frozen evidence — experiment IDs, contracts,
receipts, wandb history — is never renamed, and neither are the compatibility
surfaces that predate the Academy name: the `x-etude-*` wire headers,
`etude.*` browser storage keys and schema identifiers. Hosted services are
`academy-play` and `fantasia-website`, with deployment secrets in Doppler
`manabot/prd` (Jack Heart, 2026-10-14). `etude.gg` and its legacy aliases
redirect to `manabot.gg`, preserving paths and queries; hosted play is
available at both `play.manabot.gg` and `academy.manabot.gg` (equivalent
addresses selected by Jack Heart, 2026-10-14). Namecheap is the registrar and
Cloudflare manages DNS. "Sepia Etude" remains the name of the visual language.

Name a thing for what it is where it is produced (`catalog_tokens`,
`record_vocabulary` in `managym`; consumer-specific names in `manabot`), keep
the wrapped interface's noun on wrappers (`RecordedVectorEnv`), and qualify
generic roles with their domain unit (`decision_weights`, `SnapshotScorer`). Do
not reuse established terms (`recipe` for training recipes, `manifest` for
content packs, `replay` for seed-and-action replays) for unrelated concepts,
even internally. Rename published names via a deprecated alias stating the
removal release; serialized keys, checkpoint fields and frozen evidence keep
their spelling.

Code dependencies flow `managym.academy` → `managym` and `manabot` → `managym`.
Academy never imports manabot or Torch; optional bot capabilities are discovered
through the `managym.bots` provider entry points; optional historical replay/advice
capabilities use `managym.academy.bots`.
PyPI distributes the engine as `managym` and training as `manabot`, released
together at the same version. Release builds use `scripts/release/prepare.py`
to stage self-contained source archives; direct engine sdists from the workspace
do not preserve shared asset paths. `manabot` requires that exact `managym` version;
`manabot.managym` is a compatibility alias to the same engine. manabot never imports
Academy, eagerly or lazily (`tests/test_package_boundaries.py` enforces it).
The engine wheel retains `managym/src/**/*.rs` and `managym/Cargo.lock` for
source-byte identities. Keep wheel exclusions narrow: `**/cards/**` and
`**/state/**` also match engine modules under `src/`.

## Academy's shared decision surface

Jack Heart's direction (2026-10-21): Academy should make the acting agent's
viewer-safe observations and complete legal action space understandable to a
human. Navigation, grouping and labels may clarify that shared contract; they
must not hide strategic choices, silently select actions, reveal unavailable
information or introduce human-only game semantics. Evaluate awkward decisions
at the shared observation/action authority rather than disguising them in the
browser. Preserve Command binding, stale rejection, actual pending decisions on
reconnect, replay/fork identity and accessibility. Cancellation or undo needs an
explicit shared action and cannot erase committed costs or revealed information.
Configured priority-stop automation must remain identifiable as automation,
separate from deliberate human decisions.
Play's retained `hero_*`/`villain_*` setup fields name human/opponent roles;
`GameConfig.human_seat` orders them at the native boundary. Record and project
actual native seats; Watch keeps its two-seat acting-view contract.
Academy resume credentials and the started-launch marker live in browser
`sessionStorage`. Keep them scoped together; a launch must not resume another
tab's or another launch's game.
Game-file imports send original JSON source spans: JavaScript numeric parsing and
re-encoding can round u64 identities and invalidate retained digests.
Watch history is the selected acting viewer's recorded track strictly before
that decision's `presentation_cursor`; spectator reveal never supplies history
labels. Event emphasis stays separate from choice focus and lane navigation.
This direction does not approve a particular redesign or change frozen worlds,
checkpoints or experiment evidence.

## Style

- Etude evaluation source identity reads `git ls-files`. Mark a new source file
  with `git add -N <path>` before running an evaluation fixture that fingerprints
  it; untracked files fail admission with `identity_path_without_files`.
- Package style guides live in each package README (`manabot/README.md`,
  `managym/README.md`). Avoid transient comments that denote changes; pay
  attention to file headers and README content; propose small, iterative
  changes.
- Loopflow discovers repository graphs in `.lf/graphs/`, not legacy `.lf/flows/`.
  Compose builtin graphs; use narrowly named forks for domain review placement.
  Validate changes with `uv run python scripts/check_loopflow_workflows.py`.
  See [work boundaries](docs/workflows.md).
  `lf pr publish` stages tracked edits, including edits left after a path-scoped
  commit. Temporarily preserve unrelated edits outside the checkout and restore
  them after publication.
- In a Loopflow Graph step, commit your own edits, scratch notes included,
  before finishing. A clean tree skips the separate commit-message session
  Loopflow otherwise starts for what a step leaves behind.

### Experiments

- Base new training and learnability trials on the current TLA frontier setup
  (`experiments/tla_frontier.py`) rather than falling back to Elemental; TLA
  does not have larger or longer games.
- Etude report curves average runs within a job. For a per-seed readout,
  select that seed's fixed checkpoint seatings in its own panel and declare
  `Seating(seeds="one")`; otherwise other seeds can make the curve a gap.
- Exploratory runs (calibration, throughput, capacity, pilots) need no
  prediction, protocol or review; the report ends them. Findings (cited claims
  about a method or playing strength) open their report with one committed
  paragraph before scoring: question, numeric prediction, refuting result and
  cost cap. When unsure, treat a run as exploratory.
- The freeze is the commit and Etude run ID. Changes keeping the question
  (capacity, crash fix, device, retry) are dated lines in the report, never new
  protocol versions. Record the launcher's cost once; never reconstruct bills,
  reserves or settlements.
- Experiments costing at most USD25 launch under `--max-usd`; higher caps need
  approval first. Rental admission includes the on-demand startup allowance
  beyond `Job.max_hours`; size job and experiment caps for the whole window (a
  price offer does not guarantee provider capacity).
- Each experiment is `experiments/<name>.py` at a commit with
  `experiments/reports/<name>.md`—no per-experiment protocol doc, serialized
  regime, digest check or schedule test suite. Put reusable machinery in
  `manabot/` or Etude. Keep paired deals, replay verification, checkpoint
  compatibility and training seeds as the unit of evidence.

### Python structure and documentation

- Design around the domain's data structures and public APIs. Keep the core
  types and entry points easy to find; separate orchestration, learning rules,
  and persistence when they have different responsibilities.
- Give substantial modules a short opening explanation of their purpose,
  principal types and entry points, and non-obvious ownership or data flow.
  Use section comments to group related code when that helps navigation.
- Explain intent, invariants, units, tensor shapes, and algorithmic choices
  beside the code that depends on them. Training code should explain what an
  estimator measures, where gradients flow, and which samples a loss uses.
  Cite a method when claiming to implement it; document meaningful departures.
- Keep docstrings concise, but include non-obvious contracts, side effects,
  and failure behavior. Do not repeat signatures in `Args:` / `Returns:`
  boilerplate or narrate obvious statements. Update comments with behavior.
- Put imports at the top; use explicit imports. Do not prefix a function with
  an underscore merely because it is used only within its module.
  Keep one implementation and one owner for each fact; use git for
  old versions rather than `_new`, `_old`, or `_backup` variants.
- Import names from their defining modules (`manabot.env.observation`, not
  `manabot.env`); package `__init__` files re-export nothing, so evaluation
  game and replay children stay Torch-free
  (`tests/test_package_boundaries.py` enforces the child modules). Launch
  evaluation children with `manabot.sim.child.ChildProcess`, not
  `multiprocessing` spawn, which re-imports the parent's `__main__`.
- Library code receives a framework backend (the module for an update, a
  `BackendPolicy` for simulation and evaluation) and never imports
  `manabot.backends.*`. Only composition roots choose one: `experiments/`,
  `scripts/`, recipe entry points, the CLI and child-process entry points, `manabot/sim/table_bots.py`, `manabot/sim/policy_provider.py` and the
  backends' tests, by import or `manabot.backends.resolve("torch")`.
  `tests/test_package_boundaries.py` enforces roots and named entry-point scopes
  in mixed CLI/library modules ([who chooses a backend](manabot/backends/README.md#who-chooses-a-backend)).
- Standalone local checkpoint admission and scalar games can run with
  `manabot.training` unavailable. Load the backend learner on first training
  use so opening a policy in these children needs only inference.
- Generic backend learning helpers (`rl`, `learner`, `arrays`) keep role-specific
  types under `TYPE_CHECKING`; importing their annotations at runtime must not
  pull in checkpoint, player, observation or training-configuration modules.
- `manabot.policy.prepare.Prepared` is the only policy input in every backend.
  Integer lookups, masks and admission run once on the host in `prepare`; do
  not add a forward that reads an observation dictionary or a second
  preparation for one framework.

Execution row limits bound prefix materialization before preparation, independently
of stream counts and logical optimizer minibatches. Accumulate actor and critic
terms with their own logical denominators and clip/step once. Collection keeps
one sampling call per original row group. A prefetch generator must release its
previous Prepared input and completed Future before submitting another chunk;
the completed Future also owns the host result.

Decision-memory capture is opt-in from positive record capacity or include-own;
search clones never append hypothetical records, while rollback snapshots preserve
capture. Raw prefix chunks can be interned across streams with identical early
content: a growing serving segment belongs to its native stream, never that shared
chunk. Historical diagnostic inputs without the capture setting mean enabled.

Profile observation retention separately from policy preparation: `_retain_obs_rows`
runs before the preparation track and includes MemoryStore capture. Capture is
nested under retention; do not add both as disjoint collection costs.

Jagged CUDA attention's static sequence bound must not exceed the physical
packed token total, even when a short execution chunk retains more padded
slots. Bound it from host shape metadata; do not read a CUDA scalar for it.
### Functions and objects (Python)

Use a class where it offers something genuinely valuable, and avoid one where
it does not. An object is a thing; a function is what happens. A class earns
its place as a value, a resource or an interchangeable part, where the type
guarantees something every user relies on. Functions do the work: anything
with steps, a loop or an order of operations. A class whose methods sequence the
work (an executor, trainer, manager, runner or collector) is the shape to
avoid; write the loop as a function whose body shows each step
(`manabot/training/loops/self_play.py`:
`row = step(settings, learner, ema, collector, counts, rng, out=run.out)`).
That step updates the learner, EMA and counters in place. Etude restores the
registered counts before calling the self-play restore callback. Register the
live exporter with `run.save(export=...)` once the collector has opened.

Where classes are valuable here, with the examples to copy:

- **Configurations and records** (`ObservationConfig`, `ModelConfig`,
  `MatchConfig`, `Learning`, `GameErrorPolicy`, `llm/schema.py`). A config
  enforces real domain and structural invariants when built (never arbitrary
  upper caps or tested-grid `Literal` sets on resource counts), compares and
  hashes by content, serializes with stable keys, and is copied rather than
  edited (`model_copy(update=...)`, `dataclasses.replace`). It may own
  validators and pure derivations, never I/O, runtime construction, or
  execution.
- **`torch.nn.Module` subclasses** (`ActorCritic` and its parts). Keep
  construction and `forward` there; training logic is functions over the
  module.
- **A resource with a lifetime** (`VLLMServer`,
  `PolicyInference`, `ChildProcess`, native `managym` environments, Etude's
  `Run`): something opened, used and closed. Make it a context manager, or a
  `@contextmanager` function when there is no state to hold. Do not wrap
  another library's handle in our own class to add methods.
- **A guarded invariant** (`BudgetLedger`: spend never exceeds the cap across
  restarts; `AdmittedRollout`: rows admitted once; `Profiler`: nested
  timers). The methods exist so no caller can break the invariant; name it
  in the docstring and keep orchestration out.
- **Interchangeable parts behind one small interface** (the LLM runtimes,
  the scenario trackers in `verify/scenarios.py`, the loop's `Tracker`
  protocol). Declare the interface as a `Protocol`; pick an implementation
  with a table or an argument, not an `isinstance` chain.
- **Exceptions** callers catch by type or that carry a typed payload.

These are examples, not a closed list. For a class of another kind, ask what
it gives a caller that a dataclass and a few functions would not; if the
answer is real, write the class.

Everything else is functions:

- Pass what a function needs and return what it produces: the next state,
  not a mutated argument. A function that must change its argument in place
  says so in its docstring; Torch modules and optimizers are the usual case.
- No hidden state: no module globals that change, no environment reads below
  the entry point, no callbacks that reach back into a caller's object.
- No operator overloading to make data act like something else, and no
  `try/except/finish` repeated in callers where a `with` would scope it.

Apply this to new and changed code. Convert an existing class when the
change already touches it; do not rewrite a module only for style. Rust in
`managym/` follows its own README.

### Python typing

- Write fully typed Python: annotate parameters and return values on all
  functions and methods, including private helpers and tests; use `-> None`
  where appropriate. Type model fields and collections with their element
  types. Local variables can rely on clear inference.
- Represent structured domain data with named types: dataclasses, existing
  Pydantic models, `TypedDict`, or small protocols as appropriate. Avoid
  passing unstructured dictionaries or positional tuples across API boundaries.
- Prefer precise unions, literals, and callable signatures to `Any`. At
  untyped library, JSON, or native-extension boundaries, validate or narrow
  values once and expose a typed interface. Keep necessary `Any`, casts, and
  type-checker suppressions local, with a reason when it is not evident.
- Use Python 3.12 forms (`list[T]`, `T | None`). A nullable annotation means
  `None` is a real possible value; a default determines whether an argument
  can be omitted. Return `None` for expected absence only when the API defines
  it; do not disguise failures as empty results.
- Apply these conventions to new and changed code incrementally. Do not claim
  repository-wide type safety without a passing configured type check, or
  weaken types merely to silence one.
- A checkpoint records the revision and Etude run that wrote it
  (`checkpoint_source`), its column names, definition keys and naming names.
  `manabot.model.interface.model_interface` checks the recorded input contract.
  The shared policy reader then binds the learned arrays by key. Academy uses
  that reader when listing checkpoints. When the checkpoint records exact decks,
  compare its non-schema binding fields exactly and admit its saved column prefixes.
  Otherwise the card and naming digests record what the writer used. Append
  input columns to named buffers. Shared host preparation projects each buffer
  to the saved prefix. Preserve column order and meaning. Renaming a Rust type,
  field or record name the token serializer hashes needs an entry in
  `managym/src/cards/token_renames.rs`; a name is never reused with a new
  meaning. Never learn a `CardDefId`: tables indexed by card or name bind by
  definition key or name. A change that alters what a stored pilot computes
  fails `tests/model/test_pinned_pilot.py` until it adds the rename or column
  or declares the break with `scripts/pinned_pilot.py record --reason`. Do not
  bump the input version for a layout change.
- The engine has no world label. New checkpoints, recordings, decision captures
  (v6) and evaluation records write none; a reader ignores one in an older file
  (`manabot.model.world.current_binding_form`, optional `world` fields omitted
  when absent) and never rewrites stored data. `managym.WORLD_VERSION` is a
  warning alias until 0.4.0. Matchup ids keep `w17` as the plain string
  `manabot.sim.matchup.SUITE_PREFIX`, and frozen experiment files write the
  literal, so retained tournaments find their records. Add no new label.
- Main carries policy input v11, the fresh default
  (`ObservationConfig.policy_input_version`, always serialized): one row per
  game thing. Card rows hold cards outside the battlefield and stack;
  permanents own one effective `CharacteristicsData` projection plus object
  identity and token status; spells own a `SpellData` projection, and abilities
  carry only source references. Old Python `.card` and flat characteristic
  getters derive from those records (removed in 0.4.0); the `stack`
  buffer holds every spell and ability. A battlefield object's or spell's card
  id indexes its permanent or stack row, so focus, relations and history name
  one row. V11 carries v10's `player_turn_state` and `card_turn_state`; a
  turn-state-free encoder refuses an observation that needs them
  (`managym.input_v10_state`) before writing buffers. Inputs v9 and v10 are a
  deprecated fork for earlier checkpoints (`agent/legacy_input.rs`,
  `manabot/env/legacy_input.py`, `InputLayout::Legacy`; removed once a v11 pilot is at least as strong as
  the best v9 one, not before 0.4.0); never extend it. A checkpoint,
  replay or saved configuration bound to any other version, or omitting it,
  raises `UnsupportedPolicyInput`, which names the revision to run it from
  (`ac601834d` is the last revision supporting v1-v8). Add no other version
  branch.
- Open-ended vocabularies (counter kinds, added subtypes, set names, used
  abilities, continuous-effect records, copy effects and their exceptions) are
  fact rows, never columns. Active effects are shared records: `effects`
  headers own bounded offset/count slices into `effect_details` and
  `effect_members`; members keep one-based public object joins. Detail and fact
  rows carry learning-token kinds and exact signed i32 values in two 16-bit
  halves; raw storage identities never enter these tokens. A stacked activation
  owns its chosen effects; catalog joins are observation provenance, never
  resolution lookup authority.
- Fact, relation and capacity overflow raise typed capacity errors. Only
  `action_overflow=first-legal` may truncate policy input. The TLA pilot
  uses it with 64 policy slots and aggregate replay-linked diagnostics
  (Jack Heart, 2026-11-04). Native legality and Academy retain all choices,
  and an omitted overflow policy means error.
- The model reads each numeric input column by a reading. A fresh model
  derives it from the column's native kind (`ValueLayout` in
  `agent/observation_layout.rs`): unscaled counts get the log squash, a
  category of declared `size` n reads `one_hot:n` and a selector `one_hot:15`
  (the last flag is always "above the size", never a real value), a count or
  number stored in tenths reads `steps:8`. Scaled player mana pools keep one
  scalar (`as_is`), and unscaled mana reads `log`. Every `rank` is rank / 128
  independent of fact and effect capacity, references are gathered and ids
  looked up. No category or index reaches a projection as a magnitude. A new
  column declares its kind (`scale` when the encoder divides it, `size` for a
  category); add no Python column table and no per-backend scaling. The one
  table is `PLACED_READINGS` (`manabot/policy/readings.py`): validity flags,
  embedded words, fact durations and join extents read by their place in a
  structured row. A sized reading is recorded as `name:size`, so a stored
  model keeps its width when a category grows. A checkpoint records its readings
  (`input_readings`) and never consults the rule; one without the record gets
  `legacy_readings`, which is never extended. The native kinds of buffers with
  their own preparation (decision memory, history events) are not used. A
  run's saved state records its model settings and readings. Resume checks both before
  restoring weights. Older runs supply their model settings from the initial raw policy.
- New observation state reaches the policy input, LLM text and learning
  tokens in the same change. `ObservationSpace` pairs the config with its
  native `ObservationLayout`. Pass `space.layout` to scalar, batch and vector
  encoding and derive buffers from it. Validate the full buffer contract
  before encoding, including exact keys and companion shapes. The layout
  supplies the input version, optional blocks and capacities. The sole encoder
  lives in `managym/src/agent/`. Read columns and row kinds by native name
  (`column("facts", "value")` in `manabot.env.observation`). Give new columns
  or rows a directed native test on a constructed position.
  `ObservationSpace.encode`, `space.encoder` and `ObservationEncoder` are
  warning aliases removed in 0.4.0; frozen experiment files keep them.
  `managym/cards/tests/cards/observability.rs` names each `Observation` field's
  tensor columns or fact kind and fails to compile when a field is unlisted.
- Decklist disclosure (open by default, or closed) lives on native
  `GameSetup.decklists` and the resolved `GameState.disclosure`, written by
  `MatchConfig.decklists` and Academy `GameConfig.decklists`. Observations, search and possible worlds read it
  through `Game::decklist_known`/`require_open_decklist`; never filter decklist
  information in an encoder or the browser. The policy input encodes the lists in
  the `decklists` block (2 seats × max cards × 5); a closed opponent's rows are zero.
  Academy's server keeps a closed opponent's deck name, the matchup's asset pack
  and unseen definition names out of its frames and viewer trace exports.
- Naming is ordinary native decisions: `Decision::NameCard` holds a
  `NameScope`; scopes over `NAME_BRANCHING` (64) names offer balanced groups of
  sorted names, and only a name choice commits. `name_candidates[A,4]` at
  ordinary action capacity holds the name row, an optional definition
  reference, then a group's first and last name rows, all one-based. Names are
  not game objects; do not use card slots or invent definitions for them.
- Fact and shared-effect ids read the card encoder's token rows
  (`ModelConfig.record_vocabulary="catalog"`, the fresh default, omitted from
  regimes); native `agent/record_vocabulary.rs` lists every name the encoders
  hash, so a new hashed name must appear there or the policy rejects it.
  Schema-specific labels use an owner prefix (for example `OpponentMemory::Role`):
  the token namespace is shared, and an unprefixed label can collide with an
  existing rules field that has a different token meaning. Optional model
  capabilities select their catalog rows explicitly so disabled capabilities
  do not grow the default model's lookup buffers or parameters.

## Commands

- **Use uv everywhere for Python installation and execution guidance.** Never
  suggest bare `pip install`, including in docs, quickstarts, CLI errors or
  troubleshooting. Use `uv venv`, `uv pip install` and `uv run`; installed
  Academy examples use `uv run --no-project managym academy`. Preserve exact
  historical quotes and frozen evidence.
- **All Python commands run through uv.** This repo's environment is
  uv-managed; bare `python`, `uvicorn`, `pytest`, `maturin` etc. will miss the
  venv or pick the wrong interpreter. Always `uv run <cmd>` (or
  `.venv/bin/<cmd>` when a script must reference the interpreter directly).
  Never emit an un-uv'ed Python command in docs, scripts, or instructions.
- Development uses ordinary CPython 3.13; consumers also support 3.12. Fresh
  development venvs: `uv venv --python 3.13`. Keep source syntax compatible with
  3.12 and validate both interpreters in isolated consumer environments. Do not
  claim free-threaded or newer interpreter support. Preserve frozen experiment
  interpreter pins and never replace another active checkout's environment.
  Public Torch requirements split identical Linux/non-Linux minimums: uv 0.9.22
  otherwise applies group-specific CPU/CUDA indexes across platform branches.
  Keep those markers aligned with the runtime groups; validate a locked sync,
  since lock resolution alone can select an unusable platform wheel.
- After changing Rust under `managym/` (the engine, `state/` or `cards/`), rebuild from the repository root so
  uv uses the pinned root environment instead of creating a nested venv:
  `uv run --extra play maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`,
  then place the matching interpreter/platform `.so` from the wheel in
  `managym/`, preserving its filename (for example,
  `_managym.cpython-313-darwin.so` for the macOS development interpreter).
  Never copy a cp312 extension into a cp313 environment or vice versa.
  Replace it atomically (write a new
  file, then rename); overwriting a mapped extension in place can trigger macOS
  code-signature termination on subsequent imports.
- Format all Rust workspace members with `cargo fmt --manifest-path managym/Cargo.toml --all`;
  formatting only the root package does not format the `state` and `cards` crates.
- Inspect a bot in the browser: `./scripts/play` (installs locked local dependencies,
  starts backend + frontend, and stops both on Ctrl-C).

## Decision ownership

`TurnState.active_player` / `Game::active_player()` identifies whose turn it is.
Native `Decision.player()` and `Action.player()` identify whose rules decision,
actions and resources apply. Published `ActionSpace.player` identifies the
chooser: `publish_action_space` resolves control once from the legal actions.
Env dispatch, observations and Command/offer actors read that chooser directly;
never remap it again. Player control grants in-game inspection through native
`shares_information`, never private sideboard access or outside-game selection.

Native `semantic_decision_frame_json` and `execute_semantic_command_json`
use action-aligned `structured_search_offers`: these offers have no choices
and reject nonempty `Command.answers`. Compound submissions belong to the
separate `compound_offers` / `step_structured` API. Do not infer supported
atomic answers from the `Command` field alone, or pass compound offer ids to
the semantic executor. Recording preserves the original native Command and
its object preconditions; Academy's presentation Command has a separate
revision scope.

Native heuristic selection consumes the live viewer `Observation`, including
`action_space.focus` (source identity for target/cost decisions). Canonical viewer
JSON omits that focus; a `toJSON`/`from_json` round trip is not an equivalent
selector input. Parity fixtures replay decisions to retain the live projection.
`Observation::for_player` omits presentation events; built-in stepping retains
the last live projection's events so combat history survives scalar calls and
exact forks. Player policies belong to Env/VectorEnv chooser seats, separate
from rules/deck PlayerConfig and world identity.

Optional statistics and private recording share native setup, movement and
committed-choice facts. Search clones omit those sinks and their copy provenance;
transactional rollback snapshots must restore them. Training supplies recorder
paths and phase markers; engine recording never discovers training scope.
Install retained vector roots with `reset_each_into_buffers(setups, seeds=seeds)`.
An ordinary reset followed by a seed reset creates a separate interrupted game
in recording and statistics. Enable native statistics before the first reset.

`protocol/*.schema.json` has no generator. Patch the files when a protocol
type changes; the Rust schema tests compare them byte for byte with what
`schemars` derives. A resolution payment's focus (ward, "unless pays") is the
permanent's `object_id`, never its permanent id.

## Forced decisions

Every game runner, benchmark and Academy session uses native trivial-decision
skipping (`skip_trivial`, on by default); a decision with one legal action is
never shown to a person or sent to a model or policy (Jack Heart, 2026-10-28).
Turning it off is for explicit debugging only. Outside tests, the only code
that does is the constructed-position fixtures listed in
`tests/env/test_skip_forced_decisions.py`; they play no game for a person,
model or policy.

## Testing

**NEVER intentionally induce load for testing**, including debugging,
benchmarks or demos, on local, CI or remote hosts. No CPU burners (`yes`),
busy loops, stress tools or artificial resource pressure. Use controlled
fixtures and injected measurements; see [TESTING.md](TESTING.md).

**No test runs for five minutes.** Keep learning tests bounded under CI's
parallel workers. Use bucketed shapes for small boards when they preserve the
comparison, and restore any Torch thread limit changed by the test.
Run targeted tests of the change and its existing consumers. For Python edits
under `managym/` with a wheel already installed, run `uv run python -m pytest`
from the repository root so the tests import the checkout’s Python modules.
When policy input width or scale changes, include model tests that call
projections directly. Dense reference tests should apply the model's saved
readings before projection and keep the control that proves the behavior.

Read [validation by change area](docs/development.md#validation-by-change-area) for the
checks required by the affected package and workflow.

Use distinct Python test filenames across test directories: the configured
pytest import mode shares their module namespace, so duplicate basenames fail
collection even when the tests pass individually.

Native `PlayerConfig::new` creates a fixed/test setup and bypasses submitted-deck
admission. Admission-rejection fixtures must mark the config as submitted
(`submit_to_world`); otherwise an invalid deck can reach game construction.

- Native isolated failure receipts name `AgentError` / `NativePanic`; PyO3
  class names are `PyAgentError` / `PanicException`. Catch the former through
  its public export `managym.AgentError` (there is no `managym.PyAgentError`).
  Replay compares these through
  `manabot.env.replay.matches_retained_error`, retaining original class names and
  full vector diagnostic payloads in evidence.
- An observation that does not fit its buffers raises
  `managym.ObservationCapacity` (a `ValueError`, message
  `observation capacity exceeded: …`) from `encode_observation_into` and
  `encode_observations_into`. Collectors that count capacity failures catch
  that type, never a message prefix, and propagate other failures. Vector
  steps keep raising `AgentError` for an unrecovered one (retained evidence
  names `PyAgentError`) and record `ObservationCapacity` when they recover.
- Frozen experiment scheduling/analysis tests use an explicit compatible-world
  fixture for synthetic outcomes, with separate real-runtime rejection tests.
  Pin declared input capacities rather than inheriting fresh defaults; never
  rebind frozen protocols or run games/training under the fixture identity.
- Experiment results under `experiments/data/` live in the artifact store. Before
  committing new results or merging a branch that tracked them, publish their
  bytes with `uv run --extra artifacts python -m manabot.data publish <path>`,
  commit `experiments/data.manifest.json`, and remove git tracking while retaining
  the local files. Verify with
  `uv run --extra dev pytest tests/infra/test_experiment_data_untracked.py -q`.
- After a world or observation-shape change, validate current-world arena
  admission and monitoring with `uv run --extra dev --extra artifacts --extra reports pytest tests/semantic/test_elemental_world.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py tests/model/test_choice_separability.py -q`.
- Artifact storage and Etude integration tests require the `artifacts` extra.
  Validate dependency changes in an isolated environment so locally installed
  extras cannot hide missing CI dependencies:
  `uv run --isolated --extra dev --extra artifacts pytest tests/training/test_artifact_storage.py tests/training/test_etude_sdk.py tests/training/test_etude_source.py tests/training/test_recipe_etude.py -q`.
- Engine speed is budgeted in CI: `Engine Budget` runs `bench_search` and
  `bench_steps` in release for the change and for the reference commit of
  `managym/perf/baseline.json` on one runner, and fails a PR more than 10%
  slower than the reference or with a changed search/decision digest. A moved
  digest (a rules change reaching the curated decks, or a change to the
  benchmarks or curated lists), an accepted cost or a kept speedup moves the
  reference by adopting the PR's own run (`gh run download`, then
  `scripts/engine_budget.py record --reason`); a reference that no longer
  builds with the change's benchmarks is recorded the same way
  ([engine performance budget](docs/development.md#engine-performance-budget));
  wall time is never compared across runners.
- **CI runs the Rust suite in debug, so validate in debug before landing.**
  The dev profile is opt-level 1 with debug assertions on; CI runs it with
  `cargo nextest run --profile ci`, which terminates any test after 5 minutes.
  Limit nextest builds with `CARGO_BUILD_JOBS=2`, and tests with
  `--test-threads 2`; nextest `-j` aliases test threads, not build jobs.
  `cargo test --release` alone is not enough: the engine guards its invariants
  with `debug_assert!`, which compiles out of release entirely, so a test can
  pass green in release and still fail CI. `create_token` is the live example —
  it `debug_assert!`s that the key resolves in the token catalog, so passing
  a card key or a printed name is release-silent and debug-fatal. Clippy does not
  catch this either.
- Finish frontend generation/check/build commands before starting Playwright's
  Vite dev servers. Running `svelte-kit sync` or a build during browser proofs
  can reload active pages and destroy in-page revision probes; use isolated
  ports and stable generated files for concurrent headless runs.

### Card definitions

Each card is one function in the set file of its first printing under
`managym/cards/src/` (the `managym-cards` crate; keys `<set>.<card>`; basics
are `lea.*`), returning the engine's own `CardDefinition` written in its rules
types (`Effect`, `Ability`, `StaticAbility`, `AdditionalCost`, `TargetSpec`,
`TriggerCondition`, …), which live in the `managym-state` crate
(`managym/state/`) and are re-exported as `managym::state`. The engine reaches
the sets only through `managym_cards::catalog()` and `token_catalog()`. Keywords are `Keyword`
variants held in a `Keywords` set (`Keywords::of(&[Keyword::Flying])`);
counters are `CounterKind` entries stored sparsely on the permanent.
That value is the only card representation: there is no authoring language,
compiler, IR or card JSON. What cards and tokens share (name, cost, types,
colors, keywords, abilities, costs, text, power/toughness) is one
`PrintedCharacteristics`, the object as printed before any continuous effect;
`Game::characteristics(id)` is the layered view computed from it. A
`CardDefinition` is `key`, a required `scryfall_id` (the Scryfall card id of
the printing in its set file) and `printed`; a transforming card's back face is
another `PrintedCharacteristics` on it. A static ability or additional cost is
a variant of `StaticAbility` or `AdditionalCost`, never a new field. Add the
function to its set file's `cards()`. New behavior is a new rules-type variant
(with a directed test in `managym/cards/tests/cards/<set>.rs`), never
card-specific logic. Rebuild the extension after card changes.

Tokens are not cards (CR 108.2, 111.1). Each is a `TokenDefinition` (`key`,
an optional `scryfall_id` and `printed`) in `managym/cards/src/tokens.rs`,
listed in its `tokens()` and reached through `managym_cards::token_catalog()`,
keyed `token.<name>`. Effects create tokens by key (`Effect::create_tokens("token.spirit", 1)`),
never by printed name, so two tokens may share a name (a plain 1/1 Spirit
beside TLA's `token.spirit`); give each its own descriptive key. A definition's
identity is its key, never its printed name: a deck entry names the one
eligible card of that name whatever tokens share it, and the learning catalog
binds manifest rows by key (`catalog_tokens()["keys"]`), by name only for the
keyless back faces and fixtures. A token's
`scryfall_id` is the id of its first token printing, or `None` only for a token
with no printing (Lita's Zeppelin). Deck admission, the world pool and card
support never accept a token. The engine registers every card face and token
as an `ObjectDefinition` (one `CardDefId` row: identity, `printed` and token
status); token status is never a printed characteristic, so it lives only on
that row and on game objects, and copy effects copy `printed`, never status.
The cards crate's `tests/cards/scryfall.rs`
checks id shape, uniqueness and token keys offline; `tests/cards/check_scryfall.py`
checks names, sets and token printings online, never in CI, and skips tokens
without an id.

Subtypes stay printed names; `managym/state/src/subtype.rs` is the one CR 205.3
table of their categories. Add a new subtype there (the cards crate's
`every_authored_subtype_has_a_category` fails otherwise), and read creature
types through `is_creature_type`, never by excluding other subtypes.

Declare a parameterless keyword once in the rules enum's `keywords!` declaration
and use `Keywords::of`. Storage grows with the vocabulary; append variants to
preserve present-keyword serialization order and values. When changing storage,
test membership, union, removal and copy independence across word boundaries
without inventing mechanics, widening frozen observations or regenerating
evidence to hide a regression. A new mechanic still needs behavior, learning
tokens and every shared observation path; non-evergreen kinds use fact rows.

Learning tokens derive from the same definitions (`cards/tokens.rs`,
`managym.catalog_tokens`), so a new variant reaches the policy without a
vocabulary edit. Every mechanic must be observable in both the observation
tensor and the tokens: extend the exhaustive guards in
`managym/cards/tests/cards/observability.rs` (keywords, permanent and player status, `Effect`,
`TriggerCondition`, `Ability`, `StaticAbility`, `AdditionalCost`) and give the
mechanic a directed test there. Static and cost rows name a real card and its
behavior test in `tests/cards/<set>.rs`, which the guard checks.

Support is definition: the world pool is every defined card from every set.
Retained `Decklist` records include the diagnostics of their original admission.
For explicit current-world re-admission, parse `Decklist(record.source,
record.filename)` before admitting; passing the retained record keeps its old
issues. Preserve the original record and source bytes as evidence.
Defined non-token cards are supported for deck entry; tokens, rules fixtures
and unknown cards are rejected. Restrict to a set with `world::set_cards` /
`managym.set_cards`, never with per-pack membership lists. Tests never qualify
or gate a card. Jack Heart clarified on 2026-11-03 that known rules bugs must
not block previously playable cards from play; track those defects as bug
Tasks. Playable does not mean bug-free. A test about one card goes in
`managym/cards/tests/cards/<set>.rs`, the set file of that card; engine rules
stay in `managym/tests/rules/`, and shared scenario helpers live in
`managym/tests/support/`. Keep decklists separate from behavior. Constructed
position helpers discard facts their injections can invalidate: direct library
reorders clear positional knowledge, and removing hand cards clears hand
minima. Ordinary rules moves update facts through the engine hooks. Viewer-knowledge writes use
`TrackedState`'s narrow accessors: they do not invalidate layered-characteristic
or mana caches. Keep long printed text in
a local `text_box` binding so rustfmt can format the definition.
The default pack registers every definition in name order (`cards/content_pack.rs`).
Card tests mirror the set files: `managym/cards/tests/cards/<set>.rs`, one module
per set under the single `tests/cards.rs` target of the cards crate (the engine is
its dev-dependency). Both test targets mount the engine's `tests/support/` (`game.rs` builds, `play.rs` acts, `inspect.rs` reads), each
primitive defined once. `five_color_setup` decks hold four copies of each listed
name; a fifth `force_permanent_on_battlefield` of that name returns an existing
copy, so list another card when a test needs more. `inspect::keywords` reads
printed and until-end-of-turn keywords only; assert static grants through the
observation. `force_permanent_on_battlefield` places a permanent without an
entry event (no triggers or turn history); `enter_and_settle` and
`play_and_settle` enter through the zone move.

A forced permanent is not summoning-sick. With exactly enough untapped lands a
manual payment finishes itself after its last object choice; calling `pay_rest`
then overruns into the next priority. Read a new card's printing with `curl`
from `api.scryfall.com` (the framework Python has no CA bundle), two seconds
apart on `cards/search`, and append its row to
`managym/tests/fixtures/scryfall_cards.json` with the fixture's eight fields.

Funded fixtures must finish native spending decisions before passing priority
or inspecting a newly cast spell. Call `spend_colorless` explicitly to preserve
the colors in `five_color_setup`; tests of color, restriction, rider or expiry
choices select their intended native offer.

An `ObjectId` names one appearance of a card, token or copy in one zone (CR
109) and goes stale when that object changes zones (CR 400.7), stack to
battlefield included; a `CardId` names the physical `Card` (CR 108: owner and
definition) for the whole game. `GameState::object_of(card)` and
`card_of(object)` are the only bridges; a token or spell copy has no card.
Hold an `ObjectId` for "that creature", a target or an event subject, a
`CardId` for ownership or "that card", and read what a departed object was
from `ObjectLkiLog`. A zone move is the one place an object is made from
another and returns the new id; `ZoneManager::successor(old)` finds what an
object became by its one next move (CR 400.7e, j). A fact crosses a zone
change only as a `CarryOver` field citing its CR 400.7 sub-rule, never as a
field copied by hand or an id that keeps working. Slots are private storage:
never build an `ObjectId` from a number, in tests included. A test that
follows one card through moves the engine made uses the support helpers
`now`, `zone_of`, `permanent_of` and `move_card`; a pre-move id passed to
`Game::move_object` or compared with an event's object does nothing or fails
([objects and zones](docs/engine.md#objects-and-zones)).

`GameState::object` and `state[object]` check appearance identity against the
zone manager; `ObjectVec` has no object-id or card-id indexing. Zone-entry
code alone uses `entry_object` before placement (new tokens and outside-game
cards), still checking the allocated generation. Departed-source consumers
read LKI and `source_base`/`source_face` definition provenance, never a later
appearance's copy state. Recorded spell, mill and discard predicates use
`recorded_object_matches_predicate` after their captured appearance departs.
`AbilitySource` holds one id and the creator's exact
reference presence; v16 serialization projects both historical source fields.

A permanent's characteristics after continuous effects (name, types, colors,
keywords, static/triggered/activated abilities, power/toughness) are one view,
`Game::characteristics(id)`, which applies static abilities and effect records
in layer, dependency and timestamp order; callers read its fields, and rules
questions (`permanent_can_tap`, lethal damage) use it rather than re-deriving.
Its `activated_abilities` follow layer/timestamp order; activation, payment and
offer code indexes only that list, never `card.activated_abilities`. Resolution
consumes the effects captured at activation. `flow/layers.rs::Context` evaluates
the board without calling the public view; `flow/statics.rs` caches that board
and finalizes activation costs and dynamic mana outputs. Each immutable view is
shared as an `Arc<Characteristics>`. Whole-board readers use `characteristics_many`
(or internal `layer_views` when activation finalization is unnecessary), preserving
their input order and validating the shared board once per immutable scan.
`Game::state` is a `TrackedState` with
`Revisions` (`characteristics`, `mana_pools`, `tapped`, `triggers`): `DerefMut`
advances every counter, and its typed accessors (`tap`, `mark_damage`,
`events_mut`, `move_object`, `mana_pool_mut`, ...) advance only the counters
their field feeds. Tap/untap advances `tapped`; layers retain their board only
when no discovered scope, condition or value predicate reads tap state. Removing
a stun counter also advances `characteristics`. Route a hot write through a typed accessor only when no
layer, static condition or trigger source reads the field; debug builds
recompute every cached view and index on read and assert they match, so a
misclassified accessor fails `cargo test` in debug. A `BoardScan` narrows this
for the raw layer board only: it validates the board when it begins, and its
shared borrow excludes writes, so reads inside it skip the repeat. Hold one
across `&self` code that reads the board per source or per ability. Continuous-effect records
live in `EffectRecords` (indexed by affected object and player); trigger
sources are looked up per event kind through `TriggerIndex`
(`flow/triggers.rs`), invalidated by trigger and layer dependencies; departed-object
facts are an `ObjectLkiLog`. A cloned `Game` carries these caches.
Layer rebuilds reuse immutable copiable source objects and static plans, bound to
exact definition identity, appearance and timestamp; copy/transform definition
changes invalidate that source. Intermediate layer objects detach on writes.
Conservative characteristic read/write summaries skip impossible dependency
probes; plausible pairs still compare instructions and generating-ability
existence. Debug rebuilds source plans independently and checks skipped pairs.
Static abilities of one source share a timestamp and apply in definition order,
so a definition lists an ability loss before the grant it keeps (Honest Work).
A change that lasts until end of turn, or for any other duration, is an
`Effect::ApplyContinuous` record (`state/effect.rs`; expiry in
`flow/effects.rs`), never a new `Permanent` or `Player` field.

An `Effect` is a verb over an `ObjectSelector` (`This`, `Target(spec)`,
`Triggering`, `Attached`, `CreatedThisWay`, `Grantor`, `Moved`, `You`,
`Chosen { predicate, except }`, `Each { of: Scope, predicate, except,
max_mana_value }`), never a `<Verb><Object>` or `<Verb>All` variant:
`PutCounters { on }`, `Sacrifice { what }`, `Move { what, to }`, `Tap { what }`,
`Destroy { what }`, `DealDamage { amount, what }`, `ApplyContinuous { affected }`.
"Each creature" is the verb over `Each`; "all other" is `except: Target`;
"another" is `except: This`. One resolver, `Game::select`, names the objects
(last known information for `This` after the source leaves); `Chosen` suspends
the instruction on `Decision::ChooseObject` for any verb ("tap a creature you
control"); a verb that depends on which object it hit is an `If` composition,
and "if you do" is `Condition::Moved` over the frame's latest `Move` or
`Sacrifice`. Flicker is two `Move`s, the second over `Moved`. A `Move` to the
library a card is already in, with a stated `library_end`, reorders it without a
zone change ("then shuffle and put that card on top"). Amounts are
`EffectValue` (`Fixed(n)` or read at resolution); a verb never has a
`<Verb>Value` twin. Use the constructors (`Effect::exile`, `return_to_hand`,
`put_counters`, `destroy`, `deal_damage`, `gain_life`, `discard`,
`create_tokens`, `add_mana`, `sacrifice_this`, `may_sacrifice`, `if_you_do`,
`ObjectSelector::each_you_control`, `chosen`) in card code and write the full
`Effect::Move { .. }` literal only for `under`, `with_counter` or `linked`.
A stack ability is no card object, so `CounterAbility` reads the clause's
`Target` directly; `Destroy` over several objects is one simultaneous event
with pre-batch trigger sources.

A `TriggerCondition` is "whenever X does Y": an event over its actor, never an
`<Actor><Event>` variant. A player actor is an `Actor` (`You`, `Opponent`,
`AnyPlayer`): `Casts { who, predicate, nth }`, `Draws { who, nth }`,
`BeginningOf { step, whose }`, `Performs { who, any_of }` for keyword actions
(search, scry, surveil, the four bends) and `PutIntoGraveyard { whose,
predicate }`, `PutsCounters { who, kind, on }`. An object actor is a
`TriggerSubject` (`This`, `EnchantedCreature`, or `Matching { controlled_by,
predicate, another }`, whose controller is the same `Actor`; card code uses
`you_control`, `another_you_control`, `any`, `another`), as in
`Dies { subject }`, `Sacrificed { subject }`,
`Attacks { subject, per, attackers }` ("alone" and "with at least one other"
are `AttackSize`) and `DealsCombatDamage { subject, to, per }`. Counts,
predicates, recipients and once-per-object versus once-per-batch ("one or
more") are fields of the event, never names. The player who acted (or, for a
step, the active player; for a graveyard, its owner) is "that player",
`PlayerSelector::TriggeringPlayer`. `ActiveIf`, `OnceEachTurn`, `FromZone`,
`InGraveyard`, `WasCast` and `WhileSourceAttacking` wrap any event. Ability
descriptions and learning tokens come from one template per event that fills
in the actor and parameters, so a new actor or count needs no new variant,
text or token. Card code writes the common forms with
`TriggerCondition::casts(who, predicate)`, `beginning_of(step, whose)` and
`attacks(subject, per)`.

Activated abilities that function from hand or graveyard set
`ActivatedAbilityDefinition.zone` (landcycling included); they are offered as
`Action::ActivateCardAbility`. An ability's index is its position in
`activated_abilities`.

Fixed mana abilities on a card are additive: multiple fixed productions
produce all listed colors together. A land that produces one chosen color uses
one activated modal mana ability, not multiple fixed mana abilities. Verify each
choice by paying a real cost and checking the remaining mana. A mana ability
with other instructions ("Add {G}{G}. You gain 2 life.") lists its output first;
the untargeted rest (`ActivatedAbilityDefinition::mana_suffix`) runs once, off
the stack, as the source's frame before the output choice (CR 605.3b).

### Decision-memory rows

Collected and bootstrap rows retain immutable prefixes; materialize raw memory
only for admission/preparation, never as configured padded buffers per row. Prepared learner rows keep
ordinary action fields for reference distributions, never a second padded memory
owner beside Prepared (including prefetch). Numerical incidents are bounded
diagnostics and must report omitted complete prefixes explicitly. Etude's game
sampler prepares the log for a memory checkpoint and forwards each such row as
its own batch; `blank_decision_memory` on a seat's config is the ablation.

Read-only native transport metadata must also be exposed through
`managym.recording._VECTOR_READS` and flattened in `ShardedVectorEnv`;
collectors use these wrappers rather than raw native vectors.
Runtime-checkable vector extension protocols list their concrete capability
methods. Python 3.12+ protocol checks use static lookup and miss the ordinary
native reads delegated by `RecordedVectorEnv.__getattr__`.
CLI subcommands import their execution runtime at the command boundary when
needed to keep migration/help available without the training SDK.

### Etude training records

A launched training job's experiment label must match the label supplied to
Etude tracking by its recipe/stage, including training in a subprocess. Etude
rejects a mismatch with `experiment_differs_from_launch` before the first update;
changing the launch label does not rename the recipe's tracking label.

For a training warm start, `run.input("start")` inside `with track(...) as run`
returns the delivered `Path` directly (Etude 0.4.3).

Etude preserves a checkpoint's source suffix. Worker tests wait for a registered
checkpoint through the SDK, rather than assuming the file ends in `.pt`.

Training row tables belong under the metric run's `artifacts/`, named by their
content digest, with `.partial` staging before atomic rename. Etude's tracker
and launcher courier carry these immutable author artifacts before metric
references; they are never checkpoints. Resolve a reference through its selected
metric block's Storage and run prefix, after attempt/update supersession, so
relocated evidence and resumed runs keep the same meaning.

Etude report labels are the text metrics `run/kind` and `run/recipe`; explicit
declaration kind/job fields take precedence. `RunData.points` reads numeric
metrics only: read text through the stored block's series columns without
decoding the entire update history.

Etude requires every declared clock on each row, table and checkpoint. A
missing checkpoint clock warns and skips recording; verify the recorded
checkpoint, including downstream recipe outputs, rather than file existence.

### Etude card evaluation

Etude page builds reload repository neighbours, including imported helpers.
Tests that build pages and retain module-level references must restore saved
`sys.modules` entries and remove neighbours first imported during the build,
while keeping new dependency imports (including Torch lazy registrations).

Etude lists a file in its experiment pages if it defines `training_jobs`,
`eval_jobs` or `report`. Standalone benchmark CLIs use `render_report` for
their offline report renderer.

Etude discovers experiment files with `runpy`; pass evaluation callbacks from
their canonical importable module, not the discovery file's `<run_path>`
namespace or a `python -m` invocation's `__main__`; the same holds for agent
loaders. Exercise discovery with `etude.controller.declarations.read_experiment` in runner tests,
or `etude.controller.declarations.read_training_jobs` for a file's `training_jobs()`. Local
evaluation must supply `ETUDE_OUTPUT_DIR` when its callback writes replay evidence;
create the directory before starting local evaluation because Etude writes progress
there before calling the experiment. Launched jobs receive it automatically.
Local evaluation and reporting also read the default tracking directory alongside
`--source`. Set `ETUDE_TRACK_DIR` to a study-specific directory for both commands
when smoke and full tournaments share a name.
Etude imports evaluation callbacks by module and qualified name, including local
runs; callbacks take `play(match, agents)`. A closure or callable instance is
not a portable entry point. Managed jobs share `manabot.eval.agents.AGENTS`;
`eval.agent_loader` alone chooses the framework/device, and game callbacks use
the supplied agents in-process. Seating configs containing records with omitted fields must pass their JSON
model dump, so a retained tournament compares alike on resume. Persistence fixtures that inject local outcomes use
`tests/fixtures/evaluation_callbacks.py` across the real game process boundary.

Etude hashes tracked source paths (`git add -N` admits new files for local
implementation checks). Eval runs are sealed blocks: each game's card evidence
is a blob of its match, named in `card_plays` by SHA256 and read back through
`etude.eval.query.read_blobs`. Card queries read only `CARD_ROLLUPS`; open the
database with `manabot.eval.card_query.card_db`, and give a changed rollup a new
name. Never put card evidence in the checkpoint namespace. Per-file runs from
earlier SDKs become readable through `etude convert-eval`, which writes blocks
beside their files and never rewrites them; converting stored data is an
explicit step of its own.

Etude evaluation tables use `seat` for the participant's index in
`Match.players`, and `game_seat` for its native game seat (0 is on the play).
Join results to `seats` on `(run, match, seat)` and to `players` on `player`;
group native play/draw reports by `game_seat`. Etude's conversion owns the
historical `player`/`seat` column rename; never rewrite retained tables for it.
Every Etude result row names a participant `seat` or a declared `group`; use
`group: all` for a receipt covering the whole match. `card_plays.game` is the
native trace's game ID, not its leg index; join evaluator rows to retained card
blobs by `trace_sha256` within the match, then validate the blob's deal and leg.

### Offline inference reports

Historical LLM receipt readers stay independent of executable model, simulation
and native imports. Load the evaluation projection only for evaluation inputs;
keep its simulation dependencies off the LLM-only path. Check this boundary in a
fresh process with those imports blocked, and exercise mixed reports through the
actual builder CLI in the existing headless Perf proof.

Etude launched source is an extracted commit without `.git`. Read its revision
through `etude.eval.source.source_commit`; direct `git rev-parse` in a training
loop cannot identify it. Local `--seconds` passes an ISO-8601 `ETUDE_DEADLINE`;
continuation targets use `ETUDE_UNTIL_CLOCK` and `ETUDE_UNTIL_VALUE`.
`etude continue` queues the child in the parent's pool with its rental settings.
A backend or device config override must be runnable in that pool.

The repository requires `etude-sdk>=0.4.6,<0.5` from PyPI for training
continuation callbacks and versioned storage listing. Etude's shared records,
storage and errors live in `etude.records`, `etude.storage` and `etude.errors`, and training
uses `etude.train.tracking`. Experiment entry points are `training_jobs()` and
`eval_jobs(state)`; discovery is in `etude.controller.declarations`, state in `etude.jobs`.
A resumed tracker uses `resume=True`; its first logged row sets the supersession
coordinate, and an attempt that logs nothing supersedes nothing. Retained
nested tracker declarations still decode through Etude's `RunRecord`.
