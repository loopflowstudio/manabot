Never add "Co-Authored by:" tags in commits

## Naming

- **Manabot** — the whole project and public brand at `manabot.gg`.
  The repository is `loopflowstudio/manabot` (renamed at Jack Heart's request
  on 2026-10-15).
- **Academy** — the browser UI for inspecting bots and replays: the `academy`
  package, its Svelte frontend in `academy/ui`, and their service namespaces. It is a client of
  manabot and secondary to training. The project website uses Manabot.
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
provisional Fantasia project name. Academy remains the name of the UI.

Boundary rule for anything those lines don't settle: if it is the browser UI,
it is Academy; if it trains or evaluates the agent, it is manabot; if it is the
world, it is managym; if it launches or records experiments, it is Etude. So
the play server, experience protocol, curated packs, and their env/error
namespaces (`ACADEMY_PLAY_*`) take Academy names, while training
infrastructure (wandb project, presets, ops jobs, `MANABOT_*` config vars)
correctly keeps manabot names. Frozen evidence — experiment IDs, contracts,
receipts, wandb history — is never renamed, and neither are the compatibility
surfaces that predate the Academy name: the `x-etude-*` wire headers,
`etude.*` browser storage keys and schema identifiers. Hosted services are `academy-play` and `fantasia-website`, with
deployment secrets in Doppler `manabot/prd` (Jack Heart, 2026-10-14).
`etude.gg` and its legacy aliases redirect to `manabot.gg`, preserving paths
and queries; hosted play is available at both `play.manabot.gg` and
`academy.manabot.gg` (equivalent addresses selected by Jack Heart, 2026-10-14).
Namecheap is the registrar and Cloudflare manages DNS. Existing service
identifiers stay unchanged.
The shared Etude experiment services retain their names. "Sepia Etude" remains
the name of the visual language.

Code dependencies flow Academy → manabot → the engine in `manabot.managym`.
PyPI distributes them in one `manabot` package; `import managym` is a compatibility
alias, not a separate distribution. manabot never imports
Academy, eagerly or lazily (`tests/test_package_boundaries.py` enforces it).

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
This direction does not approve a particular redesign or change frozen worlds,
checkpoints or experiment evidence.

## Style

- Package style guides live in each package README (`manabot/README.md`,
  `managym/README.md`). Avoid transient comments that denote changes; pay
  attention to file headers and README content; propose small, iterative
  changes.

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
- Put imports at the top; use explicit imports and underscore-prefixed private
  helpers. Keep one implementation and one owner for each fact; use git for
  old versions rather than `_new`, `_old`, or `_backup` variants.
- Import names from their defining modules (`manabot.env.observation`, not
  `manabot.env`); package `__init__` files re-export nothing, so evaluation
  game and replay children stay Torch-free
  (`tests/test_package_boundaries.py` enforces the child modules). Launch
  evaluation children with `manabot.sim.child.ChildProcess`, not
  `multiprocessing` spawn, which re-imports the parent's `__main__`.

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
- Native scalar/vector observation buffers bind continuation input semantics by
  relation width: v2/v3 have 32 columns, v4 and v5 have 33; validity remains
  column 31. At 33, the player row width tells v4 (40) from v5 (60).
  Preserve the frozen widths and derive buffers from `ObservationSpace`.
- Decklist disclosure (open by default, or closed) lives on native
  `PlayerConfig.disclosure`, written by `MatchConfig.decklists` and Academy
  `GameConfig.decklists`. Observations, search and possible worlds read it
  through `Game::decklist_known`/`require_open_decklist`; never filter decklist
  information in an encoder or the browser.
- Saved observation configurations use `ObservationConfig.from_checkpoint` so
  an absent policy-input version retains v1 semantics. Fresh construction uses
  the current version. Keep version-specific zero/default exclusions stable:
  architecture receipts hash the serialized configuration, including omissions.

## Commands

- **All Python commands run through uv.** This repo's environment is
  uv-managed; bare `python`, `uvicorn`, `pytest`, `maturin` etc. will miss the
  venv or pick the wrong interpreter. Always `uv run <cmd>` (or
  `.venv/bin/<cmd>` when a script must reference the interpreter directly).
  Never emit an un-uv'ed Python command in docs, scripts, or instructions.
- Python is pinned to 3.12 (PyO3 caps at 3.13; the venv is 3.12). Fresh venvs:
  `uv venv --python 3.12`.
- After changing Rust under `managym/src`, rebuild from the repository root so
  uv uses the pinned root environment instead of creating a nested venv:
  `uv run maturin build --release -i .venv/bin/python -m managym/Cargo.toml -o managym/target/wheels`,
  then place the cp312 `.so` from the wheel at
  `manabot/managym/_managym.cpython-312-darwin.so`. Replace it atomically (write a new
  file, then rename); overwriting a mapped extension in place can trigger macOS
  code-signature termination on subsequent imports.
- Inspect a bot in the browser: `./scripts/play` (installs locked local dependencies,
  starts backend + frontend, and stops both on Ctrl-C).

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

Read [validation by change area](docs/development/validation.md) for the
checks required by the affected package and workflow.

- Native isolated failure receipts name `AgentError` / `NativePanic`; PyO3
  exposes `PyAgentError` / `PanicException`. Replay compares these through
  `manabot.env.replay.matches_retained_error`, retaining original class names and
  full vector diagnostic payloads in evidence.
- Experiment results under `experiments/data/` live in the artifact store. Before
  committing new results or merging a branch that tracked them, publish their
  bytes with `uv run --extra artifacts python -m manabot.data publish <path>`,
  commit `experiments/data.manifest.json`, and remove git tracking while retaining
  the local files. Verify with
  `uv run --extra dev pytest tests/infra/test_experiment_data_untracked.py -q`.
- After a world or observation-shape change, validate current-world arena
  admission and monitoring with `uv run --extra dev --extra notebook --extra artifacts --extra reports pytest tests/semantic/test_elemental_world.py tests/eval/test_evaluation_models.py tests/eval/test_mirror_admission.py tests/training/test_monitor_evaluation.py tests/model/test_choice_separability.py -q`.
- Artifact storage and Etude integration tests require the `artifacts` extra.
  Validate dependency changes in an isolated environment so locally installed
  extras cannot hide missing CI dependencies:
  `uv run --isolated --extra dev --extra artifacts pytest tests/training/test_artifact_storage.py tests/training/test_etude_sdk.py tests/training/test_etude_source.py tests/training/test_etude_job.py -q`.
- **CI runs `cargo test` in debug, so validate in debug before landing.**
  `cargo test --release` alone is not enough: the engine guards its invariants
  with `debug_assert!`, which compiles out of release entirely, so a test can
  pass green in release and still fail CI. `create_token` is the live example —
  it `debug_assert!`s that the name resolves to a token definition, so passing
  an ordinary card name is release-silent and debug-fatal. Clippy does not
  catch this either.
- Finish frontend generation/check/build commands before starting Playwright's
  Vite dev servers. Running `svelte-kit sync` or a build during browser proofs
  can reload active pages and destroy in-page revision probes; use isolated
  ports and stable generated files for concurrent headless runs.

### Card definitions

Each card is one function in the set file of its first printing under
`managym/src/cardsets/` (keys `<set>.<card>`; basics are `lea.*`), returning
the engine's own `CardDefinition` written in its rules types (`Effect`,
`Ability`, `StaticAbility`, `AdditionalCost`, `TargetSpec`, `TriggerCondition`,
…). That value is the only card representation: there is no authoring language,
compiler, IR or card JSON. `CardDefinition` holds identity, characteristics and
ability/cost lists; a static ability or additional cost is a variant of
`StaticAbility` or `AdditionalCost`, never a new field. Every token is a
`CardDefinition` with `is_token: true` in `cardsets/tokens.rs` (keys `token.*`,
Scryfall id of its first token printing). Add the function to its set file's `cards()`. New behavior is a new
rules-type variant (with a directed test in `tests/cards/<set>.rs`), never
card-specific logic. Every definition carries `scryfall_id`, the Scryfall card
id of the printing in its set file (`tests/cards/scryfall.rs` checks shape and
uniqueness offline; `tests/cards/check_scryfall.py` checks names and sets
online, never in CI). Rebuild the extension after card changes.

Learning tokens derive from the same definitions (`cards/tokens.rs`,
`managym.learning_catalog`), so a new variant reaches the policy without a
vocabulary edit. Every mechanic must be observable in both the observation
tensor and the tokens: extend the exhaustive guards in
`tests/cards/observability.rs` (keywords, permanent and player status, `Effect`,
`TriggerCondition`, `Ability`, `StaticAbility`, `AdditionalCost`) and give the
mechanic a directed test there. Static and cost rows name a real card and its
behavior test in `tests/cards/<set>.rs`, which the guard checks.

Support is definition: the world pool is every defined card from every set, and
a card is supported unless its set file marks it incomplete (`tla::INCOMPLETE`,
with the missing behavior). Restrict to a set with `world::set_cards` /
`managym.set_cards`, never with per-pack membership lists. Tests never qualify
or gate a card. A test about one card goes in `managym/tests/cards/<set>.rs`, the
set file of that card; engine rules stay in `tests/rules/`, and shared scenario
helpers live in `tests/support/`. Keep decklists separate from behavior. Keep long printed text in
a local `text_box` binding so rustfmt can format the definition.
The default pack registers every definition in name order (`cards/content_pack.rs`).
Activated abilities that function from hand or graveyard set
`ActivatedAbilityDefinition.zone` (`ActivationZone`); they are offered as
`Action::ActivateCardAbility`.

Native fixed mana abilities on a card are additive: multiple fixed productions
produce all listed colors together. A land that produces one chosen color uses
one activated modal mana program, not multiple fixed mana programs. Verify each
choice by paying a real cost and checking the remaining mana.

### Etude card evaluation

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

### Offline inference reports

Historical LLM receipt readers stay independent of executable model, simulation
and native imports. Load the evaluation projection only for evaluation inputs;
keep its simulation dependencies off the LLM-only path. Check this boundary in a
fresh process with those imports blocked, and exercise mixed reports through the
actual builder CLI in the existing headless Perf proof.
