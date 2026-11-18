# MTG-353 plan

Jack Heart, 2026-11-17, on the previous version of this plan and the
versioning and mechanics readings that followed it: "sounds good. launch a
graph to get it built!"

Jack Heart, 2026-11-17 (steer): "Build scratch/plan.md: the best player keeps
loading and playing on main, with a CI pinned-pilot check and recheck. Publish
a PR and stop at human review; do not land. Follow-up work is MTG-386."

Criterion (Jack Heart): "The whole point im saying is to try to make my
trainign dollars go farther". Design history: mtg-353-compatibility.md.

The outcome and the eight parts are unchanged from the plan that was approved.
This version adds what the kickoff probes found on current main and orders the
build by them. Choices made here for review are marked **Chosen**.

The concept review of 2026-11-17 (`scratch/concept-review.html`) changed five
things, each written into the sections below: a card the pilot trained on is
read from the program stored in its checkpoint; the writer catalog is found by
the card digest the checkpoint records; the old-family pinned pilot is written
at `ab2b757d9` so CI exercises the best player's own load path; a declared
break is a reason in the recording and no separate list; `recheck` measures
strength only.

## Outcome

Today's best player (TLA frontier Small, run `59718fe34945`, update 1952,
written at `ab2b757d9`, policy input v9, sha256 `c95c61aa…bbaf8`) loads on
current main unmodified, plays any deck in the pool, and its strength on main
is measured against its record (0.72 against the first-64 heuristic, 64 games,
`experiments/reports/tla-frontier-c5fba4-results.md`). A CI check fails any
later change that would alter what a stored pilot computes, unless the change
says so.

## What the probes found (2026-11-17, main at `44f76fe987`)

The branch was 30 commits behind; `lf sync` moved it to main. Main now has
input v11 with v9/v10 as the deprecated legacy fork (PR #521), safetensors
containers and `Prepared` as the one policy input (MTG-333), and the
record-vocabulary guard (MTG-369). Probe scripts: `scratch/probes/`. Local
copy of the checkpoint, its run spec and the writing revision's naming
catalog: `~/src/manabot.mtg-349-research-how-should/mini-heart.localobe/`.

1. **Main refuses the best player for five reasons, not two.**
   `load_checkpoint` says: `agent_player 1x73 -> 1x74, opponent_player 1x73 ->
   1x74, naming_vocabulary, rules, world_digest; it records no source
   revision`.
   - The player row gained `decklists_open` (`c035d5566a`), in the legacy
     layout too. This is a real added column, so part 3 is needed for the best
     player itself, not only for future checkpoints.
   - `rules.possible_world_version` went 3 → 4 (`26593101e3`). It describes
     the search and belief space; a policy without belief inputs never reads it.
   - The checkpoint has no `source` record. Its revision comes from the report.
2. **Token keys are not stable across revisions.** Of 1,836 learned token rows,
   64 have keys that no longer exist on main (22 `Variant`, 15 `Structure`,
   5 `Field`, 2 `Text`, 18 `DefinitionRef`, 2 record names): Rust types and
   fields were renamed. 138 of the 298 TLA and basic-land definitions now
   contain at least one token the checkpoint never learned. Binding by key
   alone would leave the pilot reading blank vectors inside nearly half the
   cards it trained on. All 348 naming names still exist.
3. **`DefinitionRef` keys are positional.** `managym/src/cards/tokens.rs:419`
   writes the referenced definition's row (`CardDefId`) as the token value, and
   `(7, row)` gets a learned row. After cards are added the same key names a
   different token or back face, so key binding (and MTG-349's prototype)
   silently misreads created tokens. This is the one place a name is reused
   with a new meaning today.
4. **With keys bound, rows zero-filled and the added column ignored, the pilot
   plays.** Six complete self-play games over the training decks on main,
   1,005 decisions, no admission refusal. That is thin: MTG-349's failure came
   after 30 games. MTG-369 (`ae8e9be267`) traced that class of failure to
   runtime effects such as Koh's borrowed abilities missing vocabulary
   registrations, fixed them and added an exhaustive guard. The proof run is
   the full-game confirmation (part 4).
5. **Column names already exist natively.** `managym.observation_layout(
   policy_history=, turn_state=, legacy=)` publishes every array's named
   columns with capacities as open extents
   (`managym/src/agent/observation_layout.rs`). No column map needs writing by
   hand: a checkpoint without recorded names takes the layout of its input
   version at its recorded widths.
6. Row capacities already come from the checkpoint's own
   `observation_hypers`, so they do not differ between checkpoint and runtime;
   only column widths and catalog sizes do.
7. No binary tensor is committed anywhere in the repository; the largest test
   fixture is 226 KB. The best player's file is 3.9 MB.
8. The local laptop ran the probe at 73 s per game under other sessions' load,
   against MTG-349's 8–10 s on the mini. Measurements belong on the mini or a
   rental.

## Design

### Source of truth

One function decides whether a checkpoint runs here: `model_interface` in a
new `manabot/model/interface.py` compares what the checkpoint recorded with
the runtime and returns a typed result (compatible with a binding report, or
the named reasons it is not). The world never decides loading, so the check
does not live in `world.py`; that module keeps deck admission and the setup
binding. The three load sites (`read_container` and the two in
`manabot/backends/torch/checkpoint.py`) call `model_interface` and raise its
reasons; Academy's `checkpoint_incompatibility` (`manabot/model/catalog.py`)
returns the same reasons as text. `validate_checkpoint_world` stays only as
the setup-bound path (below); the second field table in `catalog.py` goes.
The serialized key `world_binding` keeps its spelling.

It checks, by name:

- **Input columns.** Per array, the checkpoint's ordered column names against
  `managym.observation_layout` for its input version. Equal: nothing to do.
  Runtime has more: the extra columns are dropped in `prepare`, before the
  model reads the array, so the model computes what it computed before. A
  column the checkpoint needs is gone or renamed: refuse, naming it. A
  checkpoint without recorded names has the first N names of its version's
  layout at its recorded width N. That holds because columns are only ever
  appended within an input version (`decklists_open` is index 73, the last
  player column); the old-family pinned pilot fails if one is inserted.
- **Action meaning.** `decision_version`, `naming_representation`,
  `naming_branching` equal. `possible_world_version` only when the model has
  belief inputs.
- **Vocabulary.** `learning_schema_sha256` equal (token kinds and roles);
  every learned row is bound by key (below).
- **Architecture.** The existing architecture receipt and state names.

It no longer compares `world`, `world_digest`, `naming_vocabulary` or whole
buffer shapes. Those stay in the file as recorded facts.

### Binding learned rows by key

One NumPy function in `manabot/policy/` takes a `PolicyCheckpoint` and the
runtime's tables and returns the checkpoint on the runtime catalog; torch
`.pt`, containers and MLX all load through it (the `.pt` path becomes
`from_torch_state` → bind → `assemble`, removing its separate loader).

- **Chosen: a card the pilot trained on is read as it was learned.** A
  definition whose key is in the checkpoint's catalog and still defined keeps
  the token program stored in the checkpoint's own `tokens`/`lengths` buffers
  (references rebound by definition key). Its card vector is then equal to the
  writing revision's by construction, whether Rust names were renamed or a
  variant was folded into another (`Exile` into `Move { what, to }`): a
  restructured serialization leaves no key unaccounted for and still changes
  the vector, so counting keys cannot show the pilot is intact. Every other
  definition (cards added since) reads the running catalog's program. The load
  report lists the trained cards whose running program differs from the stored
  one; those are where a later warm start (MTG-386) changes what the model
  reads. Passed over: running programs for every card, with the loss measured.
- Token and record-name rows the runtime emits (programs of added cards, fact
  and effect records) move by key, using the checkpoint's own `record_token_*`
  and `record_name_*` buffers. A learned row only stored programs use keeps
  its place in the table.
- A learned key absent from the runtime follows `managym`'s **rename table**:
  one committed list beside the token serializer of
  `(kind, old name) → (kind, new name)`, or `→ none` for a name that went
  away. The old key is the hash of the old name, so the table needs no dump at
  load. An absent key with no entry refuses the load and prints the key: every
  Rust rename is classified once.
- `DefinitionRef` rows bind through the definition's key (`token.spirit`), on
  both sides. New checkpoints record their definition keys and naming names in
  row order; the token serializer itself is unchanged.
- Naming rows bind by name.
- Every other catalog-owned buffer is rebuilt from the running catalog,
  whatever its shape.
- **Chosen:** a row for a key the checkpoint never saw starts at the mean of
  its table's learned rows, and the load report counts them. Passed over: zero
  (reads as padding inside a GRU sequence). The proof run's new-card arm is
  where this shows.

A checkpoint written before this change lacks its definition keys and naming
names. They come from a **writer catalog**: one committed file per card
digest, `manabot/model/writer_catalogs/<world_digest>.json`, holding the
definition identities and complete naming candidates in row order. It is found by the
`world_digest` the checkpoint records, never by revision: the best player has
no `source` record. What checks it: the names reproduce the checkpoint's
`naming_vocabulary` digest exactly; the key count equals the checkpoint's
definition rows; and the load report's count of trained cards whose stored and
running programs agree is near the whole catalog only when the keys are
right. The token names (`(kind, value)` → name) are dumped once from
`ab2b757d9` for authoring the 64 rename entries (finding 2) and are not
committed; `~/src/manabot.mtg-349-value-probe` is detached at that revision.

### Pinned pilots in CI

**Chosen:** two width-eight pilots committed under
`tests/fixtures/pinned_pilot/` (2.1 and 3.0 MiB as saved; the native catalog
buffers dominate, not the weights), each with one complete recorded game on
fixed decks: seed, actions taken, and the pilot's action probabilities at
every decision. The v11
pilot is written on main. The v9 pilot is written from the `ab2b757d9`
checkout (a small model saved there, its `.pt` committed as written), so it is
the best player's kind of file: 73 player columns, no column names, no
definition keys, the old token names, the same `world_digest` and so the same
writer catalog. A v9 pilot saved on main would have none of those and CI would
not cover the path the best player takes. Its recording is made on main after
binding, with its card vectors compared once against the old checkout. Passed
over: committing the best player (3.9 MB of binary), and fetching it in CI
(needs store credentials). The best player itself is covered by the same test
when its file is present locally, and by `recheck`.

`tests/model/test_pinned_pilot.py` runs in the Integration job (it already
runs `tests/model/` with the built extension). It loads each pilot through the
ordinary loader, replays the recorded actions through the engine and encoder
on the current tree, and compares. The failure says which layer moved: the
game no longer replays (rules), or it replays and the probabilities differ
(encoder, vocabulary or loader). Probabilities must agree within 1e-5; the
chosen action's rank must be equal. Bit equality across Linux and macOS is not
claimed.

A change that moves it has two ways to pass, and both are visible in review:
add the rename or column it forgot, or run
`uv run python scripts/pinned_pilot.py record --reason "<why>"`, which rewrites
the recording with that reason and the commit in its header, as
`managym/perf/baseline.json` carries its reference's reason. The file's git
history is the list of declared breaks; a changed reason in a PR is the prompt
to run `recheck`.

### recheck

`uv run manabot recheck <reference>` re-measures a stored strength reference
on the current tree and prints recorded against measured. A reference is a
small committed record (`experiments/references/<name>.json`): the checkpoint
artifact, opponent, deck pairs and legs, the recorded score with its interval
and the revision that produced it. Exit status is nonzero when the measured
score falls outside the recorded interval. It reuses the existing local
evaluation path (`checkpoint_request` in `manabot/eval/local.py`,
`evaluate_local` in `manabot/eval/evaluation.py`); it adds no evaluator.

**Chosen:** strength only. A throughput reference is a different measurement
(a training benchmark on a named machine, nothing the evaluator runs) and
would put two mechanisms behind one command; it is left for its own Task.

First references: frontier Small u1952 against the heuristic, and two earlier
checkpoints of the same curve.

### Definitions written down

One section in `docs/observation-and-model.md`, and the world-binding and
input-version lines of AGENTS.md amended to match:

- A world number is the span of revisions over which stored measurements are
  comparable. It never blocks loading. It moves when `recheck` shows a
  reference outside tolerance; a declared pinned-pilot break, a moved Engine
  Budget decision digest or a heuristic change is the prompt to run it.
- Formats (TLA Limited, Standard) are named card pools plus deck rules.
- The model interface has no number: columns, token names and action kinds are
  bound by name, a name is never reused with a new meaning, and a rename is an
  entry in the rename table. Existing version keys keep their spelling.
- Card identity is never learned. The key is the stable identity; `CardDefId`
  is a per-revision pointer. Learned tables indexed by card or name
  (`name_candidates.embedding`, generic-world `card_definitions`,
  `belief_card_embedding`) bind by key or are refused with a reason.
- A new decision keeps the engine's previous automatic behaviour as a
  selectable action or seat default.
- Anything a pinned revision can do is done there: no exact replay on main.
- The v9/v10 fork's removal (stated as 0.4.0) waits for a v11 pilot at least
  as strong as the best player.

## Acceptance check at gate

```
uv run --extra dev pytest tests/model/test_pinned_pilot.py tests/model/test_world.py \
  tests/model/test_policy_input_rejection.py tests/model/test_torch_container_load.py -q
uv run manabot recheck experiments/references/tla-frontier-small-u1952.json
```

The first passes. The second, on the mini, loads the unmodified file, plays
its reference games with no admission refusal, and prints recorded 0.72
against the measured score and interval. The AGENTS.md world and observation
suite passes unchanged.

## Absent and error states

- Learned key absent from the runtime with no rename-table entry: refused,
  key printed, with the writing revision when recorded.
- Column the checkpoint reads is absent: refused, column named.
- Old checkpoint whose `world_digest` has no writer catalog: refused, naming
  the digest, the file to add and the dump command to run at its revision.
- Writer catalog that does not reproduce the checkpoint's naming digest or
  row count: refused as the wrong catalog.
- A trained card whose running program differs from the stored one: runs on
  the stored program; the load report names the card.
- No `source` record: the message says so and the reference record carries
  the revision.
- Setup-bound checkpoints (`setups`, `content_manifest`) keep today's whole
  binding comparison through `validate_checkpoint_world`, so a fixed-match
  run's checkpoint still stops loading when cards are added.
  `record_vocabulary != "catalog"`: refused as today.
- Closed decklists for a pilot trained open: runs; the model does not read
  `decklists_open`. Out of distribution is measured, not refused.
- Pinned recording does not replay: the test names the decision and the
  `record --reason` command.
- A reference whose opponent changed since its record (the heuristic): the
  recheck output says so beside the measured score.

## Exclusions

Exact replay on main; retained encoders beyond the existing fork; per-seat
interfaces; warm start, grow and teacher (MTG-386); changes to v9/v10/v11
encoders beyond recording column names; setup-bound bindings; throughput
references; a v11 retrain of the best player; landing.

## What goes

- For world-bound checkpoints, the equality gates on `world`, `world_digest`,
  `naming_vocabulary` and whole `input_schema` shapes, and
  `binding_difference`'s shape listing where the column check replaces it.
- `_BINDING_FIELDS` and the mirrored logic in `checkpoint_incompatibility`.
- The separate `.pt` state-dict load in `load_checkpoint`.
- Tests in `tests/model/test_world.py` and
  `tests/model/test_policy_input_rejection.py` that assert refusal on a card
  digest or naming digest; their layout-refusal cases stay, restated by column.
- AGENTS.md: "The world binding compares every input buffer's shape … Do not
  bump the input version for a layout change or add a loading path for an
  older layout", and README lines saying a changed player-buffer shape fails
  world binding.
- MTG-349's untracked `experiments/adopt_pilot.py` is never landed.

Survives untouched: recorded binding fields in existing files, frozen
experiment evidence, `UnsupportedPolicyInput` for v1–v8, replay's card-digest
requirement, the legacy encoder.

## Slices

1. **The best player loads and plays on main (complete).** Writer catalog
   export and the first comparison are complete. The unmodified u1952 file
   contains 360 definitions. Comparing its expanded program buffers to main,
   with DefinitionRef tokens joined through definition identity, gives 224
   identical, zero rename-only and 136 with a different token-kind sequence.
   Of the 296 writer TLA/basic rows, 181 match and 115 changed structure.
   All 360 identities still exist. Resolving references by identity accounts
   for the 18 positional reference misses; 44 token keys and two record
   names remain to classify. The stored naming digest hashes both the name
   and its optional definition row. Names alone give 6ca6ae4d…52cdb;
   the export reproduces f3510fb8…ef7be5d. Nine keyless definitions use
   registry names, as the semantic pack does. The historical exporter command
   and the committed catalog are verified against the writer engine.
   The native table accounts for all 46 missing keys: `opponent_picks` →
   `picks`, 43 retired token keys and two retired record names. The model
   interface admits the unmodified u1952 file. Its 360 trained-card vectors
   match an independent GRU computation from the original tensors within
   1.79e-7 maximum absolute error. The bind finds 136 changed programs,
   initializes 396 token rows and 103 name rows, and has zero unexplained keys.
   On mini-heart.local, seed 353 sampled pairs from all 134 training lists:
   30/30 terminated, zero truncations or refusals, 4,346 decisions in 214.2 s.
   Original PyTorch → safetensors → MLX also served a v9 decision on this
   checkout. Re-exporting the bound container preserved every Torch tensor.
   Slice 3 below records strength on the original and new-card decks.
   General appended-column projection beyond the two player buffers and catalog
   display of tensor-level refusals were completed in slice 4.

   Fact and effect records under retired names start unseen. Slice 3 measures
   overall play strength; it cannot isolate the cost of a single row.
2. **Pinned pilots (complete).** Width-eight v9 and v11 pilots were saved
   by the ab2b757d9 and 44f76fe987 writers. Padded catalog buffers put the
   originals at 2.1 and 3.0 MiB (172 and 202 KiB as compressed Git objects).
   Both replay 71 decisions. The optional original u1952 file replays 53.
   All 360 bound v9 vectors matched the historical writer at zero maximum
   absolute difference. Removing the first retired record alias refused key
   5779523629; the historical name dump identifies it as
   `execution/6/land choices remaining`. Renaming the running fact name
   `flying` to `flight` refused key 6371843220; adding that successor restored
   the full v9 replay. Inserting a named column at index
   72 failed v9's recorded player prefix. Three replay cases passed locally.
   The existing Integration model job collects the test. Cross-platform
   agreement awaits CI.
3. **recheck and the proof run (complete).** References bind the unmodified
   u25, u800 and u1952 bytes, the exact 16 historical deck pairs, the first-64
   heuristic and the unrounded intervals. All three checkpoints passed ordinary
   admission. On the mini, u1952 scored 87/128 (0.680, 95% interval
   0.586–0.758) against the recorded 0.719 (0.609–0.813). Original deal seeds
   gave 42/64, additional deals 45/64. The first eight eligible trophy decks
   with a new main-deck card gave 23/32 (0.719, 0.500–0.938). All 160 games
   finished and passed replay with zero integrity counts. USD 0. The opponent
   changed; attribution to the pilot alone is impossible. Report and receipts:
   `experiments/reports/best-player-on-main.md`.
4. **New checkpoints and docs (complete).** New files carry the writer catalog,
   ordered column names and source. Setup-bound files retain strict admission.
   The save/load regression uses a writer digest with no catalog on disk.
   A wider engine reproduces the original outputs and re-exports the same
   prefixes. Torch, MLX and Academy use the shared tensor admission. Both file
   formats expose the same unexplained-key refusal in Academy. The two small
   pilots and the original u1952 replayed locally. Linux verification is at CI.

5. **The load report is a record (built).** The candidate's admitted checkpoint
   carries its catalog binding report through local preparation to recheck.
   The INFO log and CLI show counts. `--out` retains the full report in
   `binding.json`. Academy continues to show admission refusals.
   The CLI fixture admitted the pinned v9 pilot in a child and retained
   136 changed programs out of 360, including `tla.aang_swift_savior`.
   The v11 fixture produced no binding line. Saving and reloading a rebound
   container preserved action probabilities exactly on the fixture decision.
   Both targeted test files passed locally (9 tests).
   The next step is human review of PR #558.

## Done when

- `load_checkpoint` on main returns the best player from its original file,
  and it finishes full games on its training decks and on decks with cards it
  never saw.
- The report states its measured strength on main beside 0.72.
- A PR that renames a token the runtime emits, drops or moves a column a
  pilot reads, or changes what a pilot computes fails CI until it adds the
  rename or declares the break, for a pilot saved the way the best player
  was.
- One page says what a world, a format, the model interface and an execution
  record are, and which changes refuse a model.

## Follow-up: MTG-386, start new experiments from the best player

Filed separately; not part of this run. Grow (explicit operation writing a new
checkpoint with lineage), per-seat interface, teacher, the warm-start versus
from-scratch experiment (cap USD 25), and the deckbuilding run using the best
player as pilot.

## Docs written first (2026-11-17)

The docs below describe the intended behaviour and are the reference for the
build; slice 4 reconciles them with what was built (report wording, counts,
record field names).

- `docs/observation-and-model.md`: "Running a checkpoint on a later revision"
  (load report, what a checkpoint is bound to, the terms, rename table, writer
  catalogs); superseded binding lines in Policy inputs, decklists and naming.
- `docs/evaluation.md`: "Rechecking a recorded strength".
- `docs/development.md`: "Pinned pilots"; validation and release lines.
- `docs/learning.md`, `docs/training.md`, `docs/engine.md`, `docs/academy.md`,
  `README.md`, `manabot/README.md`, `WORLDS.md`, `AGENTS.md`: one statement
  each brought into line.

Choices the writing settled are 14-19 in `questions.md`.
