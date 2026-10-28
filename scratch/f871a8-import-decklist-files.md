# PR D: import decklist files into shared human/bot play

## Current direction: training deck collections — 2026-10-28

Jack Heart accepted Academy option 1a with “a”, then clarified the priority:
“im less worried about academy and more worried about setting up training to be
able to generate random matches between differenet decks (to start, taken from
17lands after we finish all of TLA)”. Training setup is the primary outcome;
Academy import is a secondary consumer. Do not launch the Academy redesign as
the next main implementation effort. Its accepted per-deck-source direction
remains useful, but it must not displace the training path.

Accepted intent: use complete externally supplied decklists to generate varied
training matchups. Initial real-world deck source is 17Lands after full TLA
support. This does not authorize a training campaign, downloading a large
dataset now, another card-rules registry, arbitrary card-pile generation, or
changing MTG-175's reserved curriculum experiments. The existing no-spend,
no-training, one-cohesive-PR-for-review boundary remains.

### Experience and proposed usage

Prepare a local deck collection once, validate it against an explicitly selected
qualified world/content pool, then reference the resulting versioned collection
from the ordinary training configuration. At each new game, choose two complete
decks and their play/draw seats with reproducible randomness. Invalid decks fail
collection preparation with their source and line diagnostics intact; do not
silently discard cards or silently train on a filtered subset. Once TLA support
is complete, an offline 17Lands adapter feeds the same admission path. Preserve
source records even when multiple records describe the same normalized deck.

Proposed configuration sketch (NOT an available API):

```yaml
deck_corpus: artifacts/tla-decks/manifest.json
matchup_sampling: uniform-distinct
seed: 17402
```

The first plumbing proof can use several fully eligible supported decks now.
It must run through the actual training collector's reset path, not just the
simulation CLI or two scalar games. No policy training or random-game search is
needed to establish deterministic root selection, replay and rejection.

### Concepts and ownership

- **Content pool:** which definitions and programs the engine admits, with exact
  world/content/learning identity. Owned by content Task 892c47 and later TLA
  batches. This task consumes it.
- **Deck collection:** admitted complete main/sideboard lists, stable normalized
  deck identities, retained source references and a digest of the membership.
  File parsing is an input adapter, not the training distribution itself.
- **Matchup sampler:** chooses deck IDs and ordered seats per native game root
  from an explicit distribution. Record its version, corpus digest and seed.
  Do not allocate all N-squared pairs just to sample two decks.

Observed code already owns per-game native root sampling and replayable seeds:
`VectorEnv.set_uniform_roster`, `current_matchup_indices`, and reset handling in
`managym/src/agent/vector_env.rs`; `SeatRoutedCollector` wires it into training.
The restriction is explicit: `elemental_roster` admits exactly ten authored
Elemental setups; `validate_regime` also pins the Elemental policy input.
Generalizing only the parser or only `MatchConfig` therefore does not deliver
this outcome. A new corpus-backed training path must bind the whole selected
content vocabulary once, not reconstruct or resize policy inputs for each pair.
Existing frozen Elemental/curriculum contracts and receipts keep their meaning.

MTG-175 / PR383 owns vector curriculum changes and its experimental distributions.
Coordinate with that current branch before touching vector environment/bindings
or collector integration. Reuse its generic root/reset/sampler authority where
available; do not build a competing sampler. PR389 / Task 19d240 owns shared
simulation/evaluation changes. Its accepted interfaces remain dependencies.

### Accepted distribution and next proof

Jack Heart selected **2a**: equal probability for each distinct normalized
decklist. He expects nearly all imported lists to be unique and anticipates
curriculum learning over decklists later, but explicitly not initially.
The initial implementation therefore uses uniform sampling, without adaptive
weights, performance-based selection or a curriculum experiment. Retain stable
deck IDs, corpus membership/provenance and the sampling policy's identity so a
future curriculum can extend the same native root path. Do not build a general
curriculum framework now or change MTG-175's existing experimental policies.

This resolves the distribution question and authorizes the next training-focused
implementation pass under the existing implement/test/publish-for-review brief.
Preserve the file importer already implemented; the new work is corpus-backed
training setup, not a repeat of the parser implementation or an Academy-first
redesign. Real 17Lands ingestion remains sequenced after full TLA support.

Working proposal samples distinct deck identities for the two seats, following
the request for different decks; a mirror policy remains explicit configuration,
not an accidental effect of independently drawing two indices. Play/draw is
unbiased and reproducible. This sampling detail is proposed, not a claim that
the user selected all statistical choices.

Acceptance should cover three or more admitted non-preset decks; varying native
roots across resets; identical pair/seat/seed sequence after recovery; invalid
or unsupported input rejected before collection starts; corpus/content/schema
mismatch rejected on resume; actual sampled deck IDs in failure/capture records;
and a single compatible policy vocabulary across all decks. Record game starts
and transitions separately: uniform game starts do not guarantee uniform learner
transition exposure. A bounded fixture proof is sufficient before full-TLA data
ingestion becomes possible. Real 17Lands acquisition/schema mapping and held-out
corpus policy remain later design work, not claims about this importer today.

## Historical Academy concept review — 2026-10-28

Jack Heart invoked concept-review after asking for progress, accepted 1a, then
made training the primary concern as recorded above. This section retains the
Academy rationale; it does not select the next implementation effort.
Implementation and compression finished. The stopped sync was recovered with a
checkpoint and successful `lf sync --manual`; final gate and publication remain.
Content checkpoint 1e0d3c998 is integrated. Recorded focused validation is 50
Python tests, three Rust admission tests, frontend check/build and one browser
import/reconnect proof. No PR is published. Copied research remains local.

Accepted requirements remain: import valid novel decks into Academy and manabot
through one native admission contract; preserve editable source and line errors;
use the qualified pool and named development ruleset; reject unsupported content
and incompatible checkpoints; preserve viewer safety, Commands, reconnect,
replay/fork identity. No tournament-legality claim, sideboarding UI, training,
automatic merge or broader TLA implementation is added by this review.

**Mismatch:** Academy has two simultaneous matchup descriptions. `DeckSelector`
still edits `gameStore.decks` while enabled `deckImport` overrides both seats in
the server. A successful deck check says Ready without checking the selected
opponent; a checkpoint choice then fails on New Game. Importing one person's
deck requires preparing the other person's file too. These are consequences of
treating import as a separate whole-match mode, not a deck source.

**Accepted 1a (secondary priority):** each visible deck slot selects a preset or an
imported file. An explicit “Use the same list” opponent option can copy an
admitted list into the other slot; it is never silently selected. One visible
match draft is the source of New Game. Presets mixed with imports must satisfy
the same pool's eligibility; existing curated permission is not sufficient.
Until such presets exist, import a second file or explicitly choose the mirror.
No saved deck library, deckbuilder or automatic card replacement is proposed.

**Alternative 1b:** retain a dedicated two-file matchup mode. Hide the ordinary
deck selectors while it is active and label it “Import a matchup.” Require two
files as today. This is narrower work but makes import an advanced setup tool.

Both choices must expose opponent compatibility before claiming the match can
start, retain a previously selected incompatible opponent with an explanation
rather than silently switching it, and leave the current game intact on failure.
Both keep source/editor state after failed checks. Deck validation alone should
say “Decks valid”; “Ready to play” includes opponent compatibility. World hashes
remain recorded and inspectable, without making a raw pack key required setup UI.

### Proposed usage documentation (1a; not implemented)

Choose **Your deck → Import file** and select `my-deck.txt`. Correct errors in
the source editor; for example, “Your deck, line 3: quantity must be positive”
keeps the filename, comments and other lines. Choose the opponent's deck by
importing another file or explicitly choosing **Use the same list**. Choose a
built-in opponent. The setup summary shows both actual decks and whether that
opponent can play them. **New Game** rechecks the setup and starts it; a failed
check keeps the current game and editable draft. Current imports use development
construction rules and the qualified supported pool, not tournament legality.
Checkpoint opponents remain unavailable for imported setups until compatibility
admission exists. Reconnect restores the game's pending decision; replay uses
the admitted setup identity, not the current editor contents.

### Consequences and next proof (pending choice)

For 1a, model one `MatchDraft` with two `DeckSource` values (preset or imported
source) and an opponent selection; remove the competing deckImport override.
Mirror is an explicit draft operation that copies a deck, not a new game rule.
Validation resolves sources and checks match/opponent compatibility; New Game
revalidates atomically and creates the existing `ImportedMatch`/native setup.
Keep source text, normalized entries and immutable content identity separate;
do not persist the whole content catalog or add another rules registry. CLI
file flags and native admission stay owned by manabot/managym. Existing preset
only matches retain their admitted setup identity. Private imported provenance
stays out of public replay/viewer projections.

Counterexample: importing a qualified deck next to an old curated preset does
not make that preset eligible for the new pool. Show an actionable incompatibility
instead of falling back. Likewise a previously selected checkpoint must not
become a different opponent automatically. Proof should cover a single-file
explicit mirror, two distinct imports, invalid line correction, incompatible
presets/opponents, and no live-game replacement on failure, then shared identity,
Command, reconnect, replay and fork checks. The mirror proof is proposed, not a
claim about the current implementation.

Resolved: Jack Heart selected **1a**. The subsequent training-first direction
governs the next work. Working screen: `scratch/concept-review.html`.

Accepted by Jack Heart on 2026-10-28 as the separate parallel decklist PR within an eight-PR TLA delivery. Automatically implement, test, refine and publish one cohesive PR, then stop for Jack Heart's review. No automatic merge, paid compute, training or follow-up launches. Source exploration: loopflow://manabot/task/0d9ba3. This brief governs implementation; copied research is evidence, not a competing research-only instruction.

Outcome: a person can supply a decklist file and use a valid novel deck combination with Academy and manabot through one native setup/admission contract. Begin with a qualified supported pool; expanding TLA support expands eligibility without changing the parser. Curated-only matchup matching is not an acceptable final path. A parser disconnected from playable setup is not completion.

Define a documented, small text dialect supporting counts/card names and main/sideboard sections, retaining original source and line references. Normalize into named typed structures, resolve against the selected versioned pool, and return actionable line-level unknown/ambiguous/unsupported/count/format issues without silently substituting cards or discarding source. Card legality and format validation are shared domain behavior, never browser-only checks. No network card lookup at runtime. Both consumers use the same admitted setup/content identity. Preserve source on errors so correction is possible.

Inspect existing deck construction rules and expose them as an explicit named, versioned ruleset. Specific tournament-format selection is not settled: do not assert Standard legality or silently invent tournament restrictions. Use an explicitly labeled development format based on established repository rules for the initial end-to-end proof; isolate construction policy so review can select a product format without changing parsing or native rules. Main/sideboard parsing does not authorize a sideboarding UI, retail draft/sealed collation, Commander or multiplayer. Deliver a concise Academy file-import flow and ordinary manabot configuration/CLI entry point.

Parallel PR1 owns content authoring, semantic IR/runtime lowering, definition eligibility and content identity. This Task owns parsing, normalized deck models, shared pool/format admission, and setup entry points. Consume a small typed interface of resolved definition IDs, versioned content/world identity and eligibility; do not build a second rules registry or rewrite cardsets. Agree contract details early and integrate current reviewed upstream changes; initial fixtures can use the existing qualified supported pool. Admission must reject missing programs/faces/tokens and unsupported cards rather than falling back to native generic definitions. Existing curated setups remain presets.

Proof: import a valid deck combination absent from authored setups and start it through both consumers; show the same resolved identities and legal action authority. Exercise malformed lines, bad quantities, ambiguous/unknown names, known unsupported TLA cards, sideboard/format violations, and retained editable source. Preserve Command/reconnect/replay/fork binding and viewer safety. No policy training is necessary. Keep model/checkpoint compatibility explicit; do not silently accept a checkpoint incompatible with the chosen content/observation schema.

Overlap check 2026-10-28: PR389, Task 19d240, rewrites shared evaluation and simulation consumers including manabot/sim/native_game.py, match.py, serving.py, semantic/policy.py and Academy watch/villain. PR383, MTG-175, owns adaptive curriculum changes in vector_env.rs, vector_env_bindings.rs and sim/rollout.py; its reserved curriculum decisions stay there. Recheck these before editing overlapping entry points. MTG-177 owns target-resolution changes; this importer must not repair those independently.

Use repository conventions and validation.md. All Python through uv; debug Rust with two Cargo jobs and atomic extension replacement where needed; frontend generation/check/build before browser proofs. No artificial load, random search, heavy benchmarks, paid services, frozen-evidence rewrites or new training. Include a simple real import demo and focused independent error/admission tests in the final review.

## Implementation — 2026-10-28

Content contract coordinated with Task 892c47: native `content_pool(pack_key)`
provides world/content/compiled identity and eligible/unsupported definition rows.
That Task owns eligibility and closure; importer owns setup admission and parsing.
PR389 and PR383 remain open; no edits planned in their reserved files.

Delete — do not maintain: replace explicit-pack authored-setup-only selection
in `semantic/content.rs` with shared qualified pool admission. Preserve implicit
preset selection and checkpoint exact-world binding. No card authoring deletions
belong to this Task. Remove unused `env.match.parse_deck` and its exclusive
`TestDeckParsing`/`deck_formats` fixture; production call-site search found none.
Existing structured native fixture configurations remain; file imports always
carry explicit pool admission and never use their generic registry path.

Implemented: typed counted-lines-v1 parser, source/digests/line diagnostics,
front/full DFC identity, native development-v1 construction/eligibility, retained
match identity, sim file options, Academy editable import and validation endpoint,
private trace provenance, replay/rematch binding and focused tests.

Remaining (gate/demo owners): complete changed-area suites and browser proof
`decklist-import.spec.ts`; revalidate against final upstream content delivery;
publish only the importer diff against its content dependency, then review.
No automatic merge. No importer code gap is currently known.

Check result — focused debug native admission/catalog tests (7), Python startup/
identity/reconnect/fork/stale/source/DFC checks, TypeScript check + focused store
test, frontend build, CI selection (40) and combined test collection passed.
The documented CLI ran one fixed-seed, four-decision local startup proof, admitted
setup 0471df58fab1310ced94add4190c76509abb878f8df191c601c12bcafc8fe3fa;
its configured step limit deliberately ended the game unfinished. Native extension
rebuilt in release and replaced atomically. Full gate suites/browser remain deferred.

Historical full-TLA/audit notes and duplicate handoff preserved in
`.lf/tmp/decklist-evidence/decklist-research-20261028.tar.gz`; source exploration 0d9ba3,
base c5d406e3d2a701525c15c8500a30f7da7896cc40.

Final research 0d9ba3 reviewed 2026-10-28; full source-preserving copy is at
`.lf/tmp/decklist-evidence/full-tla-final/` (plan digest
7cdb6308f8750ec6d1e30ca485e1676c53af5456401fffa3cb88f10c496c07ef).
Its D1 contract confirms one parser, native shared admission, source correction
and explicit development policy. Imported source text preserves UTF-8 bytes,
including CRLF/BOM; raw legacy structured fixture configs remain unchanged.

Integrated content dependency 73865361f through `lf sync` (merge 2806d03d5).
It provides `tla-pool-v1`, world w16 and conservative eligibility; this is a
dependency checkpoint, not final upstream qualification/review. Importer PR must
be reviewed against the content branch (or revalidated after it lands), not
present its 81k generated content lines as importer-owned work.
