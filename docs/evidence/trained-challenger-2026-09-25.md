# Local challenger and shared history — 2026-09-25

This record preserves accepted decisions and dated branch evidence for
[Challenge an identified trained bot through complete games · ETU-79](https://linear.app/loopflow/issue/ETU-79/challenge-an-identified-trained-bot-through-complete-games).
It is not Task completion, a strength result or deployment approval. The
[operator guide](../local-trained-challenger.md) owns the repeatable commands.
Detailed working notes remain in `scratch/`; local `.runs/` paths below locate
original evidence but are not portable artifacts guaranteed by a repository clone.
This curation did not rerun training, browser checks or the affected suite.

## Binding decisions

The User wants to train the best-supported existing recipe now, play its exact
output through the normal product and record how it feels. Initial compute is
local on the M4 Max MacBook, with nearly zero new spend. ETU-79 supersedes ETU-78's
separate pipeline work; ETU-80 consumes the same receipts. Do not create another
runner, evaluator or receipt store, or revive deferred ETU-31 experiments.

The built Svelte SPA and `deploy.play.asgi:app` are the local product proof path.
Server configuration supplies the opponent; the player sees name, fingerprint
and availability. Search, Random and Passive remain named controls. Checkpoint
bytes, world and inference configuration stay pinned through a match and rematch.
Invalid replacement preserves the current game. Omitted seeds generate a fresh
deal; requested unsigned 64-bit seeds reproduce both environment and policy.

The User extended the design to a global game log “across everyone,” inspired by
17Lands, with both players identified as Human/Bot and easy search/filtering.
My games narrows the same shared history. Two seat snapshots retain stable public
IDs, recorded names, decks and exact bot versions/configurations. Renames and
duplicate labels do not merge identities or rewrite history. Bot lineage is
declared independently from producer and loader location; absent lineage stays
exact-digest only. This does not add accounts, new match modes or a registry.

GameSession and AttemptStore own one SQLite attempt/Trace transaction, including
seat metadata. Legacy JSON remains readable without a second writer. Retain
active, stopped, interrupted and completed attempts with their real ending
reasons. Disconnect is not abandonment; restart does not invent a result.
Recording failure must be visible, including when cleanup also fails.

New attempts are shared listings; completed shared replay uses the existing
seat-0 viewer projection and authoritative terminal reveal guard. Unfinished
prefixes and feedback stay private. Existing private SQLite history migrates
privately with original trace JSON unchanged. Legacy JSON has unknown player
identity. Listing permission is applied before SQL pagination and does not
confer private replay, feedback or Retry permission. Summaries omit private
credentials, seed, checkpoint path and mixed-view canonical evidence.

Future graphs and training exports read these same schema-versioned records,
with actual world/source, origin and canonical decisions. Origin distinguishes
automated browser validation from human exploration; a Human seat is insufficient.
Missing provenance stays unknown. Any future dataset manifest must pin attempt
IDs and revisions or content digests, selection query, schema/world versions,
viewer projection and transformation version. Pin or exclude active prefixes;
separate decision-time information from terminal reveal/outcomes and use
whole-game splits/deduplication. No dataset exporter or graph platform is built.

## Retained execution evidence

The existing search-64 behavior-cloning recipe is supported by historical Exp-03;
Exp-07 cautions against weakening its teacher. Neither ranks the tiny current
candidate as the strongest bot. The generic local PPO smoke uses another matchup.
The chosen recipe uses eight teacher games with alternating assignments, eight
epochs, learning rate 0.001, batch 64 and whole-game 20% validation. Select the
final epoch without a sweep. Fixed observation bounds reject overflow rather
than dropping choices. Each run is capped at 600 seconds, 4 GiB process-tree RSS
and one CPU worker/thread on the 128 GiB M4 Max.

| Original run directory under `.runs/` | Result | Worker wall seconds | Peak RSS bytes | Checkpoint SHA-256 |
| --- | --- | ---: | ---: | --- |
| `etu79-challenger-1` | 24-game attempt timed out; no promoted model | 600.177 | 1718779904 | none |
| `etu79-challenger-2` | Eight games, seed 79, complete | 130.496 | 1502216192 | `cc4e6f99149fbf7eb2f0ef0a2272c2eca9ee370d61cf42f8589f3ea56102ef18` |
| `etu79-challenger-3` | Eight games, seed 80, complete | 123.462 | 1491238912 | `269fc159b14b24790a896dee1f6d7192d1c3b8d00740e8e1e4c238f9d3c9f3ef` |

All runs used the pre-ETU-75 world. Actual new cloud spend was $0; electricity
cost is unknown. Seed 80 overlapped browser work, so timings are workflow costs,
not controlled performance comparisons. Run 1 lacks phase timings. Completed
runs preceded exact `sources.zip` capture and retain digests only; do not imply
their dirty source bytes were archived. The final recorder's archive and budget
failure checks passed, but completed corrected-world repetitions remain pending.

The 2026-09-25 review served the built SPA through a fresh hosted ASGI process on
isolated port 18083 using seed-79 bytes and `automated_validation` origin. After
repairing generated-seed policy drift, both browser specs passed in 15.9 seconds.
They exercised both assignments, keyboard play/search, reconnect/rematch, two
named players, shared history/pairing, private notes, incomplete replay privacy,
public completed replay and preserved filter state. The isolated server stopped
after proof; this is not a current preview availability claim.

Final trained attempts in `.runs/etu79-review-history-records/play.sqlite`:

| Attempt | Hero deck | Revision | Winner seat |
| --- | --- | ---: | ---: |
| `8Us7lJmLGVtrqaxoADlvCg` | UR Lessons | 146 | 1 |
| `3JhsbtO1QZXnUt9zX9cflw` | GW Allies | 63 | 0 |

All six completed validation games reconstructed every observation and terminal
observation from retained seed, deck and legal action indices without policies.
Restart preserved records and changed an active prefix to interrupted without
winner or invented end time. Witnesses and logs are under
`.runs/etu79-review-history-evidence/`; SQLite remains authoritative.

The affected backend run reported **289 passed, 5 failed**. It imported source
before the seed repair; the post-repair focused run reported **24 passed** and
the restarted browser checks passed. Frontend reported **92 unit tests passed**,
zero type-check errors/warnings and a successful static build. The five remaining
checkpoint-advice failures concern fixture/request identity, historical
`max_conditions` loader compatibility and parity-receipt digest drift. Frozen
evidence was not changed. Optional dependency imports and `ed2` Study-address
validation were repaired separately; `ed2` remains pre-command, with independent
landmark offer/Command checks and retained `erd1` validation.

## Human evidence and next acceptance

On 2026-09-24 the User said “game is good, but no lesson select,” “demo is playing
fine,” and “transitions need work.” Attempt `9-QZ6XaIIk0wgJUpH_PPfw` was observed
active at revision 39, UR Lessons versus GW Allies, against seed-79 bytes. Its
last retained Command precedes the Learn prompt; it is not an exact address of
the reported Learn decision. The engine implemented Learn as discard/draw
without a sideboard in this world, consistent with the missing Lesson selection.
The transition report did not identify a specific sequence.

No terminal human outcome, saved in-product note, replay visit or voluntary
rematch was confirmed. The historical Ask ID
`ask_8ca31f31455f4aad96d802ed9d8dc964` later disappeared from session listing and
a ready handoff failed because it no longer existed. Neither observation proves
completion or cancellation. Do not count automated notes or create a duplicate
human request solely from that disappearance.

ETU-79 remains open for corrected-world repetitions with final source capture,
attributable human completion/return and a passing applicable gate. ETU-75 owns
the corrected rules/Lesson pool. Interaction proof builds on
[Make large legal decisions navigable without awkward workarounds · ETU-77](https://linear.app/loopflow/issue/ETU-77/make-large-legal-decisions-navigable-without-awkward-workarounds);
transition repair belongs to
[Make action consequences and the next decision legible · ETU-76](https://linear.app/loopflow/issue/ETU-76/make-action-consequences-and-the-next-decision-legible).
Their existing scopes cover the observed friction; no duplicate Task is needed.
Evidence capture alone is not repair. Scored strength/return claims require an
agreed protocol. Hosted deployment additionally needs persistent storage and a
runtime/candidate bundle; local proof does not authorize it.

`lf status game --json` on 2026-09-25 confirms this checkout's ETU-79 belongs to
Game and all three current KRs remain false. The current chapter is
`20260924-trained-challengers`, internal Project
`b5f82d53-ff4b-409d-96f2-23c5efd8a956`; metric targets are empty and the recommended
Flow is unset. The older Project ID in GOAL is historical chapter metadata.
The mandate, KR wording, targets and Flow are unchanged by this curation; do not
infer outcome acceptance from the status surface's chapter phase `complete`.

The update-wave pass refreshed ETU-79's brief with the accepted shared-history
requirement, this evidence location and the remaining acceptance gaps. A second
status read verified the exact description, open Task and unchanged chapter.
No Task was closed or created. Scratch remains because the detailed design and
unfinished acceptance work still use it. Repository changes are documentation
and review guidance only; new local links and whitespace were checked.

## Gate follow-up — 2026-09-25

Fresh gate evidence is retained under `.runs/etu79-gate/`. The branch remains
unpublished and is not ship-ready: five checkpoint-advice checks still fail,
corrected-world repetition is pending, and no additional human acceptance was
obtained. No frozen checkpoint, fixture or receipt was changed.

The gate rejected `--seconds nan` and `--seconds inf`, which bypassed the
runner's positive-limit check. A regression guarded all subprocess launches:
before repair two cases failed and two passed; afterward all four pass, rejecting
invalid limits before output creation. No training worker ran for this test.
The human launch example now declares `human_exploratory` origin explicitly.
Ruff reformatted five branch files; AST comparison confirmed no behavior change.

Current checks:

- `uv run pytest tests/etude tests/env tests/sim/test_distill.py -q`:
  **331 passed, 5 failed**, 185.30 seconds.
- Encoder consumers (`tests/agent`, `tests/model`, distill datagen, structured
  policy and benchmark contracts): **43 passed**, 6.09 seconds.
- Runner budget admission: **4 passed**. Frontend: **92 passed**, zero Svelte
  errors/warnings, successful static build. Affected Ruff lint/format and
  whitespace checks passed.
- The same five advice failures concern frozen request/receipt identity and
  the historical checkpoint's rejected `max_conditions` field. Relevant failing
  tests, loader, hypers, fixture generator and identity source are unchanged
  against main; this gate did not run a separate baseline checkout. Frozen
  evidence reconciliation remains a gate blocker, not permission to refresh it.

Built SPA/ASGI proof used isolated port 18085 and
`.runs/etu79-gate/records/play.sqlite` with `automated_validation` origin and the
original seed-79 bytes. Both browser specs passed in **13.4 seconds**, exercising
keyboard play, both assignments, reconnect/rematch, two identities, shared
history, private notes/prefixes, public completed replay and URL-preserved return.

| Attempt | State | Transitions | Winner seat |
| --- | --- | ---: | ---: |
| `MEP45hGkXR_Gi1ouydoa-g` | Trained bot, UR Lessons hero, completed | 122 | 1 |
| `mCl9BHOnHIcezI8nT3f1jA` | Trained bot, GW Allies hero, completed | 49 | 0 |
| `rvDyxtv2pYcVndIS4IlFGg` | Passive control, completed | 45 | 0 |
| `39kky_IDLkDOH1Mape3X_A` | Passive control, active prefix at inspection | 0 | unknown |

Read-only SQL confirmed four attempts and three linked notes. Independent
reconstruction of all three completed games from stored seeds/decks/actions
matched every recorded observation and terminal observation with legal choices.
Witnesses are in `replay-witnesses.json`; SQLite remains authority. The isolated
server was stopped after proof. Its incomplete prefix is retained and will be
marked interrupted on a subsequent server open. No human store was touched.

Accessible DOM snapshots cover populated history, empty results, a simulated
HTTP 503 and successful retry (`history-ui-evidence.json`). This is interaction
proof, not pinned visual certification. With four records on the warm local
server, one navigation reached visible rows in **50.9 ms**; 20 serial public
history reads including JSON parsing measured **1.6 ms median / 2.2 ms p95**.
This is a small local observation, not a before/after improvement or live SLO.

There is no deployed Games performance signal. Proposal for the Game wave:
track history API latency/error rate by row count and filter class, browser
navigation/filter-to-visible-rows latency, and surfaced recording failures.
Exclude names, credentials and search text. Provisional p95 goals of 200 ms
server / 500 ms warm filter interaction need calibration on the hosted workload;
no remote plan or metric target was changed here.

The full CI matrix remains with CI: debug Rust, cross-platform integration,
cold clean-machine startup and pinned Linux visual references. No native code
or launcher ordering changed. All selected checks ran fresh; no cached result
was reused. No new challenger training, paid compute, public deployment or
human completion/return claim was made.
