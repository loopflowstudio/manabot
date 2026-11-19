# MTG-353 — choices made without a person present (kickoff, 2026-11-17)

The exploration questions of 2026-11-17 (mechanism A/B/C, payment level,
per-card rows, row counts, vision C) were settled by the plan Jack Heart
approved; per-seat interfaces and payment level moved to MTG-386.

Made for review in `plan.md`; each is reversible:

1. Pinned pilots are two small committed containers (v9 and v11) written by
   this change, not the 3.9 MB best player. The best player is checked by
   `recheck` and by the same test when its file is present.
2. Pinned-pilot equality is 1e-5 on probabilities plus equal rank of the
   chosen action, not bit identity, so Linux CI and macOS agree.
3. A row for a token the checkpoint never saw starts at the mean of its
   table's learned rows (zero passed over).
4. Renamed token names are carried by a committed rename table in managym; a
   learned key missing from it refuses the load.
5. `possible_world_version` binds only models with belief inputs.
6. Setup-bound checkpoints keep refusing as today.
7. The branch was moved to current main with `lf sync` (it had no commits).

Slice 1 answered the open question: all 46 missing keys are classified in
`managym/src/cards/token_renames.rs` (one rename, 45 retired), and the
unmodified best player loads with trained-card vectors matching the
checkpoint-only computation. Strength on main is slice 3's measurement.

## Concept review, 2026-11-17 (nobody present; each reversible)

8. A card the pilot trained on reads the program stored in its checkpoint;
   cards added since read the running catalog. Passed over: running programs
   for every card with the loss measured. Not yet measured: how many of the
   298 trained definitions differ by rename against restructure; slice 1
   counts it first.
9. The writer catalog is one file per `world_digest`, holding definition keys
   and naming names only. Token names are an authoring dump, not committed.
10. The v9 pinned pilot is a small model saved from the `ab2b757d9` checkout.
11. No `breaks.json`: the reason and commit live in the recording's header.
12. `recheck` is strength only. The approved plan had one throughput
    reference for the mini; it is left out as its own Task. Put it back if
    that was wanted in this PR.
13. Left as it was, and shown on the page as a mismatch: a fixed-match
    (setup-bound) checkpoint still stops loading when cards are added.

## Docs written first, 2026-11-17 (nobody present; each reversible)

14. The load report is one log record from `load_checkpoint` (its return type
    is unchanged), printed in full by `recheck` and summarised per row in
    Academy. Passed over: a new return value or a separate `inspect` command.
15. The rename table is `managym/src/cards/token_renames.rs`, beside the
    serializer in `managym/src/cards/tokens.rs`.
16. Two small commands the refusal messages name, so each refusal carries its
    fix: `scripts/pinned_pilot.py names --at <revision>` (names behind
    unexplained keys) and `python -m manabot.model.writer_catalog` (dump a
    writer catalog at the writing revision).
17. `recheck` exits 1 outside the interval or on an incomplete cohort, takes
    `--out` and `--seconds`, and its reference names a matchups file and deal
    seeds. Field names in the example record are a proposal.
18. `record --reason` rewrites recordings only and cannot pass a pilot the
    loader refuses.
19. v9/v10 removal is restated as "once a v11 player is at least as strong,
    not before 0.4.0" where docs said "removed in 0.4.0".

Open: WORLDS.md's per-world sections (w4-w16) still say earlier checkpoints
"are rejected". They are history of those revisions and were left as written;
say so if they should carry a note.

20. Split the writer catalog and actual program comparison out first. The
    export stores complete naming candidates because names alone cannot
    reproduce the historical checkpoint digest. Keyless rows follow the
    semantic pack's existing registry-name convention.

21. Classified the original `opponent_picks` field as `picks` inside
    `DistinctSearchEnd::OpponentSplits`. Classified the two search labels
    and their record aliases as retired: the new decision carries arbitrary
    object IDs, a predicate and a structured ending. Trigger shorthands and
    ManaUse were also restructured. Their rows remain available to stored
    programs. Validate each architecture receipt on its writer's arrays
    before growing the runtime tables.

22. Keep both checkpoint files byte-for-byte as saved. The native catalog
    buffers dominate the file size even at hidden width eight. Use one SGD
    update on a real opening decision for each fixture. Main's Python writer
    ran with this branch's native extension (main's catalog plus rename export).
    Record u1952 separately and select its optional local bytes by environment.

23. Select updates 25 and 800 as early and middle reference points. Use exact
    unrounded intervals from the retained page. For u1952 double the historical
    cohort to 128 games by adding 16 to each pair's deal. For the new-card arm
    use the first eight eligible trophy lists in source order, paired adjacently
    for 32 games. Compare the heuristic with the first-64 cap in both arms.

24. Academy runs tensor admission once per cached file identity when it lists
    bots. Hashing streams the file and the cached summary retains no weights.

## Concept review of the built change, 2026-11-18 (nobody present; reversible)

25. The load report becomes a value on the bound checkpoint and a
    `binding.json` beside a recheck's measurement; the log line keeps counts
    only. Academy shows refusals only: the report summary on a loaded bot's
    row (choice 14) is passed over as a new UI surface.
26. Saving a carried player keeps the implicit metadata rewrite in
    `write_container`, with a round-trip test; naming that action and its
    lineage is left to MTG-386.

27. A catalog admission with zero changes still writes its complete report
    under `--out` and leaves the console quiet. Registration constructs the
    backend policy from the same admitted tensors that supplied the report,
    avoiding a second file read.

## Compress, 2026-11-18 (nobody present; reversible)

28. `checkpoint_request` stays a wrapper over `local_comparison` with the same
    keywords. Returning the report from it would change
    `experiments/etude_execution.py`, whose source Etude fingerprints.

## Concept review, second built pass, 2026-11-18 (nobody present; reversible)

29. The written/admitted split of `PolicyCheckpoint` (one type, four readers,
    `load_policy` accepting either) is left for MTG-386 with `grow`. Passed
    over: two types in this PR.
30. `docs/academy.md` quotes pre-build refusal examples; left for gate to
    replace with the loader's real sentences.

31. Both reproduction commands explicitly choose their arm. The historical
    arm derives its pairs from the existing frontier protocol and committed
    deck source. A direct comparison matched all 16 previous pair records.

32. New registrations with no binding report count the saved card program
    rows used directly by the loader (zero for a model without card programs).
    A retained registration lacking the count stays unknown. Etude already
    stores these fields in the match participant's registration config.

## Concept review of the revised build, 2026-11-18 (nobody present; reversible)

33. The participant record's writing commit is named `writer_revision`, not
    `checkpoint_source`, which elsewhere names the commit, dirty flag and
    Etude run together. Passed over: storing that whole record.

34. Etude change needed for the corpus query (2026-11-18). Its training reader
    selects logged tables correctly. Its evaluation query database ingests
    training checkpoints but skips training metrics and logged row tables.
    Expose selected training rows by experiment, run and clock coordinate to
    queries, including historical scalar outcome columns, preserving resume
    supersession. A checkpoint's recorded update should bound the selection.
    Manabot can then supply the deck/card join and game-specific queries.
    Decision 35 superseded waiting for it.
    Historical reports, serialized identifiers and UI control IDs retain
    their spelling.

## Realign, 2026-11-18 (nobody present; reversible)

35. The corpus query is built in manabot on `RunData.read_table` and
    `updates_from_run` rather than waiting for Etude's query database to
    ingest training rows. Both already select winning attempts. Passed over:
    holding this PR for an Etude release. Moving the relation upstream later
    changes only where the rows are read.

36. The current local and frontier recipes declare their recipe arguments and
    resolve decks inside Python. Record the resolved content IDs and the union
    of main-deck and sideboard definition keys once at update zero. Keep duplicate
    population entries in the roster and count each content ID once in matchup
    joins. The checkpoint entry point accepts Etude's published CheckpointRecord.
    SDK 0.4.1 selects table artifacts across the supplied history before the
    caller can filter on update. Include a bounded table read in the Etude
    follow-up alongside database ingestion.

## Concept review of the corpus query, 2026-11-18 (nobody present; reversible)

37. Card statistics join the corpus as a `card_statistics` relation on the
    same DuckDB connection, switched on by `statistics=path`, keyed by
    update, seat and definition key. Passed over: a separate function (its
    relation could not be joined), a deck column (rows hold main-deck copies
    only) and locating `out/statistics` from the run (Etude storage does not
    hold it). Fewer terminal statistics games than corpus games at an update
    raises; more is allowed, since superseded attempts leave rows behind.
38. To file after this PR: publish statistics rows as run artifacts with the
    attempt in their context.

39. The local statistics run includes Ally and Clue exposure without deck copies.
    Use the native definition index in the digest-selected catalog for every
    row. Games and wins follow CardPerformance's main-deck denominator, while
    generated cards retain their draws and exposure.

40. Keep the setup tables at update zero on resume. In Etude 0.4.1 the first
    row of a resumed writer cuts off its predecessor at that coordinate. Logging
    setup again at the last completed update could hide that update's outcomes.
    A restart from zero writes the tables again.
41. Ancestry uses the checkpoint registration's run, update and SHA-256. An
    unidentifiable parent leaves ancestry unknown. A single local statistics
    path cannot prove statistics coverage across a chain. Attempt-aware
    statistics publication remains question 38.

42. For historical training settings, report a stage setting only when every
    declared stage records the same value. The settings table has one row per
    run, so a value that varies by stage has no single run-wide value.

## Concept review of the built corpus, 2026-11-18 (nobody present; reversible)

43. `training_corpus` returns the DuckDB connection with `corpus` as a view
    on it. As built it returns the `corpus` relation and the other tables
    are reached through `relation.query("corpus", sql)`. Passed over: a
    wrapper type around the connection, a second function for the side
    tables, and leaving it until after release. `decks` and `matchups` stay
    on the connection undocumented. Statistics across a parent chain stay
    refused until question 38.

44. In the sync onto main, use the same observation capacity switch for
    fresh and rebound decision-memory catalog tables. This keeps the
    log-enabled control and its memory branch on the same learned rows.

45. Compress, 2026-11-18: a run that predates the update-zero deck table now
    supplies `deck_cards` rows too, derived from its declared deck source
    through the writer catalog (the same inputs its roster already used).
    That left one deck path in the reader: `decks` is built from `deck_cards`,
    and `corpus_decks` is gone. Reversible; the guide no longer says older
    runs have no per-board rows.

## Concept review, sixth pass, 2026-11-18 (nobody present; reversible)

46. Remove the Python `roster` reader. After choice 45 every run has
    `deck_cards` rows, so the roster is `deck_cards` grouped by entry; the
    function re-reads the deck table outside the connection and only the
    guide calls it. Passed over: keeping it as a convenience (a second name
    for one fact, and a published function needs a warning period to remove
    later), and documenting the `decks` table in its place (it drops
    duplicate entries).


## Concept review, seventh pass, 2026-11-18 (nobody present; reversible until merge)

47. Corpus settings are two relations split by one rule, whether the value
    can differ between two games of one run: `runs` (one row per run) and
    `game_settings` (update, seat, setting, value, games). Passed over:
    every setting in the long relation (model facts repeated each update
    and typed columns lost), and a column per setting on the outcome rows
    (a reshape for each new setting, #559 included).
48. `decklists` leaves the per-run row. Its only run-wide spelling for a
    varying run was the word `mixed`; the counts replace it. Passed over:
    keeping both (one fact with two owners that can disagree).
49. The per-run relation is named `runs`, not `settings`: it holds the
    parent checkpoint and model facts, and the name `settings` would
    otherwise cover two tables. The logged table becomes `training_run`.
    Unpublished, so no alias.
50. Whole-game settings carry a NULL seat; per-seat settings carry the
    native seat. Passed over: writing a whole-game value once per seat
    (sums to twice the games) and seat-suffixed setting names (a query per
    seat). `seat` follows `card_statistics`; outcome rows keep `position`.
51. Setting counts live in the outcome counter and are logged with the
    outcomes, and the reader refuses an update whose setting counts do not
    sum to its games. Passed over: a second counter drained separately
    (nothing ties its games to the outcome rows across snapshots and
    resumes). Older runs get derived rows only for declared `open` or
    `closed`; `mixed` and `skip_trivial` stay unrecorded.

## Implement, corpus settings, 2026-11-18

52. A run that logged outcome tables but no setting table at any update reads
    as predating the table (no `game_settings` rows, or derived `decklists`
    rows), not as `UnknownCoverage`: the reader cannot tell it from an older
    run. A run that recorded a setting at any update must account for every
    game at every update. Passed over: a writer-version marker in `runs`.
53. `SelfPlayCounter.record` takes settings with a default of none, so tests
    of outcome routing stay as they were; the collector always passes them.

## Implement, human review third round, 2026-11-18

54. Seat spelling (item 4, "c: fix it in this PR"): the column is `seat`, the
    native seat as an integer, 0 on the play, on every per-seat row: stored
    outcome rows (`SelfPlayOutcome.seat` replaces `position`),
    `game_settings`, `card_statistics` and the report's `SelfPlayPoint`
    (`position` stays as a warning alias until 0.4.0). `corpus` keeps `play`,
    `draw`, `play_cards` and `draw_cards`: its row is about a game and names
    both seats, and the guide says `play` is seat 0. Passed over: `deck_0` /
    `deck_1` columns on `corpus` (every guide query rewritten for a row that
    is not about one seat). Reversible until merge.
55. Stored runs keep their bytes. `manabot.reports.update_metrics.outcome_rows`
    is the one reader: scalar counts named `play`/`draw`, tables with
    `position`, and tables with `seat` all read into rows with `seat`. The
    offline projection (`updates_from_run`) keeps the scalar metric names
    `self_play_outcomes/<deck>/<opponent>/<play|draw>/<count>` for every run:
    stored series already carry them and the self-play panels read them.
56. Corpus from a checkpoint file (item 1): `training_corpus(path,
    source=storage)` reads the container header only
    (`manabot.policy.checkpoint.read_metadata`), so the eval library stays
    backend-free; a PyTorch `.pt` file raises `NotAContainer` and is converted
    first. New files record `source.etude_experiment` from `ETUDE_EXPERIMENT`
    beside `etude_run`. An `experiment=` that contradicts the file's record
    raises. Not checked: that the file's bytes match a checkpoint the run
    registered (the raw policy is never registered, only the EMA one).

## 57. QA at 6fa4efee06: three repairs to the file form (reversible)

`training_corpus` accepts the checkpoint file as a string as well as a `Path`
(`source` already did). A storage that holds no record of the selected run
raises `UnknownCoverage: <source> holds no run <experiment>/<run>` instead of
"resolved training decks are unknown". The guide's first example names real
paths (`~/.etude/track/local/out/policies/...`) and its statistics example
uses the same file; `etude_home` was undefined and frontier Small has no
statistics.

## Implement, human review fourth round (remove the world label), 2026-11-18

58. Decision capture: the capture's SHA-256 covers every field, the label
    included, so the field could not be dropped in place. New captures are
    version 6 without `world_version`; `DecisionInput` reads v3-v6 and keeps
    an optional, unread `world_version` so a stored v3-v5 capture validates
    and its digest still recomputes from its own bytes. Reversible.
59. Evaluation records: `PlayerRegistration.world` and `EvaluationKey.world`
    are optional and omitted when absent, so a retained record validates and
    hashes as before and a new one writes none. Consequence, not checked
    against Etude: a registration made by this revision differs from the one a
    retained tournament holds for the same player (no `world`), so a
    tournament resumed across this revision may treat its participants as new.
    Frozen experiment files that passed `world=managym.WORLD_VERSION` pass
    `"w17"` and keep their old records. The same optional form is used for
    `SelfPlay.world` (recorded settings) and `BaselineReport.world`.
60. Other places the label was part of a stored id or digest, kept readable
    without rewriting: `manabot.sim.matchup.SUITE_PREFIX = "w17"` (authored,
    deck and selected suite ids); a card-evaluation `Environment` whose
    `files` hash covered `world` (read back as one more identity field when
    its config has it); a scenario root, game record `world` dict, receipt
    identities and native failure context (the key is skipped); Academy demo
    fixtures' `manifest.world_version` (skipped in the comparison; the
    fixtures were not rewritten, and the next refresh drops it).
61. Not read any more: `CheckpointSummary.world` and Academy's "World" detail
    row are removed, since a current checkpoint has no label to show. The
    browser still prints `provenance.world` of a recorded game when present.
62. `experiments/protocols/mtg186-identity.json`, the conformance manifests and
    every JSON fixture keep the label they were written with.

## Tenth concept review (after the label removal), 2026-11-18

63. Choices made for review, all reversible: (1) the four guide sentences that
    still say a world separates measurements (`README.md` 199, 284, 308;
    `docs/engine.md` 1307) are rewritten by gate, not by another implement
    pass; (2) question 59 is accepted as the ordinary cost of resuming a
    tournament across revisions, since `source_sha256`, `writer_revision` and
    `stored_programs` already change a registration's hash there; not run
    against Etude; (3) the word "world" for the pool, the card identity and
    the old label is left as it is. Passed over: writing `world="w17"` into
    new registrations to keep hashes equal (keeps the label alive).
