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
