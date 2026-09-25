# Curated rules evidence

Check the executable semantic source and installed pack manifest before using
historical wave prose as an exact setup. At ETU-75 kickoff (2026-09-24), the
selected UR list contains 41 cards and GW contains 40; both are sometimes
described historically as 40-card decks. Do not rebalance them incidentally.

Resolve named setups within their compiled pack: the selected matchup and
Jeong share a UR deck name but intentionally have different sideboards. Carry
both lists through seat reversal and recorded-root reconstruction. Reconnect
retains the live engine; restoring a stored root must use its full setup, not
infer sideboards from a matching main deck. Versioned presentation manifests
are immutable; add the next version and check its setup against native exports.
Keep sideboard maps local to manifest validation; runtime callers resolve them
through `CuratedPack.player_config`, so the pack needs no extra stored copies.

`Env.reset(options={"match": ...})` retains the replacement for later resets.
Seat-balanced callers must pass the intended Match on both seats; omitting
options does not restore the original setup. Check actual engine configs as
well as recorded seat labels when evaluating those callers.

When a rules change reveals a card entering a hidden zone, inspect both
`Game::determinize` and `PossibleWorldSpace` as consumers of that information.
An event rendered correctly in Etude does not establish search correctness.
Event subject IDs have typed domains: a Definition subject already names its
definition and must bypass object-to-definition lookup. Numeric collisions
between card objects and definitions otherwise produce plausible wrong evidence.
Zero fallback counters alone do not prove complete consumer meaning; inspect
actual offer labels, previews, and reveal metadata through the session path.

Do not assume `journal_player` snapshots every field of `Player`: its
`PlayerVitals` intentionally captures only mutable scalar state. New mutable
collections need an explicit rollback strategy and state-witness coverage.

Changed effect meaning requires compatible evidence and checkpoint identity,
even if an opcode or tensor shape stays the same. Preserve frozen receipts;
use a newly registered cohort for the corrected world.

A native JSON frame parsing in Python does not by itself prove contract
compatibility. Decision frames, Observations and receipts must reject an old
schema; frames and receipts must validate their public commitments even when
consumers retain dictionary projections. Exercise old-version rejection,
commitment/history consumers and real session setup separately during a migration.
Possible-world reader fixtures must carry explicit known-hand counts, including
an empty map; omitting that field must not silently erase public information.
Keep public-commitment value validation in its typed constructor. Payload
readers additionally enforce canonical field sets; they need not repeat the
constructor's card-name checks.

For choice rollback proofs, use normal Commands or branch-driver `apply`.
Direct resolution helpers bypass the transition-queue journaling boundary and
cannot establish the public transaction contract.

Possible-world enumeration order is part of the projected index and identity
contract. When refactoring it, compare the complete ordered projection as well
as support sizes and total weights; equal totals alone do not prove parity.

Distinguish generated coverage metadata from frozen conformance evidence.
Inspect stale coverage output before refreshing it: a source/artifact hash
update does not certify changed rules or authorize rewriting historical tapes.

After a shared contract migration, run the consuming belief suite as well as
new Learn tests. Constructor validation can move rejection before history or
tracker code. Encoding receipts bind world/content identity as well as numeric
arrays: retain historical expectations separately, reject old bindings, and
check the numerical projection before pinning current unit-test receipts.

Sideboard projection checks should compare fixed-viewer native JSON with the
Python observation JSON, including known-hand and public remaining-definition
counts. Keep private outside candidates separate from game-zone cards and
prove their programs are available before Learn. A successful raw projection
is not evidence that bounded tensors or the configured policy use those fields.
At a real Learn decision, inspect the action-kind bits and candidate-to-focus
rows from both encoders: matching tensors can agree on a missing feature. A
larger action capacity does not repair missing action kinds or outside objects.
When replacing encoder clipping with capacity errors, migrate existing
truncation-warning tests as well as adding Learn checks. For structural encoder
reductions, compare complete tensors, dtypes and object-index maps over both
viewers; matching action counts alone misses padding and focus regressions.

For viewer DTOs whose fields already match their JSON representation, derive
serialization instead of repeating field lists. Keep explicit mappings for
numeric enums and normalized values. Compare exact observation bytes across
the legal prefix and choice outcomes before accepting serializer reductions;
native/Python agreement alone can miss a shared serialization regression.

- Additive product projection fields must omit empty defaults when serializing
  old frames; inserting empty maps into retained frames changes their canonical
  replay addresses. This preserves historical bytes, not rules compatibility.
- Projection changes also affect schemas that embed ExperienceFrame: canonical
  replay, Study artifact/index/recorded decisions, and advice. Regenerate each
  authority's schema and extend each viewer-safe reader's privacy checks for new
  private fields. Live redaction alone does not protect imported artifacts.
  See `protocol/README.md` for the complete regeneration commands.
- A demo that replays normal GameSession Commands must return an initial recovery
  envelope at its final decision. The last Command update alone is not a valid
  new-game response on the browser transport. Verify the actual URL/browser path.
- Pack migrations must update both the shell wrapper and `scripts/play.py`.
  Check the launcher's actual installed manifest against the server pack, then
  exercise `./scripts/play` itself: parser tests and browser harnesses that start
  servers directly do not cover launcher admission. Include new interaction
  suites explicitly in `playwright.release.config.ts` when its selection is
  narrower than the development configuration.

Preserve human acceptance with its exact scope in durable repository records.
ETU-75's “works. approved” accepts the working Learn interaction, without
supplying per-attempt tapes or certifying policy/checkpoint/replay requirements.
Do not let earlier pending-human notes reopen an accepted gate. The curated
decision and evidence boundary is in `docs/rules/learn-lesson.md`; keep detailed
scratch while the same Task still owns unfinished implementation.
