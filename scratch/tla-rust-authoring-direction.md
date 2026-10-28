# Rust card authoring — accepted direction, 2026-10-28

Jack Heart clarified in the primary Task conversation that cards should be clear
code in Rust set files, with minimal configuration, rather than the current
semantic JSON authoring arrangement. Separate scripts directories may help build
and generate files; generated files should be checked in.

Implementation direction: make typed Rust set definitions the single authored
card authority. Generate the representations needed by learning and Academy from
that authority, retain generated artifacts in git, and check their freshness.
Remove the competing manually authored semantic definition/configuration path
instead of adding another source. Keep ordinary decklist data separate from card
behavior. Preserve the independent rules witnesses, explicit unsupported-card
eligibility, shared pool interface, historical evidence and one native runtime.

This supersedes the JSON-as-authority implementation choice in the earlier
handoff. It does not change the one-PR publication/demo boundary, authorize merge,
or transfer decklist/admission ownership from f871a8. Reconcile the current design
and implementation before publication; do not publish the superseded authoring
architecture as the accepted result.

## Recovery and delivered implementation

The preceding Graph stopped at automatic sync before receiving this direction.
The installed sync checkpoint failed on its ignored `.lf/tmp` exclusion pathspec.
The operator preserved the cleanup and direction in commit `080289907`, completed
`lf sync --manual` and `lf sync --continue`, retained world w16, and preserved both
scratch histories. Main's resolution repair `7d7736c6a` is now integrated locally;
the Rust revision reassessed Allies at Last with independent directed witnesses.
Checks predating that integration remain historical evidence.

Implementation and compression are complete at `0a0d22577`; manual sync succeeded.
The 2026-10-28 Task status reports no running Graph. The launcher failure was
workspace routing discovery, not evidence that implementation must be repeated.
The current outcome and retained validation are in
[892c47-author-tla-cards.md](892c47-author-tla-cards.md).

The remaining pre-demo boundary is concept review and publication assessment,
then an update to existing PR394. Full gate/CI checks remain explicitly deferred;
they are not a step in the saved pursue procedure. The recorded draft publication
still describes the superseded implementation. Jack Heart's accepted boundary
remains one updated PR for demo review, without merge or additional Task launches.
