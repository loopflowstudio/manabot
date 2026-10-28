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

## Recovery for the Rust implementation

The preceding Graph stopped at automatic sync before receiving this direction.
The installed sync checkpoint failed on its ignored `.lf/tmp` exclusion pathspec.
The operator preserved the cleanup and direction in commit `080289907`, completed
`lf sync --manual` and `lf sync --continue`, retained world w16, and preserved both
scratch histories. Main's resolution repair `7d7736c6a` is now integrated locally;
reassess affected qualification and validation on that combination. Earlier passing
checks are not evidence for the integrated runtime or the forthcoming rewrite.

Update existing draft PR #394 after implementation and required checks; do not
create a second PR or repeat the superseded JSON implementation. The remaining
work is the Rust authoring revision, validation, publication and demo. If main
moves again before automatic sync, recover with the supported manual sync path
rather than repeatedly invoking the broken checkpoint. Keep Cargo to two jobs
and avoid unnecessary concurrent checks on the loaded machine.
