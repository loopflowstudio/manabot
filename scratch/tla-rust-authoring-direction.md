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
