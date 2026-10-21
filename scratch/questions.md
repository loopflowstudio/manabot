# Academy redesign assumptions and review choices — 2026-10-21

## Current direction

Jack Heart, 2026-10-21: “otherwise lg2m, ship it.” Publication and waited landing
are now authorized after the remaining grouped/fixed battlefield work and its
checks. This supersedes the earlier no-publication hold below. Latest layout:
native-type creatures and lands in separate lanes, artifact tokens in a corner;
bounded field dimensions, compact cards and explicit paging rather than page
growth. Preserve exact choices, all object identities and frozen evidence.

Academy UI move delivered, 2026-10-21: `academy/ui` owns the Svelte project;
launcher, Docker, CI, protocol/test imports, package readers and active docs use
that path. No root frontend directory or alias remains. Python wheels include
only the immutable packs from UI, not Node sources/dependencies. Four manifests
are byte-identical. Checks: 103 UI tests, clean Svelte/Ruff/diff checks, production
build and Python wheel, 30 remaining path/pack/compiler tests (earlier launcher
and clean-machine tests retained their passing result), and all three live
Precision browser cases passed. Rendered screenshots confirm Scryfall art and
the board above choices, with shared-table/Study tools collapsed below play.
All 37 cached images moved intact, including the card back; artist/source credits
are available in the footer. The image downloader now respects 429 cooldowns.
Demo remains at http://127.0.0.1:5173/?demo=precision for Jack Heart's review.

Jack Heart, 2026-10-21: move the frontend into Academy. Selected location is
`academy/ui`; preserve the existing app, downloaded Scryfall cache and frozen
artifact bytes. Update active launcher, deployment, CI, documentation and test
paths. No parallel frontend directory or compatibility symlink is needed.

Demo corrections delivered, 2026-10-21: board first at all widths, full-width
play selection next, history/consequences/advice collapsed, one initial start
button and explicit pending state with repeat starts suppressed. Checked demo
setup retains the full Command tape and Study roots but persists the completed
setup once. Live WebSocket startup measured 0.77 seconds after restarting the
dev backend. Verification: four decision-surface Python tests, 15 socket tests,
clean Svelte/Ruff/diff checks, and three real browser scenarios against the live
dev server passed (390px, 1500px, alternative mana cost; 13.2 seconds total).
The demo is available for Jack Heart's review. Frozen receipt compatibility is
a separate existing shipping limitation below, not fixed by this UI patch.

Live demo feedback, Jack Heart, 2026-10-21: the board should be at the top,
and “the UI for selecting plays should be the primary thing.” This supersedes
decision-first narrow ordering: show board first for orientation, followed
immediately by prominent legal play selection; setup, history and stop controls
are secondary. The initial screen has duplicate New Game buttons and clicking
appears inert. Fix startup and verify against the actual running dev server,
with visible pending/error feedback and a single initial start action.
Decision log, consequence history and ancillary details stay available but
collapsed by default (Jack Heart, same demo). Do not auto-expand them.

Jack Heart, 2026-10-21: “just build it and then i will demo”. This supersedes
the design-only hold below and authorizes implementing the full proposed
interaction model, including explicit offer commitment after inspection,
acting-view Watch by default, and the decision-first narrow layout. Build and
verify the real runtime, then leave the demonstration to Jack Heart here.
No publication, deployment, paid compute or training is requested.
The earlier design-only boundary below is historical, not current direction.

Implementation review finding: keep internal protocol language out of the
product. DecisionSurface currently says “submit a Command” and renders raw
“object N” IDs and an underscored stage converted into a heading. Use normal
player language (“choose an action”, “selected target”, “reserved sacrifice”),
readable stage labels and board highlighting/accessible copy to distinguish
duplicates. Exact IDs remain internal bindings or optional diagnostics, not
the primary product explanation. Gate should check this across all modes.

## Implementation assumptions and discoveries

Academy uses the existing native `skip_trivial=False` setting so a sole legal
mana activation, color or payment is surfaced and recoverable. The prior default
collapsed the complete Treasure payment before the browser could observe it.
Configured priority-stop automation remains a separate, recorded Academy path.
Training defaults and frozen w9 worlds/checkpoints/erd1 evidence remain unchanged.
Current Learn/Precision demo prefixes explicitly bind this runtime setting;
attempt provenance is `automated_validation`, including human continuation. Old
Learn/navigation fixtures remain frozen compatibility evidence. The retained
navigation reproduction test explicitly selects its historical native preset.

The explanatory native API summarizes prior actor choices and current visible
state; it adds no tensor features, legal choices or mutable transaction state.
Public ownership/kind and committed facts can cross viewers; private source,
target selection, reservation and inspection fields cannot. New optional fields
are omitted by historical serializers/readers, preserving old frame hashes.

No supplied real checkpoint bytes or human usability acceptance is claimed.
Publication, landing, deployment, training and Task completion remain excluded.

## Gate findings — 2026-10-21

Follow-up gate: the historical-collapse diagnostic fails at revision 18
(`DECLARE_ATTACKER`, tape offer 2, available offers 0/1). Loading the unmodified
HEAD versions of `academy/server.py`, `academy/presentation.py` and
`academy/learn_demo.py` in an isolated diagnostic process reproduces that same
divergence. The v1 public-commitment tape therefore already has a producer/world
binding problem at this checkout's base; changing the Academy preset alone is
insufficient. This does not certify the current explicit runtime or waive the
strict historical verifier. No historical tape or receipt was changed.

Resolved the recorded product-language finding: shared context now has readable
stage labels, card locations and copy positions; exact IDs stay internal. Card
accessible names retain duplicate-copy positions. Historical empty-search copy
and live navigation prefixes were repaired without changing frozen fixtures.

Gate blocker: frozen public-commitment Commands are rejected by the current
explicit-decision runtime (`tests/academy/test_public_commitment_parity.py`).
A diagnostic run with the original native collapse preset still encountered
`Villain offer policy selected unknown offer: 2`; the fixed authored authority
receipt also differs in its new offer subjects/presentation facts. An internal
compatibility-preset experiment was removed rather than leaving a partial
compatibility path. No frozen receipts or world identities were rewritten.
Resolve how those historical release verifiers bind their producer/preset before
calling this branch ready to ship. Existing retained frame/reader tests remain
separate evidence; they cannot certify live regeneration of a frozen tape.

Observation scope assumption: this branch adds optional Academy explanation and
presentation fields over existing native observations. It changes no model tensor
shape, legal rules, training preset or world identity. Arena/monitor checks were
included; full learnability training was not launched under the brief's explicit
no-training boundary. This is not a learnability pass.

Final current-runtime checks: 91 scoped Python integration checks passed (one
fixed-match summary test excluded), 18 focused surface/setup/persistence tests
passed, and all current browser scenarios have passing focused runs. The owned
Study helper also retains its actual returned participant credential, so the
permission checks are exercised without weakening production access. Historical
receipt regeneration remains the gate blocker.

Grouped bounded battlefield delivered before shipping: native optional permanent
card types and token metadata drive creature/land/other/token lanes. Animated
lands remain in creatures; artifact creatures are creatures. Fixed viewport-based
field dimensions stay unchanged as objects enter/leave. Cards compact to 44px,
then page explicitly; focusing a legal offer reveals its object's page. No merged
identities or inferred card-name rules. Historical serialization omits the new
metadata. Zone labels are accessible groups to avoid duplicate player landmarks.
Checks: 105 UI unit tests, clean Svelte check and production build, 19 affected
Python schema/surface tests, four Rust debug schema/reader targets passed.
Real Precision browser interaction checks passed after the accessible-group fix
(see /tmp/academy-battlefield-browser.log for final result). Prior receipt blocker
still requires an actual repair before landing; tests/receipts must not be weakened.
