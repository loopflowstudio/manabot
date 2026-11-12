# MTG-287 decisions and open release question

Jack Heart, 2026-11-11: "my goal would be to sync manabo and managym's publishing,
so that we basically only ever publish them together and at the same versions
for now". The current Task steer says: "Still no tagging, upload or publication."
Licence, hidden labels, schema/defaults, capacity and card text moved to MTG-288
through MTG-292 and are outside this change.

Reversible choices: use the proposed uv workspace, canonical `managym` import,
pure-Python `manabot` and exact matching engine dependency. Retain the requested
`manabot.managym` alias, including submodule and stub identity. Keep the current
0.2.1 metadata while developing; this does not select the first paired release.

Release question: "Which shared version should the first paired managym/manabot
release use?" Both trusted publishers and explicit publication authorization
remain delivery prerequisites. No publication was attempted.

Build finding: maturin with `python-source = ".."` puts shared content under the
staging parent's name, breaking Rust `include_str!` paths. `prepare.py` now stages
a self-contained engine project with `managym/` and `content/` children. Only the
staged training project includes its generated source receipt in its wheel;
local editable builds have no generated receipt. The receipt covers the moved
engine adapters as well as training Python and the protocol.

Inherited MTG-285 history remains in `scratch/questions.md` at checkout base
`4c041d6cefe829c5b1b52273ba19991d663ee26b`; it is not authorization for this Task.
Historical experiment paths and observation-comparison receipt keys remain
unchanged. Current launch/setup and source-binding consumers use the new path.

Later steer, Jack Heart 2026-11-11: "publication of 0.2.1 is pending and PyPI
still lists only 0.2.0"; "What remains for this Task is the separate `managym`
distribution and publishing both packages together at one version, built on
that release tooling." The implementation retains the existing release tools
and extends their inventories; it does not redo hand snapshots or publish.
