# Gate choices — 2026-11-11

The prior design and QA notes were removed in ab40fe19f; gate read them from
its parent rather than recreating the old working files. Publication remains
outside this Task: the retained direction says "Still no tagging, upload or
publication."

For the relocated source manifest, use the same comparison policy as
`public_commitment_parity.verify_receipt`: keep the historical receipt intact
and compare all behavior exactly, allowing only source provenance to differ.
A fresh replay demonstrated that `identity.relevant_source` was the only
authored-match receipt difference. This is a reversible implementation choice,
not permission to rewrite frozen evidence or change rules.
