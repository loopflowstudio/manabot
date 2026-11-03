# TLA trophy pilot after the five-card repair

This prospective identity incorporates [PR #415](https://github.com/loopflowstudio/manabot/pull/415),
which merged while PR #416 was being published. Its corrected card definitions
change the native digest. V3 explicitly binds that runtime; it does not relabel
v1/v2 protocols, attempts, checkpoints, evaluation or capacity-replay evidence.

The sync baseline is `a6d481ba9f97eca5f819ffc07616060881a5491e`, merging main
`c823c9c8f53df06c723e93d6d3e7a8676a71fa9d` with the reviewed capacity repair.
Freeze the exact subsequent verified v3 commit through Etude before launch.
The installed runtime must have world `w17` and card digest
`dcb9d47c674a657176c2a9d56e016288d1e175729aa09cd60f0dc333633330ed`.
The unchanged input contract is **v6, max_facts=256**. Generic v6 still defaults
to 512; frozen v5 remains 128, and excess facts remain fatal.

The v3 regime differs from v2 only in its ID. Keep the same **134 entries**,
sideboards, source-event weights, independent seat draws, learning settings,
seeds, milestones and eight-pair/four-leg evaluation schedule. PR #415's broader
admission does not expand this frozen population or erase its 866 historical
exclusions. There is no held-out-deck claim. The original corpus SHA256 remains
`2173600970d0ab9efb44c620b082e8b15a2615d4e4d801ec0c015710dc333951`.

`tla-trophy-213-v3.json` retains the v2 schema and explicitly names its predecessor
protocol hash. `experiments/tla_pilot_v3.py` is the active runner; the shared
analysis reads its schedule and rejects earlier experiment identities. Start
with fresh initialization and baseline evaluation. Historical v2 execution and
analysis remain at `d655d37996d2702cd5fee6ac37db11d855e87640`; the v2 JSON and
Markdown files stay unchanged. The earlier 140-fact replay receipts bind their
recorded runtime, not this merged one.

The [v2 scientific and allocation contract](tla-trophy-213-v2.md) otherwise
applies unchanged: the **$10 cumulative cap**, **$1 first job ceiling**, bounded
deadlines, unchanged failure allowance, complete missing-result accounting and
joint paired-bootstrap criterion. Reconcile both prior $0.7375 reservations
and actual billing before any new paid allocation. A new protocol identity
does not replenish the budget. No GPU learner update or v3 result is claimed
by preparing this protocol. Report training only after observed GPU updates.
