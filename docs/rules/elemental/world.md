# One ten-deck Elemental world (MTG-164)

Jack Heart accepted these lists and autonomous integration on 2026-10-20.
The exact [40-card lists and outside-game Lesson pools](decks.md),
[machine-readable selection](decks.json), and [cube snapshot](cube-snapshot.csv)
are retained here. Cube counts apply independently to each main deck plus its
Lesson pool; basic lands are unlimited and reuse between decks is allowed.
This is development world **w10**, with one `elemental-world-v1` compiled pack.
It does not freeze a strength baseline or certify balance.

## Setup and runtime authority

The compiled source is `content/semantic/v1/elemental_world.source.json`.
Its ten keys are `elemental_gr`, `elemental_gw`, `elemental_ur`, `elemental_ub`,
`elemental_uw`, `elemental_ug`, `elemental_wr`, `elemental_wb`, `elemental_rb`,
and `elemental_bg`. UG is the accepted Temur Lessons list, including Iroh,
Grand Lotus and Ran and Shaw. BG retains landfall and sacrifice.

`managym.authored_deck_keys(pack_key)` discovers the native roster;
`managym.authored_deck_setup(pack_key, deck_key)` returns a complete main deck,
sideboard and explicit compiled pack identity. Any two admitted setups,
including mirrors, run through the same engine. Native admission rejects an
unknown pack, a missing seat binding or a changed main deck/sideboard.
No per-matchup rules implementation or human-only action is introduced.

Academy's schema-2 presentation manifest has a roster plus a default pair
selected from it. Each row is checked against the compiled setup. It includes
local authored treatments and retained identity provenance for all 100
reachable definitions: **89 nonbasic cards, five basics, six token definitions**.
New identity records refer to the retained Scryfall fixture by semantic key;
they do not invent Oracle UUIDs or fetch third-party artwork. The earlier
schema-1 pair manifests and their exact treatments remain unchanged.

Both browser deck selectors include all ten names. The server preserves their
Lesson pools and explicit pack through new games and recorded root rebuilds.
A named Elemental setup paired with an outside setup is rejected, so it cannot
silently run the generic registry. Human and bot Commands still use the same
native legal offers, revision checks and information boundary.

Training uses the ordinary APIs, for example:

```python
from manabot.infra.hypers import AgentSpec, MatchHypers

match = MatchHypers.authored(
    "elemental-world-v1", "elemental_ug", "elemental_bg"
)
agent = AgentSpec(semantic_pack="elemental-world-v1", policy_film=True)
```

These values go into the existing environment/model/trainer configuration.
No new curriculum, heuristic, paid run or strength campaign is included.
The semantic policy joins all admitted definitions to complete typed programs;
it does not rely on transport IDs as card meaning.

An Elemental checkpoint binds all ten exact setups, the full compiled manifest,
input shapes and learning schema 6. The ordinary catalog and loader reconstruct
an explicitly bound representative root and validate that roster before loading
weights. Execution admits any pair or mirror from it and rejects a foreign or
modified setup. Compatibility does not claim that a policy trained on one pair
has learned the other pairs. Earlier checkpoints fail w10 world admission.

## Complete-card audit and ownership

`managym/tests/elemental_world_tests.rs` compares all 100 complete compiled
and native definitions: characteristics, costs, triggers, targets, static rules,
activated abilities and token closure. Only inferred single-target clauses and
trigger ordering are normalized. The selected native shells and Oracle text
remain checked against `managym/tests/fixtures/scryfall_cards.json` by the normal
Rust conformance test. The opcode fixtures check compilation and closure;
they are not an independent rules oracle.

The 65 predecessor-owned cards retain their directed rule mappings:

| Selected cards | Directed rule evidence |
| --- | --- |
| All 42 combat/Learn cards | [Elemental combat contracts](../elemental-combat.md), `managym/tests/rules/elemental_cards.rs` |
| All 17 sacrifice/landfall cards | [Sacrifice contracts](../elemental-sacrifice.md), `managym/tests/rules/sacrifice_cards.rs` |
| All six Temur cards | [Temur contracts](../elemental-temur.md), `managym/tests/rules/temur_cards.rs` |

The other 24 selected nonbasic cards are audited on the same surviving rules
path, rather than assuming that registration proves complete support:

| Selected cards | Behavior and directed owner |
| --- | --- |
| Accumulate Wisdom | Bounded look/select, Lesson graveyard threshold; `stage2_cards.rs` |
| Allies at Last | Ally affinity, controlled-creature power and target legality; `stage2_cards.rs` |
| Badgermole Cub | Earthbend and tapping-for-mana bonus, excluding waterbend taps; `tla_cards.rs`, `temur_cards.rs` |
| Compassionate Healer | Tap-triggered life and scry; `stage3_cards.rs` |
| Water Tribe Rallier | Waterbend payment and creature selection; `stage2_cards.rs` |
| White Lotus Reinforcements | Other-Allies anthem and stacking; `stage3_cards.rs` |
| Divide by Zero; Igneous Inspiration; Pop Quiz | Printed interaction/draw followed by Learn; `stage2_cards.rs` and retained Learn tests |
| Dragonfly Swarm; Fire Nation Cadets | Noncreature/Lesson triggers, firebending and scope; `tla_cards.rs`, `stage3_cards.rs` |
| Earth Kingdom Jailer | Exact-object exile link and return; `tla_cards.rs`, `stage3_cards.rs` |
| Fancy Footwork; Yip Yip! | Untap, target count, temporary stats and keywords; `stage3_cards.rs` |
| Firebending Lesson | Kicker payment, two/five damage and affordability; `stage2_cards.rs` |
| Forecasting Fortune Teller | Entry Clue and sacrifice-to-draw lifecycle; `tla_cards.rs` |
| It'll Quench Ya! | Counter-unless payment and Lesson identity; `stage2_cards.rs`, `stage3_cards.rs` |
| Invasion Reinforcements | Flash casting on the opponent's turn; `tla_cards.rs` |
| Jeong Jeong's Deserters | Target trigger, counter and lethal SBA; `tla_cards.rs` |
| Otter-Penguin | Second-draw payoff and turn reset; `tla_cards.rs` |
| South Pole Voyager | Independent second-resolution draw; `tla_cards.rs` |
| Gran-Gran | Mandatory loot on tapping and conditional noncreature generic reduction; `ur_lessons_increment.rs` |
| Combustion Technique | Graveyard-counted damage, object-local death exile, expiry and rollback; `ur_lessons_increment.rs` |
| Enter the Avatar State | Controlled target, Avatar type addition and all four temporary keywords; `stage3_cards.rs`, new incarnation/cleanup case in `elemental_cards.rs` |

The last three lacked compiled admission. Typed instructions now lower their
existing mandatory discard and death-exile rules, and close Enter the Avatar
State's explicit native omission of its Avatar subtype. A temporary subtype
belongs to a battlefield incarnation, is journaled with the permanent, affects
battlefield subtype predicates, disappears at cleanup or reentry, and is not
copied as a printed characteristic. Both viewers see it. The shared protocol,
Python/native observations and Academy card labels carry it. The permanent
encoder's new Avatar feature is index 26, with validity at 27. Hash version 7
includes this state. Frozen worlds, schemas and `erd1` evidence are untouched;
`ed2` remains a pre-command address and its selected Command remains validated
against its landmark separately.

## Bounded integration evidence

Before running, the cohort was fixed as all ten mirrors plus GR→UG, UG→GR,
UR→GW and GW→UR. Each uses game seed 164, uniform legal-offer selection from
`random.Random(164)`, and a cap of 30,000 Commands. The runner is:

```bash
uv run python -m scripts.verify_elemental_world --out conformance/elemental-world-w10-v1/command-cohort.json
```

The [retained result](../../../conformance/elemental-world-w10-v1/command-cohort.json)
records **14 terminal games / 13,126 Commands**, with **268** isolated fork and
stale-rejection checks. Every Command was replayed on a second root with equal
frames, transition receipts and state digests. Both viewer projections were
checked for opposing hand/candidate and library-order privacy at every choice.
The largest offer set in this cohort was 20; that does not prove a universal
64-action capacity bound.

The separate debug native test exercises every ordered pair and mirror: 100
cells, seed `16400 + left_index*10 + right_index` in compiled sorted-deck order,
with a 30,000-action cap per cell. All 100 completed. This is one bounded seed
per cell, not a balance or playing-strength estimate.

The focused Python checks cover all 100 named Academy pairings, exact cube
limits and Lesson pools, full policy-program input, a real Academy Command,
Avatar observation parity, complete UG/BG games in both seat orders through
the ordinary training encoder, actual CPU PPO parameter updates, and checkpoint
save/catalog/reload across the ten-setup roster. Wrong pools, outside setups,
missing explicit identity and older worlds are rejected.

The w10 semantic conformance corpus adds both GR/UG seat orders to the retained
validation orientations. Its normalized and optimized executors share native
rules primitives; it does not establish agreement with an independent engine.
The inherited Phase table is pinned source evidence. Current Academy authority,
Learn and navigation fixtures have separate w10 paths; earlier artifacts are
not rewritten to appear current.

Full affected suites, the prescribed learnability diagnostic, release browser
checks and the pinned Linux visual gate belong to the authored delivery gate.
No human demo, review or merge-click approval is required by this Task's
accepted direction. Publication, waited landing and Task completion are later
Flow steps; this implementation record does not declare them complete.

## Delivery gate evidence — 2026-10-21

At published head `316b9f76`, CI passed debug Rust, Linux integration (both
groups), Academy integration on Linux, protocol conformance, semantic coverage,
semantic conformance and the pinned Ubuntu visual reference comparison (14
browser tests). The full local learnability diagnostic also passed, including
65,536 production reward-PPO transitions: preference score increased from
0.59375 to 1.0, versus 0.640625 without updates. These results do not certify
strength. macOS CI jobs were still queued at gate handoff.

Gate adds release browser coverage for selecting UG/BG, BG/UG and UG/UG through
the real selectors and backend. All three cases passed, checking all ten menu
entries, exact visible Lesson pools, hidden opposing hands/pools, pack identity
and an advancing Command. The existing 13 local release interaction tests also
passed. Rust formatting, Clippy with warnings denied, Python Ruff lint/format
and frontend static checks passed.

**Delivery remains blocked.** An additional affected Python run passed 77 tests
and failed four inherited evidence checks. The belief-recommendation fixture
still selects the expected actions, but current visit margins are `[14, 48]`
against frozen `[66, 38]`. Two public-commitment checks fail in live replay at
`academy/public_commitment_parity.py:376` with `KeyError: 'status'`. The RUL-9
provenance check rejects its changed landed parity receipt. These failures do
not authorize replacing frozen evidence; their contract owners need to resolve
the current-world proof gaps described in the predecessor contracts.

Separately, [Clean-machine Play on this head](https://github.com/loopflowstudio/manabot/actions/runs/39680117834/job/119500174780)
reaches play and reloads with equal visible state, but its first action after
reload does not advance `data-update-seq` within 30 seconds (remains 1).
The cause is unresolved; passing Learn reconnect and visual tests does not
close this recovery failure. No clean-machine proof receipt was produced.
