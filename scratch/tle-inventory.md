# TLE inventory — source evidence

[Scryfall TLE search](https://api.scryfall.com/cards/search?q=set%3Atle&unique=cards&order=name),
both pages, fetched 2026-11-12T02:27:14+00:00: **242 distinct names and Oracle
IDs**. The complete names, paragraphs and resolved printing identities now live
in `managym/cards/tests/fixtures/tle-inventory.jsonl`; source/digest details are
in `scratch/questions.md`. This is source data, not an admission or behavior proof.

At `4c041d6cefe829c5b1b52273ba19991d663ee26b`, literal definitions in
`managym/cards/src/*.rs` matched eight names; 234 were absent:

| Card | Existing definition |
|---|---|
| Bastion of Remembrance | iko.bastion_of_remembrance |
| Fabled Passage | eld.fabled_passage |
| Fists of Flame | mh1.fists_of_flame |
| Lightning Bolt | lea.lightning_bolt |
| Mountain | lea.mountain |
| Path to Redemption | tla.path_to_redemption |
| Plains | lea.plains |
| Razor Rings | tla.razor_rings |

The original 242-row baseline table is retained at
`e19af3eb51a614f501fbcaca60129db6a2b4e221:scratch/tle-inventory.md`.
Current implementation status belongs in the active plan. The world pool derives
from definitions, never this inventory; reprints retain first-printing owners.
