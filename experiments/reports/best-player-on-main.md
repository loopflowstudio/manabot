# Frontier Small u1952 on main

Exploratory compatibility check of the original checkpoint from Etude run
`59718fe34945`. Does it still play complete games against the first-64 heuristic
on the recorded TLA lists, and on lists containing cards released into managym
after its training run? Both arms ran on the mini at USD 0 against a USD 3 cap.

The pilot scored 0.680 on the original deck pairs, inside its recorded
0.609–0.813 interval. All 160 games across both arms finished and replayed.

## Cohorts

The historical arm uses the original 16 deck pairs in their recorded order,
including the pair with two identical lists. Each pair plays its recorded
deal and a second deal at that seed plus 16. Every deal crosses the two decks
and the two seats in four legs. A draw scores one half. All games must finish
and pass Command replay. The 95% interval resamples whole deck pairs, keeping
both deals and all eight legs together.

The new-card arm takes the first eight complete trophy lists in retained
17Lands source order that managym admits today and whose main deck contains
a definition missing from the writer catalog. The sideboard is admitted
alongside the main deck. Adjacent lists form four pairs and use the same deal
schedule as the first four historical pairs. This measures the whole pilot on
those lists. It cannot isolate the contribution of a new card embedding.

The heuristic changed at `452d7b5357` and `86ad8e9c6a`. A score difference
combines possible drift in rules, policy inputs and opponent play.

## Recorded deck pairs

| Revision | Score points / games | Score | Deck-pair 95% interval |
| --- | ---: | ---: | ---: |
| Writer, `ab2b757d9` | 46 / 64 | 0.719 | 0.609–0.813 |
| Compatibility change on main, `5f12080d81` | 87 / 128 | 0.680 | 0.586–0.758 |

The measured score is inside the recorded interval. The original deal seeds
yielded 42 wins in 64 games and the additional deals yielded 45 in 64. Every
game terminated and all 23,831 decisions passed replay. The loader used the stored programs for all
360 writer definitions, including the 136 whose serialization changed,
and initialized 396 token rows and 103 naming rows. No key was unexplained.
[Recorded comparison](tla-frontier-c5fba4-results.md) ·
[Evaluation rows, Commands and replay receipts](../data/best-player-on-main/frontier.tar.gz)

## New-card decks

The eight main decks introduced Black Sun's Zenith, Scout's Warning (two
decks), Noxious Gearhulk, Blasphemous Act, Tarnished Citadel, Agent of
Treachery and Lita, Mechanical Engineer. Each pair gave the pilot turns on
both sides of both lists.

At `14a168309f` the pilot won 23 of 32 games (0.719). All 8,340 decisions
passed replay. The deck-pair 95% interval was 0.500–0.938. With only four
pairs, that interval leaves considerable uncertainty about strength on other
new-card lists. [New-card lists, evaluation rows, Commands and replay receipts](../data/best-player-on-main/new-cards.tar.gz)

## Reproduce

Use the experiment checkout at `14a168309f` for this v9 pilot.

Both arms load the original 3,945,456-byte policy from run `59718fe34945`
by its recorded digest and play through local evaluation. With access to the
artifact bucket, run from the repository root and choose fresh output directories.

```sh
uv run --extra artifacts python -m experiments.best_player_on_main --arm historical --out .runs/u1952-frontier
uv run --extra artifacts python -m manabot.data fetch experiments/data/mtg213/corpus-v1/corpus.json
uv run --extra artifacts python -m experiments.best_player_on_main --arm new-cards --out .runs/u1952-new-cards
```

Deck lists derive from [17Lands public data](https://www.17lands.com/public_datasets),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
