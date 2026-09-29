# manabot.study

Read how a player behaves from complete recorded games: does it take its land
drop, does it attack when nothing can block, which deck wins on the play. The
answers describe behavior. Strength and promotion claims belong to the arena
and its frozen protocols.

## Run the Allies versus Lessons report

```bash
uv run python -m manabot.study.allies_lessons --out .runs/study/first
```

This records the demo Search-64 opponent against itself, against Random, and
Random against itself. The default set is 580 games and takes about ten minutes
on ten workers. Each run writes three files to `--out`:

| File | Holds |
|---|---|
| `study.json` | Every parameter of the run |
| `games.jsonl.gz` | One recorded game per line |
| `report.html` | The report, self-contained |

## Re-run with a change

Start from a saved configuration and override the fields that differ:

```bash
uv run python -m manabot.study.allies_lessons --out .runs/study/second \
    --config .runs/study/first/study.json \
    --set mirror_deals=100 --set 'gw_allies=-2 Badgermole Cub, +2 Plains'

# Rebuild the report from an existing record, without playing
uv run python -m manabot.study.allies_lessons --out .runs/study/first \
    --games .runs/study/first/games.jsonl.gz
```

`--help` lists the fields. Deck changes are written as signed counts from the
authored main decks. Set `keep_decisions=false` to keep outcomes only, which is
enough for win rates and keeps a large sweep small.

## Explore in a notebook

```bash
uv sync --extra dev --extra notebook
uv run jupyter lab experiments/study/allies-lessons-play-study.ipynb
```

The notebook reads the same `study.json` and record, shows the measures as
DataFrames (`manabot.study.frames`), and embeds the report. Its first code cell
holds the parameters. Keep committed notebooks free of outputs.

## Compare decklists

```bash
# One pair of lists
uv run python -m manabot.study.allies_lessons.balance \
    --gw "-2 Badgermole Cub, +2 Plains" --deals 100

# Several candidates from a file, saved and rendered
uv run python -m manabot.study.allies_lessons.balance \
    --candidates experiments/study/gw-allies-weaken-2026-09-29.json \
    --out .runs/study/balance.jsonl --html .runs/study/balance.html
```

The rate is UR Lessons' share of wins with one player in both seats.
Candidates run on the same deal seeds, so they can be compared with each other.
Candidate files live in `experiments/study/`; results stay under `.runs/`.

## Layout

| Module | Holds |
|---|---|
| `record.py` | `GameSpec` in, `GameRecord` out. Every decision keeps the offers, the choice and the acting player's view of the board. |
| `measures.py` | Pure functions from records to `Rate`s, each with a Wilson interval and examples of the misses. |
| `report.py` | HTML building blocks: sections, rate tables, miss lists. |
| `frames.py` | Records as pandas DataFrames, one row per game, seat or decision. |
| `allies_lessons/` | The matchup's setup and seatings, saved configuration, Learn choices, decklist comparison, the report and its command. |

Put a question in `measures.py` when it would make sense for any matchup, and
in the matchup's subpackage when it names that matchup's cards or mechanics.

## Ask a new question

A measure filters decisions and yields whether the player did the thing:

```python
def held_full_grip(games, label):
    """Own turns that ended with seven or more cards in hand."""
    def outcomes():
        for game, seat in seats(games, label):
            for turn, decisions in own_turns(game, seat).items():
                yield decisions[-1].hand < 7, Example(game.game_id, turn, "7+ cards")
    return rate_of(outcomes())
```

If the record lacks what the question needs, add the field to `Decision` in
`record.py` and record again. Measures never reach back into the engine.

## Limits

- A record holds the acting player's view. The opponent's hand is never stored.
- "Could attack freely" is judged from visible untapped creatures and flying or
  reach. Combat tricks and what an attack leaves undefended are not modelled.
- Intervals treat games as independent. Games sharing a deal seed are not, so
  real uncertainty is somewhat wider.
- Study records are exploratory. They are written under `.runs/` and are not
  arena evidence.
