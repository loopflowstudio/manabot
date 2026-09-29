# manabot.study

Read how a player behaves from complete recorded games: does it take its land
drop, does it attack when nothing can block, which deck wins on the play. The
answers describe behavior. Strength and promotion claims belong to the arena
and its frozen protocols.

## Run the Allies versus Lessons report

```bash
uv run python -m manabot.study.allies_lessons --out .runs/study/allies-lessons
```

This records the demo Search-64 opponent against itself, against Random, and
Random against itself, then writes `games.jsonl` and `report.html` to `--out`.
The default set is 580 games and takes about ten minutes on ten workers.

```bash
# Rebuild the report from an existing record, without playing
uv run python -m manabot.study.allies_lessons --games .runs/study/allies-lessons/games.jsonl

# Study another player
uv run python -m manabot.study.allies_lessons --subject greedy --mirror-deals 20
```

## Layout

| Module | Holds |
|---|---|
| `record.py` | `GameSpec` in, `GameRecord` out. Every decision keeps the offers, the choice and the acting player's view of the board. |
| `measures.py` | Pure functions from records to `Rate`s, each with a Wilson interval and examples of the misses. |
| `report.py` | HTML building blocks: sections, rate tables, miss lists. |
| `allies_lessons/` | The matchup's setup and seatings, Learn choices, the report and its command. |

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
