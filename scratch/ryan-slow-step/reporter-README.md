# manabot 0.2.1: one engine step takes 2 to 120 seconds on boards with many mana sources

## What happens

A single call to `VectorEnv.step_isolated_into_buffers` (and to `Env.step` under flat search) occasionally takes
between 2 and 120 seconds, where the median step for 64 games is about 4 ms. The process uses one core for the whole
time. It is not an infinite loop: every slow step observed did return.

Every stack sampled during a slow step was in the same place:

```text
PyVectorEnv::step_isolated_into_buffers
  managym::flow::tick::Game::tick_priority
    managym::flow::mana::Game::payment_model
      Vec<ManaSource>::from_iter(... iter_battlefield_permanents ... layer_views ... characteristics_many ...)
```

## When

- Only on boards where one player has about 19 or more permanents. Smaller boards never showed it.
- In the one case traced to a single game, the action was `activate_ability` at priority, and the next decision
  offered a long list of `choose_mode` options (mana colours). The acting player had 11 lands (Forest x4, Island x4,
  Plains, Mountain, Kyoshi Village) plus Hermitic Herbalist, Bender's Waterskin and five other permanents.
- It does not need a trained policy. A scripted policy that prefers `play_land` and `cast` reproduces it. Uniformly
  random play did not show it in several hours, probably because random play rarely builds such boards.
- The table below comes from 64 games at once on `VectorEnv`, played by that scripted policy on human TLA decks
  from 17Lands.

## Measurements (manabot 0.2.1 from PyPI, Python 3.13, Apple M3 Max, 64 games, TLA human decks from 17Lands)

| Seed | Steps played | Slow steps (> 0.25 s) | Longest step |
| ---- | ------------ | --------------------- | ------------ |
| 2    | 6,660        | 9                     | 38.9 s (122 s in another run under load) |
| 4    | 5,674        | 8                     | 91.6 s       |
| 5    | 11,670       | 28                    | 27.6 s       |
| 1, 3 | —            | —                     | did not finish within 7 minutes |

In seed 4's run, 22% of finished games had one side pass 18 permanents and 12% pass 20.

## Reproduce

Run `manabot_slow_step.py`, which needs only manabot:

```sh
pip install manabot==0.2.1
python manabot_slow_step.py
```

It replays one game of two 40-card TLA decks on a scalar `managym.Env` from a fixed seed and a fixed list of 426
actions, and times every step. On an M3 Max it prints:

```text
step 242 of 426: action 2 took 0.65 s
step 243 of 426: action 2 took 0.32 s
step 426 of 426: action 3 took 48.48 s
slowest step: 48.48 s
```

The last action activates a mana ability with 11 lands and two other mana sources in play.

## Also seen

- `Env.flat_mc_scores` playouts stall in the same function: 4 of 40 games against random play ran past 120 s, and 11
  of 20 against a trained player ran past 40 s.
- Not checked on 0.2.2.
