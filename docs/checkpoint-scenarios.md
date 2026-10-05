# Current-world checkpoint tactical scoring

Score retained S1–S5 positions through the ordinary `checkpoint` player without
training a new policy:

```bash
uv run python -m manabot.verify.checkpoint_scenarios --checkpoints checkpoints.json --out .runs/tactics-001
uv run python -m manabot.verify.checkpoint_scenarios --out .runs/tactics-001 --report-only
```

The JSON file maps scenario names to saved checkpoint paths. Omitted scenarios
appear as unsupported; paths are relative to the command's working directory.
For example:

```json
{
  "s1_counter_the_bomb": ".runs/s1-policy.pt",
  "s2_hold_the_wipe": ".runs/s2-policy.pt",
  "s3_bolt_the_threat": ".runs/s3-policy.pt",
  "s4_race_vs_block": ".runs/s4-policy.pt",
  "s5_hold_up_quench": ".runs/s5-policy.pt"
}
```

Each checkpoint must already bind the exact scenario's hero/villain deck pair
and empty sideboards from `manabot.verify.competency.SCENARIOS`. The ordinary
world/setup/input admission checks remain unchanged. **An Allies/Lessons
checkpoint cannot score these different decks.** No transfer permission is
inferred from equal tensor dimensions. A mismatched checkpoint is a failed
admission, not a tactical loss; missing assignments are unsupported. These are
current-runtime executions of retained custom-deck positions, not newly authored
selected-match positions or replacements for historical results.

The saved observation capacity is used at every decision, including the injected
root. Overflow, illegal actions, root mutation, wrong world/setup, interrupted
execution and Command-cap exhaustion fail explicitly. Belief-enabled policies
are unsupported because these injected roots lack authentic prior history.
Ordinary observation-only and compound checkpoint policies use their existing
player behavior; compound suffixes reset at each root. The runner uses CPU,
one thread, deterministic action selection, seed 2 and at most 1,000 Commands
per attempt by default. `--seed` and `--max-steps` make these choices explicit.
No weights are updated.

## What the report means

Before scoring, reference and contrast scripts revalidate each premise against
resolved engine state. Failure makes the scenario unsupported for that run.
The policy run then measures countering bait versus a bomb (S1), delayed versus
early wipe resolution (S2), held removal and target selection (S3), flying attacks
and ground blocking (S4), and passing with response mana versus tapping out (S5).
Scores require both the recorded reference behavior and resolved effects. S4
therefore requires the flying attack/ground hold as well as a win, and S5 requires
the main-phase mana hold as well as countering the threat. S2 checks actual
removal after resolution, rather than scoring cast intent alone.
A turn-bounded run that never demonstrates the reference behavior records false;
a technical failure records no score. The Command cap is a technical guard, not
a tactical stopping rule.

These are local behaviors against a particular scripted continuation. The
contrasts establish that the intended effects are executable; they do not prove
globally optimal play, reachability under a natural opening, hidden-information
planning quality, or checkpoint strength. Injection bypasses costs and ETB
triggers. Do not pool different scenario setups as a selected-match strength
number. ETU-99 owns independent-seed attacks and empirical robustness;
ETU-85 retains demo-opponent comparison ownership. No ETU-91 allocation applies.

## Evidence and reproduction

`results.json` retains premise validation, every attempt's status and error,
checkpoint digest, behavioral measurements, separate premise-validation cost,
elapsed attempt time including replay and
trace identity. `traces.jsonl.gz` uses arena serialization with source-versioned
scenario roots, native world/setup/input binding, declared observation capacity,
seed, initial state digest and each actor's viewer frame, legal offer, Command
and pre/post state digests. These are private diagnostic artifacts.

Offline report regeneration reconstructs roots and replays Commands without
loading checkpoint weights. Artifact/source/world mismatches fail. Prefix replay
is explicitly requested; the ordinary arena still requires complete games.
A passing prefix receipt proves the retained trajectory, not game completion or
the correctness of a strategic interpretation. Historical experiment bytes and
scores remain unchanged. Reproduction requires the matching fixture source and
rules runtime. Failed attempts preserve completed Commands and cost; Python or
native process death is not a resumable execution contract.

Deterministically initialized, untrained fixture checkpoints exercise all five
scenarios in `tests/verify/test_checkpoint_scenarios.py`. Coverage includes
world/setup/capacity rejection, hidden-hand swap invariance, repeated identical
traces, tamper detection, failed-prefix retention and report regeneration after
removing checkpoint files. These are software acceptance checks only.
