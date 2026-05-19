# First Light Report — run 1

## Recommendation

**Stay in first-light: creature-play signal remains weak**

- Win rate delta vs baseline: +18.0% (baseline 82.0%, final 100.0%).
- Pass-vs-land choice state: land 65.2%, pass 2.7%.
- Land probability in pass+land states: 66.1% vs pass 1.7%.
- Spell opportunity retention: 0.41x baseline with cast_when_able 99.7%.

## Key metrics and deltas vs untrained baseline

| Metric | Baseline | Final | Delta |
| --- | ---: | ---: | ---: |
| Win rate | 82.0% | 100.0% | +18.0% |
| Win CI lower | 69.2% | 92.9% | +23.7% |
| Landed when able | 43.3% | 65.2% | +21.9% |
| Cast when able | 57.7% | 99.7% | +42.0% |
| Passed when able | 38.3% | 1.3% | -37.0% |
| Pass+land: pass rate | 39.2% | 2.7% | -36.6% |
| Pass+land: land rate | 43.3% | 65.2% | +21.9% |
| Mean pass prob in pass+land | 38.9% | 1.7% | -37.1% |
| Mean land prob in pass+land | 43.8% | 66.1% | +22.3% |

## Interpretation of causal-chain metrics

- Final landed_when_able: 65.2%
- Final pass+land split: land 65.2%, pass 2.7%
- Final spell opportunity count: 655
- Final explained variance: 0.434

## Experiment config

- Label: first-run
- Mode: dev
- Seed: 42
- Opponent: random
- Recipe: first_light_shaped_v1
- Total timesteps: 262144
- Eval interval: 65536
- Eval games: 50
- Git branch: jack-heart.first-light.20260305_1906
- Git commit: 74dcf316ca381656b5907979b15444f963b7b9be
- Tree dirty: True

## Optional detailed notes / tables

```json
{
  "agent": {
    "attention_on": false,
    "hidden_dim": 64,
    "num_attention_heads": 4
  },
  "experiment": {
    "device": "cpu",
    "exp_name": "first-light-first-run-bootstrap",
    "log_level": "INFO",
    "profiler_enabled": false,
    "runs_dir": "/Users/jack/src/manabot.first-light/.runs",
    "seed": 42,
    "torch_deterministic": true,
    "wandb": false,
    "wandb_project_name": "manabot"
  },
  "match": {
    "hero": "gaea",
    "hero_deck": {
      "Forest": 12,
      "Grey Ogre": 18,
      "Llanowar Elves": 18,
      "Mountain": 12
    },
    "villain": "urza",
    "villain_deck": {
      "Forest": 12,
      "Grey Ogre": 18,
      "Llanowar Elves": 18,
      "Mountain": 12
    }
  },
  "observation": {
    "max_actions": 20,
    "max_cards_per_player": 60,
    "max_events": 32,
    "max_focus_objects": 2,
    "max_permanents_per_player": 30
  },
  "reward": {
    "creature_play_reward": 0.06,
    "land_play_reward": 0.03,
    "lose_reward": -1.0,
    "managym": false,
    "opponent_life_loss_reward": 0.01,
    "trivial": false,
    "win_reward": 1.0
  },
  "train": {
    "anneal_lr": true,
    "clip_coef": 0.1,
    "clip_vloss": true,
    "ent_coef": 0.01,
    "eval_interval": 100,
    "eval_num_games": 50,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "learning_rate": 0.00025,
    "max_grad_norm": 0.5,
    "norm_adv": true,
    "num_envs": 4,
    "num_minibatches": 4,
    "num_steps": 128,
    "opponent_policy": "random",
    "target_kl": null,
    "total_timesteps": 262144,
    "update_epochs": 4,
    "vf_coef": 0.5
  }
}
```
