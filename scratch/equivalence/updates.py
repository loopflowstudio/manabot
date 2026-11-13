"""Item 9: byte-identical update equivalence before/after the backend argument.

uv run python scratch/equivalence/updates.py OUT.json
Dumps every diagnostic (repr) and every per-parameter float64 sum after one
update for PPO (KL on, both references), damped scalar and damped categorical.
"""
import inspect
import os
from pathlib import Path
import json
import sys

import numpy as np
import torch

import manabot.backends.torch as backend
from manabot.backends.torch.learner import Learner
from manabot.backends.torch.policy import ActorCritic
from manabot.env.match import Match, Reward
from manabot.env.observation import ObservationSpace
from manabot.infra.config import MatchConfig, ModelConfig, RewardConfig
from manabot.training.regime import DampedPolicyLearning, Learning
from manabot.training.rollout.collection import SeatRoutedCollector
from manabot.training.updates.damped_policy import update_damped_iteration
from manabot.training.updates.ppo import update_iteration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "speed"))
from paired import select_arm
select_arm(os.environ.get("LOOKUP_ARM", "candidate"))

torch.set_num_threads(1)
TAKES_BACKEND = "backend" in inspect.signature(update_iteration).parameters


def run(model, learning, iteration=1):
    torch.manual_seed(7)
    space = ObservationSpace()
    agent = ActorCritic(space, model)
    collector = SeatRoutedCollector(
        space, Match(MatchConfig()), Reward(RewardConfig()), num_envs=2, seed=7,
        opponent_mode="self", learning_seats="one", batch_closure="exact-quota",
        device="cpu",
    )
    batch = collector.collect(agent, 6)
    learner = Learner(agent, "cpu", minibatch_rows=2)
    lead = (backend,) if TAKES_BACKEND else ()
    if isinstance(learning, DampedPolicyLearning):
        out = update_damped_iteration(*lead, learner, batch, learning, iteration)
    else:
        out = update_iteration(
            *lead, learner, batch, learning, 0.3, np.random.default_rng(3),
            iteration=iteration,
        )
    params = {n: repr(float(p.detach().double().sum())) for n, p in agent.named_parameters()}
    residuals = None if batch.target_residuals is None else batch.target_residuals.tobytes().hex()
    return {
        "diagnostics": repr(sorted(out.items(), key=lambda kv: kv[0])),
        "params": params,
        "target_residuals": residuals,
    }


small = dict(hidden_dim=16, num_attention_heads=2)
damped = dict(gradient="damped_policy", advantage_quantile=0.5, min_advantage=0)
result = {
    "ppo_kl_uniform": run(ModelConfig(**small), Learning(retained_fraction=0.5)),
    "ppo_kl_action_type": run(
        ModelConfig(**small),
        Learning(retained_fraction=0.5, reference="action_type_uniform"),
    ),
    "damped_scalar": run(ModelConfig(**small), DampedPolicyLearning(**damped)),
    "damped_scalar_actor_scope": run(
        ModelConfig(**small), DampedPolicyLearning(**damped, filter_scope="actor")
    ),
    "damped_categorical": run(
        ModelConfig(**small, value_kind="categorical_wdl"), DampedPolicyLearning(**damped)
    ),
}
assert result["ppo_kl_uniform"]["diagnostics"] != "[]"
json.dump(result, open(sys.argv[1], "w"), indent=1, sort_keys=True)
print({k: v["diagnostics"][:120] for k, v in result.items()})
