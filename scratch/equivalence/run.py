"""Capture policy outputs, gradients and trained weights for one checkout.

Usage: python run.py OUT.pt  (with PYTHONPATH pointing at the checkout)
Imports only names that resolve at both 05c1203e9 and MTG-325 HEAD.
"""

import sys
import warnings

import numpy as np
import torch

warnings.simplefilter("ignore", DeprecationWarning)
torch.set_num_threads(1)

import manabot  # noqa: E402
from manabot.env.match import Match  # noqa: E402
from manabot.env.observation import ObservationSpace  # noqa: E402
from manabot.infra.config import ModelConfig, ObservationConfig  # noqa: E402
from manabot.model.actor_critic import ActorCritic  # noqa: E402
from manabot.model.admission import admit_observation  # noqa: E402
from manabot.model.policy_distribution import legal_mask, policy_logs  # noqa: E402
from manabot.training.loops.minimal import train  # noqa: E402
import managym  # noqa: E402

print("manabot from", manabot.__file__, file=sys.stderr)
W = managym.WORLD_SETUP
CONFIGS = {
    "default": (ModelConfig(hidden_dim=16, world=W), 9),
    "masked_mean": (ModelConfig(hidden_dim=16, world=W, value_aggregation="masked_mean"), 9),
    "history": (ModelConfig(hidden_dim=16, world=W, recent_events=True), "history"),
    "wdl": (ModelConfig(hidden_dim=16, world=W, value_kind="categorical_wdl"), 9),
    "film": (ModelConfig(hidden_dim=16, world=W, policy_film=True), 9),
    "pooled": (ModelConfig(hidden_dim=16, world=W, token_stream="pooled"), 9),
    "no_attention": (ModelConfig(hidden_dim=16, world=W, attention_on=False), 9),
    "two_layers": (ModelConfig(hidden_dim=16, world=W, attention_layers=2), 9),
    "hashed": (ModelConfig(hidden_dim=16, world=W, record_vocabulary="hashed"), 9),
    "no_world": (ModelConfig(hidden_dim=16), 9),
    "v10": (ModelConfig(hidden_dim=16, world=W), 10),
}


def observations(space: ObservationSpace, count: int) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(3)
    rows: list[dict[str, np.ndarray]] = []
    env = managym.Env(seed=11)
    env.reset(Match().to_rust())
    while len(rows) < count:
        agent = env.current_agent_index()
        obs = env.observation_for_player(agent)
        if len(rows) < count and rng.random() < 0.5:
            rows.append(space.encode(obs))
        actions = obs.action_space.actions
        _, _, terminated, truncated, _ = env.step(int(rng.integers(len(actions))))
        if terminated or truncated:
            env.reset(Match().to_rust())
    return {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0]}


def capture() -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    spaces = {
        9: ObservationConfig(),
        10: ObservationConfig(policy_input_version=10),
        "history": ObservationConfig(policy_history_version=1),
    }
    for version, observation_config in spaces.items():
        space = ObservationSpace(observation_config)
        batch = observations(space, 24)
        for name, (config, cfg_version) in CONFIGS.items():
            if cfg_version != version:
                continue
            for bucketed in (False, True) if config.token_stream == "tables" else (False,):
                key = f"{name}/{'bucketed' if bucketed else 'padded'}"
                torch.manual_seed(0)
                model = ActorCritic(space, config, bucketed_shapes=bucketed)
                model.train()
                obs = admit_observation(model, batch)
                logits, values = model(obs)
                logs = policy_logs(logits, legal_mask(obs["actions_valid"]))
                chosen = legal_mask(obs["actions_valid"]).float().argmax(-1)
                loss = -logs.gather(-1, chosen[:, None]).mean() + values.float().square().mean()
                loss.backward()
                out[f"{key}/logits"] = logits.detach()
                out[f"{key}/values"] = values.detach()
                out[f"{key}/loss"] = loss.detach()
                for pname, p in model.named_parameters():
                    if p.grad is not None:
                        out[f"{key}/grad/{pname}"] = p.grad.detach().clone()
    for name, config in {
        "masked_mean": ModelConfig(hidden_dim=16, value_aggregation="masked_mean"),
        "default": ModelConfig(hidden_dim=16),
    }.items():
        trained = train(2, streams=2, decisions=24, seed=5, model=config)
        for pname, value in trained.state_dict().items():
            out[f"train/{name}/{pname}"] = value.detach().clone()
    return out


if __name__ == "__main__":
    torch.save(capture(), sys.argv[1])
