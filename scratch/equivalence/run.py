"""Capture policy outputs, gradients and trained weights for one checkout.

Usage: uv run python run.py OUT.pt  (with PYTHONPATH pointing at the checkout)
EQUIV_DEVICE=cuda runs the forward/backward configs on CUDA with TF32 matmul
and cuDNN allowed (`precision="tf32"`'s switches); `minimal.train` stays CPU.
Imports names shared by the integrated baseline 105e7e2aa and MTG-333.
"""

import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

warnings.simplefilter("ignore", DeprecationWarning)
torch.set_num_threads(1)
DEVICE = torch.device(os.environ.get("EQUIV_DEVICE", "cpu"))
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import manabot  # noqa: E402
from manabot.backends.torch.policy import ActorCritic  # noqa: E402
from manabot.env.match import Match  # noqa: E402
from manabot.env.observation import ObservationSpace  # noqa: E402
from manabot.infra.config import ModelConfig, ObservationConfig  # noqa: E402

try:
    from manabot.model.admission import admit_observation
    from manabot.model.inference import PolicyInference

    PREPARED_INPUT = False
except ImportError:
    from manabot.backends.torch.arrays import device_of, on_device
    from manabot.backends.torch.policy import PolicyInference

    PREPARED_INPUT = True

    def admit_observation(model, batch):
        return on_device(
            model.prepare(
                {key: value.detach().cpu().numpy() for key, value in batch.items()},
                inference=not torch.is_grad_enabled(),
            ),
            device_of(model),
        )


try:  # noqa: E402
    from manabot.backends.torch.rl import legal_mask, policy_logs
except ImportError:  # before the updates take the backend (PR A)
    from manabot.model.policy_distribution import legal_mask, policy_logs
from manabot.training.loops.minimal import train  # noqa: E402
import managym  # noqa: E402

print("manabot from", manabot.__file__, file=sys.stderr)
W = managym.WORLD_SETUP
CONFIGS = {
    "default": (ModelConfig(hidden_dim=16, world=W), 9),
    "masked_mean": (
        ModelConfig(hidden_dim=16, world=W, value_aggregation="masked_mean"),
        9,
    ),
    "history": (ModelConfig(hidden_dim=16, world=W, recent_events=True), "history"),
    "wdl": (ModelConfig(hidden_dim=16, world=W, value_kind="categorical_wdl"), 9),
    "film": (ModelConfig(hidden_dim=16, world=W, policy_film=True), 9),
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


def pooled_mana_batch(space: ObservationSpace) -> dict[str, torch.Tensor]:
    """Exercise player-owned fact programs even when sampled games omit them."""
    env = managym.Env(seed=163)
    env.reset(Match().to_rust())
    actor = env.current_agent_index()
    payload = json.loads(env.observation_for_player(actor).toJSON())
    for seat in ("agent", "opponent"):
        payload[seat]["pooled_mana_classes"] = [
            {
                "combat": False,
                "effect": "a +1/+1 counter on a non-Human creature spell it casts",
                "mana": [0, 0, 0, 0, 1, 0],
                "program": [[4, 0], [4, 1], [4, 2]],
            }
        ]
    encoded = space.encode(managym.Observation.from_json(json.dumps(payload)))
    return {key: torch.from_numpy(value[None]) for key, value in encoded.items()}


def belief_batch(
    count: int,
) -> tuple[ObservationSpace, ModelConfig, dict[str, torch.Tensor]]:
    """Real belief rows, optionally retained once for identical cross-revision input.

    EQUIV_BELIEF_INPUT names the shared corpus file. Upstream schema generation
    can change independently of a forward, so comparisons must reuse its bytes.
    """
    path = Path(os.environ["EQUIV_BELIEF_INPUT"]) if "EQUIV_BELIEF_INPUT" in os.environ else None
    if path is not None and path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        batch = saved["batch"]
        if any(value.shape[0] != count for value in batch.values()):
            raise ValueError("retained belief corpus has a different batch size")
        return (
            ObservationSpace(ObservationConfig.model_validate(saved["observation"])),
            ModelConfig.model_validate(saved["model"]),
            batch,
        )
    from manabot.belief import demo
    from manabot.belief.encoding import encode_belief
    from manabot.belief.runtime import ManabotPlayer
    from manabot.belief.state import condition_belief
    from managym.possible_worlds import WorldQuery

    env, observation = demo._runtime_env()
    viewer = int(env.last_raw_obs.agent.player_index)
    schema, policy_value = demo._schema_and_agent(env._engine, viewer)
    player = ManabotPlayer(policy_value, belief_schema=schema)
    player.start_game(env, viewer)
    player.act(env, observation)
    generated = player.last_step.belief_update.belief
    beliefs = [
        generated,
        condition_belief(generated, WorldQuery.has("Lightning Bolt")),
        condition_belief(generated, WorldQuery.lacks("Lightning Bolt")),
    ]
    views = [
        encode_belief(b, schema).to_torch_observation(batch_size=1, device="cpu")
        for b in beliefs
    ]
    rng = np.random.default_rng(7)
    rows: list[dict[str, np.ndarray]] = []
    while len(rows) < count:
        if rng.random() < 0.5:
            rows.append(observation)
        actions = int(np.asarray(observation["actions_valid"]).sum())
        observation, _, terminated, truncated, _ = env.step(int(rng.integers(actions)))
        if terminated or truncated:
            observation, _ = env.reset(seed=int(rng.integers(1 << 30)))
    batch = {
        k: torch.from_numpy(np.stack([np.asarray(r[k]) for r in rows])) for k in rows[0]
    }
    for key in views[0]:
        batch[key] = torch.cat([views[i % 3][key] for i in range(count)]).contiguous()
    config = ModelConfig(
        hidden_dim=16,
        belief_count_buckets=schema.count_buckets,
        belief_card_vocab_size=1 + max(row.card_def_id for row in schema.rows),
    )
    if path is not None:
        torch.save({
            "observation": env.observation_space.encoder.config.model_dump(mode="json"),
            "model": config.model_dump(mode="json"),
            "batch": batch,
        }, path)
    return env.observation_space, config, batch


def forward_backward(out, key, model, batch) -> None:
    model.to(DEVICE)
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    obs = admit_observation(model, batch)
    logits, values = model(obs)
    valid = obs.legal if PREPARED_INPUT else legal_mask(obs["actions_valid"])
    logs = policy_logs(logits, valid)
    chosen = valid.float().argmax(-1)
    loss = -logs.gather(-1, chosen[:, None]).mean() + values.float().square().mean()
    loss.backward()
    out[f"{key}/logits"] = logits.detach().cpu()
    out[f"{key}/values"] = values.detach().cpu()
    out[f"{key}/loss"] = loss.detach().cpu()
    for pname, p in model.named_parameters():
        if p.grad is not None:
            out[f"{key}/grad/{pname}"] = p.grad.detach().cpu().clone()


def inference(out, key, model, batch) -> None:
    """The serving path: a PolicyInference scope, no autograd, eval mode."""
    model.eval()
    for rows in (24, 1):
        with torch.no_grad(), PolicyInference(model) as scope:
            host = {k: v[:rows].clone() for k, v in batch.items()}
            obs = scope.prepare(
                {k: v.cpu().numpy() for k, v in host.items()}
                if PREPARED_INPUT
                else host
            )
            logits, values = model(obs)
        out[f"{key}/infer{rows}/logits"] = logits.cpu()
        out[f"{key}/infer{rows}/values"] = values.cpu()
    model.train()


def capture() -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    space = ObservationSpace()
    for bucketed in (False, True):
        torch.manual_seed(0)
        model = ActorCritic(space, CONFIGS["default"][0], bucketed_shapes=bucketed)
        key = f"pooled_mana/{'bucketed' if bucketed else 'padded'}"
        forward_backward(out, key, model, pooled_mana_batch(space))
        inference(out, key, model, pooled_mana_batch(space))
    space, config, batch = belief_batch(24)
    for bucketed in (False, True):
        torch.manual_seed(0)
        model = ActorCritic(space, config, bucketed_shapes=bucketed)
        model.train()
        forward_backward(
            out, f"belief/{'bucketed' if bucketed else 'padded'}", model, batch
        )
        inference(out, f"belief/{'bucketed' if bucketed else 'padded'}", model, batch)
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
            for bucketed in (False, True):
                for live in (False, True) if config.attention_on else (False,):
                    key = f"{name}/{'bucketed' if bucketed else 'padded'}{'-live' if live else ''}"
                    torch.manual_seed(0)
                    model = ActorCritic(
                        space, config, bucketed_shapes=bucketed, live_tokens=live
                    )
                    model.train()
                    forward_backward(out, key, model, batch)
                    inference(out, key, model, batch)
    for name, config in {
        "masked_mean": ModelConfig(hidden_dim=16, value_aggregation="masked_mean"),
        "default": ModelConfig(hidden_dim=16),
    }.items():
        trained = train(2, streams=2, decisions=24, seed=5, model=config)
        for pname, value in trained.state_dict().items():
            out[f"train/{name}/{pname}"] = value.detach().clone()
    return out


if __name__ == "__main__":
    if sys.argv[1] == "--belief-input":
        os.environ["EQUIV_BELIEF_INPUT"] = sys.argv[2]
        belief_batch(24)
    else:
        torch.save(capture(), sys.argv[1])
