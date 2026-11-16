"""Torch-specific headless opponent-memory policy and update proofs.

The CLI selects this backend; native replay and text verification stay in
manabot.verify.opponent_memory. Controlled FiLM weights test connectivity,
while the ordinary local recipe separately exercises one unmodified update.
"""

from pathlib import Path

import numpy as np
import torch

from manabot.backends import resolve
from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from manabot.env.opponent_memory import blank_memory
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.recipes.local import local_smoke
from manabot.training.execution import execute_regime
from manabot.training.regime import TrainingRegime, TrainSelfPlay
from manabot.verify.store import VerifyStore
import managym


def prove_policies(
    space: ObservationSpace,
    obs: dict[str, np.ndarray],
    out: Path,
    player_configs: list[managym.PlayerConfig],
) -> list[dict[str, object]]:
    torch.set_num_threads(1)
    blank = blank_memory(obs)
    for key in obs.keys() - {"opponent_records", "opponent_details"}:
        np.testing.assert_array_equal(obs[key], blank[key])
    results: list[dict[str, object]] = []
    for encoder in ("transformer", "gru"):
        for summaries in (1, 4):
            torch.manual_seed(221)
            model = ActorCritic(
                space,
                ModelConfig.model_validate(
                    {
                        "world": managym.WORLD_SETUP,
                        "hidden_dim": 16,
                        "opponent_memory": encoder,
                        "opponent_memory_summaries": summaries,
                        "attention_on": True,
                        "attention_layers": 2,
                        "policy_film": True,
                    }
                ),
            ).eval()
            # FiLM initializes at zero. A declared controlled weight intervention
            # exercises the learned actor path, without a claim about training.
            assert model.policy_film is not None
            with torch.no_grad():
                model.policy_film.weight.normal_(0, 0.1)
            logits, value = model.forward_distribution(
                on_device(model.prepare(obs), "cpu")
            )
            blank_logits, _ = model.forward_distribution(
                on_device(model.prepare(blank), "cpu")
            )
            delta = float(
                (logits.softmax(-1) - blank_logits.softmax(-1)).abs().max().detach()
            )
            assert delta > 1e-8
            gradients: dict[str, float] = {}
            assert model.opponent_memory is not None
            temporal = (
                model.opponent_memory.gru
                if encoder == "gru"
                else model.opponent_memory.transformer
            )
            assert temporal is not None
            for name, loss in (
                ("actor", logits.softmax(-1)[0, 0]),
                ("value", value.sum()),
            ):
                model.zero_grad(set_to_none=True)
                loss.backward(retain_graph=True)
                grads = [p.grad for p in temporal.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                gradients[name] = sum(float(g.abs().sum()) for g in grads)
                assert gradients[name] > 0
            path = out / f"{encoder}-k{summaries}.safetensors"
            save_checkpoint(model, space, path, player_configs=player_configs)
            restored, space = load_checkpoint(str(path))
            assert space == space
            actual = restored.forward_distribution(
                on_device(restored.prepare(obs), "cpu")
            )
            for a, b in zip(actual, (logits, value), strict=True):
                torch.testing.assert_close(a, b)
            results.append(
                {
                    "encoder": encoder,
                    "summaries": summaries,
                    "legal_probability_max_delta": delta,
                    "temporal_gradient_l1": gradients,
                    "checkpoint_reload_matches": True,
                }
            )
    return results


def ordinary_update(out: Path) -> dict[str, object]:
    regime = local_smoke(steps=1, device="cpu")
    regime.id = "opponent-memory-software-demo"
    regime.agent.opponent_memory = "gru"
    regime.agent.opponent_memory_summaries = 1
    regime.observation = ObservationConfig(
        max_opponent_records=1024, max_opponent_details=1048576
    )
    stage = regime.stages[0]
    assert isinstance(stage, TrainSelfPlay)
    stage.execution.collection_rows = 2
    stage.execution.bootstrap_rows = 2
    stage.execution.learner_rows = 2
    regime = TrainingRegime.model_validate(regime.model_dump())
    with VerifyStore(out / "training.sqlite") as store:
        run = execute_regime(
            regime, 1, out / "training", store, backend=resolve("torch")
        )
    assert run.status == "completed" and run.updates_through() == 1
    model, _ = load_checkpoint(run.stages[0].artifacts["raw"]["path"])
    assert model.opponent_memory is not None
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    return {
        "status": run.status,
        "updates": 1,
        "learner_transitions": run.stages[0].learner_transitions,
        "checkpoint_reload_finite": True,
    }
