"""Torch-specific headless decision-memory policy and update proofs.

The CLI selects this backend; native replay and text verification stay in
manabot.verify.decision_memory. Controlled FiLM weights test connectivity,
while the ordinary local recipe separately exercises one unmodified update.
"""

from pathlib import Path

import numpy as np
import torch

from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.checkpoint import load_checkpoint, save_checkpoint
from manabot.backends.torch.policy import ActorCritic
from manabot.env.decision_memory import MEMORY_FIELDS
from manabot.env.match import Match
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from manabot.recipes.builders import gray_ogre_self_play
from manabot.training.loops.self_play import (
    Counts,
    SelfPlay,
    create,
    root_match,
    self_play_collector,
    step,
)
from manabot.training.rollout.collection import close_collector
import managym


def prove_policies(
    space: ObservationSpace,
    obs: managym.ObservationBatch,
    out: Path,
    player_configs: managym.GameSetup,
) -> list[dict[str, object]]:
    torch.set_num_threads(1)
    blank = obs.blank_decision_memory()
    original, ablated = obs.tables(), blank.tables()
    for key in original.keys() - set(MEMORY_FIELDS):
        assert original[key] == ablated[key], key
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
                        "decision_memory": encoder,
                        "decision_memory_summaries": summaries,
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
                for layer in (model.attention, *model.extra_attention):
                    assert layer.memory_gate is not None
                    layer.memory_gate.fill_(0.5)
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
            assert model.decision_memory is not None
            temporal = (
                model.decision_memory.gru
                if encoder == "gru"
                else model.decision_memory.transformer
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
            restored, restored_space = load_checkpoint(str(path))
            assert restored_space == space
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
    """One unmodified self-play update of a GRU-memory policy, saved and reloaded."""
    settings = gray_ogre_self_play(horizon=1).model_copy(
        update={
            "model": ModelConfig(
                world=managym.WORLD_SETUP,
                hidden_dim=16,
                num_attention_heads=2,
                attention_on=True,
                policy_film=True,
                decision_memory="gru",
            ),
            "observation": ObservationConfig(
                max_decision_records=1024,
                max_decision_reference_details=8192,
                max_decision_references=256,
                max_decision_members=16384,
            ),
            "collection_rows": 2,
            "bootstrap_rows": 2,
            "learner_rows": 2,
        }
    )
    settings = SelfPlay.model_validate(settings.model_dump())
    learner, ema = create(settings, seed=221)
    counts = Counts()
    collector = self_play_collector(settings, seed=221, start=0)
    try:
        step(
            settings,
            learner,
            ema,
            collector,
            counts,
            np.random.default_rng(221),
            out=out / "training",
        )
    finally:
        close_collector(collector)
    path = out / "training" / "raw.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        learner.agent,
        ObservationSpace(settings.observation),
        path,
        player_configs=Match(root_match(settings)).to_rust(),
    )
    model, _ = load_checkpoint(str(path))
    assert model.decision_memory is not None
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    return {
        "status": "completed",
        "updates": counts.updates,
        "learner_transitions": counts.learner_transitions,
        "checkpoint_reload_finite": True,
    }
