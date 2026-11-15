"""Random draft/construction policy and separately bound artifacts.

Both phases share one attention network. The policy reads Prepared rows only;
all legality and catalog projection runs on the host. These are draft artifacts,
never game-policy checkpoints or a compatibility path for them.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from manabot.backends.torch.arrays import on_device
from manabot.infra.source import source_commit, source_dirty
from manabot.policy.draft import DraftPrepared

if TYPE_CHECKING:
    from manabot.backends.torch.learner import Learner
    from manabot.policy.draft import DraftTrajectory


@dataclass(frozen=True)
class DraftConfig:
    catalog_size: int
    seats: int = 8
    width: int = 64
    layers: int = 2
    heads: int = 4


class DraftPolicy(nn.Module):
    def __init__(self, config: DraftConfig, cards: np.ndarray, binding: str) -> None:
        super().__init__()
        self.config, self.binding = config, binding
        self.register_buffer("cards", torch.as_tensor(cards.copy(), dtype=torch.long))
        self.embedding = nn.Embedding(config.catalog_size, config.width)
        self.features = nn.Linear(4 + config.seats, config.width)
        self.position = nn.Linear(7, config.width)
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, config.width * 2, dropout=0, batch_first=True
        )
        self.attention = nn.TransformerEncoder(
            layer, config.layers, enable_nested_tensor=False
        )
        self.output = nn.Linear(config.width, 1)

    @property
    def identity(self) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {"config": asdict(self.config), "binding": self.binding}, sort_keys=True
            ).encode()
        )
        for name, tensor in self.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def forward(self, batch: DraftPrepared[Tensor]) -> Tensor:
        if (
            not isinstance(batch, DraftPrepared)
            or batch.binding != self.binding
            or not torch.equal(batch.cards, self.cards)
        ):
            raise ValueError("draft Prepared binding/card axis differs")
        # Fixed scales preserve counts and seat identities, without treating capacity as vocabulary.
        tokens = (
            self.embedding(batch.cards)[None]
            + self.features(batch.features / 45)
            + self.position(batch.position / 360)[:, None]
        )
        return (
            self.output(self.attention(tokens))
            .squeeze(-1)
            .masked_fill(~batch.legal, -torch.inf)
        )

    def logits(self, batch: DraftPrepared[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            return self(on_device(batch, self.cards.device)).cpu().numpy()

    def log_probability(
        self, batches: list[DraftPrepared[np.ndarray]], actions: list[int]
    ) -> Tensor:
        """Recompute in bounded chunks; checkpoint activations, never detach credit."""
        total = torch.zeros((), device=self.cards.device)
        for offset in range(0, len(batches), 8):
            group = batches[offset : offset + 8]
            batch = DraftPrepared(
                group[0].cards,
                np.concatenate([b.features for b in group]),
                np.concatenate([b.position for b in group]),
                np.concatenate([b.legal for b in group]),
                self.binding,
            )
            moved = on_device(batch, self.cards.device)
            chosen = torch.as_tensor(
                actions[offset : offset + 8], device=self.cards.device
            )

            def score(
                features: Tensor, position: Tensor, legal: Tensor, chosen: Tensor
            ) -> Tensor:
                prepared = DraftPrepared(
                    self.cards, features, position, legal, self.binding
                )
                return self(prepared).log_softmax(-1).gather(1, chosen[:, None]).sum()

            total = total + checkpoint(
                score,
                moved.features,
                moved.position,
                moved.legal,
                chosen,
                use_reentrant=False,
            )
        return total


def save(path: Path, policy: DraftPolicy) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema": "rotisserie-policy-1",
            "source_commit": source_commit(),
            "source_dirty": source_dirty(),
            "config": asdict(policy.config),
            "binding": policy.binding,
            "state": policy.state_dict(),
            "parameter_digest": policy.identity,
        },
        temporary,
    )
    os.replace(temporary, path)


def load(path: Path, *, binding: str, cards: np.ndarray) -> DraftPolicy:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (
        data["schema"] != "rotisserie-policy-1"
        or data["binding"] != binding
        or not np.array_equal(data["state"]["cards"].numpy(), cards)
    ):
        raise ValueError("draft artifact schema/catalog/printing binding differs")
    policy = DraftPolicy(DraftConfig(**data["config"]), cards, binding)
    policy.load_state_dict(data["state"], strict=True)
    return policy.eval()


def update(
    learner: "Learner",
    trajectories: list["DraftTrajectory"],
    rewards: list[float],
    baseline: float,
) -> float:
    """REINFORCE terminal returns; each reward multiplies only its seat's log sum.

    No critic, bootstraps, advantage normalization or imitation term. All
    non-learner and failed-bundle rows are absent before building this graph.
    """
    if not isinstance(learner.agent, DraftPolicy):
        raise TypeError("rotisserie update requires a draft policy")
    policy = learner.agent
    learner.set_learning_rate(1e-4)

    def objective() -> tuple[Tensor, ...]:
        losses = [
            (baseline - reward)
            * policy.log_probability(list(t.inputs), list(t.actions))
            for t, reward in zip(trajectories, rewards, strict=True)
        ]
        return (torch.stack(losses).mean(),)

    (loss,) = learner.loss(objective)
    learner.step(loss, 1)
    return float(loss.detach())


def create(
    config: DraftConfig, cards: np.ndarray, binding: str, seed: int
) -> DraftPolicy:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return DraftPolicy(config, cards, binding)
