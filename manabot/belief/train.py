"""Fit the belief sampler and atomically publish an admitted checkpoint."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from manabot.belief.checkpoint import load_belief_sampler, save_belief_sampler
from manabot.belief.protocol import SamplerCodec
from manabot.belief.sampling_data import read_dataset
from manabot.belief.sampling_fit import BeliefTrainer, fit_belief_sampler


class SamplerConfig(BaseModel):
    """How the sampler is fit (`fit_belief_sampler`)."""

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(default=32, ge=1)
    batch_size: int = Field(default=32, ge=1)
    hidden_size: int = Field(default=32, ge=2)
    learning_rate: float = Field(default=0.001, gt=0)
    history_dropout: float = Field(default=0, ge=0, le=1)
    evaluation_samples: int = Field(default=32, ge=1)


def train_belief(
    histories: Path,
    out: Path,
    config: SamplerConfig,
    *,
    seed: int,
    backend: BeliefTrainer,
    codec: SamplerCodec,
) -> Path:
    """Fit on `histories` and write the sampler to `out`; the policy is untouched.

    `backend` fits the sampler; `codec` exports and restores its parameters.
    """
    dataset = read_dataset(histories)
    result = fit_belief_sampler(
        dataset, **config.model_dump(), seed=seed, backend=backend
    )
    identity = save_belief_sampler(out, result, dataset, codec=codec)
    load_belief_sampler(
        out,
        codec=codec,
        expected_dataset_identity=dataset.identity,
        expected_policy_identity=dataset.policy_identity,
        expected_schema_identity=dataset.schema.identity,
        expected_world_identity=dataset.world_identity,
        expected_checkpoint_identity=identity,
    )
    return out
