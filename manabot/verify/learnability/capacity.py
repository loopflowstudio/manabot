"""One mixed supervised fit per seed, then independent frozen-policy assertions.

Loss is mean cross entropy over every fit case; all parameters receive gradients.
Public life selects either preference. This is representational capacity only.
Held-out native seeds/life totals never enter optimization. Input ablations remove
only the audited identity channels; they are not historical checkpoint evidence.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from manabot.env import ObservationSpace
from manabot.infra.hypers import AgentSpec
from manabot.model import Agent
from manabot.verify.action_cases import (
    DecisionCase,
    curriculum,
    identity_removed,
    reorder_objects,
    reorder_offers,
)
from manabot.verify.learnability.report import save
from manabot.verify.learnability.schema import Config, Result, Score
from manabot.verify.learning_control import run_positive_control
import managym


def _digest_cases(cases: list[DecisionCase]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(
            f"{case.family}/{case.preference}/{case.split}/{case.variant}".encode()
        )
        for key, value in sorted(case.observation.items()):
            digest.update(key.encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _score(
    agent: Agent, case: DecisionCase, seed: int, initial: float, config: Config
) -> Score:
    obs = case.observation
    logits, _ = agent(obs)
    probabilities = logits.softmax(-1)
    ablated, _ = agent(identity_removed(case))
    a, b = case.candidates[:2]
    gap = float((ablated[0, a] - ablated[0, b]).abs())
    count = int(obs["actions_valid"].sum())
    # Seeded shuffle, independently of preference; never a positional label.
    order = torch.randperm(
        count, generator=torch.Generator().manual_seed(seed + case.variant)
    )
    actual, _ = agent(reorder_offers(obs, order))
    offer_error = float(
        (actual.softmax(-1)[:, :count] - probabilities[:, order]).abs().max()
    )
    actual, _ = agent(reorder_objects(obs))
    object_error = float((actual.softmax(-1) - probabilities).abs().max())
    trained = float(probabilities[0, case.preferred])
    return Score(
        seed=seed,
        family=case.family,
        split=case.split,
        variant=case.variant,
        preference=case.preference,
        native=case.native,
        initial=initial,
        no_update=initial,
        trained=trained,
        identity_removed=float(ablated.softmax(-1)[0, case.preferred]),
        ablation_gap=gap,
        offer_error=offer_error,
        object_error=object_error,
        passed=trained >= config.probability
        and gap <= config.invariance_tolerance
        and max(offer_error, object_error) <= config.invariance_tolerance,
    )


def run(out: Path, result: Result) -> None:
    torch.set_num_threads(1)
    result.world = managym.WORLD_VERSION
    result.native_sha256 = hashlib.sha256(
        Path(managym._managym.__file__).read_bytes()
    ).hexdigest()
    result.torch_version = torch.__version__
    cases = curriculum()
    result.cases_digest = _digest_cases(cases)
    case_path = out / "cases.json"
    case_path.write_text(
        json.dumps(
            [
                {
                    "family": c.family,
                    "split": c.split,
                    "variant": c.variant,
                    "preference": c.preference,
                    "candidates": c.candidates,
                    "native": c.native,
                    "outcomes": [asdict(o) for o in c.outcomes],
                }
                for c in cases
            ],
            indent=2,
        )
    )
    result.artifacts[case_path.name] = hashlib.sha256(
        case_path.read_bytes()
    ).hexdigest()
    fit = [c for c in cases if c.split == "fit"]
    result.fit_cases = len(fit)
    result.held_out_cases = len(cases) - len(fit)
    batch = {
        key: torch.cat([c.observation[key] for c in fit]) for key in fit[0].observation
    }
    labels = torch.tensor([c.preferred for c in fit])
    spec = AgentSpec(
        hidden_dim=result.config.hidden_dim,
        num_attention_heads=2,
        attention_feedforward_dim=64,
    )
    for seed in result.config.seeds:
        torch.manual_seed(seed)
        agent = Agent(ObservationSpace(), spec)
        agent.eval()
        with torch.no_grad():
            initial = [
                float(agent(c.observation)[0].softmax(-1)[0, c.preferred])
                for c in cases
            ]
        optimizer = torch.optim.Adam(agent.parameters(), lr=result.config.learning_rate)
        started = time.perf_counter()
        agent.train()
        for _ in range(result.config.updates):
            logits, _ = agent(batch)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        result.training_seconds[str(seed)] = time.perf_counter() - started
        path = out / f"capacity-{seed}.pt"
        # Deliberately diagnostic: no fake single-match world binding. The PPO
        # layer below supplies ordinary production checkpoint admission/reload.
        torch.save(
            {
                "state_dict": agent.state_dict(),
                "agent": spec.model_dump(),
                "config": result.config.model_dump(),
                "source": result.source,
                "source_digest": result.source_digest,
                "world": result.world,
                "cases_digest": result.cases_digest,
                "seed": seed,
            },
            path,
        )
        restored = Agent(ObservationSpace(), spec)
        restored.load_state_dict(torch.load(path, weights_only=True)["state_dict"])
        restored.eval()
        result.artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        result.scores.extend(
            _score(restored, case, seed, before, result.config)
            for case, before in zip(cases, initial, strict=True)
        )
        save(out, result)
    if result.config.ppo:
        ppo = run_positive_control(out / "ppo", deadline=time.perf_counter() + 150)
        result.ppo = ppo
        path = out / "ppo" / ppo.checkpoint
        result.artifacts[str(path.relative_to(out))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    else:
        result.skips.append("Production PPO omitted in explicit PR capacity subset.")
    result.status = (
        "passed"
        if all(s.passed for s in result.scores)
        and (result.ppo is None or result.ppo.passed)
        else "failed"
    )
