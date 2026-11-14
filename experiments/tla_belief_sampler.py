"""Inspect closed TLA sampler evidence and reproduce the saved seed357 demo.

Collection and admission require a compatible Etude controller and the original
frozen worker bundle. They are not available from this CLI yet. Demo reads only
closed evidence and saved candidates; it never collects games or optimizes.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch

from manabot.belief.checkpoint import load_belief_sampler
from manabot.belief.sampling import sample_physical_deal
from manabot.belief.sampling_collection import _write_private
from manabot.belief.sampling_evidence import EvidenceRoot
from manabot.infra.torch import deterministic_cpu
from manabot.reports.belief import report_saved_sampler
from managym.decision import PUBLIC_COMMITMENT_KINDS


def demo(directory: Path) -> Path:
    """Verify the declared demo candidate and retain private reports/public inputs.

    Choose the first saved test observation with nonempty public history,
    irrespective of score or predicted sensitivity. Predictions contain no labels, seeds
    or replay commands. History is the model's public count vector, not a trace.
    """
    if not directory.is_dir():
        raise FileNotFoundError("saved evidence root is unavailable")
    root = EvidenceRoot(directory)
    dataset = root.dataset()
    fit = directory / "fits/357"
    receipt = json.loads((fit / "complete.json").read_text())
    binding = json.loads((fit / "started.json").read_text())
    if (
        receipt["binding"] != binding
        or binding["dataset_identity"] != dataset.identity
        or binding["seed"] != 357
    ):
        raise ValueError("demo fit binding changed")
    checkpoint = fit / "sampler.pt"
    with deterministic_cpu(), torch.random.fork_rng(devices=[]):
        model = load_belief_sampler(
            checkpoint,
            expected_dataset_identity=dataset.identity,
            expected_policy_identity=dataset.policy_identity,
            expected_schema_identity=dataset.schema.identity,
            expected_world_identity=dataset.world_identity,
            expected_checkpoint_identity=receipt["checkpoint_identity"],
        )
        output = directory / "demo"
        output.mkdir(exist_ok=True, mode=0o700)
        report = output / "quality.json"
        report_binding = output / "quality-binding.json"
        if not report.exists():
            quality = report_saved_sampler(
                directory / "dataset.json",
                checkpoint=checkpoint,
                checkpoint_identity=receipt["checkpoint_identity"],
                training_dataset_path=directory / "dataset.json",
                sample_counts=(16, 64),
                seed=357,
            )
            if quality.training_seed != 357:
                raise ValueError("saved checkpoint is not the declared demo fit seed")
            _write_private(report, asdict(quality))
            _write_private(
                report_binding,
                {
                    "dataset_identity": dataset.identity,
                    "checkpoint_identity": receipt["checkpoint_identity"],
                    "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                },
            )
        if json.loads(report_binding.read_text()) != {
            "dataset_identity": dataset.identity,
            "checkpoint_identity": receipt["checkpoint_identity"],
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        }:
            raise ValueError("saved demo report identity changed")
        inputs = next(
            (
                row.inputs
                for game in dataset.games
                if game.split == "test"
                for row in game.examples
                if any(row.inputs.history_features)
            ),
            None,
        )
        if inputs is None:
            raise ValueError(
                "history example unavailable: no retained test row has public commitments"
            )
        zero = replace(inputs, history_features=(0.0,) * len(inputs.history_features))
        # Matched sampling seeds describe sensitivity, not a controlled ablation.
        with torch.no_grad():
            predictions = {
                "prior": sample_physical_deal(
                    [inputs] * 64, generator=torch.Generator().manual_seed(357)
                ),
                "learned": model.sample(
                    [inputs] * 64, generator=torch.Generator().manual_seed(357)
                ),
                "zero_history": model.sample(
                    [zero] * 64, generator=torch.Generator().manual_seed(357)
                ),
            }
        public = {
            "description": "First retained test observation with public commitments; 64 samples per arm. Zero history illustrates sensitivity, not an ablation.",
            "card_names": dataset.schema.card_names,
            "public_inputs": asdict(inputs),
            "public_commitment_features": [
                {
                    "actor": ("viewer", "opponent")[
                        index
                        // (
                            len(PUBLIC_COMMITMENT_KINDS)
                            * (len(dataset.schema.card_names) + 1)
                        )
                    ],
                    "kind": PUBLIC_COMMITMENT_KINDS[
                        (index // (len(dataset.schema.card_names) + 1))
                        % len(PUBLIC_COMMITMENT_KINDS)
                    ],
                    "card": (None, *dataset.schema.card_names)[
                        index % (len(dataset.schema.card_names) + 1)
                    ],
                    "normalized_count": value,
                }
                for index, value in enumerate(inputs.history_features)
                if value
            ],
            "inclusion_probabilities": {
                name: (draws > 0).double().mean(dim=0).tolist()
                for name, draws in predictions.items()
            },
        }
        path = output / "public-example.json"
        if path.exists():
            if json.loads(path.read_text()) != json.loads(json.dumps(public)):
                raise ValueError("saved public demo differs from admitted candidate")
        else:
            _write_private(path, public)
        lines = [
            "# Saved sampler example",
            "",
            public["description"],
            "",
            "Public history features are normalized commitment counts; event order is unavailable.",
            "",
            "| Card | Prior | Learned | Zero history |",
            "| --- | ---: | ---: | ---: |",
        ]
        probabilities = public["inclusion_probabilities"]
        for index, name in enumerate(dataset.schema.card_names):
            if inputs.pool_counts[index]:
                values = [
                    probabilities[arm][index]
                    for arm in ("prior", "learned", "zero_history")
                ]
                lines.append(
                    f"| {name} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |"
                )
        lines.extend(["", "Public commitment features:", ""])
        lines.extend(
            f"- {feature['actor']}: {feature['kind']}; {feature['card'] or 'no card'}; normalized count {feature['normalized_count']:.6g}"
            for feature in public["public_commitment_features"]
        )
        text = "\n".join(lines) + "\n"
        markdown = output / "public-example.md"
        if markdown.exists():
            if markdown.read_text() != text:
                raise ValueError("saved public text demo changed")
        else:
            with markdown.open("x") as destination:
                destination.write(text)
        return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    saved = commands.add_parser(
        "demo", help="reload seed357 and report saved observations"
    )
    saved.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args(argv)
    print(demo(args.artifacts))


if __name__ == "__main__":
    main()
