"""Exploratory equal-update head-to-head, with each native decision mode explicit.

Run on a bounded worker or the Mac mini. The parent launches one
subprocess per decision mode so native switches are set before engine import.
Each pair retains ordinary evaluation requests, Commands and replay results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


@dataclass(frozen=True)
class CheckpointInput:
    name: str
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class PanelInputs:
    updates: int
    training_seed: int
    baseline: CheckpointInput
    candidate: CheckpointInput


def checkpoint_inputs(directory: Path) -> PanelInputs:
    """Read explicit hash-bound inputs, preserving the original u50 defaults."""
    manifest = os.environ.get("QUICK_EVAL_MANIFEST")
    if manifest is None:
        return PanelInputs(
            50,
            26502,
            CheckpointInput(
                "baseline-u50",
                directory / "baseline-u50.pt",
                "7250ccd5fec785bd2b305e9a180c93af2e70ade1ca0b0f5a40a54d010e584853",
                4927595,
            ),
            CheckpointInput(
                "mana-taps-u50",
                directory / "mana-taps-u50.pt",
                "6a9b27d0c2b334ad0fe00256474dfcb358918c141774f0aead7e4d1fdd70f4e2",
                4927595,
            ),
        )
    raw = json.loads(Path(manifest).read_text())
    if (
        not isinstance(raw, dict)
        or type(raw.get("updates")) is not int
        or type(raw.get("training_seed")) is not int
    ):
        raise ValueError("checkpoint manifest needs integer updates and training_seed")

    def read(role: str) -> CheckpointInput:
        row = raw.get(role)
        if (
            not isinstance(row, dict)
            or not all(isinstance(row.get(k), str) for k in ("name", "file", "sha256"))
            or type(row.get("bytes")) is not int
        ):
            raise ValueError(f"invalid checkpoint manifest role: {role}")
        if Path(row["file"]).name != row["file"] or row["bytes"] <= 0:
            raise ValueError("checkpoint file must be a basename with positive bytes")
        return CheckpointInput(
            row["name"], directory / row["file"], row["sha256"], row["bytes"]
        )

    result = PanelInputs(
        raw["updates"], raw["training_seed"], read("baseline"), read("candidate")
    )
    if result.updates <= 0 or result.baseline.name == result.candidate.name:
        raise ValueError("checkpoint coordinates and distinct player names required")
    return result


def panel(mode: str, output: Path, deadline: float) -> None:
    import torch

    from manabot.eval.admission import register_checkpoint
    from manabot.eval.evaluation import EvaluationRequest, evaluate
    from manabot.eval.models import EvaluationKey
    from manabot.infra.identity import canonical_sha256
    from manabot.infra.files import atomic_json
    from manabot.sim.execution import EvalExecution
    from manabot.sim.matchup import DeckMatchup
    from managym import Deck, WORLD_VERSION

    torch.set_num_threads(1)
    source = Path(__file__).resolve().parents[1]
    held = json.loads(
        (source / "experiments/regimes/decision-shapes-held-out.json").read_text()
    )
    excluded = "Yue, the Moon Spirit"
    kept = [
        index
        for index, deck in enumerate(held["decks"])
        if excluded not in deck["decklist"] and excluded not in deck["sideboard"]
    ]
    pair_count = int(os.environ.get("QUICK_EVAL_PAIRS", "8"))
    if not 1 <= pair_count <= 64:
        raise ValueError("pair count must be between one and64")
    pairs = random.Random(26593001).sample(list(combinations(kept, 2)), pair_count)
    inputs = Path(os.environ["ETUDE_INPUTS_DIR"])
    declared = checkpoint_inputs(inputs)
    baseline, candidate = declared.baseline.name, declared.candidate.name
    paths = {
        checkpoint.name: str(checkpoint.path)
        for checkpoint in (declared.baseline, declared.candidate)
    }
    summary: dict[str, object] = {
        "mode": mode,
        "updates": declared.updates,
        "training_seed": declared.training_seed,
        "players": [baseline, candidate],
        "held_out_total": len(held["decks"]),
        "held_out_retained": len(kept),
        "excluded_card": excluded,
        "expected_games": len(pairs) * 4,
        "pairs": [],
    }
    rows: list[dict[str, object]] = []
    output.mkdir(parents=True, exist_ok=True)
    for number, (first, second) in enumerate(pairs):
        if time.time() >= deadline - 150:
            break
        matchup = DeckMatchup(
            population_sha256=held["corpus_sha256"],
            entries=(held["entry_ids"][first], held["entry_ids"][second]),
            lists=(
                Deck(
                    held["decks"][first]["decklist"], held["decks"][first]["sideboard"]
                ),
                Deck(
                    held["decks"][second]["decklist"],
                    held["decks"][second]["sideboard"],
                ),
            ),
        )
        registrations = {
            name: register_checkpoint(
                Path(path),
                match=matchup.match(),
                content_suite=matchup.content_suite,
                player_id=name,
                display_name=name,
                training_seed=declared.training_seed,
                artifact_prefix=f"quick-eval/{name}",
                expected_sha256=checkpoint.sha256,
                expected_bytes=checkpoint.size_bytes,
            ).registration
            for checkpoint in (declared.baseline, declared.candidate)
            for name, path in [(checkpoint.name, str(checkpoint.path))]
        }
        key = EvaluationKey(
            world=WORLD_VERSION,
            content_suite=matchup.content_suite,
            viewer_boundary="acting-viewer",
            arena_version=f"decision-shapes-quick-u{declared.updates}-{mode}",
            rating_model_version="unrated",
            rating_prior_sha256=canonical_sha256({}),
            anchor_cohort_sha256=registrations[baseline].identity_sha256,
            evaluation_compute_envelope_id="policy-cpu-one-thread-one-pass",
        )
        result = evaluate(
            EvaluationRequest(
                key=key,
                candidate=registrations[candidate],
                opponent=registrations[baseline],
                checkpoint_paths=paths,
                deal_seeds=(26594000 + number,),
                purpose="exploratory-equal-update-head-to-head",
                authored_matchup=matchup,
                game_seconds=120,
                max_commands=10000,
                bootstrap_seed=26595001,
            ),
            out_dir=output / f"pair-{number:02d}",
            execution=EvalExecution(
                device="cuda"
                if os.environ.get("QUICK_EVAL_DEVICE") == "cuda"
                else "cpu",
                deadline_unix=deadline,
                max_active_games=2,
                max_batch_rows=16,
            ),
        )
        valid = [row for row in result.rows if row.valid]
        rows.append(
            {
                "pair": number,
                "entries": list(matchup.entries),
                "status": result.status,
                "attempted": len(result.rows),
                "valid": len(valid),
                "complete_deals": list(result.complete_deals()),
                "mana_score_sum": sum(row.score_a for row in valid),
                "error": result.error,
            }
        )
        summary["pairs"] = rows
        atomic_json(output / "summary.json", summary)
        print(json.dumps(rows[-1]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "mana-taps"))
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    output = Path(os.environ["ETUDE_OUTPUT_DIR"])
    if args.mode is not None:
        if args.deadline is None:
            raise ValueError("mode requires an allocation deadline")
        panel(args.mode, output / args.mode, args.deadline)
        return
    for mode in ("baseline", "mana-taps"):
        environment = dict(os.environ)
        environment.update(
            MANAGYM_EXPERIMENT_COMBAT_PICK="0",
            MANAGYM_EXPERIMENT_SOURCE_TAPS=str(int(mode == "mana-taps")),
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.decision_shapes_quick_eval",
                "--mode",
                mode,
                "--deadline",
                str(time.time() + 2100),
            ],
            env=environment,
            check=True,
            timeout=2160,
        )


if __name__ == "__main__":
    main()
