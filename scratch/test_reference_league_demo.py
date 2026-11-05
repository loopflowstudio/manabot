"""Headless demo adapted from the existing sealed-league fixture.

Only the authored synthetic endpoint score differs between cases. The real
report CLI runs in a fresh subprocess without compatibility patches. No native
games, policies or training execute. Raw stores stay in .runs, outside scientific
evidence and checkpoint stores.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from etude.eval.match import evaluate as seal_matches
from etude.eval.records import Code, Environment, Model
from etude.experiment import ExperimentState, observe
from etude.train.read import RunData, read_all
from etude.train.records import Step
from etude.train.run import track
import pytest

from experiments import (
    growing_pool as entry,
    growing_pool_runtime as growing_pool,
)
from experiments.growing_pool import BINDING
from experiments.growing_pool_analysis import AnalysisResult
from experiments.growing_pool_protocol import (
    ROOT,
    Protocol,
    World,
)
from manabot.eval.evaluation import EvaluationRequest, EvaluationResult
from manabot.infra.identity import canonical_sha256, file_sha256
from manabot.infra.provenance import source_bundle_sha256
from manabot.sim.card_evidence import GameEvidence, retain_evidence
from manabot.sim.matchup import DeckMatchup
from manabot.sim.registration import PlayerRegistration
from tests.eval.test_card_evaluation import card_fixture
from tests.eval.test_local_evaluation import _replayed_rows
from tests.training.test_training_exposure import receipt as training_receipt

EXPERIMENT = BINDING.experiment
PROTOCOL = BINDING.protocol_path


@dataclass
class SealedLeague:
    protocol: Protocol
    source: Path
    state: ExperimentState
    training: list[RunData]


@pytest.fixture(scope="module", params=["positive", "flat"])
def league(request: pytest.FixtureRequest) -> Iterator[SealedLeague]:
    root = ROOT / ".runs/growing-pool-demo-20261105" / str(request.param)
    root.mkdir(parents=True, exist_ok=False)
    improvement = request.param == "positive"
    protocol = Protocol.model_validate_json(PROTOCOL.read_bytes())
    # Explicit synthetic compatible world: scheduling only; never train/replay.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(World, "current", classmethod(lambda cls: protocol.world))
        patch.setenv("ETUDE_HOME", str(root / "home"))
        for name in (
            "ETUDE_RUN",
            "ETUDE_TRACK_DIR",
            "ETUDE_EXPERIMENT",
            "ETUDE_CONTINUES",
        ):
            patch.delenv(name, raising=False)
        paths = sorted((ROOT / "manabot").rglob("*.py"))
        identities = {
            "world": protocol.world.world_version,
            "engine_extension_sha256": "e" * 64,
            "engine_source_sha256": protocol.world.engine_source_sha256,
            "content_digest": protocol.world.card_digest,
            "deck_source_sha256": protocol.manifest.source().digest,
            "training_source_sha256": source_bundle_sha256(paths),
        }
        heuristic_source = canonical_sha256(
            {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}
        )
        source = root / "store"
        spec = protocol.analysis
        for index, seed in enumerate(protocol.budget.seeds):
            run = track(
                experiment=EXPERIMENT,
                run=f"learner-{index}",
                seed=seed,
                clocks=[Step("updates", "completed updates")],
                directory=source,
                config={
                    "training": {
                        "run_id": f"learner-{index}",
                        "regime": protocol.regime().model_dump(mode="json"),
                        "identities": identities,
                    }
                },
                strict=True,
            )
            try:
                for update in range(protocol.budget.updates + 1):
                    if update:
                        metrics = {name: 0.1 for name in spec.finite_metrics}
                        metrics.update({name: 0.0 for name in spec.zero_metrics})
                        metrics.update({name: 1.0 for name in spec.positive_metrics})
                        if update in spec.sampled_updates:
                            metrics.update(
                                {name: 0.1 for name in spec.sampled_finite_metrics}
                            )
                            metrics[spec.parameter_change] = 0.01
                        run.log(metrics, updates=update)
                    if update in (
                        0,
                        *protocol.budget.milestones,
                        protocol.budget.updates,
                    ):
                        path = root / f"seed-{seed}-u{update}.pt"
                        path.write_text(f"synthetic export {seed} {update}")
                        run.checkpoint(path, updates=update)
            finally:
                run.finish()
        state = observe(source, EXPERIMENT)
        coordinates = {
            c.artifact.sha256: (c.clocks["updates"], i)
            for i, r in enumerate(state.runs)
            for c in r.checkpoints
        }

        def registration(
            player: Model | Code, name: str, matrix: DeckMatchup, protocol: Protocol
        ) -> tuple[PlayerRegistration, Path | None]:
            common = dict(
                player_id=name,
                display_name=player.name,
                role="challenger" if name == "candidate" else "anchor",
                world=protocol.world.world_version,
                content_suite=matrix.content_suite,
                information_boundary="acting-viewer",
                observation_abi_sha256="a" * 64,
                action_abi_sha256="b" * 64,
                matchup_sha256=canonical_sha256(matrix.match().model_dump(mode="json")),
                player_seed_derivation_id="arena-pair-deal-player-v1",
                compute_class_id="policy-cpu-one-thread-one-pass",
            )
            if isinstance(player, Code):
                return PlayerRegistration(
                    **common,
                    runner_kind="code",
                    player_spec={"kind": "heuristic"},
                    source_sha256=heuristic_source,
                ), None
            return PlayerRegistration(
                **common,
                runner_kind="checkpoint",
                player_spec={
                    "kind": "checkpoint",
                    "device": "cpu",
                    "batch_size": 1,
                    "deterministic": False,
                    "allow_deck_repetition": True,
                },
                checkpoint_sha256=player.checkpoint.artifact.sha256,
                checkpoint_bytes=player.checkpoint.artifact.size_bytes,
                parameter_count=1,
                artifact_id=f"mtg245/{player.checkpoint.artifact.sha256}",
            ), Path("synthetic.pt")

        def synthetic(
            request: EvaluationRequest, *, out_dir: Path, card_evidence_dir: Path
        ) -> EvaluationResult:
            out_dir.mkdir(parents=True)
            matrix = request.authored_matchup
            assert matrix is not None
            update, seed_index = coordinates[request.candidate.checkpoint_sha256]
            rows = _replayed_rows(request)
            refs = []
            for row in rows:
                # Mean gain .0625/.125, with zero-gain blocks: lower CI is zero.
                extra_win = (
                    improvement
                    and update == protocol.budget.updates
                    and row.deal_seed % 100 == 0
                    and row.leg < 1 + seed_index
                )
                score = float(row.leg >= 2 or extra_win)
                winner = row.leg % 2 if score else 1 - row.leg % 2
                players = (
                    ("candidate", "opponent")
                    if row.leg % 2 == 0
                    else ("opponent", "candidate")
                )
                game = card_fixture(
                    winner=winner, leg=row.leg, deal=row.deal_seed, players=players
                )
                raw = game.model_dump(mode="json", by_alias=True)
                raw["seat_decks"] = list(matrix.decks(row.leg))
                setup = matrix.match(row.leg)
                setup.hero, setup.villain = players
                raw["setup"] = setup.model_dump(mode="json")
                raw["cards"]["capture"]["content_digest"] = protocol.world.card_digest
                trace = raw["trace"]["data"]
                trace.pop("game_trace_sha256")
                trace.update(
                    seat_decks=raw["seat_decks"],
                    match_hypers=raw["setup"],
                    observation_hypers=protocol.observation.model_dump(mode="json"),
                    runtime_identities={
                        **identities,
                        "arena_source_sha256": identities["training_source_sha256"],
                        "observation_abi_sha256": "a" * 64,
                        "action_abi_sha256": "b" * 64,
                    },
                )
                trace["game_trace_sha256"] = canonical_sha256(trace)
                raw["trace_sha256"] = trace["game_trace_sha256"]
                game = GameEvidence.model_validate(raw)
                refs.append(retain_evidence(card_evidence_dir, game))
                row.score_a = score
                row.__pydantic_extra__ = {"trace_sha256": game.trace_sha256}
            (out_dir / "trace.json").write_text(
                json.dumps({"card_evidence": [ref.model_dump() for ref in refs]})
            )
            return EvaluationResult(request=request, rows=rows, status="completed")

        patch.setattr(growing_pool, "_registration", registration)
        patch.setattr(growing_pool, "evaluate", synthetic)
        patch.setenv("ETUDE_OUTPUT_DIR", str(root / "eval"))
        matches = [
            match
            for run in state.runs
            for slot in growing_pool.comparisons(run, protocol, experiment=EXPERIMENT)
            if slot.matchup
            for match in slot.matchup.matches()
        ]
        seal_matches(
            experiment=EXPERIMENT,
            environment=Environment(
                name=protocol.id,
                files=protocol.digest,
                config={"status": protocol.status},
            ),
            matches=matches,
            play=entry.play,
            directory=source,
            run="synthetic-evaluation",
        )
        yield SealedLeague(protocol, source, state, read_all(source))


def test_reference_league_cli_demo(league: SealedLeague) -> None:
    """Present independent completeness and learning outcomes from retained bytes."""
    root = league.source.parent
    evidence = root / "exposure"
    evidence.mkdir()
    prototype = training_receipt(league.protocol)
    expected = league.protocol.budget.training_games // len(
        league.protocol.budget.seeds
    )
    for data in league.training:
        if data.file is None or data.file.declaration.kind == "evaluation":
            continue
        assert data.file.declaration.config_json is not None
        config = json.loads(data.file.declaration.config_json)["training"]
        for seed in range(expected):
            captured = prototype.model_copy(
                update={
                    "game": prototype.game.model_copy(update={"seed": seed}),
                    "stream": seed % league.protocol.budget.streams,
                    "context": {
                        **prototype.context,
                        "run": config["run_id"],
                        "seed": data.file.declaration.seed,
                        "identities": config["identities"],
                    },
                }
            )
            body = (
                json.dumps(
                    captured.model_dump(mode="json", by_alias=True),
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            (evidence / f"{captured.sha256}.jsonl").write_text(body)

    summaries = []
    for include_exposure in (True, False):
        name = "complete" if include_exposure else "missing-exposure"
        out = root / f"{name}.json"
        command = [
            sys.executable,
            "-m",
            "experiments.growing_pool_analysis",
            "--source",
            str(league.source),
            "--out",
            str(out),
        ]
        if include_exposure:
            command.extend(["--training-evidence", str(evidence)])
        result = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "ETUDE_HOME": str(root / "reader-home")},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        (root / f"{name}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr
        report = AnalysisResult.model_validate_json(out.read_bytes())
        assert report.counts.expected == report.counts.valid == 800
        assert report.complete is include_exposure
        assert report.training_exposure_complete is include_exposure
        assert report.modest_check is (root.name == "positive")
        assert all(seed.health.passed is True for seed in report.seeds)
        assert all(
            seed.exposure.terminal_games == (256 if include_exposure else 0)
            for seed in report.seeds
        )
        summaries.append(
            {
                "case": f"{root.name}-{name}",
                "report": str(out.relative_to(ROOT)),
                "complete": report.complete,
                "modest_check": report.modest_check,
                "training_exposure_complete": report.training_exposure_complete,
                "evaluation": report.counts.model_dump(),
                "terminal_games": [s.exposure.terminal_games for s in report.seeds],
                "anchor_gains": [
                    c.gain.model_dump()
                    for seed in report.seeds
                    for c in seed.contrasts
                    if c.panel == "anchors"
                    and c.update == 32
                    and c.anchor == "heuristic"
                    and c.gain is not None
                ],
                "issues": [s.issues for s in report.seeds],
            }
        )
    (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
