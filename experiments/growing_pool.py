"""Frozen v1 reference-league entry point; prospective batches bind new modules."""

from pathlib import Path

from etude.eval.match import Result
from etude.eval.records import Match
from etude.experiment import ExperimentState
from etude.launch.records import Job, Run
from etude.report import Report

from experiments import growing_pool_runtime as league
from experiments.growing_pool_protocol import ROOT, ProtocolBinding

BINDING = ProtocolBinding(
    module="experiments.growing_pool",
    protocol_path=ROOT / "experiments/protocols/growing-pool-245-v1.json",
    protocol_sha256="1c306b93bdde4e4d9ee17774c88511fc675f73429d5b6367e88514cf7d7a87f2",
)
BINDING.check_entry(Path(__file__))


def runs(revision: str) -> list[Run]:
    return league.training_runs(BINDING, revision)


def eval(state: ExperimentState) -> list[Job]:
    return league.evaluation_jobs(BINDING, state)


def play(match: Match) -> Result:
    return league.play_match(BINDING, match)


def report() -> Report:
    return league.league_report()
