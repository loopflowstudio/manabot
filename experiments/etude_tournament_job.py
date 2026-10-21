"""Launch the w6 round robin on rented CPU machines, one share of its matches per run.

    etude launch experiments/etude_tournament_job.py --pool cpu8 --cohort w6-finals \
        --parallel 2 --max-usd 1 --deals 6 --shares 2

Each run's config (`ShareConfig`) says which share of the round robin it
plays and where to download the two checkpoints. Jobs hold no bucket keys, so
the launching computer signs a short-lived GET link per checkpoint
(`aws s3 presign`); setup verifies every file's SHA-256 before play.

The launcher loads this file with `runpy`, outside the repository's import
path, so it imports only Etude and the standard library; the adapter
(`experiments.etude_tournament`) reads `ShareConfig` from here on the machine.
"""

from __future__ import annotations

from decimal import Decimal
import subprocess

from etude.launch.records import Job, Run
from pydantic import BaseModel, ConfigDict

# Where the GPU runs' final checkpoints were delivered, with their digests and seeds.
BUCKET = "s3://etudefantasia/manabot/etude-launch"
FINALS = {
    "9945cc53ff41d4418520b2bb37c3c66be66fbf887f26a53a8513d68baba833e2": (
        "gpu-14353d64c8d3-final.pt",
        12261,
        "14353d64c8d3/attempts/bb60b709e5d6/training/self-play-raw.pt",
    ),
    "68a6b512df0b959adf5bcd731df82e2594c11c00bb9a81f8d8806403ba5641b4": (
        "gpu-cb14c42a3338-final.pt",
        12262,
        "cb14c42a3338/attempts/c65f82053677/training/self-play-raw.pt",
    ),
}
# Long enough for a machine to boot and set up; the links expire on their own.
LINK_SECONDS = 3 * 3600
# The Etude client the job's own environment installs: public, pinned by commit.
ETUDE = "e9f26336d09af5d6851dce1e80c4fdc968e10a94"

SETUP = f"""set -eu
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
. "$HOME/.cargo/env"
uv sync --frozen --no-default-groups --group play-runtime
uv pip install 'etude-sdk[eval] @ git+https://github.com/loopflowstudio/etude@{ETUDE}'
uv run --no-sync maturin build --release -i .venv/bin/python \
  -m managym/Cargo.toml -o managym/target/wheels
uv run --no-sync python -m experiments.etude_tournament install-managym
uv run --no-sync python -m experiments.etude_tournament fetch-checkpoints
"""


class Download(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    file: str
    training_seed: int
    url: str


class ShareConfig(BaseModel):
    """One launched run's part of the round robin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deals: int
    # (k, n): this run plays the matches whose key is k modulo n.
    share: tuple[int, int]
    # By SHA-256.
    checkpoints: dict[str, Download]


def _presign(key: str) -> str:
    # The AWS CLI signs with the launching computer's credentials; nothing is printed.
    signed = subprocess.run(
        ["aws", "s3", "presign", f"{BUCKET}/{key}", "--expires-in", str(LINK_SECONDS)],
        capture_output=True,
        text=True,
        check=True,
    )
    return signed.stdout.strip()


def runs(deals: int = 6, shares: int = 2) -> list[Run]:
    """One run per share of the `deals`-deal round robin."""
    checkpoints = {
        sha: Download(file=file, training_seed=seed, url=_presign(key))
        for sha, (file, seed, key) in FINALS.items()
    }
    return [
        Run(
            job=Job(
                name=f"w6-finals-{deals}-share-{k}-of-{shares}",
                setup=SETUP,
                command="uv run --no-sync python -m experiments.etude_tournament"
                " hold-share",
                config=ShareConfig(
                    deals=deals, share=(k, shares), checkpoints=checkpoints
                ).model_dump_json(),
                repo=".",
                # Exactly the commit: a snapshot would take untracked local runs too.
                revision="HEAD",
                max_hours=1,
                max_usd=Decimal("0.50"),
                max_attempts=1,
            )
        )
        for k in range(shares)
    ]
