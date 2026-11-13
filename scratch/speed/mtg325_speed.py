"""MTG-325 speed check (steer 449b1a8e) on one rented L4.

One machine, one venv and one native build. Each arm directory holds its own
`manabot/` beside symlinks to every other checkout entry: `head` is the job
revision's, and every `<arm>.patch` here gives one more arm (`base.patch`:
main a343904ee's `manabot/`, the merge-base; `backend.patch`: #503's).
`bench.py` runs the arms interleaved, `REPEATS` processes each, then
`compare.py` writes `summary.txt`.

Patches are generated, not tracked: commit them only on the launch commit
(`git diff HEAD a343904ee -- manabot > scratch/speed/base.patch`).

    uv run --with "etude-sdk[s3,eval]==0.2.15" etude launch \\
        scratch/speed/mtg325_speed.py --rent l4 --revision COMMIT \\
        --cohort mtg325-speed --max-usd 10
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from etude.launch.records import Job, Run

from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup

ROOT = Path(__file__).resolve().parents[2]
_HERE = "scratch/speed"

_COMMAND = f"""{TOOLCHAIN_ENV}
set -eu
out="$ETUDE_OUTPUT_DIR"; mkdir -p "$out"
src=$PWD; work=$(mktemp -d)
arms=$(ls {_HERE}/*.patch | xargs -n1 basename | sed 's/.patch$//')
for arm in $arms head; do
  mkdir -p "$work/$arm"; cp -r manabot "$work/$arm/"
  for entry in "$src"/*; do test "$entry" = "$src/manabot" || ln -s "$entry" "$work/$arm/"; done
done
for arm in $arms; do
  (cd "$work/$arm" && git apply "$src/{_HERE}/$arm.patch")
  find "$work/$arm/manabot" -type d -empty -delete
done
uv run --locked python -c "import torch; print('GPU', torch.cuda.get_device_name(), torch.__version__, torch.version.cuda)"
for i in $(seq 1 ${{REPEATS:-2}}); do
  for arm in $arms head; do
    echo "== $arm $i"
    PYTHONPATH="$work/$arm" uv run --locked python {_HERE}/bench.py "$out/$arm-$i.json"
  done
done
uv run --locked python {_HERE}/compare.py "$out" | tee "$out/compare.txt"""


def runs(revision: str, hours: float = 2.0, usd: float = 4.0) -> list[Run]:
    """The one L4 job at `revision`."""
    return Job(
        name="mtg325-speed",
        setup=machine_setup(""),
        command=_COMMAND,
        repo=str(ROOT),
        revision=revision,
        max_hours=hours,
        max_usd=Decimal(str(usd)),
        max_attempts=1,
    ).runs(0)
