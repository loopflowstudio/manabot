"""MTG-325 speed check (steer 449b1a8e) on one rented L4.

One machine, one venv and one native build. Two arm directories each hold
their own `manabot/` beside symlinks to every other checkout entry: `head` is
the job revision's, `base` is that with `base.patch` applied (main
a343904ee's `manabot/`, the merge-base). `bench.py` runs ABAB, `REPEATS` processes per arm,
then `compare.py` writes `summary.txt`.

`base.patch` is generated, not tracked: commit it only on the launch commit
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
for arm in base head; do
  mkdir -p "$work/$arm"; cp -r manabot "$work/$arm/"
  for entry in "$src"/*; do test "$entry" = "$src/manabot" || ln -s "$entry" "$work/$arm/"; done
done
if command -v git >/dev/null; then (cd "$work/base" && git apply "$src/{_HERE}/base.patch")
else (cd "$work/base" && patch -p1 -s < "$src/{_HERE}/base.patch"); fi
find "$work/base/manabot" -type d -empty -delete
uv run --locked python -c "import torch; print('GPU', torch.cuda.get_device_name(), torch.__version__, torch.version.cuda)"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu --format=csv || true
for i in $(seq 1 ${{REPEATS:-3}}); do
  for arm in base head; do
    echo "== $arm $i"
    PYTHONPATH="$work/$arm" uv run --locked python {_HERE}/bench.py "$out/$arm-$i.json"
  done
done
uv run --locked python {_HERE}/compare.py "$out" | tee "$out/compare.txt"
"""


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
