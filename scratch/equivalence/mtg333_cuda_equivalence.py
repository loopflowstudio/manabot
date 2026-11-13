"""MTG-333: the equivalence capture on one rented L4.

One machine, one venv and one native build run `run.py` three times under
`EQUIV_DEVICE=cuda` (TF32 allowed): with `manabot/` as at ef28f3a58 (this
revision's tree with `base.patch` reversed), with this revision's `manabot/`,
and with this revision's again (the run-to-run floor on CUDA). Each arm is a
directory holding its own `manabot/` beside symlinks to every other entry of
the checkout, so only `manabot/` differs between the arms.

    uv run --with "etude-sdk[s3,eval]==0.2.15" etude launch \\
        scratch/equivalence/mtg333_cuda_equivalence.py --rent l4 --revision COMMIT
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from etude.launch.records import Job, Run

from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup

ROOT = Path(__file__).resolve().parents[2]
_HERE = "scratch/equivalence"

_COMMAND = f"""{TOOLCHAIN_ENV}
set -eu
out="$ETUDE_OUTPUT_DIR"; mkdir -p "$out"
src=$PWD; work=$(mktemp -d)
mkdir -p "$work/base" "$work/head"
for arm in base head; do
  cp -r manabot "$work/$arm/"
  for entry in "$src"/*; do test "$entry" = "$src/manabot" || ln -s "$entry" "$work/$arm/"; done
done
if command -v git >/dev/null; then (cd "$work/base" && git apply -R "$src/{_HERE}/base.patch")
else (cd "$work/base" && patch -R -p1 -s < "$src/{_HERE}/base.patch"); fi
uv run --locked python -c "import torch; print('GPU', torch.cuda.get_device_name(), torch.__version__, torch.version.cuda)"
for arm in base head head; do
  name=$arm; test -e "$out/$arm.pt" && name=${{arm}}_repeat
  EQUIV_DEVICE=cuda PYTHONPATH="$work/$arm" uv run --locked python {_HERE}/run.py "$out/$name.pt"
done
echo "== base vs head"; uv run --locked python {_HERE}/compare.py "$out/base.pt" "$out/head.pt" | tee "$out/base_vs_head.txt"
echo "== head vs head repeat"; uv run --locked python {_HERE}/compare.py "$out/head.pt" "$out/head_repeat.pt" | tee "$out/head_vs_repeat.txt"
"""


def runs(revision: str, hours: float = 1.0, usd: float = 2.5) -> list[Run]:
    """The one L4 job at `revision`."""
    return Job(
        name="mtg333-cuda-equivalence",
        setup=machine_setup(""),
        command=_COMMAND,
        repo=str(ROOT),
        revision=revision,
        max_hours=hours,
        max_usd=Decimal(str(usd)),
        max_attempts=1,
    ).runs(0)
