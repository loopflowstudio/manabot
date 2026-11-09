"""Quote or launch the immutable same-device timing diagnostic, capped at USD0.75."""
from decimal import Decimal
from pathlib import Path
import sys

from etude.launch.client import launch_cohort, queue_cohort, quote
from etude.launch.records import Job

from manabot.experiment.job import TOOLCHAIN_ENV, machine_setup

root = Path.cwd()
script = (root / 'scratch/vocabulary_batched_probe.py').read_text()
candidate = (root / 'manabot/model/shared_effects.py').read_text()
script = 'from pathlib import Path\nPath("/tmp/c5fba4_shared_effects.py").write_text(' + repr(candidate) + ')\n' + script.replace('from __future__ import annotations\n', '')
job = Job(
    name='c5fba4-vocabulary-batched-timing',
    setup=machine_setup(' --extra artifacts --extra evaluation'),
    command=f"{TOOLCHAIN_ENV}\nuv run --locked --extra artifacts --extra evaluation python - <<'PROBE'\n{script}\nPROBE\n",
    repo=str(root), revision='bf408ec48c207fb034ab130341be96c61c544dad',
    max_hours=0.75, max_usd=Decimal('0.75'), max_attempts=1,
)
runs = job.runs(21501)
cohort = 'c5fba4-vocabulary-batched-20261108'
if '--queue' in sys.argv:
    print(queue_cohort(cohort, runs, rent='l4', max_usd=Decimal('0.75')))
elif '--launch' in sys.argv:
    print(launch_cohort(cohort, runs, rent='l4', max_usd=Decimal('0.75')))
else:
    print(quote(runs, rent='l4', cohort=cohort, max_usd=Decimal('0.75')).model_dump_json(indent=2))
