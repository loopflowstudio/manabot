"""Time the configured database catch-up and page generation separately."""
import json
from pathlib import Path
import time

from manabot.eval.card_page import write_page
from manabot.eval.card_query import CardSelection, card_db

source = Path('/private/tmp/claude-501/-Users-jack-src-etude-eb70f8-keep-each-eval/af5d648d-451c-4dce-9ad4-003076c612d7/scratchpad/demo/runs')
selection = CardSelection(experiment='mtg183-elemental-eval', subjects=('u005950',), opponents=('heuristic',))
wall, cpu = time.perf_counter(), time.process_time()
with card_db(selection.experiment, source) as db:
    opening = {'wall_seconds': time.perf_counter() - wall, 'cpu_seconds': time.process_time() - cpu}
    wall, cpu = time.perf_counter(), time.process_time()
    write_page(db, selection, Path('scratch/mtg206-demo/u005950-vs-heuristic.html'))
    page = {'wall_seconds': time.perf_counter() - wall, 'cpu_seconds': time.process_time() - cpu}
result = {'source': str(source), 'selection': selection.model_dump(mode='json'), 'database_catch_up': opening, 'page': page}
Path('scratch/mtg206-demo/pair-timing.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
