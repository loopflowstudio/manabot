"""A managym-only record exports through the independently pinned Arena reader."""

import json
from pathlib import Path
import subprocess
import sys


def test_record_exports_without_training_imports(tmp_path: Path) -> None:
    script = """
import builtins
from pathlib import Path
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'manabot', 'torch', 'etude'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import managym
from managym.recording import ScalarCapture, read_record
from managym.export_arena import export_log, coverage
path = Path(__import__('sys').argv[1])
env = managym.Env(seed=7)
capture = ScalarCapture(seed=7, games_path=path / 'game.jsonl')
capture.reset(env, [managym.PlayerConfig(name=str(i), decklist={'Mountain': 12}) for i in (0, 1)])
while not env.is_game_over():
    capture.step(env, 0)
capture.close()
record = read_record(path / 'game.jsonl')
log = export_log(record)
assert record.sha256 in log
(path / 'game.log').write_text(log)
assert coverage()['definitions'] == coverage()['mapped'] + coverage()['unmapped']
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], check=True, timeout=30
    )
    harness = Path(__file__).parents[1] / "eval" / "arena_log_harness.py"
    result = subprocess.run(
        [sys.executable, str(harness), str(tmp_path / "game.log")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    parsed = json.loads(result.stdout)
    assert not parsed["errors"]
    assert len(parsed["results"]) == 1
    game = parsed["results"][0]
    assert len(game["opening_hand"]) == 7
    assert len(game["maindeck_card_ids"]) == 12
    assert game["history"]["seat_id"] == 1
