import json
import sys

import pytest

import managym
from scripts.bench_sharded_collection import main


def test_sharded_collection_continues_across_short_games(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def short_deck(setup: str, key: str) -> managym.PlayerConfig:
        return managym.PlayerConfig(key, {"Forest": 8})

    monkeypatch.setattr(managym, "authored_deck_setup", short_deck)
    monkeypatch.setattr(managym, "ALLOCATION_COUNTING", False)
    monkeypatch.setattr(sys, "argv", ["bench_sharded_collection"])
    main()
    assert json.loads(capsys.readouterr().out)["steps"] == 2048


@pytest.mark.parametrize(
    ("counting", "expected", "message"),
    [
        (True, "system", "allocation counting is enabled"),
        (False, "mimalloc", "expected allocator mimalloc, loaded system"),
    ],
)
def test_collection_rejects_incompatible_build(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    counting: bool,
    expected: str,
    message: str,
) -> None:
    monkeypatch.setattr(managym, "ALLOCATION_COUNTING", counting)
    monkeypatch.setattr(managym, "NATIVE_ALLOCATOR", "system")
    monkeypatch.setattr(
        sys, "argv", ["bench_sharded_collection", "--expect-allocator", expected]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert message in output.err
