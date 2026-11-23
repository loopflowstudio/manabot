import sys

import pytest

import managym
from scripts.bench_sharded_collection import main


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
