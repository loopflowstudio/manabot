"""Operator budget validation must reject unbounded runs before creating output."""

import pytest

from scripts import train_challenger


@pytest.mark.parametrize("seconds", ["nan", "inf", "-inf", "0"])
def test_rejects_invalid_wall_budget_before_starting_worker(
    seconds, tmp_path, monkeypatch, capsys
):
    out = tmp_path / "invalid-budget"
    monkeypatch.setattr(
        "sys.argv", ["train_challenger", "--out", str(out), f"--seconds={seconds}"]
    )

    def unexpected_worker(*args, **kwargs):
        pytest.fail("Invalid wall budget must not launch a training worker")

    monkeypatch.setattr(train_challenger.subprocess, "Popen", unexpected_worker)
    monkeypatch.setattr(train_challenger.platform, "platform", lambda: "test host")
    with pytest.raises(SystemExit) as error:
        train_challenger.main()
    assert error.value.code == 2
    assert "resource limits" in capsys.readouterr().err
    assert not out.exists()
