from pathlib import Path

from experiments.runners import run_int18_arena_rating as int18

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    ROOT
    / "experiments/data/int-18-first-world-pinned-arena-v1/sha256"
    / "af0c3f56745ba4f60e5e3f612787b11c65d6b125917ec1e59b1835e113765b2b"
    / "result"
)


def test_verify_only_authenticates_retained_exact_range_receipt(
    monkeypatch,
) -> None:
    def fail_on_live_derivation() -> None:
        raise AssertionError("verify-only re-derived mutable exact-range evidence")

    monkeypatch.setattr(int18, "_exact_range_wait", fail_on_live_derivation)

    receipt = int18.verify("production", RESULT_ROOT)

    assert receipt["verified"] is True
    assert receipt["no_generation"] is True
    assert receipt["no_replay"] is True
    assert receipt["exact_range_status"] == "evidence_wait"
