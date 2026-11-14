"""Engine command line: `managym academy` launches the packaged Academy browser UI.

The commands live in `managym.academy.cli`, built with Typer, which the
`academy` extra provides with the server. Importing them lazily keeps
`import managym` free of both and lets a bare install say what to install.
"""

import json
import sys

_ACADEMY_EXTRA = {"click", "fastapi", "starlette", "typer", "uvicorn"}


def _missing_extra(exc: ModuleNotFoundError) -> int:
    if not exc.name or exc.name.split(".")[0] not in _ACADEMY_EXTRA:
        raise exc
    record = {
        "code": "extra.missing",
        "message": "Install the server dependencies: uv pip install 'managym[academy]'",
    }
    print("ACADEMY_PLAY_ERROR " + json.dumps(record, sort_keys=True), file=sys.stderr)
    return 2


def academy(argv: list[str] | None = None) -> int:
    """Run `managym academy` with `argv`; returns the exit status."""
    try:
        from managym.academy.cli import academy as run
    except ModuleNotFoundError as exc:
        return _missing_extra(exc)
    return run(argv)


def main(argv: list[str] | None = None) -> int:
    """The `managym` console script."""
    try:
        from managym.academy.cli import main as run
    except ModuleNotFoundError as exc:
        return _missing_extra(exc)
    return run(argv)
