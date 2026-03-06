# Open Questions / Blockers

- Could not run `pytest tests/gui/test_server.py tests/gui/test_trace_api.py` in this sandbox because project test dependencies are not installed and `uv run` panics in this environment (`system-configuration` panic). I validated syntax with `py_compile` and ran lightweight smoke checks for `gui/trace.py` and `_mini_fastapi.py` HTTP behavior using `.venv/bin/python`.
