# Open Questions / Blockers

- Could not run `pip install -e managym` exactly as specified in the design doc because the local `pip` (Python 3.9) does not support editable installs for non-setuptools `pyproject.toml` projects.
- Could not run `pytest tests/agent/test_managym.py tests/env/` because `pytest` is not installed in this environment (`python3 -m pytest` reports `No module named pytest`).

Workaround used for local smoke testing:
- Built a temporary extension artifact via `cargo rustc --features python --crate-type cdylib -- -C link-arg=-undefined -C link-arg=dynamic_lookup` and loaded it as `managym/_managym.so` to verify API behavior and enum/value parity.
