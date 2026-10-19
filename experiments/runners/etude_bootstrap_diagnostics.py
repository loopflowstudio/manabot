"""Expose stock-image startup phases before an Etude job can publish progress.

The bounded launcher opts into this adapter before creating a rental. It adds
stdout events to Etude 0.1.1's provider entrypoint, leaving the guardian, package
verification, installation and scientific job unchanged. No credentials or URLs
enter the events. Provider logs must still be retained before pod deletion.
"""

from pathlib import Path

_IMPORT = "from etude.launch.providers.runpod.pod import main\n"
_CALL = "raise SystemExit(main(['pod','boot']))"
_EVENTS = """import json as _json, os as _os, re as _re, subprocess as _subprocess, time as _time
def _phase(name: str) -> None:
    print("ETUDE_BOOT_PHASE " + _json.dumps({
        "phase": name, "unix_seconds": _time.time(),
    }), flush=True)
_phase("entrypoint")
"""
_WRAPPERS = """from etude.launch.providers.runpod import pod as _pod
_phase("guardian_imported")
_original_uv = _pod.ensure_uv
_original_install = _pod.install
def _ensure_uv() -> None:
    _phase("uv_start")
    _original_uv()
    _phase("uv_ready")
def _install(code: bytes) -> _pod.Path:
    _phase("client_install_start")
    try:
        result = _original_install(code)
    except Exception as error:
        _phase("client_install_failed")
        event = {"error_type": type(error).__name__}
        if isinstance(error, _subprocess.TimeoutExpired):
            event["timeout_seconds"] = error.timeout
            stderr = error.stderr or b""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            # Sanitize before truncation, so a boundary cannot split a secret.
            for value in _os.environ.values():
                if len(value) > 8:
                    stderr = stderr.replace(value, "<redacted>")
            stderr = _re.sub(r"https?://\\S+", "<redacted-url>", stderr)
            event["stderr_tail"] = stderr[-4096:]
        print("ETUDE_BOOT_FAILURE " + _json.dumps(event), flush=True)
        raise
    _phase("client_install_ready")
    return result
_pod.ensure_uv = _ensure_uv
_pod.install = _install
"""

_WORKER_WRAPPERS = """
_worker_script: _pod.Path | None = None
_original_worker_install = _pod.install
_original_worker_call = _pod.subprocess.call
def _worker_install(code: bytes) -> _pod.Path:
    global _worker_script
    root = _original_worker_install(code)
    _worker_script = root / "worker-startup-observer.py"
    _worker_script.write_text(_worker_source)
    return root
def _worker_call(argv: list[str], *args: object, **kwargs: object) -> int:
    if len(argv) == 6 and argv[1:5] == ["-m", "etude", "worker", "--bootstrap"]:
        if _worker_script is None:
            raise RuntimeError("Worker observer requires completed client installation")
        _phase("worker_process_start")
        argv = [argv[0], str(_worker_script), *argv[3:]]
    return _original_worker_call(argv, *args, **kwargs)
_pod.install = _worker_install
_pod.subprocess.call = _worker_call
"""


def instrument_bootstrap(source: str, *, worker_source: str | None = None) -> str:
    """Fail closed if the installed SDK entrypoint has a different structure."""
    if source.count(_IMPORT) != 1 or source.count(_CALL) != 1:
        raise ValueError(
            "Unsupported Etude bootstrap; inspect it before instrumentation"
        )
    wrappers = _WRAPPERS
    if worker_source is not None:
        compile(worker_source, "worker-startup-observer.py", "exec")
        wrappers += (
            "\n_worker_source = " + repr(worker_source) + "\n" + _WORKER_WRAPPERS
        )
    return _EVENTS + source.replace(_IMPORT, wrappers).replace(
        _CALL, "raise SystemExit(_pod.main(['pod','boot']))"
    )


def enable(*, worker: bool = False) -> str:
    """Instrument this launcher's future rentals; return the entrypoint for audit.

    The private SDK seam is deliberately version checked rather than silently
    altering another bootstrap. Existing machines and archived SDK code stay as
    launched. A second call is idempotent.
    """
    from etude.launch.providers.runpod import machines

    if machines._BOOT.startswith(_EVENTS):
        if worker and _WORKER_WRAPPERS not in machines._BOOT:
            raise ValueError(
                "Enable worker diagnostics before bootstrap instrumentation"
            )
    else:
        worker_source = (
            Path(__file__).with_name("etude_worker_diagnostics.py").read_text()
            if worker
            else None
        )
        machines._BOOT = instrument_bootstrap(
            machines._BOOT, worker_source=worker_source
        )
    return machines._BOOT
