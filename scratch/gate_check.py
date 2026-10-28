"""Bound a gate command and retain output plus exact input identities."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main() -> None:
    name, seconds, *command = sys.argv[1:]
    root = Path('.cache/rules-gate')
    root.mkdir(parents=True, exist_ok=True)
    tracked = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
    hashes = {
        path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in tracked
        if path and Path(path).is_file()
        and not path.startswith(('scratch/', 'docs/', '.lf/'))
    }
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    (root / f'{name}-inputs.json').write_text(json.dumps(hashes, indent=2) + '\n')
    started = time.monotonic()
    timed_out = False
    with (root / f'{name}.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=int(seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    result = dict(command=command, elapsed_seconds=round(time.monotonic()-started, 3),
                  exit_code=process.returncode, timed_out=timed_out, input_sha256=fingerprint,
                  head=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                  environment={key: os.environ[key] for key in (
                      'CARGO_PROFILE_DEV_OPT_LEVEL', 'CARGO_BUILD_JOBS', 'RUST_TEST_THREADS',
                      'OMP_NUM_THREADS', 'ACADEMY_API_PORT', 'ACADEMY_FRONTEND_PORT',
                  ) if key in os.environ})
    (root / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
    print((root / f'{name}.log').read_text()[-6000:])
    print(json.dumps(result))
    sys.exit(124 if timed_out else process.returncode)


if __name__ == '__main__':
    main()
