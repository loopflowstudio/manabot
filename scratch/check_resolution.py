"""Bound one local check and retain its output and exact source digests."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main() -> None:
    receipt, seconds, *command = sys.argv[1:]
    paths = [*Path('managym/src').rglob('*.rs'), *Path('managym/tests').rglob('*.rs')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    start = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    timed_out = False
    try:
        output, _ = process.communicate(timeout=int(seconds))
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
    result = dict(command=command, elapsed_seconds=round(time.monotonic()-start, 3), exit_code=process.returncode, timed_out=timed_out, source_sha256=hashes, output=output)
    Path(receipt).write_text(json.dumps(result, indent=2) + '\n')
    print(output[-6000:])
    print(f'receipt={receipt}; exit={process.returncode}; timeout={timed_out}')
    sys.exit(124 if timed_out else process.returncode)


if __name__ == '__main__':
    main()
