"""Local bounded command receipts; no background retries."""
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

label, limit, *command = sys.argv[1:]
started = time.monotonic()
path = Path('scratch') / f'{label}.log'
env = dict(os.environ, CARGO_BUILD_JOBS='2', CARGO_PROFILE_DEV_OPT_LEVEL='1')
with path.open('w') as log:
    try:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        code = process.wait(timeout=int(limit))
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        code = 124
receipt = {'label':label,'command':command,'limit_seconds':int(limit),'elapsed_seconds':round(time.monotonic()-started,3),'exit_code':code,'log':str(path),'cargo_build_jobs':2,'debug_assertions': ('--release' not in command) if ('cargo' in command or 'maturin' in command) and 'fmt' not in command else None}
Path(f'scratch/{label}.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
print(path.read_text()[-3000:])
