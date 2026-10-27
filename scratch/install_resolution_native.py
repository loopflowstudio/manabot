"""Atomically install the cp312 extension from this repair's successful build."""

import hashlib
import json
from pathlib import Path
import re
import zipfile


def main() -> None:
    build = json.loads(Path('scratch/resolution-native-build.json').read_text())
    assert build['exit_code'] == 0 and not build['timed_out']
    match = re.search(r'Built wheel for CPython 3.12 to (.+\.whl)', build['output'])
    assert match is not None
    wheel = Path(match[1])
    target = Path('manabot/managym/_managym.cpython-312-darwin.so')
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.endswith('/' + target.name)]
        assert len(members) == 1, members
        data = archive.read(members[0])
    temporary = target.with_suffix('.so.new')
    temporary.write_bytes(data)
    temporary.replace(target)
    result = dict(wheel=str(wheel), target=str(target), wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(), extension_sha256=hashlib.sha256(data).hexdigest())
    Path('scratch/resolution-native-install.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
