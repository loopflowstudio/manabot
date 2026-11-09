"""Read this cohort's complete retained prefixes without scanning unrelated experiments."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from functools import cache
from pathlib import Path

from etude.cache import Cache
from etude.experiment import observe
from etude.run.storage import Storage, storage_from_uri

from experiments.tla_frontier import with_launcher_ends
from experiments.tla_frontier_vocabulary import EXPERIMENT, read_check


class CohortStorage:
    def __init__(self, source: Storage, prefixes: list[str]) -> None:
        self.source = source
        self.name = 'c5fba4-vocabulary-retained-20261108'
        with ThreadPoolExecutor(max_workers=4) as pool:
            self.listing = {key: version for values in pool.map(source.versions, prefixes) for key, version in values.items()}

    def keys(self, prefix: str) -> list[str]:
        return sorted(key for key in self.listing if key.startswith(prefix))

    def versions(self, prefix: str) -> dict[str, str]:
        return {key: value for key, value in self.listing.items() if key.startswith(prefix)}

    @cache
    def get(self, key: str) -> bytes:
        return self.source.get(key)

    def get_range(self, key: str, start: int, length: int) -> bytes:
        return self.get(key)[start : start + length]

    def uri(self, key: str) -> str:
        return self.source.uri(key)

    def put(self, key: str, data: bytes) -> None:
        raise RuntimeError('Read only')

    def put_file(self, key: str, path: Path) -> None:
        raise RuntimeError('Read only')

    def delete(self, key: str) -> bool:
        raise RuntimeError('Read only')


def main() -> None:
    costs = json.loads(Path('.runs/c5fba4-20261108/c5fba4-costs.json').read_text())
    prefixes = [row['run'] + '/' for row in costs if 'vocabulary' in row['name'] or 'vocabulary' in row['run']]
    remote = storage_from_uri('r2://manabot/jobs')
    remote.cache = Cache(Path('.runs/c5fba4-20261108/read-cache'))
    source = CohortStorage(remote, prefixes)
    Path('/tmp/c5fba4-reading-prefixes.json').write_text(json.dumps({'prefixes': prefixes, 'keys': source.keys('')}, indent=2))
    print('Listed', len(source.listing), 'retained keys', flush=True)
    state = with_launcher_ends(observe(source, EXPERIMENT))
    print([(run.name, run.state) for run in state.runs], flush=True)
    reading = read_check(state, source)
    Path('/tmp/c5fba4-vocabulary-scoped.json').write_text(reading.model_dump_json(indent=2))
    print(reading.model_dump_json(exclude={'arms': {'__all__': {'value_loss'}}, 'early': True}, indent=2))


if __name__ == '__main__':
    main()
