"""Install the pinned local cp312 wheel extension atomically, then import it."""
from pathlib import Path
import zipfile


def main() -> None:
    wheels = list(Path('managym/target/wheels').glob('manabot-*cp312*.whl'))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as wheel:
        members = [name for name in wheel.namelist() if name.endswith('.so')]
        assert len(members) == 1, members
        target = Path('manabot/managym/_managym.cpython-312-darwin.so')
        temporary = target.with_suffix('.so.new')
        temporary.write_bytes(wheel.read(members[0]))
        temporary.replace(target)
    import manabot.managym as engine
    import managym
    assert managym is engine
    print('Native import succeeded:', engine.__file__)


if __name__ == '__main__':
    main()
