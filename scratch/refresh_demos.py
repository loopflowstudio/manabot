"""Record fresh current-world demo prefixes through ordinary Commands."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from academy.server import GameSession


def main() -> None:
    fixtures = Path('academy/fixtures')
    for stem in (sys.argv[1:] or ['learn-demo-explicit', 'precision-demo', 'precision-mana-demo']):
        previous = json.loads((fixtures / f'{stem}-w11.json').read_text())
        with TemporaryDirectory() as directory:
            session = GameSession(Path(directory))
            try:
                update = session.new_game(previous['config'])
                steps: list[dict[str, object]] = []
                for ordinal, old in enumerate(previous['prefix']):
                    frame = update['frame']
                    offer = next(o for o in frame['offers'] if o['id'] == old['offer_id'])
                    assert offer['label'] == old['label'], (stem, ordinal, offer)
                    steps.append({'before_digest': session.env.state_digest(), 'offer_id': offer['id'], 'label': offer['label']})
                    result = session.hero_command({'command_id': f'{stem}-w12.{ordinal}', 'match_id': frame['match_id'], 'expected_revision': frame['revision'], 'prompt_id': frame['prompt']['id'], 'offer_id': offer['id'], 'answers': []})
                    assert result['status'] == 'accepted', result
                    update = result['update']
                assert update['frame']['action_space'] == previous.get('decision_kind', 'LEARN')
                current = {**previous, 'manifest': session.env.content_pack_manifest(), 'prefix': steps, 'decision_digest': session.env.state_digest(), 'decision_revision': update['frame']['revision']}
                (fixtures / f'{stem}-w12.json').write_text(json.dumps(current, indent=2)+'\n')
                print(stem, len(steps), current['decision_digest'], flush=True)
            finally:
                session.close('test')


if __name__ == '__main__':
    main()
