import { expect, it } from 'vitest';
import { playerIdentity, recordedIdentity } from './player-identity';

it('shortens only path-shaped bot names and retains exact provenance', () => {
  const name = 'ci-fix-528/consumer/release-smoke/checkpoints/learner_transitions-8';
  const digest = '7ac5d442da' + '0'.repeat(54);
  expect(playerIdentity({ name, kind: 'bot', digest })).toMatchObject({
    compact: 'ci-fix-528/…/learner_transitions-8', full: name, digest,
    text: 'ci-fix-528/…/learner_transitions-8 · 7ac5d442da',
    description: `${name} · ${digest}`,
  });
});

it.each(['run/model', 'Tournament winner', 'x'.repeat(60)])('keeps non-path names whole: %s', name => {
  expect(playerIdentity({ name, kind: 'bot' }).compact).toBe(name);
});

it('keeps human names whole and missing versions absent', () => {
  const name = 'my/name/is/human';
  expect(playerIdentity({ name, kind: 'human' })).toMatchObject({ compact: name, digest: null, text: name });
});

it('distinguishes checkpoints with the same compact name by recorded version', () => {
  const first = recordedIdentity({ name: 'run/one/model', kind: 'bot', player_id: 'checkpoint:a', version: 'a'.repeat(64) });
  const second = recordedIdentity({ name: 'run/two/model', kind: 'bot', player_id: 'checkpoint:b', version: 'b'.repeat(64) });
  expect(first.compact).toBe(second.compact);
  expect(first.text).not.toBe(second.text);
});

it('treats a control bot version as settings, not a digest', () => {
  const search = recordedIdentity({ name: 'Search', kind: 'bot', player_id: 'control:search',
    version: 'c'.repeat(64), inference: { policy: 'search' } });
  expect(search).toMatchObject({ digest: null, text: 'Search' });
  const checkpoint = recordedIdentity({ name: 'run/x/model', kind: 'bot', player_id: 'p',
    version: 'd'.repeat(64), inference: { policy: 'checkpoint' } });
  expect(checkpoint.text).toBe('run/…/model · dddddddddd');
});
