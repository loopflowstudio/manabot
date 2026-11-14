import { expect, it } from 'vitest';
import { playerIdentity } from './player-identity';

it('shortens only path-shaped bot names and retains exact provenance', () => {
  const name = 'ci-fix-528/consumer/release-smoke/checkpoints/learner_transitions-8';
  const digest = '7ac5d442da' + '0'.repeat(54);
  expect(playerIdentity({ name, kind: 'bot', digest })).toMatchObject({
    compact: 'ci-fix-528/…/learner_transitions-8', full: name, digest,
    shortDigest: '7ac5d442da', text: 'ci-fix-528/…/learner_transitions-8 · 7ac5d442da',
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
  const first = playerIdentity({ name: 'run/one/model', kind: 'bot', version: 'a'.repeat(64) });
  const second = playerIdentity({ name: 'run/two/model', kind: 'bot', version: 'b'.repeat(64) });
  expect(first.compact).toBe(second.compact);
  expect(first.text).not.toBe(second.text);
});
