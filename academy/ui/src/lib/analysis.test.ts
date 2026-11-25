import { describe, expect, it } from 'vitest';

import { LIVE_ANALYSIS_OFF, followShown, keyMoments, nextLiveAnalysis, type AnalysedDecision, type Analysis } from './analysis';

function decided(event: number, played: number, probabilities: number[], value: number): AnalysedDecision {
  return { event, played, annotation: { probabilities, value }, semantic: 'signed_outcome', after: probabilities.map(() => null) };
}
const analysis = (decisions: AnalysedDecision[]): Analysis => ({
  id: 'a', seat: 0, policy: { name: 'Reader' } as Analysis['policy'], status: 'complete', reason: null, total: decisions.length, created_at: '', decisions,
});

describe('key moments', () => {
  it('finds where you and the policy differed and where its estimate moved, in game order', () => {
    expect(keyMoments(analysis([
      decided(2, 0, [0.5, 0.5], 0),
      // Differs, and the estimate moves after it: listed once, as a difference.
      decided(5, 1, [0.9, 0.1], 0),
      decided(9, 0, [0.7, 0.3], 0.6),
      decided(12, 1, [0.4, 0.6], -0.2),
    ]))).toEqual([
      { event: 5, kind: 'differs', reason: 'Differs · 90% on its first choice, 10% on yours' },
      { event: 9, kind: 'moved', reason: 'Its expected score moved 80 → 40 by your next decision' },
    ]);
  });

  it('keeps the largest of each kind', () => {
    const many = Array.from({ length: 9 }, (_, i) => decided(i, 1, [0.5 + i / 20, 0.5 - i / 20], 0));
    expect(keyMoments(analysis(many)).map(moment => moment.event)).toEqual([4, 5, 6, 7, 8]);
  });

  it('finds none in a flat reading', () => {
    expect(keyMoments(analysis([decided(0, 1, [0.51, 0.49], 0), decided(1, 0, [0.5, 0.5], 0.02)]))).toEqual([]);
  });
});

describe('analysis while you play', () => {
  const reader = { sha256: 'r'.repeat(64), name: 'Reader' };
  const on = nextLiveAnalysis({ ...LIVE_ANALYSIS_OFF, pending: true }, { type: 'analysis_state', shown: true, policy: reader, policies: [reader], unavailable: null });
  const reading = { type: 'analysis' as const, policy: reader, event: 4, revision: 9, annotation: { probabilities: [0.25, 0.75], value: 0.2 }, semantic: 'signed_outcome', after: [0.4, null] };

  it('keeps the reading for the decision published at its revision, choosing nothing', () => {
    const read = nextLiveAnalysis(on, reading);
    expect(read.reading).toEqual({ revision: 9, decision: { event: 4, played: -1, annotation: reading.annotation, semantic: 'signed_outcome', after: [0.4, null] } });
  });

  it('drops a reading once you turned analysis off, or while that request is on its way', () => {
    expect(nextLiveAnalysis(LIVE_ANALYSIS_OFF, reading).reading).toBeNull();
    expect(nextLiveAnalysis({ ...on, pending: true }, reading).reading).toBeNull();
  });

  it('says why an analysis stopped, and follows frames: on with its policy after a reload, off in a new game', () => {
    expect(nextLiveAnalysis(on, { type: 'analysis', policy: reader, stopped: 'The policy failed at step 3.' }).stopped).toBe('The policy failed at step 3.');
    expect(followShown(LIVE_ANALYSIS_OFF, reader)).toMatchObject({ shown: true, policy: reader });
    expect(followShown(on, null)).toMatchObject({ shown: false, policy: null, policies: [reader] });
  });
});
