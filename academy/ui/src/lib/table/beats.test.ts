import { describe, expect, it } from 'vitest';

import { BEAT_LIMIT, beatsFor } from './beats';
import type { RecentFact } from './recent';

const fact = (seq: number, by: RecentFact['by'], text: string, subjects: number[] = []): RecentFact => ({ seq, by, text, spell: null, subjects });

describe("the opponent's moves as beats", () => {
  it('reads out what the opponent did, with its consequences, and not the person\'s own moves', () => {
    const facts = [
      fact(1, 'you', 'You cast Shock, targeting Ally', [7]),
      fact(2, null, 'Shock dealt 2 to Ally', [7]),
      fact(3, 'opponent', 'Heuristic cast Yip Yip!, targeting Jailer', [9]),
      fact(4, 'opponent', 'Heuristic attacked with Jailer', [9]),
      fact(5, null, 'Jailer dealt 3 to You'),
    ];
    expect(beatsFor(facts, 0).map(beat => beat.seq)).toEqual([2, 3, 4, 5]);
    expect(beatsFor(facts, 3).map(beat => [beat.text, beat.subjects])).toEqual([
      ['Heuristic attacked with Jailer', [9]], ['Jailer dealt 3 to You', []],
    ]);
  });

  it('plays nothing for the consequences of the person\'s own move alone', () => {
    expect(beatsFor([fact(1, 'you', 'You attacked with Ally'), fact(2, null, 'Ally dealt 2 to Heuristic')], 0)).toEqual([]);
    expect(beatsFor([fact(3, 'opponent', 'Heuristic cast X')], 3)).toEqual([]);
  });

  it('keeps a long turn to its newest beats', () => {
    const facts = Array.from({ length: BEAT_LIMIT + 4 }, (_, i) => fact(i + 1, 'opponent', `move ${i + 1}`));
    const beats = beatsFor(facts, 0);
    expect(beats).toHaveLength(BEAT_LIMIT);
    expect(beats[0].seq).toBe(5);
  });
});
