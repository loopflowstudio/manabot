import { describe, expect, it } from 'vitest';

import { BEAT_LIMIT, beatsFor, loadBeatPace, saveBeatPace } from './beats';
import type { FactKind, RecentFact } from './recent';

const NAMES = { you: 'You', opponent: 'Heuristic' };
const fact = (seq: number, by: RecentFact['by'], text: string, subjects: number[] = [], kind: FactKind = 'cast'): RecentFact =>
  ({ seq, by, kind, text, spell: null, subjects });

describe("the opponent's moves as beats", () => {
  it('reads out what the opponent did, with its consequences, and not the person\'s own moves', () => {
    const facts = [
      fact(1, 'you', 'You cast Shock, targeting Ally', [7]),
      fact(2, null, 'Shock dealt 2 to Ally', [7], 'damage'),
      fact(3, 'opponent', 'Heuristic cast Yip Yip!, targeting Jailer', [9]),
      fact(4, 'opponent', 'Heuristic attacked with Jailer', [9], 'attacked'),
      fact(5, null, 'Jailer dealt 3 to you', [], 'damage'),
    ];
    expect(beatsFor(facts, 0, NAMES).map(beat => beat.seq)).toEqual([2, 3, 4, 5]);
    expect(beatsFor(facts, 3, NAMES).map(beat => [beat.text, beat.subjects])).toEqual([
      ['Heuristic attacked with Jailer', [9]], ['Jailer dealt 3 to you', []],
    ]);
  });

  it('plays nothing for the consequences of the person\'s own move alone', () => {
    expect(beatsFor([fact(1, 'you', 'You attacked with Ally'), fact(2, null, 'Ally dealt 2 to Heuristic', [], 'damage')], 0, NAMES)).toEqual([]);
    expect(beatsFor([fact(3, 'opponent', 'Heuristic cast X')], 3, NAMES)).toEqual([]);
  });

  it('groups a busy turn into sentences: one per run of a player\'s moves or of one kind of consequence', () => {
    const facts = [
      fact(1, 'opponent', 'Heuristic played Plains', [1], 'land'),
      fact(2, 'opponent', 'Heuristic attacked with Ally and Kyoshi Warriors', [2, 3], 'attacked'),
      fact(3, null, 'Ally dealt 2 to you', [2], 'damage'),
      fact(4, null, 'Kyoshi Warriors dealt 3 to you', [3], 'damage'),
      fact(5, null, 'Ally died', [2], 'died'),
    ];
    expect(beatsFor(facts, 0, NAMES).map(beat => [beat.text, beat.subjects])).toEqual([
      ['Heuristic played Plains and attacked with Ally and Kyoshi Warriors', [1, 2, 3]],
      ['Ally dealt 2 to you and Kyoshi Warriors dealt 3 to you', [2, 3]],
      ['Ally died', [2]],
    ]);
    // Four facts or fewer still read one at a time.
    expect(beatsFor(facts.slice(0, 4), 0, NAMES)).toHaveLength(4);
  });

  it('keeps a long turn to its newest beats', () => {
    const facts = Array.from({ length: BEAT_LIMIT + 4 }, (_, i) => fact(i + 1, i % 2 ? 'opponent' : null, `move ${i + 1}`, [], i % 2 ? 'cast' : (['damage', 'died'] as const)[i % 4 / 2]));
    const beats = beatsFor(facts, 0, NAMES);
    expect(beats).toHaveLength(BEAT_LIMIT);
    expect(beats[0].seq).toBe(5);
  });
});

describe('beat pace', () => {
  it('is quick unless the browser remembers another', () => {
    const stored = new Map<string, string>();
    (globalThis as { localStorage?: unknown }).localStorage = {
      getItem: (key: string) => stored.get(key) ?? null,
      setItem: (key: string, value: string) => { stored.set(key, value); },
    };
    expect(loadBeatPace()).toBe('quick');
    saveBeatPace('off');
    expect(loadBeatPace()).toBe('off');
    localStorage.setItem('academy.table.beat-pace', 'nonsense');
    expect(loadBeatPace()).toBe('quick');
    delete (globalThis as { localStorage?: unknown }).localStorage;
    expect(loadBeatPace()).toBe('quick');
  });
});
