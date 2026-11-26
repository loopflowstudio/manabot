import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type { BotDecision, Position, RecordedPosition } from '$lib/recording';
import type { ExperienceFrame, PresentationEvent } from '$lib/types';
import { markText } from '$lib/analysis';
import { findBoard } from './boards';
import { buildHistory, legacyMoveHistory } from './history';
import {
  analysedEvidence, analysedMarks, analysisShownMoves, bottomSeat, entryPosition, historyAnalysisShown, historyKeyAt, historyMoveMarks, moveAt, moveMarks, nameSeats, offerEvidence, perspectiveOf, planReviewJump, positionHeading, positionKeyForEvent, recordedFrame, replayLog, reviewTransitionBoards, sinceRevision, stripMarks, withLiveAnalysisPoints, type StripMark,
} from './replay';
import { boardView } from './view';

const PLAYERS = { you: 'You', opponent: 'Heuristic' };

const FIXTURES = new URL('../../../../../managym/academy/fixtures/', import.meta.url);
const navigation = (JSON.parse(readFileSync(new URL('choice-navigation-w17-smart-payment.json', FIXTURES), 'utf8')) as {
  positions: Record<string, { frame: ExperienceFrame }>;
}).positions;
const frame = Object.values(navigation).map(position => position.frame).find(candidate => candidate.offers.length >= 3)!;

function position(kind: Position['kind'], extra: Partial<Position> = {}): Position {
  return { key: `${kind}:${Math.random()}`, kind, turn: 1, phase: 'MAIN', step: 'MAIN', actor: null, automatic: false, can_reveal: false, ...extra };
}

function recorded(base: Position, decision: ExperienceFrame | null): RecordedPosition {
  const viewer = decision?.projection.agent.player_index ?? 0;
  return {
    ...base, description: null, note_key: base.key, observation: decision?.projection ?? frame.projection,
    decision: decision ? {
      ordinal: 0, viewer, source: 'policy', revision: decision.revision, prompt_id: decision.prompt?.id ?? 0,
      offer_id: decision.offers[1].id, command_id: 'c', presentation_cursor: 0, frame: decision, offer: decision.offers[1],
      command: { command_id: 'c', match_id: decision.match_id, expected_revision: decision.revision, prompt_id: 0, offer_id: decision.offers[1].id, answers: [] },
      address: base.key,
    } : null,
    outputs: null, viewer, revealed: false, history: [], history_labels: [], history_before: null, cursor: 0, transition_start: 0, consequences_recorded: true,
  };
}

describe('recordedFrame', () => {
  it('draws a recorded choice with exactly its offers, in their order', () => {
    const seat = frame.projection.agent.player_index as 0 | 1;
    const drawn = recordedFrame(recorded(position('choice', { actor: seat }), frame), { id: 'r', revision: 9, winner: null }, seat);
    expect(drawn.offers.map(offer => offer.id)).toEqual(frame.offers.map(offer => offer.id));
    expect(drawn.prompt).toEqual(frame.prompt);
    expect(boardView(drawn, PLAYERS).list).toHaveLength(frame.offers.length);
  });

  it('draws a transition as the board alone, one revision on, with the other seat at the bottom when asked', () => {
    const seat = frame.projection.agent.player_index as 0 | 1;
    const drawn = recordedFrame(recorded(position('consequences', { revision: 4 }), null), { id: 'r', revision: 9, winner: null }, (1 - seat) as 0 | 1);
    expect(drawn.offers).toEqual([]);
    expect(drawn.prompt).toBeNull();
    expect(drawn.revision).toBe(5);
    expect(drawn.projection.agent.player_index).toBe(1 - seat);
  });
});

describe('perspective', () => {
  const positions = [position('choice', { actor: 1 }), position('consequences', { actor: 1 }), position('choice', { actor: 0 })];
  it('keeps a Play game at the person\'s seat and follows the chooser in a bot game', () => {
    expect(perspectiveOf(null, { players: [{ seat: 1, kind: 'human' } as never] })).toBe(1);
    expect(perspectiveOf(null, { players: [] })).toBe('chooser');
    expect(perspectiveOf('0', { players: [] })).toBe(0);
  });
  it('holds a chooser at the bottom through its consequences', () => {
    expect([0, 1, 2].map(at => bottomSeat('chooser', positions, at))).toEqual([1, 1, 0]);
    expect(bottomSeat(0, positions, 0)).toBe(0);
  });
});

describe('offerEvidence', () => {
  const outputs: BotDecision = {
    action: 1, probabilities: frame.offers.map((_, i) => (i === 1 ? 0.7 : 0.3 / (frame.offers.length - 1))),
    value: 0.25, value_semantic: 'win probability', search_scores: [], unavailable: null, factors: [], belief: null, update: null,
  };
  it('keys recorded probabilities by offer id, aligned with the frame', () => {
    const evidence = offerEvidence(frame, outputs, frame.offers[1].id);
    expect(evidence.probability.get(frame.offers[1].id)).toBe(0.7);
    expect(evidence.probability.size).toBe(frame.offers.length);
    expect(evidence.chosen).toBe(frame.offers[1].id);
    expect(evidence.summary).toEqual(['Value 0.250 · win probability', 'Update not retained']);
  });
  it('says what was not retained', () => {
    const evidence = offerEvidence(frame, null, frame.offers[0].id);
    expect(evidence.probability.size).toBe(0);
    expect(evidence.summary).toEqual(['Value not retained', 'Update not retained', 'Policy not retained']);
  });
});

describe('the move list and strip read the index', () => {
  const positions: Position[] = [
    position('choice', { key: 'c1', actor: 0, label: 'Play Island', revision: 3 }),
    position('consequences', { key: 'a1', actor: 0, label: 'Play Island', revision: 3 }),
    position('consequences', { key: 'a2', actor: 1, label: 'Cast Shock, targeting Seat 1', revision: 4 }),
    position('consequences', { key: 'a3', actor: 0, automatic: true, label: 'Auto: Pass priority', revision: 5, turn: 2 }),
    position('choice', { key: 'c2', actor: 0, label: 'Cast Shock', revision: 7, turn: 2, value: 0.4 }),
    position('consequences', { key: 'a4', actor: 0, label: 'Cast Shock, targeting Seat 2', revision: 8, turn: 2 }),
    position('state', { key: 'latest', turn: 2 }),
  ];
  const entries = replayLog(positions, 0, seat => (seat === 0 ? 'You' : 'Heuristic'));

  it('lists each move once, keyed by the position it jumps to', () => {
    // An opponent's step without narration is part of the move that follows it.
    expect(entries.map(entry => [entry.id, entry.actor, entry.text])).toEqual([
      ['c1', 'hero', 'Play Island'], ['a2', 'villain', 'Cast Shock, targeting you'], ['a3', 'auto', 'Pass priority'], ['c2', 'hero', 'Cast Shock, targeting Heuristic'],
    ]);
    const turns = legacyMoveHistory(entries, { you: 'Seat 1', opponent: 'Seat 2' }, new Map(), []);
    expect(turns.flatMap(turn => turn.lines.map(line => line.id))).toEqual(['c1', 'a2', 'a3', 'c2']);
    expect(positionKeyForEvent(positions, { from_revision: 3 }, true)).toBe('c1');
    expect(positionKeyForEvent(positions, { from_revision: 3 }, false)).toBe('a1');
    expect(historyKeyAt(new Set(['c1', 'a2']), positions, 3)).toBe('a2');
  });

  it('names your own multi-step move once, as narrated, at the decision where it began', () => {
    const steps = replayLog([
      position('choice', { key: 'cast', actor: 0, label: 'Cast Divide by Zero', revision: 3 }),
      position('consequences', { key: 'after-cast', actor: 0, label: null, revision: 3 }),
      position('choice', { key: 'target', actor: 0, label: 'Target Tiger-Seal', revision: 4 }),
      position('consequences', { key: 'after-target', actor: 0, label: null, revision: 4 }),
      position('choice', { key: 'pay', actor: 0, label: 'Pay the cost', revision: 5 }),
      position('consequences', { key: 'after-pay', actor: 0, label: 'Cast Divide by Zero, targeting Tiger-Seal', revision: 5 }),
      position('choice', { key: 'order', actor: 0, label: 'Put on stack next — Ally — …', revision: 6 }),
      position('consequences', { key: 'after-order', actor: 0, label: 'Ordered triggers: Ally, then Tiger-Seal', revision: 6 }),
    ], 0, seat => (seat === 0 ? 'You' : 'Heuristic'));
    expect(steps.map(entry => [entry.id, entry.text])).toEqual([
      ['cast', 'Cast Divide by Zero, targeting Tiger-Seal'], ['order', 'Ordered triggers: Ally, then Tiger-Seal'],
    ]);
  });

  it('calls the person "you" mid-sentence, and "yourself" in their own move', () => {
    const names = (seat: 0 | 1): string => (seat === 0 ? 'You' : 'Heuristic');
    expect(nameSeats('Cast Shock, targeting Seat 1', names)).toBe('Cast Shock, targeting you');
    expect(nameSeats('Cast Shock, targeting Seat 1', names, true)).toBe('Cast Shock, targeting yourself');
    expect(nameSeats('Cast Shock, targeting Seat 2', names, true)).toBe('Cast Shock, targeting Heuristic');
  });

  it('names players a recorded target choice calls Hero and Villain', () => {
    const targets = replayLog([
      position('choice', { key: 't1', actor: 0, label: 'Target Villain', revision: 3 }),
      position('choice', { key: 't2', actor: 0, label: 'Target Hero', revision: 4 }),
    ], 0, seat => (seat === 0 ? 'You' : 'Heuristic'));
    expect(targets.map(entry => entry.text)).toEqual(['Target Heuristic', 'Target yourself']);
  });

  it('notes where an imported record branched, without seeds, and jumps to that choice', () => {
    const branched = replayLog([
      position('choice', { key: 'd1', actor: 1, label: 'Play Island', revision: 3 }),
      position('choice', { key: 'd2', actor: 0, label: 'Pass priority', revision: 4, branched: ['determinize', 'shuffle_libraries'] }),
    ], 0, seat => (seat === 0 ? 'Seat 1' : 'Seat 2'));
    expect(branched.map(entry => [entry.id, entry.actor, entry.text])).toEqual([
      ['d1', 'villain', 'Play Island'],
      ['branch:d2', 'system', 'Branch · hidden cards re-dealt, libraries reshuffled'],
      ['d2', 'hero', 'Pass priority'],
    ]);
    expect(entryPosition('branch:d2')).toBe('d2');
    expect(entryPosition('d1')).toBe('d1');
  });

  it('marks your moves chosen with analysis shown, a multi-step move at the decision where it began', () => {
    const steps: Position[] = [
      position('choice', { key: 'cast', actor: 0, label: 'Cast Shock', revision: 3 }),
      position('consequences', { key: 'after-cast', actor: 0, label: null, revision: 3 }),
      position('choice', { key: 'target', actor: 0, label: 'Target Villain', revision: 4, analysis_shown: true }),
      position('consequences', { key: 'after-target', actor: 0, label: 'Cast Shock, targeting Seat 2', revision: 4 }),
      position('consequences', { key: 'reply', actor: 1, label: 'Play Forest', revision: 5 }),
      position('choice', { key: 'land', actor: 0, label: 'Play Island', revision: 6 }),
      position('consequences', { key: 'after-land', actor: 0, label: 'Play Island', revision: 6 }),
    ];
    const log = replayLog(steps, 0, seat => (seat === 0 ? 'You' : 'Heuristic'));
    expect(log.map(entry => entry.id)).toEqual(['cast', 'reply', 'land']);
    expect([...analysisShownMoves(log, steps)]).toEqual(['cast']);
  });

  it('finds the move a consequence belongs to', () => {
    expect(moveAt(entries, positions, 1)).toBe('c1');
    expect(moveAt(entries, positions, 5)).toBe('c2');
    expect(moveAt(entries, positions, 6)).toBe('c2');
  });

  it('marks turns, decisions and recorded values', () => {
    const marks = stripMarks(positions, 0);
    expect(marks.filter(mark => mark.turnStart).map(mark => mark.key)).toEqual(['c1', 'a3']);
    expect(marks.filter(mark => mark.decision).map(mark => [mark.side, mark.value])).toEqual([['you', null], ['you', 0.4]]);
    expect(marks[2].side).toBe('opponent');
  });

  it('starts the recap at the chooser\'s previous decision, or at the transition', () => {
    expect(sinceRevision(positions, 4)).toBe(3);
    expect(sinceRevision(positions, 2)).toBe(4);
    expect(sinceRevision(positions, 0)).toBeNull();
  });
});

describe('a position says what it is', () => {
  const names = (seat: 0 | 1): string => (seat === 0 ? 'You' : 'Heuristic');
  it('names the chooser, the mover and the end', () => {
    expect(positionHeading({ kind: 'choice', actor: 0, automatic: false }, names, false, null)).toBe('Your choice');
    expect(positionHeading({ kind: 'choice', actor: 1, automatic: false }, names, false, null)).toBe("Heuristic's choice");
    expect(positionHeading({ kind: 'consequences', actor: 1, automatic: false }, names, false, null)).toBe("After Heuristic's move");
    expect(positionHeading({ kind: 'consequences', actor: 0, automatic: false }, names, false, null)).toBe('After your move');
    expect(positionHeading({ kind: 'consequences', actor: null, automatic: true }, names, false, null)).toBe('After an automatic pass');
    expect(positionHeading({ kind: 'state', actor: null, automatic: false }, names, true, 1)).toBe('Game over · Heuristic wins');
    expect(positionHeading({ kind: 'state', actor: null, automatic: false }, names, true, 0)).toBe('Game over · You win');
    expect(positionHeading({ kind: 'state', actor: null, automatic: false }, names, true, null)).toBe('Game over · Draw');
  });
});

describe('Review transition boards and jumps', () => {
  it('rebuilds transition boards via mergeBoards before orienting, preserves capped fallbacks, and surfaces patch errors', () => {
    const baseObs = structuredClone(frame.projection);
    const seat = baseObs.agent.player_index as 0 | 1;
    const otherSeat = (1 - seat) as 0 | 1;
    baseObs.turn.step = 'END_STEP';
    const ev = (seq: number): PresentationEvent => ({
      seq,
      from_revision: 4,
      to_revision: 5,
      caused_by: null,
      group: 10,
      importance: 'normal',
      suggested_ms: 0,
      sound: null,
      kind: { kind: 'drew', player: 1 },
    });

    const rebuilt = reviewTransitionBoards(
      {
        kind: 'consequences',
        revision: 4,
        observation: baseObs,
        history: [ev(10), ev(11), ev(12), ev(13)],
        // Capped before seq 12: seq 10..11 fall back to seq 12's board; seq 13 gets post-step observation.
        boards: [{ first: 12, last: 12, patch: { o: { turn: { o: { step: { '=': 'COMBAT_DAMAGE_STEP' } } } } } }],
      },
      otherSeat,
      null,
    );
    expect(rebuilt.error).toBeNull();
    expect(findBoard(rebuilt.boards, 10)).toMatchObject({ first: 10, last: 11, fallback: true });
    expect(findBoard(rebuilt.boards, 10)?.projection.turn.step).toBe('COMBAT_DAMAGE_STEP');
    expect(findBoard(rebuilt.boards, 10)?.projection.agent.player_index).toBe(otherSeat);
    expect(findBoard(rebuilt.boards, 12)?.fallback).toBeUndefined();
    expect(findBoard(rebuilt.boards, 13)?.projection.turn.step).toBe('END_STEP');
    expect(findBoard(rebuilt.boards, 13)?.projection.agent.player_index).toBe(otherSeat);

    // Corrupt patch reports an explicit error rather than silently falling back.
    const corrupt = reviewTransitionBoards(
      {
        kind: 'consequences',
        revision: 4,
        observation: baseObs,
        history: [ev(10)],
        boards: [{ first: 10, last: 10, patch: { o: { agent: { o: { battlefield: { a: [{ k: 999999 }] } } } } } }],
      },
      seat,
      null,
    );
    expect(corrupt.boards).toEqual([]);
    expect(corrupt.error).toMatch(/Board patch names no item/);
  });

  it('jumps to a player move choice even when already on its consequences transition', () => {
    const list: Pick<Position, 'key' | 'kind'>[] = [
      { key: 'choice:10', kind: 'choice' },
      { key: 'after:10', kind: 'consequences' },
      { key: 'after:11', kind: 'consequences' },
    ];
    // Already on `after:10`: clicking a player move whose key is `choice:10` must jump to `choice:10`.
    expect(planReviewJump(list, 1, 'after:10', 'choice:10', 41)).toEqual({
      kind: 'seek',
      index: 0,
      pendingSubCursor: null,
    });
    // Clicking another line belonging to `after:10` updates subCursor directly.
    expect(planReviewJump(list, 1, 'after:10', 'after:10', 45)).toEqual({
      kind: 'subcursor',
      seq: 45,
    });
    // Clicking a consequence belonging to a different transition seeks there with pendingSubCursor.
    expect(planReviewJump(list, 1, 'after:10', 'after:11', 52)).toEqual({
      kind: 'seek',
      index: 2,
      pendingSubCursor: 52,
    });
  });

  it('anchors a multi-step track cast move at its starting choice and groups its analysis marks', () => {
    const steps: Position[] = [
      position('choice', { key: 'cast', actor: 0, event: 3, label: 'Cast Shock', revision: 3 }),
      position('consequences', { key: 'after-cast', actor: 0, event: 3, label: null, revision: 3 }),
      position('choice', { key: 'target', actor: 0, event: 4, label: 'Target Villain', revision: 4, analysis_shown: true }),
      position('consequences', { key: 'after-target', actor: 0, event: 4, label: 'Cast Shock, targeting Seat 2', revision: 4 }),
    ];
    expect(positionKeyForEvent(steps, { from_revision: 4 }, true)).toBe('cast');
    const castEvent: PresentationEvent = {
      seq: 1,
      from_revision: 4,
      to_revision: 5,
      caused_by: 'cmd-1',
      group: 1,
      importance: 'normal',
      suggested_ms: 0,
      sound: null,
      labels: { objects: { '10:1': 'Shock' }, stacks: { '10': 'Shock' } },
      kind: { kind: 'cast', object: { entity: 10, incarnation: 1 }, controller: 0, stack: 10 },
    };
    const turns = buildHistory([castEvent], () => ({ objects: {}, players: {}, stacks: {} }), { youSeat: 0, you: 'You', opponent: 'Reader' }, {
      jumpKeyFor: (event, isMove) => positionKeyForEvent(steps, event, isMove),
    });
    expect(turns[0].lines[0].jumpKey).toBe('cast');
    const offers = frame.offers.length;
    const probabilities = frame.offers.map((_, i) => (i === offers - 1 ? 0.6 : 0.4 / (offers - 1)));
    const decision = { event: 3, played: 1, annotation: { probabilities, value: 0.2 }, semantic: 'signed_outcome', after: frame.offers.map((_, i) => i / offers) };
    const marks = historyMoveMarks(turns, steps, 0, new Map([[3, decision], [4, { ...decision, event: 4, played: offers - 1 }]]));
    expect(marks.get(turns[0].lines[0].id)).toMatchObject({ agreed: 1, of: 2 });
    expect([...historyAnalysisShown(turns, steps, 0)]).toEqual([turns[0].lines[0].id]);
  });
});

describe('an analysis read beside your choices', () => {
  const offers = frame.offers.length;
  // The policy prefers the last offer; you played the second.
  const probabilities = frame.offers.map((_, i) => (i === offers - 1 ? 0.6 : 0.4 / (offers - 1)));
  const decision = { event: 3, played: 1, annotation: { probabilities, value: 0.2 }, semantic: 'signed_outcome', after: frame.offers.map((_, i) => i / offers) };

  it('rates every option with its probability and its estimate after it, naming its choice and yours', () => {
    const evidence = analysedEvidence(frame, decision, frame.offers[1].id, 'Reader');
    expect(evidence.heading).toBe('Analysis by Reader');
    expect([...evidence.probability.values()]).toEqual(probabilities);
    expect(evidence.estimate.get(frame.offers[2].id)).toBeCloseTo(2 / offers);
    expect(evidence.first).toBe(frame.offers[offers - 1].id);
    expect(evidence.chosen).toBe(frame.offers[1].id);
    expect(evidence.summary).toContain("Reader's estimate here: 60");
    // Neutral words: what the policy would do, never a verdict on yours.
    expect(evidence.summary.join(' ')).not.toMatch(/mistake|blunder|inaccura|best|loss/i);
  });

  it('draws your analysed decisions on a 0–1 score and marks where you chose otherwise', () => {
    const positions = [position('choice', { actor: 0, event: 3 }), position('consequences', { actor: 0, event: 3 }), position('choice', { actor: 1, event: 4, value: 0.9 })];
    const marks = analysedMarks(stripMarks(positions, 0), positions, new Map([[3, decision]]));
    expect(marks.map(mark => mark.value)).toEqual([0.6, null, null]);
    expect(marks[0].differs).toBe(true);
  });

  it('marks each of your moves same or differs, with both estimates', () => {
    const positions = [position('choice', { actor: 0, event: 3 }), position('choice', { actor: 0, event: 5 })];
    const same = { ...decision, event: 5, played: offers - 1 };
    const entries = positions.map(p => ({ id: p.key, actor: 'hero' as const, turn: 1, text: 'Move' }));
    const marks = moveMarks(entries, positions, new Map([[3, decision], [5, same]]));
    expect(marks.get(positions[0].key)).toEqual({ agreed: 0, of: 1, yours: 1 / offers, its: (offers - 1) / offers });
    expect(marks.get(positions[1].key)).toMatchObject({ agreed: 1, of: 1 });
  });

  it("counts a cast's target choice with the cast, so marks add up to the summary", () => {
    const positions = [
      position('choice', { key: 'cast', actor: 0, event: 3, label: 'Cast Shock' }),
      position('consequences', { key: 'after-cast', actor: 0, event: 3, label: null }),
      position('choice', { key: 'target', actor: 0, event: 5, label: 'Target Villain' }),
      position('consequences', { key: 'after-target', actor: 0, event: 5, label: 'Cast Shock, targeting Seat 2' }),
    ];
    const log = replayLog(positions, 0, seat => (seat === 0 ? 'You' : 'Reader'));
    expect(log.map(entry => entry.id)).toEqual(['cast']);
    const marks = moveMarks(log, positions, new Map([[3, decision], [5, { ...decision, event: 5, played: offers - 1 }]]));
    const mark = marks.get('cast')!;
    expect([mark.agreed, mark.of]).toEqual([1, 2]);
    expect(markText(mark)).toBe('1 of 2 same');
    expect(mark.yours).toBe(1 / offers);
  });
});

describe('withLiveAnalysisPoints', () => {
  const base: StripMark[] = [
    { at: 0, key: 'a', turn: 1, turnStart: true, decision: true, side: 'opponent', value: null, label: 'Turn 1 · Opponent plays Island' },
    { at: 1, key: 'b', turn: 1, turnStart: false, decision: true, side: 'you', value: null, label: 'Turn 1 · You play Mountain' },
    { at: 2, key: 'c', turn: 2, turnStart: true, decision: false, side: null, value: null, label: 'Turn 2 · Untap' },
    { at: 3, key: 'd', turn: 2, turnStart: false, decision: true, side: 'you', value: null, label: 'Turn 2 · You cast Shock' },
  ];

  it('places each reading where the track stood, a pass that emits no move included', () => {
    const marks = withLiveAnalysisPoints(base, [
      { event: 2, at: 1, score: 0.6, semantic: 'signed_outcome' },
      { event: 3, at: 1, score: 0.55, semantic: 'signed_outcome' },
      { event: 8, at: 4, score: 0.3, semantic: 'signed_outcome' },
    ]);
    expect(marks.map(mark => [mark.key, mark.value])).toEqual([['a', null], ['read-2', 0.6], ['read-3', 0.55], ['b', null], ['c', null], ['d', null], ['read-8', 0.3]]);
    expect(marks.map(mark => mark.at)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(marks.at(-1)).toMatchObject({ side: 'you', decision: true, turn: 2, label: 'Turn 2 · after You cast Shock' });
    expect(marks[1].label).toBe('Turn 1 · after Opponent plays Island');
  });

  it('places a decision read after the fact at its turn, named by the move you chose', () => {
    const marks = withLiveAnalysisPoints(base, [
      { event: 6, at: null, score: 0.4, semantic: 'signed_outcome', turn: 2, label: 'Cast Shock' },
      { event: 1, at: null, score: 0.7, semantic: 'signed_outcome', turn: 1, label: 'Play Mountain' },
      { event: 9, at: 4, score: 0.3, semantic: 'signed_outcome' },
    ]);
    expect(marks.map(mark => mark.key)).toEqual(['read-1', 'a', 'b', 'read-6', 'c', 'd', 'read-9']);
    expect(marks[0].label).toBe('Turn 1 · Play Mountain');
    expect(marks[3]).toMatchObject({ turn: 2, value: 0.4, label: 'Turn 2 · Cast Shock' });
  });
});
