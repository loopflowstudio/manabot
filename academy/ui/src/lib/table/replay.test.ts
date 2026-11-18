import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type { BotDecision, Position, RecordedPosition } from '$lib/recording';
import type { ExperienceFrame } from '$lib/types';
import { moveList } from './moves';
import { bottomSeat, moveAt, offerEvidence, perspectiveOf, recordedFrame, replayLog, sinceRevision, stripMarks } from './replay';
import { tableView } from './view';

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
    expect(tableView(drawn).list).toHaveLength(frame.offers.length);
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
    position('consequences', { key: 'a4', actor: 1, label: null, revision: 8, turn: 2 }),
    position('state', { key: 'latest', turn: 2 }),
  ];
  const entries = replayLog(positions, 0, seat => (seat === 0 ? 'You' : 'Heuristic'));

  it('lists each move once, keyed by the position it jumps to', () => {
    // An opponent's step without narration is part of the move that follows it.
    expect(entries.map(entry => [entry.id, entry.actor, entry.text])).toEqual([
      ['c1', 'hero', 'Play Island'], ['a2', 'villain', 'Cast Shock, targeting you'], ['a3', 'auto', 'Pass priority'], ['c2', 'hero', 'Cast Shock'],
    ]);
    const turns = moveList(entries, { you: 'Seat 1', opponent: 'Seat 2' }, new Map(), []);
    expect(turns.flatMap(turn => turn.moves.map(move => move.id))).toEqual(['c1', 'a2', 'a3', 'c2']);
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
