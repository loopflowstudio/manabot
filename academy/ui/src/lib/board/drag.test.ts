import { describe, expect, it } from 'vitest';

import type { CardState, InteractionOffer } from '$lib/types';
import { carriedAt, cancel, DRAG_THRESHOLD_PX, dropLabel, IDLE, move, playable, press, promote, release, type Gesture, type PlayDrag } from './drag';
import type { PlacedOffer } from './view';

const offer = (id: number, verb: InteractionOffer['verb'], label: string): PlacedOffer => ({
  offer: { id, actor: 0, verb, source: null, label, help: null, choices: [], confirm_label: label, action_type: 'PRIORITY_CAST_SPELL', focus: [7] },
  number: id + 1,
  anchor: { kind: 'object', id: 7 },
});
const bolt = { id: 7, name: 'Lightning Bolt' } as CardState;
const cast = offer(0, 'cast', 'Cast Lightning Bolt');
const cycle = offer(1, 'activate', 'Cycle Lightning Bolt');
const opener = {} as HTMLElement;

function pressed(offers: PlacedOffer[] = [cast]): Gesture {
  return press({ kind: 'play', overField: false, card: bolt, offers, width: 140, grab: { x: 20, y: 30 }, opener }, 1, 5, { x: 100, y: 500 });
}
const field = (overField: boolean): Parameters<typeof move>[3] => drag => drag.kind === 'play' ? { ...drag, overField } : drag;
function dragged(offers: PlacedOffer[] = [cast], overField = true): Gesture {
  return move(pressed(offers), 1, { x: 100, y: 300 }, field(overField));
}

describe('a press becomes a drag', () => {
  it('only past the threshold, so a short press stays a click', () => {
    const short = move(pressed(), 1, { x: 100 + DRAG_THRESHOLD_PX - 1, y: 500 }, field(true));
    expect(short.phase).toBe('pressed');
    expect(release(short, 1)).toEqual({ gesture: IDLE, release: { outcome: 'click' } });
    expect(move(pressed(), 1, { x: 100, y: 500 - DRAG_THRESHOLD_PX }, field(true)).phase).toBe('dragging');
  });

  it('follows its own pointer only', () => {
    const gesture = dragged();
    expect(move(gesture, 2, { x: 0, y: 0 }, field(false))).toBe(gesture);
    expect(release(gesture, 2).release.outcome).toBe('none');
    expect(move(pressed(), 2, { x: 0, y: 0 }, field(true)).phase).toBe('pressed');
  });

  it('carries the card where it was gripped', () => {
    expect(carriedAt(pressed())).toBeNull();
    expect(carriedAt(dragged())).toEqual({ x: 80, y: 270 });
  });
});

describe('releasing a carried card', () => {
  it('over the battlefield commits its one offer and holds it where it was dropped', () => {
    const { gesture, release: result } = release(dragged(), 1);
    expect(result).toMatchObject({ outcome: 'commit', at: { x: 80, y: 270 } });
    expect(gesture).toMatchObject({ phase: 'holding', reason: 'pending', revision: 5, at: { x: 80, y: 270 } });
  });

  it('over the battlefield opens several offers, and choosing one makes it pending', () => {
    const { gesture, release: result } = release(dragged([cast, cycle]), 1);
    expect(result.outcome).toBe('choose');
    expect(gesture).toMatchObject({ phase: 'holding', reason: 'menu' });
    expect(promote(gesture)).toMatchObject({ phase: 'holding', reason: 'pending' });
  });

  it('anywhere else sends nothing and settles back', () => {
    const { gesture, release: result } = release(dragged([cast], false), 1);
    expect(gesture).toBe(IDLE);
    expect(result).toMatchObject({ outcome: 'cancel', at: { x: 80, y: 270 } });
  });
});

it('releases a connection on its last recipient, without holding the source', () => {
  const target = { endpoint: { kind: 'object', id: 9 } as const, blocker: 3 };
  const start = press({ kind: 'connect', style: 'block', from: { kind: 'object', id: 3 }, blockerId: 3, target: null }, 1, 5, { x: 0, y: 0 });
  const over = move(start, 1, { x: 20, y: 30 }, drag => ({ ...drag, target }));
  expect(release(over, 1)).toEqual({ gesture: IDLE, release: { outcome: 'connect', target } });
  const off = move(over, 1, { x: 20, y: 80 }, drag => ({ ...drag, target: null }));
  expect(release(off, 1)).toEqual({ gesture: IDLE, release: { outcome: 'connect', target: null } });
});

describe('cancelling', () => {
  it('ends a press or a drag, never a held drop', () => {
    expect(cancel(pressed())).toBe(IDLE);
    expect(cancel(dragged())).toBe(IDLE);
    const held = release(dragged(), 1).gesture;
    expect(cancel(held)).toBe(held);
  });
});

describe('what a drag can carry and says', () => {
  it('starts only from a play or a cast', () => {
    expect(playable([cast])).toBe(true);
    expect(playable([offer(2, 'play_land', 'Play Island')])).toBe(true);
    expect(playable([offer(3, 'choose', 'Discard Lightning Bolt')])).toBe(false);
    expect(playable([])).toBe(false);
  });

  it('names what releasing does', () => {
    const at = (gesture: Gesture): PlayDrag => (gesture.phase === 'dragging' && gesture.drag.kind === 'play' ? gesture.drag : null)!;
    expect(dropLabel(at(dragged()))).toBe('Cast Lightning Bolt');
    expect(dropLabel(at(dragged([cast, cycle])))).toBe('Choose how to play Lightning Bolt');
    expect(dropLabel(at(dragged([cast], false)))).toBe('Release to cancel');
  });
});
