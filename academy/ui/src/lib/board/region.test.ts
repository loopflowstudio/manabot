import { describe, expect, it } from 'vitest';

import { isFocusNavigationKey, nearest } from './region';

const rect = (x: number, y: number, w = 60, h = 84): DOMRect => ({ left: x, top: y, width: w, height: h, right: x + w, bottom: y + h, x, y, toJSON: () => ({}) }) as DOMRect;
const key = (k: string, mods: Partial<{ metaKey: boolean; ctrlKey: boolean; altKey: boolean }> = {}) => ({
  key: k,
  metaKey: false,
  ctrlKey: false,
  altKey: false,
  ...mods,
});

describe('arrow keys move to the nearest control that way on screen', () => {
  // Two rows: creatures above, lands below; a plate at the left.
  const plate = rect(0, 0, 120, 200);
  const creatures = [rect(140, 10), rect(210, 10), rect(280, 10)];
  const lands = [rect(140, 110), rect(210, 110)];
  const all = [plate, ...creatures, ...lands];

  it('steps along a row and between rows', () => {
    expect(nearest(creatures[0], all, 'right')).toBe(2);
    expect(nearest(creatures[1], all, 'down')).toBe(5);
    expect(nearest(lands[0], all, 'up')).toBe(1);
    expect(nearest(creatures[0], all, 'left')).toBe(0);
  });

  it('finds nothing past the edge, so the caller steps in document order', () => {
    expect(nearest(creatures[2], all, 'right')).toBeNull();
    expect(nearest(creatures[0], all, 'up')).toBeNull();
  });
});

describe('isFocusNavigationKey', () => {
  it('recognizes bare focus-movement keys', () => {
    for (const name of ['Tab', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End']) {
      expect(isFocusNavigationKey(key(name))).toBe(true);
    }
  });

  it('ignores action shortcuts and modified chords so pointer play never sprouts a focus ring', () => {
    for (const event of [
      key(' '),
      key('1'),
      key('Escape'),
      key('Enter'),
      key('4', { metaKey: true }),
      key('Tab', { ctrlKey: true }),
      key('ArrowLeft', { metaKey: true }),
      key('ArrowDown', { altKey: true }),
    ]) {
      expect(isFocusNavigationKey(event)).toBe(false);
    }
  });
});
