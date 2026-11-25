import { describe, expect, it } from 'vitest';
import { LONGEST_TRAVEL_MS, Travel } from './motion';

// A card drawn on a fake page: a layout slot, plus the offset of the travel
// animation in progress at `clock`. Only translation is modelled, linearly;
// that is enough to read where a card is drawn against where its slot is.

let clock = 0;

interface FakeAnimation {
  id: string;
  from: { x: number; y: number };
  start: number;
  duration: number;
  cancelled: boolean;
  cancel(): void;
}

class FakeCard {
  dataset: Record<string, string>;
  slot: { x: number; y: number };
  animations: FakeAnimation[] = [];
  constructor(
    id: string,
    name: string,
    slot: { x: number; y: number },
    extra: Record<string, string> = {},
  ) {
    this.dataset = { travelId: id, travelName: name, ...extra };
    this.slot = slot;
  }
  running(): FakeAnimation[] {
    return this.animations.filter(animation => !animation.cancelled && clock < animation.start + animation.duration);
  }
  /** Where the card is drawn: the latest running animation wins, as WAAPI composites `translate`. */
  drawn(): { x: number; y: number } {
    const animation = this.running().at(-1);
    if (!animation) return this.slot;
    const left = 1 - (clock - animation.start) / animation.duration;
    return { x: this.slot.x + animation.from.x * left, y: this.slot.y + animation.from.y * left };
  }
  getBoundingClientRect(): DOMRect {
    const { x, y } = this.drawn();
    return { left: x, top: y, width: 120, height: 90, right: x + 120, bottom: y + 90, x, y, toJSON: () => ({}) } as DOMRect;
  }
  getAnimations(): FakeAnimation[] { return this.running(); }
  animate(keyframes: Keyframe[], options: KeyframeAnimationOptions): FakeAnimation {
    const [x = 0, y = 0] = String(keyframes[0].translate ?? '0 0').split(' ').map(part => parseFloat(part));
    const animation: FakeAnimation = {
      id: options.id ?? '', from: { x, y }, start: clock, duration: Number(options.duration), cancelled: false,
      cancel() { this.cancelled = true; },
    };
    this.animations.push(animation);
    return animation;
  }
}

function page(cards: FakeCard[]): HTMLElement {
  return { querySelectorAll: (selector: string) => (selector === '[data-travel-id]' ? cards : []) } as unknown as HTMLElement;
}

describe('card travel', () => {
  it('continues an interrupted travel from where the card is drawn into its new slot', () => {
    clock = 0;
    const cub = new FakeCard('perm:9', 'Badgermole Cub', { x: 100, y: 100 });
    const root = page([cub]);
    const travel = new Travel();

    // A frame moves the card's slot; it starts travelling there.
    travel.snapshot(root);
    cub.slot = { x: 500, y: 100 };
    travel.play(root, false);
    expect(cub.drawn()).toEqual({ x: 100, y: 100 });

    // Halfway (90 ms of 180 ms), another frame moves its slot again.
    clock = 90;
    expect(cub.drawn().x).toBeCloseTo(300);
    travel.snapshot(root);
    cub.slot = { x: 600, y: 100 };
    travel.play(root, false);

    // It goes on from where it was drawn, with one travel, not back to its old slot.
    expect(cub.running()).toHaveLength(1);
    expect(cub.drawn().x).toBeCloseTo(300);
    clock = 1000;
    expect(cub.running()).toHaveLength(0);
    expect(cub.drawn()).toEqual({ x: 600, y: 100 });
  });

  it('keeps travel within 180 ms and leaves stack abilities from their source card', () => {
    expect(LONGEST_TRAVEL_MS).toBeLessThanOrEqual(180);
    clock = 0;
    const reaper = new FakeCard('perm:14', 'Midnight Reaper', { x: 220, y: 140 }, { travelSide: 'opponent' });
    const trigger = new FakeCard('stack:91', 'Midnight Reaper', { x: 960, y: 80 }, {
      travelSide: 'opponent',
      travelSource: 'perm:14',
    });
    const travel = new Travel();
    travel.snapshot(page([reaper]));
    expect(travel.seenBefore('perm:14')?.side).toBe('opponent');
    travel.play(page([reaper, trigger]), false);

    expect(trigger.running()).toHaveLength(1);
    expect(trigger.running()[0].duration).toBeLessThanOrEqual(180);
    expect(trigger.drawn()).toEqual({ x: 220, y: 140 });
    clock = 180;
    expect(trigger.running()).toHaveLength(0);
    expect(trigger.drawn()).toEqual({ x: 960, y: 80 });
  });

  it('remembers a departed card position for death rule chips', () => {
    clock = 0;
    const ogre = new FakeCard('perm:19', 'Gray Ogre', { x: 340, y: 120 }, { travelSide: 'opponent' });
    const travel = new Travel();
    travel.snapshot(page([ogre]));
    travel.play(page([]), true);
    expect(travel.seenBefore('perm:19')?.rect.left).toBe(340);
    expect(travel.knownPosition('perm:19')?.side).toBe('opponent');
  });

  it('puts a travelling card home when the next frame does not animate', () => {
    clock = 0;
    const cub = new FakeCard('perm:9', 'Badgermole Cub', { x: 100, y: 100 });
    const root = page([cub]);
    const travel = new Travel();
    travel.snapshot(root);
    cub.slot = { x: 500, y: 100 };
    travel.play(root, false);

    clock = 90;
    travel.snapshot(root);
    travel.play(root, true);
    expect(cub.running()).toHaveLength(0);
    expect(cub.drawn()).toEqual({ x: 500, y: 100 });
  });
});
