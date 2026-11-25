import { describe, expect, it } from 'vitest';
import type { Observation, PresentationEvent, PresentationLabels } from '$lib/types';
import { collectDeathMarks, LONGEST_TRAVEL_MS, Travel } from './motion';

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

  it('collects death markers for destroyed creatures and tokens whose side comes from prior stepBoards', () => {
    const emptyPlayer = (id: number) => ({
      id,
      player_index: id,
      life: 20,
      poison: 0,
      hand_count: 0,
      library_count: 40,
      mana_pool: [0, 0, 0, 0, 0, 0],
      hand: [],
      graveyard: [],
      exile: [],
      stack: [],
      battlefield: [],
    });
    const baseProjection = {
      game_over: false,
      won: false,
      turn: { turn_number: 4, phase: 'Main1', step: 'Main', active_player_id: 0, agent_player_id: 0 },
      agent: emptyPlayer(0),
      opponent: emptyPlayer(1),
    } as unknown as Observation;
    const priorProjection = structuredClone(baseProjection);
    priorProjection.agent.battlefield.push({ id: 77, name: 'Ally', power: 1, toughness: 1 } as never);
    priorProjection.opponent.battlefield.push({ id: 88, name: 'Gray Ogre', power: 2, toughness: 2 } as never);

    const events: PresentationEvent[] = [
      {
        seq: 10,
        from_revision: 4,
        to_revision: 5,
        caused_by: null,
        group: 10,
        importance: 'critical',
        suggested_ms: 650,
        sound: 'creature.died',
        kind: { kind: 'died', objects: [{ entity: 77, incarnation: 0 }] },
        rule: 'lethal_damage',
        labels: { objects: { '77:0': 'Ally' } },
      },
      {
        seq: 11,
        from_revision: 4,
        to_revision: 5,
        caused_by: null,
        group: 11,
        importance: 'critical',
        suggested_ms: 650,
        sound: 'creature.died',
        kind: { kind: 'destroyed', objects: [{ entity: 88, incarnation: 0 }] },
        labels: { objects: { '88:0': 'Gray Ogre' } },
      },
    ];
    const emptyLabels: PresentationLabels = { objects: {}, players: {}, stacks: {} };
    const rect = { left: 0, top: 0, width: 600, height: 400, right: 600, bottom: 400, x: 0, y: 0, toJSON: () => ({}) } as DOMRect;
    const rootEl = { querySelector: () => ({ getBoundingClientRect: () => rect }) } as unknown as HTMLElement;
    const boardEl = { getBoundingClientRect: () => rect } as unknown as HTMLElement;

    const marks = collectDeathMarks({
      root: rootEl,
      boardEl,
      travel: new Travel(),
      projection: baseProjection,
      stepBoards: [{ first: 1, last: 9, projection: priorProjection }],
      events,
      labelsAt: () => emptyLabels,
      activeSeqs: new Set([10, 11]),
      cursor: 10,
      sinceRevision: null,
      matchId: 'm1',
      revision: 5,
      reduced: false,
    });
    expect(marks).toHaveLength(2);
    expect(marks[0]).toMatchObject({ entity: 77, name: 'Ally', side: 'you', rule: 'lethal damage' });
    expect(marks[1]).toMatchObject({ entity: 88, name: 'Gray Ogre', side: 'opponent', rule: 'destroyed' });
  });
});
