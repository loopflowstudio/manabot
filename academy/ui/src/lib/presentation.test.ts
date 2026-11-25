import { describe, expect, it } from 'vitest';

import {
  validatePresentationEvents,
  validatePresentationTail,
  validatePresentationUpdate,
} from './presentation';
import { createPresentationPlayer } from './presentation.svelte';
import { COMBAT_TO_TURN_PRESENTATION, LIGHTNING_BOLT_PRESENTATION } from './presentation-fixtures';

describe('Lightning Bolt presentation fixture', () => {
  it('records cast, target, resolution, damage, and death as ordered semantic facts', () => {
    expect(LIGHTNING_BOLT_PRESENTATION.events.map((event) => event.kind.kind)).toEqual([
      'cast',
      'targeted',
      'resolved',
      'damage',
      'died',
    ]);
    expect(LIGHTNING_BOLT_PRESENTATION.events.map((event) => event.seq)).toEqual([
      900, 901, 902, 903, 904,
    ]);
    expect(
      LIGHTNING_BOLT_PRESENTATION.events.every(
        (event) => event.from_revision === 42 && event.to_revision === 43,
      ),
    ).toBe(true);
  });

  it('recovers exact names from the event when the card has left the board', () => {
    const event = structuredClone(LIGHTNING_BOLT_PRESENTATION.events[0]);
    event.kind = { kind: 'drew', player: 1, object: { entity: 14, incarnation: 3 } };
    event.labels = { objects: { '14:3': 'Island' } };
    const player = createPresentationPlayer();
    player.recover([event], { objects: { '14:3': 'Mountain' }, players: {}, stacks: {} });
    expect(player.labelsAt(0).objects).toEqual({ '14:3': 'Island' });
  });

  it('rejects unordered input instead of silently inventing an order', () => {
    const reversed = [...LIGHTNING_BOLT_PRESENTATION.events].reverse();
    expect(() => validatePresentationEvents(reversed)).toThrow(/strictly ordered/);
  });

  it('rejects an unknown wire kind instead of rendering undefined theater', () => {
    const malformed = structuredClone(LIGHTNING_BOLT_PRESENTATION.events);
    malformed[0].kind = { kind: 'snapshot_changed' } as never;
    expect(() => validatePresentationEvents(malformed)).toThrow(
      /unsupported kind snapshot_changed/,
    );
  });

  it('accepts passed, rules, and earlier causes, and rejects invalid causes or rules', () => {
    const events = structuredClone(LIGHTNING_BOLT_PRESENTATION.events);
    const died = events.find((event) => event.kind.kind === 'died')!;
    died.cause = events[0].seq;
    died.rule = 'lethal_damage';
    events.push({
      ...died,
      seq: died.seq + 100,
      kind: { kind: 'triggered', source: { entity: 9, incarnation: 1 }, controller: 1, stack: 40 },
      cause: died.seq,
      rule: undefined,
    });
    events.push({
      ...died,
      seq: died.seq + 101,
      kind: { kind: 'passed', player: 0 },
      cause: undefined,
      rule: undefined,
    });
    expect(() => validatePresentationEvents(events)).not.toThrow();

    const forward = structuredClone(events);
    forward[0].cause = forward[0].seq;
    expect(() => validatePresentationEvents(forward)).toThrow(/invalid cause/);
    const unknownRule = structuredClone(events);
    unknownRule[0].rule = 'bad_luck' as never;
    expect(() => validatePresentationEvents(unknownRule)).toThrow(/invalid rule/);
  });

  it('accepts ordered per-action transitions inside one authoritative FrameUpdate', () => {
    expect(() =>
      validatePresentationUpdate(LIGHTNING_BOLT_PRESENTATION.events, 41, 43),
    ).not.toThrow();
    expect(() =>
      validatePresentationUpdate(LIGHTNING_BOLT_PRESENTATION.events, 42, 43),
    ).not.toThrow();
    const stepped = structuredClone(LIGHTNING_BOLT_PRESENTATION.events);
    stepped[0].from_revision = 41;
    stepped[0].to_revision = 42;
    expect(() => validatePresentationUpdate(stepped, 41, 43)).not.toThrow();
    for (const event of stepped.slice(1)) event.seq += 1;
    expect(() => validatePresentationUpdate(stepped, 41, 43)).toThrow(/sequence gap/);
  });

  it('requires a contiguous authority-addressed recovery tail', () => {
    expect(validatePresentationTail(LIGHTNING_BOLT_PRESENTATION.events, 900)).toBe(905);
    expect(validatePresentationTail(LIGHTNING_BOLT_PRESENTATION.events.slice(3), 903)).toBe(905);
    expect(validatePresentationTail([], 905)).toBe(905);
    expect(() =>
      validatePresentationTail(LIGHTNING_BOLT_PRESENTATION.events.slice(1), 900),
    ).toThrow(/cursor gap/);
  });
});

describe('PresentationPlayer track store', () => {
  it('keeps the board and replay on the same ordered semantic facts', () => {
    const live = createPresentationPlayer();
    const replay = createPresentationPlayer();
    live.enqueue(COMBAT_TO_TURN_PRESENTATION.events, COMBAT_TO_TURN_PRESENTATION.labels);
    replay.load(COMBAT_TO_TURN_PRESENTATION);
    expect(live.events).toEqual(replay.events);
    expect(live.events.map((event) => event.kind.kind)).toEqual([
      'attack_group',
      'blocked',
      'damage',
      'damage',
      'died',
      'turn_started',
    ]);
  });

  it('keeps pre-transition labels so a dead target remains named', () => {
    const player = createPresentationPlayer();
    player.enqueue(
      LIGHTNING_BOLT_PRESENTATION.events.slice(0, 2),
      LIGHTNING_BOLT_PRESENTATION.labels,
    );
    const postResolutionLabels = structuredClone(LIGHTNING_BOLT_PRESENTATION.labels);
    delete postResolutionLabels.objects['77:0'];
    player.enqueue(
      LIGHTNING_BOLT_PRESENTATION.events.slice(2),
      postResolutionLabels,
    );
    expect(player.labelsAt(3).objects['77:0']).toBe('Ally token');
  });

  it('rejects a stale live batch by sequence number', () => {
    const player = createPresentationPlayer();
    player.enqueue(LIGHTNING_BOLT_PRESENTATION.events, LIGHTNING_BOLT_PRESENTATION.labels);
    expect(() =>
      player.enqueue(
        [LIGHTNING_BOLT_PRESENTATION.events[0]],
        LIGHTNING_BOLT_PRESENTATION.labels,
      ),
    ).toThrow(/not after existing event/);
  });

  it('merges recovery without erasing retained history and flags unavailable prefixes', () => {
    const player = createPresentationPlayer();
    player.setScope('live:one');
    player.enqueue(LIGHTNING_BOLT_PRESENTATION.events.slice(0, 3), LIGHTNING_BOLT_PRESENTATION.labels);
    player.recover(LIGHTNING_BOLT_PRESENTATION.events.slice(1), LIGHTNING_BOLT_PRESENTATION.labels);
    expect(player.events.map(event => event.seq)).toEqual([900, 901, 902, 903, 904]);
    expect(player.unavailablePrefix).toBe(true);
    player.reject();
    expect(player.incomplete).toBe(true);
    expect(player.events).toHaveLength(5);
  });

  it('never mixes games, branch names, or later labels into a retained event', () => {
    const player = createPresentationPlayer();
    player.setScope('live:one');
    player.enqueue(LIGHTNING_BOLT_PRESENTATION.events.slice(0, 1), LIGHTNING_BOLT_PRESENTATION.labels);
    const renamed = structuredClone(LIGHTNING_BOLT_PRESENTATION.labels);
    renamed.objects['31:0'] = 'A later name';
    renamed.objects['999:0'] = 'Unrelated private card';
    player.enqueue(LIGHTNING_BOLT_PRESENTATION.events.slice(1, 2), renamed);
    expect(player.labelsAt(0).objects['31:0']).toBe('Lightning Bolt');
    expect(player.labels.objects['999:0']).toBeUndefined();
    player.setScope('live:two');
    expect(player.events).toEqual([]);
    expect(player.labels.objects).toEqual({});
    expect(player.incomplete).toBe(false);
  });

  it('rebuilds a canonical prefix on backward/forward seek and after Study', () => {
    const player = createPresentationPlayer();
    const batches = LIGHTNING_BOLT_PRESENTATION.events.map(event => ({ events: [event], labels: LIGHTNING_BOLT_PRESENTATION.labels }));
    player.loadPrefix(batches, 'replay:one');
    expect(player.events).toHaveLength(5);
    player.loadPrefix(batches.slice(0, 2), 'replay:one');
    expect(player.events.map(event => event.seq)).toEqual([900, 901]);
    const branchLabels = structuredClone(LIGHTNING_BOLT_PRESENTATION.labels);
    branchLabels.objects['31:0'] = 'Study only';
    player.loadEvents([LIGHTNING_BOLT_PRESENTATION.events[0]], branchLabels, 'study:one');
    expect(player.labelsAt(0).objects['31:0']).toBe('Study only');
    player.loadPrefix(batches, 'replay:one');
    expect(player.events).toHaveLength(5);
    expect(player.labels.objects['31:0']).toBe('Lightning Bolt');
  });

  it('labels an unavailable empty recovery prefix', () => {
    const player = createPresentationPlayer();
    player.recover([], { objects: {}, players: {}, stacks: {} }, 12);
    expect(player.unavailablePrefix).toBe(true);
    player.setScope('another-match');
    expect(player.unavailablePrefix).toBe(false);
  });
});
