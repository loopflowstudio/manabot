import { describe, expect, it } from 'vitest';

import {
  presentationBeat,
  validatePresentationEvents,
  validatePresentationTail,
  validatePresentationUpdate,
} from './presentation';
import { createPresentationPlayer } from './presentation.svelte';
import { historyContext } from './presentation-history';
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

  it('projects table copy from the canonical events', () => {
    const tableBeats = LIGHTNING_BOLT_PRESENTATION.events.map((event) =>
      presentationBeat(event, LIGHTNING_BOLT_PRESENTATION.labels),
    );
    expect(tableBeats.map((beat) => beat.detail)).toEqual([
      'Hero casts Lightning Bolt.',
      'Lightning Bolt targets Ally token.',
      'Lightning Bolt resolves.',
      'Lightning Bolt deals 3 damage to Ally token.',
      'Ally token dies.',
    ]);
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

describe('curated combat-to-turn presentation fixture', () => {
  it('keeps table and replay on the same ordered semantic facts', () => {
    const live = createPresentationPlayer();
    const replay = createPresentationPlayer();
    live.enqueue(COMBAT_TO_TURN_PRESENTATION.events, COMBAT_TO_TURN_PRESENTATION.labels);
    replay.load(COMBAT_TO_TURN_PRESENTATION);

    expect(COMBAT_TO_TURN_PRESENTATION.events.map((event) => event.kind.kind)).toEqual([
      'attack_group',
      'blocked',
      'damage',
      'damage',
      'died',
      'turn_started',
    ]);

    // Live motion may catch up, but semantic history remains identical.
    const liveBeats = live.events.map((event, index) => presentationBeat(event, live.labelsAt(index)));
    const replayBeats = replay.events.map((event, index) => presentationBeat(event, replay.labelsAt(index)));
    expect(liveBeats).toEqual(replayBeats);
    expect(liveBeats.map((beat) => beat.detail)).toEqual([
      'Badgermole Cub attacks Hero.',
      'Otter-Penguin blocks Badgermole Cub.',
      'Otter-Penguin deals 2 damage to Badgermole Cub.',
      'Badgermole Cub deals 1 damage to Otter-Penguin.',
      'Otter-Penguin, Badgermole Cub die.',
      "Hero's turn begins.",
    ]);
  });

  it('supports skip, fast-forward, and reduced motion without changing combat facts', () => {
    const player = createPresentationPlayer();
    player.load(COMBAT_TO_TURN_PRESENTATION);
    const facts = structuredClone(player.events);

    player.setFastForward(true);
    expect(player.effectiveDurationMs).toBe(125);
    player.setReducedMotion(true);
    expect(player.effectiveDurationMs).toBe(125);
    player.skipCurrent();
    expect(player.currentEvent?.kind.kind).toBe('blocked');
    player.finishSequence();

    expect(player.events).toEqual(facts);
    expect(player.remaining).toBe(0);
  });
});

describe('PresentationPlayer', () => {
  it('gives live and replay paths the same first semantic beat', () => {
    const live = createPresentationPlayer();
    const replay = createPresentationPlayer();

    live.enqueue(LIGHTNING_BOLT_PRESENTATION.events.slice(0, 2), LIGHTNING_BOLT_PRESENTATION.labels);
    replay.load(LIGHTNING_BOLT_PRESENTATION);

    expect(live.currentBeat).toEqual(replay.currentBeat);
    live.advance();
    replay.advance();
    expect(live.currentBeat).toEqual(replay.currentBeat);
  });

  it('supports skip, fast-forward, and reduced-motion timing without changing facts', () => {
    const player = createPresentationPlayer();
    player.load(LIGHTNING_BOLT_PRESENTATION);
    const first = player.currentBeat;
    const normalDuration = player.effectiveDurationMs;

    player.setFastForward(true);
    expect(player.effectiveDurationMs).toBeLessThan(normalDuration);
    expect(player.currentBeat).toEqual(first);

    const fastDuration = player.effectiveDurationMs;
    player.setReducedMotion(true);
    expect(player.effectiveDurationMs).toBe(fastDuration);
    expect(player.currentBeat).toEqual(first);

    player.skipCurrent();
    expect(player.currentEvent?.seq).toBe(901);
    player.finishSequence();
    expect(player.currentEvent).toBeNull();
    expect(player.remaining).toBe(0);
    expect(player.events).toHaveLength(5);
  });

  it('pauses reading without changing the cursor or losing skipped consequences', () => {
    const player = createPresentationPlayer();
    player.load(COMBAT_TO_TURN_PRESENTATION);
    const first = player.currentBeat;
    player.togglePlayback();
    expect(player.playing).toBe(false);
    expect(player.currentBeat).toEqual(first);
    player.togglePlayback();
    expect(player.playing).toBe(true);
    const duration = player.effectiveDurationMs;
    player.setReducedMotion(true);
    expect(player.effectiveDurationMs).toBe(duration);
    player.finishSequence();
    expect(player.events).toEqual(COMBAT_TO_TURN_PRESENTATION.events);
    player.togglePlayback();
    expect(player.playing).toBe(false);
  });

  it('keeps narration paused when another committed update arrives', () => {
    const player = createPresentationPlayer();
    player.enqueue(LIGHTNING_BOLT_PRESENTATION.events.slice(0, 2), LIGHTNING_BOLT_PRESENTATION.labels);
    player.togglePlayback();
    player.enqueue(LIGHTNING_BOLT_PRESENTATION.events.slice(2), LIGHTNING_BOLT_PRESENTATION.labels);
    expect(player.playing).toBe(false);
    expect(player.currentEvent?.seq).toBe(902);
    expect(player.events).toHaveLength(5);
  });

  it('keeps pre-transition labels so a dead target remains named', () => {
    const player = createPresentationPlayer();
    player.enqueue(
      LIGHTNING_BOLT_PRESENTATION.events.slice(0, 2),
      LIGHTNING_BOLT_PRESENTATION.labels,
    );
    player.finishSequence();

    const postResolutionLabels = structuredClone(LIGHTNING_BOLT_PRESENTATION.labels);
    delete postResolutionLabels.objects['77:0'];
    player.enqueue(
      LIGHTNING_BOLT_PRESENTATION.events.slice(2),
      postResolutionLabels,
    );
    player.advance();

    expect(player.currentBeat?.detail).toBe(
      'Lightning Bolt deals 3 damage to Ally token.',
    );
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
});

describe('scoped retained history', () => {
  it('bounds paused live admission by count and duration, retaining all facts', () => {
    const player = createPresentationPlayer();
    const events = Array.from({ length: 18 }, (_, seq) => ({ ...LIGHTNING_BOLT_PRESENTATION.events[0], seq, suggested_ms: 80 }));
    player.enqueue(events.slice(0, 1), LIGHTNING_BOLT_PRESENTATION.labels);
    player.togglePlayback();
    player.enqueue(events.slice(1), LIGHTNING_BOLT_PRESENTATION.labels);
    expect(player.playing).toBe(false);
    expect(player.remaining).toBe(12);
    expect(player.pendingDurationMs).toBe(960);
    expect(player.events).toHaveLength(18);
    player.enqueue([{ ...events[0], seq: 18, suggested_ms: 5000 }], LIGHTNING_BOLT_PRESENTATION.labels);
    expect(player.remaining).toBe(1);
    expect(player.pendingDurationMs).toBe(2000);
    expect(player.events).toHaveLength(19);
  });

  it('reapplies the live duration cap when leaving fast-forward', () => {
    const player = createPresentationPlayer();
    player.setFastForward(true);
    player.enqueue(LIGHTNING_BOLT_PRESENTATION.events, LIGHTNING_BOLT_PRESENTATION.labels);
    expect(player.remaining).toBe(5);
    player.setFastForward(false);
    expect(player.pendingDurationMs).toBeLessThanOrEqual(2000);
    expect(player.remaining).toBe(3);
    expect(player.events).toHaveLength(5);
  });

  it('merges recovery without repeating seen motion or erasing reading history', () => {
    const player = createPresentationPlayer();
    player.setScope('live:one');
    player.enqueue(LIGHTNING_BOLT_PRESENTATION.events.slice(0, 3), LIGHTNING_BOLT_PRESENTATION.labels);
    player.recover(LIGHTNING_BOLT_PRESENTATION.events.slice(1), LIGHTNING_BOLT_PRESENTATION.labels);
    expect(player.events.map(event => event.seq)).toEqual([900, 901, 902, 903, 904]);
    expect(player.currentEvent?.seq).toBe(903);
    player.recover(LIGHTNING_BOLT_PRESENTATION.events, LIGHTNING_BOLT_PRESENTATION.labels);
    expect(player.currentEvent).toBeNull();
    expect(player.events).toHaveLength(5);
    expect(player.unavailablePrefix).toBe(true);
    player.enqueue([{ ...LIGHTNING_BOLT_PRESENTATION.events[0], seq: 910 }], LIGHTNING_BOLT_PRESENTATION.labels);
    expect(historyContext(player.events).at(-1)?.gap).toEqual({ first: 905, last: 909 });
    player.reject();
    expect(player.incomplete).toBe(true);
    expect(player.currentEvent).toBeNull();
    expect(player.events).toHaveLength(6);
  });

  it('never mixes games, branch names, or later labels into a retained event', () => {
    const player = createPresentationPlayer();
    player.setScope('live:one');
    player.enqueue(LIGHTNING_BOLT_PRESENTATION.events.slice(0, 1), LIGHTNING_BOLT_PRESENTATION.labels);
    const renamed = structuredClone(LIGHTNING_BOLT_PRESENTATION.labels);
    renamed.objects['31:0'] = 'A later name';
    renamed.objects['999:0'] = 'Unrelated private card';
    player.enqueue(LIGHTNING_BOLT_PRESENTATION.events.slice(1, 2), renamed);
    expect(presentationBeat(player.events[0], player.labelsAt(0)).detail).toContain('Lightning Bolt');
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
    expect(player.currentEvent?.seq).toBe(904);
    player.loadPrefix(batches.slice(0, 2), 'replay:one');
    expect(player.events.map(event => event.seq)).toEqual([900, 901]);
    const branchLabels = structuredClone(LIGHTNING_BOLT_PRESENTATION.labels);
    branchLabels.objects['31:0'] = 'Study only';
    player.loadEvents([LIGHTNING_BOLT_PRESENTATION.events[0]], branchLabels, 'study:one');
    expect(player.currentBeat?.detail).toContain('Study only');
    player.loadPrefix(batches, 'replay:one', false);
    expect(player.events).toHaveLength(5);
    expect(player.currentEvent).toBeNull();
    expect(player.labels.objects['31:0']).toBe('Lightning Bolt');
  });
});


describe('recovery history gaps', () => {
  it('fills previously unavailable rows by identity without replaying seen facts', () => {
    const player = createPresentationPlayer();
    const fixture = LIGHTNING_BOLT_PRESENTATION;
    player.enqueue([fixture.events[0]], fixture.labels);
    player.recover(fixture.events.slice(3), fixture.labels);
    expect(historyContext(player.events).at(-2)?.gap).toEqual({ first: 901, last: 902 });
    player.recover(fixture.events, fixture.labels);
    expect(player.events.map(event => event.seq)).toEqual([900, 901, 902, 903, 904]);
    expect(historyContext(player.events).every(row => row.gap === null)).toBe(true);
    expect(player.currentEvent?.seq).toBe(901);
    expect(player.remaining).toBe(2);
    player.recover(fixture.events, fixture.labels);
    expect(player.remaining).toBe(0);
  });

  it('labels an unavailable empty recovery prefix', () => {
    const player = createPresentationPlayer();
    player.recover([], { objects: {}, players: {}, stacks: {} }, 12);
    expect(player.unavailablePrefix).toBe(true);
    player.setScope('another-match');
    expect(player.unavailablePrefix).toBe(false);
  });
});
