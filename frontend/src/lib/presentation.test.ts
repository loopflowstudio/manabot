import { describe, expect, it } from 'vitest';

import {
  COMBAT_TO_TURN_PRESENTATION,
  LIGHTNING_BOLT_PRESENTATION,
  presentationBeat,
  presentationInspectorRows,
  validatePresentationEvents,
  validatePresentationTail,
  validatePresentationUpdate,
} from './presentation';
import { createPresentationPlayer } from './presentation.svelte';

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

  it('projects table and inspector copy from the same canonical events', () => {
    const tableBeats = LIGHTNING_BOLT_PRESENTATION.events.map((event) =>
      presentationBeat(event, LIGHTNING_BOLT_PRESENTATION.labels),
    );
    const inspectorRows = presentationInspectorRows(
      LIGHTNING_BOLT_PRESENTATION.events,
      LIGHTNING_BOLT_PRESENTATION.labels,
    );

    expect(inspectorRows.map(({ event: _event, causedBy: _causedBy, ...beat }) => beat)).toEqual(
      tableBeats,
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
  it('keeps table, replay, and inspector on the same ordered semantic facts', () => {
    const live = createPresentationPlayer();
    const replay = createPresentationPlayer();
    live.enqueue(COMBAT_TO_TURN_PRESENTATION.events, COMBAT_TO_TURN_PRESENTATION.labels);
    replay.load(COMBAT_TO_TURN_PRESENTATION);

    const inspector = presentationInspectorRows(
      COMBAT_TO_TURN_PRESENTATION.events,
      COMBAT_TO_TURN_PRESENTATION.labels,
    );
    expect(COMBAT_TO_TURN_PRESENTATION.events.map((event) => event.kind.kind)).toEqual([
      'attack_group',
      'blocked',
      'damage',
      'damage',
      'died',
      'turn_started',
    ]);
    expect(inspector.map((row) => row.event)).toEqual(COMBAT_TO_TURN_PRESENTATION.events);

    const liveBeats = [];
    const replayBeats = [];
    while (live.currentBeat && replay.currentBeat) {
      liveBeats.push(live.currentBeat);
      replayBeats.push(replay.currentBeat);
      live.advance();
      replay.advance();
    }
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
    expect(player.effectiveDurationMs).toBe(100);
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

    live.enqueue(LIGHTNING_BOLT_PRESENTATION.events, LIGHTNING_BOLT_PRESENTATION.labels);
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

    player.setReducedMotion(true);
    expect(player.effectiveDurationMs).toBe(100);
    expect(player.currentBeat).toEqual(first);

    player.skipCurrent();
    expect(player.currentEvent?.seq).toBe(901);
    player.finishSequence();
    expect(player.currentEvent).toBeNull();
    expect(player.remaining).toBe(0);
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
