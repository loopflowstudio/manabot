import { describe, expect, it } from 'vitest';
import { defaultStops } from '$lib/stops';
import { emptyOnce, nextHeldOnce, nextStopState, onceHeldAt, phaseTrack, pruneOnce, withStopState } from './phases';

const turn = (step: string) => ({ turn_number: 3, active_player: 0, active_player_id: 1, agent_player: 0, step }) as never;

describe('stops per step, for both turns', () => {
  it('cycles off, every time, once, off', () => {
    let stops = defaultStops();
    let once = emptyOnce();
    const states: string[] = [];
    for (let press = 0; press < 3; press++) {
      const state = phaseTrack('PRECOMBAT_MAIN_STEP', stops, true, once).marks.find(mark => mark.key === 'upkeep')!.state.opponent;
      ({ stops, once } = withStopState(stops, once, 'opponent', 'upkeep', nextStopState(state)));
      states.push(phaseTrack('PRECOMBAT_MAIN_STEP', stops, true, once).marks.find(mark => mark.key === 'upkeep')!.state.opponent);
    }
    expect(states).toEqual(['every', 'once', 'off']);
    expect(stops).toEqual(defaultStops());
    expect(once).toEqual(emptyOnce());
  });

  it('holds a once stop in the config the server sees', () => {
    const { stops } = withStopState(defaultStops(), emptyOnce(), 'my', 'upkeep', 'once');
    expect(stops.my).toContain('upkeep');
  });

  it('holds a once stop only at a priority window on the turn it belongs to', () => {
    const { stops, once } = withStopState(defaultStops(), emptyOnce(), 'opponent', 'upkeep', 'once');
    expect(onceHeldAt(turn('BEGINNING_UPKEEP'), false, stops, once)).toEqual({ side: 'opponent', step: 'upkeep' });
    expect(onceHeldAt(turn('BEGINNING_UPKEEP'), true, stops, once)).toBeNull();
    expect(onceHeldAt(turn('BEGINNING_DRAW'), false, stops, once)).toBeNull();
    // An every-time stop is never held once.
    expect(onceHeldAt(turn('ENDING_END'), false, stops, once)).toBeNull();
    // In Full control every priority choice pauses; nothing is held once.
    expect(onceHeldAt(turn('BEGINNING_UPKEEP'), false, { ...stops, auto_pass: false }, once)).toBeNull();
    // Non-priority decisions (declaring attackers/blockers) never hold a once stop.
    const combatOnce = withStopState(defaultStops(), emptyOnce(), 'my', 'declare_attackers', 'once');
    expect(onceHeldAt(turn('COMBAT_DECLARE_ATTACKERS'), true, combatOnce.stops, combatOnce.once, 'DECLARE_ATTACKER')).toBeNull();
    expect(onceHeldAt(turn('COMBAT_DECLARE_ATTACKERS'), true, combatOnce.stops, combatOnce.once, 'PRIORITY')).toEqual({
      side: 'my',
      step: 'declare_attackers',
    });
  });

  it('spends a held once stop when the person leaves its window, not on arrival', () => {
    const attackers = { match: 'm', turn: 3, side: 'my' as const, step: 'declare_attackers' };
    const arrived = { side: 'my' as const, step: 'declare_attackers' };
    // Arriving holds it; nothing is spent while the person is still there.
    let next = nextHeldOnce(null, attackers, arrived);
    expect(next).toEqual({ held: attackers, spend: null });
    // Casting, targeting and paying within the window keep it held.
    next = nextHeldOnce(next.held, attackers, null);
    expect(next).toEqual({ held: attackers, spend: null });
    // Leaving for blockers spends it.
    next = nextHeldOnce(next.held, { ...attackers, step: 'declare_blockers' }, null);
    expect(next).toEqual({ held: null, spend: arrived });
    // A new match, later turn, or the other side's turn also leaves the window.
    expect(nextHeldOnce(attackers, { ...attackers, match: 'n' }, null).spend).toEqual(arrived);
    expect(nextHeldOnce(attackers, { ...attackers, turn: 5 }, null).spend).toEqual(arrived);
    expect(nextHeldOnce(attackers, { ...attackers, side: 'opponent' }, null).spend).toEqual(arrived);
  });

  it('forgets once for a stop the Menu turned off', () => {
    const { stops, once } = withStopState(defaultStops(), emptyOnce(), 'my', 'main2', 'once');
    expect(pruneOnce(once, { ...stops, my: ['main1'] })).toEqual(emptyOnce());
  });

  it('marks the current step and rules each phase', () => {
    const track = phaseTrack('COMBAT_DECLARE_BLOCKERS', defaultStops(), false);
    expect(track.marks.filter(mark => mark.current).map(mark => mark.key)).toEqual(['declare_blockers']);
    expect(track.marks.filter(mark => mark.phaseStart).map(mark => mark.short)).toEqual(['Upkeep', 'Main', 'Combat', 'Main 2', 'End']);
    expect(track.active).toBe('opponent');
  });
});
