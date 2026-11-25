import { describe, expect, it } from 'vitest';
import { defaultStops } from '$lib/stops';
import { emptyOnce, nextStopState, phaseTrack, pruneOnce, spentOnce, withStopState } from './phases';

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

  it('spends a once stop only where the game stopped, on the turn it belongs to', () => {
    const { stops, once } = withStopState(defaultStops(), emptyOnce(), 'opponent', 'upkeep', 'once');
    expect(spentOnce(turn('BEGINNING_UPKEEP'), false, stops, once)).toEqual({ side: 'opponent', step: 'upkeep' });
    expect(spentOnce(turn('BEGINNING_UPKEEP'), true, stops, once)).toBeNull();
    expect(spentOnce(turn('BEGINNING_DRAW'), false, stops, once)).toBeNull();
    // An every-time stop is never spent.
    expect(spentOnce(turn('ENDING_END'), false, stops, once)).toBeNull();
    // In Full control every priority choice pauses; nothing is spent.
    expect(spentOnce(turn('BEGINNING_UPKEEP'), false, { ...stops, auto_pass: false }, once)).toBeNull();
    // Non-priority decisions (declaring attackers/blockers) never spend a once stop.
    const combatOnce = withStopState(defaultStops(), emptyOnce(), 'my', 'declare_attackers', 'once');
    expect(spentOnce(turn('COMBAT_DECLARE_ATTACKERS'), true, combatOnce.stops, combatOnce.once, 'DECLARE_ATTACKER')).toBeNull();
    expect(spentOnce(turn('COMBAT_DECLARE_ATTACKERS'), true, combatOnce.stops, combatOnce.once, 'PRIORITY')).toEqual({
      side: 'my',
      step: 'declare_attackers',
    });
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
