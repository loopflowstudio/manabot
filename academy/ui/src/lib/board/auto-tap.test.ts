import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

import type { ExperienceFrame } from '$lib/types';
import { autoTapNote, autoTapOffer, nextFullControl } from './auto-tap';
import { boardView } from './view';

const PLAYERS = { you: 'You', opponent: 'Manabot' };
const fixture = JSON.parse(
  readFileSync(new URL('../../../../../managym/academy/fixtures/choice-navigation-w17-smart-payment.json', import.meta.url), 'utf8'),
) as { positions: Record<string, { frame: ExperienceFrame }> };

const payFrame = fixture.positions.PAY_OR_NOT.frame;

describe('auto-tap', () => {
  it('auto-submits mandatory pure-mana PAY_OR_NOT prompts when enabled and not in full control', () => {
    const view = boardView(payFrame, PLAYERS);
    const offer = autoTapOffer(payFrame, view, { autoTap: true, fullControl: 'off' });
    expect(offer?.offer.verb).toBe('pay');
    expect(autoTapNote(payFrame)).toMatch(/^Auto-tapped \{/);
  });

  it('pauses when auto-tap is off or Full Control (once or hold) is active', () => {
    const view = boardView(payFrame, PLAYERS);
    expect(autoTapOffer(payFrame, view, { autoTap: false, fullControl: 'off' })).toBeNull();
    expect(autoTapOffer(payFrame, view, { autoTap: true, fullControl: 'once' })).toBeNull();
    expect(autoTapOffer(payFrame, view, { autoTap: true, fullControl: 'hold' })).toBeNull();
  });

  it('never auto-pays optional costs (decline present) or non-mana choices (e.g. convoke/sacrifice)', () => {
    const optional = structuredClone(payFrame);
    optional.offers.push({
      ...optional.offers[0],
      id: 999,
      verb: 'choose',
      action_type: 'DECLINE',
      label: 'Decline',
    });
    expect(autoTapOffer(optional, boardView(optional, PLAYERS), { autoTap: true, fullControl: 'off' })).toBeNull();

    const sacrifice = structuredClone(payFrame);
    sacrifice.offers.push({
      ...sacrifice.offers[0],
      id: 998,
      verb: 'pay',
      action_type: 'SACRIFICE_PERMANENT',
      label: 'Sacrifice Clue',
    });
    expect(autoTapOffer(sacrifice, boardView(sacrifice, PLAYERS), { autoTap: true, fullControl: 'off' })).toBeNull();
  });

  it('transitions Full Control on Ctrl and Shift+Ctrl like Arena', () => {
    expect(nextFullControl('off', false)).toBe('once');
    expect(nextFullControl('once', false)).toBe('off');
    expect(nextFullControl('hold', false)).toBe('off');
    expect(nextFullControl('off', true)).toBe('hold');
    expect(nextFullControl('once', true)).toBe('hold');
    expect(nextFullControl('hold', true)).toBe('off');
  });
});
