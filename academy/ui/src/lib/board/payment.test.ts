import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import type { PaymentPreview } from '$lib/payment-preview';
import type { ExperienceFrame } from '$lib/types';
import { payLabel, paymentPlan, poolSymbols } from './payment';

const fixture = JSON.parse(readFileSync(new URL('../../../../../managym/academy/fixtures/choice-navigation-w17-smart-payment.json', import.meta.url), 'utf8')) as {
  positions: Record<string, { frame: ExperienceFrame }>;
};
const pay = fixture.positions.PAY_OR_NOT.frame;
const [mountain, island] = [88, 128];

function preview(state: PaymentPreview['state'], sources: [number, number[]][], pool: number[] = [0, 0, 0, 0, 0, 0]): PaymentPreview {
  return {
    state, explanation: '', cost: '{1}{U}', contributions: [],
    sources: sources.map(([id, amounts]) => ({ source: { kind: 'object', id: { entity: id, incarnation: 2 } }, label: 'Land', output: { amounts, classes: [] } })),
    pool_after_production: { amounts: pool, classes: [] }, pool_after_spending: null,
  };
}

describe('paymentPlan', () => {
  it('marks each land the payer would tap with its order and output', () => {
    const plan = paymentPlan(preview('exact', [[mountain, [0, 0, 0, 1, 0, 0]], [island, [0, 1, 0, 0, 0, 0]]]), pay)!;
    expect([...plan.sources]).toEqual([[mountain, { order: 1, output: '{R}' }], [island, { order: 2, output: '{U}' }]]);
    expect(plan.names).toEqual(['Mountain', 'Island']);
    expect(payLabel('{1}{U}', plan, null)).toBe('Pay {1}{U} · Mountain, Island');
  });

  it('counts repeated sources on the button', () => {
    const plan = paymentPlan(preview('exact', [[island, [0, 1, 0, 0, 0, 0]], [mountain, [0, 0, 0, 1, 0, 0]], [117, [0, 0, 0, 1, 0, 0]], [99, [0, 0, 0, 1, 0, 0]]]), pay)!;
    expect(payLabel('{3}{U}', plan, null)).toBe('Pay {3}{U} · Island, Mountain ×3');
  });

  it('marks nothing unless the preview is exact', () => {
    expect(paymentPlan(preview('needs_choice', [[mountain, [0, 0, 0, 1, 0, 0]]]), pay)).toBeNull();
    expect(paymentPlan(null, pay)).toBeNull();
    expect(payLabel('{1}{U}', null, null)).toBe('Pay {1}{U}');
  });

  it('says floating mana pays when no source is tapped', () => {
    const answer = preview('exact', [], [0, 2, 0, 0, 0, 0]);
    expect(payLabel('{1}{U}', paymentPlan(answer, pay), answer)).toBe('Pay {1}{U} · floating mana');
  });

  it('writes pools as symbols in WUBRGC order', () => {
    expect(poolSymbols([1, 0, 0, 2, 0, 1])).toBe('{W}{R}{R}{C}');
    expect(poolSymbols([0, 0, 0, 0, 0, 0])).toBe('');
  });
});
