import { readFileSync } from 'node:fs';
import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';

import {
  adviceMeta,
  computeDeltas,
  type AdviceArtifact,
} from '$lib/advice';
import DecisionAdvice from './DecisionAdvice.svelte';

const fixture = JSON.parse(
  readFileSync(
    new URL('../../../../protocol/fixtures/advice-curated-decision.json', import.meta.url),
    'utf8',
  ),
) as AdviceArtifact;

const meta = adviceMeta(fixture);
const landmarkA = fixture.artifact.landmarks.find((lm) => lm.id === 'advice-scenario-a')!;
const landmarkB = fixture.artifact.landmarks.find((lm) => lm.id === 'advice-scenario-b')!;
const deltas = computeDeltas(landmarkA.evidence, landmarkB.evidence);

function defaultProps() {
  return {
    mode: 'live' as const,
    scenarios: meta.scenarios,
    selectedScenarioId: 'advice-scenario-a',
    frame: landmarkA.frame,
    offers: [...landmarkA.frame.offers],
    evidence: landmarkA.evidence,
    deltas,
    status: 'ok' as const,
    reason: null,
    advisorId: meta.identity.advisor_id,
    computeId: meta.identity.compute_id,
    onSelectScenario: () => {},
  };
}

describe('DecisionAdvice', () => {
  it('renders four distinct regions with the shared action vocabulary in live mode', () => {
    const { body } = render(DecisionAdvice, { props: defaultProps() });
    expect(body).toContain('data-testid="decision-advice"');
    expect(body).toContain('data-mode="live"');
    expect(body).toContain('data-testid="advice-beliefs"');
    expect(body).toContain('data-testid="advice-facts"');
    expect(body).toContain('data-testid="advice-advice"');
    expect(body).toContain('data-testid="advice-deltas"');
    expect(body).toContain('data-testid="advice-footer"');
    // Two belief scenarios, identity-pinned by landmark id.
    expect(body.match(/data-testid="advice-scenario-option"/g)).toHaveLength(2);
    expect(body).toContain('data-scenario-id="advice-scenario-a"');
    expect(body).toContain('data-scenario-id="advice-scenario-b"');
    // Two action rows keyed by the shared action vocabulary.
    expect(body.match(/data-testid="advice-action-row"/g)).toHaveLength(2);
    expect(body).toContain('data-action-id="offer-0"');
    expect(body).toContain('data-action-id="offer-1"');
    expect(body).toContain('Play Mountain');
    expect(body).toContain('Pass priority');
  });

  it('renders the advisor identity and advisory-only footer', () => {
    const { body } = render(DecisionAdvice, { props: defaultProps() });
    expect(body).toContain('flat-mc-search-v1');
    expect(body).toContain('1w-8r-16s');
    expect(body).toContain('advisory only');
  });

  it('states the strategy is conditional on the selected belief, not unconditional', () => {
    const { body } = render(DecisionAdvice, { props: defaultProps() });
    // Truthful conditional-vs-unconditional wording: the Advice region frames
    // the distribution as conditional on the belief, and the footer reinforces
    // it. No hidden-truth access and no parallel rules/search meaning.
    expect(body).toContain('Advice given this belief');
    expect(body).toContain('conditional on the selected belief');
    expect(body).toContain('not an unconditional advisor verdict');
    // The Advice region's ARIA label also names the conditional framing.
    expect(body).toContain('aria-label="Advisor evidence conditional on the selected belief"');
  });

  it('exposes study mode and a reduced-motion data attribute', () => {
    const { body } = render(
      DecisionAdvice,
      { props: { ...defaultProps(), mode: 'study', reducedMotion: true } },
    );
    expect(body).toContain('data-mode="study"');
    expect(body).toContain('data-reduced-motion="true"');
  });

  it('renders a typed unavailable state with no evidence on identity mismatch', () => {
    const { body } = render(
      DecisionAdvice,
      {
        props: {
          ...defaultProps(),
          frame: null,
          offers: [],
          evidence: null,
          deltas: null,
          status: 'unavailable',
          reason: 'identity_mismatch',
        },
      },
    );
    expect(body).toContain('data-testid="advice-unavailable"');
    expect(body).toContain('data-reason="identity_mismatch"');
    expect(body).not.toContain('data-testid="advice-facts"');
    expect(body).not.toContain('data-testid="advice-advice"');
  });

  it('renders facts from the public board without opponent hand identities', () => {
    const { body } = render(DecisionAdvice, { props: defaultProps() });
    expect(body).toContain('Hero life');
    expect(body).toContain('Opponent life');
    expect(body).toContain('(hidden)');
    // The opponent hand is viewer-safe (empty); no private card name leaks.
    expect(body).not.toContain('Secret Counterspell');
  });
});
