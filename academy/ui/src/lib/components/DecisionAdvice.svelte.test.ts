import { readFileSync } from 'node:fs';
import { render } from 'svelte/server';
import { describe, expect, it } from 'vitest';

import type { AdviceScenarioSummary } from '$lib/advice';
import type { DecisionEvidence, StudyArtifact } from '$lib/study-protocol';
import DecisionAdvice from './DecisionAdvice.svelte';

const fixture = JSON.parse(
  readFileSync(
    new URL('../../../../../protocol/fixtures/advice-curated-decision.json', import.meta.url),
    'utf8',
  ),
) as { artifact: StudyArtifact; scenarios: AdviceScenarioSummary[] };

const scenarios = fixture.scenarios.map(
  ({ landmark_id, label, description, inferred_range, belief_kind }) => ({
    landmark_id,
    label,
    description,
    inferred_range,
    belief_kind,
  }),
);
const landmarkA = fixture.artifact.landmarks.find((lm) => lm.id === 'advice-scenario-a')!;
const landmarkB = fixture.artifact.landmarks.find((lm) => lm.id === 'advice-scenario-b')!;
const deltas = scenarioDeltas(landmarkA.evidence, landmarkB.evidence);

function scenarioDeltas(
  left: DecisionEvidence,
  right: DecisionEvidence,
): Record<string, Record<string, number>> {
  const deltas: Record<string, Record<string, number>> = {};
  for (const row of left.policy_mass) {
    const alternative = row.alternative;
    const value = (evidence: DecisionEvidence) => ({
      policy_mass: evidence.policy_mass.find((r) => r.alternative === alternative)?.probability ?? 0,
      search_value:
        evidence.search_value.find((r) => r.alternative === alternative)?.expected_match_points ?? 0,
      uncertainty:
        evidence.uncertainty.find((r) => r.alternative === alternative)?.standard_error ?? 0,
    });
    const a = value(left);
    const b = value(right);
    deltas[alternative] = {
      policy_mass: a.policy_mass - b.policy_mass,
      search_value: a.search_value - b.search_value,
      uncertainty: a.uncertainty - b.uncertainty,
    };
  }
  return deltas;
}

function defaultProps() {
  return {
    mode: 'live' as const,
    scenarios,
    selectedScenarioId: 'advice-scenario-a',
    frame: landmarkA.frame,
    offers: [...landmarkA.frame.offers],
    evidence: landmarkA.evidence,
    deltas,
    status: 'ok' as const,
    reason: null,
    advisorId: fixture.artifact.identity.model.id,
    computeId: fixture.artifact.identity.analysis_budget.id,
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
    expect(body).toContain('conditional-determinized-puct-v1');
    expect(body).toContain('2w-16s-paired-seed-197');
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
          reason: 'legacy_identity_incomplete',
        },
      },
    );
    expect(body).toContain('data-testid="advice-unavailable"');
    expect(body).toContain('data-reason="legacy_identity_incomplete"');
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
