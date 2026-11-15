import {readFileSync} from 'node:fs';
import type {StudyArtifact} from '$lib/study-protocol';
import { render } from 'svelte/server';
import { expect, it } from 'vitest';
import RecordedDecisionEvidence from './RecordedDecisionEvidence.svelte';

  it('distinguishes retained base and updated search policy with exact seed evidence', () => {
    const { body } = render(RecordedDecisionEvidence, { props: {
      offers: (JSON.parse(readFileSync(new URL('../../../../../protocol/fixtures/advice-curated-decision.json', import.meta.url), 'utf8')) as {artifact:StudyArtifact}).artifact.landmarks[0].frame.offers, recorded: {
        action: 0, probabilities: [0.8, 0.2], value: 0.25,
        value_semantic: 'signed_outcome', search_scores: [0.5, -0.5],
        unavailable: null, factors: [],
        belief: { model: 'managym.determinize/physical-deal/v1', digest: 'abc', condition_mass: 1, sampled_worlds: 2 },
        mirror: { algorithm: 'regularized-local-update', worlds: 2, depth: 3,
          alpha: 0.02, beta: 1, rollouts: 4, seconds: 0.123, decision_seconds: 30,
          seed: 'ffffffffffffffff', prior: [0.5, 0.5], receipt_sha256: 'f'.repeat(64) },
      },
    } });
    expect(body).toContain('Raw policy');
    expect(body).toContain('Updated policy');
    expect(body).toContain('Search values');
    expect(body).toContain('Search:');
    expect(body).toContain('managym.determinize/physical-deal/v1');
    expect(body).toContain('50.00%');
    expect(body).toContain('80.00%');
    expect(body).toContain('Uniform belief (physical deal)');
    expect(body).toContain('ffffffffffffffff');
    expect(body).toContain('0.123 s');
  });
