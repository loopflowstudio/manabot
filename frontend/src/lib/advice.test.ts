import { readFileSync } from 'node:fs';

import Ajv2020 from 'ajv/dist/2020.js';
import { describe, expect, it } from 'vitest';

import {
  adviceMeta,
  assertViewerSafeAdviceArtifact,
  computeDeltas,
  expectedIdentity,
  selectAdvice,
  type AdviceArtifact,
} from './advice';

const schema = JSON.parse(
  readFileSync(new URL('../../../protocol/study-v1.schema.json', import.meta.url), 'utf8'),
);
const fixture = JSON.parse(
  readFileSync(
    new URL('../../../protocol/fixtures/advice-curated-decision.json', import.meta.url),
    'utf8',
  ),
) as AdviceArtifact;
const validate = new Ajv2020({ strict: false, validateFormats: false }).compile(
  schema,
);

const SCENARIO_A = 'advice-scenario-a';
const SCENARIO_B = 'advice-scenario-b';

describe('advice fixture and adapter twin', () => {
  it('validates the artifact through the study schema and viewer-safety', () => {
    expect(validate(fixture.artifact), JSON.stringify(validate.errors)).toBe(true);
    expect(() => assertViewerSafeAdviceArtifact(fixture)).not.toThrow();
    expect(fixture.artifact.identity.model.id).toBe('conditional-determinized-puct-v1');
    expect(fixture.artifact.identity.analysis_budget.id).toBe('2w-16s-paired-seed-197');
  });

  it('pins two scenarios at one decision with distinct conditional evidence', () => {
    const landmarks = fixture.artifact.landmarks;
    expect(landmarks).toHaveLength(2);
    expect(new Set(landmarks.map((lm) => lm.decision_id)).size).toBe(1);
    expect(new Set(landmarks.map((lm) => lm.id))).toEqual(new Set([SCENARIO_A, SCENARIO_B]));
    const massA = landmarks[0].evidence.policy_mass.map((r) => r.probability);
    const massB = landmarks[1].evidence.policy_mass.map((r) => r.probability);
    expect(massA).not.toEqual(massB);
    expect(
      [massA, massB].some((mass) => Math.max(...mass) !== Math.min(...mass)),
    ).toBe(true);
    expect(new Set(fixture.scenarios.map((scenario) => scenario.seed_plan))).toEqual(
      new Set(['paired-seed-197']),
    );
    expect(new Set(landmarks.map((lm) => lm.evidence.provenance.producer))).toEqual(
      new Set(['conditional-determinized-puct:v1:paired-seed-197']),
    );
  });

  it('cross-references scenarios to landmarks and keeps opponent hands empty', () => {
    const landmarkIds = new Set(fixture.artifact.landmarks.map((lm) => lm.id));
    const scenarioIds = new Set(fixture.scenarios.map((s) => s.landmark_id));
    expect(scenarioIds).toEqual(landmarkIds);
    for (const landmark of fixture.artifact.landmarks) {
      expect(landmark.frame.projection.opponent.hand).toEqual([]);
    }
  });

  it('rejects cross-scenario seed-plan and producer drift', () => {
    const seedDrift = JSON.parse(JSON.stringify(fixture)) as AdviceArtifact;
    seedDrift.scenarios[1].seed_plan = 'different-seed-plan';
    expect(() => assertViewerSafeAdviceArtifact(seedDrift)).toThrow(
      'advice scenarios must share one paired seed plan',
    );

    const producerDrift = JSON.parse(JSON.stringify(fixture)) as AdviceArtifact;
    producerDrift.artifact.landmarks[1].evidence.provenance.producer = 'different-producer';
    expect(() => assertViewerSafeAdviceArtifact(producerDrift)).toThrow(
      'advice scenarios must share one producer identity',
    );
  });

  it('bootstraps the pinned address, scenarios, and expected identity', () => {
    const meta = adviceMeta(fixture);
    expect(meta.address.startsWith('erd1.')).toBe(true);
    expect(meta.scenarios).toHaveLength(2);
    expect(meta.identity.advisor_id).toBe('conditional-determinized-puct-v1');
    expect(meta.identity.compute_id).toBe('2w-16s-paired-seed-197');
  });

  it('returns real evidence and deltas for each scenario', () => {
    const meta = adviceMeta(fixture);
    for (const scenarioId of [SCENARIO_A, SCENARIO_B]) {
      const response = selectAdvice(fixture, meta.address, scenarioId, meta.identity);
      expect(response.status).toBe('ok');
      expect(response.reason).toBeNull();
      expect(response.frame).not.toBeNull();
      expect(response.offers.length).toBeGreaterThan(0);
      expect(response.scenario?.landmark_id).toBe(scenarioId);
      expect(response.evidence).not.toBeNull();
      expect(response.deltas).not.toBeNull();
    }
  });

  it('fails closed on identity mismatch, unknown scenario, and wrong address', () => {
    const meta = adviceMeta(fixture);
    const wrong = { ...meta.identity, advisor_id: 'not-the-advisor' };
    expect(selectAdvice(fixture, meta.address, SCENARIO_A, wrong)).toMatchObject({
      status: 'unavailable',
      reason: 'legacy_identity_incomplete',
      evidence: null,
    });
    expect(selectAdvice(fixture, meta.address, 'advice-scenario-z', meta.identity)).toMatchObject({
      status: 'unavailable',
      reason: 'scenario_not_found',
      evidence: null,
    });
    expect(selectAdvice(fixture, 'erd1.wrong', SCENARIO_A, meta.identity)).toMatchObject({
      status: 'unavailable',
      reason: 'decision_not_found',
      evidence: null,
    });
  });

  it('computes antisymmetric non-zero deltas between scenarios', () => {
    const a = fixture.artifact.landmarks.find((lm) => lm.id === SCENARIO_A)!;
    const b = fixture.artifact.landmarks.find((lm) => lm.id === SCENARIO_B)!;
    const forward = computeDeltas(a.evidence, b.evidence);
    const reverse = computeDeltas(b.evidence, a.evidence);
    const alternatives = a.alternatives.map((alt) => alt.id);
    expect([...Object.keys(forward)].sort()).toEqual([...alternatives].sort());
    let anyNonZero = false;
    for (const alt of alternatives) {
      expect(Object.keys(forward[alt]).sort()).toEqual(
        ['policy_mass', 'search_value', 'uncertainty'].sort(),
      );
      for (const metric of ['policy_mass', 'search_value', 'uncertainty'] as const) {
        expect(forward[alt][metric]).toBeCloseTo(-reverse[alt][metric], 10);
        if (Math.abs(forward[alt][metric]) > 1e-9) anyNonZero = true;
      }
    }
    expect(anyNonZero).toBe(true);
  });

  it('keeps the expected identity stable against the fixture', () => {
    expect(expectedIdentity(fixture)).toEqual(adviceMeta(fixture).identity);
  });
});
