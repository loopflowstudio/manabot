import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import {
  serializeReplayDecisionAddress,
  type CanonicalReplayProjectionV1,
  type CanonicalReplayProjectionResponseV1,
  type RestoredReplayDecision,
} from './replay-index';
import { bindStudyArtifact, buildStudyPlans } from './study-runtime';
import type { StudyArtifact } from './study-protocol';

function fixture<T>(name: string): T {
  return JSON.parse(
    readFileSync(new URL(`../../../protocol/fixtures/${name}`, import.meta.url), 'utf8'),
  ) as T;
}

const replay = fixture<CanonicalReplayProjectionV1>('canonical-replay-player-0.json');
const artifact = fixture<StudyArtifact>('study-curated-decision.json');

function projectionAndRestored(): {
  projection: CanonicalReplayProjectionResponseV1;
  restored: RestoredReplayDecision;
} {
  const addressedDecisions = replay.decisions.map((row) => ({
    ...structuredClone(row),
    address: serializeReplayDecisionAddress({
      version: 1,
      replay_id: replay.replay_id,
      match_id: replay.match_id,
      ordinal: String(row.ordinal),
      viewer: String(row.viewer),
      revision: String(row.revision),
      prompt_id: String(row.prompt_id),
      offer_id: String(row.offer_id),
      command_id: row.command_id,
      presentation_cursor: String(row.presentation_cursor),
      decision_sha256: '0'.repeat(64),
    }),
  }));
  addressedDecisions[0].address = artifact.landmarks[0].decision_id;
  const projection: CanonicalReplayProjectionResponseV1 = {
    ...structuredClone(replay),
    decisions: addressedDecisions,
  };
  const row = projection.decisions[0];
  return {
    projection,
    restored: {
      address: row.address,
      ordinal: row.ordinal,
      viewer: row.viewer,
      revision: row.revision,
      presentation_cursor: row.presentation_cursor,
      frame: structuredClone(row.frame),
      offer: structuredClone(row.offer),
      command: structuredClone(row.command),
      continuation: [],
    },
  };
}

describe('Study runtime consumer', () => {
  it('joins exact evidence and keeps Played, Policy, and Search distinct', async () => {
    const { projection, restored } = projectionAndRestored();
    const landmark = await bindStudyArtifact(artifact, projection, restored);
    const plans = buildStudyPlans(landmark);

    expect(plans.map(({ label }) => label)).toEqual(['Played', 'Policy', 'Search']);
    expect(plans[0]).toMatchObject({
      kind: 'played',
      policyProbability: null,
      expectedMatchPoints: null,
      visits: null,
    });
    expect(plans[1]).toMatchObject({
      kind: 'policy',
      policyProbability: 0.5,
      expectedMatchPoints: 0,
      visits: 1,
    });
    expect(plans[2]).toMatchObject({
      kind: 'search',
      policyProbability: 0.5,
      expectedMatchPoints: 0,
      visits: 1,
      favorableWorlds: 1,
      sampledWorlds: 1,
      standardError: 0,
      uncertaintyMethod: 'fixture',
    });
  });

  it('fails closed when replay or restored identities drift', async () => {
    const { projection, restored } = projectionAndRestored();
    const driftedArtifact = structuredClone(artifact);
    driftedArtifact.identity.source_replay_sha256 = 'f'.repeat(64);
    await expect(
      bindStudyArtifact(driftedArtifact, projection, restored),
    ).rejects.toThrow(/identity differs/);

    const driftedRestored = structuredClone(restored);
    driftedRestored.command.command_id = 'different-command';
    await expect(
      bindStudyArtifact(artifact, projection, driftedRestored),
    ).rejects.toThrow(/landmark differs/);
  });
});
