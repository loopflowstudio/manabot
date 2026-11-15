import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vitest';

import {
  canonicalReplayProjectionSha256,
  type CanonicalReplayProjectionV1,
  type RestoredReplayDecision,
} from './replay-index';
import { bindStudyArtifact, buildStudyPlans, type StudySource } from './study-runtime';
import type { StudyArtifact } from './study-protocol';

function fixture<T>(name: string): T {
  return JSON.parse(
    readFileSync(new URL(`../../../../protocol/fixtures/${name}`, import.meta.url), 'utf8'),
  ) as T;
}

const replay = fixture<CanonicalReplayProjectionV1>('canonical-replay-player-0.json');
const artifact = fixture<StudyArtifact>('study-curated-decision.json');

async function projectionAndRestored(): Promise<{
  projection: StudySource;
  restored: RestoredReplayDecision;
}> {
  const row = replay.decisions[0];
  const projection: StudySource = {
    replay_id: replay.replay_id,
    match_id: replay.match_id,
    viewer: replay.viewer,
    content_hash: replay.content_hash,
    asset_manifest_hash: replay.asset_manifest_hash,
    sha256: await canonicalReplayProjectionSha256(replay),
  };
  return {
    projection,
    restored: {
      address: artifact.landmarks[0].decision_id,
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
    const { projection, restored } = await projectionAndRestored();
    const landmark = bindStudyArtifact(artifact, projection, restored);
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
    const { projection, restored } = await projectionAndRestored();
    const driftedArtifact = structuredClone(artifact);
    driftedArtifact.identity.source_replay_sha256 = 'f'.repeat(64);
    expect(() =>
      bindStudyArtifact(driftedArtifact, projection, restored),
    ).toThrow(/identity differs/);

    const driftedRestored = structuredClone(restored);
    driftedRestored.command.command_id = 'different-command';
    expect(() =>
      bindStudyArtifact(artifact, projection, driftedRestored),
    ).toThrow(/landmark differs/);
  });
});
