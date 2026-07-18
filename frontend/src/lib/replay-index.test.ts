import { readFileSync } from 'node:fs';

import Ajv2020 from 'ajv/dist/2020.js';
import { describe, expect, it } from 'vitest';

import {
  assertAddressBindsReplayDecision,
  assertRestoredReplayDecisionBinds,
  assertViewerSafeReplayProjection,
  assertViewerSafeReplayProjectionResponse,
  canonicalReplayProjectionSha256,
  parseReplayDecisionAddress,
  serializeReplayDecisionAddress,
  type CanonicalReplayProjectionV1,
  type CanonicalReplayProjectionResponseV1,
  type RestoredReplayDecision,
} from './replay-index';
import type { StudyArtifact } from './study-protocol';

function fixture<T>(name: string): T {
  return JSON.parse(
    readFileSync(new URL(`../../../protocol/fixtures/${name}`, import.meta.url), 'utf8'),
  ) as T;
}

const playerZero = fixture<CanonicalReplayProjectionV1>(
  'canonical-replay-player-0.json',
);
const playerOne = fixture<CanonicalReplayProjectionV1>(
  'canonical-replay-player-1.json',
);
const metadata = fixture<{ decisions: Array<{ ordinal: number }> }>(
  'canonical-replay-authority-metadata.json',
);
const study = fixture<StudyArtifact>('study-curated-decision.json');
const schema = fixture<object>('../canonical-replay-v1.schema.json');
const validate = new Ajv2020({ strict: false, validateFormats: false })
  .compile<CanonicalReplayProjectionV1>(schema);

function addressedProjection(): CanonicalReplayProjectionResponseV1 {
  return {
    ...structuredClone(playerZero),
    decisions: playerZero.decisions.map((row) => ({
      ...structuredClone(row),
      address: serializeReplayDecisionAddress({
        version: 1,
        replay_id: playerZero.replay_id,
        match_id: playerZero.match_id,
        ordinal: String(row.ordinal),
        viewer: String(row.viewer),
        revision: String(row.revision),
        prompt_id: String(row.prompt_id),
        offer_id: String(row.offer_id),
        command_id: row.command_id,
        presentation_cursor: String(row.presentation_cursor),
        decision_sha256: '0'.repeat(64),
      }),
    })),
  };
}

describe('canonical replay viewer projection', () => {
  it('validates both isolated viewers whose ordinals form the authority timeline', () => {
    expect(validate(playerZero), JSON.stringify(validate.errors)).toBe(true);
    expect(validate(playerOne), JSON.stringify(validate.errors)).toBe(true);
    expect(() => assertViewerSafeReplayProjection(playerZero)).not.toThrow();
    expect(() => assertViewerSafeReplayProjection(playerOne)).not.toThrow();

    const union = [...playerZero.decisions, ...playerOne.decisions]
      .map(({ ordinal }) => ordinal)
      .sort((left, right) => left - right);
    expect(union).toEqual(metadata.decisions.map(({ ordinal }) => ordinal));
    expect(playerZero.decisions.every(({ viewer }) => viewer === 0)).toBe(true);
    expect(playerOne.decisions.every(({ viewer }) => viewer === 1)).toBe(true);
  });

  it('round-trips Study’s erd1 address without converting u64 fields to Number', () => {
    const encoded = study.landmarks[0].decision_id;
    const address = parseReplayDecisionAddress(encoded);
    expect(serializeReplayDecisionAddress(address)).toBe(encoded);
    const row = playerZero.decisions.find(
      ({ ordinal }) => String(ordinal) === address.ordinal,
    );
    expect(row).toBeDefined();
    if (row) {
      expect(() => assertAddressBindsReplayDecision(address, playerZero, row)).not.toThrow();
    }
    expect(() => parseReplayDecisionAddress(`${encoded}=`)).toThrow(
      /invalid replay decision address/,
    );
  });

  it('rejects mixed viewers, duplicate ordinals, and private opponent hands', () => {
    const mixed = structuredClone(playerZero);
    mixed.decisions.push(structuredClone(playerOne.decisions[0]));
    expect(() => assertViewerSafeReplayProjection(mixed)).toThrow(/mixes viewer/);

    const duplicate = structuredClone(playerZero);
    duplicate.decisions[1].ordinal = duplicate.decisions[0].ordinal;
    expect(() => assertViewerSafeReplayProjection(duplicate)).toThrow(/not increasing/);

    const privateProjection = structuredClone(playerZero);
    privateProjection.decisions[0].frame.projection.opponent.hand.push(
      structuredClone(privateProjection.decisions[0].frame.projection.agent.hand[0]),
    );
    expect(() => assertViewerSafeReplayProjection(privateProjection)).toThrow(
      /decision binding drifted/,
    );
  });

  it('rejects an empty replay namespace and drifted projection identity', () => {
    const emptyReplayId = structuredClone(playerZero);
    emptyReplayId.replay_id = '';
    expect(() => assertViewerSafeReplayProjection(emptyReplayId)).toThrow(
      /requires replay_id/,
    );

    for (const field of ['match_id', 'content_hash', 'asset_manifest_hash'] as const) {
      const drifted = structuredClone(playerZero);
      drifted[field] = field === 'match_id'
        ? 'different-match'
        : '0'.repeat(64);
      expect(() => assertViewerSafeReplayProjection(drifted)).toThrow(
        /decision binding drifted/,
      );
    }
  });

  it('uses one semantic viewer-projection digest across replay and Study', async () => {
    await expect(canonicalReplayProjectionSha256(playerZero)).resolves.toBe(
      study.identity.source_replay_sha256,
    );
  });

  it('validates addressed responses and exact restored decisions', () => {
    const projection = addressedProjection();
    expect(() => assertViewerSafeReplayProjectionResponse(projection)).not.toThrow();

    const row = projection.decisions[0];
    const restored: RestoredReplayDecision = {
      address: row.address,
      ordinal: row.ordinal,
      viewer: row.viewer,
      revision: row.revision,
      presentation_cursor: row.presentation_cursor,
      frame: structuredClone(row.frame),
      offer: structuredClone(row.offer),
      command: structuredClone(row.command),
      continuation: [],
    };
    expect(() => assertRestoredReplayDecisionBinds(projection, restored)).not.toThrow();

    const drifted = structuredClone(projection);
    drifted.decisions[0].address = projection.decisions[1].address;
    expect(() => assertViewerSafeReplayProjectionResponse(drifted)).toThrow(
      /identity drifted/,
    );

    restored.offer.label = 'Client-authored label';
    expect(() => assertRestoredReplayDecisionBinds(projection, restored)).toThrow(
      /identity drifted/,
    );
  });
});
